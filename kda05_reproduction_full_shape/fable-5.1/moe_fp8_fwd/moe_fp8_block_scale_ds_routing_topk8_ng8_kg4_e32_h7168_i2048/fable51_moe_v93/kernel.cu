// MoE FP8 block-scale (DeepSeek-V3 no-aux routing) for NVIDIA B200 (sm_100a).
// Phase 1: routing -> scan/scatter -> grouped GEMM1+SwiGLU (fp8 mma.sync, fp32 block-scale
// promotion) -> grouped GEMM2 -> combine.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int H = 7168;       // hidden size
constexpr int I = 2048;       // intermediate size
constexpr int NE = 256;       // global experts
constexpr int NLOCAL = 32;    // local experts
constexpr int TOPK = 8;
constexpr int KB1 = H / 128;  // 56 K-blocks for GEMM1
constexpr int KB2 = I / 128;  // 16 K-blocks for GEMM2
constexpr int NT1 = I / 64;   // 32 N-tiles for GEMM1 (64 cols of x1 + 64 cols of x2)
constexpr int ROUTE_WARPS = 8;  // warps per routing block

// ----------------------------------------------------------------------------------------------
// PTX helpers
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, bool valid) {
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src, bool valid) {
  int sz = valid ? 4 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// D = A(16x32 e4m3, row) * B(32x8 e4m3, col) + C, fp32 accumulate.
__device__ __forceinline__ void mma_fp8(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D = A * B (no accumulator input): starts a fresh promotion window without zeroing registers.
__device__ __forceinline__ void mma_fp8_zc(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}

__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// silu(x) = x * sigmoid(x) = 0.5 x (1 + tanh(x/2)) with one MUFU op
__device__ __forceinline__ float silu_fast(float x) { return 0.5f * x * (1.0f + tanh_approx(0.5f * x)); }

__device__ __forceinline__ float bf16_to_f32(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
// v59: 16-byte vector bf16 reduction into global memory (8 bf16 adds, no return value)
__device__ __forceinline__ void red_v4_bf16x2(void* p, uint4 v) {
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
// v92: 256-bit store (sm_100a: STG.E.256); p must be 32-B aligned
__device__ __forceinline__ void st_v8(void* p, uint4 a, uint4 b) {
  asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n" ::"l"(p), "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// ----------------------------------------------------------------------------------------------
// Kernel 1: fused routing + expert compaction. One warp per token (tpw tokens per warp).
// Each block computes top-k for its tokens, publishes per-expert counts, then performs a
// decoupled look-back over all blocks (flags with a per-call epoch) to obtain the global expert
// offsets and its own exclusive prefix, and finally scatters its own pairs. Block 0 also writes
// expert_off[33] / mtile_prefix[33] for the GEMM kernels.
// ----------------------------------------------------------------------------------------------
constexpr int ROUTE_TPW_MAX = 16;
#ifdef MOE_TRACE
__device__ unsigned long long* g_route_trace = nullptr;
#define RTRACE(k) do { if (g_route_trace && threadIdx.x == 0 && blockIdx.x == 0) g_route_trace[k] = gtimer2(); } while (0)
__device__ __forceinline__ unsigned long long gtimer2() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#else
#define RTRACE(k) do {} while (0)
#endif
constexpr int ROUTE_ENTRIES = ROUTE_WARPS * ROUTE_TPW_MAX * TOPK;  // per-block pair slots

__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// One warp routes one token (lane l owns global experts [8l, 8l+8)): DeepSeek-V3 no-aux top-k, identical
// arithmetic to routing_kernel. lv/b: the lane's 8 logits / biases. Outputs slot[i] (rank 0..7 among the
// selected experts, -1 if not selected) and w[i] = normalized weight * scale_factor.
__device__ __forceinline__ void route_token(const float* lv, const float* b, int lane, float scale_factor,
                                            int* slot, float* w) {
  const int g = lane >> 2;
  float s[8], sb[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s[i] = 1.0f / (1.0f + expf(-lv[i]));
    sb[i] = s[i] + b[i];
  }
  float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (sb[i] > m1) {
      m2 = m1;
      m1 = sb[i];
    } else if (sb[i] > m2) {
      m2 = sb[i];
    }
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    float o1 = __shfl_xor_sync(0xffffffffu, m1, off);
    float o2 = __shfl_xor_sync(0xffffffffu, m2, off);
    float n1 = fmaxf(m1, o1);
    float n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
    m1 = n1;
    m2 = n2;
  }
  const float gscore = m1 + m2;
  int rank = 0;
#pragma unroll
  for (int h = 0; h < 8; ++h) {
    float gs = __shfl_sync(0xffffffffu, gscore, h * 4);
    rank += (gs > gscore) || (gs == gscore && h < g);
  }
  const bool keep = rank < 4;
#pragma unroll
  for (int i = 0; i < 8; ++i) sb[i] = keep ? sb[i] : -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; ++i) slot[i] = -1;
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t u = __float_as_uint(sb[i]);
    key[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  }
#pragma unroll 1
  for (int r = 0; r < TOPK; ++r) {
    uint32_t bk = 0;
    int bi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (key[i] > bk) {
        bk = key[i];
        bi = i;
      }
    }
    const uint32_t wmax = __reduce_max_sync(0xffffffffu, bk);
    const uint32_t ballot = __ballot_sync(0xffffffffu, bk == wmax);
    const int owner = __ffs(ballot) - 1;
    if (lane == owner) {
      // predicated updates keep key/slot in registers (a runtime index would spill them to local memory)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (i == bi) {
          key[i] = 0;
          slot[i] = r;
        }
      }
    }
  }
  float wsum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) wsum += (slot[i] >= 0) ? s[i] : 0.f;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
  const float inv = scale_factor / (wsum + 1e-20f);
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = s[i] * inv;
}

__device__ __forceinline__ uint32_t f2key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// v66: one CTA (>= 8 warps, all threads call) routes one token with the same selection semantics as route_token
// (ties: lower expert index first): warp g scores group g (lane = expert 32g + lane; top-2 by two redux rounds),
// the 4 kept groups park their 32 keys in smem, warp 0 selects the top-8 with 8 redux rounds and writes the
// outputs (route_scratch pairs, token_npair, zero rows). Scratch: s_s[256] f32, s_gs[8] f32, s_kept[4] int,
// s_cand[128] u32. Ends with a __syncthreads (scratch reusable).
__device__ __forceinline__ void route_token_cta(const float* __restrict__ logits_row, const uint16_t* __restrict__ bias,
                                                int warp, int lane, float scale_factor, int local_off,
                                                float* s_s, float* s_gs, int* s_kept, uint32_t* s_cand, int t,
                                                int* route_scratch, int* token_npair, uint16_t* out, int combine_atomic) {
  uint32_t key = 0;
  if (warp < 8) {
    const int e = warp * 32 + lane;
    const float lv = __ldg(logits_row + e);
    const float b = bf16_to_f32(__ldg(bias + e));
    const float sv = 1.0f / (1.0f + expf(-lv));
    const float sb = sv + b;
    s_s[e] = sv;
    key = f2key(sb);
    const uint32_t k1 = __reduce_max_sync(0xffffffffu, key);
    const int o1 = __ffs(__ballot_sync(0xffffffffu, key == k1)) - 1;
    const uint32_t k2 = __reduce_max_sync(0xffffffffu, lane == o1 ? 0u : key);
    if (lane == 0) s_gs[warp] = key2f(k1) + key2f(k2);
  }
  __syncthreads();
  if (warp < 8) {
    const float gsc = s_gs[warp];
    int rank = 0;
#pragma unroll
    for (int h = 0; h < 8; ++h) {
      const float gs = s_gs[h];
      rank += (gs > gsc) || (gs == gsc && h < warp);
    }
    if (rank < 4) {
      s_cand[rank * 32 + lane] = key;
      if (lane == 0) s_kept[rank] = warp;
    }
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t c0 = s_cand[lane], c1 = s_cand[32 + lane], c2 = s_cand[64 + lane], c3 = s_cand[96 + lane];
    const int g0 = s_kept[0] * 32 + lane, g1 = s_kept[1] * 32 + lane, g2 = s_kept[2] * 32 + lane, g3 = s_kept[3] * 32 + lane;
    int sel = 0;
#pragma unroll 1
    for (int r = 0; r < TOPK; ++r) {
      uint32_t bk = c0;
      int bq = 0;
      if (c1 > bk) { bk = c1; bq = 1; }
      if (c2 > bk) { bk = c2; bq = 2; }
      if (c3 > bk) { bk = c3; bq = 3; }
      const uint32_t wmax = __reduce_max_sync(0xffffffffu, bk);
      const int my_e = (bk == wmax) ? (bq == 0 ? g0 : (bq == 1 ? g1 : (bq == 2 ? g2 : g3))) : 1024;
      const int we = __reduce_min_sync(0xffffffffu, my_e);
      if (my_e == we) {
        if (bq == 0) c0 = 0;
        if (bq == 1) c1 = 0;
        if (bq == 2) c2 = 0;
        if (bq == 3) c3 = 0;
      }
      if (lane == r) sel = we;
    }
    float sv = (lane < TOPK) ? s_s[sel] : 0.f;
    float wsum = sv;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
    const float inv = scale_factor / (wsum + 1e-20f);
    int nloc = 0;
    if (lane < TOPK) {
      const int le = sel - local_off;
      const int lev = (le >= 0 && le < NLOCAL) ? le : -1;
      nloc = lev >= 0 ? 1 : 0;
      *reinterpret_cast<int2*>(route_scratch + 2 * (t * TOPK + lane)) = make_int2(lev, __float_as_int(sv * inv));
    }
    nloc = __reduce_add_sync(0xffffffffu, nloc);
    if (lane == 0) token_npair[t] = nloc;
    if (out != nullptr && (nloc == 0 || (combine_atomic && nloc >= 2))) {  // v86: nullptr -> zero-filled later
      uint4* orow = reinterpret_cast<uint4*>(out + (size_t)t * H);
#pragma unroll 4
      for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(ROUTE_WARPS * 32, 4) routing_kernel(
    const float* __restrict__ logits, const uint16_t* __restrict__ bias_bf16, int T, int tpw,
    int local_off, float scale_factor, int MT, int epoch, int* __restrict__ blk_count,
    int* __restrict__ flags, int* __restrict__ expert_off, int* __restrict__ mtile_prefix,
    int* __restrict__ pair_token, float* __restrict__ pair_w, int* __restrict__ token_pairs,
    const uint8_t* __restrict__ hidden, const float* __restrict__ hidden_scale,
    uint8_t* __restrict__ act_in, float* __restrict__ act_in_scale, int gather, int p_stride,
    int fixed_cap, int* __restrict__ expert_count, int* __restrict__ blk_pre, int* __restrict__ token_npair,
    uint16_t* __restrict__ out, int* __restrict__ multi_list, int combine_atomic) {
  __shared__ int s_count[NLOCAL];
  __shared__ int s_chunk[ROUTE_WARPS * 32];
  __shared__ int s_last;
  __shared__ int s_tot[NLOCAL];
  __shared__ int s_pre[NLOCAL];
  __shared__ int s_off[NLOCAL + 1];
  __shared__ int8_t s_le[ROUTE_ENTRIES];
  __shared__ int16_t s_rank[ROUTE_ENTRIES];
  __shared__ float s_w[ROUTE_ENTRIES];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int B = gridDim.x;
  const int tok_per_blk = ROUTE_WARPS * tpw;
  const int tok0 = blockIdx.x * tok_per_blk;
  const int tok1 = min(T, tok0 + tok_per_blk);
  if (threadIdx.x < NLOCAL) s_count[threadIdx.x] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) expert_count[2 * NLOCAL + 2 + ((epoch + 1) & 1)] = 0;  // v59: next call's multi-list slot
  __syncthreads();
  RTRACE(0);
  pdl_launch_dependents();

  // bias for this lane's 8 experts
  float b[8];
  {
    uint4 bv = *reinterpret_cast<const uint4*>(bias_bf16 + lane * 8);
    uint32_t w[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      b[2 * i] = bf16_to_f32((uint16_t)(w[i] & 0xffff));
      b[2 * i + 1] = bf16_to_f32((uint16_t)(w[i] >> 16));
    }
  }
  const int g = lane >> 2;  // group of this lane

  // software-pipelined logits loads: the next token's 32 B per lane are requested before the
  // current token's selection runs (hides the DRAM latency of the uncached logits rows)
  float4 nv0 = make_float4(0.f, 0.f, 0.f, 0.f), nv1 = nv0;
  if (tok0 + warp * tpw < T) {
    const float4* lp = reinterpret_cast<const float4*>(logits + (size_t)(tok0 + warp * tpw) * NE + lane * 8);
    nv0 = __ldg(lp);
    nv1 = __ldg(lp + 1);
  }
  for (int ti = 0; ti < tpw; ++ti) {
    const int tl = warp * tpw + ti;  // token index within block
    const int tok = tok0 + tl;
    if (tok >= T) break;
    float s[8], sb[8];
    {
      const float4 v0 = nv0, v1 = nv1;
      if (ti + 1 < tpw && tok + 1 < T) {
        const float4* lp = reinterpret_cast<const float4*>(logits + (size_t)(tok + 1) * NE + lane * 8);
        nv0 = __ldg(lp);
        nv1 = __ldg(lp + 1);
      }
      float lv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        s[i] = 1.0f / (1.0f + expf(-lv[i]));
        sb[i] = s[i] + b[i];
      }
    }
    // group score: sum of top-2 biased scores within the group (4 lanes x 8 experts)
    float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (sb[i] > m1) {
        m2 = m1;
        m1 = sb[i];
      } else if (sb[i] > m2) {
        m2 = sb[i];
      }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      float o1 = __shfl_xor_sync(0xffffffffu, m1, off);
      float o2 = __shfl_xor_sync(0xffffffffu, m2, off);
      float n1 = fmaxf(m1, o1);
      float n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
      m1 = n1;
      m2 = n2;
    }
    const float gscore = m1 + m2;
    int rank = 0;
#pragma unroll
    for (int h = 0; h < 8; ++h) {
      float gs = __shfl_sync(0xffffffffu, gscore, h * 4);
      rank += (gs > gscore) || (gs == gscore && h < g);
    }
    const bool keep = rank < 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) sb[i] = keep ? sb[i] : -INFINITY;

    // top-8 experts over the kept groups: 8 rounds of warp-wide max via redux.sync on an
    // order-preserving 32-bit key (exact float compare), ties -> lowest expert index.
    int slot[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) slot[i] = -1;
    uint32_t key[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t u = __float_as_uint(sb[i]);
      key[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // monotonic map (NaN-free inputs)
    }
#pragma unroll 1
    for (int r = 0; r < TOPK; ++r) {
      uint32_t bk = 0;
      int bi = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (key[i] > bk) {
          bk = key[i];
          bi = i;
        }
      }
      const uint32_t wmax = __reduce_max_sync(0xffffffffu, bk);
      const uint32_t ballot = __ballot_sync(0xffffffffu, bk == wmax);
      const int owner = __ffs(ballot) - 1;  // lowest lane -> lowest expert index among ties
      if (lane == owner) {
        // predicated updates keep key/slot in registers (a runtime index would spill them to local memory)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (i == bi) {
            key[i] = 0;
            slot[i] = r;
          }
        }
      }
    }
    float wsum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) wsum += (slot[i] >= 0) ? s[i] : 0.f;
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
    const float inv = scale_factor / (wsum + 1e-20f);
    int nloc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (slot[i] >= 0) {
        const int ge = lane * 8 + i;
        const int le = ge - local_off;
        const int idx = tl * TOPK + slot[i];
        if (le >= 0 && le < NLOCAL) {
          int rk = atomicAdd(&s_count[le], 1);
          s_le[idx] = (int8_t)le;
          s_rank[idx] = (int16_t)rk;
          s_w[idx] = s[i] * inv;
          ++nloc;
        } else {
          s_le[idx] = -1;
        }
      }
    }
    nloc = __reduce_add_sync(0xffffffffu, nloc);
    if (lane == 0) {
      token_npair[tok] = nloc;
      if (nloc >= 2) {  // tokens that need the combine kernel (compact list, parity slot by epoch)
        const int mi = atomicAdd(expert_count + 2 * NLOCAL + 2 + (epoch & 1), 1);
        multi_list[mi] = tok;
      }
    }
    if (nloc == 0 || (combine_atomic && nloc >= 2)) {
      // no local expert: the output row is zero; multi-pair rows (atomic combine) start from zero and
      // receive the GEMM2 rows by vector reductions
      uint4* orow = reinterpret_cast<uint4*>(out + (size_t)tok * H);
#pragma unroll 4
      for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  __syncthreads();
  RTRACE(1);
  // ---- publish this block's aggregate -------------------------------------------------------
  if (fixed_cap > 0) {
    // fixed-capacity slots: expert e owns rows [e*cap, (e+1)*cap); one atomic per (block, expert)
    // yields this block's offset inside the expert segment (order between blocks is arbitrary).
    if (threadIdx.x < NLOCAL) {
      const int c = s_count[threadIdx.x];
      s_pre[threadIdx.x] = c ? atomicAdd(expert_count + threadIdx.x, c) : 0;
      s_off[threadIdx.x] = threadIdx.x * fixed_cap;
      s_tot[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) s_off[NLOCAL] = NLOCAL * fixed_cap;
  } else if (B == 1) {
    // single block: everything is already in shared memory
    if (threadIdx.x < NLOCAL) {
      s_tot[threadIdx.x] = s_count[threadIdx.x];
      s_pre[threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int o = 0, m = 0;
#pragma unroll
      for (int e = 0; e < NLOCAL; ++e) {
        s_off[e] = o;
        expert_off[e] = o;
        mtile_prefix[e] = m;
        o += s_tot[e];
        m += (s_tot[e] + MT - 1) / MT;
      }
      s_off[NLOCAL] = o;
      expert_off[NLOCAL] = o;
      mtile_prefix[NLOCAL] = m;
    }
  } else {
  if (threadIdx.x < NLOCAL) {
    __stcg(blk_count + blockIdx.x * NLOCAL + threadIdx.x, s_count[threadIdx.x]);
  }
  __syncthreads();
  int* route_ctr = expert_count + 2 * NLOCAL;   // zeroed by the combine kernel of the previous call
  int* done_flag = expert_count + 2 * NLOCAL + 1;  // epoch-tagged
  if (threadIdx.x == 0) {
    __threadfence();
    const int old = atomicAdd(route_ctr, 1);
    s_last = (old == B - 1) ? 1 : 0;
    if (s_last) *route_ctr = 0;  // v59: every block has arrived; reset for the next call
  }
  __syncthreads();
  if (s_last) {
    // ---- last block: one global reduction + per-block exclusive prefixes ------------------
    __threadfence();
    constexpr int TPE = ROUTE_WARPS * 32 / NLOCAL;  // threads per expert
    const int e = threadIdx.x % NLOCAL, part = threadIdx.x / NLOCAL;
    const int chunk = (B + TPE - 1) / TPE;
    const int j0 = part * chunk, j1 = min(B, j0 + chunk);
    int sum = 0;
    for (int j = j0; j < j1; ++j) sum += __ldcg(blk_count + j * NLOCAL + e);
    s_chunk[part * NLOCAL + e] = sum;
    __syncthreads();
    int base = 0, total = 0;
#pragma unroll
    for (int q = 0; q < TPE; ++q) {
      const int v = s_chunk[q * NLOCAL + e];
      if (q < part) base += v;
      total += v;
    }
    int run = base;
    for (int j = j0; j < j1; ++j) {
      const int c = __ldcg(blk_count + j * NLOCAL + e);
      __stcg(blk_pre + j * NLOCAL + e, run);
      run += c;
    }
    if (part == 0) s_tot[e] = total;
    __syncthreads();
    if (threadIdx.x == 0) {
      int o = 0, m = 0;
#pragma unroll
      for (int x = 0; x < NLOCAL; ++x) {
        expert_off[x] = o;
        mtile_prefix[x] = m;
        o += s_tot[x];
        m += (s_tot[x] + MT - 1) / MT;
      }
      expert_off[NLOCAL] = o;
      mtile_prefix[NLOCAL] = m;
      __threadfence();
      asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(done_flag), "r"(epoch) : "memory");
    }
  }
  // ---- all blocks: wait for the global prefix, read own row --------------------------------
  if (threadIdx.x == 0) {
    int f;
    do {
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(f) : "l"(done_flag) : "memory");
      if (f != epoch) __nanosleep(64);
    } while (f != epoch);
  }
  __syncthreads();
  if (threadIdx.x < NLOCAL) {
    s_pre[threadIdx.x] = __ldcg(blk_pre + blockIdx.x * NLOCAL + threadIdx.x);
    s_off[threadIdx.x] = __ldcg(expert_off + threadIdx.x);
    s_tot[threadIdx.x] = 0;
  }
  if (threadIdx.x == 0) s_off[NLOCAL] = __ldcg(expert_off + NLOCAL);
  }  // B > 1
  __syncthreads();
  RTRACE(2);
  // ---- scatter this block's pairs -----------------------------------------------------------
  const int n = (tok1 - tok0) * TOPK;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    const int le = s_le[idx];
    const int gidx = tok0 * TOPK + idx;
    if (le >= 0) {
      const int pos = s_off[le] + s_pre[le] + s_rank[idx];
      pair_token[pos] = tok0 + idx / TOPK;
      pair_w[pos] = s_w[idx];
      token_pairs[gidx] = pos;
    } else {
      token_pairs[gidx] = -1;
    }
  }
  RTRACE(3);
  if (gather) {
    // large-M path: copy each local pair's activation row and its 56 scales into contiguous
    // (expert-grouped) buffers so the GEMM can TMA-load them. Work is flattened over
    // (pair, 16B chunk) so that loads from many rows are in flight concurrently.
    __shared__ int s_lidx[ROUTE_ENTRIES];
    __shared__ int s_lpos[ROUTE_ENTRIES];
    __shared__ int s_nl;
    if (threadIdx.x == 0) s_nl = 0;
    __syncthreads();
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
      const int le = s_le[idx];
      if (le >= 0) {
        const int k = atomicAdd(&s_nl, 1);
        s_lidx[k] = tok0 + idx / TOPK;
        s_lpos[k] = s_off[le] + s_pre[le] + s_rank[idx];
      }
    }
    __syncthreads();
    const int nl = s_nl;
    constexpr int CPR = H / 16;  // 448 chunks per row
    const int total = nl * CPR;
    int w = threadIdx.x;
    for (; w + 3 * (int)blockDim.x < total; w += 4 * blockDim.x) {
      uint4 v[4];
      const uint4* srcp[4];
      uint4* dstp[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ww = w + q * blockDim.x;
        const int j = ww / CPR, c = ww - j * CPR;
        srcp[q] = reinterpret_cast<const uint4*>(hidden + (size_t)s_lidx[j] * H) + c;
        dstp[q] = reinterpret_cast<uint4*>(act_in + (size_t)s_lpos[j] * H) + c;
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = __ldg(srcp[q]);
#pragma unroll
      for (int q = 0; q < 4; ++q) *dstp[q] = v[q];
    }
    for (; w < total; w += blockDim.x) {
      const int j = w / CPR, c = w - j * CPR;
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(hidden + (size_t)s_lidx[j] * H) + c);
      *(reinterpret_cast<uint4*>(act_in + (size_t)s_lpos[j] * H) + c) = v;
    }
    for (int w = threadIdx.x; w < nl * KB1; w += blockDim.x) {
      const int j = w / KB1, kb = w - j * KB1;
      act_in_scale[(size_t)kb * p_stride + s_lpos[j]] = __ldg(hidden_scale + (size_t)kb * T + s_lidx[j]);
    }
  }
  __syncthreads();
  RTRACE(4);
}

// ----------------------------------------------------------------------------------------------
// v63: split routing for T >= 512 (no inter-block look-back, any grid size)
// ----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(ROUTE_WARPS * 32) route_topk_kernel(
    const float* __restrict__ logits, const uint16_t* __restrict__ bias_bf16, int T, int local_off,
    float scale_factor, int epoch, int* __restrict__ blk_count, int* __restrict__ expert_count,
    int* __restrict__ token_npair, uint16_t* __restrict__ out, int* __restrict__ multi_list, int combine_atomic,
    int4* __restrict__ scratch) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tok = blockIdx.x * ROUTE_WARPS + warp;
  if (blockIdx.x == 0 && threadIdx.x == 0) expert_count[2 * NLOCAL + 2 + ((epoch + 1) & 1)] = 0;
  pdl_launch_dependents();
  if (tok < T) {
    float b[8];
    {
      uint4 bv = *reinterpret_cast<const uint4*>(bias_bf16 + lane * 8);
      uint32_t wv[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        b[2 * i] = bf16_to_f32((uint16_t)(wv[i] & 0xffff));
        b[2 * i + 1] = bf16_to_f32((uint16_t)(wv[i] >> 16));
      }
    }
    const float4* lp = reinterpret_cast<const float4*>(logits + (size_t)tok * NE + lane * 8);
    const float4 v0 = __ldg(lp), v1 = __ldg(lp + 1);
    const float lv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    int slot[8];
    float w[8];
    route_token(lv, b, lane, scale_factor, slot, w);
    int nloc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (slot[i] >= 0) {
        const int le = lane * 8 + i - local_off;
        const int idx = tok * TOPK + slot[i];
        if (le >= 0 && le < NLOCAL) {
          const int rk = atomicAdd(expert_count + le, 1);  // v63b: global rank within the expert (reset by the scatter kernel)
          scratch[idx] = make_int4(le, rk, __float_as_int(w[i]), 0);
          ++nloc;
        } else {
          scratch[idx] = make_int4(-1, 0, 0, 0);
        }
      }
    }
    nloc = __reduce_add_sync(0xffffffffu, nloc);
    if (lane == 0) {
      token_npair[tok] = nloc;
      if (nloc >= 2) {
        const int mi = atomicAdd(expert_count + 2 * NLOCAL + 2 + (epoch & 1), 1);
        multi_list[mi] = tok;
      }
    }
  }
}

__global__ void __launch_bounds__(32) route_prefix_kernel(int MT, const int* __restrict__ expert_count,
                                                          int* __restrict__ expert_off, int* __restrict__ mtile_prefix) {
  pdl_wait();
  pdl_launch_dependents();
  const int c = __ldcg(expert_count + threadIdx.x);  // this call's expert totals (v63b global atomic ranks)
  int m = (c + MT - 1) / MT, o = c;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int vo = __shfl_up_sync(0xffffffffu, o, d), vm = __shfl_up_sync(0xffffffffu, m, d);
    if ((int)threadIdx.x >= d) {
      o += vo;
      m += vm;
    }
  }
  expert_off[threadIdx.x + 1] = o;
  mtile_prefix[threadIdx.x + 1] = m;
  if (threadIdx.x == 0) {
    expert_off[0] = 0;
    mtile_prefix[0] = 0;
  }
}

__global__ void __launch_bounds__(ROUTE_WARPS * 32) route_scatter_kernel(
    int T, const int4* __restrict__ scratch, const int* __restrict__ expert_off, int* __restrict__ expert_count,
    const int* __restrict__ token_npair, uint16_t* __restrict__ out, int combine_atomic,
    int* __restrict__ pair_token, float* __restrict__ pair_w, int* __restrict__ token_pairs,
    const uint8_t* __restrict__ hidden, const float* __restrict__ hidden_scale, uint8_t* __restrict__ act_in,
    float* __restrict__ act_in_scale, int gather, int p_stride, int zero_rows) {
  __shared__ int s_off[NLOCAL];
  __shared__ int s_lidx[ROUTE_WARPS * TOPK];
  __shared__ int s_lpos[ROUTE_WARPS * TOPK];
  __shared__ int s_nl;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  pdl_launch_dependents();
  pdl_wait();
  if (threadIdx.x < NLOCAL) s_off[threadIdx.x] = __ldcg(expert_off + threadIdx.x);
  if (blockIdx.x == 0 && threadIdx.x < NLOCAL) expert_count[threadIdx.x] = 0;  // ranks of the next call start at 0
  if (threadIdx.x == 0) s_nl = 0;
  const int tok0 = blockIdx.x * ROUTE_WARPS;
  {
    // zero rows (no local expert, or multi-pair rows that receive vector reductions): one warp per token
    const int tok = tok0 + warp;
    if (tok < T && zero_rows) {
      const int nloc = __ldcg(token_npair + tok);
      if (nloc == 0 || (combine_atomic && nloc >= 2)) {
        uint4* orow = reinterpret_cast<uint4*>(out + (size_t)tok * H);
#pragma unroll 4
        for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
  __syncthreads();
  const int n = min(ROUTE_WARPS, T - tok0) * TOPK;
  for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
    const int4 v = __ldcg(scratch + tok0 * TOPK + idx);
    const int gidx = tok0 * TOPK + idx;
    if (v.x >= 0) {
      const int pos = s_off[v.x] + v.y;
      pair_token[pos] = tok0 + idx / TOPK;
      pair_w[pos] = __int_as_float(v.z);
      token_pairs[gidx] = pos;
      const int k = atomicAdd(&s_nl, 1);
      s_lidx[k] = tok0 + idx / TOPK;
      s_lpos[k] = pos;
    } else {
      token_pairs[gidx] = -1;
    }
  }
  __syncthreads();
  if (gather) {
    const int nl = s_nl;
    constexpr int CPR = H / 16;  // 448 chunks per row
    const int total = nl * CPR;
    int w = threadIdx.x;
#ifndef MOE_SCATTER_U
#define MOE_SCATTER_U 4  // v82: 16-B loads in flight per thread (8 and 16 measured neutral)
#endif
    constexpr int SU = MOE_SCATTER_U;
    for (; w + (SU - 1) * (int)blockDim.x < total; w += SU * blockDim.x) {
      uint4 v[SU];
      const uint4* srcp[SU];
      uint4* dstp[SU];
#pragma unroll
      for (int q = 0; q < SU; ++q) {
        const int ww = w + q * blockDim.x;
        const int j = ww / CPR, c = ww - j * CPR;
        srcp[q] = reinterpret_cast<const uint4*>(hidden + (size_t)s_lidx[j] * H) + c;
        dstp[q] = reinterpret_cast<uint4*>(act_in + (size_t)s_lpos[j] * H) + c;
      }
#pragma unroll
      for (int q = 0; q < SU; ++q) v[q] = __ldg(srcp[q]);
#pragma unroll
      for (int q = 0; q < SU; ++q) *dstp[q] = v[q];
    }
    for (; w < total; w += blockDim.x) {
      const int j = w / CPR, c = w - j * CPR;
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(hidden + (size_t)s_lidx[j] * H) + c);
      *(reinterpret_cast<uint4*>(act_in + (size_t)s_lpos[j] * H) + c) = v;
    }
    for (int w = threadIdx.x; w < nl * KB1; w += blockDim.x) {
      const int j = w / KB1, kb = w - j * KB1;
      act_in_scale[(size_t)kb * p_stride + s_lpos[j]] = __ldg(hidden_scale + (size_t)kb * T + s_lidx[j]);
    }
  }
}

// ----------------------------------------------------------------------------------------------
// Grouped GEMM (persistent CTAs, TMA weight tiles + cp.async activation rows, fp8 mma.sync,
// fp32 block-scale promotion in registers).
// ----------------------------------------------------------------------------------------------
struct GemmParams {
  const uint8_t* A;        // fp8 activations [a_rows][K]
  const float* A_scale;    // [K/ASK][a_rows_stride]
  int a_rows_stride;
  const float* W_scale;    // [NLOCAL][N_total/128][K/128]
  const int* pair_token;   // GEMM1: row -> token gather
  const int* expert_off;   // [33]
  const int* mtile_prefix; // [33]
  int K;
  int N_total;
  int n_tiles;
  // GEMM1 outputs
  uint8_t* act;            // fp8 [P_cap][I]
  float* act_scale;        // [I/64][P_cap]
  int p_cap;
  // GEMM2 outputs
  const float* pair_w;
  uint16_t* pair_out;      // bf16 [P_cap][H]
  const int* token_npair;  // [T] local pairs per token (GEMM2: single-pair tokens write the output directly)
  uint16_t* out;           // [T][H] bf16 output
  // split-K workspace (used only when the unit count is small)
  float* partial;          // [sub_unit][consumer thread][MI*NJ*4]
  int* counters;           // [unit], zero between calls
  unsigned long long* trace;  // optional per-CTA timestamps (MOE_TRACE)
  int split_mult;          // split-K may oversubscribe the grid by this factor
  int b_evict_first;       // UMMA path: weight tiles are not reused (1 m-tile per expert) -> evict_first
  int fixed_cap;           // > 0: expert e rows start at e*fixed_cap, counts in expert_count
  const uint8_t* W;        // v74: raw weights [NLOCAL][N_total][K] (L2 touch prefetch)
  int T;                   // v80: tokens
  int zero_rows;           // v80: GEMM1 warp 3 zeroes the output rows that need it (large path)
  const int* expert_count; // [NLOCAL] (fixed_cap mode)
  int combine_atomic;      // v59: multi-pair rows are red-added into the zeroed output (no combine kernel)
  int a_gather;            // v62: UMMA GEMM1 gathers A rows itself (A = hidden_states, A_scale = hidden_states_scale[kb][T])
  int a_box;               // v69: UMMA A tiles use a 32/64/128-row TMA box chosen from the unit's valid rows
  int acc_warp_release;    // v72: acc_empty count 8 (one arrive per epilogue warp) instead of 256
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#ifndef MOE_TRACE_CTAS
#define MOE_TRACE_CTAS 64  // CTAs with trace slots (s8: -DMOE_TRACE_CTAS=512 for the whole grid; main.py sizes the buffer for 512)
#endif
#ifdef MOE_TRACE
#define TRACE(p, k) do { if ((p).trace && threadIdx.x % 32 == 0 && blockIdx.x < MOE_TRACE_CTAS) (p).trace[(blockIdx.x * 16 + (k)) * 1] = gtimer(); } while (0)
#define TACC(p, k, t0) do { if ((p).trace && blockIdx.x < MOE_TRACE_CTAS) (p).trace[blockIdx.x * 16 + (k)] += gtimer() - (t0); } while (0)
#else
#define TRACE(p, k) do {} while (0)
#define TACC(p, k, t0) do {} while (0)
#endif
#ifdef MOE_TRACE
#define UT_BEGIN() unsigned long long _ut0 = gtimer()
#define UT_END(p, k, cond) do { if ((p).trace && (cond) && blockIdx.x < MOE_TRACE_CTAS) (p).trace[blockIdx.x * 16 + (k)] += gtimer() - _ut0; } while (0)
#else
#define UT_BEGIN() do {} while (0)
#define UT_END(p, k, cond) do {} while (0)
#endif

// mbarrier / TMA helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar,
                                            int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], "
      "[%2];\n" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
// L2 cache policies for TMA loads: evict_first for stream-once weight tiles, evict_last for reused tiles
__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint64_t l2_policy_evict_normal() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma_load_2d_hint(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1,
                                                 uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], "
      "[%2], %5;\n" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
// v84: Blackwell TMA tile::gather4 -- four arbitrary rows (coordinates r0..r3 of dimension 1) x one 128-B column box
// (dimension 0 coordinate c0) land contiguously (4 x box bytes) in smem, swizzled like the tiled mode
__device__ __forceinline__ void tma_gather4_hint(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int r0,
                                                 int r1, int r2, int r3, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1, {%3, %4, %5, %6, %7}], [%2], %8;\n" ::"r"(dst),
      "l"(map), "r"(bar), "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
template <int ID, int COUNT>
__device__ __forceinline__ void named_bar_sync() {
  asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "n"(COUNT) : "memory");
}
__device__ __forceinline__ void tma_prefetch_2d_hint(const CUtensorMap* map, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0, {%1, %2}], %3;\n" ::"l"(map), "r"(c0),
               "r"(c1), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void tma_prefetch_desc(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n" ::"l"(map) : "memory");
}
__device__ __forceinline__ void pf_l2(const void* ptr) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(ptr)); }
__device__ __forceinline__ uint32_t ld_cg_u32(const void* ptr) {
  uint32_t v;
  asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(v) : "l"(ptr));
  return v;
}
// v75: bulk async smem -> global copies / reduce-adds (bulk_group completion)
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_store(void* gdst, uint32_t ssrc, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_red_bf16(void* gdst, uint32_t ssrc, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;\n" ::"l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

template <int MT, int NT, int WARPS_M, int WARPS_N, int STAGES, int ASK, bool GEMM1>
struct GemmCfg {
  static constexpr int NTHREADS = (WARPS_M * WARPS_N + 1) * 32;  // + producer warp
  static constexpr int WM = MT / WARPS_M;
  static constexpr int WN = NT / WARPS_N;
  static constexpr int MI = WM / 16;
  static constexpr int NJ = WN / 8;
  static constexpr int NSUB = 128 / ASK;  // A-scale sub-blocks per 128-K stage
  static constexpr int NSW = GEMM1 ? 2 : NT / 128;  // W-scale values per stage
  static constexpr int W_BYTES = NT * 128;
  static constexpr int A_BYTES = MT * 128;
  static constexpr int SA_BYTES = NSUB * MT * 4;
  static constexpr int SW_BYTES = 32;  // padded
  static constexpr int STAGE_RAW = W_BYTES + A_BYTES + SA_BYTES + SW_BYTES;
  static constexpr int STAGE_BYTES = (STAGE_RAW + 1023) / 1024 * 1024;  // 1 KB aligned (TMA swizzle)
  static constexpr int XBUF_BYTES = GEMM1 ? MT * (NT + 4) * 4 : 0;
  static constexpr int MISC_BYTES = 2 * (NLOCAL + 1) * 4 + 2 * STAGES * 8 + 64;
  static constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES + XBUF_BYTES + MISC_BYTES;
  static_assert(WM % 16 == 0 && WN % 16 == 0, "warp tile");
  static_assert(NJ % 2 == 0, "ldmatrix x4 pairs two n8 subtiles");
  static_assert(GEMM1 ? (WN <= NT / 2 && (NT / 2) % WN == 0) : (WN <= 128 && 128 % WN == 0),
                "each warp must stay inside one scale block");
  static_assert(GEMM1 ? NT == 128 : NT % 128 == 0, "TMA box layout");
};

struct UnitInfo {
  int e, nt, row0, rows_valid;
};

// Fill s_off[0..32], s_mtp[0..32], s_end[0..31] from either the routing prefix arrays or, in
// fixed-capacity mode, from the 32 expert counts (warp 0 does an inclusive shuffle scan).
template <int MT>
__device__ __forceinline__ void load_expert_tables(const GemmParams& p, int* s_off, int* s_mtp, int* s_end) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (p.fixed_cap > 0) {
    if (warp == 0) {
      const int c = (lane < NLOCAL) ? __ldcg(p.expert_count + lane) : 0;
      int m = (c + MT - 1) / MT;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, m, o);
        if (lane >= o) m += v;
      }
      s_mtp[lane + 1] = m;
      s_off[lane] = lane * p.fixed_cap;
      s_end[lane] = lane * p.fixed_cap + c;
      if (lane == 0) {
        s_mtp[0] = 0;
        s_off[NLOCAL] = NLOCAL * p.fixed_cap;
      }
    }
  } else if (tid < 2 * (NLOCAL + 1)) {
    const int v = (tid <= NLOCAL) ? p.expert_off[tid] : p.mtile_prefix[tid - NLOCAL - 1];
    if (tid <= NLOCAL) s_off[tid] = v; else s_mtp[tid - NLOCAL - 1] = v;
    if (tid >= 1 && tid <= NLOCAL) s_end[tid - 1] = v;
  }
}

template <int MT, bool MTMAJOR = false>
__device__ __forceinline__ UnitInfo decode_unit(int u, int mtile_total, const int* s_off,
                                                const int* s_mtp, int n_tiles = 0,
                                                const int* s_end = nullptr) {
  UnitInfo ui;
  int mtg;
  if (MTMAJOR) {
    mtg = u / n_tiles;
    ui.nt = u - mtg * n_tiles;
  } else {
    ui.nt = u / mtile_total;
    mtg = u - ui.nt * mtile_total;
  }
  int e = 0;
#pragma unroll 4
  for (int x = 1; x <= NLOCAL; ++x) e += (s_mtp[x] <= mtg) ? 1 : 0;
  ui.e = e;
  ui.row0 = s_off[e] + (mtg - s_mtp[e]) * MT;
  ui.rows_valid = min(MT, (s_end ? s_end[e] : s_off[e + 1]) - ui.row0);
  return ui;
}

template <int MT, int NT, int WARPS_M, int WARPS_N, int STAGES, int ASK, bool GEMM1>
__global__ void __launch_bounds__((WARPS_M* WARPS_N + 1) * 32, (MT >= 64 ? 1 : 2))
    grouped_gemm_kernel(const __grid_constant__ CUtensorMap tmap, const GemmParams p) {
  using C = GemmCfg<MT, NT, WARPS_M, WARPS_N, STAGES, ASK, GEMM1>;
  constexpr int NCW = WARPS_M * WARPS_N;      // consumer warps
  constexpr int NCT = NCW * 32;               // consumer threads
  constexpr int MI = C::MI, NJ = C::NJ;
  constexpr int ACH = (MT * 8 + 31) / 32;     // activation chunks per producer lane
  constexpr int SCH = (C::NSUB * MT + 31) / 32;  // A-scale items per producer lane
  static_assert(C::NSW <= 32, "scale loads");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw_u32 = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw_u32 & 1023)) & 1023;
  uint8_t* smem = smem_raw + pad;
  const uint32_t smem_base = raw_u32 + pad;
  float* xbuf = reinterpret_cast<float*>(smem + STAGES * C::STAGE_BYTES);
  int* s_off = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full_base = smem_base + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty_base = full_base + STAGES * 8;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TRACE(p, 0);
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full_base + s * 8, 1 + 32);
      mbar_init(empty_base + s * 8, NCW);
    }
    fence_barrier_init();
    tma_prefetch_desc(&tmap);
  }
  pdl_wait();  // routing / previous GEMM results become visible here
  if (tid == 0) TRACE(p, 1);
  pdl_launch_dependents();
  if (tid < 2 * (NLOCAL + 1)) {
    s_off[tid] = (tid <= NLOCAL) ? p.expert_off[tid] : p.mtile_prefix[tid - NLOCAL - 1];
  }
  __syncthreads();
  const int mtile_total = s_mtp[NLOCAL];
  const int total_units = mtile_total * p.n_tiles;
  const int KB = p.K / 128;
  // split-K factor: only when there are too few units to fill the grid
  int S = 1;
  const int cap = (int)gridDim.x * p.split_mult;
  if (total_units * 8 <= cap && (KB % 8) == 0) S = 8;
  else if (total_units * 4 <= cap && (KB % 4) == 0) S = 4;
  else if (total_units * 2 <= cap && (KB % 2) == 0) S = 2;
  const int KBS = KB / S;
  const int total_sub = total_units * S;
  if ((int)blockIdx.x >= total_sub) return;
  const int my_units = (total_sub - blockIdx.x + gridDim.x - 1) / gridDim.x;
  int* s_flag = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4 + 2 * STAGES * 8);

  if (warp == NCW) {
    // =========================== producer warp ==============================================
    int a_row[ACH];
    uint32_t a_dst[ACH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int idx = lane + i * 32;
      a_row[i] = idx >> 3;
      const int ch = idx & 7;
      a_dst[i] = C::W_BYTES + a_row[i] * 128 + ((ch ^ (a_row[i] & 7)) << 4);
    }
    int stage = 0;
    uint32_t phase = 1;  // empty barriers start out "released"
    for (int ui = 0; ui < my_units; ++ui) {
      const int su = blockIdx.x + ui * gridDim.x;
      const int u = su / S;
      const int kb0 = (su - u * S) * KBS;
      const UnitInfo lu = decode_unit<MT>(u, mtile_total, s_off, s_mtp);
      int w_row0, w_row1 = 0;
      if (GEMM1) {
        w_row0 = lu.e * p.N_total + lu.nt * (NT / 2);
        w_row1 = w_row0 + I;
      } else {
        w_row0 = lu.e * p.N_total + lu.nt * NT;
      }
      const uint8_t* a_src[ACH];
      bool a_valid[ACH];
#pragma unroll
      for (int i = 0; i < ACH; ++i) {
        const int idx = lane + i * 32;
        a_valid[i] = (idx < MT * 8) && (a_row[i] < lu.rows_valid);
        int arow = 0;
        if (a_valid[i]) arow = GEMM1 ? p.pair_token[lu.row0 + a_row[i]] : (lu.row0 + a_row[i]);
        a_src[i] = p.A + (size_t)arow * p.K + (idx & 7) * 16 + (size_t)kb0 * 128;
      }
      const float* sa_src[SCH];
      bool sa_valid[SCH];
#pragma unroll
      for (int i = 0; i < SCH; ++i) {
        const int idx = lane + i * 32;
        const int sub = idx / MT, row = idx - sub * MT;
        sa_valid[i] = (idx < C::NSUB * MT) && (row < lu.rows_valid);
        int arow = 0;
        if (sa_valid[i]) arow = GEMM1 ? p.pair_token[lu.row0 + row] : (lu.row0 + row);
        sa_src[i] = p.A_scale + (size_t)(kb0 * C::NSUB + sub) * p.a_rows_stride + arow;
      }
      const float* sw_src = p.W_scale;
      if (lane < C::NSW) {
        int nb;
        if (GEMM1) {
          nb = (lane == 0) ? (lu.nt * (NT / 2)) / 128 : (I + lu.nt * (NT / 2)) / 128;
        } else {
          nb = (lu.nt * NT) / 128 + lane;
        }
        sw_src = p.W_scale + ((size_t)lu.e * (p.N_total / 128) + nb) * KB + kb0;
      }
      if (ui == 0 && lane == 0) TRACE(p, 2);
      for (int kb = kb0; kb < kb0 + KBS; ++kb) {
#ifdef MOE_TRACE
        { unsigned long long t0 = gtimer(); mbar_wait(empty_base + stage * 8, phase);
          if (p.trace && lane == 0 && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 8] += gtimer() - t0; }
#else
        mbar_wait(empty_base + stage * 8, phase);
#endif
        const uint32_t base = smem_base + stage * C::STAGE_BYTES;
        const uint32_t full = full_base + stage * 8;
        if (lane == 0) {
          mbar_arrive_expect_tx(full, C::W_BYTES);
          if (GEMM1) {
            tma_load_2d(base, &tmap, full, kb * 128, w_row0);
            tma_load_2d(base + (NT / 2) * 128, &tmap, full, kb * 128, w_row1);
          } else {
#pragma unroll
            for (int b = 0; b < NT / 128; ++b)
              tma_load_2d(base + b * 128 * 128, &tmap, full, kb * 128, w_row0 + b * 128);
          }
        }
#ifndef MOE_NO_A
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
          if (lane + i * 32 < MT * 8) {
            cp_async_16(base + a_dst[i], a_src[i], a_valid[i]);
            a_src[i] += 128;
          }
        }
#endif
#ifndef MOE_NO_ASCALE
#pragma unroll
        for (int i = 0; i < SCH; ++i) {
          if (lane + i * 32 < C::NSUB * MT) {
            cp_async_4(base + C::W_BYTES + C::A_BYTES + (lane + i * 32) * 4, sa_src[i], sa_valid[i]);
            sa_src[i] += C::NSUB * p.a_rows_stride;
          }
        }
        if (lane < C::NSW) {
          cp_async_4(base + C::W_BYTES + C::A_BYTES + C::SA_BYTES + lane * 4, sw_src, true);
          sw_src += 1;
        }
#endif
        cp_async_mbar_arrive_noinc(full);
        if (stage + 1 == STAGES) {
          stage = 0;
          phase ^= 1;
        } else {
          ++stage;
        }
      }
    }
    return;
  }

  // ============================= consumer warps =============================================
  const int warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
  const int lm_mi = lane >> 3, lm_ri = lane & 7;
  uint32_t a_ld[4], b_ld[NJ / 2][4];
  {
    const int row = warp_m * C::WM + lm_ri + 8 * (lm_mi & 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int ch = 2 * ks + (lm_mi >> 1);
      a_ld[ks] = C::W_BYTES + row * 128 + ((ch ^ (row & 7)) << 4);
    }
#pragma unroll
    for (int jp = 0; jp < NJ / 2; ++jp) {
      const int brow = warp_n * C::WN + jp * 16 + lm_ri + 8 * (lm_mi >> 1);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int ch = 2 * ks + (lm_mi & 1);
        b_ld[jp][ks] = brow * 128 + ((ch ^ (brow & 7)) << 4);
      }
    }
  }
  const int sw_idx = GEMM1 ? ((warp_n * C::WN >= NT / 2) ? 1 : 0) : (warp_n * C::WN) / 128;
  const int sa_r0 = warp_m * C::WM + (lane >> 2);

  float acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  int stage = 0;
  uint32_t phase = 0;
  for (int ui = 0; ui < my_units; ++ui) {
    const int su = blockIdx.x + ui * gridDim.x;
    const int u = su / S;
    const int ssplit = su - u * S;
    const UnitInfo cu = decode_unit<MT>(u, mtile_total, s_off, s_mtp);
    for (int kb = 0; kb < KBS; ++kb) {
#ifdef MOE_TRACE
      { unsigned long long t0 = gtimer(); mbar_wait(full_base + stage * 8, phase);
        if (p.trace && tid == 0 && blockIdx.x < 64) { p.trace[blockIdx.x * 16 + 9] += gtimer() - t0; p.trace[blockIdx.x * 16 + 10] += 1; } }
#else
      mbar_wait(full_base + stage * 8, phase);
#endif
      if (ui == 0 && kb == 0 && warp == 0) TRACE(p, 3);
      if (ui == 0 && kb == KBS - 1 && warp == 0) TRACE(p, 4);
#ifdef MOE_TRACE
      const unsigned long long tc0 = gtimer();
#endif
      const uint8_t* st = smem + stage * C::STAGE_BYTES;
      const uint32_t sbase = smem_base + stage * C::STAGE_BYTES;
      const float* sa = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES);
      const float wsc = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES + C::SA_BYTES)[sw_idx];
#ifndef MOE_NO_MMA
      // fragments for ks=0 (double-buffered in registers; next ks is prefetched before the MMAs)
      uint32_t afrag[2][MI][4], bfrag[2][NJ / 2][4];
      float part[MI][NJ][4];
#pragma unroll
      for (int i = 0; i < MI; ++i)
        ldmatrix_x4(sbase + a_ld[0] + i * 16 * 128, afrag[0][i][0], afrag[0][i][1], afrag[0][i][2],
                    afrag[0][i][3]);
#pragma unroll
      for (int jp = 0; jp < NJ / 2; ++jp)
        ldmatrix_x4(sbase + b_ld[jp][0], bfrag[0][jp][0], bfrag[0][jp][1], bfrag[0][jp][2], bfrag[0][jp][3]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int cur = ks & 1, nxt = cur ^ 1;
        if (ks < 3) {
#pragma unroll
          for (int i = 0; i < MI; ++i)
            ldmatrix_x4(sbase + a_ld[ks + 1] + i * 16 * 128, afrag[nxt][i][0], afrag[nxt][i][1],
                        afrag[nxt][i][2], afrag[nxt][i][3]);
#pragma unroll
          for (int jp = 0; jp < NJ / 2; ++jp)
            ldmatrix_x4(sbase + b_ld[jp][ks + 1], bfrag[nxt][jp][0], bfrag[nxt][jp][1], bfrag[nxt][jp][2],
                        bfrag[nxt][jp][3]);
        }
        // single accumulation chain per promotion window: the first k-step of a window uses C = 0
        // (no register zeroing, no per-chain summation); i-outer / j-inner reuses the A fragment
        const bool first = ((ks * 32) % ASK) == 0;
#pragma unroll
        for (int i = 0; i < MI; ++i) {
#pragma unroll
          for (int jp = 0; jp < NJ / 2; ++jp) {
            if (first) {
              mma_fp8_zc(part[i][2 * jp], afrag[cur][i], bfrag[cur][jp]);
              mma_fp8_zc(part[i][2 * jp + 1], afrag[cur][i], bfrag[cur][jp] + 2);
            } else {
              mma_fp8(part[i][2 * jp], afrag[cur][i], bfrag[cur][jp]);
              mma_fp8(part[i][2 * jp + 1], afrag[cur][i], bfrag[cur][jp] + 2);
            }
          }
        }
        if (((ks + 1) * 32) % ASK == 0) {
          const int sub = (ks * 32) / ASK;
#pragma unroll
          for (int i = 0; i < MI; ++i) {
            const float s0 = sa[sub * MT + sa_r0 + i * 16] * wsc;
            const float s1 = sa[sub * MT + sa_r0 + i * 16 + 8] * wsc;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
              acc[i][j][0] = fmaf(part[i][j][0], s0, acc[i][j][0]);
              acc[i][j][1] = fmaf(part[i][j][1], s0, acc[i][j][1]);
              acc[i][j][2] = fmaf(part[i][j][2], s1, acc[i][j][2]);
              acc[i][j][3] = fmaf(part[i][j][3], s1, acc[i][j][3]);
            }
          }
        }
      }
#endif
      __syncwarp();
#ifdef MOE_TRACE
      if (p.trace && tid == 0 && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 11] += gtimer() - tc0;
#endif
      if (lane == 0) mbar_arrive(empty_base + stage * 8);
      if (stage + 1 == STAGES) {
        stage = 0;
        phase ^= 1;
      } else {
        ++stage;
      }
    }
    if (ui == 0 && warp == 0) TRACE(p, 5);
    // ---- split-K fixup ----------------------------------------------------------------------
    bool do_epi = true;
    if (S > 1) {
      constexpr int PT = MI * NJ * 4;  // floats per thread
      float* pp = p.partial + ((size_t)su * NCT + tid) * PT;
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
          __stcg(reinterpret_cast<float4*>(pp + (i * NJ + j) * 4),
                 make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]));
      __threadfence();
      named_bar_sync<1, NCT>();
      if (tid == 0) {
        const int old = atomicAdd(p.counters + u, 1);
        const int last = (old == S - 1) ? 1 : 0;
        if (last) p.counters[u] = 0;
        *s_flag = last;
      }
      named_bar_sync<1, NCT>();
      do_epi = (*s_flag != 0);
      if (do_epi) {
        __threadfence();
        for (int o = 0; o < S; ++o) {
          if (o == ssplit) continue;
          const float* op = p.partial + ((size_t)(u * S + o) * NCT + tid) * PT;
#pragma unroll
          for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
              const float4 v = __ldcg(reinterpret_cast<const float4*>(op + (i * NJ + j) * 4));
              acc[i][j][0] += v.x;
              acc[i][j][1] += v.y;
              acc[i][j][2] += v.z;
              acc[i][j][3] += v.w;
            }
        }
      }
    }
    // ---- epilogue ---------------------------------------------------------------------------
    if (!do_epi) {
    } else if constexpr (GEMM1) {
      constexpr int XS = NT + 4;  // padded row stride (floats)
      static_assert(XS * MT * 4 <= C::XBUF_BYTES, "xbuf size");
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int r0 = warp_m * C::WM + i * 16 + (lane >> 2);
          const int c0 = warp_n * C::WN + j * 8 + (lane & 3) * 2;
          *reinterpret_cast<float2*>(xbuf + r0 * XS + c0) = make_float2(acc[i][j][0], acc[i][j][1]);
          *reinterpret_cast<float2*>(xbuf + (r0 + 8) * XS + c0) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
      named_bar_sync<1, NCT>();
      constexpr int CG = NT / 16;  // 8-column groups per row (x1 half only)
      static_assert(CG == 8, "epilogue reduction assumes 8 lanes per row");
      static_assert((MT * CG) % 32 == 0, "whole warps in the epilogue loop");
      for (int item = tid; item < MT * CG; item += NCT) {
        const int row = item / CG, cg = item - row * CG;
        float v[8];
        float amax = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const float x1 = xbuf[row * XS + cg * 8 + q];
          const float x2 = xbuf[row * XS + NT / 2 + cg * 8 + q];
          v[q] = x2 * __frcp_rn(1.0f + __expf(-x2)) * x1;
          amax = fmaxf(amax, fabsf(v[q]));
        }
#pragma unroll
        for (int off = 4; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float scale = (amax > 0.f) ? amax * (1.0f / 448.0f) : 1.0f;
        const float inv = __frcp_rn(scale);
        if (row < cu.rows_valid) {
          const int pair = cu.row0 + row;
          uint2 packed;
          packed.x = (uint32_t)pack_e4m3x2(v[0] * inv, v[1] * inv) |
                     ((uint32_t)pack_e4m3x2(v[2] * inv, v[3] * inv) << 16);
          packed.y = (uint32_t)pack_e4m3x2(v[4] * inv, v[5] * inv) |
                     ((uint32_t)pack_e4m3x2(v[6] * inv, v[7] * inv) << 16);
          *reinterpret_cast<uint2*>(p.act + (size_t)pair * I + cu.nt * (NT / 2) + cg * 8) = packed;
          if (cg == 0) p.act_scale[(size_t)cu.nt * p.p_cap + pair] = scale;
        }
      }
      named_bar_sync<1, NCT>();
    } else {
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int r0 = warp_m * C::WM + i * 16 + (lane >> 2);
        const int r1 = r0 + 8;
        const bool v0 = r0 < cu.rows_valid, v1 = r1 < cu.rows_valid;
        const float w0 = v0 ? p.pair_w[cu.row0 + r0] : 0.f;
        const float w1 = v1 ? p.pair_w[cu.row0 + r1] : 0.f;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int c0 = cu.nt * NT + warp_n * C::WN + j * 8 + (lane & 3) * 2;
          if (v0)
            *reinterpret_cast<uint32_t*>(p.pair_out + (size_t)(cu.row0 + r0) * H + c0) =
                pack_bf16x2(acc[i][j][0] * w0, acc[i][j][1] * w0);
          if (v1)
            *reinterpret_cast<uint32_t*>(p.pair_out + (size_t)(cu.row0 + r1) * H + c0) =
                pack_bf16x2(acc[i][j][2] * w1, acc[i][j][3] * w1);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
      for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
    if (ui == 0 && warp == 0) TRACE(p, 6);
    if (ui == my_units - 1 && warp == 0) TRACE(p, 7);
  }
}

// ----------------------------------------------------------------------------------------------
// Small-M grouped GEMM in swap-AB form (v31): the MMA computes D[weight rows][tokens], so the m16
// dimension is spent on weight rows (always full) and the n8 dimension on tokens (MT = 8 or 16).
// Stage = 128 weight rows x 128 K (TMA, 128B swizzle) + MT gathered activation rows (cp.async).
// Consumer warp w owns weight rows [16w, 16w+16) for the whole 128-K of every stage; per stage it
// issues 4 ldmatrix.x4 (weights), 4 ldmatrix.x2/x4 (tokens), 4*MT/8 MMAs and MT/8*4 FMAs.
// ----------------------------------------------------------------------------------------------
template <int MT, int STAGES, int ASK, bool GEMM1>
struct SwCfg {
  static constexpr int NT = 128;
  static constexpr int NCW = NT / 16;               // 8 consumer warps
  static constexpr int NCT = NCW * 32;
  static constexpr int NTHREADS = (NCW + 1) * 32;   // + producer warp
  static constexpr int NTOK = MT / 8;               // n8 token subtiles
  static constexpr int NSUB = 128 / ASK;            // A-scale sub-blocks per stage
  static constexpr int NSW = GEMM1 ? 2 : 1;         // W-scale values per stage
  static constexpr int W_BYTES = NT * 128;
  static constexpr int A_BYTES = MT * 128;
  static constexpr int SA_BYTES = NSUB * MT * 4;
  static constexpr int SW_BYTES = 32;
  static constexpr int STAGE_RAW = W_BYTES + A_BYTES + SA_BYTES + SW_BYTES;
  static constexpr int STAGE_BYTES = (STAGE_RAW + 1023) / 1024 * 1024;
  // epilogue staging: GEMM1 = x2 exchange [MT][64] f32 + amax [4][MT] + fp8 [MT][64]; GEMM2 = bf16 [MT][128]
  static constexpr int XBUF_BYTES = GEMM1 ? (MT * 64 * 4 + 4 * MT * 4 + MT * 64) : (MT * NT * 2 + 64);
  static constexpr int MISC_BYTES = 2 * (NLOCAL + 1) * 4 + 2 * STAGES * 8 + 64 + NLOCAL * 4 + 32 + (NLOCAL + 1) * 4 + 32;  // v86: + s_rem
  static constexpr int SMEM_BYTES = 1024 + STAGES * STAGE_BYTES + XBUF_BYTES + MISC_BYTES;
  static_assert(MT == 8 || MT == 16, "token tile");
};

template <int MT, int STAGES, int ASK, bool GEMM1>
__global__ void __launch_bounds__(SwCfg<MT, STAGES, ASK, GEMM1>::NTHREADS, 2)
    swab_gemm_kernel(const __grid_constant__ CUtensorMap tmap, const GemmParams p) {
  using C = SwCfg<MT, STAGES, ASK, GEMM1>;
  constexpr int NT = C::NT, NCW = C::NCW, NCT = C::NCT, NTOK = C::NTOK;
  constexpr int ACH = (MT * 8 + 31) / 32;         // activation 16 B chunks per producer lane
  constexpr int SCH = (C::NSUB * MT + 31) / 32;   // A-scale items per producer lane
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw_u32 = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw_u32 & 1023)) & 1023;
  uint8_t* smem = smem_raw + pad;
  const uint32_t smem_base = raw_u32 + pad;
  uint8_t* xbuf = smem + STAGES * C::STAGE_BYTES;
  int* s_off = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full_base = smem_base + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty_base = full_base + STAGES * 8;
  int* s_flag = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4 + 2 * STAGES * 8);
  int* s_end = s_flag + 16;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TRACE(p, 0);
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full_base + s * 8, 1 + 32);
      mbar_init(empty_base + s * 8, NCW);
    }
    fence_barrier_init();
    tma_prefetch_desc(&tmap);
  }
  pdl_wait();
  if (tid == 0) TRACE(p, 1);
  pdl_launch_dependents();
  load_expert_tables<MT>(p, s_off, s_mtp, s_end);
  __syncthreads();
  const int mtile_total = s_mtp[NLOCAL];
  const int total_units = mtile_total * p.n_tiles;
  const int KB = p.K / 128;
  int S = 1;
  const int cap = (int)gridDim.x * p.split_mult;
  if (total_units * 8 <= cap && (KB % 8) == 0) S = 8;
  else if (total_units * 4 <= cap && (KB % 4) == 0) S = 4;
  else if (total_units * 2 <= cap && (KB % 2) == 0) S = 2;
  const int KBS = KB / S;
  const int total_sub = total_units * S;
  if ((int)blockIdx.x >= total_sub) return;
  const int my_units = (total_sub - blockIdx.x + gridDim.x - 1) / gridDim.x;
#ifdef MOE_TRACE
  if (tid == 0 && p.trace && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 13] = my_units * 100 + S;
#endif

  if (warp == NCW) {
    // =========================== producer warp ==============================================
    int a_row[ACH];
    uint32_t a_dst[ACH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int idx = lane + i * 32;
      a_row[i] = idx >> 3;
      const int ch = idx & 7;
      a_dst[i] = C::W_BYTES + a_row[i] * 128 + ((ch ^ (a_row[i] & 7)) << 4);
    }
    int stage = 0;
    uint32_t phase = 1;
    const uint64_t pol_w = l2_policy_evict_first();
    for (int ui = 0; ui < my_units; ++ui) {
      const int su = blockIdx.x + ui * gridDim.x;
      const int u = su / S;
      const int kb0 = (su - u * S) * KBS;
      const UnitInfo lu = decode_unit<MT>(u, mtile_total, s_off, s_mtp, 0, s_end);
      int w_row0, w_row1 = 0;
      if (GEMM1) {
        w_row0 = lu.e * p.N_total + lu.nt * (NT / 2);
        w_row1 = w_row0 + I;
      } else {
        w_row0 = lu.e * p.N_total + lu.nt * NT;
      }
      const uint8_t* a_src[ACH];
      bool a_valid[ACH];
#pragma unroll
      for (int i = 0; i < ACH; ++i) {
        const int idx = lane + i * 32;
        a_valid[i] = (idx < MT * 8) && (a_row[i] < lu.rows_valid);
        int arow = 0;
        if (a_valid[i]) arow = GEMM1 ? p.pair_token[lu.row0 + a_row[i]] : (lu.row0 + a_row[i]);
        a_src[i] = p.A + (size_t)arow * p.K + (idx & 7) * 16 + (size_t)kb0 * 128;
      }
      const float* sa_src[SCH];
      bool sa_valid[SCH];
#pragma unroll
      for (int i = 0; i < SCH; ++i) {
        const int idx = lane + i * 32;
        const int sub = idx / MT, row = idx - sub * MT;
        sa_valid[i] = (idx < C::NSUB * MT) && (row < lu.rows_valid);
        int arow = 0;
        if (sa_valid[i]) arow = GEMM1 ? p.pair_token[lu.row0 + row] : (lu.row0 + row);
        sa_src[i] = p.A_scale + (size_t)(kb0 * C::NSUB + sub) * p.a_rows_stride + arow;
      }
      const float* sw_src = p.W_scale;
      if (lane < C::NSW) {
        int nb;
        if (GEMM1) {
          nb = (lane == 0) ? (lu.nt * (NT / 2)) / 128 : (I + lu.nt * (NT / 2)) / 128;
        } else {
          nb = (lu.nt * NT) / 128 + lane;
        }
        sw_src = p.W_scale + ((size_t)lu.e * (p.N_total / 128) + nb) * KB + kb0;
      }
      if (ui == 0 && lane == 0) TRACE(p, 2);
      for (int kb = kb0; kb < kb0 + KBS; ++kb) {
        { UT_BEGIN(); mbar_wait(empty_base + stage * 8, phase); UT_END(p, 8, lane == 0); }
        const uint32_t base = smem_base + stage * C::STAGE_BYTES;
        const uint32_t full = full_base + stage * 8;
        if (lane == 0) {
          mbar_arrive_expect_tx(full, C::W_BYTES);
          if (GEMM1) {
            tma_load_2d_hint(base, &tmap, full, kb * 128, w_row0, pol_w);
            tma_load_2d_hint(base + (NT / 2) * 128, &tmap, full, kb * 128, w_row1, pol_w);
          } else {
            tma_load_2d_hint(base, &tmap, full, kb * 128, w_row0, pol_w);
          }
        }
#ifndef MOE_ABL_NOA
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
          if (lane + i * 32 < MT * 8) {
            cp_async_16(base + a_dst[i], a_src[i], a_valid[i]);
            a_src[i] += 128;
          }
        }
#pragma unroll
        for (int i = 0; i < SCH; ++i) {
          if (lane + i * 32 < C::NSUB * MT) {
            cp_async_4(base + C::W_BYTES + C::A_BYTES + (lane + i * 32) * 4, sa_src[i], sa_valid[i]);
            sa_src[i] += C::NSUB * p.a_rows_stride;
          }
        }
        if (lane < C::NSW) {
          cp_async_4(base + C::W_BYTES + C::A_BYTES + C::SA_BYTES + lane * 4, sw_src, true);
          sw_src += 1;
        }
#endif
        cp_async_mbar_arrive_noinc(full);
        if (stage + 1 == STAGES) {
          stage = 0;
          phase ^= 1;
        } else {
          ++stage;
        }
      }
    }
    return;
  }

  // ============================= consumer warps =============================================
  const int lm_mi = lane >> 3, lm_ri = lane & 7;
  const int g = lane >> 2, t = lane & 3;  // accumulator row (within m16) / token pair
  uint32_t w_ld[4], t_ld[4];
  {
    const int row = warp * 16 + lm_ri + 8 * (lm_mi & 1);
    const int tok = lm_ri + ((NTOK == 2) ? 8 * (lm_mi >> 1) : 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int chw = 2 * ks + (lm_mi >> 1);
      w_ld[ks] = row * 128 + ((chw ^ (row & 7)) << 4);
      const int cht = 2 * ks + (lm_mi & 1);
      t_ld[ks] = C::W_BYTES + tok * 128 + ((cht ^ (tok & 7)) << 4);
    }
  }
  const int sw_idx = GEMM1 ? (warp >= NCW / 2 ? 1 : 0) : 0;
  float acc[NTOK][4];
#pragma unroll
  for (int j = 0; j < NTOK; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  int stage = 0;
  uint32_t phase = 0;
  for (int ui = 0; ui < my_units; ++ui) {
    const int su = blockIdx.x + ui * gridDim.x;
    const int u = su / S;
    const int ssplit = su - u * S;
    const UnitInfo cu = decode_unit<MT>(u, mtile_total, s_off, s_mtp, 0, s_end);
    for (int kb = 0; kb < KBS; ++kb) {
      { UT_BEGIN(); mbar_wait(full_base + stage * 8, phase); UT_END(p, 9, tid == 0); }
      if (ui == 0 && kb == 0 && tid == 0) TRACE(p, 3);
      if (ui == 0 && kb == KBS - 1 && tid == 0) TRACE(p, 4);
      UT_BEGIN();
      const uint8_t* st = smem + stage * C::STAGE_BYTES;
      const uint32_t sbase = smem_base + stage * C::STAGE_BYTES;
      const float* sa = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES);
      const float wsc = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES + C::SA_BYTES)[sw_idx];
      uint32_t wf[2][4], tf[2][2 * NTOK];
      float part[2][NTOK][4];  // two independent accumulation chains (even / odd k-steps)
#ifndef MOE_ABL_NOCONSUME
      ldmatrix_x4(sbase + w_ld[0], wf[0][0], wf[0][1], wf[0][2], wf[0][3]);
      if (NTOK == 1) ldmatrix_x2(sbase + t_ld[0], tf[0][0], tf[0][1]);
      else ldmatrix_x4(sbase + t_ld[0], tf[0][0], tf[0][1], tf[0][2], tf[0][3]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int cur = ks & 1, nxt = cur ^ 1;
        if (ks < 3) {
          ldmatrix_x4(sbase + w_ld[ks + 1], wf[nxt][0], wf[nxt][1], wf[nxt][2], wf[nxt][3]);
          if (NTOK == 1) ldmatrix_x2(sbase + t_ld[ks + 1], tf[nxt][0], tf[nxt][1]);
          else ldmatrix_x4(sbase + t_ld[ks + 1], tf[nxt][0], tf[nxt][1], tf[nxt][2], tf[nxt][3]);
        }
        // window start = first k-step of a scale window; each chain starts fresh (C = 0) on its first
        // k-step inside the window
        const int kw = (ks * 32) % ASK / 32;  // k-step index within the window
        const int ch = kw & 1;
        const bool first = kw < 2;
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          if (first) mma_fp8_zc(part[ch][j], wf[cur], tf[cur] + 2 * j);
          else mma_fp8(part[ch][j], wf[cur], tf[cur] + 2 * j);
        }
        if (((ks + 1) * 32) % ASK == 0) {
          constexpr int KSW = ASK / 32;  // k-steps per window (2 or 4)
          const int sub = (ks * 32) / ASK;
#pragma unroll
          for (int j = 0; j < NTOK; ++j) {
            const float s0 = sa[sub * MT + j * 8 + 2 * t] * wsc;
            const float s1 = sa[sub * MT + j * 8 + 2 * t + 1] * wsc;
            float p0 = part[0][j][0], p1 = part[0][j][1], p2 = part[0][j][2], p3 = part[0][j][3];
            if (KSW >= 2) { p0 += part[1][j][0]; p1 += part[1][j][1]; p2 += part[1][j][2]; p3 += part[1][j][3]; }
            acc[j][0] = fmaf(p0, s0, acc[j][0]);
            acc[j][1] = fmaf(p1, s1, acc[j][1]);
            acc[j][2] = fmaf(p2, s0, acc[j][2]);
            acc[j][3] = fmaf(p3, s1, acc[j][3]);
          }
        }
      }
#else
      (void)sbase; (void)sa; (void)wsc;
#endif
      __syncwarp();
      UT_END(p, 11, tid == 0);
      if (lane == 0) mbar_arrive(empty_base + stage * 8);
      if (stage + 1 == STAGES) {
        stage = 0;
        phase ^= 1;
      } else {
        ++stage;
      }
    }
    if (ui == 0 && tid == 0) TRACE(p, 5);
    // ---- split-K fixup ----------------------------------------------------------------------
    bool do_epi = true;
    if (S > 1) {
      constexpr int PT = NTOK * 4;
      float* pp = p.partial + ((size_t)su * NCT + tid) * PT;
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
        __stcg(reinterpret_cast<float4*>(pp + j * 4), make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]));
      __threadfence();
      named_bar_sync<1, NCT>();
      if (tid == 0) {
        const int old = atomicAdd(p.counters + u, 1);
        const int last = (old == S - 1) ? 1 : 0;
        if (last) p.counters[u] = 0;
        *s_flag = last;
      }
      named_bar_sync<1, NCT>();
      do_epi = (*s_flag != 0);
      if (do_epi) {
        __threadfence();
        for (int o = 0; o < S; ++o) {
          if (o == ssplit) continue;
          const float* op = p.partial + ((size_t)(u * S + o) * NCT + tid) * PT;
#pragma unroll
          for (int j = 0; j < NTOK; ++j) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(op + j * 4));
            acc[j][0] += v.x;
            acc[j][1] += v.y;
            acc[j][2] += v.z;
            acc[j][3] += v.w;
          }
        }
      }
    }
    // ---- epilogue ---------------------------------------------------------------------------
    // lane holds D[row g][tok 2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1] per token subtile j
    if (!do_epi) {
    } else if constexpr (GEMM1) {
      float* x2buf = reinterpret_cast<float*>(xbuf);                 // [MT][64]
      float* amaxbuf = x2buf + MT * 64;                               // [4][MT]
      uint8_t* qbuf = reinterpret_cast<uint8_t*>(amaxbuf + 4 * MT);  // [MT][64]
      const int c0 = (warp & 3) * 16 + g;  // column within the 64-column block (x1 or x2 row index)
      if (warp >= 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          x2buf[tok * 64 + c0] = acc[j][0];
          x2buf[(tok + 1) * 64 + c0] = acc[j][1];
          x2buf[tok * 64 + c0 + 8] = acc[j][2];
          x2buf[(tok + 1) * 64 + c0 + 8] = acc[j][3];
        }
      }
      named_bar_sync<1, NCT>();
      float v[NTOK][4];
      if (warp < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          const float x2a = x2buf[tok * 64 + c0], x2b = x2buf[(tok + 1) * 64 + c0];
          const float x2c = x2buf[tok * 64 + c0 + 8], x2d = x2buf[(tok + 1) * 64 + c0 + 8];
          v[j][0] = x2a * __frcp_rn(1.0f + __expf(-x2a)) * acc[j][0];
          v[j][1] = x2b * __frcp_rn(1.0f + __expf(-x2b)) * acc[j][1];
          v[j][2] = x2c * __frcp_rn(1.0f + __expf(-x2c)) * acc[j][2];
          v[j][3] = x2d * __frcp_rn(1.0f + __expf(-x2d)) * acc[j][3];
          float m0 = fmaxf(fabsf(v[j][0]), fabsf(v[j][2]));  // token 2t
          float m1 = fmaxf(fabsf(v[j][1]), fabsf(v[j][3]));  // token 2t+1
#pragma unroll
          for (int off = 4; off <= 16; off <<= 1) {
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
          }
          if (g == 0) {
            amaxbuf[warp * MT + tok] = m0;
            amaxbuf[warp * MT + tok + 1] = m1;
          }
        }
      }
      named_bar_sync<1, NCT>();
      if (warp < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          float m0 = amaxbuf[tok], m1 = amaxbuf[tok + 1];
#pragma unroll
          for (int w = 1; w < 4; ++w) {
            m0 = fmaxf(m0, amaxbuf[w * MT + tok]);
            m1 = fmaxf(m1, amaxbuf[w * MT + tok + 1]);
          }
          const float sc0 = (m0 > 0.f) ? m0 * (1.0f / 448.0f) : 1.0f;
          const float sc1 = (m1 > 0.f) ? m1 * (1.0f / 448.0f) : 1.0f;
          const float i0 = __frcp_rn(sc0), i1 = __frcp_rn(sc1);
          const uint16_t q0 = pack_e4m3x2(v[j][0] * i0, v[j][2] * i0);  // token 2t: cols c0, c0+8
          const uint16_t q1 = pack_e4m3x2(v[j][1] * i1, v[j][3] * i1);  // token 2t+1
          qbuf[tok * 64 + c0] = (uint8_t)(q0 & 0xff);
          qbuf[tok * 64 + c0 + 8] = (uint8_t)(q0 >> 8);
          qbuf[(tok + 1) * 64 + c0] = (uint8_t)(q1 & 0xff);
          qbuf[(tok + 1) * 64 + c0 + 8] = (uint8_t)(q1 >> 8);
          if (warp == 0 && g == 0) {
            if (tok < cu.rows_valid) p.act_scale[(size_t)cu.nt * p.p_cap + cu.row0 + tok] = sc0;
            if (tok + 1 < cu.rows_valid) p.act_scale[(size_t)cu.nt * p.p_cap + cu.row0 + tok + 1] = sc1;
          }
        }
      }
      named_bar_sync<1, NCT>();
      // coalesced store: MT rows x 64 B
      for (int i = tid; i < MT * 4; i += NCT) {
        const int tok = i >> 2, ch = i & 3;
        if (tok < cu.rows_valid) {
          const uint4 q = *reinterpret_cast<const uint4*>(qbuf + tok * 64 + ch * 16);
          *reinterpret_cast<uint4*>(p.act + (size_t)(cu.row0 + tok) * I + cu.nt * (NT / 2) + ch * 16) = q;
        }
      }
      named_bar_sync<1, NCT>();
    } else {
      uint16_t* obuf = reinterpret_cast<uint16_t*>(xbuf);  // [MT][128] bf16
      const int c0 = warp * 16 + g;
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int tok = j * 8 + 2 * t;
        const float w0 = (tok < cu.rows_valid) ? p.pair_w[cu.row0 + tok] : 0.f;
        const float w1 = (tok + 1 < cu.rows_valid) ? p.pair_w[cu.row0 + tok + 1] : 0.f;
        obuf[tok * NT + c0] = __bfloat16_as_ushort(__float2bfloat16(acc[j][0] * w0));
        obuf[(tok + 1) * NT + c0] = __bfloat16_as_ushort(__float2bfloat16(acc[j][1] * w1));
        obuf[tok * NT + c0 + 8] = __bfloat16_as_ushort(__float2bfloat16(acc[j][2] * w0));
        obuf[(tok + 1) * NT + c0 + 8] = __bfloat16_as_ushort(__float2bfloat16(acc[j][3] * w1));
      }
      int* s_dst = reinterpret_cast<int*>(obuf + MT * NT);  // per row: token index if it has one local pair, else -1
      if (tid < MT) {
        int d = -1;
        if (tid < cu.rows_valid) {
          const int tk = p.pair_token[cu.row0 + tid];
          if (p.token_npair[tk] == 1) d = tk;
        }
        s_dst[tid] = d;
      }
      named_bar_sync<1, NCT>();
      for (int i = tid; i < MT * 16; i += NCT) {
        const int tok = i >> 4, ch = i & 15;
        if (tok < cu.rows_valid) {
          const uint4 q = *reinterpret_cast<const uint4*>(obuf + tok * NT + ch * 8);
          const int d = s_dst[tok];
          uint16_t* dst = (d >= 0) ? (p.out + (size_t)d * H) : (p.pair_out + (size_t)(cu.row0 + tok) * H);
          *reinterpret_cast<uint4*>(dst + cu.nt * NT + ch * 8) = q;
        }
      }
      named_bar_sync<1, NCT>();
    }
#pragma unroll
    for (int j = 0; j < NTOK; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
    if (ui == 0 && tid == 0) TRACE(p, 6);
    if (ui == my_units - 1 && tid == 0) TRACE(p, 7);
  }
}

// ----------------------------------------------------------------------------------------------
// Fused persistent small-path kernel (v42): all GEMM1 sub-units, then all GEMM2 sub-units in one
// launch. GEMM2 units prefetch their weight tiles before waiting on the per-expert GEMM1 counter.
// ----------------------------------------------------------------------------------------------
struct FusedParams {
  GemmParams g1, g2;
  int* g1_done;  // [NLOCAL] GEMM1 units completed per expert (zeroed by the combine kernel)
  int* work_ctr; // v56: dynamic unit counter (zeroed by the combine kernel)
  int sched_target;  // v56: > 0 -> split so that sub-units >= sched_target * grid (else legacy rule)
  // v58: in-kernel routing (T < 512)
  int route_inkernel;
  const float* logits;
  const uint16_t* bias;
  int T, local_off;
  float scale_factor;
  int* pair_token_w;
  float* pair_w_w;
  int* token_pairs;
  int* token_npair;
  uint16_t* out;
  int* zf_done;  // v59: CTAs that finished zero-filling (in-kernel routing); GEMM2 reds wait for gridDim.x
  int* fin_ctr;  // v59: finished CTAs; the last one resets g1_done / work_ctr / zf_done
  // v60: cooperative routing (route_coop): CTA b routes tokens b, b+grid, ...; grid barrier on grid_bar
  int route_coop;
  int epoch;
  int* route_scratch;  // [T*8][2] int32: local expert (or -1), weight bits
  int* grid_bar;       // monotonic arrival counter (epoch * gridDim.x = all arrived)
  // v65: work-conserving two-queue scheduler: GEMM2 sub-units are dequeued as soon as their expert's GEMM1 is
  // complete; sched_wc = 1 + 4 * (reserve of GEMM1 sub-units, in grids) that must remain before a ready GEMM2
  // unit is preferred (0 = legacy single queue)
  int sched_wc;
  int* work_ctr2;
  int s1_mult, s2_mult;  // v65: split-K so that sub-units >= mult * grid (power of two <= 8); 0 = legacy rule
  int route_cta;         // v66: CTA-wide routing (one token at a time per CTA) instead of one warp per token
  int early_w;           // v67: the producer issues the first unit's weight stages during the table build
  int spec_pf;           // v68: K-blocks of the first-wave GEMM1 weight tiles prefetched into L2 before routing (0 = off)
  int spec_w;            // v86: first-wave weight stages issued into smem before routing (expert-space first wave)
  int g2_pf;             // v73: GEMM2 units (own + next ones) a waiting GEMM2 CTA prefetches into L2 (0 = off)
  int pf_ahead;          // v73: K-blocks prefetched into L2 ahead of the smem ring for every unit (0 = off)
  int pf_pol;            // v73: L2 policy of those prefetches: 0 normal, 1 evict_last, 2 evict_first
};

__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_relaxed_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <int MT, int STAGES>
__global__ void __launch_bounds__(SwCfg<MT, STAGES, 64, true>::NTHREADS, 2)
    swab_fused_kernel(const __grid_constant__ CUtensorMap tm1, const __grid_constant__ CUtensorMap tm2,
                      const FusedParams fp) {
  using C = SwCfg<MT, STAGES, 64, true>;  // NSUB = 2 and the GEMM1 xbuf: superset layout for both phases
  constexpr int NT = C::NT, NCW = C::NCW, NCT = C::NCT, NTOK = C::NTOK;
  constexpr int ACH = (MT * 8 + 31) / 32;
  constexpr int SCH = (C::NSUB * MT + 31) / 32;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw_u32 = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw_u32 & 1023)) & 1023;
  uint8_t* smem = smem_raw + pad;
  const uint32_t smem_base = raw_u32 + pad;
  uint8_t* xbuf = smem + STAGES * C::STAGE_BYTES;
  int* s_off = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full_base = smem_base + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty_base = full_base + STAGES * 8;
  int* s_flag = reinterpret_cast<int*>(smem + STAGES * C::STAGE_BYTES + C::XBUF_BYTES + 2 * (NLOCAL + 1) * 4 + 2 * STAGES * 8);
  int* s_end = s_flag + 16;
  int* s_unit = s_end + NLOCAL;  // v56: unit-id ring (8 slots), producer -> consumers
  int* s_rem = s_unit + 8;       // v86: [33] prefix of the GEMM1 units outside the first wave (spec mode)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) TRACE(fp.g1, 0);
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < STAGES; ++st) {
      mbar_init(full_base + st * 8, 1 + 32);
      mbar_init(empty_base + st * 8, NCW);
    }
    fence_barrier_init();
    tma_prefetch_desc(&tm1);
    tma_prefetch_desc(&tm2);
  }
  pdl_wait();
  if (tid == 0) TRACE(fp.g1, 1);
  pdl_launch_dependents();
  int pre0 = 0;  // v67: weight stages of the first unit already issued in the prologue (producer warp)
#ifndef MOE_ROUTE_RED_T
#define MOE_ROUTE_RED_T 1  // v81: T=1 routes redundantly in every CTA (43.5 -> 41.9 us); T=7 redundant measured 79 us
#endif
  // ---- v86: speculative first-wave weight issue (before routing) -----------------------------------
  // CTA b assumes GEMM1 unit (expert b / n_tiles, n-tile b % n_tiles, m-tile 0) with no split and issues its
  // first weight stages now; the routing scratch is laid out in the top stages so stages [0, spec_pre) are free.
  const bool spec_w = fp.spec_w != 0 && fp.route_inkernel && fp.route_cta && fp.T > MOE_ROUTE_RED_T &&
                      fp.g1.combine_atomic != 0 && fp.sched_target == 0 && fp.sched_wc == 0 && fp.s1_mult == 0;
  int spec_pre = 0;     // stages free for the preload (= stages issued if the expert exists)
  int spec_issued = 0;  // stages issued speculatively by this CTA
  uint32_t scr0 = 0;    // routing scratch base (byte offset into the stage area)
  if (spec_w) {
    const int nch0 = (fp.T * TOPK + 31) / 32;
    const int scr = ((fp.T * TOPK + 127) & ~127) + fp.T * TOPK * 4 + nch0 * NLOCAL * 4 + 2048;
    int ns = (scr + C::STAGE_BYTES - 1) / C::STAGE_BYTES;
    if (ns > STAGES) ns = STAGES;
    spec_pre = STAGES - ns;
    scr0 = (uint32_t)spec_pre * C::STAGE_BYTES;
    const int e = (int)blockIdx.x / fp.g1.n_tiles;
    if (e < NLOCAL && spec_pre > 0) spec_issued = spec_pre;  // issued after this CTA's routing (see below)
    // every CTA pulls the whole logits array into L2 first: CTAs that start late (launch ramp) must not find
    // their logits loads queued behind the speculative weight traffic in the DRAM (measured: routing 4.8 -> 9.4 us)
    {
      const char* lp = reinterpret_cast<const char*>(fp.logits);
      const int nl = (fp.T * NE * 4) / 128;
      for (int i = tid; i < nl; i += (NCW + 1) * 32) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(lp + (size_t)i * 128));
    }
  }
  // v86: the speculative issue itself (producer warp lane 0); placed after the CTA's own routing so that the
  // routing's logits loads are not queued behind 32 MB of weight traffic in the DRAM
  auto spec_issue = [&]() {
    if (spec_issued > 0) __syncthreads();  // barrier init (tid 0) visible to the producer warp
    if (spec_issued > 0 && warp == NCW && lane == 0) {
      const int e = (int)blockIdx.x / fp.g1.n_tiles, nt = (int)blockIdx.x - e * fp.g1.n_tiles;
      const int w_row0 = e * fp.g1.N_total + nt * (NT / 2), w_row1 = w_row0 + I;
      const uint64_t pol = l2_policy_evict_first();
      for (int q = 0; q < spec_issued; ++q) {
        const uint32_t base = smem_base + q * C::STAGE_BYTES, full = full_base + q * 8;
        mbar_arrive_expect_tx(full, C::W_BYTES);
        tma_load_2d_hint(base, &tm1, full, q * 128, w_row0, pol);
        tma_load_2d_hint(base + (NT / 2) * 128, &tm1, full, q * 128, w_row1, pol);
      }
    }
  };
  spec_issue();  // v86e: at entry; the routing no longer zero-fills output rows, so its fence stays cheap
  if (fp.spec_pf > 0 && warp == NCW && lane == 0) {
    // v68: speculative L2 prefetch (before routing) of the first-wave GEMM1 weight tiles, assuming every local
    // expert has one m-tile: unit blockIdx.x -> (expert u / n_tiles, n-tile u % n_tiles), K-blocks [0, spec_pf)
    const int u = (int)blockIdx.x;
    const int e = u / fp.g1.n_tiles, nt = u - e * fp.g1.n_tiles;
    if (e < NLOCAL) {
      const int w_row0 = e * fp.g1.N_total + nt * (NT / 2), w_row1 = w_row0 + I;
      const uint64_t pol = l2_policy_evict_first();
      for (int q = 0; q < fp.spec_pf; ++q) {
        tma_prefetch_2d_hint(&tm1, q * 128, w_row0, pol);
        tma_prefetch_2d_hint(&tm1, q * 128, w_row1, pol);
      }
    }
  }
  if (fp.route_inkernel) {
    // ---- v58: in-kernel routing: every CTA computes the same deterministic result -----------
    constexpr int NW = NCW + 1;
    const int T = fp.T;
    int8_t* s_le = reinterpret_cast<int8_t*>(smem + scr0);           // [T*8] local expert or -1 (stage scratch)
    float* s_w = reinterpret_cast<float*>(smem + scr0 + (spec_w ? ((T * TOPK + 127) & ~127) : 4096));  // [T*8] weights
    int* s_cnt = reinterpret_cast<int*>(spec_w ? reinterpret_cast<uint8_t*>(s_w) + T * TOPK * 4 : smem + 4096 + 16384);  // [nchunks][32]
    for (int i = tid; i < ((T * TOPK + 31) / 32) * NLOCAL; i += NW * 32) s_cnt[i] = 0;  // v60b: match_any leaders fill it
    float b[8];
    {
      uint4 bv = *reinterpret_cast<const uint4*>(fp.bias + lane * 8);
      uint32_t wv[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        b[2 * i] = bf16_to_f32((uint16_t)(wv[i] & 0xffff));
        b[2 * i + 1] = bf16_to_f32((uint16_t)(wv[i] >> 16));
      }
    }
#ifndef MOE_ROUTE_RED_T
#define MOE_ROUTE_RED_T 1  // v81: T=1 routes redundantly in every CTA (43.5 -> 41.9 us); T=7 redundant measured 79 us
#endif
    // v81: for T <= MOE_ROUTE_RED_T every CTA routes all tokens itself (no grid barrier, no global scratch)
    const bool redundant = fp.route_cta && T <= MOE_ROUTE_RED_T;
    int* s_scr = reinterpret_cast<int*>(smem + 45056);  // [T*8] int2 (local expert, weight) staging (redundant mode)
    if (fp.route_cta) {
      // ---- v66: CTA-wide routing: this CTA routes tokens blockIdx.x + k * gridDim.x, one at a time ----------
      float* r_s = spec_w ? reinterpret_cast<float*>(s_cnt + ((T * TOPK + 31) / 32) * NLOCAL) : reinterpret_cast<float*>(smem + 40960);
      float* r_gs = r_s + NE;
      int* r_kept = reinterpret_cast<int*>(r_gs + 8);
      uint32_t* r_cand = reinterpret_cast<uint32_t*>(r_kept + 8);
      if (redundant) {
        for (int t = 0; t < T; ++t)
          route_token_cta(fp.logits + (size_t)t * NE, fp.bias, warp, lane, fp.scale_factor, fp.local_off, r_s, r_gs,
                          r_kept, r_cand, t, s_scr, fp.token_npair, fp.out, fp.g1.combine_atomic);
      } else {
        for (int t = (int)blockIdx.x; t < T; t += (int)gridDim.x)
          route_token_cta(fp.logits + (size_t)t * NE, fp.bias, warp, lane, fp.scale_factor, fp.local_off, r_s, r_gs,
                          r_kept, r_cand, t, fp.route_scratch, fp.token_npair, spec_w ? nullptr : fp.out,
                          fp.g1.combine_atomic);
      }
    } else {
      // ---- v60: cooperative: this CTA routes tokens t = blockIdx.x + k * gridDim.x (warp k) ----------
      for (int t = (int)blockIdx.x + warp * (int)gridDim.x; t < T; t += NW * (int)gridDim.x) {
        const float4* lp = reinterpret_cast<const float4*>(fp.logits + (size_t)t * NE + lane * 8);
        const float4 v0 = __ldg(lp), v1 = __ldg(lp + 1);
        const float lv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        int slot[8];
        float w[8];
        route_token(lv, b, lane, fp.scale_factor, slot, w);
        int nloc = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (slot[i] >= 0) {
            const int le = lane * 8 + i - fp.local_off;
            const int idx = t * TOPK + slot[i];
            const int lev = (le >= 0 && le < NLOCAL) ? le : -1;
            if (lev >= 0) ++nloc;
            *reinterpret_cast<int2*>(fp.route_scratch + 2 * idx) = make_int2(lev, __float_as_int(w[i]));
          }
        }
        nloc = __reduce_add_sync(0xffffffffu, nloc);
        if (lane == 0) fp.token_npair[t] = nloc;
        if (nloc == 0 || (fp.g1.combine_atomic && nloc >= 2)) {
          uint4* orow = reinterpret_cast<uint4*>(fp.out + (size_t)t * H);
#pragma unroll 4
          for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
    if (tid == 0) TRACE(fp.g1, 11);
    if (redundant) {
      __syncthreads();
      for (int i = tid; i < T * TOPK; i += NW * 32) {
        const int2 v = reinterpret_cast<const int2*>(s_scr)[i];
        s_le[i] = (int8_t)v.x;
        s_w[i] = __int_as_float(v.y);
      }
    } else {
      __threadfence();
      __syncthreads();
      if (tid == 0) TRACE(fp.g1, 12);
      if (tid == 0) {
        atomicAdd(fp.grid_bar, 1);
        const int target = fp.epoch * (int)gridDim.x;
        while (ld_acquire_gpu(fp.grid_bar) < target) __nanosleep(64);
        TRACE(fp.g1, 15);
      }
      __syncthreads();
      for (int i = tid; i < T * TOPK; i += NW * 32) {
        const int2 v = __ldcg(reinterpret_cast<const int2*>(fp.route_scratch) + i);
        s_le[i] = (int8_t)v.x;
        s_w[i] = __int_as_float(v.y);
      }
    }
    __syncthreads();
    if (tid == 0) TRACE(fp.g1, 8);
    const int n = T * TOPK, nchunks = (n + 31) / 32;
    // deterministic ranks: per 32-entry chunk, per expert, count (pass 1) -> exclusive prefix (pass 2) ->
    // rank = prefix + popc(ballot & lanemask_lt) (pass 3)
    for (int c = warp; c < nchunks; c += NW) {
      const int i = c * 32 + lane;
      const int le = (i < n) ? (int)s_le[i] : -1;
      const unsigned m = __match_any_sync(0xffffffffu, le);  // v60b: one match per chunk instead of 32 ballots
      if (le >= 0 && lane == __ffs(m) - 1) s_cnt[c * NLOCAL + le] = __popc(m);
    }
    __syncthreads();
    if (tid == 0) TRACE(fp.g1, 9);
    if (tid < NLOCAL) {
      int run = 0;
      for (int c = 0; c < nchunks; ++c) {
        const int v = s_cnt[c * NLOCAL + tid];
        s_cnt[c * NLOCAL + tid] = run;
        run += v;
      }
      s_end[tid] = run;  // expert count (temporary)
    }
    __syncthreads();
    if (tid == 0) {
      int o = 0, m = 0;
#pragma unroll
      for (int e = 0; e < NLOCAL; ++e) {
        const int c = s_end[e];
        s_off[e] = o;
        s_mtp[e] = m;
        o += c;
        m += (c + MT - 1) / MT;
      }
      s_off[NLOCAL] = o;
      s_mtp[NLOCAL] = m;
    }
    __syncthreads();
    if (tid < NLOCAL) s_end[tid] = s_off[tid + 1];
    if (spec_w && tid == NCT) {
      // v86: prefix over experts of the GEMM1 units outside the first wave (first wave = ids [0, grid) in expert
      // space: expert e owns ids [e * n_tiles, (e + 1) * n_tiles))
      const int n1 = fp.g1.n_tiles, nfull = (int)gridDim.x / n1, remw = (int)gridDim.x - nfull * n1;
      int r = 0;
#pragma unroll 1
      for (int e = 0; e < NLOCAL; ++e) {
        s_rem[e] = r;
        const int mt = s_mtp[e + 1] - s_mtp[e];
        if (mt > 0) r += mt * n1 - (e < nfull ? n1 : (e == nfull ? remw : 0));
      }
      s_rem[NLOCAL] = r;
    }
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const bool early_mode = fp.early_w && fp.sched_target >= 0 && !spec_w;  // v67 (dynamic scheduling: first unit = blockIdx.x)
    if (early_mode && warp == NCW) {
      // v67: the producer warp issues the weight TMA loads of its first unit's first stages now (the A gather
      // needs the pair tables, the weights only (expert, n-tile)); the main loop skips these stages (pre0)
      const int mtt = s_mtp[NLOCAL];
      const int un1 = mtt * fp.g1.n_tiles, kb1 = fp.g1.K / 128;
      int S = 1;
      if (fp.s1_mult > 0) {
        while (S < 8 && un1 * S < fp.s1_mult * (int)gridDim.x && (kb1 % (2 * S)) == 0) S *= 2;
      } else {
        const int capx = (int)gridDim.x * fp.g1.split_mult;
        if (un1 * 8 <= capx && (kb1 % 8) == 0) S = 8;
        else if (un1 * 4 <= capx && (kb1 % 4) == 0) S = 4;
        else if (un1 * 2 <= capx && (kb1 % 2) == 0) S = 2;
      }
      if ((int)blockIdx.x < un1 * S) {
        const int u = (int)blockIdx.x / S, split = (int)blockIdx.x - u * S, KBS = kb1 / S, kb0 = split * KBS;
        const int mtg = u / fp.g1.n_tiles, nt = u - mtg * fp.g1.n_tiles;
        int e = 0;
#pragma unroll 4
        for (int x = 1; x <= NLOCAL; ++x) e += (s_mtp[x] <= mtg) ? 1 : 0;
        const int w_row0 = e * fp.g1.N_total + nt * (NT / 2), w_row1 = w_row0 + I;
        const int npre = KBS < STAGES ? KBS : STAGES;
        if (lane == 0) {
          const uint64_t pol = l2_policy_evict_first();
          for (int q = 0; q < npre; ++q) {
            const uint32_t base = smem_base + q * C::STAGE_BYTES, full = full_base + q * 8;
            mbar_arrive_expect_tx(full, C::W_BYTES);
            tma_load_2d_hint(base, &tm1, full, (kb0 + q) * 128, w_row0, pol);
            tma_load_2d_hint(base + (NT / 2) * 128, &tm1, full, (kb0 + q) * 128, w_row1, pol);
          }
        }
        pre0 = npre;
      }
      if (lane == 0) TRACE(fp.g1, 12);  // producer after the early weight issue (overwrites 'fenced')
    }
    const int cw = early_mode ? NCW : NW;
    for (int c = warp; c < nchunks && warp < cw; c += cw) {
      const int i = c * 32 + lane;
      const int le = (i < n) ? (int)s_le[i] : -1;
      int pos = -1;
      const unsigned m = __match_any_sync(0xffffffffu, le);
      if (le >= 0) pos = s_off[le] + s_cnt[c * NLOCAL + le] + __popc(m & lanemask_lt);
      if (i < n) {
        if (pos >= 0) {
          fp.pair_token_w[pos] = i / TOPK;
          fp.pair_w_w[pos] = s_w[i];
        }
        fp.token_pairs[i] = pos;
      }
    }
    __syncthreads();  // tables + this CTA's global pair writes visible to all its threads (own writes only)
    if (tid == 0) TRACE(fp.g1, 10);
    // v64: the zero rows of every CTA were fenced before the grid barrier -> no zf_done handshake needed
  } else {
    load_expert_tables<MT>(fp.g1, s_off, s_mtp, s_end);
    __syncthreads();
  }
  // v59: last-CTA reset of the per-call counters (all other CTAs have fenced their final writes)
  auto cta_finish_last = [&]() {
    const int old = atomicAdd(fp.fin_ctr, 1);
    if (old == (int)gridDim.x - 1) {
#pragma unroll
      for (int e = 0; e < NLOCAL; ++e) fp.g1_done[e] = 0;
      *fp.work_ctr = 0;
      *fp.work_ctr2 = 0;
      *fp.zf_done = 0;
      *fp.fin_ctr = 0;
      __threadfence();
    }
  };
  const int mtile_total = s_mtp[NLOCAL];
  const int cap = (int)gridDim.x * fp.g1.split_mult;
  const int target = (int)gridDim.x * fp.sched_target;
  auto split_of = [&](int units, int KB, bool by_target) {
    int S = 1;
    if (by_target) {
      // v64: smallest power-of-two split that yields >= target sub-units (GEMM2 tail balance), at most 4
      while (S < 4 && units * S < target && (KB % (2 * S)) == 0) S *= 2;
      if (units * 8 <= cap && (KB % 8) == 0) S = 8;  // tiny T: the legacy rule splits more (few units)
      else if (units * 4 <= cap && (KB % 4) == 0) S = S < 4 ? 4 : S;
      else if (units * 2 <= cap && (KB % 2) == 0) S = S < 2 ? 2 : S;
    } else {
      if (units * 8 <= cap && (KB % 8) == 0) S = 8;
      else if (units * 4 <= cap && (KB % 4) == 0) S = 4;
      else if (units * 2 <= cap && (KB % 2) == 0) S = 2;
    }
    return S;
  };
  auto split_mult_of = [&](int units, int KB, int mult) {
    // v65: smallest power-of-two split (<= 8) with units * S >= mult * grid
    int S = 1;
    while (S < 8 && units * S < mult * (int)gridDim.x && (KB % (2 * S)) == 0) S *= 2;
    return S;
  };
  // v86: spec mode = expert-space first wave (needs the legacy split S1 == 1, i.e. units1 * 2 > grid)
  const bool spec_mode = spec_w && mtile_total * fp.g1.n_tiles * 2 > (int)gridDim.x;
  const int units1 = spec_mode ? ((int)gridDim.x + s_rem[NLOCAL]) : mtile_total * fp.g1.n_tiles, KB1 = fp.g1.K / 128;
  const int S1 = spec_mode ? 1 : (fp.s1_mult > 0 ? split_mult_of(units1, KB1, fp.s1_mult) : split_of(units1, KB1, false));
  const int KBS1 = KB1 / S1, subs1 = units1 * S1;
  const int units2 = mtile_total * fp.g2.n_tiles, KB2 = fp.g2.K / 128;
  const int S2 = fp.s2_mult > 0 ? split_mult_of(units2, KB2, fp.s2_mult) : split_of(units2, KB2, fp.sched_target > 0);
  const int KBS2 = KB2 / S2, subs2 = units2 * S2;
  const int total_sub = subs1 + subs2;
  // v56d: sched_target < 0 -> contiguous static unit ranges balanced by K-block cost (no atomics);
  // otherwise dynamic fetch from the global counter (first unit static)
  const bool contiguous = fp.sched_target == -1;
  const bool static_il = fp.sched_target <= -2;  // static interleaved (blockIdx + i * grid) through the ring
  int su_begin = blockIdx.x, su_end = total_sub;
  if (contiguous) {
    const int c1 = KBS1 + 2, c2 = KBS2 + 2;  // per-unit cost: K-blocks + epilogue/ramp allowance
    const long long tot = (long long)subs1 * c1 + (long long)subs2 * c2;
    auto range_start = [&](int b) -> int {
      const long long tc = tot * b / (long long)gridDim.x;
      if (tc < (long long)subs1 * c1) return (int)(tc / c1);
      return subs1 + (int)((tc - (long long)subs1 * c1) / c2);
    };
    su_begin = range_start(blockIdx.x);
    su_end = ((int)blockIdx.x + 1 == (int)gridDim.x) ? total_sub : range_start(blockIdx.x + 1);
  }
  // v86: zero rows of this CTA's routed tokens (no local expert / multi-pair red-add targets), written after the
  // grid barrier instead of inside the routing (keeps the pre-barrier fence cheap under speculative traffic)
  auto spec_zero_fill = [&](int t0, int nthr) {
    for (int t = (int)blockIdx.x; t < fp.T; t += (int)gridDim.x) {
      const int np = __ldcg(fp.token_npair + t);
      if (np == 0 || np >= 2) {
        uint4* orow = reinterpret_cast<uint4*>(fp.out + (size_t)t * H);
        for (int c = t0; c < H / 8; c += nthr) orow[c] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  if (su_begin >= su_end) {
    if (spec_w) {
      spec_zero_fill(tid, (NCW + 1) * 32);
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicAdd(fp.zf_done, 1);
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) cta_finish_last();
    return;
  }

  // per sub-unit decode shared by producer and consumers
  struct Sub {
    int phase, u, split, kb0, KBS, S;
    UnitInfo ui;
  };
  auto decode_sub = [&](int su) {
    Sub d;
    if (spec_mode && su < subs1) {
      // v86: ids [0, grid): expert-space first wave; ids >= grid: remaining units in expert order
      const int n1 = fp.g1.n_tiles;
      d.phase = 1; d.S = 1; d.u = su; d.split = 0; d.KBS = KB1; d.kb0 = 0;
      int e, l;
      if (su < (int)gridDim.x) {
        e = su / n1;
        l = su - e * n1;
      } else {
        const int v = su - (int)gridDim.x;
        e = 0;
#pragma unroll 4
        for (int x = 1; x <= NLOCAL; ++x) e += (s_rem[x] <= v) ? 1 : 0;
        const int nfull = (int)gridDim.x / n1, remw = (int)gridDim.x - nfull * n1;
        l = v - s_rem[e] + (e < nfull ? n1 : (e == nfull ? remw : 0));
      }
      const int mt = l / n1;
      d.ui.e = e;
      d.ui.nt = l - mt * n1;
      d.ui.row0 = s_off[e] + mt * MT;
      const int rv = s_end[e] - d.ui.row0;
      d.ui.rows_valid = rv > MT ? MT : rv;  // <= 0: null unit (expert without tokens, first wave only)
      return d;
    }
    if (su < subs1) {
      d.phase = 1; d.S = S1; d.u = su / S1; d.split = su - d.u * S1; d.KBS = KBS1; d.kb0 = d.split * KBS1;
    } else {
      const int s2 = su - subs1;
      d.phase = 2; d.S = S2; d.u = s2 / S2; d.split = s2 - d.u * S2; d.KBS = KBS2; d.kb0 = d.split * KBS2;
    }
    // expert-major unit order (m-tile group major, n-tile fastest): an expert's GEMM1 units finish early
    // in the GEMM1 phase so its GEMM2 units overlap the remaining GEMM1 work
    d.ui = decode_unit<MT, true>(d.u, mtile_total, s_off, s_mtp, d.phase == 1 ? fp.g1.n_tiles : fp.g2.n_tiles, s_end);
    return d;
  };

  // v65: work-conserving fetch. Queue 1 = GEMM1 sub-units [0, subs1), queue 2 = GEMM2 sub-units [subs1, total).
  // The static first units are ids 0..grid-1 of the combined order, so queue 1 continues at min(grid, subs1) and
  // queue 2 at max(0, grid - subs1). A GEMM2 unit whose expert's GEMM1 is complete is preferred while more than
  // `reserve` GEMM1 sub-units remain (long GEMM1 units must not become the tail); otherwise GEMM1 first, GEMM2
  // (waiting) only when GEMM1 is exhausted. Returns total_sub when both queues are empty.
  const int q1_base = (int)gridDim.x < subs1 ? (int)gridDim.x : subs1;
  const int q2_base = (int)gridDim.x > subs1 ? (int)gridDim.x - subs1 : 0;
  const int wc_reserve = ((fp.sched_wc - 1) * (int)gridDim.x) / 4;
  auto fetch_wc = [&]() -> int {
    const int h1 = q1_base + ld_relaxed_gpu(fp.work_ctr);
    const int h2 = q2_base + ld_relaxed_gpu(fp.work_ctr2);
    bool take2 = false;
    if (h2 < subs2 && (h1 >= subs1 || subs1 - h1 > wc_reserve)) {
      const Sub d2 = decode_sub(subs1 + h2);
      const int e = d2.ui.e;
      const int need = fp.g1.n_tiles * (s_mtp[e + 1] - s_mtp[e]);
      take2 = ld_acquire_gpu(fp.g1_done + e) >= need;
    }
    if (!take2 && h1 < subs1) {
      const int v = q1_base + atomicAdd(fp.work_ctr, 1);
      if (v < subs1) return v;
    }
    const int v2 = q2_base + atomicAdd(fp.work_ctr2, 1);
    return v2 < subs2 ? subs1 + v2 : total_sub;
  };

#ifdef MOE_CONT_STREAM
  if (warp == NCW) {
    // =========================== producer warp ==============================================
    int a_row[ACH];
    uint32_t a_dst[ACH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int idx = lane + i * 32;
      a_row[i] = idx >> 3;
      const int ch = idx & 7;
      a_dst[i] = C::W_BYTES + a_row[i] * 128 + ((ch ^ (a_row[i] & 7)) << 4);
    }
    int stage = 0;
    uint32_t phase = 1;
    int uq = 0;
    const uint64_t pol_w = l2_policy_evict_first();
    const uint64_t pol_pf = fp.pf_pol == 1 ? l2_policy_evict_last() : (fp.pf_pol == 2 ? l2_policy_evict_first() : l2_policy_evict_normal());
    // v85: continuous unit stream. The producer's per-unit state (unit decode, token rows of the A chunks and of
    // the scale items, GEMM1-done snapshot) is prepared while the current unit's last STAGES K-blocks are in
    // flight: the next unit id is fetched STAGES K-blocks before the end (its atomic latency hides behind the
    // ring), the decode and its global loads two K-blocks later, so the next unit's first stage is issued the
    // moment a ring slot frees up and the smem ring never drains between units (measured 2.3 us per unit
    // transition before). Unit ids are still published through the smem ring before the unit's first stage.
    // The global loads of the decode (pair_token rows, g1_done) are kept raw and only consumed at the unit start,
    // so the producer never stalls on them inside the K loop (v85a consumed them at the fetch point: +2 %).
    struct PUnit {
      Sub d;
      int arow[ACH];     // pair row (GEMM2) / token row (GEMM1) of each A chunk (row clamped into the unit)
      bool avalid[ACH];  // chunk inside the unit's valid rows
      int sarow[SCH];    // row of each A-scale item
      bool savalid[SCH];
      int g1_cnt;        // GEMM2: expert's g1_done counter when sampled (lane 0)
    };
    auto decode_pu = [&](int su_) {
      PUnit u;
      u.d = decode_sub(su_);
      const bool G1_ = (u.d.phase == 1);
      const GemmParams& p_ = G1_ ? fp.g1 : fp.g2;
      const UnitInfo& lu_ = u.d.ui;
      const int rmax = lu_.rows_valid > 0 ? lu_.rows_valid - 1 : 0;
#pragma unroll
      for (int i = 0; i < ACH; ++i) {
        const int idx = lane + i * 32;
        u.avalid[i] = (idx < MT * 8) && (a_row[i] < lu_.rows_valid);
        const int r = a_row[i] < rmax ? a_row[i] : rmax;
        u.arow[i] = G1_ ? p_.pair_token[lu_.row0 + r] : (lu_.row0 + r);
      }
      const int NSUB_ = G1_ ? 1 : 2;
#pragma unroll
      for (int i = 0; i < SCH; ++i) {
        const int idx = lane + i * 32;
        const int sub = idx / MT, row = idx - sub * MT;
        u.savalid[i] = (idx < NSUB_ * MT) && (row < lu_.rows_valid);
        const int r = row < rmax ? row : rmax;
        u.sarow[i] = G1_ ? p_.pair_token[lu_.row0 + r] : (lu_.row0 + r);
      }
      u.g1_cnt = 0x7fffffff;
      if (!G1_ && lane == 0) u.g1_cnt = ld_acquire_gpu(fp.g1_done + lu_.e);
      return u;
    };
    // first unit: static (blockIdx.x)
    int su = su_begin;
    PUnit pu;
    if (su < su_end) pu = decode_pu(su);
    int nxt_raw = 0;
    for (;;) {
      if (su >= su_end) {
        mbar_wait(empty_base + stage * 8, phase);
        if (lane == 0) s_unit[uq & 7] = -1;
        __syncwarp();
        mbar_arrive(full_base + stage * 8);
        if (lane == 0) mbar_arrive(full_base + stage * 8);
        break;
      }
      if (lane == 0) s_unit[uq & 7] = su;
#ifdef MOE_TRACE
      if (fp.g1.trace && lane == 0 && blockIdx.x < 64) { fp.g1.trace[blockIdx.x * 16 + 13] += 1; if (uq < 4) fp.g1.trace[blockIdx.x * 16 + 2 + uq] = gtimer(); }
#endif
      ++uq;
      const Sub d = pu.d;
      const bool G1 = (d.phase == 1);
#ifdef MOE_TRACE
      if (fp.g1.trace && lane == 0 && blockIdx.x < 64 && !G1 && fp.g1.trace[blockIdx.x * 16 + 14] == 0) fp.g1.trace[blockIdx.x * 16 + 14] = gtimer();
#endif
      const GemmParams& p = G1 ? fp.g1 : fp.g2;
      const CUtensorMap* tmap = G1 ? &tm1 : &tm2;
      const int NSUB = G1 ? 1 : 2;   // A-scale sub-blocks per stage (ASK 128 / 64)
      const int NSW = G1 ? 2 : 1;
      const UnitInfo lu = d.ui;
      int w_row0, w_row1 = 0;
      if (G1) {
        w_row0 = lu.e * p.N_total + lu.nt * (NT / 2);
        w_row1 = w_row0 + I;
      } else {
        w_row0 = lu.e * p.N_total + lu.nt * NT;
      }
      const uint8_t* a_src[ACH];
      bool a_valid[ACH];
#pragma unroll
      for (int i = 0; i < ACH; ++i) {
        const int idx = lane + i * 32;
        a_valid[i] = pu.avalid[i];
        a_src[i] = p.A + (size_t)pu.arow[i] * p.K + (idx & 7) * 16 + (size_t)d.kb0 * 128;
      }
      const float* sa_src[SCH];
      bool sa_valid[SCH];
#pragma unroll
      for (int i = 0; i < SCH; ++i) {
        const int idx = lane + i * 32;
        const int sub = idx / MT;
        sa_valid[i] = pu.savalid[i];
        sa_src[i] = p.A_scale + (size_t)(d.kb0 * NSUB + sub) * p.a_rows_stride + pu.sarow[i];
      }
      const float* sw_src = p.W_scale;
      if (lane < NSW) {
        int nb;
        if (G1) nb = (lane == 0) ? (lu.nt * (NT / 2)) / 128 : (I + lu.nt * (NT / 2)) / 128;
        else nb = (lu.nt * NT) / 128 + lane;
        sw_src = p.W_scale + ((size_t)lu.e * (p.N_total / 128) + nb) * (p.K / 128) + d.kb0;
      }
      // GEMM2: prefetch the weight tiles of the first stages, then wait for the expert's GEMM1
      int pre = pre0;  // v67: stages of the first unit already issued in the prologue
      pre0 = 0;
      int pq = STAGES;  // v73: next K-block (unit-relative) to prefetch into L2 ahead of the ring
      if (!G1) {
        const int npre = d.KBS < STAGES ? d.KBS : STAGES;
        for (int q = 0; q < npre; ++q) {
          const int st_q = (stage + q) % STAGES;
          const uint32_t ph_q = ((stage + q) >= STAGES) ? (phase ^ 1) : phase;
          mbar_wait(empty_base + st_q * 8, ph_q);
          const uint32_t base = smem_base + st_q * C::STAGE_BYTES;
          const uint32_t full = full_base + st_q * 8;
          if (lane == 0) {
            mbar_arrive_expect_tx(full, C::W_BYTES);
            tma_load_2d_hint(base, tmap, full, (d.kb0 + q) * 128, w_row0, pol_w);
          }
        }
        pre = npre;
        {
          const int need = fp.g1.n_tiles * (s_mtp[lu.e + 1] - s_mtp[lu.e]);
          int waiting = (lane == 0) ? (pu.g1_cnt >= need ? 0 : (ld_acquire_gpu(fp.g1_done + lu.e) < need ? 1 : 0)) : 0;
          waiting = __shfl_sync(0xffffffffu, waiting, 0);
          if (waiting && fp.g2_pf > 0) {
            // v73: this CTA idles until the expert's GEMM1 is complete -> pull the rest of its weight tile and the
            // tiles of the GEMM2 units it will probably take next (stride = statically assigned GEMM2 CTAs) into L2
            const int stride = (int)gridDim.x - subs1;
            for (int kx = 0; kx < fp.g2_pf; ++kx) {
              const int su_k = su + kx * stride;
              if (kx > 0 && (stride <= 0 || su_k >= total_sub)) break;
              const Sub dk = (kx == 0) ? d : decode_sub(su_k);
              if (dk.phase != 2) break;
              const int wr = dk.ui.e * p.N_total + dk.ui.nt * NT;
              const int q0 = (kx == 0) ? npre : 0;
              for (int q = q0 + lane; q < dk.KBS; q += 32) tma_prefetch_2d_hint(tmap, (dk.kb0 + q) * 128, wr, pol_pf);
            }
            pq = d.KBS;  // the whole unit is already on its way into L2
          }
          if (waiting && lane == 0) {
            while (ld_acquire_gpu(fp.g1_done + lu.e) < need) {
              __nanosleep(200);
            }
          }
          __syncwarp();
        }
      }
      // v85: K-block at which the next unit id is fetched / decoded (both inside this unit's K loop)
      const int q_fetch = d.KBS > STAGES ? d.KBS - STAGES : 0;
      const int q_dec = (q_fetch + 2 < d.KBS) ? q_fetch + 2 : d.KBS - 1;
      int su_next = su_end;
      PUnit pu_next = pu;
      for (int kb = d.kb0; kb < d.kb0 + d.KBS; ++kb) {
        const int q = kb - d.kb0;
        const uint32_t base = smem_base + stage * C::STAGE_BYTES;
        const uint32_t full = full_base + stage * 8;
        if (q == q_fetch) {
          // v56c/v85: the next unit id comes from the global counter (first unit static); fetched early so that
          // the atomic's latency hides behind the ring
          if (contiguous) {
            nxt_raw = su + 1;
          } else if (static_il) {
            nxt_raw = su + (int)gridDim.x;
          } else if (fp.sched_wc > 0) {
            if (lane == 0) nxt_raw = fetch_wc();
          } else if (lane == 0) {
            nxt_raw = atomicAdd(fp.work_ctr, 1);  // + gridDim.x applied at the decode point (raw value: no stall)
          }
        }
        if (q == q_dec) {
          su_next = __shfl_sync(0xffffffffu, nxt_raw, 0);
          if (!contiguous && !static_il && fp.sched_wc <= 0) su_next += (int)gridDim.x;
          if (su_next < su_end) pu_next = decode_pu(su_next);
        }
        if (fp.pf_ahead > 0 && lane == 0) {
          // v73: keep pf_ahead K-blocks beyond the smem ring in flight towards L2
          const int lim = min(d.KBS, q + STAGES + fp.pf_ahead);
          for (; pq < lim; ++pq) {
            tma_prefetch_2d_hint(tmap, (d.kb0 + pq) * 128, w_row0, pol_pf);
            if (G1) tma_prefetch_2d_hint(tmap, (d.kb0 + pq) * 128, w_row1, pol_pf);
          }
        }
        if (q >= pre) {
          mbar_wait(empty_base + stage * 8, phase);
          if (lane == 0) {
            mbar_arrive_expect_tx(full, C::W_BYTES);
            if (G1) {
              tma_load_2d_hint(base, tmap, full, kb * 128, w_row0, pol_w);
              tma_load_2d_hint(base + (NT / 2) * 128, tmap, full, kb * 128, w_row1, pol_w);
            } else {
              tma_load_2d_hint(base, tmap, full, kb * 128, w_row0, pol_w);
            }
          }
        }
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
          if (lane + i * 32 < MT * 8) {
            cp_async_16(base + a_dst[i], a_src[i], a_valid[i]);
            a_src[i] += 128;
          }
        }
#pragma unroll
        for (int i = 0; i < SCH; ++i) {
          if (lane + i * 32 < NSUB * MT) {
            cp_async_4(base + C::W_BYTES + C::A_BYTES + (lane + i * 32) * 4, sa_src[i], sa_valid[i]);
            sa_src[i] += NSUB * p.a_rows_stride;
          }
        }
        if (lane < NSW) {
          cp_async_4(base + C::W_BYTES + C::A_BYTES + C::SA_BYTES + lane * 4, sw_src, true);
          sw_src += 1;
        }
        cp_async_mbar_arrive_noinc(full);
        if (stage + 1 == STAGES) {
          stage = 0;
          phase ^= 1;
        } else {
          ++stage;
        }
      }
      su = su_next;
      pu = pu_next;
    }
    return;
  }
#else
  if (warp == NCW) {
    // =========================== producer warp ==============================================
    int a_row[ACH];
    uint32_t a_dst[ACH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
      const int idx = lane + i * 32;
      a_row[i] = idx >> 3;
      const int ch = idx & 7;
      a_dst[i] = C::W_BYTES + a_row[i] * 128 + ((ch ^ (a_row[i] & 7)) << 4);
    }
    int stage = 0;
    uint32_t phase = 1;
    int uq = 0;
    const uint64_t pol_w = l2_policy_evict_first();
    const uint64_t pol_pf = fp.pf_pol == 1 ? l2_policy_evict_last() : (fp.pf_pol == 2 ? l2_policy_evict_first() : l2_policy_evict_normal());
    // v56c: the first unit is static (blockIdx.x); later unit ids come from the global counter, fetched
    // when the current unit's last stage has been issued. Ids are published through the smem ring
    // before the unit's first stage (visible to the consumers through the stage's full barrier);
    // su >= total_sub -> sentinel stage.
    int su = su_begin;
    int nxt = 0;
    if (lane == 0) TRACE(fp.g1, 6);  // producer at the unit loop entry
    if (spec_issued > 0) {
      // v86: the speculative stages belong to unit blockIdx.x only in spec mode and if the expert has tokens;
      // otherwise they are drained through a pseudo-unit (consumers wait + release, no compute)
      bool ok = spec_mode;
      if (ok) ok = decode_sub(su).ui.rows_valid > 0;
      if (ok) {
        pre0 = spec_issued;
      } else {
        if (lane == 0) s_unit[uq & 7] = -100 - spec_issued;
        __syncwarp();
        ++uq;
        for (int q = 0; q < spec_issued; ++q) {
          cp_async_mbar_arrive_noinc(full_base + stage * 8);
          if (stage + 1 == STAGES) { stage = 0; phase ^= 1; } else { ++stage; }
        }
        if (spec_mode) {  // the null first-wave id is skipped; in legacy order unit blockIdx.x stays this CTA's
          if (lane == 0) nxt = atomicAdd(fp.work_ctr, 1) + (int)gridDim.x;
          su = __shfl_sync(0xffffffffu, nxt, 0);
        }
        pre0 = 0;
      }
    }
    for (;;) {
      if (su >= su_end) {
        mbar_wait(empty_base + stage * 8, phase);
        if (lane == 0) s_unit[uq & 7] = -1;
        __syncwarp();
        mbar_arrive(full_base + stage * 8);
        if (lane == 0) mbar_arrive(full_base + stage * 8);
        break;
      }
      if (lane == 0) s_unit[uq & 7] = su;
#ifdef MOE_TRACE
      if (fp.g1.trace && lane == 0 && blockIdx.x < 64) { fp.g1.trace[blockIdx.x * 16 + 13] += 1; if (uq < 4) fp.g1.trace[blockIdx.x * 16 + 2 + uq] = gtimer(); }
#endif
      ++uq;
      const Sub d = decode_sub(su);
      const bool G1 = (d.phase == 1);
#ifdef MOE_TRACE
      if (fp.g1.trace && lane == 0 && blockIdx.x < 64 && !G1 && fp.g1.trace[blockIdx.x * 16 + 14] == 0) fp.g1.trace[blockIdx.x * 16 + 14] = gtimer();
#endif
      const GemmParams& p = G1 ? fp.g1 : fp.g2;
      const CUtensorMap* tmap = G1 ? &tm1 : &tm2;
      const int NSUB = G1 ? 1 : 2;   // A-scale sub-blocks per stage (ASK 128 / 64)
      const int NSW = G1 ? 2 : 1;
      const UnitInfo lu = d.ui;
      int w_row0, w_row1 = 0;
      if (G1) {
        w_row0 = lu.e * p.N_total + lu.nt * (NT / 2);
        w_row1 = w_row0 + I;
      } else {
        w_row0 = lu.e * p.N_total + lu.nt * NT;
      }
      const uint8_t* a_src[ACH];
      bool a_valid[ACH];
#pragma unroll
      for (int i = 0; i < ACH; ++i) {
        const int idx = lane + i * 32;
        a_valid[i] = (idx < MT * 8) && (a_row[i] < lu.rows_valid);
        int arow = 0;
        if (a_valid[i]) arow = G1 ? p.pair_token[lu.row0 + a_row[i]] : (lu.row0 + a_row[i]);
        a_src[i] = p.A + (size_t)arow * p.K + (idx & 7) * 16 + (size_t)d.kb0 * 128;
      }
      const float* sa_src[SCH];
      bool sa_valid[SCH];
#pragma unroll
      for (int i = 0; i < SCH; ++i) {
        const int idx = lane + i * 32;
        const int sub = idx / MT, row = idx - sub * MT;
        sa_valid[i] = (idx < NSUB * MT) && (row < lu.rows_valid);
        int arow = 0;
        if (sa_valid[i]) arow = G1 ? p.pair_token[lu.row0 + row] : (lu.row0 + row);
        sa_src[i] = p.A_scale + (size_t)(d.kb0 * NSUB + sub) * p.a_rows_stride + arow;
      }
      const float* sw_src = p.W_scale;
      if (lane < NSW) {
        int nb;
        if (G1) nb = (lane == 0) ? (lu.nt * (NT / 2)) / 128 : (I + lu.nt * (NT / 2)) / 128;
        else nb = (lu.nt * NT) / 128 + lane;
        sw_src = p.W_scale + ((size_t)lu.e * (p.N_total / 128) + nb) * (p.K / 128) + d.kb0;
      }
      // GEMM2: prefetch the weight tiles of the first stages, then wait for the expert's GEMM1
      int pre = pre0;  // v67: stages of the first unit already issued in the prologue
      pre0 = 0;
      int pq = STAGES;  // v73: next K-block (unit-relative) to prefetch into L2 ahead of the ring
      if (!G1) {
        const int npre = d.KBS < STAGES ? d.KBS : STAGES;
        for (int q = 0; q < npre; ++q) {
          const int st_q = (stage + q) % STAGES;
          const uint32_t ph_q = ((stage + q) >= STAGES) ? (phase ^ 1) : phase;
          mbar_wait(empty_base + st_q * 8, ph_q);
          const uint32_t base = smem_base + st_q * C::STAGE_BYTES;
          const uint32_t full = full_base + st_q * 8;
          if (lane == 0) {
            mbar_arrive_expect_tx(full, C::W_BYTES);
            tma_load_2d_hint(base, tmap, full, (d.kb0 + q) * 128, w_row0, pol_w);
          }
        }
        pre = npre;
        {
          const int need = fp.g1.n_tiles * (s_mtp[lu.e + 1] - s_mtp[lu.e]);
          int waiting = (lane == 0) ? (ld_acquire_gpu(fp.g1_done + lu.e) < need ? 1 : 0) : 0;
          waiting = __shfl_sync(0xffffffffu, waiting, 0);
          if (waiting && fp.g2_pf > 0) {
            // v73: this CTA idles until the expert's GEMM1 is complete -> pull the rest of its weight tile and the
            // tiles of the GEMM2 units it will probably take next (stride = statically assigned GEMM2 CTAs) into L2
            const int stride = (int)gridDim.x - subs1;
            for (int kx = 0; kx < fp.g2_pf; ++kx) {
              const int su_k = su + kx * stride;
              if (kx > 0 && (stride <= 0 || su_k >= total_sub)) break;
              const Sub dk = (kx == 0) ? d : decode_sub(su_k);
              if (dk.phase != 2) break;
              const int wr = dk.ui.e * p.N_total + dk.ui.nt * NT;
              const int q0 = (kx == 0) ? npre : 0;
              for (int q = q0 + lane; q < dk.KBS; q += 32) tma_prefetch_2d_hint(tmap, (dk.kb0 + q) * 128, wr, pol_pf);
            }
            pq = d.KBS;  // the whole unit is already on its way into L2
          }
          if (lane == 0) {
            while (ld_acquire_gpu(fp.g1_done + lu.e) < need) {
              __nanosleep(200);
            }
          }
          __syncwarp();
        }
      }
      for (int kb = d.kb0; kb < d.kb0 + d.KBS; ++kb) {
        const int q = kb - d.kb0;
        const uint32_t base = smem_base + stage * C::STAGE_BYTES;
        const uint32_t full = full_base + stage * 8;
        if (fp.pf_ahead > 0 && lane == 0) {
          // v73: keep pf_ahead K-blocks beyond the smem ring in flight towards L2
          const int lim = min(d.KBS, q + STAGES + fp.pf_ahead);
          for (; pq < lim; ++pq) {
            tma_prefetch_2d_hint(tmap, (d.kb0 + pq) * 128, w_row0, pol_pf);
            if (G1) tma_prefetch_2d_hint(tmap, (d.kb0 + pq) * 128, w_row1, pol_pf);
          }
        }
        if (q >= pre) {
          mbar_wait(empty_base + stage * 8, phase);
          if (lane == 0) {
            mbar_arrive_expect_tx(full, C::W_BYTES);
            if (G1) {
              tma_load_2d_hint(base, tmap, full, kb * 128, w_row0, pol_w);
              tma_load_2d_hint(base + (NT / 2) * 128, tmap, full, kb * 128, w_row1, pol_w);
            } else {
              tma_load_2d_hint(base, tmap, full, kb * 128, w_row0, pol_w);
            }
          }
        }
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
          if (lane + i * 32 < MT * 8) {
            cp_async_16(base + a_dst[i], a_src[i], a_valid[i]);
            a_src[i] += 128;
          }
        }
#pragma unroll
        for (int i = 0; i < SCH; ++i) {
          if (lane + i * 32 < NSUB * MT) {
            cp_async_4(base + C::W_BYTES + C::A_BYTES + (lane + i * 32) * 4, sa_src[i], sa_valid[i]);
            sa_src[i] += NSUB * p.a_rows_stride;
          }
        }
        if (lane < NSW) {
          cp_async_4(base + C::W_BYTES + C::A_BYTES + C::SA_BYTES + lane * 4, sw_src, true);
          sw_src += 1;
        }
        cp_async_mbar_arrive_noinc(full);
        if (stage + 1 == STAGES) {
          stage = 0;
          phase ^= 1;
        } else {
          ++stage;
        }
      }
      // v56c: fetch the next unit only now (after this unit's last stage was issued): the atomic's latency
      // hides behind the consumers' drain of the ring, and a CTA never holds a unit it cannot start yet
      if (contiguous) {
        nxt = su + 1;
      } else if (static_il) {
        nxt = su + (int)gridDim.x;
      } else if (fp.sched_wc > 0) {
        if (lane == 0) nxt = fetch_wc();
      } else if (lane == 0) {
        nxt = atomicAdd(fp.work_ctr, 1) + (int)gridDim.x;
      }
      su = __shfl_sync(0xffffffffu, nxt, 0);
    }
    return;
  }
#endif

  // ============================= consumer warps =============================================
  const int lm_mi = lane >> 3, lm_ri = lane & 7;
  const int g = lane >> 2, t = lane & 3;
  uint32_t w_ld[4], t_ld[4];
  {
    const int row = warp * 16 + lm_ri + 8 * (lm_mi & 1);
    const int tok = lm_ri + ((NTOK == 2) ? 8 * (lm_mi >> 1) : 0);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int chw = 2 * ks + (lm_mi >> 1);
      w_ld[ks] = row * 128 + ((chw ^ (row & 7)) << 4);
      const int cht = 2 * ks + (lm_mi & 1);
      t_ld[ks] = C::W_BYTES + tok * 128 + ((cht ^ (tok & 7)) << 4);
    }
  }
  float acc[NTOK][4];
#pragma unroll
  for (int j = 0; j < NTOK; ++j)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;

  int stage = 0;
  uint32_t phase = 0;
  int uq = 0;
  bool zf_ok = !spec_w;  // v64: zero rows are complete after the routing kernel (PDL) or the in-kernel grid barrier
  bool zf_signaled = !spec_w;  // v86: this CTA's zero rows published (zf_done) at its first epilogue / exit
  if (spec_w) spec_zero_fill(tid, NCT);
  auto zf_signal = [&]() {
    if (!zf_signaled) {
      __threadfence();
      named_bar_sync<1, NCT>();
      if (tid == 0) atomicAdd(fp.zf_done, 1);
      zf_signaled = true;
    }
  };
  for (;;) {
    mbar_wait(full_base + stage * 8, phase);  // first stage of the next unit (or the sentinel)
    const int su = s_unit[uq & 7];
    ++uq;
    if (su <= -100) {
      // v86: drain of speculative stages (data unused)
      const int n = -100 - su;
      for (int q = 0; q < n; ++q) {
        if (q > 0) mbar_wait(full_base + stage * 8, phase);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_base + stage * 8);
        if (stage + 1 == STAGES) { stage = 0; phase ^= 1; } else { ++stage; }
      }
      continue;
    }
    if (su < 0) {
      zf_signal();
      break;
    }
    const Sub d = decode_sub(su);
    const bool G1 = (d.phase == 1);
    const GemmParams& p = G1 ? fp.g1 : fp.g2;
    const int ASK = G1 ? 128 : 64;
    const int sw_idx = G1 ? (warp >= NCW / 2 ? 1 : 0) : 0;
    const UnitInfo cu = d.ui;
    for (int kb = 0; kb < d.KBS; ++kb) {
      if (kb > 0) mbar_wait(full_base + stage * 8, phase);
#ifndef MOE_ABL_F_NOMMA
      const uint8_t* st = smem + stage * C::STAGE_BYTES;
      const uint32_t sbase = smem_base + stage * C::STAGE_BYTES;
      const float* sa = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES);
      const float wsc = reinterpret_cast<const float*>(st + C::W_BYTES + C::A_BYTES + C::SA_BYTES)[sw_idx];
      uint32_t wf[2][4], tf[2][2 * NTOK];
      float part[2][NTOK][4];
      ldmatrix_x4(sbase + w_ld[0], wf[0][0], wf[0][1], wf[0][2], wf[0][3]);
      if (NTOK == 1) ldmatrix_x2(sbase + t_ld[0], tf[0][0], tf[0][1]);
      else ldmatrix_x4(sbase + t_ld[0], tf[0][0], tf[0][1], tf[0][2], tf[0][3]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int cur = ks & 1, nxt = cur ^ 1;
        if (ks < 3) {
          ldmatrix_x4(sbase + w_ld[ks + 1], wf[nxt][0], wf[nxt][1], wf[nxt][2], wf[nxt][3]);
          if (NTOK == 1) ldmatrix_x2(sbase + t_ld[ks + 1], tf[nxt][0], tf[nxt][1]);
          else ldmatrix_x4(sbase + t_ld[ks + 1], tf[nxt][0], tf[nxt][1], tf[nxt][2], tf[nxt][3]);
        }
        // window = ASK/32 k-steps; chains alternate; the first k-step of each chain in a window uses C=0
        const int kw = G1 ? ks : (ks & 1);
        const int ch = kw & 1;
        const bool first = kw < 2;
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          if (first) mma_fp8_zc(part[ch][j], wf[cur], tf[cur] + 2 * j);
          else mma_fp8(part[ch][j], wf[cur], tf[cur] + 2 * j);
        }
        const bool promote = G1 ? (ks == 3) : ((ks & 1) == 1);
        if (promote) {
          const int sub = G1 ? 0 : (ks >> 1);
#pragma unroll
          for (int j = 0; j < NTOK; ++j) {
            const float s0 = sa[sub * MT + j * 8 + 2 * t] * wsc;
            const float s1 = sa[sub * MT + j * 8 + 2 * t + 1] * wsc;
            const float p0 = part[0][j][0] + part[1][j][0], p1 = part[0][j][1] + part[1][j][1];
            const float p2 = part[0][j][2] + part[1][j][2], p3 = part[0][j][3] + part[1][j][3];
            acc[j][0] = fmaf(p0, s0, acc[j][0]);
            acc[j][1] = fmaf(p1, s1, acc[j][1]);
            acc[j][2] = fmaf(p2, s0, acc[j][2]);
            acc[j][3] = fmaf(p3, s1, acc[j][3]);
          }
        }
      }
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_base + stage * 8);
      if (stage + 1 == STAGES) {
        stage = 0;
        phase ^= 1;
      } else {
        ++stage;
      }
    }
    // ---- split-K fixup ----------------------------------------------------------------------
    bool do_epi = true;
    if (d.S > 1) {
      constexpr int PT = NTOK * 4;
      const int cidx = G1 ? d.u : (units1 + d.u);  // distinct counters for the two phases
      float* pp = p.partial + ((size_t)su * NCT + tid) * PT;
#pragma unroll
      for (int j = 0; j < NTOK; ++j)
        __stcg(reinterpret_cast<float4*>(pp + j * 4), make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]));
      __threadfence();
      named_bar_sync<1, NCT>();
      if (tid == 0) {
        const int old = atomicAdd(p.counters + cidx, 1);
        const int last = (old == d.S - 1) ? 1 : 0;
        if (last) p.counters[cidx] = 0;
        *s_flag = last;
      }
      named_bar_sync<1, NCT>();
      do_epi = (*s_flag != 0);
      if (do_epi) {
        __threadfence();
        const int sub_base = G1 ? (d.u * d.S) : (subs1 + d.u * d.S);
        for (int o = 0; o < d.S; ++o) {
          if (o == d.split) continue;
          const float* op = p.partial + ((size_t)(sub_base + o) * NCT + tid) * PT;
#pragma unroll
          for (int j = 0; j < NTOK; ++j) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(op + j * 4));
            acc[j][0] += v.x;
            acc[j][1] += v.y;
            acc[j][2] += v.z;
            acc[j][3] += v.w;
          }
        }
      }
    }
    // ---- epilogue ---------------------------------------------------------------------------
#if defined(MOE_ABL_F_NOEPI) || defined(MOE_ABL_F_G1NOEPI)
    if (do_epi && G1) { named_bar_sync<1, NCT>(); if (tid == 0) atomicAdd(fp.g1_done + cu.e, 1); }
    const bool do_epi_g1 = false;
#else
    const bool do_epi_g1 = do_epi;
#endif
#if defined(MOE_ABL_F_NOEPI) || defined(MOE_ABL_F_G2NOEPI)
    const bool do_epi_g2 = false;
#else
    const bool do_epi_g2 = do_epi;
#endif
    if (!(G1 ? do_epi_g1 : do_epi_g2)) {
    } else if (G1) {
      float* x2buf = reinterpret_cast<float*>(xbuf);
      float* amaxbuf = x2buf + MT * 64;
      uint8_t* qbuf = reinterpret_cast<uint8_t*>(amaxbuf + 4 * MT);
      const int c0 = (warp & 3) * 16 + g;
      if (warp >= 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          x2buf[tok * 64 + c0] = acc[j][0];
          x2buf[(tok + 1) * 64 + c0] = acc[j][1];
          x2buf[tok * 64 + c0 + 8] = acc[j][2];
          x2buf[(tok + 1) * 64 + c0 + 8] = acc[j][3];
        }
      }
      named_bar_sync<1, NCT>();
      float v[NTOK][4];
      if (warp < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          const float x2a = x2buf[tok * 64 + c0], x2b = x2buf[(tok + 1) * 64 + c0];
          const float x2c = x2buf[tok * 64 + c0 + 8], x2d = x2buf[(tok + 1) * 64 + c0 + 8];
          v[j][0] = silu_fast(x2a) * acc[j][0];
          v[j][1] = silu_fast(x2b) * acc[j][1];
          v[j][2] = silu_fast(x2c) * acc[j][2];
          v[j][3] = silu_fast(x2d) * acc[j][3];
          float m0 = fmaxf(fabsf(v[j][0]), fabsf(v[j][2]));
          float m1 = fmaxf(fabsf(v[j][1]), fabsf(v[j][3]));
#pragma unroll
          for (int off = 4; off <= 16; off <<= 1) {
            m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, off));
            m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, off));
          }
          if (g == 0) {
            amaxbuf[warp * MT + tok] = m0;
            amaxbuf[warp * MT + tok + 1] = m1;
          }
        }
      }
      named_bar_sync<1, NCT>();
      if (warp < 4) {
#pragma unroll
        for (int j = 0; j < NTOK; ++j) {
          const int tok = j * 8 + 2 * t;
          float m0 = amaxbuf[tok], m1 = amaxbuf[tok + 1];
#pragma unroll
          for (int w = 1; w < 4; ++w) {
            m0 = fmaxf(m0, amaxbuf[w * MT + tok]);
            m1 = fmaxf(m1, amaxbuf[w * MT + tok + 1]);
          }
          const float sc0 = (m0 > 0.f) ? m0 * (1.0f / 448.0f) : 1.0f;
          const float sc1 = (m1 > 0.f) ? m1 * (1.0f / 448.0f) : 1.0f;
          const float i0 = __frcp_rn(sc0), i1 = __frcp_rn(sc1);
          const uint16_t q0 = pack_e4m3x2(v[j][0] * i0, v[j][2] * i0);
          const uint16_t q1 = pack_e4m3x2(v[j][1] * i1, v[j][3] * i1);
          qbuf[tok * 64 + c0] = (uint8_t)(q0 & 0xff);
          qbuf[tok * 64 + c0 + 8] = (uint8_t)(q0 >> 8);
          qbuf[(tok + 1) * 64 + c0] = (uint8_t)(q1 & 0xff);
          qbuf[(tok + 1) * 64 + c0 + 8] = (uint8_t)(q1 >> 8);
          if (warp == 0 && g == 0) {
            if (tok < cu.rows_valid) p.act_scale[(size_t)cu.nt * p.p_cap + cu.row0 + tok] = sc0;
            if (tok + 1 < cu.rows_valid) p.act_scale[(size_t)cu.nt * p.p_cap + cu.row0 + tok + 1] = sc1;
          }
        }
      }
      named_bar_sync<1, NCT>();
      for (int i = tid; i < MT * 4; i += NCT) {
        const int tok = i >> 2, ch = i & 3;
        if (tok < cu.rows_valid) {
          const uint4 q = *reinterpret_cast<const uint4*>(qbuf + tok * 64 + ch * 16);
          *reinterpret_cast<uint4*>(p.act + (size_t)(cu.row0 + tok) * I + cu.nt * (NT / 2) + ch * 16) = q;
        }
      }
      // publish: this unit's activations are visible to GEMM2 units of the expert
      __threadfence();
      named_bar_sync<1, NCT>();
      if (tid == 0) atomicAdd(fp.g1_done + cu.e, 1);
      if (!zf_signaled) {
        if (tid == 0) atomicAdd(fp.zf_done, 1);
        zf_signaled = true;
      }
    } else {
      uint16_t* obuf = reinterpret_cast<uint16_t*>(xbuf);
      const int c0 = warp * 16 + g;
#pragma unroll
      for (int j = 0; j < NTOK; ++j) {
        const int tok = j * 8 + 2 * t;
        const float w0 = (tok < cu.rows_valid) ? p.pair_w[cu.row0 + tok] : 0.f;
        const float w1 = (tok + 1 < cu.rows_valid) ? p.pair_w[cu.row0 + tok + 1] : 0.f;
        obuf[tok * NT + c0] = __bfloat16_as_ushort(__float2bfloat16(acc[j][0] * w0));
        obuf[(tok + 1) * NT + c0] = __bfloat16_as_ushort(__float2bfloat16(acc[j][1] * w1));
        obuf[tok * NT + c0 + 8] = __bfloat16_as_ushort(__float2bfloat16(acc[j][2] * w0));
        obuf[(tok + 1) * NT + c0 + 8] = __bfloat16_as_ushort(__float2bfloat16(acc[j][3] * w1));
      }
      int* s_dst = reinterpret_cast<int*>(obuf + MT * NT);  // per row: token | (0x40000000 if red-added), -1 -> pair_out
      if (tid < MT) {
        int d = -1;
        if (tid < cu.rows_valid) {
          const int tk = p.pair_token[cu.row0 + tid];
          const int np = p.token_npair[tk];
          if (np == 1) d = tk;
          else if (p.combine_atomic) d = tk | 0x40000000;
        }
        s_dst[tid] = d;
      }
      named_bar_sync<1, NCT>();
      zf_signal();
      if (p.combine_atomic && !zf_ok) {
        if (tid == 0) {
          while (ld_acquire_gpu(fp.zf_done) < (int)gridDim.x) __nanosleep(100);
        }
        named_bar_sync<1, NCT>();
        zf_ok = true;
      }
      for (int i = tid; i < MT * 16; i += NCT) {
        const int tok = i >> 4, ch = i & 15;
        if (tok < cu.rows_valid) {
          const uint4 q = *reinterpret_cast<const uint4*>(obuf + tok * NT + ch * 8);
          const int d = s_dst[tok];
          if (d >= 0) {
            uint16_t* dst = p.out + (size_t)(d & 0x3fffffff) * H + cu.nt * NT + ch * 8;
            if (d & 0x40000000) red_v4_bf16x2(dst, q);
            else *reinterpret_cast<uint4*>(dst) = q;
          } else {
            *reinterpret_cast<uint4*>(p.pair_out + (size_t)(cu.row0 + tok) * H + cu.nt * NT + ch * 8) = q;
          }
        }
      }
      named_bar_sync<1, NCT>();
    }
#pragma unroll
    for (int j = 0; j < NTOK; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[j][r] = 0.f;
  }
  if (tid == 0) TRACE(fp.g1, 7);
  __threadfence();
  named_bar_sync<1, NCT>();
  if (tid == 0) cta_finish_last();
}

// ----------------------------------------------------------------------------------------------
// Large-M grouped GEMM on 5th-generation tensor cores (tcgen05 / UMMA, TMEM accumulators).
// CTA tile 128(M) x 128(N) x 128(K); warp 0 = TMA producer, warp 1 = MMA issuer, warp 2 = TMEM
// allocator, warps 4..7 = promotion + epilogue (thread <-> TMEM lane <-> output row).
// Both operands are K-major fp8 tensors loaded by TMA with 128B swizzle (UMMA canonical layout).
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(dst_smem), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(ncols));
}
#ifdef MOE_ABL_NOFENCE
__device__ __forceinline__ void tc_fence_after() {}
__device__ __forceinline__ void tc_fence_before() {}
#else
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
#endif
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void umma_f8f6f4(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                            uint32_t accum) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_ld_32x32b_x64(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(taddr));
}
// Packed fp32x2 FMA (sm_100): d = v * s + d on two floats held in one 64-bit register pair.
__device__ __forceinline__ uint64_t pack_f32x2(float lo, float hi) {
  return ((uint64_t)__float_as_uint(hi) << 32) | (uint64_t)__float_as_uint(lo);
}
__device__ __forceinline__ void ffma2(uint64_t& d, uint64_t v, uint64_t s) {
  asm("fma.rn.f32x2 %0, %1, %2, %0;\n" : "+l"(d) : "l"(v), "l"(s));
}
// acc2[i] (cols 2i, 2i+1) += v[2i..2i+1] * s2 for a 32-column chunk
__device__ __forceinline__ void fma2_chunk32(uint64_t* acc2, const uint32_t* v, uint64_t s2) {
#pragma unroll
  for (int i = 0; i < 16; ++i) ffma2(acc2[i], ((uint64_t)v[2 * i + 1] << 32) | (uint64_t)v[2 * i], s2);
}

// acc2[i] (cols 2i, 2i+1) += unpack_f16x2(v[i]) * s2 for a 64-column chunk held as 32 packed words
__device__ __forceinline__ void fma2_chunk32h(uint64_t* acc2, const uint32_t* v, uint64_t s2) {
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float lo, hi;
    asm("{\n.reg .b16 l, u;\nmov.b32 {l, u}, %2;\ncvt.f32.f16 %0, l;\ncvt.f32.f16 %1, u;\n}\n"
        : "=f"(lo), "=f"(hi) : "r"(v[i]));
    ffma2(acc2[i], pack_f32x2(lo, hi), s2);
  }
}

// K-major SWIZZLE_128B smem matrix descriptor (version 1, SBO = 1024 B between 8-row atoms)
__device__ __forceinline__ uint64_t umma_desc(uint32_t smem_addr) {
  uint64_t d = (uint64_t)((smem_addr >> 4) & 0x3fff);
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// arrive on the same-offset mbarrier of CTA `cta` of the cluster
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t bar, uint32_t cta) {
  asm volatile(
      "{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n" ::"r"(bar), "r"(cta) : "memory");
}
// 2-SM TMA load: issued by both CTAs of the pair; the mbarrier address has the peer bit cleared so that
// the transaction bytes of both CTAs count on CTA 0's barrier
__device__ __forceinline__ void tma_load_2d_2sm_hint(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0,
                                                     int c1, uint64_t pol) {
  const uint32_t bar0 = bar & 0xFEFFFFFFu;
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1, {%3, %4}], [%2], %5;\n" ::"r"(dst),
      "l"(map), "r"(bar0), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tmem_alloc2(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(dst_smem), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish2() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::);
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ void umma_commit2_mc(uint32_t bar, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n" ::"r"(bar),
      "h"(mask)
      : "memory");
}
__device__ __forceinline__ void umma2_f8f6f4(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                             uint32_t accum) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accum));
}

constexpr int U_BM = 128, U_BN = 256, U_BK = 128, U_STAGES = 4;
#ifdef MOE_ACC_WARP_RELEASE
constexpr bool g_acc_warp_release = true;  // v72: measured neutral (14107 971.8 vs 972.1 us) -> off
#else
constexpr bool g_acc_warp_release = false;
#endif
#ifdef MOE_ACC_F16
// v57 (rejected): fp16 partial accumulators. Measured on B200 (scripts/umma_test_f16.cu RAW mode): a 16-bit
// accumulator still occupies one 32-bit TMEM column per element (value in the low half), so the TMEM
// footprint does not shrink and this path's packed-column addressing is wrong. Kept for reference only.
constexpr bool U_F16 = true;
#else
constexpr bool U_F16 = false;
#endif
constexpr int U_NBUF = U_F16 ? 4 : 2;            // TMEM partial-accumulator buffers (512 columns total)
constexpr int U_ACC_COLS = U_F16 ? U_BN / 2 : U_BN;  // TMEM columns per 128x256 partial accumulator
constexpr int U2_NBUF = 2;  // 2-SM kernel: fp32 accumulators, 2 x 256 columns
constexpr int U_STAGE_BYTES = U_BM * U_BK + U_BN * U_BK;  // 48 KB
constexpr int U_MISC_BYTES = 1024;  // offsets/prefix (264 B), barriers, tmem ptr; s_end at 384 (1-SM) / 512 (2-SM), s_pp at 768
constexpr int U_XCH_BYTES = U_BM * (U_BN / 2) * 2;  // SwiGLU exchange [128][128] bf16 (32 KB)
constexpr int U_AMAX_BYTES = U_BM * 2 * 4;  // per-row amax of each column half (1 KB)
constexpr int U_TOK_BYTES = U_BM * 4;  // v84: token row of every tile row for the gather4 producer (lives in the alignment slack)
constexpr int U_SMEM_BYTES = 1024 + U_STAGES * U_STAGE_BYTES + U_MISC_BYTES + U_XCH_BYTES + U_AMAX_BYTES;  // = 227 KB
// 2-SM variant: per CTA a stage holds its 128 A rows + its 128-row N-half of B (32 KB), 6 stages
constexpr int U2_STAGES = 6;
constexpr int U2_STAGE_BYTES = U_BM * U_BK + (U_BN / 2) * U_BK;
constexpr int U2_SMEM_BYTES = 1024 + U2_STAGES * U2_STAGE_BYTES + U_MISC_BYTES + U_XCH_BYTES + U_AMAX_BYTES;
constexpr int U2_MIN_T = 2048;  // experts have several m-tiles -> pairs; 901 stays on the 1-SM kernel

// 2-SM unit: (expert, m-tile pair, n-tile); CTA `rank` of the pair owns m-tile 2*pair + rank
__device__ __forceinline__ UnitInfo decode_unit2(int u, const int* s_off, const int* s_pp, int n_tiles,
                                                 const int* s_end, uint32_t rank) {
  UnitInfo ui;
  const int pg = u / n_tiles;
  ui.nt = u - pg * n_tiles;
  int e = 0;
#pragma unroll 4
  for (int x = 1; x <= NLOCAL; ++x) e += (s_pp[x] <= pg) ? 1 : 0;
  ui.e = e;
  const int pi = pg - s_pp[e];
  ui.row0 = s_off[e] + (2 * pi + (int)rank) * U_BM;
  const int rv = s_end[e] - ui.row0;
  ui.rows_valid = rv < 0 ? 0 : (rv > U_BM ? U_BM : rv);
  return ui;
}
constexpr int U_THREADS = 384;
constexpr int U_NT1 = I / (U_BN / 2);  // 16 GEMM1 n-tiles (128 x1 + 128 x2 weight rows each)
constexpr int U_NT2 = H / U_BN;        // 28 GEMM2 n-tiles

__device__ __forceinline__ void setmaxnreg_dec40() { asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::); }
__device__ __forceinline__ void setmaxnreg_inc232() { asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::); }
// v84: the gather4 producer warp keeps 32 lanes of state (4 token rows each) -> 64 registers for warps 0..3 and
// 216 for the 8 promotion warps (the increment must fit the registers freed by the decrement; compiled for 168)
__device__ __forceinline__ void setmaxnreg_dec64() { asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::); }
__device__ __forceinline__ void setmaxnreg_inc224() { asm volatile("setmaxnreg.inc.sync.aligned.u32 216;\n" ::); }  // 256 x (216-168) = 12288 <= 128 x (168-64)

// v72: acc_empty release with one arrive per warp (the fence::before_thread_sync of every thread precedes __syncwarp)
__device__ __forceinline__ void acc_release1(uint32_t bar) {
  if (g_acc_warp_release) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) mbar_arrive(bar);
  } else {
    mbar_arrive(bar);
  }
}
template <int ASK, bool GEMM1>
__global__ void __launch_bounds__(U_THREADS, 1)
    umma_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                     const __grid_constant__ CUtensorMap tmA64, const __grid_constant__ CUtensorMap tmA32,
                     const GemmParams p) {
  static_assert(ASK == 128, "one promotion per 128-K stage");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw & 1023)) & 1023;
  const uint32_t sbase = raw + pad;
  int* s_off = reinterpret_cast<int*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full = sbase + U_STAGES * U_STAGE_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty = full + U_STAGES * 8;
  const uint32_t acc_full = empty + U_STAGES * 8;
  const uint32_t acc_empty = acc_full + U_NBUF * 8;
  uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES + 2 * (NLOCAL + 1) * 4 + (2 * U_STAGES + 2 * U_NBUF) * 8);
  int* s_end = reinterpret_cast<int*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES + 512);  // [NLOCAL] (barriers end at 396 with 4 buffers)
  volatile int* s_prog = reinterpret_cast<volatile int*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES + 1000);  // v74: K-blocks consumed by the MMA issuer
  uint16_t* xch = reinterpret_cast<uint16_t*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES + U_MISC_BYTES);  // [128][128] bf16
  float* amaxx = reinterpret_cast<float*>(smem_raw + pad + U_STAGES * U_STAGE_BYTES + U_MISC_BYTES + U_XCH_BYTES);  // [128][2]
  int* s_tok = reinterpret_cast<int*>(smem_raw + U_SMEM_BYTES - U_TOK_BYTES);  // [128] v84: the 1024-B alignment slack (pad is 0: extern smem is 1024-aligned)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < U_STAGES; ++s) {
      mbar_init(full + s * 8, (GEMM1 && p.a_gather == 1) ? 1 + 64 : 1);  // v62: + one cp.async arrival per producer thread (warps 0, 3)
      mbar_init(empty + s * 8, 1);
    }
#pragma unroll
    for (int b = 0; b < U_NBUF; ++b) {
      mbar_init(acc_full + b * 8, 1);
      mbar_init(acc_empty + b * 8, p.acc_warp_release ? 8 : 256);  // v72: one arrive per warp
    }
    fence_barrier_init();
    tma_prefetch_desc(&tmA);
    tma_prefetch_desc(&tmB);
  }
  if (warp == 2) {
    tmem_alloc(smem_u32(tmem_ptr), U_NBUF * U_ACC_COLS);
    tmem_relinquish();
  }
  if (tid == 0) *s_prog = 0;
  if (tid == 0) TRACE(p, 0);
  pdl_wait();
  if (tid == 0) TRACE(p, 1);
  pdl_launch_dependents();
  load_expert_tables<U_BM>(p, s_off, s_mtp, s_end);
  __syncthreads();
  const uint32_t tmem_base = *tmem_ptr;
  const int mtile_total = s_mtp[NLOCAL];
  const int total_units = mtile_total * p.n_tiles;
  const int my_units = ((int)blockIdx.x < total_units) ? (total_units - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  const int KB = p.K / 128;

  if (warp < 4) {
    setmaxnreg_dec40();
    if (GEMM1 && p.a_gather == 2 && warp == 0) {
      // ================= v84: producer warp 0 (32 lanes): TMA weights + TMA gather4 of the A rows ==========
      // lane i gathers rows 4i..4i+3 of the m-tile straight from hidden_states (token rows), 512 B per lane per
      // K-block into the 128B-swizzled canonical layout; no scatter copy of the activations is needed
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      const uint64_t pol_a = l2_policy_evict_last();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        const int b_row0 = u.e * p.N_total + u.nt * (U_BN / 2);
        const int b_row1 = b_row0 + I;
        // the token row of every tile row (invalid rows: any valid token) -> smem; the TMA coordinates must be
        // warp-uniform (R2UR), so lane 0 alone issues the 32 gather4 loads of a K-block from this table
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int r = lane * 4 + j;
          s_tok[r] = (r < u.rows_valid) ? __ldg(p.pair_token + u.row0 + r) : 0;
        }
        __syncwarp();
        if (lane == 0) {
          for (int kb = 0; kb < KB; ++kb) {
            const uint32_t st = sbase + stage * U_STAGE_BYTES;
            const uint32_t fb = full + stage * 8;
            mbar_wait(empty + stage * 8, phase);
            mbar_arrive_expect_tx(fb, U_BM * U_BK + U_BN * U_BK);
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
            tma_load_2d_hint(st + U_BM * U_BK + (U_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
#pragma unroll 4
            for (int g = 0; g < U_BM / 4; ++g) {
              const int4 t = *reinterpret_cast<const int4*>(s_tok + 4 * g);
              tma_gather4_hint(st + g * 512, &tmA, fb, kb * U_BK, t.x, t.y, t.z, t.w, pol_a);
            }
            if (++stage == U_STAGES) {
              stage = 0;
              phase ^= 1;
            }
          }
        }
        __syncwarp();
        stage = __shfl_sync(0xffffffffu, stage, 0);
        phase = __shfl_sync(0xffffffffu, phase, 0);
      }
    } else if (GEMM1 && p.a_gather == 1 && warp != 1 && warp != 2) {
      // ================= v62: producer warps 0 and 3 (64 threads): TMA weights + cp.async A rows ==========
      // thread t (0..63) gathers rows t and t+64 of the m-tile: 8 x 16 B per row per K-block, 128B-swizzled
      // (16-B chunk c of row r at r*128 + ((c ^ (r & 7)) << 4)), the TMA SWIZZLE_128B / UMMA canonical layout.
      const int pt = (warp == 0) ? lane : 32 + lane;  // 0..63
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        const int b_row0 = u.e * p.N_total + u.nt * (U_BN / 2);
        const int b_row1 = b_row0 + I;
        const uint8_t* src[2];
        bool valid[2];
        uint32_t dst[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = pt + 64 * h;
          valid[h] = r < u.rows_valid;
          const int tok = valid[h] ? p.pair_token[u.row0 + r] : 0;
          src[h] = p.A + (size_t)tok * p.K;
          dst[h] = r * 128;
        }
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(empty + stage * 8, phase); UT_END(p, 12, true); }
          const uint32_t st = sbase + stage * U_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
          if (pt == 0) {
            mbar_arrive_expect_tx(fb, U_BN * U_BK);
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
            tma_load_2d_hint(st + U_BM * U_BK + (U_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
          }
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (valid[h]) {
              const uint8_t* srow = src[h] + (size_t)kb * 128;
              const int r = pt + 64 * h;
#pragma unroll
              for (int c = 0; c < 8; ++c) cp_async_16(st + dst[h] + ((c ^ (r & 7)) << 4), srow + c * 16, true);
            }
          }
          cp_async_mbar_arrive_noinc(fb);
          if (++stage == U_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    } else if (warp == 0 && lane == 0) {
      // =========================== TMA producer =============================================
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      const uint64_t pol_a = l2_policy_evict_last();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        int b_row0, b_row1 = 0;
#ifdef MOE_ABL_BHOT
        const int b_e = 0;  // ablation: every unit reads expert 0's weights (L2-resident feed)
#else
        const int b_e = u.e;
#endif
        if (GEMM1) {
          b_row0 = b_e * p.N_total + u.nt * (U_BN / 2);
          b_row1 = b_row0 + I;
        } else {
          b_row0 = b_e * p.N_total + u.nt * U_BN;
        }
        // v69: A box sized to the valid rows; rows >= rows_valid hold stale smem whose results are discarded
        const CUtensorMap* ta = &tmA;
        uint32_t a_bytes = U_BM * U_BK;
        if (p.a_box) {
          if (u.rows_valid <= 32) { ta = &tmA32; a_bytes = 32 * U_BK; }
          else if (u.rows_valid <= 64) { ta = &tmA64; a_bytes = 64 * U_BK; }
        }
#ifdef MOE_ABL_AHOT
        const int a_row0 = 0;  // ablation: every unit reads activation rows [0,128) (L2-resident feed)
#else
        const int a_row0 = u.row0;
#endif
#ifdef MOE_U_PF
        // v74 experiment: B K-blocks prefetched into L2 MOE_U_PF K-blocks ahead of the TMA ring
        for (int q = 0; q < MOE_U_PF && q < KB; ++q) {
          tma_prefetch_2d_hint(&tmB, q * U_BK, b_row0, pol_w);
          if (GEMM1) tma_prefetch_2d_hint(&tmB, q * U_BK, b_row1, pol_w);
        }
#endif
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(empty + stage * 8, phase); UT_END(p, 12, true); }
          const uint32_t st = sbase + stage * U_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
#ifdef MOE_U_PF
          if (kb + MOE_U_PF < KB) {
            tma_prefetch_2d_hint(&tmB, (kb + MOE_U_PF) * U_BK, b_row0, pol_w);
            if (GEMM1) tma_prefetch_2d_hint(&tmB, (kb + MOE_U_PF) * U_BK, b_row1, pol_w);
          }
#endif
          mbar_arrive_expect_tx(fb, a_bytes + U_BN * U_BK);
          tma_load_2d_hint(st, ta, fb, kb * U_BK, a_row0, pol_a);
          if (GEMM1) {
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
            tma_load_2d_hint(st + U_BM * U_BK + (U_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
          } else {
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
          }
          if (++stage == U_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    } else if (warp == 1 && lane == 0) {
      // =========================== MMA issuer ===============================================
      const uint32_t idesc = ((U_F16 ? 0u : 1u) << 4) | ((U_BN >> 3) << 17) | ((U_BM >> 4) << 24);  // F16/F32 += E4M3 * E4M3, K-major
      int stage = 0;
      uint32_t phase = 0;
      uint32_t kctr = 0;  // global K-block counter (TMEM buffer = kctr % U_NBUF)
      for (int ui = 0; ui < my_units; ++ui) {
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(full + stage * 8, phase); UT_END(p, 8, true); }
#ifdef MOE_U_PFW
          *s_prog = (int)kctr + 1;
#endif
          tc_fence_after();
          const uint32_t st = sbase + stage * U_STAGE_BYTES;
          const uint64_t adesc = umma_desc(st), bdesc = umma_desc(st + U_BM * U_BK);
          const int buf = kctr % U_NBUF;
          { UT_BEGIN(); mbar_wait(acc_empty + buf * 8, ((kctr / U_NBUF) & 1) ^ 1); UT_END(p, 9, true); }
#ifdef MOE_TRACE
          if (p.trace && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 13] += 1;
#endif
          tc_fence_after();
          const uint32_t d_tmem = tmem_base + buf * U_ACC_COLS;
#ifndef MOE_ABL_NOMMA
#pragma unroll
          for (int k = 0; k < U_BK / 32; ++k) {
            const uint32_t koff = (uint32_t)(k * 32) >> 4;
            umma_f8f6f4(d_tmem, adesc + koff, bdesc + koff, idesc, k > 0 ? 1u : 0u);
          }
#endif
          umma_commit(acc_full + buf * 8);
          umma_commit(empty + stage * 8);
          ++kctr;
          if (++stage == U_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
    else if (warp == 3 && GEMM1 && p.zero_rows) {
      // v80: output rows without a local expert, and multi-pair rows that receive red-adds, are zeroed here
      // (overlapped with the GEMM) instead of in the scatter kernel ahead of it
      for (int t = (int)blockIdx.x; t < p.T; t += (int)gridDim.x) {
        const int np = __ldg(p.token_npair + t);
        if (np == 0 || (p.combine_atomic && np >= 2)) {
          uint4* orow = reinterpret_cast<uint4*>(p.out + (size_t)t * H);
#pragma unroll 4
          for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
#ifdef MOE_U_PFW
    else if (warp == 3) {
      // v74: warp 3 touches the B weight lines of K-block g + MOE_U_PFW (LSU prefetch / load) so that the TMA ring
      // finds them in L2; paced by the ring's full barriers (the TMA L2 prefetch instruction measured slower)
      const uint8_t* Wb = p.W;
      int cu = 0, ck = 0, cb0 = 0, cb1 = 0;
      uint32_t sink = 0;
      auto load_unit = [&](int ui) {
        const UnitInfo uu = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        if (GEMM1) { cb0 = uu.e * p.N_total + uu.nt * (U_BN / 2); cb1 = cb0 + I; }
        else { cb0 = uu.e * p.N_total + uu.nt * U_BN; cb1 = cb0 + (U_BN / 2); }
      };
      auto touch = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int rr = lane + 32 * i;
          const uint8_t* a0 = Wb + (size_t)(cb0 + rr) * p.K + (size_t)ck * U_BK;
          const uint8_t* a1 = Wb + (size_t)(cb1 + rr) * p.K + (size_t)ck * U_BK;
#ifdef MOE_U_PF_LD
          sink += ld_cg_u32(a0) + ld_cg_u32(a1);
#else
          pf_l2(a0);
          pf_l2(a1);
#endif
        }
        if (++ck == KB) { ck = 0; ++cu; if (cu < my_units) load_unit(cu); }
      };
      if (my_units > 0) {
        load_unit(0);
        const int total = my_units * KB;
        int issued = 0;
        for (; issued < MOE_U_PFW && issued < total; ++issued) touch();
        for (int g = 0; g < total; ++g) {
          if (lane == 0) {
            while (*s_prog < g + 1) __nanosleep(100);  // v74c: paced by the MMA issuer's progress (no barrier aliasing)
          }
          __syncwarp();
          if (issued < total) { touch(); ++issued; }
        }
      }
      if (sink == 0xFFFFFFFFu) p.counters[4095] = sink;  // keep the loads alive
    }
#endif
  } else {
    // =========================== promotion + epilogue warpgroups ==========================
    // warps 4..7 own columns [0,128) of every row (x1 / first weight block), warps 8..11 own [128,256).
    setmaxnreg_inc232();
    constexpr int HC = U_BN / 2;  // 128 columns per thread
    const int half = (warp >= 8) ? 1 : 0;
    const int r = (warp & 3) * 32 + lane;  // row within the tile == TMEM lane
    const uint32_t lane_base = ((uint32_t)((warp & 3) * 32)) << 16;
    const uint32_t col_base = U_F16 ? half * (HC / 2) : half * HC;  // TMEM column of this thread's first element
    uint64_t acc2[HC / 2];  // packed fp32 pairs (cols 2i, 2i+1)
    float acc[HC];
    uint32_t kctr = 0;
    // v77: the scale pointers of a unit (A scale of row r, weight scale of this thread's 128-column half)
    auto scale_ptrs = [&](const UnitInfo& uu, const float*& sa, const float*& swp) {
      const int ar = uu.row0 + r;
      sa = p.A_scale + ((GEMM1 && p.a_gather) ? ((r < uu.rows_valid) ? p.pair_token[ar] : 0) : ar);
      int nbx;
      if (GEMM1) nbx = (half == 0) ? uu.nt : (I / 128 + uu.nt);
      else nbx = uu.nt * (U_BN / 128) + half;
      swp = p.W_scale + ((size_t)uu.e * (p.N_total / 128) + nbx) * KB;
    };
#ifdef MOE_NEXT_PF
    float nw0 = 0.f, ns0 = 0.f, nw1 = 0.f, ns1 = 0.f, nw2 = 0.f, ns2 = 0.f, nw3 = 0.f, ns3 = 0.f;  // next unit's first scales
    float n_w2 = 0.f, c_w2 = 0.f;  // GEMM2 row metadata of the next / current unit
    int n_tk = 0, n_np = 0, n_rv = 0, c_tk = 0, c_np = 0;
#endif
    for (int ui = 0; ui < my_units; ++ui) {
      const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
      const float* sa_ptr;
      const float* sw;
      scale_ptrs(u, sa_ptr, sw);
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) acc2[i] = 0ull;
      // block-scale prefetch: 4 K-blocks ahead in rotating scalars (no unrolling, no spills)
      float w0, s0, w1 = 0.f, s1 = 0.f, w2 = 0.f, s2 = 0.f, w3 = 0.f, s3 = 0.f;
#ifdef MOE_NEXT_PF
      if (ui > 0 && KB >= 4) {
        w0 = nw0; s0 = ns0; w1 = nw1; s1 = ns1; w2 = nw2; s2 = ns2; w3 = nw3; s3 = ns3;
        c_w2 = n_w2; c_tk = n_tk; c_np = n_np;
      } else
#endif
      {
        w0 = __ldg(sw); s0 = __ldg(sa_ptr);
        if (KB > 1) { w1 = __ldg(sw + 1); s1 = __ldg(sa_ptr + (size_t)1 * p.a_rows_stride); }
        if (KB > 2) { w2 = __ldg(sw + 2); s2 = __ldg(sa_ptr + (size_t)2 * p.a_rows_stride); }
        if (KB > 3) { w3 = __ldg(sw + 3); s3 = __ldg(sa_ptr + (size_t)3 * p.a_rows_stride); }
      }
      uint32_t va[32], vb[32];
      {
        // first K-block of the unit: chunks 0,1 in flight before the loop
        const int buf = kctr % U_NBUF;
        { UT_BEGIN(); mbar_wait(acc_full + buf * 8, (kctr / U_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
        tc_fence_after();
        const uint32_t tb = tmem_base + lane_base + buf * U_ACC_COLS + col_base;
        tmem_ld_32x32b_x32(tb, va);
        tmem_ld_32x32b_x32(tb + 32, vb);
      }
      if (ui == 0 && warp == 4 && lane == 0) TRACE(p, 3);
      for (int kb = 0; kb < KB; ++kb) {
#ifdef MOE_ABL_NOEPI
        {
          const int bufx = kctr % U_NBUF;
          if (kb > 0) mbar_wait(acc_full + bufx * 8, (kctr / U_NBUF) & 1);
          tmem_wait_ld();
          tc_fence_after();
          tc_fence_before();
          acc_release1(acc_empty + bufx * 8);
          ++kctr;
          continue;
        }
#endif
#ifdef MOE_ABL_NOSCALE
        const float sc = 1.0f;
#else
        const float sc = s0 * w0;
#endif
        s0 = s1; w0 = w1; s1 = s2; w1 = w2; s2 = s3; w2 = w3;
        if (kb + 4 < KB) {
          w3 = __ldg(sw + kb + 4);
          s3 = __ldg(sa_ptr + (size_t)(kb + 4) * p.a_rows_stride);
        }
#ifdef MOE_NEXT_PF
        else if (kb == KB - 4 && ui + 1 < my_units) {
          // v77: the next unit's first four scales (and GEMM2 row metadata) are fetched while this unit finishes
          const UnitInfo un = decode_unit<U_BM, true>(blockIdx.x + (ui + 1) * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
          const float* san;
          const float* swn;
          scale_ptrs(un, san, swn);
          nw0 = __ldg(swn); ns0 = __ldg(san);
          nw1 = __ldg(swn + 1); ns1 = __ldg(san + (size_t)1 * p.a_rows_stride);
          nw2 = __ldg(swn + 2); ns2 = __ldg(san + (size_t)2 * p.a_rows_stride);
          nw3 = __ldg(swn + 3); ns3 = __ldg(san + (size_t)3 * p.a_rows_stride);
          if (!GEMM1) {
            n_rv = (r < un.rows_valid) ? 1 : 0;
            const int pn = un.row0 + r;
            n_w2 = n_rv ? p.pair_w[pn] : 0.f;
            n_tk = n_rv ? p.pair_token[pn] : 0;
          }
        } else if (kb == KB - 2 && ui + 1 < my_units && !GEMM1) {
          n_np = n_rv ? p.token_npair[n_tk] : 0;
        }
#endif
        uint64_t s2;
        {
          UT_BEGIN();
          s2 = pack_f32x2(sc, sc);
#ifdef MOE_TRACE
          asm volatile("" : "+l"(s2));  // force the scale dependency inside this window
#endif
          UT_END(p, 15, warp == 4 && lane == 0);
        }
        const int buf = kctr % U_NBUF;
        const uint32_t tb = tmem_base + lane_base + buf * U_ACC_COLS + col_base;
        UT_BEGIN();
        if constexpr (U_F16) {
          // v57: the whole 128-column half is already in va/vb (64 packed f16x2 words)
          tmem_wait_ld();
          fma2_chunk32h(acc2 + 0, va, s2);
          fma2_chunk32h(acc2 + 32, vb, s2);
          UT_END(p, 11, warp == 4 && lane == 0);
          tc_fence_before();
          acc_release1(acc_empty + buf * 8);
          if (kb + 1 < KB) {
            const int nbuf = (kctr + 1) % U_NBUF;
            { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
            tc_fence_after();
            const uint32_t tbn = tmem_base + lane_base + nbuf * U_ACC_COLS + col_base;
            tmem_ld_32x32b_x32(tbn, va);
            tmem_ld_32x32b_x32(tbn + 32, vb);
          }
        } else {
#ifdef MOE_ABL_HALFREAD
        // ablation: read only chunks 0,1 (half the TMEM bytes), emulating fp16 accumulators
        tmem_wait_ld();
        fma2_chunk32(acc2 + 0, va, s2);
        fma2_chunk32(acc2 + 16, vb, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          const int nbuf = (kctr + 1) % U_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * U_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
        tc_fence_before();
        acc_release1(acc_empty + buf * 8);
#elif defined(MOE_LATE_RELEASE)
        tmem_wait_ld();                  // chunks 0,1 of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        tmem_ld_32x32b_x32(tb + 64, va);
        fma2_chunk32(acc2 + 16, vb, s2);
        tmem_ld_32x32b_x32(tb + 96, vb);
        tmem_wait_ld();                  // chunks 2,3
        fma2_chunk32(acc2 + 32, va, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          // prefetch chunks 0,1 of the next K-block (other TMEM buffer) behind the remaining FMAs
          const int nbuf = (kctr + 1) % U_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * U_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
        }
        fma2_chunk32(acc2 + 48, vb, s2);
        if (more) tmem_ld_32x32b_x32(tbn + 32, vb);
        {
          UT_BEGIN();
          tc_fence_before();
          acc_release1(acc_empty + buf * 8);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
#else
        // v55: finish this K-block and release its TMEM buffer *before* waiting for the next MMA, so
        // the issuer can enqueue K-block kb+2 while kb+1 is still executing (no tensor-pipe bubble)
        tmem_wait_ld();                  // chunks 0,1 of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        tmem_ld_32x32b_x32(tb + 64, va);
        fma2_chunk32(acc2 + 16, vb, s2);
        tmem_ld_32x32b_x32(tb + 96, vb);
        tmem_wait_ld();                  // chunks 2,3
#ifdef MOE_EARLY_RELEASE
        // v79: every chunk of this buffer is in registers -> hand the buffer back before the remaining FMAs
        {
          UT_BEGIN();
          tc_fence_before();
          acc_release1(acc_empty + buf * 8);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
        fma2_chunk32(acc2 + 32, va, s2);
        fma2_chunk32(acc2 + 48, vb, s2);
        UT_END(p, 11, warp == 4 && lane == 0);
#else
        fma2_chunk32(acc2 + 32, va, s2);
        fma2_chunk32(acc2 + 48, vb, s2);
        UT_END(p, 11, warp == 4 && lane == 0);
        {
          UT_BEGIN();
          tc_fence_before();
          acc_release1(acc_empty + buf * 8);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
#endif
        const bool more = (kb + 1 < KB);
        if (more) {
          const int nbuf = (kctr + 1) % U_NBUF;
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          const uint32_t tbn = tmem_base + lane_base + nbuf * U_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
#endif
        }  // !U_F16
        ++kctr;
      }
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) {
        acc[2 * i] = __uint_as_float((uint32_t)acc2[i]);
        acc[2 * i + 1] = __uint_as_float((uint32_t)(acc2[i] >> 32));
      }
      // ---- epilogue for row r ------------------------------------------------------------
      const int pair = u.row0 + r;
#ifdef MOE_ABL_G1NOEPI
      if (GEMM1 && p.n_tiles != 12345) continue;  // ablation: no GEMM1 epilogue (acc stays live through the guard)
#endif
#ifdef MOE_ABL_G2NOEPI
      if (!GEMM1 && p.n_tiles != 12345) continue;
#endif
      if constexpr (GEMM1) {
        // export: half 0 -> x1[64..127], half 1 -> x2[0..63] (bf16, 16 B chunks XOR-swizzled by row)
        {
          // v61: literal register bases per half (a runtime base would move `acc` to local memory)
          auto export8 = [&](int c, float v0, float v1, float v2_, float v3, float v4, float v5, float v6, float v7) {
            uint4 pk;
            pk.x = pack_bf16x2(v0, v1);
            pk.y = pack_bf16x2(v2_, v3);
            pk.z = pack_bf16x2(v4, v5);
            pk.w = pack_bf16x2(v6, v7);
            const int ch = (half << 3) + (c >> 3);  // half h writes chunks [8h, 8h+8) of the 16-chunk row
            *reinterpret_cast<uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3)) = pk;
          };
          if (half == 0) {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[64 + c], acc[64 + c + 1], acc[64 + c + 2], acc[64 + c + 3], acc[64 + c + 4], acc[64 + c + 5], acc[64 + c + 6], acc[64 + c + 7]);
          } else {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[c], acc[c + 1], acc[c + 2], acc[c + 3], acc[c + 4], acc[c + 5], acc[c + 6], acc[c + 7]);
          }
        }
        named_bar_sync<1, 256>();
        // half 0 quantizes columns [0,64): v = x1[0..63] (own) * silu(x2[0..63]) (from half 1)
        // half 1 quantizes columns [64,128): v = x1[64..127] (from half 0) * silu(x2[64..127]) (own)
        // v61: SwiGLU values are computed twice (amax pass, quantization pass) instead of being held in a
        // 64-float array; each half indexes `acc` with literal bases
        const int ch0 = (half == 0) ? 8 : 0;   // chunks written by the other half
        auto swiglu8 = [&](int c, float* a) {
          const int ch = ch0 + (c >> 3);
          const uint4 pk = *reinterpret_cast<const uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3));
          const uint32_t w4[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float o0 = bf16_to_f32((uint16_t)(w4[q] & 0xffff));
            const float o1 = bf16_to_f32((uint16_t)(w4[q] >> 16));
            if (half == 0) {  // own = x1 (cols 0..63), other = x2
              a[2 * q] = acc[c + 2 * q] * silu_fast(o0);
              a[2 * q + 1] = acc[c + 2 * q + 1] * silu_fast(o1);
            } else {          // own = x2 (cols 64..127), other = x1
              a[2 * q] = o0 * silu_fast(acc[64 + c + 2 * q]);
              a[2 * q + 1] = o1 * silu_fast(acc[64 + c + 2 * q + 1]);
            }
          }
        };
        float amax = 0.f;
#ifndef MOE_SWIGLU_TWICE
        // v90: SwiGLU computed once (64 values stay in the registers freed by the export); default (14107 955 -> 942 us)
        float sw[64];
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          swiglu8(c, sw + c);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(sw[c + q]));
        }
#else
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          float a[8];
          swiglu8(c, a);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(a[q]));
        }
#endif
        amaxx[r * 2 + half] = amax;
        named_bar_sync<1, 256>();
        if (r < u.rows_valid) {
          const float am = fmaxf(amaxx[r * 2], amaxx[r * 2 + 1]);
          const float scale = (am > 0.f) ? am * (U_F16 ? (1.0f / 128.0f) : (1.0f / 448.0f)) : 1.0f;
          const float inv = __frcp_rn(scale);
          uint8_t* dst = p.act + (size_t)pair * I + u.nt * HC + half * 64;
#pragma unroll
          for (int c = 0; c < 64; c += 16) {
            float a[16];
#ifndef MOE_SWIGLU_TWICE
#pragma unroll
            for (int q = 0; q < 16; ++q) a[q] = sw[c + q];
#else
            swiglu8(c, a);
            swiglu8(c + 8, a + 8);
#endif
            uint4 pk;
            pk.x = (uint32_t)pack_e4m3x2(a[0] * inv, a[1] * inv) | ((uint32_t)pack_e4m3x2(a[2] * inv, a[3] * inv) << 16);
            pk.y = (uint32_t)pack_e4m3x2(a[4] * inv, a[5] * inv) | ((uint32_t)pack_e4m3x2(a[6] * inv, a[7] * inv) << 16);
            pk.z = (uint32_t)pack_e4m3x2(a[8] * inv, a[9] * inv) | ((uint32_t)pack_e4m3x2(a[10] * inv, a[11] * inv) << 16);
            pk.w = (uint32_t)pack_e4m3x2(a[12] * inv, a[13] * inv) | ((uint32_t)pack_e4m3x2(a[14] * inv, a[15] * inv) << 16);
            *reinterpret_cast<uint4*>(dst + c) = pk;
          }
          if (half == 0) p.act_scale[(size_t)u.nt * p.p_cap + pair] = scale;  // 128-column blocks: [16][p_cap]
        }
        named_bar_sync<1, 256>();  // xch / amaxx may be overwritten by the next unit only after they were read
      } else {
#ifdef MOE_G2_BULK
        // v75: the bf16 tile is staged in smem (64 rows per round; the 16-B chunks of row rr are rotated by rr so
        // that a warp's 32 row writes spread over the banks) and every row leaves with one or two bulk async
        // copies (single-pair rows / pair_out rows) or bulk reduce-adds (multi-pair rows into the zeroed output)
        const bool rv = r < u.rows_valid;
        const float w2 = rv ? p.pair_w[pair] : 0.f;
        const int tk = rv ? p.pair_token[pair] : 0;
        const int np = rv ? p.token_npair[tk] : 0;
        const bool rd = p.combine_atomic && np >= 2;
        uint16_t* dst = ((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * U_BN;
        uint8_t* stg = reinterpret_cast<uint8_t*>(xch);  // [64][512 B]
#pragma unroll
        for (int round = 0; round < 2; ++round) {
          const int rr = r - round * 64;
          const bool mine = (rr >= 0) && (rr < 64);
          if (mine) {
#pragma unroll
            for (int c = 0; c < HC; c += 8) {
              uint4 pk;
              pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
              pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
              pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
              pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
              const int ch = ((col_base + c) >> 3) + rr;
              *reinterpret_cast<uint4*>(stg + rr * 512 + ((ch & 31) << 4)) = pk;
            }
            fence_proxy_async_smem();
          }
          named_bar_sync<1, 256>();
          if (mine && half == 0 && rv) {
            const uint32_t src = smem_u32(stg + rr * 512);
            const int rot = rr & 31;
            const uint32_t n0 = (uint32_t)(32 - rot) * 16;
            if (rd) {
              bulk_red_bf16(dst, src + rot * 16, n0);
              if (rot) bulk_red_bf16(dst + n0 / 2, src, rot * 16);
            } else {
              bulk_store(dst, src + rot * 16, n0);
              if (rot) bulk_store(dst + n0 / 2, src, rot * 16);
            }
            bulk_commit();
            bulk_wait_read0();
          }
          named_bar_sync<1, 256>();
        }
#elif defined(MOE_G2_COAL)
        // v76: the bf16 tile is staged in smem (64 rows per round; the 16-B chunks of row rr are rotated by rr so a
        // warp's 32 row writes spread over the banks) and each warp then writes 8 rows per round as 512-B
        // coalesced vector stores (single-pair / pair_out rows) or vector red-adds (multi-pair rows)
        const bool rv = r < u.rows_valid;
        const float w2 = rv ? p.pair_w[pair] : 0.f;
        uint8_t* stg = reinterpret_cast<uint8_t*>(xch);                          // [64][512 B]
        unsigned long long* s_dst = reinterpret_cast<unsigned long long*>(amaxx);  // [64] row destinations (0 = skip)
        int* s_rd = reinterpret_cast<int*>(amaxx) + 128;                           // [64] red-add flags
#pragma unroll
        for (int round = 0; round < 2; ++round) {
          const int rr = r - round * 64;
          const bool mine = (rr >= 0) && (rr < 64);
          if (mine) {
#pragma unroll
            for (int c = 0; c < HC; c += 8) {
              uint4 pk;
              pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
              pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
              pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
              pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
              const int ch = ((col_base + c) >> 3) + rr;
              *reinterpret_cast<uint4*>(stg + rr * 512 + ((ch & 31) << 4)) = pk;
            }
            if (half == 0) {
              unsigned long long d = 0ull;
              int rd = 0;
              if (rv) {
                const int tk = p.pair_token[pair];
                const int np = p.token_npair[tk];
                rd = (p.combine_atomic && np >= 2) ? 1 : 0;
                d = reinterpret_cast<unsigned long long>(((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * U_BN);
              }
              s_dst[rr] = d;
              s_rd[rr] = rd;
            }
          }
          named_bar_sync<1, 256>();
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int row = (warp - 4) * 8 + i;
            const unsigned long long d = s_dst[row];
            if (d != 0ull) {
              const uint4 v = *reinterpret_cast<const uint4*>(stg + row * 512 + (((lane + row) & 31) << 4));
              uint16_t* drow = reinterpret_cast<uint16_t*>(d) + lane * 8;
              if (s_rd[row]) red_v4_bf16x2(drow, v);
              else *reinterpret_cast<uint4*>(drow) = v;
            }
          }
          named_bar_sync<1, 256>();
        }
#else
        if (r < u.rows_valid) {
          float w2;
          int tk, np;
#ifdef MOE_NEXT_PF
          if (ui > 0 && KB >= 4) { w2 = c_w2; tk = c_tk; np = c_np; } else
#endif
          { w2 = p.pair_w[pair]; tk = p.pair_token[pair]; np = p.token_npair[tk]; }
          const bool rd = p.combine_atomic && np >= 2;  // v59: red-add into the zeroed output row
          uint16_t* dst = ((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * U_BN + col_base;
#ifndef MOE_ST_V4_G2
          // v92: 32-B stores (8 per thread) for stored rows; red-adds stay 16 B (14107 950 -> 942-945 us)
#pragma unroll
          for (int c = 0; c < HC; c += 16) {
            uint4 pk, pq;
            pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
            pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
            pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
            pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
            pq.x = pack_bf16x2(acc[c + 8] * w2, acc[c + 9] * w2);
            pq.y = pack_bf16x2(acc[c + 10] * w2, acc[c + 11] * w2);
            pq.z = pack_bf16x2(acc[c + 12] * w2, acc[c + 13] * w2);
            pq.w = pack_bf16x2(acc[c + 14] * w2, acc[c + 15] * w2);
            if (rd) { red_v4_bf16x2(dst + c, pk); red_v4_bf16x2(dst + c + 8, pq); }
            else st_v8(dst + c, pk, pq);
          }
#else
#pragma unroll
          for (int c = 0; c < HC; c += 8) {
            uint4 pk;
            pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
            pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
            pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
            pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
            if (rd) red_v4_bf16x2(dst + c, pk);
            else *reinterpret_cast<uint4*>(dst + c) = pk;
          }
#endif
        }
#endif
      }
    }
#ifdef MOE_G2_BULK
    if (!GEMM1) bulk_wait0();
#endif
  }
  if (tid == 0) TRACE(p, 7);
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem_base, U_NBUF * U_ACC_COLS);
}

// ----------------------------------------------------------------------------------------------
// v71: GEMM2-only 128x128-tile variant (generated from umma_gemm_kernel by scripts/patch_v71.py): 4 TMEM
// accumulator buffers (4 x 128 columns) and 6 smem stages of 32 KB, so three MMAs are in flight while a
// K-block is promoted (the 128x256 kernel has only 2 buffers and is bound by the promotion handshake chain).
// ----------------------------------------------------------------------------------------------
constexpr int V_BN = 128, V_NBUF = 4, V_ACC_COLS = 128, V_STAGES = 6;
constexpr int V_STAGE_BYTES = U_BM * U_BK + V_BN * U_BK;  // 32 KB
constexpr int V_XCH_BYTES = U_BM * (V_BN / 2) * 2;
constexpr int V_SMEM_BYTES = 1024 + V_STAGES * V_STAGE_BYTES + U_MISC_BYTES + V_XCH_BYTES + U_AMAX_BYTES;
template <int ASK, bool GEMM1>
__global__ void __launch_bounds__(U_THREADS, 1)
    umma_n128_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                     const __grid_constant__ CUtensorMap tmA64, const __grid_constant__ CUtensorMap tmA32,
                     const GemmParams p) {
  static_assert(ASK == 128, "one promotion per 128-K stage");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw & 1023)) & 1023;
  const uint32_t sbase = raw + pad;
  int* s_off = reinterpret_cast<int*>(smem_raw + pad + V_STAGES * V_STAGE_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full = sbase + V_STAGES * V_STAGE_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty = full + V_STAGES * 8;
  const uint32_t acc_full = empty + V_STAGES * 8;
  const uint32_t acc_empty = acc_full + V_NBUF * 8;
  uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(smem_raw + pad + V_STAGES * V_STAGE_BYTES + 2 * (NLOCAL + 1) * 4 + (2 * V_STAGES + 2 * V_NBUF) * 8);
  int* s_end = reinterpret_cast<int*>(smem_raw + pad + V_STAGES * V_STAGE_BYTES + 512);  // [NLOCAL] (barriers end at 396 with 4 buffers)
  uint16_t* xch = reinterpret_cast<uint16_t*>(smem_raw + pad + V_STAGES * V_STAGE_BYTES + U_MISC_BYTES);  // [128][128] bf16
  float* amaxx = reinterpret_cast<float*>(smem_raw + pad + V_STAGES * V_STAGE_BYTES + U_MISC_BYTES + V_XCH_BYTES);  // [128][2]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < V_STAGES; ++s) {
      mbar_init(full + s * 8, (GEMM1 && p.a_gather) ? 1 + 64 : 1);  // v62: + one cp.async arrival per producer thread (warps 0, 3)
      mbar_init(empty + s * 8, 1);
    }
#pragma unroll
    for (int b = 0; b < V_NBUF; ++b) {
      mbar_init(acc_full + b * 8, 1);
      mbar_init(acc_empty + b * 8, 256);
    }
    fence_barrier_init();
    tma_prefetch_desc(&tmA);
    tma_prefetch_desc(&tmB);
  }
  if (warp == 2) {
    tmem_alloc(smem_u32(tmem_ptr), V_NBUF * V_ACC_COLS);
    tmem_relinquish();
  }
  if (tid == 0) TRACE(p, 0);
  pdl_wait();
  if (tid == 0) TRACE(p, 1);
  pdl_launch_dependents();
  load_expert_tables<U_BM>(p, s_off, s_mtp, s_end);
  __syncthreads();
  const uint32_t tmem_base = *tmem_ptr;
  const int mtile_total = s_mtp[NLOCAL];
  const int total_units = mtile_total * p.n_tiles;
  const int my_units = ((int)blockIdx.x < total_units) ? (total_units - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  const int KB = p.K / 128;

  if (warp < 4) {
    setmaxnreg_dec40();
    if (GEMM1 && p.a_gather && warp != 1 && warp != 2) {
      // ================= v62: producer warps 0 and 3 (64 threads): TMA weights + cp.async A rows ==========
      // thread t (0..63) gathers rows t and t+64 of the m-tile: 8 x 16 B per row per K-block, 128B-swizzled
      // (16-B chunk c of row r at r*128 + ((c ^ (r & 7)) << 4)), the TMA SWIZZLE_128B / UMMA canonical layout.
      const int pt = (warp == 0) ? lane : 32 + lane;  // 0..63
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        const int b_row0 = u.e * p.N_total + u.nt * (V_BN / 2);
        const int b_row1 = b_row0 + I;
        const uint8_t* src[2];
        bool valid[2];
        uint32_t dst[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = pt + 64 * h;
          valid[h] = r < u.rows_valid;
          const int tok = valid[h] ? p.pair_token[u.row0 + r] : 0;
          src[h] = p.A + (size_t)tok * p.K;
          dst[h] = r * 128;
        }
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(empty + stage * 8, phase); UT_END(p, 12, true); }
          const uint32_t st = sbase + stage * V_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
          if (pt == 0) {
            mbar_arrive_expect_tx(fb, V_BN * U_BK);
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
            tma_load_2d_hint(st + U_BM * U_BK + (V_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
          }
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (valid[h]) {
              const uint8_t* srow = src[h] + (size_t)kb * 128;
              const int r = pt + 64 * h;
#pragma unroll
              for (int c = 0; c < 8; ++c) cp_async_16(st + dst[h] + ((c ^ (r & 7)) << 4), srow + c * 16, true);
            }
          }
          cp_async_mbar_arrive_noinc(fb);
          if (++stage == V_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    } else if (warp == 0 && lane == 0) {
      // =========================== TMA producer =============================================
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      const uint64_t pol_a = l2_policy_evict_last();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        int b_row0, b_row1 = 0;
        if (GEMM1) {
          b_row0 = u.e * p.N_total + u.nt * (V_BN / 2);
          b_row1 = b_row0 + I;
        } else {
          b_row0 = u.e * p.N_total + u.nt * V_BN;
        }
        // v69: A box sized to the valid rows; rows >= rows_valid hold stale smem whose results are discarded
        const CUtensorMap* ta = &tmA;
        uint32_t a_bytes = U_BM * U_BK;
        if (p.a_box) {
          if (u.rows_valid <= 32) { ta = &tmA32; a_bytes = 32 * U_BK; }
          else if (u.rows_valid <= 64) { ta = &tmA64; a_bytes = 64 * U_BK; }
        }
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(empty + stage * 8, phase); UT_END(p, 12, true); }
          const uint32_t st = sbase + stage * V_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
          mbar_arrive_expect_tx(fb, a_bytes + V_BN * U_BK);
          tma_load_2d_hint(st, ta, fb, kb * U_BK, u.row0, pol_a);
          if (GEMM1) {
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
            tma_load_2d_hint(st + U_BM * U_BK + (V_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
          } else {
            tma_load_2d_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_row0, pol_w);
          }
          if (++stage == V_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    } else if (warp == 1 && lane == 0) {
      // =========================== MMA issuer ===============================================
      const uint32_t idesc = ((U_F16 ? 0u : 1u) << 4) | ((V_BN >> 3) << 17) | ((U_BM >> 4) << 24);  // F16/F32 += E4M3 * E4M3, K-major
      int stage = 0;
      uint32_t phase = 0;
      uint32_t kctr = 0;  // global K-block counter (TMEM buffer = kctr % V_NBUF)
      for (int ui = 0; ui < my_units; ++ui) {
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(full + stage * 8, phase); UT_END(p, 8, true); }
          tc_fence_after();
          const uint32_t st = sbase + stage * V_STAGE_BYTES;
          const uint64_t adesc = umma_desc(st), bdesc = umma_desc(st + U_BM * U_BK);
          const int buf = kctr % V_NBUF;
          { UT_BEGIN(); mbar_wait(acc_empty + buf * 8, ((kctr / V_NBUF) & 1) ^ 1); UT_END(p, 9, true); }
#ifdef MOE_TRACE
          if (p.trace && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 13] += 1;
#endif
          tc_fence_after();
          const uint32_t d_tmem = tmem_base + buf * V_ACC_COLS;
#ifndef MOE_ABL_NOMMA
#pragma unroll
          for (int k = 0; k < U_BK / 32; ++k) {
            const uint32_t koff = (uint32_t)(k * 32) >> 4;
            umma_f8f6f4(d_tmem, adesc + koff, bdesc + koff, idesc, k > 0 ? 1u : 0u);
          }
#endif
          umma_commit(acc_full + buf * 8);
          umma_commit(empty + stage * 8);
          ++kctr;
          if (++stage == V_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // =========================== promotion + epilogue warpgroups ==========================
    // warps 4..7 own columns [0,128) of every row (x1 / first weight block), warps 8..11 own [128,256).
    setmaxnreg_inc232();
    constexpr int HC = V_BN / 2;  // 128 columns per thread
    const int half = (warp >= 8) ? 1 : 0;
    const int r = (warp & 3) * 32 + lane;  // row within the tile == TMEM lane
    const uint32_t lane_base = ((uint32_t)((warp & 3) * 32)) << 16;
    const uint32_t col_base = U_F16 ? half * (HC / 2) : half * HC;  // TMEM column of this thread's first element
    uint64_t acc2[HC / 2];  // packed fp32 pairs (cols 2i, 2i+1)
    float acc[HC];
    uint32_t kctr = 0;
    for (int ui = 0; ui < my_units; ++ui) {
      const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
      const int arow = u.row0 + r;  // gathered activation row (valid if r < rows_valid)
      // v62: with gather-on-demand the A scale lives in hidden_states_scale[kb][token]
      const float* sa_ptr = p.A_scale + ((GEMM1 && p.a_gather) ? ((r < u.rows_valid) ? p.pair_token[arow] : 0) : arow);
      int nb;
      if (GEMM1) {
        nb = (half == 0) ? u.nt : (I / 128 + u.nt);
      } else {
        nb = u.nt;  // V_BN = 128: both column halves share one 128-column W-scale block
      }
      const float* sw = p.W_scale + ((size_t)u.e * (p.N_total / 128) + nb) * KB;
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) acc2[i] = 0ull;
      // block-scale prefetch: 4 K-blocks ahead in rotating scalars (no unrolling, no spills)
      float w0 = __ldg(sw), s0 = __ldg(sa_ptr);
      float w1 = 0.f, s1 = 0.f, w2 = 0.f, s2 = 0.f, w3 = 0.f, s3 = 0.f;
      if (KB > 1) { w1 = __ldg(sw + 1); s1 = __ldg(sa_ptr + (size_t)1 * p.a_rows_stride); }
      if (KB > 2) { w2 = __ldg(sw + 2); s2 = __ldg(sa_ptr + (size_t)2 * p.a_rows_stride); }
      if (KB > 3) { w3 = __ldg(sw + 3); s3 = __ldg(sa_ptr + (size_t)3 * p.a_rows_stride); }
      uint32_t va[32], vb[32];
      {
        // first K-block of the unit: chunks 0,1 in flight before the loop
        const int buf = kctr % V_NBUF;
        { UT_BEGIN(); mbar_wait(acc_full + buf * 8, (kctr / V_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
        tc_fence_after();
        const uint32_t tb = tmem_base + lane_base + buf * V_ACC_COLS + col_base;
        tmem_ld_32x32b_x32(tb, va);
        tmem_ld_32x32b_x32(tb + 32, vb);
      }
      if (ui == 0 && warp == 4 && lane == 0) TRACE(p, 3);
      for (int kb = 0; kb < KB; ++kb) {
#ifdef MOE_ABL_NOEPI
        {
          const int bufx = kctr % V_NBUF;
          if (kb > 0) mbar_wait(acc_full + bufx * 8, (kctr / V_NBUF) & 1);
          tmem_wait_ld();
          tc_fence_after();
          tc_fence_before();
          mbar_arrive(acc_empty + bufx * 8);
          ++kctr;
          continue;
        }
#endif
#ifdef MOE_ABL_NOSCALE
        const float sc = 1.0f;
#else
        const float sc = s0 * w0;
#endif
        s0 = s1; w0 = w1; s1 = s2; w1 = w2; s2 = s3; w2 = w3;
        if (kb + 4 < KB) {
          w3 = __ldg(sw + kb + 4);
          s3 = __ldg(sa_ptr + (size_t)(kb + 4) * p.a_rows_stride);
        }
        uint64_t s2;
        {
          UT_BEGIN();
          s2 = pack_f32x2(sc, sc);
#ifdef MOE_TRACE
          asm volatile("" : "+l"(s2));  // force the scale dependency inside this window
#endif
          UT_END(p, 15, warp == 4 && lane == 0);
        }
        const int buf = kctr % V_NBUF;
        const uint32_t tb = tmem_base + lane_base + buf * V_ACC_COLS + col_base;
        UT_BEGIN();
        if constexpr (U_F16) {
          // v57: the whole 128-column half is already in va/vb (64 packed f16x2 words)
          tmem_wait_ld();
          fma2_chunk32h(acc2 + 0, va, s2);
          fma2_chunk32h(acc2 + 32, vb, s2);
          UT_END(p, 11, warp == 4 && lane == 0);
          tc_fence_before();
          mbar_arrive(acc_empty + buf * 8);
          if (kb + 1 < KB) {
            const int nbuf = (kctr + 1) % V_NBUF;
            { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / V_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
            tc_fence_after();
            const uint32_t tbn = tmem_base + lane_base + nbuf * V_ACC_COLS + col_base;
            tmem_ld_32x32b_x32(tbn, va);
            tmem_ld_32x32b_x32(tbn + 32, vb);
          }
        } else {
#ifdef MOE_ABL_HALFREAD
        // ablation: read only chunks 0,1 (half the TMEM bytes), emulating fp16 accumulators
        tmem_wait_ld();
        fma2_chunk32(acc2 + 0, va, s2);
        fma2_chunk32(acc2 + 16, vb, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          const int nbuf = (kctr + 1) % V_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / V_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * V_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
        tc_fence_before();
        mbar_arrive(acc_empty + buf * 8);
#elif defined(MOE_LATE_RELEASE)
        tmem_wait_ld();                  // chunks 0,1 of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        tmem_ld_32x32b_x32(tb + 64, va);
        fma2_chunk32(acc2 + 16, vb, s2);
        tmem_ld_32x32b_x32(tb + 96, vb);
        tmem_wait_ld();                  // chunks 2,3
        fma2_chunk32(acc2 + 32, va, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          // prefetch chunks 0,1 of the next K-block (other TMEM buffer) behind the remaining FMAs
          const int nbuf = (kctr + 1) % V_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / V_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * V_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
        }
        fma2_chunk32(acc2 + 48, vb, s2);
        if (more) tmem_ld_32x32b_x32(tbn + 32, vb);
        {
          UT_BEGIN();
          tc_fence_before();
          mbar_arrive(acc_empty + buf * 8);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
#else
        // v55: finish this K-block and release its TMEM buffer *before* waiting for the next MMA, so
        // the issuer can enqueue K-block kb+2 while kb+1 is still executing (no tensor-pipe bubble)
        tmem_wait_ld();                  // chunks 0,1 = the whole 64-column half of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        fma2_chunk32(acc2 + 16, vb, s2);
        UT_END(p, 11, warp == 4 && lane == 0);
        {
          UT_BEGIN();
          tc_fence_before();
          mbar_arrive(acc_empty + buf * 8);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
        const bool more = (kb + 1 < KB);
        if (more) {
          const int nbuf = (kctr + 1) % V_NBUF;
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / V_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          const uint32_t tbn = tmem_base + lane_base + nbuf * V_ACC_COLS + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
#endif
        }  // !U_F16
        ++kctr;
      }
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) {
        acc[2 * i] = __uint_as_float((uint32_t)acc2[i]);
        acc[2 * i + 1] = __uint_as_float((uint32_t)(acc2[i] >> 32));
      }
      // ---- epilogue for row r ------------------------------------------------------------
      const int pair = u.row0 + r;
      if constexpr (GEMM1) {
        // export: half 0 -> x1[64..127], half 1 -> x2[0..63] (bf16, 16 B chunks XOR-swizzled by row)
        {
          // v61: literal register bases per half (a runtime base would move `acc` to local memory)
          auto export8 = [&](int c, float v0, float v1, float v2_, float v3, float v4, float v5, float v6, float v7) {
            uint4 pk;
            pk.x = pack_bf16x2(v0, v1);
            pk.y = pack_bf16x2(v2_, v3);
            pk.z = pack_bf16x2(v4, v5);
            pk.w = pack_bf16x2(v6, v7);
            const int ch = (half << 3) + (c >> 3);  // half h writes chunks [8h, 8h+8) of the 16-chunk row
            *reinterpret_cast<uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3)) = pk;
          };
          if (half == 0) {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[64 + c], acc[64 + c + 1], acc[64 + c + 2], acc[64 + c + 3], acc[64 + c + 4], acc[64 + c + 5], acc[64 + c + 6], acc[64 + c + 7]);
          } else {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[c], acc[c + 1], acc[c + 2], acc[c + 3], acc[c + 4], acc[c + 5], acc[c + 6], acc[c + 7]);
          }
        }
        named_bar_sync<1, 256>();
        // half 0 quantizes columns [0,64): v = x1[0..63] (own) * silu(x2[0..63]) (from half 1)
        // half 1 quantizes columns [64,128): v = x1[64..127] (from half 0) * silu(x2[64..127]) (own)
        // v61: SwiGLU values are computed twice (amax pass, quantization pass) instead of being held in a
        // 64-float array; each half indexes `acc` with literal bases
        const int ch0 = (half == 0) ? 8 : 0;   // chunks written by the other half
        auto swiglu8 = [&](int c, float* a) {
          const int ch = ch0 + (c >> 3);
          const uint4 pk = *reinterpret_cast<const uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3));
          const uint32_t w4[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float o0 = bf16_to_f32((uint16_t)(w4[q] & 0xffff));
            const float o1 = bf16_to_f32((uint16_t)(w4[q] >> 16));
            if (half == 0) {  // own = x1 (cols 0..63), other = x2
              a[2 * q] = acc[c + 2 * q] * silu_fast(o0);
              a[2 * q + 1] = acc[c + 2 * q + 1] * silu_fast(o1);
            } else {          // own = x2 (cols 64..127), other = x1
              a[2 * q] = o0 * silu_fast(acc[64 + c + 2 * q]);
              a[2 * q + 1] = o1 * silu_fast(acc[64 + c + 2 * q + 1]);
            }
          }
        };
        float amax = 0.f;
#ifndef MOE_SWIGLU_TWICE
        // v90: SwiGLU computed once (64 values stay in the registers freed by the export); default (14107 955 -> 942 us)
        float sw[64];
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          swiglu8(c, sw + c);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(sw[c + q]));
        }
#else
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          float a[8];
          swiglu8(c, a);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(a[q]));
        }
#endif
        amaxx[r * 2 + half] = amax;
        named_bar_sync<1, 256>();
        if (r < u.rows_valid) {
          const float am = fmaxf(amaxx[r * 2], amaxx[r * 2 + 1]);
          const float scale = (am > 0.f) ? am * (U_F16 ? (1.0f / 128.0f) : (1.0f / 448.0f)) : 1.0f;
          const float inv = __frcp_rn(scale);
          uint8_t* dst = p.act + (size_t)pair * I + u.nt * HC + half * 64;
#pragma unroll
          for (int c = 0; c < 64; c += 16) {
            float a[16];
#ifndef MOE_SWIGLU_TWICE
#pragma unroll
            for (int q = 0; q < 16; ++q) a[q] = sw[c + q];
#else
            swiglu8(c, a);
            swiglu8(c + 8, a + 8);
#endif
            uint4 pk;
            pk.x = (uint32_t)pack_e4m3x2(a[0] * inv, a[1] * inv) | ((uint32_t)pack_e4m3x2(a[2] * inv, a[3] * inv) << 16);
            pk.y = (uint32_t)pack_e4m3x2(a[4] * inv, a[5] * inv) | ((uint32_t)pack_e4m3x2(a[6] * inv, a[7] * inv) << 16);
            pk.z = (uint32_t)pack_e4m3x2(a[8] * inv, a[9] * inv) | ((uint32_t)pack_e4m3x2(a[10] * inv, a[11] * inv) << 16);
            pk.w = (uint32_t)pack_e4m3x2(a[12] * inv, a[13] * inv) | ((uint32_t)pack_e4m3x2(a[14] * inv, a[15] * inv) << 16);
            *reinterpret_cast<uint4*>(dst + c) = pk;
          }
          if (half == 0) p.act_scale[(size_t)u.nt * p.p_cap + pair] = scale;  // 128-column blocks: [16][p_cap]
        }
        named_bar_sync<1, 256>();  // xch / amaxx may be overwritten by the next unit only after they were read
      } else {
        if (r < u.rows_valid) {
          const float w2 = p.pair_w[pair];
          const int tk = p.pair_token[pair];
          const int np = p.token_npair[tk];
          const bool rd = p.combine_atomic && np >= 2;  // v59: red-add into the zeroed output row
          uint16_t* dst = ((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * V_BN + col_base;
#pragma unroll
          for (int c = 0; c < HC; c += 8) {
            uint4 pk;
            pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
            pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
            pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
            pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
            if (rd) red_v4_bf16x2(dst + c, pk);
            else *reinterpret_cast<uint4*>(dst + c) = pk;
          }
        }
      }
    }
  }
  if (tid == 0) TRACE(p, 7);
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem_base, V_NBUF * V_ACC_COLS);
}


// ----------------------------------------------------------------------------------------------
// 2-SM variant (cta_group::2), v70: generated from umma_gemm_kernel by scripts/patch_v70.py (same producer /
// promotion / epilogue code; the leader CTA issues 256x256 MMAs over both CTAs' smem; each CTA promotes its own
// 128-row half from its own TMEM). acc_empty is released with one arrive per warp (count 32) on the leader.
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ void mbar_arrive_cnt(uint32_t bar, uint32_t cnt) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster_cnt(uint32_t bar, uint32_t cta, uint32_t cnt) {
  asm volatile(
      "{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\n"
      "mbarrier.arrive.shared::cluster.b64 _, [ra], %2;\n}\n" ::"r"(bar), "r"(cta), "r"(cnt) : "memory");
}
__device__ __forceinline__ void acc_release2(uint32_t bar, uint32_t crank) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    if (crank == 0) mbar_arrive_cnt(bar, 32);
    else mbar_arrive_cluster_cnt(bar, 0, 32);
  }
}
template <int ASK, bool GEMM1>
__global__ void __launch_bounds__(U_THREADS, 1) __cluster_dims__(2, 1, 1)
    umma2_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                     const __grid_constant__ CUtensorMap tmA64, const __grid_constant__ CUtensorMap tmA32,
                     const GemmParams p) {
  static_assert(ASK == 128, "one promotion per 128-K stage");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw & 1023)) & 1023;
  const uint32_t sbase = raw + pad;
  int* s_off = reinterpret_cast<int*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES);
  int* s_mtp = s_off + NLOCAL + 1;
  const uint32_t full = sbase + U2_STAGES * U2_STAGE_BYTES + 2 * (NLOCAL + 1) * 4;
  const uint32_t empty = full + U2_STAGES * 8;
  const uint32_t acc_full = empty + U2_STAGES * 8;
  const uint32_t acc_empty = acc_full + U2_NBUF * 8;
  uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES + 2 * (NLOCAL + 1) * 4 + (2 * U2_STAGES + 2 * U2_NBUF) * 8);
  int* s_end = reinterpret_cast<int*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES + 512);  // [NLOCAL] (barriers end at 396 with 4 buffers)
  uint16_t* xch = reinterpret_cast<uint16_t*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES + U_MISC_BYTES);  // [128][128] bf16
  float* amaxx = reinterpret_cast<float*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES + U_MISC_BYTES + U_XCH_BYTES);  // [128][2]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < U2_STAGES; ++s) {
      mbar_init(full + s * 8, 1);
      mbar_init(empty + s * 8, 1);
    }
#pragma unroll
    for (int b = 0; b < U2_NBUF; ++b) {
      mbar_init(acc_full + b * 8, 1);
      mbar_init(acc_empty + b * 8, 2 * 256);  // 8 warps x count 32 per CTA; both CTAs release on the leader
    }
    fence_barrier_init();
    tma_prefetch_desc(&tmA);
    tma_prefetch_desc(&tmB);
  }
  if (warp == 2) {
    tmem_alloc2(smem_u32(tmem_ptr), U2_NBUF * U_BN);
    tmem_relinquish2();
  }
  if (tid == 0) TRACE(p, 0);
  pdl_wait();
  if (tid == 0) TRACE(p, 1);
  pdl_launch_dependents();
  load_expert_tables<U_BM>(p, s_off, s_mtp, s_end);
  __syncthreads();
  // m-tile pairs per expert (a 2-SM unit covers two consecutive m-tiles); lives in the dynamic smem misc area
  int* s_pp = reinterpret_cast<int*>(smem_raw + pad + U2_STAGES * U2_STAGE_BYTES + 768);
  if (warp == 0) {
    const int mt = (lane < NLOCAL) ? (s_mtp[lane + 1] - s_mtp[lane]) : 0;
    int m = (mt + 1) >> 1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, m, o);
      if (lane >= o) m += v;
    }
    s_pp[lane + 1] = m;
    if (lane == 0) s_pp[0] = 0;
  }
  __syncthreads();
  cluster_sync();  // peer barriers initialized and TMEM allocated in both CTAs
  const uint32_t crank = cluster_ctarank();
  const int cidx = (int)(blockIdx.x >> 1), ncl = (int)(gridDim.x >> 1);
  const uint32_t tmem_base = *tmem_ptr;
  const int total_units = s_pp[NLOCAL] * p.n_tiles;
  const int my_units = (cidx < total_units) ? (total_units - cidx + ncl - 1) / ncl : 0;
  const int KB = p.K / 128;

  if (warp < 4) {
    setmaxnreg_dec40();
    if (warp == 0 && lane == 0) {
      // =========================== TMA producer =============================================
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      const uint64_t pol_a = l2_policy_evict_last();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit2(cidx + ui * ncl, s_off, s_pp, p.n_tiles, s_end, crank);
        // this CTA's N-half of B: x1 (rank 0) / x2 (rank 1) for GEMM1, W2 rows nt*256 + 128*rank for GEMM2
        const int b_rowh = GEMM1 ? (u.e * p.N_total + u.nt * (U_BN / 2) + (int)crank * I)
                                 : (u.e * p.N_total + u.nt * U_BN + (int)crank * (U_BN / 2));
        // v69/v70: A box sized to the valid rows of each CTA's m-tile (no rows -> no A load; stale smem rows are
        // discarded by the epilogue); the leader's full barrier expects the bytes of both CTAs
        auto a_sel = [&](int rv, const CUtensorMap*& m) -> uint32_t {
          m = &tmA;
          if (rv <= 0) return 0u;
          if (p.a_box && rv <= 32) { m = &tmA32; return 32u * U_BK; }
          if (p.a_box && rv <= 64) { m = &tmA64; return 64u * U_BK; }
          return (uint32_t)U_BM * U_BK;
        };
        const CUtensorMap* ta;
        const uint32_t a_bytes = a_sel(u.rows_valid, ta);
        const UnitInfo up = decode_unit2(cidx + ui * ncl, s_off, s_pp, p.n_tiles, s_end, crank ^ 1u);
        const CUtensorMap* tap;
        const uint32_t tx_total = a_bytes + a_sel(up.rows_valid, tap) + 2u * (U_BN / 2) * U_BK;
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(empty + stage * 8, phase); UT_END(p, 12, true); }
          const uint32_t st = sbase + stage * U2_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
          if (crank == 0) mbar_arrive_expect_tx(fb, tx_total);  // both CTAs' bytes count here
          if (a_bytes) tma_load_2d_2sm_hint(st, ta, fb, kb * U_BK, u.row0, pol_a);
          tma_load_2d_2sm_hint(st + U_BM * U_BK, &tmB, fb, kb * U_BK, b_rowh, pol_w);
          if (++stage == U2_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    } else if (warp == 1 && lane == 0 && crank == 0) {
      // =========================== MMA issuer (leader CTA only) =============================
      const uint32_t idesc = (1u << 4) | ((U_BN >> 3) << 17) | (((2 * U_BM) >> 4) << 24);  // M = 256 (pair), N = 256
      int stage = 0;
      uint32_t phase = 0;
      uint32_t kctr = 0;  // global K-block counter (TMEM buffer = kctr % U2_NBUF)
      for (int ui = 0; ui < my_units; ++ui) {
        for (int kb = 0; kb < KB; ++kb) {
          { UT_BEGIN(); mbar_wait(full + stage * 8, phase); UT_END(p, 8, true); }
          tc_fence_after();
          const uint32_t st = sbase + stage * U2_STAGE_BYTES;
          const uint64_t adesc = umma_desc(st), bdesc = umma_desc(st + U_BM * U_BK);
          const int buf = kctr % U2_NBUF;
          { UT_BEGIN(); mbar_wait(acc_empty + buf * 8, ((kctr / U2_NBUF) & 1) ^ 1); UT_END(p, 9, true); }
#ifdef MOE_TRACE
          if (p.trace && blockIdx.x < 64) p.trace[blockIdx.x * 16 + 13] += 1;
#endif
          tc_fence_after();
          const uint32_t d_tmem = tmem_base + buf * U_BN;
#ifndef MOE_ABL_NOMMA
#pragma unroll
          for (int k = 0; k < U_BK / 32; ++k) {
            const uint32_t koff = (uint32_t)(k * 32) >> 4;
            umma2_f8f6f4(d_tmem, adesc + koff, bdesc + koff, idesc, k > 0 ? 1u : 0u);
          }
#endif
          umma_commit2_mc(acc_full + buf * 8, 0x3);
          umma_commit2_mc(empty + stage * 8, 0x3);
          ++kctr;
          if (++stage == U2_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // =========================== promotion + epilogue warpgroups ==========================
    // warps 4..7 own columns [0,128) of every row (x1 / first weight block), warps 8..11 own [128,256).
    setmaxnreg_inc232();
    constexpr int HC = U_BN / 2;  // 128 columns per thread
    const int half = (warp >= 8) ? 1 : 0;
    const int r = (warp & 3) * 32 + lane;  // row within the tile == TMEM lane
    const uint32_t lane_base = ((uint32_t)((warp & 3) * 32)) << 16;
    const uint32_t col_base = U_F16 ? half * (HC / 2) : half * HC;  // TMEM column of this thread's first element
    uint64_t acc2[HC / 2];  // packed fp32 pairs (cols 2i, 2i+1)
    float acc[HC];
    uint32_t kctr = 0;
    for (int ui = 0; ui < my_units; ++ui) {
      const UnitInfo u = decode_unit2(cidx + ui * ncl, s_off, s_pp, p.n_tiles, s_end, crank);
      const int arow = u.row0 + r;  // gathered activation row (valid if r < rows_valid)
      // v62: with gather-on-demand the A scale lives in hidden_states_scale[kb][token]
      const float* sa_ptr = p.A_scale + arow;
      int nb;
      if (GEMM1) {
        nb = (half == 0) ? u.nt : (I / 128 + u.nt);
      } else {
        nb = u.nt * (U_BN / 128) + half;
      }
      const float* sw = p.W_scale + ((size_t)u.e * (p.N_total / 128) + nb) * KB;
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) acc2[i] = 0ull;
      // block-scale prefetch: 4 K-blocks ahead in rotating scalars (no unrolling, no spills)
      float w0 = __ldg(sw), s0 = __ldg(sa_ptr);
      float w1 = 0.f, s1 = 0.f, w2 = 0.f, s2 = 0.f, w3 = 0.f, s3 = 0.f;
      if (KB > 1) { w1 = __ldg(sw + 1); s1 = __ldg(sa_ptr + (size_t)1 * p.a_rows_stride); }
      if (KB > 2) { w2 = __ldg(sw + 2); s2 = __ldg(sa_ptr + (size_t)2 * p.a_rows_stride); }
      if (KB > 3) { w3 = __ldg(sw + 3); s3 = __ldg(sa_ptr + (size_t)3 * p.a_rows_stride); }
      uint32_t va[32], vb[32];
      {
        // first K-block of the unit: chunks 0,1 in flight before the loop
        const int buf = kctr % U2_NBUF;
        { UT_BEGIN(); mbar_wait(acc_full + buf * 8, (kctr / U2_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
        tc_fence_after();
        const uint32_t tb = tmem_base + lane_base + buf * U_BN + col_base;
        tmem_ld_32x32b_x32(tb, va);
        tmem_ld_32x32b_x32(tb + 32, vb);
      }
      if (ui == 0 && warp == 4 && lane == 0) TRACE(p, 3);
      for (int kb = 0; kb < KB; ++kb) {
#ifdef MOE_ABL_NOEPI
        {
          const int bufx = kctr % U2_NBUF;
          if (kb > 0) mbar_wait(acc_full + bufx * 8, (kctr / U2_NBUF) & 1);
          tmem_wait_ld();
          tc_fence_after();
          tc_fence_before();
          acc_release2(acc_empty + bufx * 8, crank);
          ++kctr;
          continue;
        }
#endif
#ifdef MOE_ABL_NOSCALE
        const float sc = 1.0f;
#else
        const float sc = s0 * w0;
#endif
        s0 = s1; w0 = w1; s1 = s2; w1 = w2; s2 = s3; w2 = w3;
        if (kb + 4 < KB) {
          w3 = __ldg(sw + kb + 4);
          s3 = __ldg(sa_ptr + (size_t)(kb + 4) * p.a_rows_stride);
        }
        uint64_t s2;
        {
          UT_BEGIN();
          s2 = pack_f32x2(sc, sc);
#ifdef MOE_TRACE
          asm volatile("" : "+l"(s2));  // force the scale dependency inside this window
#endif
          UT_END(p, 15, warp == 4 && lane == 0);
        }
        const int buf = kctr % U2_NBUF;
        const uint32_t tb = tmem_base + lane_base + buf * U_BN + col_base;
        UT_BEGIN();
        if constexpr (U_F16) {
          // v57: the whole 128-column half is already in va/vb (64 packed f16x2 words)
          tmem_wait_ld();
          fma2_chunk32h(acc2 + 0, va, s2);
          fma2_chunk32h(acc2 + 32, vb, s2);
          UT_END(p, 11, warp == 4 && lane == 0);
          tc_fence_before();
          acc_release2(acc_empty + buf * 8, crank);
          if (kb + 1 < KB) {
            const int nbuf = (kctr + 1) % U2_NBUF;
            { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U2_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
            tc_fence_after();
            const uint32_t tbn = tmem_base + lane_base + nbuf * U_BN + col_base;
            tmem_ld_32x32b_x32(tbn, va);
            tmem_ld_32x32b_x32(tbn + 32, vb);
          }
        } else {
#ifdef MOE_ABL_HALFREAD
        // ablation: read only chunks 0,1 (half the TMEM bytes), emulating fp16 accumulators
        tmem_wait_ld();
        fma2_chunk32(acc2 + 0, va, s2);
        fma2_chunk32(acc2 + 16, vb, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          const int nbuf = (kctr + 1) % U2_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U2_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * U_BN + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
        tc_fence_before();
        acc_release2(acc_empty + buf * 8, crank);
#elif defined(MOE_LATE_RELEASE)
        tmem_wait_ld();                  // chunks 0,1 of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        tmem_ld_32x32b_x32(tb + 64, va);
        fma2_chunk32(acc2 + 16, vb, s2);
        tmem_ld_32x32b_x32(tb + 96, vb);
        tmem_wait_ld();                  // chunks 2,3
        fma2_chunk32(acc2 + 32, va, s2);
        const bool more = (kb + 1 < KB);
        uint32_t tbn = 0;
        if (more) {
          // prefetch chunks 0,1 of the next K-block (other TMEM buffer) behind the remaining FMAs
          const int nbuf = (kctr + 1) % U2_NBUF;
          UT_END(p, 11, warp == 4 && lane == 0);
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U2_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          tbn = tmem_base + lane_base + nbuf * U_BN + col_base;
          tmem_ld_32x32b_x32(tbn, va);
        }
        fma2_chunk32(acc2 + 48, vb, s2);
        if (more) tmem_ld_32x32b_x32(tbn + 32, vb);
        {
          UT_BEGIN();
          tc_fence_before();
          acc_release2(acc_empty + buf * 8, crank);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
#else
        // v55: finish this K-block and release its TMEM buffer *before* waiting for the next MMA, so
        // the issuer can enqueue K-block kb+2 while kb+1 is still executing (no tensor-pipe bubble)
        tmem_wait_ld();                  // chunks 0,1 of this K-block
        fma2_chunk32(acc2 + 0, va, s2);
        tmem_ld_32x32b_x32(tb + 64, va);
        fma2_chunk32(acc2 + 16, vb, s2);
        tmem_ld_32x32b_x32(tb + 96, vb);
        tmem_wait_ld();                  // chunks 2,3
        fma2_chunk32(acc2 + 32, va, s2);
        fma2_chunk32(acc2 + 48, vb, s2);
        UT_END(p, 11, warp == 4 && lane == 0);
        {
          UT_BEGIN();
          tc_fence_before();
          acc_release2(acc_empty + buf * 8, crank);
          UT_END(p, 14, warp == 4 && lane == 0);
        }
        const bool more = (kb + 1 < KB);
        if (more) {
          const int nbuf = (kctr + 1) % U2_NBUF;
          { UT_BEGIN(); mbar_wait(acc_full + nbuf * 8, ((kctr + 1) / U2_NBUF) & 1); UT_END(p, 10, warp == 4 && lane == 0); }
          tc_fence_after();
          const uint32_t tbn = tmem_base + lane_base + nbuf * U_BN + col_base;
          tmem_ld_32x32b_x32(tbn, va);
          tmem_ld_32x32b_x32(tbn + 32, vb);
        }
#endif
        }  // !U_F16
        ++kctr;
      }
#pragma unroll
      for (int i = 0; i < HC / 2; ++i) {
        acc[2 * i] = __uint_as_float((uint32_t)acc2[i]);
        acc[2 * i + 1] = __uint_as_float((uint32_t)(acc2[i] >> 32));
      }
      // ---- epilogue for row r ------------------------------------------------------------
      const int pair = u.row0 + r;
      if constexpr (GEMM1) {
        // export: half 0 -> x1[64..127], half 1 -> x2[0..63] (bf16, 16 B chunks XOR-swizzled by row)
        {
          // v61: literal register bases per half (a runtime base would move `acc` to local memory)
          auto export8 = [&](int c, float v0, float v1, float v2_, float v3, float v4, float v5, float v6, float v7) {
            uint4 pk;
            pk.x = pack_bf16x2(v0, v1);
            pk.y = pack_bf16x2(v2_, v3);
            pk.z = pack_bf16x2(v4, v5);
            pk.w = pack_bf16x2(v6, v7);
            const int ch = (half << 3) + (c >> 3);  // half h writes chunks [8h, 8h+8) of the 16-chunk row
            *reinterpret_cast<uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3)) = pk;
          };
          if (half == 0) {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[64 + c], acc[64 + c + 1], acc[64 + c + 2], acc[64 + c + 3], acc[64 + c + 4], acc[64 + c + 5], acc[64 + c + 6], acc[64 + c + 7]);
          } else {
#pragma unroll
            for (int c = 0; c < 64; c += 8)
              export8(c, acc[c], acc[c + 1], acc[c + 2], acc[c + 3], acc[c + 4], acc[c + 5], acc[c + 6], acc[c + 7]);
          }
        }
        named_bar_sync<1, 256>();
        // half 0 quantizes columns [0,64): v = x1[0..63] (own) * silu(x2[0..63]) (from half 1)
        // half 1 quantizes columns [64,128): v = x1[64..127] (from half 0) * silu(x2[64..127]) (own)
        // v61: SwiGLU values are computed twice (amax pass, quantization pass) instead of being held in a
        // 64-float array; each half indexes `acc` with literal bases
        const int ch0 = (half == 0) ? 8 : 0;   // chunks written by the other half
        auto swiglu8 = [&](int c, float* a) {
          const int ch = ch0 + (c >> 3);
          const uint4 pk = *reinterpret_cast<const uint4*>(xch + r * HC + ((ch ^ (r & 15)) << 3));
          const uint32_t w4[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float o0 = bf16_to_f32((uint16_t)(w4[q] & 0xffff));
            const float o1 = bf16_to_f32((uint16_t)(w4[q] >> 16));
            if (half == 0) {  // own = x1 (cols 0..63), other = x2
              a[2 * q] = acc[c + 2 * q] * silu_fast(o0);
              a[2 * q + 1] = acc[c + 2 * q + 1] * silu_fast(o1);
            } else {          // own = x2 (cols 64..127), other = x1
              a[2 * q] = o0 * silu_fast(acc[64 + c + 2 * q]);
              a[2 * q + 1] = o1 * silu_fast(acc[64 + c + 2 * q + 1]);
            }
          }
        };
        float amax = 0.f;
#ifndef MOE_SWIGLU_TWICE
        // v90: SwiGLU computed once (64 values stay in the registers freed by the export); default (14107 955 -> 942 us)
        float sw[64];
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          swiglu8(c, sw + c);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(sw[c + q]));
        }
#else
#pragma unroll
        for (int c = 0; c < 64; c += 8) {
          float a[8];
          swiglu8(c, a);
#pragma unroll
          for (int q = 0; q < 8; ++q) amax = fmaxf(amax, fabsf(a[q]));
        }
#endif
        amaxx[r * 2 + half] = amax;
        named_bar_sync<1, 256>();
        if (r < u.rows_valid) {
          const float am = fmaxf(amaxx[r * 2], amaxx[r * 2 + 1]);
          const float scale = (am > 0.f) ? am * (U_F16 ? (1.0f / 128.0f) : (1.0f / 448.0f)) : 1.0f;
          const float inv = __frcp_rn(scale);
          uint8_t* dst = p.act + (size_t)pair * I + u.nt * HC + half * 64;
#pragma unroll
          for (int c = 0; c < 64; c += 16) {
            float a[16];
#ifndef MOE_SWIGLU_TWICE
#pragma unroll
            for (int q = 0; q < 16; ++q) a[q] = sw[c + q];
#else
            swiglu8(c, a);
            swiglu8(c + 8, a + 8);
#endif
            uint4 pk;
            pk.x = (uint32_t)pack_e4m3x2(a[0] * inv, a[1] * inv) | ((uint32_t)pack_e4m3x2(a[2] * inv, a[3] * inv) << 16);
            pk.y = (uint32_t)pack_e4m3x2(a[4] * inv, a[5] * inv) | ((uint32_t)pack_e4m3x2(a[6] * inv, a[7] * inv) << 16);
            pk.z = (uint32_t)pack_e4m3x2(a[8] * inv, a[9] * inv) | ((uint32_t)pack_e4m3x2(a[10] * inv, a[11] * inv) << 16);
            pk.w = (uint32_t)pack_e4m3x2(a[12] * inv, a[13] * inv) | ((uint32_t)pack_e4m3x2(a[14] * inv, a[15] * inv) << 16);
            *reinterpret_cast<uint4*>(dst + c) = pk;
          }
          if (half == 0) p.act_scale[(size_t)u.nt * p.p_cap + pair] = scale;  // 128-column blocks: [16][p_cap]
        }
        named_bar_sync<1, 256>();  // xch / amaxx may be overwritten by the next unit only after they were read
      } else {
        if (r < u.rows_valid) {
          const float w2 = p.pair_w[pair];
          const int tk = p.pair_token[pair];
          const int np = p.token_npair[tk];
          const bool rd = p.combine_atomic && np >= 2;  // v59: red-add into the zeroed output row
          uint16_t* dst = ((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * U_BN + col_base;
#pragma unroll
          for (int c = 0; c < HC; c += 8) {
            uint4 pk;
            pk.x = pack_bf16x2(acc[c] * w2, acc[c + 1] * w2);
            pk.y = pack_bf16x2(acc[c + 2] * w2, acc[c + 3] * w2);
            pk.z = pack_bf16x2(acc[c + 4] * w2, acc[c + 5] * w2);
            pk.w = pack_bf16x2(acc[c + 6] * w2, acc[c + 7] * w2);
            if (rd) red_v4_bf16x2(dst + c, pk);
            else *reinterpret_cast<uint4*>(dst + c) = pk;
          }
        }
      }
    }
  }
  if (tid == 0) TRACE(p, 7);
  __syncthreads();
  cluster_sync();  // the peer may still arrive on our barriers / read TMEM until both are done
  if (warp == 2) tmem_dealloc2(tmem_base, U2_NBUF * U_BN);
}


// ----------------------------------------------------------------------------------------------
// Kernel 5: combine per-pair rows into the token output (bf16).
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ void bf16x8_add(float* acc, const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    acc[2 * q] += __uint_as_float(w[q] << 16);
    acc[2 * q + 1] += __uint_as_float(w[q] & 0xffff0000u);
  }
}

// Small T: grid-stride over all 16-byte chunks; tokens with <= 1 local pair were already written
// (zero rows by routing, single-pair rows by the GEMM2 epilogue).
__global__ void __launch_bounds__(256) combine_small_kernel(int T, const int* __restrict__ token_pairs,
                                                            const uint16_t* __restrict__ pair_out,
                                                            uint16_t* __restrict__ out,
                                                            int* __restrict__ expert_count, int epoch) {
  pdl_wait();
  if (blockIdx.x == 0 && threadIdx.x < 2 * NLOCAL + 2 && expert_count) expert_count[threadIdx.x] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0 && expert_count) {
    expert_count[2 * NLOCAL + 2 + ((epoch + 1) & 1)] = 0;
    expert_count[2 * NLOCAL + 4] = 0;  // v56 work counter
  }
  constexpr int CPT = H / 8;
  const int total = T * CPT;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
    const int tok = idx / CPT;
    const int c = (idx - tok * CPT) * 8;
    const int4 pa = __ldg(reinterpret_cast<const int4*>(token_pairs + tok * TOPK));
    const int4 pb = __ldg(reinterpret_cast<const int4*>(token_pairs + tok * TOPK + 4));
    const int prs[TOPK] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    int valid[TOPK];
    int np = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (prs[k] >= 0) valid[np++] = prs[k];
    }
    if (np <= 1) continue;
    float acc[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) acc[q] = 0.f;
    if (np == 2) {
      const uint4 v0 = __ldg(reinterpret_cast<const uint4*>(pair_out + (size_t)valid[0] * H + c));
      const uint4 v1 = __ldg(reinterpret_cast<const uint4*>(pair_out + (size_t)valid[1] * H + c));
      bf16x8_add(acc, v0);
      bf16x8_add(acc, v1);
    } else {
      uint4 v[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k)
        if (k < np) v[k] = __ldg(reinterpret_cast<const uint4*>(pair_out + (size_t)valid[k] * H + c));
#pragma unroll
      for (int k = 0; k < TOPK; ++k)
        if (k < np) bf16x8_add(acc, v[k]);
    }
    uint4 o;
    o.x = pack_bf16x2(acc[0], acc[1]);
    o.y = pack_bf16x2(acc[2], acc[3]);
    o.z = pack_bf16x2(acc[4], acc[5]);
    o.w = pack_bf16x2(acc[6], acc[7]);
    *reinterpret_cast<uint4*>(out + (size_t)tok * H + c) = o;
  }
}

// Large T: one block per listed multi-pair token (bounded grid walking the list).
__global__ void __launch_bounds__(256) combine_kernel(int T, const int* __restrict__ token_pairs,
                                                      const uint16_t* __restrict__ pair_out,
                                                      uint16_t* __restrict__ out,
                                                      int* __restrict__ expert_count,
                                                      const int* __restrict__ multi_list, int epoch) {
  pdl_wait();
  // reset the counters used by the next call (expert counts, GEMM1 done, route ctr, done flag) and the
  // other parity slot of the multi-pair token counter; this call's slot is read by every block below
  if (blockIdx.x == 0 && threadIdx.x < 2 * NLOCAL + 2 && expert_count) expert_count[threadIdx.x] = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0 && expert_count) {
    expert_count[2 * NLOCAL + 2 + ((epoch + 1) & 1)] = 0;
    expert_count[2 * NLOCAL + 4] = 0;  // v56 work counter
  }
  const int cnt = __ldcg(expert_count + 2 * NLOCAL + 2 + (epoch & 1));
  constexpr int CPT = H / 8;  // 896 16-byte chunks per row
  for (int li = blockIdx.x; li < cnt; li += gridDim.x) {
    const int tok = __ldg(multi_list + li);
    const int4 pa = __ldg(reinterpret_cast<const int4*>(token_pairs + tok * TOPK));
    const int4 pb = __ldg(reinterpret_cast<const int4*>(token_pairs + tok * TOPK + 4));
    const int prs[TOPK] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    int valid[TOPK];
    int np = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (prs[k] >= 0) valid[np++] = prs[k];
    }
    // two chunks per iteration with all pair loads in flight
    for (int c0 = threadIdx.x; c0 < CPT; c0 += 2 * blockDim.x) {
      const int c1 = c0 + blockDim.x;
      uint4 v0[TOPK], v1[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (k < np) {
          v0[k] = __ldg(reinterpret_cast<const uint4*>(pair_out + (size_t)valid[k] * H + c0 * 8));
          if (c1 < CPT) v1[k] = __ldg(reinterpret_cast<const uint4*>(pair_out + (size_t)valid[k] * H + c1 * 8));
        }
      }
      float a0[8], a1[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) a0[q] = a1[q] = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        if (k < np) {
          bf16x8_add(a0, v0[k]);
          if (c1 < CPT) bf16x8_add(a1, v1[k]);
        }
      }
      uint4 o;
      o.x = pack_bf16x2(a0[0], a0[1]); o.y = pack_bf16x2(a0[2], a0[3]);
      o.z = pack_bf16x2(a0[4], a0[5]); o.w = pack_bf16x2(a0[6], a0[7]);
      *reinterpret_cast<uint4*>(out + (size_t)tok * H + c0 * 8) = o;
      if (c1 < CPT) {
        o.x = pack_bf16x2(a1[0], a1[1]); o.y = pack_bf16x2(a1[2], a1[3]);
        o.z = pack_bf16x2(a1[4], a1[5]); o.w = pack_bf16x2(a1[6], a1[7]);
        *reinterpret_cast<uint4*>(out + (size_t)tok * H + c1 * 8) = o;
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------
// Host side
// ----------------------------------------------------------------------------------------------
// Small-M config (decode / mid-size): 16-row tiles, 8 warps each owning 16 columns.
constexpr int S_MT = 16, S_NT = 128, S_WM = 1, S_WN = 8, S_STAGES = 5;
// Large-M configs (prefill): 64-row tiles (2x4 warps, 32x32 warp tiles) and 128-row tiles
// (2x4 warps, 64x32 warp tiles); one CTA per SM.
constexpr int L_MT = 64, L_NT = 128, L_WM = 2, L_WN = 4, L_STAGES = 6;
constexpr int H_MT = 128, H_NT = 128, H_WM = 2, H_WN = 4, H_STAGES = 4;
using CfgH1 = GemmCfg<H_MT, H_NT, H_WM, H_WN, H_STAGES, 128, true>;
using CfgH2 = GemmCfg<H_MT, H_NT, H_WM, H_WN, H_STAGES, 64, false>;
// Tiny-T config (decode): same tiles as small, but one CTA per SM with an 11-stage pipeline.
constexpr int X_MT = 16, X_NT = 128, X_WM = 1, X_WN = 8, X_STAGES = 11;
using CfgX1 = GemmCfg<X_MT, X_NT, X_WM, X_WN, X_STAGES, 128, true>;
using CfgX2 = GemmCfg<X_MT, X_NT, X_WM, X_WN, X_STAGES, 64, false>;
using CfgS1 = GemmCfg<S_MT, S_NT, S_WM, S_WN, S_STAGES, 128, true>;
using CfgS2 = GemmCfg<S_MT, S_NT, S_WM, S_WN, S_STAGES, 64, false>;
using CfgL1 = GemmCfg<L_MT, L_NT, L_WM, L_WN, L_STAGES, 128, true>;
using CfgL2 = GemmCfg<L_MT, L_NT, L_WM, L_WN, L_STAGES, 64, false>;

// Swap-AB small-path configs (v31): 8-token tiles; 2 CTAs/SM with SW_STAGES stages (small), 1 CTA/SM
// with SX_STAGES stages (tiny).
constexpr int SW_MT = 8, SW_STAGES = 6, SX_STAGES = 11;
using CfgW1 = SwCfg<SW_MT, SW_STAGES, 128, true>;
using CfgW2 = SwCfg<SW_MT, SW_STAGES, 64, false>;
using CfgY1 = SwCfg<SW_MT, SX_STAGES, 128, true>;
using CfgY2 = SwCfg<SW_MT, SX_STAGES, 64, false>;

int g_num_sms = 0;
bool g_attr_set = false;

typedef CUresult (*PFN_encodeTiled_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                      const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                      const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                      CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
PFN_encodeTiled_t g_encode = nullptr;

// ----------------------------------------------------------------------------------------------
// v83: hardware block-scaled UMMA path (tcgen05.mma.kind::mxf8f6f4.block_scale with UE8M0 scale factors in
// TMEM). The accumulator stays in TMEM for the whole K loop (no per-K-block fp32 promotion): the power-of-two
// part of every fp32 block scale goes to the hardware scale factors and the mantissa residual (with the sign)
// is folded into the fp8 A tile, re-quantized per 128-column N block by the convert warps:
//   sa = +-2^ea * ra, sb_j = +-2^eb_j * rb_j (ra, rb in [1,2))
//   A'_j[m][k] = fp8(A[m][k] * ra[m] * rb_j * sign / 4),  SFA[m] = 2^(ea+2),  SFB_j[n] = 2^eb_j
//   D[m][n] += SFA[m] * SFB_j[n] * sum_k A'_j[m][k] * B[n][k]   (exact apart from one fp8 rounding of A')
// CTA tile 128 x 256 (two 128-column N blocks, two MMAs per 32-K chunk) x 128 K. smem stage = raw A tile
// (converted in place into A'_0) | A'_1 | B (64 KB), 3 stages. Warp roles: 0 TMA, 1 MMA issuer, 2 TMEM
// allocator, 3 output zero-fill (v80, GEMM1), 4..7 convert (thread <-> row), 8..11 epilogue (thread <-> row).
// TMEM: columns [0,256) accumulator, [256, 256+4*MX_STAGES) SFA (4 columns per stage: rows 32c..32c+31 in
// column c, lane = row % 32 of the subpartition), then SFB (8 columns per stage: 4 per N block).
// ----------------------------------------------------------------------------------------------
constexpr int MX_STAGES = 3;
constexpr int MX_A_BYTES = U_BM * U_BK;                      // 16 KB raw A / A'_0
constexpr int MX_STAGE_BYTES = 2 * MX_A_BYTES + U_BN * U_BK;  // A'_0 | A'_1 | B = 64 KB
constexpr int MX_SMEM_BYTES = 1024 + MX_STAGES * MX_STAGE_BYTES + U_MISC_BYTES;
constexpr int MX_SFA_COL = 256;
constexpr int MX_SFB_COL = 256 + 4 * MX_STAGES;
constexpr int MX_TMEM_COLS = 512;
constexpr int MX_CONV_ARRIVALS = 128;  // every convert thread arrives (release of its own smem / TMEM writes)
constexpr int MX_EPI_ARRIVALS = 128;

__device__ __forceinline__ void umma_mx(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
                                        uint32_t sfa, uint32_t sfb, uint32_t accum) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::"r"(d_tmem),
      "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accum));
}
__device__ __forceinline__ void tmem_st_32x32b_x1(uint32_t taddr, uint32_t v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n" ::"r"(taddr), "r"(v));
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t v) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"((uint16_t)v));
  return r;
}
__device__ __forceinline__ uint32_t cvt_f16x2_e4m3x2(uint32_t v) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(v));
  return (uint32_t)r;
}
__device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ uint32_t f16x2_bcast(float q) {
  const __half h = __float2half_rn(q);
  const uint32_t b = (uint32_t)__half_as_ushort(h);
  return b | (b << 16);
}
// one 32-bit word of four e4m3 values -> the A'_0 / A'_1 words (q0h / q1h: f16x2 broadcast residual factors)
__device__ __forceinline__ void mx_convert_word(uint32_t w, uint32_t q0h, uint32_t q1h, uint32_t& o0, uint32_t& o1) {
  const uint32_t f0 = cvt_e4m3x2_f16x2(w & 0xffffu);
  const uint32_t f1 = cvt_e4m3x2_f16x2(w >> 16);
  o0 = cvt_f16x2_e4m3x2(mul_f16x2(f0, q0h)) | (cvt_f16x2_e4m3x2(mul_f16x2(f1, q0h)) << 16);
  o1 = cvt_f16x2_e4m3x2(mul_f16x2(f0, q1h)) | (cvt_f16x2_e4m3x2(mul_f16x2(f1, q1h)) << 16);
}

template <bool GEMM1>
__global__ void __launch_bounds__(U_THREADS, 1)
    umma_mx_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, const GemmParams p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t pad = (1024 - (raw & 1023)) & 1023;
  const uint32_t sbase = raw + pad;
  uint8_t* misc = smem_raw + pad + MX_STAGES * MX_STAGE_BYTES;
  int* s_off = reinterpret_cast<int*>(misc);  // [33]
  int* s_mtp = s_off + NLOCAL + 1;            // [33]
  int* s_end = s_mtp + NLOCAL + 1;            // [32]
  const uint32_t full = sbase + MX_STAGES * MX_STAGE_BYTES + 512;
  const uint32_t empty = full + MX_STAGES * 8;
  const uint32_t conv = empty + MX_STAGES * 8;
  const uint32_t acc_full = conv + MX_STAGES * 8;
  const uint32_t acc_empty = acc_full + 8;
  const uint32_t dbg_bar = acc_empty + 8;  // MOE_MX_DBG_SYNC: MMA completion barrier
  uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(misc + 512 + (3 * MX_STAGES + 3) * 8);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < MX_STAGES; ++s) {
      mbar_init(full + s * 8, 1);
      mbar_init(empty + s * 8, 1);
      mbar_init(conv + s * 8, MX_CONV_ARRIVALS);
    }
    mbar_init(acc_full, 1);
    mbar_init(acc_empty, MX_EPI_ARRIVALS);
    mbar_init(dbg_bar, 1);
    fence_barrier_init();
    tma_prefetch_desc(&tmA);
    tma_prefetch_desc(&tmB);
  }
  if (warp == 2) {
    tmem_alloc(smem_u32(tmem_ptr), MX_TMEM_COLS);
    tmem_relinquish();
  }
  pdl_wait();
  pdl_launch_dependents();
  load_expert_tables<U_BM>(p, s_off, s_mtp, s_end);
  __syncthreads();
  const uint32_t tmem_base = *tmem_ptr;
  const int mtile_total = s_mtp[NLOCAL];
  const int total_units = mtile_total * p.n_tiles;
  const int my_units = ((int)blockIdx.x < total_units) ? (total_units - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
  const int KB = p.K / 128;

  if (warp == 0) {
    if (lane == 0) {
      // =========================== TMA producer =============================================
      int stage = 0;
      uint32_t phase = 1;
      const uint64_t pol_w = p.b_evict_first ? l2_policy_evict_first() : l2_policy_evict_normal();
      const uint64_t pol_a = l2_policy_evict_last();
      for (int ui = 0; ui < my_units; ++ui) {
        const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
        int b_row0, b_row1;
        if (GEMM1) {
          b_row0 = u.e * p.N_total + u.nt * (U_BN / 2);
          b_row1 = b_row0 + I;
        } else {
          b_row0 = u.e * p.N_total + u.nt * U_BN;
          b_row1 = b_row0 + (U_BN / 2);
        }
        for (int kb = 0; kb < KB; ++kb) {
          mbar_wait(empty + stage * 8, phase);
          const uint32_t st = sbase + stage * MX_STAGE_BYTES;
          const uint32_t fb = full + stage * 8;
          mbar_arrive_expect_tx(fb, MX_A_BYTES + U_BN * U_BK);
          tma_load_2d_hint(st, &tmA, fb, kb * U_BK, u.row0, pol_a);
          tma_load_2d_hint(st + 2 * MX_A_BYTES, &tmB, fb, kb * U_BK, b_row0, pol_w);
          tma_load_2d_hint(st + 2 * MX_A_BYTES + (U_BN / 2) * U_BK, &tmB, fb, kb * U_BK, b_row1, pol_w);
          if (++stage == MX_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      // =========================== MMA issuer ===============================================
      // block-scaled instruction descriptor: E4M3 x E4M3 (K-major), N = 128, UE8M0 scale factors (bit 23),
      // M = 128, scale-factor ids 0 (byte 0 of the TMEM scale words)
      const uint32_t idesc = (((uint32_t)(U_BN / 2) >> 3) << 17) | (1u << 23) | (((uint32_t)U_BM >> 4) << 24);
      int stage = 0;
      uint32_t phase = 0;
      uint32_t dbg_phase = 0;
      (void)dbg_phase;
      for (int ui = 0; ui < my_units; ++ui) {
        mbar_wait(acc_empty, ((uint32_t)ui & 1u) ^ 1u);
        tc_fence_after();
        for (int kb = 0; kb < KB; ++kb) {
#ifdef MOE_MX_DBG_FULL
          mbar_wait(full + stage * 8, phase);
#endif
          mbar_wait(conv + stage * 8, phase);
          tc_fence_after();
          const uint32_t st = sbase + stage * MX_STAGE_BYTES;
          const uint64_t adesc0 = umma_desc(st), adesc1 = umma_desc(st + MX_A_BYTES);
          const uint64_t bdesc0 = umma_desc(st + 2 * MX_A_BYTES), bdesc1 = umma_desc(st + 2 * MX_A_BYTES + (U_BN / 2) * U_BK);
          const uint32_t sfa = tmem_base + MX_SFA_COL + 4 * stage;
          const uint32_t sfb0 = tmem_base + MX_SFB_COL + 8 * stage, sfb1 = sfb0 + 4;
#pragma unroll
          for (int c = 0; c < U_BK / 32; ++c) {
            const uint32_t koff = (uint32_t)(c * 32) >> 4;
            const uint32_t accum = (kb > 0 || c > 0) ? 1u : 0u;
            umma_mx(tmem_base, adesc0 + koff, bdesc0 + koff, idesc, sfa, sfb0, accum);
            umma_mx(tmem_base + (U_BN / 2), adesc1 + koff, bdesc1 + koff, idesc, sfa, sfb1, accum);
          }
          umma_commit(empty + stage * 8);
          if (kb == KB - 1) umma_commit(acc_full);
#ifdef MOE_MX_DBG_SYNC
          umma_commit(dbg_bar);
          mbar_wait(dbg_bar, dbg_phase);
          dbg_phase ^= 1u;
#endif
          if (++stage == MX_STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else if (warp == 3) {
    if (GEMM1 && p.zero_rows) {
      // v80: output rows without a local expert, and multi-pair rows that receive red-adds, are zeroed here
      for (int t = (int)blockIdx.x; t < p.T; t += (int)gridDim.x) {
        const int np = __ldg(p.token_npair + t);
        if (np == 0 || (p.combine_atomic && np >= 2)) {
          uint4* orow = reinterpret_cast<uint4*>(p.out + (size_t)t * H);
#pragma unroll 4
          for (int c = lane; c < H / 8; c += 32) orow[c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  } else if (warp >= 4 && warp < 8) {
    // =========================== convert warps (thread <-> row) ===========================
    const int cw = warp - 4;
    const int r = cw * 32 + lane;
    const uint32_t lane_base = ((uint32_t)(cw * 32)) << 16;
    int stage = 0;
    uint32_t phase = 0;
    for (int ui = 0; ui < my_units; ++ui) {
      const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
      const int ar = (r < u.rows_valid) ? (u.row0 + r) : u.row0;
      const float* sa_ptr = p.A_scale + ar;
      int nbx0, nbx1;
      if (GEMM1) { nbx0 = u.nt; nbx1 = I / 128 + u.nt; }
      else { nbx0 = u.nt * 2; nbx1 = nbx0 + 1; }
      const float* sw0 = p.W_scale + ((size_t)u.e * (p.N_total / 128) + nbx0) * KB;
      const float* sw1 = p.W_scale + ((size_t)u.e * (p.N_total / 128) + nbx1) * KB;
      // rotating 4-deep prefetch of the per-K-block scales
      float a0, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0, b1 = 0.f, b2 = 0.f, b3 = 0.f, c0, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      a0 = __ldg(sa_ptr); b0 = __ldg(sw0); c0 = __ldg(sw1);
      if (KB > 1) { a1 = __ldg(sa_ptr + (size_t)1 * p.a_rows_stride); b1 = __ldg(sw0 + 1); c1 = __ldg(sw1 + 1); }
      if (KB > 2) { a2 = __ldg(sa_ptr + (size_t)2 * p.a_rows_stride); b2 = __ldg(sw0 + 2); c2 = __ldg(sw1 + 2); }
      if (KB > 3) { a3 = __ldg(sa_ptr + (size_t)3 * p.a_rows_stride); b3 = __ldg(sw0 + 3); c3 = __ldg(sw1 + 3); }
      for (int kb = 0; kb < KB; ++kb) {
        const float sa = a0, sb0 = b0, sb1 = c0;
        a0 = a1; a1 = a2; a2 = a3; b0 = b1; b1 = b2; b2 = b3; c0 = c1; c1 = c2; c2 = c3;
        if (kb + 4 < KB) {
          a3 = __ldg(sa_ptr + (size_t)(kb + 4) * p.a_rows_stride);
          b3 = __ldg(sw0 + kb + 4);
          c3 = __ldg(sw1 + kb + 4);
        }
        // exponent / residual split (biased exponent field = UE8M0 of 2^floor(log2|s|))
        const uint32_t ba = __float_as_uint(sa), bb0 = __float_as_uint(sb0), bb1 = __float_as_uint(sb1);
        const float ra = __uint_as_float((ba & 0x7fffffu) | 0x3f800000u);
        const float rb0 = __uint_as_float((bb0 & 0x7fffffu) | 0x3f800000u);
        const float rb1 = __uint_as_float((bb1 & 0x7fffffu) | 0x3f800000u);
        float q0 = ra * rb0 * 0.25f, q1 = ra * rb1 * 0.25f;
        if ((ba ^ bb0) & 0x80000000u) q0 = -q0;
        if ((ba ^ bb1) & 0x80000000u) q1 = -q1;
        const uint32_t sfa_w = min(((ba >> 23) & 0xffu) + 2u, 254u);
        const uint32_t sfb0_w = (bb0 >> 23) & 0xffu, sfb1_w = (bb1 >> 23) & 0xffu;
        const uint32_t q0h = f16x2_bcast(q0), q1h = f16x2_bcast(q1);
        mbar_wait(full + stage * 8, phase);
        // scale factors of this stage: SFA column cw (rows 32cw..32cw+31), SFB columns 0..3 of each N block
        tmem_st_32x32b_x1(tmem_base + lane_base + MX_SFA_COL + 4 * stage + cw, sfa_w);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          tmem_st_32x32b_x1(tmem_base + lane_base + MX_SFB_COL + 8 * stage + c, sfb0_w);
          tmem_st_32x32b_x1(tmem_base + lane_base + MX_SFB_COL + 8 * stage + 4 + c, sfb1_w);
        }
        // row r of the raw A tile (128B-swizzled: chunk c at r*128 + ((c ^ (r & 7)) << 4)) -> A'_0 in place, A'_1
        uint8_t* arow = smem_raw + pad + stage * MX_STAGE_BYTES + r * 128;
#ifdef MOE_MX_DBG_DUMP
        // dump of unit 0 / K-block 0 of CTA 0: per row 32 raw words, 32 A'_0 words, 32 A'_1 words, 8 scalars
        const bool dumping = (blockIdx.x == 0 && ui == 0 && kb == 0);
        uint32_t* dump = reinterpret_cast<uint32_t*>(p.partial) + r * 104;
        if (dumping) {
          dump[96] = __float_as_uint(sa); dump[97] = __float_as_uint(sb0); dump[98] = __float_as_uint(sb1);
          dump[99] = __float_as_uint(q0); dump[100] = __float_as_uint(q1); dump[101] = sfa_w; dump[102] = sfb0_w; dump[103] = (uint32_t)u.row0;
        }
#endif
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const int off = (c ^ (r & 7)) << 4;
          const uint4 v = *reinterpret_cast<const uint4*>(arow + off);
          uint4 o0, o1;
#ifdef MOE_MX_ABL_NOCONV
          o0 = v; o1 = v;  // ablation: no conversion arithmetic (wrong results)
          (void)q0h; (void)q1h;
#else
          mx_convert_word(v.x, q0h, q1h, o0.x, o1.x);
          mx_convert_word(v.y, q0h, q1h, o0.y, o1.y);
          mx_convert_word(v.z, q0h, q1h, o0.z, o1.z);
          mx_convert_word(v.w, q0h, q1h, o0.w, o1.w);
#endif
#ifdef MOE_MX_DBG_DUMP
          if (dumping) {
            *reinterpret_cast<uint4*>(dump + c * 4) = v;
            *reinterpret_cast<uint4*>(dump + 32 + c * 4) = o0;
            *reinterpret_cast<uint4*>(dump + 64 + c * 4) = o1;
          }
#endif
          *reinterpret_cast<uint4*>(arow + off) = o0;
          *reinterpret_cast<uint4*>(arow + MX_A_BYTES + off) = o1;
        }
        tmem_wait_st();
        fence_proxy_async_smem();
        tc_fence_before();
        mbar_arrive(conv + stage * 8);
        if (++stage == MX_STAGES) {
          stage = 0;
          phase ^= 1;
        }
      }
    }
  } else if (warp >= 8) {
    // =========================== epilogue warps (thread <-> row) ==========================
    const int ew = warp - 8;
    const int r = ew * 32 + lane;
    const uint32_t tb = tmem_base + (((uint32_t)(ew * 32)) << 16);
    for (int ui = 0; ui < my_units; ++ui) {
      const UnitInfo u = decode_unit<U_BM, true>(blockIdx.x + ui * gridDim.x, mtile_total, s_off, s_mtp, p.n_tiles, s_end);
      const int pair = u.row0 + r;
      const bool rv = r < u.rows_valid;
      mbar_wait(acc_full, (uint32_t)ui & 1u);
      tc_fence_after();
      if constexpr (GEMM1) {
        // pass 1: v = x1 * silu(x2) (columns [0,128) / [128,256)) -> bf16 pairs in registers, row amax
        uint32_t vb[64];
        float amax = 0.f;
#pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {
          uint32_t x1[32], x2[32];
          tmem_ld_32x32b_x32(tb + c0 * 32, x1);
          tmem_ld_32x32b_x32(tb + (U_BN / 2) + c0 * 32, x2);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const float v0 = __uint_as_float(x1[i]) * silu_fast(__uint_as_float(x2[i]));
            const float v1 = __uint_as_float(x1[i + 1]) * silu_fast(__uint_as_float(x2[i + 1]));
            amax = fmaxf(amax, fmaxf(fabsf(v0), fabsf(v1)));
            vb[c0 * 16 + (i >> 1)] = pack_bf16x2(v0, v1);
          }
        }
        tc_fence_before();
        mbar_arrive(acc_empty);
        if (rv) {
          const float scale = (amax > 0.f) ? amax * (1.0f / 448.0f) : 1.0f;
          const float inv = __frcp_rn(scale);
          uint8_t* dst = p.act + (size_t)pair * I + u.nt * (U_BN / 2);
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            float a[16];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              a[2 * q] = bf16_to_f32((uint16_t)(vb[c * 8 + q] & 0xffffu)) * inv;
              a[2 * q + 1] = bf16_to_f32((uint16_t)(vb[c * 8 + q] >> 16)) * inv;
            }
            uint4 pk;
            pk.x = (uint32_t)pack_e4m3x2(a[0], a[1]) | ((uint32_t)pack_e4m3x2(a[2], a[3]) << 16);
            pk.y = (uint32_t)pack_e4m3x2(a[4], a[5]) | ((uint32_t)pack_e4m3x2(a[6], a[7]) << 16);
            pk.z = (uint32_t)pack_e4m3x2(a[8], a[9]) | ((uint32_t)pack_e4m3x2(a[10], a[11]) << 16);
            pk.w = (uint32_t)pack_e4m3x2(a[12], a[13]) | ((uint32_t)pack_e4m3x2(a[14], a[15]) << 16);
            *reinterpret_cast<uint4*>(dst + c * 16) = pk;
          }
          p.act_scale[(size_t)u.nt * p.p_cap + pair] = scale;  // 128-column blocks: [16][p_cap]
        }
      } else {
        float w2 = 0.f;
        bool rd = false;
        uint16_t* dst = nullptr;
        if (rv) {
          w2 = p.pair_w[pair];
          const int tk = p.pair_token[pair];
          const int np = p.token_npair[tk];
          rd = p.combine_atomic && np >= 2;  // v59: red-add into the zeroed output row
          dst = ((np == 1 || rd) ? (p.out + (size_t)tk * H) : (p.pair_out + (size_t)pair * H)) + u.nt * U_BN;
#ifdef MOE_MX_DBG_DUMP
          if (tk == 0 && u.nt == 0) {
            uint32_t* wr = reinterpret_cast<uint32_t*>(p.partial) + 128 * 104 + 128 * 256 + 64;
            const int slot = atomicAdd(reinterpret_cast<int*>(wr), 1);
            if (slot < 16) {
              wr[8 + slot * 6 + 0] = blockIdx.x; wr[8 + slot * 6 + 1] = ui; wr[8 + slot * 6 + 2] = r;
              wr[8 + slot * 6 + 3] = u.e; wr[8 + slot * 6 + 4] = u.row0; wr[8 + slot * 6 + 5] = (uint32_t)pair;
            }
          }
#endif
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t v[128];
#pragma unroll
          for (int c0 = 0; c0 < 4; ++c0) tmem_ld_32x32b_x32(tb + half * (U_BN / 2) + c0 * 32, v + c0 * 32);
          tmem_wait_ld();
#ifdef MOE_MX_DBG_DUMP
          if (blockIdx.x == 0 && ui == 0) {
            float* dumpacc = p.partial + 128 * 104 + (size_t)r * 256 + half * 128;
#pragma unroll
            for (int c = 0; c < 128; ++c) dumpacc[c] = __uint_as_float(v[c]);
          }
#endif
          if (half == 1) {
            tc_fence_before();
            mbar_arrive(acc_empty);
          }
          if (rv) {
#pragma unroll
            for (int c = 0; c < 128; c += 8) {
              uint4 pk;
              pk.x = pack_bf16x2(__uint_as_float(v[c]) * w2, __uint_as_float(v[c + 1]) * w2);
              pk.y = pack_bf16x2(__uint_as_float(v[c + 2]) * w2, __uint_as_float(v[c + 3]) * w2);
              pk.z = pack_bf16x2(__uint_as_float(v[c + 4]) * w2, __uint_as_float(v[c + 5]) * w2);
              pk.w = pack_bf16x2(__uint_as_float(v[c + 6]) * w2, __uint_as_float(v[c + 7]) * w2);
              if (rd) red_v4_bf16x2(dst + half * (U_BN / 2) + c, pk);
              else *reinterpret_cast<uint4*>(dst + half * (U_BN / 2) + c) = pk;
#ifdef MOE_MX_DBG_DUMP
              if (blockIdx.x == 0 && ui == 0 && r == 0 && half == 0 && c == 0) {
                uint32_t* d2 = reinterpret_cast<uint32_t*>(p.partial) + 128 * 104 + 128 * 256;
                d2[0] = (uint32_t)(reinterpret_cast<uintptr_t>(dst) & 0xffffffffu);
                d2[1] = (uint32_t)(reinterpret_cast<uintptr_t>(dst) >> 32);
                d2[2] = (uint32_t)(reinterpret_cast<uintptr_t>(p.out) & 0xffffffffu);
                d2[3] = (uint32_t)(reinterpret_cast<uintptr_t>(p.out) >> 32);
                d2[4] = (uint32_t)p.pair_token[pair]; d2[5] = (uint32_t)p.token_npair[p.pair_token[pair]]; d2[6] = __float_as_uint(w2); d2[7] = rd ? 1u : 0u;
                d2[8] = pk.x; d2[9] = pk.y; d2[10] = pk.z; d2[11] = pk.w;
                __threadfence();
                uint4 rb;
                asm volatile("ld.global.cv.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(rb.x), "=r"(rb.y), "=r"(rb.z), "=r"(rb.w) : "l"(dst));
                d2[12] = rb.x; d2[13] = rb.y; d2[14] = rb.z; d2[15] = rb.w;
                d2[16] = (uint32_t)u.nt; d2[17] = (uint32_t)u.row0; d2[18] = (uint32_t)u.rows_valid; d2[19] = (uint32_t)u.e;
              }
#endif
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem_base, MX_TMEM_COLS);
}

bool make_tmap(CUtensorMap* m, const void* ptr, uint64_t rows, uint64_t k, uint32_t box_rows) {
  cuuint64_t gdim[2] = {k, rows};
  cuuint64_t gstride[1] = {k};
  cuuint32_t box[2] = {128, box_rows};
  cuuint32_t es[2] = {1, 1};
  CUresult r = g_encode(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), gdim, gstride, box,
                        es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
#if defined(MOE_L2PROMO) && MOE_L2PROMO == 0
                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
#elif defined(MOE_L2PROMO) && MOE_L2PROMO == 1
                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
#else
                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
#endif
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

template <int MT, int NT, int WM, int WN, int ST, int ASK, bool G1>
void set_smem_attr() {
  using C = GemmCfg<MT, NT, WM, WN, ST, ASK, G1>;
  cudaFuncSetAttribute(grouped_gemm_kernel<MT, NT, WM, WN, ST, ASK, G1>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES);
}

template <int MT, int ST, int ASK, bool G1>
void launch_swab(const CUtensorMap& tmap, const GemmParams& p, int grid, cudaStream_t stream) {
  using C = SwCfg<MT, ST, ASK, G1>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(C::NTHREADS);
  cfg.dynamicSmemBytes = C::SMEM_BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, swab_gemm_kernel<MT, ST, ASK, G1>, tmap, p);
}

template <int MT, int ST>
void launch_fused(const CUtensorMap& tm1, const CUtensorMap& tm2, const FusedParams& fp, int grid, cudaStream_t stream) {
  using C = SwCfg<MT, ST, 64, true>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(C::NTHREADS);
  cfg.dynamicSmemBytes = C::SMEM_BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, swab_fused_kernel<MT, ST>, tm1, tm2, fp);
}

template <int MT, int NT, int WM, int WN, int ST, int ASK, bool G1>
void launch_gemm(const CUtensorMap& tmap, const GemmParams& p, int grid, cudaStream_t stream) {
  using C = GemmCfg<MT, NT, WM, WN, ST, ASK, G1>;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(C::NTHREADS);
  cfg.dynamicSmemBytes = C::SMEM_BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, grouped_gemm_kernel<MT, NT, WM, WN, ST, ASK, G1>, tmap, p);
}

}  // namespace

extern "C" int moe_run(const void* routing_logits, const void* routing_bias, const void* hidden,
                       const void* hidden_scale, const void* w1, const void* w1s, const void* w2,
                       const void* w2s, int T, int local_off, float scale_factor, void* output,
                       // workspace
                       void* blk_count, void* flags, void* expert_off, void* mtile_prefix,
                       void* pair_token, void* pair_w, void* token_pairs, void* act, void* act_scale,
                       void* pair_out, void* partial, void* counters, void* trace, void* act_in,
                       void* act_in_scale, int p_cap, int tpw, int route_blocks, int large, int epoch,
                       int split_mult, int fixed_cap, void* expert_count, int fused_small, void* blk_pre,
                       void* token_npair, void* multi_list, int u2_enable, int sched_target, int route_inkernel,
                       int combine_atomic, int route_coop, void* route_scratch, int a_gather, int route_split,
                       int sched_wc, int s1_mult, int s2_mult, int route_cta, int early_w, int spec_pf, int a_box, int g2_n128, int g2_pf, int pf_ahead, int pf_pol, int mx_mode, int mx_min_t, int spec_w, void* stream_v) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);
#ifdef MOE_TRACE
  if (trace) {
    unsigned long long* rt = (unsigned long long*)trace + 2 * 64 * 16;
    cudaMemcpyToSymbol(g_route_trace, &rt, sizeof(rt));
  }
#endif
  if (!g_attr_set) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
    set_smem_attr<S_MT, S_NT, S_WM, S_WN, S_STAGES, 128, true>();
    set_smem_attr<S_MT, S_NT, S_WM, S_WN, S_STAGES, 64, false>();
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SW_STAGES, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CfgW1::SMEM_BYTES);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SW_STAGES, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, CfgW2::SMEM_BYTES);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SX_STAGES, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CfgY1::SMEM_BYTES);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SX_STAGES, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, CfgY2::SMEM_BYTES);
    cudaFuncSetAttribute(swab_fused_kernel<SW_MT, SW_STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SwCfg<SW_MT, SW_STAGES, 64, true>::SMEM_BYTES);
    cudaFuncSetAttribute(swab_fused_kernel<SW_MT, SX_STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SwCfg<SW_MT, SX_STAGES, 64, true>::SMEM_BYTES);
    cudaFuncSetAttribute(swab_fused_kernel<SW_MT, SW_STAGES>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(swab_fused_kernel<SW_MT, SX_STAGES>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    set_smem_attr<L_MT, L_NT, L_WM, L_WN, L_STAGES, 128, true>();
    set_smem_attr<L_MT, L_NT, L_WM, L_WN, L_STAGES, 64, false>();
    set_smem_attr<X_MT, X_NT, X_WM, X_WN, X_STAGES, 128, true>();
    set_smem_attr<X_MT, X_NT, X_WM, X_WN, X_STAGES, 64, false>();
    set_smem_attr<H_MT, H_NT, H_WM, H_WN, H_STAGES, 128, true>();
    set_smem_attr<H_MT, H_NT, H_WM, H_WN, H_STAGES, 64, false>();
    cudaFuncSetAttribute(umma_gemm_kernel<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, U_SMEM_BYTES);
    cudaFuncSetAttribute(umma_gemm_kernel<128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, U_SMEM_BYTES);
    cudaFuncSetAttribute(umma_n128_kernel<128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, V_SMEM_BYTES);
    cudaFuncSetAttribute(umma_n128_kernel<128, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma_mx_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, MX_SMEM_BYTES);
    cudaFuncSetAttribute(umma_mx_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, MX_SMEM_BYTES);
    cudaFuncSetAttribute(umma_mx_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma_mx_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma2_gemm_kernel<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, U2_SMEM_BYTES);
    cudaFuncSetAttribute(umma2_gemm_kernel<128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, U2_SMEM_BYTES);
    cudaFuncSetAttribute(umma2_gemm_kernel<128, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma2_gemm_kernel<128, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(routing_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(combine_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(combine_small_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SW_STAGES, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SW_STAGES, 64, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SX_STAGES, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(swab_gemm_kernel<SW_MT, SX_STAGES, 64, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma_gemm_kernel<128, true>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(umma_gemm_kernel<128, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
    if (fn == nullptr) return -100;
    g_encode = reinterpret_cast<PFN_encodeTiled_t>(fn);
    g_attr_set = true;
  }
  // large: 0 = small cfg, 2 = tiny cfg, 1 = tcgen05 path (MT=128), 3 = mma.sync MT=64 fallback,
  // 4 = mma.sync MT=128 fallback
  // large: 0 = swap-AB small cfg, 2 = swap-AB tiny cfg, 5/6 = legacy MT=16 small/tiny cfgs
  const int MT = (large == 1) ? U_BM : ((large == 3) ? L_MT : ((large == 4) ? H_MT : ((large == 5 || large == 6) ? S_MT : SW_MT)));
  CUtensorMap tm1, tm2, tmA1, tmA2, tmB1u, tmB2u, tmA1_64, tmA1_32, tmA2_64, tmA2_32;
  if (!make_tmap(&tm1, w1, (uint64_t)NLOCAL * 2 * I, H, 64)) return -101;
  if (!make_tmap(&tm2, w2, (uint64_t)NLOCAL * H, I, 128)) return -102;
  if (large == 1) {
    if (!a_gather && !make_tmap(&tmA1, act_in, (uint64_t)p_cap + 128, H, 128)) return -103;
    if (a_gather == 2 && !make_tmap(&tmA1, hidden, (uint64_t)T, H, 1)) return -111;  // v84: gather4 map (1-row boxes)
    if (!a_gather && !make_tmap(&tmA1_64, act_in, (uint64_t)p_cap + 128, H, 64)) return -107;
    if (!a_gather && !make_tmap(&tmA1_32, act_in, (uint64_t)p_cap + 128, H, 32)) return -108;
    if (a_gather) tmA1_64 = tmA1_32 = tm1;
    if (a_gather == 1) tmA1 = tm1;  // unused by the cp.async-gathering GEMM1 (any valid map); mode 2 keeps the gather4 map
    if (!make_tmap(&tmA2, act, (uint64_t)p_cap + 128, I, 128)) return -104;
    if (!make_tmap(&tmA2_64, act, (uint64_t)p_cap + 128, I, 64)) return -109;
    if (!make_tmap(&tmA2_32, act, (uint64_t)p_cap + 128, I, 32)) return -110;
    if (!make_tmap(&tmB1u, w1, (uint64_t)NLOCAL * 2 * I, H, 128)) return -105;
    if (!make_tmap(&tmB2u, w2, (uint64_t)NLOCAL * H, I, 256)) return -106;
  }

  const bool inkernel = route_inkernel && (large == 0 || large == 2) && fused_small && T < 512;
  const bool split = (large == 1) && route_split;
  if (split) {
    const int B1 = (T + ROUTE_WARPS - 1) / ROUTE_WARPS;
    route_topk_kernel<<<B1, ROUTE_WARPS * 32, 0, stream>>>(
        (const float*)routing_logits, (const uint16_t*)routing_bias, T, local_off, scale_factor, epoch,
        (int*)blk_count, (int*)expert_count, (int*)token_npair, (uint16_t*)output, (int*)multi_list, combine_atomic,
        (int4*)route_scratch);
    cudaLaunchConfig_t rc = {};
    rc.blockDim = dim3(256);
    rc.stream = stream;
    cudaLaunchAttribute rattr[1];
    rattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    rattr[0].val.programmaticStreamSerializationAllowed = 1;
    rc.attrs = rattr;
    rc.numAttrs = 1;
    rc.gridDim = dim3(1);
    rc.blockDim = dim3(32);
    cudaLaunchKernelEx(&rc, route_prefix_kernel, MT, (const int*)expert_count, (int*)expert_off, (int*)mtile_prefix);
    rc.gridDim = dim3(B1);
    rc.blockDim = dim3(ROUTE_WARPS * 32);
    cudaLaunchKernelEx(&rc, route_scatter_kernel, T, (const int4*)route_scratch, (const int*)expert_off,
                       (int*)expert_count, (const int*)token_npair, (uint16_t*)output, combine_atomic,
                       (int*)pair_token, (float*)pair_w, (int*)token_pairs,
                       (const uint8_t*)hidden, (const float*)hidden_scale, (uint8_t*)act_in, (float*)act_in_scale,
                       a_gather ? 0 : 1, p_cap,
#ifdef MOE_ZF_SCATTER
                       1);
#else
                       0);
#endif
  } else if (!inkernel)
  routing_kernel<<<route_blocks, ROUTE_WARPS * 32, 0, stream>>>(
      (const float*)routing_logits, (const uint16_t*)routing_bias, T, tpw, local_off, scale_factor,
      MT, epoch, (int*)blk_count, (int*)flags, (int*)expert_off, (int*)mtile_prefix,
      (int*)pair_token, (float*)pair_w, (int*)token_pairs, (const uint8_t*)hidden,
      (const float*)hidden_scale, (uint8_t*)act_in, (float*)act_in_scale, (large == 1 && !a_gather) ? 1 : 0, p_cap,
      fixed_cap, (int*)expert_count, (int*)blk_pre, (int*)token_npair, (uint16_t*)output, (int*)multi_list,
      combine_atomic);

  GemmParams p1{};
  p1.A = (const uint8_t*)hidden;
  p1.A_scale = (const float*)hidden_scale;
  p1.a_rows_stride = T;
  p1.W_scale = (const float*)w1s;
  p1.W = (const uint8_t*)w1;
  p1.T = T;
  p1.out = (uint16_t*)output;             // v80: zero-fill by GEMM1 warp 3
  p1.token_npair = (const int*)token_npair;
#ifdef MOE_ZF_SCATTER
  p1.zero_rows = 0;
#else
  p1.zero_rows = (large == 1 && route_split) ? 1 : 0;  // v80: zero-fill by GEMM1 warp 3 (14107 967 -> 956 us)
#endif
  p1.pair_token = (const int*)pair_token;
  p1.expert_off = (const int*)expert_off;
  p1.mtile_prefix = (const int*)mtile_prefix;
  p1.K = H;
  p1.N_total = 2 * I;
  p1.n_tiles = NT1;
  p1.act = (uint8_t*)act;
  p1.act_scale = (float*)act_scale;
  p1.p_cap = p_cap;
  p1.partial = (float*)partial;
  p1.counters = (int*)counters;
  p1.trace = (unsigned long long*)trace;
  p1.split_mult = split_mult;
  p1.fixed_cap = fixed_cap;
  p1.b_evict_first = (T < 2048) ? 1 : 0;  // at large T every expert has several m-tiles that share B via L2
  p1.expert_count = (const int*)expert_count;
  p1.combine_atomic = combine_atomic;

  GemmParams p2{};
  p2.A = (const uint8_t*)act;
  p2.A_scale = (const float*)act_scale;
  p2.a_rows_stride = p_cap;
  p2.W_scale = (const float*)w2s;
  p2.W = (const uint8_t*)w2;
  p2.T = T;
  p2.zero_rows = 0;
  p2.pair_token = (const int*)pair_token;
  p2.token_npair = (const int*)token_npair;
  p2.out = (uint16_t*)output;
  p2.expert_off = (const int*)expert_off;
  p2.mtile_prefix = (const int*)mtile_prefix;
  p2.K = I;
  p2.N_total = H;
  p2.n_tiles = H / 128;
  p2.pair_w = (const float*)pair_w;
  p2.pair_out = (uint16_t*)pair_out;
  p2.p_cap = p_cap;
  p2.partial = (float*)partial;
  p2.counters = (int*)counters;
  p2.trace = trace ? (unsigned long long*)trace + 64 * 16 : nullptr;
  p2.split_mult = split_mult;
  p2.fixed_cap = fixed_cap;
  p2.b_evict_first = (T < 2048) ? 1 : 0;
  p2.expert_count = (const int*)expert_count;
  p2.combine_atomic = combine_atomic;

  const int grid = g_num_sms * 2;
  if (large == 1) {
    GemmParams q1 = p1;
    q1.a_gather = a_gather;
    q1.a_box = (a_gather == 2) ? 0 : a_box;
    q1.acc_warp_release = g_acc_warp_release ? 1 : 0;
    if (!a_gather) {
      q1.A = (const uint8_t*)act_in;
      q1.A_scale = (const float*)act_in_scale;
      q1.a_rows_stride = p_cap;
    }
    q1.n_tiles = U_NT1;
    GemmParams q2 = p2;
    q2.a_box = a_box;
    q2.acc_warp_release = g_acc_warp_release ? 1 : 0;
    q2.n_tiles = U_NT2;  // act_scale produced by the UMMA GEMM1 uses 128-column blocks [16][p_cap]
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(g_num_sms);
    cfg.blockDim = dim3(U_THREADS);
    cfg.dynamicSmemBytes = U_SMEM_BYTES;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    if (T >= U2_MIN_T && u2_enable) {
      // 2-SM clusters: grid of 2*k CTAs, 32 KB per K-block per SM (A rows + N-half of B)
      cudaLaunchConfig_t c2 = cfg;
      c2.gridDim = dim3((g_num_sms / 2) * 2);
      c2.dynamicSmemBytes = U2_SMEM_BYTES;
      cudaLaunchAttribute attr2[2];
      attr2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr2[0].val.programmaticStreamSerializationAllowed = 1;
      attr2[1].id = cudaLaunchAttributeClusterDimension;
      attr2[1].val.clusterDim.x = 2;
      attr2[1].val.clusterDim.y = 1;
      attr2[1].val.clusterDim.z = 1;
      c2.attrs = attr2;
      c2.numAttrs = 2;
      cudaLaunchKernelEx(&c2, umma2_gemm_kernel<128, true>, tmA1, tmB1u, tmA1_64, tmA1_32, q1);
      cudaLaunchKernelEx(&c2, umma2_gemm_kernel<128, false>, tmA2, tm2, tmA2_64, tmA2_32, q2);  // tm2: 128-row W2 boxes
    } else if (mx_mode > 0 && T >= mx_min_t) {
      // v83: hardware block-scaled UMMA kernels (mx_mode 1: GEMM2 only, 2: both GEMMs); 128-row W boxes
      cudaLaunchConfig_t cx = cfg;
      cx.dynamicSmemBytes = MX_SMEM_BYTES;
#ifdef MOE_MX_DBG_NOPDL
      cx.numAttrs = 0;
#endif
      if (mx_mode >= 2) cudaLaunchKernelEx(&cx, umma_mx_kernel<true>, tmA1, tmB1u, q1);
      else cudaLaunchKernelEx(&cfg, umma_gemm_kernel<128, true>, tmA1, tmB1u, tmA1_64, tmA1_32, q1);
      cudaLaunchKernelEx(&cx, umma_mx_kernel<false>, tmA2, tm2, q2);
    } else {
      cudaLaunchKernelEx(&cfg, umma_gemm_kernel<128, true>, tmA1, tmB1u, tmA1_64, tmA1_32, q1);
      if (g2_n128) {
        // v71: 128x128 GEMM2 tiles (4 TMEM buffers, 6 stages); W2 boxes of 128 rows (tm2), 56 n-tiles
        cudaLaunchConfig_t c3 = cfg;
        c3.dynamicSmemBytes = V_SMEM_BYTES;
        GemmParams q3 = q2;
        q3.n_tiles = H / V_BN;
        cudaLaunchKernelEx(&c3, umma_n128_kernel<128, false>, tmA2, tm2, tmA2_64, tmA2_32, q3);
      } else {
        cudaLaunchKernelEx(&cfg, umma_gemm_kernel<128, false>, tmA2, tmB2u, tmA2_64, tmA2_32, q2);
      }
    }
  } else if (large == 3) {
    launch_gemm<L_MT, L_NT, L_WM, L_WN, L_STAGES, 128, true>(tm1, p1, g_num_sms, stream);
    launch_gemm<L_MT, L_NT, L_WM, L_WN, L_STAGES, 64, false>(tm2, p2, g_num_sms, stream);
  } else if (large == 4) {
    launch_gemm<H_MT, H_NT, H_WM, H_WN, H_STAGES, 128, true>(tm1, p1, g_num_sms, stream);
    launch_gemm<H_MT, H_NT, H_WM, H_WN, H_STAGES, 64, false>(tm2, p2, g_num_sms, stream);
  } else if (large == 6) {
    launch_gemm<X_MT, X_NT, X_WM, X_WN, X_STAGES, 128, true>(tm1, p1, g_num_sms, stream);
    launch_gemm<X_MT, X_NT, X_WM, X_WN, X_STAGES, 64, false>(tm2, p2, g_num_sms, stream);
  } else if (large == 5) {
    launch_gemm<S_MT, S_NT, S_WM, S_WN, S_STAGES, 128, true>(tm1, p1, grid, stream);
    launch_gemm<S_MT, S_NT, S_WM, S_WN, S_STAGES, 64, false>(tm2, p2, grid, stream);
  } else if (large == 2 || large == 0) {
    FusedParams fp;
    fp.g1 = p1;
    fp.g2 = p2;
    fp.g1_done = (int*)expert_count + NLOCAL;
    fp.work_ctr = (int*)expert_count + 2 * NLOCAL + 4;
    fp.sched_target = sched_target;
    fp.route_inkernel = inkernel ? 1 : 0;
    fp.logits = (const float*)routing_logits;
    fp.bias = (const uint16_t*)routing_bias;
    fp.T = T;
    fp.local_off = local_off;
    fp.scale_factor = scale_factor;
    fp.pair_token_w = (int*)pair_token;
    fp.pair_w_w = (float*)pair_w;
    fp.token_pairs = (int*)token_pairs;
    fp.token_npair = (int*)token_npair;
    fp.out = (uint16_t*)output;
    fp.zf_done = (int*)expert_count + 2 * NLOCAL + 5;
    fp.fin_ctr = (int*)expert_count + 2 * NLOCAL + 6;
    fp.route_coop = route_coop;
    fp.epoch = epoch;
    fp.route_scratch = (int*)route_scratch;
    fp.grid_bar = (int*)expert_count + 2 * NLOCAL + 7;
    fp.sched_wc = sched_wc;
    fp.work_ctr2 = (int*)expert_count + 2 * NLOCAL + 8;
    fp.s1_mult = s1_mult;
    fp.s2_mult = s2_mult;
    fp.route_cta = route_cta;
    fp.early_w = early_w;
    fp.spec_pf = spec_pf;
    fp.spec_w = spec_w;
    fp.g2_pf = g2_pf;
    fp.pf_ahead = pf_ahead;
    fp.pf_pol = pf_pol;
    if (fused_small) {
      if (large == 2) launch_fused<SW_MT, SX_STAGES>(tm1, tm2, fp, g_num_sms, stream);
      else launch_fused<SW_MT, SW_STAGES>(tm1, tm2, fp, grid, stream);
    } else if (large == 2) {
      launch_swab<SW_MT, SX_STAGES, 128, true>(tm1, p1, g_num_sms, stream);
      launch_swab<SW_MT, SX_STAGES, 64, false>(tm2, p2, g_num_sms, stream);
    } else {
      launch_swab<SW_MT, SW_STAGES, 128, true>(tm1, p1, grid, stream);
      launch_swab<SW_MT, SW_STAGES, 64, false>(tm2, p2, grid, stream);
    }
  }

  if (combine_atomic) {
    // v59: no combine kernel (GEMM2 epilogues reduce into the output; counters reset by the producers)
  } else if (large != 1) {
    cudaLaunchConfig_t cfg = {};
    const long long total_chunks = (long long)T * (H / 8);
    int cgrid = (int)((total_chunks + 255) / 256);
    if (cgrid > g_num_sms * 8) cgrid = g_num_sms * 8;
    cfg.gridDim = dim3(cgrid);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, combine_small_kernel, T, (const int*)token_pairs, (const uint16_t*)pair_out,
                       (uint16_t*)output, (int*)expert_count, epoch);
  } else {
    cudaLaunchConfig_t cfg = {};
    int cgrid = T < g_num_sms * 8 ? T : g_num_sms * 8;  // bounded grid walking the multi-pair token list
    if (cgrid < 1) cgrid = 1;
    cfg.gridDim = dim3(cgrid);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, combine_kernel, T, (const int*)token_pairs, (const uint16_t*)pair_out,
                       (uint16_t*)output, (int*)expert_count, (const int*)multi_list, epoch);
  }
  return (int)cudaGetLastError();
}
