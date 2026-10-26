// FP8 block-scale MoE (DeepSeek-V3 routing, top-8, 8 groups, keep 4) for NVIDIA B200 (sm_100a).
// Self-contained CUDA C++ + inline PTX: TMA (cp.async.bulk.tensor), tcgen05.mma (kind::f8f6f4),
// TMEM accumulators with DeepGEMM-style per-128-k-block FP32 scale promotion.
//
// Pipeline (one host call):
//   K1 routing + compaction (last-CTA pattern)
//   K2 gather X rows into expert-sorted buffer
//   K3 grouped GEMM1 (W13) + SwiGLU + fp8 quant (per token / 128-block scales)
//   K4 grouped GEMM2 (W2) -> bf16 partials
//   K5 finalize: weighted sum of partials per token -> bf16 output
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace moe {

// ----------------------------------------------------------------------------------------------
// Problem constants
// ----------------------------------------------------------------------------------------------
constexpr int E_GLOBAL = 256;
constexpr int E_LOCAL = 32;
constexpr int TOPK = 8;
constexpr int N_GROUP = 8;
constexpr int TOPK_GROUP = 4;
constexpr int HIDDEN = 7168;
constexpr int INTER = 2048;
constexpr int KB_HIDDEN = HIDDEN / 128;   // 56
constexpr int KB_INTER = INTER / 128;     // 16
constexpr int MB_GEMM1 = INTER / 128;     // 16 j-blocks (each covers gate rows + up rows)
constexpr int MB_GEMM2 = HIDDEN / 128;    // 56 h-blocks
constexpr int N_PAD = 16;                 // tokens per expert padded to a multiple of 16
constexpr int N_MAX = 128;                // max tokens per item (N of UMMA)

// meta layout (int32 array)
enum MetaIdx { META_NUM_ROWS = 0, META_NUM_ITEMS1 = 1, META_NUM_ITEMS2 = 2, META_DONE = 3, META_NMAX = 4, META_EPOCH = 5, META_NEXT = 6, META_SIZE = 8 };

// ----------------------------------------------------------------------------------------------
// PTX helpers
// ----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" ::"r"(smem_u32(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar);
  asm volatile(
      "{\n"
      " .reg .pred p;\n"
      " LAB_WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @p bra DONE;\n"
      " bra LAB_WAIT;\n"
      " DONE:\n"
      "}" ::"r"(addr), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_prefetch_desc(const void* desc) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(desc)) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const void* desc, uint64_t* bar, int c0, int c1, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(smem_dst)),
      "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const void* desc, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(reinterpret_cast<uint64_t>(desc)),
               "r"(c0), "r"(c1), "r"(c2)
               : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* smem_dst, const void* desc, uint64_t* bar, int c0, int c1, int c2, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(smem_u32(smem_dst)),
      "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "l"(policy)
      : "memory");
}

// ---- tcgen05 ----
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(smem_dst)), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
// D[tmem] (+)= A[smem desc] * B[smem desc], kind::f8f6f4 (e4m3 x e4m3 -> f32)
__device__ __forceinline__ void umma_f8(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n"
      " .reg .pred p;\n"
      " setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
      "}" ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]),
        "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// UMMA shared-memory descriptor: K-major, SWIZZLE_128B, tile base must be 1024B aligned.
__device__ __forceinline__ uint64_t make_smem_desc_sw128(uint32_t smem_addr) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr & 0x3FFFF) >> 4);        // start address [0,14)
  d |= (uint64_t)(16 >> 4) << 16;                     // leading byte offset (unused for SW128 K-major)
  d |= (uint64_t)(1024 >> 4) << 32;                   // stride byte offset: 8 rows * 128B
  d |= (uint64_t)1 << 46;                             // descriptor version (sm100)
  d |= (uint64_t)2 << 61;                             // layout type: SWIZZLE_128B
  return d;
}
// Instruction descriptor: kind::f8f6f4, A/B = e4m3, D = f32, both K-major, M=128, N=n.
__device__ __forceinline__ uint32_t make_idesc_f8(uint32_t n) {
  uint32_t d = 0;
  d |= 1u << 4;              // c_format = F32
  d |= 0u << 7;              // a_format = E4M3
  d |= 0u << 10;             // b_format = E4M3
  d |= (n >> 3) << 17;       // N >> 3
  d |= (128u >> 4) << 24;    // M >> 4
  return d;
}

__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
__device__ __forceinline__ uint32_t named_bar_sync(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
  return 0;
}


// ----------------------------------------------------------------------------------------------
// K1: routing (one warp per token). Each CTA aggregates its local-expert counts in smem, reserves
// slots with one global atomic per expert, and the last CTA computes offsets/prefixes (32 entries).
// ----------------------------------------------------------------------------------------------
struct RoutingParams {
  const float* logits;        // [T, 256]
  const __nv_bfloat16* bias;  // [256]
  int T;
  int local_offset;
  float rsf;
  int* topk_ids;              // [T, 8]
  float* topk_w;              // [T, 8]
  int* pair_slot;             // [T, 8]  slot within local expert, -1 if not local
  int* expert_count;          // [32]  (zero between calls; accumulated with atomics)
  int* expert_offset;         // [33] padded offsets
  int* item_prefix1;          // [33]
  int* item_prefix2;          // [33]
  int* meta;                  // [META_SIZE]
  int* act_done;              // [32] per-expert act-item completion counters (reset here)
};

constexpr int ROUTE_WARPS = 8;

__global__ void __launch_bounds__(ROUTE_WARPS * 32) routing_kernel(RoutingParams p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * ROUTE_WARPS + warp;
  __shared__ int s_count[E_LOCAL];
  __shared__ int s_base[E_LOCAL];
  __shared__ int s_sel[ROUTE_WARPS * TOPK];   // local expert id or -1
  __shared__ int s_rank[ROUTE_WARPS * TOPK];
  __shared__ int s_last;

  if (threadIdx.x < E_LOCAL) s_count[threadIdx.x] = 0;
  __syncthreads();

  if (t < p.T) {
    // lane holds experts lane*8 .. lane*8+7 ; group g = lanes 4g..4g+3
    float s[8], sb[8];
    const float4* lg = reinterpret_cast<const float4*>(p.logits + (size_t)t * E_GLOBAL + lane * 8);
    float4 a = lg[0], b = lg[1];
    float lv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s[i] = 1.0f / (1.0f + expf(-lv[i]));
      sb[i] = s[i] + __bfloat162float(p.bias[lane * 8 + i]);
    }
    float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = sb[i];
      if (v > m1) { m2 = m1; m1 = v; }
      else if (v > m2) { m2 = v; }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      float o1 = __shfl_xor_sync(0xffffffff, m1, o);
      float o2 = __shfl_xor_sync(0xffffffff, m2, o);
      float n1 = fmaxf(m1, o1);
      float n2 = fminf(fmaxf(m1, o1), fmaxf(fminf(m1, o1), fmaxf(m2, o2)));
      m1 = n1; m2 = n2;
    }
    float gscore = m1 + m2;
    const int g = lane >> 2;
    int rank = 0;
#pragma unroll
    for (int gg = 0; gg < N_GROUP; ++gg) {
      float other = __shfl_sync(0xffffffff, gscore, gg * 4);
      if (other > gscore || (other == gscore && gg < g)) ++rank;
    }
    const bool kept = rank < TOPK_GROUP;
    float cand[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cand[i] = kept ? sb[i] : -INFINITY;
    float wsum = 0.f;
    int my_id = 0;
    float my_s = 0.f;
#pragma unroll
    for (int r = 0; r < TOPK; ++r) {
      float bv = -INFINITY; int bi = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (cand[i] > bv) { bv = cand[i]; bi = i; }
      }
      int bidx = lane * 8 + bi;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, o);
        int oi = __shfl_xor_sync(0xffffffff, bidx, o);
        if (ov > bv || (ov == bv && oi < bidx)) { bv = ov; bidx = oi; }
      }
      float mys = s[0];
#pragma unroll
      for (int i = 1; i < 8; ++i) if (i == (bidx & 7)) mys = s[i];
      float sv = __shfl_sync(0xffffffff, mys, bidx >> 3);
      wsum += sv;
      if (lane == r) { my_id = bidx; my_s = sv; }
      if ((bidx >> 3) == lane) {
#pragma unroll
        for (int i = 0; i < 8; ++i) if (i == (bidx & 7)) cand[i] = -INFINITY;
      }
    }
    const float inv = p.rsf / (wsum + 1e-20f);
    if (lane < TOPK) {
      const int id = my_id;
      const float sv = my_s;
      p.topk_ids[t * TOPK + lane] = id;
      p.topk_w[t * TOPK + lane] = sv * inv;
      int e = id - p.local_offset;
      int rk = -1;
      if (e >= 0 && e < E_LOCAL) rk = atomicAdd(&s_count[e], 1); else e = -1;
      s_sel[warp * TOPK + lane] = e;
      s_rank[warp * TOPK + lane] = rk;
    }
  } else if (lane < TOPK) {
    s_sel[warp * TOPK + lane] = -1;
    s_rank[warp * TOPK + lane] = -1;
  }
  __syncthreads();
  if (threadIdx.x < E_LOCAL) {
    int c = s_count[threadIdx.x];
    s_base[threadIdx.x] = c > 0 ? atomicAdd(p.expert_count + threadIdx.x, c) : 0;
  }
  __syncthreads();
  if (threadIdx.x < ROUTE_WARPS * TOPK) {
    const int tt = blockIdx.x * ROUTE_WARPS + threadIdx.x / TOPK;
    if (tt < p.T) {
      int e = s_sel[threadIdx.x];
      p.pair_slot[tt * TOPK + (threadIdx.x % TOPK)] = e >= 0 ? s_base[e] + s_rank[threadIdx.x] : -1;
    }
  }

  // ---- last CTA: offsets / item prefixes ----
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    int prev = atomicAdd(p.meta + META_DONE, 1);
    s_last = (prev == (int)gridDim.x - 1) ? 1 : 0;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (threadIdx.x < E_LOCAL) s_count[threadIdx.x] = __ldcg(p.expert_count + threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) {
    int off = 0, it1 = 0, it2 = 0, nmax = N_PAD;
    for (int e = 0; e < E_LOCAL; ++e) {
      int c = s_count[e];
      int padded = (c + N_PAD - 1) / N_PAD * N_PAD;
      int chunks = (padded + N_MAX - 1) / N_MAX;
      nmax = max(nmax, min(padded, N_MAX));
      p.expert_offset[e] = off;
      p.item_prefix1[e] = it1;
      p.item_prefix2[e] = it2;
      off += padded;
      it1 += chunks * MB_GEMM1;
      it2 += chunks * MB_GEMM2;
    }
    p.expert_offset[E_LOCAL] = off;
    p.item_prefix1[E_LOCAL] = it1;
    p.item_prefix2[E_LOCAL] = it2;
    p.meta[META_NUM_ROWS] = off;
    p.meta[META_NUM_ITEMS1] = 2 * it1;
    p.meta[META_NUM_ITEMS2] = it2;
    p.meta[META_NMAX] = nmax;
    p.meta[META_EPOCH] = __ldcg(p.meta + META_EPOCH) + 1;
    p.meta[META_NEXT] = 0;
    for (int e = 0; e < E_LOCAL; ++e) p.act_done[e] = 0;
    __threadfence();
    p.meta[META_DONE] = 0;  // reset for the next call (all CTAs already arrived)
  }
}

// ----------------------------------------------------------------------------------------------
// K2: gather X rows (fp8) and scales into expert-sorted layout. One CTA per (token, k) pair, plus
// E_LOCAL*N_PAD CTAs that zero the padding rows. Also resets expert_count for the next call.
// ----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(224) gather_kernel(const uint8_t* __restrict__ X, const float* __restrict__ xs, int T,
                                                     const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                                                     const int* __restrict__ pair_slot, const int* __restrict__ expert_offset,
                                                     const int* __restrict__ expert_count, int local_offset,
                                                     int* __restrict__ row_token, float* __restrict__ row_w,
                                                     int* __restrict__ pair_row,
                                                     uint8_t* __restrict__ Xp, float* __restrict__ xsp, int R_max) {
  const int i = blockIdx.x;
  const int total = T * TOPK;
  if (i < total) {
    const int slot = pair_slot[i];
    if (slot < 0) {
      if (threadIdx.x == 0) pair_row[i] = -1;
      return;
    }
    const int e = topk_ids[i] - local_offset;
    const int row = expert_offset[e] + slot;
    const int t = i / TOPK;
    {
      const uint4* src = reinterpret_cast<const uint4*>(X + (size_t)t * HIDDEN);
      uint4* dst = reinterpret_cast<uint4*>(Xp + (size_t)row * HIDDEN);
      const uint4 a = __ldg(src + threadIdx.x), b = __ldg(src + 224 + threadIdx.x);
      dst[threadIdx.x] = a;
      dst[224 + threadIdx.x] = b;
    }
    if (threadIdx.x < KB_HIDDEN) xsp[threadIdx.x * R_max + row] = xs[threadIdx.x * T + t];
    if (threadIdx.x == 0) {
      row_token[row] = t;
      row_w[row] = topk_w[i];
      pair_row[i] = row;
    }
  } else {
    const int j = i - total;              // padding slot: expert j/16, rank count + j%16
    const int e = j / N_PAD;
    const int cnt = expert_count[e];
    const int padded = expert_offset[e + 1] - expert_offset[e];
    const int r = cnt + (j % N_PAD);
    if (r < padded) {
      const int row = expert_offset[e] + r;
      uint4* dst = reinterpret_cast<uint4*>(Xp + (size_t)row * HIDDEN);
      dst[threadIdx.x] = make_uint4(0, 0, 0, 0);
      dst[224 + threadIdx.x] = make_uint4(0, 0, 0, 0);
      if (threadIdx.x < KB_HIDDEN) xsp[threadIdx.x * R_max + row] = 0.f;
      if (threadIdx.x == 0) {
        row_token[row] = -1;
        row_w[row] = 0.f;
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------
// K3: fused persistent grouped GEMM (tcgen05). Work list (static round-robin over CTAs):
//   [0, N1)        lin items  (e, 128-wide j-block): D = X W13_lin^T  -> bf16 scratch L + flag
//   [N1, 2N1)      act items  (e, j-block):          D = X W13_act^T  -> SwiGLU with L, fp8 C + scale
//   [2N1, 2N1+N2)  down items (e, 128-wide h-block): D = C W2^T      -> bf16 partials P
// Down items of expert e wait until all act items of e are done (per-expert counter).
// Tokens are the UMMA A operand (M=128 = TMEM lanes), weights are B (N=128). A stage holds KPER=2
// k-blocks (W tile + X rows each); the two k-block accumulators use separate TMEM column ranges and
// are promoted by two epilogue warpgroups (64 columns each, one token row per thread).
// CTA = 384 threads: warp 0 TMA producer, warp 1 MMA issuer (warps 2,3 idle), warps 4..11 epilogue.
// ----------------------------------------------------------------------------------------------
struct XMaps {
  CUtensorMap m[8];  // box rows 16, 32, ..., 128
};

struct GemmParams {
  const int* expert_offset;   // [33]
  const int* item_prefix1;    // [33] GEMM1 items per expert (lin or act)
  const int* item_prefix2;    // [33] down items per expert
  int* meta;
  int* act_done;              // [32]
  const float* w1_scale;      // [32][32][56]
  const float* w2_scale;      // [32][56][16]
  const float* xs;            // [56][R_max]
  const float* cs;            // [16][R_max]
  int R_max;
  __nv_bfloat16* L;           // [R_max][2048] bf16 lin scratch
  int* flags;                 // [4096] lin-item completion flags (epoch-tagged)
  uint8_t* C;                 // [R_max][2048] fp8
  float* c_scale;             // [16][R_max]
  __nv_bfloat16* P;           // [R_max][7168]
};

constexpr int GEMM_THREADS = 384;
constexpr int TILE_BYTES = 128 * 128;  // 16 KB fp8 tile (128 rows x 128 B)
constexpr int SMEM_MAX_DYN = 232448;   // 227 KB
constexpr int SMEM_MISC_BYTES = 4096;
constexpr int MAX_STAGES = 8;
constexpr int KPER = 2;
constexpr int NBUF = 2;                // TMEM accumulator buffers (256 columns each)
constexpr int BUF_COLS = KPER * 128;
constexpr int TMEM_COLS = 512;
constexpr int SMEM_AMAX_OFF = 0;       // [2][128] f32
constexpr int SMEM_WS_OFF = 1024;      // [64] f32
constexpr int SMEM_TAB_OFF = 1280;     // item_prefix1[33], expert_offset[33], item_prefix2[33]
constexpr int SMEM_BAR_OFF = 1792;     // full[8], empty[8], tfull[2], tempty[2], qfull[8], qempty[8]
constexpr int SMEM_TMEMPTR_OFF = 2304;
constexpr int SMEM_Q_OFF = 2368;        // item queue [QD]
constexpr int QD = 8;
constexpr int SMEM_PIPE_OFF = SMEM_MISC_BYTES;

// runtime pipeline layout from the largest padded expert chunk (n_max <= 128)
struct PipeLayout {
  int xb;           // bytes of X rows per k-block (n_max * 128)
  int kb_bytes;     // W tile + X rows
  int stage_bytes;  // KPER * kb_bytes
  int nstages;
};
__device__ __forceinline__ PipeLayout make_layout(int nmax) {
  PipeLayout L;
  L.xb = nmax * 128;
  L.kb_bytes = TILE_BYTES + L.xb;
  L.stage_bytes = KPER * L.kb_bytes;
  const int slack = TILE_BYTES - L.xb;   // the MMA reads 128 rows of X from the last tile
  const int budget = SMEM_MAX_DYN - SMEM_MISC_BYTES - 1024 - slack;
  int n = budget / L.stage_bytes;
  L.nstages = n > MAX_STAGES ? MAX_STAGES : n;
  return L;
}

enum ItemType { IT_LIN = 0, IT_ACT = 1, IT_DOWN = 2 };
struct ItemInfo {
  int e, mb, n, n_base, type, idx, kb;
};

// tables in smem: tab[0..32] = item_prefix1, tab[33..65] = expert_offset, tab[66..98] = item_prefix2
__device__ __forceinline__ ItemInfo decode_item(int item, const int* tab) {
  const int* prefix1 = tab;
  const int* expert_offset = tab + 33;
  const int* prefix2 = tab + 66;
  const int N1 = prefix1[E_LOCAL];
  int type = IT_LIN, MB = MB_GEMM1;
  const int* prefix = prefix1;
  if (item >= 2 * N1) { type = IT_DOWN; item -= 2 * N1; prefix = prefix2; MB = MB_GEMM2; }
  else if (item >= N1) { type = IT_ACT; item -= N1; }
  int e = 0;
#pragma unroll 1
  for (int i = 1; i < E_LOCAL; ++i) e += (item >= prefix[i]) ? 1 : 0;
  int local = item - prefix[e];
  int chunk = local / MB;
  int mb = local - chunk * MB;
  int off = expert_offset[e];
  int padded = expert_offset[e + 1] - off;
  int n_base = off + chunk * N_MAX;
  int n = min(N_MAX, padded - chunk * N_MAX);
  return ItemInfo{e, mb, n, n_base, type, item, type == IT_DOWN ? KB_INTER : KB_HIDDEN};
}

__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]),
        "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15]),
        "=f"(v[16]), "=f"(v[17]), "=f"(v[18]), "=f"(v[19]), "=f"(v[20]), "=f"(v[21]), "=f"(v[22]), "=f"(v[23]),
        "=f"(v[24]), "=f"(v[25]), "=f"(v[26]), "=f"(v[27]), "=f"(v[28]), "=f"(v[29]), "=f"(v[30]), "=f"(v[31])
      : "r"(taddr)
      : "memory");
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }

#ifndef MOE_TRACE
#define MOE_TRACE 0
#endif
constexpr int TRACE_N = 256;
__device__ long long g_trace[2][3][TRACE_N][4];
__device__ long long g_cta_time[2][256][2];      // [kernel][cta][start,end] globaltimer ns
__device__ long long g_item_time[2][256][32][8]; // [kernel][cta][item idx][phase stamps] globaltimer ns
__device__ __forceinline__ long long clk() { long long c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c; }
__device__ __forceinline__ long long gtimer() { long long c; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c)); return c; }
#define TR(K, ROLE, I, J) do { if (MOE_TRACE && blockIdx.x == 0 && (I) < TRACE_N) g_trace[K][ROLE][I][J] = clk(); } while (0)
#ifndef MOE_ABL_NO_EPI
#define MOE_ABL_NO_EPI 0
#endif
#ifndef MOE_REG_PROD
#define MOE_REG_PROD 40
#endif
#ifndef MOE_REG_EPI
#define MOE_REG_EPI 232
#endif
#ifndef MOE_KROT
#define MOE_KROT 1
#endif
#ifndef MOE_BF16ACC
#define MOE_BF16ACC 0
#endif
// per-CTA rotation of the k-block walk so that CTAs do not hit the same row offsets in lockstep
__device__ __forceinline__ int krot(int kb, int KB) {
  if (!MOE_KROT) return kb;
  const int r = 2 * (blockIdx.x % (KB / 2));
  int k = kb + r;
  return k >= KB ? k - KB : k;
}

__device__ __forceinline__ bool item_ready(const GemmParams& p, const ItemInfo& it, int epoch, const int* tab) {
  if (it.type == IT_ACT) return ld_acquire_gpu(p.flags + it.idx) == epoch;
  if (it.type == IT_DOWN) {
    const int chunks = (tab[33 + it.e + 1] - tab[33 + it.e] + N_MAX - 1) / N_MAX;
    return ld_acquire_gpu(p.act_done + it.e) >= chunks * MB_GEMM1;
  }
  return true;
}

__global__ void __launch_bounds__(GEMM_THREADS, 1)
fused_gemm_kernel(const __grid_constant__ CUtensorMap tmap_w1, const __grid_constant__ CUtensorMap tmap_w2,
                  const __grid_constant__ XMaps xmaps_x, const __grid_constant__ XMaps xmaps_c, GemmParams p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + SMEM_BAR_OFF);
  uint64_t* empty_bar = full_bar + MAX_STAGES;
  uint64_t* tfull_bar = empty_bar + MAX_STAGES;
  uint64_t* tempty_bar = tfull_bar + NBUF;
  uint64_t* qfull_bar = tempty_bar + NBUF;
  uint64_t* qempty_bar = qfull_bar + QD;
  uint32_t* s_tmem_ptr = reinterpret_cast<uint32_t*>(smem + SMEM_TMEMPTR_OFF);
  volatile int* s_q = reinterpret_cast<volatile int*>(smem + SMEM_Q_OFF);
  float* s_amax = reinterpret_cast<float*>(smem + SMEM_AMAX_OFF);   // [2][128]
  float* s_ws = reinterpret_cast<float*>(smem + SMEM_WS_OFF);       // [64]
  int* s_tab = reinterpret_cast<int*>(smem + SMEM_TAB_OFF);         // [99]
  const PipeLayout lay = make_layout(__ldg(p.meta + META_NMAX));
  uint8_t* pipe = smem + SMEM_PIPE_OFF;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int N1 = __ldg(p.item_prefix1 + E_LOCAL);
  const int num_items = 2 * N1 + __ldg(p.item_prefix2 + E_LOCAL);
  const int epoch = __ldg(p.meta + META_EPOCH);

  if (threadIdx.x < 33) s_tab[threadIdx.x] = __ldg(p.item_prefix1 + threadIdx.x);
  else if (threadIdx.x < 66) s_tab[threadIdx.x] = __ldg(p.expert_offset + threadIdx.x - 33);
  else if (threadIdx.x < 99) s_tab[threadIdx.x] = __ldg(p.item_prefix2 + threadIdx.x - 66);
  if (warp == 0 && lane == 0) {
    tma_prefetch_desc(&tmap_w1);
    tma_prefetch_desc(&tmap_w2);
    for (int i = 0; i < 8; ++i) { tma_prefetch_desc(&xmaps_x.m[i]); tma_prefetch_desc(&xmaps_c.m[i]); }
    for (int i = 0; i < MAX_STAGES; ++i) {
      mbar_init(&full_bar[i], 1);
      mbar_init(&empty_bar[i], 1);
    }
    for (int i = 0; i < NBUF; ++i) {
      mbar_init(&tfull_bar[i], 1);
      mbar_init(&tempty_bar[i], 8);  // one arrival per epilogue warp
    }
    for (int i = 0; i < QD; ++i) {
      mbar_init(&qfull_bar[i], 1);
      mbar_init(&qempty_bar[i], 9);  // MMA warp + 8 epilogue warps
    }
    fence_barrier_init();
  }
  if (warp == 1) {
    tmem_alloc(s_tmem_ptr, TMEM_COLS);
    tmem_relinquish();
  }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem_base = *s_tmem_ptr;

  if (warp < 4) {
    setmaxnreg_dec<MOE_REG_PROD>();
    if (warp == 0 && lane == 0) {
      // ===================== TMA producer (also claims work items) =====================
      const uint64_t pol_w = policy_evict_first();
      const uint64_t pol_x = policy_evict_last();
      int stage = 0;
      uint32_t phase = 0;
      int pidx = 0;
      int qseq = 0;  // number of items published
      auto claim = [&]() {
        const int qi = qseq % QD;
        mbar_wait(&qempty_bar[qi], (uint32_t)(((qseq / QD) & 1) ^ 1));
        int v = atomicAdd(p.meta + META_NEXT, 1);
        if (v >= num_items) v = -1;
        s_q[qi] = v;
        mbar_arrive(&qfull_bar[qi]);
        ++qseq;
      };
      claim();
      claim();
      for (int seq = 0;; ++seq, ++pidx) {
        const int item = s_q[seq % QD];
        if (item < 0) break;
        claim();  // keep two items claimed ahead of the one being issued
        ItemInfo it = decode_item(item, s_tab);
        const bool down = it.type == IT_DOWN;
        const CUtensorMap* wmap = down ? &tmap_w2 : &tmap_w1;
        const CUtensorMap* xmap = down ? &xmaps_c.m[it.n / 16 - 1] : &xmaps_x.m[it.n / 16 - 1];
        const int wrow = down ? it.mb * 128 : ((it.type == IT_ACT ? INTER : 0) + it.mb * 128);
        const uint32_t tx = KPER * (TILE_BYTES + it.n * 128);
        while (!item_ready(p, it, epoch, s_tab)) { __nanosleep(200); }
        const int nst = it.kb / KPER;
        for (int si = 0; si < nst; ++si) {
          const int tri = pidx * 28 + si;
          TR(down, 0, tri, 0);
          mbar_wait(&empty_bar[stage], phase ^ 1);
          TR(down, 0, tri, 1);
          uint8_t* st = pipe + stage * lay.stage_bytes;
          mbar_arrive_expect_tx(&full_bar[stage], tx);
#pragma unroll
          for (int k = 0; k < KPER; ++k) {
            const int kb = krot(si * KPER + k, it.kb);
            tma_load_3d(st + k * lay.kb_bytes, wmap, &full_bar[stage], kb * 128, wrow, it.e, pol_w);
            tma_load_2d(st + k * lay.kb_bytes + TILE_BYTES, xmap, &full_bar[stage], kb * 128, it.n_base, pol_x);
          }
          TR(down, 0, tri, 2);
          if (++stage == lay.nstages) { stage = 0; phase ^= 1; }
        }
      }
    } else if (warp == 1 && lane == 0) {
      // ===================== MMA issuer =====================
      int stage = 0;
      uint32_t phase = 0;
      int buf = 0;
      uint32_t bphase = 0;
      const uint32_t idesc = make_idesc_f8(128u);
      for (int seq = 0;; ++seq) {
        const int qi = seq % QD;
        mbar_wait(&qfull_bar[qi], (uint32_t)((seq / QD) & 1));
        const int item = s_q[qi];
        mbar_arrive(&qempty_bar[qi]);
        if (item < 0) break;
        const ItemInfo it = decode_item(item, s_tab);
        const int nst = it.kb / KPER;
        for (int si = 0; si < nst; ++si) {
          const int tri = seq * 28 + si;
          TR(it.type == IT_DOWN, 1, tri, 0);
          mbar_wait(&full_bar[stage], phase);
          TR(it.type == IT_DOWN, 1, tri, 1);
          mbar_wait(&tempty_bar[buf], bphase ^ 1);
          TR(it.type == IT_DOWN, 1, tri, 2);
          tc_fence_after();
          const uint32_t st_addr = smem_u32(pipe + stage * lay.stage_bytes);
          {
            const uint64_t a0 = make_smem_desc_sw128(st_addr + TILE_BYTES);                 // tokens, k-block 0
            const uint64_t b0 = make_smem_desc_sw128(st_addr);                              // features, k-block 0
            const uint64_t a1 = make_smem_desc_sw128(st_addr + lay.kb_bytes + TILE_BYTES);  // k-block 1
            const uint64_t b1 = make_smem_desc_sw128(st_addr + lay.kb_bytes);
            const uint32_t d0 = tmem_base + buf * BUF_COLS;
            const uint32_t d1 = d0 + 128;
#ifndef MOE_ABL_NO_MMA
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {   // interleave the two independent accumulators
              umma_f8(d0, a0 + 2 * kk, b0 + 2 * kk, idesc, kk > 0 ? 1u : 0u);
              umma_f8(d1, a1 + 2 * kk, b1 + 2 * kk, idesc, kk > 0 ? 1u : 0u);
            }
#endif
          }
          tc_commit(&tfull_bar[buf]);
          tc_commit(&empty_bar[stage]);
          TR(it.type == IT_DOWN, 1, tri, 3);
          if (++buf == NBUF) { buf = 0; bphase ^= 1; }
          if (++stage == lay.nstages) { stage = 0; phase ^= 1; }
        }
      }
    }
  } else {
    // ===================== Epilogue warpgroups =====================
    setmaxnreg_inc<MOE_REG_EPI>();
    const int wg = (warp >> 2) - 1;        // 0 or 1: owns columns [64*wg, 64*wg+64)
    const int q = warp & 3;                // TMEM lane group
    const int tid = threadIdx.x - 128;     // 0..255
    const int row = q * 32 + lane;         // token row within the item (TMEM lane)
    const uint32_t tmem_lane_off = (uint32_t)(q * 32) << 16;
    const int col0 = wg * 64;
    int buf = 0;
    uint32_t bphase = 0;
    float acc[64];
#if MOE_BF16ACC
    __nv_bfloat162 acc2[32];
#endif
    if (MOE_TRACE && warp == 4 && lane == 0) g_cta_time[0][blockIdx.x][0] = gtimer();

    auto ws_ptr = [&](const ItemInfo& it) -> const float* {
      if (it.type == IT_DOWN) return p.w2_scale + ((size_t)it.e * 56 + it.mb) * KB_INTER;
      return p.w1_scale + ((size_t)it.e * 32 + (it.type == IT_ACT ? 16 : 0) + it.mb) * KB_HIDDEN;
    };
    auto xs_ptr = [&](const ItemInfo& it) -> const float* {
      const float* base = it.type == IT_DOWN ? p.cs : p.xs;
      return base + ((row < it.n) ? (it.n_base + row) : it.n_base);
    };
    auto q_read = [&](int seq) -> int {   // all lanes read after the warp waits on the slot
      const int qi = seq % QD;
      mbar_wait(&qfull_bar[qi], (uint32_t)((seq / QD) & 1));
      const int v = s_q[qi];
      __syncwarp();
      if (lane == 0) mbar_arrive(&qempty_bar[qi]);
      return v;
    };
    int seq = 0;
    int item = q_read(0);
    ItemInfo it = (item >= 0) ? decode_item(item, s_tab) : ItemInfo{};
    float ws_reg = 0.f;
    float xs_r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) xs_r[i] = 0.f;
    const float* xsp = nullptr;
    if (item >= 0) {
      if (tid == 0) { while (!item_ready(p, it, epoch, s_tab)) { __nanosleep(200); } }
      named_bar_sync(1, 256);
      if (tid < it.kb) ws_reg = __ldg(ws_ptr(it) + tid);
      xsp = xs_ptr(it);
#pragma unroll
      for (int i = 0; i < 8; ++i) xs_r[i] = __ldg(xsp + (size_t)krot(i, it.kb) * p.R_max);
    }
    int item_idx = 0;
    for (; item >= 0; ++item_idx, ++seq) {
      const int grow = it.n_base + row;
      const bool valid = row < it.n;
      const int KBI = it.kb;
      const int tk = it.type == IT_DOWN ? 1 : 0;
#define ITEM_STAMP(J) do { if (MOE_TRACE && warp == 4 && lane == 0 && item_idx < 32) g_item_time[tk][blockIdx.x][item_idx][J] = gtimer(); } while (0)
      ITEM_STAMP(0);
      named_bar_sync(1, 256);  // previous item finished with s_ws / s_amax
      if (tid < KBI) s_ws[tid] = ws_reg;
      named_bar_sync(1, 256);
      // next item (prefetch targets); its inputs must be ready before the scale prefetch
      const int next_item = q_read(seq + 1);
      const bool has_next = next_item >= 0;
      ItemInfo nit = has_next ? decode_item(next_item, s_tab) : it;
      if (has_next) {
        if (tid == 0) { while (!item_ready(p, nit, epoch, s_tab)) { __nanosleep(200); } }
        named_bar_sync(1, 256);
        if (tid < nit.kb) ws_reg = __ldg(ws_ptr(nit) + tid);
      }
      const float* nxsp = has_next ? xs_ptr(nit) : xsp;
      const int NKB = has_next ? nit.kb : KBI;
#pragma unroll
      for (int c = 0; c < 64; ++c) acc[c] = 0.f;
#if MOE_BF16ACC
#pragma unroll
      for (int c = 0; c < 32; ++c) acc2[c] = __float2bfloat162_rn(0.f);
#endif
      if (it.type == IT_ACT && valid) prefetch_l2(p.L + (size_t)grow * INTER + it.mb * 128 + col0);

      // one stage = 2 k-blocks (KB0, KB0+1) in TMEM buffer `buf`
#if MOE_BF16ACC
#define MOE_ACCUMULATE(V, SC)                                                                          \
          {                                                                                            \
            const __nv_bfloat162 sc2 = __float2bfloat162_rn(SC);                                       \
            _Pragma("unroll")                                                                          \
            for (int j = 0; j < 32; ++j) acc2[j] = __hfma2(__floats2bfloat162_rn(V[2 * j], V[2 * j + 1]), sc2, acc2[j]); \
          }
#else
#define MOE_ACCUMULATE(V, SC)                                                                          \
          {                                                                                            \
            _Pragma("unroll")                                                                          \
            for (int j = 0; j < 64; ++j) acc[j] = fmaf(V[j], SC, acc[j]);                              \
          }
#endif
#define MOE_STAGE_BODY(KB0, XS0, XS1)                                                                  \
      {                                                                                                \
        const int tri = (warp == 4 && lane == 0) ? item_idx * 28 + (KB0) / 2 : TRACE_N;                \
        TR(tk, 2, tri, 0);                                                                             \
        mbar_wait(&tfull_bar[buf], bphase);                                                            \
        TR(tk, 2, tri, 1);                                                                             \
        tc_fence_after();                                                                              \
        if (!MOE_ABL_NO_EPI) {                                                                         \
          const uint32_t tbase = tmem_base + tmem_lane_off + buf * BUF_COLS + col0;                    \
          float v[64];                                                                                 \
          tmem_ld32(tbase, v);                                                                         \
          tmem_ld32(tbase + 32, v + 32);                                                               \
          const float sc0 = s_ws[krot(KB0, KBI)] * (XS0);                                              \
          const float sc1 = s_ws[krot(KB0, KBI) + 1] * (XS1);                                          \
          tmem_ld_wait();                                                                              \
          MOE_ACCUMULATE(v, sc0)                                                                       \
          tmem_ld32(tbase + 128, v);                                                                   \
          tmem_ld32(tbase + 160, v + 32);                                                              \
          tmem_ld_wait();                                                                              \
          TR(tk, 2, tri, 2);                                                                           \
          tc_fence_before();                                                                           \
          __syncwarp();                                                                                \
          if (lane == 0) mbar_arrive(&tempty_bar[buf]);                                                \
          MOE_ACCUMULATE(v, sc1)                                                                       \
          TR(tk, 2, tri, 3);                                                                           \
        } else {                                                                                       \
          tc_fence_before();                                                                           \
          __syncwarp();                                                                                \
          if (lane == 0) mbar_arrive(&tempty_bar[buf]);                                                \
        }                                                                                              \
        if (++buf == NBUF) { buf = 0; bphase ^= 1; }                                                   \
      }
      // ring: xs for k-blocks kb..kb+7 live in xs_r[0..7]; after consuming kb..kb+3 reload them 8 ahead
#pragma unroll 1
      for (int kb = 0; kb < KBI; kb += 8) {
        {
          const bool more = kb + 8 < KBI;
          MOE_STAGE_BODY(kb, xs_r[0], xs_r[1])
          MOE_STAGE_BODY(kb + 2, xs_r[2], xs_r[3])
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kn = more ? kb + 8 + i : kb + 8 + i - KBI;
            xs_r[i] = __ldg((more ? xsp : nxsp) + (size_t)krot(kn, more ? KBI : NKB) * p.R_max);
          }
        }
        {
          const bool more = kb + 12 < KBI;
          MOE_STAGE_BODY(kb + 4, xs_r[4], xs_r[5])
          MOE_STAGE_BODY(kb + 6, xs_r[6], xs_r[7])
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int kn = more ? kb + 12 + i : kb + 12 + i - KBI;
            xs_r[4 + i] = __ldg((more ? xsp : nxsp) + (size_t)krot(kn, more ? KBI : NKB) * p.R_max);
          }
        }
      }
#undef MOE_STAGE_BODY
#undef MOE_ACCUMULATE
#if MOE_BF16ACC
#pragma unroll
      for (int c = 0; c < 32; ++c) { float2 f = __bfloat1622float2(acc2[c]); acc[2 * c] = f.x; acc[2 * c + 1] = f.y; }
#endif
      ITEM_STAMP(1);

      // ---- item epilogue ----
      if (it.type == IT_LIN) {
        if (valid) {
          __nv_bfloat16* dst = p.L + (size_t)grow * INTER + it.mb * 128 + col0;
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            uint4 o;
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
            for (int j = 0; j < 4; ++j) o2[j] = __floats2bfloat162_rn(acc[c * 8 + j * 2], acc[c * 8 + j * 2 + 1]);
            *reinterpret_cast<uint4*>(dst + c * 8) = o;
          }
        }
        named_bar_sync(1, 256);
        if (tid == 0) { __threadfence(); st_release_gpu(p.flags + it.idx, epoch); }
      } else if (it.type == IT_ACT) {
        float amax = 0.f;
        if (valid) {
          const __nv_bfloat16* src = p.L + (size_t)grow * INTER + it.mb * 128 + col0;
          uint4 lraw[8];
#pragma unroll
          for (int c = 0; c < 8; ++c) lraw[c] = __ldcg(reinterpret_cast<const uint4*>(src + c * 8));
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const __nv_bfloat162* l2 = reinterpret_cast<const __nv_bfloat162*>(&lraw[c]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              float2 lin = __bfloat1622float2(l2[j]);
              const float x2a = acc[c * 8 + j * 2], x2b = acc[c * 8 + j * 2 + 1];
              const float ea = __expf(fminf(-x2a, 80.f)), eb = __expf(fminf(-x2b, 80.f));
              const float va = lin.x * x2a * __fdividef(1.0f, 1.0f + ea);
              const float vb = lin.y * x2b * __fdividef(1.0f, 1.0f + eb);
              acc[c * 8 + j * 2] = va;
              acc[c * 8 + j * 2 + 1] = vb;
              amax = fmaxf(amax, fmaxf(fabsf(va), fabsf(vb)));
            }
          }
        }
        s_amax[wg * 128 + row] = amax;
        named_bar_sync(1, 256);
        amax = fmaxf(amax, s_amax[(wg ^ 1) * 128 + row]);
        const float scale = amax > 0.f ? amax * (1.0f / 448.0f) : 1.0f;
        const float inv = 1.0f / scale;
        if (valid) {
          uint8_t* dst = p.C + (size_t)grow * INTER + it.mb * 128 + col0;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            uint32_t packed[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              __nv_fp8x4_e4m3 q4(make_float4(acc[c * 16 + j * 4] * inv, acc[c * 16 + j * 4 + 1] * inv,
                                             acc[c * 16 + j * 4 + 2] * inv, acc[c * 16 + j * 4 + 3] * inv));
              packed[j] = *reinterpret_cast<uint32_t*>(&q4);
            }
            *reinterpret_cast<uint4*>(dst + c * 16) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
          }
          if (wg == 0) p.c_scale[(size_t)it.mb * p.R_max + grow] = scale;
        }
        named_bar_sync(1, 256);
        if (tid == 0) { __threadfence(); atomicAdd(p.act_done + it.e, 1); }
      } else {
        if (valid) {
          __nv_bfloat16* dst = p.P + (size_t)grow * HIDDEN + it.mb * 128 + col0;
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            uint4 o;
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
            for (int j = 0; j < 4; ++j) o2[j] = __floats2bfloat162_rn(acc[c * 8 + j * 2], acc[c * 8 + j * 2 + 1]);
            *reinterpret_cast<uint4*>(dst + c * 8) = o;
          }
        }
      }
      ITEM_STAMP(7);
      item = next_item;
      it = nit;
      xsp = nxsp;
    }
    if (MOE_TRACE && warp == 4 && lane == 0) g_cta_time[0][blockIdx.x][1] = gtimer();
  }

  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    tmem_dealloc(tmem_base, TMEM_COLS);
  }
}

// ----------------------------------------------------------------------------------------------
// K5: finalize. One CTA per token: out[t] = sum_k w[t,k] * P[row(t,k)]
// ----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(896) finalize_kernel(const __nv_bfloat16* __restrict__ P, const int* __restrict__ pair_row,
                                                       const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ out,
                                                       int* expert_count) {
  const int t = blockIdx.x;
  if (t == 0 && threadIdx.x < E_LOCAL) expert_count[threadIdx.x] = 0;  // reset for the next call
  int rows[TOPK];
  float w[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    rows[k] = __ldg(pair_row + t * TOPK + k);
    w[k] = __ldg(topk_w + t * TOPK + k);
  }
  const int v = threadIdx.x;
  uint4 raw[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    raw[k] = make_uint4(0, 0, 0, 0);
    if (rows[k] >= 0) raw[k] = __ldg(reinterpret_cast<const uint4*>(P + (size_t)rows[k] * HIDDEN + v * 8));
  }
  float accv[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) accv[j] = 0.f;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    if (rows[k] >= 0) {
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw[k]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(b2[j]);
        accv[2 * j] = fmaf(w[k], f.x, accv[2 * j]);
        accv[2 * j + 1] = fmaf(w[k], f.y, accv[2 * j + 1]);
      }
    }
  }
  uint4 o;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int j = 0; j < 4; ++j) o2[j] = __floats2bfloat162_rn(accv[2 * j], accv[2 * j + 1]);
  *reinterpret_cast<uint4*>(out + (size_t)t * HIDDEN + v * 8) = o;
}

// ----------------------------------------------------------------------------------------------
// Host side
// ----------------------------------------------------------------------------------------------
typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static PFN_encodeTiled get_encode_fn() {
  static PFN_encodeTiled fn = nullptr;
  if (!fn) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult qres;
    cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &qres);
    TORCH_CHECK(err == cudaSuccess && ptr != nullptr, "cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<PFN_encodeTiled>(ptr);
  }
  return fn;
}

// rank-2 or rank-3 fp8 tensor map, box = {128 (inner), box_rows, 1}
static CUtensorMap make_tmap(const void* base, int rank, const cuuint64_t* dims, const cuuint64_t* strides_bytes, int box_rows) {
  CUtensorMap m;
  cuuint32_t box[3] = {128u, (cuuint32_t)box_rows, 1u};
  cuuint32_t estr[3] = {1u, 1u, 1u};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides_bytes, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

struct Workspace {
  int T = 0;
  int R_max = 0;
  torch::Tensor topk_ids, topk_w, pair_slot, pair_row, expert_count, expert_offset, item_prefix1, item_prefix2, row_token, row_w, meta;
  torch::Tensor Xp, xsp, C, cs, P, L, flags, act_done;
  XMaps xmaps_x, xmaps_c;
  bool tmaps_ready = false;
};

static Workspace& get_ws(int T, torch::Device dev) {
  static Workspace ws;
  if (ws.T < T) {
    ws.T = T;
    ws.R_max = T * TOPK + E_LOCAL * N_PAD;
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto bf = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
    ws.topk_ids = torch::empty({T, TOPK}, i32);
    ws.topk_w = torch::empty({T, TOPK}, f32);
    ws.pair_slot = torch::empty({T, TOPK}, i32);
    ws.pair_row = torch::empty({T, TOPK}, i32);
    ws.expert_count = torch::zeros({E_LOCAL}, i32);
    ws.expert_offset = torch::empty({E_LOCAL + 1}, i32);
    ws.item_prefix1 = torch::empty({E_LOCAL + 1}, i32);
    ws.item_prefix2 = torch::empty({E_LOCAL + 1}, i32);
    ws.row_token = torch::empty({ws.R_max}, i32);
    ws.row_w = torch::empty({ws.R_max}, f32);
    ws.meta = torch::zeros({META_SIZE}, i32);
    ws.Xp = torch::zeros({ws.R_max, HIDDEN}, u8);
    ws.xsp = torch::zeros({KB_HIDDEN, ws.R_max}, f32);
    ws.C = torch::zeros({ws.R_max, INTER}, u8);
    ws.cs = torch::zeros({KB_INTER, ws.R_max}, f32);
    ws.P = torch::empty({ws.R_max, HIDDEN}, bf);
    ws.L = torch::empty({ws.R_max, INTER}, bf);
    ws.flags = torch::zeros({4096}, i32);
    ws.act_done = torch::zeros({E_LOCAL}, i32);
    cuuint64_t dx[2] = {(cuuint64_t)HIDDEN, (cuuint64_t)ws.R_max};
    cuuint64_t sx[2] = {(cuuint64_t)HIDDEN, 0};
    cuuint64_t dc[2] = {(cuuint64_t)INTER, (cuuint64_t)ws.R_max};
    cuuint64_t sc[2] = {(cuuint64_t)INTER, 0};
    for (int i = 0; i < 8; ++i) {
      ws.xmaps_x.m[i] = make_tmap(ws.Xp.data_ptr(), 2, dx, sx, 16 * (i + 1));
      ws.xmaps_c.m[i] = make_tmap(ws.C.data_ptr(), 2, dc, sc, 16 * (i + 1));
    }
    ws.tmaps_ready = true;
  }
  return ws;
}

static CUtensorMap& weight_tmap(const torch::Tensor& w, int rows, int cols, int box_rows) {
  static std::unordered_map<const void*, CUtensorMap> cache;
  const void* ptr = w.data_ptr();
  auto itr = cache.find(ptr);
  if (itr != cache.end()) return itr->second;
  cuuint64_t d[3] = {(cuuint64_t)cols, (cuuint64_t)rows, (cuuint64_t)E_LOCAL};
  cuuint64_t s[3] = {(cuuint64_t)cols, (cuuint64_t)cols * rows, 0};
  CUtensorMap m = make_tmap(ptr, 3, d, s, box_rows);
  return cache.emplace(ptr, m).first->second;
}

static int num_sms() {
  static int n = 0;
  if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  return n;
}

void run(torch::Tensor routing_logits, torch::Tensor routing_bias, torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
         torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale, torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
         int64_t local_expert_offset, double routed_scaling_factor, torch::Tensor output) {
  const int T = routing_logits.size(0);
  c10::cuda::CUDAGuard guard(hidden_states.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Workspace& ws = get_ws(T, hidden_states.device());

  // K1
  {
    RoutingParams rp;
    rp.logits = routing_logits.data_ptr<float>();
    rp.bias = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
    rp.T = T;
    rp.local_offset = (int)local_expert_offset;
    rp.rsf = (float)routed_scaling_factor;
    rp.topk_ids = ws.topk_ids.data_ptr<int>();
    rp.topk_w = ws.topk_w.data_ptr<float>();
    rp.pair_slot = ws.pair_slot.data_ptr<int>();
    rp.expert_count = ws.expert_count.data_ptr<int>();
    rp.expert_offset = ws.expert_offset.data_ptr<int>();
    rp.item_prefix1 = ws.item_prefix1.data_ptr<int>();
    rp.item_prefix2 = ws.item_prefix2.data_ptr<int>();
    rp.meta = ws.meta.data_ptr<int>();
    rp.act_done = ws.act_done.data_ptr<int>();
    const int grid = (T + ROUTE_WARPS - 1) / ROUTE_WARPS;
    routing_kernel<<<grid, ROUTE_WARPS * 32, 0, stream>>>(rp);
  }
  // K2
  gather_kernel<<<T * TOPK + E_LOCAL * N_PAD, 224, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(hidden_states.data_ptr()), hidden_states_scale.data_ptr<float>(), T,
      ws.topk_ids.data_ptr<int>(), ws.topk_w.data_ptr<float>(), ws.pair_slot.data_ptr<int>(), ws.expert_offset.data_ptr<int>(),
      ws.expert_count.data_ptr<int>(), (int)local_expert_offset, ws.row_token.data_ptr<int>(), ws.row_w.data_ptr<float>(),
      ws.pair_row.data_ptr<int>(), ws.Xp.data_ptr<uint8_t>(), ws.xsp.data_ptr<float>(), ws.R_max);
  // K3 (fused GEMM1 + GEMM2)
  {
    static bool attr = false;
    if (!attr) {
      cudaFuncSetAttribute(fused_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX_DYN);
      attr = true;
    }
    GemmParams gp{};
    gp.expert_offset = ws.expert_offset.data_ptr<int>();
    gp.item_prefix1 = ws.item_prefix1.data_ptr<int>();
    gp.item_prefix2 = ws.item_prefix2.data_ptr<int>();
    gp.meta = ws.meta.data_ptr<int>();
    gp.act_done = ws.act_done.data_ptr<int>();
    gp.w1_scale = gemm1_weights_scale.data_ptr<float>();
    gp.w2_scale = gemm2_weights_scale.data_ptr<float>();
    gp.xs = ws.xsp.data_ptr<float>();
    gp.cs = ws.cs.data_ptr<float>();
    gp.R_max = ws.R_max;
    gp.L = reinterpret_cast<__nv_bfloat16*>(ws.L.data_ptr());
    gp.flags = ws.flags.data_ptr<int>();
    gp.C = ws.C.data_ptr<uint8_t>();
    gp.c_scale = ws.cs.data_ptr<float>();
    gp.P = reinterpret_cast<__nv_bfloat16*>(ws.P.data_ptr());
    CUtensorMap& tw1 = weight_tmap(gemm1_weights, 2 * INTER, HIDDEN, 128);
    CUtensorMap& tw2 = weight_tmap(gemm2_weights, HIDDEN, INTER, 128);
    fused_gemm_kernel<<<num_sms(), GEMM_THREADS, SMEM_MAX_DYN, stream>>>(tw1, tw2, ws.xmaps_x, ws.xmaps_c, gp);
  }
  // K5
  finalize_kernel<<<T, 896, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(ws.P.data_ptr()), ws.pair_row.data_ptr<int>(),
                                         ws.topk_w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                                         ws.expert_count.data_ptr<int>());
}

torch::Tensor debug_trace() {
  auto t = torch::empty({2, 3, TRACE_N, 4}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  cudaMemcpyFromSymbol(t.data_ptr(), g_trace, sizeof(long long) * 2 * 3 * TRACE_N * 4);
  return t;
}
std::vector<torch::Tensor> debug_times() {
  auto a = torch::zeros({2, 256, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto b = torch::zeros({2, 256, 32, 8}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  cudaMemcpyFromSymbol(a.data_ptr(), g_cta_time, sizeof(long long) * 2 * 256 * 2);
  cudaMemcpyFromSymbol(b.data_ptr(), g_item_time, sizeof(long long) * 2 * 256 * 32 * 8);
  return {a, b};
}

// Debug accessors
std::vector<torch::Tensor> debug_buffers() {
  Workspace& ws = get_ws(1, torch::Device(torch::kCUDA, 0));
  return {ws.topk_ids, ws.topk_w, ws.pair_slot, ws.expert_count, ws.expert_offset, ws.row_token, ws.row_w, ws.meta,
          ws.Xp, ws.xsp, ws.C, ws.cs, ws.P};
}

}  // namespace moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &moe::run, "fp8 block-scale MoE (sm_100a)");
  m.def("debug_buffers", &moe::debug_buffers, "workspace buffers");
  m.def("debug_trace", &moe::debug_trace, "in-kernel trace");
  m.def("debug_times", &moe::debug_times, "per-CTA / per-item times");
}
