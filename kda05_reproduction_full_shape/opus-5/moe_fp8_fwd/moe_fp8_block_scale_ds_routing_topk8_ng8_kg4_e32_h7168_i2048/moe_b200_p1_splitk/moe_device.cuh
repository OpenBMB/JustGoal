#pragma once
// FP8 block-scale MoE for NVIDIA B200 (sm_100), CUDA 13.0.
//
// Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
//
// Design notes (measurements behind them are in docs/draft.md):
//
//  * 14 of 19 workloads have seq_len <= 62. There, total local token-expert
//    pairs ~= seq_len, so an active expert owns only 1-18 tokens and the kernel
//    is purely weight-bandwidth bound: the job is "stream the selected experts'
//    weights once at peak HBM bandwidth", not "do a big GEMM".
//  * Hence the MMA operands are *swapped*: weight rows are the MMA M dimension
//    and tokens are N. mma.sync.m16n8k32 takes N=8 tokens natively, so a 1-8
//    token expert wastes nothing, and `row.col` (both operands K-contiguous) is
//    exactly the natural layout of gemm1_weights[e] ([4096,7168]),
//    gemm2_weights[e] ([7168,2048]) and hidden_states ([T,7168]).
//  * The m16n8k32 e4m3 fragment layouts were decoded from the hardware
//    (dev/decode_mma.cu) because the published tables did not match. Only
//    self-consistency between how A/B are filled and how D is read is needed.
//    With g = lane>>2, j = lane&3:
//        A: reg0=A[2g][8j..8j+3]   reg1=A[2g+1][8j..8j+3]
//           reg2=A[2g][8j+4..+7]   reg3=A[2g+1][8j+4..+7]
//        B: reg0=B[g][8j..8j+3]    reg1=B[g][8j+4..+7]
//        D: reg0=D[2g][2j] reg1=D[2g][2j+1] reg2=D[2g+1][2j] reg3=D[2g+1][2j+1]
//    Lanes j=0..3 cover 32 contiguous bytes of a row, so weight fragments load
//    global->register directly with no smem staging; that keeps occupancy high,
//    and >=4 CTAs/SM is required to reach peak HBM bandwidth on this part.
//  * At tiny seq_len only a handful of experts are live, so tiling by
//    (expert, m_tile, n_tile) alone leaves most SMs idle. The K dimension is
//    therefore split `ksplit` ways, chosen on the host so the tile count always
//    fills the machine. Split-K forces GEMM1's SwiGLU into its own pass, which
//    is cheap: the intermediate is only `pairs x 4096`.
//  * CUPTI charges us the span from the first kernel's start to the last one's
//    end, so there is no host sync anywhere and the work queue is built on the
//    device by the routing kernel's last block.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace moe_b200 {

// ---------------------------------------------------------------- constants

constexpr int H = 7168;        // hidden_size
constexpr int I_SZ = 2048;     // intermediate_size
constexpr int G1N = 4096;      // gemm1_out_size
constexpr int E_LOCAL = 32;    // num_local_experts
constexpr int E_GLOBAL = 256;  // num_experts
constexpr int TOPK = 8;
constexpr int NGROUP = 8;
constexpr int KGROUP = 4;
constexpr int BLK = 128;                  // quantisation block
constexpr int HB = H / BLK;               // 56  k-blocks of gemm1 / n-blocks of gemm2
constexpr int IB = I_SZ / BLK;            // 16  k-blocks of gemm2
constexpr int G1B = G1N / BLK;            // 32
constexpr int GSIZE = E_GLOBAL / NGROUP;  // 32 experts per routing group

constexpr unsigned FULL = 0xffffffffu;
constexpr float NEG_INF = -3.0e38f;
constexpr float FP8_MAX = 448.0f;

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int NROWS = WARPS * 16;      // weight rows per CTA (one 16-row tile per warp)
constexpr int NTILE1 = G1N / NROWS;    // 64
constexpr int NTILE2 = H / NROWS;      // 112
constexpr int SM_COUNT = 148;

constexpr int ROUTE_WARPS = 8;
constexpr int ROUTE_THREADS = ROUTE_WARPS * 32;

// ---------------------------------------------------------------- workspace

struct Work {
  int* counts;    // [E_LOCAL]        routing scratch; always zero between calls
  int* cnt2;      // [E_LOCAL]        snapshot of counts taken by k_route
  int* offs;      // [E_LOCAL + 1]    exclusive scan -> pair index base; offs[32] = pairs
  int* g1_start;  // [E_LOCAL + 1]    gemm1 work-tile start per expert
  int* g2_start;  // [E_LOCAL + 1]    gemm2 work-tile start per expert
  int* meta;      // [8] {n_g1_tiles, n_g2_tiles, done_route, done_gemm2}
  int* elist;     // [E_LOCAL * cap]  token ids, expert-major
  float* ewt;     // [E_LOCAL * cap]  routing weights, expert-major
  __nv_bfloat16* g1p;  // [ksplit][pairs_cap][G1N]  split-K partials of gemm1
  __nv_fp8_e4m3* cq;   // [pairs_cap][I_SZ]         quantised SwiGLU output
  float* cs;           // [IB][pairs_cap]           its 1x128 scales
  int cap;
  int pairs_cap;
  int ksplit1;
  int ksplit2;
};

// ---------------------------------------------------------------- mma helpers

__device__ __forceinline__ void mma_m16n8k32(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// --- k-class permutation -------------------------------------------------
//
// The decoded fragment layout says lane (g,j) supplies k-classes [8j, 8j+8) of
// a 32-wide MMA step. Which *memory* byte backs which k-class is entirely up to
// us, as long as A and B agree: the MMA just sums over the 32 classes.
//
// The naive choice (class c of step ks <- byte 32*ks + c) makes each lane read
// four separate 8-byte pieces per 128-byte k block, and NCU showed that pins
// the kernel at 67-85% L1/TEX throughput while DRAM sits at ~22%.
//
// Instead use   class (8j + m) of step ks  <-  byte  32*j + 8*ks + m.
// Over ks = 0..3 and j = 0..3 that is still a bijection onto the 128 bytes, but
// now lane (g,j) owns the 32 *contiguous* bytes [32j, 32j+32) of the block and
// loads them as two uint4. Four lanes cover a full 128-byte line, so one warp
// instruction touches 8 rows x 128 B instead of 8 rows x 32 B.
//
// `w[n]` holds bytes [32j + 4n, +4), so MMA step ks takes w[2*ks] and w[2*ks+1].
__device__ __forceinline__ void load32(uint32_t* w, const uint8_t* p) {
  uint4 v0 = *reinterpret_cast<const uint4*>(p);
  uint4 v1 = *reinterpret_cast<const uint4*>(p + 16);
  w[0] = v0.x;
  w[1] = v0.y;
  w[2] = v0.z;
  w[3] = v0.w;
  w[4] = v1.x;
  w[5] = v1.y;
  w[6] = v1.z;
  w[7] = v1.w;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ---------------------------------------------------------------- routing

// One warp per token. Lane L owns experts [8L, 8L+8), so DeepSeek group g = L>>2
// is exactly lanes 4g..4g+3 and the per-group top-2 is a local top-2 of 8 plus
// two shuffles.
template <int MT>
__global__ __launch_bounds__(ROUTE_THREADS) void k_route(
    const float* __restrict__ logits, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out, int T, int local_off, float rsf, Work w) {
  __shared__ int sel_e[ROUTE_WARPS][TOPK];
  __shared__ float sel_s[ROUTE_WARPS][TOPK];
  __shared__ bool is_last;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * ROUTE_WARPS + warp;

  if (t < T) {
    // Zero this token's output row: the DPS output tensor is torch.empty, and
    // gemm2 only accumulates into it.
    uint4* orow = reinterpret_cast<uint4*>(out + (size_t)t * H);
    const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll 4
    for (int i = lane; i < H * 2 / 16; i += 32) orow[i] = z;

    const float4* lg = reinterpret_cast<const float4*>(logits + (size_t)t * E_GLOBAL + 8 * lane);
    float4 l0 = lg[0], l1 = lg[1];
    float s[8], swb[8];
    s[0] = 1.0f / (1.0f + expf(-l0.x));
    s[1] = 1.0f / (1.0f + expf(-l0.y));
    s[2] = 1.0f / (1.0f + expf(-l0.z));
    s[3] = 1.0f / (1.0f + expf(-l0.w));
    s[4] = 1.0f / (1.0f + expf(-l1.x));
    s[5] = 1.0f / (1.0f + expf(-l1.y));
    s[6] = 1.0f / (1.0f + expf(-l1.z));
    s[7] = 1.0f / (1.0f + expf(-l1.w));
#pragma unroll
    for (int i = 0; i < 8; ++i) swb[i] = s[i] + __bfloat162float(bias[8 * lane + i]);

    // per-group top-2 -> group score
    float m1 = NEG_INF, m2 = NEG_INF;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = swb[i];
      if (v > m1) {
        m2 = m1;
        m1 = v;
      } else if (v > m2) {
        m2 = v;
      }
    }
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      float o1 = __shfl_xor_sync(FULL, m1, off);
      float o2 = __shfl_xor_sync(FULL, m2, off);
      float n1 = fmaxf(m1, o1);
      float cand = (m1 >= o1) ? m2 : o2;
      float n2 = fmaxf(fminf(m1, o1), cand);
      m1 = n1;
      m2 = n2;
    }
    // Use the group leader's value so all 4 lanes of a group agree even if the
    // xor merge resolved a tie asymmetrically.
    float gscore = __shfl_sync(FULL, m1 + m2, lane & ~3);

    float gs[NGROUP];
#pragma unroll
    for (int g = 0; g < NGROUP; ++g) gs[g] = __shfl_sync(FULL, gscore, 4 * g);

    unsigned keep = 0;
#pragma unroll
    for (int it = 0; it < KGROUP; ++it) {
      float bv = NEG_INF;
      int bi = 0;
#pragma unroll
      for (int g = 0; g < NGROUP; ++g) {
        if (!((keep >> g) & 1u) && gs[g] > bv) {
          bv = gs[g];
          bi = g;
        }
      }
      keep |= 1u << bi;
    }
    if (!((keep >> (lane >> 2)) & 1u)) {
#pragma unroll
      for (int i = 0; i < 8; ++i) swb[i] = NEG_INF;
    }

    unsigned removed = 0;
    for (int it = 0; it < TOPK; ++it) {
      float bv = NEG_INF;
      int bi = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (!((removed >> i) & 1u) && swb[i] > bv) {
          bv = swb[i];
          bi = i;
        }
      }
      int gidx = 8 * lane + bi;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        float ov = __shfl_xor_sync(FULL, bv, off);
        int oi = __shfl_xor_sync(FULL, gidx, off);
        if (ov > bv || (ov == bv && oi < gidx)) {
          bv = ov;
          gidx = oi;
        }
      }
      if ((gidx >> 3) == lane) removed |= 1u << (gidx & 7);
      float sv = 0.0f;
      if ((gidx >> 3) == lane) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if ((gidx & 7) == i) sv = s[i];
      }
      sv = __shfl_sync(FULL, sv, gidx >> 3);
      if (lane == 0) {
        sel_e[warp][it] = gidx;
        sel_s[warp][it] = sv;
      }
    }
    __syncwarp();

    float wsum = 0.0f;
#pragma unroll
    for (int i = 0; i < TOPK; ++i) wsum += sel_s[warp][i];
    if (lane < TOPK) {
      int le = sel_e[warp][lane] - local_off;
      if (le >= 0 && le < E_LOCAL) {
        float wgt = sel_s[warp][lane] / (wsum + 1e-20f) * rsf;
        int pos = atomicAdd(&w.counts[le], 1);
        w.elist[le * w.cap + pos] = t;
        w.ewt[le * w.cap + pos] = wgt;
      }
    }
  }

  // The last block builds the work queue for both GEMMs.
  __threadfence();
  if (threadIdx.x == 0) is_last = (atomicAdd(&w.meta[2], 1) == gridDim.x - 1);
  __syncthreads();
  if (!is_last) return;

  if (threadIdx.x < 32) {
    const int l = threadIdx.x;
    const int c = w.counts[l];
    // Snapshot the counts and immediately clear them, so `counts` is already
    // zeroed for the next call and the GEMMs have no reset duty.
    w.cnt2[l] = c;
    w.counts[l] = 0;
    const int nmt = (c + MT - 1) / MT;
    int sc = c;
    int s1 = nmt * NTILE1 * w.ksplit1;
    int s2 = nmt * NTILE2 * w.ksplit2;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int a = __shfl_up_sync(FULL, sc, off);
      int b = __shfl_up_sync(FULL, s1, off);
      int d = __shfl_up_sync(FULL, s2, off);
      if (l >= off) {
        sc += a;
        s1 += b;
        s2 += d;
      }
    }
    w.offs[l + 1] = sc;
    w.g1_start[l + 1] = s1;
    w.g2_start[l + 1] = s2;
    if (l == 0) {
      w.offs[0] = 0;
      w.g1_start[0] = 0;
      w.g2_start[0] = 0;
      w.meta[2] = 0;  // ready for the next call
    }
    if (l == 31) {
      w.meta[0] = s1;
      w.meta[1] = s2;
    }
  }
}

// ---------------------------------------------------------------- gemm1

// Work tile = (expert, m_tile, n_tile, k_chunk) with n_tile fastest so that
// neighbouring tiles share the same token rows and k range (L2 reuse of the
// activations). Writes bf16 split-K partials; k_swiglu reduces them.
template <int MT, int CTAS>
__global__ __launch_bounds__(THREADS, CTAS) void k_gemm1(
    const __nv_fp8_e4m3* __restrict__ hs, const float* __restrict__ hs_scale,
    const __nv_fp8_e4m3* __restrict__ w13, const float* __restrict__ w13_scale, int T, Work w) {
  constexpr int TT = MT / 8;  // token tiles of 8
  __shared__ int s_start[E_LOCAL + 1];
  __shared__ int s_tok[MT];

  for (int i = threadIdx.x; i <= E_LOCAL; i += THREADS) s_start[i] = w.g1_start[i];
  __syncthreads();

  const int n_tiles = w.meta[0];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, j = lane & 3;
  const int kper = (HB + w.ksplit1 - 1) / w.ksplit1;

  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    int e = 0;
    while (e < E_LOCAL - 1 && tile >= s_start[e + 1]) ++e;
    int loc = tile - s_start[e];
    const int mt = loc / (NTILE1 * w.ksplit1);
    loc -= mt * NTILE1 * w.ksplit1;
    const int kc = loc / NTILE1;
    const int nt = loc - kc * NTILE1;

    const int kb0 = kc * kper;
    const int kb1 = min(HB, kb0 + kper);
    const int m0 = mt * MT;
    const int M = min(MT, w.cnt2[e] - m0);
    if (M <= 0 || kb0 >= kb1) continue;

    __syncthreads();
    if (threadIdx.x < MT)
      s_tok[threadIdx.x] = (threadIdx.x < M) ? w.elist[e * w.cap + m0 + threadIdx.x] : 0;
    __syncthreads();

    const int row0 = nt * NROWS + warp * 16;  // this warp's 16 weight rows
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w13) + (size_t)e * G1N * H +
                        (size_t)(row0 + 2 * g) * H + 32 * j;
    const uint8_t* hb = reinterpret_cast<const uint8_t*>(hs) + 32 * j;
    // weight-scale block: a 64-row tile never straddles a 128-row scale block
    const float* ws = w13_scale + ((size_t)e * G1B + row0 / BLK) * HB;

    int tk[TT][2];
    const uint8_t* bp[TT];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
      tk[tt][0] = s_tok[tt * 8 + 2 * j];
      tk[tt][1] = s_tok[tt * 8 + 2 * j + 1];
      bp[tt] = hb + (size_t)s_tok[tt * 8 + g] * H;
    }

    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[tt][q] = 0.0f;

#pragma unroll 2
    for (int kb = kb0; kb < kb1; ++kb) {
      float raw[TT][4];
#pragma unroll
      for (int tt = 0; tt < TT; ++tt)
#pragma unroll
        for (int q = 0; q < 4; ++q) raw[tt][q] = 0.0f;

      // one 128-byte k block: 32 contiguous bytes per lane per row
      const int koff = kb * BLK;
      uint32_t aw0[8], aw1[8], bw[TT][8];
      load32(aw0, wp + koff);
      load32(aw1, wp + koff + H);
#pragma unroll
      for (int tt = 0; tt < TT; ++tt) load32(bw[tt], bp[tt] + koff);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t a[4] = {aw0[2 * ks], aw1[2 * ks], aw0[2 * ks + 1], aw1[2 * ks + 1]};
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
          const uint32_t b[2] = {bw[tt][2 * ks], bw[tt][2 * ks + 1]};
          mma_m16n8k32(raw[tt], a, b);
        }
      }

      const float sw = ws[kb];
      const float* hsk = hs_scale + (size_t)kb * T;
#pragma unroll
      for (int tt = 0; tt < TT; ++tt) {
        const float s0 = sw * hsk[tk[tt][0]];
        const float s1 = sw * hsk[tk[tt][1]];
        acc[tt][0] += raw[tt][0] * s0;
        acc[tt][1] += raw[tt][1] * s1;
        acc[tt][2] += raw[tt][2] * s0;
        acc[tt][3] += raw[tt][3] * s1;
      }
    }

    // store partials: row = G1N column, col = token
    __nv_bfloat16* dst = w.g1p + ((size_t)kc * w.pairs_cap + w.offs[e] + m0) * G1N;
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int mm = tt * 8 + 2 * j + (q & 1);
        if (mm >= M) continue;
        dst[(size_t)mm * G1N + row0 + 2 * g + (q >> 1)] = __float2bfloat16(acc[tt][q]);
      }
    }
  }
}

// ---------------------------------------------------------------- swiglu

// One warp per (pair, i_tile of 128 columns): reduce the split-K partials,
// apply SwiGLU, then quantise the 128-wide block to FP8 with one scale.
__global__ __launch_bounds__(256) void k_swiglu(Work w) {
  const int pairs = w.offs[E_LOCAL];
  const int total = pairs * IB;
  const int warp = (blockIdx.x * 256 + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = (gridDim.x * 256) >> 5;
  const int ks = w.ksplit1;

  for (int idx = warp; idx < total; idx += nwarps) {
    const int pair = idx / IB;
    const int it = idx - pair * IB;
    const int c0 = it * BLK + lane * 4;  // 4 columns per lane

    float x1[4] = {0, 0, 0, 0}, x2[4] = {0, 0, 0, 0};
    for (int k = 0; k < ks; ++k) {
      const __nv_bfloat16* p = w.g1p + ((size_t)k * w.pairs_cap + pair) * G1N;
      const __nv_bfloat162* q1 = reinterpret_cast<const __nv_bfloat162*>(p + c0);
      const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(p + I_SZ + c0);
      __nv_bfloat162 a0 = q1[0], a1 = q1[1], b0 = q2[0], b1 = q2[1];
      x1[0] += __low2float(a0);
      x1[1] += __high2float(a0);
      x1[2] += __low2float(a1);
      x1[3] += __high2float(a1);
      x2[0] += __low2float(b0);
      x2[1] += __high2float(b0);
      x2[2] += __low2float(b1);
      x2[3] += __high2float(b1);
    }

    float c[4], mx = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c[i] = silu(x2[i]) * x1[i];
      mx = fmaxf(mx, fabsf(c[i]));
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, off));

    const float sc = (mx > 0.0f) ? mx / FP8_MAX : 1.0f;
    const float inv = (mx > 0.0f) ? FP8_MAX / mx : 0.0f;
    if (lane == 0) w.cs[(size_t)it * w.pairs_cap + pair] = sc;

    __nv_fp8_e4m3 o8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o8[i] = __nv_fp8_e4m3(c[i] * inv);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(w.cq) + (size_t)pair * I_SZ + c0) =
        *reinterpret_cast<const uint32_t*>(o8);
  }
}

// ---------------------------------------------------------------- gemm2

// Work tile = (expert, m_tile, n_tile, k_chunk). The output accumulation is
// already atomic, so split-K costs nothing extra here.
template <int MT, int CTAS>
__global__ __launch_bounds__(THREADS, CTAS) void k_gemm2(
    const __nv_fp8_e4m3* __restrict__ w2, const float* __restrict__ w2_scale,
    __nv_bfloat16* __restrict__ out, Work w) {
  constexpr int TT = MT / 8;
  __shared__ int s_start[E_LOCAL + 1];
  __shared__ int s_tok[MT];
  __shared__ float s_wt[MT];

  for (int i = threadIdx.x; i <= E_LOCAL; i += THREADS) s_start[i] = w.g2_start[i];
  __syncthreads();

  const int n_tiles = w.meta[1];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, j = lane & 3;
  const int kper = (IB + w.ksplit2 - 1) / w.ksplit2;
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out);

  for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
    int e = 0;
    while (e < E_LOCAL - 1 && tile >= s_start[e + 1]) ++e;
    int loc = tile - s_start[e];
    const int mt = loc / (NTILE2 * w.ksplit2);
    loc -= mt * NTILE2 * w.ksplit2;
    const int kc = loc / NTILE2;
    const int nt = loc - kc * NTILE2;

    const int kb0 = kc * kper;
    const int kb1 = min(IB, kb0 + kper);
    const int m0 = mt * MT;
    const int M = min(MT, w.cnt2[e] - m0);
    if (M <= 0 || kb0 >= kb1) continue;

    __syncthreads();
    if (threadIdx.x < MT) {
      const bool ok = threadIdx.x < M;
      s_tok[threadIdx.x] = ok ? w.elist[e * w.cap + m0 + threadIdx.x] : 0;
      s_wt[threadIdx.x] = ok ? w.ewt[e * w.cap + m0 + threadIdx.x] : 0.0f;
    }
    __syncthreads();

    const size_t pbase = (size_t)w.offs[e] + m0;
    const int row0 = nt * NROWS + warp * 16;
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w2) + (size_t)e * H * I_SZ +
                        (size_t)(row0 + 2 * g) * I_SZ + 32 * j;
    const uint8_t* cb = reinterpret_cast<const uint8_t*>(w.cq) + 32 * j;
    const float* ws = w2_scale + ((size_t)e * HB + row0 / BLK) * IB;

    float acc[TT][4];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[tt][q] = 0.0f;

#pragma unroll 2
    for (int kb = kb0; kb < kb1; ++kb) {
      float raw[TT][4];
#pragma unroll
      for (int tt = 0; tt < TT; ++tt)
#pragma unroll
        for (int q = 0; q < 4; ++q) raw[tt][q] = 0.0f;

      const int koff = kb * BLK;
      uint32_t aw0[8], aw1[8], bw[TT][8];
      load32(aw0, wp + koff);
      load32(aw1, wp + koff + I_SZ);
#pragma unroll
      for (int tt = 0; tt < TT; ++tt)
        load32(bw[tt], cb + (pbase + tt * 8 + g) * I_SZ + koff);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t a[4] = {aw0[2 * ks], aw1[2 * ks], aw0[2 * ks + 1], aw1[2 * ks + 1]};
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
          const uint32_t b[2] = {bw[tt][2 * ks], bw[tt][2 * ks + 1]};
          mma_m16n8k32(raw[tt], a, b);
        }
      }

      const float sw = ws[kb];
      const float* csk = w.cs + (size_t)kb * w.pairs_cap + pbase;
#pragma unroll
      for (int tt = 0; tt < TT; ++tt) {
        const float s0 = sw * csk[tt * 8 + 2 * j];
        const float s1 = sw * csk[tt * 8 + 2 * j + 1];
        acc[tt][0] += raw[tt][0] * s0;
        acc[tt][1] += raw[tt][1] * s1;
        acc[tt][2] += raw[tt][2] * s0;
        acc[tt][3] += raw[tt][3] * s1;
      }
    }

    // Lane holds output columns (row0 + 2g) and (row0 + 2g + 1) -- adjacent, so
    // each token's pair packs into one 4-byte-aligned bfloat162 atomic.
    const int hcol = row0 + 2 * g;
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int mm = tt * 8 + 2 * j + half;
        if (mm >= M) continue;
        const float wt = s_wt[mm];
        atomicAdd(&o2[((size_t)s_tok[mm] * H + hcol) >> 1],
                  __floats2bfloat162_rn(acc[tt][half] * wt, acc[tt][2 + half] * wt));
      }
    }
  }
}


// ---------------------------------------------------------------- launch glue

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// Expected number of live local experts; used only to size the K split.
inline int est_active(int T) {
  double pairs = (double)T;  // T * topk * E_local / E_global
  double p = 1.0 - pow(1.0 - 1.0 / (double)E_LOCAL, pairs);
  int a = (int)(E_LOCAL * p + 0.5);
  return a < 1 ? 1 : (a > E_LOCAL ? E_LOCAL : a);
}

template <int MT, int CTAS>
void launch(int T, const Work& w, cudaStream_t stream, const void* logits, const void* bias,
            const void* hs, const void* hs_scale, const void* w13, const void* w13_scale,
            const void* w2, const void* w2_scale, void* out, int local_off, float rsf) {
  const int route_blocks = (T + ROUTE_WARPS - 1) / ROUTE_WARPS;
  k_route<MT><<<route_blocks, ROUTE_THREADS, 0, stream>>>(
      static_cast<const float*>(logits), static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(out), T, local_off, rsf, w);

  const int grid = SM_COUNT * CTAS;
  k_gemm1<MT, CTAS><<<grid, THREADS, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(hs), static_cast<const float*>(hs_scale),
      static_cast<const __nv_fp8_e4m3*>(w13), static_cast<const float*>(w13_scale), T, w);

  k_swiglu<<<SM_COUNT * 2, 256, 0, stream>>>(w);

  k_gemm2<MT, CTAS><<<grid, THREADS, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(w2), static_cast<const float*>(w2_scale),
      static_cast<__nv_bfloat16*>(out), w);
}

}  // namespace moe_b200
