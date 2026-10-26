// FP8 block-scale MoE (DeepSeek-V3 no-aux routing) for NVIDIA B200 / sm_100.
//
// definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
//
// Design notes (full analysis in docs/draft.md):
//
//  * Each token selects 8 of 256 experts but only 32 are local, so the expected
//    number of local (token, expert) pairs is exactly seq_len. The arithmetic is
//    tiny while the weight footprint is fixed at 44 MB/expert (1409 MB for all
//    32). 17 of 19 workloads are purely weight-bandwidth bound, so this is a
//    weight streamer that happens to do some MMAs.
//
//  * Tensor cores via `mma.sync.m16n8k32.e4m3`; tcgen05 needs sm_100a, which the
//    tvm-ffi builder does not emit. Measured 2.0 PFLOP/s at 8 warps/SM = 216
//    weight-bytes/cycle/SM against the 25 bytes/cycle/SM that 7.3 TB/s demands.
//
//  * The weights are NOT staged through shared memory. Each weight byte is read
//    by exactly one warp, so SMEM would only buy coalescing -- and measurement
//    (dev/probe/micro4.cu) shows the direct global->register fragment pattern
//    reaches 6.55 TB/s, matching the best staged pattern, while freeing all the
//    SMEM. That lifts occupancy from 1 CTA/SM (12.5% of warps) to 4 CTA/SM and
//    removes the barrier and SMEM round-trip stalls that NCU flagged on v1.
//
//  * Both MMA operands are read as ONE 16-byte contiguous chunk per thread,
//    covering two k-steps: the K axis is a pure summation index, so relabelling
//    it identically in A and B leaves the dot product unchanged. The
//    relabelling stays inside a 64-byte window, which is why the intermediate
//    tensor is quantised with one scale per 64 i-values -- a finer block would
//    straddle the window and corrupt the promotion.

#ifndef MOE_STANDALONE
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#endif

#include <cuda_runtime.h>
#include <cuda.h>          // CUtensorMap for the tcgen05 path's TMA descriptors
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled_v12000
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace moe {

// ------------------------------------------------------------------ geometry
constexpr int H = 7168;
constexpr int IIN = 2048;              // intermediate_size
constexpr int G1O = 4096;              // 2 * intermediate_size
constexpr int EG = 256;                // global experts
constexpr int EL = 32;                 // local experts
constexpr int TOPK = 8;
constexpr int NGROUP = 8;
constexpr int GSZ = EG / NGROUP;       // 32
constexpr int TOPKG = 4;
constexpr int BLK = 128;               // weight quantisation block
constexpr int HB = H / BLK;            // 56
constexpr int IB = IIN / BLK;          // 16

constexpr int NWARP = 8;               // GEMM2's CTA width
constexpr int NTHR = NWARP * 32;
constexpr int WN = 8;                  // weight rows per warp == one MMA N tile
constexpr int NI = NWARP * WN;         // 64 weight rows per CTA in GEMM1
// The intermediate C is quantised with one scale per 128 i-values, matching
// W2's own block size. GEMM1 no longer has to own a whole scale block, because
// the amax is taken in the separate SwiGLU/quantise pass.
constexpr int CSCALE = 128;
constexpr int CSB = IIN / CSCALE;      // 16 intermediate-scale blocks per pair
constexpr int WIN = 64;                // bytes of K per fragment read (relabel window)

// GEMM1 stages the activations over K; 7168 = ACH * NACH must hold and ACH
// must be a multiple of BLK so scale blocks never straddle a stage.
#ifndef MOE_ACH
#define MOE_ACH 1024
#endif
constexpr int ACH = MOE_ACH;           // bytes of K per activation SMEM stage
constexpr int ASS = ACH + 64;          // activation SMEM row stride (64 mod 128)
constexpr int AVEC = ACH / 16;
static_assert(H % ACH == 0, "ACH must divide the hidden size");
static_assert(ACH % BLK == 0, "ACH must be a multiple of the quant block");

// Weight prefetch depth in 64-byte windows, per kernel. dev/probe/micro5.cu:
// what matters is (streams x PFW) uint4 in flight per thread, traded against
// the register budget that sets CTAs/SM. GEMM1 reads two streams (the two
// halves of W13) so it needs half the depth of GEMM2.
#ifndef MOE_PFW1
#define MOE_PFW1 4
#endif
#ifndef MOE_PFW2
#define MOE_PFW2 4
#endif
#ifndef MOE_MINBLK1
#define MOE_MINBLK1 2
#endif
#ifndef MOE_MINBLK2
#define MOE_MINBLK2 2
#endif

constexpr float FP8_MAX = 448.0f;

// ------------------------------------------------------------------- helpers
__device__ __forceinline__ void cp_async_ca16(void *dst, const void *src) {
  uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Streaming weight read. This must be an intrinsic rather than inline asm:
// inline asm forces the address into a 64-bit register pair for every load,
// which cost ~3 extra integer ops per load and drove the ALU pipe to 50% of
// peak (NCU: 66 instructions per 64-byte window instead of ~20). __ldcg lets
// nvcc fold compile-time offsets into the LDG immediate field, and .cg still
// keeps the stream out of L1.
__device__ __forceinline__ int4 ld_w(const uint8_t *p) {
  return __ldcg(reinterpret_cast<const int4 *>(p));
}

__device__ __forceinline__ void mma_e4m3(float *d, const uint32_t *a, const uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// monotone float -> uint32 so (value, -index) packs into one comparable key
__device__ __forceinline__ uint64_t pack_key(float v, int e) {
  uint32_t b = __float_as_uint(v);
  uint32_t u = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return (static_cast<uint64_t>(u) << 32) | static_cast<uint32_t>(EG - 1 - e);
}

// =========================================================================
//  K1: no-aux routing + output zeroing + per-expert bucket build
// =========================================================================
constexpr int ROUTE_WARPS = 8;
constexpr int ROUTE_THREADS = ROUTE_WARPS * 32;

__global__ __launch_bounds__(ROUTE_THREADS) void route_kernel(
    const float *__restrict__ logits,         // [T, 256]
    const __nv_bfloat16 *__restrict__ bias,   // [256]
    __nv_bfloat16 *__restrict__ out,          // [T, H] -> zeroed
    int *__restrict__ cnt,                    // [EL]  (pre-zeroed)
    int *__restrict__ cnt_next,               // [EL]  zeroed here for next call
    int *__restrict__ pair_tok,               // [EL, cap]
    float *__restrict__ pair_w,               // [EL, cap]
    int T, int cap, int expert_off, float rsf) {
  if (threadIdx.x < EL) cnt_next[threadIdx.x] = 0;

  {  // zero the output (grid-stride, 16 B stores)
    const size_t nvec = static_cast<size_t>(T) * (H / 8);
    const uint4 z = make_uint4(0, 0, 0, 0);
    uint4 *p = reinterpret_cast<uint4 *>(out);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nvec;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
      p[i] = z;
  }

  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * ROUTE_WARPS + (threadIdx.x >> 5);
  if (token >= T) return;

  const float *lg = logits + static_cast<size_t>(token) * EG;

  float s[NGROUP], sb[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; ++g) {
    const float v = lg[g * GSZ + lane];
    const float sig = 1.0f / (1.0f + __expf(-v));
    s[g] = sig;
    sb[g] = sig + __bfloat162float(bias[g * GSZ + lane]);
  }

  // group score = sum of the top-2 selection scores in the group
  float gs[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; ++g) {
    float m1 = sb[g], m2 = -INFINITY;
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      const float o1 = __shfl_xor_sync(0xffffffffu, m1, d);
      const float o2 = __shfl_xor_sync(0xffffffffu, m2, d);
      const float n1 = fmaxf(m1, o1);
      const float n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
      m1 = n1;
      m2 = n2;
    }
    gs[g] = m1 + m2;
  }

  // group bookkeeping is kept in bit masks: a bool array indexed by a runtime
  // value (the argmax position) would be placed in local memory, and NCU
  // showed that turning the whole route kernel into a long_scoreboard stall
  // chain -- 18.9 us of GPU time at seq_len = 1.
  uint32_t kept = 0;
#pragma unroll
  for (int r = 0; r < TOPKG; ++r) {
    float best = -INFINITY;
    int bi = 0;
#pragma unroll
    for (int g = 0; g < NGROUP; ++g)
      if (!((kept >> g) & 1u) && gs[g] > best) {
        best = gs[g];
        bi = g;
      }
    kept |= 1u << bi;
  }

  uint32_t sel = 0;
#pragma unroll 1
  for (int r = 0; r < TOPK; ++r) {
    uint64_t key = 0;
#pragma unroll
    for (int g = 0; g < NGROUP; ++g)
      if (((kept >> g) & 1u) && !((sel >> g) & 1u)) {
        const uint64_t k = pack_key(sb[g], g * GSZ + lane);
        key = k > key ? k : key;
      }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
      const uint64_t o = __shfl_xor_sync(0xffffffffu, key, d);
      key = o > key ? o : key;
    }
    const int e_sel = EG - 1 - static_cast<int>(key & 0xffffffffu);
    if ((e_sel & (GSZ - 1)) == lane) sel |= 1u << (e_sel >> 5);
  }

  float mysum = 0.f;
#pragma unroll
  for (int g = 0; g < NGROUP; ++g)
    if ((sel >> g) & 1u) mysum += s[g];
#pragma unroll
  for (int d = 16; d; d >>= 1) mysum += __shfl_xor_sync(0xffffffffu, mysum, d);
  const float inv = rsf / (mysum + 1e-20f);

#pragma unroll
  for (int g = 0; g < NGROUP; ++g) {
    if (!((sel >> g) & 1u)) continue;
    const int le = g * GSZ + lane - expert_off;
    if (le < 0 || le >= EL) continue;
    const int slot = atomicAdd(&cnt[le], 1);
    if (slot < cap) {
      pair_tok[le * cap + slot] = token;
      pair_w[le * cap + slot] = s[g] * inv;
    }
  }
}

// =========================================================================
//  Work-item plan shared by the two GEMM kernels
// =========================================================================
struct Plan {
  int off[EL];
  int nm[EL];
  int item_pre[EL + 1];
  int total;
};

template <int MROWS>
__device__ void build_plan(const int *__restrict__ cnt, int nb_per_e, Plan &p, int lane) {
  // 32 lanes load the counts in parallel, then one inclusive warp scan
  const int c = cnt[lane];
  const int nm = (c + MROWS - 1) / MROWS;
  const int it = nb_per_e * nm;
  int sp = c, si = it;
#pragma unroll
  for (int d = 1; d < EL; d <<= 1) {
    const int op = __shfl_up_sync(0xffffffffu, sp, d);
    const int oi = __shfl_up_sync(0xffffffffu, si, d);
    if (lane >= d) {
      sp += op;
      si += oi;
    }
  }
  p.off[lane] = sp - c;          // exclusive
  p.nm[lane] = nm;
  p.item_pre[lane] = si - it;    // exclusive
  if (lane == EL - 1) {
    p.item_pre[EL] = si;
    p.total = si;
  }
}

__device__ __forceinline__ void decode_item(const Plan &p, int item, int &e, int &nb, int &mb) {
  // branchless binary search for the last expert whose prefix <= item
  int lo = 0;
#pragma unroll
  for (int step = EL >> 1; step; step >>= 1) {
    const int mid = lo + step;
    lo = (mid < EL && p.item_pre[mid] <= item) ? mid : lo;
  }
  e = lo;
  const int local = item - p.item_pre[e];
  const int nm = p.nm[e];
  nb = local / nm;
  mb = local - nb * nm;
}

// =========================================================================
//  K2: GEMM1 (gate/up projection), one half of W13 per work item
// =========================================================================
// Splitting the two halves of W13 into separate work items -- instead of doing
// both in one CTA so that SwiGLU could stay warp-local -- buys three things:
//   * twice as many work items, which is what tiny seq_len needs: with only
//     2-5 experts touched there were too few CTAs to cover the 148 SMs;
//   * half the live accumulators, so the token tile can grow to 128 rows
//     without spilling, which is what stops the large-prefill shapes from
//     re-reading the weights once per token block;
//   * the intermediate's scale block is no longer tied to this kernel's row
//     block, so it can match W2's 128.
// The cost is one extra kernel and a bf16 round trip through G1.
template <int MT, int ASTG, int PFW, int MINB, int ACHV, int NW1>
__global__ __launch_bounds__(NW1 * 32, MINB) void gemm1_kernel(
    const uint8_t *__restrict__ Ad,        // [T, H]
    const float *__restrict__ As,          // [HB, T]
    const uint8_t *__restrict__ W13,       // [EL, G1O, H]
    const float *__restrict__ S13,         // [EL, 32, HB]
    const int *__restrict__ cnt, const int *__restrict__ pair_tok,
    __nv_bfloat16 *__restrict__ G1,        // [P, G1O]
    int T, int cap) {
  constexpr int NTHR1 = NW1 * 32;
  constexpr int NI1 = NW1 * WN;            // weight rows per CTA
  constexpr int MROWS = 16 * MT;
  constexpr int NBH = IIN / NI1;           // row blocks per half
  constexpr int NB = 2 * NBH;              // row blocks per expert
  constexpr int NKW = H / WIN;             // 112 k-windows
  constexpr int NGRP = NKW / PFW;
  constexpr int ASSV = ACHV + 64;          // SMEM row stride (== 64 mod 128)
  constexpr int AVECV = ACHV / 16;
  constexpr int W_PER_A = ACHV / WIN;
  constexpr int GRP_PER_A = W_PER_A / PFW;
  constexpr int NACH = H / ACHV;
  constexpr int ALD = (MROWS * AVECV) / NTHR1;
  static_assert(H % ACHV == 0 && ACHV % BLK == 0, "bad activation stage size");
  static_assert(GRP_PER_A >= 1, "prefetch group must fit in an activation stage");
  // kb is derived as kb0 + (pp >> 1), so a group must cover whole scale blocks
  static_assert(PFW % 2 == 0, "prefetch depth must be even");
  static_assert(NKW % PFW == 0, "prefetch depth must divide the k-window count");
  static_assert(ASTG >= 2 && ASTG <= NACH, "activation stages must fit the K range");
  static_assert(MROWS * AVECV % NTHR1 == 0, "activation load must divide evenly");

  extern __shared__ __align__(16) char smem[];
  uint8_t *sA = reinterpret_cast<uint8_t *>(smem);                    // [ASTG][MROWS][ASSV]
  float *s_as = reinterpret_cast<float *>(sA + ASTG * MROWS * ASSV);  // [HB][MROWS]
  float *s_s13 = s_as + MROWS * HB;                                   // [HB]
  int *s_tok = reinterpret_cast<int *>(s_s13 + HB);
  int *s_pair = s_tok + MROWS;
  Plan *s_plan = reinterpret_cast<Plan *>(s_pair + MROWS);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  if (warp == 0) build_plan<MROWS>(cnt, NB, *s_plan, lane);
  __syncthreads();
  const int total = s_plan->total;

  int prev_key = -1;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    int e, nb, mb;
    decode_item(*s_plan, item, e, nb, mb);
    const int half = nb / NBH;             // low NBH blocks are the first half
    const int i0 = (nb - half * NBH) * NI1;
    const int slot0 = mb * MROWS;
    const int ce = cnt[e];
    const int poff = s_plan->off[e];
    const int key = e * 65536 + mb;

    __syncthreads();
    if (key != prev_key) {
      for (int j = tid; j < MROWS; j += NTHR1) {
        const int slot = slot0 + j;
        const bool ok = slot < ce;
        s_tok[j] = ok ? pair_tok[e * cap + slot] : 0;
        s_pair[j] = ok ? (poff + slot) : -1;
      }
      __syncthreads();
      // k-block-major so that for a fixed k-block the 16 lanes needing
      // distinct token scales hit 16 consecutive banks
      for (int j = tid; j < MROWS * HB; j += NTHR1) {
        const int kb = j / MROWS, r = j - kb * MROWS;
        s_as[kb * MROWS + r] = As[static_cast<size_t>(kb) * T + s_tok[r]];
      }
      prev_key = key;
    }
    for (int j = tid; j < HB; j += NTHR1)
      s_s13[j] = S13[(static_cast<size_t>(e) * 32 + half * IB + i0 / BLK) * HB + j];

    const uint8_t *wp =
        W13 + (static_cast<size_t>(e) * G1O + half * IIN + i0 + warp * WN + g) * H + 16 * t;

    auto load_a = [&](int st, int k0) {
#pragma unroll
      for (int u = 0; u < ALD; ++u) {
        const int idx = tid + u * NTHR1;
        const int r = idx / AVECV, c = idx - r * AVECV;
        cp_async_ca16(sA + (static_cast<size_t>(st) * MROWS + r) * ASSV + c * 16,
                      Ad + static_cast<size_t>(s_tok[r]) * H + k0 + c * 16);
      }
      cp_async_commit();
    };

    float acc[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][j] = 0.f;

    __syncthreads();
#pragma unroll
    for (int st = 0; st < ASTG - 1; ++st) load_a(st, st * ACHV);

    int4 q[PFW];
#pragma unroll
    for (int pp = 0; pp < PFW; ++pp) q[pp] = ld_w(wp + pp * WIN);
    const uint8_t *wnext = wp + PFW * WIN;

#pragma unroll 1
    for (int grp = 0; grp < NGRP; ++grp) {
      const int st = (grp / GRP_PER_A) % ASTG;
      if (grp % GRP_PER_A == 0) {
        const int ach = grp / GRP_PER_A;
        cp_async_wait<ASTG - 2>();
        __syncthreads();
        const int nxt = ach + ASTG - 1;
        if (nxt < NACH) load_a(nxt % ASTG, nxt * ACHV);
      }
      const uint8_t *pab = sA + static_cast<size_t>(st) * MROWS * ASSV +
                           (grp % GRP_PER_A) * (PFW * WIN) + 16 * t + g * ASSV;
      const bool more = grp + 1 < NGRP;
      const int kb0 = grp * (PFW / 2);

#pragma unroll
      for (int pp = 0; pp < PFW; ++pp) {
        const int4 cw = q[pp];
        if (more) q[pp] = ld_w(wnext + pp * WIN);

        uint32_t bb[2] = {(uint32_t)cw.x, (uint32_t)cw.y};
        uint32_t bb2[2] = {(uint32_t)cw.z, (uint32_t)cw.w};
        // both 64-byte windows of a 128-element scale block share one scale, so
        // promoting per window is numerically identical and keeps only one raw
        // accumulator alive
        const int kb = kb0 + (pp >> 1);
        const float f = s_s13[kb];
        const float *asb = s_as + kb * MROWS + g;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const uint4 vlo = *reinterpret_cast<const uint4 *>(pab + (m * 16) * ASSV + pp * WIN);
          const uint4 vhi =
              *reinterpret_cast<const uint4 *>(pab + (m * 16 + 8) * ASSV + pp * WIN);
          uint32_t aa[4] = {vlo.x, vhi.x, vlo.y, vhi.y};
          uint32_t aa2[4] = {vlo.z, vhi.z, vlo.w, vhi.w};
          float raw[4] = {0.f, 0.f, 0.f, 0.f};
          mma_e4m3(raw, aa, bb);
          mma_e4m3(raw, aa2, bb2);
          const float flo = f * asb[m * 16];
          const float fhi = f * asb[m * 16 + 8];
          acc[m][0] = fmaf(raw[0], flo, acc[m][0]);
          acc[m][1] = fmaf(raw[1], flo, acc[m][1]);
          acc[m][2] = fmaf(raw[2], fhi, acc[m][2]);
          acc[m][3] = fmaf(raw[3], fhi, acc[m][3]);
        }
      }
      wnext += PFW * WIN;
    }
    cp_async_wait<0>();

    // store the raw GEMM1 output as bf16; SwiGLU and re-quantisation happen in
    // the next kernel, which sees a whole pair at once
    const int n = half * IIN + i0 + warp * WN + 2 * t;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int plo = s_pair[m * 16 + g], phi = s_pair[m * 16 + g + 8];
      if (plo >= 0)
        *reinterpret_cast<__nv_bfloat162 *>(G1 + static_cast<size_t>(plo) * G1O + n) =
            __floats2bfloat162_rn(acc[m][0], acc[m][1]);
      if (phi >= 0)
        *reinterpret_cast<__nv_bfloat162 *>(G1 + static_cast<size_t>(phi) * G1O + n) =
            __floats2bfloat162_rn(acc[m][2], acc[m][3]);
    }
  }
}

// =========================================================================
//  K2b: SwiGLU + FP8 re-quantisation of the intermediate
// =========================================================================
// One warp per (pair, 128-element block of i): reads both halves of G1, forms
// C = silu(second half) * first half, takes the block amax, writes FP8 + scale.
constexpr int SQ_WARPS = 8;
constexpr int SQ_THREADS = SQ_WARPS * 32;

__global__ __launch_bounds__(SQ_THREADS) void swiglu_q_kernel(
    const __nv_bfloat16 *__restrict__ G1,  // [P, G1O]
    const int *__restrict__ cnt, uint8_t *__restrict__ Cq, float *__restrict__ Cs) {
  __shared__ int s_total;
  if (threadIdx.x < 32) {
    const int c = cnt[threadIdx.x];
    int v = c;
#pragma unroll
    for (int d = 1; d < EL; d <<= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (threadIdx.x == 0) s_total = v;
  }
  __syncthreads();
  const int nunits = s_total * CSB;

  const int lane = threadIdx.x & 31;
  const int stride = gridDim.x * SQ_WARPS;
  for (int unit = blockIdx.x * SQ_WARPS + (threadIdx.x >> 5); unit < nunits;
       unit += stride) {
    const int pair = unit / CSB, blk = unit - pair * CSB;
    const int i0 = blk * CSCALE;

    const __nv_bfloat16 *g1 = G1 + static_cast<size_t>(pair) * G1O + i0;
    float cv[4];
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int j = lane * 2 + r * 64;
      const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162 *>(g1 + j);
      const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162 *>(g1 + IIN + j);
      const float a0 = __bfloat162float(__low2bfloat16(x1));
      const float a1 = __bfloat162float(__high2bfloat16(x1));
      const float b0 = __bfloat162float(__low2bfloat16(x2));
      const float b1 = __bfloat162float(__high2bfloat16(x2));
      cv[r * 2 + 0] = a0 * (b0 / (1.0f + __expf(-b0)));
      cv[r * 2 + 1] = a1 * (b1 / (1.0f + __expf(-b1)));
      amax = fmaxf(amax, fmaxf(fabsf(cv[r * 2 + 0]), fabsf(cv[r * 2 + 1])));
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    const float sc = amax > 0.f ? amax * (1.0f / FP8_MAX) : 1.f;
    const float inv = 1.0f / sc;

    uint8_t *cq = Cq + static_cast<size_t>(pair) * IIN + i0;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint8_t q0 = __nv_cvt_float_to_fp8(cv[r * 2 + 0] * inv, __NV_SATFINITE, __NV_E4M3);
      const uint8_t q1 = __nv_cvt_float_to_fp8(cv[r * 2 + 1] * inv, __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t *>(cq + lane * 2 + r * 64) =
          static_cast<uint16_t>(q0) | static_cast<uint16_t>(static_cast<uint16_t>(q1) << 8);
    }
    if (lane == 0) Cs[static_cast<size_t>(pair) * CSB + blk] = sc;
  }
}

// =========================================================================
//  K3: GEMM2 (down projection) + routed-weight accumulation into the output
// =========================================================================
// GEMM2. Two structural choices here, both driven by NCU:
//
//  * K is only IIN = 2048, so the whole activation (the re-quantised
//    intermediate C) fits in SMEM. Staging it would add a cp.async pipeline
//    and barriers per work item for nothing.
//
//  * Each warp owns RN2 MMA N tiles rather than one. L1TEX was the binding
//    resource at 81% of peak and the per-window request count is 1 global
//    (weights) + 2 shared (activations) however many weight rows a warp
//    covers, so more rows per warp amortises both the L1TEX requests and the
//    FP8->FP16 conversions per weight byte. Measured optimum is RN2 = 2: at 4
//    the accumulators and prefetch registers spill and it loses again. H = 7168
//    leaves plenty of row blocks per expert either way.
#ifndef MOE_RN2
#define MOE_RN2 2
#endif
constexpr int RN2 = MOE_RN2;           // MMA N tiles per warp in GEMM2
constexpr int WN2 = 8 * RN2;           // weight rows per warp
constexpr int NI2 = NWARP * WN2;       // h-rows per CTA
constexpr int NSB2 = (NI2 + BLK - 1) / BLK;   // 128-row weight-scale blocks per CTA

template <int MT, int ASTG, int PFW, int MINB>
__global__ __launch_bounds__(NTHR, MINB) void gemm2_kernel(
    const uint8_t *__restrict__ Cq,        // [P, IIN]
    const float *__restrict__ Cs,          // [P, CSB]
    const uint8_t *__restrict__ W2,        // [EL, H, IIN]
    const float *__restrict__ S2,          // [EL, HB, IB]
    const int *__restrict__ cnt, const int *__restrict__ pair_tok,
    const float *__restrict__ pair_w, __nv_bfloat16 *__restrict__ out, int T, int cap) {
  constexpr int MROWS = 16 * MT;
  constexpr int NB = H / NI2;              // 28 row blocks per expert
  constexpr int NW = IIN / WIN;            // 32 k-windows
  constexpr int NGRP = NW / PFW;
  constexpr int CSS = IIN + 64;            // SMEM row stride for the whole-K tile
  constexpr int CVEC = IIN / 16;
  constexpr int ALD = (MROWS * CVEC) / NTHR;
  static_assert(NW % PFW == 0, "prefetch depth must divide the k-window count");
  static_assert(MROWS * CVEC % NTHR == 0, "activation load must divide evenly");
  static_assert(WIN % CSCALE == 0 || CSCALE % WIN == 0,
                "a k-window must not straddle two intermediate scale blocks");

  extern __shared__ __align__(16) char smem[];
  uint8_t *sA = reinterpret_cast<uint8_t *>(smem);                // [MROWS][CSS]
  float *s_cs = reinterpret_cast<float *>(sA + MROWS * CSS);      // [CSB][MROWS]
  float *s_s2 = s_cs + MROWS * CSB;                               // [NSB2][IB]
  float *s_w = s_s2 + NSB2 * IB;                                  // [MROWS]
  int *s_tok = reinterpret_cast<int *>(s_w + MROWS);
  int *s_pair = s_tok + MROWS;
  Plan *s_plan = reinterpret_cast<Plan *>(s_pair + MROWS);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int hblk = (warp * WN2) / BLK;     // which 128-row scale block this warp is in

  if (warp == 0) build_plan<MROWS>(cnt, NB, *s_plan, lane);
  __syncthreads();
  const int total = s_plan->total;

  int prev_key = -1;
  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    int e, nb, mb;
    decode_item(*s_plan, item, e, nb, mb);
    const int h0 = nb * NI2;
    const int slot0 = mb * MROWS;
    const int ce = cnt[e];
    const int poff = s_plan->off[e];
    const int key = e * 65536 + mb;

    __syncthreads();  // every warp is done with the previous item's SMEM
    if (key != prev_key) {
      for (int j = tid; j < MROWS; j += NTHR) {
        const int slot = slot0 + j;
        const bool ok = slot < ce;
        s_tok[j] = ok ? pair_tok[e * cap + slot] : 0;
        s_pair[j] = ok ? (poff + slot) : -1;
        s_w[j] = ok ? pair_w[e * cap + slot] : 0.f;
      }
      __syncthreads();
      for (int j = tid; j < MROWS * CSB; j += NTHR) {
        const int bb = j / MROWS, r = j - bb * MROWS;
        const int pr = s_pair[r];
        s_cs[bb * MROWS + r] = pr >= 0 ? Cs[static_cast<size_t>(pr) * CSB + bb] : 0.f;
      }
#pragma unroll
      for (int u = 0; u < ALD; ++u) {
        const int idx = tid + u * NTHR;
        const int r = idx / CVEC, c = idx - r * CVEC;
        const int pr = s_pair[r] >= 0 ? s_pair[r] : 0;
        cp_async_ca16(sA + r * CSS + c * 16, Cq + static_cast<size_t>(pr) * IIN + c * 16);
      }
      cp_async_commit();
      prev_key = key;
    }
    for (int j = tid; j < NSB2 * IB; j += NTHR) {
      const int blkid = j / IB, ib = j - blkid * IB;
      s_s2[j] = S2[(static_cast<size_t>(e) * HB + h0 / BLK + blkid) * IB + ib];
    }

    const uint8_t *wp =
        W2 + (static_cast<size_t>(e) * H + h0 + warp * WN2 + g) * IIN + 16 * t;
    const uint8_t *pab = sA + 16 * t + g * CSS;
    const float *s2w = s_s2 + hblk * IB;

    float acc[MT][RN2][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int n = 0; n < RN2; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.f;

    int4 q[PFW][RN2];
#pragma unroll
    for (int pp = 0; pp < PFW; ++pp)
#pragma unroll
      for (int n = 0; n < RN2; ++n)
        q[pp][n] = ld_w(wp + static_cast<size_t>(n) * 8 * IIN + pp * WIN);
    const uint8_t *wnext = wp + PFW * WIN;

    cp_async_wait<0>();
    __syncthreads();

#pragma unroll 1
    for (int grp = 0; grp < NGRP; ++grp) {
      const int kw0 = grp * PFW;
      const bool more = grp + 1 < NGRP;
      const float *csb0 = s_cs + g;
      const uint8_t *pag = pab + kw0 * WIN;

#pragma unroll
      for (int pp = 0; pp < PFW; ++pp) {
        int4 cw[RN2];
#pragma unroll
        for (int n = 0; n < RN2; ++n) cw[n] = q[pp][n];
        if (more) {
#pragma unroll
          for (int n = 0; n < RN2; ++n)
            q[pp][n] = ld_w(wnext + static_cast<size_t>(n) * 8 * IIN + pp * WIN);
        }
        const float wq = s2w[((kw0 + pp) * WIN) / BLK];
        const float *csb = csb0 + (((kw0 + pp) * WIN) / CSCALE) * MROWS;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const uint4 vlo = *reinterpret_cast<const uint4 *>(pag + (m * 16) * CSS + pp * WIN);
          const uint4 vhi =
              *reinterpret_cast<const uint4 *>(pag + (m * 16 + 8) * CSS + pp * WIN);
          uint32_t aa[4] = {vlo.x, vhi.x, vlo.y, vhi.y};
          uint32_t aa2[4] = {vlo.z, vhi.z, vlo.w, vhi.w};
          const float flo = wq * csb[m * 16];
          const float fhi = wq * csb[m * 16 + 8];
#pragma unroll
          for (int n = 0; n < RN2; ++n) {
            float raw[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t bb[2] = {(uint32_t)cw[n].x, (uint32_t)cw[n].y};
            uint32_t bb2[2] = {(uint32_t)cw[n].z, (uint32_t)cw[n].w};
            mma_e4m3(raw, aa, bb);
            mma_e4m3(raw, aa2, bb2);
            acc[m][n][0] = fmaf(raw[0], flo, acc[m][n][0]);
            acc[m][n][1] = fmaf(raw[1], flo, acc[m][n][1]);
            acc[m][n][2] = fmaf(raw[2], fhi, acc[m][n][2]);
            acc[m][n][3] = fmaf(raw[3], fhi, acc[m][n][3]);
          }
        }
      }
      wnext += PFW * WIN;
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int rl = m * 16 + g, rh = m * 16 + g + 8;
      const bool okl = s_pair[rl] >= 0, okh = s_pair[rh] >= 0;
      const float wl = okl ? s_w[rl] : 0.f, wh = okh ? s_w[rh] : 0.f;
#pragma unroll
      for (int n = 0; n < RN2; ++n) {
        const int h = h0 + warp * WN2 + n * 8 + 2 * t;
        if (okl) {
          const __nv_bfloat162 v = __floats2bfloat162_rn(acc[m][n][0] * wl, acc[m][n][1] * wl);
          atomicAdd(
              reinterpret_cast<__nv_bfloat162 *>(out + static_cast<size_t>(s_tok[rl]) * H + h),
              v);
        }
        if (okh) {
          const __nv_bfloat162 v = __floats2bfloat162_rn(acc[m][n][2] * wh, acc[m][n][3] * wh);
          atomicAdd(
              reinterpret_cast<__nv_bfloat162 *>(out + static_cast<size_t>(s_tok[rh]) * H + h),
              v);
        }
      }
    }
  }
}

// =========================================================================
//  K3: native-FP8 tcgen05 path, used for the prefill buckets
// =========================================================================
// The mma.sync path above is capped by ptxas emulating FP8 with
// HMMA.16816 + F2FP (one conversion per 64 operand bytes per warp), which
// tops out near 0.3 PFLOP/s -- fine while weight bandwidth binds, fatal for
// the prefill shapes where arithmetic binds. tcgen05 is native FP8 and
// dev/probe/umma_rate.cu measures 4.4 PFLOP/s for back-to-back
// tcgen05.mma.cta_group::1, the full B200 dense peak.
//
// The structural cost of block scaling is that the accumulator cannot stay in
// TMEM for the whole K loop: the scale changes every 128 elements of K. So
// each 128-K block is accumulated into one of two TMEM buffers, then read out,
// scaled by as[row,kblock] * ws[nblock,kblock] and promoted into an fp32
// register accumulator, while the next block's MMA runs in the other buffer.
//
// Descriptor encodings follow CUTLASS cute::UMMA (BSD-3-Clause,
// cute/arch/mma_sm100_desc.hpp, cute/atom/mma_traits_sm100.hpp) and are
// validated bit-exact against a CPU reference in dev/probe/umma_fp8.cu.
namespace tc {

constexpr int MTMAX = 128;            // largest supported CTA tile rows
constexpr int NT = 128;               // CTA tile cols == one weight scale block
constexpr int MMA_K = 32;             // kind::f8f6f4 is a K=32 instruction
constexpr int NMMA = BLK / MMA_K;     // 4 MMAs per 128-K scale block
constexpr int WARPS = 4;              // exactly covers the 128 TMEM lanes
constexpr int THR = WARPS * 32;
#ifndef MOE_TC_STAGES
#define MOE_TC_STAGES 3
#endif
constexpr int STG = MOE_TC_STAGES;    // >= 3: a slot may only be refilled after
                                      // the mbarrier of the MMA reading it
static_assert(STG >= 3, "a stage slot is freed one block late, so STG >= 3");
// 3 is the measured best (seq_len=901: 366 us at 3 vs 521 at 6, 550 at 7) and
// is the only value this ships with. Treat it as fixed, not as a tuning knob:
// STG=4 and STG=5 *deadlock* in dev/harness.cu while 3, 6 and 7 run, which is
// not explained by the stage accounting (each slot is provably waited before it
// is refilled, for any STG >= 3) nor by shared memory or occupancy, and was not
// diagnosed. Changing it needs that understood first.
constexpr int NKB1 = H / BLK;         // 56 k-blocks in GEMM1
constexpr int NKB2 = IIN / BLK;       // 16 k-blocks in GEMM2
constexpr int NB1 = G1O / NT;         // 32 n-tiles per expert in GEMM1
constexpr int NB2 = H / NT;           // 56 n-tiles per expert in GEMM2
// bytes landing per stage, for a tile of MTV rows
template <int MTV>
__device__ __host__ constexpr uint32_t txb() {
  return static_cast<uint32_t>((MTV + NT) * BLK);
}

// Accumulator row -> TMEM lane. For M=128 it is the identity. For M=64,
// dev/probe/umma_m64.cu recovers lane = (row >> 4) * 32 + (row & 15): each
// 32-lane warp slice carries 16 rows in its lower half and nothing in its
// upper half, so only 64 of the 128 lanes hold data. A warp therefore owns
// MTV/4 rows either way, and for M=64 half its lanes idle through the
// promotion -- which is still the right trade wherever the padded MMA work of
// an M=128 tile exceeds the weight-bandwidth floor.
template <int MTV>
__device__ __forceinline__ int rows_per_warp() {
  return MTV / 4;
}

__device__ __forceinline__ uint32_t sm32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cute::UMMA::SmemDescriptor for a K-major SWIZZLE_128B FP8 tile of
// (rows x 128 bytes). In uint128_t units cute's canonical Major-K B128 form is
// ((8,n),2):((8,SBO),LBO), which gives LBO = 1 and SBO = 64 regardless of the
// row count. version 1 = Blackwell, layout_type 2 = SWIZZLE_128B. The tile
// must be 1024-byte aligned so that base_offset can stay 0.
__device__ __forceinline__ uint64_t sdesc(uint32_t a) {
  uint64_t d = static_cast<uint64_t>((a >> 4) & 0x3FFFu);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(64) << 32;
  d |= static_cast<uint64_t>(1) << 46;
  d |= static_cast<uint64_t>(2) << 61;
  return d;
}

// cute::UMMA::InstrDescriptor: dense e4m3 x e4m3 -> f32, both operands K-major.
// c_format=F32 at [4,6), a/b_format=E4M3 at [7,10)/[10,13) are zero,
// n_dim = N>>3 at [17,23), m_dim = M>>4 at [24,29).
template <int M_, int N_>
__device__ __forceinline__ uint32_t idesc() {
  return (1u << 4) | (static_cast<uint32_t>(N_ >> 3) << 17) |
         (static_cast<uint32_t>(M_ >> 4) << 24);
}

__device__ __forceinline__ void bar_init(void *b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sm32(b)), "r"(n));
}
__device__ __forceinline__ void bar_wait(void *b, int par) {
  asm volatile("{ .reg .pred p;\n"
               "  BW%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "  @!p bra BW%=;\n}" ::"r"(sm32(b)), "r"(par));
}
__device__ __forceinline__ void expect_tx(void *b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(sm32(b)), "r"(bytes));
}
__device__ __forceinline__ void tma2(uint32_t dst, const void *map, int c0, int c1, void *b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(sm32(b))
               : "memory");
}
__device__ __forceinline__ void tma3(uint32_t dst, const void *map, int c0, int c1, int c2,
                                     void *b) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile"
               ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2),
                 "r"(sm32(b))
               : "memory");
}
__device__ __forceinline__ void mma(uint32_t acc, uint64_t da, uint64_t db, uint32_t id,
                                    int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0;\n"
               "  tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
               ::"r"(acc), "l"(da), "l"(db), "r"(id), "r"(accumulate));
}
__device__ __forceinline__ void mma_commit(void *b) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" ::"r"(sm32(b)));
}

#define TC_LD32(dst, addr)                                                                    \
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "                                      \
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                      \
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"   \
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]),        \
                 "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]), "=r"(dst[8]), "=r"(dst[9]),        \
                 "=r"(dst[10]), "=r"(dst[11]), "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]),   \
                 "=r"(dst[15]), "=r"(dst[16]), "=r"(dst[17]), "=r"(dst[18]), "=r"(dst[19]),   \
                 "=r"(dst[20]), "=r"(dst[21]), "=r"(dst[22]), "=r"(dst[23]), "=r"(dst[24]),   \
                 "=r"(dst[25]), "=r"(dst[26]), "=r"(dst[27]), "=r"(dst[28]), "=r"(dst[29]),   \
                 "=r"(dst[30]), "=r"(dst[31])                                                 \
               : "r"(addr))

// Read this warp's 32 TMEM lanes x NT columns, scale, add into acc. All the
// loads are issued before the single wait: one wait per 32 columns puts a TMEM
// round trip on the critical path four times per k-block, and the MMA-issuing
// thread runs this code too.
__device__ __forceinline__ void promote(float *acc, float sc, uint32_t base) {
  uint32_t v[NT];
#pragma unroll
  for (int c = 0; c < NT; c += 32) TC_LD32((v + c), base + static_cast<uint32_t>(c));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = fmaf(sc, __int_as_float(v[i]), acc[i]);
}

}  // namespace tc

// -------------------------------------------------------------------------
// K3a: gather the routed activations into the compacted pair space.
// TMA addresses rectangular tiles, so it cannot gather; doing the gather once
// here instead of inside each GEMM1 CTA also stops the same token row being
// re-gathered by all 32 n-tile CTAs.
__global__ __launch_bounds__(256) void tc_gather_kernel(
    const uint8_t *__restrict__ Ad,        // [T, H]
    const float *__restrict__ As,          // [HB, T]
    const int *__restrict__ cnt, const int *__restrict__ pair_tok,
    uint8_t *__restrict__ Ag,              // [P, H]
    float *__restrict__ Ags,               // [HB, P]
    int T, int cap, int Pcap) {
  __shared__ Plan pl;
  if (threadIdx.x < 32) build_plan<1>(cnt, 1, pl, threadIdx.x);
  __syncthreads();
  const int total = pl.total;
  const int tid = threadIdx.x;

  for (int p = blockIdx.x; p < total; p += gridDim.x) {
    int e, nb, slot;
    decode_item(pl, p, e, nb, slot);
    const int tok = pair_tok[e * cap + slot];
    const uint4 *src = reinterpret_cast<const uint4 *>(Ad + static_cast<size_t>(tok) * H);
    uint4 *dst = reinterpret_cast<uint4 *>(Ag + static_cast<size_t>(p) * H);
    for (int c = tid; c < H / 16; c += 256) dst[c] = src[c];
    if (tid < HB)
      Ags[static_cast<size_t>(tid) * Pcap + p] = As[static_cast<size_t>(tid) * T + tok];
  }

  // A 128-row TMA box starting at the last expert's base can overhang `total`
  // by up to 127 rows. The activation bytes there are zero-filled by TMA, but
  // the scales are read with an ordinary load, so zero them: a zero scale makes
  // those rows contribute nothing instead of propagating uninitialised values.
  for (int p = total + (int)blockIdx.x; p < min(total + tc::MTMAX, Pcap); p += gridDim.x)
    if (tid < HB) Ags[static_cast<size_t>(tid) * Pcap + p] = 0.f;
}

// -------------------------------------------------------------------------
// K3b: GEMM1. One work item is (expert, 128-row token tile, 128-column tile of
// W13); both halves of W13 are just n-tiles here, so SwiGLU stays in K2b.
template <int MTV>
__global__ __launch_bounds__(tc::THR, 1) void tc_gemm1_kernel(
    const __grid_constant__ CUtensorMap mapA,   // Ag  [P, H]
    const __grid_constant__ CUtensorMap mapW,   // W13 [EL, G1O, H]
    const float *__restrict__ Ags,              // [HB, P]
    const float *__restrict__ S13,              // [EL, 32, HB]
    const int *__restrict__ cnt,
    __nv_bfloat16 *__restrict__ G1,             // [P, G1O]
    int Pcap) {
  extern __shared__ char smem_raw[];
  // the UMMA descriptor assumes base_offset == 0, i.e. a 1024-byte aligned tile
  const uint32_t pad = ((tc::sm32(smem_raw) + 1023u) & ~1023u) - tc::sm32(smem_raw);
  uint8_t *sA = reinterpret_cast<uint8_t *>(smem_raw) + pad;
  uint8_t *sB = sA + tc::STG * MTV * BLK;

  __shared__ __align__(16) uint32_t tmem_addr;
  __shared__ __align__(8) uint64_t tbar[tc::STG];
  __shared__ __align__(8) uint64_t mbar[2];
  __shared__ Plan pl;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // a warp owns MTV/4 consecutive rows, in the low lanes of its TMEM slice
  constexpr int RPW = MTV / 4;
  const int my_row = warp * RPW + lane;
  const bool row_ok = lane < RPW;
  const int srow = row_ok ? my_row : 0;

  if (warp == 0) build_plan<MTV>(cnt, tc::NB1, pl, lane);
  if (tid < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 ::"r"(tc::sm32(&tmem_addr)), "r"(2 * tc::NT));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  __syncthreads();
  const uint32_t tmem = tmem_addr;
  const uint32_t id = tc::idesc<MTV, tc::NT>();
  const int nitems = pl.total;

  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
    int e, nb, mb;
    decode_item(pl, item, e, nb, mb);
    const int m_base = pl.off[e] + mb * MTV;
    const int rows = min(MTV, cnt[e] - mb * MTV);
    const int n0 = nb * tc::NT;

    __syncthreads();  // previous item's epilogue is done reading sA
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < tc::STG; ++s) tc::bar_init(&tbar[s], 1);
      tc::bar_init(&mbar[0], 1);
      tc::bar_init(&mbar[1], 1);
    }
    __syncthreads();

    auto load = [&](int j, int s) {
      tc::expect_tx(&tbar[s], tc::txb<MTV>());
      tc::tma2(tc::sm32(sA + s * MTV * BLK), &mapA, j * BLK, m_base, &tbar[s]);
      tc::tma3(tc::sm32(sB + s * tc::NT * BLK), &mapW, j * BLK, n0, e, &tbar[s]);
    };

    if (tid == 0)
#pragma unroll
      for (int s = 0; s < tc::STG - 1; ++s) load(s, s);

    float acc[tc::NT];
#pragma unroll
    for (int i = 0; i < tc::NT; ++i) acc[i] = 0.f;

    const float *as_col = Ags + m_base + srow;
    const float *ws_col = S13 + (static_cast<size_t>(e) * 32 + nb) * HB;
    int tpar[tc::STG];
#pragma unroll
    for (int s = 0; s < tc::STG; ++s) tpar[s] = 0;
    int mpar[2] = {0, 0};

    for (int j = 0; j < tc::NKB1; ++j) {
      const int s = j % tc::STG;
      const int buf = j & 1;
      if (tid == 0) {
        tc::bar_wait(&tbar[s], tpar[s]);
        tpar[s] ^= 1;
        const uint64_t da = tc::sdesc(tc::sm32(sA + s * MTV * BLK));
        const uint64_t db = tc::sdesc(tc::sm32(sB + s * tc::NT * BLK));
        const uint32_t ap = tmem + static_cast<uint32_t>(buf * tc::NT);
#pragma unroll
        for (int kk = 0; kk < tc::NMMA; ++kk) {
          const uint32_t stp = static_cast<uint32_t>(kk * tc::MMA_K / 16);
          tc::mma(ap, da + stp, db + stp, id, kk);
        }
        tc::mma_commit(&mbar[buf]);
      }
      if (j > 0) {
        const int pb = buf ^ 1;
        tc::bar_wait(&mbar[pb], mpar[pb]);
        mpar[pb] ^= 1;
        const float sc = as_col[static_cast<size_t>(j - 1) * Pcap] * ws_col[j - 1];
        tc::promote(acc, sc, tmem + (static_cast<uint32_t>(warp * 32) << 16) +
                                 static_cast<uint32_t>(pb * tc::NT));
      }
      // The slot that held block j-1 is provably free now (every thread ran the
      // mbarrier wait above and the MMA is its only reader). This must run on
      // every iteration including j == 0, where it loads block STG-1 into a
      // slot the priming loop never filled.
      const int nj = j + tc::STG - 1;
      if (tid == 0 && nj < tc::NKB1) load(nj, nj % tc::STG);
    }
    {
      const int pb = (tc::NKB1 - 1) & 1;
      tc::bar_wait(&mbar[pb], mpar[pb]);
      const float sc = as_col[static_cast<size_t>(tc::NKB1 - 1) * Pcap] * ws_col[tc::NKB1 - 1];
      tc::promote(acc, sc, tmem + (static_cast<uint32_t>(warp * 32) << 16) +
                               static_cast<uint32_t>(pb * tc::NT));
    }

    // Epilogue. The accumulator has TMEM lane == token row, so a warp's 32
    // threads own 32 rows 8 KB apart in G1: storing straight from registers
    // costs one 32-byte sector per 4 useful bytes. Transposing 32 columns at a
    // time through SMEM (reusing the stage buffer, which is dead by now) makes
    // each warp write 32 contiguous bf16 per row instead.
    constexpr int SS = 33;  // odd stride: bank = (row + col) % 32, conflict-free
    float *st = reinterpret_cast<float *>(sA);
#pragma unroll
    for (int c0 = 0; c0 < tc::NT; c0 += 32) {
      __syncthreads();
      if (row_ok) {
#pragma unroll
        for (int i = 0; i < 32; ++i) st[my_row * SS + i] = acc[c0 + i];
      }
      __syncthreads();
      for (int r = warp; r < rows; r += tc::WARPS)
        G1[static_cast<size_t>(m_base + r) * G1O + n0 + c0 + lane] =
            __float2bfloat16(st[r * SS + lane]);
    }
  }

  __syncthreads();
  if (tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 ::"r"(tmem), "r"(2 * tc::NT));
}

// -------------------------------------------------------------------------
// K3c: GEMM2 plus the weighted scatter-accumulate into the output.
template <int MTV>
__global__ __launch_bounds__(tc::THR, 1) void tc_gemm2_kernel(
    const __grid_constant__ CUtensorMap mapC,   // Cq [P, IIN]
    const __grid_constant__ CUtensorMap mapW,   // W2 [EL, H, IIN]
    const float *__restrict__ Cs,               // [P, CSB]
    const float *__restrict__ S2,               // [EL, HB, IB]
    const int *__restrict__ cnt, const int *__restrict__ pair_tok,
    const float *__restrict__ pair_w, __nv_bfloat16 *__restrict__ out, int cap) {
  extern __shared__ char smem_raw[];
  const uint32_t pad = ((tc::sm32(smem_raw) + 1023u) & ~1023u) - tc::sm32(smem_raw);
  uint8_t *sA = reinterpret_cast<uint8_t *>(smem_raw) + pad;
  uint8_t *sB = sA + tc::STG * MTV * BLK;

  __shared__ __align__(16) uint32_t tmem_addr;
  __shared__ __align__(8) uint64_t tbar[tc::STG];
  __shared__ __align__(8) uint64_t mbar[2];
  __shared__ Plan pl;
  __shared__ int s_tok[tc::MTMAX];
  __shared__ float s_w[tc::MTMAX];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int RPW = MTV / 4;
  const int my_row = warp * RPW + lane;
  const bool row_ok = lane < RPW;
  const int srow = row_ok ? my_row : 0;

  if (warp == 0) build_plan<MTV>(cnt, tc::NB2, pl, lane);
  if (tid < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 ::"r"(tc::sm32(&tmem_addr)), "r"(2 * tc::NT));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
  }
  __syncthreads();
  const uint32_t tmem = tmem_addr;
  const uint32_t id = tc::idesc<MTV, tc::NT>();
  const int nitems = pl.total;

  for (int item = blockIdx.x; item < nitems; item += gridDim.x) {
    int e, nb, mb;
    decode_item(pl, item, e, nb, mb);
    const int m_base = pl.off[e] + mb * MTV;
    const int rows = min(MTV, cnt[e] - mb * MTV);
    const int n0 = nb * tc::NT;

    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int s = 0; s < tc::STG; ++s) tc::bar_init(&tbar[s], 1);
      tc::bar_init(&mbar[0], 1);
      tc::bar_init(&mbar[1], 1);
    }
    if (tid < MTV) {
      const int slot = mb * MTV + tid;
      const bool ok = tid < rows;
      s_tok[tid] = ok ? pair_tok[e * cap + slot] : 0;
      s_w[tid] = ok ? pair_w[e * cap + slot] : 0.f;
    }
    __syncthreads();

    auto load = [&](int j, int s) {
      tc::expect_tx(&tbar[s], tc::txb<MTV>());
      tc::tma2(tc::sm32(sA + s * MTV * BLK), &mapC, j * BLK, m_base, &tbar[s]);
      tc::tma3(tc::sm32(sB + s * tc::NT * BLK), &mapW, j * BLK, n0, e, &tbar[s]);
    };

    if (tid == 0)
#pragma unroll
      for (int s = 0; s < tc::STG - 1; ++s) load(s, s);

    float acc[tc::NT];
#pragma unroll
    for (int i = 0; i < tc::NT; ++i) acc[i] = 0.f;

    // Cs is pair-major [P, CSB], so unlike GEMM1's k-major scales this is a
    // scattered read: the 32 threads of a warp are 64 bytes apart. Reading it
    // per k-block anyway -- holding all CSB in a register array would need a
    // runtime index, which puts the array in local memory and spills acc with
    // it. It is one scalar per thread per k-block, 16 in total.
    const float *cs_row = Cs + static_cast<size_t>(m_base + srow) * CSB;
    const float *ws_col = S2 + (static_cast<size_t>(e) * HB + nb) * IB;

    int tpar[tc::STG];
#pragma unroll
    for (int s = 0; s < tc::STG; ++s) tpar[s] = 0;
    int mpar[2] = {0, 0};

    for (int j = 0; j < tc::NKB2; ++j) {
      const int s = j % tc::STG;
      const int buf = j & 1;
      if (tid == 0) {
        tc::bar_wait(&tbar[s], tpar[s]);
        tpar[s] ^= 1;
        const uint64_t da = tc::sdesc(tc::sm32(sA + s * MTV * BLK));
        const uint64_t db = tc::sdesc(tc::sm32(sB + s * tc::NT * BLK));
        const uint32_t ap = tmem + static_cast<uint32_t>(buf * tc::NT);
#pragma unroll
        for (int kk = 0; kk < tc::NMMA; ++kk) {
          const uint32_t stp = static_cast<uint32_t>(kk * tc::MMA_K / 16);
          tc::mma(ap, da + stp, db + stp, id, kk);
        }
        tc::mma_commit(&mbar[buf]);
      }
      if (j > 0) {
        const int pb = buf ^ 1;
        tc::bar_wait(&mbar[pb], mpar[pb]);
        mpar[pb] ^= 1;
        tc::promote(acc, cs_row[j - 1] * ws_col[j - 1],
                    tmem + (static_cast<uint32_t>(warp * 32) << 16) +
                        static_cast<uint32_t>(pb * tc::NT));
      }
      const int nj = j + tc::STG - 1;
      if (tid == 0 && nj < tc::NKB2) load(nj, nj % tc::STG);
    }
    {
      const int pb = (tc::NKB2 - 1) & 1;
      tc::bar_wait(&mbar[pb], mpar[pb]);
      tc::promote(acc, cs_row[tc::NKB2 - 1] * ws_col[tc::NKB2 - 1],
                  tmem + (static_cast<uint32_t>(warp * 32) << 16) +
                      static_cast<uint32_t>(pb * tc::NT));
    }

    // Epilogue: same SMEM transpose as GEMM1, but 64 columns per pass so each
    // lane can issue one bf16x2 atomic and a warp covers 128 contiguous bytes.
    constexpr int SS = 66;  // even, so the float2 read is 8-byte aligned
    float *st = reinterpret_cast<float *>(sA);
#pragma unroll
    for (int c0 = 0; c0 < tc::NT; c0 += 64) {
      __syncthreads();
      if (row_ok) {
#pragma unroll
        for (int i = 0; i < 64; ++i) st[my_row * SS + i] = acc[c0 + i];
      }
      __syncthreads();
      for (int r = warp; r < rows; r += tc::WARPS) {
        const float2 v = *reinterpret_cast<const float2 *>(&st[r * SS + 2 * lane]);
        const float w = s_w[r];
        atomicAdd(reinterpret_cast<__nv_bfloat162 *>(out + static_cast<size_t>(s_tok[r]) * H +
                                                     n0 + c0 + 2 * lane),
                  __floats2bfloat162_rn(v.x * w, v.y * w));
      }
    }
  }

  __syncthreads();
  if (tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 ::"r"(tmem), "r"(2 * tc::NT));
}

// =========================================================================
//  Host side
// =========================================================================
static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

template <int MT, int ASTG, int ACHV>
static constexpr size_t g1_smem() {
  constexpr int MROWS = 16 * MT;
  return static_cast<size_t>(ASTG) * MROWS * (ACHV + 64) +
         sizeof(float) * (MROWS * HB + HB) + sizeof(int) * 2 * MROWS + sizeof(Plan) + 64;
}
template <int MT, int ASTG>
static constexpr size_t g2_smem() {
  constexpr int MROWS = 16 * MT;
  constexpr int CSS = IIN + 64;
  return static_cast<size_t>(MROWS) * CSS +
         sizeof(float) * (MROWS * CSB + NSB2 * IB + MROWS) + sizeof(int) * 2 * MROWS +
         sizeof(Plan) + 64;
}

struct MoeArgs {
  const float *logits;
  const __nv_bfloat16 *bias;
  const uint8_t *Ad;
  const float *As;
  const uint8_t *W13;
  const float *S13;
  const uint8_t *W2;
  const float *S2;
  __nv_bfloat16 *out;
  int T;
  int expert_off;
  float rsf;
};

#ifndef MOE_T_TC
#define MOE_T_TC 160          // seq_len above which the tcgen05 path is used
#endif
#ifndef MOE_T_TC_MT128
#define MOE_T_TC_MT128 2000   // ... and above which it uses a 128-row tile
#endif
// A token can route to at most TOPK local experts, so the number of routed
// (token, expert) pairs is bounded by T * TOPK.
static size_t moe_ws_bytes(int T, size_t *o_cnt, size_t *o_tok, size_t *o_w, size_t *o_cq,
                           size_t *o_cs, size_t *o_g1, size_t *o_ag, size_t *o_ags) {
  const int cap = T;
  const size_t P = static_cast<size_t>(T) * TOPK + 256;
  size_t o = 0;
  *o_cnt = o;  o = align_up(o + sizeof(int) * 2 * EL, 256);
  *o_tok = o;  o = align_up(o + sizeof(int) * static_cast<size_t>(EL) * cap, 256);
  *o_w = o;    o = align_up(o + sizeof(float) * static_cast<size_t>(EL) * cap, 256);
  *o_cq = o;   o = align_up(o + P * IIN, 1024);
  *o_cs = o;   o = align_up(o + sizeof(float) * P * CSB, 256);
  *o_g1 = o;   o = align_up(o + sizeof(__nv_bfloat16) * P * G1O, 1024);
  // The gathered activations only exist for the tcgen05 path, and P*H is large
  // (810 MB at seq_len=14107), so they are not reserved for the shapes that
  // never take that path. TMA needs the base 1024-byte aligned.
  *o_ag = o;
  if (T > MOE_T_TC) o = align_up(o + P * H, 1024);
  *o_ags = o;
  if (T > MOE_T_TC) o = align_up(o + sizeof(float) * P * HB, 256);
  return o;
}

// CTAs/SM: dev/probe/micro4.cu shows the direct fragment-read pattern needs
// ~4 CTAs/SM of 8 warps to saturate HBM.
#ifndef MOE_G1_CTAS_PER_SM
#define MOE_G1_CTAS_PER_SM 2
#endif
#ifndef MOE_G2_CTAS_PER_SM
#define MOE_G2_CTAS_PER_SM 2
#endif

// ---- tcgen05 path: TMA descriptors and dispatch -------------------------
#ifndef MOE_TC_CTAS
#define MOE_TC_CTAS 2         // CTAs per SM for the tcgen05 GEMMs
#endif

// Resolved through the runtime API so the solution needs no -lcuda link flag.
static PFN_cuTensorMapEncodeTiled_v12000 tc_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void *p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault,
                                         &q) == cudaSuccess &&
        q == cudaDriverEntryPointSuccess)
      fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// A K-major FP8 operand map: the innermost (contiguous) dimension is K, the box
// is 128 bytes of K by `box_rows` rows, and the swizzle is the SWIZZLE_128B
// that the UMMA shared-memory descriptor in namespace tc assumes.
static bool tc_map(CUtensorMap *m, void *base, int rank, const uint64_t *dim,
                   const uint64_t *str, int box_rows) {
  PFN_cuTensorMapEncodeTiled_v12000 fn = tc_encode();
  if (fn == nullptr) return false;
  cuuint64_t gd[3];
  cuuint64_t gs[2] = {0, 0};
  const cuuint32_t bd[3] = {static_cast<cuuint32_t>(BLK), static_cast<cuuint32_t>(box_rows), 1u};
  const cuuint32_t es[3] = {1u, 1u, 1u};
  for (int i = 0; i < rank; ++i) gd[i] = dim[i];
  for (int i = 0; i < rank - 1; ++i) gs[i] = str[i];
  return fn(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, static_cast<cuuint32_t>(rank), base, gd, gs, bd,
            es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

struct TcMaps {
  CUtensorMap a1, w1, c2, w2;
  const void *p_ag, *p_w13, *p_cq, *p_w2;
  size_t p_rows;
  int p_mt;
  bool ok;
};

// Rebuilt only when a pointer or the row count changes. The benchmark's window
// only spans CUPTI kernel activity, so host work before the first launch costs
// nothing measurable even on the calls where this does run.
static TcMaps *tc_maps_for(const uint8_t *Ag, const uint8_t *W13, const uint8_t *Cq,
                           const uint8_t *W2, size_t rows, int mt) {
  // one slot per tile height, since the box height is baked into the descriptor
  static TcMaps slots[2] = {};
  TcMaps &m = slots[mt == 64 ? 0 : 1];
  if (m.ok && m.p_ag == Ag && m.p_w13 == W13 && m.p_cq == Cq && m.p_w2 == W2 &&
      m.p_rows == rows && m.p_mt == mt)
    return &m;
  const uint64_t d_a1[2] = {H, rows};
  const uint64_t s_a1[1] = {H};
  const uint64_t d_w1[3] = {H, G1O, EL};
  const uint64_t s_w1[2] = {H, static_cast<uint64_t>(H) * G1O};
  const uint64_t d_c2[2] = {IIN, rows};
  const uint64_t s_c2[1] = {IIN};
  const uint64_t d_w2[3] = {IIN, H, EL};
  const uint64_t s_w2[2] = {IIN, static_cast<uint64_t>(IIN) * H};
  const bool ok =
      tc_map(&m.a1, const_cast<uint8_t *>(Ag), 2, d_a1, s_a1, mt) &&
      tc_map(&m.w1, const_cast<uint8_t *>(W13), 3, d_w1, s_w1, tc::NT) &&
      tc_map(&m.c2, const_cast<uint8_t *>(Cq), 2, d_c2, s_c2, mt) &&
      tc_map(&m.w2, const_cast<uint8_t *>(W2), 3, d_w2, s_w2, tc::NT);
  m.ok = ok;
  if (ok) {
    m.p_ag = Ag;
    m.p_w13 = W13;
    m.p_cq = Cq;
    m.p_w2 = W2;
    m.p_rows = rows;
    m.p_mt = mt;
  }
  return &m;
}

template <int MTV>
constexpr size_t tc_smem() {
  return static_cast<size_t>(tc::STG) * (MTV + tc::NT) * BLK + 1024;
}

// Returns false if the tcgen05 path is unavailable, so the caller falls back to
// the mma.sync kernels rather than failing.
template <int MTV>
static bool tc_launch(const MoeArgs &a, int cap, int phase, cudaStream_t stream, int *d_cnt,
                      int *d_tok, float *d_w, uint8_t *d_cq, float *d_cs, __nv_bfloat16 *d_g1,
                      uint8_t *d_ag, float *d_ags, int sq_grid) {
  const int T = a.T;
  const size_t Prow = static_cast<size_t>(T) * TOPK + 256;
  TcMaps *mp = tc_maps_for(d_ag, a.W13, d_cq, a.W2, Prow, MTV);
  if (!mp->ok) return false;

  constexpr size_t s = tc_smem<MTV>();
  static_assert(s <= 227 * 1024, "tcgen05 shared memory over budget");
  static bool init = false;
  if (!init) {
    if (cudaFuncSetAttribute(tc_gemm1_kernel<MTV>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(s)) != cudaSuccess ||
        cudaFuncSetAttribute(tc_gemm2_kernel<MTV>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(s)) != cudaSuccess)
      return false;
    init = true;
  }

  if (phase != 3) {
    tc_gather_kernel<<<1184, 256, 0, stream>>>(a.Ad, a.As, d_cnt, d_tok, d_ag, d_ags, T, cap,
                                               static_cast<int>(Prow));
    tc_gemm1_kernel<MTV><<<148 * MOE_TC_CTAS, tc::THR, s, stream>>>(
        mp->a1, mp->w1, d_ags, a.S13, d_cnt, d_g1, static_cast<int>(Prow));
    swiglu_q_kernel<<<sq_grid, SQ_THREADS, 0, stream>>>(d_g1, d_cnt, d_cq, d_cs);
  }
  if (phase == 0 || phase == 3)
    tc_gemm2_kernel<MTV><<<148 * MOE_TC_CTAS, tc::THR, s, stream>>>(
        mp->c2, mp->w2, d_cs, a.S2, d_cnt, d_tok, d_w, a.out, cap);
  return true;
}

// phase: 0 = all, 1 = route only, 2 = route+gemm1+swiglu, 3 = gemm2 only
// (needs a prior full run to have populated the workspace), 4 = nothing, which
// lets the harness measure and subtract its own timing overhead.
// Used by dev/harness.cu.
void moe_launch(const MoeArgs &a, char *ws, int parity, cudaStream_t stream, int phase = 0) {
  const int T = a.T;
  const int cap = T;
  size_t o_cnt, o_tok, o_w, o_cq, o_cs, o_g1, o_ag, o_ags;
  moe_ws_bytes(T, &o_cnt, &o_tok, &o_w, &o_cq, &o_cs, &o_g1, &o_ag, &o_ags);

  int *cnt_bank = reinterpret_cast<int *>(ws + o_cnt);
  int *d_cnt = cnt_bank + parity * EL;
  int *d_cnt_next = cnt_bank + (1 - parity) * EL;
  int *d_tok = reinterpret_cast<int *>(ws + o_tok);
  float *d_w = reinterpret_cast<float *>(ws + o_w);
  uint8_t *d_cq = reinterpret_cast<uint8_t *>(ws + o_cq);
  float *d_cs = reinterpret_cast<float *>(ws + o_cs);
  __nv_bfloat16 *d_g1 = reinterpret_cast<__nv_bfloat16 *>(ws + o_g1);

  const int route_grid = (T + ROUTE_WARPS - 1) / ROUTE_WARPS;
  // swiglu_q strides over units, so an estimate is enough: 2*T pairs covers
  // every observed routing, and the stride loop stays correct beyond it.
  const long long sq_units = 2LL * T * CSB;
  const int sq_grid =
      static_cast<int>(sq_units / SQ_WARPS + 1 < 1184 ? sq_units / SQ_WARPS + 1 : 1184);
  if (phase == 0 || phase == 1 || phase == 2)
    route_kernel<<<route_grid, ROUTE_THREADS, 0, stream>>>(
        a.logits, a.bias, a.out, d_cnt, d_cnt_next, d_tok, d_w, T, cap, a.expert_off, a.rsf);
  if (phase == 1) return;

  // The prefill shapes are arithmetic-bound, where the emulated FP8 mma.sync
  // path caps out near 0.3 PFLOP/s; hand them to the native-FP8 tcgen05
  // kernels. Falls through to mma.sync if the TMA descriptors cannot be built.
  // Tile height is a per-bucket choice. A 128-row tile only pays off when the
  // experts are full enough to use it: at seq_len=14107 the mean is 504 tokens
  // per expert, but at 901 it is 41.6 and in the mid bucket 2-4, where an
  // M=128 tile spends most of the tensor core on padding. A 64-row tile halves
  // that padding; it also halves the useful TMEM lanes, so it is not free, and
  // for the huge bucket it would double both the weight re-reads and the
  // promotion work. See docs/draft.md section 9.3.
  if (T > MOE_T_TC) {
    uint8_t *d_ag = reinterpret_cast<uint8_t *>(ws + o_ag);
    float *d_ags = reinterpret_cast<float *>(ws + o_ags);
    const bool big_tile = T > MOE_T_TC_MT128;
    if (big_tile) {
      if (tc_launch<128>(a, cap, phase, stream, d_cnt, d_tok, d_w, d_cq, d_cs, d_g1, d_ag,
                         d_ags, sq_grid))
        return;
    } else {
      if (tc_launch<64>(a, cap, phase, stream, d_cnt, d_tok, d_w, d_cq, d_cs, d_g1, d_ag,
                        d_ags, sq_grid))
        return;
    }
  }

#define MOE_LAUNCH(MT1, MT2, ST1, ST2, PF1, PF2, MB1, MB2, AC1, NW1)                           \
  do {                                                                                         \
    constexpr size_t s1 = g1_smem<MT1, ST1, AC1>();                                            \
    constexpr size_t s2 = g2_smem<MT2, ST2>();                                                 \
    static_assert(s1 <= 227 * 1024 && s2 <= 227 * 1024, "shared memory over budget");          \
    static bool init = false;                                                                  \
    if (!init) {                                                                               \
      cudaFuncSetAttribute(gemm1_kernel<MT1, ST1, PF1, MB1, AC1, NW1>,                         \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s1);              \
      cudaFuncSetAttribute(gemm2_kernel<MT2, ST2, PF2, MB2>,                                   \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s2);              \
      init = true;                                                                             \
    }                                                                                          \
    if (phase != 3) {                                                                          \
      gemm1_kernel<MT1, ST1, PF1, MB1, AC1, NW1><<<148 * (MB1), (NW1) * 32, s1, stream>>>(     \
          a.Ad, a.As, a.W13, a.S13, d_cnt, d_tok, d_g1, T, cap);                                \
      swiglu_q_kernel<<<sq_grid, SQ_THREADS, 0, stream>>>(d_g1, d_cnt, d_cq, d_cs);             \
    }                                                                                          \
    if (phase == 0 || phase == 3)                                                              \
      gemm2_kernel<MT2, ST2, PF2, MB2><<<148 * (MB2), NTHR, s2, stream>>>(                     \
          d_cq, d_cs, a.W2, a.S2, d_cnt, d_tok, d_w, a.out, T, cap);                            \
  } while (0)
// one extra expansion so a MOE_CFG_* comma list counts as seven arguments
#define MOE_LAUNCH_CFG(...) MOE_LAUNCH(__VA_ARGS__)

  // Shape-specialised dispatch. The buckets and their knobs come from the
  // sweeps in dev/sweep_buckets.sh: at tiny seq_len there are few work items,
  // so what matters is bytes in flight *per CTA* (deep register prefetch, one
  // big CTA per SM); at larger seq_len there are plenty of items and raw
  // occupancy wins instead.
// Per-bucket knobs. Arguments are
//   (gemm1 token tiles, gemm2 token tiles, gemm1 activation stages,
//    gemm2 unused, gemm1 prefetch windows, gemm2 prefetch windows,
//    gemm1 CTAs/SM, gemm2 CTAs/SM, gemm1 activation stage bytes).
// The values come from dev/sweep_buckets.sh against the representative shape
// of each bucket; see docs/draft.md section 7 for the tables. The pattern is
// that tiny seq_len wants deep prefetch and one big CTA per SM (few work items
// ⇒ bytes in flight per CTA is what matters), while larger seq_len wants
// shallower prefetch and more CTAs (plenty of items ⇒ occupancy wins).
#ifndef MOE_T_MT1
#define MOE_T_MT1 1
#endif
#ifndef MOE_T_PF1
#define MOE_T_PF1 16
#endif
#ifndef MOE_T_PF2
#define MOE_T_PF2 8
#endif
#ifndef MOE_T_MB1
#define MOE_T_MB1 1
#endif
#ifndef MOE_T_MB2
#define MOE_T_MB2 1
#endif
#ifndef MOE_T_ST1
#define MOE_T_ST1 3
#endif
#ifndef MOE_T_AC1
#define MOE_T_AC1 1024
#endif
// Narrower CTAs would double the work-item count, which tiny seq_len wants
// (3 experts give only 192 items for 148 SMs), but measured worse everywhere:
// seq_len=1 55.5 -> 65.6 us, seq_len=16 143.7 -> 174.1 us. Halving the rows per
// CTA halves the weight bytes per item while the activation staging per item is
// unchanged, so the staging cost per weight byte doubles.
#ifndef MOE_T_NW1
#define MOE_T_NW1 8
#endif
#define MOE_CFG_TINY MOE_T_MT1, 1, MOE_T_ST1, 2, MOE_T_PF1, MOE_T_PF2, MOE_T_MB1, \
                     MOE_T_MB2, MOE_T_AC1, \
                     MOE_T_NW1

#ifndef MOE_S_PF1
#define MOE_S_PF1 16
#endif
#ifndef MOE_S_PF2
#define MOE_S_PF2 8
#endif
#ifndef MOE_S_MB1
#define MOE_S_MB1 1
#endif
#ifndef MOE_S_MB2
#define MOE_S_MB2 2
#endif
#ifndef MOE_S_ST1
#define MOE_S_ST1 3
#endif
#ifndef MOE_S_AC1
#define MOE_S_AC1 1024
#endif
#ifndef MOE_S_NW1
#define MOE_S_NW1 8
#endif
#define MOE_CFG_SMALL 1, 1, MOE_S_ST1, 2, MOE_S_PF1, MOE_S_PF2, MOE_S_MB1, MOE_S_MB2, \
                      MOE_S_AC1, \
                     MOE_S_NW1

#ifndef MOE_M_PF1
#define MOE_M_PF1 8
#endif
#ifndef MOE_M_PF2
#define MOE_M_PF2 4
#endif
#ifndef MOE_M_MB1
#define MOE_M_MB1 2
#endif
#ifndef MOE_M_MB2
#define MOE_M_MB2 2
#endif
#ifndef MOE_M_ST1
#define MOE_M_ST1 2
#endif
#ifndef MOE_M_AC1
#define MOE_M_AC1 512
#endif
#ifndef MOE_M_NW1
#define MOE_M_NW1 8
#endif
#define MOE_CFG_MID 1, 1, MOE_M_ST1, 2, MOE_M_PF1, MOE_M_PF2, MOE_M_MB1, MOE_M_MB2, \
                    MOE_M_AC1, \
                     MOE_M_NW1

#ifndef MOE_B_MT1
#define MOE_B_MT1 2
#endif
#ifndef MOE_B_MT2
#define MOE_B_MT2 2
#endif
#ifndef MOE_B_ST1
#define MOE_B_ST1 2
#endif
#ifndef MOE_B_PF1
#define MOE_B_PF1 4
#endif
#ifndef MOE_B_PF2
#define MOE_B_PF2 4
#endif
#ifndef MOE_B_MB1
#define MOE_B_MB1 2
#endif
#ifndef MOE_B_MB2
#define MOE_B_MB2 2
#endif
#ifndef MOE_B_AC1
#define MOE_B_AC1 512
#endif
#ifndef MOE_B_NW1
#define MOE_B_NW1 8
#endif
#define MOE_CFG_BIG MOE_B_MT1, MOE_B_MT2, MOE_B_ST1, 2, MOE_B_PF1, MOE_B_PF2, MOE_B_MB1, \
                    MOE_B_MB2, MOE_B_AC1, \
                     MOE_B_NW1

// The two large-prefill shapes have max M in the hundreds, so they want a
// bigger token tile than seq_len=901 does (which has max M = 77): 6296 vs
// 6560 us at seq_len=14107, but 836 vs 687 us at seq_len=901.
#ifndef MOE_H_MT1
#define MOE_H_MT1 4
#endif
#ifndef MOE_H_MT2
#define MOE_H_MT2 2
#endif
#ifndef MOE_H_ST1
#define MOE_H_ST1 2
#endif
#ifndef MOE_H_PF1
#define MOE_H_PF1 4
#endif
#ifndef MOE_H_PF2
#define MOE_H_PF2 4
#endif
#ifndef MOE_H_MB1
#define MOE_H_MB1 2
#endif
#ifndef MOE_H_MB2
#define MOE_H_MB2 2
#endif
#ifndef MOE_H_AC1
#define MOE_H_AC1 512
#endif
#ifndef MOE_H_NW1
#define MOE_H_NW1 8
#endif
#define MOE_CFG_HUGE MOE_H_MT1, MOE_H_MT2, MOE_H_ST1, 2, MOE_H_PF1, MOE_H_PF2, MOE_H_MB1, \
                     MOE_H_MB2, MOE_H_AC1, \
                     MOE_H_NW1

#ifndef MOE_T_TINY
#define MOE_T_TINY 10
#endif
#ifndef MOE_T_SMALL
#define MOE_T_SMALL 40
#endif
#ifndef MOE_T_MID
#define MOE_T_MID 160
#endif
#ifndef MOE_T_BIG
#define MOE_T_BIG 2000
#endif
  if (T <= MOE_T_TINY) {
    MOE_LAUNCH_CFG(MOE_CFG_TINY);
  } else if (T <= MOE_T_SMALL) {
    MOE_LAUNCH_CFG(MOE_CFG_SMALL);
  } else if (T <= MOE_T_MID) {
    MOE_LAUNCH_CFG(MOE_CFG_MID);
  } else if (T <= MOE_T_BIG) {
    MOE_LAUNCH_CFG(MOE_CFG_BIG);
  } else {
    MOE_LAUNCH_CFG(MOE_CFG_HUGE);
  }
#undef MOE_LAUNCH
#undef MOE_LAUNCH_CFG
}

#ifndef MOE_STANDALONE
static void *g_ws = nullptr;
static size_t g_ws_bytes = 0;
static int g_parity = 0;

static void *ws_get(size_t need) {
  if (need > g_ws_bytes) {
    if (g_ws) cudaFree(g_ws);
    const size_t want = need + (need >> 2) + (1 << 20);
    const cudaError_t err = cudaMalloc(&g_ws, want);
    if (err != cudaSuccess) {
      g_ws = nullptr;
      g_ws_bytes = 0;
      TVM_FFI_THROW(RuntimeError) << "moe workspace alloc failed: " << cudaGetErrorString(err);
    }
    g_ws_bytes = want;
    cudaMemset(g_ws, 0, sizeof(int) * 2 * EL);  // once, outside the hot path
    g_parity = 0;
  }
  return g_ws;
}

void moe_run(tvm::ffi::TensorView routing_logits, tvm::ffi::TensorView routing_bias,
             tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView hidden_states_scale,
             tvm::ffi::TensorView gemm1_weights, tvm::ffi::TensorView gemm1_weights_scale,
             tvm::ffi::TensorView gemm2_weights, tvm::ffi::TensorView gemm2_weights_scale,
             int64_t local_expert_offset, double routed_scaling_factor,
             tvm::ffi::TensorView output) {
  const int T = static_cast<int>(routing_logits.size(0));
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), EG);
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), H);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), HB);
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), T);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(0), EL);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), G1O);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(1), H);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(2), IIN);
  TVM_FFI_ICHECK_EQ(output.size(0), T);
  TVM_FFI_ICHECK_EQ(output.size(1), H);

  const DLDevice dev = hidden_states.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));

  size_t q1, q2, q3, q4, q5, q6, q7, q8;
  char *ws =
      static_cast<char *>(ws_get(moe_ws_bytes(T, &q1, &q2, &q3, &q4, &q5, &q6, &q7, &q8)));

  MoeArgs ar;
  ar.logits = static_cast<const float *>(routing_logits.data_ptr());
  ar.bias = static_cast<const __nv_bfloat16 *>(routing_bias.data_ptr());
  ar.Ad = static_cast<const uint8_t *>(hidden_states.data_ptr());
  ar.As = static_cast<const float *>(hidden_states_scale.data_ptr());
  ar.W13 = static_cast<const uint8_t *>(gemm1_weights.data_ptr());
  ar.S13 = static_cast<const float *>(gemm1_weights_scale.data_ptr());
  ar.W2 = static_cast<const uint8_t *>(gemm2_weights.data_ptr());
  ar.S2 = static_cast<const float *>(gemm2_weights_scale.data_ptr());
  ar.out = static_cast<__nv_bfloat16 *>(output.data_ptr());
  ar.T = T;
  ar.expert_off = static_cast<int>(local_expert_offset);
  ar.rsf = static_cast<float>(routed_scaling_factor);

  const int parity = g_parity;
  g_parity ^= 1;
  moe_launch(ar, ws, parity, stream);
}
#endif  // MOE_STANDALONE

}  // namespace moe

#ifndef MOE_STANDALONE
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_run, moe::moe_run);
#endif
