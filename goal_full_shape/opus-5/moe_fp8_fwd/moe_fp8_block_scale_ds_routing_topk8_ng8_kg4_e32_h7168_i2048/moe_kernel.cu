// FP8 block-scale MoE (DeepSeek-V3 no-aux routing) for NVIDIA Blackwell.
//
//   routing  -> per-local-expert token buckets (+ gathered activation scales)
//   gemm1    -> A[tok] @ W13[e]^T, SwiGLU, fp8 block quantise  (+ zero output)
//   gemm2    -> C @ W2[e]^T, weighted accumulate into the bf16 output
//
// Both GEMMs are persistent kernels.  A CTA owns a strided slice of the work
// list and runs ONE cp.async pipeline across all of its work items: every
// per-item input (token ids, activation block scales, weight block scales) is
// fetched by cp.async inside a stage group, so crossing from one work item to
// the next costs no extra barrier and never drains the memory pipeline.
//
// Tiles are consumed by register-accumulator mma.sync m16n8k32 e4m3
// instructions fed through ldmatrix.  Warp layout: 8 warps, warp w owns mma
// n-tile w (8 columns) of the 64-column output block and walks every 16-row
// m-step, so no warp idles and B fragments are reused across m-steps.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define HID 7168
#define INTE 2048
#define EGLOB 256
#define QBLK 128
#define NKBLK 56
#define NIBLK 16

#ifdef MOE_MINBLK
#define MOE_LB __launch_bounds__(256, MOE_MINBLK)
#else
#define MOE_LB __launch_bounds__(256)
#endif

namespace {

__device__ __forceinline__ uint32_t sm32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------- routing --
__global__ void routing_kernel(const float *__restrict__ logits,
                               const __nv_bfloat16 *__restrict__ bias,
                               const float *__restrict__ Ascale,
                               int *__restrict__ cnt, int *__restrict__ tokids,
                               float *__restrict__ tokw, float *__restrict__ ascg,
                               int T, int TP, int local_off, float scale_f, int EL) {
  const int t = blockIdx.x;
  const int e = threadIdx.x;
  const int lane = e & 31;
  const int w = e >> 5;

  __shared__ float sgs[8];
  __shared__ unsigned long long red[8];
  __shared__ float ssum;
  __shared__ int npair;
  __shared__ int plist[16];

  float lg = logits[(size_t)t * EGLOB + e];
  float s = 1.f / (1.f + __expf(-lg));
  float swb = s + __bfloat162float(bias[e]);

  float v1 = swb, v2 = -INFINITY;
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    float o1 = __shfl_xor_sync(0xffffffffu, v1, off);
    float o2 = __shfl_xor_sync(0xffffffffu, v2, off);
    float hi = fmaxf(v1, o1);
    float lo = fminf(v1, o1);
    v1 = hi;
    v2 = fmaxf(lo, fmaxf(v2, o2));
  }
  if (lane == 0) sgs[w] = v1 + v2;
  if (e == 0) npair = 0;
  __syncthreads();

  float g[8];
#pragma unroll
  for (int i = 0; i < 8; i++) g[i] = sgs[i];
  int keep = 0;
#pragma unroll
  for (int it = 0; it < 4; it++) {
    int best = 0;
    float bv = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; i++)
      if (!((keep >> i) & 1) && g[i] > bv) { bv = g[i]; best = i; }
    keep |= 1 << best;
  }

  float sc = ((keep >> w) & 1) ? swb : -INFINITY;
  uint32_t fb = __float_as_uint(sc);
  uint32_t key = (fb & 0x80000000u) ? ~fb : (fb | 0x80000000u);
  unsigned long long mypk =
      ((unsigned long long)key << 32) | (unsigned long long)(EGLOB - 1 - e);

  int selected = 0;
#pragma unroll
  for (int it = 0; it < 8; it++) {
    unsigned long long m = mypk;
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      unsigned long long o = __shfl_xor_sync(0xffffffffu, m, off);
      m = o > m ? o : m;
    }
    if (lane == 0) red[w] = m;
    __syncthreads();
    unsigned long long gm = red[0];
#pragma unroll
    for (int i = 1; i < 8; i++) gm = red[i] > gm ? red[i] : gm;
    int we = EGLOB - 1 - (int)(gm & 0xffffffffu);
    if (e == we) { selected = 1; mypk = 0ull; }
    __syncthreads();
  }

  if (e == 0) ssum = 0.f;
  __syncthreads();
  if (selected) atomicAdd(&ssum, s);
  __syncthreads();

  int le = e - local_off;
  if (selected && le >= 0 && le < EL) {
    float wn = s / (ssum + 1e-20f) * scale_f;
    int pos = atomicAdd(&cnt[le], 1);
    tokids[(size_t)le * TP + pos] = t;
    tokw[(size_t)le * TP + pos] = wn;
    int k = atomicAdd(&npair, 1);
    if (k < 8) {
      plist[2 * k] = le;
      plist[2 * k + 1] = pos;
    }
  }
  __syncthreads();

  // Gather this token's 56 activation block scales next to each local pair, so
  // a gemm1 tile reads its scales as one contiguous 16B-aligned run.
  const int np = min(npair, 8);
  for (int i = e; i < np * NKBLK; i += EGLOB) {
    int j = i / NKBLK, kb = i - j * NKBLK;
    ascg[((size_t)plist[2 * j] * TP + plist[2 * j + 1]) * NKBLK + kb] =
        Ascale[(size_t)kb * T + t];
  }
}


// Warp-per-token routing: no block barriers at all, so the latency for short
// sequences is a few hundred cycles instead of a few microseconds.
template <int BT>
__global__ __launch_bounds__(32 * BT) void routing_warp_kernel(
    const float *__restrict__ logits, const __nv_bfloat16 *__restrict__ bias,
    const float *__restrict__ Ascale, int *__restrict__ cnt,
    int *__restrict__ tokids, float *__restrict__ tokw, float *__restrict__ ascg,
    int T, int TP, int local_off, float scale_f, int EL, int want_ascg) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * BT + warp;
  if (t >= T) return;

  float sg[8], v[8];
#pragma unroll
  for (int g = 0; g < 8; g++) {
    const int e = g * 32 + lane;
    const float lg = logits[(size_t)t * EGLOB + e];
    sg[g] = 1.f / (1.f + __expf(-lg));
    v[g] = sg[g] + __bfloat162float(bias[e]);
  }

  float hi[8], lo[8];
#pragma unroll
  for (int g = 0; g < 8; g++) { hi[g] = v[g]; lo[g] = -INFINITY; }
#pragma unroll
  for (int off = 16; off; off >>= 1) {
#pragma unroll
    for (int g = 0; g < 8; g++) {
      const float oh = __shfl_xor_sync(0xffffffffu, hi[g], off);
      const float ol = __shfl_xor_sync(0xffffffffu, lo[g], off);
      const float a = fmaxf(hi[g], oh);
      const float b = fminf(hi[g], oh);
      hi[g] = a;
      lo[g] = fmaxf(b, fmaxf(lo[g], ol));
    }
  }
  int keep = 0;
  {
    float gs[8];
#pragma unroll
    for (int g = 0; g < 8; g++) gs[g] = hi[g] + lo[g];
#pragma unroll
    for (int it = 0; it < 4; it++) {
      int best = 0;
      float bv = -INFINITY;
#pragma unroll
      for (int g = 0; g < 8; g++)
        if (!((keep >> g) & 1) && gs[g] > bv) { bv = gs[g]; best = g; }
      keep |= 1 << best;
    }
  }
#pragma unroll
  for (int g = 0; g < 8; g++)
    if (!((keep >> g) & 1)) v[g] = -INFINITY;

  int selmask = 0;
  float wsum = 0.f;
#pragma unroll
  for (int it = 0; it < 8; it++) {
    unsigned long long best = 0ull;
#pragma unroll
    for (int g = 0; g < 8; g++) {
      const uint32_t fb = __float_as_uint(v[g]);
      const uint32_t key = (fb & 0x80000000u) ? ~fb : (fb | 0x80000000u);
      const unsigned long long pk =
          ((unsigned long long)key << 32) |
          (unsigned long long)(EGLOB - 1 - (g * 32 + lane));
      best = pk > best ? pk : best;
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
      best = o > best ? o : best;
    }
    const int we = EGLOB - 1 - (int)(best & 0xffffffffu);
    if (lane == (we & 31)) {
      const int g = we >> 5;
      selmask |= 1 << g;
      wsum += sg[g];
      v[g] = -INFINITY;
    }
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
  const float inv = scale_f / (wsum + 1e-20f);

#pragma unroll
  for (int g = 0; g < 8; g++) {
    if ((selmask >> g) & 1) {
      const int le = g * 32 + lane - local_off;
      if (le >= 0 && le < EL) {
        const int pos = atomicAdd(&cnt[le], 1);
        tokids[(size_t)le * TP + pos] = t;
        tokw[(size_t)le * TP + pos] = sg[g] * inv;
        if (want_ascg) {
          float *dst = ascg + ((size_t)le * TP + pos) * NKBLK;
          for (int kb = 0; kb < NKBLK; kb++) dst[kb] = Ascale[(size_t)kb * T + t];
        }
      }
    }
  }
}

// Same routing, but the two warp-wide selections run on the hardware warp
// reduction (redux.sync) instead of a butterfly of shuffles: one instruction
// instead of five dependent shuffle rounds.  Ordering is identical - a total
// order on (value, expert index), ties going to the lower expert index.
__device__ __forceinline__ uint32_t ford(float x) {
  const uint32_t fb = __float_as_uint(x);
  return (fb & 0x80000000u) ? ~fb : (fb | 0x80000000u);
}
__device__ __forceinline__ float fund(uint32_t k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

template <int BT>
__global__ __launch_bounds__(32 * BT) void routing_fast_kernel(
    const float *__restrict__ logits, const __nv_bfloat16 *__restrict__ bias,
    int *__restrict__ cnt, int *__restrict__ tokids, float *__restrict__ tokw,
    int *__restrict__ nloc, int *__restrict__ tslot, int T, int TP,
    int local_off, float scale_f, int EL) {
  const unsigned FULL = 0xffffffffu;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * BT + warp;
  if (t >= T) return;

  float sg[8];
  uint32_t key[8];
#pragma unroll
  for (int g = 0; g < 8; g++) {
    const int e = g * 32 + lane;
    const float lg = logits[(size_t)t * EGLOB + e];
    sg[g] = 1.f / (1.f + __expf(-lg));
    key[g] = ford(sg[g] + __bfloat162float(bias[e]));
  }

  // per group: sum of its top two biased scores
  float gs[8];
#pragma unroll
  for (int g = 0; g < 8; g++) {
    const uint32_t k1 = __reduce_max_sync(FULL, key[g]);
    const int L = __ffs(__ballot_sync(FULL, key[g] == k1)) - 1;
    const uint32_t k2 = __reduce_max_sync(FULL, lane == L ? 0u : key[g]);
    gs[g] = fund(k1) + fund(k2);
  }

  int keep = 0;
#pragma unroll
  for (int it = 0; it < 4; it++) {
    int best = 0;
    float bv = -INFINITY;
#pragma unroll
    for (int g = 0; g < 8; g++)
      if (!((keep >> g) & 1) && gs[g] > bv) { bv = gs[g]; best = g; }
    keep |= 1 << best;
  }

  // keys of the surviving candidates; 0 means "not a candidate" (a finite or
  // -inf score never maps to 0 under ford())
  uint32_t lk[8];
#pragma unroll
  for (int g = 0; g < 8; g++) lk[g] = ((keep >> g) & 1) ? key[g] : 0u;

  int selmask = 0;
  float wsum = 0.f;
#pragma unroll
  for (int it = 0; it < 8; it++) {
    uint32_t mk = 0u;
#pragma unroll
    for (int g = 0; g < 8; g++) mk = lk[g] > mk ? lk[g] : mk;
    const uint32_t k = __reduce_max_sync(FULL, mk);
    uint32_t myidx = 0xffffffffu;
#pragma unroll
    for (int g = 7; g >= 0; g--)
      if (lk[g] == k) myidx = (uint32_t)(g * 32 + lane);
    const uint32_t gi = __reduce_min_sync(FULL, myidx);
    if ((gi & 31u) == (uint32_t)lane) {
      const int g = (int)(gi >> 5);
      selmask |= 1 << g;
      wsum += sg[g];
      lk[g] = 0u;
    }
  }

#pragma unroll
  for (int off = 16; off; off >>= 1) wsum += __shfl_xor_sync(FULL, wsum, off);
  const float inv = scale_f / (wsum + 1e-20f);

  // inverse map for the optional gather-reduce output path: for this token,
  // tslot[le] is 1 + the position it was appended at in local expert le's
  // bucket, or 0 if that expert was not selected
  if (tslot) {
    if (lane < EL) tslot[(size_t)t * EL + lane] = 0;
    __syncwarp();
  }
  int nl = 0;
#pragma unroll
  for (int g = 0; g < 8; g++) {
    if ((selmask >> g) & 1) {
      const int le = g * 32 + lane - local_off;
      if (le >= 0 && le < EL) {
        nl++;
        const int pos = atomicAdd(&cnt[le], 1);
        tokids[(size_t)le * TP + pos] = t;
        tokw[(size_t)le * TP + pos] = sg[g] * inv;
        if (tslot) tslot[(size_t)t * EL + le] = pos + 1;
      }
    }
  }
  // how many of this token's experts live on this rank: lets gemm1 skip
  // zeroing rows that gemm2 will overwrite, and lets gemm2 store instead of
  // accumulate when it is the only contribution to the row
  if (nloc) {
#pragma unroll
    for (int off = 16; off; off >>= 1) nl += __shfl_xor_sync(FULL, nl, off);
    if (lane == 0) nloc[t] = nl;
  }
}

// --------------------------------------------------------------- helpers ---
template <int RB>
__device__ __forceinline__ int swz(int r, int k) {
  return r * RB + (((k >> 4) ^ (r & 7)) << 4) + (k & 15);
}
__device__ __forceinline__ void cpa16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void mma_f8(float (&d)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void waitg(int rem) {
  switch (rem) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::); break;
    case 7: asm volatile("cp.async.wait_group 7;\n" ::); break;
    case 8: asm volatile("cp.async.wait_group 8;\n" ::); break;
    case 9: asm volatile("cp.async.wait_group 9;\n" ::); break;
    case 10: asm volatile("cp.async.wait_group 10;\n" ::); break;
    case 11: asm volatile("cp.async.wait_group 11;\n" ::); break;
    default: asm volatile("cp.async.wait_group 12;\n" ::); break;
  }
}

#include "moe_ffma.cuh"

struct Dec {
  int e, ml, nt, rows;
};

__device__ __forceinline__ Dec decomp(const int *mt, const int *pre, const int *cntv,
                                      int EL, int wi, int NT, int BM) {
  int q = wi / NT;
  int e = 0;
  for (int i = 0; i < EL; i++)
    if (mt[i] > 0 && pre[i] <= q) e = i;
  int mte = mt[e];
  int loc = wi - pre[e] * NT;
  Dec d;
  d.e = e;
  d.nt = loc / mte;
  d.ml = loc - d.nt * mte;
  d.rows = min(BM, cntv[e] - d.ml * BM);
  return d;
}

// ----------------------------------------------------------------- gemm1 ---
template <int BM, int RB, int STG>
__global__ MOE_LB void gemm1_kernel(
    const __nv_fp8_e4m3 *__restrict__ A, const __nv_fp8_e4m3 *__restrict__ W13,
    const float *__restrict__ S13, __nv_fp8_e4m3 *__restrict__ Cq,
    float *__restrict__ Cs, const int *__restrict__ cnt,
    const int *__restrict__ tokids, __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ ascg, int T, int TP, long long out_numel, int EL,
    int NCTA) {
  constexpr int BNI = 64;
  constexpr int NT = INTE / BNI;
  constexpr int KPS = RB / QBLK;
  constexpr int NST = NKBLK / KPS;
  constexpr int MAXM = BM / 16;

  extern __shared__ char smem[];
  __nv_fp8_e4m3 *As = reinterpret_cast<__nv_fp8_e4m3 *>(smem);
  __nv_fp8_e4m3 *Ws = As + (size_t)STG * BM * RB;
  float *sa_s = reinterpret_cast<float *>(Ws + (size_t)STG * 2 * BNI * RB);
  float *sg_s = sa_s + (size_t)2 * NKBLK * BM;      // 2 * 2 * NKBLK
  float *amax_s = sg_s + 4 * NKBLK;                 // 8 * BM
  int *tok_s = reinterpret_cast<int *>(amax_s + 8 * BM);  // 2 * BM
  int *mt = tok_s + 2 * BM;
  int *pre = mt + 32;
  int *cntv = pre + 32;
  int *pslot = cntv + 32;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  {  // zero the output; the kernel boundary orders this before gemm2
    long long n4 = out_numel >> 3;
    for (long long i = (long long)blockIdx.x * 256 + tid; i < n4;
         i += (long long)NCTA * 256)
      reinterpret_cast<int4 *>(out)[i] = make_int4(0, 0, 0, 0);
  }

  __shared__ int s_total;
  if (tid == 0) {
    int tt = 0, as = 0;
    for (int i = 0; i < EL; i++) {
      int c = cnt[i];
      cntv[i] = c;
      mt[i] = (c + BM - 1) / BM;
      pre[i] = tt;
      pslot[i] = as;
      tt += mt[i];
      as += mt[i] * BM;
    }
    s_total = tt;
  }
  __syncthreads();
  const int nwork = s_total * NT;
  if (blockIdx.x >= (unsigned)nwork) return;
  const int nloc = (nwork - (int)blockIdx.x + NCTA - 1) / NCTA;
  const int tot = nloc * NST;

  // token ids of the first work item (everything later rides the pipeline)
  Dec d0 = decomp(mt, pre, cntv, EL, blockIdx.x, NT, BM);
  if (tid < BM)
    tok_s[tid] = tid < d0.rows ? tokids[(size_t)d0.e * TP + d0.ml * BM + tid] : 0;
  __syncthreads();

  int qitem = -1;
  Dec qd = d0;
  auto issue_stage = [&](int q) {
    const int jq = q / NST;
    const int ks = q - jq * NST;
    if (jq != qitem) {
      qitem = jq;
      qd = decomp(mt, pre, cntv, EL, blockIdx.x + jq * NCTA, NT, BM);
    }
    const int sl = jq & 1;
    const __nv_fp8_e4m3 *wg =
        W13 + (size_t)qd.e * (2 * (size_t)INTE * HID) + (size_t)(qd.nt * BNI) * HID;
    const __nv_fp8_e4m3 *wu = wg + (size_t)INTE * HID;
    const int koff = ks * RB;
    const int st = q % STG;
    constexpr int NCW = 2 * BNI * RB / 16;
    __nv_fp8_e4m3 *dw = Ws + (size_t)st * 2 * BNI * RB;
#pragma unroll
    for (int i = tid; i < NCW; i += 256) {
      int r = i / (RB / 16), c = i & (RB / 16 - 1);
      const __nv_fp8_e4m3 *src =
          (r < BNI ? wg : wu) + (size_t)(r & (BNI - 1)) * HID + koff + c * 16;
      cpa16(sm32(dw + swz<RB>(r, c * 16)), src);
    }
    __nv_fp8_e4m3 *da = As + (size_t)st * BM * RB;
    const int *tk = tok_s + sl * BM;
    const int nca = qd.rows * (RB / 16);
    for (int i = tid; i < nca; i += 256) {
      int r = i / (RB / 16), c = i & (RB / 16 - 1);
      cpa16(sm32(da + swz<RB>(r, c * 16)), A + (size_t)tk[r] * HID + koff + c * 16);
    }
    if (ks == 0) {
      // this item's activation + weight block scales
      const float *asrc = ascg + ((size_t)qd.e * TP + qd.ml * BM) * NKBLK;
      float *adst = sa_s + (size_t)sl * NKBLK * BM;
      for (int i = tid; i < NKBLK * qd.rows / 4; i += 256)
        cpa16(sm32(adst + i * 4), asrc + i * 4);
      const int nbg = (qd.nt * BNI) / QBLK;
      const int nbu = (INTE + qd.nt * BNI) / QBLK;
      const float *g0 = S13 + (size_t)qd.e * (32 * NKBLK) + (size_t)nbg * NKBLK;
      const float *g1 = S13 + (size_t)qd.e * (32 * NKBLK) + (size_t)nbu * NKBLK;
      float *gdst = sg_s + sl * 2 * NKBLK;
      for (int i = tid; i < NKBLK / 4; i += 256) {
        cpa16(sm32(gdst + i * 4), g0 + i * 4);
        cpa16(sm32(gdst + NKBLK + i * 4), g1 + i * 4);
      }
      if (tid == 0) {
        for (int i = (NKBLK / 4) * 4; i < NKBLK; i++) {
          gdst[i] = g0[i];
          gdst[NKBLK + i] = g1[i];
        }
      }
      // token ids of the NEXT item (needed STG-1 stages before it is consumed)
      if (jq + 1 < nloc) {
        Dec nd = decomp(mt, pre, cntv, EL, blockIdx.x + (jq + 1) * NCTA, NT, BM);
        const int *tsrc = tokids + (size_t)nd.e * TP + nd.ml * BM;
        int *tdst = tok_s + ((jq + 1) & 1) * BM;
        for (int i = tid; i < BM / 4; i += 256) cpa16(sm32(tdst + i * 4), tsrc + i * 4);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

#pragma unroll 1
  for (int p = 0; p < STG - 1; p++)
    if (p < tot) issue_stage(p);

  float accg[MAXM][4], accu[MAXM][4];
#pragma unroll
  for (int m = 0; m < MAXM; m++)
#pragma unroll
    for (int i = 0; i < 4; i++) { accg[m][i] = 0.f; accu[m][i] = 0.f; }

  Dec cd = d0;
  for (int j = 0; j < nloc; j++) {
    if (j > 0) cd = decomp(mt, pre, cntv, EL, blockIdx.x + j * NCTA, NT, BM);
    const int sl = j & 1;
    const int rows = cd.rows;
    const int nms = (rows + 15) >> 4;
    const float *sap = sa_s + (size_t)sl * NKBLK * BM;
    const float *sgp = sg_s + sl * 2 * NKBLK;
    const int pbase = j * NST;

    for (int ks = 0; ks < NST; ks++) {
      const int p = pbase + ks;
      waitg(min(STG - 2, tot - 1 - p));
      __syncthreads();
      if (p + STG - 1 < tot) issue_stage(p + STG - 1);
      const int st = p % STG;
      const __nv_fp8_e4m3 *Ap = As + (size_t)st * BM * RB;
      const __nv_fp8_e4m3 *Wp = Ws + (size_t)st * 2 * BNI * RB;
#pragma unroll
      for (int kb = 0; kb < KPS; kb++) {
        const int kbase = kb * QBLK;
        const int kqi = ks * KPS + kb;
        const float sgv = sgp[kqi], suv = sgp[NKBLK + kqi];
        constexpr int KS = MAXM >= 4 ? 1 : (MAXM == 2 ? 2 : 4);
        float cg[KS][MAXM][4], cu[KS][MAXM][4];
#pragma unroll
        for (int q2 = 0; q2 < KS; q2++)
#pragma unroll
          for (int m = 0; m < MAXM; m++)
#pragma unroll
            for (int i = 0; i < 4; i++) { cg[q2][m][i] = 0.f; cu[q2][m][i] = 0.f; }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          const int k0 = kbase + kk * 32;
          const int brow = warp * 8 + (lane & 7);
          const int bko = k0 + (((lane >> 3) & 1) << 4);
          uint32_t bg[2], bu[2];
          ldm_x2(bg, sm32(Wp + swz<RB>(brow, bko)));
          ldm_x2(bu, sm32(Wp + swz<RB>(BNI + brow, bko)));
          const int arow = lane & 15;
          const int ako = k0 + (((lane >> 4) & 1) << 4);
#pragma unroll
          for (int m = 0; m < MAXM; m++) {
            if (m < nms) {
              uint32_t a[4];
              ldm_x4(a, sm32(Ap + swz<RB>(m * 16 + arow, ako)));
              mma_f8(cg[kk % KS][m], a, bg);
              mma_f8(cu[kk % KS][m], a, bu);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
          if (m < nms) {
            const int r0 = m * 16 + (lane >> 2);
            const int r1 = min(r0 + 8, rows - 1);
            const float s0 = sap[r0 * NKBLK + kqi];
            const float s1 = sap[r1 * NKBLK + kqi];
            float tg[4], tu[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { tg[i] = cg[0][m][i]; tu[i] = cu[0][m][i]; }
#pragma unroll
            for (int q2 = 1; q2 < KS; q2++)
#pragma unroll
              for (int i = 0; i < 4; i++) { tg[i] += cg[q2][m][i]; tu[i] += cu[q2][m][i]; }
            accg[m][0] += tg[0] * (s0 * sgv);
            accg[m][1] += tg[1] * (s0 * sgv);
            accg[m][2] += tg[2] * (s1 * sgv);
            accg[m][3] += tg[3] * (s1 * sgv);
            accu[m][0] += tu[0] * (s0 * suv);
            accu[m][1] += tu[1] * (s0 * suv);
            accu[m][2] += tu[2] * (s1 * suv);
            accu[m][3] += tu[3] * (s1 * suv);
          }
        }
      }
    }
    __syncthreads();
    {

    // ---- SwiGLU + fp8 block quantise (one scale per 64-column block) ----
    {
      float cv[MAXM][4], am[MAXM][2];
#pragma unroll
      for (int m = 0; m < MAXM; m++) {
        am[m][0] = 0.f;
        am[m][1] = 0.f;
        if (m < nms) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            float u = accu[m][i];
            cv[m][i] = (u / (1.f + __expf(-u))) * accg[m][i];
          }
          float x0 = fmaxf(fabsf(cv[m][0]), fabsf(cv[m][1]));
          float x1 = fmaxf(fabsf(cv[m][2]), fabsf(cv[m][3]));
          x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
          x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
          x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
          x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
          am[m][0] = x0;
          am[m][1] = x1;
          if ((lane & 3) == 0) {
            amax_s[warp * BM + m * 16 + (lane >> 2)] = x0;
            amax_s[warp * BM + m * 16 + (lane >> 2) + 8] = x1;
          }
        }
      }
      __syncthreads();
      const long long sbase = pslot[cd.e] + (long long)cd.ml * BM;
#pragma unroll
      for (int m = 0; m < MAXM; m++) {
        if (m < nms) {
          const int r0 = m * 16 + (lane >> 2);
          const int r1 = r0 + 8;
          float m0 = am[m][0], m1 = am[m][1];
#pragma unroll
          for (int gq = 0; gq < 8; gq++) {
            m0 = fmaxf(m0, amax_s[gq * BM + r0]);
            if (r1 < rows) m1 = fmaxf(m1, amax_s[gq * BM + r1]);
          }
          float s0 = m0 * (1.f / 448.f), s1 = m1 * (1.f / 448.f);
          float i0 = s0 > 0.f ? 1.f / s0 : 0.f;
          float i1 = s1 > 0.f ? 1.f / s1 : 0.f;
          const int cc = cd.nt * BNI + warp * 8 + (lane & 3) * 2;
          if (r0 < rows) {
            Cq[(sbase + r0) * INTE + cc] = __nv_fp8_e4m3(cv[m][0] * i0);
            Cq[(sbase + r0) * INTE + cc + 1] = __nv_fp8_e4m3(cv[m][1] * i0);
            if (warp == 0 && (lane & 3) == 0) Cs[(sbase + r0) * 32 + cd.nt] = s0;
          }
          if (r1 < rows) {
            Cq[(sbase + r1) * INTE + cc] = __nv_fp8_e4m3(cv[m][2] * i1);
            Cq[(sbase + r1) * INTE + cc + 1] = __nv_fp8_e4m3(cv[m][3] * i1);
            if (warp == 0 && (lane & 3) == 0) Cs[(sbase + r1) * 32 + cd.nt] = s1;
          }
        }
      }
#pragma unroll
      for (int m = 0; m < MAXM; m++)
#pragma unroll
        for (int i = 0; i < 4; i++) { accg[m][i] = 0.f; accu[m][i] = 0.f; }
      __syncthreads();
    }
    }
  }
}

// ----------------------------------------------------------------- gemm2 ---
template <int BM, int RB, int STG>
__global__ MOE_LB void gemm2_kernel(
    const __nv_fp8_e4m3 *__restrict__ Cq, const float *__restrict__ Cs,
    const __nv_fp8_e4m3 *__restrict__ W2, const float *__restrict__ S2,
    __nv_bfloat16 *__restrict__ out, const int *__restrict__ cnt,
    int *__restrict__ cnt_clear, const int *__restrict__ tokids,
    const float *__restrict__ tokw, int T, int TP, int EL, int NCTA) {
  constexpr int BH = 64;
  constexpr int CSB = 64;
  constexpr int NH = HID / BH;
  constexpr int CPS = RB / CSB;
  constexpr int NST = INTE / RB;
  constexpr int MAXM = BM / 16;
  constexpr int NSC = INTE / CSB;

  extern __shared__ char smem[];
  __nv_fp8_e4m3 *Cm = reinterpret_cast<__nv_fp8_e4m3 *>(smem);
  __nv_fp8_e4m3 *Ws = Cm + (size_t)STG * BM * RB;
  float *sc_s = reinterpret_cast<float *>(Ws + (size_t)STG * BH * RB);
  float *s2_s = sc_s + (size_t)2 * NSC * BM;
  int *tok_s = reinterpret_cast<int *>(s2_s + 2 * NIBLK);
  int *mt = tok_s + 2 * BM;
  int *pre = mt + 32;
  int *cntv = pre + 32;
  int *pslot = cntv + 32;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ int s_total;
  if (tid == 0) {
    int tt = 0, as = 0;
    for (int i = 0; i < EL; i++) {
      int c = cnt[i];
      cntv[i] = c;
      mt[i] = (c + BM - 1) / BM;
      pre[i] = tt;
      pslot[i] = as;
      tt += mt[i];
      as += mt[i] * BM;
    }
    s_total = tt;
  }
  if (blockIdx.x < (unsigned)EL && tid == 0) cnt_clear[blockIdx.x] = 0;
  __syncthreads();
  const int nwork = s_total * NH;
  if (blockIdx.x >= (unsigned)nwork) return;
  const int nloc = (nwork - (int)blockIdx.x + NCTA - 1) / NCTA;
  const int tot = nloc * NST;

  Dec d0 = decomp(mt, pre, cntv, EL, blockIdx.x, NH, BM);
  if (tid < BM)
    tok_s[tid] = tid < d0.rows ? tokids[(size_t)d0.e * TP + d0.ml * BM + tid] : 0;
  __syncthreads();

  int qitem = -1;
  Dec qd = d0;
  auto issue_stage = [&](int q) {
    const int jq = q / NST;
    const int ks = q - jq * NST;
    if (jq != qitem) {
      qitem = jq;
      qd = decomp(mt, pre, cntv, EL, blockIdx.x + jq * NCTA, NH, BM);
    }
    const int sl = jq & 1;
    const __nv_fp8_e4m3 *w2 =
        W2 + (size_t)qd.e * ((size_t)HID * INTE) + (size_t)(qd.nt * BH) * INTE;
    const long long sbase = pslot[qd.e] + (long long)qd.ml * BM;
    const int koff = ks * RB;
    const int st = q % STG;
    constexpr int NCW = BH * RB / 16;
    __nv_fp8_e4m3 *dw = Ws + (size_t)st * BH * RB;
#pragma unroll
    for (int i = tid; i < NCW; i += 256) {
      int r = i / (RB / 16), c = i & (RB / 16 - 1);
      cpa16(sm32(dw + swz<RB>(r, c * 16)), w2 + (size_t)r * INTE + koff + c * 16);
    }
    __nv_fp8_e4m3 *dc = Cm + (size_t)st * BM * RB;
    const int nca = qd.rows * (RB / 16);
    for (int i = tid; i < nca; i += 256) {
      int r = i / (RB / 16), c = i & (RB / 16 - 1);
      cpa16(sm32(dc + swz<RB>(r, c * 16)), Cq + (sbase + r) * INTE + koff + c * 16);
    }
    if (ks == 0) {
      const float *csrc = Cs + sbase * 32;
      float *cdst = sc_s + (size_t)sl * NSC * BM;
      for (int i = tid; i < NSC * qd.rows / 4; i += 256)
        cpa16(sm32(cdst + i * 4), csrc + i * 4);
      const float *s2src =
          S2 + (size_t)qd.e * (NKBLK * NIBLK) + (size_t)((qd.nt * BH) / QBLK) * NIBLK;
      float *s2dst = s2_s + sl * NIBLK;
      for (int i = tid; i < NIBLK / 4; i += 256) cpa16(sm32(s2dst + i * 4), s2src + i * 4);
      if (jq + 1 < nloc) {
        Dec nd = decomp(mt, pre, cntv, EL, blockIdx.x + (jq + 1) * NCTA, NH, BM);
        const int *tsrc = tokids + (size_t)nd.e * TP + nd.ml * BM;
        int *tdst = tok_s + ((jq + 1) & 1) * BM;
        for (int i = tid; i < BM / 4; i += 256) cpa16(sm32(tdst + i * 4), tsrc + i * 4);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

#pragma unroll 1
  for (int p = 0; p < STG - 1; p++)
    if (p < tot) issue_stage(p);

  float acc[MAXM][4];
#pragma unroll
  for (int m = 0; m < MAXM; m++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[m][i] = 0.f;

  Dec cd = d0;
  for (int j = 0; j < nloc; j++) {
    if (j > 0) cd = decomp(mt, pre, cntv, EL, blockIdx.x + j * NCTA, NH, BM);
    const int sl = j & 1;
    const int rows = cd.rows;
    const int nms = (rows + 15) >> 4;
    const float *scp = sc_s + (size_t)sl * NSC * BM;
    const float *s2p = s2_s + sl * NIBLK;
    const int pbase = j * NST;

    for (int ks = 0; ks < NST; ks++) {
      const int p = pbase + ks;
      waitg(min(STG - 2, tot - 1 - p));
      __syncthreads();
      if (p + STG - 1 < tot) issue_stage(p + STG - 1);
      const int st = p % STG;
      const __nv_fp8_e4m3 *Cp = Cm + (size_t)st * BM * RB;
      const __nv_fp8_e4m3 *Wp = Ws + (size_t)st * BH * RB;
#pragma unroll
      for (int cb = 0; cb < CPS; cb++) {
        const int kbase = cb * CSB;
        const int sci = ks * CPS + cb;
        const float s2v = s2p[(sci * CSB) / QBLK];
        constexpr int KS2 = MAXM >= 4 ? 1 : 2;
        float c[KS2][MAXM][4];
#pragma unroll
        for (int q2 = 0; q2 < KS2; q2++)
#pragma unroll
          for (int m = 0; m < MAXM; m++)
#pragma unroll
            for (int i = 0; i < 4; i++) c[q2][m][i] = 0.f;
#pragma unroll
        for (int kk = 0; kk < CSB / 32; kk++) {
          const int k0 = kbase + kk * 32;
          const int brow = warp * 8 + (lane & 7);
          const int bko = k0 + (((lane >> 3) & 1) << 4);
          uint32_t b[2];
          ldm_x2(b, sm32(Wp + swz<RB>(brow, bko)));
          const int arow = lane & 15;
          const int ako = k0 + (((lane >> 4) & 1) << 4);
#pragma unroll
          for (int m = 0; m < MAXM; m++) {
            if (m < nms) {
              uint32_t a[4];
              ldm_x4(a, sm32(Cp + swz<RB>(m * 16 + arow, ako)));
              mma_f8(c[kk % KS2][m], a, b);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
          if (m < nms) {
            const int r0 = m * 16 + (lane >> 2);
            const int r1 = min(r0 + 8, rows - 1);
            const float sc0 = scp[r0 * NSC + sci] * s2v;
            const float sc1 = scp[r1 * NSC + sci] * s2v;
            float t[4];
#pragma unroll
            for (int i = 0; i < 4; i++) t[i] = c[0][m][i];
#pragma unroll
            for (int q2 = 1; q2 < KS2; q2++)
#pragma unroll
              for (int i = 0; i < 4; i++) t[i] += c[q2][m][i];
            acc[m][0] += t[0] * sc0;
            acc[m][1] += t[1] * sc0;
            acc[m][2] += t[2] * sc1;
            acc[m][3] += t[3] * sc1;
          }
        }
      }
    }
    __syncthreads();

    {
      const int cc = cd.nt * BH + warp * 8 + (lane & 3) * 2;
      const int *tk = tokids + (size_t)cd.e * TP + cd.ml * BM;
#pragma unroll
      for (int m = 0; m < MAXM; m++) {
        if (m < nms) {
          const int r0 = m * 16 + (lane >> 2);
          const int r1 = r0 + 8;
          if (r0 < rows) {
            float w0 = tokw[(size_t)cd.e * TP + cd.ml * BM + r0];
            __nv_bfloat162 v = __floats2bfloat162_rn(acc[m][0] * w0, acc[m][1] * w0);
            atomicAdd(reinterpret_cast<__nv_bfloat162 *>(out + (size_t)tk[r0] * HID + cc),
                      v);
          }
          if (r1 < rows) {
            float w1 = tokw[(size_t)cd.e * TP + cd.ml * BM + r1];
            __nv_bfloat162 v = __floats2bfloat162_rn(acc[m][2] * w1, acc[m][3] * w1);
            atomicAdd(reinterpret_cast<__nv_bfloat162 *>(out + (size_t)tk[r1] * HID + cc),
                      v);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < MAXM; m++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[m][i] = 0.f;
      __syncthreads();
    }
  }
}

}  // namespace

// ------------------------------------------------------------------ host ---
template <int BM, int RB, int STG>
static void run_g1(int grid, cudaStream_t st, const void *A, const void *W,
                   const float *S, void *Cq, float *Cs, const int *cnt, const int *tok,
                   void *out, const float *ascg, int T, int TP, long long on, int EL) {
  constexpr size_t sm = (size_t)STG * BM * RB + (size_t)STG * 128 * RB +
                        4 * (size_t)2 * NKBLK * BM + 4 * 4 * NKBLK + 4 * 8 * BM +
                        4 * 2 * BM + 4 * 128 + 256;
  auto k = gemm1_kernel<BM, RB, STG>;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    once = true;
  }
  k<<<grid, 256, sm, st>>>((const __nv_fp8_e4m3 *)A, (const __nv_fp8_e4m3 *)W, S,
                           (__nv_fp8_e4m3 *)Cq, Cs, cnt, tok, (__nv_bfloat16 *)out, ascg,
                           T, TP, on, EL, grid);
}

template <int BM, int RB, int STG>
static void run_g2(int grid, cudaStream_t st, const void *Cq, const float *Cs,
                   const void *W, const float *S, void *out, const int *cnt,
                   int *cnt_clear, const int *tok, const float *tw, int T, int TP,
                   int EL) {
  constexpr size_t sm = (size_t)STG * BM * RB + (size_t)STG * 64 * RB +
                        4 * (size_t)2 * (INTE / 64) * BM + 4 * 2 * NIBLK + 4 * 2 * BM +
                        4 * 128 + 256;
  auto k = gemm2_kernel<BM, RB, STG>;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    once = true;
  }
  k<<<grid, 256, sm, st>>>((const __nv_fp8_e4m3 *)Cq, Cs, (const __nv_fp8_e4m3 *)W, S,
                           (__nv_bfloat16 *)out, cnt, cnt_clear, tok, tw, T, TP, EL,
                           grid);
}

template <int MAXC, int WPB>
static void run_ffma(int nc1, int nc2, cudaStream_t st, const void *A, const void *W1,
                     const float *S1, float *Cf, void *Cq, float *Cs, const void *W2,
                     const float *S2, const int *cnt, int *cnt_clear, const int *tok,
                     const float *tw, const float *ascg, void *out, int T, int TP,
                     long long on, int EL) {
  gemm1_ffma_kernel<MAXC, WPB><<<nc1, 32 * WPB, 0, st>>>(
      (const __nv_fp8_e4m3 *)A, (const __nv_fp8_e4m3 *)W1, S1, Cf, cnt, tok, ascg,
      (__nv_bfloat16 *)out, T, TP, on, EL, nc1);
  quantc_kernel<<<nc2, 256, 0, st>>>(Cf, (__nv_fp8_e4m3 *)Cq, Cs, cnt, EL);
  gemm2_ffma_kernel<MAXC, WPB><<<nc2, 32 * WPB, 0, st>>>(
      (const __nv_fp8_e4m3 *)Cq, Cs, (const __nv_fp8_e4m3 *)W2, S2,
      (__nv_bfloat16 *)out, cnt, cnt_clear, tok, tw, T, TP, EL, nc2);
}

torch::Tensor moe_run(torch::Tensor routing_logits, torch::Tensor routing_bias,
                      torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                      torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                      torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
                      int64_t local_expert_offset, double routed_scaling_factor,
                      torch::Tensor cnt, torch::Tensor cnt_clear, torch::Tensor tokids,
                      torch::Tensor tokw, torch::Tensor Cq, torch::Tensor Cs,
                      torch::Tensor ascg, torch::Tensor Cf, int64_t pairs_cap,
                      int64_t bm, int64_t nc1, int64_t nc2, int64_t cfg1,
                      int64_t cfg2, int64_t use_ffma) {
  const int T = (int)routing_logits.size(0);
  const int TP = (int)tokids.size(1);
  const int EL = (int)gemm1_weights.size(0);
  auto out = torch::empty(
      {T, HID},
      torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));
  const long long on = (long long)T * HID;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  (void)pairs_cap;

  routing_warp_kernel<8><<<(T + 7) / 8, 256, 0, st>>>(
      routing_logits.data_ptr<float>(), (const __nv_bfloat16 *)routing_bias.data_ptr(),
      hidden_states_scale.data_ptr<float>(), cnt.data_ptr<int>(),
      tokids.data_ptr<int>(), tokw.data_ptr<float>(), ascg.data_ptr<float>(), T, TP,
      (int)local_expert_offset, (float)routed_scaling_factor, EL, 1);

  const void *A = hidden_states.data_ptr();
  const void *W1 = gemm1_weights.data_ptr();
  const float *S1 = gemm1_weights_scale.data_ptr<float>();
  const void *W2 = gemm2_weights.data_ptr();
  const float *S2 = gemm2_weights_scale.data_ptr<float>();
  void *Cqp = Cq.data_ptr();
  float *Csp = Cs.data_ptr<float>();
  const int *cp = cnt.data_ptr<int>();
  int *ccp = cnt_clear.data_ptr<int>();
  const int *tp = tokids.data_ptr<int>();
  const float *twp = tokw.data_ptr<float>();
  void *op = out.data_ptr();
  const float *agp = ascg.data_ptr<float>();

  if (use_ffma) {
    float *Cfp = Cf.data_ptr<float>();
    if (use_ffma == 1)
      run_ffma<8, 8>((int)nc1, (int)nc2, st, A, W1, S1, Cfp, Cqp, Csp, W2, S2, cp, ccp,
                      tp, twp, agp, op, T, TP, on, EL);
    else
      run_ffma<16, 8>((int)nc1, (int)nc2, st, A, W1, S1, Cfp, Cqp, Csp, W2, S2, cp, ccp,
                      tp, twp, agp, op, T, TP, on, EL);
    return out;
  }

#define CASE1(BMV, RBV, STGV) \
  run_g1<BMV, RBV, STGV>((int)nc1, st, A, W1, S1, Cqp, Csp, cp, tp, op, agp, T, TP, on, EL)
#define CASE2(BMV, RBV, STGV) \
  run_g2<BMV, RBV, STGV>((int)nc2, st, Cqp, Csp, W2, S2, op, cp, ccp, tp, twp, T, TP, EL)

  switch (bm * 2 + cfg1) {
    case 16 * 2 + 0: CASE1(16, 128, 5); break;
    case 16 * 2 + 1: CASE1(16, 256, 4); break;
    case 32 * 2 + 0: CASE1(32, 128, 5); break;
    case 32 * 2 + 1: CASE1(32, 256, 3); break;
    case 64 * 2 + 0: CASE1(64, 128, 4); break;
    case 64 * 2 + 1: CASE1(64, 256, 3); break;
    case 128 * 2 + 0: CASE1(128, 128, 3); break;
    default: CASE1(128, 128, 4); break;
  }
  switch (bm * 2 + cfg2) {
    case 16 * 2 + 0: CASE2(16, 128, 8); break;
    case 16 * 2 + 1: CASE2(16, 256, 6); break;
    case 32 * 2 + 0: CASE2(32, 128, 8); break;
    case 32 * 2 + 1: CASE2(32, 256, 6); break;
    case 64 * 2 + 0: CASE2(64, 128, 6); break;
    case 64 * 2 + 1: CASE2(64, 256, 4); break;
    case 128 * 2 + 0: CASE2(128, 128, 5); break;
    default: CASE2(128, 256, 3); break;
  }
  return out;
}

// Routing only: used when the GEMMs are provided by the Triton path.
void moe_routing(torch::Tensor routing_logits, torch::Tensor routing_bias,
                 torch::Tensor hidden_states_scale, torch::Tensor cnt,
                 torch::Tensor tokids, torch::Tensor tokw, torch::Tensor ascg,
                 torch::Tensor nloc, torch::Tensor tslot,
                 int64_t local_expert_offset,
                 double routed_scaling_factor, int64_t EL, int64_t want_ascg,
                 int64_t kind) {
  const int T = (int)routing_logits.size(0);
  const int TP = (int)tokids.size(1);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (want_ascg == 0 && kind == 1) {
    routing_fast_kernel<8><<<(T + 7) / 8, 256, 0, st>>>(
        routing_logits.data_ptr<float>(),
        (const __nv_bfloat16 *)routing_bias.data_ptr(), cnt.data_ptr<int>(),
        tokids.data_ptr<int>(), tokw.data_ptr<float>(),
        nloc.numel() > 1 ? nloc.data_ptr<int>() : nullptr,
        tslot.numel() > 1 ? tslot.data_ptr<int>() : nullptr, T, TP,
        (int)local_expert_offset, (float)routed_scaling_factor, (int)EL);
    return;
  }
  routing_warp_kernel<8><<<(T + 7) / 8, 256, 0, st>>>(
      routing_logits.data_ptr<float>(), (const __nv_bfloat16 *)routing_bias.data_ptr(),
      hidden_states_scale.data_ptr<float>(), cnt.data_ptr<int>(),
      tokids.data_ptr<int>(), tokw.data_ptr<float>(), ascg.data_ptr<float>(), T, TP,
      (int)local_expert_offset, (float)routed_scaling_factor, (int)EL,
      (int)want_ascg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_run", &moe_run);
  m.def("moe_routing", &moe_routing);
}
