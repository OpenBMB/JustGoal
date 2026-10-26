// MLA paged prefill causal, H=16, d_ckv=512, d_kpe=64, page_size=1, B200 (sm_100).
// Final. One CTA per (query token, output-dim half); 16 warps.
//   * page indices + Q staged with cp.async while the KV gather is issued
//   * KV (48 rows x 576) staged in two cp.async groups so QK on the first
//     256 dims overlaps the transfer of the rest
//   * QK^T: M=16 heads, N=keys, K=576 split across 8 warps, mma.m16n8k16 fed
//     by ldmatrix; per-warp partials reduced in shared memory
//   * softmax in fp32 (exp2 domain), P written back as bf16
//   * P@V: ldmatrix.trans supplies V^T fragments straight from the same KV tile
//   * output staged in shared memory so the global store is fully coalesced
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

#define DEVINL __device__ __forceinline__
#ifndef NWARPS_CFG
#define NWARPS_CFG 8
#endif
#ifndef SPLIT_CFG
#define SPLIT_CFG 256
#endif
#ifndef SD_CFG
#define SD_CFG 2
#endif
#ifndef QKWARPS_CFG
#define QKWARPS_CFG 5
#endif

namespace {

constexpr int NH       = 16;
constexpr int DCKV     = 512;
constexpr int DPE      = 64;
constexpr int DTOT     = 576;
constexpr int SQS      = 584;
constexpr int NKVPAD   = 48;
constexpr int NTILES   = 5;
constexpr int KSTEPS   = 3;
constexpr int NCHUNK   = DTOT / 32;         // 18
constexpr int SPLIT    = SPLIT_CFG;          // stage-0 depth
constexpr int NC0      = SPLIT / 32;        // 9 chunks in stage 0
constexpr int NWARPS   = NWARPS_CFG;
constexpr int QKWARPS  = QKWARPS_CFG;
constexpr int NTHREADS = NWARPS * 32;
constexpr int SPS      = 56;
constexpr int SSS      = 52;
constexpr int SOS      = (DCKV / SD_CFG) + 8;
constexpr int TPR      = NTHREADS / NH;     // softmax threads per head row
constexpr int SCOLS    = KSTEPS * 16;       // 48
constexpr int CPT      = (SCOLS + TPR - 1) / TPR;
constexpr int SD       = SD_CFG;                 // output-dim grid split
constexpr int DTCTA    = (DCKV / 8) / SD;        // d-tiles per CTA
constexpr int DPW      = DTCTA / NWARPS;         // d-tiles per warp

static_assert(QKWARPS <= NWARPS, "QKWARPS must not exceed NWARPS");
static_assert((DCKV / 8) % (SD * NWARPS) == 0, "d-tiles must divide evenly over SD*NWARPS");
static_assert(DPW % 2 == 0, "need an even number of d-tiles per warp");
static_assert(NCHUNK * 32 == DTOT, "chunking mismatch");
static_assert(SPLIT % 32 == 0 && SPLIT > 0 && SPLIT < DCKV, "bad pipeline split");
constexpr int BW = (DPW >= 4) ? 4 : 2;      // d-tiles per PV inner iteration

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;

DEVINL uint32_t sptr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVINL void ldm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVINL void ldm_x4_t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVINL void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVINL void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
}

#define CPA_COMMIT  asm volatile("cp.async.commit_group;")
#define CPA_WAIT(N) asm volatile("cp.async.wait_group " #N ";")

__global__ __launch_bounds__(NTHREADS) void mla_v6(
    const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv,    const __nv_bfloat16* __restrict__ kpe,
    const int* __restrict__ kv_indices,
    __nv_bfloat16* __restrict__ out, float* __restrict__ lse_out,
    float sm_scale, int qlen, int kvlen, unsigned long long* __restrict__ tms)
{
  const int qi = blockIdx.x, sd = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#ifdef TIMEIT
#define TSTAMP(i) do { __syncthreads(); if (tid == 0) tms[(size_t)qi * 8 + (i)] = clock64(); } while (0)
#else
#define TSTAMP(i) do {} while (0)
#endif
  TSTAMP(0);

  int n_kv = qi + (kvlen - qlen) + 1;
  if (n_kv > kvlen) n_kv = kvlen;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sKV = sQ  + NH * SQS;
  __nv_bfloat16* sP  = sKV + NKVPAD * SQS;
  float*         sSp = reinterpret_cast<float*>(sP + NH * SPS);
  __nv_bfloat16* sO  = reinterpret_cast<__nv_bfloat16*>(sSp + QKWARPS * NH * SSS);
  int*           sIdx= reinterpret_cast<int*>(sO + NH * SOS);
  float*         sL  = reinterpret_cast<float*>(sIdx + NKVPAD);
  float*         sM  = sL + NH;

#define ISSUE(N, ...) do {                                        \
    _Pragma("unroll")                                              \
    for (int r = 0; r < ((N) + NTHREADS - 1) / NTHREADS; ++r) {    \
      const int c = tid + r * NTHREADS;                            \
      if ((N) % NTHREADS == 0 || c < (N)) { __VA_ARGS__ }                 \
    } } while (0)

  if (tid < NKVPAD) sIdx[tid] = kv_indices[tid < n_kv ? tid : n_kv - 1];

  { const __nv_bfloat16* qn = q_nope + (size_t)qi * NH * DCKV;
    ISSUE(NH * (DCKV / 8), { int h = c >> 6, o = (c & 63) << 3;
                             cpa16(sptr(&sQ[h * SQS + o]), qn + h * DCKV + o); });
    const __nv_bfloat16* qp = q_pe + (size_t)qi * NH * DPE;
    ISSUE(NH * (DPE / 8), { int h = c >> 3, o = (c & 7) << 3;
                            cpa16(sptr(&sQ[h * SQS + DCKV + o]), qp + h * DPE + o); }); }
  __syncthreads();
  TSTAMP(1);

  { constexpr int H0 = SPLIT / 8, H1 = (DCKV - SPLIT) / 8;
    ISSUE(NKVPAD * H0, { int j = c / H0, o = (c % H0) << 3;
                         cpa16(sptr(&sKV[j * SQS + o]), ckv + (size_t)sIdx[j] * DCKV + o); });
    CPA_COMMIT;                                            // ---- group 0 ----
    ISSUE(NKVPAD * H1, { int j = c / H1, o = SPLIT + ((c % H1) << 3);
                         cpa16(sptr(&sKV[j * SQS + o]), ckv + (size_t)sIdx[j] * DCKV + o); });
    ISSUE(NKVPAD * (DPE / 8), { int j = c >> 3, o = (c & 7) << 3;
                                cpa16(sptr(&sKV[j * SQS + DCKV + o]), kpe + (size_t)sIdx[j] * DPE + o); });
    CPA_COMMIT; }                                          // ---- group 1 ----
  CPA_WAIT(1);
  __syncthreads();
  TSTAMP(2);

  // ---------------- QK^T ----------------
  {
    const int ar = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int ac = ((lane >> 4) & 1) << 3;
    const int br = lane & 7;
    const int bc = ((lane >> 3) & 3) << 3;
    const __nv_bfloat16* abase = &sQ[(size_t)ar * SQS + ac];
    float acc[NTILES][4];
#pragma unroll
    for (int t = 0; t < NTILES; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u) acc[t][u] = 0.f;

    constexpr int NC1  = NCHUNK - NC0;
    constexpr int CMX0 = (NC0 + QKWARPS - 1) / QKWARPS;
    constexpr int CMX1 = (NC1 + QKWARPS - 1) / QKWARPS;
    constexpr int CMAX = CMX0 > CMX1 ? CMX0 : CMX1;
#pragma unroll 1
    for (int st = 0; st < 2; ++st) {
      if (warp < QKWARPS) {
        const int base = st ? NC0 : 0;
        const int span = st ? NC1 : NC0;
        const int c0 = base + (warp * span) / QKWARPS;
        const int c1 = base + ((warp + 1) * span) / QKWARPS;
#pragma unroll
        for (int r = 0; r < CMAX; ++r) {
          const int c = c0 + r;
          if (c >= c1) break;
          const int k0 = c << 5;
          uint32_t a0[4], a1[4], b[NTILES][4];
          ldm_x4(a0, sptr(abase + k0));
          ldm_x4(a1, sptr(abase + k0 + 16));
#pragma unroll
          for (int t = 0; t < NTILES; ++t)
            ldm_x4(b[t], sptr(&sKV[(size_t)((t << 3) + br) * SQS + bc + k0]));
#pragma unroll
          for (int t = 0; t < NTILES; ++t) {
            mma16816(acc[t], a0, &b[t][0]);
            mma16816(acc[t], a1, &b[t][2]);
          }
        }
      }
      if (st == 0) { CPA_WAIT(0); __syncthreads(); }
    }

    if (warp < QKWARPS) {
      const int ch = lane >> 2, cj = (lane & 3) << 1;
      float* dst = sSp + (size_t)warp * NH * SSS;
#pragma unroll
      for (int t = 0; t < NTILES; ++t) {
        const int j0 = (t << 3) + cj;
        *reinterpret_cast<float2*>(&dst[ch * SSS + j0])       = make_float2(acc[t][0], acc[t][1]);
        *reinterpret_cast<float2*>(&dst[(ch + 8) * SSS + j0]) = make_float2(acc[t][2], acc[t][3]);
      }
    }
  }
  __syncthreads();
  TSTAMP(3);

  // ---------------- softmax ----------------
  {
    const int row = tid / TPR, sub = tid % TPR;
    const float scl = sm_scale * LOG2E;
    float v[CPT];
    float mx = -CUDART_INF_F;
#pragma unroll
    for (int t = 0; t < CPT; ++t) {
      const int j = sub + t * TPR;
      float x = -CUDART_INF_F;
      if (j < NTILES * 8 && j < n_kv) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < QKWARPS; ++w) s += sSp[(size_t)w * NH * SSS + row * SSS + j];
        x = s * scl;
      }
      v[t] = x;
      mx = fmaxf(mx, x);
    }
#pragma unroll
    for (int o = 1; o < TPR; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < CPT; ++t) { v[t] = exp2f(v[t] - mx); sum += v[t]; }
#pragma unroll
    for (int o = 1; o < TPR; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
#pragma unroll
    for (int t = 0; t < CPT; ++t) {
      const int j = sub + t * TPR;
      if (j < SCOLS) sP[row * SPS + j] = __float2bfloat16(v[t]);
    }
    if (sub == 0) { sL[row] = 1.f / sum; sM[row] = (mx + log2f(sum)) * LN2; }
  }
  __syncthreads();
  TSTAMP(4);

  // ---------------- P @ V ----------------
  {
    const int ar = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int ac = ((lane >> 4) & 1) << 3;
    uint32_t af[KSTEPS][4];
#pragma unroll
    for (int k = 0; k < KSTEPS; ++k) ldm_x4(af[k], sptr(&sP[(size_t)ar * SPS + (k << 4) + ac]));
    const int g = lane >> 3;
    const int brw = ((g & 1) << 3) + (lane & 7);
    const int bcl = (g >> 1) << 3;
    const int ch = lane >> 2, cj = (lane & 3) << 1;
    const float rl0 = sL[ch], rl1 = sL[ch + 8];
#pragma unroll
    for (int it = 0; it < DPW / BW; ++it) {
      const int d0 = (sd * DTCTA + warp * DPW + it * BW) << 3;
      float acc[BW][4];
#pragma unroll
      for (int t = 0; t < BW; ++t)
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[t][u] = 0.f;
      uint32_t bb[KSTEPS][BW / 2][4];
#pragma unroll
      for (int k = 0; k < KSTEPS; ++k)
#pragma unroll
        for (int u = 0; u < BW / 2; ++u)
          ldm_x4_t(bb[k][u], sptr(&sKV[(size_t)((k << 4) + brw) * SQS + d0 + (u << 4) + bcl]));
#pragma unroll
      for (int k = 0; k < KSTEPS; ++k)
#pragma unroll
        for (int u = 0; u < BW / 2; ++u) {
          mma16816(acc[u * 2],     af[k], &bb[k][u][0]);
          mma16816(acc[u * 2 + 1], af[k], &bb[k][u][2]);
        }
#pragma unroll
      for (int t = 0; t < BW; ++t) {
        const int dd = d0 + (t << 3) + cj;
        *reinterpret_cast<__nv_bfloat162*>(&sO[ch * SOS + dd - sd * DTCTA * 8]) =
            __floats2bfloat162_rn(acc[t][0] * rl0, acc[t][1] * rl0);
        *reinterpret_cast<__nv_bfloat162*>(&sO[(ch + 8) * SOS + dd - sd * DTCTA * 8]) =
            __floats2bfloat162_rn(acc[t][2] * rl1, acc[t][3] * rl1);
      }
    }
  }
  __syncthreads();
  TSTAMP(5);

  { __nv_bfloat16* ob = out + (size_t)qi * NH * DCKV + sd * (DCKV / SD);
    constexpr int DW = DCKV / SD;
    ISSUE(NH * (DW / 8), { int h = c / (DW / 8), o = (c % (DW / 8)) << 3;
        *reinterpret_cast<uint4*>(ob + (size_t)h * DCKV + o) =
            *reinterpret_cast<const uint4*>(&sO[h * SOS + o]); });
    if (sd == 0 && tid < NH) lse_out[(size_t)qi * NH + tid] = sM[tid]; }
  TSTAMP(6);
}

}  // namespace

void mla_prefill_v6(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
                    torch::Tensor kv_indices, torch::Tensor out, torch::Tensor lse, torch::Tensor tms,
                    double sm_scale, int64_t qlen, int64_t kvlen)
{
  const size_t smem_bytes =
      (size_t)NH * SQS * 2 + (size_t)NKVPAD * SQS * 2 + (size_t)NH * SPS * 2 +
      (size_t)QKWARPS * NH * SSS * 4 + (size_t)NH * SOS * 2 + (size_t)NKVPAD * 4 + 2 * NH * 4;
  static bool cfg = false;
  if (!cfg) { cudaFuncSetAttribute(mla_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg = true; }
  mla_v6<<<dim3((unsigned)qlen, SD), NTHREADS, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr()),
      kv_indices.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
      (float)sm_scale, (int)qlen, (int)kvlen,
      reinterpret_cast<unsigned long long*>(tms.data_ptr()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mla_prefill_v6", &mla_prefill_v6, "MLA v6"); }
