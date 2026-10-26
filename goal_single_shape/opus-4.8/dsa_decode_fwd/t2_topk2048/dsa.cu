#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define D 512
#define DPE 64
#define H 16
#define INV_LN2 1.4426950408889634f

#ifndef NW
#define NW 8      // kernelspan-tuned: NW=8/12 ~2x faster than NW=16 (occupancy: 2 blocks/SM)
#endif
#define NT (NW * 32)
#ifndef HPB
#define HPB 1              // heads per block (cuts K-load redundancy HPB x)
#endif
#define TOPK_MAX 2048
#ifndef PF
#define PF 1
#endif

// grid = (H/HPB, T). One block handles HPB heads of token t -> loads each K row once for
// HPB heads. No global scratch. warp-per-row flash, non-atomic warp-combine, write out.
__global__ void dsa_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ Kc,
    const __nv_bfloat16* __restrict__ Kp,
    const int* __restrict__ idx,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    float scale, int TOPK) {
  const int hb = blockIdx.x;          // head-group
  const int t = blockIdx.y;
  const int h0 = hb * HPB;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int DK = D / 32;
  const int PK = DPE / 32;

  extern __shared__ char smem[];
  int* clist = reinterpret_cast<int*>(smem);
  float* sacc = reinterpret_cast<float*>(clist + TOPK_MAX);   // [NW][HPB][D]
  float* sm = sacc + NW * HPB * D;                            // [NW][HPB]
  float* sl = sm + NW * HPB;                                  // [NW][HPB]
  __shared__ int nv_s;

  float qn[HPB][DK], qp[HPB][PK];
#pragma unroll
  for (int hh = 0; hh < HPB; hh++) {
    int th = t * H + h0 + hh;
    const float4* p = reinterpret_cast<const float4*>(q_nope + (long)th * D + lane * DK);
    float4 a = p[0], b = p[1];
    const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(&a);
    const __nv_bfloat16* hbp = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
    for (int j = 0; j < 8; j++) { qn[hh][j] = __bfloat162float(ha[j]); qn[hh][j + 8] = __bfloat162float(hbp[j]); }
    const __nv_bfloat16* qpp = q_pe + (long)th * DPE + lane * PK;
#pragma unroll
    for (int k = 0; k < PK; k++) qp[hh][k] = __bfloat162float(qpp[k]);
  }

  if (tid == 0) nv_s = 0;
  __syncthreads();
  const int* idx_row = idx + (long)t * TOPK;
  for (int p = tid; p < TOPK; p += NT) {
    int r = idx_row[p];
    if (r >= 0) clist[atomicAdd(&nv_s, 1)] = r;
  }
  __syncthreads();
  int nv = nv_s;

  float m[HPB], l[HPB], acc[HPB][DK];
#pragma unroll
  for (int hh = 0; hh < HPB; hh++) {
    m[hh] = -1e30f; l[hh] = 0.f;
#pragma unroll
    for (int k = 0; k < DK; k++) acc[hh][k] = 0.f;
  }

  int nrow = (nv > wid) ? (nv - wid + NW - 1) / NW : 0;
  for (int base = 0; base < nrow; base += PF) {
    // issue PF independent row loads together (ILP hides Kc latency at low occupancy)
    float kcv[PF][DK], kpv[PF][PK];
    int valid[PF];
#pragma unroll
    for (int sp = 0; sp < PF; sp++) {
      int ri = base + sp;
      valid[sp] = ri < nrow;
      int r = valid[sp] ? clist[wid + ri * NW] : 0;
      const float4* p = reinterpret_cast<const float4*>(Kc + (long)r * D + lane * DK);
      float4 a = p[0], b = p[1];
      const __nv_bfloat16* ha = reinterpret_cast<const __nv_bfloat16*>(&a);
      const __nv_bfloat16* hbp = reinterpret_cast<const __nv_bfloat16*>(&b);
#pragma unroll
      for (int k = 0; k < 8; k++) { kcv[sp][k] = __bfloat162float(ha[k]); kcv[sp][k + 8] = __bfloat162float(hbp[k]); }
      const __nv_bfloat16* kp = Kp + (long)r * DPE + lane * PK;
#pragma unroll
      for (int k = 0; k < PK; k++) kpv[sp][k] = __bfloat162float(kp[k]);
    }
#pragma unroll
    for (int sp = 0; sp < PF; sp++) {
      if (!valid[sp]) continue;
#pragma unroll
      for (int hh = 0; hh < HPB; hh++) {
        float dot = 0.f;
#pragma unroll
        for (int k = 0; k < DK; k++) dot += qn[hh][k] * kcv[sp][k];
#pragma unroll
        for (int k = 0; k < PK; k++) dot += qp[hh][k] * kpv[sp][k];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
        float logit = dot * scale;
        float mnew = fmaxf(m[hh], logit);
        float alpha = __expf(m[hh] - mnew);
        float pval = __expf(logit - mnew);
        l[hh] = l[hh] * alpha + pval;
#pragma unroll
        for (int k = 0; k < DK; k++) acc[hh][k] = acc[hh][k] * alpha + pval * kcv[sp][k];
        m[hh] = mnew;
      }
    }
  }

#pragma unroll
  for (int hh = 0; hh < HPB; hh++) {
    if (lane == 0) { sm[wid * HPB + hh] = m[hh]; sl[wid * HPB + hh] = l[hh]; }
#pragma unroll
    for (int k = 0; k < DK; k++) sacc[(wid * HPB + hh) * D + lane * DK + k] = acc[hh][k];
  }
  __syncthreads();

  // combine warps, per head, write output (threads split over HPB*D outputs)
  for (int e = tid; e < HPB * D; e += NT) {
    int hh = e / D, d = e - hh * D;
    float gm = -1e30f;
#pragma unroll
    for (int w = 0; w < NW; w++) gm = fmaxf(gm, sm[w * HPB + hh]);
    float gl = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) gl += sl[w * HPB + hh] * __expf(sm[w * HPB + hh] - gm);
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) o += sacc[(w * HPB + hh) * D + d] * __expf(sm[w * HPB + hh] - gm);
    bool ok = gl > 0.f;
    int th = t * H + h0 + hh;
    out[(long)th * D + d] = __float2bfloat16(ok ? o / gl : 0.f);
    if (d == 0) lse[th] = ok ? (gm + __logf(gl)) * INV_LN2 : 0.f;
  }
}

std::vector<torch::Tensor> dsa_forward(
    torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
    torch::Tensor idx, double sm_scale) {
  int T = q_nope.size(0);
  int TOPK = idx.size(1);
  auto fopt = torch::dtype(torch::kFloat32).device(q_nope.device());
  auto out = torch::empty({T, H, D}, q_nope.options());
  auto lse = torch::empty({T, H}, fopt);
  auto stream = at::cuda::getCurrentCUDAStream();
  int smem = TOPK_MAX * 4 + NW * HPB * D * 4 + NW * HPB * 4 * 2;
  static bool set = false;
  if (!set) { cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
  dim3 grid(H / HPB, T);
  dsa_kernel<<<grid, NT, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr()),
      idx.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      lse.data_ptr<float>(), (float)sm_scale, TOPK);
  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsa_forward", &dsa_forward, "dsa forward");
}
