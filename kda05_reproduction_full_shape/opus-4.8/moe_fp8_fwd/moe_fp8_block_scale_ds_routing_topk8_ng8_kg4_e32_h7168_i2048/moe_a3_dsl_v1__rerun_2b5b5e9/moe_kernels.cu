// MoE FP8 block-scale kernels for NVIDIA B200 (sm_100a).
// Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
//
// Pipeline (all on-GPU, no host syncs):
//   1. routing_kernel       : DeepSeek-V3 no-aux routing -> per-token top-8 + weights,
//                             local-expert filtering, per-expert counts.
//   2. build_offsets_kernel : exclusive scan of counts -> row offsets + flat m-tile list.
//   3. scatter_kernel       : place (token, weight) rows grouped by local expert.
//   4. gemm1_swiglu_kernel  : grouped GEMM1 (gate & up) + SwiGLU -> C[R, 2048] (bf16).
//   5. gemm2_kernel         : grouped GEMM2 -> weighted atomic accumulate into acc[T, H] (fp32).
//   6. finalize_kernel      : cast acc -> output (bf16).
//
// GEMMs: cp.async double-buffered load of FP8 tiles -> dequant to bf16 (block-scale constant
// within each 128-K block) -> bf16 WMMA with fp32 accumulation. ~0.4% rel error << MoE tol.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <math_constants.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

#define H        7168
#define I_SIZE   2048
#define G1OUT    4096
#define E_GLOBAL 256
#define E_LOCAL  32
#define BLK      128
#define NHB      56   // H / 128
#define NIB      16   // I / 128
#define NG1B     32   // G1OUT / 128
#define TOPK     8
#define NGROUP   8
#define GROUPSZ  32   // 256 / 8
#define TOPKGROUP 4

#define BM 64
#define BN 64
#define NTHREADS 128
#define NTHREADS1 256
#define FULLMASK 0xffffffffu

__device__ __forceinline__ uint32_t ld32(const __nv_fp8_e4m3* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
// D[16x8] += A[16x32] * B[32x8]  (A row-major, B col-major), e4m3 operands, f32 accum.
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D[16x8] += A[16x16] * B[16x8]  (A row-major, B col-major), bf16 operands, f32 accum.
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t ldbf32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
// pack two fp8 (dequantized by sc) into a bf16x2 register.
__device__ __forceinline__ uint32_t packW2(const __nv_fp8_e4m3* p, float sc) {
  __nv_bfloat16 x0 = __float2bfloat16(float(p[0]) * sc);
  __nv_bfloat16 x1 = __float2bfloat16(float(p[1]) * sc);
  return (uint32_t)__bfloat16_as_ushort(x0) | ((uint32_t)__bfloat16_as_ushort(x1) << 16);
}

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem_ptr) {
  __pipeline_memcpy_async(smem_ptr, gmem_ptr, 16);
}
__device__ __forceinline__ void cp_commit() { __pipeline_commit(); }
template <int N>
__device__ __forceinline__ void cp_wait() { __pipeline_wait_prior(N); }

// ---------------------------------------------------------------------------
// 1. Routing  (one warp per token)
// ---------------------------------------------------------------------------
__global__ void routing_kernel(const float* __restrict__ logits,
                               const __nv_bfloat16* __restrict__ bias,
                               int T, int local_off, float scale,
                               int* __restrict__ sel_le, float* __restrict__ sel_w,
                               int* __restrict__ expert_count) {
  int t = blockIdx.x;
  int lane = threadIdx.x;
  if (t >= T) return;
  const float* lg = logits + (size_t)t * E_GLOBAL;

  float s[NGROUP], sb[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; g++) {
    int e = GROUPSZ * g + lane;
    float x = lg[e];
    float sig = 1.0f / (1.0f + __expf(-x));
    s[g] = sig;
    sb[g] = sig + __bfloat162float(bias[e]);
  }

  float gscore[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; g++) {
    float v = sb[g];
    float m1 = v;
    int am = lane;
    for (int o = 16; o > 0; o >>= 1) {
      float om = __shfl_xor_sync(FULLMASK, m1, o);
      int oa = __shfl_xor_sync(FULLMASK, am, o);
      if (om > m1 || (om == m1 && oa < am)) { m1 = om; am = oa; }
    }
    float v2 = (lane == am) ? -CUDART_INF_F : v;
    float m2 = v2;
    for (int o = 16; o > 0; o >>= 1) {
      float om = __shfl_xor_sync(FULLMASK, m2, o);
      if (om > m2) m2 = om;
    }
    gscore[g] = m1 + m2;
  }

  int keep[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; g++) {
    int rank = 0;
#pragma unroll
    for (int h = 0; h < NGROUP; h++)
      if (gscore[h] > gscore[g] || (gscore[h] == gscore[g] && h < g)) rank++;
    keep[g] = (rank < TOPKGROUP) ? 1 : 0;
  }
  float sbm[NGROUP];
#pragma unroll
  for (int g = 0; g < NGROUP; g++) sbm[g] = keep[g] ? sb[g] : -CUDART_INF_F;

  int sel_e[TOPK];
  float sel_s[TOPK];
  float sum_s = 0.0f;
  for (int it = 0; it < TOPK; it++) {
    float lb = -CUDART_INF_F;
    int lg = 0;
#pragma unroll
    for (int g = 0; g < NGROUP; g++)
      if (sbm[g] > lb) { lb = sbm[g]; lg = g; }
    float bestv = lb;
    int bestp = lane * NGROUP + lg;
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(FULLMASK, bestv, o);
      int op = __shfl_xor_sync(FULLMASK, bestp, o);
      if (ov > bestv || (ov == bestv && op < bestp)) { bestv = ov; bestp = op; }
    }
    int winlane = bestp / NGROUP;
    int wing = bestp % NGROUP;
    int expert = GROUPSZ * wing + winlane;
    float ws = __shfl_sync(FULLMASK, (lane == winlane) ? s[wing] : 0.0f, winlane);
    if (lane == winlane) sbm[wing] = -CUDART_INF_F;
    sel_e[it] = expert;
    sel_s[it] = ws;
    sum_s += ws;
  }

  if (lane == 0) {
    float norm = sum_s + 1e-20f;
#pragma unroll
    for (int it = 0; it < TOPK; it++) {
      int ge = sel_e[it];
      int le = ge - local_off;
      float w = sel_s[it] / norm * scale;
      if (le >= 0 && le < E_LOCAL) {
        sel_le[t * TOPK + it] = le;
        sel_w[t * TOPK + it] = w;
        atomicAdd(&expert_count[le], 1);
      } else {
        sel_le[t * TOPK + it] = -1;
        sel_w[t * TOPK + it] = 0.0f;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Offsets + m-tile list
// ---------------------------------------------------------------------------
__global__ void build_offsets_kernel(const int* __restrict__ expert_count,
                                      int* __restrict__ expert_offset,
                                      int* __restrict__ mtile_expert,
                                      int* __restrict__ mtile_mstart,
                                      int* __restrict__ mtile_msize,
                                      int* __restrict__ total_mtiles, int bm) {
  int off = 0, nt = 0;
  for (int e = 0; e < E_LOCAL; e++) {
    expert_offset[e] = off;
    int cnt = expert_count[e];
    int ntiles = (cnt + bm - 1) / bm;
    for (int mt = 0; mt < ntiles; mt++) {
      mtile_expert[nt] = e;
      mtile_mstart[nt] = mt * bm;
      int ms = cnt - mt * bm;
      mtile_msize[nt] = ms < bm ? ms : bm;
      nt++;
    }
    off += cnt;
  }
  expert_offset[E_LOCAL] = off;
  *total_mtiles = nt;
}

// ---------------------------------------------------------------------------
// 3. Scatter
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(const int* __restrict__ sel_le, const float* __restrict__ sel_w,
                               int T, const int* __restrict__ expert_offset,
                               int* __restrict__ wptr, int* __restrict__ row_token,
                               float* __restrict__ row_weight) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
#pragma unroll
  for (int it = 0; it < TOPK; it++) {
    int le = sel_le[t * TOPK + it];
    if (le >= 0) {
      int pos = expert_offset[le] + atomicAdd(&wptr[le], 1);
      row_token[pos] = t;
      row_weight[pos] = sel_w[t * TOPK + it];
    }
  }
}

// ---------------------------------------------------------------------------
// 4. GEMM1 + SwiGLU  -> C[R, I_SIZE] bf16   (cp.async double-buffered)
//    tile: BM rows x BN columns (of the up/gate half); computes up & gate, fuses SwiGLU.
// ---------------------------------------------------------------------------
// WROWS warp-rows x (8/WROWS) warp-cols, BM = 16*WROWS. WROWS=1 (BM=16) for small/mid T
// avoids wasted warp-rows; WROWS=4 (BM=64) amortizes weight reads for large prefill.
template <int WROWS, int BNT>
__global__ __launch_bounds__(NTHREADS1) void gemm1_swiglu_tmpl(
    const __nv_fp8_e4m3* __restrict__ A, const float* __restrict__ hscale,
    const __nv_fp8_e4m3* __restrict__ W, const float* __restrict__ wscale,
    const int* __restrict__ row_token, const int* __restrict__ expert_offset,
    const int* __restrict__ mtile_expert, const int* __restrict__ mtile_mstart,
    const int* __restrict__ mtile_msize, const int* __restrict__ total_mtiles_ptr,
    int* __restrict__ counter, __nv_bfloat16* __restrict__ C, int T) {
  constexpr int BM_T = WROWS * 16;
  constexpr int WCOLS = 8 / WROWS;
  constexpr int WN = BNT / WCOLS;          // cols per warp-col
  constexpr int NSUB = WN / 8;             // n-subtiles per warp
  extern __shared__ unsigned char smem[];
  auto Af = reinterpret_cast<__nv_fp8_e4m3(*)[BM_T][BLK]>(smem);
  auto Wuf = reinterpret_cast<__nv_fp8_e4m3(*)[BNT][BLK]>(smem + 2 * BM_T * BLK);
  auto Wgf = reinterpret_cast<__nv_fp8_e4m3(*)[BNT][BLK]>(smem + 2 * BM_T * BLK + 2 * BNT * BLK);
  int* rtok = reinterpret_cast<int*>(smem + 2 * BM_T * BLK + 4 * BNT * BLK);
  __shared__ int s_tile;

  const int NNT = I_SIZE / BNT;
  int total = (*total_mtiles_ptr) * NNT;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int wr = warp / WCOLS, wc = warp % WCOLS;
  int gid = lane >> 2, t4 = lane & 3;
  int wm = wr * 16, wn = wc * WN;
  int arow0 = wm + gid, arow1 = wm + gid + 8;

  while (true) {
    if (tid == 0) s_tile = atomicAdd(counter, 1);
    __syncthreads();
    int tile = s_tile;
    if (tile >= total) break;

    int mt = tile / NNT, nt = tile % NNT;
    int e = mtile_expert[mt];
    int msize = mtile_msize[mt];
    int rowbase = expert_offset[e] + mtile_mstart[mt];
    int n0 = nt * BNT;
    int nbu = n0 / BLK;
    int nbg = (I_SIZE + n0) / BLK;
    for (int r = tid; r < BM_T; r += NTHREADS1) rtok[r] = (r < msize) ? row_token[rowbase + r] : 0;
    __syncthreads();

    float cmu[NSUB][4], cmg[NSUB][4];
#pragma unroll
    for (int j = 0; j < NSUB; j++)
#pragma unroll
      for (int el = 0; el < 4; el++) { cmu[j][el] = 0.0f; cmg[j][el] = 0.0f; }

    auto issue = [&](int buf, int kb) {
      size_t ko = (size_t)kb * BLK;
      for (int c = tid; c < BM_T * 8; c += NTHREADS1) {
        int r = c >> 3, ch = c & 7;
        cp_async16(&Af[buf][r][ch * 16], A + (size_t)rtok[r] * H + ko + ch * 16);
      }
      for (int c = tid; c < BNT * 8; c += NTHREADS1) {
        int nn = c >> 3, ch = c & 7;
        cp_async16(&Wuf[buf][nn][ch * 16], W + ((size_t)e * G1OUT + n0 + nn) * H + ko + ch * 16);
        cp_async16(&Wgf[buf][nn][ch * 16], W + ((size_t)e * G1OUT + I_SIZE + n0 + nn) * H + ko + ch * 16);
      }
    };

    issue(0, 0);
    cp_commit();
    for (int kb = 0; kb < NHB; kb++) {
      int buf = kb & 1;
      if (kb + 1 < NHB) { issue((kb + 1) & 1, kb + 1); cp_commit(); cp_wait<1>(); }
      else cp_wait<0>();
      __syncthreads();

      float cpu[NSUB][4], cpg[NSUB][4];
#pragma unroll
      for (int j = 0; j < NSUB; j++)
#pragma unroll
        for (int el = 0; el < 4; el++) { cpu[j][el] = 0.0f; cpg[j][el] = 0.0f; }
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        int k0 = ks * 32;
        uint32_t a[4];
        a[0] = ld32(&Af[buf][arow0][k0 + t4 * 4]);
        a[1] = ld32(&Af[buf][arow1][k0 + t4 * 4]);
        a[2] = ld32(&Af[buf][arow0][k0 + 16 + t4 * 4]);
        a[3] = ld32(&Af[buf][arow1][k0 + 16 + t4 * 4]);
#pragma unroll
        for (int j = 0; j < NSUB; j++) {
          int ncol = wn + j * 8 + gid;
          uint32_t bu[2], bg[2];
          bu[0] = ld32(&Wuf[buf][ncol][k0 + t4 * 4]);
          bu[1] = ld32(&Wuf[buf][ncol][k0 + 16 + t4 * 4]);
          bg[0] = ld32(&Wgf[buf][ncol][k0 + t4 * 4]);
          bg[1] = ld32(&Wgf[buf][ncol][k0 + 16 + t4 * 4]);
          mma_e4m3(cpu[j], a, bu);
          mma_e4m3(cpg[j], a, bg);
        }
      }
      float as0 = (arow0 < msize) ? hscale[(size_t)kb * T + rtok[arow0]] : 0.0f;
      float as1 = (arow1 < msize) ? hscale[(size_t)kb * T + rtok[arow1]] : 0.0f;
      float wsu = wscale[((size_t)e * NG1B + nbu) * NHB + kb];
      float wsg = wscale[((size_t)e * NG1B + nbg) * NHB + kb];
#pragma unroll
      for (int j = 0; j < NSUB; j++) {
        cmu[j][0] += cpu[j][0] * as0 * wsu; cmu[j][1] += cpu[j][1] * as0 * wsu;
        cmu[j][2] += cpu[j][2] * as1 * wsu; cmu[j][3] += cpu[j][3] * as1 * wsu;
        cmg[j][0] += cpg[j][0] * as0 * wsg; cmg[j][1] += cpg[j][1] * as0 * wsg;
        cmg[j][2] += cpg[j][2] * as1 * wsg; cmg[j][3] += cpg[j][3] * as1 * wsg;
      }
      __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NSUB; j++) {
      int col = n0 + wn + j * 8 + t4 * 2;
      if (arow0 < msize) {
        C[(size_t)(rowbase + arow0) * I_SIZE + col] =
            __float2bfloat16((cmg[j][0] / (1.0f + __expf(-cmg[j][0]))) * cmu[j][0]);
        C[(size_t)(rowbase + arow0) * I_SIZE + col + 1] =
            __float2bfloat16((cmg[j][1] / (1.0f + __expf(-cmg[j][1]))) * cmu[j][1]);
      }
      if (arow1 < msize) {
        C[(size_t)(rowbase + arow1) * I_SIZE + col] =
            __float2bfloat16((cmg[j][2] / (1.0f + __expf(-cmg[j][2]))) * cmu[j][2]);
        C[(size_t)(rowbase + arow1) * I_SIZE + col + 1] =
            __float2bfloat16((cmg[j][3] / (1.0f + __expf(-cmg[j][3]))) * cmu[j][3]);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// 5. GEMM2  -> weighted atomic accumulate into acc[T, H]   (cp.async double-buffered)
// ---------------------------------------------------------------------------
// Quantize C[R, I_SIZE] bf16 -> Cq fp8 (e4m3) + per-(row, 128-block) scale cscale[R, NIB].
__global__ void quantize_c_kernel(const __nv_bfloat16* __restrict__ C,
                                  __nv_fp8_e4m3* __restrict__ Cq, float* __restrict__ cscale,
                                  const int* __restrict__ expert_offset) {
  int R = expert_offset[E_LOCAL];
  int total = R * NIB;
  int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (int u = gwarp; u < total; u += nwarps) {
    int row = u / NIB, ib = u % NIB;
    size_t base = (size_t)row * I_SIZE + (size_t)ib * BLK;
    float v[4], amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      v[i] = __bfloat162float(C[base + lane * 4 + i]);
      amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULLMASK, amax, o));
    float scale = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
    float inv = 1.0f / scale;
#pragma unroll
    for (int i = 0; i < 4; i++) Cq[base + lane * 4 + i] = __nv_fp8_e4m3(v[i] * inv);
    if (lane == 0) cscale[row * NIB + ib] = scale;
  }
}

// GEMM2 as fp8 mma.sync m16n8k32 + per-128-K promotion. A = Cq (fp8, scale cscale[row][kb]),
// B = W2 (fp8, scale w2scale[e][hblk][kb]). Weighted atomic-accumulate into acc[T, H].
template <int WROWS, int BNT>
__global__ __launch_bounds__(NTHREADS1) void gemm2_fp8_tmpl(
    const __nv_fp8_e4m3* __restrict__ Cq, const float* __restrict__ cscale,
    const __nv_fp8_e4m3* __restrict__ W2, const float* __restrict__ w2scale,
    const int* __restrict__ row_token, const float* __restrict__ row_weight,
    const int* __restrict__ expert_offset, const int* __restrict__ mtile_expert,
    const int* __restrict__ mtile_mstart, const int* __restrict__ mtile_msize,
    const int* __restrict__ total_mtiles_ptr, int* __restrict__ counter,
    float* __restrict__ acc, int T) {
  constexpr int BM_T = WROWS * 16;
  constexpr int WCOLS = 8 / WROWS;
  constexpr int WN = BNT / WCOLS;
  constexpr int NSUB = WN / 8;
  extern __shared__ unsigned char smem[];
  auto Af = reinterpret_cast<__nv_fp8_e4m3(*)[BM_T][BLK]>(smem);
  auto Wf = reinterpret_cast<__nv_fp8_e4m3(*)[BNT][BLK]>(smem + 2 * BM_T * BLK);
  int* rtok = reinterpret_cast<int*>(smem + 2 * BM_T * BLK + 2 * BNT * BLK);
  float* rwt = reinterpret_cast<float*>(smem + 2 * BM_T * BLK + 2 * BNT * BLK + BM_T * 4);
  __shared__ int s_tile;

  const int NNT = H / BNT;
  int total = (*total_mtiles_ptr) * NNT;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int wr = warp / WCOLS, wc = warp % WCOLS;
  int gid = lane >> 2, t4 = lane & 3;
  int wm = wr * 16, wn = wc * WN;
  int arow0 = wm + gid, arow1 = wm + gid + 8;

  while (true) {
    if (tid == 0) s_tile = atomicAdd(counter, 1);
    __syncthreads();
    int tile = s_tile;
    if (tile >= total) break;

    int mt = tile / NNT, nt = tile % NNT;
    int e = mtile_expert[mt];
    int msize = mtile_msize[mt];
    int rowbase = expert_offset[e] + mtile_mstart[mt];
    int n0 = nt * BNT;
    int hb = n0 / BLK;
    for (int r = tid; r < BM_T; r += NTHREADS1) {
      rtok[r] = (r < msize) ? row_token[rowbase + r] : 0;
      rwt[r] = (r < msize) ? row_weight[rowbase + r] : 0.0f;
    }
    __syncthreads();

    float cm[NSUB][4];
#pragma unroll
    for (int j = 0; j < NSUB; j++)
#pragma unroll
      for (int el = 0; el < 4; el++) cm[j][el] = 0.0f;

    auto issue = [&](int buf, int kb) {
      size_t ko = (size_t)kb * BLK;
      for (int c = tid; c < BM_T * 8; c += NTHREADS1) {
        int r = c >> 3, ch = c & 7;
        cp_async16(&Af[buf][r][ch * 16],
                   Cq + (size_t)(rowbase + (r < msize ? r : 0)) * I_SIZE + ko + ch * 16);
      }
      for (int c = tid; c < BNT * 8; c += NTHREADS1) {
        int nn = c >> 3, ch = c & 7;
        cp_async16(&Wf[buf][nn][ch * 16], W2 + ((size_t)e * H + n0 + nn) * I_SIZE + ko + ch * 16);
      }
    };

    issue(0, 0);
    cp_commit();
    for (int kb = 0; kb < NIB; kb++) {
      int buf = kb & 1;
      if (kb + 1 < NIB) { issue((kb + 1) & 1, kb + 1); cp_commit(); cp_wait<1>(); }
      else cp_wait<0>();
      __syncthreads();

      float cp[NSUB][4];
#pragma unroll
      for (int j = 0; j < NSUB; j++)
#pragma unroll
        for (int el = 0; el < 4; el++) cp[j][el] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        int k0 = ks * 32;
        uint32_t a[4];
        a[0] = ld32(&Af[buf][arow0][k0 + t4 * 4]);
        a[1] = ld32(&Af[buf][arow1][k0 + t4 * 4]);
        a[2] = ld32(&Af[buf][arow0][k0 + 16 + t4 * 4]);
        a[3] = ld32(&Af[buf][arow1][k0 + 16 + t4 * 4]);
#pragma unroll
        for (int j = 0; j < NSUB; j++) {
          int ncol = wn + j * 8 + gid;
          uint32_t b[2];
          b[0] = ld32(&Wf[buf][ncol][k0 + t4 * 4]);
          b[1] = ld32(&Wf[buf][ncol][k0 + 16 + t4 * 4]);
          mma_e4m3(cp[j], a, b);
        }
      }
      float as0 = (arow0 < msize) ? cscale[(rowbase + arow0) * NIB + kb] : 0.0f;
      float as1 = (arow1 < msize) ? cscale[(rowbase + arow1) * NIB + kb] : 0.0f;
      float wsc = w2scale[((size_t)e * NHB + hb) * NIB + kb];
#pragma unroll
      for (int j = 0; j < NSUB; j++) {
        cm[j][0] += cp[j][0] * as0 * wsc; cm[j][1] += cp[j][1] * as0 * wsc;
        cm[j][2] += cp[j][2] * as1 * wsc; cm[j][3] += cp[j][3] * as1 * wsc;
      }
      __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NSUB; j++) {
      int col = n0 + wn + j * 8 + t4 * 2;
      if (arow0 < msize) {
        atomicAdd(&acc[(size_t)rtok[arow0] * H + col], cm[j][0] * rwt[arow0]);
        atomicAdd(&acc[(size_t)rtok[arow0] * H + col + 1], cm[j][1] * rwt[arow0]);
      }
      if (arow1 < msize) {
        atomicAdd(&acc[(size_t)rtok[arow1] * H + col], cm[j][2] * rwt[arow1]);
        atomicAdd(&acc[(size_t)rtok[arow1] * H + col + 1], cm[j][3] * rwt[arow1]);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// 6. Finalize
// ---------------------------------------------------------------------------
__global__ void finalize_kernel(const float* __restrict__ acc, __nv_bfloat16* __restrict__ out,
                                size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = __float2bfloat16(acc[idx]);
}

// ===========================================================================
// A3 path: C++ routing/scatter/gather/swiglu-quant/scatter-back around the
// CuTe DSL tcgen05 grouped GEMMs (driven from main.py). Per-expert rows are
// padded to a multiple of 128 (contiguous grouped-GEMM requirement).
// ===========================================================================

// Padded exclusive scan: pexpert_offset[e] = sum_{e'<e} ceil(count[e']/128)*128.
__global__ void build_padded_offsets_kernel(const int* __restrict__ expert_count,
                                            int* __restrict__ pexpert_offset,
                                            int* __restrict__ vmpad) {
  int off = 0;
  for (int e = 0; e < E_LOCAL; e++) {
    pexpert_offset[e] = off;
    int c = expert_count[e];
    off += ((c + BLK - 1) / BLK) * BLK;
  }
  pexpert_offset[E_LOCAL] = off;
  *vmpad = off;
}

// Fill gidx for every padded row and pre-mark all rows as pad (token=-1, w=0).
__global__ void fill_gidx_kernel(const int* __restrict__ pexpert_offset,
                                 int* __restrict__ gidx, int* __restrict__ prow_token,
                                 float* __restrict__ prow_weight) {
  int e = blockIdx.x;  // E_LOCAL blocks
  int s = pexpert_offset[e], en = pexpert_offset[e + 1];
  for (int r = s + threadIdx.x; r < en; r += blockDim.x) {
    gidx[r] = e;
    prow_token[r] = -1;
    prow_weight[r] = 0.0f;
  }
}

// Scatter (token,weight) into each local expert's padded block.
__global__ void scatter_padded_kernel(const int* __restrict__ sel_le,
                                      const float* __restrict__ sel_w, int T,
                                      const int* __restrict__ pexpert_offset,
                                      int* __restrict__ wptr, int* __restrict__ prow_token,
                                      float* __restrict__ prow_weight,
                                      int* __restrict__ sel_prow) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
#pragma unroll
  for (int it = 0; it < TOPK; it++) {
    int le = sel_le[t * TOPK + it];
    if (le >= 0) {
      int base = pexpert_offset[le];
      int cap = pexpert_offset[le + 1] - base;   // per-expert block size (max_m)
      int rank = atomicAdd(&wptr[le], 1);
      if (rank < cap) {                           // guard: never overflow into next block
        int pos = base + rank;
        prow_token[pos] = t;
        prow_weight[pos] = sel_w[t * TOPK + it];
        // forward map (token,slot) -> padded row, for the token-indexed combine.
        if (sel_prow) sel_prow[t * TOPK + it] = pos;
      }
    }
  }
}

// Gather hidden rows + per-row hidden scale into dense padded buffers for TMA.
// Apad[r, :H] = hidden[token, :H] (0 if pad);  sfa_pool[kb, r] = hscale[kb, token].
// (Tried skipping the pad-row zero-writes 2026-10-22 — proven correct but ZERO measured
// wall-clock benefit on real data: the zero-writes are hidden under the weight-bandwidth-
// bound GEMMs / not on the graph critical path. Kept the explicit zeroing for safety.)
__global__ void gather_a_kernel(const __nv_fp8_e4m3* __restrict__ A,
                                const float* __restrict__ hscale,
                                const int* __restrict__ prow_token,
                                const int* __restrict__ vmpad_ptr, int T,
                                __nv_fp8_e4m3* __restrict__ Apad,
                                float* __restrict__ sfa_pool, int VM_cap,
                                int skip_pad) {
  int VM = *vmpad_ptr;
  for (int r = blockIdx.x; r < VM; r += gridDim.x) {
    int tok = prow_token[r];
    // skip_pad: masked grouped GEMM never consumes pad rows beyond the last active
    // 128-tile, and swiglu_quant_padded zeros Cq for every pad row by position, so pad
    // Apad/sfa are never read into the output -> leave them stale (buffer is zero-inited
    // once at alloc so the straddle pad rows in the last active tile stay finite).
    if (tok < 0 && skip_pad) continue;
    const uint4* src = (tok >= 0) ? reinterpret_cast<const uint4*>(A + (size_t)tok * H) : nullptr;
    uint4* dst = reinterpret_cast<uint4*>(Apad + (size_t)r * H);
    uint4 z = make_uint4(0, 0, 0, 0);
    for (int c = threadIdx.x; c < H / 16; c += blockDim.x) dst[c] = (tok >= 0) ? src[c] : z;
    for (int kb = threadIdx.x; kb < NHB; kb += blockDim.x)
      sfa_pool[(size_t)kb * VM_cap + r] = (tok >= 0) ? hscale[(size_t)kb * T + tok] : 0.0f;
  }
}

// SwiGLU over raw GEMM1 output G1[VM, G1OUT] then per-128-block fp8 quantize.
// out[i] = silu(G1[row, I_SIZE+i]) * G1[row, i];  cscale2 is [NIB, VM_cap] (vm-contiguous).
// expert_count/block_size (A3s static-block layout only; nullptr disables): skip pad
// rows (row%block_size >= expert_count[row/block_size]) -> write Cq=0 (matches the
// G1==0 pad behaviour) without the G1 reads+SwiGLU. Big win at small/mid T where most
// of the VM=E_LOCAL*128 rows are pad.
__global__ void swiglu_quant_padded_kernel(const __nv_bfloat16* __restrict__ G1,
                                           __nv_fp8_e4m3* __restrict__ Cq,
                                           float* __restrict__ cscale2, int VM, int VM_cap,
                                           const int* __restrict__ expert_count, int block_size) {
  int total = VM * NIB;
  int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  int lane = threadIdx.x & 31;
  for (int u = gwarp; u < total; u += nwarps) {
    int row = u / NIB, ib = u % NIB;
    if (expert_count && block_size > 0 && (row % block_size) >= expert_count[row / block_size]) {
      size_t qb = (size_t)row * I_SIZE + ib * BLK;
#pragma unroll
      for (int i = 0; i < 4; i++) Cq[qb + lane * 4 + i] = __nv_fp8_e4m3(0.0f);
      if (lane == 0) cscale2[(size_t)ib * VM_cap + row] = 1.0f;
      continue;
    }
    size_t gbase = (size_t)row * G1OUT;
    int col0 = ib * BLK;
    float v[4], amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int ic = col0 + lane * 4 + i;
      float gate = __bfloat162float(G1[gbase + ic]);
      float up = __bfloat162float(G1[gbase + I_SIZE + ic]);
      float o = (up / (1.0f + __expf(-up))) * gate;
      v[i] = o;
      amax = fmaxf(amax, fabsf(o));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULLMASK, amax, o));
    float scale = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
    float inv = 1.0f / scale;
    size_t qbase = (size_t)row * I_SIZE + col0;
#pragma unroll
    for (int i = 0; i < 4; i++) Cq[qbase + lane * 4 + i] = __nv_fp8_e4m3(v[i] * inv);
    if (lane == 0) cscale2[(size_t)ib * VM_cap + row] = scale;
  }
}

// Weighted scatter-accumulate GEMM2 output into acc[T, H] (fp32, atomic).
__global__ void scatter_back_kernel(const __nv_bfloat16* __restrict__ C2,
                                    const int* __restrict__ prow_token,
                                    const float* __restrict__ prow_weight, int VM,
                                    float* __restrict__ acc) {
  for (int row = blockIdx.x; row < VM; row += gridDim.x) {
    int tok = prow_token[row];
    if (tok < 0) continue;
    float w = prow_weight[row];
    const __nv_bfloat16* c2 = C2 + (size_t)row * H;
    float* a = acc + (size_t)tok * H;
    for (int h = threadIdx.x; h < H; h += blockDim.x)
      atomicAdd(&a[h], w * __bfloat162float(c2[h]));
  }
}

// Token-indexed combine: one block per output token gathers its (<=TOPK) local
// expert rows from C2[padded_row, :H], weight-sums them, and writes output[tok, :H]
// once as bf16. No atomics, no fp32 acc buffer, no separate finalize pass -- each
// C2 row is read exactly once and the output is written once (pure bandwidth).
// sel_prow[t*TOPK+k] = padded row for token t's k-th selection (-1 if not local /
// dropped); sel_w[t*TOPK+k] = its routing weight. H is a multiple of 8 so we move
// 8 bf16 (one uint4) per thread per column-group.
__global__ void combine_kernel(const __nv_bfloat16* __restrict__ C2,
                               const int* __restrict__ sel_prow,
                               const float* __restrict__ sel_w, int T,
                               __nv_bfloat16* __restrict__ out) {
  __shared__ int s_prow[TOPK];
  __shared__ float s_w[TOPK];
  const int HV = H / 8;
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    if (threadIdx.x < TOPK) {
      s_prow[threadIdx.x] = sel_prow[t * TOPK + threadIdx.x];
      s_w[threadIdx.x] = sel_w[t * TOPK + threadIdx.x];
    }
    __syncthreads();
    for (int hv = threadIdx.x; hv < HV; hv += blockDim.x) {
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = 0.0f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        int pr = s_prow[k];
        if (pr >= 0) {
          float w = s_w[k];
          uint4 raw = reinterpret_cast<const uint4*>(C2 + (size_t)pr * H)[hv];
          const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
          for (int i = 0; i < 8; i++) acc[i] += w * __bfloat162float(bp[i]);
        }
      }
      uint4 outraw;
      __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(&outraw);
#pragma unroll
      for (int i = 0; i < 8; i++) op[i] = __float2bfloat16(acc[i]);
      reinterpret_cast<uint4*>(out + (size_t)t * H)[hv] = outraw;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void moe_forward(torch::Tensor routing_logits, torch::Tensor routing_bias,
                 torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                 torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                 torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
                 int64_t local_expert_offset, double routed_scaling_factor,
                 torch::Tensor sel_le, torch::Tensor sel_w, torch::Tensor expert_count,
                 torch::Tensor expert_offset, torch::Tensor wptr, torch::Tensor mtile_expert,
                 torch::Tensor mtile_mstart, torch::Tensor mtile_msize, torch::Tensor total_mtiles,
                 torch::Tensor counter, torch::Tensor row_token, torch::Tensor row_weight,
                 torch::Tensor C, torch::Tensor Cq, torch::Tensor cscale, torch::Tensor acc,
                 torch::Tensor output, int64_t T) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const float* logits = routing_logits.data_ptr<float>();
  const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const __nv_fp8_e4m3* A = reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const float* hscale = hidden_states_scale.data_ptr<float>();
  const __nv_fp8_e4m3* W1 = reinterpret_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr());
  const float* w1scale = gemm1_weights_scale.data_ptr<float>();
  const __nv_fp8_e4m3* W2 = reinterpret_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr());
  const float* w2scale = gemm2_weights_scale.data_ptr<float>();

  int* d_sel_le = sel_le.data_ptr<int>();
  float* d_sel_w = sel_w.data_ptr<float>();
  int* d_expert_count = expert_count.data_ptr<int>();
  int* d_expert_offset = expert_offset.data_ptr<int>();
  int* d_wptr = wptr.data_ptr<int>();
  int* d_mtile_expert = mtile_expert.data_ptr<int>();
  int* d_mtile_mstart = mtile_mstart.data_ptr<int>();
  int* d_mtile_msize = mtile_msize.data_ptr<int>();
  int* d_total_mtiles = total_mtiles.data_ptr<int>();
  int* d_counter = counter.data_ptr<int>();
  int* d_row_token = row_token.data_ptr<int>();
  float* d_row_weight = row_weight.data_ptr<float>();
  __nv_bfloat16* d_C = reinterpret_cast<__nv_bfloat16*>(C.data_ptr());
  __nv_fp8_e4m3* d_Cq = reinterpret_cast<__nv_fp8_e4m3*>(Cq.data_ptr());
  float* d_cscale = cscale.data_ptr<float>();
  float* d_acc = acc.data_ptr<float>();
  __nv_bfloat16* d_out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

  cudaMemsetAsync(d_expert_count, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_wptr, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_acc, 0, (size_t)T * H * sizeof(float), stream);

  // Shape-specialized dispatch by seq_len -> warp-rows (BM=16*WROWS):
  //   T<=128    -> WROWS=1 (BM=16): tiny per-expert M, no wasted warp-rows, high occupancy
  //   128<T<=1024 -> WROWS=2 (BM=32): mid prefill, ~tens of rows/expert
  //   T>1024    -> WROWS=4 (BM=64): large prefill, amortize weight reads
  bool small = (T <= 128);
  int bm = small ? 16 : 64;

  routing_kernel<<<T, 32, 0, stream>>>(logits, bias, (int)T, (int)local_expert_offset,
                                        (float)routed_scaling_factor, d_sel_le, d_sel_w,
                                        d_expert_count);
  build_offsets_kernel<<<1, 1, 0, stream>>>(d_expert_count, d_expert_offset, d_mtile_expert,
                                            d_mtile_mstart, d_mtile_msize, d_total_mtiles, bm);
  scatter_kernel<<<(T + 127) / 128, 128, 0, stream>>>(d_sel_le, d_sel_w, (int)T, d_expert_offset,
                                                      d_wptr, d_row_token, d_row_weight);

  // Mid prefill benefits from ~4 blocks/SM (more in-flight weight loads); tiny T has too few
  // tiles so extra blocks just add atomic-counter contention.
  const int GRID = (T > 16 && T <= 128) ? 592 : 304;
  // small/mid (T<=128): BM=16, BN=128 (NSUB=2 -> better smem-load:mma ratio);
  // large: BM=64, BN=64.
  const int SMEM1_S = 69696, SMEM1_L = 49408;
  const int SMEM2_S = 36992, SMEM2_L = 33280;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gemm1_swiglu_tmpl<1, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM1_S);
    cudaFuncSetAttribute(gemm1_swiglu_tmpl<4, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM1_L);
    cudaFuncSetAttribute(gemm2_fp8_tmpl<1, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2_S);
    cudaFuncSetAttribute(gemm2_fp8_tmpl<4, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2_L);
    attr_set = true;
  }
  cudaMemsetAsync(d_counter, 0, sizeof(int), stream);
  if (small)
    gemm1_swiglu_tmpl<1, 128><<<GRID, NTHREADS1, SMEM1_S, stream>>>(
        A, hscale, W1, w1scale, d_row_token, d_expert_offset, d_mtile_expert, d_mtile_mstart,
        d_mtile_msize, d_total_mtiles, d_counter, d_C, (int)T);
  else
    gemm1_swiglu_tmpl<4, 64><<<GRID, NTHREADS1, SMEM1_L, stream>>>(
        A, hscale, W1, w1scale, d_row_token, d_expert_offset, d_mtile_expert, d_mtile_mstart,
        d_mtile_msize, d_total_mtiles, d_counter, d_C, (int)T);
  quantize_c_kernel<<<GRID, 256, 0, stream>>>(d_C, d_Cq, d_cscale, d_expert_offset);
  cudaMemsetAsync(d_counter, 0, sizeof(int), stream);
  if (small)
    gemm2_fp8_tmpl<1, 128><<<GRID, NTHREADS1, SMEM2_S, stream>>>(
        d_Cq, d_cscale, W2, w2scale, d_row_token, d_row_weight, d_expert_offset, d_mtile_expert,
        d_mtile_mstart, d_mtile_msize, d_total_mtiles, d_counter, d_acc, (int)T);
  else
    gemm2_fp8_tmpl<4, 64><<<GRID, NTHREADS1, SMEM2_L, stream>>>(
        d_Cq, d_cscale, W2, w2scale, d_row_token, d_row_weight, d_expert_offset, d_mtile_expert,
        d_mtile_mstart, d_mtile_msize, d_total_mtiles, d_counter, d_acc, (int)T);

  size_t nout = (size_t)T * H;
  finalize_kernel<<<(nout + 255) / 256, 256, 0, stream>>>(d_acc, d_out, nout);
}

// ---------------------------------------------------------------------------
// A3 host entry points (DSL GEMMs run between these from main.py).
// ---------------------------------------------------------------------------
// Stage 1: routing -> padded offsets -> scatter -> gather dense Apad + sfa_pool.
void moe_a3_stage1(torch::Tensor routing_logits, torch::Tensor routing_bias,
                   torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                   int64_t local_expert_offset, double routed_scaling_factor, int64_t T,
                   torch::Tensor sel_le, torch::Tensor sel_w, torch::Tensor expert_count,
                   torch::Tensor pexpert_offset, torch::Tensor wptr, torch::Tensor gidx,
                   torch::Tensor prow_token, torch::Tensor prow_weight, torch::Tensor vmpad,
                   torch::Tensor Apad, torch::Tensor sfa_pool, int64_t VM_cap,
                   torch::Tensor sel_prow) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const float* logits = routing_logits.data_ptr<float>();
  const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const __nv_fp8_e4m3* A = reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const float* hscale = hidden_states_scale.data_ptr<float>();
  int* d_sel_le = sel_le.data_ptr<int>();
  float* d_sel_w = sel_w.data_ptr<float>();
  int* d_expert_count = expert_count.data_ptr<int>();
  int* d_pexpert_offset = pexpert_offset.data_ptr<int>();
  int* d_wptr = wptr.data_ptr<int>();
  int* d_gidx = gidx.data_ptr<int>();
  int* d_prow_token = prow_token.data_ptr<int>();
  float* d_prow_weight = prow_weight.data_ptr<float>();
  int* d_vmpad = vmpad.data_ptr<int>();
  __nv_fp8_e4m3* d_Apad = reinterpret_cast<__nv_fp8_e4m3*>(Apad.data_ptr());
  float* d_sfa = sfa_pool.data_ptr<float>();

  int* d_sel_prow = sel_prow.data_ptr<int>();

  cudaMemsetAsync(d_expert_count, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_wptr, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_sel_prow, 0xFF, (size_t)T * TOPK * sizeof(int), stream);  // -1 = none
  routing_kernel<<<T, 32, 0, stream>>>(logits, bias, (int)T, (int)local_expert_offset,
                                        (float)routed_scaling_factor, d_sel_le, d_sel_w,
                                        d_expert_count);
  build_padded_offsets_kernel<<<1, 1, 0, stream>>>(d_expert_count, d_pexpert_offset, d_vmpad);
  fill_gidx_kernel<<<E_LOCAL, 128, 0, stream>>>(d_pexpert_offset, d_gidx, d_prow_token,
                                                d_prow_weight);
  scatter_padded_kernel<<<(T + 127) / 128, 128, 0, stream>>>(
      d_sel_le, d_sel_w, (int)T, d_pexpert_offset, d_wptr, d_prow_token, d_prow_weight,
      d_sel_prow);
  gather_a_kernel<<<1024, 256, 0, stream>>>(A, hscale, d_prow_token, d_vmpad, (int)T, d_Apad,
                                            d_sfa, (int)VM_cap, /*skip_pad=*/0);
}

// Stage 2: SwiGLU(raw GEMM1 out) + per-128 fp8 quantize -> Cq, cscale2.
// expert_count (optional) + block_size: when set (A3s static-block layout), pad rows
// are skipped (Cq=0) instead of recomputed; pass an undefined tensor to disable.
void moe_a3_swiglu_quant(torch::Tensor G1, torch::Tensor Cq, torch::Tensor cscale2,
                         int64_t vm, int64_t VM_cap, torch::Tensor expert_count,
                         int64_t block_size) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* d_G1 = reinterpret_cast<const __nv_bfloat16*>(G1.data_ptr());
  __nv_fp8_e4m3* d_Cq = reinterpret_cast<__nv_fp8_e4m3*>(Cq.data_ptr());
  float* d_cscale2 = cscale2.data_ptr<float>();
  const int* d_ec = expert_count.defined() ? expert_count.data_ptr<int>() : nullptr;
  int blocks = (int)((vm * NIB + 7) / 8);  // 8 warps/block -> one warp per (row,ib)
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  swiglu_quant_padded_kernel<<<blocks, 256, 0, stream>>>(d_G1, d_Cq, d_cscale2, (int)vm,
                                                         (int)VM_cap, d_ec, (int)block_size);
}

// Stage 3: weighted scatter-back GEMM2 out -> acc(fp32) -> output(bf16).
// Stage 3: token-indexed weighted combine of GEMM2 output -> output (bf16).
// sel_prow[T,TOPK] = padded row per (token,slot); sel_w[T,TOPK] = routing weight.
void moe_a3_scatter_back(torch::Tensor C2, torch::Tensor sel_prow, torch::Tensor sel_w,
                         torch::Tensor output, int64_t T) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* d_C2 = reinterpret_cast<const __nv_bfloat16*>(C2.data_ptr());
  const int* d_sel_prow = sel_prow.data_ptr<int>();
  const float* d_sel_w = sel_w.data_ptr<float>();
  __nv_bfloat16* d_out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  int blocks = (int)T;
  if (blocks < 1) blocks = 1;
  if (blocks > 32768) blocks = 32768;
  combine_kernel<<<blocks, 256, 0, stream>>>(d_C2, d_sel_prow, d_sel_w, (int)T, d_out);
}

// ---------------------------------------------------------------------------
// A3s (static) path: each local expert gets a FIXED 128-row block, so the padded
// row count vm = E_LOCAL*128 is a compile-time constant (no D2H sync). pexpert_offset
// (= e*128) and gidx (= row/128) are static and pre-filled once in the workspace;
// this stage only resets the per-call state and scatters/gathers. Fully
// CUDA-graph-capturable -> the whole small/mid-T pipeline replays as one launch,
// collapsing inter-kernel bubbles. Valid for T<=128 (no expert can exceed 128 rows).
void moe_a3s_stage1(torch::Tensor routing_logits, torch::Tensor routing_bias,
                    torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                    int64_t local_expert_offset, double routed_scaling_factor, int64_t T,
                    torch::Tensor sel_le, torch::Tensor sel_w, torch::Tensor expert_count,
                    torch::Tensor pexpert_offset, torch::Tensor wptr, torch::Tensor vmfix,
                    torch::Tensor prow_token, torch::Tensor prow_weight,
                    torch::Tensor Apad, torch::Tensor sfa_pool, int64_t VM_fixed, int64_t VM_cap,
                    torch::Tensor sel_prow, int64_t skip_pad) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const float* logits = routing_logits.data_ptr<float>();
  const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const __nv_fp8_e4m3* A = reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const float* hscale = hidden_states_scale.data_ptr<float>();
  int* d_sel_le = sel_le.data_ptr<int>();
  float* d_sel_w = sel_w.data_ptr<float>();
  int* d_expert_count = expert_count.data_ptr<int>();
  int* d_pexpert_offset = pexpert_offset.data_ptr<int>();  // static [0,128,..,VM_fixed]
  int* d_wptr = wptr.data_ptr<int>();
  int* d_vmfix = vmfix.data_ptr<int>();                    // static == VM_fixed
  int* d_prow_token = prow_token.data_ptr<int>();
  float* d_prow_weight = prow_weight.data_ptr<float>();
  __nv_fp8_e4m3* d_Apad = reinterpret_cast<__nv_fp8_e4m3*>(Apad.data_ptr());
  float* d_sfa = sfa_pool.data_ptr<float>();

  int* d_sel_prow = sel_prow.data_ptr<int>();

  cudaMemsetAsync(d_expert_count, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_wptr, 0, E_LOCAL * sizeof(int), stream);
  cudaMemsetAsync(d_prow_token, 0xFF, (size_t)VM_fixed * sizeof(int), stream);  // -1 = pad
  cudaMemsetAsync(d_prow_weight, 0, (size_t)VM_fixed * sizeof(float), stream);
  cudaMemsetAsync(d_sel_prow, 0xFF, (size_t)T * TOPK * sizeof(int), stream);  // -1 = none
  routing_kernel<<<T, 32, 0, stream>>>(logits, bias, (int)T, (int)local_expert_offset,
                                        (float)routed_scaling_factor, d_sel_le, d_sel_w,
                                        d_expert_count);
  scatter_padded_kernel<<<(T + 127) / 128, 128, 0, stream>>>(
      d_sel_le, d_sel_w, (int)T, d_pexpert_offset, d_wptr, d_prow_token, d_prow_weight,
      d_sel_prow);
  gather_a_kernel<<<1024, 256, 0, stream>>>(A, hscale, d_prow_token, d_vmfix, (int)T, d_Apad,
                                            d_sfa, (int)VM_cap, (int)skip_pad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_forward", &moe_forward, "MoE FP8 block-scale forward (v6 mma.sync path)");
  m.def("moe_a3_stage1", &moe_a3_stage1, "A3 stage1: route/scatter/gather");
  m.def("moe_a3s_stage1", &moe_a3s_stage1, "A3s static stage1: route/scatter/gather (fixed 128/expert)");
  m.def("moe_a3_swiglu_quant", &moe_a3_swiglu_quant, "A3 stage2: SwiGLU + fp8 quantize");
  m.def("moe_a3_scatter_back", &moe_a3_scatter_back, "A3 stage3: weighted scatter-back + finalize");
}
