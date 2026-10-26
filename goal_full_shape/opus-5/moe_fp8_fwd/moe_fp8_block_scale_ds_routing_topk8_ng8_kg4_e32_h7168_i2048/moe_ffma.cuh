// FFMA (CUDA-core) MoE GEMM path.
//
// On sm_100 `mma.sync ... e4m3` is emulated (ptxas expands it into
// F2FP e4m3->f16 conversions plus HMMA), which costs several times more than
// the real work for the skinny GEMMs this operator produces.  For every
// workload where (expert,token) pairs <= ~4.9x active experts - which is every
// seq_len up to 80 here - plain CUDA-core FMAs are memory bound instead, carry
// no m-padding at all, and need no shared-memory staging or barriers, so the
// kernel can run at full occupancy and reach the streaming bandwidth ceiling.
//
// One warp owns one (active expert, output column) pair and walks the whole
// reduction dimension; the 32 lanes split k into 16-byte chunks, so every
// weight load is a fully coalesced 512-byte warp transaction.  The 16 products
// of a chunk are accumulated with f16x2 FMAs (fp8 -> f16 is exact and the
// weight side is pre-scaled by 2^-6, so a 16-term f16 sum provably cannot
// overflow: 448 * 7 * 16 = 50176 < 65504), then promoted to fp32 and scaled by
// the block scales.  Lane partials are combined with a warp butterfly.

#define FFMA_WSCALE 0.015625f   // 2^-6, exact in fp16
#define FFMA_WUNSCALE 64.0f

__device__ __forceinline__ void fp8x16_to_h2(const int4 &v, __half2 (&h)[8]) {
  const uint32_t *u = reinterpret_cast<const uint32_t *>(&v);
#pragma unroll
  for (int q = 0; q < 4; q++) {
    uint32_t w = u[q];
    h[q * 2 + 0] = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(w & 0xffffu), __NV_E4M3);
    h[q * 2 + 1] = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(w >> 16), __NV_E4M3);
  }
}

__device__ __forceinline__ float h2_hsum(const __half2 &x) {
  float2 f = __half22float2(x);
  return f.x + f.y;
}

// Build the list of active local experts plus their packed slot bases.
struct ActList {
  int nact;
};

__device__ __forceinline__ int build_act(const int *__restrict__ cnt, int EL, int tid,
                                         int *act, int *acnt, int *abase) {
  __shared__ int s_nact;
  if (tid == 0) {
    int n = 0, b = 0;
    for (int e = 0; e < EL; e++) {
      int c = cnt[e];
      if (c > 0) {
        act[n] = e;
        acnt[n] = c;
        abase[n] = b;
        b += c;
        n++;
      }
    }
    s_nact = n;
  }
  __syncthreads();
  return s_nact;
}

// ---------------------------------------------------------- gemm1 (ffma) ---
template <int MAXC, int WPB>
__global__ __launch_bounds__(32 * WPB) void gemm1_ffma_kernel(
    const __nv_fp8_e4m3 *__restrict__ A, const __nv_fp8_e4m3 *__restrict__ W13,
    const float *__restrict__ S13, float *__restrict__ Cf,
    const int *__restrict__ cnt, const int *__restrict__ tokids,
    const float *__restrict__ ascg, __nv_bfloat16 *__restrict__ out, int T, int TP,
    long long out_numel, int EL, int NCTA) {
  __shared__ int act[32], acnt[32], abase[32];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  {  // zero the output; ordered before gemm2 by the kernel boundary
    long long n4 = out_numel >> 3;
    for (long long x = (long long)blockIdx.x * (32 * WPB) + tid; x < n4;
         x += (long long)NCTA * (32 * WPB))
      reinterpret_cast<int4 *>(out)[x] = make_int4(0, 0, 0, 0);
  }

  const int nact = build_act(cnt, EL, tid, act, acnt, abase);
  if (nact == 0) return;

  const int nwarp = NCTA * WPB;
  const long long nitem = (long long)nact * INTE;
  for (long long wi = (long long)blockIdx.x * WPB + warp; wi < nitem; wi += nwarp) {
    const int ai = (int)(wi / INTE);
    const int i = (int)(wi - (long long)ai * INTE);
    const int e = act[ai];
    const int ce = acnt[ai];
    const int sb = abase[ai];
    const __nv_fp8_e4m3 *wgp = W13 + (size_t)e * (2 * (size_t)INTE * HID) + (size_t)i * HID;
    const __nv_fp8_e4m3 *wup = wgp + (size_t)INTE * HID;
    const float *sgp = S13 + (size_t)e * (32 * NKBLK) + (size_t)(i >> 7) * NKBLK;
    const float *sup = S13 + (size_t)e * (32 * NKBLK) + (size_t)(16 + (i >> 7)) * NKBLK;

    for (int c0 = 0; c0 < ce; c0 += MAXC) {
      const int cc = min(MAXC, ce - c0);
      int tk[MAXC];
#pragma unroll
      for (int t = 0; t < MAXC; t++)
        tk[t] = tokids[(size_t)e * TP + (t < cc ? c0 + t : 0)];
      const float *sabase = ascg + ((size_t)e * TP + c0) * NKBLK;
      float accg[MAXC], accu[MAXC];
#pragma unroll
      for (int t = 0; t < MAXC; t++) { accg[t] = 0.f; accu[t] = 0.f; }

      const int koff = lane * 16;
#pragma unroll 7
      for (int j = 0; j < HID / 512; j++) {
        const int k0 = j * 512 + koff;
        const int kb = (k0 >> 7);
        __half2 hg[8], hu[8];
        fp8x16_to_h2(*reinterpret_cast<const int4 *>(wgp + k0), hg);
        fp8x16_to_h2(*reinterpret_cast<const int4 *>(wup + k0), hu);
        const __half2 ws = __float2half2_rn(FFMA_WSCALE);
#pragma unroll
        for (int q = 0; q < 8; q++) {
          hg[q] = __hmul2(hg[q], ws);
          hu[q] = __hmul2(hu[q], ws);
        }
        const float sgv = sgp[kb] * FFMA_WUNSCALE;
        const float suv = sup[kb] * FFMA_WUNSCALE;
#pragma unroll
        for (int t = 0; t < MAXC; t++) {
          if (t < cc) {
            __half2 ha[8];
            fp8x16_to_h2(*reinterpret_cast<const int4 *>(A + (size_t)tk[t] * HID + k0),
                         ha);
            __half2 pg = __float2half2_rn(0.f), pu = __float2half2_rn(0.f);
#pragma unroll
            for (int q = 0; q < 8; q++) {
              pg = __hfma2(ha[q], hg[q], pg);
              pu = __hfma2(ha[q], hu[q], pu);
            }
            const float sa = sabase[(size_t)t * NKBLK + kb];
            accg[t] += h2_hsum(pg) * (sa * sgv);
            accu[t] += h2_hsum(pu) * (sa * suv);
          }
        }
      }

      // combine the 32 lane partials
#pragma unroll
      for (int t = 0; t < MAXC; t++) {
        if (t < cc) {
#pragma unroll
          for (int off = 16; off; off >>= 1) {
            accg[t] += __shfl_xor_sync(0xffffffffu, accg[t], off);
            accu[t] += __shfl_xor_sync(0xffffffffu, accu[t], off);
          }
        }
      }
      if (lane < cc) {
        float g = 0.f, u = 0.f;
#pragma unroll
        for (int t = 0; t < MAXC; t++)
          if (t == lane) { g = accg[t]; u = accu[t]; }
        Cf[(size_t)(sb + c0 + lane) * INTE + i] = (u / (1.f + __expf(-u))) * g;
      }
    }
  }
}

// ------------------------------------------------- fp8 block quantise of C --
// One CTA per packed pair; 32 scale blocks of 64 columns each.
__global__ __launch_bounds__(256) void quantc_kernel(
    const float *__restrict__ Cf, __nv_fp8_e4m3 *__restrict__ Cq,
    float *__restrict__ Cs, const int *__restrict__ cnt, int EL) {
  __shared__ int s_pairs;
  const int tid = threadIdx.x;
  if (tid == 0) {
    int p = 0;
    for (int e = 0; e < EL; e++) p += cnt[e];
    s_pairs = p;
  }
  __syncthreads();
  const int npair = s_pairs;
  for (int slot = blockIdx.x; slot < npair; slot += gridDim.x) {
    const float *src = Cf + (size_t)slot * INTE;
    __nv_fp8_e4m3 *dst = Cq + (size_t)slot * INTE;
    const int warp = tid >> 5, lane = tid & 31;
    // 8 warps x 4 blocks of 64 columns
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int blk = warp * 4 + r;
      const int c0 = blk * 64 + lane * 2;
      float v0 = src[c0], v1 = src[c0 + 1];
      float m = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
      for (int off = 16; off; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
      float sc = m * (1.f / 448.f);
      float inv = sc > 0.f ? 1.f / sc : 0.f;
      dst[c0] = __nv_fp8_e4m3(v0 * inv);
      dst[c0 + 1] = __nv_fp8_e4m3(v1 * inv);
      if (lane == 0) Cs[(size_t)slot * 32 + blk] = sc;
    }
  }
}

// ---------------------------------------------------------- gemm2 (ffma) ---
// One warp per (active expert, pair of output columns).
template <int MAXC, int WPB>
__global__ __launch_bounds__(32 * WPB) void gemm2_ffma_kernel(
    const __nv_fp8_e4m3 *__restrict__ Cq, const float *__restrict__ Cs,
    const __nv_fp8_e4m3 *__restrict__ W2, const float *__restrict__ S2,
    __nv_bfloat16 *__restrict__ out, const int *__restrict__ cnt,
    int *__restrict__ cnt_clear, const int *__restrict__ tokids,
    const float *__restrict__ tokw, int T, int TP, int EL, int NCTA) {
  __shared__ int act[32], acnt[32], abase[32];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (blockIdx.x < (unsigned)EL && tid == 0) cnt_clear[blockIdx.x] = 0;

  const int nact = build_act(cnt, EL, tid, act, acnt, abase);
  if (nact == 0) return;

  constexpr int NHP = HID / 2;
  const int nwarp = NCTA * WPB;
  const long long nitem = (long long)nact * NHP;
  for (long long wi = (long long)blockIdx.x * WPB + warp; wi < nitem; wi += nwarp) {
    const int ai = (int)(wi / NHP);
    const int hp = (int)(wi - (long long)ai * NHP);
    const int h0 = hp * 2;
    const int e = act[ai];
    const int ce = acnt[ai];
    const int sb = abase[ai];
    const __nv_fp8_e4m3 *w0p = W2 + (size_t)e * ((size_t)HID * INTE) + (size_t)h0 * INTE;
    const __nv_fp8_e4m3 *w1p = w0p + INTE;
    const float *s2p = S2 + (size_t)e * (NKBLK * NIBLK) + (size_t)(h0 >> 7) * NIBLK;

    for (int c0 = 0; c0 < ce; c0 += MAXC) {
      const int cc = min(MAXC, ce - c0);
      float acc0[MAXC], acc1[MAXC];
#pragma unroll
      for (int t = 0; t < MAXC; t++) { acc0[t] = 0.f; acc1[t] = 0.f; }
      const int koff = lane * 16;
#pragma unroll 4
      for (int j = 0; j < INTE / 512; j++) {
        const int i0 = j * 512 + koff;
        __half2 h0v[8], h1v[8];
        fp8x16_to_h2(*reinterpret_cast<const int4 *>(w0p + i0), h0v);
        fp8x16_to_h2(*reinterpret_cast<const int4 *>(w1p + i0), h1v);
        const __half2 ws = __float2half2_rn(FFMA_WSCALE);
#pragma unroll
        for (int q = 0; q < 8; q++) {
          h0v[q] = __hmul2(h0v[q], ws);
          h1v[q] = __hmul2(h1v[q], ws);
        }
        const float s2v = s2p[i0 >> 7] * FFMA_WUNSCALE;
        const int ib = i0 >> 6;
#pragma unroll
        for (int t = 0; t < MAXC; t++) {
          if (t < cc) {
            const size_t slot = (size_t)(sb + c0 + t);
            __half2 hc[8];
            fp8x16_to_h2(*reinterpret_cast<const int4 *>(Cq + slot * INTE + i0), hc);
            __half2 p0 = __float2half2_rn(0.f), p1 = __float2half2_rn(0.f);
#pragma unroll
            for (int q = 0; q < 8; q++) {
              p0 = __hfma2(hc[q], h0v[q], p0);
              p1 = __hfma2(hc[q], h1v[q], p1);
            }
            const float sc = Cs[slot * 32 + ib] * s2v;
            acc0[t] += h2_hsum(p0) * sc;
            acc1[t] += h2_hsum(p1) * sc;
          }
        }
      }
#pragma unroll
      for (int t = 0; t < MAXC; t++) {
        if (t < cc) {
#pragma unroll
          for (int off = 16; off; off >>= 1) {
            acc0[t] += __shfl_xor_sync(0xffffffffu, acc0[t], off);
            acc1[t] += __shfl_xor_sync(0xffffffffu, acc1[t], off);
          }
        }
      }
      if (lane < cc) {
        float v0 = 0.f, v1 = 0.f;
#pragma unroll
        for (int t = 0; t < MAXC; t++)
          if (t == lane) { v0 = acc0[t]; v1 = acc1[t]; }
        const int pos = c0 + lane;
        const float w = tokw[(size_t)e * TP + pos];
        const int tok = tokids[(size_t)e * TP + pos];
        __nv_bfloat162 vv = __floats2bfloat162_rn(v0 * w, v1 * w);
        atomicAdd(reinterpret_cast<__nv_bfloat162 *>(out + (size_t)tok * HID + h0), vv);
      }
    }
  }
}
