#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define FULLMASK 0xffffffffu
#ifndef NT
#define NT 256
#endif
#ifndef U1
#define U1 4
#endif
#ifndef U2
#define U2 4
#endif
#ifndef WIN
#define WIN 8
#endif
#ifndef BPSM
#define BPSM (1024/NT)
#endif
#define NW (NT/32)
#define HID 7168
#define INTSZ 2048
#define TWOI 4096
#define EG 4
#define XSCALE 512.0f

__device__ __forceinline__ unsigned cvtf16(unsigned short s) {
  unsigned d; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(s)); return d;
}

__device__ __forceinline__ float dot16h(const uint4 *X, int stride, int g, const uint4 &wv) {
  uint4 xa = X[g], xb = X[stride + g];
  const unsigned *w = reinterpret_cast<const unsigned *>(&wv);
  const unsigned *pa = reinterpret_cast<const unsigned *>(&xa);
  const unsigned *pb = reinterpret_cast<const unsigned *>(&xb);
  __half2 acc = __halves2half2(__float2half(0.f), __float2half(0.f));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    unsigned lo = cvtf16((unsigned short)(w[i] & 0xffffu));
    unsigned hi = cvtf16((unsigned short)(w[i] >> 16));
    unsigned xl = (i < 2) ? pa[2*i] : pb[2*(i-2)];
    unsigned xh = (i < 2) ? pa[2*i+1] : pb[2*(i-2)+1];
    acc = __hfma2(*reinterpret_cast<__half2*>(&lo), *reinterpret_cast<__half2*>(&xl), acc);
    acc = __hfma2(*reinterpret_cast<__half2*>(&hi), *reinterpret_cast<__half2*>(&xh), acc);
  }
  return __half2float(__low2half(acc)) + __half2float(__high2half(acc));
}

__device__ __forceinline__ float wred(float a) {
#pragma unroll
  for (int off = 16; off; off >>= 1) a += __shfl_down_sync(FULLMASK, a, off);
  return a;
}

extern "C" __global__ __launch_bounds__(NT, BPSM)
void moe_fused(const float *__restrict__ logits,
               const __nv_bfloat16 *__restrict__ bias,
               const uint8_t *__restrict__ hs,
               const float *__restrict__ hss,
               const uint8_t *__restrict__ w1,
               const float *__restrict__ w1s,
               const uint8_t *__restrict__ w2,
               const float *__restrict__ w2s,
               int eoff, float rsf,
               float *__restrict__ gbuf,
               float *__restrict__ obuf,
               __nv_bfloat16 *__restrict__ out,
               unsigned int *__restrict__ ctr,
               int nctas) {
  extern __shared__ __align__(16) char smem_raw[];
  uint4 *X = reinterpret_cast<uint4 *>(smem_raw);
  __shared__ float s_hmax;
  __shared__ int s_nhit, s_le[8], s_last;
  __shared__ int s_cnt;
  __shared__ float s_wt[8];
  __shared__ float s_red[EG * NW];
  __shared__ float s_cinv[EG];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int cta = blockIdx.x;
#if RSEL==0
  const int RWARP = 0;
#elif RSEL==1
  const int RWARP = (cta & 3) * 4;
#elif RSEL==2
  const int RWARP = ((cta >> 1) & 3) * 4;
#elif RSEL==3
  const int RWARP = ((cta * 5) & 3) * 4;
#else
  const int RWARP = (cta % 7) * 2;
#endif
  const long W = (long)nctas * NW;
  const long gw = (long)cta * NW + warp;

  // ---------------- routing (warp 0) ----------------
#ifdef NOROUTE
  if (tid == 0) { s_nhit = 3; s_le[0]=22; s_le[1]=30; s_le[2]=10; s_wt[0]=0.3f; s_wt[1]=0.3f; s_wt[2]=0.3f; }
  if (false)
#endif
  if (warp == RWARP) {
    if (lane == 0) s_cnt = 0;
    __syncwarp();
    const float4 *lg4 = reinterpret_cast<const float4 *>(logits);
    float4 la = lg4[lane * 2], lb = lg4[lane * 2 + 1];
    uint4 bpk = reinterpret_cast<const uint4 *>(bias)[lane];
    const __nv_bfloat16 *bh = reinterpret_cast<const __nv_bfloat16 *>(&bpk);
    float lv[8] = {la.x, la.y, la.z, la.w, lb.x, lb.y, lb.z, lb.w};
    float sg[8], sbv[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      sg[t] = __frcp_rn(1.0f + __expf(-lv[t]));
      sbv[t] = sg[t] + __bfloat162float(bh[t]);
    }
    float m1 = -3e38f, m2 = -3e38f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      float x = sbv[t];
      if (x > m1) { m2 = m1; m1 = x; } else if (x > m2) { m2 = x; }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      float o1 = __shfl_xor_sync(FULLMASK, m1, off);
      float o2 = __shfl_xor_sync(FULLMASK, m2, off);
      float n1 = fmaxf(m1, o1);
      float n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
      m1 = n1; m2 = n2;
    }
    float mygsc = m1 + m2;
    int mygrp = lane >> 2;
    int gcnt = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float gk = __shfl_sync(FULLMASK, mygsc, k * 4);
      if (gk > mygsc || (gk == mygsc && k < mygrp)) ++gcnt;
    }
    bool gsel = (gcnt < 4);
    unsigned key[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      unsigned u = __float_as_uint(gsel ? sbv[t] : -3.0e38f);
      key[t] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    unsigned pre = 0;
#pragma unroll 1
    for (int b = 31; b >= 0; --b) {
      unsigned cand = pre | (1u << b);
      unsigned c = 0;
#pragma unroll
      for (int t = 0; t < 8; ++t) c += (key[t] >= cand) ? 1u : 0u;
      c = __reduce_add_sync(FULLMASK, c);
      if (c >= 8u) { pre = cand; if (c == 8u) break; }
    }
    int nsel = 0;
    bool sel[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) { sel[t] = (key[t] >= pre); nsel += sel[t] ? 1 : 0; }
    int scan = nsel;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int o = __shfl_up_sync(FULLMASK, scan, off);
      if (lane >= off) scan += o;
    }
    int rk = scan - nsel;
    float wloc = 0.f;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      if (sel[t]) {
        if (rk < 8) {
          wloc += sg[t];
          int e = lane * 8 + t;
          if (e >= eoff && e < eoff + 32) {
            int p = atomicAdd(&s_cnt, 1);
            s_le[p] = e - eoff; s_wt[p] = sg[t];
          }
        }
        ++rk;
      }
    }
#pragma unroll
    for (int off = 16; off; off >>= 1) wloc += __shfl_xor_sync(FULLMASK, wloc, off);
    __syncwarp();
    if (lane == 0) {
      float inv = rsf * __frcp_rn(wloc + 1e-20f);
      int n = s_cnt;
#pragma unroll
      for (int k = 0; k < 8; ++k) if (k < n) s_wt[k] *= inv;
      s_nhit = n;
    }
  }
  float hmax;
  {
    float a = (lane < 56) ? fabsf(hss[lane]) : 0.f;
    float b = (lane + 32 < 56) ? fabsf(hss[lane + 32]) : 0.f;
    float m = fmaxf(a, b);
#pragma unroll
    for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(FULLMASK, m, off));
    hmax = (m > 0.f) ? m : 1.0f;
  }
  if (tid == 0) s_hmax = hmax;
  {
    float inv = __frcp_rn(hmax);
    for (int g = tid; g < 448; g += NT) {
      uint4 vv = *reinterpret_cast<const uint4 *>(hs + g * 16);
      float sc = __ldg(&hss[g >> 3]) * inv;
      __half2 iv = __halves2half2(__float2half(sc), __float2half(sc));
      const unsigned *w = reinterpret_cast<const unsigned *>(&vv);
      uint4 a, b;
      unsigned *pa = reinterpret_cast<unsigned *>(&a), *pb = reinterpret_cast<unsigned *>(&b);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        unsigned lo = cvtf16((unsigned short)(w[i] & 0xffffu));
        unsigned hi = cvtf16((unsigned short)(w[i] >> 16));
        __half2 L = __hmul2(*reinterpret_cast<__half2*>(&lo), iv);
        __half2 H = __hmul2(*reinterpret_cast<__half2*>(&hi), iv);
        if (i < 2) { pa[2*i] = *reinterpret_cast<unsigned*>(&L); pa[2*i+1] = *reinterpret_cast<unsigned*>(&H); }
        else { pb[2*(i-2)] = *reinterpret_cast<unsigned*>(&L); pb[2*(i-2)+1] = *reinterpret_cast<unsigned*>(&H); }
      }
      X[g] = a; X[448 + g] = b;
    }
  }
  __syncthreads();
  const int nhit0 = s_nhit;
  for (int i = cta * NT + tid; i < nhit0 * TWOI; i += nctas * NT) gbuf[i] = 0.f;
  for (int i = cta * NT + tid; i < HID; i += nctas * NT) obuf[i] = 0.f;
  __syncthreads();

  const int nhit = s_nhit;
  const int myq = (nhit > 0) ? (cta % nhit) : 0;
  const int iq  = (nhit > 0) ? (cta / nhit) : 0;
  const int ncq = (nhit > 0) ? ((nctas - myq + nhit - 1) / nhit) : nctas;
  const long Wq = (long)ncq * NW;
  const long gwq = (long)iq * NW + warp;

  // ---------------- phase 1: GEMM1 ----------------
#ifndef SKIP_P1
  if (nhit > 0) {
    const int lh = lane >> 3;
    const long CPE = (long)TWOI * 14;
    int le = s_le[myq];
    const uint8_t *base = w1 + (size_t)le * TWOI * HID + lane * 16;
    float *gq = gbuf + ((long)myq << 12);
    long c0 = CPE * gwq / Wq, c1 = CPE * (gwq + 1) / Wq;
    int r = (int)(c0 / 14), ck = (int)(c0 % 14);
    long c = c0;
    while (c < c1) {
      int ckE = ck + (int)(c1 - c); if (ckE > 14) ckE = 14;
      const uint8_t *pr = base + (size_t)r * HID;
      const float *wsr = w1s + ((size_t)le * 32 + (r >> 7)) * 56;
      float acc = 0.f;
      if (ckE - ck == 14) {
        uint4 b[WIN];
#pragma unroll
        for (int k = 0; k < WIN; ++k) b[k] = *reinterpret_cast<const uint4 *>(pr + (size_t)k * 512);
#pragma unroll
        for (int k = 0; k < 14; ++k) {
          uint4 cur = b[k % WIN];
          if (k + WIN < 14) b[k % WIN] = *reinterpret_cast<const uint4 *>(pr + (size_t)(k + WIN) * 512);
          acc = fmaf(wsr[k*4+lh], dot16h(X, 448, k*32+lane, cur), acc);
        }
      } else {
        int k = ck;
        for (; k + 4 <= ckE; k += 4) {
          uint4 b[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) b[j] = *reinterpret_cast<const uint4 *>(pr + (size_t)(k + j) * 512);
#pragma unroll
          for (int j = 0; j < 4; ++j) acc = fmaf(wsr[(k+j)*4+lh], dot16h(X, 448, (k+j)*32+lane, b[j]), acc);
        }
        for (; k < ckE; ++k) {
          uint4 bv = *reinterpret_cast<const uint4 *>(pr + (size_t)k * 512);
          acc = fmaf(wsr[k*4+lh], dot16h(X, 448, k*32+lane, bv), acc);
        }
      }
      float t = wred(acc);
      if (lane == 0) atomicAdd(&gq[r], t * s_hmax);
      c += ckE - ck; ck = 0; ++r;
    }
  }
#endif

  // ---------------- grid barrier ----------------
  __syncthreads();
#ifndef NOBAR
  if (tid == 0 && nhit > 0) {
    __threadfence();
    unsigned *cp = ctr + 2 + myq;
    atomicAdd(cp, 1u);
    while (*(volatile unsigned *)cp < (unsigned)ncq) __nanosleep(32);
    __threadfence();
  }
#endif
  __syncthreads();

  // ---------------- phase 2 ----------------
#ifndef SKIP_P2
  if (nhit > 0) {
    const long CPE = (long)HID * 4;
    const int lh = lane >> 3;
    uint4 *Hb = reinterpret_cast<uint4 *>(smem_raw);
    __syncthreads();
    {
      const int KK = INTSZ / NT;
      const float *gp = gbuf + ((long)myq << 12);
      float ga[KK], gc[KK];
#pragma unroll
      for (int k = 0; k < KK; ++k) { ga[k] = __ldcg(&gp[tid + k*NT]); gc[k] = __ldcg(&gp[INTSZ + tid + k*NT]); }
      float lm = 0.f;
#pragma unroll
      for (int k = 0; k < KK; ++k) {
        float b2 = gc[k];
        gc[k] = (b2 * __frcp_rn(1.0f + __expf(-b2))) * ga[k];
        lm = fmaxf(lm, fabsf(gc[k]));
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) lm = fmaxf(lm, __shfl_xor_sync(FULLMASK, lm, off));
      if (lane == 0) s_red[warp] = lm;
      __syncthreads();
      float m = 0.f;
#pragma unroll
      for (int k = 0; k < NW; ++k) m = fmaxf(m, s_red[k]);
      int ex = (int)((__float_as_uint(m) >> 23) & 0xffu) - 126;
      if (m == 0.f) ex = 0;
      float scl = __uint_as_float((unsigned)(127 - ex) << 23);
      if (tid == 0) s_cinv[0] = __uint_as_float((unsigned)(127 + ex) << 23);
      __half *hp = reinterpret_cast<__half *>(Hb);
#pragma unroll
      for (int k = 0; k < KK; ++k) {
        int i = tid + k*NT;
        hp[(((i >> 3) & 1) * 128 + (i >> 4)) * 8 + (i & 7)] = __float2half(gc[k] * scl);
      }
      __syncthreads();
    }
#ifndef SKIP_G2
    {
      int le = s_le[myq];
      float wq = s_wt[myq] * s_cinv[0];
      const uint8_t *base = w2 + (size_t)le * HID * INTSZ + lane * 16;
      long c0 = CPE * gwq / Wq, c1 = CPE * (gwq + 1) / Wq;
      int r = (int)(c0 >> 2), ck = (int)(c0 & 3);
      long c = c0;
      while (c < c1) {
        int ckE = ck + (int)(c1 - c); if (ckE > 4) ckE = 4;
        const uint8_t *pr = base + (size_t)r * INTSZ;
        const float *wsr = w2s + ((size_t)le * 56 + (r >> 7)) * 16;
        uint4 b[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) if (k >= ck && k < ckE) b[k] = *reinterpret_cast<const uint4 *>(pr + (size_t)k * 512);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) if (k >= ck && k < ckE)
          acc = fmaf(wsr[k*4+lh], dot16h(Hb, 128, k*32+lane, b[k]), acc);
        float t = wred(acc);
        if (lane == 0) atomicAdd(&obuf[r], wq * t);
        c += ckE - ck; ck = 0; ++r;
      }
    }
#endif
  }
#endif

  // ---------------- finish ----------------
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(ctr + 1, 1u) == (unsigned)(nctas - 1)) ? 1 : 0;
  }
  __syncthreads();
#ifdef NOEPI
  if (false) {
#else
  if (s_last) {
#endif
    const float4 *op = reinterpret_cast<const float4 *>(obuf);
    __nv_bfloat162 *o2 = reinterpret_cast<__nv_bfloat162 *>(out);
    const int NE = (HID / 4 + NT - 1) / NT;
    float4 vv[NE];
#pragma unroll
    for (int k = 0; k < NE; ++k) { int i = tid + k * NT; if (i < HID/4) vv[k] = op[i]; }
#pragma unroll
    for (int k = 0; k < NE; ++k) { int i = tid + k * NT; if (i < HID/4) {
      o2[2*i]   = __floats2bfloat162_rn(vv[k].x, vv[k].y);
      o2[2*i+1] = __floats2bfloat162_rn(vv[k].z, vv[k].w); } }

    __syncthreads();
    if (tid < 10) ctr[tid] = 0u;
  }
}
