// One warp per token routing for the DeepSeek-V3 no-aux scheme: sigmoid+bias, group
// top-2 sum, top-4 of 8 groups, global top-8, weights from sigmoid normalised x RSF.
// Every reduction is a warp shuffle, so there is no shared-memory round trip; ties are
// broken by the lower expert/group index, matching an iterative argmax.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEV __device__ __forceinline__
constexpr int WT = 4;            // tokens (warps) per block

DEV float sig(float x) { return 1.f / (1.f + __expf(-x)); }

// (max, count-of-max, second-max) merge, used for the group score
DEV void m2_merge(float &m1, int &c1, float &m2, float om1, int oc1, float om2) {
  if (om1 > m1) {
    m2 = fmaxf(m2, m1);
    m1 = om1;
    c1 = oc1;
    m2 = fmaxf(m2, om2);
  } else if (om1 < m1) {
    m2 = fmaxf(m2, om1);
    m2 = fmaxf(m2, om2);
  } else {
    c1 += oc1;
    m2 = fmaxf(m2, om2);
  }
}

extern "C" __global__ void k_route_cu(
    const float *__restrict__ logits, const __nv_bfloat16 *__restrict__ bias,
    float *__restrict__ tokw, int *__restrict__ cnt, int *__restrict__ ptok,
    float *__restrict__ pw, __nv_bfloat16 *__restrict__ out, int *__restrict__ tokr,
    int *__restrict__ flag,
    int *__restrict__ base, int *__restrict__ mt, int64_t *__restrict__ pt,
    int64_t p2, int64_t p3, int64_t p4, int64_t p5, int64_t p6, int64_t p7,
    int T, int E, int EL, int ELP, int OFF, float RSF, int PMAX, int H, int NPROG,
    int NWARP, int ZOUT) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tok = blockIdx.x * WT + warp;
  if (blockIdx.x == 0 && threadIdx.x == 0) {   // publish the replayed kernels' addresses
    pt[2] = p2; pt[3] = p3; pt[4] = p4; pt[5] = p5; pt[6] = p6; pt[7] = p7;
  }
  constexpr int EPL = 8;                       // experts per lane (E == 256)
  float s[8], sc[8];
  if (tok < T) {
    const float4 *lp = (const float4 *)(logits + (size_t)tok * E + lane * EPL);
    const float4 x0 = lp[0], x1 = lp[1];       // 32 B per lane: two coalesced 512 B/warp
    const float4 b4 = *(const float4 *)(bias + lane * EPL);
    const __nv_bfloat162 *bb = (const __nv_bfloat162 *)&b4;
    const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = sig(xs[j]);
      float bv = (j & 1) ? __high2float(bb[j >> 1]) : __low2float(bb[j >> 1]);
      sc[j] = s[j] + bv;
    }
    // ---- group score: max + (count>=2 ? max : second max), over 32 experts = 4 lanes
    float m1 = -INFINITY, m2 = -INFINITY;
    int c1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (sc[j] > m1) { m2 = m1; m1 = sc[j]; c1 = 1; }
      else if (sc[j] == m1) { ++c1; }
      else if (sc[j] > m2) { m2 = sc[j]; }
    }
    const int LPG = 32 / 8;                    // lanes per group (4 for NGRP=8)
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      float om1 = __shfl_xor_sync(0xffffffff, m1, d);
      int oc1 = __shfl_xor_sync(0xffffffff, c1, d);
      float om2 = __shfl_xor_sync(0xffffffff, m2, d);
      m2_merge(m1, c1, m2, om1, oc1, om2);
    }
    float gs = m1 + (c1 >= 2 ? m1 : m2);
    // ---- top-4 of the 8 group scores (ties -> lower group index); bitmasks only, a
    // runtime index into a register array would spill the array to local memory
    float gall[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) gall[k] = __shfl_sync(0xffffffff, gs, k * LPG);
    uint32_t gsel = 0;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      float best = -INFINITY;
      int bi = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (!((gsel >> k) & 1u) && gall[k] > best) { best = gall[k]; bi = k; }
      gsel |= 1u << bi;
    }
    const int mygrp = lane / LPG;
    const bool ging = (gsel >> mygrp) & 1u;
    // ---- global top-8 among the selected groups (ties -> lower expert index)
    uint32_t selm = 0;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      float lb = -INFINITY;
      int lj = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (ging && !((selm >> j) & 1u) && sc[j] > lb) { lb = sc[j]; lj = j; }
      float gb = lb;                         // max only: one shuffle per step, then the
#pragma unroll                                 // lowest-index holder comes from a ballot
      for (int d = 1; d < 32; d <<= 1) gb = fmaxf(gb, __shfl_xor_sync(0xffffffff, gb, d));
      uint32_t have = __ballot_sync(0xffffffff, lb == gb);
      if (lane == __ffs(have) - 1) selm |= 1u << lj;
    }
    // ---- weights
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += ((selm >> j) & 1u) ? s[j] : 0.f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, d);
    float inv = RSF / (sum + 1e-20f);
    // ---- local experts: count, signed weights, per-expert counts
    int nloc = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int e = lane * 8 + j;
      if (((selm >> j) & 1u) && e >= OFF && e < OFF + EL) ++nloc;
    }
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) nloc += __shfl_xor_sync(0xffffffff, nloc, d);
    float sgn = (nloc == 1) ? -1.f : 1.f;
    const bool mylocal = (lane * 8 >= OFF && lane * 8 + 8 <= OFF + EL);
    int rk[8];
    if (mylocal) {
      float wv[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        wv[j] = ((selm >> j) & 1u) ? s[j] * inv * sgn : 0.f;
      float4 *dst = (float4 *)(tokw + (size_t)tok * ELP + (lane * 8 - OFF));
      dst[0] = make_float4(wv[0], wv[1], wv[2], wv[3]);
      dst[1] = make_float4(wv[4], wv[5], wv[6], wv[7]);
      // the counter atomic already returns this slot's rank inside its expert bucket, so
      // the permutation kernel needs no atomics of its own
#pragma unroll
      for (int j = 0; j < 8; ++j)
        rk[j] = ((selm >> j) & 1u) ? atomicAdd(cnt + lane * 8 + j - OFF, 1) : 0;
    }
    // ---- clear the output row when gemm2 will accumulate into it; this also covers the
    // latency of the rank atomics above
    if (ZOUT && nloc != 1) {
      uint4 z = make_uint4(0, 0, 0, 0);
      for (int c = lane * 8; c < H; c += 32 * 8)
        *(uint4 *)(out + (size_t)tok * H + c) = z;
    }
    if (mylocal) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if ((selm >> j) & 1u) tokr[(size_t)tok * ELP + lane * 8 + j - OFF] = rk[j];
    }
  }
  // ---- clear the permutation arrays
  for (int o = blockIdx.x * 128 + threadIdx.x; o < PMAX; o += NPROG * 128) {
    ptok[o] = 0;
    pw[o] = 0.f;
  }
  // ---- the last warp to arrive builds the padded bucket prefix sum.  Counting warps
  // instead of blocks avoids two __syncthreads() that the whole block would wait on.
  __syncwarp();
  int mine = 0;
  if (lane == 0) {
    __threadfence();             // release: this warp's counter updates must be visible
    mine = (atomicAdd(flag, 1) == NWARP - 1);
  }
  if (__shfl_sync(0xffffffff, mine, 0)) {
    __threadfence();             // acquire: see every other warp's counter updates
    const int e = lane;
    const int c = (e < EL) ? cnt[e] : 0;
    const int pad = ((c + 127) / 128) * 128;
    int x = pad;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int y = __shfl_up_sync(0xffffffff, x, d);
      if (e >= d) x += y;
    }
    if (e < EL) {
      base[e] = x - pad;
      cnt[e] = 0;
    }
    if (e == EL - 1) {
      base[EL] = x;
      mt[0] = x / 128;
      mt[1] = x;
    }
    if (e == 0) *flag = 0;
  }
}

void launch_route(const float *logits, const void *bias, float *tokw, int *cnt, int *ptok,
                  float *pw, void *out, int *tokr, int *flag, int *base, int *mt, void *pt, int64_t p2,
                  int64_t p3, int64_t p4, int64_t p5, int64_t p6, int64_t p7, int T, int E,
                  int EL, int ELP, int OFF, float RSF, int PMAX, int H, int ZOUT,
                  cudaStream_t stream) {
  int nprog = (T + WT - 1) / WT;
  k_route_cu<<<nprog, WT * 32, 0, stream>>>(logits, (const __nv_bfloat16 *)bias, tokw, cnt,
                                 ptok, pw, (__nv_bfloat16 *)out, tokr, flag, base, mt,
                                 (int64_t *)pt, p2, p3,
                                 p4, p5, p6, p7, T, E, EL, ELP, OFF, RSF, PMAX, H, nprog,
                                 nprog * WT, ZOUT);
}
