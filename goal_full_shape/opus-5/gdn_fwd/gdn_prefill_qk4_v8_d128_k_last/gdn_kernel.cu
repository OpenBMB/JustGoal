// Fully fused Gated Delta Net prefill: one block per (sequence, v-head,
// v-slice), walking the chunks of a sequence in order with the recurrent state
// resident in registers for the whole sequence, so no intermediate ever reaches
// HBM.  See gdn_common.cuh for the GEMM primitives and notes/APPROACH.md for the
// derivation; gdn_split.cu holds the two-kernel variant used for long sequences.
//
// Per chunk, with S_p the incoming state and L the within-chunk log-decay prefix:
//   A   = beta_i e^{L_i-L_j} (k_i . k_j), j<i      (one gram, routed per level)
//   T   = (I+A)^{-1}                               (blocked forward substitution)
//   U   = T diag(beta) V - T diag(beta e^L) (K S_p)
//   O   = scale diag(e^L) (Q S_p) + ((Q K^T) * e^{L_i-L_j})_{j<=i} U
//   S  <- e^{L_last} S_p + (diag(e^{L_last-L}) K)^T U
// K S_p and Q S_p are formed in the gram phase (one stacked GEMM, no part of T
// needed), which is what lets U be a single GEMM and keeps W = T diag(beta e^L)K
// out of the chunk entirely -- see notes/APPROACH.md section 4.
#include "gdn_common.cuh"

template <int DV>
__global__ __launch_bounds__(NTH, MINBLK) void gdn_fused(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const float *__restrict__ S0, const float *__restrict__ ALOG, const bf16 *__restrict__ AA,
    const float *__restrict__ DTB, const bf16 *__restrict__ BB, const int64_t *__restrict__ CU,
    bf16 *__restrict__ O, float *__restrict__ NS, float scale) {
  constexpr int LDV = DV + 8;
  extern __shared__ char smem[];
  bf16 *sShi = (bf16 *)smem;            // [DV][LDD]  bf16 operand copy of the state
  bf16 *sK = sShi + DV * LDD;           // [C][LDD]
  bf16 *sQ = sK + C * LDD;              // [C][LDD]
  bf16 *sV = sQ + C * LDD;              // [C][LDV]   (reused to stage the output)
  bf16 *sKS = sV + C * LDV;             // [C][LDV]   K S_p, stacked under V
  bf16 *sW = sKS + C * LDV;             // [C][LDD]   only the fallback path uses it
  bf16 *sUT = sW + C * LDD;             // [DV][LDC]  U transposed
  bf16 *sUTd = sUT + DV * LDC;          // [DV][LDC]  the same, decayed by e^{L_last-L}
  bf16 *sMsk = sUTd + DV * LDC;         // [NMSK][C][LDC] P_s blocks, then [Tb | -Tbe]
  constexpr int LDC2 = 2 * C + 8;       // [Tb | -Tbe] lives in the first two of them
  bf16 *sTT = sMsk;                     // [C][LDC2]
  static_assert((size_t)C * LDC2 <= (size_t)2 * C * LDC, "Tb|-Tbe does not fit");
  bf16 *sZ2 = sMsk + NMSK * C * LDC;    // T ping
  bf16 *sZ3 = sZ2 + C * LDC;            // T pong
  bf16 *sX = sZ3 + C * LDC;             // T P_s
  bf16 *sAtt = sX + C * LDC;
  float *sGl = (float *)(sAtt + C * LDC);
  float *sBe = sGl + C;
  float *sL = sBe + C;
  float *sEl = sL + C;
  float *sEd = sEl + C;
  float *sEgl = sEd + C;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int vb = blockIdx.x, h = blockIdx.y, sidx = blockIdx.z;
  const int hq = h / (HV / HQ), iv0 = vb * DV;
  const int s0 = (int)CU[sidx], n = (int)CU[sidx + 1] - s0;
  const float alog = expf(ALOG[h]), dtb = DTB[h];
  const bool live = (n > 0);

  constexpr int TQ = C * (D / 8), TV = C * (DV / 8);
  constexpr int NITQ = (TQ + NTH - 1) / NTH, NITV = (TV + NTH - 1) / NTH;
  uint4 rk[NITQ], rq[NITQ], rv[NITV];
  float rg = 0.f, rb = 0.f, rL = 0.f, rEl = 0.f, rEd = 0.f, rEgl = 0.f;
#define GDN_PREFETCH(CC0, NVX)                                                         \
  {                                                                                    \
    const int nvx_ = (NVX);                                                            \
    _Pragma("unroll") for (int u = 0; u < NITQ; ++u) {                                 \
      const int it = tid + u * NTH;                                                    \
      rk[u] = make_uint4(0, 0, 0, 0);                                                  \
      rq[u] = rk[u];                                                                   \
      if (it < TQ) {                                                                   \
        const int r = it / (D / 8), cc = (it % (D / 8)) * 8;                            \
        if (r < nvx_) {                                                                \
          const size_t off = ((size_t)(s0 + (CC0) + r) * HQ + hq) * D + cc;             \
          rk[u] = *reinterpret_cast<const uint4 *>(K + off);                            \
          rq[u] = *reinterpret_cast<const uint4 *>(Q + off);                            \
        }                                                                              \
      }                                                                                \
    }                                                                                  \
    _Pragma("unroll") for (int u = 0; u < NITV; ++u) {                                 \
      const int it = tid + u * NTH;                                                    \
      rv[u] = make_uint4(0, 0, 0, 0);                                                  \
      if (it < TV) {                                                                   \
        const int r = it / (DV / 8), cc = (it % (DV / 8)) * 8;                           \
        if (r < nvx_)                                                                  \
          rv[u] = *reinterpret_cast<const uint4 *>(                                    \
              V + ((size_t)(s0 + (CC0) + r) * HV + h) * D + iv0 + cc);                  \
      }                                                                                \
    }                                                                                  \
    rg = 0.f;                                                                          \
    rb = 0.f;                                                                          \
    if (tid < C && tid < nvx_) {                                                       \
      const size_t off = (size_t)(s0 + (CC0) + tid) * HV + h;                           \
      rg = -alog * softplus(__bfloat162float(AA[off]) + dtb);                           \
      rb = 1.f / (1.f + __expf(-__bfloat162float(BB[off])));                            \
    }                                                                                  \
    if (warp == 0) {                                                                   \
      float v_ = (lane < C) ? rg : 0.f;                                                 \
      _Pragma("unroll") for (int d_ = 1; d_ < C; d_ <<= 1) {                           \
        const float t_ = __shfl_up_sync(0xffffffff, v_, d_);                            \
        if (lane >= d_) v_ += t_;                                                       \
      }                                                                                \
      const float gl_ = __shfl_sync(0xffffffff, v_, C - 1);                             \
      rL = v_;                                                                          \
      rEl = __expf(v_);                                                                 \
      rEd = __expf(gl_ - v_);                                                           \
      rEgl = __expf(gl_);                                                               \
    }                                                                                  \
  }

  if (n > 0) GDN_PREFETCH(0, min(C, n))

  // ---- the state lives in registers as the persistent mma accumulator of
  //      S <- e^g S + U^T Kd, with a bf16 copy in shared memory for operands --
  constexpr int SNEL = TL<DV, D, NW>::NEL;
  float Sacc[SNEL];
  {
    const float *sp = (S0 && live) ? (S0 + ((size_t)sidx * HV + h) * D * D) : nullptr;
#pragma unroll
    for (int e = 0; e < SNEL; e += 2) {
      int r, c;
      accpos<DV, D, NW>(e, warp, lane, r, c);
      float2 z = make_float2(0.f, 0.f);
      if (sp) z = *reinterpret_cast<const float2 *>(sp + (size_t)(iv0 + r) * D + c);
      Sacc[e] = z.x;
      Sacc[e + 1] = z.y;
    }
#pragma unroll
    for (int e = 0; e < SNEL; e += 2) {
      int r, c;
      accpos<DV, D, NW>(e, warp, lane, r, c);
      bf16 two[2] = {__float2bfloat16(Sacc[e]), __float2bfloat16(Sacc[e + 1])};
      *reinterpret_cast<uint32_t *>(sShi + r * LDD + c) = *reinterpret_cast<uint32_t *>(two);
    }
  }

  __syncthreads();

  for (int c0 = 0; c0 < n; c0 += C) {
    const int nv = min(C, n - c0);
#pragma unroll
    for (int u = 0; u < NITQ; ++u) {
      const int it = tid + u * NTH;
      if (it < TQ) {
        const int r = it / (D / 8), cc = (it % (D / 8)) * 8;
        *reinterpret_cast<uint4 *>(sK + r * LDD + cc) = rk[u];
        *reinterpret_cast<uint4 *>(sQ + r * LDD + cc) = rq[u];
      }
    }
#pragma unroll
    for (int u = 0; u < NITV; ++u) {
      const int it = tid + u * NTH;
      if (it < TV) {
        const int r = it / (DV / 8), cc = (it % (DV / 8)) * 8;
        *reinterpret_cast<uint4 *>(sV + r * LDV + cc) = rv[u];
      }
    }
    if (tid < C) sBe[tid] = rb;
    if (C <= 32) {
      if (warp == 0 && lane < C) {
        sL[lane] = rL;
        sEl[lane] = rEl;
        sEd[lane] = rEd;
        if (lane == 0) *sEgl = rEgl;
      }
    } else {
      if (tid < C) sGl[tid] = rg;
    }
    __syncthreads();
    if (C > 32) {                      // wide chunks: scan through shared memory
      if (warp == 0) {
        float v = (lane < C) ? sGl[lane] : 0.f;
        float v2 = (lane + 32 < C) ? sGl[lane + 32] : 0.f;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const float t = __shfl_up_sync(0xffffffff, v, d);
          if (lane >= d) v += t;
          const float t2 = __shfl_up_sync(0xffffffff, v2, d);
          if (lane >= d) v2 += t2;
        }
        const float base = __shfl_sync(0xffffffff, v, 31);
        if (lane < C) { sL[lane] = v; }
        if (lane + 32 < C) { sL[lane + 32] = base + v2; }
      }
      __syncthreads();
      const float gl = sL[C - 1];
      if (tid < C) { sEl[tid] = __expf(sL[tid]); sEd[tid] = __expf(gl - sL[tid]); }
      if (tid == 0) *sEgl = __expf(gl);
      __syncthreads();
    }
    const float egl = *sEgl;
    if (c0 + C < n) GDN_PREFETCH(c0 + C, min(C, n - c0 - C))

    // ---- the two grams (both read only sQ/sK, so they share one segment):
    //      A routed into its per-level blocks P_s, the closed-form 2x2 seed of
    //      T, and att ----
    // K S_p and Q S_p share the right operand and need no part of T, so they go
    // here as one [2C][DV] GEMM (sK and sQ are adjacent with the same pitch).
    // Rows < C are K S_p, which replaces W = Tbe K entirely: Z = W S_p is
    // Tbe (K S_p), a K = C GEMM instead of a K = D one, and the whole W phase
    // and its barrier disappear.  Rows >= C are Y = Q S_p, and they land in the
    // same warps and registers that the att GEMM later accumulates into.
    // It only works when the [2C][DV] tile's accumulator layout over NW warps
    // matches the [C][DV] tile's over NW/2 -- true for every shape this
    // dispatches, but not for every DV the launcher can instantiate, so the
    // original three-GEMM form stays as a fallback.
    constexpr int NWH = NW / 2;
    const int wh = warp & (NWH - 1);
    typedef TL<2 * C, DV, NW> TS;
    typedef TL<C, DV, NWH> TH;
    constexpr bool STK = TS::NEL == TH::NEL && TS::NWM == 2 * TH::NWM &&
                         TS::NWN == TH::NWN && TS::NWA == NW;
    float accO[TH::NEL];
    {
      float accG[TL<C, C, NW>::NEL], accA[TL<C, C, NW>::NEL];
      gemm<C, C, D, true, true, NW>(accG, sK, LDD, sK, LDD, warp, lane);
      gemm<C, C, D, true, true, NW>(accA, sQ, LDD, sK, LDD, warp, lane);
      if constexpr (STK) {
        gemm<2 * C, DV, D, true, true, NW>(accO, sK, LDD, sShi, LDD, warp, lane);
        PAIRLOOP(2 * C, DV, {
          if (r < C) {
            st2(sKS + r * LDV + c, accO[e], accO[e + 1]);
          } else {
            const float f = scale * sEl[r - C];
            accO[e] *= f;
            accO[e + 1] *= f;
          }
        })
      }
      PAIRLOOP(C, C, {
        float va = 0.f, vb = 0.f;
        if (c <= r) va = scale * accA[e] * __expf(fminf(sL[r] - sL[c], 0.f));
        if (c + 1 <= r) vb = scale * accA[e + 1] * __expf(fminf(sL[r] - sL[c + 1], 0.f));
        st2(sAtt + r * LDC + c, va, vb);
      })
      PAIRLOOP(C, C, {
        float a0 = 0.f, a1 = 0.f;
        if (c < r) a0 = sBe[r] * accG[e] * __expf(fminf(sL[r] - sL[c], 0.f));
        if (c + 1 < r) a1 = sBe[r] * accG[e + 1] * __expf(fminf(sL[r] - sL[c + 1], 0.f));
        // level of an entry = position of the highest bit where r and c differ
        const int l0 = (c < r) ? (31 - __clz(r ^ c)) : -1;
        const int l1 = (c + 1 < r) ? (31 - __clz(r ^ (c + 1))) : -1;
#pragma unroll
        for (int lv = 1; lv <= NMSK; ++lv)
          st2(sMsk + (size_t)(lv - 1) * C * LDC + r * LDC + c, l0 == lv ? a0 : 0.f,
              l1 == lv ? a1 : 0.f);
        float t0 = (r == c) ? 1.f : 0.f, t1v = (r == c + 1) ? 1.f : 0.f;
        if (l0 == 0) t0 = -a0;
        if (l1 == 0) t1v = -a1;
        st2(sZ2 + r * LDC + c, t0, t1v);
      })
    }
    __syncthreads();
    // ---- T = (I+A)^{-1}, left as [Tb | -Tbe] in one [C][2C] matrix ----
    tri_inv<STK ? LDC2 : LDC, STK>(sZ2, sZ3, sX, sMsk, sTT, STK ? sTT + C : sMsk + C * LDC,
                                   sBe, sEl, nv, warp, lane);
    if constexpr (STK) {
      // U = Tb V - Tbe (K S_p) is then a single GEMM over the concatenated K
      // dimension, since V and K S_p are adjacent with the same pitch.  Only
      // half the warps can tile a C x DV output; the rest are done here.
      if (warp < NWH) {
        float accU[TH::NEL];
        gemm<C, DV, 2 * C, false, true, NWH>(accU, sTT, LDC2, sV, LDV, wh, lane);
        ACCLOOP2(C, DV, NWH, wh, { sUT[c * LDC + r] = __float2bfloat16(accU[e]); })
      }
    } else {
      {
        float accW[TL<C, D, NW>::NEL];
        gemm<C, D, C, false, true, NW>(accW, sMsk + C * LDC, LDC, sK, LDD, warp, lane);
        ACCLOOP(C, D, { sW[r * LDD + c] = __float2bfloat16(accW[e]); })
      }
      __syncthreads();
      if (warp < NWH) {
        float accU[TH::NEL], accZ[TH::NEL];
        gemm<C, DV, C, false, true, NWH>(accU, sMsk, LDC, sV, LDV, wh, lane);
        gemm<C, DV, D, true, true, NWH>(accZ, sW, LDD, sShi, LDD, wh, lane);
#pragma unroll
        for (int e = 0; e < TH::NEL; ++e) accU[e] -= accZ[e];
        ACCLOOP2(C, DV, NWH, wh, { sUT[c * LDC + r] = __float2bfloat16(accU[e]); })
      } else {
        gemm<C, DV, D, true, true, NWH>(accO, sQ, LDD, sShi, LDD, wh, lane);
        ACCLOOP2(C, DV, NWH, wh, { accO[e] *= scale * sEl[r]; })
      }
    }
    __syncthreads();
    // ---- O = scale e^{L} (Q S_p) + att U ----
    {
      if (warp >= NWH) {
        gemm<C, DV, C, true, false, NWH>(accO, sAtt, LDC, sUT, LDC, wh, lane);
        ACCLOOP2(C, DV, NWH, wh, { sV[r * LDV + c] = __float2bfloat16(accO[e]); })
      } else {
        // the warps that sit out the att GEMM fold the decay into a second copy
        // of U^T, which is what the state update wants -- doing it here instead
        // of in a pass of its own costs nothing and saves two barriers
        for (int it = tid * 2; it < DV * C; it += NTH) {
          const int v = it / C, t0 = it % C;
          float u0, u1;
          ld2(sUT + v * LDC + t0, u0, u1);
          st2(sUTd + v * LDC + t0, u0 * sEd[t0], u1 * sEd[t0 + 1]);
        }
      }
      __syncthreads();
      for (int it = tid; it < TV; it += NTH) {
        const int r = it / (DV / 8), cc = (it % (DV / 8)) * 8;
        if (r < nv)
          *reinterpret_cast<uint4 *>(O + ((size_t)(s0 + c0 + r) * HV + h) * D + iv0 + cc) =
              *reinterpret_cast<const uint4 *>(sV + r * LDV + cc);
      }
    }
    // ---- S <- e^{L_last} S + Kd^T U, in registers (overlaps the flush) ----
    {
#pragma unroll
      for (int e = 0; e < SNEL; ++e) Sacc[e] *= egl;
      gemm<DV, D, C, false, false, NW>(Sacc, sUTd, LDC, sK, LDD, warp, lane);
#pragma unroll
      for (int e = 0; e < SNEL; e += 2) {
        int r, c;
        accpos<DV, D, NW>(e, warp, lane, r, c);
        bf16 two[2] = {__float2bfloat16(Sacc[e]), __float2bfloat16(Sacc[e + 1])};
        *reinterpret_cast<uint32_t *>(sShi + r * LDD + c) = *reinterpret_cast<uint32_t *>(two);
      }
      __syncthreads();
    }
  }
  {
    float *np = NS + ((size_t)sidx * HV + h) * D * D;
#pragma unroll
    for (int e = 0; e < SNEL; e += 2) {
      int r, c;
      accpos<DV, D, NW>(e, warp, lane, r, c);
      float2 z = make_float2(live ? Sacc[e] : 0.f, live ? Sacc[e + 1] : 0.f);
      *reinterpret_cast<float2 *>(np + (size_t)(iv0 + r) * D + c) = z;
    }
  }
}

template <int DV>
static size_t smem_bytes() {
  constexpr int LDV = DV + 8;
  return (size_t)DV * LDD * 2 + (size_t)3 * C * LDD * 2 + (size_t)2 * C * LDV * 2 +
         (size_t)2 * DV * LDC * 2 + (size_t)(NMSK + 4) * C * LDC * 2 + 5 * C * 4 + 256;
}

extern "C" void gdn_launch(const void *q, const void *k, const void *v, const void *state,
                           const void *alog, const void *a, const void *dtb, const void *b,
                           const void *cu, void *o, void *ns, float scale, int T, int N, int vsplit,
                           int stages, cudaStream_t stream) {
#define LAUNCH(DVV)                                                                    \
  do {                                                                                 \
    const size_t sm = smem_bytes<DVV>();                                               \
    static bool done = false;                                                          \
    if (!done) {                                                                       \
      cudaFuncSetAttribute((void *)gdn_fused<DVV>,                                     \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);      \
      cudaFuncSetAttribute((void *)gdn_fused<DVV>,                                     \
                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);       \
      done = true;                                                                     \
    }                                                                                  \
    gdn_fused<DVV><<<dim3(D / DVV, HV, N), NTH, sm, stream>>>(                         \
        (const bf16 *)q, (const bf16 *)k, (const bf16 *)v, (const float *)state,       \
        (const float *)alog, (const bf16 *)a, (const float *)dtb, (const bf16 *)b,     \
        (const int64_t *)cu, (bf16 *)o, (float *)ns, scale);                           \
  } while (0)
  if (vsplit == 1) LAUNCH(128);
  else if (vsplit == 2) LAUNCH(64);
  else if (vsplit == 4) LAUNCH(32);
  else LAUNCH(16);
#undef LAUNCH
}
