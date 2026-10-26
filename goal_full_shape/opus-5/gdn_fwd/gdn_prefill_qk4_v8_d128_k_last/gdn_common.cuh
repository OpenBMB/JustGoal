// Gated Delta Net prefill -- fused chunked (WY / UT-transform) CUDA kernel.
//
// One block per (sequence, v-head, v-slice).  The recurrent state stays in
// shared memory for the whole sequence, so nothing round-trips through HBM.
// GEMMs are warp-level mma.sync.m16n8k16 (bf16 operands, fp32 accumulate) fed
// by ldmatrix.  Chunk inputs are prefetched one chunk ahead into registers.
//
// Derivation (per sequence and v-head; the state S is [K,V]):
//     S_i = g_i S_{i-1} + k_i^T u_i,  u_i = beta_i (v_i - k_i g_i S_{i-1})
//     o_i = scale * q_i S_i
// With L_i = sum_{j<=i} log g_j inside a chunk of C tokens and S_p the state
// entering the chunk,
//     S_i         = e^{L_i} S_p + sum_{j<=i} e^{L_i-L_j} k_j^T u_j
//     g_i S_{i-1} = e^{L_i} S_p + sum_{j<i}  e^{L_i-L_j} k_j^T u_j
// so U = [u_0..u_{C-1}] solves the unit lower triangular system
//     (I + A) U = beta*V - diag(beta e^L) K S_p,
//     A_ij = beta_i e^{L_i-L_j} (k_i . k_j)   for j < i, else 0,
// and with T = (I+A)^{-1}, W = T diag(beta e^L) K, Uv = T (beta*V):
//     U     = Uv - W S_p
//     O     = scale [ diag(e^L) Q S_p + ((Q K^T) * e^{L_i-L_j})_{j<=i} U ]
//     S_out = e^{L_{C-1}} S_p + (diag(e^{L_{C-1}-L_j}) K)^T U
// Every exponent is <= 0, so all factors are bounded by 1.  T is built by
// exact blocked forward substitution (T <- T - T P_s T), which is stable; a
// Neumann/doubling series is not, because intermediate powers of A can blow
// up long before nilpotency kicks in.
//
// The state is kept V-major, S_v[v][k].  That makes the global<->shared copy a
// plain contiguous copy and lets every GEMM touching S use the cheaper
// "B stored transposed" operand form.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

typedef __nv_bfloat16 bf16;
#define DEVI __device__ __forceinline__

DEVI uint32_t sm32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void ldm_x4(uint32_t *r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldm_x2(uint32_t *r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
DEVI void ldm_x2t(uint32_t *r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
// Asynchronous 16-byte global->shared copy.  It needs no registers to hold the
// data in flight and the queue is far deeper than a register prefetch, which is
// what the serial scan needs: with one block per SM there are only eight warps
// to keep memory-level parallelism up.
DEVI void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// ... and the form with an explicit source size: 0 zero-fills the destination,
// which is exactly the masking a partial chunk needs.
DEVI void cp_async16s(uint32_t dst, const void *src, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

DEVI void mma16816(float *d, const uint32_t *a, const uint32_t *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Pick the warp grid (NWM x NWN) for an M x N output.  Every mma operand comes
// from shared memory, and one warp tile costs MT ldmatrix.x4 + NT ldmatrix.x2
// per k-step for MT*NT mmas -- so (4*MT + 2*NT)/(MT*NT) shared-memory cycles per
// mma.  Fat tiles are therefore much cheaper in shared bandwidth, which is the
// binding resource here; prefer tilings that keep every warp busy, and among
// those the one with the lowest operand cost.  Warps beyond NWM*NWN sit out.
constexpr int cpick(int MTT, int NTT, int NW) {
  int best = -1;
  long bc = 1L << 60;
  for (int nwm = 1; nwm <= NW; nwm <<= 1)
    for (int nwn = 1; nwn * nwm <= NW; nwn <<= 1) {
      if (MTT % nwm || NTT % nwn) continue;
      const int mt = MTT / nwm, nt = NTT / nwn;
      const int idle = NW - nwm * nwn;          // warps with nothing to do
      const long opcost = (4L * mt + 2L * nt) * 1024 / (mt * nt);
      const long c = (long)idle * 1024 * 1024 + opcost;
      if (c < bc) { bc = c; best = nwm * 1024 + nwn; }
    }
  return best;
}

template <int M, int N, int NW>
struct TL {
  static constexpr int MTT = M / 16, NTT = N / 8;
  static constexpr int PK = cpick(MTT, NTT, NW);
  static_assert(PK > 0, "no valid warp tiling for this shape");
  static constexpr int NWM = PK / 1024, NWN = PK % 1024;
  static constexpr int NWA = NWM * NWN;
  static constexpr int MT = MTT / NWM, NT = NTT / NWN;
  static constexpr int NEL = MT * NT * 4;
};

// C[M][N] (+)= A[M][K] * (BNT ? B[N][K]^T : B[K][N]); bf16 operands in shared
// memory with the last index contiguous.
template <int M, int N, int K, bool BNT, bool ZERO, int NW>
DEVI void gemm(float *acc, const bf16 *A, int lda, const bf16 *B, int ldb, int warp, int lane) {
  typedef TL<M, N, NW> T;
  constexpr int MT = T::MT, NT = T::NT, NWN = T::NWN;
  if (T::NWA < NW && warp >= T::NWA) return;
  const int wm = warp / NWN, wn = warp % NWN;
  uint32_t aad[MT], bad[NT];
#pragma unroll
  for (int i = 0; i < MT; ++i)
    aad[i] = sm32(A + (size_t)((wm * MT + i) * 16 + (lane & 15)) * lda + 8 * (lane >> 4));
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int n0 = (wn * NT + j) * 8;
    if (BNT) bad[j] = sm32(B + (size_t)(n0 + (lane & 7)) * ldb + ((lane & 8) ? 8 : 0));
    else     bad[j] = sm32(B + (size_t)(lane & 15) * ldb + n0);
  }
  if (ZERO) {
#pragma unroll
    for (int i = 0; i < MT * NT * 4; ++i) acc[i] = 0.f;
  }
#pragma unroll
  for (int kk = 0; kk < K; kk += 16) {
    uint32_t af[MT][4], bfr[NT][2];
#pragma unroll
    for (int i = 0; i < MT; ++i) ldm_x4(af[i], aad[i] + (uint32_t)(kk * 2));
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      if (BNT) ldm_x2(bfr[j], bad[j] + (uint32_t)(kk * 2));
      else     ldm_x2t(bfr[j], bad[j] + (uint32_t)(kk * ldb * 2));
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) mma16816(acc + (i * NT + j) * 4, af[i], bfr[j]);
  }
}

template <int M, int N, int NW>
DEVI void accpos(int idx, int warp, int lane, int &row, int &col) {
  typedef TL<M, N, NW> T;
  const int wm = warp / T::NWN, wn = warp % T::NWN;
  const int i = idx / (T::NT * 4), j = (idx / 4) % T::NT, e = idx & 3;
  row = (wm * T::MT + i) * 16 + (lane >> 2) + ((e >> 1) ? 8 : 0);
  col = (wn * T::NT + j) * 8 + 2 * (lane & 3) + (e & 1);
}

// softplus without a libdevice call: the series branch keeps the small-x
// absolute accuracy that log(1+z) loses once z falls below ~1e-2.
DEVI float softplus(float x) {
  const float z = __expf(x);
  if (x > 20.f) return x;
  if (x < -15.f) return z;
  if (x < -3.f) return z * (1.f - z * (0.5f - z * (0.3333333333f - z * 0.25f)));
  return __logf(1.f + z);
}

DEVI void st2(bf16 *p, float a, float b) {
  const __nv_bfloat162 x = __floats2bfloat162_rn(a, b);
  *reinterpret_cast<__nv_bfloat162 *>(p) = x;
}
DEVI void ld2(const bf16 *p, float &a, float &b) {
  const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162 *>(p);
  a = __low2float(x);
  b = __high2float(x);
}

// ACCLOOP/PAIRLOOP variants for a sub-group of warps: NWX warps, with WX the
// warp index inside the group.  Used to run two independent GEMMs on two warp
// halves when an output shape is too small to tile over every warp.
#define ACCLOOP2(M, N, NWX, WX, ...)                              \
  if (TL<M, N, NWX>::NWA >= (NWX) || (WX) < TL<M, N, NWX>::NWA) {  \
    constexpr int NEL_ = TL<M, N, NWX>::NEL;                      \
    _Pragma("unroll") for (int e = 0; e < NEL_; ++e) {            \
      int r, c;                                                   \
      accpos<M, N, NWX>(e, (WX), lane, r, c);                     \
      __VA_ARGS__;                                                \
    }                                                             \
  }

#define PAIRLOOP2(M, N, NWX, WX, ...)                             \
  if (TL<M, N, NWX>::NWA >= (NWX) || (WX) < TL<M, N, NWX>::NWA) {  \
    constexpr int NEL_ = TL<M, N, NWX>::NEL;                      \
    _Pragma("unroll") for (int e = 0; e < NEL_; e += 2) {         \
      int r, c;                                                   \
      accpos<M, N, NWX>(e, (WX), lane, r, c);                     \
      __VA_ARGS__;                                                \
    }                                                             \
  }

// Like ACCLOOP but visits accumulator elements in adjacent-column pairs so
// that bf16 results can be packed into one 32-bit shared-memory store.
#define PAIRLOOP(M, N, ...)                                  \
  if (TL<M, N, NW>::NWA >= NW || warp < TL<M, N, NW>::NWA) {  \
    constexpr int NEL_ = TL<M, N, NW>::NEL;                   \
    _Pragma("unroll") for (int e = 0; e < NEL_; e += 2) {     \
      int r, c;                                               \
      accpos<M, N, NW>(e, warp, lane, r, c);                  \
      __VA_ARGS__;                                            \
    }                                                         \
  }

#ifndef CHUNK
#define CHUNK 32
#endif
#ifndef NWARPS
#define NWARPS 8
#endif
#ifndef MINBLK
#define MINBLK 1          // requested minimum resident blocks per SM
#endif
constexpr int D = 128, HV = 8, HQ = 4, C = CHUNK, NW = NWARPS;
constexpr int LDD = D + 8, LDC = C + 8, NTH = NW * 32;
constexpr int NLEV = C == 16 ? 4 : (C == 32 ? 5 : (C == 64 ? 6 : 3));   // log2(C)
constexpr int NMSK = NLEV - 1;      // matmul levels of the triangular inverse

#define ACCLOOP(M, N, ...)                                   \
  if (TL<M, N, NW>::NWA >= NW || warp < TL<M, N, NW>::NWA) {  \
    constexpr int NEL_ = TL<M, N, NW>::NEL;                   \
    _Pragma("unroll") for (int e = 0; e < NEL_; ++e) {        \
      int r, c;                                               \
      accpos<M, N, NW>(e, warp, lane, r, c);                  \
      __VA_ARGS__;                                            \
    }                                                         \
  }

// T = (I+A)^{-1} for strictly lower triangular A, by exact blocked forward
// substitution:  T <- T - T P_s T  over the power-of-two block levels, where
// P_s holds the sub-diagonal block joining the two halves of every 2s block.
// An entry A[r][c] belongs to exactly one level, namely the position of the
// highest bit where r and c differ, so all the P_s can be laid out in one pass
// (see the caller) and each level then costs only two barriers.
// Mbase holds P_2, P_4, ... contiguously; T must arrive holding the closed-form
// inverse of the 2x2 diagonal blocks.  Returns the buffer holding the result.
// The only two things the callers want from T are Tb = T diag(beta) and
// Tbe = T diag(beta e^L), so the last level writes those directly: that saves a
// whole phase (read T, scale, store twice) and the barrier in front of it.  Tb
// and Tbe land in the first two P_s buffers, which the early levels have long
// since consumed -- safe as long as there are at least three levels.
// Level lv joins the two halves of each 2^(lv+1) block, so a chunk with only nv
// valid tokens has nothing in the levels from 2^(lv+1) >= nv up: ceil(log2(nv))-1
// of them suffice (at least one, which is where Tb/Tbe get written).  Every
// sequence ends in a partial chunk, and a short sequence is nothing else.
template <int LDT = LDC, bool NEGE = false>
DEVI void tri_inv(bf16 *Tc, bf16 *Tn, bf16 *sX, const bf16 *Mbase, bf16 *Tb, bf16 *Tbe,
                  const float *sBe, const float *sEl, int nv, int warp, int lane) {
  static_assert(NMSK >= 3, "Tb/Tbe would alias a P_s block still in use");
  float t1[TL<C, C, NW>::NEL], t2[TL<C, C, NW>::NEL];
#ifndef INVLEV
#define INVLEV NMSK       // experiment knob: truncate the inverse (wrong maths)
#endif
  const int nl = max(1, min(INVLEV, 31 - __clz(max(nv - 1, 1))));
  for (int lv = 0; lv < nl; ++lv) {
    const bf16 *P = Mbase + (size_t)lv * C * LDC;
    gemm<C, C, C, false, true, NW>(t1, Tc, LDC, P, LDC, warp, lane);
    PAIRLOOP(C, C, { st2(sX + r * LDC + c, t1[e], t1[e + 1]); })
    __syncthreads();
    gemm<C, C, C, false, true, NW>(t2, sX, LDC, Tc, LDC, warp, lane);
    if (lv == nl - 1) {
      PAIRLOOP(C, C, {
        float o0, o1;
        ld2(Tc + r * LDC + c, o0, o1);
        const float b0 = (o0 - t2[e]) * sBe[c], b1 = (o1 - t2[e + 1]) * sBe[c + 1];
        constexpr float sg = NEGE ? -1.f : 1.f;
        st2(Tb + r * LDT + c, b0, b1);
        st2(Tbe + r * LDT + c, sg * b0 * sEl[c], sg * b1 * sEl[c + 1]);
      })
    } else {
      PAIRLOOP(C, C, {
        float o0, o1;
        ld2(Tc + r * LDC + c, o0, o1);
        st2(Tn + r * LDC + c, o0 - t2[e], o1 - t2[e + 1]);
      })
    }
    __syncthreads();
    bf16 *x = Tc;
    Tc = Tn;
    Tn = x;
  }
}
