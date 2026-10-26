// Two-phase ("split") variant of the chunked gated-delta-rule algorithm.
//
// Everything in the WY / UT transform of a chunk is independent of the
// recurrent state, so it is hoisted into ut_body, which runs one block per
// (chunk, v-head) and therefore has enough blocks in flight to hide latency.
// Only the state recursion stays sequential, in scan_body, which per chunk does
//     U = Uv - W S_p ;  O = Qg S_p + att U ;  S = e^{L_last} S_p + Kd^T U
// with Qg = scale diag(e^L) Q and Kd = diag(e^{L_last-L}) K both pre-scaled by
// the UT phase, so the scan has no per-token gate arithmetic left at all.
//
// The scan slices the V dimension freely (none of its math is duplicated by
// that split), which is also how the (seq, head) chains get balanced over SMs.
//
// Both bodies are __device__ functions, called either as two kernels (gdn_ut
// then gdn_scan) or as one cooperative kernel with a grid-wide barrier between
// the phases (gdn_coop).  The second form exists because a kernel launch costs
// several microseconds under this operator's timing definition, which for short
// and medium inputs is most of the time; see notes/APPROACH.md.
#include <cooperative_groups.h>

#include "gdn_common.cuh"

#ifndef AHEAD
#define AHEAD 1          // chunks of staging in flight in the serial scan.  Two
                         // measures the same as one -- the copies are already
                         // enough to saturate what the memory system will give
                         // eight warps -- and one keeps the shared footprint
                         // small, which the fused cooperative kernel needs.
#endif
#ifndef NSTG
#define NSTG 3           // experiment knob: how many of Kd/Qg/W to stage
#endif

namespace cg = cooperative_groups;

// Chunk ids are numbered per sequence and concatenated; both kernels derive the
// same numbering from cu_seqlens alone.  sh[0..N] ends up holding the exclusive
// prefix sum of ceil(len_i / C), with sh[N] = the total number of chunks.
DEVI void chunk_prefix(const int64_t *CU, int N, int tid, int *sh) {
  for (int i = tid; i <= N; i += NTH) {
    int c = 0;
    if (i < N) c = ((int)(CU[i + 1] - CU[i]) + C - 1) / C;
    sh[i] = c;
  }
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int i = 0; i <= N; ++i) {
      const int c = sh[i];
      sh[i] = acc;
      acc += c;
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------- kernel 1 --
// One chunk's UT transform, as a device function so that both the standalone
// kernel and the fused cooperative kernel can call it.  sIx is the chunk-number
// prefix sum, computed once by the caller.
DEVI void ut_body(char *smem, const int *sIx, int gid, int h,
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const float *__restrict__ ALOG, const bf16 *__restrict__ AA, const float *__restrict__ DTB,
    const bf16 *__restrict__ BB, const int64_t *__restrict__ CU, bf16 *__restrict__ WB,
    bf16 *__restrict__ UVB, bf16 *__restrict__ ATB, bf16 *__restrict__ QGB,
    bf16 *__restrict__ KDB, float *__restrict__ GB, float scale, int N, int us) {
  bf16 *sK = (bf16 *)smem;           // [C][LDD]
  bf16 *sQ = sK + C * LDD;           // [C][LDD]
  bf16 *sV = sQ + C * LDD;           // [C][LDD]  (reused to stage Uv)
  bf16 *sWb = sV + C * LDD;          // [C][LDD]  staging for W
  bf16 *sMsk = sWb + C * LDD;        // [NMSK][C][LDC] P_s blocks, then Tb/Tbe
  bf16 *sT0 = sMsk + NMSK * C * LDC; // T ping
  bf16 *sT1 = sT0 + C * LDC;         // T pong
  bf16 *sX = sT1 + C * LDC;          // T P_s
  bf16 *sAtt = sX + C * LDC;         // att, staged for the flush
  float *sGl = (float *)(sAtt + C * LDC);
  float *sBe = sGl + C;
  float *sL = sBe + C;
  float *sEl = sL + C;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int hq = h / (HV / HQ);
  if (us == 51) return;            // bisection points for the kernel's prologue
  int seq = -1, loc = 0;
  for (int i = 0; i < N; ++i)
    if (gid >= sIx[i] && gid < sIx[i + 1]) { seq = i; loc = gid - sIx[i]; }
  if (seq < 0) return;
  const int s0 = (int)CU[seq], len = (int)CU[seq + 1] - s0;
  const int c0 = loc * C;
  const int nv = min(C, len - c0);
  if (nv <= 0) return;
  if (us == 52) return;
  const float alog = expf(ALOG[h]), dtb = DTB[h];
  if (us == 53) return;

  // Staged with cp.async: no registers, no store-to-shared, and the rows past
  // the end of a partial chunk are zero-filled by asking for zero source bytes
  // (the address is clamped so it stays inside the tensor either way).
  if (us > 1)
  for (int it = tid; it < C * (D / 8); it += NTH) {
    const int r = it / (D / 8), cc = (it % (D / 8)) * 8;
    const int rc = min(r, nv - 1), sz = (r < nv) ? 16 : 0;
    const size_t off = ((size_t)(s0 + c0 + rc) * HQ + hq) * D + cc;
    cp_async16s(sm32(sK + r * LDD + cc), K + off, sz);
    cp_async16s(sm32(sQ + r * LDD + cc), Q + off, sz);
    cp_async16s(sm32(sV + r * LDD + cc), V + ((size_t)(s0 + c0 + rc) * HV + h) * D + cc, sz);
  }
  if (tid < C) {
    float gl = 0.f, be = 0.f;
    if (tid < nv) {
      const size_t off = (size_t)(s0 + c0 + tid) * HV + h;
      gl = -alog * softplus(__bfloat162float(AA[off]) + dtb);
      be = 1.f / (1.f + __expf(-__bfloat162float(BB[off])));
    }
    sGl[tid] = gl;
    sBe[tid] = be;
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  if (us == 54) return;
  if (warp == 0) {
    float v = (lane < C) ? sGl[lane] : 0.f;
    float v2 = (C > 32 && lane + 32 < C) ? sGl[lane + 32] : 0.f;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float t = __shfl_up_sync(0xffffffff, v, d);
      if (lane >= d) v += t;
      if (C > 32) {
        const float t2 = __shfl_up_sync(0xffffffff, v2, d);
        if (lane >= d) v2 += t2;
      }
    }
    const float base = __shfl_sync(0xffffffff, v, 31);
    if (lane < C) { sL[lane] = v; sEl[lane] = __expf(v); }
    if (C > 32 && lane + 32 < C) {
      sL[lane + 32] = base + v2;
      sEl[lane + 32] = __expf(base + v2);
    }
  }
  __syncthreads();
  const float glast = sL[C - 1];
  if (tid == 0) GB[(size_t)gid * HV + h] = __expf(glast);
  if (us == 55) return;

  if (us > 2) {
    // K K^T and Q K^T share the right operand, and sK/sQ are adjacent with the
    // same pitch, so they are a single [2C][C] GEMM: twice the M tiles per warp,
    // hence 1.5x fewer shared-memory operand loads per mma, and this kernel is
    // shared-bandwidth bound.  Rows < C feed A, rows >= C are att.
    float accGA[TL<2 * C, C, NW>::NEL];
    gemm<2 * C, C, D, true, true, NW>(accGA, sK, LDD, sK, LDD, warp, lane);
    PAIRLOOP(2 * C, C, {
      if (r >= C) {
        const int rr = r - C;
        float va = 0.f, vb = 0.f;
        if (c <= rr) va = scale * accGA[e] * __expf(fminf(sL[rr] - sL[c], 0.f));
        if (c + 1 <= rr) vb = scale * accGA[e + 1] * __expf(fminf(sL[rr] - sL[c + 1], 0.f));
        st2(sAtt + rr * LDC + c, va, vb);
      } else {
        float a0 = 0.f, a1 = 0.f;
        if (c < r) a0 = sBe[r] * accGA[e] * __expf(fminf(sL[r] - sL[c], 0.f));
        if (c + 1 < r) a1 = sBe[r] * accGA[e + 1] * __expf(fminf(sL[r] - sL[c + 1], 0.f));
        const int l0 = (c < r) ? (31 - __clz(r ^ c)) : -1;
        const int l1 = (c + 1 < r) ? (31 - __clz(r ^ (c + 1))) : -1;
#pragma unroll
        for (int lv = 1; lv <= NMSK; ++lv)
          st2(sMsk + (size_t)(lv - 1) * C * LDC + r * LDC + c, l0 == lv ? a0 : 0.f,
              l1 == lv ? a1 : 0.f);
        float t0 = (r == c) ? 1.f : 0.f, t1v = (r == c + 1) ? 1.f : 0.f;
        if (l0 == 0) t0 = -a0;
        if (l1 == 0) t1v = -a1;
        st2(sT0 + r * LDC + c, t0, t1v);
      }
    })
  }
  __syncthreads();
  if (us > 3)                     // leaves Tb in sMsk[0] and Tbe in sMsk[1]
    tri_inv(sT0, sT1, sX, sMsk, sMsk, sMsk + C * LDC, sBe, sEl, nv, warp, lane);
  if (us > 4) {
    float accW[TL<C, D, NW>::NEL];
    gemm<C, D, C, false, true, NW>(accW, sMsk + C * LDC, LDC, sK, LDD, warp, lane);
    PAIRLOOP(C, D, { st2(sWb + r * LDD + c, accW[e], accW[e + 1]); })
  }
  if (us > 5) {
    float accU[TL<C, D, NW>::NEL];
    gemm<C, D, C, false, true, NW>(accU, sMsk, LDC, sV, LDD, warp, lane);
    __syncthreads();
    PAIRLOOP(C, D, { st2(sV + r * LDD + c, accU[e], accU[e + 1]); })
  }
  __syncthreads();
  // Pre-scale Qg = scale diag(e^L) Q and Kd = diag(e^{L_last-L}) K in place, so
  // the serial scan has no per-token gate arithmetic left at all.
  for (int it = tid; it < C * (D / 2); it += NTH) {
    const int r = it / (D / 2), cc = (it % (D / 2)) * 2;
    const float sq = scale * sEl[r], sk = __expf(glast - sL[r]);
    float a0, a1;
    ld2(sQ + r * LDD + cc, a0, a1);
    st2(sQ + r * LDD + cc, sq * a0, sq * a1);
    ld2(sK + r * LDD + cc, a0, a1);
    st2(sK + r * LDD + cc, sk * a0, sk * a1);
  }
  __syncthreads();
  if (us > 7) {
    const size_t gh = (size_t)gid * HV + h;
    bf16 *dst[4] = {WB + gh * C * D, UVB + gh * C * D, QGB + gh * C * D, KDB + gh * C * D};
    const bf16 *src[4] = {sWb, sV, sQ, sK};
#pragma unroll
    for (int a = 0; a < 4; ++a)
      for (int it = tid; it < C * (D / 8); it += NTH) {
        const int r = it / (D / 8), cc = (it % (D / 8)) * 8;
        *reinterpret_cast<uint4 *>(dst[a] + r * D + cc) =
            *reinterpret_cast<const uint4 *>(src[a] + r * LDD + cc);
      }
    bf16 *ap = ATB + gh * C * C;
    for (int it = tid; it < C * C / 8; it += NTH) {
      const int r = it / (C / 8), cc = (it % (C / 8)) * 8;
      *reinterpret_cast<uint4 *>(ap + r * C + cc) =
          *reinterpret_cast<const uint4 *>(sAtt + r * LDC + cc);
    }
  }
}

// Shared-memory prefix for the chunk-number table, 128-byte aligned so that
// everything after it keeps its alignment.
DEVI int ixbytes(int N) { return (((N + 2) * 4) + 127) & ~127; }

__global__ __launch_bounds__(NTH, MINBLK) void gdn_ut(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const float *__restrict__ ALOG, const bf16 *__restrict__ AA, const float *__restrict__ DTB,
    const bf16 *__restrict__ BB, const int64_t *__restrict__ CU, bf16 *__restrict__ WB,
    bf16 *__restrict__ UVB, bf16 *__restrict__ ATB, bf16 *__restrict__ QGB,
    bf16 *__restrict__ KDB, float *__restrict__ GB, float scale, int N, int us) {
  extern __shared__ char smem[];
  int *sIx = (int *)smem;
  chunk_prefix(CU, N, threadIdx.x, sIx);
  ut_body(smem + ixbytes(N), sIx, blockIdx.x, blockIdx.y, Q, K, V, ALOG, AA, DTB, BB, CU, WB,
          UVB, ATB, QGB, KDB, GB, scale, N, us);
}

// ---------------------------------------------------------------- kernel 2 --
// One (v-slice, head, sequence) chain of the state recursion, again as a device
// function so the fused cooperative kernel can call it after a grid barrier.
template <int DV>
DEVI void scan_body(char *smem, const int *sIx, int vb, int h, int sidx,
    const float *__restrict__ S0, const int64_t *__restrict__ CU, const bf16 *__restrict__ WB,
    const bf16 *__restrict__ UVB, const bf16 *__restrict__ ATB, const bf16 *__restrict__ QGB,
    const bf16 *__restrict__ KDB, const float *__restrict__ GB, bf16 *__restrict__ O,
    float *__restrict__ NS, int N, int stages) {
  constexpr int LDV = DV + 8;
  // The five per-chunk workspace arrays are staged with cp.async, AHEAD chunks
  // in flight: the copies cost no registers (which is what used to cap this
  // kernel at one resident block per SM) and they replace an explicit
  // load-to-register plus store-to-shared.  Staging is what this kernel is made
  // of -- with the GEMMs switched off it still takes 84% of the time, stalled on
  // global-memory latency at 31% of L2's peak -- so what matters is how many
  // copies are in flight, and two chunks' worth fits in the same number of
  // resident blocks as one.
  constexpr int STG = 3 * C * LDD + C * LDV + C * LDC;   // elements per buffer
  constexpr int NBUF = AHEAD + 1;
  bf16 *sShi = (bf16 *)smem;        // [DV][LDD]
  bf16 *sStg = sShi + DV * LDD;     // [NBUF][STG]:  Kd, Qg, W, Uv, att
  bf16 *sUT = sStg + NBUF * STG;    // [DV][LDC]  U transposed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int iv0 = vb * DV;
  const int s0 = (int)CU[sidx], n = (int)CU[sidx + 1] - s0;
  const bool live = (n > 0);
  const int gbase = sIx[sidx];

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
      st2(sShi + r * LDD + c, Sacc[e], Sacc[e + 1]);
    }
  }

  constexpr int TQ = C * (D / 8), TV = C * (DV / 8), TA = C * C / 8;
#define SCAN_STAGE(LOC, BUF)                                                            \
  {                                                                                     \
    const size_t gh_ = ((size_t)(gbase + (LOC)) * HV + h);                              \
    bf16 *d_ = sStg + (size_t)(BUF) * STG;                                              \
    const bf16 *src_[3] = {KDB + gh_ * C * D, QGB + gh_ * C * D, WB + gh_ * C * D};     \
    _Pragma("unroll") for (int a_ = 0; a_ < NSTG; ++a_) {                                  \
      for (int it = tid; it < TQ; it += NTH) {                                          \
        const int r_ = it / (D / 8), c_ = (it % (D / 8)) * 8;                           \
        cp_async16(sm32(d_ + (size_t)a_ * C * LDD + r_ * LDD + c_), src_[a_] + r_ * D + c_); \
      }                                                                                 \
    }                                                                                   \
    bf16 *du_ = d_ + 3 * C * LDD;                                                       \
    for (int it = tid; it < TV; it += NTH) {                                            \
      const int r_ = it / (DV / 8), c_ = (it % (DV / 8)) * 8;                           \
      cp_async16(sm32(du_ + r_ * LDV + c_), UVB + gh_ * C * D + r_ * D + iv0 + c_);     \
    }                                                                                   \
    bf16 *da_ = du_ + C * LDV;                                                          \
    for (int it = tid; it < TA; it += NTH) {                                            \
      const int r_ = it / (C / 8), c_ = (it % (C / 8)) * 8;                             \
      cp_async16(sm32(da_ + r_ * LDC + c_), ATB + gh_ * C * C + r_ * C + c_);           \
    }                                                                                   \
    cp_commit();                                                                        \
  }

  // Exactly one group is committed per chunk, empty if there is nothing left to
  // stage, so that cp.async's group count stays uniform and waiting for "all but
  // AHEAD-1" always means "the chunk I am about to use has landed".
#pragma unroll
  for (int j = 0; j < AHEAD; ++j) {
    if (j * C < n) SCAN_STAGE(j, j)
    else cp_commit();
  }
  cp_wait<AHEAD - 1>();
  __syncthreads();
  int loc = 0;
  for (int c0 = 0; c0 < n; c0 += C, ++loc) {
    const int nv = min(C, n - c0);
    const int buf = loc % NBUF;
    // e^{L_last} is wanted only by the state update at the end of the chunk, so
    // an ordinary load here has the whole chunk to cover its latency.
    const float egl = GB[(size_t)(gbase + loc) * HV + h];
    bf16 *sKd = sStg + (size_t)buf * STG;
    bf16 *sQg = sKd + C * LDD;
    bf16 *sW = sQg + C * LDD;
    bf16 *sUv = sW + C * LDD;          // (reused to stage the output)
    bf16 *sAtt = sUv + C * LDV;
    if (c0 + AHEAD * C < n) SCAN_STAGE(loc + AHEAD, (loc + AHEAD) % NBUF)
    else cp_commit();

    // W S_p and Q S_p are independent, so they issue as one segment; only att U
    // then has to wait for U = Uv - W S_p.
    float accO[TL<C, DV, NW>::NEL];
    if (stages > 1) {
      float accZ[TL<C, DV, NW>::NEL];
      gemm<C, DV, D, true, true, NW>(accZ, sW, LDD, sShi, LDD, warp, lane);
      gemm<C, DV, D, true, true, NW>(accO, sQg, LDD, sShi, LDD, warp, lane);
      PAIRLOOP(C, DV, {
        float u0, u1;
        ld2(sUv + r * LDV + c, u0, u1);
        sUT[c * LDC + r] = __float2bfloat16(u0 - accZ[e]);
        sUT[(c + 1) * LDC + r] = __float2bfloat16(u1 - accZ[e + 1]);
      })
    }
    __syncthreads();
    // O = Qg S_p + att U
    if (stages > 2) {
      gemm<C, DV, C, true, false, NW>(accO, sAtt, LDC, sUT, LDC, warp, lane);
      PAIRLOOP(C, DV, { st2(sUv + r * LDV + c, accO[e], accO[e + 1]); })
      __syncthreads();
      for (int it = tid; it < TV; it += NTH) {
        const int r = it / (DV / 8), cc = (it % (DV / 8)) * 8;
        if (r < nv)
          *reinterpret_cast<uint4 *>(O + ((size_t)(s0 + c0 + r) * HV + h) * D + iv0 + cc) =
              *reinterpret_cast<const uint4 *>(sUv + r * LDV + cc);
      }
    }
    // S <- e^{L_last} S + Kd^T U
    if (stages > 4) {
#pragma unroll
      for (int e = 0; e < SNEL; ++e) Sacc[e] *= egl;
      gemm<DV, D, C, false, false, NW>(Sacc, sUT, LDC, sKd, LDD, warp, lane);
    }
    if (stages > 5) {
#pragma unroll
      for (int e = 0; e < SNEL; e += 2) {
        int r, c;
        accpos<DV, D, NW>(e, warp, lane, r, c);
        st2(sShi + r * LDD + c, Sacc[e], Sacc[e + 1]);
      }
    }
    cp_wait<AHEAD - 1>();   // chunk loc+1's copies have had this chunk to fly
    __syncthreads();
  }
  {
    float *np = NS + ((size_t)sidx * HV + h) * D * D;
#pragma unroll
    for (int e = 0; e < SNEL; e += 2) {
      int r, c;
      accpos<DV, D, NW>(e, warp, lane, r, c);
      const float2 z = make_float2(live ? Sacc[e] : 0.f, live ? Sacc[e + 1] : 0.f);
      *reinterpret_cast<float2 *>(np + (size_t)(iv0 + r) * D + c) = z;
    }
  }
}

template <int DV>
__global__ __launch_bounds__(NTH, MINBLK) void gdn_scan(
    const float *__restrict__ S0, const int64_t *__restrict__ CU, const bf16 *__restrict__ WB,
    const bf16 *__restrict__ UVB, const bf16 *__restrict__ ATB, const bf16 *__restrict__ QGB,
    const bf16 *__restrict__ KDB, const float *__restrict__ GB, bf16 *__restrict__ O,
    float *__restrict__ NS, int N, int stages) {
  extern __shared__ char smem[];
  int *sIx = (int *)smem;
  chunk_prefix(CU, N, threadIdx.x, sIx);
  scan_body<DV>(smem + ixbytes(N), sIx, blockIdx.x, blockIdx.y, blockIdx.z, S0, CU, WB, UVB,
                ATB, QGB, KDB, GB, O, NS, N, stages);
}

// Both phases in one cooperative launch, separated by a grid-wide barrier.  A
// kernel launch measured the way this operator is measured (CUPTI span, cold L2)
// costs 4-6 us before it does any work, and there is another ~1.8 us gap between
// two launches on a stream, so for short and medium sequences that floor is most
// of the time.  One launch pays it once.  Each block walks its share of the
// chunks, then its share of the (v-slice, head, sequence) chains.
template <int DV>
__global__ __launch_bounds__(NTH, MINBLK) void gdn_coop(
    const bf16 *__restrict__ Q, const bf16 *__restrict__ K, const bf16 *__restrict__ V,
    const float *__restrict__ S0, const float *__restrict__ ALOG, const bf16 *__restrict__ AA,
    const float *__restrict__ DTB, const bf16 *__restrict__ BB, const int64_t *__restrict__ CU,
    bf16 *__restrict__ WB, bf16 *__restrict__ UVB, bf16 *__restrict__ ATB,
    bf16 *__restrict__ QGB, bf16 *__restrict__ KDB, float *__restrict__ GB,
    bf16 *__restrict__ O, float *__restrict__ NS, float scale, int N, int nchub, int stages) {
  extern __shared__ char smem[];
  int *sIx = (int *)smem;
  chunk_prefix(CU, N, threadIdx.x, sIx);
  char *base = smem + ixbytes(N);
  const int nut = nchub * HV;
  for (int job = blockIdx.x; job < nut; job += gridDim.x) {
    __syncthreads();                  // the previous job's shared memory is spent
    ut_body(base, sIx, job / HV, job % HV, Q, K, V, ALOG, AA, DTB, BB, CU, WB, UVB, ATB, QGB,
            KDB, GB, scale, N, stages);
  }
  if (stages == 200) return;        // the UT phase alone, for bisection
  cg::this_grid().sync();
  if (stages == 201) return;        // + the grid barrier
  const int nvb = D / DV, nsc = nvb * HV * N;
  for (int job = blockIdx.x; job < nsc; job += gridDim.x) {
    __syncthreads();
    scan_body<DV>(base, sIx, job % nvb, (job / nvb) % HV, job / (nvb * HV), S0, CU, WB, UVB,
                  ATB, QGB, KDB, GB, O, NS, N, stages);
  }
}

static size_t ixb(int N) { return (size_t)((((N + 2) * 4) + 127) & ~127); }

static size_t ut_smem(int N) {
  return ixb(N) + (size_t)4 * C * LDD * 2 + (size_t)(NMSK + 4) * C * LDC * 2 + 4 * C * 4 + 64;
}
template <int DV>
static size_t scan_smem(int N) {
  constexpr int LDV = DV + 8;
  return ixb(N) + (size_t)DV * LDD * 2 +
         (size_t)(AHEAD + 1) * (3 * C * LDD + C * LDV + C * LDC) * 2 + (size_t)DV * LDC * 2 + 256;
}

template <int DV>
static size_t coop_smem(int N) {
  const size_t a = ut_smem(N), b = scan_smem<DV>(N);
  return a > b ? a : b;
}

extern "C" int gdn_chunk_size() { return C; }

extern "C" size_t gdn_ws_bytes(int nchub) {
  return (size_t)nchub * HV * (4 * C * D + C * C) * 2 + (size_t)nchub * HV * 4 + 256;
}

extern "C" void gdn_launch_split(const void *q, const void *k, const void *v, const void *state,
                                 const void *alog, const void *a, const void *dtb, const void *b,
                                 const void *cu, void *o, void *ns, void *wsp, float scale, int T,
                                 int N, int nchub, int vsplit, int stages, cudaStream_t stream) {
  bf16 *WB = (bf16 *)wsp;
  bf16 *UVB = WB + (size_t)nchub * HV * C * D;
  bf16 *ATB = UVB + (size_t)nchub * HV * C * D;
  bf16 *QGB = ATB + (size_t)nchub * HV * C * C;
  bf16 *KDB = QGB + (size_t)nchub * HV * C * D;
  float *GB = (float *)(KDB + (size_t)nchub * HV * C * D);
  // stages: 99 = everything.  Lower values cut the kernels short for bisection
  // measurements; 100 runs gdn_ut alone, 101 gdn_scan alone, 110 + k gdn_scan
  // alone truncated to k, and 120 + k gdn_ut alone truncated to k.
  int uts = stages, scs = stages;
  if (stages == 100) scs = -1;
  else if (stages == 101) uts = 0;
  else if (stages >= 110 && stages < 120) { uts = 0; scs = stages - 110; }
  else if (stages >= 120) { uts = stages - 120; scs = -1; }
  if (uts > 0) {
    const size_t sm = ut_smem(N);
    static size_t done = 0;
    if (done < sm) {
      cudaFuncSetAttribute((void *)gdn_ut, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      done = sm;
    }
    gdn_ut<<<dim3(nchub, HV), NTH, sm, stream>>>(
        (const bf16 *)q, (const bf16 *)k, (const bf16 *)v, (const float *)alog, (const bf16 *)a,
        (const float *)dtb, (const bf16 *)b, (const int64_t *)cu, WB, UVB, ATB, QGB, KDB, GB,
        scale, N, stages);
  }
#define LAUNCH_SCAN(DVV)                                                                   \
  do {                                                                                     \
    const size_t sm = scan_smem<DVV>(N);                                                   \
    static size_t done = 0;                                                                \
    if (done < sm) {                                                                       \
      cudaFuncSetAttribute((void *)gdn_scan<DVV>,                                           \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);           \
      done = sm;                                                                           \
    }                                                                                      \
    gdn_scan<DVV><<<dim3(D / DVV, HV, N), NTH, sm, stream>>>(                               \
        (const float *)state, (const int64_t *)cu, WB, UVB, ATB, QGB, KDB, GB, (bf16 *)o,   \
        (float *)ns, N, scs);                                                               \
  } while (0)
  if (scs < 0) return;                     // gdn_ut alone
  if (vsplit == 1) LAUNCH_SCAN(128);
  else if (vsplit == 2) LAUNCH_SCAN(64);
  else if (vsplit == 4) LAUNCH_SCAN(32);
  else LAUNCH_SCAN(16);
#undef LAUNCH_SCAN
}

// Returns the grid it would use, or 0 if the fused cooperative launch is not
// usable for this shape (the whole grid has to be co-resident).
static int coop_grid(void *fn, size_t sm, int nut, int nsc) {
  int nb = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, NTH, sm) != cudaSuccess || nb < 1)
    return 0;
  static int nsm = 0;
  if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int need = nut > nsc ? nut : nsc;
  const int cap = nb * nsm;
  return need < cap ? need : cap;
}

extern "C" int gdn_launch_coop(const void *q, const void *k, const void *v, const void *state,
                               const void *alog, const void *a, const void *dtb, const void *b,
                               const void *cu, void *o, void *ns, void *wsp, float scale, int T,
                               int N, int nchub, int vsplit, int stages, cudaStream_t stream) {
  bf16 *WB = (bf16 *)wsp;
  bf16 *UVB = WB + (size_t)nchub * HV * C * D;
  bf16 *ATB = UVB + (size_t)nchub * HV * C * D;
  bf16 *QGB = ATB + (size_t)nchub * HV * C * C;
  bf16 *KDB = QGB + (size_t)nchub * HV * C * D;
  float *GB = (float *)(KDB + (size_t)nchub * HV * C * D);
#define LAUNCH_COOP(DVV)                                                                     \
  do {                                                                                        \
    const size_t sm = coop_smem<DVV>(N);                                                      \
    static size_t done = 0;                                                                   \
    if (done < sm) {                                                                          \
      cudaFuncSetAttribute((void *)gdn_coop<DVV>,                                             \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);             \
      done = sm;                                                                              \
    }                                                                                         \
    const int g = coop_grid((void *)gdn_coop<DVV>, sm, nchub * HV, (D / DVV) * HV * N);        \
    if (g <= 0) return 0;                                                                     \
    const bf16 *q_ = (const bf16 *)q, *k_ = (const bf16 *)k, *v_ = (const bf16 *)v;            \
    const float *s0_ = (const float *)state, *al_ = (const float *)alog;                       \
    const bf16 *a_ = (const bf16 *)a;                                                          \
    const float *dt_ = (const float *)dtb;                                                     \
    const bf16 *b_ = (const bf16 *)b;                                                          \
    const int64_t *cu_ = (const int64_t *)cu;                                                  \
    bf16 *o_ = (bf16 *)o;                                                                      \
    float *ns_ = (float *)ns;                                                                  \
    void *args[] = {&q_,  &k_,  &v_,  &s0_, &al_, &a_,  &dt_, &b_,     &cu_,    &WB,          \
                    &UVB, &ATB, &QGB, &KDB, &GB,  &o_,  &ns_, &scale,  &N,      &nchub,       \
                    &stages};                                                                  \
    if (cudaLaunchCooperativeKernel((void *)gdn_coop<DVV>, dim3(g), dim3(NTH), args, sm,        \
                                    stream) != cudaSuccess) {                                   \
      cudaGetLastError();            /* clear it, the caller falls back */                      \
      return 0;                                                                                \
    }                                                                                          \
    return 1;                                                                                  \
  } while (0)
  if (vsplit == 1) LAUNCH_COOP(128);
  else if (vsplit == 2) LAUNCH_COOP(64);
  else if (vsplit == 4) LAUNCH_COOP(32);
  else LAUNCH_COOP(16);
#undef LAUNCH_COOP
}

// A kernel that does nothing, to measure what a launch costs under the harness
// this operator is timed with (CUPTI span, cold L2 before every call).
__global__ void gdn_noop(float *o) {
  extern __shared__ char sm[];
  if (threadIdx.x == 1024) o[0] = (float)sm[0];
}

extern "C" void gdn_launch_noop(void *o, int blocks, int smem, int nk, cudaStream_t stream) {
  static int done = 0;
  if (done < smem) {
    cudaFuncSetAttribute((void *)gdn_noop, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    done = smem;
  }
  for (int i = 0; i < nk; ++i) gdn_noop<<<blocks, NTH, smem, stream>>>((float *)o);
}
