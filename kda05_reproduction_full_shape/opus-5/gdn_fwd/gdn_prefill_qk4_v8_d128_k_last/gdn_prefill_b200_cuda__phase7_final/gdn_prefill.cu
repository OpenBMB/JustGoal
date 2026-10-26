// Gated Delta Net prefill, k-last state layout, NVIDIA B200 (sm_100a).
//
// definition: gdn_prefill_qk4_v8_d128_k_last
//
// One CTA owns (sequence, value head, V-slice) and walks its sequence in
// chunks of C tokens, keeping the recurrent state in shared memory in the
// definition's own k-last layout S[V][K].
//
// Every bf16 matrix lives in shared memory in exactly ONE layout and is read
// either straight or transposed with `ldmatrix` / `ldmatrix.trans`, so there
// are no transposing stores and no duplicated buffers. Rows are padded by
// PADH bf16 so that the eight row addresses one `ldmatrix` octet supplies land
// in eight distinct 4-byte-bank groups (the row stride in 32-bit words is
// 4, 12 or 20 mod 32, all coprime-enough with 32 to cover all 32 banks).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Development-only ablation mask (-DGDN_ABL=...): each bit removes one stage so
// its share of the per-chunk critical path can be measured. Results are wrong
// for any non-zero value.
// Bring the initial state tile in with cp.async instead of a blocking
// global->register->shared copy, and build its bf16 mirror from shared memory
// afterwards.  dev/floor2.py measures the old path at 2.40us of a 6.64us
// T=6 workload; dev/floor3.py measures this one ~1us cheaper.
#ifndef GDN_CPST
#define GDN_CPST 1
#endif

// Dev-only: pad the main kernel's shared-memory block by this many KB, to buy
// an occupancy level without changing anything else.  Used to decide whether
// the main kernel is occupancy-limited before paying to restructure it.
#ifndef GDN_XPAD
#define GDN_XPAD 0
#endif

// Programmatic dependent launch: the main kernel's grid is launched while the
// pre-kernel is still running, so its prologue (the CTA -> sequence map, the
// state tile load and the first k/q/v cp.async -- none of which touch the
// workspace) overlaps the pre-kernel's tail, and `griddepcontrol.wait` is the
// only thing that waits.  Measured inter-kernel gap without it: 3-6 us of a
// 19-47 us span on the mid-size shapes.
//
// Both griddepcontrol instructions are emitted unconditionally: without the
// launch attribute, `wait` has nothing to wait for (ordinary stream ordering
// already happened) and `launch_dependents` has no dependent grid, so they are
// no-ops.  That makes PDL a pure launch-time choice, which is what allows a
// paired A/B inside one process -- and the measurements need it, because the
// split path's cost is bimodal *per process* (the same build measures 294/3 at
// 17.9 us in one run and 20.4 us in another, stably within each).

#ifndef GDN_ABL
#define GDN_ABL 0
#endif
#define ABL(bit) ((GDN_ABL) & (bit))

// Second ablation mask (-DGDN_ABL2=...), because the first one is out of bits.
// Phase 7 probes for the pre-kernel's 4.27us one-wave floor (docs/draft.md
// 23.2): 1 = return right after the index math (the pure launch floor),
// 2 = skip the W/P epilogue's masking/exp/stores while keeping its gemm,
// 4 = skip publishing the gate vectors.
#ifndef GDN_ABL2
#define GDN_ABL2 0
#endif
#define AB2(bit) ((GDN_ABL2) & (bit))

#define DEVI __device__ __forceinline__

namespace {

constexpr int HQ = 4;    // num_q_heads == num_k_heads
constexpr int HV = 8;    // num_v_heads
constexpr int DH = 128;  // head_size
constexpr int PADH = 8;  // bf16 row padding (elements)
constexpr int PADF = 4;  // fp32 row padding (elements)
constexpr int ORDER_MAXN = 512;  // max num_seqs for the ORDER=1 CTA remap

using bf16 = __nv_bfloat16;
using bf2 = __nv_bfloat162;

DEVI unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// 16-byte global->shared async copy. `bytes` of 0 makes the hardware
// zero-fill, which is exactly the padding a partial chunk needs.
DEVI void cp_async16(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)),
               "l"(src), "r"(bytes)
               : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
DEVI void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------- mma / ldmatrix

DEVI void mma16x8x16(float (&d)[4], const unsigned (&a)[4], const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEVI void ldm_x4(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
DEVI void ldm_x4t(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
DEVI void ldm_x2(unsigned (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}
DEVI void ldm_x2t(unsigned (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

// A-fragment of the 16x16 tile (m0, k0).
//   TR=false: source is [M][K] row-major; TR=true: source is [K][M] row-major.
// `ldmatrix.x4` lane l supplies row (l&7) of sub-matrix (l>>3); sub-matrix
// order is (m,k), (m+8,k), (m,k+8), (m+8,k+8) which is exactly a0..a3.
template <bool TR>
DEVI void frag_a(unsigned (&a)[4], const bf16* base, int ld, int m0, int k0, int lane) {
  const int sm_ = lane >> 3, row_in = lane & 7;
  if (TR) {
    const int row = k0 + (sm_ >> 1) * 8 + row_in, col = m0 + (sm_ & 1) * 8;
    ldm_x4t(a, smem_u32(base + (size_t)row * ld + col));
  } else {
    const int row = m0 + (sm_ & 1) * 8 + row_in, col = k0 + (sm_ >> 1) * 8;
    ldm_x4(a, smem_u32(base + (size_t)row * ld + col));
  }
}

// B-fragments of the two 8-column n-tiles (n0, n0+8) at k0, i.e. b[0..1] for
// n-tile n0 and b[2..3] for n-tile n0+8.
//   TR=false: source is [N][K] row-major; TR=true: source is [K][N] row-major.
template <bool TR>
DEVI void frag_b2(unsigned (&b)[4], const bf16* base, int ld, int n0, int k0, int lane) {
  const int sm_ = lane >> 3, row_in = lane & 7;
  if (TR) {
    const int row = k0 + (sm_ & 1) * 8 + row_in, col = n0 + (sm_ >> 1) * 8;
    ldm_x4t(b, smem_u32(base + (size_t)row * ld + col));
  } else {
    const int row = n0 + (sm_ >> 1) * 8 + row_in, col = k0 + (sm_ & 1) * 8;
    ldm_x4(b, smem_u32(base + (size_t)row * ld + col));
  }
}

// B-fragment of a single 8-column n-tile.
template <bool TR>
DEVI void frag_b1(unsigned (&b)[2], const bf16* base, int ld, int n0, int k0, int lane) {
  const int sm_ = (lane >> 3) & 1, row_in = lane & 7;
  if (TR) {
    ldm_x2t(b, smem_u32(base + (size_t)(k0 + sm_ * 8 + row_in) * ld + n0));
  } else {
    ldm_x2(b, smem_u32(base + (size_t)(n0 + row_in) * ld + k0 + sm_ * 8));
  }
}

// Per-row scale of an A-fragment (a0,a2 -> row; a1,a3 -> row+8).
DEVI void scale_a(unsigned (&a)[4], float s0, float s8) {
  const bf2 f0 = __float2bfloat162_rn(s0), f8 = __float2bfloat162_rn(s8);
  bf2* p = reinterpret_cast<bf2*>(a);
  p[0] = __hmul2(p[0], f0);
  p[1] = __hmul2(p[1], f8);
  p[2] = __hmul2(p[2], f0);
  p[3] = __hmul2(p[3], f8);
}

// Per-K scale of a pair of B-fragments: b[0],b[2] hold k and k+1, b[1],b[3]
// hold k+8 and k+9.
template <int NR>
DEVI void scale_b(unsigned (&b)[NR], float c0, float c1, float c8, float c9) {
  const bf2 lo = __halves2bfloat162(__float2bfloat16_rn(c0), __float2bfloat16_rn(c1));
  const bf2 hi = __halves2bfloat162(__float2bfloat16_rn(c8), __float2bfloat16_rn(c9));
  bf2* p = reinterpret_cast<bf2*>(b);
#pragma unroll
  for (int i = 0; i < NR / 2; ++i) {
    p[2 * i] = __hmul2(p[2 * i], lo);
    p[2 * i + 1] = __hmul2(p[2 * i + 1], hi);
  }
}

// ------------------------------------------------------------- tile copies

// cp.async a C x (W*8) bf16 tile, advancing pointers instead of recomputing a
// 64-bit address per row.  The naive loop's `(size_t)(s0+cc+r) * stride` (with
// a select on r < nv to keep masked rows in bounds) was ~12% of the main
// kernel's pcsamp samples; here the row index, the global pointer and the
// shared pointer are all strength-reduced to adds.
//
// Rows >= nv get a source size of 0, which cp.async zero-fills -- exactly the
// masking a partial chunk needs -- and the source pointer is never advanced
// past the last valid row, so a masked copy's address stays inside the tensor.
// `src` must already point at the tile's row 0 (or at a safe row when nv == 0).
template <int C, int W, int NTHR>
DEVI void tile_cp(bf16* dst, int ldd, const bf16* src, int srcstride, int nv,
                  int tid) {
  constexpr int RSTEP = NTHR / W;            // rows advanced per iteration
  constexpr int NITER = (C + RSTEP - 1) / RSTEP;
  static_assert(NTHR % W == 0, "a row must be covered by a whole set of lanes");
  const int r0 = tid / W, c = (tid % W) * 8;
  if (r0 >= C) return;
  const bf16* gp = src + (size_t)((r0 < nv) ? r0 : 0) * srcstride + c;
  bf16* sp = dst + r0 * ldd + c;
#pragma unroll
  for (int it = 0; it < NITER; ++it) {
    const int r = r0 + it * RSTEP;
    const int nb = (r < nv) ? 16 : 0;
    if (NITER == 1 || r < C) cp_async16(sp, gp, nb);
    gp += ((r + RSTEP) < nv) ? RSTEP * srcstride : 0;
    sp += RSTEP * ldd;
  }
}

// Same, for a fully contiguous source block of C rows of `W*8` elements (the
// per-chunk Tm / P tiles in the workspace), so there is no masking at all.
template <int C, int W, int NTHR>
DEVI void tile_cp_full(bf16* dst, int ldd, const bf16* src, int srcstride, int tid) {
  constexpr int RSTEP = NTHR / W;
  constexpr int NITER = (C + RSTEP - 1) / RSTEP;
  static_assert(NTHR % W == 0, "a row must be covered by a whole set of lanes");
  const int r0 = tid / W, c = (tid % W) * 8;
  if (r0 >= C) return;
  const bf16* gp = src + (size_t)r0 * srcstride + c;
  bf16* sp = dst + r0 * ldd + c;
#pragma unroll
  for (int it = 0; it < NITER; ++it) {
    if (NITER == 1 || r0 + it * RSTEP < C) cp_async16(sp, gp, 16);
    gp += RSTEP * srcstride;
    sp += RSTEP * ldd;
  }
}

// ------------------------------------------------------------- warp tiling

// Pick the per-warp output tile (WMG m-tiles x WNG n-tiles) that minimises the
// number of ldmatrix instructions per mma, i.e. WMG + ceil(WNG/2) for
// WMG*WNG mmas, subject to tiling the MxN output exactly over NWARP warps.
constexpr int pick_wmg(int MT, int NT, int NWARP) {
  int best = 0, bestcost = 1 << 20;
  for (int wmg = 1; wmg <= MT; ++wmg) {
    if (MT % wmg) continue;
    const int mg = MT / wmg;
    if (NWARP % mg) continue;
    const int ng = NWARP / mg;
    if (ng > NT || NT % ng) continue;
    const int wng = NT / ng;
    const int cost = (wmg + (wng + 1) / 2) * 1024 / (wmg * wng);
    if (cost < bestcost) {
      bestcost = cost;
      best = wmg;
    }
  }
  return best;
}

template <int M, int N, int NWARP>
struct Tile {
  static constexpr int MT = M / 16;
  static constexpr int NT = N / 8;
  static constexpr int WMG = pick_wmg(MT, NT, NWARP);
  static constexpr int MG = MT / (WMG ? WMG : 1);          // m-groups of warps
  static constexpr int NG = NWARP / (MG ? MG : 1);         // n-groups of warps
  static constexpr int WNG = NT / (NG ? NG : 1);           // n-tiles per warp
  static_assert(WMG > 0, "no valid warp tiling for this (M,N,NWARP)");
};

#define TILE_IDS(T)                       \
  const int wm0 = (warp % T::MG) * T::WMG; \
  const int wn0 = (warp / T::MG) * T::WNG

// acc[WMG][WNG][4] += A[M][K] * B[N][K]^T.  TA/TB select a transposed source.
template <int M, int N, int K, int NWARP, bool TA, bool TB>
DEVI void gemm(float (*acc)[Tile<M, N, NWARP>::WNG][4], const bf16* A, int lda,
               const bf16* B, int ldb, int warp, int lane) {
  using T = Tile<M, N, NWARP>;
  TILE_IDS(T);
#pragma unroll
  for (int k0 = 0; k0 < K; k0 += 16) {
    unsigned bf[T::WNG][2];
#pragma unroll
    for (int ni = 0; ni + 1 < T::WNG; ni += 2)
      frag_b2<TB>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), B, ldb, (wn0 + ni) * 8, k0, lane);
    if (T::WNG & 1)
      frag_b1<TB>(bf[T::WNG - 1], B, ldb, (wn0 + T::WNG - 1) * 8, k0, lane);
#pragma unroll
    for (int mi = 0; mi < T::WMG; ++mi) {
      unsigned af[4];
      frag_a<TA>(af, A, lda, (wm0 + mi) * 16, k0, lane);
#pragma unroll
      for (int ni = 0; ni < T::WNG; ++ni) mma16x8x16(acc[mi][ni], af, bf[ni]);
    }
  }
}

template <int M, int N, int NWARP>
DEVI void zero_acc(float (*acc)[Tile<M, N, NWARP>::WNG][4]) {
  using T = Tile<M, N, NWARP>;
#pragma unroll
  for (int mi = 0; mi < T::WMG; ++mi)
#pragma unroll
    for (int ni = 0; ni < T::WNG; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.f;
}

// Row/col of accumulator element e of tile (mi, ni) for this lane.
#define ACC_ROW(T, mi, e) ((wm0 + (mi)) * 16 + (lane >> 2) + ((e) >> 1) * 8)
#define ACC_COL(T, ni, e) ((wn0 + (ni)) * 8 + (lane & 3) * 2 + ((e) & 1))

// A <- I, as one 16-byte store per thread rather than C*C/NTHR scalar 2-byte
// stores (which also cost a div and a mod each, and bank-conflict).  The value
// is built arithmetically so nothing is address-taken -- an indexed write into
// a local array would spill the whole thing to local memory.
//
// This is the identity seed of the triangular inverse, and it depends on
// nothing: not on the k/q tiles, not on the gates, not on the Gram.  Phase 7's
// per-line profile of the pre-kernel puts this loop at 11.4 % of samples (its
// top line) sitting *after* the Gram, while the cold `cp.async` wait for the
// k/q tiles immediately before the Gram stalls at 7.4 % with nothing to fill
// it.  With AIH = 1 the call is hoisted above that wait.
template <int C, int NTHR>
DEVI void init_identity(bf16* A, int LD, int tid, bool off) {
  for (int i = off ? C * (C / 8) : tid; i < C * (C / 8); i += NTHR) {
    const int r = i / (C / 8), c = (i % (C / 8)) * 8;
    const int k = r - c;  // 0..7 exactly on the diagonal's 8-element run
    const unsigned val = ((unsigned)k < 8u) ? (0x3F80u << ((k & 1) * 16)) : 0u;
    const int w = (unsigned)k < 8u ? (k >> 1) : -1;
    int4 v;
    v.x = (w == 0) ? (int)val : 0;
    v.y = (w == 1) ? (int)val : 0;
    v.z = (w == 2) ? (int)val : 0;
    v.w = (w == 3) ? (int)val : 0;
    *reinterpret_cast<int4*>(&A[r * LD + c]) = v;
  }
}

// --------------------------------------------- triangular inverse, small levels

// The 16x16 diagonal blocks of (I+W)^-1, one block per warp, by forward
// substitution with lane j owning column j of its block:
//
//     T[j][j] = 1 ;  T[i][j] = -sum_{p=j..i-1} W[i][p] T[p][j]   (i > j)
//
// The whole column stays in registers, so this is ONE barrier-free pass where
// the doubling levels mm = 1, 2, 4, 8 were four two-stage passes and eight
// __syncthreads() -- measured at 20% of the fused kernel's per-chunk time.  It
// is also strictly more accurate: the doubling levels round A to bf16 between
// levels, this accumulates the whole column in fp32.
//
// W[i][p] is the same address for every lane, so each load is a broadcast.
// Rows at or past nvalid have W = 0 (the W epilogue masked them), so their
// columns come out as the identity, exactly as the doubling levels left them.
// The 16x16 diagonal blocks of (I+W)^-1, one block per warp, by forward
// substitution with lane j owning column j of its block:
//
//     T[j][j] = 1 ;  T[i][j] = -sum_{p=j..i-1} W[i][p] T[p][j]   (i > j)
//
// The whole column stays in registers, so this is ONE barrier-free pass where
// the doubling levels mm = 1, 2, 4, 8 were four two-stage passes and eight
// __syncthreads().  It is also strictly more accurate: the doubling levels
// round A to bf16 between levels, this accumulates each column in fp32.
//
// W[i][p] is the same address for every lane, so each load is a broadcast, and
// the loop is fully unrolled so ptxas can hoist all 120 of them across the
// chain.  Stopping the chain at nvalid (a uniform `break`, which saves up to
// 8x of the FFMAs on a partial chunk) was built and measured: it is 1.04x at
// T=6 and 2-5% SLOWER on full chunks, because the per-iteration branch stops
// the loads being scheduled ahead -- and splitting it into a FULL and a partial
// instantiation does not recover that either (the extra code costs ~2% even on
// the shapes that take the unchanged path).
template <int C, int NWARP>
DEVI void inv_diag16(bf16* A, const bf16* Wm, int LD, int nvalid, int warp, int lane) {
  constexpr int NBLK = C / 16;
  static_assert(NBLK <= NWARP, "one warp per 16x16 diagonal block");
  if (warp >= NBLK) return;
  const int base = warp * 16;
  if (base >= nvalid) return;  // block is entirely masked; A is already I there
  const int j = lane & 15;     // lanes 16..31 recompute the same columns
  float t[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) t[i] = (i == j) ? 1.f : 0.f;
#pragma unroll
  for (int i = 1; i < 16; ++i) {
    float sum = 0.f;
#pragma unroll
    for (int pp = 0; pp < i; ++pp)
      sum = fmaf(__bfloat162float(Wm[(base + i) * LD + base + pp]), t[pp], sum);
    if (i > j) t[i] = -sum;
  }
  if (lane < 16) {
#pragma unroll
    for (int i = 0; i < 16; ++i)
      A[(base + i) * LD + base + j] = __float2bfloat16_rn(t[i]);
  }
}

// ----------------------------------------------- block-sparse triangular inverse

// One level of the block-recursive 2x2 inversion of L = I + W, done
// block-sparse and in place.
//
// A holds L^-1 for the mm x mm diagonal blocks (everything else zero).  For the
// pair at base = 2*p*mm, with A11 = A[base:+mm, base:+mm],
// A22 = A[base+mm:+mm, base+mm:+mm] and L21 = W[base+mm:+mm, base:+mm],
//
//     A[base+mm:+mm, base:+mm]  =  -A22 . L21 . A11 ,
//
// two mm x mm x mm matmuls.  The target block is still zero and nothing in the
// level reads it, so there is no Acur/Anxt ping-pong.  The Phase 2 version got
// the same band out of two masked C x C x C matmuls, i.e. 2*C^3 FLOP per level
// where the mathematics needs 2*C*mm^2 -- 8x to 128x more, and 60% of the
// pre-kernel's mma instructions.
//
// Warps are split over pairs (WPP per pair) rather than over the full width, so
// every level is a handful of mma instructions: 8 at (C=64, mm=16), 32 at
// (C=64, mm=32), 4 at (C=32, mm=16).
// Split into two barrier-free halves so that a caller which interleaves this
// with an independent instruction stream can place the barriers itself (the
// pipelined fused kernel does; `inv_level` below keeps the original shape).
// `act` already contains the `base + MM < nvalid` test, so an empty band makes
// both halves no-ops without an early return -- which is what lets the
// pipelined path keep a compile-time-constant barrier count per chunk.
template <int C, int MM, int NWARP>
DEVI void inv_level_a(bf16* A, const bf16* Wm, bf16* Z, int LD, int nvalid,
                      int warp_in, int lane) {
  static_assert(MM >= 16, "block-sparse inverse levels need mm >= 16 (MM_ELEM >= 8)");
  constexpr int NPAIR = C / (2 * MM);
  constexpr int TILES = (MM / 16) * (MM / 8);
  constexpr int WR = NWARP / NPAIR;
  constexpr int WPP = WR < 1 ? 1 : (WR > TILES ? TILES : WR);
  using T = Tile<MM, MM, WPP>;
  const int p = warp_in / WPP;
  const int warp = warp_in % WPP;
  const int base = 2 * p * MM;
  const bool act = (p < NPAIR) && (base + MM < nvalid);
  TILE_IDS(T);
  if (!act) return;
  float acc[T::WMG][T::WNG][4];  // Z = A22 . L21
  zero_acc<MM, MM, WPP>(acc);
  gemm<MM, MM, MM, WPP, false, true>(acc, A + (base + MM) * LD + base + MM, LD,
                                     Wm + (base + MM) * LD + base, LD, warp, lane);
#pragma unroll
  for (int mi = 0; mi < T::WMG; ++mi)
#pragma unroll
    for (int ni = 0; ni < T::WNG; ++ni)
#pragma unroll
      for (int half = 0; half < 2; ++half)
        *reinterpret_cast<bf2*>(
            &Z[(base + MM + ACC_ROW(T, mi, half * 2)) * LD + base +
               ACC_COL(T, ni, 0)]) =
            __floats2bfloat162_rn(acc[mi][ni][half * 2], acc[mi][ni][half * 2 + 1]);
}

template <int C, int MM, int NWARP>
DEVI void inv_level_b(bf16* A, const bf16* Wm, const bf16* Z, int LD, int nvalid,
                      int warp_in, int lane) {
  constexpr int NPAIR = C / (2 * MM);
  constexpr int TILES = (MM / 16) * (MM / 8);
  constexpr int WR = NWARP / NPAIR;
  constexpr int WPP = WR < 1 ? 1 : (WR > TILES ? TILES : WR);
  using T = Tile<MM, MM, WPP>;
  const int p = warp_in / WPP;
  const int warp = warp_in % WPP;
  const int base = 2 * p * MM;
  const bool act = (p < NPAIR) && (base + MM < nvalid);
  TILE_IDS(T);
  if (!act) return;
  float acc[T::WMG][T::WNG][4];  // A[.] = -Z . A11
  zero_acc<MM, MM, WPP>(acc);
  gemm<MM, MM, MM, WPP, false, true>(acc, Z + (base + MM) * LD + base, LD,
                                     A + base * LD + base, LD, warp, lane);
#pragma unroll
  for (int mi = 0; mi < T::WMG; ++mi)
#pragma unroll
    for (int ni = 0; ni < T::WNG; ++ni)
#pragma unroll
      for (int half = 0; half < 2; ++half)
        *reinterpret_cast<bf2*>(
            &A[(base + MM + ACC_ROW(T, mi, half * 2)) * LD + base +
               ACC_COL(T, ni, 0)]) =
            __floats2bfloat162_rn(-acc[mi][ni][half * 2], -acc[mi][ni][half * 2 + 1]);
}

template <int C, int MM, int NWARP>
DEVI void inv_level(bf16* A, const bf16* Wm, bf16* Z, int LD, int nvalid, int warp_in,
                    int lane) {
  inv_level_a<C, MM, NWARP>(A, Wm, Z, LD, nvalid, warp_in, lane);
  __syncthreads();
  inv_level_b<C, MM, NWARP>(A, Wm, Z, LD, nvalid, warp_in, lane);
  __syncthreads();
}

// Levels MM0, 2*MM0, ... < C.  mm >= nvalid makes the band empty, and nvalid is
// CTA-uniform, so the early return is the same uniform skip the elementwise
// levels use.
template <int C, int MM, int NWARP>
DEVI void inv_mm_levels(bf16* A, const bf16* Wm, bf16* Z, int LD, int nvalid, int warp,
                        int lane) {
  if constexpr (MM < C) {
    if (MM >= nvalid) return;
    inv_level<C, MM, NWARP>(A, Wm, Z, LD, nvalid, warp, lane);
    inv_mm_levels<C, 2 * MM, NWARP>(A, Wm, Z, LD, nvalid, warp, lane);
  }
}

// -------------------------------------------------------------- shared memory

template <int C, int VB>
struct alignas(16) Smem {
  static constexpr int LDK = DH + PADH;   // k, q   : [C][DH]
  static constexpr int LDV = VB + PADH;   // v/rhs/o: [C][VB]
  static constexpr int LDSB = DH + PADH;  // Sb     : [VB][DH] bf16
  static constexpr int LDC = C + PADH;    // CxC
  static constexpr int LDU = VB + PADH;   // U      : [C][VB]

  static constexpr int N_SB = VB * LDSB;
  static constexpr int N_K = C * LDK;
  static constexpr int N_V = C * LDV;
  static constexpr int N_CC = C * LDC;
  static constexpr int N_U = C * LDU;
  // phase 1 wants {W, Z}; phase 2 wants {U}
  static constexpr int N_UNION = (2 * N_CC > N_U) ? 2 * N_CC : N_U;

  bf16 Sb[N_SB];
  bf16 K[2 * N_K];  // double-buffered: cp.async fills [buf^1] while [buf] is used
  bf16 Q[2 * N_K];
  bf16 V[2 * N_V];  // v tile -> rhs (in place) -> output staging
  bf16 A[N_CC];
  bf16 P[N_CC];
  bf16 Ud[N_U];  // U with dkd folded into its rows (the state update's A operand)
  bf16 un[N_UNION];
  float cl[C], beta[C], dcl[C], dkd[C], scal[4];  // scal[0] = d_last
};

// Write the bf16 mirror of the register-resident fp32 state.  Sb is the B
// operand ([N=v][K=k]) of k.S and q.S; the fp32 copy never leaves registers.
template <int C, int VB, int NWARP, class SMT>
DEVI void sreg_to_Sb(SMT& sm,
                     const float (&sreg)[Tile<VB, DH, NWARP>::WMG]
                                        [Tile<VB, DH, NWARP>::WNG][4],
                     int sm0, int sn0, int lane) {
  using TVD = Tile<VB, DH, NWARP>;
  using SM = SMT;
#pragma unroll
  for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
    for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int vv = (sm0 + mi) * 16 + (lane >> 2) + half * 8;
        const int kk = (sn0 + ni) * 8 + (lane & 3) * 2;
        *reinterpret_cast<bf2*>(&sm.Sb[vv * SM::LDSB + kk]) = __floats2bfloat162_rn(
            sreg[mi][ni][half * 2], sreg[mi][ni][half * 2 + 1]);
      }
}

DEVI float softplus(float x) {
  // matches torch.nn.functional.softplus to fp32 without overflowing
  return fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x)));
}

// ------------------------------------------------------------------ the kernel

template <int C, int VB, int NWARP, int MINB, int MM_ELEM, int BAR = 1,
          int AIH = 0>
__global__ __launch_bounds__(NWARP * 32, MINB) void gdn_kernel(
    const bf16* __restrict__ gq, const bf16* __restrict__ gk,
    const bf16* __restrict__ gv, const float* __restrict__ gstate,
    const float* __restrict__ gAlog, const bf16* __restrict__ ga,
    const float* __restrict__ gdtb, const bf16* __restrict__ gb,
    const int64_t* __restrict__ gcu, bf16* __restrict__ gout,
    float* __restrict__ gns, float scale, int has_state, int num_seqs,
    int total_seq_len, int tri) {
  using SM = Smem<C, VB>;
  extern __shared__ char smem_raw[];
  SM& sm = *reinterpret_cast<SM*>(smem_raw);

  const int n = blockIdx.x, h = blockIdx.y, vs = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int NTHR = NWARP * 32;
  using TVD = Tile<VB, DH, NWARP>;
  const int hq = h >> 1;
  const int v0 = vs * VB;

  // The state tile address depends only on the block index, so issue it before
  // touching cu_seqlens -- otherwise the whole CTA waits out a dependent global
  // round trip before its first useful load. That latency is ~0.5us, which is
  // 7% of a one-chunk workload.
  // The fp32 state lives in REGISTERS for the whole sequence, in the
  // accumulator layout of S <- d_last*S + U^T.Kd, so that gemm accumulates onto
  // it: no fp32 shared array and no read-modify-write epilogue (which the
  // per-line profile had at 9.4% of this kernel).  Shared memory keeps only the
  // bf16 mirror, which is the B operand of k.S and q.S.
  const int sm0 = (warp % TVD::MG) * TVD::WMG;
  const int sn0 = (warp / TVD::MG) * TVD::WNG;
  float sreg[TVD::WMG][TVD::WNG][4];
#define SREG_V(mi, e) ((sm0 + (mi)) * 16 + (lane >> 2) + (((e) >> 1) * 8))
#define SREG_K(ni) ((sn0 + (ni)) * 8 + (lane & 3) * 2)
  {
    const float* src = gstate + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          float2 x = make_float2(0.f, 0.f);
          if (has_state && !ABL(268435456))
            x = *reinterpret_cast<const float2*>(
                src + (size_t)SREG_V(mi, e) * DH + SREG_K(ni));
          sreg[mi][ni][e] = x.x;
          sreg[mi][ni][e + 1] = x.y;
        }
  }

  // With one sequence, cu_seqlens is {0, total_seq_len} -- both host-known -- so
  // the dependent (and L2-cold) gcu load disappears from the kernel's critical
  // path entirely.  20 of the 100 workloads have num_seqs == 1.
  const bool one = num_seqs == 1;
  const int s0 = one ? 0 : (int)gcu[n];
  const int seqlen = one ? total_seq_len : ((int)gcu[n + 1] - s0);

  // the reference leaves new_state at zero for empty sequences
  if (seqlen <= 0) zero_acc<VB, DH, NWARP>(sreg);

  // phase-1 (inverse) and phase-2 (U) views of the aliased scratch region
  bf16* sW = sm.un;
  bf16* sZ = sm.un + SM::N_CC;

  bf16* sU = sm.un;

  const float nexp_alog = -__expf(gAlog[h]);
  const float dtb = gdtb[h];

  using TCC = Tile<C, C, NWARP>;
  using TCV = Tile<C, VB, NWARP>;

  // Issue the chunk-0 tile copies, then one chunk ahead on every iteration.
  auto issue_tiles = [&](int cc, int buf) {
    const int nv = (cc < seqlen && !ABL(128)) ? min(C, seqlen - cc) : 0;
    const int t0 = s0 + (nv ? cc : 0);  // tile row 0, safe when nv == 0
    tile_cp<C, DH / 8, NTHR>(sm.K + buf * SM::N_K, SM::LDK,
                             gk + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, DH / 8, NTHR>(sm.Q + buf * SM::N_K, SM::LDK,
                             gq + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, VB / 8, NTHR>(sm.V + buf * SM::N_V, SM::LDV,
                             gv + (size_t)t0 * (HV * DH) + (size_t)h * DH + v0,
                             HV * DH, nv, tid);
    cp_async_commit();
  };
  if (seqlen > 0) issue_tiles(0, 0);

  // Gate inputs are 2-byte strided reads (stride HV), so their latency does not
  // hide behind cp.async; warp 0 keeps one chunk of a/b in registers ahead.
  constexpr int PER = (C + 31) / 32;  // lanes >= C contribute 0
  float a_pf[PER], b_pf[PER];
  if (warp == 0) {
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int j = p * 32 + lane;
      const int t = s0 + j;
      const bool ok = j < C && j < seqlen && !ABL(256);
      a_pf[p] = ok ? __bfloat162float(ga[(size_t)t * HV + h]) : 0.f;
      b_pf[p] = ok ? __bfloat162float(gb[(size_t)t * HV + h]) : 0.f;
    }
  }

  int buf = 0;
  for (int c0 = 0; c0 < (ABL(134217728) ? 0 : seqlen); c0 += C, buf ^= 1) {
    const int nvalid = min(C, seqlen - c0);
    bf16* bK = sm.K + buf * SM::N_K;
    bf16* bQ = sm.Q + buf * SM::N_K;
    bf16* bV = sm.V + buf * SM::N_V;
    // Order matters here. The last reader of buffer `buf^1` is the *previous*
    // iteration (its state update reads k through ldmatrix.trans), and this
    // iteration's prefetch overwrites that buffer -- so the copies must be
    // issued after the barrier that closes the previous iteration, not before
    // it. With the issue moved after the wait, waiting for *all* groups is what
    // keeps exactly one chunk of prefetch distance: the group issued below
    // overlaps this whole chunk's compute and is waited for one iteration later.
    // The identity seed of this chunk's inverse depends on nothing, so with
    // AIH = 1 it runs here and fills the cold tile wait below instead of
    // sitting between the Gram epilogue and inv_diag16.  sm.A holds the
    // previous chunk's Tm, whose last reader (U = Tm.rhs) finished before the
    // barrier that closed that iteration.
    if constexpr (AIH)
      init_identity<C, NTHR>(sm.A, SM::LDC, tid, ABL(268435456));
    cp_async_wait<0>();
    __syncthreads();
    issue_tiles(c0 + C, buf ^ 1);

    // The bf16 mirror of the initial state.  Placed here so the global loads
    // that fill sreg have the whole gate block to land in, and so the barrier
    // that closes the gate block publishes it.
    if (c0 == 0) sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);

    // ---- warp 0 computes the gates and the inclusive cumsum of log g in-warp
    if (warp == 0) {
      float a_cur[PER], b_cur[PER];
#pragma unroll
      for (int p = 0; p < PER; ++p) {
        a_cur[p] = a_pf[p];
        b_cur[p] = b_pf[p];
      }
#pragma unroll
      for (int p = 0; p < PER; ++p) {  // issue the next chunk's gate reads early
        const int j = c0 + C + p * 32 + lane;
        const int t = s0 + j;
        const bool ok = (p * 32 + lane) < C && j < seqlen && !ABL(256);
        a_pf[p] = ok ? __bfloat162float(ga[(size_t)t * HV + h]) : 0.f;
        b_pf[p] = ok ? __bfloat162float(gb[(size_t)t * HV + h]) : 0.f;
      }
      float carry = 0.f;
#pragma unroll
      for (int p = 0; p < PER; ++p) {
        const int j = p * 32 + lane;
        float lg = 0.f, be = 0.f;
        if (j < nvalid) {
          lg = nexp_alog * softplus(a_cur[p] + dtb);
          be = 1.f / (1.f + __expf(-b_cur[p]));
        }
        if (j < C) sm.beta[j] = be;
        float x = lg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const float y = __shfl_up_sync(0xffffffffu, x, d);
          if (lane >= d) x += y;
        }
        x += carry;
        if (j < C) {
          sm.cl[j] = x;
          sm.dcl[j] = __expf(x);
        }
        carry = __shfl_sync(0xffffffffu, x, 31);
      }
      const float cl_last = carry;
#pragma unroll
      for (int p = 0; p < PER; ++p) {
        const int j = p * 32 + lane;
        if (j < C) sm.dkd[j] = __expf(fminf(cl_last - sm.cl[j], 0.f));
      }
      if (lane == 0) sm.scal[0] = __expf(cl_last);  // d_last, broadcast via smem
    }
    __syncthreads();
    const float d_last = sm.scal[0];

    // ---- W = strict_lower( beta_i (k_i.k_j) E(i,j) ) -> sW
    //      P = lower_incl(        (q_i.k_j) E(i,j) ) -> sm.P
    // Both contract against the same B operand (k) and need the same E(i,j), so
    // the B fragments are fetched once and the exp is evaluated once.
    {
      float wacc[TCC::WMG][TCC::WNG][4], pacc[TCC::WMG][TCC::WNG][4];
      zero_acc<C, C, NWARP>(wacc);
      zero_acc<C, C, NWARP>(pacc);
      TILE_IDS(TCC);
      // Both outputs of this gemm are masked -- W by `ii > jj`, P by `ii >= jj` --
      // so a warp whose entire output block lies in the strict upper triangle
      // writes zeros no matter what the gemm produced.  For every instantiated
      // (C, NWARP) here, exactly one warp in four is in that position
      // (C=32/NWARP=4: warp 2; C=64/NWARP=4: warp 2; C=64/NWARP=8: warps 4, 5),
      // so this skips a quarter of the Gram work.  The epilogue below still runs
      // and still writes the zeros.  This is the shape of CTA-uniform skip that
      // pays: it removes a whole stage, its `ldmatrix` loads included, rather
      // than interrupting a scheduled dependency chain the way the rejected
      // `inv_diag16` early exit did.
      const bool tri_dead = tri && (wm0 + TCC::WMG - 1) * 16 + 15 < wn0 * 8;
      if (!tri_dead) {
#pragma unroll
      for (int k0 = 0; k0 < (ABL(2) ? 0 : DH); k0 += 16) {
        unsigned bf[TCC::WNG][2];
#pragma unroll
        for (int ni = 0; ni + 1 < TCC::WNG; ni += 2)
          frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bK, SM::LDK,
                         (wn0 + ni) * 8, k0, lane);
        if (TCC::WNG & 1)
          frag_b1<false>(bf[TCC::WNG - 1], bK, SM::LDK, (wn0 + TCC::WNG - 1) * 8, k0,
                         lane);
#pragma unroll
        for (int mi = 0; mi < TCC::WMG; ++mi) {
          unsigned ak[4], aq[4];
          frag_a<false>(ak, bK, SM::LDK, (wm0 + mi) * 16, k0, lane);
          frag_a<false>(aq, bQ, SM::LDK, (wm0 + mi) * 16, k0, lane);
#pragma unroll
          for (int ni = 0; ni < TCC::WNG; ++ni) {
            mma16x8x16(wacc[mi][ni], ak, bf[ni]);
            mma16x8x16(pacc[mi][ni], aq, bf[ni]);
          }
        }
      }
      }
#pragma unroll
      for (int mi = 0; mi < TCC::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TCC::WNG; ++ni)
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int ii = ACC_ROW(TCC, mi, half * 2);
            const int jj = ACC_COL(TCC, ni, 0);
            const float ci = sm.cl[ii], bi = sm.beta[ii];
            const float e0 = __expf(fminf(ci - sm.cl[jj], 0.f));
            const float e1 = __expf(fminf(ci - sm.cl[jj + 1], 0.f));
            *reinterpret_cast<bf2*>(&sW[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
                ii > jj ? bi * wacc[mi][ni][half * 2] * e0 : 0.f,
                ii > jj + 1 ? bi * wacc[mi][ni][half * 2 + 1] * e1 : 0.f);
            *reinterpret_cast<bf2*>(&sm.P[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
                ii >= jj ? pacc[mi][ni][half * 2] * e0 : 0.f,
                ii >= jj + 1 ? pacc[mi][ni][half * 2 + 1] * e1 : 0.f);
          }
    }

    // ---- Tm = (I+W)^-1 : block-recursive 2x2 inversion, ping-ponged.
    //      A <- I ; for m = 1,2,4,...:  Lo = W masked to {m <= i^j < 2m}
    //                                   A <- A - (A*Lo)*A
    if constexpr (!AIH)
      init_identity<C, NTHR>(sm.A, SM::LDC, tid, ABL(268435456));
    __syncthreads();

    bf16* Acur = sm.A;
    if (!ABL(1) && !ABL(536870912)) {
      inv_diag16<C, NWARP>(Acur, sW, SM::LDC, nvalid, warp, lane);
      __syncthreads();
    }
    if (!ABL(1) && !ABL(1073741824))
      inv_mm_levels<C, 2 * MM_ELEM, NWARP>(Acur, sW, sZ, SM::LDC, nvalid, warp, lane);
    // ---- rhs[j][v] = beta_j*v[j][v] - (kbd . S)[j][v], written over bV, plus
    //      the state half of the output, oacc = (d*q) . S. Both contract over
    //      the same B operand (Sb), so its fragments are fetched once; oacc
    //      stays in registers until P.U is added to it below.
    float oacc[TCV::WMG][TCV::WNG][4];
    zero_acc<C, VB, NWARP>(oacc);
    {
      float acc[TCV::WMG][TCV::WNG][4];
      zero_acc<C, VB, NWARP>(acc);
      TILE_IDS(TCV);
#pragma unroll
      for (int k0 = 0; k0 < (ABL(8) ? 0 : DH); k0 += 16) {
        unsigned bf[TCV::WNG][2];
#pragma unroll
        for (int ni = 0; ni + 1 < TCV::WNG; ni += 2)
          frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), sm.Sb, SM::LDSB,
                         (wn0 + ni) * 8, k0, lane);
        if (TCV::WNG & 1)
          frag_b1<false>(bf[TCV::WNG - 1], sm.Sb, SM::LDSB,
                         (wn0 + TCV::WNG - 1) * 8, k0, lane);
#pragma unroll
        for (int mi = 0; mi < TCV::WMG; ++mi) {
          const int m0 = (wm0 + mi) * 16;
          unsigned ak[4], aq[4];
          // UNSCALED (see the main kernel): beta_i and d_i are per-row, so they
          // go on the fp32 accumulator in the epilogue rather than on the bf16
          // A fragments.
          frag_a<false>(ak, bK, SM::LDK, m0, k0, lane);
          frag_a<false>(aq, bQ, SM::LDK, m0, k0, lane);
#pragma unroll
          for (int ni = 0; ni < TCV::WNG; ++ni) {
            mma16x8x16(acc[mi][ni], ak, bf[ni]);
            mma16x8x16(oacc[mi][ni], aq, bf[ni]);
          }
        }
      }
      // in place: the thread that owns acc[j][v] also owns bV[j][v]
#pragma unroll
      for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int jj = ACC_ROW(TCV, mi, half * 2);
            const int vv = ACC_COL(TCV, ni, 0);
            bf2* p = reinterpret_cast<bf2*>(&bV[jj * SM::LDV + vv]);
            const bf2 v = *p;
            const float bj = sm.beta[jj], dj = sm.dcl[jj];
            *p = __floats2bfloat162_rn(
                bj * (__bfloat162float(__low2bfloat16(v)) -
                      dj * acc[mi][ni][half * 2]),
                bj * (__bfloat162float(__high2bfloat16(v)) -
                      dj * acc[mi][ni][half * 2 + 1]));
            oacc[mi][ni][half * 2] *= dj;
            oacc[mi][ni][half * 2 + 1] *= dj;
          }
    }
    __syncthreads();

    // ---- U = Tm * rhs  -> sU [C][VB]  (sU aliases the phase-1 union)
    {
      float acc[TCV::WMG][TCV::WNG][4];
      zero_acc<C, VB, NWARP>(acc);
      TILE_IDS(TCV);
      if (!ABL(16))
        gemm<C, VB, C, NWARP, false, true>(acc, Acur, SM::LDC, bV, SM::LDV, warp, lane);
      // sU aliases sW/sZ, but every sW/sZ access belongs to the inverse, which
      // already closed with a barrier (inv_level's trailing one, or
      // inv_diag16's when the levels are empty), and this gemm reads only Acur
      // and bV.  So with BAR = 0 the barrier is dropped as ordering nothing.
      if constexpr (BAR) __syncthreads();
#pragma unroll
      for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int jj = ACC_ROW(TCV, mi, half * 2), vv = ACC_COL(TCV, ni, 0);
            *reinterpret_cast<bf2*>(&sU[jj * SM::LDU + vv]) =
                __floats2bfloat162_rn(acc[mi][ni][half * 2],
                                      acc[mi][ni][half * 2 + 1]);
            // Kd[j,:] = dkd_j * k_j and j is the state update's contraction
            // index, so dkd folds into U's rows here, not into its fragments.
            const float dk = sm.dkd[jj];
            *reinterpret_cast<bf2*>(&sm.Ud[jj * SM::LDU + vv]) =
                __floats2bfloat162_rn(dk * acc[mi][ni][half * 2],
                                      dk * acc[mi][ni][half * 2 + 1]);
          }
    }
    __syncthreads();

    // ---- O = scale * ( oacc + P.U ) -> staged in bV, then stored ---------
    {
      TILE_IDS(TCV);
      if (!ABL(32))
        gemm<C, VB, C, NWARP, false, true>(oacc, sm.P, SM::LDC, sU, SM::LDU, warp, lane);
      // bV's last reader was the U gemm, two barriers back, and each warp now
      // writes only its own output block from its own accumulator -- the
      // barrier before the store is what publishes bV.  Dropped at BAR = 0.
      if constexpr (BAR) __syncthreads();
#pragma unroll
      for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
          for (int half = 0; half < 2; ++half)
            *reinterpret_cast<bf2*>(&bV[ACC_ROW(TCV, mi, half * 2) * SM::LDV +
                                        ACC_COL(TCV, ni, 0)]) =
                __floats2bfloat162_rn(oacc[mi][ni][half * 2] * scale,
                                      oacc[mi][ni][half * 2 + 1] * scale);
    }
    __syncthreads();
    for (int i = tid; i < C * (VB / 8); i += NTHR) {
      const int r = i / (VB / 8), c = (i % (VB / 8)) * 8;
      if (r < nvalid)
        *reinterpret_cast<int4*>(gout + (size_t)(s0 + c0 + r) * (HV * DH) +
                                 (size_t)h * DH + v0 + c) =
            *reinterpret_cast<const int4*>(&bV[r * SM::LDV + c]);
    }

    // ---- S <- d_last*S + U^T * Kd    (M=VB, N=DH, K=C) -------------------
    //      A = U^T  : transposed read of sU   [C][VB]
    //      B = Kd   : transposed read of bK [C][DH], K-scaled by dkd
    {
#pragma unroll
      for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
          for (int e = 0; e < 4; ++e) sreg[mi][ni][e] *= d_last;
      const int wm0 = sm0, wn0 = sn0;
      (void)wm0; (void)wn0;
#pragma unroll
      for (int k0 = 0; k0 < (ABL(64) ? 0 : C); k0 += 16) {
        unsigned bf[TVD::WNG][2];
#pragma unroll
        for (int ni = 0; ni + 1 < TVD::WNG; ni += 2)
          frag_b2<true>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bK, SM::LDK,
                        (wn0 + ni) * 8, k0, lane);
        if (TVD::WNG & 1)
          frag_b1<true>(bf[TVD::WNG - 1], bK, SM::LDK, (wn0 + TVD::WNG - 1) * 8, k0,
                        lane);
#pragma unroll
        for (int mi = 0; mi < TVD::WMG; ++mi) {
          unsigned af[4];
          frag_a<true>(af, sm.Ud, SM::LDU, (wm0 + mi) * 16, k0, lane);
#pragma unroll
          for (int ni = 0; ni < TVD::WNG; ++ni) mma16x8x16(sreg[mi][ni], af, bf[ni]);
        }
      }
      sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);
    }
  }

  // ---- final state, already in k-last layout
  if (!ABL(4096)) {
    float* dst = gns + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          *reinterpret_cast<float2*>(dst + (size_t)SREG_V(mi, e) * DH + SREG_K(ni)) =
              make_float2(sreg[mi][ni][e], sreg[mi][ni][e + 1]);
        }
  }
#undef SREG_V
#undef SREG_K
}

template <int C, int VB, int NWARP, int MINB, int MM_ELEM, int BAR = 1,
          int AIH = 0>
cudaError_t launch(const void* q, const void* k, const void* v, const void* state,
                   const void* Alog, const void* a, const void* dtb, const void* b,
                   const void* cu, void* out, void* ns, int num_seqs,
                   int total_seq_len, float scale, int has_state, int tri,
                   cudaStream_t stream) {
  constexpr int smem = (int)sizeof(Smem<C, VB>);
  auto kern = gdn_kernel<C, VB, NWARP, MINB, MM_ELEM, BAR, AIH>;
  static bool configured = false;
  if (!configured) {
    cudaError_t e =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  dim3 grid(num_seqs, HV, DH / VB);
  kern<<<grid, NWARP * 32, smem, stream>>>(
      (const bf16*)q, (const bf16*)k, (const bf16*)v, (const float*)state,
      (const float*)Alog, (const bf16*)a, (const float*)dtb, (const bf16*)b,
      (const int64_t*)cu, (bf16*)out, (float*)ns, scale, has_state, num_seqs,
      total_seq_len, tri);
  return cudaGetLastError();
}

// ======================================================= PIPELINED FUSED path
//
// P1 (docs/draft.md 23.5).  A fused chunk is two halves of the same measured
// size (23.4):
//
//   SI  gates, K.K^T / Q.K^T + the masked epilogue, A <- I, (I+W)^-1
//       ~1.20 us/chunk at C=32 -- and it depends on NOTHING the state chain
//       produces, only on this chunk's k, q and gate rows
//   SD  k.S / (d*q).S, rhs, U = Tm.rhs, Ud, O += P.U, O, S <- d*S + Ud^T.Kd, Sb
//       ~1.11 us/chunk
//
// The unpipelined kernel runs them back to back, so a chunk costs SI + SD and
// the per-chunk barrier regions number ~9.  Here SI of chunk i+1 shares its
// barrier regions with SD of chunk i, so a chunk costs max(SI, SD) once the
// pipeline is full and the region count halves.  The three previous attempts to
// exploit the same independence all tried to *share* SI between the 8 V-slice
// CTAs of a head and all lost to the cost of communicating it (draft 17.3);
// this one communicates nothing and adds no arithmetic.  It only reorders.
//
// What it costs: A, P and the gate block double-buffered, and a THREE-deep
// K/Q/V ring -- in iteration i, ring slot i is live for SD(i) (stage 1's A
// operand and stage 4's B operand) while slot i+1 is live for SI(i+1)'s Gram,
// so the prefetch for i+2 needs a third slot.
template <int C, int VB>
struct alignas(16) SmemP {
  static constexpr int LDK = DH + PADH;
  static constexpr int LDV = VB + PADH;
  static constexpr int LDSB = DH + PADH;
  static constexpr int LDC = C + PADH;
  static constexpr int LDU = VB + PADH;
  static constexpr int N_SB = VB * LDSB;
  static constexpr int N_K = C * LDK;
  static constexpr int N_V = C * LDV;
  static constexpr int N_CC = C * LDC;
  static constexpr int N_U = C * LDU;

  bf16 Sb[N_SB];
  bf16 K[3 * N_K];  // ring of 3: SD(i) reads slot i, SI(i+1) slot i+1,
  bf16 Q[3 * N_K];  // and the prefetch for i+2 lands in slot i+2
  bf16 V[3 * N_V];
  bf16 A[2 * N_CC];  // Tm, per-chunk double buffer
  bf16 P[2 * N_CC];
  bf16 W[N_CC];  // SI scratch: only one SI is ever in flight, so single
  bf16 Z[N_CC];
  bf16 U[N_U];  // SD scratch (no longer aliased onto W)
  bf16 Ud[N_U];
  float cl[2][C], beta[2][C], dcl[2][C], dkd[2][C], scal[2][4];
};

// SI step 1: W = strict_lower(beta_i (k_i.k_j) E) and P = lower_incl((q_i.k_j) E),
// then A <- I.  Identical arithmetic to the unpipelined kernel's Gram block,
// parameterised on which ring slot the k/q tile is in and which A/P/gate buffer
// to write.
template <int C, int VB, int NWARP>
DEVI void pipe_si_gram(SmemP<C, VB>& sm, const bf16* bK, const bf16* bQ, bf16* Ag,
                       bf16* Pg, const float* cl, const float* beta, int nvalid,
                       int tri, int warp, int lane, int tid) {
  using SM = SmemP<C, VB>;
  using TCC = Tile<C, C, NWARP>;
  constexpr int NTHR = NWARP * 32;
  {
    float wacc[TCC::WMG][TCC::WNG][4], pacc[TCC::WMG][TCC::WNG][4];
    zero_acc<C, C, NWARP>(wacc);
    zero_acc<C, C, NWARP>(pacc);
    TILE_IDS(TCC);
    const bool tri_dead = tri && (wm0 + TCC::WMG - 1) * 16 + 15 < wn0 * 8;
    if (!tri_dead && nvalid > 0) {
#pragma unroll
      for (int k0 = 0; k0 < (ABL(2) ? 0 : DH); k0 += 16) {
        unsigned bf[TCC::WNG][2];
#pragma unroll
        for (int ni = 0; ni + 1 < TCC::WNG; ni += 2)
          frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bK, SM::LDK,
                         (wn0 + ni) * 8, k0, lane);
        if (TCC::WNG & 1)
          frag_b1<false>(bf[TCC::WNG - 1], bK, SM::LDK, (wn0 + TCC::WNG - 1) * 8, k0,
                         lane);
#pragma unroll
        for (int mi = 0; mi < TCC::WMG; ++mi) {
          unsigned ak[4], aq[4];
          frag_a<false>(ak, bK, SM::LDK, (wm0 + mi) * 16, k0, lane);
          frag_a<false>(aq, bQ, SM::LDK, (wm0 + mi) * 16, k0, lane);
#pragma unroll
          for (int ni = 0; ni < TCC::WNG; ++ni) {
            mma16x8x16(wacc[mi][ni], ak, bf[ni]);
            mma16x8x16(pacc[mi][ni], aq, bf[ni]);
          }
        }
      }
    }
#pragma unroll
    for (int mi = 0; mi < TCC::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TCC::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int ii = ACC_ROW(TCC, mi, half * 2);
          const int jj = ACC_COL(TCC, ni, 0);
          const float ci = cl[ii], bi = beta[ii];
          const float e0 = __expf(fminf(ci - cl[jj], 0.f));
          const float e1 = __expf(fminf(ci - cl[jj + 1], 0.f));
          *reinterpret_cast<bf2*>(&sm.W[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
              ii > jj ? bi * wacc[mi][ni][half * 2] * e0 : 0.f,
              ii > jj + 1 ? bi * wacc[mi][ni][half * 2 + 1] * e1 : 0.f);
          *reinterpret_cast<bf2*>(&Pg[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
              ii >= jj ? pacc[mi][ni][half * 2] * e0 : 0.f,
              ii >= jj + 1 ? pacc[mi][ni][half * 2 + 1] * e1 : 0.f);
        }
  }
  // A <- I, one 16-byte store per thread (see the unpipelined kernel)
  for (int i = tid; i < C * (C / 8); i += NTHR) {
    const int r = i / (C / 8), c = (i % (C / 8)) * 8;
    const int k = r - c;
    const unsigned val = ((unsigned)k < 8u) ? (0x3F80u << ((k & 1) * 16)) : 0u;
    const int w = (unsigned)k < 8u ? (k >> 1) : -1;
    int4 v;
    v.x = (w == 0) ? (int)val : 0;
    v.y = (w == 1) ? (int)val : 0;
    v.z = (w == 2) ? (int)val : 0;
    v.w = (w == 3) ? (int)val : 0;
    *reinterpret_cast<int4*>(&Ag[r * SM::LDC + c]) = v;
  }
}

template <int C, int VB, int NWARP, int MINB, int MM_ELEM>
__global__ __launch_bounds__(NWARP * 32, MINB) void gdn_pipe_kernel(
    const bf16* __restrict__ gq, const bf16* __restrict__ gk,
    const bf16* __restrict__ gv, const float* __restrict__ gstate,
    const float* __restrict__ gAlog, const bf16* __restrict__ ga,
    const float* __restrict__ gdtb, const bf16* __restrict__ gb,
    const int64_t* __restrict__ gcu, bf16* __restrict__ gout,
    float* __restrict__ gns, float scale, int has_state, int num_seqs,
    int total_seq_len, int tri) {
  using SM = SmemP<C, VB>;
  extern __shared__ char smem_raw[];
  SM& sm = *reinterpret_cast<SM*>(smem_raw);

  const int n = blockIdx.x, h = blockIdx.y, vs = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int NTHR = NWARP * 32;
  using TVD = Tile<VB, DH, NWARP>;
  using TCV = Tile<C, VB, NWARP>;
  const int hq = h >> 1;
  const int v0 = vs * VB;

  // Number of 16->32->... doubling levels, hence the SI step count.  SI has
  // 3 + 2*NLEV barrier-free steps and SD has 5, so a chunk occupies
  // max(3 + 2*NLEV, 5) regions instead of the unpipelined 8 + 2*NLEV.
  constexpr int NLEV = (C >= 64) ? 2 : ((C >= 32) ? 1 : 0);
  constexpr int NSI = 3 + 2 * NLEV;
  constexpr int NREG = NSI > 5 ? NSI : 5;

  const int sm0 = (warp % TVD::MG) * TVD::WMG;
  const int sn0 = (warp / TVD::MG) * TVD::WNG;
  float sreg[TVD::WMG][TVD::WNG][4];
#define SREG_V(mi, e) ((sm0 + (mi)) * 16 + (lane >> 2) + (((e) >> 1) * 8))
#define SREG_K(ni) ((sn0 + (ni)) * 8 + (lane & 3) * 2)
  {
    const float* src = gstate + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          float2 x = make_float2(0.f, 0.f);
          if (has_state && !ABL(268435456))
            x = *reinterpret_cast<const float2*>(
                src + (size_t)SREG_V(mi, e) * DH + SREG_K(ni));
          sreg[mi][ni][e] = x.x;
          sreg[mi][ni][e + 1] = x.y;
        }
  }

  const bool one = num_seqs == 1;
  const int s0 = one ? 0 : (int)gcu[n];
  const int seqlen = one ? total_seq_len : ((int)gcu[n + 1] - s0);
  if (seqlen <= 0) zero_acc<VB, DH, NWARP>(sreg);
  const int nchunk = (seqlen + C - 1) / C;

  const float nexp_alog = -__expf(gAlog[h]);
  const float dtb = gdtb[h];

  auto issue_tiles = [&](int cc, bf16* dK, bf16* dQ, bf16* dV) {
    const int nv = (cc < seqlen && !ABL(128)) ? min(C, seqlen - cc) : 0;
    const int t0 = s0 + (nv ? cc : 0);
    tile_cp<C, DH / 8, NTHR>(dK, SM::LDK,
                             gk + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, DH / 8, NTHR>(dQ, SM::LDK,
                             gq + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, VB / 8, NTHR>(dV, SM::LDV,
                             gv + (size_t)t0 * (HV * DH) + (size_t)h * DH + v0,
                             HV * DH, nv, tid);
    cp_async_commit();
  };

  // Gate rows are 2-byte strided reads, so warp 0 keeps one chunk ahead in
  // registers exactly as the unpipelined kernel does.
  constexpr int PER = (C + 31) / 32;
  float a_pf[PER], b_pf[PER];
  auto gate_prefetch = [&](int cc) {
    if (warp != 0) return;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int j = cc + p * 32 + lane;
      const int t = s0 + j;
      const bool ok = (p * 32 + lane) < C && j < seqlen && !ABL(256);
      a_pf[p] = ok ? __bfloat162float(ga[(size_t)t * HV + h]) : 0.f;
      b_pf[p] = ok ? __bfloat162float(gb[(size_t)t * HV + h]) : 0.f;
    }
  };

  // SI step 0: the gate block for chunk `cc` into buffer `g`, consuming the
  // register prefetch and re-filling it for chunk cc+1.
  auto si_gates = [&](int cc, float* gbeta, float* gcl, float* gdcl, float* gdkd,
                      float* gscal, int nvalid) {
    if (warp != 0) return;
    float a_cur[PER], b_cur[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      a_cur[p] = a_pf[p];
      b_cur[p] = b_pf[p];
    }
    gate_prefetch(cc + C);
    float carry = 0.f;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int j = p * 32 + lane;
      float lg = 0.f, be = 0.f;
      if (j < nvalid) {
        lg = nexp_alog * softplus(a_cur[p] + dtb);
        be = 1.f / (1.f + __expf(-b_cur[p]));
      }
      if (j < C) gbeta[j] = be;
      float x = lg;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
      }
      x += carry;
      if (j < C) {
        gcl[j] = x;
        gdcl[j] = __expf(x);
      }
      carry = __shfl_sync(0xffffffffu, x, 31);
    }
    const float cl_last = carry;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
      const int j = p * 32 + lane;
      if (j < C) gdkd[j] = __expf(fminf(cl_last - gcl[j], 0.f));
    }
    if (lane == 0) gscal[0] = __expf(cl_last);
  };

  // ---- the five state-dependent steps, for chunk `ci`
  float oacc[TCV::WMG][TCV::WNG][4];

  // acc = K.S ; oacc = (d*q).S ; rhs = beta*(V - d*acc) written over bV
  auto sd0 = [&](bf16* bV, const bf16* bK, const bf16* bQ, const float* cbeta,
                 const float* cdcl) {
    zero_acc<C, VB, NWARP>(oacc);
    float acc[TCV::WMG][TCV::WNG][4];
    zero_acc<C, VB, NWARP>(acc);
    TILE_IDS(TCV);
#pragma unroll
    for (int k0 = 0; k0 < (ABL(8) ? 0 : DH); k0 += 16) {
      unsigned bf[TCV::WNG][2];
#pragma unroll
      for (int ni = 0; ni + 1 < TCV::WNG; ni += 2)
        frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), sm.Sb, SM::LDSB,
                       (wn0 + ni) * 8, k0, lane);
      if (TCV::WNG & 1)
        frag_b1<false>(bf[TCV::WNG - 1], sm.Sb, SM::LDSB, (wn0 + TCV::WNG - 1) * 8,
                       k0, lane);
#pragma unroll
      for (int mi = 0; mi < TCV::WMG; ++mi) {
        unsigned ak[4], aq[4];
        frag_a<false>(ak, bK, SM::LDK, (wm0 + mi) * 16, k0, lane);
        frag_a<false>(aq, bQ, SM::LDK, (wm0 + mi) * 16, k0, lane);
#pragma unroll
        for (int ni = 0; ni < TCV::WNG; ++ni) {
          mma16x8x16(acc[mi][ni], ak, bf[ni]);
          mma16x8x16(oacc[mi][ni], aq, bf[ni]);
        }
      }
    }
#pragma unroll
    for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int jj = ACC_ROW(TCV, mi, half * 2), vv = ACC_COL(TCV, ni, 0);
          bf2* pp = reinterpret_cast<bf2*>(&bV[jj * SM::LDV + vv]);
          const bf2 v = *pp;
          const float bj = cbeta[jj], dj = cdcl[jj];
          *pp = __floats2bfloat162_rn(
              bj * (__bfloat162float(__low2bfloat16(v)) - dj * acc[mi][ni][half * 2]),
              bj * (__bfloat162float(__high2bfloat16(v)) -
                    dj * acc[mi][ni][half * 2 + 1]));
          oacc[mi][ni][half * 2] *= dj;
          oacc[mi][ni][half * 2 + 1] *= dj;
        }
  };

  auto sd1 = [&](const bf16* bV, const bf16* Ac, const float* cdkd) {
    float acc[TCV::WMG][TCV::WNG][4];
    zero_acc<C, VB, NWARP>(acc);
    TILE_IDS(TCV);
    if (!ABL(16))
      gemm<C, VB, C, NWARP, false, true>(acc, Ac, SM::LDC, bV, SM::LDV, warp, lane);
#pragma unroll
    for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int jj = ACC_ROW(TCV, mi, half * 2), vv = ACC_COL(TCV, ni, 0);
          *reinterpret_cast<bf2*>(&sm.U[jj * SM::LDU + vv]) = __floats2bfloat162_rn(
              acc[mi][ni][half * 2], acc[mi][ni][half * 2 + 1]);
          const float dk = cdkd[jj];
          *reinterpret_cast<bf2*>(&sm.Ud[jj * SM::LDU + vv]) = __floats2bfloat162_rn(
              dk * acc[mi][ni][half * 2], dk * acc[mi][ni][half * 2 + 1]);
        }
  };

  auto sd2 = [&](const bf16* Pc) {  // oacc += P.U
    if (!ABL(32))
      gemm<C, VB, C, NWARP, false, true>(oacc, Pc, SM::LDC, sm.U, SM::LDU, warp, lane);
  };

  auto sd3 = [&](bf16* bV) {  // O = scale*oacc -> bV
    TILE_IDS(TCV);
#pragma unroll
    for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half)
          *reinterpret_cast<bf2*>(&bV[ACC_ROW(TCV, mi, half * 2) * SM::LDV +
                                      ACC_COL(TCV, ni, 0)]) =
              __floats2bfloat162_rn(oacc[mi][ni][half * 2] * scale,
                                    oacc[mi][ni][half * 2 + 1] * scale);
  };

  // store O ; S <- d_last*S + Ud^T.Kd ; refresh the bf16 state mirror
  auto sd4 = [&](const bf16* bV, const bf16* bK, const float* cscal, int c0,
                 int nvalid) {
    if (!ABL(1024))
      for (int i = tid; i < C * (VB / 8); i += NTHR) {
        const int r = i / (VB / 8), c = (i % (VB / 8)) * 8;
        if (r < nvalid)
          *reinterpret_cast<int4*>(gout + (size_t)(s0 + c0 + r) * (HV * DH) +
                                   (size_t)h * DH + v0 + c) =
              *reinterpret_cast<const int4*>(&bV[r * SM::LDV + c]);
      }
    const float d_last = cscal[0];
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) sreg[mi][ni][e] *= d_last;
#pragma unroll
    for (int k0 = 0; k0 < (ABL(64) ? 0 : C); k0 += 16) {
      unsigned bf[TVD::WNG][2];
#pragma unroll
      for (int ni = 0; ni + 1 < TVD::WNG; ni += 2)
        frag_b2<true>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bK, SM::LDK,
                      (sn0 + ni) * 8, k0, lane);
      if (TVD::WNG & 1)
        frag_b1<true>(bf[TVD::WNG - 1], bK, SM::LDK, (sn0 + TVD::WNG - 1) * 8, k0,
                      lane);
#pragma unroll
      for (int mi = 0; mi < TVD::WMG; ++mi) {
        unsigned af[4];
        frag_a<true>(af, sm.Ud, SM::LDU, (sm0 + mi) * 16, k0, lane);
#pragma unroll
        for (int ni = 0; ni < TVD::WNG; ++ni) mma16x8x16(sreg[mi][ni], af, bf[ni]);
      }
    }
    sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);
  };

  if (seqlen > 0 && !ABL(134217728)) {
    // Ring slots as rotating POINTERS, not `ci % 3` indices: a runtime modulo
    // puts a division on the chunk's critical path and makes every ldmatrix
    // base in the gemms a runtime offset.  Rotating three pointers keeps all
    // of that in registers.  (Measured: the index form was 1-5% SLOWER than
    // the unpipelined kernel at C=32/64 despite halving the barrier count.)
    bf16* Kc = sm.K;                   // chunk i   -- SD's A and B operand
    bf16* Kn = sm.K + SM::N_K;         // chunk i+1 -- SI's Gram
    bf16* Kp = sm.K + 2 * SM::N_K;     // chunk i+2 -- in flight
    bf16* Qc = sm.Q;
    bf16* Qn = sm.Q + SM::N_K;
    bf16* Qp = sm.Q + 2 * SM::N_K;
    bf16* Vc = sm.V;
    bf16* Vn = sm.V + SM::N_V;
    bf16* Vp = sm.V + 2 * SM::N_V;
    bf16* Ac = sm.A;                   // Tm of chunk i
    bf16* An = sm.A + SM::N_CC;        // being built for chunk i+1
    bf16* Pc = sm.P;
    bf16* Pn = sm.P + SM::N_CC;
    float* bec = sm.beta[0];
    float* ben = sm.beta[1];
    float* clc = sm.cl[0];
    float* cln = sm.cl[1];
    float* dcc = sm.dcl[0];
    float* dcn = sm.dcl[1];
    float* dkc = sm.dkd[0];
    float* dkn = sm.dkd[1];
    float* sclc = sm.scal[0];
    float* scln = sm.scal[1];

    // ---- prologue: tiles for chunks 0 and 1, then the whole SI(0) chain
    issue_tiles(0, Kc, Qc, Vc);
    issue_tiles(C, Kn, Qn, Vn);
    gate_prefetch(0);
    sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);
    cp_async_wait<1>();  // chunk 0's tiles; chunk 1's stay in flight
    __syncthreads();
    {
      const int nv0 = min(C, seqlen);
      si_gates(0, bec, clc, dcc, dkc, sclc, nv0);
      __syncthreads();
      pipe_si_gram<C, VB, NWARP>(sm, Kc, Qc, Ac, Pc, clc, bec, nv0, tri, warp, lane,
                                 tid);
      __syncthreads();
      if (!ABL(1) && !ABL(536870912))
        inv_diag16<C, NWARP>(Ac, sm.W, SM::LDC, nv0, warp, lane);
      __syncthreads();
      if (!ABL(1) && !ABL(1073741824)) {
        if constexpr (NLEV >= 1) {
          inv_level_a<C, 16, NWARP>(Ac, sm.W, sm.Z, SM::LDC, nv0, warp, lane);
          __syncthreads();
          inv_level_b<C, 16, NWARP>(Ac, sm.W, sm.Z, SM::LDC, nv0, warp, lane);
          __syncthreads();
        }
        if constexpr (NLEV >= 2) {
          inv_level_a<C, 32, NWARP>(Ac, sm.W, sm.Z, SM::LDC, nv0, warp, lane);
          __syncthreads();
          inv_level_b<C, 32, NWARP>(Ac, sm.W, sm.Z, SM::LDC, nv0, warp, lane);
          __syncthreads();
        }
      }
    }

    // ---- the pipelined loop.  Iteration i runs SD(i) and SI(i+1) in the same
    // NREG barrier regions.  Every predicate is CTA-uniform, so the region
    // count is a compile-time constant and the barriers stay matched.
    for (int ci = 0; ci < nchunk; ++ci) {
      const int c0 = ci * C, nvalid = min(C, seqlen - c0);
      const int sic = c0 + C;  // the chunk SI works on
      const int sinv = (ci + 1 < nchunk) ? min(C, seqlen - sic) : 0;

      // region 0 : SD stage 1 + rhs  ||  SI gates(i+1) ; prefetch chunk i+2
      sd0(Vc, Kc, Qc, bec, dcc);
      si_gates(sic, ben, cln, dcn, dkn, scln, sinv);
      issue_tiles(sic + C, Kp, Qp, Vp);
      cp_async_wait<1>();  // chunk i+1's tiles, needed by SI's Gram below
      __syncthreads();

      // region 1 : SD U = Tm.rhs  ||  SI Gram(i+1) + A <- I
      sd1(Vc, Ac, dkc);
      pipe_si_gram<C, VB, NWARP>(sm, Kn, Qn, An, Pn, cln, ben, sinv, tri, warp, lane,
                                 tid);
      __syncthreads();

      // region 2 : SD oacc += P.U  ||  SI inv_diag16
      sd2(Pc);
      if (!ABL(1) && !ABL(536870912))
        inv_diag16<C, NWARP>(An, sm.W, SM::LDC, sinv, warp, lane);
      __syncthreads();

      // region 3 : SD O -> bV  ||  SI level 16a
      sd3(Vc);
      if constexpr (NLEV >= 1)
        if (!ABL(1) && !ABL(1073741824))
          inv_level_a<C, 16, NWARP>(An, sm.W, sm.Z, SM::LDC, sinv, warp, lane);
      __syncthreads();

      // region 4 : SD store + state update + Sb  ||  SI level 16b
      sd4(Vc, Kc, sclc, c0, nvalid);
      if constexpr (NLEV >= 1)
        if (!ABL(1) && !ABL(1073741824))
          inv_level_b<C, 16, NWARP>(An, sm.W, sm.Z, SM::LDC, sinv, warp, lane);
      __syncthreads();

      // regions 5,6 : SI level 32 only (C = 64), with no SD work left to pair
      if constexpr (NLEV >= 2) {
        if (!ABL(1) && !ABL(1073741824))
          inv_level_a<C, 32, NWARP>(An, sm.W, sm.Z, SM::LDC, sinv, warp, lane);
        __syncthreads();
        if (!ABL(1) && !ABL(1073741824))
          inv_level_b<C, 32, NWARP>(An, sm.W, sm.Z, SM::LDC, sinv, warp, lane);
        __syncthreads();
      }

      // rotate: chunk i+1 becomes current, the in-flight slot becomes next
      bf16* t;
      t = Kc; Kc = Kn; Kn = Kp; Kp = t;
      t = Qc; Qc = Qn; Qn = Qp; Qp = t;
      t = Vc; Vc = Vn; Vn = Vp; Vp = t;
      t = Ac; Ac = An; An = t;
      t = Pc; Pc = Pn; Pn = t;
      float* f;
      f = bec; bec = ben; ben = f;
      f = clc; clc = cln; cln = f;
      f = dcc; dcc = dcn; dcn = f;
      f = dkc; dkc = dkn; dkn = f;
      f = sclc; sclc = scln; scln = f;
    }
  }

  if (!ABL(4096)) {
    float* dst = gns + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          *reinterpret_cast<float2*>(dst + (size_t)SREG_V(mi, e) * DH + SREG_K(ni)) =
              make_float2(sreg[mi][ni][e], sreg[mi][ni][e + 1]);
        }
  }
#undef SREG_V
#undef SREG_K
}

template <int C, int VB, int NWARP, int MINB, int MM_ELEM>
cudaError_t launch_pipe(const void* q, const void* k, const void* v,
                        const void* state, const void* Alog, const void* a,
                        const void* dtb, const void* b, const void* cu, void* out,
                        void* ns, int num_seqs, int total_seq_len, float scale,
                        int has_state, int tri, cudaStream_t stream) {
  constexpr int smem = (int)sizeof(SmemP<C, VB>);
  auto kern = gdn_pipe_kernel<C, VB, NWARP, MINB, MM_ELEM>;
  static bool configured = false;
  if (!configured) {
    cudaError_t e =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  dim3 grid(num_seqs, HV, DH / VB);
  kern<<<grid, NWARP * 32, smem, stream>>>(
      (const bf16*)q, (const bf16*)k, (const bf16*)v, (const float*)state,
      (const float*)Alog, (const bf16*)a, (const float*)dtb, (const bf16*)b,
      (const int64_t*)cu, (bf16*)out, (float*)ns, scale, has_state, num_seqs,
      total_seq_len, tri);
  return cudaGetLastError();
}

// ================================================================== SPLIT path
//
// Everything in a chunk except four matmuls is *state-independent*: the gates,
// K.Kt, Q.Kt, W, P and the five levels of the triangular inverse. Only
// kbd.S, Tm.rhs, P.U (+output) and the state update need the running state, and
// those are the only part that has to be walked in order. So this path splits
// the operator into two kernels launched back to back from one C entry point:
//
//   gdn_pre_kernel   fully parallel over (chunk, head): gates, W, P, Tm
//                    -> workspace (Tm, P and the four gate vectors per chunk)
//   gdn_main_kernel  one CTA per (sequence, head, V-slice), as before, but the
//                    chunk body is reduced to the four state-dependent matmuls
//
// The chunk's barrier-region count drops from ~16 to ~5, so the critical path
// -- which is always the single longest sequence (8-15x the mean length in this
// workload set) -- drops by about 3x. The price is one extra kernel launch.
// Measured: two launches issued from python cost ~16 us of span because of the
// per-call python overhead, but two launches issued from one C entry point cost
// only 5-7 us, which is what makes this worthwhile above total_seq_len ~ 100.
//
// Chunks are addressed without a prefix sum: slot(n, ci) = cu[n]/C + ci + n is
// injective and monotone, because
//   nc(n) = ceil(L/C) <= floor((s+L)/C) - floor(s/C) + 1,
// so sequence n's slot range never reaches sequence n+1's start. The slot count
// is bounded by floor(total_seq_len/C) + num_seqs + 1, which the host knows.

template <int C>
struct Ws {  // workspace layout, per (slot, head)
  static constexpr int NTM = C * C;        // Tm, bf16, unpadded
  static constexpr int NP = C * C;         // P,  bf16, unpadded
  static constexpr int NG = 3 * C + 4;     // beta, dcl, dkd, then d_last
  static constexpr size_t BYTES = (size_t)(NTM + NP) * 2 + (size_t)NG * 4;
  static constexpr size_t ORD_BYTES = (size_t)ORDER_MAXN * 16;

  DEVI static bf16* tm(void* base, long nslot, long j) {
    return (bf16*)base + (size_t)j * NTM;
  }
  DEVI static bf16* p(void* base, long nslot, long j) {
    return (bf16*)base + (size_t)nslot * 8 * NTM + (size_t)j * NP;
  }
  DEVI static float* g(void* base, long nslot, long j) {
    return (float*)((char*)base + (size_t)nslot * 8 * (NTM + NP) * 2) + (size_t)j * NG;
  }
  // (sequence index, cu[n], seqlen) per main-kernel dispatch rank, published by
  // the pre-kernel so the main kernel's CTA -> sequence map costs one 16-byte
  // load instead of an O(num_seqs^2) cooperative pass, two barriers and two
  // dependent gcu round trips (5.5% of the main kernel's samples at 8192/48).
  DEVI static int4* ord(void* base, long nslot) {
    return (int4*)((char*)base +
                   (size_t)nslot * 8 * ((size_t)(NTM + NP) * 2 + (size_t)NG * 4));
  }
};

// ---------------------------------------------------------------- pre-kernel

// One CTA per (chunk, v-head), on an (nslot, HV) grid.  One chunk per CTA
// (rather than a CTA walking a strided subset of its sequence's chunks) means
// the k/q/gate tiles need no double buffer, and k/q are dead once W and P are
// written, so the inverse's scratch aliases them: C=64 fits in 66 KB, against
// 118 KB for the (num_seqs, HV, G) grid this replaces -- which also launched
// 3840 CTAs for 152 chunks of real work at 8192/48.
//
// `num_q_heads == 4` against `num_v_heads == 8` means the two v-heads of a pair
// (hq = h >> 1) see the same k/q tiles and compute bit-identical Gram matrices
// k_i.k_j and q_i.k_j, so one CTA could serve both and halve the only big gemm
// here.  Built and measured: 1.06x on the pre-kernel and *negative* end to end
// (geomean 21.043 -> 21.123 us), because halving the CTA count raises per-CTA
// latency by ~1.4x and the pre-kernel is latency-bound on every shape with
// nslot*HV below a few hundred.  See docs/draft.md 15.5.
template <int C, int NWARP, int MM_ELEM, int PMINB = 1>
struct alignas(16) SmemPre {
  static constexpr int LDK = DH + PADH;
  static constexpr int LDC = C + PADH;
  static constexpr int N_K = C * LDK;
  static constexpr int N_CC = C * LDC;
  static_assert(N_K >= N_CC, "aliased inverse scratch does not fit in K");

  bf16 K[N_K], Q[N_K];
  bf16 Ga[C * HV], Gb[C * HV];
  bf16 W[N_CC], A[N_CC], P[N_CC];
  float cl[C], beta[C], dcl[C], dkd[C], scal[4];

  DEVI bf16* Z() { return K; }
};

template <int C, int NWARP, int MM_ELEM, int PMINB, int PTF, int AIH = 0>
__global__ __launch_bounds__(NWARP * 32, PMINB) void gdn_pre_kernel(
    const bf16* __restrict__ gq, const bf16* __restrict__ gk,
    const float* __restrict__ gAlog, const bf16* __restrict__ ga,
    const float* __restrict__ gdtb, const bf16* __restrict__ gb,
    const int64_t* __restrict__ gcu, void* __restrict__ gws, long nslot,
    int num_seqs, int total_seq_len, int tri) {
  using SM = SmemPre<C, NWARP, MM_ELEM, PMINB>;
  using W = Ws<C>;
  extern __shared__ char smem_raw[];
  SM& sm = *reinterpret_cast<SM*>(smem_raw);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int NTHR = NWARP * 32;

  // slot(n, ci) = cu[n]/C + ci + n is injective and monotone, so a CTA recovers
  // its (n, ci) from the slot index with one cooperative pass over cu_seqlens.
  const int slot = blockIdx.x;
  const int h = blockIdx.y;
  const int hq = h >> 1;

  if (blockIdx.x == 0 && blockIdx.y == 0 && num_seqs > 1) {
    // Rank the sequences longest-first, ties by ascending index, and publish
    // (n, cu[n], seqlen) per rank.  O(num_seqs^2) in one CTA, which the rest of
    // this kernel's thousands of CTAs cover; the main kernel then reads its
    // sequence with a single load.  sm.K is dead until the tile copy below.
    int* sl = reinterpret_cast<int*>(sm.K);
    for (int j = tid; j <= num_seqs; j += NTHR) sl[j] = (int)gcu[j];
    __syncthreads();
    int4* o = W::ord(gws, nslot);
    for (int j = tid; j < num_seqs; j += NTHR) {
      const int a0 = sl[j], lj = sl[j + 1] - a0;
      int r = 0;
      for (int i = 0; i < num_seqs; ++i) {
        const int li = sl[i + 1] - sl[i];
        r += (li > lj) | ((li == lj) & (i < j));
      }
      o[r] = make_int4(j, a0, lj, 0);
    }
    __syncthreads();  // sm.K is about to be overwritten by the tile copy
  }

  int n, ci;
  if (num_seqs == 1) {  // cu_seqlens = {0, total_seq_len}, both host-known
    n = 0;
    ci = slot;
    if (ci >= (total_seq_len + C - 1) / C) return;
  } else {
    __shared__ int s_n, s_ci;
    if (ABL(32768)) { n = 0; ci = 0; goto have_nci; }
    if (tid == 0) s_n = -1;
    __syncthreads();
    for (int j = tid; j < num_seqs; j += NTHR) {
      const int a0 = (int)gcu[j], a1 = (int)gcu[j + 1];
      const int nc = (a1 - a0 + C - 1) / C;
      const int sl0 = a0 / C + j;
      if (slot >= sl0 && slot < sl0 + nc) {
        s_n = j;
        s_ci = slot - sl0;
      }
    }
    __syncthreads();
    if (s_n < 0) return;  // slot belongs to no sequence (the map leaves gaps)
    n = s_n;
    ci = s_ci;
  }
have_nci:;

  if (AB2(1)) return;  // pure launch floor

  const bool one = num_seqs == 1;
  const int s0 = one ? 0 : (int)gcu[n];
  const int seqlen = one ? total_seq_len : ((int)gcu[n + 1] - s0);
  const int slot0 = s0 / C + n;
  const int nvalid = min(C, seqlen - ci * C);

  using TCC = Tile<C, C, NWARP>;

  {  // k, q and the a/b gate rows (16 B = one token x all 8 heads)
    const int nv = ABL(128) ? 0 : nvalid;
    const int t0 = s0 + ci * C;
    tile_cp<C, DH / 8, NTHR>(sm.K, SM::LDK,
                             gk + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, DH / 8, NTHR>(sm.Q, SM::LDK,
                             gq + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, nv, tid);
    tile_cp<C, 1, NTHR>(sm.Ga, HV, ga + (size_t)t0 * HV, HV, nv, tid);
    tile_cp<C, 1, NTHR>(sm.Gb, HV, gb + (size_t)t0 * HV, HV, nv, tid);
    cp_async_commit();
  }
  // Hoisted identity seed (AIH = 1): independent of the tiles, and this is the
  // kernel's only tile-independent work, so it is the only thing that can fill
  // the cold k/q wait.  sm.A does not alias K/Q/Ga/Gb (the inverse's Z scratch
  // does, but that is read later).
  if constexpr (AIH) init_identity<C, NTHR>(sm.A, SM::LDC, tid, ABL(65536));
  cp_async_wait<0>();
  __syncthreads();

  // ---- the two Gram matrices, shared by both v-heads of the pair.  Both
  //      contract over the same B operand (k), so its fragments are fetched
  //      once; the lower-triangular mask and exp(E) are applied per head in the
  //      epilogue below.
  float wacc[TCC::WMG][TCC::WNG][4], pacc[TCC::WMG][TCC::WNG][4];
  zero_acc<C, C, NWARP>(wacc);
  zero_acc<C, C, NWARP>(pacc);
  {
    TILE_IDS(TCC);
    // Both outputs of this gemm are masked -- W by `ii > jj`, P by `ii >= jj` --
    // so a warp whose entire output block lies in the strict upper triangle
    // writes zeros no matter what the gemm produced.  For every instantiated
    // (C, NWARP) here, exactly one warp in four is in that position
    // (C=32/NWARP=4: warp 2; C=64/NWARP=4: warp 2; C=64/NWARP=8: warps 4, 5),
    // so this skips a quarter of the Gram work.  The epilogue below still runs
    // and still writes the zeros.  This is the shape of CTA-uniform skip that
    // pays: it removes a whole stage, its `ldmatrix` loads included, rather
    // than interrupting a scheduled dependency chain the way the rejected
    // `inv_diag16` early exit did.
    const bool tri_dead = tri && (wm0 + TCC::WMG - 1) * 16 + 15 < wn0 * 8;
    if (!tri_dead) {
#pragma unroll
    for (int k0 = 0; k0 < (ABL(262144) ? 0 : DH); k0 += 16) {
      unsigned bf[TCC::WNG][2];
#pragma unroll
      for (int ni = 0; ni + 1 < TCC::WNG; ni += 2)
        frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), sm.K, SM::LDK,
                       (wn0 + ni) * 8, k0, lane);
      if (TCC::WNG & 1)
        frag_b1<false>(bf[TCC::WNG - 1], sm.K, SM::LDK, (wn0 + TCC::WNG - 1) * 8, k0,
                       lane);
#pragma unroll
      for (int mi = 0; mi < TCC::WMG; ++mi) {
        unsigned ak[4], aq[4];
        frag_a<false>(ak, sm.K, SM::LDK, (wm0 + mi) * 16, k0, lane);
        frag_a<false>(aq, sm.Q, SM::LDK, (wm0 + mi) * 16, k0, lane);
#pragma unroll
        for (int ni = 0; ni < TCC::WNG; ++ni) {
          mma16x8x16(wacc[mi][ni], ak, bf[ni]);
          mma16x8x16(pacc[mi][ni], aq, bf[ni]);
        }
      }
    }
    }
  }

  {
    if (warp == 0) {  // gates and the inclusive cumsum of log g, in-warp
      const float nexp_alog = -__expf(gAlog[h]);
      const float dtb = gdtb[h];
      constexpr int PER = (C + 31) / 32;
      float carry = 0.f;
#pragma unroll
      for (int pz = 0; pz < PER; ++pz) {
        const int j = pz * 32 + lane;
        float lg = 0.f, be = 0.f;
        if (j < nvalid) {
          const float ga = __bfloat162float(sm.Ga[j * HV + h]) + dtb;
          const float gb = __bfloat162float(sm.Gb[j * HV + h]);
          lg = ABL(2048) ? -0.01f * ga : nexp_alog * softplus(ga);
          be = ABL(2048) ? 0.5f * gb : 1.f / (1.f + __expf(-gb));
        }
        if (j < C) sm.beta[j] = be;
        float x = lg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const float y = __shfl_up_sync(0xffffffffu, x, d);
          if (lane >= d) x += y;
        }
        x += carry;
        if (j < C) {
          sm.cl[j] = x;
          sm.dcl[j] = ABL(2048) ? x : __expf(x);
        }
        carry = __shfl_sync(0xffffffffu, x, 31);
      }
#pragma unroll
      for (int pz = 0; pz < PER; ++pz) {
        const int j = pz * 32 + lane;
        if (j < C)
          sm.dkd[j] = ABL(2048) ? (carry - sm.cl[j]) : __expf(fminf(carry - sm.cl[j], 0.f));
      }
      if (lane == 0) sm.scal[0] = __expf(carry);
    }
    __syncthreads();

    {  // W = strict_lower(beta_i (k_i.k_j) E(i,j)), P = lower_incl((q_i.k_j) E)
      TILE_IDS(TCC);
#pragma unroll
      for (int mi = 0; mi < TCC::WMG; ++mi)
#pragma unroll
        for (int ni = 0; ni < TCC::WNG; ++ni)
#pragma unroll
          for (int half = 0; half < 2 && !AB2(2); ++half) {
            const int ii = ACC_ROW(TCC, mi, half * 2), jj = ACC_COL(TCC, ni, 0);
            const float ci2 = sm.cl[ii], bi = sm.beta[ii];
            const float e0 = ABL(524288) ? 1.f : __expf(fminf(ci2 - sm.cl[jj], 0.f));
            const float e1 =
                ABL(524288) ? 1.f : __expf(fminf(ci2 - sm.cl[jj + 1], 0.f));
            *reinterpret_cast<bf2*>(&sm.W[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
                ii > jj ? bi * wacc[mi][ni][half * 2] * e0 : 0.f,
                ii > jj + 1 ? bi * wacc[mi][ni][half * 2 + 1] * e1 : 0.f);
            *reinterpret_cast<bf2*>(&sm.P[ii * SM::LDC + jj]) = __floats2bfloat162_rn(
                ii >= jj ? pacc[mi][ni][half * 2] * e0 : 0.f,
                ii >= jj + 1 ? pacc[mi][ni][half * 2 + 1] * e1 : 0.f);
          }
    }
    if constexpr (!AIH) init_identity<C, NTHR>(sm.A, SM::LDC, tid, ABL(65536));
    __syncthreads();

    // ---- Tm = (I+W)^-1
    if (!ABL(1048576)) {
      inv_diag16<C, NWARP>(sm.A, sm.W, SM::LDC, nvalid, warp, lane);
      __syncthreads();
    }
    if (!ABL(131072))
      inv_mm_levels<C, 2 * MM_ELEM, NWARP>(sm.A, sm.W, sm.Z(), SM::LDC, nvalid, warp,
                                           lane);

    // ---- PTF: publish PT = P . Tm instead of P.
    //
    // The main kernel's chunk body computes U = Tm.rhs and then O += P.U, two
    // gemms in series with a barrier between them because stage 3 contracts
    // over all of U's rows while stage 2 produces only a slice per warp.  But
    // O = P.(Tm.rhs) = (P.Tm).rhs, and P and Tm are both state-independent, so
    // the product can be formed once here -- in the fully parallel kernel --
    // and the main kernel's two stages become one fused pair that shares its
    // B-operand fragments (rhs) exactly as K.K^T and Q.K^T do above.  That
    // removes one of the four per-chunk barriers from the critical path and
    // halves stages 2-3's B-fragment loads; U is still needed for the state
    // update, so it is still computed.
    //
    // Both factors are lower triangular, so PT is too, and the same
    // strict-upper-tile skip applies.  sm.W is dead once the inverse is done,
    // so it serves as the destination with no extra barrier; the levels above
    // all end in one.
    if constexpr (PTF) {
      using TPT = Tile<C, C, NWARP>;
      float ptacc[TPT::WMG][TPT::WNG][4];
      zero_acc<C, C, NWARP>(ptacc);
      {
        TILE_IDS(TPT);
        const bool dead = tri && (wm0 + TPT::WMG - 1) * 16 + 15 < wn0 * 8;
        if (!dead)
          gemm<C, C, C, NWARP, false, true>(ptacc, sm.P, SM::LDC, sm.A, SM::LDC,
                                            warp, lane);
#pragma unroll
        for (int mi = 0; mi < TPT::WMG; ++mi)
#pragma unroll
          for (int ni = 0; ni < TPT::WNG; ++ni)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int ii = ACC_ROW(TPT, mi, half * 2), jj = ACC_COL(TPT, ni, 0);
              *reinterpret_cast<bf2*>(&sm.W[ii * SM::LDC + jj]) =
                  __floats2bfloat162_rn(ii >= jj ? ptacc[mi][ni][half * 2] : 0.f,
                                        ii >= jj + 1 ? ptacc[mi][ni][half * 2 + 1]
                                                     : 0.f);
            }
      }
      __syncthreads();
    }

    // ---- publish Tm, P and the gate vectors for this chunk
    {
      const long j = (long)(slot0 + ci) * HV + h;
      bf16* wtm = W::tm(gws, nslot, j);
      bf16* wp = W::p(gws, nslot, j);
      for (int i = (ABL(512) ? C * (C / 8) : tid); i < C * (C / 8); i += NTHR) {
        const int r = i / (C / 8), c = (i % (C / 8)) * 8;
        *reinterpret_cast<int4*>(wtm + r * C + c) =
            *reinterpret_cast<const int4*>(&sm.A[r * SM::LDC + c]);
        *reinterpret_cast<int4*>(wp + r * C + c) = *reinterpret_cast<const int4*>(
            &(PTF ? sm.W : sm.P)[r * SM::LDC + c]);
      }
      float* wg = W::g(gws, nslot, j);
      for (int i = AB2(4) ? C : tid; i < C; i += NTHR) {
        wg[i] = sm.beta[i];
        wg[C + i] = sm.dcl[i];
        wg[2 * C + i] = sm.dkd[i];
      }
      // The consumer copies this vector in 16-byte units, so the three floats
      // past d_last are read (and discarded); write them so initcheck stays
      // clean on the uninitialised workspace.
      if (tid == 0) {
        wg[3 * C] = sm.scal[0];
        wg[3 * C + 1] = 0.f;
        wg[3 * C + 2] = 0.f;
        wg[3 * C + 3] = 0.f;
      }
    }
    __syncthreads();
  }
  // This CTA's workspace writes are done and visible; let the dependent grid
  // start.  The main kernel's `griddepcontrol.wait` returns once every CTA here
  // has signalled.
  __threadfence();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// --------------------------------------------------------------- main kernel

template <int C, int VB>
struct alignas(16) SmemMain {
  static constexpr int LDK = DH + PADH;
  static constexpr int LDV = VB + PADH;
  static constexpr int LDSB = DH + PADH;
  static constexpr int LDC = C + PADH;

  static constexpr int N_SB = VB * LDSB;
  static constexpr int N_K = C * LDK;
  static constexpr int N_V = C * LDV;
  static constexpr int N_CC = C * LDC;
  static constexpr int N_G = 4 * C;

  bf16 Sb[N_SB];
  bf16 K[2 * N_K], Q[2 * N_K], V[2 * N_V];
  bf16 A[2 * N_CC], P[2 * N_CC];  // Tm and P, streamed in by cp.async
  bf16 U[N_V], Ud[N_V];           // U, and U with dkd folded into its rows
  float G[2 * N_G];  // beta, dcl, dkd, d_last
#if GDN_XPAD
  char xpad[GDN_XPAD * 1024];
#endif
};

// ORDER selects the CTA -> (sequence, head, V-slice) map.
//   0: the 3D grid (num_seqs, HV, DH/VB), i.e. blockIdx.x = sequence.
//   1: a flat 1D grid in which (a) all of one sequence's CTAs are consecutive,
//      so they are dispatched together rather than one per wave, and (b) the
//      sequences are visited longest-first.  Every workload here has one
//      sequence 8-15x longer than the mean, and that sequence's chunk chain is
//      the critical path; under ORDER=0 its last CTA is dispatched in the last
//      wave, so the chain only *starts* once everything else has finished.
template <int C, int VB, int NWARP, int MINB, int ORDER, int PTF>
__global__ __launch_bounds__(NWARP * 32, MINB) void gdn_main_kernel(
    const bf16* __restrict__ gq, const bf16* __restrict__ gk,
    const bf16* __restrict__ gv, const float* __restrict__ gstate,
    const int64_t* __restrict__ gcu, bf16* __restrict__ gout,
    float* __restrict__ gns, const void* __restrict__ gws, long nslot, float scale,
    int has_state, int num_seqs, int total_seq_len) {
  using SM = SmemMain<C, VB>;
  using W = Ws<C>;
  extern __shared__ char smem_raw[];
  SM& sm = *reinterpret_cast<SM*>(smem_raw);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int NTHR = NWARP * 32;
  constexpr int NVS = DH / VB;

  // With one sequence, cu_seqlens is {0, total_seq_len} -- both host-known -- so
  // the dependent (and L2-cold) gcu load disappears entirely; 20 of the 100
  // workloads have num_seqs == 1.  Otherwise, under ORDER = 1 the sequence (and
  // its bounds) come from the one record the pre-kernel published for this
  // CTA's rank.
  // Which sequence this CTA owns.  When cu_seqlens is host-known (one sequence)
  // or the map is the plain 3-D grid (ORDER = 0), this costs nothing and runs
  // before `griddepcontrol.wait`, so under PDL the whole prologue -- index
  // math, the state tile load, the first k/q/v cp.async -- overlaps the
  // pre-kernel.  Otherwise it is the (n, cu[n], seqlen) record the pre-kernel
  // published for this rank, which lives in the workspace, so the wait comes
  // first.  (Ranking the sequences here instead, to put that work in the
  // hidden prologue, was measured slower on every shape -- only the first
  // wave's prologue is hidden, and the ranking costs two barriers.)
  const bool host_known = (num_seqs == 1) || (ORDER == 0);
  if (!host_known) asm volatile("griddepcontrol.wait;" ::: "memory");
  int n, h, vs, s0, seqlen;
  if (num_seqs == 1) {
    n = 0;
    vs = blockIdx.x % NVS;
    h = (blockIdx.x / NVS) % HV;
    if (ORDER == 0) { h = blockIdx.y; vs = blockIdx.z; }
    s0 = 0;
    seqlen = total_seq_len;
  } else if (ORDER == 0) {
    n = blockIdx.x; h = blockIdx.y; vs = blockIdx.z;
    s0 = (int)gcu[n];
    seqlen = (int)gcu[n + 1] - s0;
  } else {
    const int flat = blockIdx.x;
    vs = flat % NVS;
    h = (flat / NVS) % HV;
    const int4 o = W::ord((void*)gws, nslot)[flat / (NVS * HV)];
    n = o.x;
    s0 = o.y;
    seqlen = o.z;
  }
  const int hq = h >> 1;
  const int v0 = vs * VB;

  using TCV = Tile<C, VB, NWARP>;
  using TVD = Tile<VB, DH, NWARP>;

  // The fp32 state lives in REGISTERS for the whole sequence, in exactly the
  // accumulator layout of the state-update gemm S <- d_last*S + U^T.Kd, so that
  // gemm accumulates straight on top of it: no shared fp32 array, and no
  // read-modify-write epilogue (which the per-line profile had at 6% of the
  // kernel).  Shared memory keeps only the bf16 mirror Sb, which is the B
  // operand of k.S and q.S.
  const int sm0 = (warp % TVD::MG) * TVD::WMG;
  const int sn0 = (warp / TVD::MG) * TVD::WNG;
  float sreg[TVD::WMG][TVD::WNG][4];
  // Element e of tile (mi, ni) is state column v = SREG_V, row k = SREG_K.
#define SREG_V(mi, e) ((sm0 + (mi)) * 16 + (lane >> 2) + (((e) >> 1) * 8))
#define SREG_K(ni) ((sn0 + (ni)) * 8 + (lane & 3) * 2)
  {
    const float* src = gstate + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
    // Each float2 is 8 B and four consecutive lanes cover 32 contiguous bytes,
    // so every sector is fully used.
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          float2 x = make_float2(0.f, 0.f);
          if (has_state)
            x = *reinterpret_cast<const float2*>(
                src + (size_t)SREG_V(mi, e) * DH + SREG_K(ni));
          sreg[mi][ni][e] = x.x;
          sreg[mi][ni][e + 1] = x.y;
        }
  }

  if (seqlen <= 0) zero_acc<VB, DH, NWARP>(sreg);
  const int nchunk = (seqlen + C - 1) / C;
  const int slot0 = s0 / C + n;
  auto cc_of = [](int ci) { return ci * C; };

  auto issue = [&](int ci, int buf) {
    const int nv = (ci < nchunk) ? min(C, seqlen - cc_of(ci)) : 0;
    const int t0 = s0 + (nv ? ci * C : 0);  // tile row 0, safe when nv == 0
    tile_cp<C, DH / 8, NTHR>(sm.K + buf * SM::N_K, SM::LDK,
                             gk + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, ABL(67108864) ? 0 : nv, tid);
    tile_cp<C, DH / 8, NTHR>(sm.Q + buf * SM::N_K, SM::LDK,
                             gq + (size_t)t0 * (HQ * DH) + (size_t)hq * DH,
                             HQ * DH, ABL(67108864) ? 0 : nv, tid);
    tile_cp<C, VB / 8, NTHR>(sm.V + buf * SM::N_V, SM::LDV,
                             gv + (size_t)t0 * (HV * DH) + (size_t)h * DH + v0,
                             HV * DH, nv, tid);
    // Tm, P and the gate vectors for this chunk
    const long j = (long)(slot0 + (nv ? ci : 0)) * HV + h;
    if (!ABL(33554432)) {
      tile_cp_full<C, C / 8, NTHR>(sm.A + buf * SM::N_CC, SM::LDC,
                                   W::tm((void*)gws, nslot, j), C, tid);
      tile_cp_full<C, C / 8, NTHR>(sm.P + buf * SM::N_CC, SM::LDC,
                                   W::p((void*)gws, nslot, j), C, tid);
    }
    const float* wg = W::g((void*)gws, nslot, j);
    float* dG = sm.G + buf * SM::N_G;
    for (int i = tid * 4; i < 3 * C + 4; i += NTHR * 4)
      cp_async16(&dG[i], wg + i, 16);
    cp_async_commit();
  };

  sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);
  __syncthreads();

  // Everything above this point reads only kernel inputs (when the wait was not
  // already taken above); `issue` is the first thing that touches the
  // pre-kernel's workspace.
  if (host_known) asm volatile("griddepcontrol.wait;" ::: "memory");

  if (seqlen > 0) {
    issue(0, 0);
    int buf = 0;
    for (int ci = 0; ci < nchunk; ++ci, buf ^= 1) {
      const int c0 = ci * C, nvalid = min(C, seqlen - c0);
      const bf16* bK = sm.K + buf * SM::N_K;
      const bf16* bQ = sm.Q + buf * SM::N_K;
      bf16* bV = sm.V + buf * SM::N_V;
      const bf16* bTm = sm.A + buf * SM::N_CC;
      const bf16* bP = sm.P + buf * SM::N_CC;
      const float* bG = sm.G + buf * SM::N_G;
      cp_async_wait<0>();
      __syncthreads();
      issue(ci + 1, buf ^ 1);
      const float* bBeta = bG;
      const float* bDcl = bG + C;
      const float* bDkd = bG + 2 * C;
      const float d_last = bG[3 * C];

      // ---- rhs = beta*V - (kbd.S) over bV, and oacc = (d*q).S
      float oacc[TCV::WMG][TCV::WNG][4];
      zero_acc<C, VB, NWARP>(oacc);
      {
        float acc[TCV::WMG][TCV::WNG][4];
        zero_acc<C, VB, NWARP>(acc);
        TILE_IDS(TCV);
#pragma unroll
        for (int k0 = 0; k0 < (ABL(2097152) ? 0 : DH); k0 += 16) {
          unsigned bf[TCV::WNG][2];
#pragma unroll
          for (int ni = 0; ni + 1 < TCV::WNG; ni += 2)
            frag_b2<false>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), sm.Sb, SM::LDSB,
                           (wn0 + ni) * 8, k0, lane);
          if (TCV::WNG & 1)
            frag_b1<false>(bf[TCV::WNG - 1], sm.Sb, SM::LDSB,
                           (wn0 + TCV::WNG - 1) * 8, k0, lane);
#pragma unroll
          for (int mi = 0; mi < TCV::WMG; ++mi) {
            const int m0 = (wm0 + mi) * 16;
            unsigned ak[4], aq[4];
            // UNSCALED: beta_i and d_i are per-row, so they are applied to the
            // fp32 accumulator in the epilogue instead of to the bf16 A
            // fragments (`mul.bf16x2` was 7-11% of the kernel's samples, and
            // scaling after the gemm also removes a bf16 rounding step).
            frag_a<false>(ak, bK, SM::LDK, m0, k0, lane);
            frag_a<false>(aq, bQ, SM::LDK, m0, k0, lane);
#pragma unroll
            for (int ni = 0; ni < TCV::WNG; ++ni) {
              mma16x8x16(acc[mi][ni], ak, bf[ni]);
              mma16x8x16(oacc[mi][ni], aq, bf[ni]);
            }
          }
        }
#pragma unroll
        for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
          for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int jj = ACC_ROW(TCV, mi, half * 2), vv = ACC_COL(TCV, ni, 0);
              bf2* pp = reinterpret_cast<bf2*>(&bV[jj * SM::LDV + vv]);
              const bf2 v = *pp;
              const float bj = bBeta[jj], dj = bDcl[jj];
              // rhs = beta*(V - d*(k.S));  and the output half gets its own d.
              *pp = __floats2bfloat162_rn(
                  bj * (__bfloat162float(__low2bfloat16(v)) -
                        dj * acc[mi][ni][half * 2]),
                  bj * (__bfloat162float(__high2bfloat16(v)) -
                        dj * acc[mi][ni][half * 2 + 1]));
              oacc[mi][ni][half * 2] *= dj;
              oacc[mi][ni][half * 2 + 1] *= dj;
            }
      }
      __syncthreads();

      {  // U = Tm . rhs, and (PTF) O += PT . rhs in the same pass
        float acc[TCV::WMG][TCV::WNG][4];
        zero_acc<C, VB, NWARP>(acc);
        TILE_IDS(TCV);
        if constexpr (PTF) {
          // Both gemms contract `rhs` over the chunk index, so the B fragments
          // are fetched once for the pair -- the same trick as the pre-kernel's
          // K.K^T / Q.K^T.  oacc already holds (d*q).S from stage 1.
#pragma unroll
          for (int k0 = 0; k0 < (ABL(4194304) ? 0 : C); k0 += 16) {
            unsigned bf[TCV::WNG][2];
#pragma unroll
            for (int ni = 0; ni + 1 < TCV::WNG; ni += 2)
              frag_b2<true>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bV, SM::LDV,
                            (wn0 + ni) * 8, k0, lane);
            if (TCV::WNG & 1)
              frag_b1<true>(bf[TCV::WNG - 1], bV, SM::LDV, (wn0 + TCV::WNG - 1) * 8,
                            k0, lane);
#pragma unroll
            for (int mi = 0; mi < TCV::WMG; ++mi) {
              unsigned at[4], ap[4];
              frag_a<false>(at, bTm, SM::LDC, (wm0 + mi) * 16, k0, lane);
              frag_a<false>(ap, bP, SM::LDC, (wm0 + mi) * 16, k0, lane);
#pragma unroll
              for (int ni = 0; ni < TCV::WNG; ++ni) {
                mma16x8x16(acc[mi][ni], at, bf[ni]);
                mma16x8x16(oacc[mi][ni], ap, bf[ni]);
              }
            }
          }
        } else if (!ABL(4194304))
          gemm<C, VB, C, NWARP, false, true>(acc, bTm, SM::LDC, bV, SM::LDV, warp,
                                             lane);
#pragma unroll
        for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
          for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int jj = ACC_ROW(TCV, mi, half * 2), vv = ACC_COL(TCV, ni, 0);
              *reinterpret_cast<bf2*>(&sm.U[jj * SM::LDV + vv]) =
                  __floats2bfloat162_rn(acc[mi][ni][half * 2],
                                        acc[mi][ni][half * 2 + 1]);
              // Kd[j,:] = dkd_j * k_j, and j is the contraction index of the
              // state update, so dkd folds into U's rows here instead of into
              // the K fragments inside that gemm.
              const float dk = bDkd[jj];
              *reinterpret_cast<bf2*>(&sm.Ud[jj * SM::LDV + vv]) =
                  __floats2bfloat162_rn(dk * acc[mi][ni][half * 2],
                                        dk * acc[mi][ni][half * 2 + 1]);
            }
      }
      __syncthreads();

      {  // O = scale*(oacc + P.U) staged over bV; then the state update
        TILE_IDS(TCV);
        if constexpr (!PTF) {
          if (!ABL(8388608))
            gemm<C, VB, C, NWARP, false, true>(oacc, bP, SM::LDC, sm.U, SM::LDV,
                                               warp, lane);
          __syncthreads();
        }
#pragma unroll
        for (int mi = 0; mi < TCV::WMG; ++mi)
#pragma unroll
          for (int ni = 0; ni < TCV::WNG; ++ni)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              *reinterpret_cast<bf2*>(&bV[ACC_ROW(TCV, mi, half * 2) * SM::LDV +
                                          ACC_COL(TCV, ni, 0)]) =
                  __floats2bfloat162_rn(oacc[mi][ni][half * 2] * scale,
                                        oacc[mi][ni][half * 2 + 1] * scale);

        // sreg is pre-scaled by d_last and then IS the accumulator, so
        // S <- d_last*S + U^T.Kd needs no fp32 shared round trip.
#pragma unroll
        for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
          for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) sreg[mi][ni][e] *= d_last;
#pragma unroll
        for (int k0 = 0; k0 < (ABL(16777216) ? 0 : C); k0 += 16) {
          unsigned bf[TVD::WNG][2];
#pragma unroll
          for (int ni = 0; ni + 1 < TVD::WNG; ni += 2)
            frag_b2<true>(reinterpret_cast<unsigned(&)[4]>(bf[ni]), bK, SM::LDK,
                          (sn0 + ni) * 8, k0, lane);
          if (TVD::WNG & 1)
            frag_b1<true>(bf[TVD::WNG - 1], bK, SM::LDK, (sn0 + TVD::WNG - 1) * 8, k0,
                          lane);
#pragma unroll
          for (int mi = 0; mi < TVD::WMG; ++mi) {
            unsigned af[4];
            frag_a<true>(af, sm.Ud, SM::LDV, (sm0 + mi) * 16, k0, lane);
#pragma unroll
            for (int ni = 0; ni < TVD::WNG; ++ni) mma16x8x16(sreg[mi][ni], af, bf[ni]);
          }
        }
        sreg_to_Sb<C, VB, NWARP>(sm, sreg, sm0, sn0, lane);
      }
      __syncthreads();
      if (!ABL(1024))
      for (int i = tid; i < C * (VB / 8); i += NTHR) {
        const int r = i / (VB / 8), c = (i % (VB / 8)) * 8;
        if (r < nvalid)
          *reinterpret_cast<int4*>(gout + (size_t)(s0 + c0 + r) * (HV * DH) +
                                   (size_t)h * DH + v0 + c) =
              *reinterpret_cast<const int4*>(&bV[r * SM::LDV + c]);
      }
    }
  }

  if (!ABL(4096)) {
    float* dst = gns + ((size_t)n * HV + h) * DH * DH + (size_t)v0 * DH;
#pragma unroll
    for (int mi = 0; mi < TVD::WMG; ++mi)
#pragma unroll
      for (int ni = 0; ni < TVD::WNG; ++ni)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int e = half * 2;
          *reinterpret_cast<float2*>(dst + (size_t)SREG_V(mi, e) * DH + SREG_K(ni)) =
              make_float2(sreg[mi][ni][e], sreg[mi][ni][e + 1]);
        }
  }
#undef SREG_V
#undef SREG_K
}

template <int C, int VB, int NWARP, int MINB, int MM_ELEM, int PREW, int ORDER,
          int PMINB, int PTF, int AIH = 0>
cudaError_t launch_split(const void* q, const void* k, const void* v,
                         const void* state, const void* Alog, const void* a,
                         const void* dtb, const void* b, const void* cu, void* out,
                         void* ns, void* ws, int num_seqs, int total_seq_len,
                         float scale, int has_state, int pdl, int tri,
                         cudaStream_t stream) {
  constexpr int smem_pre = (int)sizeof(SmemPre<C, PREW, MM_ELEM, PMINB>);
  constexpr int smem_main = (int)sizeof(SmemMain<C, VB>);
  auto kpre = gdn_pre_kernel<C, PREW, MM_ELEM, PMINB, PTF, AIH>;
  auto kmain = gdn_main_kernel<C, VB, NWARP, MINB, ORDER, PTF>;
  static bool configured = false;
  if (!configured) {
    cudaError_t e = cudaFuncSetAttribute(
        kpre, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pre);
    if (e != cudaSuccess) return e;
    e = cudaFuncSetAttribute(kmain, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_main);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  const long nslot = (long)(total_seq_len / C) + num_seqs + 1;
  const dim3 gpre((unsigned)nslot, HV, 1);
  if (!ABL(8192))
  gdn_pre_kernel<C, PREW, MM_ELEM, PMINB, PTF, AIH>
      <<<gpre, PREW * 32, smem_pre, stream>>>(
      (const bf16*)q, (const bf16*)k, (const float*)Alog, (const bf16*)a,
      (const float*)dtb, (const bf16*)b, (const int64_t*)cu, ws, nslot, num_seqs,
      total_seq_len, tri);
  const dim3 gmain = ORDER ? dim3(num_seqs * HV * (DH / VB), 1, 1)
                           : dim3(num_seqs, HV, DH / VB);
  if (!ABL(16384)) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = gmain;
    lc.blockDim = dim3(NWARP * 32, 1, 1);
    lc.dynamicSmemBytes = smem_main;
    lc.stream = stream;
    cudaLaunchAttribute la[1];
    la[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    la[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = la;
    lc.numAttrs = pdl ? 1 : 0;
    cudaError_t le = cudaLaunchKernelEx(
        &lc, kmain, (const bf16*)q, (const bf16*)k, (const bf16*)v,
        (const float*)state, (const int64_t*)cu, (bf16*)out, (float*)ns,
        (const void*)ws, nslot, scale, has_state, num_seqs, total_seq_len);
    if (le != cudaSuccess) return le;
  }
  return cudaGetLastError();
}


}  // namespace

// id, chunk C, V-slice VB, warps, min CTAs/SM, elementwise inverse levels.
// Only 0 and 1 are reachable from _pick_cfg; 2 and 3 are the nearest
// alternatives from the sweep in dev/pick.py, kept so the dispatch boundary
// can be re-measured without editing the kernel. Each entry is a separate
// kernel instantiation, and the solution JIT-compiles on first call inside the
// evaluator's timeout, so the list stays short.
#define GDN_CFGS              \
  GDN_CFG(0, 32, 16, 4, 1, 8) \
  GDN_CFG(1, 32, 32, 4, 1, 8) \
  GDN_CFG(7, 16, 16, 2, 1, 8) \
  GDN_CFG(8, 64, 16, 8, 1, 8)

// DISPATCHED (Phase 7): 50/57/58 are 0/7/8 with the two redundant per-chunk
// barriers dropped (BAR = 0) AND the inverse's identity seed hoisted above the
// cold tile wait (AIH = 1).  Both halves are pure reorderings; paired A/B
// 1.0033 / 1.0019 / 1.0067.  docs/draft.md 23.9, 23.11.
#define GDN_P7_CFGS           \
  GDN_P7(50, 32, 16, 4, 1, 8) \
  GDN_P7(57, 16, 16, 2, 1, 8) \
  GDN_P7(58, 64, 16, 8, 1, 8)

// Dev-only comparison configs, behind -DGDN_DEVCFG=1: 14 extra kernel
// instantiations, and the solution JIT-compiles on first call inside the
// evaluator's per-workload timeout (a cold build is 7 s without them and 25 s
// with), so the release build leaves them out.  40/47/48 are P5's barrier
// removal alone, superseded by the dispatched 50/57/58; 30/37/38 are P1's
// rejected pipelined kernel; 9xx are P7's rejected split-path hoist.
// docs/draft.md 23.8-23.11.
#ifndef GDN_DEVCFG
#define GDN_DEVCFG 0
#endif

#if GDN_DEVCFG
#define GDN_BAR_CFGS           \
  GDN_BAR(40, 32, 16, 4, 1, 8) \
  GDN_BAR(47, 16, 16, 2, 1, 8) \
  GDN_BAR(48, 64, 16, 8, 1, 8)

// Pipelined fused path (P1): same (C, VB, NWARP) as the dispatched fused ids,
// with SI(chunk i+1) sharing its barrier regions with SD(chunk i).  30 is the
// counterpart of 0, 37 of 7, 38 of 8.
#define GDN_PIPE_CFGS           \
  GDN_PIPE(30, 32, 16, 4, 1, 8) \
  GDN_PIPE(37, 16, 16, 2, 1, 8) \
  GDN_PIPE(38, 64, 16, 8, 1, 8)
#else
#define GDN_BAR_CFGS
#define GDN_PIPE_CFGS
#endif

// Split path: id, chunk C, V-slice VB, main-kernel warps, min CTAs/SM,
// elementwise inverse levels, pre-kernel warps, CTA order, pre-kernel grid,
// pre-kernel min CTAs/SM.  Every entry is a separate kernel instantiation and
// the solution JIT-compiles on first call inside the evaluator's timeout, so
// this list is kept to the dispatchable configs plus their nearest measured
// alternatives (dev/cfglist.py).
//
// MM_ELEM = 8 everywhere: inv_diag16() inverts the 16x16 diagonal blocks
// directly (so everything below mm = 16 is covered in one pass) and
// inv_mm_levels() takes it from 2*MM_ELEM = 16 upwards with two mm x mm x mm
// matmuls per pair.  No other value is meaningful any more.
#define GDN_SPLIT_CFGS                               \
  GDN_SPLIT(700, 32, 16, 4, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(701, 32, 32, 4, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(705, 64, 16, 4, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(725, 64, 32, 4, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(751, 32, 32, 4, 1, 8, 4, 0, 1, 0)     \
  GDN_SPLIT(755, 64, 16, 4, 1, 8, 4, 0, 1, 0)     \
  GDN_SPLIT(765, 64, 64, 4, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(706, 64, 16, 8, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(726, 64, 32, 8, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(702, 32, 32, 8, 1, 8, 4, 1, 1, 0)     \
  GDN_SPLIT(805, 64, 16, 4, 1, 8, 4, 1, 1, 1)     \
  GDN_SPLIT(825, 64, 32, 4, 1, 8, 4, 1, 1, 1)     \
  GDN_SPLIT(855, 64, 16, 4, 1, 8, 4, 0, 1, 1)     \
  GDN_SPLIT(801, 32, 32, 4, 1, 8, 4, 1, 1, 1)

// P7: the four dispatched split ids with the pre-kernel's identity seed hoisted
// above its cold k/q wait (AIH = 1).  Measured a wash (0.996-1.003 over the
// four) because the pre-kernel runs at 2 CTAs/SM, so dev-only.
#if GDN_DEVCFG
#define GDN_AIH_CFGS                                 \
  GDN_AIH(905, 64, 16, 4, 1, 8, 4, 1, 1, 1)     \
  GDN_AIH(925, 64, 32, 4, 1, 8, 4, 1, 1, 1)     \
  GDN_AIH(955, 64, 16, 4, 1, 8, 4, 0, 1, 1)     \
  GDN_AIH(901, 32, 32, 4, 1, 8, 4, 1, 1, 1)
#else
#define GDN_AIH_CFGS
#endif

extern "C" long gdn_prefill_ws_bytes(int cfg, int num_seqs, int total_seq_len) {
  int C = 32;
  switch (cfg) {
#define GDN_SPLIT(id, CC, VB, NW, MB, ME, PW, OR, PB, PF) \
  case id:                                                \
    C = CC; break;
    GDN_SPLIT_CFGS
#undef GDN_SPLIT
#define GDN_AIH(id, CC, VB, NW, MB, ME, PW, OR, PB, PF) \
  case id:                                              \
    C = CC; break;
    GDN_AIH_CFGS
#undef GDN_AIH
    default: return 0;
  }
  const long nslot = (long)(total_seq_len / C) + num_seqs + 1;
  const long per = (long)(C * C) * 2 * 2 + (long)(3 * C + 4) * 4;
  return nslot * HV * per + (long)ORDER_MAXN * 16;
}

extern "C" int gdn_prefill_launch(const void* q, const void* k, const void* v,
                                  const void* state, const void* Alog, const void* a,
                                  const void* dtb, const void* b, const void* cu,
                                  void* out, void* ns, int num_seqs, int total_seq_len,
                                  float scale, int has_state, int cfg,
                                  void* stream_raw, void* ws, int pdl) {
  cudaStream_t stream = (cudaStream_t)stream_raw;
#define ARGS \
  q, k, v, state, Alog, a, dtb, b, cu, out, ns, num_seqs, total_seq_len, scale, \
      has_state, tri, stream
  cudaError_t e;
  // dev-only: cfg >= 10000 launches cfg-10000 with the triangular Gram skip
  // DISABLED, so one process can alternate the two and measure the skip with
  // the paired protocol (dev/paircfg.py) instead of across runs.  Same trick as
  // PDL: make the knob a launch-time choice, not a build flag.
  const int tri = cfg >= 10000 ? 0 : 1;
  if (cfg >= 10000) cfg -= 10000;
  // dev-only: cfg >= 1000 launches cfg-1000 twice from here, to measure the
  // inter-kernel gap charged to the harness's span metric when both launches
  // come from one C entry point (rather than two python calls).
  const int reps = cfg >= 1000 ? 2 : 1;
  if (cfg >= 1000) cfg -= 1000;
  for (int rep = 0; rep < reps; ++rep)
  switch (cfg) {
#define GDN_CFG(id, C, VB, NW, MB, ME) \
  case id:                             \
    e = launch<C, VB, NW, MB, ME>(ARGS); break;
    GDN_CFGS
#undef GDN_CFG
#define GDN_BAR(id, C, VB, NW, MB, ME) \
  case id:                             \
    e = launch<C, VB, NW, MB, ME, 0>(ARGS); break;
    GDN_BAR_CFGS
#undef GDN_BAR
#define GDN_P7(id, C, VB, NW, MB, ME) \
  case id:                            \
    e = launch<C, VB, NW, MB, ME, 0, 1>(ARGS); break;
    GDN_P7_CFGS
#undef GDN_P7
#define GDN_PIPE(id, C, VB, NW, MB, ME) \
  case id:                              \
    e = launch_pipe<C, VB, NW, MB, ME>(ARGS); break;
    GDN_PIPE_CFGS
#undef GDN_PIPE
#define GDN_SPLIT(id, CC, VB, NW, MB, ME, PW, OR, PB, PF)                          \
  case id:                                                                          \
    e = launch_split<CC, VB, NW, MB, ME, PW, OR, PB, PF>(q, k, v, state, Alog, a, \
                                             dtb, b, cu, out, ns, ws, num_seqs,     \
                                             total_seq_len, scale, has_state, pdl,   \
                                             tri, stream);                               \
    break;
    GDN_SPLIT_CFGS
#undef GDN_SPLIT
#define GDN_AIH(id, CC, VB, NW, MB, ME, PW, OR, PB, PF)                            \
  case id:                                                                          \
    e = launch_split<CC, VB, NW, MB, ME, PW, OR, PB, PF, 1>(q, k, v, state, Alog, a, \
                                             dtb, b, cu, out, ns, ws, num_seqs,     \
                                             total_seq_len, scale, has_state, pdl,   \
                                             tri, stream);                               \
    break;
    GDN_AIH_CFGS
#undef GDN_AIH
    default: return -2;
  }
#undef ARGS
  return (int)e;
}

// Reports (smem bytes, registers, max active CTAs/SM) for the split path's two
// kernels, so an occupancy claim can be checked rather than inferred.
extern "C" int gdn_prefill_occ(int cfg, int which, int field) {
  int smem = 0, nthr = 0, occ = 0, regs = 0;
  cudaFuncAttributes at;
  switch (cfg) {
#define GDN_SPLIT(id, CC, VB, NW, MB, ME, PW, OR, PB, PF)                      \
  case id: {                                                                   \
    if (which == 0) {                                                          \
      smem = (int)sizeof(SmemPre<CC, PW, ME, PB>);                     \
      nthr = PW * 32;                                                          \
      auto f = gdn_pre_kernel<CC, PW, ME, PB, PF>;                     \
      cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      cudaFuncGetAttributes(&at, f);                                           \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nthr, smem);      \
    } else {                                                                   \
      smem = (int)sizeof(SmemMain<CC, VB>);                                    \
      nthr = NW * 32;                                                          \
      auto f = gdn_main_kernel<CC, VB, NW, MB, OR, PF>;                        \
      cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      cudaFuncGetAttributes(&at, f);                                           \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nthr, smem);      \
    }                                                                          \
    regs = at.numRegs;                                                         \
    break;                                                                     \
  }
    GDN_SPLIT_CFGS
#undef GDN_SPLIT
    default: return -1;
  }
  return field == 0 ? smem : field == 1 ? regs : occ;
}

extern "C" int gdn_prefill_smem(int cfg) {
  switch (cfg) {
#define GDN_CFG(id, C, VB, NW, MB, ME) \
  case id:                             \
    return (int)sizeof(Smem<C, VB>);
    GDN_CFGS
#undef GDN_CFG
#define GDN_BAR(id, C, VB, NW, MB, ME) \
  case id:                             \
    return (int)sizeof(Smem<C, VB>);
    GDN_BAR_CFGS
#undef GDN_BAR
#define GDN_P7(id, C, VB, NW, MB, ME) \
  case id:                            \
    return (int)sizeof(Smem<C, VB>);
    GDN_P7_CFGS
#undef GDN_P7
#define GDN_PIPE(id, C, VB, NW, MB, ME) \
  case id:                              \
    return (int)sizeof(SmemP<C, VB>);
    GDN_PIPE_CFGS
#undef GDN_PIPE
    default: return -1;
  }
}
