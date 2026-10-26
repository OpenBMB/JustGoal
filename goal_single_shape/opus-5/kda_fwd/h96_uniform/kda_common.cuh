// Common definitions for the KDA forward kernel (SM100 / tcgen05).
#pragma once
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

#define DD   128           // head dim (K and V)
#define CB    64           // chunk length (tokens)
#ifdef NTHR_OVR
#define NTHR NTHR_OVR
#else
#define NTHR 512           // threads per CTA
#endif
#define NWG   (NTHR / 32)  // 32-thread groups
#define NTG   (NTHR / 128) // 128-thread groups used for TMEM traffic
// Prologue thread map: tid = tg*CGR + cg, cg in [0,CGR) covering the head dim,
// tg in [0,NPG) covering the chunk's tokens.
//
// TPT * CPT = 16 always (CB * DD / NTHR), so CGR only trades tokens per
// thread against channels per thread.  CGR = 16 (CPT = 8, TPT = 2) makes the
// g/k/q loads LDG.E.128 and the four tile stores STS.128 -- measured 5.4%
// fewer instructions overall and 14% fewer LSU instructions, for exactly the
// same bytes and the same coalescing -- and is **2.7% slower**, because TPT
// is the token loop's ILP dimension and halving it costs more than halving
// the load/store count saves.  CGR = 32 (CPT = 4, TPT = 4) is the measured
// optimum of the family; (8,2) and (1,16) are worse still on both counts.
#define CGR   32           // threads spanning the head dim
#define CPT   (DD / CGR)   // channels per thread = 8
#define NPG   (NTHR / CGR) // token groups = 32
#define TPT   (CB / NPG)   // tokens per thread = 2

// All four use the .ftz form.  Without it these are *not* single-MUFU
// instructions: PTX specifies ex2/rcp/rsqrt.approx.f32 to produce correct
// denormal results, so ptxas wraps every MUFU in a range check and a fixup
// path -- `FSETP.GEU.AND P0, PT, x, -126` plus a halve-then-square sequence.
// Each of the 20 exp2 call sites in the prologue paid for one.  The .ftz
// form flushes denormals and lowers to the bare MUFU.
//
// Safe here because nothing these see can be denormal.  The decay exponents
// are bounded by +-92 by construction (that bound is exactly what forces
// C = 64), so e and 1/e live in [2^-92, 2^92]; rsqrt gets sk + 1e-12 > 0;
// tanh has no .ftz form, so it keeps the plain one (its output is in
// [-1,1] and needs no fixup anyway).
//
// sigmoid(x) = 0.5 + 0.5*tanh(x/2): one MUFU instead of ex2 + rcp.
__device__ __forceinline__ float sigmoid_fast(float x) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}
__device__ __forceinline__ float exp2_fast(float x) {
  float e; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x)); return e;
}
__device__ __forceinline__ float rcp_fast(float x) {
  float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rsqrt_fast(float x) {
  float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}

// ---- 128B-swizzled shared-memory addressing, matching
// UMMA::tile_to_mma_shape(Layout_{K,MN}_SW128_Atom<bf16>, partition_shape_{A,B}(...)).
// Verified element-by-element against CuTe in cuda/t_layout2.cu.
__device__ __forceinline__ int sw128(int x) { return x ^ (((x >> 6) & 7) << 3); }
// K-major [M,K] tile: contraction index `c` is the fast one.
template <int M>
__device__ __forceinline__ int offK(int r, int c) {
  return sw128((c >> 6) * (M * 64) + r * 64 + (c & 63));
}
// MN-major [N,K] tile: the MN index `n` is the fast one.
template <int N>
__device__ __forceinline__ int offMN(int n, int k) {
  return sw128((k >> 4) * (N * 16) + (((k >> 3) & 1) * (N / 64) + (n >> 6)) * 512
               + (k & 7) * 64 + (n & 63));
}
// Cheap cursors for the two layouts above.  offK/offMN recompute the whole
// block decomposition *and* the swizzle XOR on every access, which measured at
// 36% of the kernel's dynamic instructions.  Both simplify exactly when the
// row (K-major) or the contraction index (MN-major) is loop invariant:
//   offK<M>(r,c)   == KCur<M>(r)(c)     for every c   (M % 8 == 0)
//   offMN<N>(n,k)  == MCur<N>(k)(n)     for every n
// so the XOR and the k/r arithmetic are hoisted once per row instead of once
// per store.  Verified exhaustively against offK/offMN for M,N in {64,128}.
template <int M> struct KCur {
  int r64, xv;
  __device__ __forceinline__ explicit KCur(int r) : r64(r * 64), xv((r & 7) << 3) {}
  __device__ __forceinline__ int operator()(int c) const {
    return (c >> 6) * (M * 64) + r64 + ((c & 63) ^ xv);
  }
};
template <int N> struct MCur {
  int kb, xv;
  __device__ __forceinline__ explicit MCur(int k)
      : kb((k >> 4) * (N * 16) + ((k >> 3) & 1) * (N / 64) * 512 + (k & 7) * 64),
        xv((((k >> 4) * (N / 4) + (k & 7)) & 7) << 3) {}
  __device__ __forceinline__ int operator()(int n) const {
    return kb + (n >> 6) * 512 + ((n & 63) ^ xv);
  }
};

// tcgen05.mma reads SMEM through the async proxy: ordinary (generic-proxy)
// stores must be made visible to it with a proxy fence before the barrier.
// Plain barrier: use where the data being published is read by other *threads*
// only.  The proxy fence in smem_sync() exists to make generic-proxy writes
// visible to the async (tcgen05) proxy, and is only needed when an MMA will
// read what was just written.
__device__ __forceinline__ void tsync() { __syncthreads(); }
// Per-barrier cost probe.  -DNOBAR=n turns barrier n into a no-op: numerically
// meaningless, but the cycle delta is that barrier's warp-arrival skew.
#ifndef NOBAR
#define NOBAR 0
#endif
// NOBAR=99 removes every barrier at once, giving the total.
#define BAR(n)  do { if ((NOBAR) != (n) && (NOBAR) != 99) tsync();     } while (0)
#define SBAR(n) do { if ((NOBAR) != (n) && (NOBAR) != 99) smem_sync(); } while (0)
__device__ __forceinline__ void smem_sync() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}
// 16-byte async global->shared copy (bypasses registers, so the prologue's
// global-load latency can be covered by the previous chunk's work).
__device__ __forceinline__ void kda_cp16(void* dst, const void* src) {
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" :: "r"(d), "l"(src));
}
__device__ __forceinline__ void kda_cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void kda_cp_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N));
}
struct alignas(16) bf16x8 { bf16 v[8]; };
// cvt.rn.bf16x2.f32 converts and packs a pair in one instruction; doing it
// element-wise costs two converts plus the packing shifts, and the kernel
// does ~85k of these per chunk.
__device__ __forceinline__ uint32_t cvt2(float lo, float hi) {
  uint32_t d; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
}
struct alignas(8)  bf16x4 { bf16 v[4]; };
struct alignas(4)  bf16x2 { bf16 v[2]; };
// All four tile stores write CPT consecutive channels of one token (KQ/KB are
// K-major with the channel fast; KBT is MN-major with the channel as its fast
// MN index).  The width must therefore track CPT, not TPT -- they happened to
// be equal at CPT = TPT = 4, which hid the distinction.
#if   CPT == 8
typedef bf16x8 bf16xC;
#define packC pack8
#elif CPT == 4
typedef bf16x4 bf16xC;
#define packC pack4
#else
#error "unsupported CPT"
#endif
__device__ __forceinline__ bf16x8 pack8(const float* x) {
  bf16x8 r; uint32_t* o = reinterpret_cast<uint32_t*>(&r);
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) o[i] = cvt2(x[2 * i], x[2 * i + 1]);
  return r;
}
__device__ __forceinline__ bf16x4 pack4(const float* x) {
  bf16x4 r; uint32_t* o = reinterpret_cast<uint32_t*>(&r);
  o[0] = cvt2(x[0], x[1]); o[1] = cvt2(x[2], x[3]);
  return r;
}
