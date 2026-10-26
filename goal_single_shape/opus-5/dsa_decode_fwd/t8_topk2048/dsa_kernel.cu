// DSA sparse MLA decode forward -- from-scratch B200 implementation.
//
// out[t] = softmax( (q_nope[t] @ Kc^T + q_pe[t] @ Kp^T) * sm_scale ) @ Kc
// lse[t] = log2( sum_j exp2(logit_j * sm_scale * log2e) )
//
// Flash-decoding style split over the KV (topk) axis:
//   kernel 1 ("split")   : one CTA per (token, kv-chunk of R rows) -> (m, l, O/l)
//   kernel 2 ("combine") : rescale + reduce the partials per (token, head)
// Partial O is stored normalized in bf16: the combine forms a weighted *mean*
// of the partials, so the bf16 rounding error averages out instead of piling up.
// A monotonically increasing stamp marks which chunks produced work, so no
// scratch buffer ever has to be reset between launches.

#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

#define HEADS 16
#define DCKV 512
// ints between adjacent token counters, so each owns a 128 B line
#define CNT_STRIDE 32
// Scratch past the arrival counters, used only by the STAGE 10 timestamp probe.
#define T_CNT 256
// Per-block arrival flags start here, one 128 B line each.
#define FLAG_BASE 768
// Blocks in a balanced launch.  One per SM keeps the arrival spin single-wave.
#define NBAL 128
// Plan scratch entries per token (count, block span, block base, tile count).
#define MAXT 16

__device__ __forceinline__ unsigned gtimer() {
  unsigned long long g;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
  return (unsigned)g;
}
#define DKPE 64
#define DKTOT (DCKV + DKPE)  // 576
#define TOPKC 2048
#define LDK (DKTOT + 8)  // 584: de-aliases smem banks across rows

// --------------------------------------------------------------------------
// PTX primitives.  nvcc's wmma wrappers lower bf16 fragment loads to generic
// `LD` instructions here, so the tensor-core path is driven by hand.
// --------------------------------------------------------------------------
__device__ __forceinline__ unsigned sm_addr(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sm_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Bulk (TMA) copy + mbarrier.  A gathered KV row is 1024 contiguous bytes, so
// cp.async would split it into 64 separate 16 B requests; cp.async.bulk moves
// it as one.  With R = 128 rows that is 256 requests per CTA instead of 9216,
// which is what the gather is actually limited by.
// Plain relaxed load / fire-and-forget add on a device-scope flag.  The add
// needs no return value, so it retires at issue instead of waiting out an L2
// round trip, and the release carries every store the block published before it.
__device__ __forceinline__ int ld_relaxed_gpu(const int *p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_add_release_gpu(int *p) {
  asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(p) : "memory");
}
__device__ __forceinline__ void mbar_init(unsigned bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect(unsigned bar, unsigned tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(tx)
               : "memory");
}
// Standalone, additive transaction count.  Unlike arrive.expect_tx this does
// not consume an arrival, so every producer thread can declare exactly the
// bytes its own copy will move instead of one thread declaring the block's
// total -- which is what forces a reduction onto the critical path.
__device__ __forceinline__ void mbar_expect(unsigned bar, unsigned tx) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(tx) : "memory");
}
__device__ __forceinline__ void tma_load(unsigned dst, const void *src, unsigned bytes,
                                         unsigned bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned bar) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "W%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra W%=;\n}\n" ::"r"(bar) : "memory");
}

// Compile-time counted loop: cp.async.wait_group needs a literal operand, which
// `#pragma unroll` alone does not provide.
template <int I>
struct CInt {
  static constexpr int value = I;
};
template <int N, class F>
__device__ __forceinline__ void seq_for(F &f) {
  if constexpr (N > 0) {
    seq_for<N - 1>(f);
    f(CInt<N - 1>{});
  }
}

__device__ __forceinline__ unsigned bf2u(__nv_bfloat162 v) {
  return *reinterpret_cast<unsigned *>(&v);
}

__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(unsigned (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldm_x2(unsigned (&r)[2], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
// D(16x8) += A(16x16) * B(16x8);  A row-major, B col-major.
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float warp_max4(float v) {  // over the 4 lanes sharing lane/4
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float warp_sum4(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---------------------------------------------------------------------------
// kernel 1: per (token, kv-chunk) partial attention.
//   STAGE 0 = empty (launch floor), 1 = gather, 2 = + QK/softmax, 3 = full,
//   4 = QK mma only, 5 = full but every chunk writes over token slot 0 so the
//   partial-buffer store bandwidth drops out and only the PV mma is left
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Phase 2: rescale + reduce the partials.  One 64-thread group per (token,
// head), one thread per 8-dim slice, so a thread owns its whole output element:
// no smem, no barrier, no cross-warp tree.  The (m, l) fetch and all NSPLIT
// partial fetches are issued back to back -- their addresses are independent,
// so the two memory round trips overlap instead of chaining.
// ---------------------------------------------------------------------------
template <int NSPLIT, int DPT, int CSG, int LZ, int OLAY = 0>
__device__ __forceinline__ void combine_item(const __nv_bfloat16 *__restrict__ o_part,
                                             const float *__restrict__ ml_part,
                                             __nv_bfloat16 *__restrict__ out,
                                             float *__restrict__ lse, int t, int h, int tid,
                                             int sg) {
  constexpr int NT = DCKV / DPT;   // threads covering one (token, head) row
  constexpr int SPG = NSPLIT / CSG;  // partials one thread group folds
  static_assert(NSPLIT % 4 == 0 && NSPLIT % CSG == 0 && DCKV % DPT == 0 && DPT >= 2 && DPT <= 8,
                "combine shape");
  using vec_t = typename std::conditional<DPT == 8, float4,
                typename std::conditional<DPT == 4, float2, float>::type>::type;

  // OLAY 1 is head-major: the NSPLIT partials of one (token, head) sit back to
  // back, so a folding block reads one contiguous NSPLIT*DCKV span instead of
  // NSPLIT 1 KB pieces spread HEADS*DCKV apart.
  constexpr size_t SSTR = OLAY ? DCKV : (size_t)HEADS * DCKV;
  const __nv_bfloat16 *bp =
      OLAY ? o_part + ((size_t)t * HEADS + h) * (NSPLIT * DCKV) + tid * DPT
           : o_part + (size_t)t * NSPLIT * (HEADS * DCKV) + (size_t)h * DCKV + tid * DPT;
  // Every thread reads the whole (m, l) vector from the same address, so this
  // broadcasts out of L1.  Doing it per thread removes the serial shfl chain
  // that a warp-level reduction would put in front of the fma loop.
  const float *mm = ml_part + ((size_t)(t * 2) * HEADS) * NSPLIT;
  float mv[NSPLIT], w[NSPLIT];
  // Issue this group's partial loads up front: nothing below feeds their
  // addresses, so they overlap with the (m, l) fetch instead of chaining after
  // it.
  vec_t raw[SPG];
  if constexpr (!LZ) {
#pragma unroll
    for (int i = 0; i < SPG; ++i)
      raw[i] = *reinterpret_cast<const vec_t *>(bp + (size_t)(sg * SPG + i) * SSTR);
  }
#pragma unroll
  for (int s = 0; s < NSPLIT; s += 4) {
    *reinterpret_cast<float4 *>(&mv[s]) = *reinterpret_cast<const float4 *>(mm + (size_t)h * NSPLIT + s);
    *reinterpret_cast<float4 *>(&w[s]) =
        *reinterpret_cast<const float4 *>(mm + (size_t)(HEADS + h) * NSPLIT + s);
  }
  // w[] still holds l; l == 0 marks a chunk that held no rows (m is garbage).
  float M = -INFINITY;
#pragma unroll
  for (int s = 0; s < NSPLIT; ++s) M = (w[s] > 0.f) ? fmaxf(M, mv[s]) : M;
#pragma unroll
  for (int s = 0; s < NSPLIT; ++s) w[s] = (w[s] > 0.f) ? exp2f(mv[s] - M) * w[s] : 0.f;
  // Only a third of the chunks hold rows for this workload, and an empty one's
  // o_part slot is never read.  Issuing the loads here -- after the weights are
  // known, before the sum they feed -- skips two thirds of the traffic while
  // still leaving just one dependent memory trip on the critical path.  The
  // predicate is uniform across the warp (its lanes share one (t, head)), so
  // nothing diverges.
  if constexpr (LZ) {
#pragma unroll
    for (int i = 0; i < SPG; ++i) {
      // An empty chunk reads chunk 0's row instead of its own.  Its weight is
      // zero either way, so the value is inert, and redirecting rather than
      // branching keeps every load unconditional and coalesced while cutting
      // the traffic to the chunks that actually hold data -- most of the grid
      // is empty on a sparse workload.
      const int s = sg * SPG + i;
      raw[i] = *reinterpret_cast<const vec_t *>(bp + (size_t)((w[s] > 0.f) ? s : 0) * SSTR);
    }
  }
  float L = 0.f;
#pragma unroll
  for (int s = 0; s < NSPLIT; ++s) L += w[s];
  const float inv = (L > 0.f) ? 1.f / L : 0.f;

  float acc[DPT];
#pragma unroll
  for (int k = 0; k < DPT; ++k) acc[k] = 0.f;
#pragma unroll
  for (int i = 0; i < SPG; ++i) {
    const float wv = w[sg * SPG + i] * inv;
    // A chunk with no rows never writes its partial, so that slot still holds
    // whatever was in the buffer -- possibly NaN.  Skipping it is a correctness
    // requirement; the predicate is warp-uniform so it costs no divergence.
    if (wv == 0.f) continue;
    const __nv_bfloat16 *v = reinterpret_cast<const __nv_bfloat16 *>(&raw[i]);
#pragma unroll
    for (int k = 0; k < DPT; ++k) acc[k] = fmaf(wv, __bfloat162float(v[k]), acc[k]);
  }
  // Splitting the partials across CSG thread groups is the only knob that adds
  // warps without adding loads: the byte count per block is fixed, and at one
  // group the block is 4 warps of pure gather with nothing to hide L2 misses
  // behind.  Group 0 folds the rest in through smem, laid out thread-major so
  // each warp's slice is contiguous.
  if constexpr (CSG > 1) {
    __shared__ float red[(CSG - 1) * NT * DPT];
    if (sg > 0) {
#pragma unroll
      for (int k = 0; k < DPT; ++k) red[(sg - 1) * (NT * DPT) + tid * DPT + k] = acc[k];
    }
    __syncthreads();
    if (sg > 0) return;
#pragma unroll
    for (int j = 0; j < CSG - 1; ++j)
#pragma unroll
      for (int k = 0; k < DPT; ++k) acc[k] += red[j * (NT * DPT) + tid * DPT + k];
  }
  __nv_bfloat16 o8[DPT];
#pragma unroll
  for (int k = 0; k < DPT; ++k) o8[k] = __float2bfloat16(acc[k]);
  *reinterpret_cast<vec_t *>(out + (size_t)(t * HEADS + h) * DCKV + tid * DPT) =
      *reinterpret_cast<const vec_t *>(o8);
  if (tid == 0) lse[t * HEADS + h] = (L > 0.f) ? (log2f(L) + M) : -INFINITY;
}

// Fold the `sub` blocks holding token `t`'s partials, which occupy slots
// [base, base + sub).  fold_token cannot serve here: its split count is a
// template constant feeding per-thread register arrays, and a balanced launch
// only knows the count at run time.
template <int DPT, int NTHREAD>
__device__ __forceinline__ void bal_fold(const __nv_bfloat16 *__restrict__ o_part,
                                         const float *__restrict__ ml_part,
                                         __nv_bfloat16 *__restrict__ out, float *__restrict__ lse,
                                         int *__restrict__ cnt, int t, int base, int sub, int j,
                                         int tid) {
  using vec_t = typename std::conditional<DPT == 8, float4,
                typename std::conditional<DPT == 4, float2, float>::type>::type;
  __shared__ int sgo;
  int *fl = cnt + FLAG_BASE + base * CNT_STRIDE;
  if (tid == 0) {
    // One lane's release publishes the whole block's stores (fence
    // cumulativity), so 256 lanes do not each need a membar.
    __threadfence();
    sgo = atomicAdd(fl + j * CNT_STRIDE, 1) + 1;
  }
  __syncthreads();
  const int v = sgo;
  // Flags are never reset: each launch adds one, so a block's own new value is
  // exactly what it must see on every sibling.
  if (tid < sub) {
    volatile int *fp = fl + tid * CNT_STRIDE;
    while (*fp < v) {
    }
  }
  // One cooperative read of the (m, l) vectors, then every thread reads them
  // out of smem.  Straight from global each thread walks the same run-time
  // loop of dependent L2 loads, which is what the fold ends up waiting on.
  __shared__ float sml[2 * HEADS * 64];
  const int nml = 2 * HEADS * sub;
  for (int i = tid; i < nml; i += NTHREAD) {
    const int hh = i / sub, sx = i - hh * sub;
    sml[i] = ml_part[(size_t)hh * NBAL + base + sx];
  }
  __syncthreads();
  // Head / d-slice assignment.  With at least HEADS blocks each folds one head
  // over a 1/HC slice of d; with fewer, a block sweeps several heads over the
  // full range.  Either way a block moves DCKV * sub elements, so the fold is
  // balanced whichever side of HEADS `sub` falls on.
  const int big = sub >= HEADS;
  const int HC = big ? (sub / HEADS) : 1;
  if (big && j >= HC * HEADS) return;  // the remainder blocks have no slice
  const int hs = big ? HEADS : sub;
  const int dw = DCKV / HC;
  const int d0 = big ? (j / HEADS) * dw : 0;
  const int nt = dw / DPT;
  if (tid >= nt) return;
  const size_t sstr = (size_t)HEADS * DCKV;
  for (int h = big ? (j % HEADS) : j; h < HEADS; h += hs) {
    const float *mm = sml + h * sub;
    const float *ll = sml + (HEADS + h) * sub;
    // Every thread reads the whole (m, l) vector: it broadcasts out of L1 and
    // costs less than a shared-memory reduction plus the barrier it needs.
    float M = -INFINITY;
    for (int sx = 0; sx < sub; ++sx)
      if (ll[sx] > 0.f) M = fmaxf(M, mm[sx]);
    float L = 0.f;
    for (int sx = 0; sx < sub; ++sx)
      if (ll[sx] > 0.f) L += exp2f(mm[sx] - M) * ll[sx];
    const float inv = (L > 0.f) ? 1.f / L : 0.f;
    float acc[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) acc[k] = 0.f;
    const __nv_bfloat16 *bp =
        o_part + (size_t)base * sstr + (size_t)h * DCKV + d0 + tid * DPT;
#pragma unroll 4
    for (int sx = 0; sx < sub; ++sx) {
      const float l = ll[sx];
      // Read unconditionally: a branch here makes each load depend on the
      // previous iteration's test, and the fold then waits out `sub` L2
      // latencies end to end.  A chunk with no rows never wrote its partial,
      // so its slot may hold NaN -- selected away below rather than scaled by
      // a zero weight, which would propagate the NaN.
      const vec_t raw = *reinterpret_cast<const vec_t *>(bp + (size_t)sx * sstr);
      const __nv_bfloat16 *vp = reinterpret_cast<const __nv_bfloat16 *>(&raw);
      const float wv = (l > 0.f) ? exp2f(mm[sx] - M) * l * inv : 0.f;
#pragma unroll
      for (int k = 0; k < DPT; ++k)
        acc[k] += (wv == 0.f) ? 0.f : wv * __bfloat162float(vp[k]);
    }
    __nv_bfloat16 o8[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) o8[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<vec_t *>(out + (size_t)(t * HEADS + h) * DCKV + d0 + tid * DPT) =
        *reinterpret_cast<const vec_t *>(o8);
    if (tid == 0 && d0 == 0) lse[t * HEADS + h] = (L > 0.f) ? (log2f(L) + M) : -INFINITY;
  }
}

// Fold in place instead of in a second kernel.  Every block arrives at its
// token's counter once its partial is visible, then spins until all NSPLIT
// chunks of that token have arrived; block (t, c) then folds head c, which
// works out exactly because NSPLIT == HEADS here.  The spin is safe only
// because the grid fits in one resident wave (checked on the host): with
// 142 KB of smem each SM holds one block, and T*NSPLIT <= SM count means no
// block is ever waiting on one that has not been scheduled.
//
// The counter is never reset.  Each launch's arrivals occupy one generation of
// NSPLIT consecutive values, so a block derives its own generation's end from
// the value atomicAdd handed back -- no reset means no race against a block
// still spinning, and no ABA across back-to-back launches.
template <int NSPLIT, int DPT, int NTHREAD, int MODE, int TS, int OLAY = 0>
__device__ __forceinline__ void fold_token(const __nv_bfloat16 *__restrict__ o_part,
                                           const float *__restrict__ ml_part,
                                           __nv_bfloat16 *__restrict__ out,
                                           float *__restrict__ lse, int *__restrict__ cnt, int t,
                                           int c, int tid, int gen) {
  // Slot 3 is the barrier's entry stamp; slot 2 is its exit.  Together with the
  // block's own start (slot 0) and end (slot 1) that splits the block into
  // body / barrier / fold per block, which differencing grid-wide maxima cannot.
  if (TS && tid == 0) cnt[T_CNT + (t * NSPLIT + c) * 4 + 3] = (int)gtimer();
  // There are NSPLIT blocks per token but only HEADS heads, so when the split
  // is finer than the head count each head is folded by HC blocks, one d-slice
  // each.  HC == 1 is the square case where block c owns head c outright.
  constexpr int HC = NSPLIT / HEADS;
  constexpr int NT = DCKV / DPT / HC;
  static_assert(NT <= NTHREAD && HC * HEADS == NSPLIT, "block folds one d-slice of one head");
  __shared__ int sgo;
  // MODE 6 drops the block-wide membar.gl: __syncthreads already orders every
  // lane's stores within the block, and a release on the arrival carries that
  // whole set to whoever acquires the counter (release cumulativity), so one
  // lane's release replaces 256 lanes' fences.
  if constexpr (MODE < 6) __threadfence();
  __syncthreads();  // every lane's store is published before the arrival
  if constexpr (MODE == 3 || MODE == 4) {
    // The NSPLIT blocks of one token form a cluster, so the hardware cluster
    // barrier replaces the L2 spin entirely.  The threadfence above already
    // published the partials at device scope; the barrier only has to provide
    // the synchronisation edge.
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (MODE == 4) return;
    if (tid < NT)
      combine_item<NSPLIT, DPT, 1, 0, OLAY>(o_part, ml_part, out, lse, t, c / HC, (c % HC) * NT + tid, 0);
    return;
  }
  if constexpr (MODE == 12) {
    // Read the flags before publishing this block's own arrival.  A chunk whose
    // flag is already up has its partial in L2 right now, and on a sparse
    // workload that is nearly all of them -- a token's empty chunks retire
    // microseconds before its full ones.  Issuing their loads here puts an L2
    // round trip in flight across the arrival and the spin instead of behind
    // them; only the stragglers, read stale, have to be fetched again after.
    static_assert(HC == 1, "MODE 12 folds one whole head per block");
    using vec_t = typename std::conditional<DPT == 8, float4,
                  typename std::conditional<DPT == 4, float2, float>::type>::type;
    int *fl = cnt + FLAG_BASE + t * (NSPLIT * CNT_STRIDE);
    __shared__ unsigned srdy;
    if (tid < 32) {
      // Acquire, not volatile: the flag's release edge is what makes the
      // partial behind it visible, and this load is the matching acquire.
      const int x = (tid < NSPLIT) ? ld_acquire_gpu(fl + tid * CNT_STRIDE) : 0;
      const unsigned m = __ballot_sync(0xffffffffu, (tid < NSPLIT) && x >= gen);
      if (tid == 0) srdy = m;
    }
    if (tid == 0) red_add_release_gpu(fl + c * CNT_STRIDE);
    __syncthreads();
    const unsigned rdy = srdy;
    constexpr size_t SSTR = (size_t)HEADS * DCKV;
    const int h = c;
    const __nv_bfloat16 *bp = o_part + (size_t)t * NSPLIT * SSTR + (size_t)h * DCKV + tid * DPT;
    const float *mp = ml_part + ((size_t)(t * 2) * HEADS) * NSPLIT + (size_t)h * NSPLIT;
    const float *lp = mp + (size_t)HEADS * NSPLIT;
    const bool act = tid < NT;
    vec_t raw[NSPLIT];
    float mv[NSPLIT], w[NSPLIT];
    if (act) {
#pragma unroll
      for (int s = 0; s < NSPLIT; ++s)
        raw[s] = *reinterpret_cast<const vec_t *>(bp + (size_t)s * SSTR);
#pragma unroll
      for (int s = 0; s < NSPLIT; s += 4) {
        *reinterpret_cast<float4 *>(&mv[s]) = *reinterpret_cast<const float4 *>(mp + s);
        *reinterpret_cast<float4 *>(&w[s]) = *reinterpret_cast<const float4 *>(lp + s);
      }
    }
    if (tid < NSPLIT) {
      const int *fp = fl + tid * CNT_STRIDE;
      while (ld_acquire_gpu(fp) < gen) {
      }
    }
    __syncthreads();
    if (TS && tid == 0) cnt[T_CNT + (t * NSPLIT + c) * 4 + 2] = (int)gtimer();
    if (!act) return;
    // Re-read the slots that were not yet flagged: those values are stale, and
    // a chunk that had written nothing would otherwise carry last launch's l.
#pragma unroll
    for (int s = 0; s < NSPLIT; ++s)
      if (!((rdy >> s) & 1u)) {
        raw[s] = *reinterpret_cast<const vec_t *>(bp + (size_t)s * SSTR);
        mv[s] = mp[s];
        w[s] = lp[s];
      }
    float M = -INFINITY;
#pragma unroll
    for (int s = 0; s < NSPLIT; ++s) M = (w[s] > 0.f) ? fmaxf(M, mv[s]) : M;
#pragma unroll
    for (int s = 0; s < NSPLIT; ++s) w[s] = (w[s] > 0.f) ? exp2f(mv[s] - M) * w[s] : 0.f;
    float L = 0.f;
#pragma unroll
    for (int s = 0; s < NSPLIT; ++s) L += w[s];
    const float inv = (L > 0.f) ? 1.f / L : 0.f;
    float acc[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) acc[k] = 0.f;
#pragma unroll
    for (int s = 0; s < NSPLIT; ++s) {
      const float wv = w[s] * inv;
      // An empty chunk never wrote its partial, so its slot may hold NaN.
      if (wv == 0.f) continue;
      const __nv_bfloat16 *vp = reinterpret_cast<const __nv_bfloat16 *>(&raw[s]);
#pragma unroll
      for (int k = 0; k < DPT; ++k) acc[k] = fmaf(wv, __bfloat162float(vp[k]), acc[k]);
    }
    __nv_bfloat16 o8[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) o8[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<vec_t *>(out + (size_t)(t * HEADS + h) * DCKV + tid * DPT) =
        *reinterpret_cast<const vec_t *>(o8);
    if (tid == 0) lse[t * HEADS + h] = (L > 0.f) ? (log2f(L) + M) : -INFINITY;
    return;
  }
  if constexpr (MODE == 9 || MODE == 10 || MODE == 13) {
    // One flag per block instead of one counter per token.  A shared counter
    // makes the NSPLIT arrivals of a token serialise as read-modify-writes on
    // a single L2 address; separate flags let them land in parallel, and the
    // wait becomes NSPLIT lanes each polling one flag, so the whole set is read
    // in one L2 round trip instead of one per poll.  MODE 10 additionally gives
    // every flag its own line, trading a wider read for conflict-free writes.
    constexpr int FS = (MODE == 9) ? 1 : CNT_STRIDE;
    int *fl = cnt + FLAG_BASE + t * (NSPLIT * CNT_STRIDE);
    // The generation came from a load issued at block entry, so the arrival
    // needs no return value and the atomic drops off the critical path: a
    // fire-and-forget red retires at issue, where atomicAdd's result took
    // ~1.6 us to come back through L2.  __syncthreads above already published
    // every lane's partial to the block; the release carries that set on.
    if (tid == 0) red_add_release_gpu(fl + c * FS);
    const int v = gen;
    if (tid < NSPLIT) {
      volatile int *fp = fl + tid * FS;
      while (*fp < v) {
      }
    }
    __syncthreads();
    if (TS && tid == 0) cnt[T_CNT + (t * NSPLIT + c) * 4 + 2] = (int)gtimer();
    if (tid < NT)
      // MODE 13 redirects an empty chunk's load to chunk 0 instead of reading
      // its own slot: two thirds of the grid is empty here, so the redirect
      // collapses that traffic onto one line at the cost of making the loads
      // wait on the weights.
      combine_item<NSPLIT, DPT, 1, (MODE == 13), OLAY>(o_part, ml_part, out, lse, t, c / HC,
                                                       (c % HC) * NT + tid, 0);
    return;
  }
  if constexpr (MODE == 14 || MODE == 15) {
    // MODE 7's single counter, but the target was loaded ahead of time and the
    // arrival is a fire-and-forget red: neither the threadfence nor the
    // atomic's return trip is left on the critical path.  The release on the
    // red carries the partials the __syncthreads above published.  MODE 14
    // takes the target at block entry, MODE 15 just before the PV (see there).
    if (tid == 0) {
      int *ap = &cnt[t * CNT_STRIDE];
      red_add_release_gpu(ap);
      volatile int *cp = ap;
      while (*cp < gen) {
      }
      asm volatile("fence.acquire.gpu;" ::: "memory");
      sgo = 1;
    }
    __syncthreads();
    (void)sgo;
    if (TS && tid == 0) cnt[T_CNT + (t * NSPLIT + c) * 4 + 2] = (int)gtimer();
    if (tid < NT)
      combine_item<NSPLIT, DPT, 1, 0, OLAY>(o_part, ml_part, out, lse, t, c / HC,
                                            (c % HC) * NT + tid, 0);
    return;
  }
  if (tid == 0) {
    // One counter per cacheline: all NSPLIT blocks of a token hammer the same
    // address, and packing the tokens together would put every block in the
    // grid on one line.  Poll with a plain volatile load rather than an atomic
    // RMW -- a read-modify-write per spin iteration serialises against the
    // arrivals we are waiting for -- and back off so the polls do not crowd
    // them out either.
    if constexpr (MODE == 7) __threadfence();  // one lane publishes for all
    int *ap = &cnt[t * CNT_STRIDE];
    int prev;
    if constexpr (MODE == 6 || MODE == 8)
      asm volatile("atom.add.release.gpu.s32 %0, [%1], 1;" : "=r"(prev) : "l"(ap) : "memory");
    else
      prev = atomicAdd(ap, 1);
    const int target = (prev / NSPLIT + 1) * NSPLIT;
    if constexpr (MODE == 6) {
      int v;
      do {
        asm volatile("ld.acquire.gpu.s32 %0, [%1];" : "=r"(v) : "l"(ap) : "memory");
      } while (v < target);
    } else {
      volatile int *cp = ap;
      while (*cp < target) {
      }
      // MODE 8 polls with plain loads and pays for the acquire exactly once.
      if constexpr (MODE == 8) asm volatile("fence.acquire.gpu;" ::: "memory");
    }
    sgo = 1;
  }
  __syncthreads();
  (void)sgo;
  if (TS && tid == 0) cnt[T_CNT + (t * NSPLIT + c) * 4 + 2] = (int)gtimer();
  // MODE 2 stops after the barrier: it times the barrier alone (and produces
  // wrong output, so it is a probe only).
  if (MODE == 2) return;
  if (tid < NT)
    combine_item<NSPLIT, DPT, 1, MODE == 5, OLAY>(o_part, ml_part, out, lse, t, c / HC,
                                            (c % HC) * NT + tid, 0);
}

template <int R, int NWARP, int NWQ, int NGK, int GMODE, int STAGE, int KA, int KS, int PVB,
          int FUSE, int DPT, int BAL, int GORD>
__device__ __forceinline__ void split_body(
    const __nv_bfloat16 *__restrict__ q_nope, const __nv_bfloat16 *__restrict__ q_pe,
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    const int *__restrict__ indices, __nv_bfloat16 *__restrict__ o_part,
    float *__restrict__ ml_part, __nv_bfloat16 *__restrict__ out, float *__restrict__ lse,
    int *__restrict__ cnt, int nsplit, int T, float scale_l2e) {
  constexpr int NTHREAD = NWARP * 32;
  constexpr int LDP = R + 8;
  constexpr int NSPL = TOPKC / R;
  // FUSE 11 is FUSE 10's barrier paired with the head-major o_part layout.
  constexpr int OLAY = (FUSE == 11);
  constexpr int MODEF = (FUSE == 11) ? 10 : FUSE;
  // Only NWQ of the NWARP warps run the QK product.  Every participating warp
  // has to stream all of Q through ldmatrix, so Q's smem read traffic is
  // NWQ * 18 KB while K's is a fixed 147 KB -- QK is shared-memory-bandwidth
  // bound, and shrinking NWQ (more n-tiles per warp) is what makes it faster.
  // The gather and the PV product still use all NWARP warps.
  constexpr int NTQ = R / 8 / NWQ;    // QK n-tiles (8 kv rows each) per warp
  constexpr int KTP = R / 16;         // PV k-steps
  constexpr int NGP = DCKV / 16 / NWARP;  // PV groups (16 dims each) per warp
  static_assert(NTQ >= 1 && NGP >= 1 && NWQ <= NWARP, "bad tiling");
  // PVB 5 exchanges P thread-major instead of head-major.  A QK warp's n-tile
  // pair covers exactly one PV k-step (TILE_ROW is (w*NTQ+n)*8 when NB == 1),
  // and inside it the accumulator's lane map already equals the m16n8k16 A
  // fragment's, so the exchange needs no transpose at all -- see the store site.
  constexpr bool PTM = (PVB == 5);
  static_assert(!PTM || (!BAL && NTQ % 2 == 0 && NWQ * (NTQ / 2) == KTP),
                "PVB 5 needs each QK warp to own whole k-steps");
  static_assert(KTP * 32 * 16 <= HEADS * LDP * 2, "thread-major P overruns the Ps slab");

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *Qs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *Ks = Qs + HEADS * LDK;
  __nv_bfloat16 *Ps = Ks + R * LDK;
  // PVB 5's view of the same slab: KTP planes of 32 lanes holding 16 B each.
  uint4 *Pt = reinterpret_cast<uint4 *>(Ps);
  int *sidx = reinterpret_cast<int *>(Ps + HEADS * LDP);
  float *swm = reinterpret_cast<float *>(sidx + R);
  float *swl = swm + NWARP * HEADS;
  // 4 * MAXT ints of plan scratch (BAL only), then the mbarriers.
  int *splan = reinterpret_cast<int *>(swl + NWARP * HEADS);
  const unsigned bar0 = sm_addr(splan + 4 * MAXT);  // NGK mbarriers, GMODE >= 1 only
  // The k-split scratch aliases Qs, which is dead once the QK loop is done.
  // Keeping the block's smem footprint unchanged matters: growing it past the
  // Ks + Qs + Ps working set slows the PV mma down sharply.
  float *sred = reinterpret_cast<float *>(Qs);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  // GORD 1 makes a token's NSPLIT blocks consecutive in launch order: they
  // share the token's Q and index row, so neighbouring block ids keep those
  // reads in the same L2 neighbourhood.
  int t = GORD ? (int)blockIdx.y : (int)blockIdx.x;
  int c = GORD ? (int)blockIdx.x : (int)blockIdx.y;
  // Rows this block covers, and where they start inside the token's live list.
  // nrp rounds nr up to the 16-row granularity of a PV k-step, so every stage
  // below can skip whole tiles past it instead of masking row by row.
  int nr = R, nrp = R, rbase = c * R, sub = 1, sbase = 0, jsub = 0;
  if constexpr (BAL) {
    // Live rows are a prefix of each token's index list, so a per-token count
    // is the whole shape of the workload.  Every block derives the same plan
    // from it independently -- no communication, no extra launch.
    int *sS = splan + MAXT, *sO = splan + 2 * MAXT, *stl = splan + 3 * MAXT;
    for (int t2 = warp; t2 < T; t2 += NWARP) {
      const int *row = indices + (size_t)t2 * TOPKC;
      // The live tile count falls out of two 32-wide rounds -- one at 64-row
      // granularity, one at 16 inside whichever 64-row span the edge lands in.
      // Counting the whole list instead costs 16x the L2 sectors, and with
      // every block running the same plan that difference is measurable.
      constexpr int C0 = TOPKC / 32;  // rows per lane in round 0
      const unsigned m0 = __ballot_sync(0xffffffffu, row[lane * C0] < 0);
      const int k0 = m0 ? (__ffs(m0) - 1) : 32;
      int tl2 = 0;
      if (k0 > 0) {
        const int b0 = (k0 - 1) * C0;
        const unsigned m1 =
            __ballot_sync(0xffffffffu, (lane >= C0 / 16) || row[b0 + lane * 16] < 0);
        const int k1 = m1 ? (__ffs(m1) - 1) : (C0 / 16);
        tl2 = (k0 - 1) * (C0 / 16) + k1;
      }
      if (lane == 0) stl[t2] = tl2;
    }
    __syncthreads();
    if (tid == 0) {
      // Smallest per-block tile count that still fits the grid.  Descending so
      // the last assignment written is the tightest one; every divisor is a
      // compile-time constant here, which keeps this off the integer divider.
      constexpr int MTILE = R / 16;
      // Coarsest split first, so sS is defined even if no split fits the grid.
      for (int t2 = 0; t2 < T; ++t2) {
        const int st = (stl[t2] + MTILE - 1) / MTILE;
        sS[t2] = st < 1 ? 1 : st;
      }
#pragma unroll
      for (int mm = MTILE; mm >= 1; --mm) {
        int s2 = 0;
        for (int t2 = 0; t2 < T; ++t2) {
          const int st = (stl[t2] + mm - 1) / mm;
          s2 += st < 1 ? 1 : st;
        }
        if (s2 <= NBAL)
          for (int t2 = 0; t2 < T; ++t2) {
            const int st = (stl[t2] + mm - 1) / mm;
            sS[t2] = st < 1 ? 1 : st;  // an empty token still needs one writer
          }
      }
      int o = 0;
      for (int t2 = 0; t2 < T; ++t2) {
        sO[t2] = o;
        o += sS[t2];
      }
      sO[MAXT - 1] = o;  // total blocks with work
    }
    __syncthreads();
    const int b = blockIdx.x;
    if (b >= sO[MAXT - 1]) return;  // grid is sized for the worst case
    int tt = 0;
    while (tt + 1 < T && sO[tt + 1] <= b) ++tt;
    t = tt;
    c = b;
    sbase = sO[tt];
    sub = sS[tt];
    jsub = b - sbase;
    const int tl = stl[tt];
    const int q0 = (int)(((long long)jsub * tl) / sub), q1 = (int)(((long long)(jsub + 1) * tl) / sub);
    rbase = q0 * 16;
    nrp = (q1 - q0) * 16;
    nr = nrp;
  }

  // Every (token, chunk) slot is written every launch -- l == 0 marks "no rows
  // here", so the combine needs no stamp array and no scratch reset.
  // STAGE 8 runs the full path and reports per-phase cycle counts through the
  // m slots of ml_part.  Differencing truncated STAGEs does not measure a
  // phase: an early return lets the compiler drop upstream work whose only
  // consumer was downstream, so the delta understates the phase that was cut.
  long long ck0 = 0, ckg = 0, ck1 = 0, ck2 = 0, ck2b = 0, ck3 = 0;
  // Under BAL the slot is the global block id and nsplit is NBAL, so the
  // per-token stride disappears; the head stride is the same either way.
  float *mlm = ml_part + (BAL ? 0 : ((size_t)(t * 2) * HEADS) * nsplit) + c;
  float *mll = mlm + (size_t)HEADS * nsplit;
  // STAGE 9 is the same floor as STAGE 0 but asks for a token smem allocation,
  // which separates the cost of setting up the grid from the cost of handing
  // every block 142 KB of shared memory.
  if (STAGE == 0 || STAGE == 9) {
    if (tid < HEADS) {
      mlm[(size_t)tid * nsplit] = -INFINITY;
      mll[(size_t)tid * nsplit] = 0.f;
    }
    return;
  }

  // STAGE 10 stamps %globaltimer (a GPU-wide ns clock, unlike clock64 which is
  // per-SM) at block entry and exit, so the host can see how far apart the
  // blocks actually start -- the spread is invisible to any per-block timer.
  if (STAGE == 10 && tid == 0) cnt[T_CNT + (t * nsplit + c) * 4] = (int)gtimer();
  if (STAGE == 8 && tid == 0) ck0 = clock64();
  // Read this block's own arrival flag now, microseconds before the fold needs
  // it.  No other block ever writes it, so the value is this launch's
  // generation minus one, and the L2 round trip hides behind the entire body.
  __shared__ int sgen;
  if (tid == 0)
    sgen = (FUSE == 14 && !BAL)
               // MODE 14 wants the token counter's next multiple of NSPL.  This
               // block has not arrived yet, so the counter is strictly below
               // that multiple however many siblings landed in between, and
               // rounding down recovers the generation exactly.
               ? (ld_relaxed_gpu(cnt + t * CNT_STRIDE) / NSPL + 1) * NSPL
           : (FUSE >= 9 && FUSE != 15 && !BAL)
               ? ld_relaxed_gpu(cnt + FLAG_BASE + t * (NSPL * CNT_STRIDE) + c * CNT_STRIDE) + 1
               : 0;
  // Index loads land in registers first so the (independent) Q copies can be
  // issued while the index fetch is still in flight.
  constexpr int IPT = (R + NTHREAD - 1) / NTHREAD;
  // GMODE 8 gives every thread the index of the row whose copy it will issue,
  // so the half of the block that moves Kpe reads the same 512 B index line
  // itself instead of receiving it through shared memory.
  constexpr bool G8 = (GMODE == 8 || GMODE == 9);
  static_assert(!G8 || (IPT == 1 && NTHREAD == 2 * R && NGK == 1 && R % 32 == 0),
                "GMODE 8/9 want exactly one copy per thread");
  int iv[IPT];
#pragma unroll
  for (int u = 0; u < IPT; ++u) {
    int i = G8 ? (tid & (R - 1)) : (tid + u * NTHREAD);
    iv[u] = (i < nr) ? indices[(size_t)t * TOPKC + rbase + i] : -1;
  }
  {
    if constexpr (G8) {
      // Each thread declares its own transaction count below, so the init has
      // to be visible block-wide first.  The sync is free here: it sits on top
      // of the index fetch, which is still in flight and is not waited on.
      // GMODE 9 arrives once per warp instead of once per thread.
      if (tid == 0) mbar_init(bar0, (GMODE == 9) ? (NTHREAD / 32) : NTHREAD);
      __syncthreads();
    }
    const __nv_bfloat16 *qn = q_nope + (size_t)t * HEADS * DCKV;
    const __nv_bfloat16 *qp = q_pe + (size_t)t * HEADS * DKPE;
    for (int i = tid; i < HEADS * 64; i += NTHREAD)
      cp_async16(&Qs[(i >> 6) * LDK + (i & 63) * 8], qn + (size_t)(i >> 6) * DCKV + (i & 63) * 8);
    for (int i = tid; i < HEADS * 8; i += NTHREAD)
      cp_async16(&Qs[(i >> 3) * LDK + DCKV + (i & 7) * 8],
                 qp + (size_t)(i >> 3) * DKPE + (i & 7) * 8);
    cp_commit();  // oldest group: all of Q
    if (GMODE >= 1 && !G8 && tid == 0)
#pragma unroll
      for (int g = 0; g < NGK; ++g) mbar_init(bar0 + g * 8, NTHREAD);
  }
  // GMODE 8 issues the copy straight out of the register the index landed in.
  // The index list is a prefix of live pages, so "is my row live" is a local
  // test, and the per-thread expect_tx above removes the last block-wide
  // quantity the gather needed -- no smem round trip, no barrier, and no
  // serialised byte-count computation between the load and the copy.
  if constexpr (G8) {
    const int r = tid & (R - 1);
    const unsigned bytes = (tid < R) ? DCKV * 2 : DKPE * 2;
    if constexpr (GMODE == 9) {
      // Every lane of a warp moves the same number of bytes (R is a multiple of
      // the warp size, so a warp never straddles the Kc/Kpe split), so one
      // ballot collapses the warp's whole declaration into a single mbarrier
      // update -- 8 per block instead of 256, all off the index's latency.
      const unsigned live = __ballot_sync(0xffffffffu, iv[0] >= 0);
      if ((tid & 31) == 0 && live) mbar_expect(bar0, (unsigned)__popc(live) * bytes);
      __syncwarp();  // the declaration must precede every copy it accounts for
    } else if (iv[0] >= 0) {
      mbar_expect(bar0, bytes);
    }
    if (iv[0] >= 0) {
      if (tid < R)
        tma_load(sm_addr(&Ks[r * LDK]), ckv + (size_t)iv[0] * DCKV, bytes, bar0);
      else
        tma_load(sm_addr(&Ks[r * LDK + DCKV]), kpe + (size_t)iv[0] * DKPE, bytes, bar0);
    }
    if (GMODE != 9 || (tid & 31) == 0) mbar_arrive(bar0);
  }
  int any = 0;
#pragma unroll
  for (int u = 0; u < IPT; ++u) {
    int i = G8 ? (tid & (R - 1)) : (tid + u * NTHREAD);
    const bool mine = !G8 || (tid < R);  // GMODE 8/9 duplicate the index
    if (i < R && mine) sidx[i] = iv[u];
    any |= (iv[u] >= 0) && mine;
  }
  // Count the live rows, do not just test for them: the index list is a prefix
  // of valid pages, so the number of threads holding one is the chunk's row
  // count.  The gather then moves only those rows -- a chunk holding 4 of 128
  // was otherwise paying a full chunk's TMA latency and setting the pace of the
  // whole token's barrier.
  const int nlrow_ = __syncthreads_count(any);
  const int nlrow = BAL ? nrp : ((IPT == 1) ? nlrow_ : (nlrow_ ? R : 0));
  if (nlrow_ == 0) {
    cp_wait<0>();
    if (tid < HEADS) {
      mlm[(size_t)tid * nsplit] = -INFINITY;
      mll[(size_t)tid * nsplit] = 0.f;
    }
    // An empty chunk never reaches the point where FUSE 15 takes its target,
    // so it takes one here.  Nothing hides it, but this path is off the
    // critical path by construction: it retires long before a full chunk.
    int egen = 0;
    if (FUSE == 15 && tid == 0) egen = (ld_relaxed_gpu(cnt + t * CNT_STRIDE) / NSPL + 1) * NSPL;
    if constexpr (BAL)
      bal_fold<DPT, NTHREAD>(o_part, ml_part, out, lse, cnt, t, sbase, sub, jsub, tid);
    else if constexpr (FUSE)
      fold_token<NSPL, DPT, NTHREAD, MODEF, STAGE == 10, OLAY>(o_part, ml_part, out, lse, cnt, t,
                                                               c, tid, FUSE == 15 ? egen : sgen);
    if (STAGE == 10 && tid == 0) cnt[T_CNT + (t * nsplit + c) * 4 + 1] = (int)gtimer();
    return;
  }

  // Kc is fetched in NGK commit groups (Kpe is the last one), so the QK product
  // below can consume slab g while slabs > g are still in flight instead of
  // waiting for every byte up front.  Slabs must stay wide: each one fetches
  // DCKV/NGK contiguous bytes per row, and cutting that below ~512 B costs more
  // in DRAM locality than the overlap wins back.
  // Kc is fetched in NGK slabs so the QK product can consume slab g while the
  // later slabs are still in flight.  Under GMODE 1 a slab is one bulk copy per
  // row against its own mbarrier; under GMODE 0 it is a cp.async commit group.
  // Kpe rides along with the last slab.
  // GMODE 2 reads NGK as a row-batch count instead of a feature-slab count:
  // every batch carries all 576 features for R/NGK rows, so its k chain stays
  // whole and the QK below can run on batch b while batch b+1 is still in
  // flight.  NB = 1 collapses all of this to the original single-shot layout.
  constexpr int NB = (GMODE == 2 || GMODE == 4) ? NGK : 1;
  constexpr int RB = R / NB;
  constexpr int FSL = (GMODE == 2 || GMODE == 4) ? DCKV : DCKV / NGK;  // Kc per slab
  constexpr int CSL = FSL / 8;     // 16 B chunks per row per slab (GMODE 0)
  constexpr int NG = NGK + 1;      // groups after Q (cp.async numbering)
  static_assert(NB > 1 || (FSL * NGK == DCKV && (CSL & (CSL - 1)) == 0), "slab shape");
  // GMODE 3 holds slab g+1 back until slab g has landed.  Issuing every copy up
  // front does not make slab 0 arrive any sooner -- measured: with all of them
  // in flight they share DRAM bandwidth and complete together -- whereas
  // withholding half the bytes lets slab 0 land in half the time, so the QK
  // over it runs while the rest is still moving.  Aggregate bandwidth stays
  // saturated because the other resident blocks issue across the gap.
  auto issue_slab = [&](int g) {
    if (g >= NGK) return;
    const int n = (g == NGK - 1) ? 2 * R : R;  // the last slab carries Kpe too
    for (int i = tid; i < n; i += NTHREAD) {
      const int r = (i < R) ? i : i - R;
      const int row = max(sidx[r], 0);
      if (i < R)
        tma_load(sm_addr(&Ks[r * LDK + g * FSL]), ckv + (size_t)row * DCKV + g * FSL, FSL * 2,
                 bar0 + g * 8);
      else
        tma_load(sm_addr(&Ks[r * LDK + DCKV]), kpe + (size_t)row * DKPE, DKPE * 2, bar0 + g * 8);
    }
  };
  // GMODE 4 is the same staging as GMODE 3 but cut along rows instead of
  // features, so every copy still moves a whole 1152 B row.  Slab-cutting costs
  // more bandwidth than the overlap wins back (measured): a 512 B per-row copy
  // streams noticeably slower than a 1152 B one.
  auto issue_batch = [&](int b) {
    if (b >= NB) return;
    for (int i = tid; i < 2 * RB; i += NTHREAD) {
      const int half = i / RB, r = b * RB + (i - half * RB);
      const int row = max(sidx[r], 0);
      if (half == 0)
        tma_load(sm_addr(&Ks[r * LDK]), ckv + (size_t)row * DCKV, DCKV * 2, bar0 + b * 8);
      else
        tma_load(sm_addr(&Ks[r * LDK + DCKV]), kpe + (size_t)row * DKPE, DKPE * 2, bar0 + b * 8);
    }
  };
  const int grow = nlrow;
  // Rows the LSU path takes off the TMA's hands (0 = the TMA fetches all).
  constexpr int LSU_STEP = (GMODE == 6) ? 2 : (GMODE == 7) ? 4 : 0;
  const int gtma = LSU_STEP ? (grow - grow / LSU_STEP) : grow;
  if constexpr (GMODE >= 1) {
    // A masked-out row is still fetched, from row 0: QK overwrites its score
    // with -inf and PV weights it by zero, so the data is inert, every such
    // fetch hits one cache line, and no uninitialised bf16 is left in smem.
    // One thread declares the whole expected-transaction count per barrier; the
    // rest just arrive.  Every count must be registered before any copy is
    // issued -- a completing copy decrements them.  GMODE 8/9 have already
    // done both, per thread or per warp, before this point.
    if constexpr (!G8) {
    if (tid == 0) {
#pragma unroll
      for (int g = 0; g < NGK; ++g) {
        // Under NB > 1 barrier g owns row batch g, so it expects only the live
        // rows of that batch; the trailing batches of a short chunk expect
        // nothing and open on the arrive alone.
        const int gb = (NB > 1) ? min(max(grow - g * RB, 0), RB) : gtma;
        mbar_arrive_expect(bar0 + g * 8,
                           (NB > 1) ? gb * (DCKV + DKPE) * 2
                                    : gb * FSL * 2 +
                                          ((GMODE != 5 && g == NGK - 1) ? gb * DKPE * 2 : 0));
      }
    } else {
#pragma unroll
      for (int g = 0; g < NGK; ++g) mbar_arrive(bar0 + g * 8);
    }
    // 2 * R bulk copies (Kc row, Kpe row) spread over the block's threads.
    // Under GMODE 2 the index is ordered batch-major so batch 0 is issued
    // first and lands first; the QK below consumes it while batch 1 flies.
    // GMODE 2 moves each row in two copies (Kc, Kpe); the slab modes move one
    // copy per feature slab plus one Kpe.
    // GMODE 5 leaves Kpe out of the bulk-copy set: at 128 B per row it is half
    // the copies but an eighth of the bytes, and the TMA unit charges per copy,
    // not per byte.  Moving it to cp.async puts it on the LSU path, which runs
    // alongside the TMA engine instead of queueing behind it.
    constexpr int NCP = (GMODE == 2) ? 2 * R : ((GMODE == 5) ? NGK * R : (NGK + 1) * R);
    const int ncp = NCP;
    if constexpr (GMODE == 3) issue_slab(0);
    else if constexpr (GMODE == 4) issue_batch(0);
    else for (int i = tid; i < ncp; i += NTHREAD) {
      if constexpr (GMODE == 2) {
        const int b = i / (2 * RB), j = i - b * (2 * RB);
        const int half = j / RB, r = b * RB + (j - half * RB);
        const int row = sidx[r];
        if (r >= grow) {
        } else if (half == 0)
          tma_load(sm_addr(&Ks[r * LDK]), ckv + (size_t)row * DCKV, DCKV * 2, bar0 + b * 8);
        else
          tma_load(sm_addr(&Ks[r * LDK + DCKV]), kpe + (size_t)row * DKPE, DKPE * 2, bar0 + b * 8);
      } else {
        const int g = i / R, r = i - g * R;
        const int row = sidx[r];
        if (r >= grow || (LSU_STEP && (r % LSU_STEP) == LSU_STEP - 1)) {
        } else if (g < NGK)
          tma_load(sm_addr(&Ks[r * LDK + g * FSL]), ckv + (size_t)row * DCKV + g * FSL, FSL * 2,
                   bar0 + g * 8);
        else
          tma_load(sm_addr(&Ks[r * LDK + DCKV]), kpe + (size_t)row * DKPE, DKPE * 2,
                   bar0 + (NGK - 1) * 8);
      }
    }
    }
    // Rows the gather skipped are still read by QK and PV, and the mma turns
    // 0 * NaN into NaN, so they have to hold zeros rather than whatever the
    // previous launch left in smem.  Issued after the copies so it overlaps
    // them; the copies only write rows below nlrow, so the two never collide.
    // Zeroing beats shortening the PV k-loop: that would make the trip count a
    // run-time value and cost the full chunks their unrolled mma schedule.
    for (int i = tid; i < (R - nlrow) * (DKTOT / 8); i += NTHREAD) {
      const int q = i / (DKTOT / 8), j = i - q * (DKTOT / 8);
      *reinterpret_cast<float4 *>(&Ks[(nlrow + q) * LDK + j * 8]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    // GMODE 6/7 hand every LSU_STEP-th row to cp.async instead of the TMA.
    // Both engines are latency-bound long before they are bandwidth-bound at
    // this row count, so running them side by side shortens the wait rather
    // than just dividing a fixed cost; the LSU path is the slower of the two
    // per byte, which is why GMODE 7 gives it only a quarter of the rows.
    if constexpr (LSU_STEP) {
      constexpr int CPR = DCKV / 8 + DKPE / 8;  // 16 B chunks in one row
      for (int i = tid; i < (R / LSU_STEP) * CPR; i += NTHREAD) {
        const int q = i / CPR, j = i - q * CPR;
        const int r2 = q * LSU_STEP + LSU_STEP - 1;
        if (r2 >= grow) continue;
        const int row = sidx[r2];
        if (j < DCKV / 8)
          cp_async16(&Ks[r2 * LDK + j * 8], ckv + (size_t)row * DCKV + j * 8);
        else
          cp_async16(&Ks[r2 * LDK + DCKV + (j - DCKV / 8) * 8],
                     kpe + (size_t)row * DKPE + (j - DCKV / 8) * 8);
      }
      cp_commit();
    }
    if constexpr (GMODE == 5) {
      for (int i = tid; i < R * 8; i += NTHREAD) {
        const int r2 = i >> 3, j = i & 7;
        if (r2 >= grow) continue;  // the zeroing loop above owns the dead rows
        cp_async16(&Ks[r2 * LDK + DCKV + j * 8], kpe + (size_t)sidx[r2] * DKPE + j * 8);
      }
      cp_commit();
    }
    cp_wait<0>();  // Q (and Kpe under GMODE 5)
  } else {
#pragma unroll
    for (int g = 0; g < NGK; ++g) {
      for (int i = tid; i < R * CSL; i += NTHREAD) {
        int r2 = i / CSL, j = i & (CSL - 1), rw = sidx[r2];
        __nv_bfloat16 *d = &Ks[r2 * LDK + g * FSL + j * 8];
        if (rw >= 0)
          cp_async16(d, ckv + (size_t)rw * DCKV + g * FSL + j * 8);
        else
          *reinterpret_cast<float4 *>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      cp_commit();
    }
    for (int i = tid; i < R * 8; i += NTHREAD) {
      int r2 = i >> 3, j = i & 7, rw = sidx[r2];
      __nv_bfloat16 *d = &Ks[r2 * LDK + DCKV + j * 8];
      if (rw >= 0)
        cp_async16(d, kpe + (size_t)rw * DKPE + j * 8);
      else
        *reinterpret_cast<float4 *>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    cp_commit();
  }

  if (STAGE == 1) {
    if constexpr (GMODE >= 1) mbar_wait(bar0 + (NGK - 1) * 8);
    else cp_wait<0>();
    __syncthreads();
    if (tid < HEADS) {
      mlm[(size_t)tid * nsplit] = __bfloat162float(Ks[0]) + __bfloat162float(Qs[0]) +
                                  __bfloat162float(Ks[(R - 1) * LDK + DKTOT - 1]);
      mll[(size_t)tid * nsplit] = 1.f;
    }
    return;
  }

  // ---- S = Q @ [Kc|Kp]^T : M = heads, N = kv rows, K = 576 ----
  const int hA = lane >> 2;        // accumulator rows held by this lane
  const int hB = hA + 8;
  const int col = (lane & 3) * 2;  // kv-row pair within an n-tile
  // NTQ * KACC independent accumulator chains: with ~1 CTA per SM there is no
  // other warp to hide the mma latency, so ILP inside the warp is what matters.
  // The k loop stays rolled -- fully unrolling 32 k-steps x NTQ mma thrashes the
  // instruction cache and costs more than the scheduling freedom buys.
  constexpr int KACC = KA ? KA : ((NTQ >= 4) ? 1 : (4 / NTQ));
  constexpr int NTB = NTQ / NB;  // n-tiles a warp owns inside one row batch
  static_assert(NTB * NB == NTQ, "row batches must split the warp's n-tiles");
  // n-tile -> first kv row it covers.  Batch-major so tiles of a batch are
  // contiguous in n, which is what lets the QK loop below run one batch at a
  // time.  With NB == 1 this is exactly (w * NTQ + n) * 8.
#define TILE_ROW(w, n) \
  (BAL ? ((n) * NWQ + (w)) * 8 : (((n) / NTB) * RB + ((w) * NTB + (n) % NTB) * 8))
  // Split the ck0..ck1 span: under GMODE 1 the QK loop's first k-group waits on
  // the same barrier, so waiting here is a no-op for timing purposes (parity
  // waits are idempotent) but tells us how much of the span is pure fetch.
  if (STAGE == 8) {
    if constexpr (GMODE >= 1) mbar_wait(bar0);
    if (tid == 0) ckg = clock64();
  }
  float sv[NTQ][4];
  float mA = -INFINITY, mB = -INFINITY;
  {
    float ka[NTQ][KACC][4];
#pragma unroll
    for (int n = 0; n < NTQ; ++n)
#pragma unroll
      for (int a = 0; a < KACC; ++a)
#pragma unroll
        for (int u = 0; u < 4; ++u) ka[n][a][u] = 0.f;
    const int live = warp < NWQ * KS;
    const int ks = live ? warp / NWQ : 0;   // which k range this warp owns
    const int wq = live ? warp % NWQ : 0;   // which kv rows it owns
    const unsigned qbase = sm_addr(&Qs[(lane & 15) * LDK + (lane >> 4) * 8]);
    // Tiles this warp actually has rows for.  Interleaved tiles make this a
    // simple prefix: tiles 0..nlive-1 are live, the rest are skipped whole.
    // Tiles this warp has rows for.  Interleaved tiles make this a prefix:
    // tiles 0..nlive-1 are live, the rest are skipped whole.
    const int nlive = BAL ? min(NTQ, max(0, (nrp / 8 - wq + NWQ - 1) / NWQ)) : NTQ;
    unsigned kbase[NTQ];
#pragma unroll
    for (int n = 0; n < NTQ; ++n)
      kbase[n] = sm_addr(&Ks[(TILE_ROW(wq, n) + (lane & 7)) * LDK + ((lane >> 3) & 1) * 8]);
    auto qk_batch = [&](auto GC) {
      constexpr int b = decltype(GC)::value;
      // Batches 0..b have landed; later ones may still be in flight.
      mbar_wait(bar0 + b * 8);
      if constexpr (GMODE == 4) issue_batch(b + 1);
      __syncthreads();
      if (warp < NWQ) {
        for (int kb = 0; kb < DKTOT / 16; kb += KACC) {
          unsigned qa[KACC][4], kk[KACC][NTB][2];
#pragma unroll
          for (int a = 0; a < KACC; ++a) ldm_x4(qa[a], qbase + (kb + a) * 32);
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int nb = 0; nb < NTB; ++nb) ldm_x2(kk[a][nb], kbase[b * NTB + nb] + (kb + a) * 32);
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int nb = 0; nb < NTB; ++nb)
              mma16816(ka[b * NTB + nb][a], qa[a], kk[a][nb][0], kk[a][nb][1]);
        }
      }
    };
    auto qk_slab = [&](auto GC) {
      constexpr int g = decltype(GC)::value;
      constexpr int kb0 = (g < NGK) ? g * (FSL / 16) : DCKV / 16;
      constexpr int KN = (g < NGK) ? (FSL / 16) : (DKPE / 16);
      static_assert(KN % KACC == 0, "slab must hold whole chains");
      // Slabs 0..g have landed; later ones may still be in flight.
      if constexpr (GMODE >= 1) {
        if constexpr (g < NGK) {
          mbar_wait(bar0 + g * 8);
          if constexpr (GMODE == 3) issue_slab(g + 1);
        }
      } else cp_wait<NG - 1 - g>();
      __syncthreads();
      if (warp < NWQ && nlive > 0) {
        for (int kb = kb0; kb < kb0 + KN; kb += KACC) {
          // Every ldmatrix of the step is issued before the first mma consumes
          // one: an ldm_x2 feeding the mma right behind it stalls on its own
          // result, and with only ~1 CTA per SM there is no other warp to fill
          // the gap.  Costs KACC * NTQ * 2 extra registers.
          unsigned qa[KACC][4], kk[KACC][NTQ][2];
#pragma unroll
          for (int a = 0; a < KACC; ++a) ldm_x4(qa[a], qbase + (kb + a) * 32);  // 16 bf16 = 32 B
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int n = 0; n < NTQ; ++n)
              if (!BAL || n < nlive) ldm_x2(kk[a][n], kbase[n] + (kb + a) * 32);
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int n = 0; n < NTQ; ++n)
              if (!BAL || n < nlive) mma16816(ka[n][a], qa[a], kk[a][n][0], kk[a][n][1]);
        }
      }
    };
    if constexpr (KS > 1) {
      // Splitting k is what finally puts two warps on every SMSP: the n tiles
      // were already exhausted at NWQ = 4, and with one warp per SMSP the
      // ldmatrix -> mma dependency is fully exposed.  Each k group runs the
      // whole chain for its own slice of the 576 features.
      static_assert(GMODE == 1 && NGK == 1 && NWQ * KS <= NWARP, "k-split shape");
      static_assert((DKTOT / 16) % KS == 0 && ((DKTOT / 16) / KS) % KACC == 0, "k-split range");
      constexpr int KPG = (DKTOT / 16) / KS;
      mbar_wait(bar0);
      __syncthreads();
      if (live) {
        for (int kb = ks * KPG; kb < ks * KPG + KPG; kb += KACC) {
          unsigned qa[KACC][4], kk[KACC][NTQ][2];
#pragma unroll
          for (int a = 0; a < KACC; ++a) ldm_x4(qa[a], qbase + (kb + a) * 32);
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int n = 0; n < NTQ; ++n) ldm_x2(kk[a][n], kbase[n] + (kb + a) * 32);
#pragma unroll
          for (int a = 0; a < KACC; ++a)
#pragma unroll
            for (int n = 0; n < NTQ; ++n) mma16816(ka[n][a], qa[a], kk[a][n][0], kk[a][n][1]);
        }
      }
    } else if constexpr (NB > 1) seq_for<NB>(qk_batch);
    else seq_for<NG>(qk_slab);
    if (STAGE == 8 && tid == 0) ck1 = clock64();
    float sacc[NTQ][4];
    if (live) {
#pragma unroll
      for (int n = 0; n < NTQ; ++n)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          float s = ka[n][0][u];
#pragma unroll
          for (int a = 1; a < KACC; ++a) s += ka[n][a][u];
          sacc[n][u] = s;
        }
    }
    if constexpr (KS > 1) {
      // Lane-major so each (n, u) plane is 32 consecutive floats: one wavefront,
      // no bank conflict.
      __syncthreads();  // every warp is done reading Qs
      if (warp >= NWQ && live)
#pragma unroll
        for (int n = 0; n < NTQ; ++n)
#pragma unroll
          for (int u = 0; u < 4; ++u)
            sred[(warp - NWQ) * (NTQ * 4 * 32) + (n * 4 + u) * 32 + lane] = sacc[n][u];
      __syncthreads();
      // STAGE 6 drops only the read half of the cross-warp reduction: same
      // warps, same QK, same PV, wrong numbers.  It isolates whether the PV
      // slowdown comes from the reduction itself or from warps 4-7 having run
      // the QK chain at all.
      if (warp < NWQ && STAGE != 6)
#pragma unroll
        for (int j = 1; j < KS; ++j)
#pragma unroll
          for (int n = 0; n < NTQ; ++n)
#pragma unroll
            for (int u = 0; u < 4; ++u)
              sacc[n][u] +=
                  sred[((j - 1) * NWQ + warp) * (NTQ * 4 * 32) + (n * 4 + u) * 32 + lane];
    }
    if (warp < NWQ) {
      if constexpr (STAGE == 4) {
        float z = 0.f;
#pragma unroll
        for (int n = 0; n < NTQ; ++n)
#pragma unroll
          for (int u = 0; u < 4; ++u) z += sacc[n][u];
        if (lane == 0) mlm[(size_t)warp * nsplit] = z;
      }

    if constexpr (STAGE != 4) {
    // ---- base-2 softmax over the whole chunk ----
    // Specialising this loop for a full chunk (no dead rows, so no sidx read)
    // measures 95 ns slower: the duplicated tile loop costs more in scheduling
    // than the two smem reads per tile it saves.
#pragma unroll
    for (int n = 0; n < NTQ; ++n) {
      int c0 = TILE_ROW(warp, n) + col;
      bool v0 = sidx[c0] >= 0, v1 = sidx[c0 + 1] >= 0;
      sv[n][0] = v0 ? sacc[n][0] * scale_l2e : -INFINITY;
      sv[n][1] = v1 ? sacc[n][1] * scale_l2e : -INFINITY;
      sv[n][2] = v0 ? sacc[n][2] * scale_l2e : -INFINITY;
      sv[n][3] = v1 ? sacc[n][3] * scale_l2e : -INFINITY;
      mA = fmaxf(mA, fmaxf(sv[n][0], sv[n][1]));
      mB = fmaxf(mB, fmaxf(sv[n][2], sv[n][3]));
    }
      mA = warp_max4(mA);
      mB = warp_max4(mB);
      if ((lane & 3) == 0) {
        swm[warp * HEADS + hA] = mA;
        swm[warp * HEADS + hB] = mB;
      }
    }
      }
}
  if constexpr (STAGE == 4) return;
  if (STAGE == 8 && tid == 0) ck2 = clock64();
  __syncthreads();

  if (warp < NWQ) {
    float MA = -INFINITY, MB = -INFINITY;
#pragma unroll
    for (int w = 0; w < NWQ; ++w) {
      MA = fmaxf(MA, swm[w * HEADS + hA]);
      MB = fmaxf(MB, swm[w * HEADS + hB]);
    }
    float lA = 0.f, lB = 0.f;
    unsigned pt[4];  // PVB 5: the A fragment under construction
#pragma unroll
    for (int n = 0; n < NTQ; ++n) {
      float p0 = exp2f(sv[n][0] - MA), p1 = exp2f(sv[n][1] - MA);
      float p2 = exp2f(sv[n][2] - MB), p3 = exp2f(sv[n][3] - MB);
      lA += p0 + p1;
      lB += p2 + p3;
      if constexpr (PTM) {
        // The accumulator lane map IS the A-fragment lane map: lane l holds
        // heads l/4 and l/4+8 at kv columns (l%4)*2 and +1, and n-tiles 2e and
        // 2e+1 of a warp supply a[0..1] and a[2..3] of k-step w*(NTQ/2)+e.  So
        // P only has to cross warps, not be re-laid-out: buffer the even tile,
        // and when the odd one lands store the whole fragment in one 16 B write
        // that every warp reads straight back as its own operand.
        pt[(n & 1) * 2] = bf2u(__floats2bfloat162_rn(p0, p1));
        pt[(n & 1) * 2 + 1] = bf2u(__floats2bfloat162_rn(p2, p3));
        if (n & 1)
          Pt[(warp * (NTQ / 2) + (n >> 1)) * 32 + lane] = make_uint4(pt[0], pt[1], pt[2], pt[3]);
      } else {
        int c0 = TILE_ROW(warp, n) + col;
        *reinterpret_cast<__nv_bfloat162 *>(&Ps[hA * LDP + c0]) = __floats2bfloat162_rn(p0, p1);
        *reinterpret_cast<__nv_bfloat162 *>(&Ps[hB * LDP + c0]) = __floats2bfloat162_rn(p2, p3);
      }
    }
    lA = warp_sum4(lA);
    lB = warp_sum4(lB);
    if ((lane & 3) == 0) {
      swl[warp * HEADS + hA] = lA;
      swl[warp * HEADS + hB] = lB;
    }
  }
  // PVB 10..13 pull the PV pipeline's first V fragments across the barrier
  // below.  They come out of Ks, which has been final since the gather landed,
  // so the only thing that barrier actually gates is P -- issuing them here
  // buries their ldmatrix latency under it instead of paying it in front of
  // the first mma.  Issuing them one step earlier, ahead of the softmax rather
  // than behind it, measures 24 ns slower (PVB 14, removed).
  constexpr int PRE = (PVB == 10 || PVB == 12) ? 2 : (PVB == 11) ? 4 : (PVB == 13) ? 3 : 0;
  unsigned kpre[PRE ? PRE : 1][4];
  if constexpr (PRE) {
    const unsigned vb = sm_addr(&Ks[(lane & 15) * LDK + (lane >> 4) * 8]);
#pragma unroll
    for (int i = 0; i < PRE; ++i) {
      const int kn = i / NGP, gn = i - kn * NGP;
      ldm_x4_trans(kpre[i], vb + (kn * 16) * (LDK * 2) + ((warp * NGP + gn) * 16) * 2);
    }
  }
  __syncthreads();

  // FUSE 15 takes the barrier's target here instead of at block entry.  The
  // value is not needed until after the PV, so the round trip hides under it
  // either way -- but at block entry the load competes with the index fetch for
  // the same L2 ports, which is what made FUSE 14 lose 215 ns.  Nothing else
  // reads L2 between here and the fold, so here it is free.
  int lgen = 0;
  if (FUSE == 15 && tid == 0) lgen = (ld_relaxed_gpu(cnt + t * CNT_STRIDE) / NSPL + 1) * NSPL;

  float LA = 0.f, LB = 0.f;
#pragma unroll
  for (int w = 0; w < NWQ; ++w) {
    LA += swl[w * HEADS + hA];
    LB += swl[w * HEADS + hB];
  }
  if (tid < HEADS) {
    float M = -INFINITY, L = 0.f;
#pragma unroll
    for (int w = 0; w < NWQ; ++w) {
      M = fmaxf(M, swm[w * HEADS + tid]);
      L += swl[w * HEADS + tid];
    }
    mlm[(size_t)tid * nsplit] = M;
    mll[(size_t)tid * nsplit] = L;
  }
  if (STAGE == 2) return;

  // ---- O_partial = (P @ Kc) / l, dims split across warps ----
  {
    const float iA = 1.f / LA, iB = 1.f / LB;
    unsigned pa[KTP][4];
    if constexpr (PTM) {
      // One 128-bit lane-indexed read per k-step: no permute, no ldmatrix, and
      // the 32 lanes cover 512 contiguous bytes so every phase hits all banks.
#pragma unroll
      for (int k = 0; k < KTP; ++k) {
        const uint4 v = Pt[k * 32 + lane];
        pa[k][0] = v.x;
        pa[k][1] = v.y;
        pa[k][2] = v.z;
        pa[k][3] = v.w;
      }
    } else {
#pragma unroll
      for (int k = 0; k < KTP; ++k)
        if (!BAL || k * 16 < nrp)
          ldm_x4(pa[k], sm_addr(&Ps[(lane & 15) * LDP + k * 16 + (lane >> 4) * 8]));
    }
    // Split the PV span: everything before this point is pulling P back out of
    // smem (it was produced by the NWQ QK warps and is consumed by all NWARP),
    // everything after is the mma loop itself.
    if (STAGE == 8 && tid == 0) ck2b = clock64();
    const unsigned vbase = sm_addr(&Ks[(lane & 15) * LDK + (lane >> 4) * 8]);
    // Head stride and block base swap roles under OLAY: the block's DCKV slice
    // moves into the chunk axis, so the 16 heads are what stride apart here.
    __nv_bfloat16 *op =
        OLAY ? o_part + (size_t)t * (HEADS * NSPL * DCKV) + (size_t)c * DCKV
             : o_part + (size_t)(BAL ? c
                                     : ((STAGE == 5 || STAGE == 6) ? t : (t * nsplit + c))) *
                            (HEADS * DCKV);
    constexpr size_t HSTR = OLAY ? (size_t)NSPL * DCKV : DCKV;
    float oac[NGP][2][4];
#pragma unroll
    for (int g = 0; g < NGP; ++g)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int u = 0; u < 4; ++u) oac[g][j][u] = 0.f;
    // PVB 3: one flattened (k, g) step list, double-buffered.  Each mma reads a
    // fragment issued one step earlier, so the ldmatrix latency is covered by
    // the previous step's mma pair rather than stalling in front of it -- and
    // unlike PVB 1 it only ever holds two fragments, so it does not spill.
    if constexpr (PVB == 3 || PVB == 5 || (PVB >= 6 && PVB <= 8) || PVB >= 10) {
      constexpr int NST = KTP * NGP;
      // Steps the fragment fetch runs ahead of the mma that consumes it.  At
      // depth 1 the V read is only 66% of what the smem port could deliver:
      // with two warps per SMSP one mma pair is not enough to cover an
      // ldmatrix, so the deeper depths are worth pricing.
      // PRE fragments come from across the barrier; a deeper DEP tops the
      // pipeline up with fragments issued here.
      constexpr int DEP = (PVB == 12 || PVB == 13) ? 3
                          : (PVB >= 10)             ? PRE
                          : (PVB >= 6)              ? (PVB - 4)
                                                    : 1;
      constexpr int NBUF = DEP + 1;
      unsigned kb[NBUF][4];
#pragma unroll
      for (int i = 0; i < DEP; ++i) {
        if (PVB >= 10 && i < PRE) {
#pragma unroll
          for (int u = 0; u < 4; ++u) kb[i][u] = kpre[i < PRE ? i : 0][u];  // pre-barrier
        } else {
          const int kn = i / NGP, gn = i - kn * NGP;
          ldm_x4_trans(kb[i], vbase + (kn * 16) * (LDK * 2) + ((warp * NGP + gn) * 16) * 2);
        }
      }
#pragma unroll
      for (int st = 0; st < NST; ++st) {
        if (st + DEP < NST) {
          const int kn = (st + DEP) / NGP, gn = (st + DEP) - kn * NGP;
          ldm_x4_trans(kb[(st + DEP) % NBUF],
                       vbase + (kn * 16) * (LDK * 2) + ((warp * NGP + gn) * 16) * 2);
        }
        const int kc = st / NGP, gc = st - kc * NGP, b = st % NBUF;
        mma16816(oac[gc][0], pa[kc], kb[b][0], kb[b][1]);
        mma16816(oac[gc][1], pa[kc], kb[b][2], kb[b][3]);
      }
    } else if constexpr (PVB == 9) {
      // PVB 7's depth-3 fragment pipeline walked head-slice-outer instead of
      // k-outer.  Same mma count and same V traffic, but a slice's 16 output
      // dims retire every KTP steps rather than all four slices landing in the
      // final step, so the 16 KB of partials drains towards L2 across the whole
      // PV.  The release the barrier issues next then waits on a store queue
      // that is already nearly empty.  It also drops oac from NGP slices to one.
      constexpr int NST = KTP * NGP;
      constexpr int DEP = 3, NBUF = DEP + 1;
      unsigned kb[NBUF][4];
#pragma unroll
      for (int i = 0; i < DEP; ++i) {
        const int gn = i / KTP, kn = i - gn * KTP;
        ldm_x4_trans(kb[i], vbase + (kn * 16) * (LDK * 2) + ((warp * NGP + gn) * 16) * 2);
      }
      float ac[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int u = 0; u < 4; ++u) ac[j][u] = 0.f;
#pragma unroll
      for (int st = 0; st < NST; ++st) {
        if (st + DEP < NST) {
          const int gn = (st + DEP) / KTP, kn = (st + DEP) - gn * KTP;
          ldm_x4_trans(kb[(st + DEP) % NBUF],
                       vbase + (kn * 16) * (LDK * 2) + ((warp * NGP + gn) * 16) * 2);
        }
        const int gc = st / KTP, kc = st - gc * KTP, b = st % NBUF;
        mma16816(ac[0], pa[kc], kb[b][0], kb[b][1]);
        mma16816(ac[1], pa[kc], kb[b][2], kb[b][3]);
        if (kc == KTP - 1) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int d = (warp * NGP + gc) * 16 + j * 8 + col;
            *reinterpret_cast<__nv_bfloat162 *>(op + hA * HSTR + d) =
                __floats2bfloat162_rn(ac[j][0] * iA, ac[j][1] * iA);
            *reinterpret_cast<__nv_bfloat162 *>(op + hB * HSTR + d) =
                __floats2bfloat162_rn(ac[j][2] * iB, ac[j][3] * iB);
#pragma unroll
            for (int u = 0; u < 4; ++u) ac[j][u] = 0.f;
          }
        }
      }
    } else if constexpr (PVB == 4) {
      // Head slice outer, k-step inner, and each slice stores the moment its
      // KTP accumulations retire.  The barrier's fence then waits on a store
      // queue that has been draining through the whole PV rather than one
      // handed the block's entire 16 KB at the very end.
#pragma unroll
      for (int g = 0; g < NGP; ++g) {
        float ac[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int u = 0; u < 4; ++u) ac[j][u] = 0.f;
#pragma unroll
        for (int k = 0; k < KTP; ++k) {
          unsigned kb[4];
          ldm_x4_trans(kb, vbase + (k * 16) * (LDK * 2) + ((warp * NGP + g) * 16) * 2);
          mma16816(ac[0], pa[k], kb[0], kb[1]);
          mma16816(ac[1], pa[k], kb[2], kb[3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int d = (warp * NGP + g) * 16 + j * 8 + col;
          *reinterpret_cast<__nv_bfloat162 *>(op + hA * HSTR + d) =
              __floats2bfloat162_rn(ac[j][0] * iA, ac[j][1] * iA);
          *reinterpret_cast<__nv_bfloat162 *>(op + hB * HSTR + d) =
              __floats2bfloat162_rn(ac[j][2] * iB, ac[j][3] * iB);
        }
      }
    } else {
#pragma unroll
    for (int k = 0; k < KTP; ++k) {
      if constexpr (PVB == 1) {
        // Issue the whole k-step's ldmatrix set before any mma consumes it.
        // Interleaving them one by one leaves the ldm -> mma dependency fully
        // exposed, and with only two warps per SMSP there is nothing else to
        // cover it.
        unsigned kb[NGP][4];
#pragma unroll
        for (int g = 0; g < NGP; ++g)
          ldm_x4_trans(kb[g], vbase + (k * 16) * (LDK * 2) + ((warp * NGP + g) * 16) * 2);
#pragma unroll
        for (int g = 0; g < NGP; ++g) {
          mma16816(oac[g][0], pa[k], kb[g][0], kb[g][1]);
          mma16816(oac[g][1], pa[k], kb[g][2], kb[g][3]);
        }
      } else if constexpr (PVB == 2) {
        // Halfway house: two fragments in flight costs 8 registers instead of
        // PVB 1's 16, which is what made the fully batched form regress.
#pragma unroll
        for (int g0 = 0; g0 < NGP; g0 += 2) {
          unsigned kb[2][4];
#pragma unroll
          for (int u = 0; u < 2; ++u)
            ldm_x4_trans(kb[u], vbase + (k * 16) * (LDK * 2) + ((warp * NGP + g0 + u) * 16) * 2);
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            mma16816(oac[g0 + u][0], pa[k], kb[u][0], kb[u][1]);
            mma16816(oac[g0 + u][1], pa[k], kb[u][2], kb[u][3]);
          }
        }
      } else if (!BAL || k * 16 < nrp) {
#pragma unroll
        for (int g = 0; g < NGP; ++g) {
          unsigned kb[4];
          ldm_x4_trans(kb, vbase + (k * 16) * (LDK * 2) + ((warp * NGP + g) * 16) * 2);
          mma16816(oac[g][0], pa[k], kb[0], kb[1]);
          mma16816(oac[g][1], pa[k], kb[2], kb[3]);
        }
      }
    }
    }
    if constexpr (PVB != 4 && PVB != 9)
#pragma unroll
    for (int g = 0; g < NGP; ++g)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        int d = (warp * NGP + g) * 16 + j * 8 + col;
        *reinterpret_cast<__nv_bfloat162 *>(op + hA * HSTR + d) =
            __floats2bfloat162_rn(oac[g][j][0] * iA, oac[g][j][1] * iA);
        *reinterpret_cast<__nv_bfloat162 *>(op + hB * HSTR + d) =
            __floats2bfloat162_rn(oac[g][j][2] * iB, oac[g][j][3] * iB);
      }
  }
  if (STAGE == 8 && tid == 0) ck3 = clock64();
  if (STAGE == 11) return;  // body without the fold
  if constexpr (BAL)
    bal_fold<DPT, NTHREAD>(o_part, ml_part, out, lse, cnt, t, sbase, sub, jsub, tid);
  else if constexpr (FUSE)
    fold_token<NSPL, DPT, NTHREAD, MODEF, STAGE == 10, OLAY>(o_part, ml_part, out, lse, cnt, t, c,
                                                             tid, FUSE == 15 ? lgen : sgen);
  if (STAGE == 10 && tid == 0) cnt[T_CNT + (t * nsplit + c) * 4 + 1] = (int)gtimer();
  if (STAGE == 8 && tid == 0) {
    const long long ck4 = clock64();
    mlm[(size_t)5 * nsplit] = (float)(ck4 - ck3);   // barrier + fold
    mlm[0] = (float)(ckg - ck0);            // gather
    mlm[(size_t)4 * nsplit] = (float)(ck1 - ckg);    // QK
    mlm[(size_t)nsplit] = (float)(ck2 - ck1);        // softmax
    mlm[(size_t)2 * nsplit] = (float)(ck3 - ck2b);   // PV mma loop
    mlm[(size_t)6 * nsplit] = (float)(ck2b - ck2);   // P read-back
    mlm[(size_t)3 * nsplit] = (float)(ck3 - ck0);    // body total
  }
}

template <int NSPLIT, int CSTAGE, int DPT, int CSG, int DSPL>
__global__ __launch_bounds__(CSG *(DCKV / DPT / DSPL)) void dsa_combine_kernel(
    const __nv_bfloat16 *__restrict__ o_part, const float *__restrict__ ml_part,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ lse) {
  // One (token, head) row is spread over DSPL blocks along d.  Combine is
  // latency-bound at 128 blocks x 128 threads -- only ~4 warps per SM, far too
  // little MLP to cover the L2 trip -- and the fix is more blocks, not more
  // threads per block (widening the block needs a cross-thread reduction that
  // costs more than it buys).
  constexpr int NT = DCKV / DPT / DSPL;
  if constexpr (CSTAGE == 0 || CSTAGE == 4) {
    if (threadIdx.x == 0) lse[blockIdx.x * HEADS] = 0.f;
    return;
  } else
    combine_item<NSPLIT, DPT, CSG, CSTAGE == 5>(o_part, ml_part, out, lse, blockIdx.x, blockIdx.y,
                                   (int)blockIdx.z * NT + (int)threadIdx.x % NT,
                                   (int)threadIdx.x / NT);
}

// Thin launch wrapper: split_body is a device function so it can be reused.
template <int R, int NWARP, int NWQ, int NGK, int GMODE, int STAGE, int KA, int KS, int PVB,
          int FUSE, int DPT, int BAL, int GORD>
__global__ __launch_bounds__(NWARP * 32) void dsa_split_kernel(
    const __nv_bfloat16 *__restrict__ q_nope, const __nv_bfloat16 *__restrict__ q_pe,
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    const int *__restrict__ indices, __nv_bfloat16 *__restrict__ o_part,
    float *__restrict__ ml_part, __nv_bfloat16 *__restrict__ out, float *__restrict__ lse,
    int *__restrict__ cnt, int nsplit, int T, float scale_l2e) {
  split_body<R, NWARP, NWQ, NGK, GMODE, STAGE, KA, KS, PVB, FUSE, DPT, BAL, GORD>(
      q_nope, q_pe, ckv, kpe, indices, o_part, ml_part, out, lse, cnt, nsplit, T, scale_l2e);
}

// ---------------------------------------------------------------------------
// host launcher
// ---------------------------------------------------------------------------
template <int R, int NWARP>
static size_t split_smem_bytes() {
  size_t b = (size_t)HEADS * LDK * 2 + (size_t)R * LDK * 2 + (size_t)HEADS * (R + 8) * 2 +
             (size_t)R * 4 + (size_t)NWARP * HEADS * 4 * 2 + 4 * MAXT * 4 +
             32;  // plan scratch + up to 4 mbarriers
  return (b + 15) & ~size_t(15);
}

struct LaunchArgs {
  const __nv_bfloat16 *q_nope;
  const __nv_bfloat16 *q_pe;
  const __nv_bfloat16 *ckv;
  const __nv_bfloat16 *kpe;
  const int *indices;
  __nv_bfloat16 *o_part;
  float *ml_part;
  __nv_bfloat16 *out;
  float *lse;
  int *cnt;
  int T;
  float scale_l2e;
  cudaStream_t stream;
};

template <int R, int NWARP, int NWQ, int NGK, int GMODE, int STAGE, int CSTAGE = 3,
          int DPT = 8, int CSG = 1, int KA = 0, int KS = 1, int PVB = 0, int FUSE = 0,
          int DSPL = 1, int BAL = 0, int GORD = 0>
static cudaError_t run_variant(const LaunchArgs &a) {
  // A balanced launch is one flat wave of NBAL blocks; the plan inside decides
  // which token and which rows each of them takes.
  constexpr int NSPLIT = BAL ? NBAL : TOPKC / R;
  constexpr int GX = BAL ? NBAL : 0;
  static bool configured = false;
  static_assert(KS == 1 || NWQ * (KS - 1) * (R / (NWQ * 8)) * 4 * 32 * 4 <= HEADS * (DCKV + 72) * 2,
                "k-split scratch must fit inside Qs");
  size_t smem = (STAGE == 9) ? 1024 : split_smem_bytes<R, NWARP>();
  auto kern = dsa_split_kernel<R, NWARP, NWQ, NGK, GMODE, STAGE, KA, KS, PVB, FUSE, DPT, BAL, GORD>;
  if (!configured) {
    cudaError_t e =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    if constexpr (FUSE == 3 || FUSE == 4) {
      // A 16-block cluster is past the portable limit, so it must be opted in.
      e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      if (e != cudaSuccess) return e;
    }
    configured = true;
  }
  if constexpr (FUSE && FUSE != 3 && FUSE != 4) {
    int nsm = 0;
    cudaError_t de = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    if (de != cudaSuccess) return de;
    // The spin only terminates if the whole grid is resident at once.  How many
    // blocks an SM holds depends on this variant's smem and register footprint,
    // so ask rather than assume one.
    int occ = 0;
    de = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, NWARP * 32, smem);
    if (de != cudaSuccess) return de;
    if ((BAL ? GX : a.T * NSPLIT) > nsm * occ) return cudaErrorInvalidConfiguration;
  }
  if constexpr (FUSE == 3 || FUSE == 4) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(a.T, NSPLIT);
    cfg.blockDim = dim3(NWARP * 32);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = a.stream;
    cudaLaunchAttribute la[1] = {};
    la[0].id = cudaLaunchAttributeClusterDimension;
    la[0].val.clusterDim.x = 1;
    la[0].val.clusterDim.y = NSPLIT;
    la[0].val.clusterDim.z = 1;
    cfg.attrs = la;
    cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, kern, a.q_nope, a.q_pe, a.ckv, a.kpe, a.indices,
                                       a.o_part, a.ml_part, a.out, a.lse, a.cnt, NSPLIT, a.T,
                                       a.scale_l2e);
    if (e != cudaSuccess) return e;
  } else
    kern<<<BAL ? dim3(GX) : (GORD ? dim3(NSPLIT, a.T) : dim3(a.T, NSPLIT)), NWARP * 32, smem, a.stream>>>(
        a.q_nope, a.q_pe, a.ckv, a.kpe, a.indices, a.o_part, a.ml_part, a.out, a.lse, a.cnt, NSPLIT,
        a.T, a.scale_l2e);
  if (STAGE == 3 && !FUSE && !BAL)
    // CSTAGE 4 is the empty kernel at a one-block grid: it separates the fixed
    // per-kernel launch tax from whatever the 128-block grid itself costs.
    dsa_combine_kernel<NSPLIT, CSTAGE, DPT, CSG, DSPL>
        <<<(CSTAGE == 4) ? dim3(1, 1, 1) : dim3(a.T, HEADS, DSPL),
           CSG * (DCKV / DPT / DSPL), 0, a.stream>>>(
        a.o_part, a.ml_part, a.out, a.lse);
  return cudaGetLastError();
}

extern "C" int dsa_launch(const void *q_nope, const void *q_pe, const void *ckv, const void *kpe,
                          const void *indices, void *out, void *lse, void *o_part, void *ml_part,
                          void *cnt, int T, float sm_scale, int variant, void *stream) {
  LaunchArgs a;
  a.q_nope = (const __nv_bfloat16 *)q_nope;
  a.q_pe = (const __nv_bfloat16 *)q_pe;
  a.ckv = (const __nv_bfloat16 *)ckv;
  a.kpe = (const __nv_bfloat16 *)kpe;
  a.indices = (const int *)indices;
  a.o_part = (__nv_bfloat16 *)o_part;
  a.ml_part = (float *)ml_part;
  a.out = (__nv_bfloat16 *)out;
  a.lse = (float *)lse;
  a.cnt = (int *)cnt;
  a.T = T;
  a.scale_l2e = sm_scale * 1.4426950408889634f;
  a.stream = (cudaStream_t)stream;

  cudaError_t e;
  switch (variant) {
    // <R, NWARP, NWQ, NGK, GMODE, STAGE, CSTAGE, DPT, CSG, KA, KS, PVB, FUSE>
    case 0: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1>(a); break;
    case 1: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 2>(a); break;
    case 2: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 4>(a); break;
    case 3: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 8>(a); break;
    case 4: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 1>(a); break;   // fused
    case 5: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 1>(a); break;   // fused DPT 8
    case 6: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;   // fused DPT 2
    // probes: STAGE 8 reports per-phase cycles, 1 = gather only, 0 = launch floor
    case 8: e = run_variant<128, 8, 4, 1, 1, 8, 0, 4>(a); break;
    case 9: e = run_variant<128, 8, 4, 1, 1, 1, 0, 4>(a); break;
    case 10: e = run_variant<128, 8, 4, 1, 1, 0, 0, 4>(a); break;
    case 11: e = run_variant<128, 8, 4, 1, 1, 9, 0, 4>(a); break;   // floor at 1 KB smem
    case 12: e = run_variant<128, 8, 4, 1, 1, 3, 0, 4>(a); break;   // combine floor, 128 blk
    case 13: e = run_variant<128, 8, 4, 1, 1, 3, 4, 4>(a); break;   // combine floor, 1 blk
    // combine spread over DSPL blocks along d
    case 14: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 0, 2>(a); break;
    case 15: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 0, 4>(a); break;
    case 16: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 0, 2>(a); break;
    case 17: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 0, 4>(a); break;
    case 18: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 0, 2>(a); break;
    case 19: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 0, 4>(a); break;
    case 20: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 0, 8>(a); break;
    // CSTAGE 5 skips the o_part loads belonging to empty chunks
    case 21: e = run_variant<128, 8, 4, 1, 1, 3, 5, 4>(a); break;
    case 22: e = run_variant<128, 8, 4, 1, 1, 3, 5, 2, 1, 0, 1, 0, 0, 4>(a); break;
    case 23: e = run_variant<128, 8, 4, 1, 1, 3, 5, 4, 1, 0, 1, 0, 0, 2>(a); break;
    case 24: e = run_variant<128, 8, 4, 1, 1, 3, 5, 8>(a); break;
    // QK currently idles 4 of the 8 warps (live = warp < NWQ*KS); these put
    // them to work, paired with the best combine shape (DPT 2, DSPL 4).
    case 25: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 2, 0, 0, 4>(a); break;  // KS 2
    case 26: e = run_variant<128, 8, 8, 1, 1, 3, 3, 2, 1, 0, 1, 0, 0, 4>(a); break;  // NWQ 8
    case 27: e = run_variant<128, 8, 4, 1, 1, 8, 0, 2, 1, 0, 2, 0, 0, 4>(a); break;  // KS 2, timed
    case 28: e = run_variant<128, 8, 8, 1, 1, 8, 0, 2, 1, 0, 1, 0, 0, 4>(a); break;  // NWQ 8, timed
    // single-kernel: spin barrier + in-place fold, block (t, c) owns head c
    case 29: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 1>(a); break;
    case 30: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 1>(a); break;
    case 31: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 32: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 2>(a); break;  // barrier only
    case 33: e = run_variant<128, 8, 4, 1, 1, 8, 0, 2, 1, 0, 1, 0, 1>(a); break;  // fused, timed
    // FUSE 3/4: hardware cluster barrier instead of the L2 spin
    case 34: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 3>(a); break;
    case 35: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 4>(a); break;  // barrier only
    case 36: e = run_variant<128, 8, 4, 1, 1, 8, 0, 2, 1, 0, 1, 0, 3>(a); break;  // timed
    // GMODE 2 probes: does the first row batch's barrier release early?
    case 37: e = run_variant<128, 8, 4, 2, 2, 8, 0, 2>(a); break;
    case 38: e = run_variant<128, 8, 4, 4, 2, 8, 0, 2>(a); break;
    // GMODE 3: slab g+1 is issued only after slab g lands
    case 39: e = run_variant<128, 8, 4, 2, 3, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 40: e = run_variant<128, 8, 4, 4, 3, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 41: e = run_variant<128, 8, 4, 2, 3, 8, 0, 2>(a); break;
    case 42: e = run_variant<128, 8, 4, 4, 3, 8, 0, 2>(a); break;
    // GMODE 4: row batches, batch b+1 issued only after batch b lands
    case 43: e = run_variant<128, 8, 4, 2, 4, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 44: e = run_variant<128, 8, 4, 2, 4, 8, 0, 2>(a); break;
    case 45: e = run_variant<128, 8, 4, 4, 4, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 46: e = run_variant<128, 8, 4, 4, 4, 8, 0, 2>(a); break;
    // R 64: half the rows per block, so the fixed DRAM latency of the gather is
    // amortised over twice as many blocks (two fit per SM at this smem size).
    case 47: e = run_variant<64, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 1>(a); break;
    case 48: e = run_variant<64, 8, 4, 1, 1, 8, 0, 2>(a); break;
    case 49: e = run_variant<64, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 1>(a); break;
    case 50: e = run_variant<64, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 0, 4>(a); break;
    case 51: e = run_variant<128, 8, 4, 1, 1, 10, 3, 2, 1, 0, 1, 0, 1>(a); break;  // gtimer probe
    // FUSE 5: fold redirects empty chunks' loads onto chunk 0
    case 52: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 5>(a); break;
    case 53: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 5>(a); break;
    case 54: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 5>(a); break;
    // FUSE 6: release/acquire on the counter instead of a block-wide threadfence
    case 55: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 6>(a); break;
    case 56: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 6>(a); break;
    // FUSE 7: single-lane threadfence.  FUSE 8: release arrive + one acquire.
    case 57: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 7>(a); break;
    case 58: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 8>(a); break;
    // FUSE 9/10: per-block arrival flags, packed / one per cacheline
    case 59: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 9>(a); break;
    case 60: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 10>(a); break;
    case 61: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 10>(a); break;
    case 62: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 0, 10>(a); break;
    case 63: e = run_variant<128, 8, 4, 1, 1, 10, 3, 4, 1, 0, 1, 0, 10>(a); break;  // gtimer
    // FUSE 11: FUSE 10's barrier over a head-major o_part, so each folding
    // block reads one contiguous 16 KB span instead of 16 scattered 1 KB ones.
    case 64: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 11>(a); break;
    case 65: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 11>(a); break;
    case 66: e = run_variant<128, 8, 4, 1, 1, 10, 3, 4, 1, 0, 1, 0, 11>(a); break;  // gtimer
    // GMODE 5: Kpe rides cp.async instead of a bulk copy, halving the TMA
    // copy count while the bytes it moves overlap on the LSU path.
    case 67: e = run_variant<128, 8, 4, 1, 5, 3, 3, 4, 1, 0, 1, 0, 10>(a); break;
    case 68: e = run_variant<128, 8, 4, 1, 5, 8, 0, 4>(a); break;  // phase probe
    case 69: e = run_variant<128, 8, 4, 1, 5, 3, 3, 2, 1, 0, 1, 0, 7>(a); break;
    // Body cost at finer R: what a load-balanced split would buy per block.
    case 70: e = run_variant<32, 8, 4, 1, 1, 8, 0, 4>(a); break;
    case 71: e = run_variant<16, 8, 2, 1, 1, 8, 0, 4>(a); break;
    // BAL: one flat wave of NBAL blocks, rows handed out in proportion to each
    // token's live count instead of a fixed R per block.
    case 72: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 10, 1, 1>(a); break;
    // BAL bisection probes: plan only, then plan + gather.
    case 75: e = run_variant<128, 8, 4, 1, 1, 0, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 76: e = run_variant<128, 8, 4, 1, 1, 1, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 77: e = run_variant<128, 8, 4, 1, 1, 4, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 78: e = run_variant<128, 8, 4, 1, 1, 2, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 79: e = run_variant<128, 8, 4, 1, 1, 3, 4, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 80: e = run_variant<128, 8, 4, 1, 1, 8, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 81: e = run_variant<128, 8, 4, 1, 1, 11, 0, 4, 1, 0, 1, 0, 0, 1, 1>(a); break;
    case 82: e = run_variant<128, 8, 4, 1, 1, 11, 0, 4, 1, 0, 1, 0, 0, 1, 0>(a); break;
    // R 64 halves the smem per block, which lets two blocks share an SM: the
    // grid doubles to 256 but still lands in one wave, each block covers half
    // the rows, and one block's gather latency is covered by the other's work.
    case 83: e = run_variant<64, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 10>(a); break;
    case 84: e = run_variant<64, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 10>(a); break;
    case 85: e = run_variant<64, 8, 4, 1, 1, 11, 0, 4, 1, 0, 1, 0, 0>(a); break;
    // PVB 4: store each head slice as it retires, so the fence at the barrier
    // is not the first thing to see the block's 16 KB of partials.
    case 86: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 7>(a); break;
    case 87: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 4, 10>(a); break;
    case 88: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 10>(a); break;
    case 89: e = run_variant<128, 8, 4, 1, 1, 10, 3, 2, 1, 0, 1, 4, 10>(a); break;  // gtimer
    // FUSE 12: prefetch the already-flagged partials across the arrival.
    case 90: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 12>(a); break;
    case 91: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 12>(a); break;
    case 92: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 12>(a); break;
    case 93: e = run_variant<128, 8, 4, 1, 1, 10, 3, 2, 1, 0, 1, 0, 12>(a); break;  // gtimer
    case 94: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;   // GORD
    case 95: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 13>(a); break;           // LZ
    case 96: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 4, 10, 1, 0, 1>(a); break;  // both
    case 97: e = run_variant<128, 8, 4, 1, 1, 8, 0, 2, 1, 0, 1, 4, 0, 1, 0, 1>(a); break;   // phase
    case 98: e = run_variant<128, 8, 8, 1, 1, 3, 3, 2, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;   // NWQ 8
    case 99: e = run_variant<128, 8, 2, 1, 1, 3, 3, 2, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;   // NWQ 2
    case 100: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;  // DPT 4
    case 101: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 7, 1, 0, 1>(a); break;  // PVB 0
    case 102: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;  // PVB 3
    case 103: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 104: e = run_variant<128, 8, 4, 1, 0, 3, 3, 4, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;  // cp.async
    case 105: e = run_variant<128, 8, 4, 1, 5, 3, 3, 4, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;  // Kpe LSU
    case 106: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;  // DPT 8
    case 107: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 4, 10, 1, 0, 1>(a); break;
    case 108: e = run_variant<128, 8, 4, 1, 1, 8, 0, 4, 1, 0, 1, 4, 0, 1, 0, 1>(a); break;  // phase
    case 109: e = run_variant<128, 8, 4, 1, 6, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;  // 1/2 LSU
    case 110: e = run_variant<128, 8, 4, 1, 7, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;  // 1/4 LSU
    case 111: e = run_variant<128, 8, 4, 1, 6, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;  // phase
    case 112: e = run_variant<128, 8, 4, 1, 7, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;  // phase
    // Row-batched gather: barrier per batch, QK starts on batch 0 while the
    // rest of the rows are still in flight.
    case 113: e = run_variant<128, 8, 4, 2, 2, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 114: e = run_variant<128, 8, 4, 4, 2, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 115: e = run_variant<128, 8, 4, 2, 2, 3, 3, 4, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;
    case 116: e = run_variant<128, 8, 4, 2, 2, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;  // phase
    // Probes of the current best shape (case 103): clock64 phase split, then
    // the globaltimer block timeline.
    case 117: e = run_variant<128, 8, 4, 1, 1, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;
    case 118: e = run_variant<128, 8, 4, 1, 1, 10, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    // Single-variable sweep of the combine shape around case 103.
    case 119: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 120: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 121: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 2, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 122: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 0, 7, 1, 0, 1>(a); break;
    // Prefetched-generation barrier on the shared counter.
    case 123: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 14, 1, 0, 1>(a); break;
    case 124: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 3, 14, 1, 0, 1>(a); break;
    case 125: e = run_variant<128, 8, 4, 1, 1, 10, 3, 4, 1, 0, 1, 3, 14, 1, 0, 1>(a); break;
    // Feature-sliced gather: QK's k loop consumes slab 0 while slab 1 flies,
    // and unlike a row split every warp keeps all NTQ n-tiles.
    case 126: e = run_variant<128, 8, 4, 2, 1, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 127: e = run_variant<128, 8, 4, 4, 1, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 128: e = run_variant<128, 8, 4, 2, 1, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;  // phase
    // Kpe off the TMA: it is half the copies for an eighth of the bytes, and
    // the gather cost tracks the copy count, not the byte count.
    case 129: e = run_variant<128, 8, 4, 1, 5, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 130: e = run_variant<128, 8, 4, 1, 5, 3, 3, 4, 1, 0, 1, 4, 7, 1, 0, 1>(a); break;
    case 131: e = run_variant<128, 8, 4, 1, 5, 8, 0, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;  // phase
    // Two-kernel split (no grid barrier), carrying every other tuning of 103.
    case 132: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;
    case 133: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 2, 0, 1, 3, 0, 1, 0, 1>(a); break;
    case 134: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 3, 0, 1, 0, 1>(a); break;
    // Widen the fold: CSG thread groups x DPT dims each, so all 256 threads
    // take part and each one issues fewer serial partial loads.
    case 135: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 2, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 136: e = run_variant<128, 8, 4, 1, 1, 3, 3, 8, 4, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 137: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 2, 1, 3, 7, 1, 0, 1>(a); break;
    case 73: e = run_variant<128, 8, 4, 1, 1, 3, 3, 2, 1, 0, 1, 0, 10, 1, 1>(a); break;
    // 138 stops at the barrier exit (wrong output): it prices the fold alone.
    case 138: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 2, 1, 0, 1>(a); break;
    case 139: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 5, 7, 1, 0, 1>(a); break;
    // 140/141 price the cluster barrier against 138/103's L2 spin.
    case 140: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 4, 1, 0, 0>(a); break;
    case 141: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 3, 3, 1, 0, 0>(a); break;
    // 142/143/144 deepen the PV fragment pipeline to 2 / 3 / 4 steps.
    case 142: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 6, 7, 1, 0, 1>(a); break;
    case 143: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 7, 7, 1, 0, 1>(a); break;
    case 144: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 8, 7, 1, 0, 1>(a); break;
    // 146/147 take the index->TMA dependency off the critical path (GMODE 8),
    // 148/149 additionally collapse the mbarrier updates to one per warp.
    case 146: e = run_variant<128, 8, 4, 1, 8, 3, 3, 4, 1, 0, 1, 7, 7, 1, 0, 1>(a); break;
    case 147: e = run_variant<128, 8, 4, 1, 8, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    case 148: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 3, 7, 1, 0, 1>(a); break;
    // 150/151 hoist the PV pipeline's first 2 / 4 V fragments above the barrier.
    case 150: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 10, 7, 1, 0, 1>(a); break;
    case 151: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 11, 7, 1, 0, 1>(a); break;
    // 152/153 keep a depth-3 pipeline but hoist only 2 / 3 of its fragments.
    case 152: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 12, 7, 1, 0, 1>(a); break;
    case 153: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 13, 7, 1, 0, 1>(a); break;
    case 155: e = run_variant<128, 8, 4, 1, 8, 3, 3, 4, 1, 0, 1, 10, 7, 1, 0, 1>(a); break;
    // 156/157/158 re-price the arrival now that the gather is off the front:
    // fire-and-forget red + precomputed target, releasing atom, acquiring poll.
    case 156: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 10, 14, 1, 0, 1>(a); break;
    case 157: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 10, 8, 1, 0, 1>(a); break;
    case 158: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 10, 6, 1, 0, 1>(a); break;
    // 160 keeps FUSE 14's one-round-trip barrier but takes its target late,
    // where the load no longer contends with the gather's index fetch.
    case 160: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 10, 15, 1, 0, 1>(a); break;
    case 161: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 3, 15, 1, 0, 1>(a); break;
    case 149: e = run_variant<128, 8, 4, 1, 9, 3, 3, 4, 1, 0, 1, 7, 7, 1, 0, 1>(a); break;
    // 145 adds the early per-slice partial store to the depth-3 pipeline.
    case 145: e = run_variant<128, 8, 4, 1, 1, 3, 3, 4, 1, 0, 1, 9, 7, 1, 0, 1>(a); break;
    default: return -1;
  }
  return (int)e;
}

extern "C" int dsa_nsplit(int variant) {
  // Must match the R of the variant's run_variant<> above.
  const int r = ((variant >= 47 && variant <= 50) || (variant >= 83 && variant <= 85)) ? 64
                : (variant == 70)                                                       ? 32
                : (variant == 71)                                        ? 16
                                                                         : 128;
  return TOPKC / r;
}
