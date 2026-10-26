// DSA sparse-attention forward (MLA style, bf16 in / bf16 out, fp32 math) for
// definition dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.
//
// One CTA per (token, head); default build: 4 warps (one per SM sub-partition), each
// warp owning the index positions p ≡ warp (mod 4) and processing up to 5 rows at once.
// The op is a chain of dependent memory latencies, so the kernel keeps that chain short:
//   1. the token's index list (coalesced, once per CTA), the warp's first 5 positions
//      (one broadcast load each) and the q row are all loaded at entry;
//   2. as soon as those 5 positions land, their ckv/kpe rows are gathered straight into
//      registers - speculating that a decode token's few valid indices sit at the front
//      of the list (they always do for this workload). Each warp only flags whether its
//      quarter of the list holds a valid index beyond the speculative positions;
//   3. fp32 online softmax on the 5 slots: branch-free, the 5 warp reductions interleaved
//      level by level, unused slots weighted by p = 0;
//   4. partial (m, l, acc) states go through smem and are merged by all 128 threads after
//      the single CTA barrier. The speculation flags are read at that same barrier: if any
//      warp flagged an index beyond the speculative positions, the CTA re-stages the list,
//      compacts it exactly and recomputes in chunks (general path, any index pattern).
// Other DSA_* macros select the earlier design points kept for the benchmark variants.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

#ifndef DSA_WARPS
#define DSA_WARPS 4  // 1 warp per SM sub-partition: the per-warp math runs at full issue rate
#endif
#ifndef DSA_ROWS_PER_CHUNK
#define DSA_ROWS_PER_CHUNK 5  // 4 warps x 5 slots cover the 18-row token in one chunk per warp
#endif
#ifndef DSA_INTERLEAVED
#define DSA_INTERLEAVED 1
#endif
#ifndef DSA_PREFETCH
#define DSA_PREFETCH 0  // 1: L2-prefetch each valid row as soon as its index arrives
#endif
#ifndef DSA_TIMING
#define DSA_TIMING 0  // 1: thread 0 of each CTA records phase timestamps (globaltimer ns + SM cycles)
#endif
#ifndef DSA_IDX_VEC
#define DSA_IDX_VEC 0  // interleaved mode: 1 = warps own 4-index blocks (uint4 loads), 0 = single indices
#endif
#ifndef DSA_SKIP_COMPUTE
#define DSA_SKIP_COMPUTE 0  // 1: diagnostic build, keeps the memory chain but drops the softmax math
#endif
#ifndef DSA_FAST_FINAL
#define DSA_FAST_FINAL 1  // 1: approximate reciprocal / log2 in the merge (fewer instructions)
#endif
#ifndef DSA_SPEC
#define DSA_SPEC 2  // 1: speculative gather from the staged block; 2: from a direct 16B load (off the staging path)
#endif
#ifndef DSA_CTA_VERIFY
#define DSA_CTA_VERIFY 3  // spec=2: 1 = CTA-wide counts (barrier), 2 = warp-local full-list counts, 3 = deferred to the merge barrier
#endif
#ifndef DSA_SKIP_EMPTY
#define DSA_SKIP_EMPTY 0  // 1: warps without rows skip the smem merge traffic (measured: no gain)
#endif
#ifndef DSA_IDX_STAGE
#define DSA_IDX_STAGE 1  // interleaved+vec: 1 = coalesced index loads staged through smem, 0 = direct strided loads
#endif
#ifndef DSA_SOLO_SMEM_KB
#define DSA_SOLO_SMEM_KB 0  // >0: request this much (unused) dynamic smem so only one CTA fits per SM
#endif
#ifndef DSA_PAD_DUP
#define DSA_PAD_DUP 1  // 1: unused chunk slots re-load the warp's first row instead of zero-filling
#endif

constexpr int kTimingCtas = 32;
constexpr int kTimingStamps = 7;
#if DSA_TIMING
__device__ unsigned long long g_timing[kTimingCtas][2 * kTimingStamps];
__device__ __forceinline__ unsigned long long global_timer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DSA_STAMP(i)                                        \
  do {                                                      \
    if (tid == 0 && th < kTimingCtas) {                     \
      g_timing[th][i] = global_timer_ns();                  \
      g_timing[th][kTimingStamps + (i)] = clock64();        \
    }                                                       \
  } while (0)
// Forces the value into a register (i.e. waits for the load that produces it) before a stamp.
#define DSA_MATERIALIZE(v) asm volatile("" ::"r"(v))
#else
#define DSA_STAMP(i) \
  do {               \
  } while (0)
#define DSA_MATERIALIZE(v) \
  do {                     \
  } while (0)
#endif

constexpr int kHeads = 16;
constexpr int kDCkv = 512;
constexpr int kDKpe = 64;
constexpr int kTopK = 2048;
constexpr int kWarps = DSA_WARPS;
constexpr int kThreads = kWarps * 32;
constexpr int kIdxPerThread = kTopK / kThreads;   // indices scanned per thread
constexpr int kIdxVecPerThread = kIdxPerThread / 4;
constexpr int kListCap = kTopK / kWarps;          // per-warp compaction capacity (all valid)
constexpr int kRowsPerChunk = DSA_ROWS_PER_CHUNK;  // rows whose loads are in flight together per warp
constexpr int kCkvVecPerRow = kDCkv * 2 / 16;     // 64 x uint4 per ckv row
constexpr int kKpeWordPerRow = kDKpe * 2 / 4;     // 32 x uint32 per kpe row
constexpr int kDimsPerThread = kDCkv / kThreads;  // output dims per thread in the final merge

static_assert(kThreads * kIdxPerThread == kTopK, "index ownership must tile the top-k list");
static_assert(!(DSA_SPEC == 1 && !DSA_IDX_VEC), "spec=1 needs the uint4 ownership layout");
static_assert(DSA_INTERLEAVED || kIdxPerThread % 4 == 0, "blocked index scan needs whole uint4 per thread");
static_assert(kDimsPerThread * kThreads == kDCkv && kDimsPerThread % 2 == 0, "final merge layout");
static_assert(kWarps % 4 == 0 && (kWarps & (kWarps - 1)) == 0, "merge reads (m,l) as float4 and uses tree sums");

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// Lane-private slice of one KV row: ckv dims [8*lane, 8*lane+8) and
// [256+8*lane, 256+8*lane+8), kpe dims [2*lane, 2*lane+2). The same layout is
// used for q so the score is a per-lane 18-term dot product + one warp sum.
struct RowFrag {
  uint4 a;
  uint4 b;
  uint32_t p;
};

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}

__device__ __forceinline__ void unpack_frag(const RowFrag& r, float* f /* 18 */) {
  unpack8(r.a, f);
  unpack8(r.b, f + 8);
  f[16] = bf_lo(r.p);
  f[17] = bf_hi(r.p);
}

__device__ __forceinline__ float dot18(const float* q, const float* k) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;  // 4 chains keep the FMA dependency short
#pragma unroll
  for (int i = 0; i < 16; i += 4) {
    s0 = fmaf(q[i], k[i], s0);
    s1 = fmaf(q[i + 1], k[i + 1], s1);
    s2 = fmaf(q[i + 2], k[i + 2], s2);
    s3 = fmaf(q[i + 3], k[i + 3], s3);
  }
  s0 = fmaf(q[16], k[16], s0);
  s1 = fmaf(q[17], k[17], s1);
  return (s0 + s1) + (s2 + s3);
}

__device__ __forceinline__ void load_row(RowFrag& f, int row, int lane, const uint4* __restrict__ ckv,
                                         const uint32_t* __restrict__ kpe) {
  const uint4* crow = ckv + (size_t)row * kCkvVecPerRow;
  f.a = __ldg(crow + lane);
  f.b = __ldg(crow + 32 + lane);
  f.p = __ldg(kpe + (size_t)row * kKpeWordPerRow + lane);
}

// Online-softmax update of (m, l, acc) with one chunk of rows; ok[c] is warp-uniform.
// Branch-free: the kRowsPerChunk butterfly reductions are interleaved level by level so
// their shuffle latencies overlap; unused slots get s = -inf, which makes p = 0.
template <bool kFirst>  // kFirst: (m, l, acc) are still the initial (-inf, 0, 0) state -> no rescale
__device__ __forceinline__ void online_update(const RowFrag* f, const bool* ok, const float* q, float& m,
                                              float& l, float* acc) {
#if DSA_SKIP_COMPUTE
  // Diagnostic: consume the gathered bytes with a trivial dependent op, no softmax.
  uint32_t sig = 0;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) sig ^= f[c].a.x ^ f[c].b.y ^ f[c].p;
  acc[0] += bf_lo(sig) + q[0];
  m = 0.f;
  l = 1.f;
  return;
#endif
  float kf[kRowsPerChunk][18];
  float part[kRowsPerChunk];
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    unpack_frag(f[c], kf[c]);
    part[c] = dot18(q, kf[c]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
    for (int c = 0; c < kRowsPerChunk; ++c) part[c] += __shfl_xor_sync(0xffffffffu, part[c], o);
  }
  float s[kRowsPerChunk];
  float mc = -INFINITY;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    s[c] = ok[c] ? part[c] : -INFINITY;
    mc = fmaxf(mc, s[c]);
  }
  const float m_new = kFirst ? mc : fmaxf(m, mc);
  const float alpha = kFirst ? 0.f : fast_exp2(m - m_new);  // m == -inf -> 0
  float p[kRowsPerChunk];
  float psum = 0.f;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    p[c] = fast_exp2(s[c] - m_new);  // unused slot: s = -inf -> p = 0
    psum += p[c];
  }
  l = kFirst ? psum : fmaf(l, alpha, psum);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    float a = kFirst ? 0.f : acc[i] * alpha;
#pragma unroll
    for (int c = 0; c < kRowsPerChunk; ++c) a = fmaf(p[c], kf[c][i], a);
    acc[i] = a;
  }
  m = m_new;
}

// Pairwise (tree) sum of a power-of-two sized array: short dependency chains.
template <int N>
__device__ __forceinline__ float tree_sum(const float* v) {
  if constexpr (N == 1) {
    return v[0];
  } else {
    float h[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) h[i] = v[2 * i] + v[2 * i + 1];
    return tree_sum<N / 2>(h);
  }
}
template <int N>
__device__ __forceinline__ float tree_max(const float* v) {
  if constexpr (N == 1) {
    return v[0];
  } else {
    float h[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) h[i] = fmaxf(v[2 * i], v[2 * i + 1]);
    return tree_max<N / 2>(h);
  }
}

__global__ void __launch_bounds__(kThreads)
dsa_sparse_attn_kernel(const uint4* __restrict__ q_nope, const uint32_t* __restrict__ q_pe,
                       const uint4* __restrict__ ckv, const uint32_t* __restrict__ kpe,
                       const int32_t* __restrict__ indices, float scale_log2e,
                       uint32_t* __restrict__ out, float* __restrict__ lse) {
  __shared__ int s_list[kWarps][kListCap];
  __shared__ int s_cnt[kWarps];
  __shared__ int s_spec[kWarps];
  __shared__ int s_dirty[kWarps];
  __shared__ __align__(16) float s_m[kWarps];
  __shared__ __align__(16) float s_l[kWarps];
  __shared__ __align__(16) float s_acc[kWarps][kDCkv];
#if DSA_INTERLEAVED && DSA_IDX_STAGE && !(DSA_CTA_VERIFY == 2)
  __shared__ __align__(16) uint4 s_idx[kTopK / 4];  // staged index list (swizzled blocks or natural order)
#endif

  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int th = t * kHeads + h;
  const int32_t* idx = indices + (size_t)t * kTopK;
  DSA_STAMP(0);

  // ---- 1. index loads and q loads, all in flight together
  int iv[kIdxPerThread];
#if DSA_INTERLEAVED && DSA_IDX_VEC && DSA_IDX_STAGE
  // 4-index block b = kWarps*(32*j + lane) + warp: warp w owns every block b ≡ w (mod kWarps).
  // The blocks are loaded fully coalesced (thread tid loads block j*kThreads + tid, 4 lines per
  // warp instruction instead of 32) and redistributed through smem; the XOR swizzle keeps both
  // the linear stores and the per-warp strided reads at the 4-wavefront minimum.
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
  uint4 v[kIdxVecPerThread];
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) v[j] = __ldg(idx4 + j * kThreads + tid);
#if DSA_SPEC == 2
  // every lane loads the warp's first block (16B, one sector) directly, issued before the
  // staging round trip so the speculative row gather is not queued behind it
  static_assert(kRowsPerChunk == 4, "uint4 ownership: the speculative chunk is one 4-index block");
  const uint4 blk = __ldg(idx4 + warp);
  const int spec_iv[4] = {(int)blk.x, (int)blk.y, (int)blk.z, (int)blk.w};
#endif
#if DSA_SPEC == 2
  // ---- 1b. speculative gather (issued before the staging round trip): the common decode layout
  // puts a token's few valid indices at the front of the list, i.e. in this warp's first
  // positions. Start those row loads now; the exact compaction below runs while they are in
  // flight and only decides whether this guess covered every valid index.
  static_assert(DSA_IDX_STAGE, "spec=2 relies on the staged index path");
  RowFrag fs[kRowsPerChunk];
  bool spec_ok[kRowsPerChunk];
  int nspec = 0;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    const int row = spec_iv[c];
    spec_ok[c] = row >= 0;
    nspec += spec_ok[c] ? 1 : 0;
    if (spec_ok[c]) {
      load_row(fs[c], row, lane, ckv, kpe);
    } else {
      fs[c].a = make_uint4(0u, 0u, 0u, 0u);
      fs[c].b = make_uint4(0u, 0u, 0u, 0u);
      fs[c].p = 0u;
    }
  }
#endif
#if DSA_SPEC == 2 && DSA_CTA_VERIFY
  int nvalid = 0;  // this thread's valid count over the whole list slice it loaded
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j)
    nvalid += ((int)v[j].x >= 0) + ((int)v[j].y >= 0) + ((int)v[j].z >= 0) + ((int)v[j].w >= 0);
#define DSA_STAGE_UINT4()                                                         \
  do {                                                                            \
    _Pragma("unroll") for (int j = 0; j < kIdxVecPerThread; ++j) {                \
      const int b = j * kThreads + tid;                                           \
      const int slot = b / kWarps, owner = b % kWarps;                            \
      s_idx[slot * kWarps + (owner ^ (slot % kWarps))] = v[j];                    \
    }                                                                             \
    __syncthreads();                                                              \
    _Pragma("unroll") for (int j = 0; j < kIdxVecPerThread; ++j) {                \
      const int slot = 32 * j + lane;                                             \
      const uint4 u = s_idx[slot * kWarps + (warp ^ (slot % kWarps))];            \
      iv[4 * j] = (int)u.x; iv[4 * j + 1] = (int)u.y;                             \
      iv[4 * j + 2] = (int)u.z; iv[4 * j + 3] = (int)u.w;                         \
    }                                                                             \
  } while (0)
#else
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) {
    const int b = j * kThreads + tid;
    const int slot = b / kWarps, owner = b % kWarps;
    s_idx[slot * kWarps + (owner ^ (slot % kWarps))] = v[j];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) {
    const int slot = 32 * j + lane;
    const uint4 u = s_idx[slot * kWarps + (warp ^ (slot % kWarps))];
    iv[4 * j] = (int)u.x; iv[4 * j + 1] = (int)u.y; iv[4 * j + 2] = (int)u.z; iv[4 * j + 3] = (int)u.w;
  }
#endif
#elif DSA_INTERLEAVED && DSA_IDX_VEC
  // 4-index block b = kWarps*(32*j + lane) + warp: warp w owns every block b ≡ w (mod kWarps)
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) {
    const uint4 v = __ldg(idx4 + kWarps * (32 * j + lane) + warp);
    iv[4 * j] = (int)v.x; iv[4 * j + 1] = (int)v.y; iv[4 * j + 2] = (int)v.z; iv[4 * j + 3] = (int)v.w;
  }
#elif DSA_INTERLEAVED && DSA_IDX_STAGE && DSA_CTA_VERIFY == 3
  // position p = kWarps*(32*j + lane) + warp: warp w owns every position p ≡ w (mod kWarps).
  // Deferred verification: the list is loaded once per CTA (coalesced), each warp only flags
  // whether its quarter holds a valid index beyond the speculative positions, and the flags are
  // read at the merge barrier that exists anyway. The fast path therefore has a single barrier;
  // the (rare) slow path re-stages the list from registers and recomputes from scratch.
  static_assert((kWarps * kRowsPerChunk) % 4 == 0, "speculative positions must be whole blocks");
  constexpr int kSpecBlocks = kWarps * kRowsPerChunk / 4;
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
  uint4 v[kIdxVecPerThread];
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) v[j] = __ldg(idx4 + j * kThreads + tid);
  int spec_iv[kRowsPerChunk];  // this warp's first kRowsPerChunk positions, one broadcast load each
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) spec_iv[c] = __ldg(idx + kWarps * c + warp);
#if DSA_SPEC == 2
  // ---- 1b. speculative gather (issued before the staging round trip): the common decode layout
  // puts a token's few valid indices at the front of the list, i.e. in this warp's first
  // positions. Start those row loads now; the exact compaction below runs while they are in
  // flight and only decides whether this guess covered every valid index.
  static_assert(DSA_IDX_STAGE, "spec=2 relies on the staged index path");
  RowFrag fs[kRowsPerChunk];
  bool spec_ok[kRowsPerChunk];
  int nspec = 0;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    const int row = spec_iv[c];
    spec_ok[c] = row >= 0;
    nspec += spec_ok[c] ? 1 : 0;
    if (spec_ok[c]) {
      load_row(fs[c], row, lane, ckv, kpe);
    } else {
      fs[c].a = make_uint4(0u, 0u, 0u, 0u);
      fs[c].b = make_uint4(0u, 0u, 0u, 0u);
      fs[c].p = 0u;
    }
  }
#endif
  {
    int beyond = 0;  // valid entries of my blocks that lie beyond the speculative positions
#pragma unroll
    for (int j = 0; j < kIdxVecPerThread; ++j) {
      const int b = j * kThreads + tid;  // block id = positions 4b .. 4b+3
      const int nv = 4 - ((int)(v[j].x >> 31) + (int)(v[j].y >> 31) + (int)(v[j].z >> 31) + (int)(v[j].w >> 31));
      beyond += b >= kSpecBlocks ? nv : 0;
    }
    const bool any_beyond = __any_sync(0xffffffffu, beyond != 0);
    if (lane == 0) s_dirty[warp] = any_beyond ? 1 : 0;
  }
#elif DSA_INTERLEAVED && DSA_IDX_STAGE && DSA_CTA_VERIFY == 2
  // position p = kWarps*(32*j + lane) + warp: warp w owns every position p ≡ w (mod kWarps).
  // Every warp loads the whole list itself (lane l: blocks l + 32j, coalesced 512B loads), so it
  // can verify the speculation - "all valid indices sit in the first kWarps*kRowsPerChunk
  // positions" - with a warp-local count and no CTA barrier; its own positions are component
  // `warp` of those blocks, so the slow path compacts straight from registers as well.
  static_assert(kWarps == 4 && kIdxPerThread == kTopK / 4 / 32, "full-list mode: 4 warps, 16 blocks per lane");
  static_assert((kWarps * kRowsPerChunk) % 4 == 0, "speculative positions must be whole blocks");
  constexpr int kSpecBlocks = kWarps * kRowsPerChunk / 4;  // blocks 0..kSpecBlocks-1 = lanes 0.., j = 0
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
  uint4 v[kIdxPerThread];
#pragma unroll
  for (int j = 0; j < kIdxPerThread; ++j) v[j] = __ldg(idx4 + 32 * j + lane);
  int spec_iv[kRowsPerChunk];  // this warp's first kRowsPerChunk positions, one broadcast load each
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) spec_iv[c] = __ldg(idx + kWarps * c + warp);
#if DSA_SPEC == 2
  // ---- 1b. speculative gather (issued before the staging round trip): the common decode layout
  // puts a token's few valid indices at the front of the list, i.e. in this warp's first
  // positions. Start those row loads now; the exact compaction below runs while they are in
  // flight and only decides whether this guess covered every valid index.
  static_assert(DSA_IDX_STAGE, "spec=2 relies on the staged index path");
  RowFrag fs[kRowsPerChunk];
  bool spec_ok[kRowsPerChunk];
  int nspec = 0;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    const int row = spec_iv[c];
    spec_ok[c] = row >= 0;
    nspec += spec_ok[c] ? 1 : 0;
    if (spec_ok[c]) {
      load_row(fs[c], row, lane, ckv, kpe);
    } else {
      fs[c].a = make_uint4(0u, 0u, 0u, 0u);
      fs[c].b = make_uint4(0u, 0u, 0u, 0u);
      fs[c].p = 0u;
    }
  }
#endif
  // warp-local counts: valid entries in the whole list / in the speculative positions
  int nvalid = 0, nspec_all = 0;
#pragma unroll
  for (int j = 0; j < kIdxPerThread; ++j) {
    const int neg = (int)(v[j].x >> 31) + (int)(v[j].y >> 31) + (int)(v[j].z >> 31) + (int)(v[j].w >> 31);
    nvalid += 4 - neg;
    if (j == 0 && lane < kSpecBlocks) nspec_all += 4 - neg;
  }
#elif DSA_INTERLEAVED && DSA_IDX_STAGE
  // position p = kWarps*(32*j + lane) + warp: warp w owns every position p ≡ w (mod kWarps).
  // Coalesced uint4 loads, staged in natural order, read back with a 16B lane stride
  // (4 wavefronts per LDS, no swizzle needed).
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
  uint4 v[kIdxVecPerThread];
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) v[j] = __ldg(idx4 + j * kThreads + tid);
#if DSA_SPEC == 2
  int spec_iv[kRowsPerChunk];  // this warp's first kRowsPerChunk positions, one broadcast load each
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) spec_iv[c] = __ldg(idx + kWarps * c + warp);
#endif
#if DSA_SPEC == 2
  // ---- 1b. speculative gather (issued before the staging round trip): the common decode layout
  // puts a token's few valid indices at the front of the list, i.e. in this warp's first
  // positions. Start those row loads now; the exact compaction below runs while they are in
  // flight and only decides whether this guess covered every valid index.
  static_assert(DSA_IDX_STAGE, "spec=2 relies on the staged index path");
  RowFrag fs[kRowsPerChunk];
  bool spec_ok[kRowsPerChunk];
  int nspec = 0;
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    const int row = spec_iv[c];
    spec_ok[c] = row >= 0;
    nspec += spec_ok[c] ? 1 : 0;
    if (spec_ok[c]) {
      load_row(fs[c], row, lane, ckv, kpe);
    } else {
      fs[c].a = make_uint4(0u, 0u, 0u, 0u);
      fs[c].b = make_uint4(0u, 0u, 0u, 0u);
      fs[c].p = 0u;
    }
  }
#endif
#if DSA_SPEC == 2 && DSA_CTA_VERIFY
  int nvalid = 0;  // this thread's valid count over the whole list slice it loaded
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j)
    nvalid += ((int)v[j].x >= 0) + ((int)v[j].y >= 0) + ((int)v[j].z >= 0) + ((int)v[j].w >= 0);
#define DSA_STAGE_UINT4()                                                                  \
  do {                                                                                     \
    uint4* s_lin = reinterpret_cast<uint4*>(s_idx);                                        \
    _Pragma("unroll") for (int j = 0; j < kIdxVecPerThread; ++j) s_lin[j * kThreads + tid] = v[j]; \
    __syncthreads();                                                                       \
    const int* s_pos = reinterpret_cast<const int*>(s_idx);                                \
    _Pragma("unroll") for (int j = 0; j < kIdxPerThread; ++j)                              \
      iv[j] = s_pos[kWarps * (32 * j + lane) + warp];                                      \
  } while (0)
#else
  {
    uint4* s_lin = reinterpret_cast<uint4*>(s_idx);  // natural order: block b at s_lin[b]
#pragma unroll
    for (int j = 0; j < kIdxVecPerThread; ++j) s_lin[j * kThreads + tid] = v[j];
  }
  __syncthreads();
  {
    const int* s_pos = reinterpret_cast<const int*>(s_idx);
#pragma unroll
    for (int j = 0; j < kIdxPerThread; ++j) iv[j] = s_pos[kWarps * (32 * j + lane) + warp];
  }
#endif
#elif DSA_INTERLEAVED
  // position p = kWarps*(32*j + lane) + warp: warp w owns every position p ≡ w (mod kWarps)
#pragma unroll
  for (int j = 0; j < kIdxPerThread; ++j) iv[j] = __ldg(idx + kWarps * (32 * j + lane) + warp);
#else
  const uint4* idx4 = reinterpret_cast<const uint4*>(idx);
#pragma unroll
  for (int j = 0; j < kIdxVecPerThread; ++j) {
    const uint4 v = __ldg(idx4 + j * kThreads + tid);  // coalesced 512B per warp per load
    iv[4 * j] = (int)v.x; iv[4 * j + 1] = (int)v.y; iv[4 * j + 2] = (int)v.z; iv[4 * j + 3] = (int)v.w;
  }
#endif
  RowFrag qf;
  qf.a = __ldg(q_nope + (size_t)th * kCkvVecPerRow + lane);
  qf.b = __ldg(q_nope + (size_t)th * kCkvVecPerRow + 32 + lane);
  qf.p = __ldg(q_pe + (size_t)th * kKpeWordPerRow + lane);
  float q[18];
  unpack_frag(qf, q);
#pragma unroll
  for (int i = 0; i < 18; ++i) q[i] *= scale_log2e;  // scores land in the log2 domain

#if DSA_PREFETCH
#pragma unroll
  for (int r = 0; r < kIdxPerThread; ++r) {
    if (iv[r] >= 0) {  // start pulling the row toward L2 while the CTA is still compacting
      const char* crow = reinterpret_cast<const char*>(ckv + (size_t)iv[r] * kCkvVecPerRow);
#pragma unroll
      for (int line = 0; line < 8; ++line) prefetch_l2(crow + 128 * line);
      prefetch_l2(kpe + (size_t)iv[r] * kKpeWordPerRow);
    }
  }
#endif

#if DSA_INTERLEAVED && DSA_IDX_VEC && DSA_SPEC == 1
  // ---- 1b. speculative gather: the common decode layout puts a token's few valid indices at
  // the front of the list, i.e. in lane 0's first 4-index block of each warp. Broadcast those
  // four indices and start their row loads now; the full compaction below runs while the
  // loads are in flight and only decides whether this guess covered every valid index.
  RowFrag fs[kRowsPerChunk];
  bool spec_ok[kRowsPerChunk];
  int nspec = 0;
  static_assert(kRowsPerChunk >= 4, "speculative chunk holds lane 0's 4-index block");
#pragma unroll
  for (int c = 0; c < kRowsPerChunk; ++c) {
    const int row = c < 4 ? __shfl_sync(0xffffffffu, iv[c], 0) : -1;
    spec_ok[c] = row >= 0;
    nspec += spec_ok[c] ? 1 : 0;
    if (spec_ok[c]) {
      load_row(fs[c], row, lane, ckv, kpe);
    } else {
      fs[c].a = make_uint4(0u, 0u, 0u, 0u);
      fs[c].b = make_uint4(0u, 0u, 0u, 0u);
      fs[c].p = 0u;
    }
  }
#endif

  // ---- 2. per-warp compaction of valid indices (negative = padding, spec uses -1)
#if DSA_SPEC == 2 && DSA_CTA_VERIFY == 3
  DSA_STAMP(1);
  int cnt = nspec;          // provisional: verified at the merge barrier
  const bool fast = true;   // (the slow path is taken after the merge barrier if a flag is set)
  const uint32_t lt_mask = (1u << lane) - 1u;
  DSA_STAMP(2);
#elif DSA_SPEC == 2 && DSA_CTA_VERIFY == 2
  DSA_STAMP(1);  // (indices arrived in registers)
  {
    int packed = nvalid + (nspec_all << 16);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) packed += __shfl_xor_sync(0xffffffffu, packed, o);
    nvalid = packed & 0xffff;
    nspec_all = packed >> 16;
  }
  const bool fast = nvalid == nspec_all;  // identical in every warp (same data), no barrier needed
  int cnt = nspec;  // fast path: this warp owns exactly its speculative rows
  const uint32_t lt_mask = (1u << lane) - 1u;
  if (!fast) {  // general path: own positions are component `warp` of the loaded blocks
    cnt = 0;
#pragma unroll
    for (int r = 0; r < kIdxPerThread; ++r) {
      const int e = warp & 3;
      const int val = e == 0 ? (int)v[r].x : e == 1 ? (int)v[r].y : e == 2 ? (int)v[r].z : (int)v[r].w;
      iv[r] = val;
      const uint32_t b = __ballot_sync(0xffffffffu, val >= 0);
      if (val >= 0) s_list[warp][cnt + __popc(b & lt_mask)] = val;
      cnt += __popc(b);
    }
  }
  DSA_STAMP(2);  // verified / compacted
#elif DSA_SPEC == 2 && DSA_CTA_VERIFY
  // The speculative slots of all warps together are exactly positions [0, kWarps*kRowsPerChunk)
  // of the list, so the guess is right for every warp iff the CTA-wide valid count equals the
  // CTA-wide speculative count. Both are cheap to get without touching smem-staged indices.
  DSA_STAMP(1);  // (indices arrived in registers)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) nvalid += __shfl_xor_sync(0xffffffffu, nvalid, o);
  if (lane == 0) {
    s_cnt[warp] = nvalid;
    s_spec[warp] = nspec;
  }
  __syncthreads();
  int total_valid = 0, total_spec = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total_valid += s_cnt[w];
    total_spec += s_spec[w];
  }
  const bool fast = total_valid == total_spec;  // CTA-uniform
  int cnt = nspec;  // fast path: this warp owns exactly its speculative rows
  const uint32_t lt_mask = (1u << lane) - 1u;
  if (!fast) {  // general path: stage the list through smem, compact per warp
    DSA_STAGE_UINT4();
    cnt = 0;
#pragma unroll
    for (int r = 0; r < kIdxPerThread; ++r) {
      const int v = iv[r];
      const uint32_t b = __ballot_sync(0xffffffffu, v >= 0);
      if (v >= 0) s_list[warp][cnt + __popc(b & lt_mask)] = v;
      cnt += __popc(b);
    }
  }
  DSA_STAMP(2);  // verified / compacted
#else
#if DSA_TIMING
  {
    int any = 0;
#pragma unroll
    for (int r = 0; r < kIdxPerThread; ++r) any |= iv[r];
    DSA_MATERIALIZE(any);
  }
#endif
  DSA_STAMP(1);  // indices arrived
  const uint32_t lt_mask = (1u << lane) - 1u;
  int cnt = 0;
#pragma unroll
  for (int r = 0; r < kIdxPerThread; ++r) {
    const int v = iv[r];
    const uint32_t b = __ballot_sync(0xffffffffu, v >= 0);
    if (v >= 0) s_list[warp][cnt + __popc(b & lt_mask)] = v;
    cnt += __popc(b);
  }
  DSA_STAMP(2);  // indices compacted
#endif

  // ---- 3. online softmax over this warp's rows
  float m = -INFINITY, l = 0.f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;

#if DSA_INTERLEAVED
  __syncwarp();
#if DSA_SPEC && (DSA_IDX_VEC || DSA_IDX_STAGE)
#if DSA_SPEC == 2 && DSA_CTA_VERIFY
  if (fast) {  // every valid index of the token was in the speculative positions
#else
  if (cnt == nspec) {  // every valid index of this warp was in the speculative block
#endif
    if (nspec > 0) {
#if DSA_TIMING
      {
        uint32_t sig = 0;
#pragma unroll
        for (int c = 0; c < kRowsPerChunk; ++c) sig ^= fs[c].a.x ^ fs[c].b.w ^ fs[c].p;
        DSA_MATERIALIZE(sig);
        DSA_STAMP(3);  // speculative rows arrived
      }
#endif
      online_update<true>(fs, spec_ok, q, m, l, acc);
    }
    cnt = 0;  // general path below has nothing left to do
  }
#endif
  // general path: rows [0, cnt_) of this warp's compacted list, chunks of kRowsPerChunk
  auto run_general = [&](int cnt_) {
    for (int base = 0; base < cnt_; base += kRowsPerChunk) {
      RowFrag f[kRowsPerChunk];
      bool ok[kRowsPerChunk];
#if DSA_PAD_DUP
      // Every slot loads a real row: unused slots duplicate the warp's first row (finite data,
      // weighted by p = 0), which removes the zero-fill instructions from the critical path.
#pragma unroll
      for (int c = 0; c < kRowsPerChunk; ++c) {
        ok[c] = base + c < cnt_;
        load_row(f[c], s_list[warp][ok[c] ? base + c : 0], lane, ckv, kpe);
      }
#else
#pragma unroll
      for (int c = 0; c < kRowsPerChunk; ++c) {
        ok[c] = base + c < cnt_;
        if (ok[c]) {
          load_row(f[c], s_list[warp][base + c], lane, ckv, kpe);
        } else {  // keep the unused slot finite: 0 * NaN would poison acc
          f[c].a = make_uint4(0u, 0u, 0u, 0u);
          f[c].b = make_uint4(0u, 0u, 0u, 0u);
          f[c].p = 0u;
        }
      }
#endif
#if DSA_TIMING
      if (base == 0) {
        uint32_t sig = 0;
#pragma unroll
        for (int c = 0; c < kRowsPerChunk; ++c) sig ^= f[c].a.x ^ f[c].b.w ^ f[c].p;
        DSA_MATERIALIZE(sig);
        DSA_STAMP(3);  // first chunk of rows arrived
      }
#endif
      online_update<false>(f, ok, q, m, l, acc);
    }
  };
  run_general(cnt);
#else
  if (lane == 0) s_cnt[warp] = cnt;
  __syncthreads();
  int pre[kWarps + 1];
  pre[0] = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) pre[w + 1] = pre[w] + s_cnt[w];
  const int total = pre[kWarps];
  for (int base = warp; base < total; base += kWarps * kRowsPerChunk) {
    RowFrag f[kRowsPerChunk];
    bool ok[kRowsPerChunk];
#pragma unroll
    for (int c = 0; c < kRowsPerChunk; ++c) {
      const int r = base + c * kWarps;
      ok[c] = r < total;
      if (ok[c]) {
        int w = 0, off = r;  // locate the per-warp list holding global row r
#pragma unroll
        for (int i = 1; i < kWarps; ++i) {
          if (r >= pre[i]) { w = i; off = r - pre[i]; }
        }
        load_row(f[c], s_list[w][off], lane, ckv, kpe);
      } else {
        f[c].a = make_uint4(0u, 0u, 0u, 0u);
        f[c].b = make_uint4(0u, 0u, 0u, 0u);
        f[c].p = 0u;
      }
    }
    online_update<false>(f, ok, q, m, l, acc);
  }
#endif

  // ---- 4. merge the warp states through smem
  DSA_STAMP(4);  // warp 0 finished its rows (gather + online softmax)
  auto publish_partials = [&]() {
    if (lane == 0) {
      s_m[warp] = m;
      s_l[warp] = l;
    }
#if DSA_SKIP_EMPTY
    if (m != -INFINITY)  // warp-uniform: a warp without rows contributes nothing
#endif
    {
      float4* d0 = reinterpret_cast<float4*>(&s_acc[warp][8 * lane]);
      float4* d1 = reinterpret_cast<float4*>(&s_acc[warp][256 + 8 * lane]);
      d0[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      d0[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
      d1[0] = make_float4(acc[8], acc[9], acc[10], acc[11]);
      d1[1] = make_float4(acc[12], acc[13], acc[14], acc[15]);
    }
  };
  publish_partials();
  __syncthreads();
#if DSA_SPEC == 2 && DSA_CTA_VERIFY == 3
  {
    int dirty = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) dirty |= s_dirty[w];
    if (dirty) {  // CTA-uniform: some valid index lies beyond the speculative positions -> redo exactly
      m = -INFINITY;
      l = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] = 0.f;
      uint4* s_lin = reinterpret_cast<uint4*>(s_idx);  // natural order: block b at s_lin[b]
#pragma unroll
      for (int j = 0; j < kIdxVecPerThread; ++j) s_lin[j * kThreads + tid] = v[j];
      __syncthreads();
      const int* s_pos = reinterpret_cast<const int*>(s_idx);
      int cnt2 = 0;
#pragma unroll
      for (int j = 0; j < kIdxPerThread; ++j) {
        const int val = s_pos[kWarps * (32 * j + lane) + warp];
        const uint32_t b = __ballot_sync(0xffffffffu, val >= 0);
        if (val >= 0) s_list[warp][cnt2 + __popc(b & lt_mask)] = val;
        cnt2 += __popc(b);
      }
      __syncwarp();
      run_general(cnt2);
      __syncthreads();  // everyone finished reading the fast-path partials before they are overwritten
      publish_partials();
      __syncthreads();
    }
  }
#endif
  DSA_STAMP(5);  // all warps merged

  uint32_t* out_row = out + (size_t)th * (kDCkv / 2);
  float mw[kWarps], lw[kWarps];
#pragma unroll
  for (int w4 = 0; w4 < kWarps / 4; ++w4) {  // one 16B smem read per 4 warps
    const float4 mv = *reinterpret_cast<const float4*>(&s_m[4 * w4]);
    const float4 lv = *reinterpret_cast<const float4*>(&s_l[4 * w4]);
    mw[4 * w4] = mv.x; mw[4 * w4 + 1] = mv.y; mw[4 * w4 + 2] = mv.z; mw[4 * w4 + 3] = mv.w;
    lw[4 * w4] = lv.x; lw[4 * w4 + 1] = lv.y; lw[4 * w4 + 2] = lv.z; lw[4 * w4 + 3] = lv.w;
  }
  const float M = tree_max<kWarps>(mw);
  if (M == -INFINITY) {  // no valid index at all: reference semantics are zero output, lse = -inf
#pragma unroll
    for (int d = 0; d < kDimsPerThread / 2; ++d) out_row[d * kThreads + tid] = 0u;
    if (tid == 0) lse[th] = -INFINITY;
    return;
  }
  float sc[kWarps], sl[kWarps];
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    sc[w] = fast_exp2(mw[w] - M);  // warps that saw no row: m = -inf -> 0
    sl[w] = sc[w] * lw[w];
  }
  const float L = tree_sum<kWarps>(sl);
#if DSA_SKIP_EMPTY
  bool has[kWarps];
#pragma unroll
  for (int w = 0; w < kWarps; ++w) has[w] = mw[w] != -INFINITY;  // CTA-uniform
#endif
#if DSA_FAST_FINAL
  float inv;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(L));
#else
  const float inv = 1.f / L;
#endif
#pragma unroll
  for (int d = 0; d < kDimsPerThread / 2; ++d) {  // dim pair 2*(d*kThreads+tid): coalesced 4B stores
    const int col = 2 * (d * kThreads + tid);
    float2 a[kWarps];
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
#if DSA_SKIP_EMPTY
      a[w] = has[w] ? *reinterpret_cast<const float2*>(&s_acc[w][col]) : make_float2(0.f, 0.f);
#else
      a[w] = *reinterpret_cast<const float2*>(&s_acc[w][col]);
#endif
    }
    float t0[kWarps], t1[kWarps];
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      t0[w] = sc[w] * a[w].x;
      t1[w] = sc[w] * a[w].y;
    }
    const float o0 = tree_sum<kWarps>(t0) * inv;
    const float o1 = tree_sum<kWarps>(t1) * inv;
    __nv_bfloat162 o = __floats2bfloat162_rn(o0, o1);
    out_row[d * kThreads + tid] = *reinterpret_cast<uint32_t*>(&o);
  }
  DSA_STAMP(6);  // output written
#if DSA_FAST_FINAL
  if (tid == 0) {
    float lg;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(lg) : "f"(L));
    lse[th] = lg + M;  // 2-based log-sum-exp of the scaled logits (approx log2: ~2^-22 rel)
  }
#else
  if (tid == 0) lse[th] = log2f(L) + M;  // 2-based log-sum-exp of the scaled logits
#endif
}

__global__ void dsa_marker_kernel() {}

// Micro-benchmarks for the latency budget: 1 CTA, one thread, 1 or 2 dependent global loads;
// chain 3 = empty kernel with the real grid / block / smem shape (launch + teardown floor).
__global__ void dsa_probe_chain1_kernel(const int32_t* __restrict__ indices, int32_t* __restrict__ sink) {
  if (threadIdx.x == 0) sink[0] = __ldg(indices);
}
__global__ void dsa_probe_chain2_kernel(const int32_t* __restrict__ indices, const uint32_t* __restrict__ ckv,
                                        uint32_t* __restrict__ sink) {
  if (threadIdx.x == 0) {
    const int row = __ldg(indices);
    sink[0] = __ldg(ckv + (size_t)row * (kDCkv / 2));
  }
}
__global__ void __launch_bounds__(kThreads) dsa_probe_grid_kernel(int32_t* __restrict__ sink) {
  __shared__ float s_dummy[kWarps * kDCkv + kWarps * kListCap / 4];
  if (threadIdx.x == 0) {
    s_dummy[blockIdx.x] = 1.f;
    sink[blockIdx.y * kHeads + blockIdx.x] = (int32_t)s_dummy[blockIdx.x];
  }
}

// Instruction-fetch probes: straight-line independent FFMAs (8 chains), executed once.
// N_FMA*16 bytes of code; duration - floor = max(issue time, code fetch time).
template <int N_FMA, bool kLoadFirst>
__global__ void dsa_probe_code_kernel(const uint32_t* __restrict__ ckv, float* __restrict__ sink) {
  float a[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) a[i] = (float)(threadIdx.x + i);
  if (kLoadFirst) a[0] += __uint_as_float(__ldg(ckv + 7000u * (kDCkv / 2)));  // stall, then code
#pragma unroll
  for (int i = 0; i < N_FMA / 8; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = fmaf(a[j], 1.0001f, 0.5f);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += a[i];
  sink[threadIdx.x] = s;
}

// L2-residency probes: two independent loads from the index buffer / one load from a KV row
// nobody touched (row 8000) - compare against chain 1 (indices[0]) under the cold protocol.
__global__ void dsa_probe_idx2_kernel(const int32_t* __restrict__ indices, int32_t* __restrict__ sink) {
  if (threadIdx.x == 0) sink[0] = __ldg(indices) + __ldg(indices + 1024);
}
__global__ void dsa_probe_untouched_kernel(const uint32_t* __restrict__ ckv, uint32_t* __restrict__ sink) {
  if (threadIdx.x == 0) sink[0] = __ldg(ckv + 8000u * (kDCkv / 2));
}

// Cold-L2 alternative: stream-read a >L2-sized buffer so L2 ends up full of *clean* lines.
__global__ void dsa_l2_read_flush_kernel(const uint4* __restrict__ p, size_t n, uint4* __restrict__ sink) {
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if (acc.x == 0x9e3779b9u) sink[threadIdx.x] = acc;  // practically never true; defeats DCE
}

// Busy-wait so the memory system quiesces after a flush while L2 stays cold.
__global__ void dsa_spin_kernel(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

__global__ void dsa_l2_flush_kernel(uint4* __restrict__ p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_uint4(1u, 2u, 3u, 4u);
}

}  // namespace

extern "C" int dsa_launch(const void* q_nope, const void* q_pe, const void* ckv_cache,
                          const void* kpe_cache, const int32_t* sparse_indices, float sm_scale,
                          void* out, float* lse, int num_tokens, cudaStream_t stream) {
  dim3 grid(kHeads, num_tokens);
  constexpr int kDynSmem = DSA_SOLO_SMEM_KB * 1024;
  if (kDynSmem > 0) {
    static bool attr_set = false;  // one-time opt-in above the 48KB default
    if (!attr_set) {
      cudaFuncSetAttribute(dsa_sparse_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmem);
      attr_set = true;
    }
  }
  dsa_sparse_attn_kernel<<<grid, kThreads, kDynSmem, stream>>>(
      reinterpret_cast<const uint4*>(q_nope), reinterpret_cast<const uint32_t*>(q_pe),
      reinterpret_cast<const uint4*>(ckv_cache), reinterpret_cast<const uint32_t*>(kpe_cache),
      sparse_indices, sm_scale * 1.4426950408889634f, reinterpret_cast<uint32_t*>(out), lse);
  return (int)cudaGetLastError();
}

extern "C" const char* dsa_config() {
#define DSA_STR2(x) #x
#define DSA_STR(x) DSA_STR2(x)
  return "warps=" DSA_STR(DSA_WARPS) " rows_per_chunk=" DSA_STR(DSA_ROWS_PER_CHUNK)
         " interleaved=" DSA_STR(DSA_INTERLEAVED) " idx_vec=" DSA_STR(DSA_IDX_VEC)
         " prefetch=" DSA_STR(DSA_PREFETCH) " skip_compute=" DSA_STR(DSA_SKIP_COMPUTE)
         " idx_stage=" DSA_STR(DSA_IDX_STAGE) " spec=" DSA_STR(DSA_SPEC) " cta_verify=" DSA_STR(DSA_CTA_VERIFY)
         " skip_empty=" DSA_STR(DSA_SKIP_EMPTY)
         " fast_final=" DSA_STR(DSA_FAST_FINAL) " pad_dup=" DSA_STR(DSA_PAD_DUP)
         " solo_smem_kb=" DSA_STR(DSA_SOLO_SMEM_KB)
         " timing=" DSA_STR(DSA_TIMING);
}

// Timing build only: copies the per-CTA phase stamps (kTimingCtas x [7 x ns, 7 x cycles]) to host.
extern "C" int dsa_timing_read(void* dst, size_t bytes) {
#if DSA_TIMING
  if (bytes != sizeof(g_timing)) return -2;
  return (int)cudaMemcpyFromSymbol(dst, g_timing, sizeof(g_timing));
#else
  (void)dst;
  (void)bytes;
  return -1;
#endif
}

// Timeline marker used by bench.py to delimit one iteration in the CUPTI trace.
extern "C" int dsa_marker_launch(cudaStream_t stream) {
  dsa_marker_kernel<<<1, 32, 0, stream>>>();
  return (int)cudaGetLastError();
}

extern "C" int dsa_probe_launch(int chain, const int32_t* indices, const void* ckv, void* sink,
                                cudaStream_t stream) {
  const uint32_t* ckv32 = reinterpret_cast<const uint32_t*>(ckv);
  switch (chain) {
    case 1: dsa_probe_chain1_kernel<<<1, 32, 0, stream>>>(indices, reinterpret_cast<int32_t*>(sink)); break;
    case 2: dsa_probe_chain2_kernel<<<1, 32, 0, stream>>>(indices, ckv32, reinterpret_cast<uint32_t*>(sink)); break;
    case 3: dsa_probe_grid_kernel<<<dim3(kHeads, 2), kThreads, 0, stream>>>(reinterpret_cast<int32_t*>(sink)); break;
    case 4: dsa_probe_code_kernel<640, false><<<1, 32, 0, stream>>>(ckv32, reinterpret_cast<float*>(sink)); break;
    case 5: dsa_probe_code_kernel<160, false><<<1, 32, 0, stream>>>(ckv32, reinterpret_cast<float*>(sink)); break;
    case 6: dsa_probe_code_kernel<640, true><<<1, 32, 0, stream>>>(ckv32, reinterpret_cast<float*>(sink)); break;
    case 7: dsa_probe_idx2_kernel<<<1, 32, 0, stream>>>(indices, reinterpret_cast<int32_t*>(sink)); break;
    case 8: dsa_probe_untouched_kernel<<<1, 32, 0, stream>>>(ckv32, reinterpret_cast<uint32_t*>(sink)); break;
    default: return -1;
  }
  return (int)cudaGetLastError();
}

// Cold-L2 protocol: overwrite (mode 0 / 2) or stream-read (mode 1) a buffer larger than L2;
// mode 2 additionally idles ~10 us afterwards so DRAM / L2 write-backs settle before the kernel.
extern "C" int dsa_l2_flush_launch(void* p, size_t bytes, int mode, void* sink, cudaStream_t stream) {
  if (mode == 1)
    dsa_l2_read_flush_kernel<<<1024, 256, 0, stream>>>(reinterpret_cast<const uint4*>(p), bytes / 16,
                                                       reinterpret_cast<uint4*>(sink));
  else
    dsa_l2_flush_kernel<<<1024, 256, 0, stream>>>(reinterpret_cast<uint4*>(p), bytes / 16);
  if (mode == 2) dsa_spin_kernel<<<1, 32, 0, stream>>>(20000);  // ~10 us at ~1.9 GHz
  return (int)cudaGetLastError();
}
