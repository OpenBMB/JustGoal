// DSA sparse attention, h16 / ckv512 / kpe64 / topk2048 / page_size64, for B200.
//
// One kernel launch (a second launch costs ~11 us of CUPTI span on this
// benchmark), flash-decoding split over KV, cross-split reduction through
// distributed shared memory inside a 16-CTA cluster.
//
//   grid = (16, num_tokens)   cluster = (16,1,1)   block = 256 thr (8 warps)
//   CTA c of token t owns sparse_indices[t, c*128 : c*128+128]
//   CTA c finalises head c (NSPLIT == num_qo_heads == 16)
//
// Both GEMMs use mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32 with M = 16 query
// heads, which fits the instruction's M exactly (100% instruction efficiency).
// Operand fragments come straight out of SMEM via ldmatrix / ldmatrix.trans, so
// the gathered ckv rows are stored once and used as both K and V.
//
// Profiling (profile/dsa_v1_baseline, profile/dsa_v15_final) showed the
// kernel is pure serialised latency -- 0.6% of compute peak, 0.05% of DRAM
// peak -- so every design choice here is about shortening dependency chains:
//
//   * all 8 warps of a CTA start within ~0 ns of each other and a dependent
//     global load lands in ~608 ns, so the index row is read with plain
//     ld.global.v4 by all 256 threads as the very first instruction: it is in
//     registers after ~1,250 cycles. Reading it with one 8 KB cp.async.bulk
//     cost ~3,000 cycles instead, because cp.async.bulk is replayed per active
//     lane (~60-100 cycles each) and the q copies issued alongside it pushed
//     the index arrival out by another ~1,000;
//   * per-chunk valid counts and the chunk's own compaction then happen
//     straight out of those registers, 16 threads per chunk;
//   * the KV gather is issued by all 8 warps, because of that same per-lane
//     replay cost: 256 copies from one warp cost ~15,000 cycles;
//   * the gather is split into eight 16-row groups with one mbarrier each, so
//     warp w starts its QK as soon as *its* rows land rather than waiting for
//     the slowest row of the tile;
//   * partials are *pushed* into each peer's inbox, so only one cluster
//     barrier is needed (a pulling reduction needs a second one to stop a CTA
//     retiring while a peer still reads its SMEM);
//   * because every thread already holds 8 slots of the index row, the valid
//     count of all 16 chunks falls out of a 16-lane shuffle reduction. Every
//     CTA therefore derives the same `nactive` from identical global data, and
//     the cluster reduction (and its barrier) is skipped entirely when there is
//     nothing to reduce -- which is the common case, not a corner case: 79 of
//     the 120 query tokens in the dataset have a single non-empty chunk;
//   * the reduction partials are normalised fp16 (O_j / l_j, with the weight
//     absorbing l_j), halving the DSMEM traffic. O_j / l_j is a convex
//     combination of ckv rows, so its magnitude is bounded by max|ckv| however
//     many slots the chunk holds -- that is what makes 16 bits safe here.
#ifndef DSA_NO_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#endif

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace dsa {

namespace cg = cooperative_groups;

constexpr int NHEAD = 16;
constexpr int DCKV = 512;
constexpr int DKPE = 64;
constexpr int DQK = DCKV + DKPE;  // 576, the QK reduction length
constexpr int NDSTEP = DQK / 16;  // 36 mma k-steps
constexpr int TOPK = 2048;

constexpr int NSPLIT = 16;            // == cluster size == NHEAD
constexpr int CHUNK = TOPK / NSPLIT;  // 128 KV slots per CTA
constexpr int NWARP = 8;
constexpr int NTHREAD = NWARP * 32;
constexpr int GROW = CHUNK / NWARP;  // 16 KV rows per gather group / per warp

// SMEM row strides, in elements. Chosen so (stride_in_4B_words % 32) == 4,
// which spreads the 32 ldmatrix addresses over all 32 banks.
constexpr int KSTRIDE = 584;  // bf16; 1168 B; 292 words
constexpr int QSTRIDE = 584;  // bf16
constexpr int PSTRIDE = 136;  // bf16; 272 B; 68 words
constexpr int SSTRIDE = 132;  // f32
constexpr int OSTRIDE = 520;  // f32, staging only

struct alignas(16) SmemStore {
  __nv_bfloat16 q[NHEAD * QSTRIDE];  // 18688  q_nope | q_pe
  __nv_bfloat16 k[CHUNK * KSTRIDE];  // 149504 gathered ckv | kpe; after PV its
                                     //        first 33280 B stage the partial O
  float s[NHEAD * SSTRIDE];          // 8448   logits
  __nv_bfloat16 p[NHEAD * PSTRIDE];  // 4352   softmax numerators
  __half inbox[NSPLIT * DCKV];       // 16384  peers push their normalised O for
                                     //        MY head; also stages the
                                     //        fast-path output (16640 B)
  __half inbox_pad[8 * 64];  // keeps the fast-path staging inside inbox
  float inm[NSPLIT];                 //        peers push their m for MY head
  float inl[NSPLIT];                 //        peers push their l for MY head
  float m[NHEAD];
  float l[NHEAD];
  int cnt[NSPLIT];  // valid slots per chunk, for the whole token
  int idx[CHUNK];   // compacted (and tail-padded) token indices for my chunk
  int nvalid;
  unsigned long long bar_k[NWARP];
};
constexpr size_t kSmemBytes = sizeof(SmemStore);

#ifdef DSA_TIMELINE
// clock64() is a per-SM counter and is NOT comparable across CTAs; globaltimer
// is GPU-wide (~32 ns granularity). Record both: clock64 for intra-CTA phase
// deltas, globaltimer for cross-CTA skew.
__device__ unsigned long long g_tl[256][12];
__device__ unsigned long long g_gt[256][12];
#define TL(ph)                                                \
  do {                                                        \
    if (threadIdx.x == 0) {                                   \
      const int cta_ = blockIdx.y * NSPLIT + blockIdx.x;      \
      unsigned long long gt_;                                 \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_)); \
      g_tl[cta_][(ph)] = clock64();                           \
      g_gt[cta_][(ph)] = gt_;                                 \
    }                                                         \
  } while (0)
#else
#define TL(ph) \
  do {         \
  } while (0)
#endif

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(void* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect(void* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(void* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// Label-free: two inlined copies of an asm block with the same PTX label would
// collide (PTX labels are function-scoped), so spin in C++ instead.
__device__ __forceinline__ bool mbar_try_wait(void* bar) {
  unsigned done;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], 0;\n"
      "selp.u32 %0, 1, 0, P;\n"
      "}\n"
      : "=r"(done)
      : "r"(smem_u32(bar))
      : "memory");
  return done != 0u;
}

__device__ __forceinline__ void mbar_wait(void* bar) {
  while (!mbar_try_wait(bar)) {
  }
}

// Bulk (non-tensor) async global -> shared copy; `bytes` must be a multiple of 16.
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, void* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldm_x4_trans(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void mma_16x8x16(float (&d)[4], const unsigned (&a)[4],
                                            const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// QK for one warp's slice of the chunk.
//
// ROWS = 16 gives 8 warps x 16 KV rows, ROWS = 32 gives 4 warps x 32 rows. The
// B operand (the gathered K tile) is read exactly once either way, but the A
// operand (Q, 18 KB) is re-read once per *participating warp*, so widening the
// tile to 32 rows cuts QK's SMEM traffic from 294 KB to 220 KB. The mma count
// is identical and stays under the SM-level mma throughput limit, so that
// traffic is the thing worth shrinking.
//
// Fragment layout (PTX mma.m16n8k16, A row-major MxK, B col-major KxN):
//   A: matrices (head 0-7, d 0-7), (head 8-15, d 0-7), (head 0-7, d 8-15),
//      (head 8-15, d 8-15)  -> one ldmatrix.x4 over Q[head][d]
//   B: B[k=d][n=kv] = K[kv][d], and the mma B fragment wants
//      (n = lane>>2, k = (lane%4)*2+{0,1}), which is exactly what a plain
//      (non-transposing) ldmatrix of K rows delivers.
template <int ROWS>
__device__ __forceinline__ void qk_tile(SmemStore& sm, int warp, int lane, int nv16) {
  constexpr int NB = ROWS / 16;  // x4 B loads per d-step
  constexpr int NT = ROWS / 8;   // n-tiles (8 KV rows each)
  const int kvbase = warp * ROWS;
  if (kvbase >= nv16) return;
#pragma unroll
  for (int b = 0; b < NB; ++b) mbar_wait(&sm.bar_k[warp * NB + b]);

  float acc[NT][4] = {};
  const __nv_bfloat16* ap =
      sm.q + ((lane & 7) + 8 * ((lane >> 3) & 1)) * QSTRIDE + 8 * ((lane >> 4) & 1);
  const __nv_bfloat16* bp =
      sm.k + (kvbase + (lane & 7) + 8 * ((lane >> 4) & 1)) * KSTRIDE + 8 * ((lane >> 3) & 1);
  unsigned A[2][4], B[2][NB][4];
  ldm_x4(A[0], ap);
#pragma unroll
  for (int b = 0; b < NB; ++b) ldm_x4(B[0][b], bp + (size_t)b * 16 * KSTRIDE);
#pragma unroll
  for (int ds = 0; ds < NDSTEP; ++ds) {
    const int cur = ds & 1, nxt = cur ^ 1;
    if (ds + 1 < NDSTEP) {
      ldm_x4(A[nxt], ap + (ds + 1) * 16);
#pragma unroll
      for (int b = 0; b < NB; ++b)
        ldm_x4(B[nxt][b], bp + (size_t)b * 16 * KSTRIDE + (ds + 1) * 16);
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      mma_16x8x16(acc[b * 2 + 0], A[cur], &B[cur][b][0]);
      mma_16x8x16(acc[b * 2 + 1], A[cur], &B[cur][b][2]);
    }
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int hh = (lane >> 2) + 8 * (i >> 1);
      const int kv = kvbase + nt * 8 + (lane & 3) * 2 + (i & 1);
      sm.s[hh * SSTRIDE + kv] = acc[nt][i];
    }
  }
}

// QK with the 576-long reduction split over all eight warps instead of over
// the KV rows. For a small tile the d-loop *is* QK -- 36 serially dependent
// ldmatrix->mma steps, ~1,200 cycles (dev/tl.sh, 48-row token) -- while six of
// the eight warps have no KV rows to work on at all. Warp w takes d-steps
// w, w+8, ... (five or four of them) and the eight partial S tiles are summed
// through the upper half of the K tile, which no tile this small can be using.
//
// This is the direction v16/v17 rejected in Phase 3, re-done as its own
// template so the full-chunk path's codegen is untouched, and dispatched only
// on the three tile sizes that are actually some workload's largest token
// (nv16 = 16, 32, 64 -> 7 of the 8 fastest workloads).
template <int NT>
__device__ __forceinline__ void qk_dsplit(SmemStore& sm, int warp, int lane, int tid) {
  constexpr int ROWS = NT * 8;
  constexpr int NB = (NT + 1) / 2;  // ldmatrix.x4 loads per d-step
  constexpr int PSP = ROWS + 4;     // partial row stride in floats; (PSP%32)!=0
  constexpr int NOUT = NHEAD * ROWS / NTHREAD > 0 ? NHEAD * ROWS / NTHREAD : 1;
  static_assert(ROWS <= CHUNK / 2, "partials would overlap live K rows");
  static_assert(NWARP * NHEAD * PSP * 4 <= (CHUNK - CHUNK / 2) * KSTRIDE * 2,
                "partials do not fit in the upper half of the K tile");
  float* const part = reinterpret_cast<float*>(sm.k + (CHUNK / 2) * KSTRIDE);

  // every warp reads every row of the tile, so wait for all of its groups
#pragma unroll
  for (int b = 0; b < (ROWS + 15) / 16; ++b) mbar_wait(&sm.bar_k[b]);

  float acc[NT][4] = {};
  const __nv_bfloat16* ap =
      sm.q + ((lane & 7) + 8 * ((lane >> 3) & 1)) * QSTRIDE + 8 * ((lane >> 4) & 1);
  const __nv_bfloat16* bp =
      sm.k + ((lane & 7) + 8 * ((lane >> 4) & 1)) * KSTRIDE + 8 * ((lane >> 3) & 1);
  unsigned A[2][4], B[2][NB][4];
  int ds = warp;
  ldm_x4(A[0], ap + ds * 16);
#pragma unroll
  for (int b = 0; b < NB; ++b) ldm_x4(B[0][b], bp + (size_t)b * 16 * KSTRIDE + ds * 16);
  for (int it = 0; ds < NDSTEP; ++it, ds += NWARP) {
    const int cur = it & 1, nxt = cur ^ 1;
    const int dn = ds + NWARP;
    if (dn < NDSTEP) {
      ldm_x4(A[nxt], ap + dn * 16);
#pragma unroll
      for (int b = 0; b < NB; ++b) ldm_x4(B[nxt][b], bp + (size_t)b * 16 * KSTRIDE + dn * 16);
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      mma_16x8x16(acc[b * 2 + 0], A[cur], &B[cur][b][0]);
      if (b * 2 + 1 < NT) mma_16x8x16(acc[b * 2 + 1], A[cur], &B[cur][b][2]);
    }
  }

#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int hh = (lane >> 2) + 8 * (i >> 1);
      const int kv = nt * 8 + (lane & 3) * 2 + (i & 1);
      part[(warp * NHEAD + hh) * PSP + kv] = acc[nt][i];
    }
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < NOUT; ++k) {
    const int o = tid + k * NTHREAD;
    const int h = o / ROWS, row = o - h * ROWS;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) sum += part[(w * NHEAD + h) * PSP + row];
    sm.s[h * SSTRIDE + row] = sum;
  }
}

__global__ void __cluster_dims__(NSPLIT, 1, 1) __launch_bounds__(NTHREAD)
    DsaKernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
              const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
              const int* __restrict__ sparse_idx, float scale2, __nv_bfloat16* __restrict__ out,
              float* __restrict__ lse, int total_kv) {
  extern __shared__ __align__(16) char smem_raw[];
  SmemStore& sm = *reinterpret_cast<SmemStore*>(smem_raw);

  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int chunk = blockIdx.x;  // == cluster rank
  const int tok = blockIdx.y;

  TL(0);
  // ---- the whole index row, straight into registers, as the first thing the
  // ---- CTA does: 8 slots per thread, 32 B per thread, 1 KB per warp.
  const int4* const gidx =
      reinterpret_cast<const int4*>(sparse_idx + (size_t)tok * TOPK) + tid * 2;
  const int4 iv0 = gidx[0];
  const int4 iv1 = gidx[1];

  // ---- q, also with plain loads issued right here. 16 KB of q_nope is 1024
  // ---- int4 (4 per thread) and 2 KB of q_pe is 128 (1 per thread for half the
  // ---- block). cp.async.bulk would need 32 lane-replays and, more
  // ---- importantly, those copies were still in flight when the ckv gather
  // ---- was issued and throttled it badly (3,171 -> see dev/sol_v10m).
  int4 qn[4];
  int4 qp;
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int ix = tid + it * NTHREAD;  // 0..1023 -> head ix/64, 16 B col ix%64
    qn[it] = *reinterpret_cast<const int4*>(
        q_nope + (size_t)(tok * NHEAD + (ix >> 6)) * DCKV + (ix & 63) * 8);
  }
  if (tid < NHEAD * 8)
    qp = *reinterpret_cast<const int4*>(q_pe + (size_t)(tok * NHEAD + (tid >> 3)) * DKPE +
                                        (tid & 7) * 8);

  // Each warp initialises and owns the barrier for the 16 ckv rows it will
  // consume, so there is no cross-warp race on the mbarrier.
  if (lane == 0) mbar_init(&sm.bar_k[warp], GROW);
  __syncwarp();

  const int vals[8] = {iv0.x, iv0.y, iv0.z, iv0.w, iv1.x, iv1.y, iv1.z, iv1.w};
  TL(1);

  // ---- valid-slot count for every chunk of this token --------------------
  // Thread tid holds slots [8*tid, 8*tid+8) of chunk tid/16, so the 16 threads
  // of a chunk are an aligned half-warp and reduce among themselves. Every CTA
  // ends up with the same view of which chunks are non-empty, which is what
  // lets them all agree on whether a cluster reduction is needed at all.
  {
    int n = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) n += (vals[b] != -1);
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) n += __shfl_xor_sync(0xffffffffu, n, off);
    if ((tid & 15) == 0) sm.cnt[tid >> 4] = n;
  }

  // ---- compact my own chunk, tail-padded to a whole mma block -----------
  // Done by the one warp that holds this chunk's slots. The eight ballots are
  // independent so they pipeline; deriving each prefix from the previous one
  // instead cost ~1,100 cycles. Pad slots duplicate a *valid* row: the softmax
  // masks them to -inf so they contribute nothing, and the PV operand is never
  // uninitialised SMEM.
  if (warp == (chunk >> 1)) {
    const bool own = (tid >> 4) == chunk;
    unsigned msk[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) msk[b] = __ballot_sync(0xffffffffu, own && vals[b] != -1);
    const unsigned lt = (1u << lane) - 1u;
    int base = 0;
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      if (own && vals[b] != -1)
        sm.idx[base + __popc(msk[b] & lt)] = min(max(vals[b], 0), total_kv - 1);
      base += __popc(msk[b]);
    }
    if (lane == 0) sm.nvalid = base;
    __syncwarp();
    const int nv16 = (base + 15) & ~15;
    if (base > 0 && lane < 16 && base + lane < nv16) sm.idx[base + lane] = sm.idx[0];
  }
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int ix = tid + it * NTHREAD;
    *reinterpret_cast<int4*>(sm.q + (ix >> 6) * QSTRIDE + (ix & 63) * 8) = qn[it];
  }
  if (tid < NHEAD * 8)
    *reinterpret_cast<int4*>(sm.q + (tid >> 3) * QSTRIDE + DCKV + (tid & 7) * 8) = qp;
  TL(2);
  __syncthreads();

  const int myn = sm.nvalid;
  const int nv16 = (myn + 15) & ~15;
  const int h = chunk;  // NSPLIT == NHEAD: CTA c finalises head c
  int nactive = 0, afirst = 0;
#pragma unroll
  for (int c = NSPLIT - 1; c >= 0; --c)
    if (sm.cnt[c] > 0) {
      ++nactive;
      afirst = c;
    }

  // ---- gather the KV rows -------------------------------------------------
  // cp.async.bulk is replayed once per *active lane* (~60 cycles each), so the
  // only thing that matters is balancing lane-copies across warps. Warp w
  // takes exactly the 16 rows it will consume in QK, on lanes 0-15, with its
  // own mbarrier -- 16 lane-replays per warp instead of 32.
  // The 128 kpe rows are only 128 B each, so 128 lane-replays would cost more
  // than moving them the ordinary way: 4 ld.global.v4 + 4 st.shared.v4 per
  // thread, again with each warp handling exactly its own 16 rows.
  const int r0 = warp * GROW;
  // kpe first: these loads are consumed in the QK block below, so issuing them
  // ahead of the ckv bulk copies lets their ~1,250-cycle latency hide behind
  // the ckv issue instead of following it.
  int4 kv[4];
#pragma unroll
  for (int it = 0; it < 4; ++it) {
    const int j = lane + it * 32;  // 0..127 -> row j/8, 16 B column j%8
    const int row = r0 + (j >> 3);
    kv[it] = (row < nv16)
                 ? *reinterpret_cast<const int4*>(kpe + (size_t)sm.idx[row] * DKPE + (j & 7) * 8)
                 : make_int4(0, 0, 0, 0);
  }
  // Row r is issued by warp (r & 7), lane (r >> 3), i.e. lane L of warp w takes
  // row 8L + w. Every warp therefore issues ceil(nv16/8) lane-replays instead
  // of the 16 that the warp owning the rows would issue alone: 4 for a 32-row
  // token, 2 for a 16-row one, and still 16 for a full chunk, so the heavy
  // path is unchanged. Group barrier r>>4 always collects exactly its 16 rows
  // (nv16 is a multiple of 16), so the dummy arrives v15 needed are gone.
  if (lane < GROW) {
    const int r = lane * NWARP + warp;
    if (r < nv16) {
      mbar_arrive_expect(&sm.bar_k[r >> 4], DCKV * 2);
      bulk_g2s(sm.k + r * KSTRIDE, ckv + (size_t)sm.idx[r] * DCKV, DCKV * 2, &sm.bar_k[r >> 4]);
    }
  }
  TL(3);

  // ---- no valid KV slot anywhere in this token --------------------------
  if (nactive == 0) {
    __nv_bfloat16* op = out + ((size_t)tok * NHEAD + h) * DCKV;
    reinterpret_cast<__nv_bfloat162*>(op)[tid] = __floats2bfloat162_rn(0.f, 0.f);
    if (tid == 0) lse[tok * NHEAD + h] = neg_inf();
    return;
  }

  // Exactly one non-empty chunk: it owns the whole token, nothing to reduce,
  // and the cluster barrier is skipped by every CTA (the decision comes from
  // identical global data, so it is cluster-uniform).
  const bool single = (nactive == 1);
  if (single && chunk != afirst) return;  // no gather was issued here

  // Writing into a peer's distributed shared memory is only legal once that
  // peer has started executing -- the programming guide requires a
  // cluster-wide sync before any remote SMEM access, and compute-sanitizer
  // racecheck flags v15's single-barrier push as "a block that might not have
  // entered yet". It is not theoretical: perturbing the instruction schedule
  // of the push loop was enough to produce NaN output (dev/sol_v25).
  //
  // The handshake is a *split* barrier so it costs nothing on the critical
  // path: arrive here, then run QK + softmax + PV (~2,000 cycles) before
  // waiting. Only the reduction path needs it, and `nactive` is derived from
  // identical global data so the branch is cluster-uniform.
  const bool reduce = !single;  // nactive >= 2
  if (reduce) asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");

  float o[8][4] = {};
  if (myn > 0) {
    // Land the kpe rows now: the store stalls on the ld.global issued above,
    // and that stall overlaps the ckv rows still arriving over cp.async.bulk.
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int j = lane + it * 32;
      const int row = r0 + (j >> 3);
      if (row < nv16)
        *reinterpret_cast<int4*>(sm.k + row * KSTRIDE + DCKV + (j & 7) * 8) = kv[it];
    }
    __syncwarp();
    TL(4);
    // ================= QK: S[16 head, kv] = Q[16,576] . K[kv,576]^T =======
    if (nv16 == CHUNK) {
      // The 32-row tile makes warp w consume rows 32w..32w+31, whose kpe
      // columns were stored by warps 2w and 2w+1 -- the ckv half is covered by
      // the group mbarriers but the kpe half is not, so __syncwarp() is not
      // enough here. compute-sanitizer racecheck reports exactly this pair
      // (kpe st.shared vs. the tile's ldmatrix) in v15. The 16-row tile needs
      // no barrier: there, warp w consumes exactly the rows it stored.
      __syncthreads();
      qk_tile<32>(sm, warp, lane, nv16);  // full chunk: 4 warps x 32 rows
    } else if (nv16 == 16) {
      __syncthreads();  // every warp now reads every row's kpe columns
      qk_dsplit<2>(sm, warp, lane, tid);
    } else if (nv16 == 32) {
      __syncthreads();
      qk_dsplit<4>(sm, warp, lane, tid);
    } else {
      qk_tile<16>(sm, warp, lane, nv16);  // partial chunk: 8 warps x 16 rows
    }
    __syncthreads();
    TL(5);

    // ================= softmax, base 2 ====================================
    // One head per 16-lane group, so the two heads of a warp share the same
    // shfl_xor instructions: 4+4 dependent shuffles instead of 5+5 per head.
    {
      const int hs = warp * 2 + (lane >> 4);
      const int j0 = (lane & 15) * 8;
      const float* sp = sm.s + hs * SSTRIDE + j0;
      const float4 lo = *reinterpret_cast<const float4*>(sp);
      const float4 hi = *reinterpret_cast<const float4*>(sp + 4);
      float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
      float mx = neg_inf();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[i] = (j0 + i < myn) ? v[i] * scale2 : neg_inf();
        mx = fmaxf(mx, v[i]);
      }
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[i] = (j0 + i < myn) ? exp2f(v[i] - mx) : 0.f;
        sum += v[i];
      }
#pragma unroll
      for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
      __align__(16) __nv_bfloat162 pb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) pb[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
      *reinterpret_cast<float4*>(sm.p + hs * PSTRIDE + j0) = *reinterpret_cast<float4*>(pb);
      if ((lane & 15) == 0) {
        sm.m[hs] = mx;
        sm.l[hs] = sum;
      }
    }
    __syncthreads();
    TL(6);

    // ================= PV: O[16 head, 512] = P[16, kv] . K[kv, 0:512] =====
    {
      const int dbase = warp * 64;
      const __nv_bfloat16* ap =
          sm.p + ((lane & 7) + 8 * ((lane >> 3) & 1)) * PSTRIDE + 8 * ((lane >> 4) & 1);
      const __nv_bfloat16* bp =
          sm.k + ((lane & 7) + 8 * ((lane >> 3) & 1)) * KSTRIDE + dbase + 8 * ((lane >> 4) & 1);
      for (int ks = 0; ks < nv16; ks += 16) {
        unsigned a[4], b[4][4];
        ldm_x4(a, ap + ks);
#pragma unroll
        for (int pr = 0; pr < 4; ++pr) ldm_x4_trans(b[pr], bp + (size_t)ks * KSTRIDE + pr * 16);
#pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
          mma_16x8x16(o[pr * 2 + 0], a, &b[pr][0]);
          mma_16x8x16(o[pr * 2 + 1], a, &b[pr][2]);
        }
      }
    }
    TL(7);
  }

  // ---- single non-empty chunk: write the final result straight out -------
  if (single) {
    // Storing 16 KB straight from the PV accumulators gives 8 scattered 4 B
    // segments per warp instruction. Stage it in the (unused) inbox region
    // first, then do fully coalesced 16 B stores: ~300 cycles instead of
    // ~1,400.
    constexpr int OBS = DCKV + 8;  // bf16 stride; keeps the staging stores off
                                   // a single bank group
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(sm.inbox);
    static_assert(sizeof(sm.inbox) + sizeof(sm.inbox_pad) >= NHEAD * (DCKV + 8) * 2,
                  "fast-path staging must fit in the inbox region");
    const int dbase = warp * 64;
    const int h0 = lane >> 2;
    const float inv0 = __fdividef(1.f, sm.l[h0]);
    const float inv1 = __fdividef(1.f, sm.l[h0 + 8]);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int d = dbase + nt * 8 + (lane & 3) * 2;
      *reinterpret_cast<__nv_bfloat162*>(ob + h0 * OBS + d) =
          __floats2bfloat162_rn(o[nt][0] * inv0, o[nt][1] * inv0);
      *reinterpret_cast<__nv_bfloat162*>(ob + (h0 + 8) * OBS + d) =
          __floats2bfloat162_rn(o[nt][2] * inv1, o[nt][3] * inv1);
    }
    if (tid < NHEAD) lse[tok * NHEAD + tid] = sm.m[tid] + __log2f(sm.l[tid]);
    __syncthreads();
    // 16 threads per head, 4 x 16 B each -> 256 B contiguous per instruction.
    const int oh = tid >> 4;
    const int4* src = reinterpret_cast<const int4*>(ob + oh * OBS);
    int4* dst = reinterpret_cast<int4*>(out + ((size_t)tok * NHEAD + oh) * DCKV);
#pragma unroll
    for (int k = 0; k < 4; ++k) dst[(tid & 15) + k * 16] = src[(tid & 15) + k * 16];
    TL(11);
    return;
  }

  // ======================================================================
  // Several non-empty chunks: push the partials into each peer's inbox, then
  // one cluster barrier, then every CTA reduces its own head from local SMEM.
  // ======================================================================
  // Peers have started; their inboxes are safe to write.
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");

  if (myn > 0) {
    __syncthreads();  // all PV reads of sm.k are done; its first 33 KB is free
    float* part = reinterpret_cast<float*>(sm.k);
    const int dbase = warp * 64;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int hh = (lane >> 2) + 8 * (i >> 1);
        const int d = dbase + nt * 8 + (lane & 3) * 2 + (i & 1);
        part[hh * OSTRIDE + d] = o[nt][i];
      }
    }
    __syncthreads();
    TL(8);
    // Warp w pushes heads 2w and 2w+1 as normalised fp16; each store
    // instruction is 32 lanes x 16 B = 512 B contiguous into one peer rank.
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int ph = warp * 2 + hh;
      const float inv = __fdividef(1.f, sm.l[ph]);
      const float* src = part + ph * OSTRIDE;
      int4* dst = reinterpret_cast<int4*>(cluster.map_shared_rank(sm.inbox, ph) + chunk * DCKV);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int d0 = (k * 32 + lane) * 8;
        const float4 a = *reinterpret_cast<const float4*>(src + d0);
        const float4 b = *reinterpret_cast<const float4*>(src + d0 + 4);
        __align__(16) __half2 hv[4];
        hv[0] = __floats2half2_rn(a.x * inv, a.y * inv);
        hv[1] = __floats2half2_rn(a.z * inv, a.w * inv);
        hv[2] = __floats2half2_rn(b.x * inv, b.y * inv);
        hv[3] = __floats2half2_rn(b.z * inv, b.w * inv);
        dst[k * 32 + lane] = *reinterpret_cast<const int4*>(hv);
      }
    }
  }
  if (tid < NSPLIT) {
    cluster.map_shared_rank(sm.inm, tid)[chunk] = (myn > 0) ? sm.m[tid] : neg_inf();
    cluster.map_shared_rank(sm.inl, tid)[chunk] = (myn > 0) ? sm.l[tid] : 0.f;
  }
  TL(9);
  // cg::cluster_group::sync() emits the *relaxed* `barrier.cluster.arrive` /
  // `barrier.cluster.wait` pair -- verified in the PTX -- which orders
  // execution but not memory. The DSMEM pushes above are therefore not
  // guaranteed to be visible to the peer that reads them after its wait, and a
  // scheduling perturbation elsewhere in the kernel was enough to turn that
  // into wrong output and NaNs (dev/sol_v23b). The release/acquire pair is the
  // documented ordering form and is what this reduction needs.
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  TL(10);

  float gmax = neg_inf();
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j) gmax = fmaxf(gmax, sm.inm[j]);

  float lsum = 0.f;
  float w[NSPLIT];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j) {
    // the pushed partials are O_j / l_j, so the weight absorbs l_j
    w[j] = (sm.inl[j] > 0.f) ? sm.inl[j] * exp2f(sm.inm[j] - gmax) : 0.f;
    lsum += w[j];
  }
  const float inv = __fdividef(1.f, lsum);

  const int d = tid * 2;
  float2 acc = make_float2(0.f, 0.f);
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j) {
    if (w[j] == 0.f) continue;
    const float2 v = __half22float2(*reinterpret_cast<const __half2*>(sm.inbox + j * DCKV + d));
    acc.x = fmaf(v.x, w[j], acc.x);
    acc.y = fmaf(v.y, w[j], acc.y);
  }
  *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)tok * NHEAD + h) * DCKV + d) =
      __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  if (tid == 0) lse[tok * NHEAD + h] = gmax + __log2f(lsum);
  TL(11);
}

// Shared by the FFI entry point and by dev/probe/standalone.cu.
// Returns 0 on success, non-zero if the launch configuration was rejected.
inline int Launch(const __nv_bfloat16* q_nope, const __nv_bfloat16* q_pe,
                  const __nv_bfloat16* ckv, const __nv_bfloat16* kpe, const int* sparse_idx,
                  float sm_scale, __nv_bfloat16* out, float* lse, int num_tokens, int total_kv,
                  cudaStream_t stream) {
  static bool configured = false;
  static cudaError_t cfg_err = cudaSuccess;
  if (!configured) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(DsaKernel),
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cfg_err = cudaFuncSetAttribute(reinterpret_cast<const void*>(DsaKernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(kSmemBytes));
    configured = true;
  }
  if (cfg_err != cudaSuccess) return 1;

  const float scale2 = sm_scale * 1.4426950408889634f;  // log2(e)
  dim3 grid(NSPLIT, num_tokens, 1);
  DsaKernel<<<grid, NTHREAD, kSmemBytes, stream>>>(q_nope, q_pe, ckv, kpe, sparse_idx, scale2,
                                                   out, lse, total_kv);
  return 0;
}

#ifndef DSA_NO_FFI
void Run(tvm::ffi::TensorView q_nope, tvm::ffi::TensorView q_pe, tvm::ffi::TensorView ckv,
         tvm::ffi::TensorView kpe, tvm::ffi::TensorView sparse_idx, double sm_scale,
         tvm::ffi::TensorView out, tvm::ffi::TensorView lse) {
  DLDevice dev = q_nope.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const int rc = Launch(static_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
                        static_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
                        static_cast<const __nv_bfloat16*>(ckv.data_ptr()),
                        static_cast<const __nv_bfloat16*>(kpe.data_ptr()),
                        static_cast<const int*>(sparse_idx.data_ptr()),
                        static_cast<float>(sm_scale),
                        static_cast<__nv_bfloat16*>(out.data_ptr()),
                        static_cast<float*>(lse.data_ptr()), static_cast<int>(q_nope.size(0)),
                        static_cast<int>(ckv.size(0)) * static_cast<int>(ckv.size(1)), stream);
  if (rc != 0) {
    TVM_FFI_THROW(RuntimeError) << "DSA kernel launch configuration failed (smem " << kSmemBytes
                                << " bytes)";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif  // DSA_NO_FFI

}  // namespace dsa
