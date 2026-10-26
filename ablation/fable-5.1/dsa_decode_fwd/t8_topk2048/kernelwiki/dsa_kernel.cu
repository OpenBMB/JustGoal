// DSA sparse attention decode kernel (B200 / sm_100a), written from scratch.
//
// v10: 16 CTAs per token, 128 rows and 512 threads each, one CTA per SM. Every warp loads its own 8
// indices (32 B) and gathers its own 8 rows; the K/V copies of a 16-row QK tile are tracked by that
// tile's own mbarrier, so a warp starts QK^T as soon as the rows it multiplies have landed instead of
// waiting for all 128 rows of the CTA. No CTA-wide index reduction, no bound bookkeeping: every CTA
// runs the same straight-line path. Split-KV reduction through NCCL-LL style (payload, epoch) words.
//
// Row ownership: sparse_indices[t] is split into 64 blocks of 32 consecutive positions; block b belongs
// to CTA (b % 16) as its local block (b / 16). So CTA r owns positions [32r, 32r+32) + 512k, k = 0..3.
// Rows are gathered in local order, and validity is a prefix of that order (the position of local row
// rr is increasing in rr), so the invalid rows of a CTA are always its tail.
//
// Per CTA:
//   1. each warp reads its own 8 indices and counts its valid rows.
//   2. TMA bulk-copy gather of Q and of this warp's valid K/V rows into padded smem; invalid rows are
//      zero-filled from registers (PV multiplies whole 16-row K tiles, and 0 * NaN would be NaN).
//   3. S = Q K^T (mma.sync, k-split over warps), masked softmax (base 2), O = P V.
//   4. write the bf16 partial O as (bf16x2, epoch) words, then the (m, l) words per head; no fence,
//      every word carries its own epoch flag.
//   5. poll the 16 (m, l) tags of the token, merge the 16 partials, write bf16 out (+ lse from CTA 0).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int H = 16;
constexpr int D = 512;
constexpr int DP = 64;
constexpr int TOPK = 2048;
constexpr int NTHREADS = 512;
constexpr int NWARPS = NTHREADS / 32;       // 16
constexpr int NCTA = 16;                    // CTAs per token
constexpr int BLK = 32;                     // positions per block
constexpr int NBLK = 4;                     // blocks per CTA
constexpr int ROWS = BLK * NBLK;            // 128 rows per CTA
constexpr int BLK_STRIDE = TOPK / NBLK;     // 512: block k of CTA r starts at position 32r + 512k
constexpr int ROWS_PER_WARP = ROWS / NWARPS;  // 8 gather rows per warp
constexpr int SLICE = D / NCTA;             // 32 output columns reduced per CTA
constexpr int SW = SLICE / 2;               // 16 (bf16x2, epoch) words per (slot, head) in a slice
static_assert(NCTA * BLK == BLK_STRIDE && NCTA * ROWS == TOPK, "blocks cover the token");

constexpr int ROW_BYTES = D * 2 + DP * 2;  // 1152
constexpr int KC_STRIDE = D * 2 + 16;      // 1040: 16B pad -> conflict-free ldmatrix
constexpr int KP_STRIDE = DP * 2 + 16;     // 144
constexpr int P_STRIDE = ROWS * 2 + 16;    // 272
// S is scattered float2 by (g, q) and read back as float4 by (h, lane). At ROWS it lands 8 row-groups
// on one bank set (g*128 % 32 == 0); +4 gives 4g % 32, eight distinct groups. Sizes S by 512 B.
constexpr int S_STRIDE = ROWS + 4;         // 132
constexpr int KSPLIT = 2;                  // warps split the 36 k-steps 2 ways
constexpr int NHALF = NWARPS / KSPLIT;     // 8 QK tiles of 16 rows (2 n-tiles) each
constexpr int KSTEPS = (D + DP) / 16;      // 36
constexpr int KPW = KSTEPS / KSPLIT;       // 18 k-steps per warp

constexpr int OFF_KC = 0;                            // [ROWS][KC_STRIDE]
constexpr int OFF_KP = OFF_KC + ROWS * KC_STRIDE;    // [ROWS][KP_STRIDE]
constexpr int OFF_QN = OFF_KP + ROWS * KP_STRIDE;    // [H][KC_STRIDE]
constexpr int OFF_QP = OFF_QN + H * KC_STRIDE;       // [H][KP_STRIDE]
constexpr int OFF_S = OFF_QP + H * KP_STRIDE;        // [KSPLIT][H][ROWS] fp32
constexpr int OFF_P = OFF_S + KSPLIT * H * S_STRIDE * 4; // [H][P_STRIDE] bf16
constexpr int OFF_ML = OFF_P + H * P_STRIDE;         // m[16] l[16] (fp32)
constexpr int OFF_IDX = OFF_ML + 256;                // ROWS int32 (local order)
constexpr int OFF_BAR = OFF_IDX + ROWS * 4;          // barQ + NHALF K-tile barriers (8 B each)
constexpr int OFF_CNT = OFF_BAR + 128;               // profiling scratch
constexpr int SMEM_TOTAL = OFF_CNT + 128;
static_assert(SMEM_TOTAL <= 232448, "one CTA per SM");

// Split-KV workspace, NCCL-LL style: every 8-byte word is (payload, epoch flag) so that a reader can
// validate each word on its own and no release fence is needed between the partial O and the (m, l).
// partial O of a (token, slot) is stored [NCTA col slices][H][SW words]: the 2 KB a reducer needs is contiguous
// STAGGER: both workspace strides were exact powers of two (4096 and 32 u64 words = 2**15 B and
// 2**8 B). A merging thread reads 8 sibling slots at that fixed pitch, so every address it touches
// differs only in bits >= 8: one L2 set, one sector slot, per iteration. Padded off the power of two
// the same bytes land in different sets. Measured -0.11 to -0.15 us, reproducible, with only the (m,l)
// stride carrying it (O alone was noise, O alone at 256 B ML was noise). Strides stay even so the
// v2.b64 accesses keep 16-byte alignment -- an O pad of 2 words cost +0.92 us, so this is not a free
// parameter. The full ladder (ML pad 0/2/4/8/16/32, O pad 0/2/4/16/32/128) is in the session notes.
constexpr int WS_O_STRIDE = H * D / 2 + 16;          // u64 words per (token, slot) partial O (bf16x2 payload)
constexpr int WS_ML_STRIDE = H * 2 + 8;              // u64 words per (token, slot): 16 x ((m, ep), (l, ep))

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}
// 16 B global -> shared copy, Ampere style. One instruction per lane, so a 1024 B row is two rounds of a
// full warp and the gather gets 32-way request parallelism instead of one TMA descriptor per row (the
// single TMA unit was draining 256 descriptors per CTA, which is what the gather time was made of).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
// Waits until every cp.async this thread issued has landed. Paired with a plain mbarrier.arrive by one
// lane per warp, so the barrier keeps the simple "one arrival per contributing warp" accounting instead
// of cp.async.mbarrier.arrive, whose interaction with the barrier's pending count is easy to get wrong.
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                              uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf162(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf162(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}
// (payload, flag) words: 64-bit scalar accesses are single-copy atomic, so a reader that sees the flag
// of a word also sees its payload.
__device__ __forceinline__ unsigned long long ll_pack(float v, uint32_t flag) {
  return ((unsigned long long)flag << 32) | __float_as_uint(v);
}
__device__ __forceinline__ unsigned long long ll_pack_u(uint32_t v, uint32_t flag) {
  return ((unsigned long long)flag << 32) | v;
}
__device__ __forceinline__ void st_ll2(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1,%2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
// Readers use L2-only (.cg) loads: the epoch flag inside every word validates it, so no ordering is needed
// beyond bypassing L1 (asm volatile keeps the compiler from reusing a value across re-reads).
__device__ __forceinline__ unsigned long long ld_ll(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void ld_ll2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.global.cg.v2.b64 {%0,%1}, [%2];" : "=l"(a), "=l"(b) : "l"(p));
}
__device__ __forceinline__ float fast_exp2(float x) {  // ex2.approx.ftz: 2 ulp, exp2(-inf) = +0
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Fire-and-forget L2 prefetch. No destination register, no scoreboard entry, no dependency chain --
// unlike a real load it cannot stall the issuing warp or occupy the register file. This is the one
// mechanism the two earlier attempts at hiding this latency did NOT use: "read before poll" issued REAL
// loads into registers from all 15 warps while warp 0 spun (10.400 us), and dropping the poll entirely
// (11.029 us) made every warp race the retry loop. A prefetch is neither -- it warms L2 without
// competing for the poll's lines, so the retry leg's round trip becomes a hit.
__device__ __forceinline__ void pf_l2(const void* a) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
}
__device__ __forceinline__ uint32_t ll_flag(unsigned long long w) { return (uint32_t)(w >> 32); }
__device__ __forceinline__ float ll_val(unsigned long long w) { return __uint_as_float((uint32_t)w); }
__device__ __forceinline__ uint32_t ll_u(unsigned long long w) { return (uint32_t)w; }

#ifdef PROF
constexpr int PROF_N = 16;
__device__ unsigned long long g_prof[NCTA * 64 * PROF_N];
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define PROF_STAMP(k)                                                                              \
  do {                                                                                             \
    if (threadIdx.x == 0) g_prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_N + (k)] = gtimer(); \
  } while (0)
#else
#define PROF_STAMP(k) \
  do {                \
  } while (0)
#endif

extern __shared__ __align__(1024) uint8_t smem[];

__global__ void __launch_bounds__(NTHREADS, 1)
dsa_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
           const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
           const int* __restrict__ sparse_idx, float scale_log2, __nv_bfloat16* __restrict__ out,
           float* __restrict__ lse, unsigned long long* __restrict__ ws_o,
           unsigned long long* __restrict__ ws_ml, int epoch) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int t = blockIdx.y, r = blockIdx.x;
  const uint32_t sbase = smem_u32(smem);
  const uint32_t barQ = sbase + OFF_BAR;
  const uint32_t barK0 = sbase + OFF_BAR + 8;  // barK0 + 8 g -> K-tile barrier of QK tile g
  int* idx_s = reinterpret_cast<int*>(smem + OFF_IDX);
  float* m_s = reinterpret_cast<float*>(smem + OFF_ML);
  float* l_s = m_s + 16;
  const uint32_t ep = (uint32_t)epoch;

  PROF_STAMP(0);
  PROF_STAMP(12);
  const int lrow = warp * ROWS_PER_WARP;
  // ---- Phase 0: issue this warp's 8 indices FIRST. This is a cold DRAM fetch of the token's sparse row
  //      list and it used to sit exposed between the barrier bring-up and the gather, costing a full
  //      ~1.0 us on every CTA. Issued here it stays in flight across the barrier init, the syncthreads
  //      and the Q copy, i.e. under ~everything the prologue does. bar.sync does not drain outstanding
  //      loads, only the consuming use does, so the value arrives where it is needed regardless.
  const int* ibase = sparse_idx + (size_t)t * TOPK + (lrow / BLK) * BLK_STRIDE + r * BLK + (lrow % BLK);
  int idl = -1;
  if (lane < 16) idl = __ldg(ibase + (lane & 7));

  // ---- Phase 1a: barrier bring-up, then the Q copy. Q depends only on (token, warp), not on the
  //      indices, so issuing it before the index value is consumed hides its TMA latency too. The
  //      barriers must be initialised and visible to every warp before any warp issues into them, hence
  //      the syncthreads between the two.
  if (tid == 0) {
    mbar_init(barQ, 1);
#pragma unroll
    for (int g = 0; g < NHALF; ++g) mbar_init(barK0 + 8 * g, 2);  // one arrival per contributing warp
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(barQ, (uint32_t)H * ROW_BYTES);
  }
  PROF_STAMP(13);
  __syncthreads();  // barriers initialised
  PROF_STAMP(14);

  if (lane == 16) {
    bulk_g2s(sbase + OFF_QN + warp * KC_STRIDE, q_nope + ((size_t)t * H + warp) * D, D * 2, barQ);
  } else if (lane == 17) {
    bulk_g2s(sbase + OFF_QP + warp * KP_STRIDE, q_pe + ((size_t)t * H + warp) * DP, DP * 2, barQ);
  }

  // ---- Phase 1b: consume the indices. Lanes 0-7 own the 8 rows this warp gathers; the ballot is
  //      masked to them so nvalid counts each row once and the K-tile barrier waits only for bytes
  //      that are actually delivered. (Lanes 8-15 used to mirror rows 0-7 to issue a parallel kpe
  //      descriptor; kpe is now the consumer's cp.async, so those lanes hold nothing.)
  const unsigned vmask = __ballot_sync(0xffffffffu, lane < 8 && idl >= 0);
  const int nvalid = __popc(vmask);                 // 0 .. 8 == number of rows actually copied
  const int lastv = vmask ? 32 - __clz(vmask) : 0;  // valid rows of this warp are [0, lastv)
  if (lane < 8) idx_s[lrow + lane] = idl;
#ifdef PROF
  if (lane == 0) atomicAdd(reinterpret_cast<unsigned*>(smem + OFF_CNT), (unsigned)nvalid);
#endif
  PROF_STAMP(1);

  // ---- Phase 2: gather. A warp arms its own K-tile barrier with exactly the bytes it will deliver, so
  //      the tile completes the moment its two gather warps are done. expect_tx is registered before the
  //      copies: a complete_tx that lands against a zero tx count is lost.
  //
  //      The ckv leg stays on TMA (8 descriptors of 1024 B). The kpe leg is 128 B per row, and a TMA
  //      descriptor costs the same to issue whether it moves 128 B or 9216 B -- measured 16 descriptors
  //      at 1152 B and 16 at 9216 B are both 2.70 us, while 8 descriptors take 1.74 us. So the 128 B kpe
  //      row was spending a whole descriptor for 1/8 the bytes. It moves to the LSU pipe as cp.async,
  //      which gets 32-way request parallelism at zero descriptor cost.
  //
  //      Ownership: only ksp==1 (warps >= NHALF) ever reads kpe -- of k-steps 18..35 only 32..35 are the
  //      kpe columns -- and warp w reads exactly tile (w & (NHALF-1)). So the CONSUMER copies, and it can
  //      issue here, before the barrier wait, overlapping the ckv TMA. Issuing from the producer instead
  //      would need a proxy fence plus an mbarrier handshake, because a cp.async is only ordered against
  //      its issuing thread's own reads and the tile is filled by a different warp.
  const uint32_t mybar = barK0 + 8 * (warp >> 1);
  const int khalf = warp & (NHALF - 1);
  if (lane == 0) mbar_expect_tx(mybar, (uint32_t)nvalid * (D * 2));
  __syncwarp();
  if (lane < 8) {
    if (idl >= 0) bulk_g2s(sbase + OFF_KC + (lrow + lane) * KC_STRIDE, ckv + (size_t)idl * D, D * 2, mybar);
  }
  if (warp >= NHALF) {
    // 16 rows x 128 B = 128 chunks of 16 B, 4 rounds of 32 lanes. The 16 indices are contiguous (one
    // 16-row run inside the 32-row block this CTA owns) and L1-hot -- the tile's producers read them in
    // Phase 0 -- so one __ldg per lane covers the tile and four shuffles broadcast them to the lanes.
    const int* cbase =
        sparse_idx + (size_t)t * TOPK + ((16 * khalf) / BLK) * BLK_STRIDE + r * BLK + ((16 * khalf) % BLK);
    const int cid = (lane < 16) ? __ldg(cbase + lane) : -1;
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
      const int row = 4 * rr + (lane >> 3);
      const int off = lane & 7;
      const int rl = __shfl_sync(0xffffffffu, cid, row);
      uint8_t* kpd = smem + OFF_KP + (16 * khalf + row) * KP_STRIDE + off * 16;
      // kpe is a bf16 pointer: address in ELEMENTS (rl * DP + off * 8), not bytes. kpd is uint8_t* so it
      // stays byte-addressed -- the two legs must not be written the same way.
      if (rl >= 0) cp_async16(smem_u32(kpd), kpe + (size_t)rl * DP + off * 8);
      // Invalid rows hold stale smem and Phase 3a reads whole 16-row tiles, so zero them.
      else *reinterpret_cast<uint4*>(kpd) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  // Invalid rows of this warp's own 8 hold stale smem, and PV reads whole 16-row tiles, so zero the ckv
  // half. They carry no copy, so this races with nothing. (kpe is the consumer's job, above.)
  if (lastv < ROWS_PER_WARP) {
#pragma unroll 1
    for (int rr = lrow + lastv; rr < lrow + ROWS_PER_WARP; ++rr) {
      uint4* kc = reinterpret_cast<uint4*>(smem + OFF_KC + rr * KC_STRIDE);
      kc[lane] = make_uint4(0u, 0u, 0u, 0u);
      kc[lane + 32] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  mbar_wait(barQ, 0);
  mbar_wait(barK0 + 8 * (warp & (NHALF - 1)), 0);  // only this warp's QK tile, not the whole CTA
  if (warp >= NHALF) {
    // The __syncwarp is load-bearing, not decoration. cp.async.wait_all only orders the chunks THAT
    // LANE issued against that same lane's later reads; Phase 3a's ldsm_x4 has every lane read the whole
    // 16x16 kpe tile, i.e. bytes copied by its neighbours. Without the syncwarp each lane can observe
    // its own chunk landed and still read a neighbour's unwritten bytes. (Observed: max_abs 3.1, 93% of
    // outputs wrong, and a slightly FASTER device time because the dependency was never enforced.)
    cp_async_wait_all();
    __syncwarp();
  }
  PROF_STAMP(2);

  const int arow = (lane & 7) + 8 * ((lane >> 3) & 1);
  const int akof = 8 * (lane >> 4);
  const int g = lane >> 2, q = lane & 3;
  // ---- Phase 3a: S = Q K^T. warp w -> k-steps [18*ksp, 18*ksp+18), rows [16*half, 16*half+16)
  {
    const int ksp = warp / NHALF, half = warp % NHALF;
    const int brow = (lane & 7) + 8 * (lane >> 4);   // ldmatrix x4 over 16 K rows x 16 k
    const int bkof = 8 * ((lane >> 3) & 1);
    float sacc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) sacc[i][0] = sacc[i][1] = sacc[i][2] = sacc[i][3] = 0.f;
    // Two-stage pipeline over the 18 k-steps was tried and rejected: 10.400 us vs the 10.379 baseline,
    // i.e. inside run-to-run noise. Two independent mmas per k-step already cover the ldmatrix latency.
#pragma unroll
    for (int j = 0; j < KPW; ++j) {
      const int kk = ksp * KPW + j;
      uint32_t a[4], b0, b1, b2, b3;
      uint32_t aaddr, baddr;
      if (kk < D / 16) {
        aaddr = sbase + OFF_QN + arow * KC_STRIDE + (kk * 16 + akof) * 2;
        baddr = sbase + OFF_KC + (half * 16 + brow) * KC_STRIDE + (kk * 16 + bkof) * 2;
      } else {
        const int kp = kk - D / 16;
        aaddr = sbase + OFF_QP + arow * KP_STRIDE + (kp * 16 + akof) * 2;
        baddr = sbase + OFF_KP + (half * 16 + brow) * KP_STRIDE + (kp * 16 + bkof) * 2;
      }
      ldsm_x4(aaddr, a[0], a[1], a[2], a[3]);
      ldsm_x4(baddr, b0, b1, b2, b3);
      mma_bf16(sacc[0], a, b0, b1);
      mma_bf16(sacc[1], a, b2, b3);
    }
    float* S_s = reinterpret_cast<float*>(smem + OFF_S) + ksp * H * S_STRIDE;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int col = half * 16 + nt * 8 + 2 * q;
      *reinterpret_cast<float2*>(S_s + g * S_STRIDE + col) = make_float2(sacc[nt][0], sacc[nt][1]);
      *reinterpret_cast<float2*>(S_s + (g + 8) * S_STRIDE + col) = make_float2(sacc[nt][2], sacc[nt][3]);
    }
  }
  __syncthreads();
  PROF_STAMP(7);
  // ---- Phase 3b: softmax (base 2), thread (h = warp, lane) owns 4 consecutive columns. Rows whose
  //      index is -1 are forced to -inf here, which also masks whatever the gather left in S.
  {
    const int h = warp, sub = lane;
    const float* S_s = reinterpret_cast<const float*>(smem + OFF_S) + h * S_STRIDE + sub * 4;
    float4 s4 = *reinterpret_cast<const float4*>(S_s);
#pragma unroll
    for (int ks = 1; ks < KSPLIT; ++ks) {
      const float4 x = *reinterpret_cast<const float4*>(S_s + ks * H * S_STRIDE);
      s4.x += x.x;
      s4.y += x.y;
      s4.z += x.z;
      s4.w += x.w;
    }
    const int4 id4 = *reinterpret_cast<const int4*>(idx_s + sub * 4);
    float v[4];
    v[0] = (id4.x >= 0) ? s4.x * scale_log2 : -INFINITY;
    v[1] = (id4.y >= 0) ? s4.y * scale_log2 : -INFINITY;
    v[2] = (id4.z >= 0) ? s4.z * scale_log2 : -INFINITY;
    v[3] = (id4.w >= 0) ? s4.w * scale_log2 : -INFINITY;
    float mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.f;
    float pf[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      pf[j] = __bfloat162float(__float2bfloat16(fast_exp2(v[j] - mx)));
      sum += pf[j];
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    uint2 pk;
    pk.x = pack_bf162(pf[0], pf[1]);
    pk.y = pack_bf162(pf[2], pf[3]);
    *reinterpret_cast<uint2*>(smem + OFF_P + h * P_STRIDE + sub * 8) = pk;
    if (sub == 0) {
      m_s[h] = mx;
      l_s[h] = sum;
    }
  }
  __syncthreads();
  PROF_STAMP(8);
  // ---- Phase 3c: O = P V  (warp w owns columns [32w, 32w+32))
  float oacc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
  {
    const int n0 = warp * 32;
#pragma unroll
    for (int ks = 0; ks < ROWS / 16; ++ks) {
      uint32_t a[4];
      ldsm_x4(sbase + OFF_P + arow * P_STRIDE + (ks * 16 + akof) * 2, a[0], a[1], a[2], a[3]);
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t b0, b1, b2, b3;
        const int vrow = ks * 16 + arow;
        const int vcol = n0 + np * 16 + akof;
        ldsm_x4_trans(sbase + OFF_KC + vrow * KC_STRIDE + vcol * 2, b0, b1, b2, b3);
        mma_bf16(oacc[2 * np], a, b0, b1);
        mma_bf16(oacc[2 * np + 1], a, b2, b3);
      }
    }
  }
  PROF_STAMP(3);
  // ---- Phase 4: partial O -> workspace as (bf16x2, epoch) words, then the (m, l) words. No fence:
  //      readers validate every word by its own flag. Lane pairs (q, q^1) swap halves so that even
  //      lanes store 4 consecutive columns of row g and odd lanes of row g+8 (one 16-B store each).
  //
  //      Every warp stores, and that is correct -- do NOT "optimise" this by having only warp r store.
  //      Warp w writes slot w, and reader r reads slot r from all 16 peers, so peer p must expose all
  //      16 slots for the 16 different readers. 16 publishers x 16 slots = 256 slot-writes against 256
  //      slot-reads: nothing here is redundant. Restricting the store to warp r writes 16 slots against
  //      the same 256 reads, and every cross-CTA read goes permanently stale (the retry loop spins
  //      forever -- this hang was observed, not theorised).
  {
    unsigned long long* O = ws_o + ((size_t)t * NCTA + r) * WS_O_STRIDE;
    const int n0 = warp * 32;
    const bool odd = q & 1;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      const uint32_t mine = odd ? pack_bf162(oacc[np][2], oacc[np][3]) : pack_bf162(oacc[np][0], oacc[np][1]);
      const uint32_t send = odd ? pack_bf162(oacc[np][0], oacc[np][1]) : pack_bf162(oacc[np][2], oacc[np][3]);
      const uint32_t recv = __shfl_xor_sync(0xffffffffu, send, 1);
      const int col = n0 + np * 8 + 4 * (q >> 1);           // first of my 4 columns
      const int widx = ((col / SLICE) * H + (odd ? (g + 8) : g)) * SW + ((col % SLICE) >> 1);
      const unsigned long long w0 = ll_pack_u(odd ? recv : mine, ep), w1 = ll_pack_u(odd ? mine : recv, ep);
      st_ll2(O + widx, w0, w1);
    }
  }
  __syncthreads();  // all O stores are queued before the (m, l) words that announce them
  if (tid < 16) {
    st_ll2(ws_ml + ((size_t)t * NCTA + r) * WS_ML_STRIDE + tid * 2, ll_pack(m_s[tid], ep), ll_pack(l_s[tid], ep));
  }
  PROF_STAMP(4);

  // ---- Phase 5+6: merge straight from L2. warp = head hh; lane -> column pair cp = lane & 15 (cols
  //      32 r + 2 cp, +1) and slots [8 sh, 8 sh + 8) with sh = lane >> 4; the two half-warps combine by
  //      one shuffle each. Every thread loads its 8 (m, l) pairs and 8 O words at once (no poll, no
  //      staging); words that do not carry this epoch yet (their store has not landed) are re-read.
  {
    const int hh = warp, cp = lane & 15, sh = lane >> 4;
    const unsigned long long* GO = ws_o + ((size_t)(t * NCTA + sh * 8)) * WS_O_STRIDE + (r * H + hh) * SW + cp;
    const unsigned long long* GML = ws_ml + (size_t)(t * NCTA + sh * 8) * WS_ML_STRIDE + hh * 2;
    unsigned long long wm[8], wl[8], wo[8];
    // No poll. This revision reads the payload and lets the retry loop below settle staleness -- every
    // word carries its own epoch flag, so an early read costs a retry, not a wrong answer. (Earlier
    // revisions polled the tags first and measured faster; that poll was removed as PROBE A and the
    // sequencing that replaced it is what the measurements below describe. The backpressure the poll
    // provided is now implicit: the payload reads are issued back-to-back and the retry loop only runs
    // for words that have not landed, which is the same serialisation without the extra pass.)
    //
    // The prefetch below is load-bearing and correctly sized. Verified 2026-09-27: deleting it costs
    // +1.408 us, cutting it from 16 slots to 2 costs +0.880 us (both bit-identical). It is NOT redundant
    // -- `hh` is multiplied by SW = 16 u64 = 128 B = one cache line, so each of the 16 warps reads a
    // different head's line, and the 16 iterations cover the 16 slots. One prefetch per line, no waste.
    // See PREFETCH_COVERAGE.md. Do not "de-duplicate" it.
    // Warm the 16 O lines this thread will need (they are 32 KB apart -- 16 distinct cold DRAM lines).
#pragma unroll
    // GO already carries the sh*8 slot offset, so this covers only this half-warp's 8 slots.
    // The retry loop for the OTHER half-warp's slots sits in a different 32 KB region and would
    // still be cold. Prefetch all 16 sibling slots explicitly.
    for (int i = 0; i < 16; ++i)
      pf_l2(ws_o + ((size_t)(t * NCTA + i)) * WS_O_STRIDE + ((size_t)r * H + hh) * SW + cp);
    // The poll only touches head 0s pair, i.e. line 0 (heads 0-7) of each slot. Warps 8-15 merge heads
    // 8-15, which live in line 1 and are therefore stone cold. Prefetch those too.
#pragma unroll
    for (int i = 0; i < 8; ++i) pf_l2(GML + (size_t)i * WS_ML_STRIDE);
    // PROBE A: poll removed. The retry loop below is self-validating, so correctness is unaffected.
    __syncthreads();
    PROF_STAMP(5);
#pragma unroll
    for (int i = 0; i < 8; ++i) ld_ll2(GML + (size_t)i * WS_ML_STRIDE, wm[i], wl[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) wo[i] = ld_ll(GO + (size_t)i * WS_O_STRIDE);
#pragma unroll 1
    for (int attempt = 0;; ++attempt) {
      uint32_t badm = 0u, bado = 0u;  // bit i: slot i's (m, l) / O word is stale
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const bool bm = ((ll_flag(wm[i]) ^ ep) | (ll_flag(wl[i]) ^ ep)) != 0u;
        // an empty slot never writes O: its O word is only needed if m is finite (or not known yet)
        const bool bo = (ll_flag(wo[i]) != ep) && (bm || ll_val(wm[i]) != -INFINITY);
        badm |= bm ? (1u << i) : 0u;
        bado |= bo ? (1u << i) : 0u;
      }
      if (!__any_sync(0xffffffffu, (badm | bado) != 0u)) break;
#ifdef PROF
      // slot 15, NOT 14: PROF_STAMP(14) above already owns slot 14 (the post-barrier-init sync),
      // and this counter was overwriting that stamp -- which is why naive slot[i+1]-slot[i] reads
      // produced negative "durations" (QK -4096). prof_verify.py has always read retry counts from
      // slot 15; this makes the kernel agree with the reader.
      if (lane == 0) atomicAdd(&g_prof[(blockIdx.y * gridDim.x + blockIdx.x) * PROF_N + 15], 1ull);
#endif
      if (attempt > 0) __nanosleep(128);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if ((badm >> i) & 1u) ld_ll2(GML + (size_t)i * WS_ML_STRIDE, wm[i], wl[i]);
        if ((bado >> i) & 1u) wo[i] = ld_ll(GO + (size_t)i * WS_O_STRIDE);
      }
    }
    PROF_STAMP(6);
    float Mh = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) Mh = fmaxf(Mh, ll_val(wm[i]));
    const float M = fmaxf(Mh, __shfl_xor_sync(0xffffffffu, Mh, 16));
    PROF_STAMP(9);
    float L = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float mi = ll_val(wm[i]);
      if (mi != -INFINITY) {
        const float w = fast_exp2(mi - M);
        const float2 o2 = unpack_bf162(ll_u(wo[i]));
        L = fmaf(w, ll_val(wl[i]), L);
        a0 = fmaf(w, o2.x, a0);
        a1 = fmaf(w, o2.y, a1);
      }
    }
    PROF_STAMP(10);
    L += __shfl_xor_sync(0xffffffffu, L, 16);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 16);
    a1 += __shfl_xor_sync(0xffffffffu, a1, 16);
    if (sh == 0) {
      const float inv = (L > 0.f) ? 1.f / L : 0.f;
      const int col = r * SLICE + 2 * cp;
      *reinterpret_cast<uint32_t*>(out + ((size_t)t * H + hh) * D + col) = pack_bf162(a0 * inv, a1 * inv);
      if (r == 0 && cp == 0) lse[t * H + hh] = (L > 0.f) ? M + log2f(L) : -INFINITY;
    }
  }
  PROF_STAMP(11);
}

}  // namespace

extern "C" int dsa_launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe,
                          const int* sparse_idx, float sm_scale, void* out, float* lse, void* ws_o,
                          void* ws_ml, int epoch, int T, cudaStream_t stream) {
  static bool attr_done = false;
  if (!attr_done) {
    cudaError_t e = cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (e != cudaSuccess) return (int)e;
    attr_done = true;
  }
  const float scale_log2 = sm_scale * 1.4426950408889634f;
  dsa_kernel<<<dim3(NCTA, T, 1), dim3(NTHREADS, 1, 1), SMEM_TOTAL, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope), reinterpret_cast<const __nv_bfloat16*>(q_pe),
      reinterpret_cast<const __nv_bfloat16*>(ckv), reinterpret_cast<const __nv_bfloat16*>(kpe), sparse_idx,
      scale_log2, reinterpret_cast<__nv_bfloat16*>(out), lse, reinterpret_cast<unsigned long long*>(ws_o),
      reinterpret_cast<unsigned long long*>(ws_ml), epoch);
  return (int)cudaGetLastError();
}

extern "C" int dsa_num_chunks() { return NCTA; }
extern "C" int dsa_smem_bytes() { return SMEM_TOTAL; }
extern "C" int dsa_prof_dump(unsigned long long* host, int n) {
#ifdef PROF
  return (int)cudaMemcpyFromSymbol(host, g_prof, (size_t)n * sizeof(unsigned long long));
#else
  return -1;
#endif
}
extern "C" int dsa_prof_clear() {
#ifdef PROF
  static unsigned long long z[sizeof(g_prof) / 8];
  return (int)cudaMemcpyToSymbol(g_prof, z, sizeof(g_prof));
#else
  return -1;
#endif
}
