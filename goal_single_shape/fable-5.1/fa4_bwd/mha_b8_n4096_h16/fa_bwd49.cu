// Causal MHA backward for B200 (sm_100a), v3: 2-CTA (cta_group::2) transposed formulation, P^T / dS^T resident in TMEM,
// S^T / dP^T computed in two q-halves, dQ lagging one tile so that the dS exchange never gates the tensor pipe.
// Layout (B, N, H, D) bf16, D = 128. A cluster of 2 CTAs owns 256 kv rows (CTA r = kv block 2m+r) of one (b, h) and walks
// the q tiles t = NT-1 .. 2m. Per tile the leader issues (cta_group::2, M = kv = 256 unless noted):
//   M1a/M1b: S^T(q half) = K Q^T   (fp32 -> S region cols 0..63 / 64..127)     M2a/M2b: dP^T(q half) = V dO^T (-> dP region)
//   M3: dV += P^T dO (A = P^T bf16, S region cols 0..63)   M4: dK += dS^T Q (A = dS^T bf16, dP region cols 0..63)
//   M5: dQ = dS K (M = q = 128 -> 64 rows per CTA folded into 64 columns; A = dS exchanged through DSMEM), issued one tile late
//       into dP region cols 64..127 (free once the right half's dS phase has read dP^T; dS^T is packed into cols 0..63:
//       slice cs at 32(cs&1) + 16(cs>>1)), so M1a/M1b(i+1) follow M3(i) immediately and never wait for the dQ drain.
// Issue order per tile i:  M3(i) M1a(i+1) M1b(i+1) M5(i-1) M4(i) M2a(i+1) M2b(i+1).
// Warpgroups: 0 = producer / issuer / peer forwarders, 1..4 = compute (q-column slice cs = wg-1; thread = kv row = TMEM lane),
// 5 = dQ drain (TMEM -> fp16 staging in the consumed dS exchange buffer -> TMA reduce-add into dq_accum (B,N,H,D) fp16).
#include <cstdio>
#include <type_traits>
#include "tc_common.cuh"

namespace {

constexpr int BR = 128, BC = 128, HD = 128;
constexpr int NUM_THREADS = 768;
// SMEM. K: O1 = own K d[0,64), O2 = own K d[64,128), O3 = K of block 2m d-half r, O4 = K of block 2m+1 d-half r (16KB each).
// Q_s half x (8KB): q rows [128t + 64x + 32r, +32), both d atoms (d[0,64) at +0, d[64,128) at +4096). Q_d: all q, d-half r (16KB).
constexpr uint32_t SK = 0, SV = 65536, SQSA = 98304, SQSB = 106496, SQD = 114688, SDOSA = 131072, SDOSB = 139264, SDOD = 147456,
                   SDSX0 = 163840, SDSX1 = 196608, SBAR = 229376, SLSE = 230400 /* lse[2][128] @+0, delta[2][128] @+1024 */;
constexpr uint32_t SMEM_BYTES = 232448;
enum { B_KV = 0, B_KV2, B_QSA, B_QSB, B_QD, B_DOSA, B_DOSB, B_DOD, B_LSE /*2*/, B_SFA = 10, B_SFB, B_DPFA, B_DPFB, B_DQF, B_DODDEAD,
       B_QDDEAD, B_KVDONE, B_KDEAD, B_DVKDONE, B_PF, B_DSF, B_DSX /*2*/, B_DSXP = 24 /*2*/, B_DQFREE = 26, B_EPIDONE,
       B_LSEFREE /*2*/, B_DLTFREE = 30 /*2*/, B_STAGERD = 32 /*2*/, B_STAGERDP = 34 /*2*/, B_DREADL = 36, B_DREADR, B_UNUSED38,
       B_KVP, B_KV2P, B_QSAP, B_QSBP, B_QDP, B_DOSAP, B_DOSBP, B_DODP, B_VDEAD, B_V, B_VP, B_EPISTORE, B_DVRD, B_DKRD, B_COUNT };
// TMEM columns.
constexpr uint32_t T_DV = 0, T_DK = 128, T_S = 256, T_D = 384, T_DQ = T_D + 64;

struct BwdParams {
  const float* lse;     // (B,H,N)
  const float* delta;   // (B,H,N)
  __nv_bfloat16* dk;
  __nv_bfloat16* dv;
  long long* dbg;
  long long* trace;
  int* dq_flag;         // (B,H,NT,2): 0 untouched, 1 first store in flight, 2 stored (DQ_FLAGS)
  int B, N, H;
  int num_clusters;
  float scale_log2e;
  float scale;
};

DEVI uint32_t pack_bf16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t pack_f16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
DEVI float exp2_poly3(float x) {   // 2^x on the FMA pipe (rel. err ~1e-4)
  x = fmaxf(x, -125.f);
  const float t = x + 12582912.0f;
  const float f = x - (t - 12582912.0f);
  float q = fmaf(f, 5.5504109e-2f, 2.4022651e-1f);
  q = fmaf(q, f, 6.9314718e-1f);
  q = fmaf(q, f, 1.0f);
  return __int_as_float(__float_as_int(q) + (__float_as_int(t) << 23));
}
DEVI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI void st_shared_v4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DEVI void st_async_v4(uint32_t raddr, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t rmbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];"
               :: "r"(raddr), "r"(a), "r"(b), "r"(c), "r"(d), "r"(rmbar) : "memory");
}
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEVI void st_global_v4(void* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DEVI void tma_reduce_add_4d(const CUtensorMap* map, uint32_t ssrc, int c0, int c1, int c2, int c3) {
  asm volatile("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
               :: "l"(map), "r"(ssrc), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tma_store_4d(const CUtensorMap* map, uint32_t ssrc, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
               :: "l"(map), "r"(ssrc), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void bulk_copy_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI int ld_acquire_gpu(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void st_release_gpu(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
DEVI void st_shared_release_u32(uint32_t addr, uint32_t v) { asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
DEVI uint32_t ld_shared_acquire_u32(uint32_t addr) { uint32_t v; asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory"); return v; }
DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N_> DEVI void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N_) : "memory"); }
template <int N_> DEVI void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N_) : "memory"); }
DEVI void named_bar_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory"); }
DEVI uint32_t cluster_ctarank() { uint32_t r; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI uint32_t mapa(uint32_t addr, uint32_t rank) { uint32_t r; asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r; }
DEVI void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI void warp_arrive(uint32_t mbar) { __syncwarp(); if ((threadIdx.x & 31) == 0) mbar_arrive(mbar); }
// Arrive on a barrier of the leader CTA (local plain arrive, or remote relaxed arrive from the peer).
DEVI void warp_arrive_leader(uint32_t mbar_local, uint32_t rank) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) { if (rank == 0) mbar_arrive(mbar_local); else mbar_arrive_remote_relaxed(mapa(mbar_local, 0)); }
}
#ifdef DBG_FWD_PF
#define ARRIVE_FWD(mb) warp_arrive(mb)                    // both CTAs arrive locally; the peer's forwarder relays one arrival
#else
#define ARRIVE_FWD(mb) warp_arrive_leader(mb, rank)
#endif

// ---- cta_group::2 tcgen05 ----
DEVI void tmem_alloc2(uint32_t smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_dst), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
DEVI void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void mma_ss2(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
               :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
DEVI void mma_ts2(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, {%5,%6,%7,%8,%9,%10,%11,%12}, p;\n}"
               :: "r"(d), "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}
DEVI void mma_commit2_mc(uint32_t mbar) {   // arrives on the barrier at the same offset in both CTAs
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mbar), "h"((uint16_t)3) : "memory");
}
DEVI void tmem_ld16_sync(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               "tcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEVI void tmem_st16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
               :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr) : "memory");
}
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// Descriptors (128B-swizzled TMA tiles). K-major (d contiguous): LBO 16, SBO 1024; k-step offsets in 16B units:
//   128-row tiles: (kk>>2)*1024 + (kk&3)*2 (second d atom at +16384); 32-row tiles: (kk>>2)*256 + (kk&3)*2 (atom at +4096).
// MN-major B (N = d contiguous, one 64-wide atom, K = rows): SBO 1024; k-step kk*128 (16 rows = 2048B).
DEVI uint64_t desc_k(uint32_t tile) { return make_smem_desc(tile, 16, 1024, 2); }
DEVI uint64_t desc_mn(uint32_t tile) { return make_smem_desc(tile, 16384, 1024, 2); }
DEVI constexpr uint64_t koff128(int kk) { return (uint64_t)((kk >> 2) * 1024 + (kk & 3) * 2); }
DEVI constexpr uint64_t koff32(int kk) { return (uint64_t)((kk >> 2) * 256 + (kk & 3) * 2); }
DEVI constexpr uint64_t koff64(int kk) { return (uint64_t)((kk >> 2) * 512 + (kk & 3) * 2); }   // 64-row tiles: second atom at +8192
DEVI constexpr uint64_t koffmn(int kk) { return (uint64_t)(kk * 128); }
// dS exchange buffer (A of M5, MN-major, no swizzle): element (q_local, kv) at (kv/8)*1024 + (q_local/8)*128 + (kv%8)*16 + (q_local%8)*2.
DEVI uint64_t desc_dsx(uint32_t tile) { return make_smem_desc(tile, 1024, 128, 0); }

#ifdef DEBUG_HANG
#define MBAR_WAIT(idx, par) do { uint32_t _n = 0; while (!mbar_test_wait(bar(idx), (par))) { if (++_n == (1u << 24)) { \
    if ((threadIdx.x & 31) == 0) printf("HANG blk %d warp %d bar %d par %u tw0 %d tw1 %d dbg %lld %lld %lld %lld %lld %lld %lld %lld iss %lld m2 dqf %lld k %lld stg %lld\n", blockIdx.x, threadIdx.x / 32, (int)(idx), (unsigned)(par), \
        (int)mbar_test_wait(bar(idx), 0), (int)mbar_test_wait(bar(idx), 1), p.dbg[0], p.dbg[1], p.dbg[2], p.dbg[3], p.dbg[4], p.dbg[5], p.dbg[6], p.dbg[7], p.dbg[8], p.dbg[9], p.dbg[10], p.dbg[11]); } \
    if (_n > (1u << 27)) __trap(); } } while (0)
#elif defined(WAIT_HINT)
#define MBAR_WAIT(idx, par) mbar_wait_hint<WAIT_HINT>(bar(idx), (par))
#else
#define MBAR_WAIT(idx, par) mbar_wait(bar(idx), (par))
#endif
#ifdef DEBUG_TRACE
#define TRACE_G0 200
#define TRACE_N 8
#define TSTAMPG(base, gg, e) do { if (blockIdx.x < 2 && (int)(gg) >= TRACE_G0 && (int)(gg) < TRACE_G0 + TRACE_N) { unsigned long long _g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_g)); p.trace[(base) + blockIdx.x * 64 + ((gg) - TRACE_G0) * 8 + (e)] = (long long)_g; } } while (0)
#define TSTAMPW(gg, e) do { if (blockIdx.x == 0 && lane == 0 && (int)(gg) >= TRACE_G0 && (int)(gg) < TRACE_G0 + TRACE_N) p.trace[1024 + (warp - 4) * 64 + ((gg) - TRACE_G0) * 8 + (e)] = clock64(); } while (0)
#define TSTAMP(base, gg, e) do { if (blockIdx.x == 0 && (int)(gg) >= TRACE_G0 && (int)(gg) < TRACE_G0 + TRACE_N) p.trace[(base) + ((gg) - TRACE_G0) * 16 + (e)] = clock64(); } while (0)
#else
#define TSTAMP(base, gg, e) ((void)0)
#define TSTAMPG(base, gg, e) ((void)0)
#define TSTAMPW(gg, e) ((void)0)
#endif
#if defined(DBG_CSPIN)
#define MBAR_WAIT_HOT(idx, par) mbar_wait_spin(bar(idx), (par))
#elif defined(DBG_CHINT)
#define MBAR_WAIT_HOT(idx, par) mbar_wait_hint<DBG_CHINT>(bar(idx), (par))
#elif defined(DBG_CNS)
#define MBAR_WAIT_HOT(idx, par) do { while (!mbar_test_wait(bar(idx), (par))) __nanosleep(DBG_CNS); } while (0)
#else
#define MBAR_WAIT_HOT(idx, par) MBAR_WAIT(idx, par)
#endif
#ifdef DEBUG_TIMING
#define TW(k, expr) do { long long _t0 = clock64(); expr; tw[k] += clock64() - _t0; } while (0)
#else
#define TW(k, expr) do { expr; } while (0)   // braces: TW(i, A; B) after an if must keep both statements conditional
#endif

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NUM_THREADS, 1)
fa_bwd49_kernel(const __grid_constant__ CUtensorMap mapQ32, const __grid_constant__ CUtensorMap mapQ128,
               const __grid_constant__ CUtensorMap mapK, const __grid_constant__ CUtensorMap mapV,
               const __grid_constant__ CUtensorMap mapDO32, const __grid_constant__ CUtensorMap mapDO128,
               const __grid_constant__ CUtensorMap mapDQ, const __grid_constant__ CUtensorMap mapDV,
               const __grid_constant__ CUtensorMap mapDK, BwdParams p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t smem_base = smem_u32(smem_raw);
  if (smem_base & 1023u) { if (threadIdx.x == 0) printf("fa_bwd49: dynamic smem not 1024B aligned (%u)\n", smem_base); __trap(); }
  const uint32_t bar_base = smem_base + SBAR;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem_raw + SBAR + 8 * B_COUNT);

  const int NT = p.N / BR;
  const int BH = p.B * p.H;
  const int cluster_id = blockIdx.x >> 1;
  const uint32_t rank = cluster_ctarank();
  const uint32_t peer = rank ^ 1u;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wg = warp / 4;
  auto bar = [&](int idx) { return bar_base + idx * 8; };

  if (warp == 0) tmem_alloc2(smem_u32(tmem_slot), 512);
  if (threadIdx.x == 0) {
    for (int i = 0; i < B_COUNT; ++i) {
      uint32_t cnt = 1;
#ifdef DBG_FWD_PF
      if (i == B_PF || i == B_DSF) cnt = (rank == 0) ? 17 : 16;              // leader: 16 local + 1 forwarded
      if (i == B_EPIDONE) cnt = 32;
#else
      if (i == B_PF || i == B_DSF || i == B_EPIDONE) cnt = 32;               // 16 compute warps of each CTA
#endif
      if (i == B_DQFREE) cnt = 8;                                             // 4 drain warps of each CTA
      if (i == B_EPISTORE) cnt = 4;                                           // dV / dK storing threads of both CTAs
      if (i == B_DVRD || i == B_DKRD) cnt = 16;                               // the 8 dV (dK) warps of each CTA (leader barrier)
      if (i == B_DSX || i == B_DSX + 1) cnt = 9;                              // expect_tx thread + my 8 local-half writer warps
      if (i == B_LSEFREE || i == B_LSEFREE + 1 || i == B_DLTFREE || i == B_DLTFREE + 1) cnt = 16;
      if (i == B_DREADL) cnt = 8;                                             // my 8 left-half warps (local)
      if (i == B_DREADR) cnt = 16;                                            // right-half warps of both CTAs (leader barrier)
      mbar_init(bar(i), cnt);
    }
    mbar_fence_init();
    *reinterpret_cast<volatile uint32_t*>(smem_raw + SBAR + 960) = 0u;   // drain progress: number of tiles whose staging read is done
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = __shfl_sync(0xffffffffu, *tmem_slot, 0);

  const int M = NT / 2, PP = (M + 1) / 2;
  const int total_pairs = BH * PP;
  auto pair_items = [&](int pi, int& bh, int& m0, int& m1) { bh = pi / PP; m0 = pi % PP; m1 = M - 1 - m0; };
#define FOR_EACH_ITEM(bh, m, ntiles) \
  for (int pi = cluster_id; pi < total_pairs; pi += p.num_clusters) \
    for (int half = 0, m0_, m1_, bh; half < 2; ++half) \
      if (pair_items(pi, bh, m0_, m1_), (half == 0 || m1_ != m0_)) \
        if (int m = half ? m1_ : m0_, ntiles = NT - 2 * m; true)

  if (wg == 0) {
    if (warp == 0 && lane == 0) {
      // ================= TMA producer =================
      tma_prefetch_desc(&mapQ32); tma_prefetch_desc(&mapQ128); tma_prefetch_desc(&mapK); tma_prefetch_desc(&mapV);
      tma_prefetch_desc(&mapDO32); tma_prefetch_desc(&mapDO128); tma_prefetch_desc(&mapDQ); tma_prefetch_desc(&mapDV); tma_prefetch_desc(&mapDK);
      uint32_t nitem = 0, g = 0;
      bool next_preloaded = false;
      const int r64 = 64 * (int)rank, r32 = 32 * (int)rank;
      FOR_EACH_ITEM(bh, m, ntiles) {
        const int b = bh / p.H, h = bh % p.H;
        const int j = 2 * m + (int)rank;
        auto tile_t = [&](int i) { return NT - 1 - i; };
        auto load_lse = [&](int i) {
          const uint32_t k = g + i, bq = k & 1;
          if (k >= 2) MBAR_WAIT(B_LSEFREE + bq, ((k - 2) >> 1) & 1);
          mbar_arrive_expect_tx(bar(B_LSE + bq), 512);
          bulk_copy_g2s(smem_base + SLSE + bq * 512, p.lse + ((size_t)bh * p.N + (size_t)tile_t(i) * BR), 512, bar(B_LSE + bq));
        };
        auto load_qs = [&](int i, int x) {   // Q rows [128t + 64x + 32r, +32), both d atoms
          const uint32_t k = g + i; const int t = tile_t(i);
          if (k > 0) MBAR_WAIT(x ? B_SFB : B_SFA, (k - 1) & 1);
          const uint32_t sq = smem_base + (x ? SQSB : SQSA), mb = bar(x ? B_QSB : B_QSA);
#ifdef DBG_NO_TMA_QDO
          mbar_arrive(mb); (void)sq; return;
#endif
          mbar_arrive_expect_tx(mb, 8192);
          tma_load_4d(sq, &mapQ32, 0, h, t * BR + 64 * x + r32, b, mb);
          tma_load_4d(sq + 4096, &mapQ32, 64, h, t * BR + 64 * x + r32, b, mb);
        };
        auto load_dos = [&](int i) {   // dO rows [128t + 64r, +64): atom d[0,64) at +0, atom d[64,128) at +8192, plus the delta row
          const uint32_t k = g + i; const int t = tile_t(i);
          if (k > 0) MBAR_WAIT(B_DPFA, (k - 1) & 1);
          const uint32_t sd = smem_base + SDOSA, mb = bar(B_DOSA);
          if (k >= 2) MBAR_WAIT(B_DLTFREE + (k & 1), ((k - 2) >> 1) & 1);
          mbar_arrive_expect_tx(mb, 16384 + 512);
          bulk_copy_g2s(smem_base + SLSE + 1024 + (k & 1) * 512, p.delta + ((size_t)bh * p.N + (size_t)t * BR), 512, mb);
          tma_load_4d(sd, &mapDO32, 0, h, t * BR + r64, b, mb);
          tma_load_4d(sd + 4096, &mapDO32, 0, h, t * BR + r64 + 32, b, mb);
          tma_load_4d(sd + 8192, &mapDO32, 64, h, t * BR + r64, b, mb);
          tma_load_4d(sd + 12288, &mapDO32, 64, h, t * BR + r64 + 32, b, mb);
        };
        auto load_dod = [&](int i) {
          const uint32_t k = g + i;
          if (k > 0) MBAR_WAIT(B_DODDEAD, (k - 1) & 1);
#ifdef DBG_NO_TMA_QDO
          mbar_arrive(bar(B_DOD)); return;
#endif
          mbar_arrive_expect_tx(bar(B_DOD), 16384);
          tma_load_4d(smem_base + SDOD, &mapDO128, r64, h, tile_t(i) * BR, b, bar(B_DOD));
        };
        auto load_qd = [&](int i) {
          const uint32_t k = g + i;
          if (k > 0) MBAR_WAIT(B_QDDEAD, (k - 1) & 1);
#ifdef DBG_NO_TMA_QDO
          mbar_arrive(bar(B_QD)); return;
#endif
          mbar_arrive_expect_tx(bar(B_QD), 16384);
          tma_load_4d(smem_base + SQD, &mapQ128, r64, h, tile_t(i) * BR, b, bar(B_QD));
        };
        // own K (O1, O2): free once M1b of the previous item's last tile is done; normally already issued from the previous item's
        // last producer iteration (ahead of that item's late Q_d load) together with lse(0) / Q_s(0)
        if (!next_preloaded) {
          if (nitem > 0) MBAR_WAIT(B_KDEAD, (nitem - 1) & 1);
          mbar_arrive_expect_tx(bar(B_KV), 2 * 16384);
          tma_load_4d(smem_base + SK, &mapK, 0, h, j * BC, b, bar(B_KV));
          tma_load_4d(smem_base + SK + 16384, &mapK, 64, h, j * BC, b, bar(B_KV));
        }
        // next item (bh, m) for the early loads at the end of this item
        int nbh = -1, nm = 0;
        if (half == 0 && m1_ != m0_) { nbh = bh; nm = m1_; }
        else if (pi + p.num_clusters < total_pairs) { int b_; pair_items(pi + p.num_clusters, nbh, nm, b_); }
        auto load_next_head = [&]() {   // K, lse(0), Q_s(0) of the next item: their buffers free at KDEAD / LSEFREE / SF(last)
          if (nbh < 0) return;
          const int nb = nbh / p.H, nh = nbh % p.H, nj = 2 * nm + (int)rank;
          const uint32_t kn = g + ntiles;   // tile count of the next item's tile 0
          MBAR_WAIT(B_KDEAD, (nitem - 1) & 1);   // nitem was incremented for this item
          mbar_arrive_expect_tx(bar(B_KV), 2 * 16384);
          tma_load_4d(smem_base + SK, &mapK, 0, nh, nj * BC, nb, bar(B_KV));
          tma_load_4d(smem_base + SK + 16384, &mapK, 64, nh, nj * BC, nb, bar(B_KV));
          const int t0 = NT - 1;
          if (kn >= 2) MBAR_WAIT(B_LSEFREE + (kn & 1), ((kn - 2) >> 1) & 1);
          mbar_arrive_expect_tx(bar(B_LSE + (kn & 1)), 512);
          bulk_copy_g2s(smem_base + SLSE + (kn & 1) * 512, p.lse + ((size_t)nbh * p.N + (size_t)t0 * BR), 512, bar(B_LSE + (kn & 1)));
          #pragma unroll
          for (int x = 0; x < 2; ++x) {
            MBAR_WAIT(x ? B_SFB : B_SFA, (kn - 1) & 1);
            const uint32_t sq = smem_base + (x ? SQSB : SQSA), mb = bar(x ? B_QSB : B_QSA);
            mbar_arrive_expect_tx(mb, 8192);
            tma_load_4d(sq, &mapQ32, 0, nh, t0 * BR + 64 * x + r32, nb, mb);
            tma_load_4d(sq + 4096, &mapQ32, 64, nh, t0 * BR + 64 * x + r32, nb, mb);
          }
          next_preloaded = true;
        };
        auto prefetch = [&](int i) {
          if (i >= ntiles) return;
          const int t = tile_t(i);
          tma_prefetch_4d(&mapQ32, 0, h, t * BR + r32, b); tma_prefetch_4d(&mapQ32, 64, h, t * BR + r32, b);
          tma_prefetch_4d(&mapQ32, 0, h, t * BR + 64 + r32, b); tma_prefetch_4d(&mapQ32, 64, h, t * BR + 64 + r32, b);
          tma_prefetch_4d(&mapQ128, r64, h, t * BR, b);
          tma_prefetch_4d(&mapDO32, 0, h, t * BR + r32, b); tma_prefetch_4d(&mapDO32, 64, h, t * BR + r32, b);
          tma_prefetch_4d(&mapDO32, 0, h, t * BR + 64 + r32, b); tma_prefetch_4d(&mapDO32, 64, h, t * BR + 64 + r32, b);
          tma_prefetch_4d(&mapDO128, r64, h, t * BR, b);
        };
        prefetch(1); prefetch(2);
        if (!next_preloaded) { load_lse(0); load_qs(0, 0); load_qs(0, 1); }   // (else already issued at the end of the previous item)
        next_preloaded = false;
        // V: free once M2 of the previous item's last tile is done
        if (nitem > 0) MBAR_WAIT(B_VDEAD, (nitem - 1) & 1);
        mbar_arrive_expect_tx(bar(B_V), 2 * 16384);
        tma_load_4d(smem_base + SV, &mapV, 0, h, j * BC, b, bar(B_V));
        tma_load_4d(smem_base + SV + 16384, &mapV, 64, h, j * BC, b, bar(B_V));
        load_dos(0); load_dod(0);
        if (nitem > 0) MBAR_WAIT(B_KVDONE, (nitem - 1) & 1);   // O3, O4: free once M5 of the previous item's last tile is done
        nitem++;
        if (ntiles == 1) load_next_head();
        load_qd(0);
        mbar_arrive_expect_tx(bar(B_KV2), 2 * 16384);
        tma_load_4d(smem_base + SK + 32768, &mapK, r64, h, (2 * m) * BC, b, bar(B_KV2));
        tma_load_4d(smem_base + SK + 49152, &mapK, r64, h, (2 * m + 1) * BC, b, bar(B_KV2));
        if (ntiles > 1) { load_lse(1); load_qs(1, 0); load_qs(1, 1); }
        for (int i = 1; i < ntiles; ++i) {
          prefetch(i + 2);
          if (i + 1 < ntiles) load_lse(i + 1);
          load_dos(i);                          // after M2(i-1)
          load_dod(i);                          // after M3(i-1)
          if (i + 1 < ntiles) { load_qs(i + 1, 0); load_qs(i + 1, 1); }   // after M1a/M1b(i)
          if (i + 1 == ntiles) load_next_head();   // next item's K / lse / Q_s ahead of this late Q_d load
          load_qd(i);                           // after M4(i-1)
        }
        g += ntiles;
      }
    } else if (warp == 1 && rank == 0 && elect_one_sync()) {
      // ================= MMA issuer (leader CTA, single thread) =================
      const uint64_t dK1 = desc_k(smem_base + SK), dV1 = desc_k(smem_base + SV);
      const uint64_t dQsa = desc_k(smem_base + SQSA), dQsb = desc_k(smem_base + SQSB), dDOsa = desc_k(smem_base + SDOSA), dDOsb = desc_k(smem_base + SDOSB);
      const uint64_t dQd = desc_mn(smem_base + SQD), dDOd = desc_mn(smem_base + SDOD), dK34 = desc_mn(smem_base + SK + 32768);
      const uint64_t dDSX0 = desc_dsx(smem_base + SDSX0), dDSX1 = desc_dsx(smem_base + SDSX1);
      const uint32_t id_h = make_idesc(256, 64, 0, 0, 1, 1, 1);    // M1x, M2x: A K-major (SMEM), B K-major, N = 64
      const uint32_t id_ts = make_idesc(256, 128, 0, 1, 1, 1, 1);  // M3, M4: A K-major (TMEM), B MN-major
      const uint32_t id_dq = make_idesc(128, 128, 1, 1, 1, 1, 1);  // M5: A MN-major (SMEM), B MN-major
      uint32_t g = 0, nitem = 0;
      bool pre_issued = false;   // M1a/M1b(0) of the current item were issued at the end of the previous one
#ifdef DEBUG_TIMING
      long long tw[16] = {0}; const long long t_start = clock64();
#endif
      auto m1 = [&](uint32_t k, int x) {
        TW(0, MBAR_WAIT(x ? B_QSB : B_QSA, k & 1); MBAR_WAIT(x ? B_QSBP : B_QSAP, k & 1));
        tc_fence_after();
        const uint64_t db = x ? dQsb : dQsa;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) mma_ss2(tmem + T_S + 64 * x, dK1 + koff128(kk), db + koff32(kk), id_h, kk > 0 ? 1u : 0u);
        mma_commit2_mc(bar(x ? B_SFB : B_SFA));
      };
      // full-width dP^T (N = 128, one MMA): V is read from SMEM once per tile
      const uint32_t id_f = make_idesc(256, 128, 0, 0, 1, 1, 1);
      auto m2 = [&](uint32_t k, int dqf) {   // dqf >= 0: dQ(dqf) occupying dP region columns 64..127 must be drained first
        TW(1, MBAR_WAIT(B_DOSA, k & 1); MBAR_WAIT(B_DOSAP, k & 1));
        if (dqf >= 0) TW(2, MBAR_WAIT(B_DQFREE, dqf & 1));
        tc_fence_after();
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) mma_ss2(tmem + T_D, dV1 + koff128(kk), dDOsa + koff64(kk), id_f, kk > 0 ? 1u : 0u);
        mma_commit2_mc(bar(B_DPFA));
      };
      auto m5 = [&](uint32_t k, uint32_t kread) {   // dQ(k) = dS(k) K -> dP region columns 64..127, once dP^T(kread)'s right half is read
        const uint32_t bx = k & 1, ph = (k >> 1) & 1;
        TW(7, MBAR_WAIT(B_DSX + bx, ph); MBAR_WAIT(B_DSXP + bx, ph));
        TW(9, MBAR_WAIT(B_DREADR, kread & 1));
        tc_fence_after();
        const uint64_t da = bx ? dDSX1 : dDSX0;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) mma_ss2(tmem + T_DQ, da + (uint64_t)kk * 128, dK34 + koffmn(kk), id_dq, kk > 0 ? 1u : 0u);
        mma_commit2_mc(bar(B_DQF));
      };
      FOR_EACH_ITEM(bh, m, ntiles) {
        (void)bh; (void)m;
        nitem++;
        int nbh = -1, nm = 0;
        if (half == 0 && m1_ != m0_) { nbh = bh; nm = m1_; }
        else if (pi + p.num_clusters < total_pairs) { int b_; pair_items(pi + p.num_clusters, nbh, nm, b_); }
        const int next_ntiles = (nbh >= 0) ? NT - 2 * nm : 0;
        if (!pre_issued) {
          TW(8, MBAR_WAIT(B_KV, (nitem - 1) & 1); MBAR_WAIT(B_KVP, (nitem - 1) & 1));
          m1(g, 0); m1(g, 1);
          if (ntiles == 1) mma_commit2_mc(bar(B_KDEAD));
        }
        pre_issued = false;
        TW(8, MBAR_WAIT(B_V, (nitem - 1) & 1); MBAR_WAIT(B_VP, (nitem - 1) & 1));
        m2(g, g > 0 ? (int)(g - 1) : -1);   // M2: dQ of the previous item's last tile must be out
        if (ntiles == 1) mma_commit2_mc(bar(B_VDEAD));
        for (int i = 0; i < ntiles; ++i) {
          const uint32_t k = g + i;
          TSTAMP(0, k, 0);
#ifdef DEBUG_HANG
          if (blockIdx.x == 2) p.dbg[8] = k;
#endif
          // M3(i): dV += P^T dO
          TW(3, MBAR_WAIT(B_DOD, k & 1)); TW(12, MBAR_WAIT(B_DODP, k & 1));
          if (i == 0) TW(10, if (nitem > 1) MBAR_WAIT(B_DVRD, (nitem - 2) & 1));   // dV of the previous item read out
          TW(4, MBAR_WAIT(B_PF, k & 1));
          TSTAMPG(896, k, 4);   // leader: PF complete
          tc_fence_after();
          #pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ts2(tmem + T_DV, tmem + T_S + 32 * (kk >> 1) + 8 * (kk & 1), dDOd + koffmn(kk), id_ts, (i > 0 || kk > 0) ? 1u : 0u);
          mma_commit2_mc(bar(B_DODDEAD));
          TSTAMP(0, k, 1);
          // M1a/M1b(i+1): the S region is free (P^T(i) consumed by M3(i), in order)
          if (i + 1 < ntiles) { m1(k + 1, 0); TSTAMP(0, k, 3); m1(k + 1, 1); if (i + 2 == ntiles) mma_commit2_mc(bar(B_KDEAD)); }
          else if (nbh >= 0) {   // the next item's tile 0 (its K / Q_s were loaded ahead of this item's last Q_d load)
            TW(8, MBAR_WAIT(B_KV, nitem & 1); MBAR_WAIT(B_KVP, nitem & 1));
            m1(k + 1, 0); m1(k + 1, 1);
            if (next_ntiles == 1) mma_commit2_mc(bar(B_KDEAD));
            pre_issued = true;
          }
          TSTAMP(0, k, 4);
          // M5(i-1): dQ of the previous tile -> dP region columns 64..127 (dP^T(i) right half read by both CTAs)
          if (i == 1) TW(8, MBAR_WAIT(B_KV2, (nitem - 1) & 1); MBAR_WAIT(B_KV2P, (nitem - 1) & 1));
          if (i > 0) m5(k - 1, k);
          TSTAMP(0, k, 2);
          // M4(i): dK += dS^T Q (A = dS^T packed into dP region columns 0..63: slice cs at 32(cs&1) + 16(cs>>1))
          TW(5, MBAR_WAIT(B_QD, k & 1); MBAR_WAIT(B_QDP, k & 1));
          TSTAMP(0, k, 5);
          TW(6, MBAR_WAIT(B_DSF, k & 1));
          if (i == 0) TW(10, if (nitem > 1) MBAR_WAIT(B_DKRD, (nitem - 2) & 1));   // dK of the previous item read out
          TSTAMPG(896, k, 5);   // leader: DSF complete
          tc_fence_after();
          #pragma unroll
          for (int kk = 0; kk < 8; ++kk) mma_ts2(tmem + T_DK, tmem + T_D + 32 * ((kk >> 1) & 1) + 16 * (kk >> 2) + 8 * (kk & 1), dQd + koffmn(kk), id_ts, (i > 0 || kk > 0) ? 1u : 0u);
          mma_commit2_mc(bar(B_QDDEAD));
          if (i == ntiles - 1) mma_commit2_mc(bar(B_DVKDONE));
          TSTAMP(0, k, 6);
          // M2a/M2b(i+1): M2b overwrites dQ(i-1), which must be drained
          if (i + 1 < ntiles) { m2(k + 1, i > 0 ? (int)(k - 1) : -1); if (i + 2 == ntiles) mma_commit2_mc(bar(B_VDEAD)); }
          TSTAMP(0, k, 7);
        }
        // flush: M5 of the item's last tile (K of this item is needed until here)
        if (ntiles == 1) TW(8, MBAR_WAIT(B_KV2, (nitem - 1) & 1); MBAR_WAIT(B_KV2P, (nitem - 1) & 1));
        if (ntiles >= 2) TW(2, MBAR_WAIT(B_DQFREE, (g + ntiles - 2) & 1));   // dQ(last-1) still occupies the dQ columns until drained
        m5(g + ntiles - 1, g + ntiles - 1);
        mma_commit2_mc(bar(B_KVDONE));
        g += ntiles;
      }
#ifdef DEBUG_TIMING
      if (blockIdx.x == 0) { tw[9] = clock64() - t_start; tw[13] = g; for (int k = 0; k < 16; ++k) p.dbg[k] = tw[k]; }
#endif
#ifdef DEBUG_TRACE
    } else if (rank == 0 && warp == 2 && lane == 0 && blockIdx.x == 0) {
      const int ids[10] = {B_SFA, B_SFB, B_DPFA, B_DPFB, B_PF, B_DSF, B_DQF, B_DQFREE, B_DSX, B_DSX + 1};
      uint32_t cnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      const long long t_end = clock64() + (1ll << 27);
      while (clock64() < t_end) {
        bool all_done = true;
        #pragma unroll
        for (int q = 0; q < 10; ++q) {
          const uint32_t lim = (q >= 8) ? (uint32_t)((TRACE_G0 + TRACE_N + 1) / 2) : (uint32_t)(TRACE_G0 + TRACE_N);
          if (cnt[q] < lim) { all_done = false; if (mbar_test_wait(bar(ids[q]), cnt[q] & 1)) {
              const int tile = (q >= 8) ? (int)(cnt[q] * 2 + (q - 8)) : (int)cnt[q];
              if (tile >= TRACE_G0 && tile < TRACE_G0 + TRACE_N) p.trace[256 + (tile - TRACE_G0) * 16 + (q >= 8 ? 8 : q)] = clock64();
              cnt[q]++; } }
        }
        if (all_done) break;
      }
#endif
#ifdef DBG_FWD_PF
    } else if (rank == 1 && (warp == 1 || warp == 2 || warp == 3) && (warp == 3 ? (lane < 2) : elect_one_sync())) {
#else
    } else if (rank == 1 && (warp == 1 || warp == 2 || warp == 3) && elect_one_sync()) {
#endif
      // ================= peer forwarders: my TMA loads / dS exchange buffer complete -> leader's *P barriers =================
      const uint32_t r_kvp = mapa(bar(B_KVP), 0), r_kv2p = mapa(bar(B_KV2P), 0), r_qsap = mapa(bar(B_QSAP), 0), r_qsbp = mapa(bar(B_QSBP), 0);
      const uint32_t r_qdp = mapa(bar(B_QDP), 0), r_dosap = mapa(bar(B_DOSAP), 0), r_dosbp = mapa(bar(B_DOSBP), 0), r_dodp = mapa(bar(B_DODP), 0);
      const uint32_t r_dsxp0 = mapa(bar(B_DSXP), 0), r_dsxp1 = mapa(bar(B_DSXP + 1), 0), r_vp = mapa(bar(B_VP), 0);
      uint32_t g = 0, nitem = 0;
      FOR_EACH_ITEM(bh, m, ntiles) {
        (void)bh; (void)m;
        if (warp == 1) { MBAR_WAIT(B_KV, nitem & 1); mbar_arrive_remote_relaxed(r_kvp); }
        for (int i = 0; i < ntiles; ++i) {
          const uint32_t k = g + i, par = k & 1;
          if (warp == 1) {
            MBAR_WAIT(B_QSA, par); mbar_arrive_remote_relaxed(r_qsap);
            MBAR_WAIT(B_QSB, par); mbar_arrive_remote_relaxed(r_qsbp);
            MBAR_WAIT(B_DOD, par); mbar_arrive_remote_relaxed(r_dodp);
            if (i == 0) { MBAR_WAIT(B_KV2, nitem & 1); mbar_arrive_remote_relaxed(r_kv2p); }
          } else if (warp == 2) {
            if (i == 0) { MBAR_WAIT(B_V, nitem & 1); mbar_arrive_remote_relaxed(r_vp); }
            MBAR_WAIT(B_DOSA, par); mbar_arrive_remote_relaxed(r_dosap);
            MBAR_WAIT(B_QD, par); mbar_arrive_remote_relaxed(r_qdp);
#ifdef DBG_FWD_PF
          } else if (warp == 3 && lane == 0) {
            MBAR_WAIT(B_SREADR, par); mbar_arrive_remote_relaxed(mapa(bar(B_SREADR), 0));
            MBAR_WAIT(B_PF, par); mbar_arrive_remote_relaxed(mapa(bar(B_PF), 0));
            MBAR_WAIT(B_DSF, par); mbar_arrive_remote_relaxed(mapa(bar(B_DSF), 0));
#endif
          } else {   // warp 3 (lane 1 under DBG_FWD_PF): dS exchange
            MBAR_WAIT(B_DSX + (k & 1), (k >> 1) & 1); mbar_arrive_remote_relaxed((k & 1) ? r_dsxp1 : r_dsxp0);
          }
        }
        nitem++;
        g += ntiles;
      }
    }
  } else if (wg == 5) {
    // ================= dQ drain warpgroup: TMEM -> fp16 staging (in the consumed dS exchange buffer) -> TMA reduce-add =================
    const int rq = warp % 4;
    const uint32_t lane_off = (uint32_t)(rq * 32) << 16;
    const int tid = threadIdx.x - 640;
    // lanes l < 64 hold q row l, d in [0,64); lanes >= 64 hold q row l-64, d in [64,128) (folded into 64 columns)
    const int dq_row = 32 * (rq & 1) + lane, dq_half = rq >> 1;
    const uint32_t r_stagerdp0 = mapa(bar(B_STAGERDP), peer), r_stagerdp1 = mapa(bar(B_STAGERDP + 1), peer);
    uint32_t g = 0;
#ifdef DQ_FLAGS
    int* fl_pending = nullptr;   // flag of a first store whose completion has not been published yet (tid 0)
#endif
    FOR_EACH_ITEM(bh, m, ntiles) {
      const int b = bh / p.H, h = bh % p.H;
      (void)m;
      for (int i = 0; i < ntiles; ++i) {
        const uint32_t k = g + i, bx = k & 1;
        const int t = NT - 1 - i;
        const uint32_t stage = smem_base + (bx ? SDSX1 : SDSX0);
        const uint32_t stage_row = stage + dq_half * 8192 + dq_row * 128;
        uint32_t v[64];
#ifdef DQ_FLAGS
        int* fl = nullptr; int old = 0;
        if (tid == 0) { fl = p.dq_flag + (((size_t)b * p.H + h) * NT + t) * 2 + rank; old = atomicCAS(fl, 0, 1); }   // issued early: latency hidden by the DQF wait
#endif
        __syncwarp();   // reconverge before the .aligned tcgen05.ld (a diverged lane 0 makes the load undefined)
#ifdef DBG_DRAIN_SPIN
        mbar_wait_spin(bar(B_DQF), k & 1);
#else
        MBAR_WAIT(B_DQF, k & 1);
#endif
        if (tid == 0) TSTAMP(768, k, 0);
#ifdef DEBUG_HANG
        if (tid == 0 && blockIdx.x >= 2 && blockIdx.x < 4) p.dbg[6 + (blockIdx.x - 2)] = k;
#endif
        tc_fence_after();
        tmem_ld64_sync(tmem + T_DQ + lane_off, v);
        tc_fence_before();
        warp_arrive_leader(bar(B_DQFREE), rank);   // the region is free as soon as it is in registers
        if (tid == 0) TSTAMP(768, k, 1);
        #pragma unroll
        for (int c = 0; c < 8; ++c) {   // chunk c: d = 8c .. 8c+7
          // dQ is reduce-added directly into the bf16 output (scale folded in): no fp16 accumulator, no conversion kernel
          const uint32_t f0 = pack_bf16x2(__uint_as_float(v[8 * c]) * p.scale, __uint_as_float(v[8 * c + 1]) * p.scale);
          const uint32_t f1 = pack_bf16x2(__uint_as_float(v[8 * c + 2]) * p.scale, __uint_as_float(v[8 * c + 3]) * p.scale);
          const uint32_t f2 = pack_bf16x2(__uint_as_float(v[8 * c + 4]) * p.scale, __uint_as_float(v[8 * c + 5]) * p.scale);
          const uint32_t f3 = pack_bf16x2(__uint_as_float(v[8 * c + 6]) * p.scale, __uint_as_float(v[8 * c + 7]) * p.scale);
          st_shared_v4(stage_row + ((c ^ (dq_row & 7)) * 16), f0, f1, f2, f3);
        }
        fence_proxy_async();
        named_bar_sync(2, 128);
        if (tid == 0) {
#ifdef DQ_FLAGS
#ifdef DBG_FLAGS
          printf("drain blk %d rank %u item(b%d h%d) t %d k %u old %d\n", blockIdx.x, rank, b, h, t, k, old);
#endif
          if (old == 0) {   // first contribution to this dq tile half: plain store (dq is not zeroed); published one tile later
            tma_store_4d(&mapDQ, stage, 0, h, t * BR + 64 * (int)rank, b);
            tma_store_4d(&mapDQ, stage + 8192, 64, h, t * BR + 64 * (int)rank, b);
          } else {
            if (old != 2) {   // the first store (by another CTA) is still in flight: publish my own pending flag first (no circular wait), then spin
              if (fl_pending) { bulk_wait<0>(); fence_proxy_async_global(); __threadfence(); st_release_gpu(fl_pending, 2); fl_pending = nullptr; }
#ifdef DEBUG_HANG
              { uint32_t _n = 0; while (ld_acquire_gpu(fl) != 2) { __nanosleep(64); if (++_n == (1u << 20)) printf("SPIN blk %d item(b%d h%d) t %d k %u old %d flag %d pending %d\n", blockIdx.x, b, h, t, k, old, ld_acquire_gpu(fl), fl_pending ? 1 : 0); } }
#else
              while (ld_acquire_gpu(fl) != 2) __nanosleep(64);
#endif
            }
            fence_proxy_async_global();
            tma_reduce_add_4d(&mapDQ, stage, 0, h, t * BR + 64 * (int)rank, b);
            tma_reduce_add_4d(&mapDQ, stage + 8192, 64, h, t * BR + 64 * (int)rank, b);
          }
          bulk_commit();
          TSTAMP(768, k, 2);
          if (fl_pending) {   // the previous tile's first store: complete by now (only the newest group may still be in flight)
            bulk_wait<1>();
            fence_proxy_async_global();
            __threadfence();
            st_release_gpu(fl_pending, 2);
            fl_pending = nullptr;
          }
          if (old == 0) fl_pending = fl;
          bulk_wait_read<0>();                       // the staging (= exchange buffer bx) may be overwritten again
#else
          tma_reduce_add_4d(&mapDQ, stage, 0, h, t * BR + 64 * (int)rank, b);
          tma_reduce_add_4d(&mapDQ, stage + 8192, 64, h, t * BR + 64 * (int)rank, b);
          bulk_commit();
          TSTAMP(768, k, 2);
          bulk_wait_read<0>();                       // the staging (= exchange buffer bx) may be overwritten again
#endif
          TSTAMP(768, k, 3);
          mbar_arrive(bar(B_STAGERD + bx));
          mbar_arrive_remote_relaxed(bx ? r_stagerdp1 : r_stagerdp0);
          st_shared_release_u32(smem_base + SBAR + 960, k + 1);   // monotonic: no parity aliasing for late waiters (epilogue)
#ifdef DEBUG_HANG
          if (blockIdx.x >= 2 && blockIdx.x < 4) { atomicAdd((unsigned long long*)&p.dbg[(blockIdx.x - 2) * 2 + bx], 1ull); p.dbg[4 + (blockIdx.x - 2)] = k; }
#endif
        }
        __syncwarp();
      }
      g += ntiles;
    }
    if (tid == 0) {
      bulk_wait<0>();
#ifdef DQ_FLAGS
      if (fl_pending) { fence_proxy_async_global(); __threadfence(); st_release_gpu(fl_pending, 2); }
#endif
    }
  } else {
    // ================= compute warpgroups (wg 1..4) =================
    const int cs = wg - 1;                      // q-column slice [32cs, 32cs+32)
    const int hx = cs >> 1;                     // q half (0: left, 1: right)
    const int rq = warp % 4;                    // row quarter: kv rows 32rq .. 32rq+31 of this CTA's block
    const int r = rq * 32 + lane;               // my kv row (TMEM lane)
    const uint32_t lane_off = (uint32_t)(rq * 32) << 16;
    const uint32_t tS = tmem + T_S + lane_off + 32 * cs;       // my S^T slice
    const uint32_t tD = tmem + T_D + lane_off + 32 * cs;       // my dP^T slice
    const uint32_t tP = tS;                                    // my P^T columns: lower 16 of my own S^T slice
    const uint32_t tDS = tmem + T_D + lane_off + 32 * (cs & 1) + 16 * (cs >> 1);   // my packed dS^T columns (dP region cols 0..63)
    // dS exchange: my 4 chunks (8 q each) go to CTA hx at (kvg/8)*1024 + (4(cs&1)+j)*128 + (kvg%8)*16 of buffer (tile & 1)
    const int kvg = 128 * (int)rank + r;
    const uint32_t dsx_off = (uint32_t)((kvg >> 3) * 1024 + (4 * (cs & 1)) * 128 + (kvg & 7) * 16);
    const bool dsx_local = ((uint32_t)hx == rank);
    const uint32_t dsx_base0 = mapa(smem_base + SDSX0, (uint32_t)hx) + dsx_off, dsx_base1 = mapa(smem_base + SDSX1, (uint32_t)hx) + dsx_off;
    const uint32_t dsx_mbar0 = mapa(bar(B_DSX), (uint32_t)hx), dsx_mbar1 = mapa(bar(B_DSX + 1), (uint32_t)hx);
    const bool dsx_tx_thread = (cs == 0 && rq == 0 && lane == 0);   // posts the expect_tx (32KB) on my CTA's DSX barrier
    uint32_t pkP[16];   // P^T of my slice, bf16 pairs (also the P used for dS)
    bool p0_done = false;   // P of the current item's tile 0 was computed before the previous item's epilogue
    uint32_t pk[16];
    uint32_t g = 0, nitem = 0;

    FOR_EACH_ITEM(bh, m, ntiles) {
      const int b = bh / p.H, h = bh % p.H;
      const int j = 2 * m + (int)rank;
      auto tile_t = [&](int i) { return NT - 1 - i; };
      // P^T(i) = exp2(S^T * scale*log2e - lse*log2e) -> bf16 into my P^T columns
      auto p_core = [&](uint32_t k, int t, int mm, auto mask_tag) {   // tile k (global count), q tile t, kv pair mm
        constexpr bool MASK = decltype(mask_tag)::value;
        const uint32_t bq = k & 1;
        const int off = 128 * (int)rank - 128 * (t - 2 * mm);   // element (r, q) masked if q < r + off
        float s[32];
        MBAR_WAIT_HOT(hx ? B_SFB : B_SFA, k & 1);
        MBAR_WAIT(B_LSE + bq, (k >> 1) & 1);
        TSTAMPW(k, 0);
        if (threadIdx.x == 128) TSTAMP(512, k, 0); if (threadIdx.x == 384) TSTAMP(640, k, 0);
        tc_fence_after();
        tmem_ld32_sync(tS, *reinterpret_cast<uint32_t(*)[32]>(&s[0]));
        TSTAMPW(k, 1);
        const float* lse_t = reinterpret_cast<const float*>(smem_raw + SLSE + bq * 512) + 32 * cs;
        if (off >= 128) {
          #pragma unroll
          for (int kq = 0; kq < 16; ++kq) pkP[kq] = 0u;
        } else {
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            const float4 l0 = *reinterpret_cast<const float4*>(lse_t + 8 * c), l1 = *reinterpret_cast<const float4*>(lse_t + 8 * c + 4);
            const float lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
            #pragma unroll
            for (int kq = 0; kq < 8; ++kq) {
              float x = fmaf(s[8 * c + kq], p.scale_log2e, -lv[kq]);   // lse pre-scaled by log2 e
              if (MASK && (32 * cs + 8 * c + kq) < r + off) x = -1e30f;
#if defined(DBG_POLY_ALL)
              s[8 * c + kq] = exp2_poly3(x);                       // FMA pipe only
#elif defined(DBG_POLY)
              s[8 * c + kq] = (kq & 1) ? exp2_poly3(x) : ex2(x);   // half MUFU, half FMA pipe
#else
              s[8 * c + kq] = ex2(x);
#endif
            }
            #pragma unroll
            for (int kq = 0; kq < 4; ++kq) pkP[4 * c + kq] = pack_bf16x2(s[8 * c + 2 * kq], s[8 * c + 2 * kq + 1]);
          }
        }
        warp_arrive(bar(B_LSEFREE + bq));
        TSTAMPW(k, 2);
        TSTAMPW(k, 3);
        tmem_st16(tP, pkP);
        tmem_st_wait();
        tc_fence_before();
        if (threadIdx.x == 384) TSTAMPG(896, k, 0);   // right-half P done (before the PF arrive), both CTAs
        ARRIVE_FWD(bar(B_PF));
        TSTAMPW(k, 4);
        if (threadIdx.x == 384) TSTAMPG(896, k, 1);   // after the arrive
        if (threadIdx.x == 128) TSTAMP(512, k, 1); if (threadIdx.x == 384) TSTAMP(640, k, 1);
      };
      // dS^T(i) = P^T (dP^T - delta): exchange chunks (st.async into the owning CTA's buffer), then bf16 into my dS^T columns
      auto ds_phase = [&](int i) {
        const uint32_t k = g + i, bq = k & 1;
        const float* dlt_t = reinterpret_cast<const float*>(smem_raw + SLSE + 1024 + bq * 512) + 32 * cs;
        MBAR_WAIT_HOT(B_DPFA, k & 1);
        TSTAMPW(k, 5);
        if (threadIdx.x == 128) TSTAMP(512, k, 2); if (threadIdx.x == 384) TSTAMP(640, k, 2);
        tc_fence_after();
        uint32_t dpv[32];
        tmem_ld32_sync(tD, dpv);
        tc_fence_before();
        if (hx == 0) warp_arrive(bar(B_DREADL)); else ARRIVE_FWD(bar(B_DREADR));   // dP^T(i) half read: left -> local (dS^T store), right -> leader (dQ)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const uint32_t* dp = dpv + 16 * hh;
          const float4 d0 = *reinterpret_cast<const float4*>(dlt_t + 16 * hh), d1 = *reinterpret_cast<const float4*>(dlt_t + 16 * hh + 4);
          const float4 d2 = *reinterpret_cast<const float4*>(dlt_t + 16 * hh + 8), d3 = *reinterpret_cast<const float4*>(dlt_t + 16 * hh + 12);
          const float dv[16] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w, d2.x, d2.y, d2.z, d2.w, d3.x, d3.y, d3.z, d3.w};
          #pragma unroll
          for (int kq = 0; kq < 8; ++kq) {
            const uint32_t dd = pack_bf16x2(__uint_as_float(dp[2 * kq]) - dv[2 * kq], __uint_as_float(dp[2 * kq + 1]) - dv[2 * kq + 1]);
            uint32_t rr; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(rr) : "r"(pkP[8 * hh + kq]), "r"(dd));
            pk[8 * hh + kq] = rr;
          }
        }
        if (threadIdx.x == 128) TSTAMP(512, k, 3); if (threadIdx.x == 384) TSTAMP(640, k, 3);
        warp_arrive(bar(B_DLTFREE + bq));
        // dS^T into TMEM first: DSF gates M4(i) (critical); the exchange only gates M5(i), issued a tile later
        if (threadIdx.x == 128) TSTAMP(512, k, 4); if (threadIdx.x == 384) TSTAMP(640, k, 4);
        if (hx == 1) { MBAR_WAIT(B_DREADL, k & 1); tc_fence_after(); }   // my dS^T columns belong to a left slice's dP^T
        tmem_st16(tDS, pk);
        tmem_st_wait();
        tc_fence_before();
        if (threadIdx.x == 384) TSTAMPG(896, k, 2);
        ARRIVE_FWD(bar(B_DSF));
        TSTAMPW(k, 6);
        if (threadIdx.x == 384) TSTAMPG(896, k, 3);
        // exchange into buffer bq of CTA hx: its previous content (tile k-2's dS, then the dQ(k-2) staging) must be consumed
        if (i == 0 && nitem > 0) {   // the previous item's dV / dK stores must have finished reading both CTAs' exchange buffers
          if (lane == 0 && rq == 0 && (cs & 1) == 0) { bulk_wait_read<0>(); mbar_arrive(bar(B_EPISTORE)); mbar_arrive_remote_relaxed(mapa(bar(B_EPISTORE), peer)); }
          __syncwarp();
          MBAR_WAIT(B_EPISTORE, (nitem - 1) & 1);
        }
        if (k >= 2) MBAR_WAIT((dsx_local ? B_STAGERD : B_STAGERDP) + bq, ((k - 2) >> 1) & 1);
#ifdef E_NOREMOTE
        if (dsx_tx_thread) mbar_arrive(bar(B_DSX + bq));
#else
        if (dsx_tx_thread) mbar_arrive_expect_tx(bar(B_DSX + bq), 16384);   // the peer's half arrives through st.async
#endif
        if (dsx_local) {   // my half of my own CTA's buffer: plain shared stores + async-proxy fence + one arrive per warp
#ifndef E_NOLOCAL
          const uint32_t base = smem_base + (bq ? SDSX1 : SDSX0) + dsx_off;
          #pragma unroll
          for (int jj = 0; jj < 4; ++jj) st_shared_v4(base + jj * 128, pk[4 * jj], pk[4 * jj + 1], pk[4 * jj + 2], pk[4 * jj + 3]);
          fence_proxy_async();
#endif
          warp_arrive(bar(B_DSX + bq));
        } else {
#ifndef E_NOREMOTE
          const uint32_t base = bq ? dsx_base1 : dsx_base0, mb = bq ? dsx_mbar1 : dsx_mbar0;
          #pragma unroll
          for (int jj = 0; jj < 4; ++jj) st_async_v4(base + jj * 128, pk[4 * jj], pk[4 * jj + 1], pk[4 * jj + 2], pk[4 * jj + 3], mb);
#endif
        }
        if (threadIdx.x == 128) TSTAMP(512, k, 5); if (threadIdx.x == 384) TSTAMP(640, k, 5);
      };
      // masking is needed only on the last 1 + rank tiles of an item (off > -128)
      auto p_any = [&](int i) { if (i >= ntiles - 1 - (int)rank) p_core(g + i, tile_t(i), m, std::true_type{}); else p_core(g + i, tile_t(i), m, std::false_type{}); };
      if (!p0_done) p_any(0);
      for (int i = 0; i < ntiles; ++i) {
        ds_phase(i);
        if (i + 1 < ntiles) p_any(i + 1);
      }
      // P of the next item's first tile before this item's epilogue (its S^T only needs the next item's K and Q_s)
      p0_done = false;
      {
        int nbh = -1, nm = 0;
        if (half == 0 && m1_ != m0_) { nbh = bh; nm = m1_; }
        else if (pi + p.num_clusters < total_pairs) { int b_; pair_items(pi + p.num_clusters, nbh, nm, b_); }
        if (nbh >= 0) {
          const int nnt = NT - 2 * nm;
          if (nnt <= 1 + (int)rank) p_core(g + ntiles, NT - 1, nm, std::true_type{}); else p_core(g + ntiles, NT - 1, nm, std::false_type{});
          p0_done = true;
        }
      }
      g += ntiles;
      // ---- epilogue: dV (wg 1, 2) / dK (wg 3, 4) -> bf16 staging in the exchange buffers (128B-swizzled) -> TMA store ----
      if (threadIdx.x == 128) TSTAMP(512, g - 1, 6); if (threadIdx.x == 384) TSTAMP(640, g - 1, 6);
      {
        const uint32_t glast = g - 1;
        // dV is final once M3(glast) is done (DODDEAD), dK once M4(glast) is done (DVKDONE)
        if (cs < 2) MBAR_WAIT(B_DODDEAD, glast & 1); else MBAR_WAIT(B_DVKDONE, nitem & 1);
        nitem++;
        if (threadIdx.x == 128) TSTAMP(512, g - 1, 7); if (threadIdx.x == 384) TSTAMP(640, g - 1, 7);
        // staging: dV in buffer (glast+1)&1 (last used by dQ(glast-1)'s drain), dK in buffer glast&1 (dQ(glast)'s drain)
        const uint32_t sbx = (cs < 2) ? ((glast + 1) & 1) : (glast & 1);
        const uint32_t sb = smem_base + (sbx ? SDSX1 : SDSX0);
        const uint32_t stg = sb + 16384 * (cs & 1) + (uint32_t)r * 128;   // 64-column half (cs & 1), my row r (128 B, swizzled 16B chunks)
        tc_fence_after();
        const uint32_t acc = tmem + (cs < 2 ? T_DV : T_DK) + lane_off + 64 * (cs & 1);
        const float mul = (cs < 2) ? 1.f : p.scale;
        uint32_t f[32];
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
          uint32_t v[32];
          tmem_ld32_sync(acc + 32 * c, v);
          #pragma unroll
          for (int q = 0; q < 16; ++q) f[16 * c + q] = pack_bf16x2(__uint_as_float(v[2 * q]) * mul, __uint_as_float(v[2 * q + 1]) * mul);
        }
        tc_fence_before();
        warp_arrive_leader(bar(cs < 2 ? B_DVRD : B_DKRD), rank);   // the accumulator is free: the next item's M3(0) / M4(0) may start
        // The drain may lag by more than one tile of a buffer here, so a parity wait could alias; spin on its monotonic counter
        // instead: dV needs the drain past tile glast-1, dK past tile glast.
        { const uint32_t need = (cs < 2) ? glast : glast + 1; while (ld_shared_acquire_u32(smem_base + SBAR + 960) < need) { __nanosleep(32); } }
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) st_shared_v4(stg + ((jj ^ (r & 7)) * 16), f[4 * jj], f[4 * jj + 1], f[4 * jj + 2], f[4 * jj + 3]);
        fence_proxy_async();
        named_bar_sync(3 + (cs >> 1), 256);         // my group's 8 warps have staged their rows
        if (lane == 0 && rq == 0 && (cs & 1) == 0) {   // warp 4 (dV) / warp 12 (dK): two 128-row x 64-column boxes
          const CUtensorMap* mp = (cs < 2) ? &mapDV : &mapDK;
          tma_store_4d(mp, sb, 0, h, j * BC, b);
          tma_store_4d(mp, sb + 16384, 64, h, j * BC, b);
          bulk_commit();
        }
        __syncwarp();
      }
      if (threadIdx.x == 128) TSTAMP(512, g - 1, 8); if (threadIdx.x == 384) TSTAMP(640, g - 1, 8);
    }
    if (lane == 0 && rq == 0 && (cs & 1) == 0) bulk_wait<0>();   // the last item's dV / dK stores
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();   // no CTA leaves while its peer may still touch its SMEM / barriers / TMEM
  tc_fence_after();
  if (warp == 0) tmem_dealloc2(tmem, 512);
}

// dq = bf16(scale * dq_accum), both (B,N,H,D); 8 elements per thread.
__global__ void __launch_bounds__(256) k_convert(const __half* dq_accum, __nv_bfloat16* dq, size_t n8, float scale) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (i >= n8) return;
  const uint4 v = __ldcs(reinterpret_cast<const uint4*>(dq_accum) + i);
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t o[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    float lo, hi;
    asm("{\n.reg .f16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}" : "=f"(lo), "=f"(hi) : "r"(w[k]));
    o[k] = pack_bf16x2(lo * scale, hi * scale);
  }
  st_global_v4(reinterpret_cast<uint4*>(dq) + i, o[0], o[1], o[2], o[3]);
}

__global__ void __launch_bounds__(256) k_pre(const __nv_bfloat16* o, const __nv_bfloat16* dout, float* delta, float* dq_accum,
                                             const float* lse, float* lse2, int N, int H, int rows) {
  for (int i = blockIdx.x * 256 + threadIdx.x; i < rows; i += gridDim.x * 256) lse2[i] = lse[i] * 1.4426950408889634f;   // rows == B*H*N
  const int lane16 = threadIdx.x % 16;
  const int row0 = (blockIdx.x * 256 + threadIdx.x) / 16;
  const int rstride = gridDim.x * 16;
#ifdef KPRE_UNROLL
  for (int row = row0; row < rows; row += 4 * rstride) {
    uint4 a[4], gq[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int rr = row + u * rstride;
      if (rr < rows) {
        const size_t off = (size_t)rr * HD + lane16 * 8;
        a[u] = __ldg(reinterpret_cast<const uint4*>(o + off));
        gq[u] = __ldg(reinterpret_cast<const uint4*>(dout + off));
      }
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int rr = row + u * rstride;
      if (rr < rows) {
        const uint32_t av[4] = {a[u].x, a[u].y, a[u].z, a[u].w}, gv[4] = {gq[u].x, gq[u].y, gq[u].z, gq[u].w};
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) s += bf16lo(av[i]) * bf16lo(gv[i]) + bf16hi(av[i]) * bf16hi(gv[i]);
        #pragma unroll
        for (int k = 8; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffff, s, k);
        if (dq_accum) { float4* z = reinterpret_cast<float4*>(dq_accum + (size_t)rr * (HD / 2)) + lane16; z[0] = make_float4(0.f, 0.f, 0.f, 0.f); }
        if (lane16 == 0) {
          const int bb = rr / (N * H), n = (rr / H) % N, hh = rr % H;
          delta[((size_t)bb * H + hh) * N + n] = s;
        }
      }
    }
  }
}
#else
  for (int row = row0; row < rows; row += rstride) {
    const size_t off = (size_t)row * HD + lane16 * 8;
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(o + off));
    const uint4 g = __ldg(reinterpret_cast<const uint4*>(dout + off));
    const uint32_t av[4] = {a.x, a.y, a.z, a.w}, gv[4] = {g.x, g.y, g.z, g.w};
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) s += bf16lo(av[i]) * bf16lo(gv[i]) + bf16hi(av[i]) * bf16hi(gv[i]);
    #pragma unroll
    for (int k = 8; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffff, s, k);
    if (dq_accum) { float4* z = reinterpret_cast<float4*>(dq_accum + (size_t)row * (HD / 2)) + lane16; z[0] = make_float4(0.f, 0.f, 0.f, 0.f); }   // zero the (bf16 dq) row
    if (lane16 == 0) {
      const int bb = row / (N * H), n = (row / H) % N, hh = row % H;
      delta[((size_t)bb * H + hh) * N + n] = s;
    }
  }
}
#endif

// Streaming pre-kernel: delta = rowsum(o * dout) with cp.async.bulk chunks of 64 rows (16 KB per tensor) through a 4-stage
// SMEM ring; also lse2 = lse * log2 e, optionally zeroes dq (bf16 rows) and the dq flags.
constexpr int KP_ROWS = 64, KP_STAGES = 4, KP_STAGE_BYTES = 2 * KP_ROWS * HD * 2;
__global__ void __launch_bounds__(256) k_pre2(const __nv_bfloat16* o, const __nv_bfloat16* dout, float* delta, float* dq_zero,
                                              const float* lse, float* lse2, int* flags, int nflags, int N, int H, int rows) {
  extern __shared__ __align__(128) unsigned char kp_smem[];
  __shared__ __align__(8) uint64_t kp_bar[KP_STAGES];
  for (int i = blockIdx.x * 256 + threadIdx.x; i < rows; i += gridDim.x * 256) lse2[i] = lse[i] * 1.4426950408889634f;
  if (flags) for (int i = blockIdx.x * 256 + threadIdx.x; i < nflags; i += gridDim.x * 256) flags[i] = 0;
  const uint32_t sbase = smem_u32(kp_smem);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int s_ = 0; s_ < KP_STAGES; ++s_) mbar_init(smem_u32(&kp_bar[s_]), 1);
    mbar_fence_init();
  }
  __syncthreads();
  const int nchunks = rows / KP_ROWS;
  auto issue = [&](int it) {
    const int c = blockIdx.x + it * gridDim.x;
    if (c >= nchunks) return;
    const int s_ = it % KP_STAGES;
    const uint32_t mb = smem_u32(&kp_bar[s_]);
    mbar_arrive_expect_tx(mb, KP_STAGE_BYTES);
    bulk_copy_g2s(sbase + s_ * KP_STAGE_BYTES, o + (size_t)c * KP_ROWS * HD, KP_STAGE_BYTES / 2, mb);
    bulk_copy_g2s(sbase + s_ * KP_STAGE_BYTES + KP_STAGE_BYTES / 2, dout + (size_t)c * KP_ROWS * HD, KP_STAGE_BYTES / 2, mb);
  };
  if (threadIdx.x == 0) { for (int it = 0; it < KP_STAGES; ++it) issue(it); }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, lane16 = lane % 16;
  for (int it = 0;; ++it) {
    const int c = blockIdx.x + it * gridDim.x;
    if (c >= nchunks) break;
    const int s_ = it % KP_STAGES;
    mbar_wait(smem_u32(&kp_bar[s_]), (it / KP_STAGES) & 1);
    const unsigned char* so = kp_smem + s_ * KP_STAGE_BYTES;
    const unsigned char* sg = so + KP_STAGE_BYTES / 2;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {   // warp: rows 8w .. 8w+7, two rows (512 contiguous bytes) per instruction
      const int rl = 8 * warp + 2 * j + (lane >> 4);
      const uint4 a = *reinterpret_cast<const uint4*>(so + rl * 256 + lane16 * 16);
      const uint4 g = *reinterpret_cast<const uint4*>(sg + rl * 256 + lane16 * 16);
      const uint32_t av[4] = {a.x, a.y, a.z, a.w}, gv[4] = {g.x, g.y, g.z, g.w};
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) s += bf16lo(av[i]) * bf16lo(gv[i]) + bf16hi(av[i]) * bf16hi(gv[i]);
      #pragma unroll
      for (int k = 8; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffff, s, k);
      const int row = c * KP_ROWS + rl;
      if (dq_zero) { float4* z = reinterpret_cast<float4*>(dq_zero + (size_t)row * (HD / 2)) + lane16; z[0] = make_float4(0.f, 0.f, 0.f, 0.f); }
      if (lane16 == 0) {
        const int bb = row / (N * H), n = (row / H) % N, hh = row % H;
        delta[((size_t)bb * H + hh) * N + n] = s;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) issue(it + KP_STAGES);
  }
}

static CUtensorMap make_map_4d(void* ptr, int B, int N, int H, uint32_t box_rows, CUtensorMapDataType dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) {
  CUtensorMap m;
  cuuint64_t gdim[4] = {(cuuint64_t)HD, (cuuint64_t)H, (cuuint64_t)N, (cuuint64_t)B};
  cuuint64_t gstride[3] = {(cuuint64_t)HD * 2, (cuuint64_t)H * HD * 2, (cuuint64_t)N * H * HD * 2};
  cuuint32_t box[4] = {64, 1, box_rows, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = get_encode_fn()(&m, dt, 4, ptr, gdim, gstride, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed %d\n", (int)r); exit(1); }
  return m;
}

}  // namespace

#ifdef DEBUG_TRACE
static void* g_trace_dev = nullptr;
extern "C" int fa_bwd49_trace_info(long long* out) { if (g_trace_dev) cudaMemcpy(out, g_trace_dev, 2048 * 8, cudaMemcpyDeviceToHost); return 0; }
#endif
#if defined(DEBUG_TIMING) || defined(DEBUG_HANG)
static void* g_dbg_dev = nullptr;
extern "C" int fa_bwd49_dbg_info(long long* out) { if (g_dbg_dev) cudaMemcpy(out, g_dbg_dev, 16 * 8, cudaMemcpyDeviceToHost); return 0; }
#endif

extern "C" int fa_bwd_launch(const void* q, const void* k, const void* v, const void* o, const void* dout, const float* lse,
                             float* delta, float* dq_accum, void* dq, void* dk, void* dv, int* counters,
                             int B, int N, int H, float scale, cudaStream_t stream, int skip_pre) {
  (void)counters;
  static int num_clusters = 0;
  if (!num_clusters) {
    cudaFuncSetAttribute(fa_bwd49_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(296); cfg.blockDim = dim3(NUM_THREADS); cfg.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, fa_bwd49_kernel, &cfg) != cudaSuccess || n <= 0) { printf("cudaOccupancyMaxActiveClusters failed\n"); n = 1; }
    num_clusters = n;
    const char* env = getenv("FA_BWD_CLUSTERS");
    if (env) num_clusters = atoi(env);
    printf("fa_bwd49: %d persistent clusters\n", num_clusters);
  }
  const int rows = B * N * H;
  static float* lse2 = nullptr; static size_t lse2_n = 0;
  if (lse2_n < (size_t)rows) { if (lse2) cudaFree(lse2); cudaMalloc(&lse2, (size_t)rows * sizeof(float)); lse2_n = rows; }
  const int NT_ = N / BR;
  static int* dq_flag = nullptr; static size_t flag_n = 0;
  const size_t nflags = (size_t)B * H * NT_ * 2;
#ifdef DQ_FLAGS
  if (flag_n < nflags) { if (dq_flag) cudaFree(dq_flag); cudaMalloc(&dq_flag, nflags * sizeof(int)); flag_n = nflags; }
  float* dq_zero = nullptr;
  if (skip_pre) cudaMemsetAsync(dq_flag, 0, nflags * sizeof(int), stream);
#else
  float* dq_zero = reinterpret_cast<float*>(dq);
  (void)flag_n; (void)nflags;
#endif
  static int nsm = 0;
  if (!nsm) { cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0); cudaFuncSetAttribute(k_pre2, cudaFuncAttributeMaxDynamicSharedMemorySize, KP_STAGES * KP_STAGE_BYTES); }
#ifdef KPRE2
  if (!skip_pre) k_pre2<<<2 * nsm, 256, KP_STAGES * KP_STAGE_BYTES, stream>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)dout, delta, dq_zero, lse, lse2, dq_flag, (int)nflags, N, H, rows);
#else
  if (!skip_pre) k_pre<<<rows / 16 / 4, 256, 0, stream>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)dout, delta, dq_zero, lse, lse2, N, H, rows);
#ifdef DQ_FLAGS
  if (!skip_pre) cudaMemsetAsync(dq_flag, 0, nflags * sizeof(int), stream);
#endif
#endif
  CUtensorMap mQ32 = make_map_4d((void*)q, B, N, H, 32), mQ128 = make_map_4d((void*)q, B, N, H, 128);
  CUtensorMap mK = make_map_4d((void*)k, B, N, H, BC), mV = make_map_4d((void*)v, B, N, H, BC);
  CUtensorMap mDO32 = make_map_4d((void*)dout, B, N, H, 32), mDO128 = make_map_4d((void*)dout, B, N, H, 128);
  CUtensorMap mDQ = make_map_4d((void*)dq, B, N, H, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);   // bf16 reduce-add into the output
  CUtensorMap mDV = make_map_4d((void*)dv, B, N, H, 128), mDK = make_map_4d((void*)dk, B, N, H, 128);   // epilogue TMA stores
  BwdParams p;
  p.lse = lse2; p.delta = delta; p.dk = (__nv_bfloat16*)dk; p.dv = (__nv_bfloat16*)dv; p.dbg = nullptr; p.trace = nullptr;
#ifdef DEBUG_TRACE
  if (!g_trace_dev) cudaMalloc(&g_trace_dev, 2048 * 8);
  cudaMemsetAsync(g_trace_dev, 0, 2048 * 8, stream);
  p.trace = reinterpret_cast<long long*>(g_trace_dev);
#endif
#if defined(DEBUG_TIMING) || defined(DEBUG_HANG)
  if (!g_dbg_dev) cudaMalloc(&g_dbg_dev, 4096);
  cudaMemsetAsync(g_dbg_dev, 0, 4096, stream);
  p.dbg = reinterpret_cast<long long*>(g_dbg_dev);
#endif
  p.dq_flag = dq_flag; p.B = B; p.N = N; p.H = H; p.scale_log2e = scale * 1.4426950408889634f; p.scale = scale;
  const int NT = N / BR;
  const int total_pairs = B * H * ((NT / 2 + 1) / 2);
  p.num_clusters = num_clusters < total_pairs ? num_clusters : total_pairs;
  fa_bwd49_kernel<<<2 * p.num_clusters, NUM_THREADS, SMEM_BYTES, stream>>>(mQ32, mQ128, mK, mV, mDO32, mDO128, mDQ, mDV, mDK, p);
  const size_t n8 = (size_t)rows * HD / 8;
  (void)n8; (void)dq_accum;   // no conversion kernel: dq accumulated in bf16 directly
  return (int)cudaGetLastError();
}
