// MLA paged prefill (causal), page_size=1, H=16, d_ckv=512, d_kpe=64. bf16 in, bf16 out, lse (log2) f32.
// Blackwell sm_100a. Written from scratch on top of umma.cuh (tcgen05 / TMA / mbarrier wrappers).
//
// v4 dataflow:
//  * Cluster of 4 CTAs = 2 MMA pairs (cta_group::2). The cluster owns 16 consecutive query positions of one request:
//    pair A (ranks 0,1) -> positions [p0, p0+8), pair B (ranks 2,3) -> [p0+8, p0+16). Each CTA owns 64 query rows
//    (4 positions x 16 heads). Both pairs stream the same KV blocks (multicast to ranks r, r+2).
//  * KV tile = 256 rows. QK (S = Q K^T, M=128 N=256 K=576): B = K, N-split -> CTA r holds kv rows [128r, 128r+128) of the
//    tile (2 blocks of 64 rows per 64-dim chunk). S lands in TMEM 2x2 layout: lanes 0..63 = rows x kv[0,128),
//    lanes 64..127 = rows x kv[128,256), 128 columns per CTA. Two S buffers.
//  * Softmax warps write P (bf16) to SMEM (K-major, 128B swizzle, 4 blocks of [64 rows][64 kv]).
//  * PV (O += P V, M=128 N=512 K=256) as two N=256 chains (chain j -> hd [128j,128j+128) from CTA0 and
//    [256+128j, +128) from CTA1): B = V N-split -> CTA r holds hd [256r, 256r+256) for all 256 kv rows of the tile,
//    streamed as (kv sub-block c, hd chunk pair) 16KB slots. O = 256 TMEM columns per CTA (lanes 0..63: hd < 256).
//  * K/V blocks flow through a ring of 7 slots x 16KB in consumption order: QK_{t+1} (9 slots), PV_t (8 slots).
//  * Online softmax with lazy rescaling (threshold 8 in log2 units), rescale applied to O in TMEM by the softmax warps.
#include "umma.cuh"
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace mla {
DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef MLA_TIMELINE
#define TL(role, idx) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && (idx) < 8192) ((unsigned long long*)P.dbg)[(rank * 4 + (role)) * 8192 + (idx)] = gtimer(); } while (0)
#define TLT(role, k) TL(role, 16 * gtile + (k))
#define TLQ(role, k) TL(role, 6800 + 4 * qt + (k))
#define SMEV(ev) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && lane == 0 && gtile < 32) ((unsigned long long*)P.dbg)[(rank * 4 + 1) * 8192 + 4000 + gtile * 128 + (warp - NUM_CTRL_WARPS) * 8 + (ev)] = gtimer(); } while (0)
#define SMEVC(ev) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && lane == 0 && gtile < 32) ((unsigned long long*)P.dbg)[(rank * 4 + 1) * 8192 + 4000 + gtile * 128 + (warp - NUM_CTRL_WARPS) * 8 + (ev)] = clock64(); } while (0)
#define SLOTG(role, use, ev) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && (use) < 200) ((unsigned long long*)P.dbg)[(rank * 4 + (role)) * 8192 + 4000 + (use) * 8 + (ev)] = gtimer(); } while (0)
#define BQ(role, k) TL(role, 6000 + 16 * qt + (k))
#define EPI(k) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && tid == 32 * NUM_CTRL_WARPS && qt < 30) ((unsigned long long*)P.dbg)[(rank * 4 + 3) * 8192 + 6500 + 16 * qt + (k)] = clock64(); } while (0)
#define FCN(k, v) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt)) ((unsigned long long*)P.dbg)[(rank * 4 + 0) * 8192 + 7600 + 16 * (gtile & 31) + (k)] = (v); } while (0)
#define FC(k) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && (gtile & 31) < 32) ((unsigned long long*)P.dbg)[(rank * 4 + 0) * 8192 + 7000 + 16 * (gtile & 31) + (k)] = clock64(); } while (0)
#define SLOTEV(use, ev) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && (use) < 200) ((unsigned long long*)P.dbg)[(rank * 4 + 2) * 8192 + 4000 + (use) * 8 + (ev)] = clock64(); } while (0)
#define WPROG(stage) do { if (P.dbg && cluster_id == (P.dbg_qt < 0 ? 0 : P.dbg_qt) && lane == 0) { ((unsigned long long*)P.dbg)[(rank * 4 + 3) * 8192 + 5000 + warp * 16 + (stage)] = gtimer(); ((unsigned long long*)P.dbg)[(rank * 4 + 3) * 8192 + 5000 + warp * 16 + 8 + (stage)] = gtile; } } while (0)
#else
#define TL(role, idx)
#define TLT(role, k)
#define TLQ(role, k)
#define SLOTEV(use, ev)
#define BQ(role, k)
#define EPI(k)
#define FC(k)
#define FCN(k, v)
#define SLOTG(role, use, ev)
#define SMEV(ev)
#define SMEVC(ev)
#define WPROG(stage)
#endif

constexpr int H = 16, D_CKV = 512, D_KPE = 64;
constexpr int ROWS = 64;            // q rows per CTA
constexpr int QPOS = 4;             // q positions per CTA
constexpr int CL_POS = 8;           // q positions per cluster (cluster = one 2-SM pair; 74 clusters fill all 148 SMs)
constexpr int CLUSTER = 2;
constexpr int KV_TILE = 256;        // kv rows per tile
constexpr int BLK_ROWS = 64;
constexpr int BLK_BYTES = BLK_ROWS * 128;   // 8192
constexpr int SLOT_BYTES = 4 * BLK_BYTES;   // 32768: QK slot = 2 hd chunks x 2 kv sub-blocks; PV slot = 1 kv sub-block x 4 hd chunks
constexpr int QBLK_BYTES = ROWS * 128;      // 8192
constexpr int NSLOT = 3;   // ring of 32KB slots; consumption order per tile: QK_{t+1} (kpe side buffer + 4 slots) then PV_t (4 slots)
constexpr int NUM_SOFTMAX_WARPS = 8;
constexpr int NUM_CTRL_WARPS = 4;   // warp 0: MMA issuer, warp 1: TMA issuer, warp 2: registrar, warp 3: P forwarder
constexpr int NUM_THREADS = 32 * (NUM_CTRL_WARPS + NUM_SOFTMAX_WARPS);
constexpr int SOFTMAX_THREADS = 32 * NUM_SOFTMAX_WARPS;
constexpr float RESCALE_THRESH = 8.0f;
constexpr int S_COLS = 128;                 // S columns per CTA (2x2 layout: N/2)
constexpr int SUB_S = 64;                   // S columns per softmax thread
constexpr int SUB_O = 128;                  // O columns per softmax thread

// SMEM map (bytes, 1024-aligned regions)
constexpr int SM_Q = 0;                                   // 9 x 8192 (chunks 0..7 ckv, 8 kpe)
constexpr int SM_P = SM_Q + 9 * QBLK_BYTES;               // 73728 : 4 x 8192 (kv blocks of 64)
constexpr int SM_SLOT = SM_P + 4 * BLK_BYTES;             // 106496 : 3 x 32768
constexpr int SM_KPE = SM_SLOT + NSLOT * SLOT_BYTES;      // 204800 : 2 x 8192 (kpe blocks of my kv half)
constexpr int SM_MISC = SM_KPE + 2 * BLK_BYTES;           // 221184
// barriers
constexpr int B_FULL = 0;                 // slot full: leader count 2 (own expect_tx arrive + peer registrar), peer count 1 [3]
constexpr int B_EMPTY = 3;                // slot consumed by both pairs [3]
constexpr int B_KFULL = 6, B_KEMPTY = 7;  // kpe side buffer
constexpr int B_SFULL = 8;                // [2]
constexpr int B_PFULL = 10, B_PLOCAL = 11, B_PEMPTY = 12, B_QFULL = 13, B_QPEER = 14, B_QEMPTY = 15, B_OFULL = 16, B_OEMPTY = 17, B_STAGED = 18, B_STAGEFREE = 19, B_LOADS = 20, NBAR = 21;
constexpr int SM_BAR = SM_MISC;
constexpr int SM_REDM = SM_BAR + 256;                     // float redm[4][64] (2048 reserved)
constexpr int SM_REDL = SM_REDM + 8 * 64 * 4;             // float redl[8][64] = 2048
constexpr int SM_ORDER = SM_REDL + 8 * 64 * 4;            // int order[32]
constexpr int SM_TMEM = SM_ORDER + 32 * 4;
constexpr int SM_TOTAL = SM_TMEM + 16;
static_assert(SM_TOTAL <= 232448 - 1024, "smem overflow");
// TMEM map (columns)
constexpr int T_O = 0;        // 256 cols
constexpr int T_S0 = 256;     // 128 cols
constexpr int T_S1 = 384;     // 128 cols

struct Params {
  const __nv_bfloat16* q_nope; const __nv_bfloat16* q_pe;
  const __nv_bfloat16* ckv; const __nv_bfloat16* kpe;
  const int* qo_indptr; const int* kv_indptr; const int* kv_indices;
  __nv_bfloat16* out; float* lse;
  float sm_scale; int B; int total_q; int nclusters; float* dbg; int dbg_qt; float* epi; const int* sched; int sched_stride;
};
struct TileInfo { int b, p0, Lq, Lk, qs, kvs; int valid; };

DEVI TileInfo get_tile(const Params& P, const int* order, int g) {   // g = q-tile slot index in this cluster's table
  TileInfo t;
  const int4* e = (const int4*)(order) + 2 * g;   // 'order' points at this cluster's table entries
  int4 a = __ldg(e), b = __ldg(e + 1);
  t.b = a.x; t.p0 = a.y; t.Lq = a.z; t.Lk = a.w; t.qs = b.x; t.kvs = b.y; t.valid = 1;
  return t;
}
DEVI int num_pv_sub_last(const TileInfo& t, int ntiles) {   // kv sub-blocks (64 rows) of the last kv tile that hold unmasked columns
  int plast = min(t.p0 + CL_POS - 1, t.Lq - 1);
  int kv_end = plast + (t.Lk - t.Lq) + 1;
  int r = kv_end - KV_TILE * (ntiles - 1);
  return (r + 63) >> 6;
}
DEVI int num_kv_tiles(const TileInfo& t) {
  int plast = min(t.p0 + CL_POS - 1, t.Lq - 1);
  int kv_end = plast + (t.Lk - t.Lq) + 1;
  return (kv_end + KV_TILE - 1) / KV_TILE;
}
DEVI void named_bar_sync(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }
DEVI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float lg2(float x) { float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

DEVI void tma_tile2d_mc(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
DEVI void tma_tile2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DEVI void st_shared_v4(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

struct Maps { CUtensorMap ckv, kpe, qn, qp, out[4]; };   // out[i]: box of 16*(i+1) rows
DEVI void tma_store2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(map), "r"(c0), "r"(c1), "r"(src) : "memory");
}
DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

__global__ void __launch_bounds__(NUM_THREADS, 1) mla_prefill_kernel(const __grid_constant__ Maps maps, Params P) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const uint32_t sbase = smem_u32(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cluster_ctarank();      // 0..3
  const int par = rank & 1;                      // kv half (QK) / hd half (PV) owned
  const uint32_t leader = rank & ~1u;            // pair leader rank
  const bool is_leader = (rank == leader);
  const int cluster_id = blockIdx.x / CLUSTER;
  auto bar = [&](int i) { return sbase + SM_BAR + i * 8; };
  auto bar_leader = [&](int i) { return is_leader ? bar(i) : mapa_shared(bar(i), leader); };
  float* redm = (float*)(smem + SM_REDM);
  float* redl = (float*)(smem + SM_REDL);
  int* order = (int*)(smem + SM_ORDER);
  uint32_t* tmem_slot = (uint32_t*)(smem + SM_TMEM);

  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(bar(B_FULL + i), is_leader ? 2 : 1); mbar_init(bar(B_EMPTY + i), 1); }
    mbar_init(bar(B_KFULL), is_leader ? 2 : 1); mbar_init(bar(B_KEMPTY), 1);
    mbar_init(bar(B_SFULL), 1); mbar_init(bar(B_SFULL + 1), 1);
    mbar_init(bar(B_PFULL), NUM_SOFTMAX_WARPS + 1); mbar_init(bar(B_PLOCAL), NUM_SOFTMAX_WARPS); mbar_init(bar(B_PEMPTY), 1);
    mbar_init(bar(B_QFULL), 1); mbar_init(bar(B_QPEER), 1); mbar_init(bar(B_QEMPTY), 1);
    mbar_init(bar(B_OFULL), 1); mbar_init(bar(B_OEMPTY), 2 * NUM_SOFTMAX_WARPS);
    mbar_init(bar(B_STAGED), NUM_SOFTMAX_WARPS / 2); mbar_init(bar(B_STAGEFREE), 1); mbar_init(bar(B_LOADS), 1);
    fence_mbar_init();
  }
  if (warp == 0) { tmem_alloc<2>(sbase + SM_TMEM, 512); tmem_relinquish<2>(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;
  const uint32_t tO = tmem + T_O, tS0 = tmem + T_S0, tS1 = tmem + T_S1;
  const float scale_log2 = P.sm_scale * 1.4426950408889634f;
  const bool getenv_nohold = (P.dbg_qt == -21);
  const bool skip_sm = (P.dbg_qt == -7 || P.dbg_qt == -8), skip_tma = (P.dbg_qt == -8 || P.dbg_qt == -17), spin_sm = (P.dbg_qt == -10), skip_mma = (P.dbg_qt == -9), spin_ctl = (P.dbg_qt != -12); const bool spin_all = (P.dbg_qt != -12); const bool nosleep = (P.dbg_qt == -15 || P.dbg_qt == -16);
  const int* my_sched = P.sched + (size_t)cluster_id * P.sched_stride; const int my_n = my_sched[0]; const int* my_tab = my_sched + 8;
#ifdef MLA_TIMELINE
  if (P.dbg && tid == 0) ((unsigned long long*)P.dbg)[3 * 8192 + 7000 + blockIdx.x] = gtimer();
#endif

  if (warp < NUM_CTRL_WARPS) {
    if (warp == 0 && lane == 0 && is_leader) {
      // ===================================================== MMA issuer (pair leader): straight-line issue in ring order
      uint32_t par_full = 0; int slot = 0; uint32_t par_kfull = 0;
      uint32_t cnt_qfull = 0, cnt_pfull = 0, cnt_oempty = 0;
      constexpr uint32_t idescQK = make_idesc_bf16(128, 256, false, false);
      constexpr uint32_t idescPV = make_idesc_bf16(128, 256, false, true);
      const uint16_t pair_mask = (uint16_t)(3u << leader);
      const uint64_t qdesc0 = make_smem_desc(sbase + SM_Q, 16, 1024, 2);
      const uint64_t kdesc0 = make_smem_desc(sbase + SM_SLOT, 16, 1024, 2);
      const uint64_t kpdesc = make_smem_desc(sbase + SM_KPE, 16, 1024, 2);
      const uint64_t pdesc0 = make_smem_desc(sbase + SM_P, 16, 1024, 2);
      const uint64_t vdesc0 = make_smem_desc(sbase + SM_SLOT, BLK_BYTES, 1024, 2);
      int gtile = 0;
      // spin until the ring slot at the head is full; returns the slot and advances
      auto take_slot = [&]() -> int {
        const int s = slot; const uint32_t pp = (par_full >> s) & 1u;
        while (!mbar_test_wait(bar(B_FULL + s), pp)) { if (!nosleep) __nanosleep(20); }
        par_full ^= 1u << s; slot = (s + 1 == NSLOT) ? 0 : s + 1; return s;
      };
      // QK for one tile: kpe chunk (side buffer) then 4 ring slots of 2 hd chunks
      auto qk_tile = [&](uint32_t tS, bool last) {
        while (!mbar_test_wait(bar(B_KFULL), par_kfull & 1u)) { if (!nosleep) __nanosleep(20); }
        par_kfull ^= 1u;
        FC(1);
        {
          const uint64_t qa = qdesc0 + (uint64_t)(8 * (QBLK_BYTES >> 4));
          #pragma unroll
          for (int s2 = 0; s2 < 4; ++s2) mma_ss<2>(tS, qa + 2 * s2, kpdesc + 2 * s2, idescQK, (uint32_t)(s2 != 0));
          mma_commit_mask(bar(B_KEMPTY), pair_mask);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int s = take_slot();
          FC(2 + i);
          const uint64_t kb = kdesc0 + (uint64_t)(s * (SLOT_BYTES >> 4));
          #pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const uint64_t qa = qdesc0 + (uint64_t)((2 * i + hh) * (QBLK_BYTES >> 4)), kh = kb + (uint64_t)(hh * ((2 * BLK_BYTES) >> 4));
            #pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) mma_ss<2>(tS, qa + 2 * s2, kh + 2 * s2, idescQK, 1u);
          }
          mma_commit_mask(bar(B_EMPTY + s), pair_mask);
          FC(6 + i);
        }
        if (last) mma_commit_mask(bar(B_QEMPTY), pair_mask);
        mma_commit_mask(bar(B_SFULL + (tS == tS1 ? 1 : 0)), pair_mask);
        TLT(0, 1);
      };
      // PV for tile t: 4 ring slots (kv sub-blocks), 2 chains each
      auto pv_tile = [&](int t, int ns) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (c >= ns) break;
          const int s = take_slot();
          FC(11 + c);
          const uint64_t vb = vdesc0 + (uint64_t)(s * (SLOT_BYTES >> 4)), pa = pdesc0 + (uint64_t)(c * (BLK_BYTES >> 4));
          const uint32_t acc0 = (t > 0 || c > 0) ? 1u : 0u;
          #pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint64_t vj = vb + (uint64_t)(j * ((2 * BLK_BYTES) >> 4));
            #pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) mma_ss<2>(tO + j * 128, pa + 2 * s2, vj + 128 * s2, idescPV, s2 == 0 ? acc0 : 1u);
          }
          mma_commit_mask(bar(B_EMPTY + s), pair_mask);
        }
        FC(15);
        mma_commit_mask(bar(B_PEMPTY), pair_mask);
        TLT(0, 3);
      };
      TileInfo ti_next = get_tile(P, my_tab, 0);
      for (int qt = 0; qt < my_n; ++qt) {
        TileInfo ti = ti_next; if (qt + 1 < my_n) ti_next = get_tile(P, my_tab, qt + 1);
        const int ntiles = num_kv_tiles(ti);
        while (!mbar_test_wait(bar(B_QFULL), cnt_qfull & 1)) { if (!nosleep) __nanosleep(20); }
        while (!mbar_test_wait(bar(B_QPEER), cnt_qfull & 1)) { if (!nosleep) __nanosleep(20); }
        cnt_qfull++;
        tc_fence_after();
        TLQ(0, 0);
        const int G0 = gtile;
        FC(0);
        qk_tile((G0 & 1) ? tS1 : tS0, ntiles == 1);
        for (int t = 0; t < ntiles; ++t) {
          if (t + 1 < ntiles) { FC(0); qk_tile(((G0 + t + 1) & 1) ? tS1 : tS0, t + 1 == ntiles - 1); }
          if (t == 0 && qt > 0) { while (!mbar_test_wait(bar(B_OEMPTY), cnt_oempty & 1)) { if (!nosleep) __nanosleep(20); } cnt_oempty++; }
          while (!mbar_test_wait(bar(B_PFULL), cnt_pfull & 1)) { if (!nosleep) __nanosleep(20); }
          cnt_pfull++; tc_fence_after();
          FC(10); TLT(0, 2);
          pv_tile(t, (t == ntiles - 1) ? num_pv_sub_last(ti, ntiles) : 4);
          ++gtile;
        }
        mma_commit_mask(bar(B_OFULL), pair_mask);
        TLQ(0, 1);
      }
    } else if (warp == 1 && lane == 0) {
      // ===================================================== TMA issuer: ranks 0,1 load kv half `par` for CTAs {par, par+2}
      uint32_t par_empty = 0, par_kempty = 0; int slot = 0;
      const uint16_t mask = (uint16_t)(1u << par);
      const bool mine_qk = (rank < 2), mine_pv = (rank < 2);
      int gtile = 0; uint32_t tuse = 0;
      auto next_slot = [&]() -> int {
        int s = slot; uint32_t pp = ((par_empty >> s) & 1u) ^ 1u;
        if (spin_ctl) { while (!mbar_test_wait(bar(B_EMPTY + s), pp)) { if (P.dbg_qt != -16) __nanosleep(20); } } else mbar_wait_poll(bar(B_EMPTY + s), pp);
        par_empty ^= 1u << s; SLOTEV(tuse, 4);
        slot = (s + 1 == NSLOT) ? 0 : s + 1; return s;
      };
      TileInfo ti_next = get_tile(P, my_tab, 0);
      for (int qt = 0; qt < my_n; ++qt) {
        TileInfo ti = ti_next; if (qt + 1 < my_n) ti_next = get_tile(P, my_tab, qt + 1);
        const int ntiles = num_kv_tiles(ti);
        auto load_qk = [&](int t) {
          if (!mine_qk) return;
          const int row0 = ti.kvs + t * KV_TILE + 128 * par;   // rows past this request's kv are harmless (masked)
          if (t == 0) BQ(2, 0);
          {  // kpe side buffer
            uint32_t pp = (par_kempty & 1u) ^ 1u;
            if (spin_ctl) { while (!mbar_test_wait(bar(B_KEMPTY), pp)) __nanosleep(20); } else mbar_wait_poll(bar(B_KEMPTY), pp);
            par_kempty ^= 1u;
            if (t == 0) BQ(2, 1);
            tma_tile2d_mc(sbase + SM_KPE, &maps.kpe, 0, row0, bar(B_KFULL), mask); if (!skip_tma) tma_tile2d_mc(sbase + SM_KPE + BLK_BYTES, &maps.kpe, 0, row0 + 64, bar(B_KFULL), mask);
            if (t == 0) BQ(2, 2);
          }
          #pragma unroll 1
          for (int i = 0; i < 4; ++i) {
            const int s = next_slot(); const uint32_t dst = sbase + SM_SLOT + s * SLOT_BYTES;
            if (t == 0 && i == 0) BQ(2, 3);
            if (skip_tma) { tma_tile2d_mc(dst, &maps.ckv, 64 * (2 * i), row0, bar(B_FULL + s), mask); continue; }
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              tma_tile2d_mc(dst + hh * 2 * BLK_BYTES, &maps.ckv, 64 * (2 * i + hh), row0, bar(B_FULL + s), mask);
              tma_tile2d_mc(dst + hh * 2 * BLK_BYTES + BLK_BYTES, &maps.ckv, 64 * (2 * i + hh), row0 + 64, bar(B_FULL + s), mask);
            }
            SLOTEV(tuse, 0); SLOTG(2, tuse, 6); tuse++;
          }
        };
        auto load_pv = [&](int t, int ns) {
          if (!mine_pv) return;
          #pragma unroll 1
          for (int c = 0; c < ns; ++c) {
            const int row0 = ti.kvs + t * KV_TILE + 64 * c;
            const int s = next_slot(); const uint32_t dst = sbase + SM_SLOT + s * SLOT_BYTES;
            if (skip_tma) { tma_tile2d_mc(dst, &maps.ckv, 64 * (4 * par), row0, bar(B_FULL + s), mask); continue; }
            #pragma unroll
            for (int h = 0; h < 4; ++h) tma_tile2d_mc(dst + h * BLK_BYTES, &maps.ckv, 64 * (4 * par + h), row0, bar(B_FULL + s), mask);
            SLOTEV(tuse, 0); SLOTG(2, tuse, 6); tuse++;
          }
        };
        load_qk(0);
        for (int t = 0; t < ntiles; ++t) {
          if (t + 1 < ntiles) load_qk(t + 1);
          if (t == 0 && qt > 0 && mine_qk) {   // the new q-tile's first QK loads are queued: the previous epilogue's stores may go behind them
            mbar_arrive(bar(B_LOADS));
          }
          TLT(2, 0);
          load_pv(t, (t == ntiles - 1) ? num_pv_sub_last(ti, ntiles) : 4);
          TLT(2, 1);
          ++gtile;
        }
      }
    } else if (warp == 2 && lane == 0) {
      // ===================================================== registrar (all CTAs): expect_tx per slot / kpe buffer, Q loads, peer notifications
      uint32_t par_full = 0, par_kfull = 0, cnt_qfull = 0, cnt_qempty = 0; int slot = 0; uint32_t ruse = 0;
      const uint32_t peer_full0 = bar_leader(B_FULL), peer_kfull = bar_leader(B_KFULL), peer_q = bar_leader(B_QPEER);
      int gtile = 0;
      auto wait_reg = [&](uint32_t b, uint32_t pp) { if (spin_ctl) { while (!mbar_test_wait(b, pp)) { if (P.dbg_qt != -16) __nanosleep(20); } } else mbar_wait_poll(b, pp); };
      auto reg_slot = [&]() {
        int s = slot;
        mbar_arrive_expect_tx(bar(B_FULL + s), skip_tma ? BLK_BYTES : SLOT_BYTES);
        wait_reg(bar(B_FULL + s), (par_full >> s) & 1u); par_full ^= 1u << s;
        SLOTEV(ruse, 5); SLOTG(2, ruse, 7); ruse++;
        if (!is_leader) mbar_arrive_remote(peer_full0 + s * 8);
        slot = (s + 1 == NSLOT) ? 0 : s + 1;
      };
      auto reg_kpe = [&]() {
        mbar_arrive_expect_tx(bar(B_KFULL), skip_tma ? BLK_BYTES : 2 * BLK_BYTES);
        wait_reg(bar(B_KFULL), par_kfull & 1u); par_kfull ^= 1u;
        if (!is_leader) mbar_arrive_remote(peer_kfull);
      };
      TileInfo ti_next = get_tile(P, my_tab, 0);
      for (int qt = 0; qt < my_n; ++qt) {
        TileInfo ti = ti_next; if (qt + 1 < my_n) ti_next = get_tile(P, my_tab, qt + 1);
        const int ntiles = num_kv_tiles(ti);
        // ---- Q for my 64 rows (issued for qt = 0 here; for qt > 0 it was prefetched during the previous q-tile's last PV slots)
        auto issue_q = [&](const TileInfo& tq) {
          int row0 = (tq.qs + tq.p0 + 8 * (rank >> 1) + QPOS * par) * H;
          mbar_arrive_expect_tx(bar(B_QFULL), 9 * QBLK_BYTES);
          for (int c = 0; c < 8; ++c) tma_tile2d(sbase + SM_Q + c * QBLK_BYTES, &maps.qn, c * 64, row0, bar(B_QFULL));
          tma_tile2d(sbase + SM_Q + 8 * QBLK_BYTES, &maps.qp, 0, row0, bar(B_QFULL));
        };
        if (qt == 0) issue_q(ti);
        wait_reg(bar(B_QFULL), cnt_qfull & 1); cnt_qfull++;
        TLQ(3, 0); BQ(3, 2);
        if (!is_leader) mbar_arrive_remote(peer_q);
        reg_kpe(); BQ(3, 3); reg_slot(); BQ(3, 4); for (int i = 1; i < 4; ++i) reg_slot();
        for (int t = 0; t < ntiles; ++t) {
          if (t + 1 < ntiles) { reg_kpe(); for (int i = 0; i < 4; ++i) reg_slot(); } TLT(3, 0);
          if (t + 1 == ntiles && qt + 1 < my_n) {   // all QK slots of this q-tile registered: prefetch the next Q as soon as QK_{n-1} is done with it
            wait_reg(bar(B_QEMPTY), cnt_qempty & 1); cnt_qempty++;
            BQ(3, 0);
            issue_q(ti_next);
            BQ(3, 1);
          }
          { const int ns = (t == ntiles - 1) ? num_pv_sub_last(ti, ntiles) : 4; for (int c = 0; c < ns; ++c) reg_slot(); } TLT(3, 1);
          ++gtile;
        }
      }
    } else if (warp == 3 && lane == 0) {
      // ===================================================== warp 3: P forwarder (peer CTA: local P complete -> arrive on the leader)
      //                                                       and epilogue store issuer (all CTAs: TMA-store the staged O half)
      uint32_t cnt = 0, cnt_staged = 0; const uint32_t pfull_leader = bar_leader(B_PFULL);
      TileInfo ti_next = get_tile(P, my_tab, 0);
      for (int qt = 0; qt < my_n; ++qt) {
        TileInfo ti = ti_next; if (qt + 1 < my_n) ti_next = get_tile(P, my_tab, qt + 1);
        const int ntiles = num_kv_tiles(ti);
        if (!is_leader) {
          for (int t = 0; t < ntiles; ++t) {
            { const uint32_t pp = cnt & 1; if (spin_ctl) { while (!mbar_test_wait(bar(B_PLOCAL), pp)) __nanosleep(20); } else mbar_wait_sleep(bar(B_PLOCAL), pp); }
            cnt++; SLOTG(3, cnt, 0); if (P.dbg_qt == -14) mbar_arrive_remote_release(pfull_leader); else mbar_arrive_remote(pfull_leader); SLOTG(3, cnt, 1);
          }
        }
        // epilogue stores for this q-tile
        { const uint32_t pp = cnt_staged & 1; while (!mbar_test_wait(bar(B_STAGED), pp)) __nanosleep(32); cnt_staged++; }
        if (qt + 1 < my_n && !getenv_nohold) { const uint32_t pp = qt & 1; while (!mbar_test_wait(bar(B_LOADS), pp)) __nanosleep(32); }
        const int pos0 = ti.p0 + 8 * (rank >> 1) + QPOS * par;
        const int nvalid = min(QPOS, ti.Lq - pos0);
        if (nvalid > 0) {
          const int row0 = (ti.qs + pos0) * H;
          const CUtensorMap* om = &maps.out[nvalid - 1];
          #pragma unroll
          for (int h = 0; h < 4; ++h) tma_store2d(om, sbase + SM_P + h * 8192, h * 64, row0);
          bulk_commit();
          bulk_wait_read0();
        }
        mbar_arrive(bar(B_STAGEFREE));
      }
      bulk_wait_all();
    }
  } else {
    // ===================================================== softmax / rescale / epilogue (8 warps: 2 per TMEM lane quadrant)
    const int quad = warp & 3;                       // hardware TMEM lane quadrant of this warp (warpid % 4)
    const int slice = (warp - NUM_CTRL_WARPS) >> 2;  // 0..1 : 64-column slice of S / 128-column slice of O
    const int rowlocal = 32 * (quad & 1) + lane;     // 0..63
    const int khalf = quad >> 1;                     // kv half (S) / hd half (O) of this quadrant
    const uint32_t lane_base = (uint32_t)(32 * quad) << 16;
    const int redslot = khalf * 2 + slice;           // 4 partial stats per row
    uint32_t par_sfull = 0, cnt_pempty = 0, cnt_ofull = 0;
    int gtile = 0;
    const uint32_t oempty_leader = bar_leader(B_OEMPTY);
    const uint32_t pbar = is_leader ? bar(B_PFULL) : bar(B_PLOCAL);
    // P destination: block kb = kv/64 = 2*khalf + slice, row rowlocal, 8 x 16B chunks (swizzled by row)
    const uint32_t p_row = sbase + SM_P + (2 * khalf + slice) * BLK_BYTES + rowlocal * 128;
    auto wait_sm = [&](uint32_t b, uint32_t pp) { if (spin_all) { while (!mbar_test_wait(b, pp)) __nanosleep(20); } else mbar_wait_sleep(b, pp); };
    TileInfo ti_next = get_tile(P, my_tab, 0);
    for (int qt = 0; qt < my_n; ++qt) {
      TileInfo ti = ti_next; if (qt + 1 < my_n) ti_next = get_tile(P, my_tab, qt + 1);
      const int ntiles = num_kv_tiles(ti);
      const int qpos = ti.p0 + 8 * (rank >> 1) + QPOS * par + (rowlocal >> 4);
      const int limit = qpos + (ti.Lk - ti.Lq);
      float m_stale = -INFINITY, l_part = 0.f;
      for (int t = 0; t < ntiles; ++t) {
        { const int sb = gtile & 1; wait_sm(bar(B_SFULL + sb), (par_sfull >> sb) & 1u); par_sfull ^= 1u << sb; }
        WPROG(0); SMEV(0);
        if (tid == 32 * NUM_CTRL_WARPS) TLT(1, 0);
        tc_fence_after();
        if (skip_sm) {
          if (!(qt == 0 && t == 0)) { wait_sm(bar(B_PEMPTY), cnt_pempty & 1); cnt_pempty++; }
          if (tid == 32 * NUM_CTRL_WARPS) TLT(1, 3);
          if (lane == 0) mbar_arrive(pbar);
          ++gtile; continue;
        }
        const uint32_t tS = (gtile & 1) ? tS1 : tS0;
        uint32_t s[SUB_S];
        SMEVC(1);
        tmem_ld32(tS + lane_base + slice * SUB_S, s);
        tmem_ld32(tS + lane_base + slice * SUB_S + 32, s + 32);
        tc_wait_ld();
        SMEVC(2);
        const int kvbase = t * KV_TILE + khalf * 128 + slice * SUB_S;
        const bool need_mask = (t * KV_TILE + KV_TILE - 1 > limit);
        float rmax = -INFINITY;
        if (need_mask) {
          #pragma unroll
          for (int c = 0; c < SUB_S; ++c) { float x = __uint_as_float(s[c]); if (kvbase + c > limit) x = -INFINITY; s[c] = __float_as_uint(x); rmax = fmaxf(rmax, x); }
        } else {
          #pragma unroll
          for (int c = 0; c < SUB_S; ++c) rmax = fmaxf(rmax, __uint_as_float(s[c]));
        }
        redm[redslot * 64 + rowlocal] = rmax;
        named_bar_sync(1, SOFTMAX_THREADS);
        WPROG(1); SMEV(3);
        float tmax = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 4; ++i) tmax = fmaxf(tmax, redm[i * 64 + rowlocal]);
        const float m_new = fmaxf(m_stale, tmax * scale_log2);
        const bool need = (t > 0) && (m_new - m_stale > RESCALE_THRESH);
        if (t == 0) m_stale = m_new;
        if (tid == 32 * NUM_CTRL_WARPS) TLT(1, 1);
        // ---- exponentials first (into packed registers), so the PEMPTY wait below overlaps the math
        float alpha = 1.f;
        if (need) { alpha = ex2(m_stale - m_new); m_stale = m_new; l_part *= alpha; }
        uint32_t pk[SUB_S / 2];
        SMEVC(6);
        {
          float rsum = 0.f;
          #pragma unroll
          for (int c = 0; c < SUB_S; c += 2) {
            const float p0 = ex2(fmaf(__uint_as_float(s[c]), scale_log2, -m_stale)), p1 = ex2(fmaf(__uint_as_float(s[c + 1]), scale_log2, -m_stale));
            rsum += p0 + p1;
            __nv_bfloat162 v = __floats2bfloat162_rn(p0, p1); pk[c >> 1] = *(uint32_t*)&v;
          }
          l_part += rsum;
        }
        // ---- previous PV finished reading P and writing O
        if (!(qt == 0 && t == 0)) { wait_sm(bar(B_PEMPTY), cnt_pempty & 1); cnt_pempty++; }
        WPROG(2); SMEV(4);
        if (tid == 32 * NUM_CTRL_WARPS) TLT(1, 2);
        if (__any_sync(0xffffffffu, need)) {   // warp-uniform branch (tcgen05.ld/st are .aligned); rows without need use alpha = 1
          tc_fence_after();
          #pragma unroll 1
          for (int cb = 0; cb < 4; ++cb) {
            uint32_t o[32];
            const uint32_t ta = tO + lane_base + slice * SUB_O + cb * 32;
            tmem_ld32(ta, o); tc_wait_ld();
            #pragma unroll
            for (int c = 0; c < 32; ++c) o[c] = __float_as_uint(__uint_as_float(o[c]) * alpha);
            tmem_st32(ta, o);
          }
          tc_wait_st();
          tc_fence_before();
        }
        if (t == 0 && qt > 0) {   // the previous epilogue's TMA store must have finished reading the P buffer
          wait_sm(bar(B_STAGEFREE), (qt - 1) & 1);
        }
        #pragma unroll
        for (int c8 = 0; c8 < SUB_S / 8; ++c8)
          st_shared_v4(p_row + ((c8 ^ (rowlocal & 7)) << 4), pk[4 * c8], pk[4 * c8 + 1], pk[4 * c8 + 2], pk[4 * c8 + 3]);
        fence_proxy_async();
        __syncwarp();
        WPROG(3); SMEV(5); SMEVC(7);
        if (tid == 32 * NUM_CTRL_WARPS) TLT(1, 3);
        if (lane == 0) mbar_arrive(pbar);
        ++gtile;
      }
      // ---- epilogue: wait final PV, normalise, store
      EPI(0);
      wait_sm(bar(B_OFULL), cnt_ofull & 1); cnt_ofull++;
      EPI(1);
      WPROG(4);
      if (tid == 32 * NUM_CTRL_WARPS) TLQ(1, 0);
      tc_fence_after();
      redl[redslot * 64 + rowlocal] = l_part;
      named_bar_sync(2, SOFTMAX_THREADS);
      EPI(2);
      float l = 0.f;
      #pragma unroll
      for (int i = 0; i < 4; ++i) l += redl[i * 64 + rowlocal];
      const float inv = 1.f / l;
      const bool rvalid = (qpos < ti.Lq);
      const int head = rowlocal & 15;
      const size_t grow = (size_t)(ti.qs + qpos) * H + head;
      if (rvalid && redslot == 0) P.lse[grow] = (m_stale + lg2(l)) * 0.6931471805599453f;
      EPI(3);
      // hd half 0 (quads 0,1): O -> bf16 -> P buffer as 4 SW128 boxes [64 rows][64 cols]; warp 3 TMA-stores them asynchronously
      // (read completion is awaited right before the next q-tile's first P store). hd half 1 (quads 2,3): stored directly with
      // shuffle-transposed, sector-aligned st.global.v4. O layout: lanes 0..63 hold hd 0..255, lanes 64..127 hold hd 256..511.
      const uint32_t stage = sbase + SM_P;
      const int pos0 = ti.p0 + 8 * (rank >> 1) + QPOS * par;
      #pragma unroll 1
      for (int cbp = 0; cbp < 2; ++cbp) {
        uint32_t packed[32];
        {
          uint32_t o[64];
          tmem_ld32(tO + lane_base + slice * SUB_O + (2 * cbp) * 32, o);
          tmem_ld32(tO + lane_base + slice * SUB_O + (2 * cbp + 1) * 32, o + 32);
          tc_wait_ld();
          #pragma unroll
          for (int c = 0; c < 32; ++c) {
            __nv_bfloat162 v = __floats2bfloat162_rn(__uint_as_float(o[2 * c]) * inv, __uint_as_float(o[2 * c + 1]) * inv); packed[c] = *(uint32_t*)&v;
          }
        }
        if (khalf == 0) {
          #pragma unroll
          for (int cb2 = 0; cb2 < 2; ++cb2) {
            const int col = slice * SUB_O + (2 * cbp + cb2) * 32;   // 0..255 within the half
            const int h = col >> 6, c0 = (col & 63) >> 3;
            const uint32_t box = stage + h * 8192 + rowlocal * 128;
            #pragma unroll
            for (int q = 0; q < 4; ++q) st_shared_v4(box + (((c0 + q) ^ (rowlocal & 7)) << 4), packed[16 * cb2 + 4 * q], packed[16 * cb2 + 4 * q + 1], packed[16 * cb2 + 4 * q + 2], packed[16 * cb2 + 4 * q + 3]);
          }
        } else {
          // store instruction kq: lane L writes row 8kq + (L>>2) of this warp, 16B chunk (L&3) of each 32-column block
          const int j = lane & 3;
          #pragma unroll
          for (int cb2 = 0; cb2 < 2; ++cb2) {
            #pragma unroll
            for (int kq = 0; kq < 4; ++kq) {
              const int src = 8 * kq + (lane >> 2);
              uint32_t sh[16];
              #pragma unroll
              for (int m = 0; m < 16; ++m) sh[m] = __shfl_sync(0xffffffffu, packed[16 * cb2 + m], src);
              uint32_t v0 = sh[0], v1 = sh[1], v2 = sh[2], v3 = sh[3];
              if (j == 1) { v0 = sh[4]; v1 = sh[5]; v2 = sh[6]; v3 = sh[7]; }
              if (j == 2) { v0 = sh[8]; v1 = sh[9]; v2 = sh[10]; v3 = sh[11]; }
              if (j == 3) { v0 = sh[12]; v1 = sh[13]; v2 = sh[14]; v3 = sh[15]; }
              const int rowr = 32 * (quad & 1) + src;          // 0..63
              const int posr = pos0 + (rowr >> 4);
              if (posr < ti.Lq) {
                const size_t growr = (size_t)(ti.qs + posr) * H + (rowr & 15);
                *(uint4*)(P.out + growr * D_CKV + 256 + slice * SUB_O + (2 * cbp + cb2) * 32 + j * 8) = make_uint4(v0, v1, v2, v3);
              }
            }
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_generic(oempty_leader, is_leader ? 0u : 1u);   // O fully read: the next q-tile's PV may start
      EPI(4);
      if (khalf == 0) { fence_proxy_async(); __syncwarp(); if (lane == 0) mbar_arrive(bar(B_STAGED)); }
      EPI(5);
      EPI(12);
      if (tid == 32 * NUM_CTRL_WARPS) TLQ(1, 1);
    }
  }
#ifdef MLA_TIMELINE
  if (P.dbg && (tid == 0 || tid == 32 || tid == 64 || tid == 96)) ((unsigned long long*)P.dbg)[3 * 8192 + 7200 + 4 * blockIdx.x + tid / 32] = gtimer();
#endif
  __syncthreads();
#ifdef MLA_TIMELINE
  if (P.dbg && tid == 0) ((unsigned long long*)P.dbg)[3 * 8192 + 7800 + blockIdx.x] = gtimer();
#endif
  cluster_sync();
  if (warp == 0) tmem_dealloc<2>(tmem, 512);
}
// Prologue: gather the paged rows of every request into contiguous [total_kv][512] / [total_kv][64] buffers.
__global__ void __launch_bounds__(256) gather_kv_kernel(const uint4* __restrict__ ckv, const uint4* __restrict__ kpe, const int* __restrict__ kv_indices,
                                                        uint4* __restrict__ kc, uint4* __restrict__ kp, int total_kv) {
  // 72 x 16B per row: 64 for ckv, 8 for kpe
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  size_t n = (size_t)total_kv * 72;
  if (i >= n) return;
  int row = (int)(i / 72), c = (int)(i % 72);
  int page = kv_indices[row];
  if (c < 64) kc[(size_t)row * 64 + c] = ckv[(size_t)page * 64 + c];
  else kp[(size_t)row * 8 + (c - 64)] = kpe[(size_t)page * 8 + (c - 64)];
}
}  // namespace mla

static PFN_cuTensorMapEncodeTiled_v12000 get_encoder() {
  return &cuTensorMapEncodeTiled;
}
static int make_map(CUtensorMap* m, const void* base, uint64_t rows, uint64_t cols, uint32_t box_cols, uint32_t box_rows) {
  cuuint64_t gdim[2] = {cols, rows}; cuuint64_t gstr[1] = {cols * 2};
  cuuint32_t box[2] = {box_cols, box_rows}; cuuint32_t es[2] = {1, 1};
  return (int)get_encoder()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim, gstr, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int mla_prefill_launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe, const int* qo_indptr,
                                  const int* kv_indptr, const int* kv_indices, void* out, float* lse, float sm_scale, int B,
                                  int total_q, int num_pages, int nclusters, cudaStream_t stream, float* dbg, int dbg_qt) {
  using namespace mla;
  Maps maps;
  int e = 0;
  // materialise contiguous K/V (scratch buffers are cached across calls; sized for total_kv rows)
  static const int* cache_qo = nullptr; static const int* cache_kv = nullptr; static int cache_B = -1, cache_tq = -1, cache_total_kv = 0;
  const bool cache_hit = (cache_qo == qo_indptr && cache_kv == kv_indptr && cache_B == B && cache_tq == total_q && !getenv("MLA_NOCACHE"));
  int total_kv = cache_total_kv;
  if (!cache_hit) { cudaMemcpyAsync(&total_kv, kv_indptr + B, 4, cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream); }
  static void* kc = nullptr; static void* kp = nullptr; static int kc_rows = 0;
  int need_rows = total_kv + KV_TILE;  // slack so tiles past the end stay in-bounds of the tensor map (zero-filled OOB otherwise)
  if (kc_rows < need_rows) { if (kc) { cudaFree(kc); cudaFree(kp); } cudaMalloc(&kc, (size_t)need_rows * 1024); cudaMalloc(&kp, (size_t)need_rows * 128); kc_rows = need_rows; }
  {
    size_t n = (size_t)total_kv * 72; int grid = (int)((n + 255) / 256);
    gather_kv_kernel<<<grid, 256, 0, stream>>>((const uint4*)ckv, (const uint4*)kpe, kv_indices, (uint4*)kc, (uint4*)kp, total_kv);
  }
  e |= make_map(&maps.ckv, kc, (uint64_t)need_rows, 512, 64, BLK_ROWS);
  e |= make_map(&maps.kpe, kp, (uint64_t)need_rows, 64, 64, BLK_ROWS);
  e |= make_map(&maps.qn, q_nope, (uint64_t)total_q * H, 512, 64, 64);
  e |= make_map(&maps.qp, q_pe, (uint64_t)total_q * H, 64, 64, 64);
  for (int i = 0; i < 4; ++i) e |= make_map(&maps.out[i], out, (uint64_t)total_q * H, 512, 64, 16 * (i + 1));
  if (e) return 1000 + e;
  static int max_clusters = 0;
  if (max_clusters == 0) {
    cudaLaunchConfig_t qcfg = {}; qcfg.blockDim = NUM_THREADS; qcfg.dynamicSmemBytes = SM_TOTAL + 1024; qcfg.gridDim = CLUSTER * 80;
    cudaLaunchAttribute qat[1]; qat[0].id = cudaLaunchAttributeClusterDimension; qat[0].val.clusterDim = {CLUSTER, 1, 1}; qcfg.attrs = qat; qcfg.numAttrs = 1;
    cudaFuncSetAttribute(mla_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SM_TOTAL + 1024));
    if (cudaOccupancyMaxActiveClusters(&max_clusters, mla_prefill_kernel, &qcfg) != cudaSuccess || max_clusters <= 0) max_clusters = 32;
    if (getenv("MLA_VERBOSE")) fprintf(stderr, "max active clusters: %d\n", max_clusters);
  }
  Params p;
  p.q_nope = (const __nv_bfloat16*)q_nope; p.q_pe = (const __nv_bfloat16*)q_pe; p.ckv = (const __nv_bfloat16*)ckv;
  p.kpe = (const __nv_bfloat16*)kpe; p.qo_indptr = qo_indptr; p.kv_indptr = kv_indptr; p.kv_indices = kv_indices;
  p.out = (__nv_bfloat16*)out; p.lse = lse; p.sm_scale = sm_scale; p.B = B; p.total_q = total_q; p.nclusters = nclusters; p.dbg = dbg; p.dbg_qt = dbg_qt;
  p.epi = nullptr;
  // ---- host-side schedule (LPT over cluster tiles, cost = kv tiles + boundary cost), cached across identical calls
  static int* dsched = nullptr; static size_t dsched_cap = 0; static int cache_stride = 0, cache_ncl = 0;
  if (cache_hit) { p.sched = dsched; p.sched_stride = cache_stride; nclusters = cache_ncl; }
  else {
    std::vector<int> hq(B + 1), hk(B + 1);
    cudaMemcpyAsync(hq.data(), qo_indptr, (B + 1) * 4, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(hk.data(), kv_indptr, (B + 1) * 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<int> ord(B); for (int i = 0; i < B; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { int La = hk[a + 1] - hk[a], Lb = hk[b + 1] - hk[b]; return La > Lb; });
    std::vector<std::pair<int, int>> tiles;   // (cost, g)
    int g = 0;
    for (int b : ord) {
      int Lq = hq[b + 1] - hq[b], Lk = hk[b + 1] - hk[b]; int nq = (Lq + CL_POS - 1) / CL_POS;
      for (int i = 0; i < nq; ++i, ++g) { int p0 = (nq - 1 - i) * CL_POS; int plast = std::min(p0 + CL_POS - 1, Lq - 1); int kv_end = plast + (Lk - Lq) + 1; int nt = (kv_end + KV_TILE - 1) / KV_TILE; int ns = ((kv_end - KV_TILE * (nt - 1)) + 63) / 64; tiles.push_back({10 * nt - (4 - ns), g}); }
    }
    std::stable_sort(tiles.begin(), tiles.end(), [](auto& a, auto& b) { return a.first > b.first; });
    int ncl = max_clusters; if (nclusters > 0 && nclusters < ncl) ncl = nclusters;
    std::vector<int> load(ncl, 0); std::vector<std::vector<int>> lists(ncl);
    int qcost = getenv("MLA_QCOST") ? atoi(getenv("MLA_QCOST")) : 14;   // boundary cost in tenths of a kv tile
    for (auto& tg : tiles) { int c = 0; for (int i = 1; i < ncl; ++i) if (load[i] < load[c]) c = i; load[c] += tg.first + qcost; lists[c].push_back(tg.second); }
    int maxn = 0; for (auto& l : lists) maxn = std::max(maxn, (int)l.size());
    int stride = maxn + 1;
    stride = 8 + 8 * maxn;
    std::vector<int> hs((size_t)ncl * stride, 0);
    // tile g -> (b, p0) in the same enumeration order used to build 'tiles'
    std::vector<std::pair<int, int>> gmap; gmap.reserve(g);
    for (int b : ord) { int Lq = hq[b + 1] - hq[b]; int nq = (Lq + CL_POS - 1) / CL_POS; for (int i = 0; i < nq; ++i) gmap.push_back({b, (nq - 1 - i) * CL_POS}); }
    for (int c = 0; c < ncl; ++c) {
      hs[(size_t)c * stride] = (int)lists[c].size();
      for (size_t i = 0; i < lists[c].size(); ++i) {
        int gi = lists[c][i]; int b = gmap[gi].first, p0 = gmap[gi].second;
        int* e = &hs[(size_t)c * stride + 8 + 8 * i];
        int Lq = hq[b + 1] - hq[b], Lk = hk[b + 1] - hk[b];
        int plast = std::min(p0 + CL_POS - 1, Lq - 1); int kv_end = plast + (Lk - Lq) + 1;
        e[0] = b; e[1] = p0; e[2] = Lq; e[3] = Lk; e[4] = hq[b]; e[5] = hk[b]; e[6] = (kv_end + KV_TILE - 1) / KV_TILE; e[7] = 0;
      }
    }
    if (dsched_cap < hs.size()) { if (dsched) cudaFree(dsched); cudaMalloc(&dsched, hs.size() * 4); dsched_cap = hs.size(); }
    cudaMemcpy(dsched, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    p.sched = dsched; p.sched_stride = stride; nclusters = ncl;
    cache_qo = qo_indptr; cache_kv = kv_indptr; cache_B = B; cache_tq = total_q; cache_total_kv = total_kv; cache_stride = stride; cache_ncl = ncl;
    if (getenv("MLA_VERBOSE")) { int mx = 0, sm = 0; for (int v : load) { mx = std::max(mx, v); sm += v; } fprintf(stderr, "schedule: %d clusters, max load %d avg %.1f\n", ncl, mx, (double)sm / ncl); }
  }
  size_t smem = SM_TOTAL + 1024;
  static bool attr_set = false;
  if (!attr_set) { cudaFuncSetAttribute(mla_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr_set = true; }
  cudaLaunchConfig_t cfg = {}; cfg.blockDim = NUM_THREADS; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim = {CLUSTER, 1, 1};
  cfg.attrs = at; cfg.numAttrs = 1;
  p.nclusters = nclusters;
  cfg.gridDim = nclusters * CLUSTER;
  return (int)cudaLaunchKernelEx(&cfg, mla_prefill_kernel, maps, p);
}
