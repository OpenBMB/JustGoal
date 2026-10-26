// From-scratch causal varlen MHA backward for Blackwell (sm_100a), bf16, D=128, layout (T,H,D).
//
// Work item = (doc, head, 128-key KV block). A persistent CTA (1 per SM) loops over its items.
// Per item: K,V are TMA-loaded to smem (SS A operands). Q/dO are streamed in 64-row sub-tiles. For each sub-tile g:
//   S'^T = [K | 1 1 0..] [Q | -lse/scale (hi,lo) 0..]^T  and  dP'^T = [V | 1 1 0..] [dO | -rowsum(dO*O) (hi,lo) 0..]^T
//   (SS, N=64, 9 k-steps: the 9th folds the per-row lse / delta subtraction into the MMA, so the softmax warps need no
//   per-column loads at all). S'^T and dP'^T are computed one after the other into the same 64-col TMEM region R[g&1].
//   The two softmax WGs ping-pong: WG b handles the sub-tiles with g&1 == b (all 64 q columns), so one WG is in its MUFU
//   (exp2) phase while the other packs / stores. P = exp2(S' * scale * log2e) (masked), dS = P * (dP' * scale);
//   P^T / dS^T -> TMEM buffer B[b] (bf16) and dS^T -> smem[b]; dV += P^T dO, dK += dS^T Q (TS, A from TMEM, N=128);
//   dQ^T = K^T dS^T (SS, N=64) accumulates into B[b] after dV/dK consumed it -> f16x2 red.global.add into dq_accum.
// TMEM columns: [0,128) dK  [128,256) dV  [256,384) R[b] = 256 + 64b: S'^T then dP'^T of sub-tile g (b = g&1)
//   [384,512) B[b] = 384 + 64b: P^T cols 0-31 | dS^T cols 32-63, then dQ^T (fp32, 64 cols) of the same sub-tile.
#include "sm100_utils.cuh"
#include <cuda_fp16.h>
#include <string.h>
#include <math_constants.h>
#ifndef NO_NAMED_BARS
#define NAMED_BARS 1
#endif
using namespace cute;
using namespace k;

namespace fa4 {

constexpr int D = 128;
constexpr int KV_BLK = 128;
constexpr int Q_SUB = 64;   // q sub-tile: 64 rows (both softmax WGs work on the same sub-tile, 32 columns each)
constexpr int QSTAGES = 4, DOSTAGES = 3;   // separate Q and dO rings (64 q rows = 16 KB + 2 KB aug slice per stage)
constexpr int NUM_THREADS = 512;   // warps 0-3: dQ readout WG (warp 0 allocs TMEM), 4-7: softmax WG0, 8-11: WG1,
                                   // 12: TMA producer, 13: MMA issuer A1 (S'^T), 14: issuer B (dV/dK then dQ^T), 15: issuer A2 (dP'^T)
constexpr int REGS_SOFTMAX = 168, REGS_OTHER = 64, REGS_DQ = 112;   // 4*(168+168+112+64) = 2048 regs/lane-slot
constexpr int ITEM_RING = 4;       // dynamic work items broadcast from the producer to the other roles

// TMEM column offsets
constexpr uint32_t T_DK = 0, T_DV = 128, T_R = 256, T_B = 384;

// smem byte offsets (all 1024-aligned)
constexpr uint32_t SM_K = 0;                        // 32 KB (K: A operand of S^T (K-major) and of dQ^T (MN-major))
constexpr uint32_t SM_V = SM_K + 32768;             // 32 KB (V: A operand of dP^T)
constexpr uint32_t SM_Q = SM_V + 32768;             // QSTAGES x 16 KB
constexpr uint32_t SM_DO = SM_Q + QSTAGES * 16384;  // DOSTAGES x 16 KB
constexpr uint32_t SM_DS = SM_DO + DOSTAGES * 16384;// 2 x 16 KB  dS^T [128 keys][64 q] bf16, SW128 rows (sub-tile parity = WG)
constexpr uint32_t SM_ONES = SM_DS + 2 * 16384;     // 4 KB: [128 rows][16] bf16 SW32, (1, 1, 0...) per row: 9th k-step A operand
constexpr uint32_t SM_QA = SM_ONES + 4096;          // QSTAGES x 2 KB: Q' slice [64 q][16] bf16 SW32 (TMA)
constexpr uint32_t SM_DOA = SM_QA + QSTAGES * 2048; // DOSTAGES x 2 KB: dO' slice
constexpr uint32_t SM_TOTAL = SM_DOA + DOSTAGES * 2048;   // 226 KB (+ 1 KB static == the 227 KB limit)
constexpr uint32_t SMEM_BYTES = SM_TOTAL;   // smem_raw is declared 1024-aligned (checked at kernel start)

struct Params {
  const __nv_bfloat16* q; const __nv_bfloat16* kk; const __nv_bfloat16* v; const __nv_bfloat16* dout;
  const __nv_bfloat16* qaug;    // (H, Tp, 16) bf16: [-lse/scale (hi), (lo), 0 x 14] per padded q row
  const __nv_bfloat16* doaug;   // (H, Tp, 16) bf16: [-rowsum(dO*O) (hi), (lo), 0 x 14]
  __half* dq_accum;     // (T, H, D) fp16 scratch, zeroed by the preprocess kernel; receives scale * dQ by f16x2 reduce-add
  __nv_bfloat16* dq; __nv_bfloat16* dk; __nv_bfloat16* dv;
  const int* cu_seqlens; int n_docs; int T; int H; int Tp;   // Tp = row count of the padded layout
  const int* pad_start; // per doc: 64-aligned start of its padded rows (computed by schedule_kernel)
  const int* n_pairs;   // device scalar
  const int* items;     // pairs doc | (n << 16): docs longest-first, kv blocks ascending (dq_accum locality + LPT-ish)
  int* counter;         // dynamic item counter (zeroed by the schedule kernel)
  float scale_log2e; float scale;
  CUtensorMap tm_q, tm_k, tm_v, tm_do, tm_qaug, tm_doaug, tm_dk, tm_dv;
};

using LayQk  = SmemLayoutK<Q_SUB, D>;        // Q/dO sub-tile as K-major [N=q(64)][K=d]
using LayQmn = SmemLayoutMN<D, Q_SUB>;       // same memory as MN-major [N=d][K=q(64)]
using LayKmn = SmemLayoutMN<D, KV_BLK>;      // K tile as MN-major A [M=d][K=key]
using LayVk  = SmemLayoutK<KV_BLK, D>;       // K/V tile as K-major A [M=key][K=d]
using LayDSmn = SmemLayoutMN<64, KV_BLK>;    // dS^T tile as MN-major B [N=q(64)][K=key]
using LayOnes = SmemLayoutK32<KV_BLK, 16>;   // ones slice as K-major A [M=key][K=16]
using LayAug  = SmemLayoutK32<Q_SUB, 16>;    // Q'/dO' slice as K-major B [N=q][K=16]

__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// bf16x2 pack by integer round-half-up on the magnitude bits (ties away from zero: negligible bias; finite inputs only).
// Avoids F2FP (4 cycles/warp-instr on the conversion pipe); IADD + PRMT run on the ALU pipes instead.
__device__ __forceinline__ uint32_t pack_bf16_rn(float lo, float hi) {
  const uint32_t a = __float_as_uint(lo) + 0x8000u, b = __float_as_uint(hi) + 0x8000u;
  return __byte_perm(a, b, 0x7632);
}

#ifdef DEBUG_WAIT
__device__ __forceinline__ void mbar_wait_dbg(uint64_t* bar, uint32_t phase, int line) {
  long long n = 0;
  while (!mbar_try_wait(bar, phase)) {
    if (++n == (1ll << 22) && (threadIdx.x & 31) == 0) { printf("HANG block %d tid %d line %d phase %u\n", blockIdx.x, threadIdx.x, line, phase); }
  }
}
#define MBW(bar, phase) mbar_wait_dbg(bar, phase, __LINE__)
#else
#define MBW(bar, phase) mbar_wait(bar, phase)
#endif

#ifdef TRACE
// Trace: 32-bit SM clock per (sub-tile, event) for sub-tiles [TRACE_G0, TRACE_G0+TRACE_W) of CTA 0, kept in a small
// static-smem ring (st.shared does not queue behind the dQ WG's global reduce-adds) and copied to g_trace at kernel end.
__device__ long long* g_trace = nullptr;   // [TRACE_N][32]
constexpr int TRACE_N = 2048, TRACE_W = 4;
#ifndef TRACE_G0
#define TRACE_G0 300
#endif
#define TR(g, e) do { if (trace_on && (g) >= TRACE_G0 && (g) < TRACE_G0 + TRACE_W) { uint32_t _c; asm volatile("mov.u32 %0, %%clock;" : "=r"(_c)); trace_ring[(g) - TRACE_G0][(e)] = _c; } } while (0)
#else
#define TR(g, e) do {} while (0)
#endif

struct Item { int doc, n, h, s0, L, kv0, j0, j1, ps; };
__device__ __forceinline__ Item decode_item(const Params& p, int i) {
  Item it;
  int pair = p.items[i / p.H];
  it.h = i % p.H;
  it.doc = pair & 0xffff; it.n = pair >> 16;
  it.s0 = p.cu_seqlens[it.doc];
  it.L = p.cu_seqlens[it.doc + 1] - it.s0;
  it.kv0 = it.n * KV_BLK;
  it.j0 = it.n * (KV_BLK / Q_SUB);
  it.j1 = (it.L + Q_SUB - 1) / Q_SUB - 1;
  it.ps = p.pad_start[it.doc];
  return it;
}

__global__ void __launch_bounds__(NUM_THREADS, 1) fa4_bwd_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~uintptr_t(1023));
  if (smem != smem_raw) __trap();   // layout assumes the dynamic smem base is 1024-aligned
  // Per-parity barriers (index b = g&1 = owning softmax WG): a WG observes only every other sub-tile and an mbarrier
  // parity wait tolerates at most a one-phase lag.
  __shared__ __align__(8) uint64_t bar_kv_full, bar_kv_empty, bar_v_full, bar_v_empty, bar_q_full[QSTAGES], bar_q_empty[QSTAGES],
      bar_do_full[DOSTAGES], bar_do_empty[DOSTAGES],
      bar_S[2], bar_dP[2], bar_spS[2], bar_spdP[2], bar_p_ready[2], bar_pds_ready[2], bar_dq_done[2], bar_dq_free[2], bar_acc_done, bar_acc_free[2], bar_epi_done,
      bar_item_full[ITEM_RING], bar_item_empty[ITEM_RING];
  __shared__ int item_ring[ITEM_RING];
  __shared__ uint32_t tmem_base_s;
#ifdef TRACE
  __shared__ uint32_t trace_ring[TRACE_W][32];
  const bool trace_on = (blockIdx.x == 0 || (*p.n_pairs) * p.H == 1);
  for (int i = threadIdx.x; i < TRACE_W * 32; i += NUM_THREADS) (&trace_ring[0][0])[i] = 0u;
#endif

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int H = p.H;
  const int n_items = (*p.n_pairs) * H;

  if (warp == 0) { tmem_alloc(&tmem_base_s, 512); tmem_relinquish(); }
  if (tid == 0) {
    mbar_init(&bar_kv_full, 1); mbar_init(&bar_kv_empty, 2);   // K: freed by A1's (last S^T) and B's (last dQ^T) commits
    mbar_init(&bar_v_full, 1); mbar_init(&bar_v_empty, 1);      // V: freed by A2's commit after the item's last dP^T
    for (int s = 0; s < QSTAGES; ++s) { mbar_init(&bar_q_full[s], 1); mbar_init(&bar_q_empty[s], 1); }     // Q empty: B's commit after dK
    for (int s = 0; s < DOSTAGES; ++s) { mbar_init(&bar_do_full[s], 1); mbar_init(&bar_do_empty[s], 1); }  // dO empty: B's commit after dV
    for (int b = 0; b < 2; ++b) {
      mbar_init(&bar_S[b], 1); mbar_init(&bar_dP[b], 1);         // S'^T / dP'^T of a sub-tile of parity b complete (A1/A2's commits)
      mbar_init(&bar_spS[b], 4); mbar_init(&bar_spdP[b], 4);     // WG b (4 warps) has read S'^T / dP'^T (mbarrier variants)
      mbar_init(&bar_p_ready[b], 4); mbar_init(&bar_pds_ready[b], 4); mbar_init(&bar_dq_done[b], 1); mbar_init(&bar_dq_free[b], 4);
      mbar_init(&bar_acc_free[b], 4);
    }
    mbar_init(&bar_acc_done, 1); mbar_init(&bar_epi_done, 1);   // epi_done: the dK/dV epilogue no longer uses the dS^T smem buffers
    for (int r = 0; r < ITEM_RING; ++r) { mbar_init(&bar_item_full[r], 1); mbar_init(&bar_item_empty[r], 6); }
    fence_barrier_init();
  }
  if (tid < KV_BLK) {   // ones slice: row = key, elements 0,1 = 1.0 (the two augmentation columns hi/lo), rest 0
    auto t = make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + SM_ONES)), LayOnes{});
    #pragma unroll
    for (int c = 0; c < 16; ++c) t(tid, c) = bfloat16_t(c < 2 ? 1.0f : 0.0f);
    fence_proxy_async_smem();
  }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = tmem_base_s;
  // release registers of the producer / issuer warps before anything can block (the other warpgroups' incs wait for the pool)
  if (warp >= 12) setmaxnreg_dec<REGS_OTHER>();

  auto next_item = [&](int it) -> int {
    const int slot = it % ITEM_RING;
    MBW(&bar_item_full[slot], (it / ITEM_RING) & 1);
    return item_ring[slot];
  };
  // exactly one thread per consumer role (MMA warps A/B1/B2 lane 0, dQ WG thread 0, WG0/WG1 thread 0) arrives -> count 6
  auto release_item = [&](int it, bool leader) { if (leader) mbar_arrive(&bar_item_empty[it % ITEM_RING]); };

  if (warp == 12) {
    // ===================== TMA producer + item fetcher =====================
    if (lane_id() == 0) {
      prefetch_tmap(&p.tm_q); prefetch_tmap(&p.tm_k); prefetch_tmap(&p.tm_v); prefetch_tmap(&p.tm_do); prefetch_tmap(&p.tm_qaug); prefetch_tmap(&p.tm_doaug);
      int g = 0;
      for (int it = 0;; ++it) {
        const int slot = it % ITEM_RING;
        MBW(&bar_item_empty[slot], ((it / ITEM_RING) & 1) ^ 1);
        int i = atomicAdd(p.counter, 1);
        if (i >= n_items) i = -1;
        item_ring[slot] = i;
        mbar_arrive(&bar_item_full[slot]);
        if (i < 0) break;
        Item w = decode_item(p, i);
#ifndef NO_L2_PREFETCH
        {   // warm L2 with this item's K/V tiles now; the smem loads below wait for the previous item's K/V to be released
          const int trow = w.s0 + w.kv0;
          tma_prefetch_3d(&p.tm_k, 0, w.h, trow); tma_prefetch_3d(&p.tm_k, 64, w.h, trow);
          tma_prefetch_3d(&p.tm_v, 0, w.h, trow); tma_prefetch_3d(&p.tm_v, 64, w.h, trow);
        }
#endif
        MBW(&bar_kv_empty, (it & 1) ^ 1);
        mbar_arrive_expect_tx(&bar_kv_full, 32768);
        uint8_t* sK = smem + SM_K; uint8_t* sV = smem + SM_V;
        const int trow = w.s0 + w.kv0;
        tma_load_3d(&p.tm_k, &bar_kv_full, sK, 0, w.h, trow);
        tma_load_3d(&p.tm_k, &bar_kv_full, sK + 16384, 64, w.h, trow);
        MBW(&bar_v_empty, (it & 1) ^ 1);
        mbar_arrive_expect_tx(&bar_v_full, 32768);
        tma_load_3d(&p.tm_v, &bar_v_full, sV, 0, w.h, trow);
        tma_load_3d(&p.tm_v, &bar_v_full, sV + 16384, 64, w.h, trow);
        for (int j = w.j1; j >= w.j0; --j, ++g) {
          const int sq = g % QSTAGES, sd = g % DOSTAGES;
          const int prow = w.ps + j * Q_SUB;   // padded row of the augmentation slices
          const int qrow = w.s0 + j * Q_SUB;
          MBW(&bar_q_empty[sq], ((g / QSTAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&bar_q_full[sq], 16384 + 2048);
          uint8_t* sQ = smem + SM_Q + sq * 16384;
          tma_load_3d(&p.tm_q, &bar_q_full[sq], sQ, 0, w.h, qrow);
          tma_load_3d(&p.tm_q, &bar_q_full[sq], sQ + 8192, 64, w.h, qrow);
          tma_load_3d(&p.tm_qaug, &bar_q_full[sq], smem + SM_QA + sq * 2048, 0, prow, w.h);
          MBW(&bar_do_empty[sd], ((g / DOSTAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&bar_do_full[sd], 16384 + 2048);
          uint8_t* sDO = smem + SM_DO + sd * 16384;
          tma_load_3d(&p.tm_do, &bar_do_full[sd], sDO, 0, w.h, qrow);
          tma_load_3d(&p.tm_do, &bar_do_full[sd], sDO + 8192, 64, w.h, qrow);
          tma_load_3d(&p.tm_doaug, &bar_do_full[sd], smem + SM_DOA + sd * 2048, 0, prow, w.h);
        }
      }
    }
  } else if (warp >= 13) {
    // ===================== MMA issuers: warp 13 = A1 (S'^T), warp 15 = A2 (dP'^T), warp 14 = B (dV/dK then dQ^T) =====================
    // S'^T(g) and dP'^T(g) are issued by different warps so that neither waits for the other WG's progress.
    const uint32_t tR0 = tmem + T_R, tB0 = tmem + T_B, tdK = tmem + T_DK, tdV = tmem + T_DV;
    // K-major Q/dO sub-tile (64 rows): k-step s (16 d) -> +32 B within the 128B row, second 64-d box at +8192
    auto stepQk = [](int s) -> uint32_t { return (s < 4) ? s * 32 : 8192 + (s - 4) * 32; };
    // K-major K/V tile (128 rows): second 64-d box at +16384
    auto stepVk = [](int s) -> uint32_t { return (s < 4) ? s * 32 : 16384 + (s - 4) * 32; };
    const uint64_t dOnes = make_desc<UMMA::Major::K>(smem + SM_ONES, LayOnes{});
    const uint32_t idesc_S = make_idesc<128, 64, UMMA::Major::K, UMMA::Major::K>();
    int g = 0;
    if (warp == 13) {
      // ---- A1: S'^T(g) into R[g&1] once WG g&1 has read dP'^T(g-2)
      const uint64_t dQk0 = make_desc<UMMA::Major::K>(smem + SM_Q, LayQk{});
      const uint64_t dKk0 = make_desc<UMMA::Major::K>(smem + SM_K, LayVk{});    // K tile as K-major A [M=key][K=d]
      const uint64_t dQA0 = make_desc<UMMA::Major::K>(smem + SM_QA, LayAug{});
      for (int it = 0;; ++it) {
        const int i = next_item(it);
        release_item(it, lane_id() == 0);
        if (i < 0) break;
        Item w = decode_item(p, i);
        MBW(&bar_kv_full, it & 1);
        tc_fence_after();
        for (int j = w.j1; j >= w.j0; --j, ++g) {
          const int st = g % QSTAGES, b = g & 1;
          TR(g, 15);
          MBW(&bar_q_full[st], (g / QSTAGES) & 1);
          TR(g, 14);
          if (g >= 2) {   // R[b] free: WG b has read dP'^T(g-2)
#ifdef NAMED_BARS
            named_bar_sync(1 + b, 160);
#else
            MBW(&bar_spdP[b], ((g - 2) >> 1) & 1);
#endif
          }
          tc_fence_after();
          TR(g, 0);
          if (elect_one()) {
            const uint64_t dQk = desc_advance(dQk0, st * 16384), dQA = desc_advance(dQA0, st * 2048);
            const uint32_t tR = tR0 + 64 * b;
            #pragma unroll
            for (int s = 0; s < 8; ++s) umma_ss(tR, desc_advance(dKk0, stepVk(s)), desc_advance(dQk, stepQk(s)), idesc_S, s != 0);
            umma_ss(tR, dOnes, dQA, idesc_S, 1);   // 9th k-step: - lse/scale broadcast over keys
            umma_commit(&bar_S[b]);
            if (j == w.j0) umma_commit(&bar_kv_empty);   // K no longer read by A1 after the item's last S'^T
          }
          __syncwarp();
          TR(g, 9);
        }
      }
    } else if (warp == 15) {
      // ---- A2: dP'^T(g) into R[g&1] once WG g&1 has read S'^T(g)
      const uint64_t dDOk0 = make_desc<UMMA::Major::K>(smem + SM_DO, LayQk{});
      const uint64_t dVk0  = make_desc<UMMA::Major::K>(smem + SM_V, LayVk{});
      const uint64_t dDOA0 = make_desc<UMMA::Major::K>(smem + SM_DOA, LayAug{});
      for (int it = 0;; ++it) {
        const int i = next_item(it);
        release_item(it, lane_id() == 0);
        if (i < 0) break;
        Item w = decode_item(p, i);
        MBW(&bar_v_full, it & 1);
        tc_fence_after();
        for (int j = w.j1; j >= w.j0; --j, ++g) {
          const int st = g % DOSTAGES, b = g & 1;
          MBW(&bar_do_full[st], (g / DOSTAGES) & 1);
#ifdef NAMED_BARS
          named_bar_sync(3 + b, 160);
#else
          MBW(&bar_spS[b], (g >> 1) & 1);
#endif
          tc_fence_after();
          if (elect_one()) {
            const uint64_t dDOk = desc_advance(dDOk0, st * 16384), dDOA = desc_advance(dDOA0, st * 2048);
            const uint32_t tR = tR0 + 64 * b;
            #pragma unroll
            for (int s = 0; s < 8; ++s) umma_ss(tR, desc_advance(dVk0, stepVk(s)), desc_advance(dDOk, stepQk(s)), idesc_S, s != 0);
            umma_ss(tR, dOnes, dDOA, idesc_S, 1);   // 9th k-step: - rowsum(dO*O) broadcast over keys
            umma_commit(&bar_dP[b]);
          }
          __syncwarp();
          TR(g, 1);
        }
        if (elect_one()) umma_commit(&bar_v_empty);   // V no longer read after the item's last dP'^T
        __syncwarp();
      }
    } else {
      // ---- B: dV += P^T dO as soon as P^T is in TMEM (p_ready), then dK += dS^T Q and dQ^T(g) = K^T dS^T(g) (dS^T smem[g&1])
      //      into the same buffer once dS is ready: same issuing thread, so the MMAs execute in order (no WAR hop).
      //      Commits: do_empty after both dV and dP, q_empty after dK, dq_done after dQ,
      //      acc_done at the item end.
      const uint64_t dQmn0  = make_desc<UMMA::Major::MN>(smem + SM_Q, LayQmn{});
      const uint64_t dDOmn0 = make_desc<UMMA::Major::MN>(smem + SM_DO, LayQmn{});
      const uint64_t dKmn0  = make_desc<UMMA::Major::MN>(smem + SM_K, LayKmn{});   // K tile as MN-major A [M=d][K=key]
      const uint64_t dDSmn0 = make_desc<UMMA::Major::MN>(smem + SM_DS, LayDSmn{}); // dS^T as MN-major B [N=q][K=key]
      const uint32_t idesc_KV = make_idesc<128, 128, UMMA::Major::K, UMMA::Major::MN>();
      const uint32_t idesc_dQ = make_idesc<128, 64, UMMA::Major::MN, UMMA::Major::MN>();
      for (int it = 0;; ++it) {
        const int i = next_item(it);
        release_item(it, lane_id() == 0);
        if (i < 0) break;
        Item w = decode_item(p, i);
        for (int j = w.j1; j >= w.j0; --j, ++g) {
          const int sq = g % QSTAGES, sd = g % DOSTAGES, b = g & 1;
          const bool first = (j == w.j1);
          const uint32_t tb = tB0 + 64 * b;   // P^T cols [0,32), dS^T cols [32,64); 8 cols per 16-q k-step
#ifdef NAMED_BARS
          named_bar_sync(5 + b, 160);
#else
          MBW(&bar_p_ready[b], (g >> 1) & 1);
#endif
          tc_fence_after();
          if (first) { MBW(&bar_acc_free[0], (it & 1) ^ 1); MBW(&bar_acc_free[1], (it & 1) ^ 1); tc_fence_after(); }
          // dV is independent of dP: wait for its own dO tile before reading.
          MBW(&bar_do_full[sd], (g / DOSTAGES) & 1);
          tc_fence_after();
          TR(g, 5);
          if (elect_one()) {
            const uint64_t dDOmn = desc_advance(dDOmn0, sd * 16384);
            #pragma unroll
            for (int s = 0; s < 4; ++s) umma_ts(tdV, tb + 8 * s, desc_advance(dDOmn, 2048 * s), idesc_KV, !(first && s == 0));
          }
          __syncwarp();
#ifdef NAMED_BARS
          named_bar_sync(7 + b, 160);
#else
          MBW(&bar_pds_ready[b], (g >> 1) & 1);
#endif
          tc_fence_after();
          if (elect_one()) {
            // pds_ready proves A2 finished dP; this commit also waits for B's dV.
            umma_commit(&bar_do_empty[sd]);
            const uint64_t dQmn = desc_advance(dQmn0, sq * 16384);
            #pragma unroll
            for (int s = 0; s < 4; ++s) umma_ts(tdK, tb + 32 + 8 * s, desc_advance(dQmn, 2048 * s), idesc_KV, !(first && s == 0));
            umma_commit(&bar_q_empty[sq]);
            TR(g, 6);
            const uint64_t dDS = desc_advance(dDSmn0, b * 16384);
            #pragma unroll
            for (int s = 0; s < 8; ++s) umma_ss(tb, desc_advance(dKmn0, 2048 * s), desc_advance(dDS, 2048 * s), idesc_dQ, s != 0);
            umma_commit(&bar_dq_done[b]);
          }
          __syncwarp();
          TR(g, 13);
        }
        if (elect_one()) { umma_commit(&bar_acc_done); umma_commit(&bar_kv_empty); }   // dK/dV complete; K no longer read by dQ^T
        __syncwarp();
      }
    }
  } else if (warp < 4) {
    // ===================== dQ readout WG: lane = d, cols = q -> red.global.add into dq_accum[(s0+q0+c)][h][d] =====================
    // dq_accum is zeroed by the preprocess kernel; every kv block reduce-adds; postprocess converts to bf16.
    if constexpr (REGS_DQ <= 128) setmaxnreg_dec<REGS_DQ>(); else setmaxnreg_inc<REGS_DQ>();   // launch allocation is 128
    const int wtid = tid;
    const uint32_t lanebase = ((warp & 3) * 32) << 16;
    const uint32_t tB0 = tmem + T_B + lanebase;
    const size_t stride = (size_t)H * D;
    int g = 0;
    for (int it = 0;; ++it) {
      const int i = next_item(it);
      release_item(it, wtid == 0);
      if (i < 0) break;
      Item w = decode_item(p, i);
      for (int j = w.j1; j >= w.j0; --j, ++g) {
        const int q0 = j * Q_SUB, b = g & 1;
        MBW(&bar_dq_done[b], (g >> 1) & 1);
        tc_fence_after();
        if (wtid == 0) TR(g, 7);
        const int odd = wtid & 1;
        // read both 32-col halves, release the buffer, then pack and reduce-add (the reds queue in the LSU for a long time)
        uint32_t dQ[64];
        tmem_ld32(tB0 + 64 * b, *reinterpret_cast<uint32_t(*)[32]>(&dQ[0]));
        tmem_ld32(tB0 + 64 * b + 32, *reinterpret_cast<uint32_t(*)[32]>(&dQ[32]));
        tmem_ld_wait();
        tc_fence_before();
        __syncwarp();
        if (lane_id() == 0) TR(g, 16 + warp);
#ifdef NAMED_BARS
        named_bar_arrive_all(9 + b, 256);
#else
        if (lane_id() == 0) mbar_arrive(&bar_dq_free[b]);
#endif
        // lane pairs (2i, 2i+1) exchange so that for rows (c, c+1) the even lane holds row c, d = (2i, 2i+1) and the
        // odd lane holds row c+1, d = (2i, 2i+1): one f16x2 reduce-add per row pair
        uint32_t h2[32];
        #pragma unroll
        for (int c = 0; c < 64; c += 2) {
          const float a = __uint_as_float(dQ[c]), bq = __uint_as_float(dQ[c + 1]);
          const float sendv = odd ? a : bq;
          const float r = __shfl_xor_sync(0xffffffff, sendv, 1);
          h2[c / 2] = odd ? pack_f16x2(r, bq) : pack_f16x2(a, r);
        }
        __half* base = p.dq_accum + ((size_t)(w.s0 + q0 + odd) * H + w.h) * D + (wtid & ~1);
        const int nvalid = min(64, w.L - q0);
        if (nvalid >= 64) {
          #pragma unroll
          for (int c = 0; c < 64; c += 2) red_add_f16x2(base + (size_t)c * stride, h2[c / 2]);
        } else if (nvalid > 0) {
          #pragma unroll
          for (int c = 0; c < 64; c += 2) if (c + odd < nvalid) red_add_f16x2(base + (size_t)c * stride, h2[c / 2]);
        }
        if (wtid == 0) TR(g, 8);
      }
      // ---- epilogue of this item: dK / dV from TMEM -> bf16 -> global (keeps the softmax WGs flowing into the next item).
      //      Full kv blocks are staged in the (idle) dS^T smem buffers and written with TMA bulk stores (coalesced);
      //      a doc's partial last block (rows beyond L must not be written) uses direct per-row stores.
      MBW(&bar_acc_done, it & 1);
      tc_fence_after();
      {
        const int key = w.kv0 + wtid;
        const bool key_valid = key < w.L;
        const bool full_blk = (w.kv0 + KV_BLK <= w.L);
        const int trow = w.s0 + w.kv0;
        if (full_blk) {
          // step k: accumulator k>>1 (dK, dV), d half k&1 -> staging buffer k&1 (16 KB) -> TMA store of one 64-d box
          #pragma unroll 1
          for (int k = 0; k < 4; ++k) {
            const uint32_t tAcc = tmem + ((k >> 1) == 0 ? T_DK : T_DV) + lanebase + 64 * (k & 1);
            const uint32_t srow = smem_u32(smem + SM_DS + (k & 1) * 16384) + wtid * 128;   // row = key, SW128 chunks
            if (k >= 2) { if (wtid == 0) bulk_wait_group_read<1>(); named_bar_sync(11, 128); }   // store k-2 has read this buffer
            #pragma unroll 1
            for (int c0 = 0; c0 < 64; c0 += 32) {
              uint32_t r[32];
              tmem_ld32(tAcc + c0, r);
              tmem_ld_wait();
              #pragma unroll
              for (int c = 0; c < 4; ++c) {
                const uint32_t chunk = (c0 / 8 + c) ^ (wtid & 7);
                sts_v4(srow + (chunk << 4),
                       make_uint4(pack_bf16_rn(__uint_as_float(r[c * 8 + 0]), __uint_as_float(r[c * 8 + 1])),
                                  pack_bf16_rn(__uint_as_float(r[c * 8 + 2]), __uint_as_float(r[c * 8 + 3])),
                                  pack_bf16_rn(__uint_as_float(r[c * 8 + 4]), __uint_as_float(r[c * 8 + 5])),
                                  pack_bf16_rn(__uint_as_float(r[c * 8 + 6]), __uint_as_float(r[c * 8 + 7]))));
              }
            }
            tc_fence_before();
            fence_proxy_async_smem();
            named_bar_sync(11, 128);
            if (wtid == 0) {
              tma_store_3d((k >> 1) == 0 ? &p.tm_dk : &p.tm_dv, smem + SM_DS + (k & 1) * 16384, 64 * (k & 1), w.h, trow);
              bulk_commit_group();
            }
            if (k & 1) { __syncwarp(); if (lane_id() == 0) mbar_arrive(&bar_acc_free[k >> 1]); }   // accumulator fully read
          }
        } else {
          #pragma unroll 1
          for (int acc = 0; acc < 2; ++acc) {
            const uint32_t tAcc = tmem + (acc == 0 ? T_DK : T_DV) + lanebase;
            __nv_bfloat16* grow = (acc == 0 ? p.dk : p.dv) + ((size_t)(w.s0 + key) * H + w.h) * D;
            #pragma unroll 1
            for (int c0 = 0; c0 < 128; c0 += 32) {
              uint32_t r[32];
              tmem_ld32(tAcc + c0, r);
              tmem_ld_wait();
              if (key_valid) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                  uint4 o;
                  o.x = pack_bf16_rn(__uint_as_float(r[c * 8 + 0]), __uint_as_float(r[c * 8 + 1]));
                  o.y = pack_bf16_rn(__uint_as_float(r[c * 8 + 2]), __uint_as_float(r[c * 8 + 3]));
                  o.z = pack_bf16_rn(__uint_as_float(r[c * 8 + 4]), __uint_as_float(r[c * 8 + 5]));
                  o.w = pack_bf16_rn(__uint_as_float(r[c * 8 + 6]), __uint_as_float(r[c * 8 + 7]));
                  *reinterpret_cast<uint4*>(grow + c0 + c * 8) = o;
                }
              }
            }
            tc_fence_before();
            __syncwarp();
            if (lane_id() == 0) mbar_arrive(&bar_acc_free[acc]);
          }
        }
        if (full_blk && wtid == 0) bulk_wait_group_read<0>();   // staging smem free again
        __syncwarp();
        if (wtid == 0) mbar_arrive(&bar_epi_done);
      }
    }
  } else {
    // ===================== softmax warpgroups (warps 4-11): WG b handles the sub-tiles with g&1 == b =====================
    setmaxnreg_inc<REGS_SOFTMAX>();
    const int wg = (warp - 4) >> 2;           // 0 or 1 == parity b of this WG's sub-tiles
    const int wtid = tid - 128 - wg * 128;    // 0..127 == TMEM lane
    const uint32_t lanebase = ((warp & 3) * 32) << 16;
    const uint32_t tR = tmem + T_R + 64 * wg + lanebase, tB = tmem + T_B + 64 * wg + lanebase;
    const uint32_t dsbase = smem_u32(smem + SM_DS + wg * 16384) + wtid * 128;   // this WG's dS^T smem buffer, row = key
    int g = 0;
    for (int it = 0;; ++it) {
      const int i = next_item(it);
      release_item(it, wtid == 0);
      if (i < 0) break;
      Item w = decode_item(p, i);
      const int key = w.kv0 + wtid;
      const bool key_valid = key < w.L;
      bool epi_wait = (it >= 1);   // the previous item's dK/dV epilogue stages through the dS^T smem buffers
      for (int j = w.j1; j >= w.j0; --j, ++g) {
        if ((g & 1) != wg) continue;
        const uint32_t ph = (g >> 1) & 1;   // phase of this WG's per-parity barriers for sub-tile g
        const int q0 = j * Q_SUB;
        const bool need_mask = (j < w.j0 + 2) || (q0 + Q_SUB > w.L);   // diagonal sub-tiles or the doc's last one
        uint32_t S[64];
        // ---- S'^T -> P
        MBW(&bar_S[wg], ph);
        tc_fence_after();
        if (wtid == 0) TR(g, 2);
        tmem_ld32(tR, *reinterpret_cast<uint32_t(*)[32]>(&S[0]));
        tmem_ld32(tR + 32, *reinterpret_cast<uint32_t(*)[32]>(&S[32]));
        tmem_ld_wait();
        tc_fence_before();
        __syncwarp();
#ifdef NAMED_BARS
        named_bar_arrive_all(3 + wg, 160);
#else
        if (lane_id() == 0) mbar_arrive(&bar_spS[wg]);
#endif
        if (wtid == 0) TR(g, 3);
        float P[64];
        if (!key_valid) {
          #pragma unroll
          for (int c = 0; c < 64; ++c) P[c] = 0.f;
        } else if (need_mask) {
          #pragma unroll
          for (int c = 0; c < 64; ++c) {
            const int qq = q0 + c;
            const float pr = ex2(__uint_as_float(S[c]) * p.scale_log2e);
            P[c] = (qq < key || qq >= w.L) ? 0.f : pr;
          }
        } else {
          #pragma unroll
          for (int c = 0; c < 64; ++c) P[c] = ex2(__uint_as_float(S[c]) * p.scale_log2e);
        }
        {
          uint32_t Pp[32];
          #pragma unroll
          for (int c = 0; c < 64; c += 2) Pp[c / 2] = pack_bf16_rn(P[c], P[c + 1]);
          if (wtid == 0) TR(g, 10);
          // buffer B[wg] is free once the dQ WG has read dQ^T(g-2) (dV/dK(g-2) consumed P^T/dS^T(g-2) before that)
          if (g >= 2) {
#ifdef NAMED_BARS
            named_bar_sync(9 + wg, 256);
#else
            MBW(&bar_dq_free[wg], ph ^ 1);
#endif
          }
          tc_fence_after();
          if (wtid == 0) TR(g, 11);
          tmem_st16(tB, *reinterpret_cast<uint32_t(*)[16]>(&Pp[0]));         // P^T cols [0,16)  = q 0-31
          tmem_st16(tB + 16, *reinterpret_cast<uint32_t(*)[16]>(&Pp[16]));   // P^T cols [16,32) = q 32-63
          tmem_st_wait();
          tc_fence_before();
          __syncwarp();
#ifdef NAMED_BARS
          named_bar_arrive_all(5 + wg, 160);   // P^T visible: B can issue dV
#else
          if (lane_id() == 0) mbar_arrive(&bar_p_ready[wg]);
#endif
        }
        // ---- dP'^T -> dS (two 32-column halves)
        MBW(&bar_dP[wg], ph);
        tc_fence_after();
        if (wtid == 0) TR(g, 28);
        // dS^T smem buffer of this WG is free once dQ(g-2) completed (it read dS^T(g-2) from there) and, for the item's
        // first store, once the previous item's epilogue has finished reading the staging smem
        if (g >= 2) MBW(&bar_dq_done[wg], ph ^ 1);
        if (epi_wait) { MBW(&bar_epi_done, (it - 1) & 1); epi_wait = false; }
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t dP[32], dSp[16];
          tmem_ld32(tR + 32 * hh, dP);
          tmem_ld_wait();
          if (hh == 1) {
            tc_fence_before();
            __syncwarp();
#ifdef NAMED_BARS
            named_bar_arrive_all(1 + wg, 160);   // R[wg] free for S'^T(g+2)
#else
            if (lane_id() == 0) mbar_arrive(&bar_spdP[wg]);
#endif
            if (wtid == 0) TR(g, 29);
          }
          #pragma unroll
          for (int c = 0; c < 32; c += 2)
            dSp[c / 2] = pack_bf16_rn(P[32 * hh + c] * (__uint_as_float(dP[c]) * p.scale), P[32 * hh + c + 1] * (__uint_as_float(dP[c + 1]) * p.scale));
          tmem_st16(tB + 32 + 16 * hh, dSp);   // dS^T cols [32+16hh, +16)
          #pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t chunk = (4 * hh + c) ^ (wtid & 7);
            sts_v4(dsbase + (chunk << 4), make_uint4(dSp[c * 4], dSp[c * 4 + 1], dSp[c * 4 + 2], dSp[c * 4 + 3]));
          }
        }
        if (lane_id() == 0) TR(g, 20 + 4 * wg + (warp & 3));
        tmem_st_wait();
        if (wtid == 0) TR(g, 12);
        tc_fence_before();
        fence_proxy_async_smem();                            // dS^T smem -> async proxy (dQ MMA)
        __syncwarp();
#ifdef NAMED_BARS
        named_bar_arrive_all(7 + wg, 160);                   // dS^T visible (TMEM + smem): B can issue dK and dQ
#else
        if (lane_id() == 0) mbar_arrive(&bar_pds_ready[wg]);
#endif
        if (wtid == 0) TR(g, 4);
      }
    }
  }
  // teardown
  __syncthreads();
  tc_fence_after();
  if (warp == 0) tmem_dealloc(tmem, 512);
#ifdef TRACE
  if (trace_on && g_trace != nullptr)
    for (int i = tid; i < TRACE_W * 32; i += NUM_THREADS) g_trace[(size_t)TRACE_G0 * 32 + i] = (long long)(&trace_ring[0][0])[i];
#endif
}

// ---------------- auxiliary kernels ----------------
// Augmentation slices (padded rows tp): qaug[h][tp] = [-lse[h][t]/scale as (hi, lo) bf16, 0 x 14],
// doaug[h][tp] = [-sum_d dO*O as (hi, lo), 0 x 14]. The main kernel's 9th MMA k-step multiplies them by a ones slice,
// which subtracts lse (resp. delta) from every key row of S^T (resp. dP^T) inside the tensor core.
// 256 threads = 16 half-warps; each half-warp handles one 256 B row per iteration (16 B per lane), 4 iterations.
__device__ __forceinline__ void split_bf16(float v, __nv_bfloat16& hi, __nv_bfloat16& lo) {
  hi = __float2bfloat16_rn(v);
  lo = __float2bfloat16_rn(v - __bfloat162float(hi));
}
__global__ void __launch_bounds__(256, 6) preprocess_kernel(const __nv_bfloat16* o, const __nv_bfloat16* dout, const float* lse,
                                  __nv_bfloat16* qaug, __nv_bfloat16* doaug, int T, int H, int Tp,
                                  const int* cu_seqlens, int n_docs, const int* pad_start, float scale, __half* dq_accum) {
  // The block handles PRE_ROWS consecutive rows (t,h) of 256 B. Phase 1 (all threads): zero dq_accum rows, stream o/dO
  // (16 lanes x 16 B per row, one row per half-warp per iteration, all loads issued up front) and reduce the row dot
  // products through shuffles into smem. Phase 2 (one thread per row): the scalar tail (doc lookup, lse, bf16 hi/lo
  // splits, 2 x 16 B stores) without lane divergence.
#ifndef PRE_ROWS
#define PRE_ROWS 64
#endif
#ifndef PRE_LD
#define PRE_LD(p) __ldg(p)
#endif
  constexpr int ITERS = PRE_ROWS / 16;
  __shared__ float sacc_s[PRE_ROWS];
  const int hw = threadIdx.x >> 4, l16 = threadIdx.x & 15;
  const int nrows = T * H;
  const int row0 = (int)blockIdx.x * PRE_ROWS;   // row indices fit in 32 bits (T*H <= 2^31)
#if !defined(PRE_MEMSET) && !defined(PRE_NO_ZERO)
  {   // zero this block's rows of dq_accum (fp16, 256 B per row): 16 B per store
    uint4* z = reinterpret_cast<uint4*>(dq_accum + (size_t)row0 * D);
    const size_t nz = (size_t)min(PRE_ROWS, nrows - row0) * D / 8;
    #pragma unroll 4
    for (size_t i = threadIdx.x; i < nz; i += 256) __stcs(z + i, make_uint4(0u, 0u, 0u, 0u));
  }
#endif
#ifdef PRE_ONLY_ZERO
  return;
#endif
  uint4 av[ITERS], bv[ITERS];
  #pragma unroll
  for (int r = 0; r < ITERS; ++r) {   // all loads in flight before any arithmetic
    const int row = min(row0 + r * 16 + hw, nrows - 1);
    av[r] = PRE_LD(reinterpret_cast<const uint4*>(o + (size_t)row * D + l16 * 8));
    bv[r] = PRE_LD(reinterpret_cast<const uint4*>(dout + (size_t)row * D + l16 * 8));
  }
  #pragma unroll
  for (int r = 0; r < ITERS; ++r) {
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&av[r]);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bv[r]);
    float sacc = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 fa = __bfloat1622float2(a2[i]), fb = __bfloat1622float2(b2[i]);
      sacc = fmaf(fa.x, fb.x, sacc); sacc = fmaf(fa.y, fb.y, sacc);
    }
    #pragma unroll
    for (int o2 = 8; o2 > 0; o2 >>= 1) sacc += __shfl_xor_sync(0xffffffff, sacc, o2);
    if (l16 == 0) sacc_s[r * 16 + hw] = sacc;
  }
  __syncthreads();
  if (threadIdx.x < PRE_ROWS) {   // one thread per row
    const int row = row0 + (int)threadIdx.x;
    if (row < nrows) {
      const int t = row / H, h = row - t * H;
      int lo = 0, hi = n_docs;   // largest doc with cu_seqlens[doc] <= t
      while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (__ldg(cu_seqlens + mid) <= t) lo = mid; else hi = mid; }
      const size_t idx = ((size_t)h * Tp + __ldg(pad_start + lo) + (t - __ldg(cu_seqlens + lo))) * 16;
      __nv_bfloat16 vhi, vlo;
      // 16 B each: (hi, lo, 0 x 6); the second 16 B of each 32 B row stay zero from the workspace initialization
      split_bf16(-__ldg(lse + (size_t)h * T + t) / scale, vhi, vlo);
      *reinterpret_cast<uint4*>(qaug + idx) = make_uint4((uint32_t)__bfloat16_as_ushort(vhi) | ((uint32_t)__bfloat16_as_ushort(vlo) << 16), 0u, 0u, 0u);
      split_bf16(-sacc_s[threadIdx.x], vhi, vlo);
      *reinterpret_cast<uint4*>(doaug + idx) = make_uint4((uint32_t)__bfloat16_as_ushort(vhi) | ((uint32_t)__bfloat16_as_ushort(vlo) << 16), 0u, 0u, 0u);
    }
  }
}

// dq = bf16(dq_accum)   (scale already folded into dS); 8 elements per thread
__global__ void __launch_bounds__(256) postprocess_kernel(const uint4* __restrict__ acc, uint4* __restrict__ dq, size_t n8) {
  const size_t i0 = (size_t)blockIdx.x * 1024 + threadIdx.x;   // 4 x 16 B per thread, 256 B apart (coalesced per instruction)
  uint4 a[4];
  #pragma unroll
  for (int k = 0; k < 4; ++k) { const size_t i = i0 + k * 256; a[k] = (i < n8) ? __ldcs(acc + i) : make_uint4(0u, 0u, 0u, 0u); }
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    const size_t i = i0 + k * 256;
    if (i >= n8) break;
    const __half2* h = reinterpret_cast<const __half2*>(&a[k]);
    uint4 o;
    float2 f;
    f = __half22float2(h[0]); o.x = pack_bf16(f.x, f.y);
    f = __half22float2(h[1]); o.y = pack_bf16(f.x, f.y);
    f = __half22float2(h[2]); o.z = pack_bf16(f.x, f.y);
    f = __half22float2(h[3]); o.w = pack_bf16(f.x, f.y);
    __stcs(dq + i, o);
  }
}

// Work order: docs sorted by length (longest first), kv blocks ascending within a doc, heads innermost
// (expanded in the main kernel). Long items start first (LPT-like balance) and concurrently running CTAs
// touch the same doc's dq_accum rows (L2 locality for the fp32 reduce-adds).
constexpr int SCHED_THREADS = 1024;
__global__ void __launch_bounds__(SCHED_THREADS) schedule_kernel(const int* cu_seqlens, int n_docs, int* items, int* n_pairs, int* counter, int* pad_start) {
  __shared__ int scan[SCHED_THREADS];
  const int tid = threadIdx.x;
  const int per = (n_docs + SCHED_THREADS - 1) / SCHED_THREADS;   // docs per thread (by rank)
  // pad_start: exclusive prefix over docs (original order) of round_up(L, 32)
  {
    int local = 0;
    for (int d = tid * per; d < min((tid + 1) * per, n_docs); ++d) local += (cu_seqlens[d + 1] - cu_seqlens[d] + Q_SUB - 1) / Q_SUB * Q_SUB;
    scan[tid] = local;
    __syncthreads();
    for (int off = 1; off < SCHED_THREADS; off <<= 1) {
      const int v = (tid >= off) ? scan[tid - off] : 0;
      __syncthreads();
      scan[tid] += v;
      __syncthreads();
    }
    int pos = scan[tid] - local;
    for (int d = tid * per; d < min((tid + 1) * per, n_docs); ++d) { pad_start[d] = pos; pos += (cu_seqlens[d + 1] - cu_seqlens[d] + Q_SUB - 1) / Q_SUB * Q_SUB; }
    if (tid == SCHED_THREADS - 1) pad_start[n_docs] = scan[tid];
    __syncthreads();
  }
  // rank of each doc = number of docs that are longer (ties: lower index first)
  // thread t handles ranks [t*per, (t+1)*per); we need nkv by rank -> first compute rank for own docs, then scatter.
  __shared__ int nkv_by_rank_dummy;   // (unused)
  // pass 1: each thread computes rank for docs d = tid, tid+1024, ... and writes rank_doc[rank] = d into items (temp)
  for (int d = tid; d < n_docs; d += SCHED_THREADS) {
    const int L = cu_seqlens[d + 1] - cu_seqlens[d];
    int rank = 0;
    for (int e = 0; e < n_docs; ++e) {
      const int Le = cu_seqlens[e + 1] - cu_seqlens[e];
      rank += (Le > L) || (Le == L && e < d);
    }
    items[rank] = d;   // temp: doc id by rank (items has >= n_docs entries)
  }
  __syncthreads();
  // pass 2: nkv per rank, local sums over this thread's rank range, block exclusive scan
  int local = 0;
  for (int r = tid * per; r < min((tid + 1) * per, n_docs); ++r) {
    const int d = items[r];
    const int L = cu_seqlens[d + 1] - cu_seqlens[d];
    local += (L + KV_BLK - 1) / KV_BLK;
  }
  scan[tid] = local;
  __syncthreads();
  for (int off = 1; off < SCHED_THREADS; off <<= 1) {
    const int v = (tid >= off) ? scan[tid - off] : 0;
    __syncthreads();
    scan[tid] += v;
    __syncthreads();
  }
  const int total = scan[SCHED_THREADS - 1];
  int pos = scan[tid] - local;   // exclusive
  // read own docs-by-rank before anyone overwrites items
  int mydocs[8];
  int nmy = 0;
  for (int r = tid * per; r < min((tid + 1) * per, n_docs) && nmy < 8; ++r) mydocs[nmy++] = items[r];
  __syncthreads();
  for (int q = 0; q < nmy; ++q) {
    const int d = mydocs[q];
    const int L = cu_seqlens[d + 1] - cu_seqlens[d];
    const int nkv = (L + KV_BLK - 1) / KV_BLK;
    for (int n = 0; n < nkv; ++n) items[pos++] = d | (n << 16);
  }
  if (tid == 0) { *n_pairs = total; *counter = 0; }
}

// ---------------- host ----------------
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn get_encode() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* ptr = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &qres);
    fn = (EncodeFn)ptr;
  }
  return fn;
}
static int make_tmap_thd(CUtensorMap* tm, const void* ptr, int T, int H, int rows) {
  cuuint64_t dims[3] = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)T};
  cuuint64_t strides[2] = {(cuuint64_t)D * 2, (cuuint64_t)H * D * 2};
  cuuint32_t box[3] = {64, 1, (cuuint32_t)rows};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

static int make_tmap_aug(CUtensorMap* tm, const void* ptr, int Tp, int H) {   // bf16 (H, Tp, 16), box (16, 64, 1), 32B swizzle
  cuuint64_t dims[3] = {16, (cuuint64_t)Tp, (cuuint64_t)H};
  cuuint64_t strides[2] = {32, (cuuint64_t)Tp * 32};
  cuuint32_t box[3] = {16, (cuuint32_t)Q_SUB, 1};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

} // namespace fa4

using namespace fa4;
// host-mapped trace buffer (readable from the host while a kernel is stuck)
extern "C" int fa4_alloc_trace_host(size_t bytes, void** host_ptr, void** dev_ptr) {
  if (cudaHostAlloc(host_ptr, bytes, cudaHostAllocMapped) != cudaSuccess) return -1;
  memset(*host_ptr, 0, bytes);
  return (int)cudaHostGetDevicePointer(dev_ptr, *host_ptr, 0);
}
extern "C" int fa4_set_trace(void* ptr) {
#ifdef TRACE
  return (int)cudaMemcpyToSymbol(g_trace, &ptr, sizeof(ptr));
#else
  return -1;
#endif
}
extern "C" int fa4_bwd_launch(const void* q, const void* kk, const void* v, const void* o, const void* dout, const void* lse,
                              const int* cu_seqlens, int n_docs, int T, int H, float scale,
                              void* dq, void* dk, void* dv,
                              __half* dq_accum, void* qaug, void* doaug, int* items, int* n_pairs, int* counter, int* pad_start,
                              int num_sms, cudaStream_t stream) {
  Params p;
  p.q = (const __nv_bfloat16*)q; p.kk = (const __nv_bfloat16*)kk; p.v = (const __nv_bfloat16*)v; p.dout = (const __nv_bfloat16*)dout;
  p.qaug = (const __nv_bfloat16*)qaug; p.doaug = (const __nv_bfloat16*)doaug; p.dq_accum = dq_accum; p.dq = (__nv_bfloat16*)dq; p.dk = (__nv_bfloat16*)dk; p.dv = (__nv_bfloat16*)dv;
  p.cu_seqlens = cu_seqlens; p.n_docs = n_docs; p.T = T; p.H = H; p.Tp = (T + Q_SUB - 1) / Q_SUB * Q_SUB + Q_SUB * n_docs; p.n_pairs = n_pairs; p.items = items; p.counter = counter; p.pad_start = pad_start;
  p.scale = scale; p.scale_log2e = scale * 1.4426950408889634f;
  int r = 0;
  r |= make_tmap_thd(&p.tm_q, q, T, H, Q_SUB);
  r |= make_tmap_thd(&p.tm_do, dout, T, H, Q_SUB);
  r |= make_tmap_thd(&p.tm_k, kk, T, H, KV_BLK);
  r |= make_tmap_thd(&p.tm_v, v, T, H, KV_BLK);
  r |= make_tmap_aug(&p.tm_qaug, qaug, p.Tp, H);
  r |= make_tmap_aug(&p.tm_doaug, doaug, p.Tp, H);
  r |= make_tmap_thd(&p.tm_dk, dk, T, H, KV_BLK);
  r |= make_tmap_thd(&p.tm_dv, dv, T, H, KV_BLK);
  if (r) return -3;

  schedule_kernel<<<1, SCHED_THREADS, 0, stream>>>(cu_seqlens, n_docs, items, n_pairs, counter, pad_start);
#ifdef PRE_MEMSET
  cudaMemsetAsync(dq_accum, 0, (size_t)T * H * D * 2, stream);
#endif
  preprocess_kernel<<<(T * H + PRE_ROWS - 1) / PRE_ROWS, 256, 0, stream>>>((const __nv_bfloat16*)o, (const __nv_bfloat16*)dout, (const float*)lse,
                                                         (__nv_bfloat16*)qaug, (__nv_bfloat16*)doaug, T, H, p.Tp, cu_seqlens, n_docs, pad_start, scale, dq_accum);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(fa4_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
    attr_set = true;
  }
  fa4_bwd_kernel<<<num_sms, NUM_THREADS, SMEM_BYTES, stream>>>(p);
  const size_t n8 = (size_t)T * H * D / 8;
  postprocess_kernel<<<(unsigned)((n8 + 1023) / 1024), 256, 0, stream>>>((const uint4*)dq_accum, (uint4*)dq, n8);
  return (int)cudaGetLastError();
}
