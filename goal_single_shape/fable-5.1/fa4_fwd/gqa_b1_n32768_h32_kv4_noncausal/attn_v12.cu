// FlashAttention-style forward for sm_100a (B200), written from scratch.
// v12: v11 + optimistic softmax: exps are computed at the pair's last known running max right after the row max; the
//      running-max decision (intra-pair proposal exchange via flag barriers, cross-pair m_done) is gathered AFTER the exps and
//      the fp32 p values are rescaled in the rare case the max moved. No rendezvous on the pre-exps path.
// v11: 2-CTA (cta_group::2) MMAs (as v10) with a decoupled S/P pipeline so the cross-CTA signalling latency leaves the
//      critical chain:
//        * S ring = 2 x 128 TMEM cols. The softmax releases an S buffer (s_free) right after loading it into registers:
//          max-first softmax (row max of the tile -> lazy running-max decision -> exps), no re-read of S ever.
//        * P ring = 2 x 64 TMEM cols (bf16), separate from S. PV(i-1) is issued after S(i+2): the MMA pipe holds
//          ~3 tile-times of work between S(i) completing and PV(i) starting, which covers softmax latency + remote arrives.
//        * O = 128 cols. O is rescaled (rarely) by the half-0 warps after PV(i-1) completes, before P(i) is published.
//      Each CTA holds its own Q and HALF of every K/V tile (K split by keys, V by head-dim): per-SM K/V traffic halves.
#include "tc.cuh"
#include <cstdio>

#ifdef SPIN_MMA
#define WAITM(b, p) { while (!mbar_test_wait(b, p)) {} }
#else
#define WAITM(b, p) mbar_wait_fast2(b, p)
#endif
#ifdef DBG_TIMING
#define CLK() clock64()
#else
#define CLK() 0ll
#endif
#ifndef ROW_DEBUG_BLK
#define ROW_DEBUG_BLK 0
#define ROW_DEBUG_ROW 0
#endif
#define WAITS(bar, par, tag) mbar_wait_fast2(bar, par)

namespace {
constexpr int D = 128;
constexpr int BM = 128;           // q rows per CTA
constexpr int BN = 128;           // keys per tile
#ifndef KST_CFG
#define KST_CFG 6
#endif
#ifndef VST_CFG
#define VST_CFG 5
#endif
constexpr int KST = KST_CFG;      // K half-tile ring (16KB stages)
constexpr int VST = VST_CFG;      // V half-tile ring (16KB stages)
constexpr int LEADK = KST - 1;    // tiles of K prefetch lead
constexpr int LEADV = VST - 1;    // tiles of V prefetch lead
constexpr int SBUF = 2;           // S ring in TMEM (128 cols each)
constexpr int PBUF = 2;           // P ring in TMEM (64 cols each)
constexpr uint32_t TM_S = 0, TM_P = SBUF * 128, TM_O = SBUF * 128 + PBUF * 64;   // 0, 256, 384
constexpr int NTHREADS = 640;     // 16 softmax warps + producer + MMA + 2 forwarders
#ifndef RESCALE_THRESH_CFG
#define RESCALE_THRESH_CFG 60.0f
#endif
constexpr float RESCALE_THRESH = RESCALE_THRESH_CFG;   // log2 units: running max moves only when the tile max exceeds it by this (p <= 2^60 is safe in fp32/bf16)
constexpr float UNDER_THRESH = 100.0f;                  // ... or falls this far below it (all p would underflow past 2^-126)
constexpr uint16_t MC_MASK = 0x3;
constexpr float M_UNSET = -1.7014118e38f;   // -2^127: 'no running max established' sentinel

struct __align__(1024) Smem {
    __nv_bfloat16 q[2][BM * 64];              // [d-block][128 rows x 64]   32KB (bf16 -> fp16 in place)
    __nv_bfloat16 k[KST][2][64 * 64];         // [stage][d-block][64 keys x 64]   16KB/stage (this CTA's key half, fp16)
    __nv_bfloat16 v[VST][128 * 64];           // [stage][128 keys x 64 d]         16KB/stage (this CTA's d half, bf16)
    float xch[2][2][2][BM];                   // [pair][slot][half][row] half-tile max exchange
    float m_run[BM];                          // shared running max per row (log2 units) or M_UNSET
    uint64_t tma_k[KST], full_k[KST], empty_k[KST];
    uint64_t tma_v[VST], full_v[VST], empty_v[VST];
    uint64_t full_q;
    uint64_t s_ready[SBUF], s_free[SBUF];
    uint64_t p_ready[PBUF][2], pv_done[PBUF];
    uint64_t m_done[2];
    uint64_t xbar[2][4];                      // [pair][warp%4]: both halves' proposals for this pair-tile are in xch (count 2)
    uint64_t q_conv;
    uint32_t tmem_base;
};

struct Params {
    __nv_bfloat16* out;
    int B, S, HQ, HKV;
    int n_qtiles, n_ktiles;
    float scale_log2;
    unsigned long long* dbg;
};
}  // namespace

extern __shared__ __align__(1024) char smem_raw[];

DEVI uint64_t pk2(float lo, float hi) { uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi)); return r; }
DEVI float lo2(uint64_t v) { float r; asm("{.reg .b32 hi; mov.b64 {%0, hi}, %1;}" : "=f"(r) : "l"(v)); return r; }
DEVI float hi2(uint64_t v) { float r; asm("{.reg .b32 lo; mov.b64 {lo, %0}, %1;}" : "=f"(r) : "l"(v)); return r; }
DEVI uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) { uint64_t r; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c)); return r; }
DEVI uint64_t add2(uint64_t a, uint64_t b) { uint64_t r; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b)); return r; }
#define EM_C0 0.9999289404f
#define EM_C1 0.6932762417f
#define EM_C2 0.2426040515f
#define EM_C3 0.0550886838f
#define EM2_C0 1.0004403309f
#define EM2_C1 0.7033803487f
#define EM2_C2 0.2383945698f
DEVI void ex2_emul2(uint64_t x, float& p0, float& p1) {
    const uint64_t MAGIC = pk2(12582912.0f, 12582912.0f), NMAGIC = pk2(-12582912.0f, -12582912.0f);
    const uint64_t NEG1 = pk2(-1.0f, -1.0f);
    const uint64_t C3 = pk2(EM_C3, EM_C3), C2 = pk2(EM_C2, EM_C2);
    const uint64_t C1 = pk2(EM_C1, EM_C1), C0 = pk2(EM_C0, EM_C0);
    x = pk2(fmaxf(lo2(x), -126.0f), fmaxf(hi2(x), -126.0f));   // lower clamp only: x <= RESCALE_THRESH (60) is guaranteed by the lazy running max
    uint64_t t = add2(x, MAGIC);
    uint64_t n = add2(t, NMAGIC);
    uint64_t f = fma2(n, NEG1, x);
#ifdef EMUL_DEG2
    uint64_t q = fma2(pk2(EM2_C2, EM2_C2), f, pk2(EM2_C1, EM2_C1));
    q = fma2(q, f, pk2(EM2_C0, EM2_C0));
#else
    uint64_t q = fma2(C3, f, C2);
    q = fma2(q, f, C1);
    q = fma2(q, f, C0);
#endif
    p0 = __uint_as_float(__float_as_uint(lo2(t)) * 8388608u + __float_as_uint(lo2(q)));
    p1 = __uint_as_float(__float_as_uint(hi2(t)) * 8388608u + __float_as_uint(hi2(q)));
}

template <int EMUL_EVERY, int DIAG>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NTHREADS, 1) attn_fwd_kernel(const __grid_constant__ CUtensorMap mapQ,
                                                              const __grid_constant__ CUtensorMap mapK,
                                                              const __grid_constant__ CUtensorMap mapV, Params p) {
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int warp = __shfl_sync(0xffffffffu, (int)(threadIdx.x / 32), 0), lane = threadIdx.x % 32;
    const int wg = warp / 4;
    const uint32_t crank = __shfl_sync(0xffffffffu, (int)cluster_ctarank(), 0);

    const int bh = blockIdx.x / p.n_qtiles;
    const int qt = blockIdx.x % p.n_qtiles;
    const int b = bh / p.HQ, h = bh % p.HQ;
    const int g = h / (p.HQ / p.HKV);
    const int q0 = qt * BM;
    const int n_ktiles = p.n_ktiles;

    if (warp == 16) { tmem_alloc2(smem_u32(&sm.tmem_base), 512); tmem_relinquish2(); }
    if (threadIdx.x == 0) {
        for (int i = 0; i < KST; ++i) { mbar_init(smem_u32(&sm.tma_k[i]), 1); mbar_init(smem_u32(&sm.full_k[i]), 2); mbar_init(smem_u32(&sm.empty_k[i]), 1); }
        for (int i = 0; i < VST; ++i) { mbar_init(smem_u32(&sm.tma_v[i]), 1); mbar_init(smem_u32(&sm.full_v[i]), 2); mbar_init(smem_u32(&sm.empty_v[i]), 1); }
        mbar_init(smem_u32(&sm.full_q), 1);
        for (int i = 0; i < SBUF; ++i) { mbar_init(smem_u32(&sm.s_ready[i]), 1); mbar_init(smem_u32(&sm.s_free[i]), 16); }
        for (int i = 0; i < PBUF; ++i) { mbar_init(smem_u32(&sm.p_ready[i][0]), 8); mbar_init(smem_u32(&sm.p_ready[i][1]), 8); mbar_init(smem_u32(&sm.pv_done[i]), 1); }
        mbar_init(smem_u32(&sm.m_done[0]), 8); mbar_init(smem_u32(&sm.m_done[1]), 8);
        for (int i = 0; i < 8; ++i) mbar_init(smem_u32(&sm.xbar[0][0]) + i * 8, 2);
        mbar_init(smem_u32(&sm.q_conv), 32);   // 16 local + 16 peer softmax warps (leader's barrier only)
        fence_mbar_init();
    }
    tc_fence_before();
    __syncthreads();
    cluster_sync_all();
    tc_fence_after();
    if (threadIdx.x == 0 && sm.tmem_base != 0) { printf("unexpected tmem base %u\n", sm.tmem_base); __trap(); }
    constexpr uint32_t tmem_base = 0;
    unsigned long long dbg_clk0 = 0, dbg_gt0 = 0;
    if (p.dbg && threadIdx.x == 0) { dbg_clk0 = clock64(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(dbg_gt0)); }

    if (wg == 4) {
        setmaxnreg_dec<32>();
        if (warp == 16) {
            // ---------------- TMA producer (each CTA loads its own half of every K/V tile) ----------------
            if (lane == 0) {
                tma_prefetch_desc(&mapQ); tma_prefetch_desc(&mapK); tma_prefetch_desc(&mapV);
                const uint32_t bar_q = smem_u32(&sm.full_q);
                mbar_expect_tx(bar_q, 2 * BM * 64 * 2);
                for (int db = 0; db < 2; ++db) tma_load_4d(&mapQ, smem_u32(sm.q[db]), bar_q, db * 64, h, q0, b);
                auto issue_k = [&](int j) {
                    const int s = j % KST, n = j / KST;
                    if (n > 0) mbar_wait_fast2(smem_u32(&sm.empty_k[s]), (n - 1) & 1);
                    if ((DIAG & 8) && n > 0) mbar_wait_fast2(smem_u32(&sm.empty_k[(j + 1) % KST]), ((j + 1) / KST - 1) & 1);   // diag: also S(j-KST+1) done
                    const uint32_t bt = smem_u32(&sm.tma_k[s]);
                    mbar_expect_tx(bt, 2 * 64 * 64 * 2);
                    for (int db = 0; db < 2; ++db) tma_load_4d(&mapK, smem_u32(sm.k[s][db]), bt, db * 64, g, j * BN + crank * 64, b);
                };
                auto issue_v = [&](int j) {
                    const int s = j % VST, n = j / VST;
                    if (n > 0) mbar_wait_fast2(smem_u32(&sm.empty_v[s]), (n - 1) & 1);
                    if ((DIAG & 16) && n > 0) mbar_wait_fast2(smem_u32(&sm.empty_v[(j + 1) % VST]), ((j + 1) / VST - 1) & 1);   // diag: also PV(j-VST+1) done
                    const uint32_t bt = smem_u32(&sm.tma_v[s]);
                    mbar_expect_tx(bt, 128 * 64 * 2);
                    tma_load_4d(&mapV, smem_u32(sm.v[s]), bt, crank * 64, g, j * BN, b);
                };
                for (int j = 0; j < LEADK && j < n_ktiles; ++j) issue_k(j);
                for (int j = 0; j < LEADV && j < n_ktiles; ++j) issue_v(j);
                for (int i = 0; i < n_ktiles; ++i) {
                    if (i + LEADK < n_ktiles) issue_k(i + LEADK);
                    if (i + LEADV < n_ktiles) issue_v(i + LEADV);
                }
            }
        } else if (warp == 18 || warp == 19) {
            // ---------------- completion forwarders: wait this CTA's local TMA barrier, arrive on the leader's full barrier ----------------
            if (lane == 0) {
                const bool isk = (warp == 18);
                const int nst = isk ? KST : VST;
                const uint32_t tma0 = isk ? smem_u32(&sm.tma_k[0]) : smem_u32(&sm.tma_v[0]);
                const uint32_t full0 = isk ? smem_u32(&sm.full_k[0]) : smem_u32(&sm.full_v[0]);
                const uint32_t full0_leader = mapa_shared(full0, 0);
                for (int i = 0; i < n_ktiles; ++i) {
                    const int st = i % nst;
                    mbar_wait_fast2(tma0 + st * 8, (i / nst) & 1);
                    if (crank == 0) mbar_arrive(full0 + st * 8); else mbar_arrive_remote(full0_leader + st * 8);
                }
            }
        } else if (warp == 17 && crank == 0) {
            // ---------------- MMA issuer (leader CTA only; cta_group::2 MMAs cover both CTAs' rows) ----------------
            {
                long long w_k = 0, w_v = 0, w_p = 0, w_sf = 0, t_loop0 = CLK();
                const uint32_t idesc_s = make_idesc_f16(256, BN, false, false, false, false);   // A = Q (fp16), B = K' (fp16), M = 2 x 128
                const uint32_t idesc_o = make_idesc_f16(256, D, true, true, false, true);       // A = P (bf16, TMEM), B = V (bf16, MN-major)
                constexpr uint32_t KSTAGE_U = (2 * 64 * 64 * 2) >> 4;   // 16KB
                constexpr uint32_t KDB_U = (64 * 64 * 2) >> 4;          // 8KB
                constexpr uint32_t VSTAGE_U = (128 * 64 * 2) >> 4;      // 16KB
                constexpr uint32_t QDB_U = (BM * 64 * 2) >> 4;          // 16KB
                const uint32_t q_lo = (smem_u32(sm.q[0]) >> 4) & 0x3FFF, k_lo = (smem_u32(sm.k[0][0]) >> 4) & 0x3FFF, v_lo = (smem_u32(sm.v[0]) >> 4) & 0x3FFF;
                constexpr uint32_t HI = (1u << 14) | (uint32_t)(1024 >> 4) | (2u << 29);
                const uint64_t DESC_K_HI = ((uint64_t)((16 >> 4)) << 16) | ((uint64_t)HI << 32);
                const uint64_t DESC_V_HI = ((uint64_t)((128 * 64 * 2) >> 4) << 16) | ((uint64_t)HI << 32);
                const uint32_t d_o = tmem_base + TM_O;
                WAITM(smem_u32(&sm.q_conv), 0);
                auto issue_S = [&](int i) {
                    const int ks = i % KST, sb = i & 1;
                    long long c0 = CLK();
                    if (i >= 2) WAITM(smem_u32(&sm.s_free[sb]), ((i - 2) >> 1) & 1);   // softmax(i-2) has loaded S(i-2)
                    long long c1 = CLK(); w_sf += c1 - c0;
                    WAITM(smem_u32(&sm.full_k[ks]), (i / KST) & 1);
                    w_k += CLK() - c1;
                    tc_fence_after();
                    const uint32_t kb = k_lo + ks * KSTAGE_U;
                    const uint32_t d_s = tmem_base + TM_S + sb * 128;
                    #pragma unroll
                    for (int db = 0; db < 2; ++db)
                        #pragma unroll
                        for (int kk = 0; kk < 4; ++kk)
                            mma2_ss_e(d_s, DESC_K_HI | (q_lo + db * QDB_U + kk * 2), DESC_K_HI | (kb + db * KDB_U + kk * 2), idesc_s, (db | kk) != 0);
                    mma2_commit_mc_e(smem_u32(&sm.s_ready[sb]), MC_MASK);
                    mma2_commit_mc_e(smem_u32(&sm.empty_k[ks]), MC_MASK);
                };
                auto issue_PV = [&](int i) {
                    const int vs = i % VST, pb = i & 1;
                    long long c1 = CLK();
                    WAITM(smem_u32(&sm.full_v[vs]), (i / VST) & 1);
                    long long c2 = CLK(); w_v += c2 - c1;
                    const uint32_t vb = v_lo + vs * VSTAGE_U;
                    const uint32_t a_p = tmem_base + TM_P + pb * 64;
                    #pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        WAITM(smem_u32(&sm.p_ready[pb][hh]), (i >> 1) & 1);
                        if (hh == 0) w_p += CLK() - c2;
                        tc_fence_after();
                        #pragma unroll
                        for (int k4 = 0; k4 < 4; ++k4) {
                            const int kk = hh * 4 + k4;
                            mma2_ts_e(d_o, a_p + hh * 32 + k4 * 8, DESC_V_HI | (vb + kk * 128), idesc_o, (i > 0) || (kk > 0));
                        }
                    }
                    mma2_commit_mc_e(smem_u32(&sm.pv_done[pb]), MC_MASK);
                    mma2_commit_mc_e(smem_u32(&sm.empty_v[vs]), MC_MASK);
                };
                for (int i = 0; i < 2 && i < n_ktiles; ++i) issue_S(i);
                for (int i = 0; i < n_ktiles; ++i) {
                    if (i + 2 < n_ktiles) issue_S(i + 2);
                    if (i >= 1) issue_PV(i - 1);
                }
                issue_PV(n_ktiles - 1);
                if (p.dbg && lane == 0) { long long t = CLK() - t_loop0; unsigned long long* d = p.dbg + gridDim.x * 5 + blockIdx.x * 8; d[0] = w_k; d[1] = w_v; d[2] = w_p; d[3] = w_sf; d[4] = t; }
            }
        }
    } else {
        // ---------------- softmax: wg = 2*pair + half; pair handles tiles i == pair (mod 2), half handles keys [64h, 64h+64) ----------------
        setmaxnreg_inc<112>();
        const int pair = wg >> 1, half = wg & 1;
        const uint32_t lane_base = (uint32_t)((warp % 4) * 32) << 16;
        const uint32_t t_o = tmem_base + lane_base + TM_O;
        const int row_in_rg = (warp % 4) * 32 + lane;
        float* m_run_p = &sm.m_run[row_in_rg];
        const uint32_t bar_mdone_mine = smem_u32(&sm.m_done[pair]), bar_mdone_other = smem_u32(&sm.m_done[pair ^ 1]);
        float m_seen = M_UNSET, m_base = 0.0f;   // raw base: exps of the pre-scaled scores need no subtraction for ordinary data
        const uint32_t bar_x = smem_u32(&sm.xbar[pair][warp % 4]);   // arrive after writing my proposal, wait (after the exps) for both
        if (wg == 0) sm.m_run[row_in_rg] = M_UNSET;
        {   // convert Q (bf16, 128 rows x 128 d = 8192 words) to fp16 in place: 512 threads x 16 words
            mbar_wait_fast2(smem_u32(&sm.full_q), 0);
            uint32_t* qw = reinterpret_cast<uint32_t*>(&sm.q[0][0]);
            const int t = threadIdx.x;   // 0..511
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t w = qw[t + j * 512];
                uint32_t r;
                asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(__uint_as_float(w & 0xffff0000u)), "f"(__uint_as_float(w << 16)));
                qw[t + j * 512] = r;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) { if (crank == 0) mbar_arrive(smem_u32(&sm.q_conv)); else mbar_arrive_remote(mapa_shared(smem_u32(&sm.q_conv), 0)); }
        }
        uint64_t lacc[2] = {0, 0};
        long long sw_wait = 0, sw_work = 0;
        long long ph[6] = {0, 0, 0, 0, 0, 0};
        auto arrive_leader = [&](uint32_t bar_local) {   // warp-collective: lane 0 arrives on the leader CTA's barrier
            __syncwarp();
            if (lane == 0) { if (crank == 0) mbar_arrive(bar_local); else mbar_arrive_remote(mapa_shared(bar_local, 0)); }
        };

        for (int i = pair; i < n_ktiles; i += 2) {
            const int sb = i & 1, pb = i & 1;
            const uint32_t t_s = tmem_base + lane_base + TM_S + sb * 128 + half * 64;
            const uint32_t t_p = tmem_base + lane_base + TM_P + pb * 64 + half * 32;
            long long sc0 = CLK();
            WAITS(smem_u32(&sm.s_ready[sb]), (i >> 1) & 1, 24);
            long long sc1 = CLK(); sw_wait += sc1 - sc0;
            tc_fence_after();
            uint32_t sr[64];
            tmem_ld_x32(t_s, sr);
            tmem_ld_x32(t_s + 32, sr + 32);
            tmem_ld_wait_pin64(sr);
            tc_fence_before();
            arrive_leader(smem_u32(&sm.s_free[sb]));          // S(i) is in registers: the buffer may be overwritten by S(i+2)
            long long tA = CLK();
            if (DIAG & 1) {
                uint32_t pp[32];
                #pragma unroll
                for (int j = 0; j < 32; ++j) pp[j] = 0x3f803f80u ^ (sr[j] & 1u);
                named_bar_sync(1 + pair, 256);
                if (i > 0) mbar_wait_fast2(bar_mdone_other, ((i - 1) >> 1) & 1);
                { __syncwarp(); if (lane == 0) mbar_arrive(bar_mdone_mine); }
                if (i >= 2) WAITS(smem_u32(&sm.pv_done[pb]), ((i - 2) >> 1) & 1, 22);
                tc_fence_after();
                tmem_st_x32(t_p, pp); tmem_st_wait_pin32(pp); tc_fence_before();
                arrive_leader(smem_u32(&sm.p_ready[pb][half]));
                lacc[0] += 1; sw_work += CLK() - sc1; continue;
            }
            // ---- row max of this half tile ----
            float m_half;
            {
                float mx[22];
                #pragma unroll
                for (int j = 0; j < 21; ++j) mx[j] = fmax3(__uint_as_float(sr[3 * j]), __uint_as_float(sr[3 * j + 1]), __uint_as_float(sr[3 * j + 2]));
                mx[21] = __uint_as_float(sr[63]);
                #pragma unroll
                for (int j = 0; j < 7; ++j) mx[j] = fmax3(mx[3 * j], mx[3 * j + 1], mx[3 * j + 2]);
                mx[7] = mx[21];
                mx[0] = fmax3(mx[0], mx[1], mx[2]); mx[1] = fmax3(mx[3], mx[4], mx[5]); mx[2] = fmaxf(mx[6], mx[7]);
                m_half = fmax3(mx[0], mx[1], mx[2]);
            }
            // ---- optimistic: exps at this pair's last decision (m_base), unless this half's max exceeds it by the threshold ----
            const uint32_t xph = (i >> 1) & 1;           // this pair's tile counter parity
            const float m_used = (m_half > m_base + RESCALE_THRESH || m_half < m_base - UNDER_THRESH) ? m_half : m_base;   // = this half's proposal
            sm.xch[pair][xph][half][row_in_rg] = m_used;
            __syncwarp();
            if (lane == 0) mbar_arrive(bar_x);
            long long tB = CLK();
            uint64_t lt0 = 0, lt1 = 0;
            uint32_t pp[32];
            const bool raw = __all_sync(0xffffffffu, m_used == 0.0f);   // warp-uniform: skip the subtraction entirely
            if (raw) {
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const uint64_t x = pk2(__uint_as_float(sr[2 * j]), __uint_as_float(sr[2 * j + 1]));
                    float p0, p1;
                    if (DIAG & 32) { p0 = lo2(x); p1 = hi2(x); }
                    else if ((EMUL_EVERY > 0) && (j % EMUL_EVERY) == EMUL_EVERY - 1) ex2_emul2(x, p0, p1);
                    else { p0 = ex2_approx(lo2(x)); p1 = ex2_approx(hi2(x)); }
                    if (DIAG & 128) { pp[j] = __float_as_uint(p0) ^ __float_as_uint(p1); }
                    else {
                        if (j & 1) lt1 = add2(lt1, pk2(p0, p1)); else lt0 = add2(lt0, pk2(p0, p1));
                        pp[j] = pack_bf16x2(p0, p1);
                    }
                }
            } else {
                const uint64_t mm = pk2(-m_used, -m_used);
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    const uint64_t x = add2(pk2(__uint_as_float(sr[2 * j]), __uint_as_float(sr[2 * j + 1])), mm);
                    float p0, p1;
                    if (DIAG & 32) { p0 = lo2(x); p1 = hi2(x); }
                    else if ((EMUL_EVERY > 0) && (j % EMUL_EVERY) == EMUL_EVERY - 1) ex2_emul2(x, p0, p1);
                    else { p0 = ex2_approx(lo2(x)); p1 = ex2_approx(hi2(x)); }
                    if (DIAG & 128) { pp[j] = __float_as_uint(p0) ^ __float_as_uint(p1); }
                    else {
                        if (j & 1) lt1 = add2(lt1, pk2(p0, p1)); else lt0 = add2(lt0, pk2(p0, p1));
                        pp[j] = pack_bf16x2(p0, p1);
                    }
                }
            }
            long long tC = CLK();
            if (DIAG & 64) {   // timing diagnostic: skip the decision gathering (keeps the barrier protocol alive)
                if (i > 0) WAITS(bar_mdone_other, ((i - 1) >> 1) & 1, 21);
                { __syncwarp(); if (lane == 0) mbar_arrive(bar_mdone_mine); }
                lacc[0] = add2(lacc[0], lt0); lacc[1] = add2(lacc[1], lt1);
                long long tD_ = CLK();
                if (i >= 2) WAITS(smem_u32(&sm.pv_done[pb]), ((i - 2) >> 1) & 1, 22);
                tc_fence_after(); tmem_st_x32(t_p, pp); tmem_st_wait_pin32(pp); tc_fence_before();
                arrive_leader(smem_u32(&sm.p_ready[pb][half]));
                long long tF_ = CLK();
                ph[0] += tA - sc1; ph[1] += tB - tA; ph[2] += tC - tB; ph[5] += tF_ - tD_; sw_work += tF_ - sc1;
                continue;
            }
            // ---- gather the decision: other pair's running max (tile i-1) and the other half's proposal ----
            if (i > 0) WAITS(bar_mdone_other, ((i - 1) >> 1) & 1, 21);
            const float m_prev = (i > 0) ? *m_run_p : M_UNSET;
            long long tC2 = CLK();
            WAITS(bar_x, xph, 20);   // both proposals present (each half arrives once per pair-tile; nobody can run 2 phases ahead)
            const float m_new = fmaxf(fmaxf(m_prev, m_used), sm.xch[pair][xph][half ^ 1][row_in_rg]);
            if (__any_sync(0xffffffffu, m_new != m_used)) {
                // rare: the max moved after the exps were taken (p <= 2^T): scale the bf16 pairs down and re-round
                const float a = ex2_approx(m_used - m_new);
                const uint64_t aa = pk2(a, a);
                lt0 = fma2(lt0, aa, 0); lt1 = fma2(lt1, aa, 0);
                #pragma unroll
                for (int j = 0; j < 32; ++j) { const uint64_t y = fma2(pk2(__uint_as_float(pp[j] << 16), __uint_as_float(pp[j] & 0xffff0000u)), aa, 0); pp[j] = pack_bf16x2(lo2(y), hi2(y)); }
            }
            const bool changed = (m_new != m_prev);
            const bool any_changed = __any_sync(0xffffffffu, changed);
#ifdef ROW_DEBUG
            if (p.dbg && blockIdx.x == ROW_DEBUG_BLK && row_in_rg == ROW_DEBUG_ROW) {
                float* ev = reinterpret_cast<float*>(p.dbg) + (1 << 20) + (i * 2 + half) * 8;
                ev[0] = (float)i; ev[1] = (float)half; ev[2] = m_prev; ev[3] = m_half; ev[4] = m_new; ev[5] = (float)changed; ev[6] = (float)(m_new != m_seen); ev[7] = m_used;
            }
#endif
#ifdef DBG_TIMING
            if (lane == 0 && any_changed) atomicAdd((unsigned long long*)(p.dbg + gridDim.x * 13), 1ull);
#endif
            if (any_changed && half == 0) {
                if (i > 0) {
                    // O holds tiles < i at scale m_prev: rescale after PV(i-1) completes (PV(i) is only issued after our p_ready)
                    const float a = changed ? ex2_approx(m_prev - m_new) : 1.0f;
                    if (i >= 2) WAITS(smem_u32(&sm.pv_done[pb]), ((i - 2) >> 1) & 1, 23);        // PV(i-2) done (unambiguous) ...
                    WAITS(smem_u32(&sm.pv_done[pb ^ 1]), ((i - 1) >> 1) & 1, 23);               // ... hence PV(i-1)'s barrier is at most one phase behind
                    tc_fence_after();
                    #pragma unroll 1
                    for (int c = 0; c < 128; c += 32) {
                        uint32_t o[32];
                        tmem_ld_x32(t_o + c, o);
                        tmem_ld_wait_pin32(o);
                        #pragma unroll
                        for (int j = 0; j < 32; ++j) o[j] = __float_as_uint(__uint_as_float(o[j]) * a);
                        tmem_st_x32(t_o + c, o);
                        tmem_st_wait_pin32(o);
                    }
                    tc_fence_before();
                }
                if (changed) *m_run_p = m_new;
            }
            { __syncwarp(); if (lane == 0) mbar_arrive(bar_mdone_mine); }      // decision for tile i published
            m_base = m_new;
            if (__any_sync(0xffffffffu, m_new != m_seen)) {
                const float a = (m_new != m_seen) ? ex2_approx(m_seen - m_new) : 1.0f;
                const uint64_t aa = pk2(a, a);
                lacc[0] = fma2(lacc[0], aa, 0); lacc[1] = fma2(lacc[1], aa, 0);
                m_seen = m_new;
            }
            lacc[0] = add2(lacc[0], lt0); lacc[1] = add2(lacc[1], lt1);
            long long tD = CLK();
            if (i >= 2) WAITS(smem_u32(&sm.pv_done[pb]), ((i - 2) >> 1) & 1, 22);   // P buffer free: PV(i-2) done
            if (DIAG & 2) { const long long t0 = clock64(); while (clock64() - t0 < 3000) {} }   // diagnostic: widen the store window
            if (DIAG & 4) { if (i >= 1) WAITS(smem_u32(&sm.pv_done[pb ^ 1]), ((i - 1) >> 1) & 1, 22); }   // diagnostic: PV(i-1) done too
            long long tE = CLK();
            tc_fence_after();
            tmem_st_x32(t_p, pp);
            tmem_st_wait_pin32(pp);
#ifdef ROW_DEBUG
            if (p.dbg && blockIdx.x == ROW_DEBUG_BLK && row_in_rg == ROW_DEBUG_ROW && i == ROW_DEBUG_TILE) { uint32_t* ev = reinterpret_cast<uint32_t*>(p.dbg) + (1 << 20) + 3200 + half * 32; for (int j = 0; j < 32; ++j) ev[j] = pp[j]; }
#endif
            tc_fence_before();
            arrive_leader(smem_u32(&sm.p_ready[pb][half]));
            long long tF = CLK();
            ph[0] += tA - sc1; ph[1] += tB - tA; ph[2] += tC - tB; ph[3] += tC2 - tC; ph[4] += tD - tC2; ph[5] += tF - tD;
            sw_work += tF - sc1;
        }
        if (p.dbg && lane == 0 && (warp == 0 || warp == 8)) { unsigned long long* d = p.dbg + gridDim.x * 5 + blockIdx.x * 8; d[5 + pair] = sw_wait; d[7] = sw_work; }
#ifdef DBG_TIMING
        if (p.dbg && lane == 0 && (warp % 4) == 0) { unsigned long long* d = p.dbg + gridDim.x * 14 + blockIdx.x * 32 + wg * 8; for (int j = 0; j < 6; ++j) d[j] = ph[j]; d[6] = sw_wait; }
#endif
        // ---- epilogue: combine the 4 partial row sums (2 pairs x 2 halves); pair 0 half 0 writes O ----
        __syncwarp();
        named_bar_sync(3, 512);   // all 16 softmax warps: every pair has published its last decision and stored its last P
        {
            const float m_fin = *m_run_p;
            if (m_fin != m_seen) { const float a = ex2_approx(m_seen - m_fin); const uint64_t aa = pk2(a, a); lacc[0] = fma2(lacc[0], aa, 0); lacc[1] = fma2(lacc[1], aa, 0); m_seen = m_fin; }
        }
        const uint64_t lsum = add2(lacc[0], lacc[1]);
        const float l_mine = lo2(lsum) + hi2(lsum);
        float* lx = &sm.xch[0][0][0][0];   // exchange buffers are idle after the last tile: fp32 scratch [pair][half][row] (512 floats)
        lx[(pair * 2 + half) * BM + row_in_rg] = l_mine;
        named_bar_sync(3, 512);
        if (pair == 0 && half == 0) {
            WAITS(smem_u32(&sm.pv_done[(n_ktiles - 1) & 1]), ((n_ktiles - 1) >> 1) & 1, 25);   // PV(n-3) is known done (P(n-1) was stored after it)
            tc_fence_after();
            const float l_tot = lx[0 * BM + row_in_rg] + lx[1 * BM + row_in_rg] + lx[2 * BM + row_in_rg] + lx[3 * BM + row_in_rg];
#ifdef ROW_DEBUG
            if (p.dbg) { float* rd = reinterpret_cast<float*>(p.dbg) + (1 << 20) + 4096 + (blockIdx.x * 128 + row_in_rg) * 4; rd[0] = m_seen; rd[1] = l_tot; rd[2] = lx[0 * BM + row_in_rg] + lx[1 * BM + row_in_rg]; rd[3] = lx[2 * BM + row_in_rg] + lx[3 * BM + row_in_rg]; }
#endif
            const float inv_l = 1.f / l_tot;
            const int row = q0 + row_in_rg;
            __nv_bfloat16* orow = p.out + ((size_t)(b * p.S + row) * p.HQ + h) * D;
            #pragma unroll 1
            for (int c = 0; c < 128; c += 32) {
                uint32_t o[32];
                tmem_ld_x32(t_o + c, o);
                tmem_ld_wait_pin32(o);
                uint32_t packed[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j) packed[j] = pack_bf16x2(__uint_as_float(o[2 * j]) * inv_l, __uint_as_float(o[2 * j + 1]) * inv_l);
                uint4* dst = reinterpret_cast<uint4*>(orow + c);
                #pragma unroll
                for (int j = 0; j < 4; ++j) dst[j] = make_uint4(packed[4 * j], packed[4 * j + 1], packed[4 * j + 2], packed[4 * j + 3]);
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (p.dbg && threadIdx.x == 0) {
        unsigned long long c1 = clock64(), g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
        unsigned int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        unsigned long long* d = p.dbg + blockIdx.x * 5;
        d[0] = dbg_clk0; d[1] = c1; d[2] = dbg_gt0; d[3] = g1; d[4] = smid;
    }
    cluster_sync_all();     // no CTA exits while its peer may still arrive on / read it
    if (warp == 16) tmem_dealloc2(sm.tmem_base, 512);
}

__global__ void scale_bf16_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, size_t n8, float c) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 v = in[i];
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float lo = __uint_as_float(w[j] << 16) * c, hi = __uint_as_float(w[j] & 0xffff0000u) * c;
        uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); w[j] = r;   // fp16x2 output
    }
    out[i] = make_uint4(w[0], w[1], w[2], w[3]);
}
static void* g_kscaled = nullptr; static size_t g_kscaled_bytes = 0;

extern "C" int attn_fwd(const void* q, const void* k, const void* v, void* out, int B, int S, int HQ, int HKV, float softmax_scale,
                        int emul_every, void* stream, void* dbg) {
    cudaStream_t st = (cudaStream_t)stream;
    const size_t kbytes = (size_t)B * S * HKV * D * 2;
    if (kbytes > g_kscaled_bytes) { if (g_kscaled) cudaFree(g_kscaled); if (cudaMalloc(&g_kscaled, kbytes) != cudaSuccess) return -5; g_kscaled_bytes = kbytes; }
    {
        const size_t n8 = kbytes / 16;
        scale_bf16_kernel<<<(unsigned)((n8 + 255) / 256), 256, 0, st>>>((const uint4*)k, (uint4*)g_kscaled, n8, softmax_scale * 1.4426950408889634f);
    }
    CUtensorMap mapQ, mapK, mapV;
    int r = make_map_bshd(&mapQ, q, B, S, HQ, D, BM);
    r |= make_map_bshd(&mapK, g_kscaled, B, S, HKV, D, 64);   // K half-tile: 64 keys per CTA per tile
    r |= make_map_bshd(&mapV, v, B, S, HKV, D, BN);
    if (r) { printf("tensormap encode failed\n"); return -1; }
    Params p;
    p.out = (__nv_bfloat16*)out; p.B = B; p.S = S; p.HQ = HQ; p.HKV = HKV;
    p.n_qtiles = S / BM; p.n_ktiles = S / BN;
    p.scale_log2 = softmax_scale * 1.4426950408889634f;
    p.dbg = (unsigned long long*)dbg;
    if (p.n_qtiles % 2 != 0 || p.n_ktiles < 1) { printf("unsupported shape\n"); return -4; }
    size_t smem = sizeof(Smem);
    dim3 grid(p.n_qtiles * B * HQ);
    int diag = emul_every / 100; emul_every %= 100;
#define LAUNCH(E, G) { cudaError_t ea = cudaFuncSetAttribute(attn_fwd_kernel<E, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); if (ea != cudaSuccess) printf("setattr error: %s\n", cudaGetErrorString(ea)); attn_fwd_kernel<E, G><<<grid, NTHREADS, smem, st>>>(mapQ, mapK, mapV, p); }
    if (diag == 0) {
        switch (emul_every) { case 0: LAUNCH(0, 0); break; case 2: LAUNCH(2, 0); break; case 3: LAUNCH(3, 0); break; case 4: LAUNCH(4, 0); break; case 6: LAUNCH(6, 0); break; case 8: LAUNCH(8, 0); break; default: printf("bad emul_every\n"); return -2; }
    } else if (diag == 1) { LAUNCH(0, 1); }
    else if (diag == 2) { LAUNCH(0, 2); }
    else if (diag == 4) { LAUNCH(0, 4); }
    else if (diag == 8) { LAUNCH(0, 8); }
    else if (diag == 32) { LAUNCH(0, 32); }
    else if (diag == 64) { LAUNCH(0, 64); }
    else if (diag == 128) { LAUNCH(0, 128); }
    else if (diag == 224) { LAUNCH(0, 224); }
    else if (diag == 16) { LAUNCH(0, 16); }
    else { printf("bad diag\n"); return -2; }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(e)); return -3; }
    return 0;
}
extern "C" int smem_bytes() { return (int)sizeof(Smem); }
