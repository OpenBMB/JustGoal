// FlashAttention forward (causal, bf16, D=128) from scratch for B200 (sm_100a) -- design W
// 2-CTA clusters, tcgen05.mma cta_group::2 (M=256 per pair), Q and P in TMEM (all MMAs read A from TMEM),
// TMEM: O[0,128) Q[128,192) P[192,256) S0[256,384) S1[384,512)
// warps: 0 producer, 1 S-issuer (leader) / K-relay (follower), 2 PV-issuer (leader) / V-relay (follower), 3 tmem alloc, 4-7 softmax
#include "ptx.cuh"
#include <cstdio>
#include <cuda_bf16.h>

namespace cfg {
constexpr int D = 128, BM = 128, BN = 128;
constexpr int NK = 6, NV = 6;
constexpr int Q_BYTES = BM * D * 2;          // 32768 staging (single buffer)
constexpr int K_BYTES = 64 * D * 2;          // 16384 : [2 subtiles][64 keys][64 d]
constexpr int V_BYTES = BN * 64 * 2;         // 16384 : [128 keys][64 d]
constexpr int OFF_Q = 0;
constexpr int OFF_K = OFF_Q + Q_BYTES;
constexpr int OFF_V = OFF_K + NK * K_BYTES;
constexpr int OFF_BAR = OFF_V + NV * V_BYTES;
enum { B_QFULL = 0, B_QEMPTY, B_QREADY, B_PFULL, B_PVDONE, B_SFULL0, B_SFULL1, B_SFREE0, B_SFREE1,
       B_KFULL, B_KEMPTY = B_KFULL + NK, B_VFULL = B_KEMPTY + NK, B_VEMPTY = B_VFULL + NV, NBAR = B_VEMPTY + NV };
constexpr int OFF_TMEM_SLOT = OFF_BAR + NBAR * 8;
constexpr int SMEM_BYTES = OFF_TMEM_SLOT + 64;
constexpr int TM_O = 0, TM_Q = 128, TM_P = 192, TM_S0 = 256;   // tmem columns; S buffer b at TM_S0 + 128*b
constexpr int NUM_THREADS = 256;
constexpr float RESCALE_THRESH = 8.0f;
}

struct Params {
    __nv_bfloat16* out;
    long long* trace;   // debug: [tile][16] events for pair 0 / CTA 0
    int B, S, H, Hkv;
    int n_blocks, n_items, n_pairs;
    float scale_log2;
};
struct ItemIter {
    int pr, n_pairs, n_items, r, i;
    DEVI void init(int pr_, int n_pairs_, int n_items_) { pr = pr_; n_pairs = n_pairs_; n_items = n_items_; r = 0; i = pr; }
    DEVI bool valid() const { return i < n_items; }
    DEVI void next() { r++; i = r * n_pairs + ((r & 1) ? (n_pairs - 1 - pr) : pr); }
};
struct Item { int bh, b_idx, h, hkv, blk, T, row0; };
DEVI Item decode(int i, const Params& p) {
    Item it;
    it.bh = i / p.n_blocks;
    it.blk = p.n_blocks - 1 - (i % p.n_blocks);
    it.b_idx = it.bh / p.H; it.h = it.bh % p.H; it.hkv = it.h / (p.H / p.Hkv);
    it.T = 2 * it.blk + 2;
    it.row0 = it.b_idx * p.S + it.blk * 256;
    return it;
}

extern __shared__ __align__(1024) uint8_t smem_raw[];
#ifdef HELPER_SLEEP
#define HW_WAIT(b, par) mbar_wait_cluster_sleep((b), (par), HELPER_SLEEP)
#define HW_WAIT_LOCAL(b, par) mbar_wait_sleep((b), (par), HELPER_SLEEP)
#else
#define HW_WAIT(b, par) mbar_wait_cluster((b), (par))
#define HW_WAIT_LOCAL(b, par) mbar_wait((b), (par))
#endif
#ifdef TRACE
#define TR(tile, ev) do { if (trace_on && (tile) < 512) p.trace[trace_off + (tile) * 16 + (ev)] = clock64(); } while (0)
#else
#define TR(tile, ev) do {} while (0)
#endif

DEVI float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// exp2 emulation on the FMA pipe: Cody-Waite split via magic-number rounding + degree-3 polynomial (rel err 7.5e-5)
DEVI float ex2_emul(float x) {
    x = fmaxf(x, -126.f);
    const float t = x + 12582912.f;
    const float n = t - 12582912.f;
    const float f = x - n;
    const float pp = fmaf(fmaf(fmaf(0.05517025f, f, 0.2426079f), f, 0.69326093f), f, 0.99992828f);
    return __uint_as_float(__float_as_uint(pp) + (__float_as_uint(t) << 23));
}
#ifndef EMUL_MOD
#define EMUL_MOD 3
#endif
DEVI float fmax3(float a, float b, float c) { float d; asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
#ifndef EMUL_EVERY
#define EMUL_EVERY 3
#endif
#include "softmax_tile.cuh"
DEVI uint32_t pack_bf16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint4 lds128(uint32_t addr) { uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr)); return v; }

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(cfg::NUM_THREADS, 1)
fa_fwd_kernel(const __grid_constant__ CUtensorMap mapQ, const __grid_constant__ CUtensorMap mapK,
              const __grid_constant__ CUtensorMap mapV, Params p) {
    using namespace cfg;
    uint8_t* smem = smem_raw;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t rank = cluster_ctarank();
    const int pr = blockIdx.x / 2;
    const uint32_t bar0 = smem_u32(smem + OFF_BAR);
    auto bar = [&](int idx) -> uint32_t { return bar0 + idx * 8; };
    const uint32_t sQ = smem_u32(smem + OFF_Q), sK = smem_u32(smem + OFF_K), sV = smem_u32(smem + OFF_V);

    if (threadIdx.x == 0) {
        tma_prefetch_desc(&mapQ); tma_prefetch_desc(&mapK); tma_prefetch_desc(&mapV);
        const uint32_t two = (rank == 0) ? 2 : 1;
        mbar_init(bar(B_QFULL), 1); mbar_init(bar(B_QEMPTY), 4); mbar_init(bar(B_QREADY), 8);
        mbar_init(bar(B_PFULL), 8); mbar_init(bar(B_PVDONE), 1);
        mbar_init(bar(B_SFULL0), 1); mbar_init(bar(B_SFULL1), 1); mbar_init(bar(B_SFREE0), 8); mbar_init(bar(B_SFREE1), 8);
        for (int i = 0; i < NK; i++) { mbar_init(bar(B_KFULL + i), two); mbar_init(bar(B_KEMPTY + i), 1); }
        for (int i = 0; i < NV; i++) { mbar_init(bar(B_VFULL + i), two); mbar_init(bar(B_VEMPTY + i), 1); }
        fence_barrier_init();
    }
    __syncthreads();
    if (warp == 3) { tmem_alloc2(smem_u32(smem + OFF_TMEM_SLOT), 512); tmem_relinquish2(); }
    __syncthreads();
    cluster_sync();
    const uint32_t tmem = *(volatile uint32_t*)(smem + OFF_TMEM_SLOT);
    const bool trace_on = (p.trace != nullptr) && (blockIdx.x < 2) && (lane == 0);
    const int trace_off = blockIdx.x * 512 * 16;

    int total = 0;
    { ItemIter it; it.init(pr, p.n_pairs, p.n_items); while (it.valid()) { total += decode(it.i, p).T; it.next(); } }

    if (warp == 0) {
        // ================= producer =================
        if (lane == 0) {
            ItemIter itQ, itK, itV; itQ.init(pr, p.n_pairs, p.n_items); itK = itQ; itV = itQ;
            Item iQ = decode(itQ.i, p), iK = iQ, iV = iQ;
            int jK = 0, jV = 0; uint32_t gq = 0, gk = 0, gv = 0;
            bool qd = !itQ.valid(), kd = qd, vd = qd;
            while (!(qd && kd && vd)) {
                bool progress = false;
                if (!qd) {
                    if (mbar_try_wait(bar(B_QEMPTY), (gq & 1) ^ 1)) {
                        mbar_expect_tx(bar(B_QFULL), Q_BYTES);
                        int row = iQ.row0 + rank * 128;
                        tma_load_3d(sQ, &mapQ, bar(B_QFULL), 0, iQ.h, row);
                        tma_load_3d(sQ + 16384, &mapQ, bar(B_QFULL), 64, iQ.h, row);
                        gq++; itQ.next(); qd = !itQ.valid(); if (!qd) iQ = decode(itQ.i, p);
                        progress = true;
                    }
                }
                if (!kd) {
                    uint32_t st = gk % NK;
                    if (mbar_try_wait(bar(B_KEMPTY + st), ((gk / NK) & 1) ^ 1)) {
                        mbar_expect_tx(bar(B_KFULL + st), K_BYTES);
                        int row = iK.b_idx * p.S + jK * 128 + rank * 64;
                        tma_load_3d(sK + st * K_BYTES, &mapK, bar(B_KFULL + st), 0, iK.hkv, row);
                        tma_load_3d(sK + st * K_BYTES + 8192, &mapK, bar(B_KFULL + st), 64, iK.hkv, row);
                        gk++; jK++;
                        if (jK == iK.T) { jK = 0; itK.next(); kd = !itK.valid(); if (!kd) iK = decode(itK.i, p); }
                        progress = true;
                    }
                }
                if (!vd) {
                    uint32_t st = gv % NV;
                    if (mbar_try_wait(bar(B_VEMPTY + st), ((gv / NV) & 1) ^ 1)) {
                        mbar_expect_tx(bar(B_VFULL + st), V_BYTES);
                        int row = iV.b_idx * p.S + jV * 128;
                        tma_load_3d(sV + st * V_BYTES, &mapV, bar(B_VFULL + st), rank * 64, iV.hkv, row);
                        gv++; jV++;
                        if (jV == iV.T) { jV = 0; itV.next(); vd = !itV.valid(); if (!vd) iV = decode(itV.i, p); }
                        progress = true;
                    }
                }
#ifdef PROD_SLEEP
                if (!progress) __nanosleep(PROD_SLEEP);
#endif
            }
        }
    } else if (warp == 1) {
        if (rank == 0) {
            // ================= S issuer (leader) =================
            if (lane == 0) {
                const uint32_t idesc_S = make_idesc_f16(256, 128, true, true, false, false, true);
                ItemIter it; it.init(pr, p.n_pairs, p.n_items);
                uint32_t t = 0, gk = 0, item_n = 0;
                for (; it.valid(); it.next(), item_n++) {
                    Item item = decode(it.i, p);
                    HW_WAIT(bar(B_QREADY), item_n & 1);
                    for (int j = 0; j < item.T; j++, t++, gk++) {
                        const uint32_t sb = t & 1;
                        TR(t, 5);
                        if (t >= 2) HW_WAIT(bar(B_SFREE0 + sb), ((t - 2) >> 1) & 1);
                        TR(t, 6);
                        const uint32_t st = gk % NK;
                        HW_WAIT(bar(B_KFULL + st), (gk / NK) & 1);
                        TR(t, 7);
                        tc_fence_after();
                        const uint32_t d_tmem = tmem + TM_S0 + 128 * sb;
                        const uint64_t bdesc = make_smem_desc(sK + st * K_BYTES, 16, 1024, 2);
                        #pragma unroll
                        for (int k = 0; k < 8; k++)
                            mma2_ts(d_tmem, tmem + TM_Q + 8 * k, bdesc + (((k >> 2) * 8192 + (k & 3) * 32) >> 4), idesc_S, k > 0);
                        mma2_commit_mc(bar(B_SFULL0 + sb), 3);
                        mma2_commit_mc(bar(B_KEMPTY + st), 3);
                        TR(t, 8);
                    }
                }
            }
        } else {
            if (lane == 0) {   // K relay
                for (uint32_t g = 0; g < (uint32_t)total; g++) {
                    uint32_t st = g % NK;
                    HW_WAIT_LOCAL(bar(B_KFULL + st), (g / NK) & 1);
                    mbar_arrive_cluster(mapa_shared(bar(B_KFULL + st), 0));
                }
            }
        }
    } else if (warp == 2) {
        if (rank == 0) {
            // ================= PV issuer (leader) =================
            if (lane == 0) {
                const uint32_t idesc_PV = make_idesc_f16(256, 128, true, true, false, true, true);
                ItemIter it; it.init(pr, p.n_pairs, p.n_items);
                uint32_t t = 0, gv = 0;
                for (; it.valid(); it.next()) {
                    Item item = decode(it.i, p);
                    for (int j = 0; j < item.T; j++, t++, gv++) {
                        TR(t, 9);
                        HW_WAIT(bar(B_PFULL), t & 1);
                        TR(t, 10);
                        const uint32_t st = gv % NV;
                        HW_WAIT(bar(B_VFULL + st), (gv / NV) & 1);
                        TR(t, 11);
                        tc_fence_after();
                        const uint64_t bdesc = make_smem_desc(sV + st * V_BYTES, 16384, 1024, 2);
                        #pragma unroll
                        for (int k = 0; k < 8; k++)
                            mma2_ts(tmem + TM_O, tmem + TM_P + 8 * k, bdesc + ((k * 2048) >> 4), idesc_PV, (j > 0) || (k > 0));
                        mma2_commit_mc(bar(B_PVDONE), 3);
                        mma2_commit_mc(bar(B_VEMPTY + st), 3);
                        TR(t, 12);
#ifdef TRACE_PVLAT
                        mbar_wait_cluster(bar(B_PVDONE), t & 1);
                        TR(t, 14);
#endif
                    }
                }
            }
        } else {
            if (lane == 0) {   // V relay
                for (uint32_t g = 0; g < (uint32_t)total; g++) {
                    uint32_t st = g % NV;
                    HW_WAIT_LOCAL(bar(B_VFULL + st), (g / NV) & 1);
                    mbar_arrive_cluster(mapa_shared(bar(B_VFULL + st), 0));
                }
            }
        }
    } else if (warp >= 4) {
        // ================= softmax warpgroup (warps 4..7) =================
        const int wq = warp - 4;
        const uint32_t lane_base = (uint32_t)(wq * 32) << 16;
        const float c = p.scale_log2;
        const uint32_t remote_base = (rank == 1) ? mapa_shared(bar0, 0) : bar0;   // leader's barriers (for arrivals)
#ifdef RELEASE_ARRIVE
        auto arrive_leader = [&](int idx) { if (rank == 0) mbar_arrive(bar0 + idx * 8); else mbar_arrive_cluster(remote_base + idx * 8); };
#else
        auto arrive_leader = [&](int idx) { if (rank == 0) mbar_arrive_relaxed(bar0 + idx * 8); else mbar_arrive_cluster_relaxed(remote_base + idx * 8); };
#endif
        ItemIter it; it.init(pr, p.n_pairs, p.n_items);
        uint32_t t = 0, item_n = 0;
        float s[128];
        const int row_local = rank * 128 + wq * 32 + lane;
        // copy Q (smem, 128B-swizzled) -> TMEM [TM_Q, +64) for this thread's row
        auto copy_q = [&](uint32_t qn) {
            mbar_wait(bar(B_QFULL), qn & 1);
            const int r = wq * 32 + lane;
            uint32_t regs[64];
            #pragma unroll
            for (int sub = 0; sub < 2; sub++) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    uint4 v = lds128(sQ + sub * 16384 + r * 128 + ((i ^ (r & 7)) << 4));
                    regs[sub * 32 + i * 4 + 0] = v.x; regs[sub * 32 + i * 4 + 1] = v.y; regs[sub * 32 + i * 4 + 2] = v.z; regs[sub * 32 + i * 4 + 3] = v.w;
                }
            }
            #pragma unroll
            for (int ch = 0; ch < 4; ch++) tmem_st_32x32b_x16(tmem + TM_Q + lane_base + ch * 16, &regs[ch * 16]);
            tmem_st_wait();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) { mbar_arrive(bar(B_QEMPTY)); arrive_leader(B_QREADY); }
        };
        copy_q(0);
        for (; it.valid(); it.next(), item_n++) {
            Item item = decode(it.i, p);
            float m_s = -INFINITY, l = 0.f;
            for (int j = 0; j < item.T; j++, t++) {
                const uint32_t sb = t & 1;
                const uint32_t s_col = tmem + TM_S0 + 128 * sb;
                if (wq == 0) TR(t, 0);
                mbar_wait_cluster(bar(B_SFULL0 + sb), (t >> 1) & 1);
                tc_fence_after();
                if (wq == 0) TR(t, 1);
                #pragma unroll
                for (int ch = 0; ch < 4; ch++) tmem_ld_32x32b_x32(s_col + lane_base + ch * 32, (uint32_t*)&s[ch * 32]);
                tmem_ld_wait();
                if (wq == 0) TR(t, 2);
                tc_fence_before();
                __syncwarp();
                if (lane == 0) arrive_leader(B_SFREE0 + sb);
                // last tile of the item: Q is dead -> stage next item's Q into TMEM
                if (j == item.T - 1) { ItemIter nx = it; nx.next(); if (nx.valid()) copy_q(item_n + 1); }
#ifdef ABL_NO_SOFTMAX
                if (true) {
                    if (j > 0) { mbar_wait_cluster(bar(B_PVDONE), (t - 1) & 1); tc_fence_after(); }
                    tc_fence_before(); __syncwarp();
                    if (lane == 0) arrive_leader(B_PFULL);
                    continue;
                }
#endif
                if (j >= item.T - 2) {
                    const int lim = row_local - 128 * (j - 2 * item.blk);
                    #pragma unroll
                    for (int i = 0; i < 128; i++) if (i > lim) s[i] = -INFINITY;
                }
                float mx = rowmax128(s);
                if (wq == 0) TR(t, 14);
#ifdef ABL_NO_MAX
                mx = (j == 0) ? mx : m_s / c;
#endif
                const float m_new = mx * c;
                const bool resc = m_new > m_s + RESCALE_THRESH;
                float alpha = 1.f;
                if (resc) { alpha = ex2(m_s - m_new); m_s = m_new; l *= alpha; }
                const float neg_m = -m_s;
                uint32_t pk[64];
                softmax_exp_tile((const uint32_t*)s, c, neg_m, pk, l);
                if (wq == 0) TR(t, 15);
                // P region is single-buffered: wait for PV(t-1) before overwriting; also O stable for rescale
                if (wq == 0) TR(t, 3);
                if (j > 0) {
                    mbar_wait_cluster(bar(B_PVDONE), (t - 1) & 1);
                    tc_fence_after();
                    if (__any_sync(0xffffffffu, resc)) {
                        #pragma unroll
                        for (int ch = 0; ch < 4; ch++) {
                            uint32_t r[32];
                            tmem_ld_32x32b_x32(tmem + TM_O + lane_base + ch * 32, r);
                            tmem_ld_wait();
                            #pragma unroll
                            for (int i = 0; i < 32; i++) r[i] = __float_as_uint(__uint_as_float(r[i]) * alpha);
                            tmem_st_32x32b_x32(tmem + TM_O + lane_base + ch * 32, r);
                        }
                    }
                }
                if (wq == 0) TR(t, 4);
                #pragma unroll
                for (int ch = 0; ch < 4; ch++) tmem_st_32x32b_x16(tmem + TM_P + lane_base + ch * 16, &pk[ch * 16]);
                tmem_st_wait();
                tc_fence_before();
                __syncwarp();
                if (lane == 0) arrive_leader(B_PFULL);
                if (wq == 0) TR(t, 13);
            }
            // epilogue
            {
                const uint32_t tl = t - 1;
                mbar_wait_cluster(bar(B_PVDONE), tl & 1);
                tc_fence_after();
                const float inv_l = 1.f / l;
                const int srow = item.row0 + row_local;
                __nv_bfloat16* orow = p.out + ((size_t)srow * p.H + item.h) * D;
                #pragma unroll
                for (int ch = 0; ch < 4; ch++) {
                    uint32_t r[32];
                    tmem_ld_32x32b_x32(tmem + TM_O + lane_base + ch * 32, r);
                    tmem_ld_wait();
                    uint32_t pk[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) pk[i] = pack_bf16x2(__uint_as_float(r[2 * i]) * inv_l, __uint_as_float(r[2 * i + 1]) * inv_l);
                    uint4* dst = (uint4*)(orow + ch * 32);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) dst[i] = make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
                }
                tc_fence_before();
            }
        }
    }
    __syncwarp();
    __syncthreads();
    cluster_sync();
    if (warp == 3) tmem_dealloc2(tmem, 512);
}

extern "C" int fa_fwd_launch(void* q, void* k, void* v, void* out, int B, int S, int H, int Hkv, float scale, int n_pairs_req, void* trace) {
    using namespace cfg;
    if (S % 256 != 0) { printf("S must be a multiple of 256\n"); return -1; }
    CUtensorMap mapQ, mapK, mapV;
    int r1 = make_map_bf16_3d(&mapQ, q, 128, H, (uint64_t)B * S, 64, 128);
    int r2 = make_map_bf16_3d(&mapK, k, 128, Hkv, (uint64_t)B * S, 64, 64);
    int r3 = make_map_bf16_3d(&mapV, v, 128, Hkv, (uint64_t)B * S, 64, 128);
    if (r1 || r2 || r3) { printf("map encode failed %d %d %d\n", r1, r2, r3); return -1; }
    static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES); attr_set = true; }
    static int n_pairs = 0;
    if (n_pairs == 0) {
        cudaLaunchConfig_t cfgq = {}; cfgq.gridDim = dim3(2); cfgq.blockDim = dim3(NUM_THREADS); cfgq.dynamicSmemBytes = SMEM_BYTES;
        cudaLaunchAttribute attr[1]; attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = 2; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
        cfgq.attrs = attr; cfgq.numAttrs = 1;
        int nc = 0; cudaOccupancyMaxActiveClusters(&nc, fa_fwd_kernel, &cfgq);
        n_pairs = nc > 0 ? nc : 74;
    }
    Params p;
    p.out = (__nv_bfloat16*)out; p.trace = (long long*)trace; p.B = B; p.S = S; p.H = H; p.Hkv = Hkv;
    p.n_blocks = S / 256; p.n_items = B * H * p.n_blocks;
    p.n_pairs = n_pairs_req > 0 ? n_pairs_req : n_pairs;
    p.scale_log2 = scale * 1.4426950408889634f;
    fa_fwd_kernel<<<p.n_pairs * 2, NUM_THREADS, SMEM_BYTES>>>(mapQ, mapK, mapV, p);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(e)); return -2; }
    return p.n_pairs;
}
