// FlashAttention forward (causal, bf16, D=128) for Blackwell sm_100a, written from scratch.
// Design: 2-CTA cluster, tcgen05.mma cta_group::2 (M=256 = 128 q rows per CTA), TMA loads,
// each CTA holds half of every K/V tile (N split), S/P/O in TMEM, one softmax warpgroup per CTA.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "tmem_ops.h"
#ifdef LD64
#define TMEM_LD_S(addr, dst) do { tmem_ld_x64((addr), (uint32_t*)(dst)); tmem_ld_x64((addr) + 64, (uint32_t*)(dst) + 64); } while (0)
#else
#define TMEM_LD_S(addr, dst) tmem_ld_x128((addr), (uint32_t*)(dst))
#endif

#ifndef STAGES
#define STAGES 4
#endif
#ifndef EMU_MOD
#define EMU_MOD 0      // emulate exp2 with FMA polynomial for elements with (c % EMU_MOD == EMU_MOD-1); 0 => never
#endif
#ifndef DEBUG_MODE
#define DEBUG_MODE 0
#endif
#ifndef WITH_MCOMMIT
#define NO_MCOMMIT 1
#endif
#ifndef EXP_DEP_DIST
#define EXP_DEP_DIST 9
#endif
#ifndef TR0
#define TR0 200
#endif
#ifndef RESCALE_THRESH
#define RESCALE_THRESH 8.0f
#endif

namespace {

constexpr int BM = 128;          // q rows per CTA
constexpr int BN = 128;          // kv columns per tile (both CTAs together)
constexpr int HD = 128;          // head dim
constexpr int NUM_THREADS = 384;
#ifndef CTRL_REGS_V
#define CTRL_REGS_V 40
#define SOFTMAX_REGS_V 232
#endif
constexpr int CTRL_REGS = CTRL_REGS_V;
constexpr int SOFTMAX_REGS = SOFTMAX_REGS_V;
constexpr int Q_BYTES = BM * HD * 2;            // 32 KB
constexpr int K_HALF_BYTES = (BN / 2) * HD * 2; // 16 KB : 64 kv rows x 128 d
constexpr int V_HALF_BYTES = BN * (HD / 2) * 2; // 16 KB : 128 kv rows x 64 d
constexpr int SMEM_Q_OFF = 0;
constexpr int SMEM_K_OFF = SMEM_Q_OFF + 2 * Q_BYTES;
constexpr int SMEM_V_OFF = SMEM_K_OFF + STAGES * K_HALF_BYTES;
constexpr int ONES_BYTES = BN * 128;                       // 128 kv rows x 128B (one MN-major SW128 atom column of ones)
constexpr int SMEM_ONES_OFF = SMEM_V_OFF + STAGES * V_HALF_BYTES;
constexpr int OST_BYTES = BM * 64 * 2;                     // 16 KB: 128 rows x 64 cols bf16 output staging (SW128, one TMA box)
constexpr int SMEM_OST_OFF = SMEM_ONES_OFF + ONES_BYTES;
constexpr int SMEM_BAR_OFF = SMEM_OST_OFF + OST_BYTES;
constexpr int NUM_BARS = 4 + 4 * STAGES + 16;
constexpr int SMEM_TMEM_OFF = SMEM_BAR_OFF + NUM_BARS * 8;
constexpr int SMEM_EX_OFF = ((SMEM_TMEM_OFF + 16 + 127) / 128) * 128;   // m_sh[128], l_sh[2][128]
constexpr int SMEM_BYTES = SMEM_EX_OFF + 512;   // m_sh[128]
constexpr int SMEM_ALLOC = SMEM_BYTES + 1024;   // slack for 1024B alignment

// barrier indices
constexpr int BAR_Q_FULL = 0;          // [2] leader: count 2 (local tx-arrive + peer forwarder); peer: count 1
constexpr int BAR_Q_FREE = 2;          // [2] commit (multicast)
constexpr int BAR_K_FULL = 4;                    // [STAGES] leader: tx + peer fwd + 8 (both CTAs' WG loaded S_{g-2}); peer: 1
constexpr int BAR_V_FULL = 4 + STAGES;           // [STAGES] leader: 1 + 1 + 8 (WG stored P_n); peer: 1
constexpr int BAR_K_FREE = 4 + 2 * STAGES;       // [STAGES] local: owning WG warp0 confirms S_j done (count 1)
constexpr int BAR_V_FREE = 4 + 3 * STAGES;       // [STAGES] local: owning WG warp0 confirms PV_j done (count 1)
constexpr int BAR_S_FULL = 4 + 4 * STAGES;       // [2] commit: S_g done, barrier g&1
constexpr int BAR_PV_DONE = 6 + 4 * STAGES;      // [2] commit: PV_g done (per P buffer / WG)
constexpr int BAR_MCOMMIT = 8 + 4 * STAGES;      // [2] local: row max committed for tile g (count 4)
constexpr int BAR_OFREE = 10 + 4 * STAGES;       // [2] leader: epilogue warps of BOTH CTAs finished reading O for item t (count 8)
constexpr int BAR_EXPDONE = 12 + 4 * STAGES;     // [8] local: warp (wg, q) finished its exp phase (count 1) -> per-SMSP ping-pong; index wg*4+q
static_assert(BAR_EXPDONE + 8 == NUM_BARS, "bar count");

// TMEM column layout (512 columns)
constexpr int TMEM_S0 = 0;      // S buffers by tile parity (WG A: even, WG B: odd)
constexpr int TMEM_S1 = 128;
constexpr int TMEM_P0 = 256;    // single P buffer (P_g stored after PV_{g-1} completed)
constexpr int TMEM_P1 = 256;
constexpr int TMEM_O = 320;     // 160 columns: [O d0..63][sum x16][O d64..127][sum x16]  -> 480 columns total
constexpr int PV_N = 160;    // 2-CTA TS MMA needs N % 32 == 0: per CTA 80 = 64 (V) + 16 (ones)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive_local(uint32_t addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_arrive_local_pred(uint32_t addr, uint32_t pred) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %1, 0;\n@p mbarrier.arrive.shared::cta.b64 _, [%0];\n}" :: "r"(addr), "r"(pred) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster_pred(uint32_t addr, uint32_t cta, uint32_t pred) {
    asm volatile("{\n.reg .pred p;\n.reg .b32 ra;\nsetp.ne.b32 p, %2, 0;\nmapa.shared::cluster.u32 ra, %0, %1;\n@p mbarrier.arrive.shared::cluster.b64 _, [ra];\n}" :: "r"(addr), "r"(cta), "r"(pred) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t addr, uint32_t cta) {
    asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.shared::cluster.b64 _, [ra];\n}" :: "r"(addr), "r"(cta) : "memory");
}
#ifdef DEBUG_HANG
__shared__ uint32_t g_bar_base;
__shared__ int g_prog[32];
#endif
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
    uint32_t done;
#ifdef DEBUG_HANG
    long long spins = 0;
#endif
    do {
#ifdef SUSPEND_HINT
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\nselp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(addr), "r"(parity), "r"((uint32_t)SUSPEND_HINT) : "memory");
#else
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(addr), "r"(parity) : "memory");
#endif
#ifdef DEBUG_HANG
        if (!done && ++spins == (1ll << 22)) {
            printf("HANG block %d warp %d lane %d waiting bar idx %d parity %u prog %d wgA %d wgB %d\n", blockIdx.x, threadIdx.x >> 5, threadIdx.x & 31, (int)(addr - g_bar_base) / 8, parity, g_prog[threadIdx.x >> 5], g_prog[4], g_prog[8]);
        }
        if (!done && spins > (1ll << 25)) __trap();
#endif
    } while (!done);
}
__device__ __forceinline__ bool mbar_test(uint32_t addr, uint32_t parity) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(addr), "r"(parity) : "memory");
    return done != 0;
}
__device__ __forceinline__ void mbar_wait_warp(uint32_t addr, uint32_t parity) {
#if DEBUG_MODE == 7 || DEBUG_MODE == 8 || defined(NO_WAITS)
    return;
#endif
#ifdef TESTWAIT_FAST
    if (mbar_test(addr, parity)) return;
#endif
#ifdef LANE0_POLL
    if ((threadIdx.x & 31) == 0) mbar_wait(addr, parity);
    __syncwarp();
#else
    mbar_wait(addr, parity);   // all lanes poll (converged)
#endif
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* tmap, uint32_t mbar, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(dst), "l"(tmap), "r"(mbar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* tmap, uint32_t src, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
                 :: "l"(tmap), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_commit_mc(uint32_t mbar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mbar), "h"(mask) : "memory");
}
__device__ __forceinline__ void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void wg_bar() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

// UMMA shared memory descriptor (SWIZZLE_128B). addr: smem byte address of the (1024B-aligned atom) + k offset.
__device__ __forceinline__ uint64_t make_desc_base(uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = 0;
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;      // version
    d |= (uint64_t)2 << 61;      // SWIZZLE_128B
    return d;
}
__device__ __forceinline__ uint64_t desc_with_addr(uint64_t base, uint32_t addr) {
    return base | (uint64_t)((addr >> 4) & 0x3FFF);
}
__device__ __forceinline__ uint32_t make_idesc(int M, int N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= 1u << 4;    // D f32
    d |= 1u << 7;    // A bf16
    d |= 1u << 10;   // B bf16
    d |= (uint32_t)a_major << 15;
    d |= (uint32_t)b_major << 16;
    d |= (uint32_t)(N >> 3) << 17;
    d |= (uint32_t)(M >> 4) << 24;
    return d;
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
// polynomial exp2 for x <= 0 (FMA pipe), rel err ~6e-4
__device__ __forceinline__ float poly_exp2(float x) {
    x = fmaxf(x, -126.0f);
    float t = __fadd_rn(x, 12582912.0f);         // 1.5 * 2^23 : t holds round(x) in low mantissa bits
    float nf = __fsub_rn(t, 12582912.0f);
    float f = __fsub_rn(x, nf);                  // [-0.5, 0.5]
    float p = fmaf(f, 0.05550411f, 0.24022652f);
    p = fmaf(p, f, 0.69314718f);
    p = fmaf(p, f, 1.0f);
    int bits = __float_as_int(p) + __float_as_int(t) * (1 << 23);
    return __int_as_float(bits);
}
__device__ __forceinline__ float fmax3(float a, float b, float c) {
    float r; asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ uint32_t ex2_bf16x2(uint32_t x) {
    uint32_t r; asm("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(r) : "r"(x)); return r;
}
__device__ __forceinline__ uint32_t pack_bf16x2_trunc(float lo, float hi) {
    uint32_t r; asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(r) : "r"(__float_as_uint(lo)), "r"(__float_as_uint(hi))); return r;
}
#ifdef PACK_TRUNC
#define PACK_P pack_bf16x2_trunc
#else
#define PACK_P pack_bf16x2
#endif
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

struct WorkItem { int b, hkv, i, hp; };
#ifdef DEBUG_TIMING
#define TSTART(v) long long v = clock64()
#define TACC(acc, v) acc += clock64() - v
#else
#define TSTART(v)
#define TACC(acc, v)
#endif

// item ordering: blocks of 4 consecutive q-tile indices (descending), (b, hkv) major inside a block
__device__ __forceinline__ WorkItem decode_item(int p, int B, int Hkv, int NT, int HP) {
    constexpr int BLK = 4;
    int per_blk = B * Hkv * BLK * HP;
    int ib = p / per_blk;
    int idx = p - ib * per_blk;
    int b = idx / (Hkv * BLK * HP);
    int rem = idx - b * (Hkv * BLK * HP);
    int hkv = rem / (BLK * HP);
    int rem2 = rem - hkv * (BLK * HP);
    int ii = rem2 / HP;
    int hp = rem2 - ii * HP;
    WorkItem w; w.b = b; w.hkv = hkv; w.i = NT - 1 - ib * BLK - ii; w.hp = hp;
    return w;
}

} // namespace


__global__ void __launch_bounds__(NUM_THREADS, 1)
fa_fwd_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k,
              const __grid_constant__ CUtensorMap tmap_v, const __grid_constant__ CUtensorMap tmap_o, __nv_bfloat16* __restrict__ out,
              int B, int S, int Hq, int Hkv, float scale_log2, long long* __restrict__ dbg)
{
    extern __shared__ uint8_t smem_raw[];
    const uint32_t smem_base_raw = smem_u32(smem_raw);
    const uint32_t smem_base = (smem_base_raw + 1023) & ~1023u;
    uint8_t* smem = smem_raw + (smem_base - smem_base_raw);

    const uint32_t rank = cluster_ctarank();       // 0 = leader
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int NT = S / BM;
    const int G = Hq / Hkv;
    const int HP = G / 2;
    const int num_items = B * Hkv * NT * HP;
    const int num_clusters = gridDim.x / 2;
    const int cluster_id = blockIdx.x / 2;

    const uint32_t bar_base = smem_base + SMEM_BAR_OFF;
    auto bar = [&](int idx) -> uint32_t { return bar_base + idx * 8; };
    uint32_t* tmem_slot = (uint32_t*)(smem + SMEM_TMEM_OFF);
    float* mx_ex = (float*)(smem + SMEM_EX_OFF);          // [2][128][2]
    float* l_ex = mx_ex + 512;                             // [128][2]

#ifdef DEBUG_HANG
    if (threadIdx.x == 0) g_bar_base = bar_base;
#endif
    if (threadIdx.x == 0) {
        const uint32_t lead = (rank == 0) ? 1u : 0u;
        for (int i = 0; i < 2; i++) { mbar_init(bar(BAR_Q_FULL + i), lead ? 2 : 1); mbar_init(bar(BAR_Q_FREE + i), 1); }
        for (int st = 0; st < STAGES; st++) {
            mbar_init(bar(BAR_K_FULL + st), lead ? 10 : 1); mbar_init(bar(BAR_V_FULL + st), lead ? 10 : 1);   // tx + peer fwd + 4 warps x 2 CTAs
            mbar_init(bar(BAR_K_FREE + st), 1); mbar_init(bar(BAR_V_FREE + st), 1);
        }
        for (int i = 0; i < 2; i++) { mbar_init(bar(BAR_S_FULL + i), 1); mbar_init(bar(BAR_PV_DONE + i), 1); mbar_init(bar(BAR_MCOMMIT + i), 4); mbar_init(bar(BAR_OFREE + i), lead ? 8 : 1); }
        for (int i = 0; i < 8; i++) mbar_init(bar(BAR_EXPDONE + i), 1);   // OFREE: both CTAs' epilogue warps arrive on the leader
        if (lead) { for (int i = 0; i < 8; i++) { mbar_arrive_local(bar(BAR_K_FULL + 0)); mbar_arrive_local(bar(BAR_K_FULL + 1)); } }   // tiles 0,1 have no S_{g-2}
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    {   // ones block (bf16 1.0) for the row-sum columns of the PV MMA
        uint32_t* ones = (uint32_t*)(smem + SMEM_ONES_OFF);
        for (int i = threadIdx.x; i < ONES_BYTES / 4; i += NUM_THREADS) ones[i] = 0x3F803F80u;
    }
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(tmem_slot)), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    cluster_sync();
    tc_fence_after();
    const uint32_t tmem_base = *tmem_slot;
    float* m_sh = (float*)(smem + SMEM_EX_OFF);          // [128]
    const uint32_t ones_smem = smem_base + SMEM_ONES_OFF;

    const uint32_t q_smem = smem_base + SMEM_Q_OFF;
    const uint32_t k_smem = smem_base + SMEM_K_OFF;
    const uint32_t v_smem = smem_base + SMEM_V_OFF;

    if (warp < 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(CTRL_REGS));
#if DEBUG_MODE == 12 || defined(CTRL_IDLE)
        if (true) {} else if (false) {
#else
        if (warp == 0 && lane == 0) {
#endif
            // ================= Q + K producer =================
            uint32_t q_iter = 0, kv_iter = 0;
            for (int p = cluster_id; p < num_items; p += num_clusters, q_iter++) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                const int h = w.hkv * G + w.hp * 2 + (int)rank;
                const int qbuf = q_iter & 1;
                if (q_iter >= 2) mbar_wait(bar(BAR_Q_FREE + qbuf), ((q_iter >> 1) - 1) & 1);
                const int qrow = w.b * S + w.i * BM;
#ifdef NO_TMA
                mbar_arrive_local(bar(BAR_Q_FULL + qbuf));
#else
                mbar_expect_tx(bar(BAR_Q_FULL + qbuf), Q_BYTES);
                tma_load_3d(q_smem + qbuf * Q_BYTES, &tmap_q, bar(BAR_Q_FULL + qbuf), 0, h, qrow);
                tma_load_3d(q_smem + qbuf * Q_BYTES + Q_BYTES / 2, &tmap_q, bar(BAR_Q_FULL + qbuf), 64, h, qrow);
#endif
                for (int j = 0; j <= w.i; j++, kv_iter++) {
                    const int stage = kv_iter % STAGES;
                    if (kv_iter >= STAGES) mbar_wait(bar(BAR_K_FREE + stage), ((kv_iter / STAGES) - 1) & 1);
                    const int kvrow = w.b * S + j * BN + (int)rank * (BN / 2);
#ifdef NO_TMA
                    mbar_arrive_local(bar(BAR_K_FULL + stage));
#else
                    mbar_expect_tx(bar(BAR_K_FULL + stage), K_HALF_BYTES);
                    tma_load_3d(k_smem + stage * K_HALF_BYTES, &tmap_k, bar(BAR_K_FULL + stage), 0, w.hkv, kvrow);
                    tma_load_3d(k_smem + stage * K_HALF_BYTES + K_HALF_BYTES / 2, &tmap_k, bar(BAR_K_FULL + stage), 64, w.hkv, kvrow);
#endif
                }
            }
#ifdef SWAP_PV
        } else if (warp == 2 && lane == 0) {
#else
        } else if (warp == 3 && lane == 0) {
#endif
            // ================= V producer =================
            uint32_t kv_iter = 0;
            for (int p = cluster_id; p < num_items; p += num_clusters) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                for (int j = 0; j <= w.i; j++, kv_iter++) {
                    const int stage = kv_iter % STAGES;
                    if (kv_iter >= STAGES) mbar_wait(bar(BAR_V_FREE + stage), ((kv_iter / STAGES) - 1) & 1);
                    const int kvrow = w.b * S + j * BN;
#ifdef NO_TMA
                    mbar_arrive_local(bar(BAR_V_FULL + stage));
#else
                    mbar_expect_tx(bar(BAR_V_FULL + stage), V_HALF_BYTES);
                    tma_load_3d(v_smem + stage * V_HALF_BYTES, &tmap_v, bar(BAR_V_FULL + stage), (int)rank * (HD / 2), w.hkv, kvrow);
#endif
                }
            }
        } else if (warp == 1 && lane == 0 && rank == 1) {
            // ================= peer: Q/K completion forwarder =================
            uint32_t q_iter = 0, kv_iter = 0;
            for (int p = cluster_id; p < num_items; p += num_clusters, q_iter++) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                const int qbuf = q_iter & 1;
                mbar_wait(bar(BAR_Q_FULL + qbuf), (q_iter >> 1) & 1);
                mbar_arrive_cluster(bar(BAR_Q_FULL + qbuf), 0);
                for (int j = 0; j <= w.i; j++, kv_iter++) {
                    const int stage = kv_iter % STAGES;
                    mbar_wait(bar(BAR_K_FULL + stage), (kv_iter / STAGES) & 1);
                    mbar_arrive_cluster(bar(BAR_K_FULL + stage), 0);
                }
            }
#ifdef SWAP_PV
        } else if (warp == 3 && lane == 0 && rank == 1) {
#else
        } else if (warp == 2 && lane == 0 && rank == 1) {
#endif
            // ================= peer: V completion forwarder =================
            uint32_t kv_iter = 0;
            for (int p = cluster_id; p < num_items; p += num_clusters) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                for (int j = 0; j <= w.i; j++, kv_iter++) {
                    const int stage = kv_iter % STAGES;
                    mbar_wait(bar(BAR_V_FULL + stage), (kv_iter / STAGES) & 1);
                    mbar_arrive_cluster(bar(BAR_V_FULL + stage), 0);
                }
            }
        } else if (warp == 1 && lane == 0 && rank == 0) {
            // ================= S = Q K^T issuer (leader) =================
            const uint64_t qdesc_base = make_desc_base(16, 1024);
            const uint64_t kdesc_base = make_desc_base(16, 1024);
            const uint32_t idesc_s = make_idesc(256, BN, 0, 0);
            uint32_t q_iter = 0, s_iter = 0;
            long long t_kr = 0, t_qr = 0, t_tot = 0, n_tiles = 0;
            TSTART(t_begin);
            for (int p = cluster_id; p < num_items; p += num_clusters, q_iter++) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                const int qbuf = q_iter & 1;
                { TSTART(tq); mbar_wait(bar(BAR_Q_FULL + qbuf), (q_iter >> 1) & 1); TACC(t_qr, tq); }
                tc_fence_after();
                const uint32_t qaddr = q_smem + qbuf * Q_BYTES;
                for (int j = 0; j <= w.i; j++, s_iter++) {
                    const uint32_t s_idx = s_iter;
#ifdef DEBUG_HANG
                    g_prog[warp] = (int)s_idx;
#endif
#ifdef DEBUG_TRACE
#define TRS(idx) do { if (dbg && blockIdx.x == 0 && s_idx >= TR0 && s_idx < TR0 + 16) dbg[2048 + (s_idx - TR0) * 16 + (idx)] = clock64(); } while (0)
#else
#define TRS(idx)
#endif
                    TRS(0);
                    { TSTART(tb); mbar_wait(bar(BAR_K_FULL + (s_idx % STAGES)), (s_idx / STAGES) & 1); TACC(t_kr, tb); }
                    TRS(1);
                    n_tiles++;
                    tc_fence_after();
                    const uint32_t kaddr = k_smem + (s_idx % STAGES) * K_HALF_BYTES;
                    const uint32_t s_tmem = tmem_base + ((s_idx & 1) ? TMEM_S1 : TMEM_S0);
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const uint32_t qa = qaddr + (k >> 2) * (Q_BYTES / 2) + (k & 3) * 32;
                        const uint32_t ka = kaddr + (k >> 2) * (K_HALF_BYTES / 2) + (k & 3) * 32;
#if DEBUG_MODE != 3 && DEBUG_MODE != 5 && !defined(SKIP_MMA)
                        mma_ss(s_tmem, desc_with_addr(qdesc_base, qa), desc_with_addr(kdesc_base, ka), idesc_s, k > 0 ? 1u : 0u);
#endif
                    }
                    TRS(2);
                    tc_commit_mc(bar(BAR_S_FULL + (s_idx & 1)), 0x3);
                    if (j == w.i) tc_commit_mc(bar(BAR_Q_FREE + qbuf), 0x3);
                }
            }
            TACC(t_tot, t_begin);
#ifdef DEBUG_TIMING
            if (dbg) { long long* d = dbg + blockIdx.x * 32; d[1] = t_kr; d[4] = t_qr; d[5] = t_tot; d[6] = n_tiles; }
#endif
#ifdef SWAP_PV
        } else if (warp == 3 && lane == 0 && rank == 0) {
#else
        } else if (warp == 2 && lane == 0 && rank == 0) {
#endif
            // ================= O += P V issuer (leader) =================
            const uint32_t idesc_pv = make_idesc(256, PV_N, 0, 1);
            uint32_t p_iter = 0, item = 0;
            long long t_vr = 0, t_tot = 0;
            TSTART(t_begin);
            for (int p = cluster_id; p < num_items; p += num_clusters, item++) {
                WorkItem w = decode_item(p, B, Hkv, NT, HP);
                for (int j = 0; j <= w.i; j++, p_iter++) {
                    const uint32_t p_idx = p_iter;
#ifdef DEBUG_HANG
                    g_prog[warp] = (int)p_idx;
#endif
#ifdef DEBUG_TRACE
#define TRP(idx) do { if (dbg && blockIdx.x == 0 && p_idx >= TR0 && p_idx < TR0 + 16) dbg[2048 + (p_idx - TR0) * 16 + (idx)] = clock64(); } while (0)
#else
#define TRP(idx)
#endif
                    TRP(4);
                    { TSTART(tb); mbar_wait(bar(BAR_V_FULL + (p_idx % STAGES)), (p_idx / STAGES) & 1); TACC(t_vr, tb); }
                    if (j == 0 && item >= 1) mbar_wait(bar(BAR_OFREE + ((item - 1) & 1)), ((item - 1) >> 1) & 1);   // previous item's O fully read
                    TRP(5);
                    tc_fence_after();
                    const uint32_t vaddr = v_smem + (p_idx % STAGES) * V_HALF_BYTES;
                    const uint64_t vdesc_base = make_desc_base(ones_smem - vaddr, 1024);   // LBO: second N-atom = ones block
                    const uint32_t p_tmem = tmem_base + TMEM_P0;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
#if DEBUG_MODE != 3 && DEBUG_MODE != 5 && !defined(SKIP_MMA)
                        mma_ts(tmem_base + TMEM_O, p_tmem + k * 8, desc_with_addr(vdesc_base, vaddr + k * 2048), idesc_pv, (j > 0 || k > 0) ? 1u : 0u);
#endif
                    }
                    TRP(6);
                    tc_commit_mc(bar(BAR_PV_DONE + (p_idx & 1)), 0x3);
                }
            }
            TACC(t_tot, t_begin);
#ifdef DEBUG_TIMING
            if (dbg) { long long* d = dbg + blockIdx.x * 32; d[3] = t_vr; d[7] = t_tot; }
#endif
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(SOFTMAX_REGS));
#if DEBUG_MODE == 12
#ifdef M12_ONLY_A
        if (warp < 8)
#endif
        {   // microbenchmark loop inside the kernel: constant inputs, no barriers
            float s[128]; uint32_t pk[64];
            for (int c = 0; c < 128; c++) s[c] = -(float)((threadIdx.x * 7 + c * 13) % 97) * 0.3f;
            long long t0 = clock64();
            for (int it = 0; it < 256; it++) {
                const float nm = -2.f - (float)it * 1e-6f;
#pragma unroll
                for (int c = 0; c < 128; c += 2) { const float x0 = fmaf(s[c], scale_log2, nm), x1 = fmaf(s[c + 1], scale_log2, nm); pk[c / 2] = pack_bf16x2(fast_exp2(x0), fast_exp2(x1)); }
#pragma unroll
                for (int c = 0; c < 128; c += 2) { s[c] += __uint_as_float(pk[c / 2] & 0xffff0000u) * 1e-9f; }
            }
            long long t1 = clock64();
            float acc = 0; for (int c = 0; c < 64; c++) acc += __uint_as_float(pk[c]);
            if (dbg && lane == 0) { dbg[blockIdx.x * 32 + 8 + (warp - 4)] = (t1 - t0) / 256; dbg[blockIdx.x * 32 + 20] = __float_as_int(acc); }
        }
        if (false)
#endif
        {   // softmax body
        // ================= softmax warpgroups: WG A (warps 4-7) even tiles, WG B (warps 8-11) odd tiles =================
        // 32x32b layout: thread owns row r = (warp&3)*32 + lane, all 128 columns.
        const int wg = (warp >= 8) ? 1 : 0;
        const int r = (warp & 3) * 32 + lane;
        const uint32_t lane_base = (uint32_t)((warp & 3) * 32) << 16;
        const uint32_t s_addr = tmem_base + lane_base + (wg ? TMEM_S1 : TMEM_S0);
        const uint32_t p_addr = tmem_base + lane_base + TMEM_P0;
        const uint32_t o_addr = tmem_base + lane_base + TMEM_O;
        const uint32_t is_l0 = (lane == 0) ? 1u : 0u;
        const uint32_t is_w0l0 = ((warp & 3) == 0 && lane == 0) ? 1u : 0u;
        const uint32_t ost_smem = smem_base + SMEM_OST_OFF;
        uint8_t* ost_ptr = smem + SMEM_OST_OFF;
        auto arrive_leader = [&](uint32_t b) { if (rank == 0) mbar_arrive_local_pred(b, is_l0); else mbar_arrive_cluster_pred(b, 0, is_l0); };
        float s[128];
        uint32_t pk[64];
        long long t_ws = 0, t_wpv = 0, t_comp = 0, t_epi = 0, t_tot = 0, n_resc = 0, t_ld = 0, t_st = 0, t_mc = 0, t_pp = 0;
#ifdef DEBUG_TRACE
#ifdef TRACE_WARPS
#define TRWW(idx) do { if (dbg && blockIdx.x == 0 && lane == 0 && g >= TR0 && g < TR0 + 16) dbg[4096 + (g - TR0) * 16 + (warp - 4) * 2 + (idx)] = clock64(); } while (0)
#define TRWF(idx)
#elif defined(TRACE_WARPS_FINE)
#define TRWW(idx)
#define TRWF(idx) do { if (dbg && blockIdx.x == 0 && lane == 0 && warp < 8 && g >= TR0 && g < TR0 + 16) dbg[4096 + (g - TR0) * 16 + (warp - 4) * 4 + (idx)] = clock64(); } while (0)
#else
#define TRWW(idx)
#define TRWF(idx)
#endif
#if defined(TRACE_WARPS) || defined(TRACE_WARPS_FINE)
#define TRW(idx)
#else
#define TRW(idx) do { if (dbg && blockIdx.x == 0 && lane == 0 && (warp == 4 || warp == 8) && g >= TR0 && g < TR0 + 16) dbg[4096 + (g - TR0) * 16 + (warp == 8 ? 8 : 0) + (idx)] = clock64(); } while (0)
#endif
#else
#define TRW(idx)
#define TRWW(idx)
#define TRWF(idx)
#endif
        TSTART(t_begin);
        uint32_t g_item = 0;       // global index of the item's first tile
        uint32_t n = 0;            // number of my tiles processed
        uint32_t item = 0;
        bool s_loaded = false;     // S of the current tile already prefetched into s[] (before an epilogue)
        for (int p = cluster_id; p < num_items; p += num_clusters, item++) {
            WorkItem w = decode_item(p, B, Hkv, NT, HP);
            const int h = w.hkv * G + w.hp * 2 + (int)rank;
            const int ni = w.i + 1;
            const int j0 = ((int)(g_item & 1) == wg) ? 0 : 1;      // my first tile in this item
            if (j0 >= ni) { g_item += ni; continue; }
            for (int j = j0; j < ni; j += 2) {
                const uint32_t g = g_item + j;
                const bool last_item = (j == w.i);
#ifdef DEBUG_HANG
                if (lane == 0) g_prog[warp] = (int)g * 8;
#endif
                TRW(0);
                if (!s_loaded) {
                    { TSTART(ta); mbar_wait_warp(bar(BAR_S_FULL + wg), (g >> 1) & 1); TACC(t_ws, ta); }
                    TRW(1); TRWF(0);
                    TSTART(tl);
                    mbar_arrive_local_pred(bar(BAR_K_FREE + (g % STAGES)), is_w0l0);
                    tc_fence_after();
#if DEBUG_MODE == 9 || defined(B_IDLE)
                    if (wg == 0)
#endif
                    { TMEM_LD_S(s_addr, s); tc_wait_ld(); }
                    TRWF(1);
                    tc_fence_before(); __syncwarp();
                    arrive_leader(bar(BAR_K_FULL + ((g + 2) % STAGES)));      // S_g loaded -> S_{g+2} (same buffer) may proceed
                    TACC(t_ld, tl);
                }
                s_loaded = false;
#if DEBUG_MODE == 9 || defined(B_IDLE)
                const bool skip_compute = (wg == 1);
#else
                const bool skip_compute = false;
#endif
                TRW(2);
                TSTART(tc);
                if (last_item) {
#pragma unroll
                    for (int c = 0; c < 128; c++) s[c] = (c > r) ? -INFINITY : s[c];
                }
                float mx = 0.f;
                if (!skip_compute) {
#ifdef MAX2
                    float m1[64];
#pragma unroll
                    for (int c = 0; c < 64; c++) m1[c] = fmaxf(s[2 * c], s[2 * c + 1]);
#pragma unroll
                    for (int c = 0; c < 32; c++) m1[c] = fmaxf(m1[2 * c], m1[2 * c + 1]);
#pragma unroll
                    for (int c = 0; c < 16; c++) m1[c] = fmaxf(m1[2 * c], m1[2 * c + 1]);
#pragma unroll
                    for (int c = 0; c < 8; c++) m1[c] = fmaxf(m1[2 * c], m1[2 * c + 1]);
                    mx = fmaxf(fmaxf(fmaxf(m1[0], m1[1]), fmaxf(m1[2], m1[3])), fmaxf(fmaxf(m1[4], m1[5]), fmaxf(m1[6], m1[7])));
#else
                    float m1[32];
#pragma unroll
                    for (int c = 0; c < 32; c++) m1[c] = fmax3(s[4 * c], s[4 * c + 1], fmaxf(s[4 * c + 2], s[4 * c + 3]));
#pragma unroll
                    for (int c = 0; c < 8; c++) m1[c] = fmax3(m1[4 * c], m1[4 * c + 1], fmaxf(m1[4 * c + 2], m1[4 * c + 3]));
                    mx = fmax3(fmax3(m1[0], m1[1], m1[2]), fmax3(m1[3], m1[4], m1[5]), fmaxf(m1[6], m1[7]));
#endif
                }
#ifdef EXP_CONST_IN
#pragma unroll
                for (int c = 0; c < 128; c++) s[c] = -(float)c * 0.01f;   // timing experiment: exp inputs not sourced from TMEM registers
#endif
                const float mxs = mx * scale_log2;
                float m_new, alpha_o = 1.f;
                bool resc = false;
#ifdef DEBUG_HANG
                if (lane == 0) g_prog[warp] = (int)g * 8 + 1;
#endif
                TRWF(2);
#ifdef NO_MCOMMIT
                // ordering: the other WG's exp phase (and thus its m_sh write) completed before our ping-pong wait; decide m after it
                TRW(6);
                { TSTART(tpp);
#ifdef PP_SPIN
                  if (wg == 0) { if (n >= 1) { while (!mbar_test(bar(BAR_EXPDONE + 4 + (warp & 3)), (n - 1) & 1)) {} } }
                  else         { while (!mbar_test(bar(BAR_EXPDONE + 0 + (warp & 3)), n & 1)) {} }
#else
                  if (wg == 0) { if (n >= 1) mbar_wait_warp(bar(BAR_EXPDONE + 4 + (warp & 3)), (n - 1) & 1); }
                  else         { mbar_wait_warp(bar(BAR_EXPDONE + 0 + (warp & 3)), n & 1); }
#endif
                  TACC(t_pp, tpp); }
                TRW(7);
                if (j == 0) {
                    m_new = mxs;
                    m_sh[r] = m_new;
                } else {
                    const float m_cur = m_sh[r];
                    m_new = m_cur;
                    if (mxs > m_cur + RESCALE_THRESH) { m_new = mxs; m_sh[r] = m_new; alpha_o = fast_exp2(m_cur - m_new); resc = true; }
                }
#else
                { TSTART(tm); if (g >= 1) mbar_wait_warp(bar(BAR_MCOMMIT + ((g - 1) & 1)), ((g - 1) >> 1) & 1); TACC(t_mc, tm); }
                if (j == 0) {
                    m_new = mxs;
                    m_sh[r] = m_new;
                } else {
                    const float m_cur = m_sh[r];
                    m_new = m_cur;
                    if (mxs > m_cur + RESCALE_THRESH) { m_new = mxs; m_sh[r] = m_new; alpha_o = fast_exp2(m_cur - m_new); resc = true; }
                }
                __syncwarp();
                mbar_arrive_local_pred(bar(BAR_MCOMMIT + (g & 1)), is_l0);
#endif
                TRW(3); TRWW(0); TRWF(3);
                if (__any_sync(0xffffffff, resc)) {
                    n_resc++;
                    mbar_wait_warp(bar(BAR_PV_DONE + ((g - 1) & 1)), ((g - 1) >> 1) & 1);   // single in-order PV issuer: all earlier PVs done
                    tc_fence_after();
#pragma unroll
                    for (int qq = 0; qq < PV_N / 32; qq++) {   // all 160 columns (O and row sums)
                        uint32_t o[32];
                        tmem_ld_x32(o_addr + qq * 32, o);
                        tc_wait_ld();
#pragma unroll
                        for (int c = 0; c < 32; c++) o[c] = __float_as_uint(__uint_as_float(o[c]) * alpha_o);
                        tmem_st_x32(o_addr + qq * 32, o);
                    }
                    tc_wait_st();
                }
#if !defined(NO_PINGPONG) && !defined(NO_MCOMMIT)
                { TSTART(tpp);
                  if (wg == 0) { if (n >= 1) mbar_wait_warp(bar(BAR_EXPDONE + 4 + (warp & 3)), (n - 1) & 1); }
                  else         { mbar_wait_warp(bar(BAR_EXPDONE + 0 + (warp & 3)), n & 1); }
                  TACC(t_pp, tpp); }
#endif
#if defined(EXP_SWP)
                if (!skip_compute) {   // software-pipelined source order: MUFUs of group k, then pack of group k-1
                    const float nm = -m_new;
                    float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
#pragma unroll
                    for (int c = 0; c < 128; c += 4) {
                        const float p0 = fast_exp2(fmaf(s[c], scale_log2, nm));
                        const float p1 = fast_exp2(fmaf(s[c + 1], scale_log2, nm));
                        const float p2 = fast_exp2(fmaf(s[c + 2], scale_log2, nm));
                        const float p3 = fast_exp2(fmaf(s[c + 3], scale_log2, nm));
                        if (c > 0) { pk[c / 2 - 2] = PACK_P(q0, q1); pk[c / 2 - 1] = PACK_P(q2, q3); }
                        q0 = p0; q1 = p1; q2 = p2; q3 = p3;
                    }
                    pk[62] = PACK_P(q0, q1); pk[63] = PACK_P(q2, q3);
                }
#elif defined(EXP_DEP)
                if (!skip_compute) {   // all MUFUs in place; each pack carries a real (zero-weight) dependency on a MUFU issued 9 elements later
                    const float nm = -m_new;
#pragma unroll
                    for (int c = 0; c < 128; c++) s[c] = fast_exp2(fmaf(s[c], scale_log2, nm));
#pragma unroll
                    for (int c = 0; c < 128; c += 2) {
                        const float dep = s[(c + EXP_DEP_DIST < 128) ? c + EXP_DEP_DIST : 127];
                        const float p1 = fmaf(dep, 0.f, s[c + 1]);
                        pk[c / 2] = PACK_P(s[c], p1);
                    }
                }
#elif defined(EXP_PHASED)
                if (!skip_compute) {   // three explicit phases: all FFMA, then all MUFU (in place, volatile), then all packs
                    const float nm = -m_new;
#pragma unroll
                    for (int c = 0; c < 128; c++) s[c] = fmaf(s[c], scale_log2, nm);
#pragma unroll
                    for (int c = 0; c < 128; c++) { float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(s[c])); s[c] = y; }
#pragma unroll
                    for (int c = 0; c < 128; c += 2) { uint32_t r; asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(s[c + 1]), "f"(s[c])); pk[c / 2] = r; }
                }
#else
                if (!skip_compute) {
                    const float nm = -m_new;
#pragma unroll
                    for (int c = 0; c < 128; c += 4) {
                        const float x0 = fmaf(s[c], scale_log2, nm);
                        const float x1 = fmaf(s[c + 1], scale_log2, nm);
                        const float x2 = fmaf(s[c + 2], scale_log2, nm);
                        const float x3 = fmaf(s[c + 3], scale_log2, nm);
#ifdef EXP_BF16X2
                        pk[c / 2] = ex2_bf16x2(pack_bf16x2(x0, x1));
                        pk[c / 2 + 1] = ex2_bf16x2(pack_bf16x2(x2, x3));
                        continue;
#endif
                        float p0, p1, p2, p3;
#if EMU_MOD == 8
                        p0 = fast_exp2(x0); p1 = fast_exp2(x1); p2 = fast_exp2(x2);
                        if (c & 4) p3 = poly_exp2(x3); else p3 = fast_exp2(x3);
#elif EMU_MOD == 4
                        p0 = fast_exp2(x0); p1 = fast_exp2(x1); p2 = fast_exp2(x2); p3 = poly_exp2(x3);
#elif EMU_MOD == 3
                        p0 = fast_exp2(x0); p2 = fast_exp2(x2); p3 = poly_exp2(x3);
                        if (c & 4) p1 = poly_exp2(x1); else p1 = fast_exp2(x1);
#elif EMU_MOD == 2
                        p0 = fast_exp2(x0); p1 = poly_exp2(x1); p2 = fast_exp2(x2); p3 = poly_exp2(x3);
#else
                        p0 = fast_exp2(x0); p1 = fast_exp2(x1); p2 = fast_exp2(x2); p3 = fast_exp2(x3);
#endif
                        pk[c / 2] = PACK_P(p0, p1);
                        pk[c / 2 + 1] = PACK_P(p2, p3);
                    }
                }
#endif
#ifndef NO_PINGPONG
                __syncwarp();
                mbar_arrive_local_pred(bar(BAR_EXPDONE + wg * 4 + (warp & 3)), is_l0);
#endif
                TACC(t_comp, tc);
                TRW(4);
                TSTART(tst);
                // ---- store P_g (buffer wg free once PV_{g-2} done) ----
                if (g >= 1) {   // single P buffer: P_g may be written once PV_{g-1} has consumed the previous P
#ifdef DEBUG_HANG
                    if (lane == 0) g_prog[warp] = (int)g * 8 + 2;
#endif
                    { TSTART(ta); mbar_wait_warp(bar(BAR_PV_DONE + ((g - 1) & 1)), ((g - 1) >> 1) & 1); TACC(t_wpv, ta); }
                    mbar_arrive_local_pred(bar(BAR_V_FREE + ((g - 1) % STAGES)), is_w0l0);   // V stage of tile g-1 is free
                }
                tc_fence_after();
                if (!skip_compute) { tmem_st_x64(p_addr, pk); tc_wait_st(); }
                tc_fence_before(); __syncwarp();
                arrive_leader(bar(BAR_V_FULL + (g % STAGES)));            // P_g ready
                TACC(t_st, tst);
                TRW(5); TRWW(1);
                n++;
                {
                    if (last_item) {
                        // ---- epilogue by the WG owning the last tile ----
                        TSTART(te);
#ifdef DEBUG_HANG
                        if (lane == 0) g_prog[warp] = (int)g * 8 + 3;
#endif
                        // prefetch my next tile's S first so the other WG / S issuer are not blocked by this epilogue
                        {
                            int pn = p + num_clusters; uint32_t gi = g_item + ni; bool found = false; int j0n = 0;
                            while (pn < num_items) {
                                const WorkItem wn = decode_item(pn, B, Hkv, NT, HP);
                                const int nin = wn.i + 1;
                                j0n = ((int)(gi & 1) == wg) ? 0 : 1;
                                if (j0n < nin) { found = true; break; }
                                gi += nin; pn += num_clusters;
                            }
#ifdef NO_EPI_PREFETCH
                            found = false;
#endif
                            if (found) {
                                const uint32_t gn = gi + j0n;
                                mbar_wait_warp(bar(BAR_S_FULL + wg), (gn >> 1) & 1);
                                mbar_arrive_local_pred(bar(BAR_K_FREE + (gn % STAGES)), is_w0l0);
                                tc_fence_after();
                                TMEM_LD_S(s_addr, s);
                                tc_wait_ld();
                                tc_fence_before(); __syncwarp();
                                arrive_leader(bar(BAR_K_FULL + ((gn + 2) % STAGES)));
                                s_loaded = true;
                            }
                        }
                        mbar_wait_warp(bar(BAR_PV_DONE + wg), (n - 1) & 1);      // PV_g done (in-order issuer => all done)
                        tc_fence_after();
                        // 2-CTA accumulator column order: [V d0..63][ones x16][V d64..127][ones x16]
                        uint32_t lsum[1];
                        tmem_ld_x1(o_addr + 64, lsum);
                        tc_wait_ld();
                        const float inv_l = 1.f / __uint_as_float(lsum[0]);
                        // O tile -> smem staging (SW128 rows) -> TMA store, 64 columns at a time
                        const int orow_g = w.b * S + w.i * BM;
                        if (is_w0l0) bulk_wait_read0();   // previous epilogue's TMA store finished reading the staging buffer
                        asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory");
#pragma unroll
                        for (int half = 0; half < 2; half++) {
                            if (half == 1) {
                                if (is_w0l0) bulk_wait_read0();
                                asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            }
#pragma unroll
                            for (int q2 = 0; q2 < 2; q2++) {
                                uint32_t o[32];
                                tmem_ld_x32(o_addr + half * (PV_N / 2) + q2 * 32, o);
                                tc_wait_ld();
                                uint32_t pkk[16];
#pragma unroll
                                for (int c = 0; c < 32; c += 2) pkk[c / 2] = pack_bf16x2(__uint_as_float(o[c]) * inv_l, __uint_as_float(o[c + 1]) * inv_l);
#pragma unroll
                                for (int c = 0; c < 4; c++) {
                                    const int chunk = q2 * 4 + c;                       // 16B chunk index within the 128B row
                                    uint4* dst = (uint4*)(ost_ptr + r * 128 + ((chunk ^ (r & 7)) << 4));
                                    *dst = make_uint4(pkk[4 * c], pkk[4 * c + 1], pkk[4 * c + 2], pkk[4 * c + 3]);
                                }
                            }
                            fence_async_smem();
                            asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory");
                            if (is_w0l0) { tma_store_3d(&tmap_o, ost_smem, half * 64, h, orow_g); bulk_commit(); }
                        }
                        tc_fence_before(); __syncwarp();
                        arrive_leader(bar(BAR_OFREE + (item & 1)));      // (both CTAs) O may be overwritten by the next item's PV_0
                        TACC(t_epi, te);
                    }
                }
            }
            g_item += ni;
        }
#ifdef DEBUG_HANG
        if (lane == 0) printf("WGDONE block %d warp %d g %u n %u\n", blockIdx.x, warp, g_item, n);
#endif
        TACC(t_tot, t_begin);
#ifdef DEBUG_TIMING
        if (dbg && warp == 4 && lane == 0) { long long* d = dbg + blockIdx.x * 32; d[8] = t_ws; d[9] = t_ld; d[10] = t_mc; d[11] = t_comp; d[12] = t_wpv; d[13] = t_epi; d[14] = t_tot; d[15] = n_resc; d[16] = t_st; d[17] = t_pp; }
        if (dbg && warp == 8 && lane == 0) { long long* d = dbg + blockIdx.x * 32; d[20] = t_ws; d[21] = t_ld; d[22] = t_mc; d[23] = t_comp; d[24] = t_wpv; d[25] = t_epi; d[26] = t_tot; d[27] = n_resc; d[28] = t_st; d[29] = t_pp; }
#endif
        }   // softmax body
    }
    bulk_wait0();
    __syncwarp();
    cluster_sync();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(512) : "memory");
    }
}

// ============================ host side ============================
static int make_tmap(CUtensorMap* tm, const void* ptr, int B, int S, int H, int box_rows) {
    cuuint64_t gdim[3] = {(cuuint64_t)HD, (cuuint64_t)H, (cuuint64_t)B * S};
    cuuint64_t gstride[2] = {(cuuint64_t)HD * 2, (cuuint64_t)H * HD * 2};
    cuuint32_t box[3] = {64, 1, (cuuint32_t)box_rows};
    cuuint32_t estride[3] = {1, 1, 1};
    CUresult r = cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)ptr, gdim, gstride, box, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)r); return -1; }
    return 0;
}

extern "C" int fa_fwd_max_grid() {
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(NUM_THREADS);
    cfg.dynamicSmemBytes = SMEM_ALLOC;
    cfg.gridDim = dim3(2);
    cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC);
    int n = 0;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)fa_fwd_kernel, &cfg);
    if (e != cudaSuccess) { fprintf(stderr, "occupancy query failed: %s\n", cudaGetErrorString(e)); return -1; }
    return n * 2;
}

extern "C" int fa_fwd(const void* q, const void* k, const void* v, void* o, int B, int S, int Hq, int Hkv, float softmax_scale, int grid, cudaStream_t stream, void* dbg) {
    if (S % BM != 0 || Hq % Hkv != 0 || ((Hq / Hkv) % 2) != 0) { fprintf(stderr, "unsupported shape\n"); return -1; }
    CUtensorMap tq, tk, tv;
    if (make_tmap(&tq, q, B, S, Hq, BM)) return -1;
    if (make_tmap(&tk, k, B, S, Hkv, BN / 2)) return -1;
    if (make_tmap(&tv, v, B, S, Hkv, BN)) return -1;
    CUtensorMap to_;
    if (make_tmap(&to_, o, B, S, Hq, BM)) return -1;
    cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_ALLOC);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(NUM_THREADS);
    cfg.dynamicSmemBytes = SMEM_ALLOC;
    cfg.stream = stream;
    cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
    const float scale_log2 = softmax_scale * 1.4426950408889634f;
    cudaError_t e = cudaLaunchKernelEx(&cfg, fa_fwd_kernel, tq, tk, tv, to_, (__nv_bfloat16*)o, B, S, Hq, Hkv, scale_log2, (long long*)dbg);
    if (e != cudaSuccess) { fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;
}
