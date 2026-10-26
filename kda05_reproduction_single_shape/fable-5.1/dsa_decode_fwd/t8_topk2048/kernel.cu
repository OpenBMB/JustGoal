// DSA sparse attention (DeepSeek V3.2 style) for B200 / sm_100a.
//   output[t,h,:] = softmax_j( (q_nope[t,h]·ckv[idx_j] + q_pe[t,h]·kpe[idx_j]) * sm_scale ) · ckv[idx_j]
//   lse[t,h]      = log2( sum_j exp(scaled logits) )
// idx_j = sparse_indices[t, j] (flat row of the paged cache, -1 = padding).
//
// v10 design (Phase 3, single launch, 64-slot tiles, every CTA is a tile worker and then a slice combiner):
//  grid = 32 tiles x T (256 for T = 8; 2 CTAs/SM so all are co-resident, checked with the occupancy API,
//  two-kernel PDL fallback otherwise), 256 threads.
//  * tile phase, CTA (token, tile): reads its own 64 indices (256 B), Q by TMA (speculatively, before the
//    validity is known), the 1 KB ckv rows by 1-D TMA bulk copies in two 32-row halves, kpe by cp.async;
//    invalid rows zero-filled. S = Q K^T with mma.sync.m16n8k16 bf16 (M = 16 heads; warp = (K quarter,
//    32-column half)), tile softmax (fp32, base 2; warp w owns 8 columns), O = P V (warp w owns 64 dims).
//    The partial is the l-normalised O in fp16 straight from the MMA C fragments, laid out
//    [token][pair][warp pair][tile][warp&1][n-tile][lane&3] so that one combiner's 32 tiles form one
//    contiguous 16 KB block; (m, l) as 16-B records [token][pair][tile], written by every CTA (l = 0 for
//    an empty tile). Then one red.release.gpu increment of the token's arrival counter.
//  * combiner phase, CTA (token, slice = tile): polls the counter (32 arrivals per call), one TMA fetch of
//    the 16 KB block + 512 B of (m, l) records, masks empty tiles by l == 0, merges with fp32 weights and
//    writes 2 heads x 128 dims of output (bf16); slice (pair, 0) also writes the pair's lse.
//  Block order is tile-major (block b -> tile b / T, token b % T): the busy (low) tiles of all tokens are
//  dispatched first, one per SM.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace dsa {

constexpr int H       = 16;    // query heads (== MMA M)
constexpr int DC      = 512;   // head_dim_ckv
constexpr int DP      = 64;    // head_dim_kpe
constexpr int DQK     = DC + DP;  // 576
constexpr int TOPK    = 2048;
constexpr int TILE    = 64;                       // kv slots per CTA
constexpr int NTILES  = TOPK / TILE;              // 32 tile CTAs per token
constexpr int NTHREADS = 256;
constexpr int NWARPS   = NTHREADS / 32;           // 8
constexpr int CTAS_PER_SM = 2;
constexpr int LDS      = DQK + 8;                 // padded smem row stride (elements): 1168 B = 73 x 16 B
constexpr int HALF     = TILE / 2;                // 32 rows per mbarrier
#ifndef DSA_CPA_ROWS
#define DSA_CPA_ROWS 8
#endif
constexpr int CPA_ROWS = DSA_CPA_ROWS;            // rows per half fetched with cp.async instead of TMA
constexpr int TMA_ROWS = HALF - CPA_ROWS;         // rows per half fetched with 1-D TMA bulk copies
static_assert(CPA_ROWS % 2 == 0 && CPA_ROWS <= HALF, "cp.async rows: 2 x CPA_ROWS x 64 chunks over 256 threads");
constexpr int KSTEPS   = DQK / 16;                // 36
constexpr int KQ       = 4;                       // K quarters in QK (warp & 3)
constexpr int KSTEPS_PER_WARP = KSTEPS / KQ;      // 9  (quarter kq covers dims [144kq, 144kq+144))
constexpr int NT_QK    = HALF / 8;                // 4 n-tiles (32 kv columns) per warp in QK
constexpr int KV_PER_WARP = TILE / NWARPS;        // 8 kv columns per warp in softmax / P
constexpr int DIMS_PER_WARP = DC / NWARPS;        // 64 output dims per warp in PV
constexpr int NT_PV    = DIMS_PER_WARP / 8;       // 8 n-tiles per warp in PV
constexpr int NPAIRS   = H / 2;                   // 8 head pairs (h, h+8) = MMA C rows (r0, r1)
constexpr int NWP      = NWARPS / 2;              // 4 warp pairs = 128-dim slices
constexpr int NSLICES  = NPAIRS * NWP;            // 32 combiner slices per token: (pair, warp pair)
static_assert(NSLICES == NTILES, "one combiner slice per tile CTA");
constexpr int CTR_STRIDE = 32;                    // one 128-B line per (token, slice) counter
constexpr int WSLICE_BYTES = NT_PV * 4 * 8;       // 256 B: one (tile, warp, pair) fp16 piece
constexpr int TSLICE_BYTES = 2 * WSLICE_BYTES;    // 512 B: one tile's piece of a (pair, warp pair) slice
constexpr int SLICE_BLOCK_BYTES = NTILES * TSLICE_BYTES;    // 16 KB: all 32 tiles of one slice
constexpr int ML_FLOATS_PER_TILE = NPAIRS * 4;              // {m_p, m_p+8, l_p, l_p+8} per pair

// shared memory carve-up (bytes)
constexpr int SQ_BYTES   = H * LDS * 2;           // 18688
constexpr int SKV_BYTES  = TILE * LDS * 2;        // 74752
constexpr int OFF_KV   = 0;
constexpr int OFF_Q    = OFF_KV + SKV_BYTES;
constexpr int OFF_IDX  = OFF_Q + SQ_BYTES;        // 64 ints: the tile's indices
constexpr int OFF_BAR  = OFF_IDX + TILE * 4;      // 5 mbarriers (kv half 0, kv half 1, Q rows, combiner O block, combiner (m,l))
constexpr int OFF_MISC = OFF_BAR + 48;            // validity mask words + poll election flag
constexpr int LDP  = TILE + 8;                                 // 72 bf16 per P row (144 B)
constexpr int OFF_P   = OFF_MISC + 64;                         // P (bf16) [16][LDP]: 2304 B
constexpr int OFF_RED = OFF_P + H * LDP * 2;                   // sM[8][16], sL[8][16]: 1 KB
constexpr int SMEM_BYTES = OFF_RED + NWARPS * H * 4 * 2;       // 97,088 -> 2 CTAs per SM
// after QK, the Q region is dead and is reused for the per-warp partial S
constexpr int LDSS = HALF + 4;                                 // 36 floats per S row
constexpr int SS_BYTES = NWARPS * H * LDSS * 4;                // 18432
constexpr int OFF_S   = OFF_Q;
static_assert(SS_BYTES <= SQ_BYTES, "partial S must fit in the Q region");
static_assert((SMEM_BYTES + 1024) * CTAS_PER_SM <= 228 * 1024, "2 CTAs per SM");
// combiner carve-up (inside the KV region, which is dead once the tile phase has published)
constexpr int OFF_C_O    = 0;                              // 16 KB slice block (TMA destination)
constexpr int OFF_C_ML   = OFF_C_O + SLICE_BLOCK_BYTES;    // float4[32] (TMA destination)
constexpr int OFF_C_W    = OFF_C_ML + NTILES * 16;         // float[2][32]
constexpr int OFF_C_RED  = OFF_C_W + 2 * NTILES * 4;       // float[4]
constexpr int OFF_C_ACC  = OFF_C_RED + 32;                 // float[3][64][4]
constexpr int SMEM_C_END = OFF_C_ACC + 3 * 64 * 16;
static_assert(SMEM_C_END <= SKV_BYTES, "combiner scratch fits in the KV region");
constexpr int SMEM_C_BYTES = OFF_MISC + 64;                // two-kernel fallback: combine kernel smem (barriers at OFF_BAR)

#ifdef DSA_TRACE
__device__ unsigned long long g_trace[8 * 64 * 16];
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ unsigned g_smid[8 * 64];
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }
__device__ unsigned long long g_clk[8 * 64 * 16];
__device__ unsigned long long g_clk2[8 * 64 * 16];
#define TRACE(k) do { if (threadIdx.x == 0) { g_trace[(token * 64 + tile) * 16 + (k)] = gtimer(); g_clk[(token * 64 + tile) * 16 + (k)] = clock64(); if ((k) == 0) g_smid[token * 64 + tile] = smid() | (blockIdx.x << 16); } } while (0)
__device__ unsigned long long g_trace2[8 * 64 * 16];
// (stamps 1 and 2 sit right after the combiner's mbarrier wait; a global store there costs ~2.5 us in the
//  trace build, so they are disabled unless DSA_TRACE_ALL is given)
#if defined(DSA_TRACE_ALL)
#define TRACE2(k) do { if (threadIdx.x == 0) { g_trace2[(token * 64 + cid) * 16 + (k)] = gtimer(); g_clk2[(token * 64 + cid) * 16 + (k)] = clock64(); } } while (0)
#else
#define TRACE2(k) do { if ((k) != 1 && (k) != 2 && threadIdx.x == 0) { g_trace2[(token * 64 + cid) * 16 + (k)] = gtimer(); g_clk2[(token * 64 + cid) * 16 + (k)] = clock64(); } } while (0)
#endif
#else
#define TRACE(k) do {} while (0)
#define TRACE2(k) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- cp.async ----
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, bool valid) {
    const int src_bytes = valid ? 16 : 0;  // src_bytes == 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ---- mbarrier + 1-D TMA bulk copies ----
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(bar), "r"(bytes) : "memory");
}
#ifndef DSA_WAIT_MODE
#define DSA_WAIT_MODE 0
#endif
#ifndef DSA_WAIT_HINT_NS
#define DSA_WAIT_HINT_NS 200
#endif
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
#if DSA_WAIT_MODE == 0
    // try_wait without a hint: the thread may be suspended up to a system-dependent limit
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" :: "r"(bar), "r"(parity) : "memory");
#elif DSA_WAIT_MODE == 1
    // busy spin on test_wait
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" :: "r"(bar), "r"(parity) : "memory");
#else
    // try_wait with a short suspend-time hint (ns)
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" :: "r"(bar), "r"(parity), "r"(DSA_WAIT_HINT_NS) : "memory");
#endif
}
__device__ __forceinline__ void tma_bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// ---- arrival counter hand-off ----
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// ---- programmatic dependent launch ----
__device__ __forceinline__ void named_bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// D(16x8 f32) += A(16x16 bf16, row) * B(16x8 bf16, col)
__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float fast_exp2(float x) { return exp2f(x); }
// hardware exp2 (ex2.approx, ~2 ulp): combiner weights only, far below bf16 output precision
__device__ __forceinline__ float ex2_approx(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
    float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

// Q -> row-padded smem: the 16 q_nope rows (1 KB) as TMA bulk copies on `barq` (issued by warp 0, whose
// lane 0 also initialises all four barriers), the 16 q_pe rows (2 KB) as one cp.async chunk per thread
// of warps 0-3 (no commit here)
#if !defined(DSA_Q_WARP7) && !defined(DSA_Q_TMA0)
// whole Q by cp.async from warps 2-7 (6 x 16 B per thread: 16 rows x 72 chunks = 1152 chunks); the TMA
// unit stays free for the kv rows and the index warps 0-1 are not involved at all
__device__ __forceinline__ void load_q(uint32_t sq, uint32_t bar0, int token, const __nv_bfloat16* __restrict__ q_nope,
                                       const __nv_bfloat16* __restrict__ q_pe, int tid) {
    if (tid == 0) {
        mbar_init(bar0, 1); mbar_init(bar0 + 8, 1); mbar_init(bar0 + 16, 1); mbar_init(bar0 + 24, 1); mbar_init(bar0 + 32, 1);
        mbar_fence_init();
        mbar_arrive_expect_tx(bar0 + 16, 0);   // barq completes with no bytes
    }
    if (tid >= 64) {
        const int t = tid - 64;   // 0..191
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int c = t + i * 192;          // chunk 0..1151: row = c / 72, 16-B chunk within the row
            const int row = c / 72, cc = c - row * 72;
            const __nv_bfloat16* src = (cc < 64) ? q_nope + (static_cast<long>(token) * H + row) * DC + cc * 8
                                                 : q_pe + (static_cast<long>(token) * H + row) * DP + (cc - 64) * 8;
            cp_async_16(sq + (row * LDS + cc * 8) * 2, src, true);
        }
    }
}
#elif defined(DSA_Q_WARP7)
// q_nope rows by TMA from the last warp (lane 0 initialises the barriers), q_pe by cp.async from warps 4-7
__device__ __forceinline__ void load_q(uint32_t sq, uint32_t bar0, int token, const __nv_bfloat16* __restrict__ q_nope,
                                       const __nv_bfloat16* __restrict__ q_pe, int tid) {
    const uint32_t barq = bar0 + 16;
    if (tid >= NTHREADS - 32) {
        const int lane = tid & 31;
        if (lane == 0) {
            mbar_init(bar0, 1); mbar_init(bar0 + 8, 1); mbar_init(barq, 1); mbar_init(bar0 + 24, 1); mbar_init(bar0 + 32, 1);
            mbar_fence_init();
            mbar_arrive_expect_tx(barq, H * DC * 2);
        }
        __syncwarp();
        if (lane < H) tma_bulk_g2s(sq + lane * (LDS * 2), q_nope + (static_cast<long>(token) * H + lane) * DC, DC * 2, barq);
    }
    if (tid >= 128) {
        const int t = tid - 128;
        cp_async_16(sq + ((t >> 3) * LDS + DC + (t & 7) * 8) * 2, q_pe + static_cast<long>(token) * H * DP + t * 8, true);
    }
}
#else
__device__ __forceinline__ void load_q(uint32_t sq, uint32_t bar0, int token, const __nv_bfloat16* __restrict__ q_nope,
                                       const __nv_bfloat16* __restrict__ q_pe, int tid) {
    const uint32_t barq = bar0 + 16;
    if (tid < 32) {
        if (tid == 0) {
            mbar_init(bar0, 1); mbar_init(bar0 + 8, 1); mbar_init(barq, 1); mbar_init(bar0 + 24, 1); mbar_init(bar0 + 32, 1);
            mbar_fence_init();
            mbar_arrive_expect_tx(barq, H * DC * 2);
        }
        __syncwarp();
        if (tid < H) tma_bulk_g2s(sq + tid * (LDS * 2), q_nope + (static_cast<long>(token) * H + tid) * DC, DC * 2, barq);
    }
    if (tid < (H * DP) / 8)
        cp_async_16(sq + ((tid >> 3) * LDS + DC + (tid & 7) * 8) * 2, q_pe + static_cast<long>(token) * H * DP + tid * 8, true);
}
#endif

// ------------------------------------------------------------------------------------------------
// Tile phase: one 64-slot tile (token, tile). Returns after the partial (or the empty record) has been
// published and the arrival counter incremented (FUSED) / the kernel's own completion is the hand-off.
// ------------------------------------------------------------------------------------------------
template <bool FUSED>
__device__ __forceinline__ void tile_phase(unsigned char* smem, int token, int tile,
                const __nv_bfloat16* __restrict__ q_nope,   // [T,16,512]
                const __nv_bfloat16* __restrict__ q_pe,     // [T,16,64]
                const __nv_bfloat16* __restrict__ ckv,      // [P*64,512]
                const __nv_bfloat16* __restrict__ kpe,      // [P*64,64]
                const int* __restrict__ sparse_indices,     // [T,2048]
                float* __restrict__ ws_ml,                  // [T][8 pairs][32 tiles][4]
                __half* __restrict__ ws_o,                  // [T][8][4][32 tiles][2][8][4] x 8 B
                unsigned* __restrict__ counters,            // [T][NSLICES][CTR_STRIDE] (fused path)
                long total_rows, float scale_log2) {
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const uint32_t skv = smem_u32(smem + OFF_KV);
    const uint32_t sq  = smem_u32(smem + OFF_Q);
    int* sIdx = reinterpret_cast<int*>(smem + OFF_IDX);
    unsigned* s_misc = reinterpret_cast<unsigned*>(smem + OFF_MISC);
    const uint32_t bar0 = smem_u32(smem + OFF_BAR);
    const uint32_t bar1 = bar0 + 8;
    const uint32_t barq = bar0 + 16;
    unsigned* ctr = counters + static_cast<long>(token) * NSLICES * CTR_STRIDE;   // + slice * CTR_STRIDE
    TRACE(0);
    // ---- the tile's own 64 indices (warps 0-1, one per lane) and, speculatively, Q ----------------
    int my_idx = -1;
    if (warp < 2) {
        const int v = __ldg(sparse_indices + static_cast<long>(token) * TOPK + tile * TILE + tid);
        my_idx = ((v >= 0) && (static_cast<long>(v) < total_rows)) ? v : -1;
    }
    load_q(sq, bar0, token, q_nope, q_pe, tid);
    cp_async_commit();
    const bool my_valid = my_idx >= 0;
    if (warp < 2) {
        const unsigned vm = __ballot_sync(0xffffffffu, my_valid);
        sIdx[tid] = my_idx;
        if (lane == 0) s_misc[warp] = vm;
    }
    if (tid == 64) s_misc[2] = 0u;   // combiner poll election flag
    __syncthreads();
    const unsigned vmask_lo = s_misc[0], vmask_hi = s_misc[1];
    const bool tile_valid = (vmask_lo | vmask_hi) != 0u;
    TRACE(1);
    if (!FUSED) pdl_launch_dependents();
    if (!tile_valid) {
        // empty tile: publish l = 0 records for the 8 pairs, acknowledge
        if (tid < NPAIRS)
            reinterpret_cast<float4*>(ws_ml)[(static_cast<long>(token) * NPAIRS + tid) * NTILES + tile] =
                make_float4(-INFINITY, -INFINITY, 0.f, 0.f);
        if (FUSED) {
            __syncthreads();
            if (warp == 0) red_release_add(ctr + lane * CTR_STRIDE, 1u);   // one arrival per slice counter
        }
        cp_async_wait<0>();
        mbar_wait(barq, 0);   // the speculative Q copies must land before the smem is reused / the CTA exits
        return;
    }
    // ---- gather: the 1 KB ckv rows go through the TMA engine (one bulk copy per row; warp w < 2 issues
    //      rows 32w + lane on barrier w); the 128 B kpe rows go through cp.async; both run concurrently
#ifndef DSA_TMA_WARPS
#define DSA_TMA_WARPS 2
#endif
    if (warp < 2) {
        const unsigned vm = (warp == 0 ? vmask_lo : vmask_hi) & ((1u << TMA_ROWS) - 1u);   // TMA rows: the first TMA_ROWS of the half
        if (lane == 0) mbar_arrive_expect_tx(warp == 0 ? bar0 : bar1, static_cast<uint32_t>(__popc(vm)) * (DC * 2));
    }
    if (DSA_TMA_WARPS == 2) {
        if (warp < 2) {
            __syncwarp();
            if (my_valid && lane < TMA_ROWS)
                tma_bulk_g2s(skv + tid * (LDS * 2), ckv + static_cast<long>(my_idx) * DC, DC * 2, warp == 0 ? bar0 : bar1);
        }
    } else if (warp < DSA_TMA_WARPS) {
        // row k = warp + DSA_TMA_WARPS * lane (interleaved so every issuing warp has rows of both halves)
        const int k = warp + DSA_TMA_WARPS * lane;
        if (k < TILE && (k & (HALF - 1)) < TMA_ROWS) {
            const int idx = sIdx[k];
            if (idx >= 0)
                tma_bulk_g2s(skv + k * (LDS * 2), ckv + static_cast<long>(idx) * DC, DC * 2, (k < HALF) ? bar0 : bar1);
        }
    }
    {
        // kpe: 64 rows x 8 chunks of 16 B (2 per thread), zero-filled for invalid rows
#pragma unroll
        for (int i = 0; i < (TILE * (DP / 8)) / NTHREADS; ++i) {
            const int c = tid + i * NTHREADS;
            const int row = c >> 3, col = c & 7;
            const int idx = sIdx[row];
            const bool valid = idx >= 0;
            cp_async_16(skv + (row * LDS + DC + col * 8) * 2, kpe + (valid ? static_cast<long>(idx) : 0L) * DP + col * 8, valid);
        }
        cp_async_commit();
    }
    if (CPA_ROWS > 0) {
#ifndef DSA_QK_HALVES
        // the last CPA_ROWS ckv rows of half nh = warp / 4 by cp.async from the 4 warps that consume that half
        const int nh = warp >> 2, t = tid & 127;
#pragma unroll
        for (int i = 0; i < CPA_ROWS / 2; ++i) {
            const int c = t + i * 128;
            const int row = nh * HALF + TMA_ROWS + (c >> 6), col = c & 63;
#else
        // the last CPA_ROWS ckv rows of each half by cp.async: 2 x CPA_ROWS x 64 chunks of 16 B over 256 threads,
        // zero-filled for invalid rows (own commit group, waited before the QK of half 0)
#pragma unroll
        for (int i = 0; i < CPA_ROWS / 2; ++i) {
            const int c = tid + i * NTHREADS;
            const int row = (c >> 6) / CPA_ROWS * HALF + TMA_ROWS + (c >> 6) % CPA_ROWS, col = c & 63;
#endif
            const int idx = sIdx[row];
            const bool valid = idx >= 0;
            cp_async_16(skv + (row * LDS + col * 8) * 2, ckv + (valid ? static_cast<long>(idx) : 0L) * DC + col * 8, valid);
        }
        cp_async_commit();
    }
    // zero-fill invalid TMA rows (warp w: rows 8w..8w+7) so P.V never touches garbage; disjoint from the
    // rows the TMA writes (cp.async rows are zero-filled by the copy itself)
    {
        const unsigned vm8 = ((warp < 4 ? vmask_lo : vmask_hi) >> ((warp & 3) * 8)) & 0xffu;
        if (vm8 != 0xffu) {
            for (int r = 0; r < 8; ++r) {
                const int row = warp * 8 + r;
                if (!((vm8 >> r) & 1u) && ((row & (HALF - 1)) < TMA_ROWS)) {
                    unsigned char* dst = smem + OFF_KV + row * (LDS * 2);
                    for (int c = lane; c < (DC * 2) / 16; c += 32)
                        *reinterpret_cast<int4*>(dst + c * 16) = make_int4(0, 0, 0, 0);
                }
            }
        }
    }
    // validity of the 2 S columns this thread masks later (columns 8*warp + 2*(lane&3) + {0,1})
    bool col_valid[2];
    {
        const int kv0 = warp * KV_PER_WARP + (lane & 3) * 2;
        const unsigned vm = (kv0 < 32) ? vmask_lo : vmask_hi;
        col_valid[0] = (vm >> (kv0 & 31)) & 1u;
        col_valid[1] = (vm >> ((kv0 & 31) + 1)) & 1u;
    }
    if (CPA_ROWS > 0) cp_async_wait<1>(); else cp_async_wait<0>();   // Q + kpe landed (the ckv row group may still be in flight)
    mbar_wait(barq, 0);
    __syncthreads();   // Q and kpe landed, zero rows written, before anybody waits on the halves
    TRACE(2);

    float s[4];   // (r0,c0) (r0,c1) (r1,c0) (r1,c1) of this thread's 2 columns, scaled, masked
#ifdef DSA_QK_FULLK
    {
        // warp w: columns [8w, 8w+8) (one n-tile) over all 36 k-steps, two interleaved accumulator chains
        float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t a_base = sq + (((lane & 15) * LDS) + (lane >> 4) * 8) * 2;
        // B: x4 covers k-steps ks (matrices 0,1: rows 8w..8w+7, k halves) and ks+1 (matrices 2,3)
        const uint32_t b_base = skv + (((warp * 8 + (lane & 7)) * LDS) + ((lane >> 3) << 3)) * 2;
        const int nh = (warp * 8) / HALF;   // the kv half holding this warp's rows
        mbar_wait(nh == 0 ? bar0 : bar1, 0);
        TRACE(3);
#pragma unroll 6
        for (int ks = 0; ks < KSTEPS; ks += 2) {
            uint32_t a0[4], a1[4], b[4];
            ldmatrix_x4(a0[0], a0[1], a0[2], a0[3], a_base + ks * 32);
            ldmatrix_x4(a1[0], a1[1], a1[2], a1[3], a_base + ks * 32 + 32);
            ldmatrix_x4(b[0], b[1], b[2], b[3], b_base + ks * 32);
            mma_bf16_16816(acc0, a0, b[0], b[1]);
            mma_bf16_16816(acc1, a1, b[2], b[3]);
        }
        TRACE(4);
        s[0] = col_valid[0] ? (acc0[0] + acc1[0]) * scale_log2 : -INFINITY;
        s[1] = col_valid[1] ? (acc0[1] + acc1[1]) * scale_log2 : -INFINITY;
        s[2] = col_valid[0] ? (acc0[2] + acc1[2]) * scale_log2 : -INFINITY;
        s[3] = col_valid[1] ? (acc0[3] + acc1[3]) * scale_log2 : -INFINITY;
    }
    __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(smem + OFF_P);   // (own region: no barrier needed here)
    float* sM = reinterpret_cast<float*>(smem + OFF_RED);   // [NWARPS][16]
    float* sL = sM + NWARPS * H;                              // [NWARPS][16]
#else
#ifndef DSA_QK_HALVES
    // ---- S = Q K^T: warp = (K quarter kq = warp & 3 -> dims [144kq, +144), kv half nh = warp >> 2 ->
    //      columns [32nh, +32)); 4 n-tiles x 9 k-steps per warp
    const int kq = warp & 3, nh = warp >> 2;
    float pacc[NT_QK][4];
#pragma unroll
    for (int i = 0; i < NT_QK; ++i) { pacc[i][0] = pacc[i][1] = pacc[i][2] = pacc[i][3] = 0.f; }
    {
        const uint32_t a_base = sq + (((lane & 15) * LDS) + (lane >> 4) * 8 + kq * KSTEPS_PER_WARP * 16) * 2;
        const uint32_t b_base = skv + (((nh * HALF + (lane & 7) + ((lane >> 4) << 3)) * LDS)
                                       + (((lane >> 3) & 1) << 3) + kq * KSTEPS_PER_WARP * 16) * 2;
#ifndef DSA_NO_QREG
        // this warp's K-quarter of Q as A fragments in registers, loaded while the kv half is still landing
        uint32_t aq[KSTEPS_PER_WARP][4];
#pragma unroll
        for (int ks = 0; ks < KSTEPS_PER_WARP; ++ks) ldmatrix_x4(aq[ks][0], aq[ks][1], aq[ks][2], aq[ks][3], a_base + ks * 32);
#endif
        if (CPA_ROWS > 0) {
            cp_async_wait<0>();                 // this half's cp.async rows (issued by its 4 warps)
            named_bar_sync(1 + nh, 128);        // ... visible to all 4 warps of the half
        }
        mbar_wait(nh == 0 ? bar0 : bar1, 0);
        TRACE(3);
#ifndef DSA_NO_QREG
        uint32_t b[2][NT_QK / 2][4];
#pragma unroll
        for (int np = 0; np < NT_QK / 2; ++np)
            ldmatrix_x4(b[0][np][0], b[0][np][1], b[0][np][2], b[0][np][3], b_base + (np * 16 * LDS) * 2);
#pragma unroll
        for (int ks = 0; ks < KSTEPS_PER_WARP; ++ks) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks + 1 < KSTEPS_PER_WARP) {
#pragma unroll
                for (int np = 0; np < NT_QK / 2; ++np)
                    ldmatrix_x4(b[nxt][np][0], b[nxt][np][1], b[nxt][np][2], b[nxt][np][3], b_base + (np * 16 * LDS) * 2 + (ks + 1) * 32);
            }
#pragma unroll
            for (int np = 0; np < NT_QK / 2; ++np) {
                mma_bf16_16816(pacc[2 * np], aq[ks], b[cur][np][0], b[cur][np][1]);
                mma_bf16_16816(pacc[2 * np + 1], aq[ks], b[cur][np][2], b[cur][np][3]);
            }
        }
#else
        uint32_t a[2][4], b[2][NT_QK / 2][4];
        ldmatrix_x4(a[0][0], a[0][1], a[0][2], a[0][3], a_base);
#pragma unroll
        for (int np = 0; np < NT_QK / 2; ++np)
            ldmatrix_x4(b[0][np][0], b[0][np][1], b[0][np][2], b[0][np][3], b_base + (np * 16 * LDS) * 2);
#pragma unroll
        for (int ks = 0; ks < KSTEPS_PER_WARP; ++ks) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks + 1 < KSTEPS_PER_WARP) {
                ldmatrix_x4(a[nxt][0], a[nxt][1], a[nxt][2], a[nxt][3], a_base + (ks + 1) * 32);
#pragma unroll
                for (int np = 0; np < NT_QK / 2; ++np)
                    ldmatrix_x4(b[nxt][np][0], b[nxt][np][1], b[nxt][np][2], b[nxt][np][3], b_base + (np * 16 * LDS) * 2 + (ks + 1) * 32);
            }
#pragma unroll
            for (int np = 0; np < NT_QK / 2; ++np) {
                mma_bf16_16816(pacc[2 * np], a[cur], b[cur][np][0], b[cur][np][1]);
                mma_bf16_16816(pacc[2 * np + 1], a[cur], b[cur][np][2], b[cur][np][3]);
            }
        }
#endif
        TRACE(4);
    }
    __syncthreads();  // every warp is done reading sQ before the partial S overwrites it
    float* sS = reinterpret_cast<float*>(smem + OFF_S);
    __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(smem + OFF_P);
    float* sM = reinterpret_cast<float*>(smem + OFF_RED);   // [NWARPS][16]
    float* sL = sM + NWARPS * H;                              // [NWARPS][16]
    {
        float* mine = sS + warp * (H * LDSS);
#pragma unroll
        for (int nt = 0; nt < NT_QK; ++nt) {
            const int col = nt * 8 + (lane & 3) * 2;
            *reinterpret_cast<float2*>(mine + (lane >> 2) * LDSS + col) = make_float2(pacc[nt][0], pacc[nt][1]);
            *reinterpret_cast<float2*>(mine + ((lane >> 2) + 8) * LDSS + col) = make_float2(pacc[nt][2], pacc[nt][3]);
        }
    }
    __syncthreads();
    // warp w owns kv columns [8w, 8w+8) (half w / 4, local columns 8 (w % 4) + ...): sum the 4 K-quarter partials
    {
        const int col = (warp & 3) * KV_PER_WARP + (lane & 3) * 2;
        const float* src = sS + (warp >> 2) * (KQ * H * LDSS);
        float2 r0 = make_float2(0.f, 0.f), r1 = make_float2(0.f, 0.f);
#pragma unroll
        for (int k = 0; k < KQ; ++k) {
            const float2 x0 = *reinterpret_cast<const float2*>(src + k * (H * LDSS) + (lane >> 2) * LDSS + col);
            const float2 x1 = *reinterpret_cast<const float2*>(src + k * (H * LDSS) + ((lane >> 2) + 8) * LDSS + col);
            r0.x += x0.x; r0.y += x0.y; r1.x += x1.x; r1.y += x1.y;
        }
        s[0] = col_valid[0] ? r0.x * scale_log2 : -INFINITY;
        s[1] = col_valid[1] ? r0.y * scale_log2 : -INFINITY;
        s[2] = col_valid[0] ? r1.x * scale_log2 : -INFINITY;
        s[3] = col_valid[1] ? r1.y * scale_log2 : -INFINITY;
    }
#else
    // ---- S = Q K^T: warp = (K quarter kq = warp & 3 -> dims [144kq, +144), n-tile pair np = warp >> 2 ->
    //      columns [16np, +16) of each half); both halves in turn, so every warp computes while half 1 lands
    const int kq = warp & 3, np2 = warp >> 2;
    float pacc[2][2][4];   // [half][n-tile of the pair][frag]
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 2; ++i) { pacc[h][i][0] = pacc[h][i][1] = pacc[h][i][2] = pacc[h][i][3] = 0.f; }
    {
        const uint32_t a_base = sq + (((lane & 15) * LDS) + (lane >> 4) * 8 + kq * KSTEPS_PER_WARP * 16) * 2;
        const uint32_t b_base = skv + (((np2 * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDS)
                                       + (((lane >> 3) & 1) << 3) + kq * KSTEPS_PER_WARP * 16) * 2;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            if (CPA_ROWS > 0 && hh == 0) {
                cp_async_wait<0>();                 // both halves' cp.async rows (issued by this half's 4 warps ...)
                __syncthreads();                    // ... visible to everybody
            }
            mbar_wait(hh == 0 ? bar0 : bar1, 0);
            if (hh == 0) TRACE(3);
            const uint32_t bh = b_base + (hh * HALF * LDS) * 2;
            uint32_t a[2][4], b[2][4];
            ldmatrix_x4(a[0][0], a[0][1], a[0][2], a[0][3], a_base);
            ldmatrix_x4(b[0][0], b[0][1], b[0][2], b[0][3], bh);
#pragma unroll
            for (int ks = 0; ks < KSTEPS_PER_WARP; ++ks) {
                const int cur = ks & 1, nxt = cur ^ 1;
                if (ks + 1 < KSTEPS_PER_WARP) {
                    ldmatrix_x4(a[nxt][0], a[nxt][1], a[nxt][2], a[nxt][3], a_base + (ks + 1) * 32);
                    ldmatrix_x4(b[nxt][0], b[nxt][1], b[nxt][2], b[nxt][3], bh + (ks + 1) * 32);
                }
                mma_bf16_16816(pacc[hh][0], a[cur], b[cur][0], b[cur][1]);
                mma_bf16_16816(pacc[hh][1], a[cur], b[cur][2], b[cur][3]);
            }
        }
        TRACE(4);   // (stamps: 3 = half 0 landed, 4 = this warp's QK done)
    }
    __syncthreads();  // every warp is done reading sQ before the partial S overwrites it
    float* sS = reinterpret_cast<float*>(smem + OFF_S);
    __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(smem + OFF_P);
    float* sM = reinterpret_cast<float*>(smem + OFF_RED);   // [NWARPS][16]
    float* sL = sM + NWARPS * H;                              // [NWARPS][16]
    {
        // partial S of warp (kq, np2): [row][hh * 16 + ntp * 8 + col]
        float* mine = sS + warp * (H * LDSS);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int ntp = 0; ntp < 2; ++ntp) {
                const int col = hh * 16 + ntp * 8 + (lane & 3) * 2;
                *reinterpret_cast<float2*>(mine + (lane >> 2) * LDSS + col) = make_float2(pacc[hh][ntp][0], pacc[hh][ntp][1]);
                *reinterpret_cast<float2*>(mine + ((lane >> 2) + 8) * LDSS + col) = make_float2(pacc[hh][ntp][2], pacc[hh][ntp][3]);
            }
    }
    __syncthreads();
    // warp w now owns kv columns [8w, 8w+8): half hh = w / 4, n-tile nt = w % 4 -> pair np = nt / 2 (warps
    // kq + 4 np hold the K-quarter partials), ntp = nt % 2; sum the 4 partials -> s[4] = (r0,c0) (r0,c1) (r1,c0) (r1,c1)
    {
        const int col = (warp >> 2) * 16 + ((warp & 3) & 1) * 8 + (lane & 3) * 2;
        const float* src = sS + (((warp & 3) >> 1) * 4) * (H * LDSS);
        float2 r0 = make_float2(0.f, 0.f), r1 = make_float2(0.f, 0.f);
#pragma unroll
        for (int k = 0; k < KQ; ++k) {
            const float2 x0 = *reinterpret_cast<const float2*>(src + k * (H * LDSS) + (lane >> 2) * LDSS + col);
            const float2 x1 = *reinterpret_cast<const float2*>(src + k * (H * LDSS) + ((lane >> 2) + 8) * LDSS + col);
            r0.x += x0.x; r0.y += x0.y; r1.x += x1.x; r1.y += x1.y;
        }
        s[0] = col_valid[0] ? r0.x * scale_log2 : -INFINITY;
        s[1] = col_valid[1] ? r0.y * scale_log2 : -INFINITY;
        s[2] = col_valid[0] ? r1.x * scale_log2 : -INFINITY;
        s[3] = col_valid[1] ? r1.y * scale_log2 : -INFINITY;
    }
#endif
#endif
    // ---- tile softmax (base 2): rows r0 = lane/4, r1 = lane/4 + 8 ------------------------------
    float tmax[2] = { fmaxf(s[0], s[1]), fmaxf(s[2], s[3]) };
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tmax[i] = fmaxf(tmax[i], __shfl_xor_sync(0xffffffffu, tmax[i], 1));
        tmax[i] = fmaxf(tmax[i], __shfl_xor_sync(0xffffffffu, tmax[i], 2));
    }
    if ((lane & 3) == 0) {
        sM[warp * H + (lane >> 2)] = tmax[0];
        sM[warp * H + (lane >> 2) + 8] = tmax[1];
    }
    __syncthreads();
    float m_r[2], msub[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = (lane >> 2) + i * 8;
        float m = sM[r];
#pragma unroll
        for (int w = 1; w < NWARPS; ++w) m = fmaxf(m, sM[w * H + r]);
        m_r[i] = m;
        msub[i] = (m == -INFINITY) ? 0.f : m;   // (cannot happen for a valid tile; keeps P finite)
    }
    const float p0 = fast_exp2(s[0] - msub[0]);
    const float p1 = fast_exp2(s[1] - msub[0]);
    const float p2 = fast_exp2(s[2] - msub[1]);
    const float p3 = fast_exp2(s[3] - msub[1]);
    float psum[2] = { p0 + p1, p2 + p3 };
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        psum[i] += __shfl_xor_sync(0xffffffffu, psum[i], 1);
        psum[i] += __shfl_xor_sync(0xffffffffu, psum[i], 2);
    }
    {
        const int col = warp * KV_PER_WARP + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(sPb + (lane >> 2) * LDP + col) = pack_bf16x2(p0, p1);
        *reinterpret_cast<uint32_t*>(sPb + ((lane >> 2) + 8) * LDP + col) = pack_bf16x2(p2, p3);
    }
    if ((lane & 3) == 0) {
        sL[warp * H + (lane >> 2)] = psum[0];
        sL[warp * H + (lane >> 2) + 8] = psum[1];
    }
    __syncthreads();  // sP, sL complete
    TRACE(5);
    float l_r[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = (lane >> 2) + i * 8;
        float l = sL[r];
#pragma unroll
        for (int w = 1; w < NWARPS; ++w) l += sL[w * H + r];
        l_r[i] = l;
    }
    // ---- O = P V for dims [64*warp, 64*warp+64) -----------------------------------------------
    float oacc[NT_PV][4];
#pragma unroll
    for (int i = 0; i < NT_PV; ++i) { oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f; }
    {
        const uint32_t pa_base = smem_u32(sPb) + ((lane & 15) * LDP + (lane >> 4) * 8) * 2;
        const uint32_t vb_base = skv + ((((lane & 7) + (((lane >> 3) & 1) << 3)) * LDS)
                                        + warp * DIMS_PER_WARP + (lane >> 4) * 8) * 2;
#ifdef DSA_NO_PV_PIPE
#pragma unroll
        for (int kk = 0; kk < TILE / 16; ++kk) {
            uint32_t a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3], pa_base + kk * 32);
#pragma unroll
            for (int np = 0; np < NT_PV / 2; ++np) {
                uint32_t b[4];
                ldmatrix_x4_trans(b[0], b[1], b[2], b[3], vb_base + (kk * 16 * LDS + np * 16) * 2);
                mma_bf16_16816(oacc[2 * np], a, b[0], b[1]);
                mma_bf16_16816(oacc[2 * np + 1], a, b[2], b[3]);
            }
        }
#else
        uint32_t a[2][4], b[2][NT_PV / 2][4];
        ldmatrix_x4(a[0][0], a[0][1], a[0][2], a[0][3], pa_base);
#pragma unroll
        for (int np = 0; np < NT_PV / 2; ++np)
            ldmatrix_x4_trans(b[0][np][0], b[0][np][1], b[0][np][2], b[0][np][3], vb_base + (np * 16) * 2);
#pragma unroll
        for (int kk = 0; kk < TILE / 16; ++kk) {
            const int cur = kk & 1, nxt = cur ^ 1;
            if (kk + 1 < TILE / 16) {
                ldmatrix_x4(a[nxt][0], a[nxt][1], a[nxt][2], a[nxt][3], pa_base + (kk + 1) * 32);
#pragma unroll
                for (int np = 0; np < NT_PV / 2; ++np)
                    ldmatrix_x4_trans(b[nxt][np][0], b[nxt][np][1], b[nxt][np][2], b[nxt][np][3], vb_base + ((kk + 1) * 16 * LDS + np * 16) * 2);
            }
#pragma unroll
            for (int np = 0; np < NT_PV / 2; ++np) {
                mma_bf16_16816(oacc[2 * np], a[cur], b[cur][np][0], b[cur][np][1]);
                mma_bf16_16816(oacc[2 * np + 1], a[cur], b[cur][np][2], b[cur][np][3]);
            }
        }
#endif
    }
    TRACE(6);
    // ---- publish: (m, l) records, l-normalised O in fp16 straight from the C fragments ---------
    {
        const int p = lane >> 2, q = lane & 3;
        if (warp == 0 && q == 0)
            reinterpret_cast<float4*>(ws_ml)[(static_cast<long>(token) * NPAIRS + p) * NTILES + tile] =
                make_float4(m_r[0], m_r[1], l_r[0], l_r[1]);
        const float inv0 = rcp_approx(l_r[0]), inv1 = rcp_approx(l_r[1]);   // l >= 1 for a valid tile (max element = 1)
#ifndef DSA_PUB8
        // unit (nt, q) of this warp -> block (token, p, wp = warp / 2), slot = tile, piece (warp & 1), 16-B unit
        // index (nt / 2) * 4 + q holding n-tiles nt, nt + 1: {h0: d, d+1 ; h1: d, d+1 ; h0: d+8, d+9 ; h1: d+8, d+9}
        uint4* po = reinterpret_cast<uint4*>(ws_o) +
                    (((((static_cast<long>(token) * NPAIRS + p) * NWP + (warp >> 1)) * NTILES + tile) * 2 + (warp & 1)) * (NT_PV / 2)) * 4 + q;
#pragma unroll
        for (int np = 0; np < NT_PV / 2; ++np) {
            uint4 o;
            o.x = pack_f16x2(oacc[2 * np][0] * inv0, oacc[2 * np][1] * inv0);
            o.y = pack_f16x2(oacc[2 * np][2] * inv1, oacc[2 * np][3] * inv1);
            o.z = pack_f16x2(oacc[2 * np + 1][0] * inv0, oacc[2 * np + 1][1] * inv0);
            o.w = pack_f16x2(oacc[2 * np + 1][2] * inv1, oacc[2 * np + 1][3] * inv1);
            po[np * 4] = o;
        }
#else
        // unit (nt, q) of this warp -> block (token, p, wp = warp / 2), slot = tile, (warp & 1, nt, q)
        uint2* po = reinterpret_cast<uint2*>(ws_o) +
                    (((((static_cast<long>(token) * NPAIRS + p) * NWP + (warp >> 1)) * NTILES + tile) * 2 + (warp & 1)) * NT_PV) * 4 + q;
#pragma unroll
        for (int nt = 0; nt < NT_PV; ++nt) {
            uint2 o;
            o.x = pack_f16x2(oacc[nt][0] * inv0, oacc[nt][1] * inv0);
            o.y = pack_f16x2(oacc[nt][2] * inv1, oacc[nt][3] * inv1);
            po[nt * 4] = o;
        }
#endif
    }
    TRACE(8);
    if (FUSED) {
        __syncthreads();   // all partial stores issued by every thread before the (cumulative) releases
        TRACE(9);
        if (warp == 0) red_release_add(ctr + lane * CTR_STRIDE, 1u);   // one arrival per slice counter
    }
    TRACE(7);
}

// ------------------------------------------------------------------------------------------------
// Combiner phase: slice cid = (pair p, warp pair wp) of one token; 256 threads.
// Everything the tile phases wrote must be visible on entry (poll acquired by thread 0 / griddepcontrol.wait).
// ------------------------------------------------------------------------------------------------
#ifndef DSA_POLL_SLEEP
#define DSA_POLL_SLEEP 100
#endif
#ifndef DSA_POLL_STAGGER
#define DSA_POLL_STAGGER 60
#endif
__device__ __forceinline__ void slice_combine(unsigned char* smem, uint32_t barc, int token, int cid,
                   const unsigned* poll, unsigned target,   // fused path: thread 0 spins until *poll >= target
                   const float* __restrict__ ws_ml, const __half* __restrict__ ws_o,
                   __nv_bfloat16* __restrict__ out, float* __restrict__ lse) {
    const int p = cid / NWP;               // head pair: heads p and p+8
    const int wp = cid % NWP;              // warp pair: dims [128wp, 128wp+128)
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int u = tid & 63;                // unit within a tile piece
    const int g = tid >> 6;                // reduction group: tiles [8g, 8g+8)
    const float4* s_ml    = reinterpret_cast<const float4*>(smem + OFF_C_ML);           // [32] (m0, m1, l0, l1)
    float (*s_W)[NTILES]  = reinterpret_cast<float (*)[NTILES]>(smem + OFF_C_W);       // [2][32]: exp2(m - M) * l
    float* s_red          = reinterpret_cast<float*>(smem + OFF_C_RED);                // M0, M1, L0, L1
    float (*s_acc)[64][4] = reinterpret_cast<float (*)[64][4]>(smem + OFF_C_ACC);      // [3][64][4] group partial sums
    const uint32_t sbase = smem_u32(smem);
    const uint32_t barml = barc + 8;
    auto issue = [&]() {
#ifndef DSA_COMBINE_TMA
        return;
#endif
        fence_proxy_async_global();
#ifdef DSA_ONE_BAR
        mbar_arrive_expect_tx(barc, SLICE_BLOCK_BYTES + NTILES * 16);
        tma_bulk_g2s(sbase + OFF_C_ML, ws_ml + (static_cast<long>(token) * NPAIRS + p) * NTILES * 4, NTILES * 16, barc);
#else
        mbar_arrive_expect_tx(barml, NTILES * 16);
        tma_bulk_g2s(sbase + OFF_C_ML, ws_ml + (static_cast<long>(token) * NPAIRS + p) * NTILES * 4, NTILES * 16, barml);
        mbar_arrive_expect_tx(barc, SLICE_BLOCK_BYTES);
#endif
        const unsigned char* ob = reinterpret_cast<const unsigned char*>(ws_o) +
            ((static_cast<long>(token) * NPAIRS + p) * NWP + wp) * SLICE_BLOCK_BYTES;
        tma_bulk_g2s(sbase + OFF_C_O, ob, SLICE_BLOCK_BYTES, barc);
    };
    if (poll == nullptr) {
        if (tid == 0) issue();
    } else {
#ifndef DSA_POLL_MULTI
        // all of warp 0 polls (one load per iteration, no divergent spinning inside the warp), lane 0 issues
        if (warp == 0) {
            while (ld_acquire_u32(poll) < target) {
                if (DSA_POLL_SLEEP > 0) __nanosleep(DSA_POLL_SLEEP);
            }
            __syncwarp();
            if (lane == 0) { TRACE2(4); issue(); }
            __syncwarp();
        }
#else
        // every warp polls as a whole (one load per warp per iteration, no intra-warp divergence), staggered in
        // time so the arrival is noticed within a fraction of one load round trip; the first warp to see it
        // wins the smem election and its lane 0 issues the fetch; the others fall through to the mbarrier
        volatile unsigned* s_flag = reinterpret_cast<volatile unsigned*>(smem + OFF_MISC + 8);
        __nanosleep(warp * DSA_POLL_STAGGER);
        for (;;) {
            if (*s_flag != 0u) break;
            const unsigned v = ld_acquire_u32(poll);
            if (v >= target) {
                unsigned won = 0u;
                if (lane == 0) won = (atomicCAS(reinterpret_cast<unsigned*>(smem + OFF_MISC + 8), 0u, 1u) == 0u) ? 1u : 0u;
                won = __shfl_sync(0xffffffffu, won, 0);
                if (won) { if (lane == 0) { TRACE2(4); issue(); } __syncwarp(); }
                break;
            }
            if (DSA_POLL_SLEEP > 0) __nanosleep(DSA_POLL_SLEEP);
        }
#endif
    }
#ifndef DSA_COMBINE_TMA
    // ---- direct loads: every thread fetches its 8 tile pieces into registers right after the poll (ordered
    //      after thread 0's acquire by the barrier), overlapping the (m, l) load and the weights below ----
    __syncthreads();
    const unsigned char* ob = reinterpret_cast<const unsigned char*>(ws_o) +
        ((static_cast<long>(token) * NPAIRS + p) * NWP + wp) * SLICE_BLOCK_BYTES;
    const float* ml0 = ws_ml + (static_cast<long>(token) * NPAIRS + p) * NTILES * 4;
    uint2 v[NTILES / 4];
#pragma unroll
    for (int jj = 0; jj < NTILES / 4; ++jj)
        v[jj] = __ldcg(reinterpret_cast<const uint2*>(ob + static_cast<long>(g * (NTILES / 4) + jj) * TSLICE_BYTES) + u);
    float4 mlv = make_float4(-INFINITY, -INFINITY, 0.f, 0.f);
#ifdef DSA_STATS_ALL
    mlv = __ldcg(reinterpret_cast<const float4*>(ml0) + lane);
#else
    if (warp == 0) mlv = __ldcg(reinterpret_cast<const float4*>(ml0) + lane);
#endif
#else
#ifdef DSA_ONE_BAR
    mbar_wait(barc, 0);
#else
    mbar_wait(barml, 0);
#endif
#endif
    TRACE2(1);
    // ---- stats (warp 0, tile j <-> lane j; or every warp redundantly with DSA_STATS_ALL) ----------
#if defined(DSA_STATS_ALL) && !defined(DSA_COMBINE_TMA)
    float M0, M1, L0, L1, Wj0, Wj1;
    {
        M0 = (mlv.z > 0.f) ? mlv.x : -INFINITY;
        M1 = (mlv.w > 0.f) ? mlv.y : -INFINITY;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            M0 = fmaxf(M0, __shfl_xor_sync(0xffffffffu, M0, o));
            M1 = fmaxf(M1, __shfl_xor_sync(0xffffffffu, M1, o));
        }
        Wj0 = (mlv.z > 0.f) ? ex2_approx(mlv.x - M0) * mlv.z : 0.f;
        Wj1 = (mlv.w > 0.f) ? ex2_approx(mlv.y - M1) * mlv.w : 0.f;
        L0 = Wj0; L1 = Wj1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            L0 += __shfl_xor_sync(0xffffffffu, L0, o);
            L1 += __shfl_xor_sync(0xffffffffu, L1, o);
        }
    }
    TRACE2(2);
#else
    if (warp == 0) {
#ifdef DSA_COMBINE_TMA
        const float4 mlv = s_ml[lane];
#endif
        float M0 = (mlv.z > 0.f) ? mlv.x : -INFINITY;
        float M1 = (mlv.w > 0.f) ? mlv.y : -INFINITY;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            M0 = fmaxf(M0, __shfl_xor_sync(0xffffffffu, M0, o));
            M1 = fmaxf(M1, __shfl_xor_sync(0xffffffffu, M1, o));
        }
        const float W0 = (mlv.z > 0.f) ? ex2_approx(mlv.x - M0) * mlv.z : 0.f;
        const float W1 = (mlv.w > 0.f) ? ex2_approx(mlv.y - M1) * mlv.w : 0.f;
        float L0 = W0, L1 = W1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            L0 += __shfl_xor_sync(0xffffffffu, L0, o);
            L1 += __shfl_xor_sync(0xffffffffu, L1, o);
        }
        s_W[0][lane] = W0;
        s_W[1][lane] = W1;
        if (lane == 0) { s_red[0] = M0; s_red[1] = M1; s_red[2] = L0; s_red[3] = L1; }
    }
    __syncthreads();
    TRACE2(2);
    const float M0 = s_red[0], M1 = s_red[1], L0 = s_red[2], L1 = s_red[3];
#endif
#if defined(DSA_COMBINE_TMA) && !defined(DSA_ONE_BAR)
    mbar_wait(barc, 0);
#endif
    // ---- reduction: thread (u = tid & 63, group g = tid >> 6) sums unit u of tiles [8g, 8g+8) ---------
    //      unit u = (warp & 1) * 32 + nt * 4 + q holds {h0: d, d+1 ; h1: d, d+1}, d = 128wp + 64(u>>5) + 8((u>>2)&7) + 2(u&3)
    float acc[4] = { 0.f, 0.f, 0.f, 0.f };
    {
#ifdef DSA_COMBINE_TMA
        const uint2* sv = reinterpret_cast<const uint2*>(smem + OFF_C_O) + (g * (NTILES / 4)) * (TSLICE_BYTES / 8) + u;
        uint2 v[NTILES / 4];
#pragma unroll
        for (int jj = 0; jj < NTILES / 4; ++jj) v[jj] = sv[jj * (TSLICE_BYTES / 8)];
#endif
#pragma unroll
        for (int jj = 0; jj < NTILES / 4; ++jj) {
            const int j = g * (NTILES / 4) + jj;
#if defined(DSA_STATS_ALL) && !defined(DSA_COMBINE_TMA)
            const float W0 = __shfl_sync(0xffffffffu, Wj0, j), W1 = __shfl_sync(0xffffffffu, Wj1, j);
#else
            const float W0 = s_W[0][j], W1 = s_W[1][j];
#endif
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&v[jj].x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&v[jj].y));
            // empty tiles (W == 0) hold stale workspace data: select it away instead of multiplying by 0
            acc[0] += (W0 > 0.f) ? W0 * f0.x : 0.f; acc[1] += (W0 > 0.f) ? W0 * f0.y : 0.f;
            acc[2] += (W1 > 0.f) ? W1 * f1.x : 0.f; acc[3] += (W1 > 0.f) ? W1 * f1.y : 0.f;
        }
    }
    if (g > 0) { s_acc[g - 1][u][0] = acc[0]; s_acc[g - 1][u][1] = acc[1]; s_acc[g - 1][u][2] = acc[2]; s_acc[g - 1][u][3] = acc[3]; }
    __syncthreads();
    if (g == 0) {
#pragma unroll
        for (int k = 0; k < 3; ++k) { acc[0] += s_acc[k][u][0]; acc[1] += s_acc[k][u][1]; acc[2] += s_acc[k][u][2]; acc[3] += s_acc[k][u][3]; }
        const float inv0 = (L0 > 0.f) ? (1.f / L0) : 0.f;
        const float inv1 = (L1 > 0.f) ? (1.f / L1) : 0.f;
#ifndef DSA_PUB8
        // 8-B unit u of a tile piece: w1 = u / 32, 16-B unit (np, q) = ((u / 2) / 4, (u / 2) % 4), half h = u % 2 -> n-tile 2 np + h
        const int d = wp * 128 + (u >> 5) * 64 + (((u >> 3) & 3) * 2 + (u & 1)) * 8 + ((u >> 1) & 3) * 2;
#else
        const int d = wp * 128 + (u >> 5) * 64 + ((u >> 2) & 7) * 8 + (u & 3) * 2;
#endif
        __nv_bfloat16* o0 = out + (static_cast<long>(token) * H + p) * DC + d;
        __nv_bfloat16* o1 = out + (static_cast<long>(token) * H + p + NPAIRS) * DC + d;
        *reinterpret_cast<uint32_t*>(o0) = pack_bf16x2(acc[0] * inv0, acc[1] * inv0);
        *reinterpret_cast<uint32_t*>(o1) = pack_bf16x2(acc[2] * inv1, acc[3] * inv1);
        if (u == 0 && wp == 0) {
            lse[static_cast<long>(token) * H + p] = (L0 > 0.f) ? (M0 + log2f(L0)) : -INFINITY;
            lse[static_cast<long>(token) * H + p + NPAIRS] = (L1 > 0.f) ? (M1 + log2f(L1)) : -INFINITY;
        }
    }
    TRACE2(3);
}

// ------------------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------------------
// fused: block b -> (token, tile), tile-major (the busy, low tiles of every token are dispatched first)
__global__ void __launch_bounds__(NTHREADS, CTAS_PER_SM)
dsa_fused_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                 const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
                 const int* __restrict__ sparse_indices,
                 float* __restrict__ ws_ml, __half* __restrict__ ws_o, unsigned* __restrict__ counters,
                 __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
                 int num_tokens, int wave_tiles, unsigned epoch, long total_rows, float scale_log2) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int bid = blockIdx.x;
#ifdef DSA_TOKEN_MAJOR
    const int token = bid / NTILES, tile = bid % NTILES;
#else
#ifndef DSA_ROT
#define DSA_ROT 2
#endif
#ifdef DSA_WAVE
    wave_tiles = DSA_WAVE;
#endif
    const int tile = bid / num_tokens;
    const int token = (bid % num_tokens + DSA_ROT * (tile / wave_tiles)) % num_tokens;
#endif
    tile_phase<true>(smem, token, tile, q_nope, q_pe, ckv, kpe, sparse_indices, ws_ml, ws_o, counters,
                     total_rows, scale_log2);
    // ---- combiner phase: slice `tile` of `token` ---------------------------------------------------
    const int cid = tile;
    const uint32_t barc = smem_u32(smem + OFF_BAR) + 24;
    TRACE2(0);
    slice_combine(smem, barc, token, cid, counters + (static_cast<long>(token) * NSLICES + cid) * CTR_STRIDE,
                  epoch * static_cast<unsigned>(NTILES), ws_ml, ws_o, out, lse);
}

// two-kernel fallback (num_tokens too large for co-residency): tile kernel + PDL combine kernel
__global__ void __launch_bounds__(NTHREADS, CTAS_PER_SM)
dsa_tile_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
                const int* __restrict__ sparse_indices,
                float* __restrict__ ws_ml, __half* __restrict__ ws_o,
                long total_rows, float scale_log2) {
    extern __shared__ __align__(128) unsigned char smem[];
    tile_phase<false>(smem, blockIdx.y, blockIdx.x, q_nope, q_pe, ckv, kpe, sparse_indices,
                      ws_ml, ws_o, nullptr, total_rows, scale_log2);
}

__global__ void __launch_bounds__(NTHREADS)
dsa_combine_kernel(const float* __restrict__ ws_ml, const __half* __restrict__ ws_o,
                   __nv_bfloat16* __restrict__ out, float* __restrict__ lse) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int cid = blockIdx.x;
    const int token = blockIdx.y;
    const uint32_t barc = smem_u32(smem + OFF_BAR) + 24;
    TRACE2(0);
    if (threadIdx.x == 0) { mbar_init(barc, 1); mbar_init(barc + 8, 1); mbar_fence_init(); }
    pdl_wait();   // everything the tile kernel wrote is visible after this
    __syncthreads();
    slice_combine(smem, barc, token, cid, nullptr, 0u, ws_ml, ws_o, out, lse);
}

}  // namespace dsa

std::vector<torch::Tensor> dsa_sparse_attention(torch::Tensor q_nope, torch::Tensor q_pe,
                                                torch::Tensor ckv_cache, torch::Tensor kpe_cache,
                                                torch::Tensor sparse_indices, double sm_scale,
                                                torch::Tensor ws_ml, torch::Tensor ws_o,
                                                torch::Tensor counters, int64_t epoch) {
    TORCH_CHECK(q_nope.is_cuda() && q_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous());
    TORCH_CHECK(ckv_cache.is_contiguous() && kpe_cache.is_contiguous());
    TORCH_CHECK(sparse_indices.is_contiguous() && sparse_indices.dtype() == torch::kInt32);
    const int T = static_cast<int>(q_nope.size(0));
    TORCH_CHECK(q_nope.size(1) == dsa::H && q_nope.size(2) == dsa::DC);
    TORCH_CHECK(q_pe.size(2) == dsa::DP);
    TORCH_CHECK(ckv_cache.size(1) == 64 && ckv_cache.size(2) == dsa::DC);
    TORCH_CHECK(sparse_indices.size(1) == dsa::TOPK);
    const long total_rows = static_cast<long>(ckv_cache.size(0)) * ckv_cache.size(1);

    const c10::cuda::OptionalCUDAGuard guard(q_nope.device());
    auto out = torch::empty({T, dsa::H, dsa::DC}, q_nope.options());
    auto lse = torch::empty({T, dsa::H}, q_nope.options().dtype(torch::kFloat32));
    if (T == 0) return {out, lse};
    TORCH_CHECK(ws_ml.numel() >= static_cast<long>(T) * dsa::NTILES * dsa::ML_FLOATS_PER_TILE);
    TORCH_CHECK(ws_o.numel() >= static_cast<long>(T) * dsa::NTILES * dsa::H * dsa::DC && ws_o.dtype() == torch::kFloat16);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(ws_o.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(ws_ml.data_ptr()) % 16 == 0);
    TORCH_CHECK(counters.numel() >= T * dsa::NSLICES * dsa::CTR_STRIDE && counters.dtype() == torch::kInt32);
    TORCH_CHECK(epoch >= 1 && epoch < (1LL << 24));
    static bool attr_set = false;
    static int max_resident = 0;   // co-resident CTAs of the fused kernel on this device
    static int num_sms = 0;
    if (!attr_set) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(dsa::dsa_tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsa::SMEM_BYTES));
        C10_CUDA_CHECK(cudaFuncSetAttribute(dsa::dsa_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsa::SMEM_BYTES));
        C10_CUDA_CHECK(cudaFuncSetAttribute(dsa::dsa_combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dsa::SMEM_C_BYTES));
        int per_sm = 0, dev = 0, sms = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dsa::dsa_fused_kernel, dsa::NTHREADS, dsa::SMEM_BYTES));
        max_resident = per_sm * sms;
        num_sms = sms;
        attr_set = true;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    const int fused_grid = dsa::NTILES * T;
    // block order: tile-major with the token order rotated every `wave_tiles` tiles (the dispatcher places
    // blocks b and ~b + num_sms on the same SM); 22 tiles per wave measured best for T = 8 on 148 SMs
    const int wave_tiles = (T == 8 && num_sms == 148) ? 22 : std::max(1, num_sms / T);
#ifndef DSA_FORCE_PDL
    if (fused_grid <= max_resident) {
        // single launch: every CTA is resident, so the polling combiner phases can never starve a tile phase
        dsa::dsa_fused_kernel<<<fused_grid, dsa::NTHREADS, dsa::SMEM_BYTES, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
            sparse_indices.data_ptr<int>(), ws_ml.data_ptr<float>(), reinterpret_cast<__half*>(ws_o.data_ptr()),
            reinterpret_cast<unsigned*>(counters.data_ptr<int>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
            T, wave_tiles, static_cast<unsigned>(epoch), total_rows, scale_log2);
        C10_CUDA_CHECK(cudaGetLastError());
        return {out, lse};
    }
#endif

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(dsa::NTILES, T, 1);
    cfg.blockDim = dim3(dsa::NTHREADS, 1, 1);
    cfg.dynamicSmemBytes = dsa::SMEM_BYTES;
    cfg.stream = stream;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsa::dsa_tile_kernel,
                                      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
                                      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
                                      reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
                                      reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
                                      sparse_indices.data_ptr<int>(),
                                      ws_ml.data_ptr<float>(), reinterpret_cast<__half*>(ws_o.data_ptr()),
                                      total_rows, scale_log2));
    // combine kernel: programmatic dependent launch, resident before kernel 1 drains
    cudaLaunchConfig_t cfg2 = {};
    cfg2.gridDim = dim3(dsa::NSLICES, T, 1);
    cfg2.blockDim = dim3(dsa::NTHREADS, 1, 1);
    cfg2.dynamicSmemBytes = dsa::SMEM_C_BYTES;
    cfg2.stream = stream;
    cudaLaunchAttribute attr2[1];
    attr2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr2[0].val.programmaticStreamSerializationAllowed = 1;
    cfg2.attrs = attr2;
    cfg2.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg2, dsa::dsa_combine_kernel,
                                      ws_ml.data_ptr<float>(), reinterpret_cast<const __half*>(ws_o.data_ptr()),
                                      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>()));
    return {out, lse};
}

torch::Tensor read_smid() {
#ifdef DSA_TRACE
    auto t = torch::empty({8 * 64}, torch::dtype(torch::kInt32));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), dsa::g_smid, sizeof(unsigned) * 8 * 64));
    return t;
#else
    return torch::empty({0}, torch::dtype(torch::kInt32));
#endif
}

torch::Tensor read_trace() {
#ifdef DSA_TRACE
    auto t = torch::empty({4 * 8 * 64 * 16}, torch::dtype(torch::kInt64));
    const size_t n = 8 * 64 * 16;
    auto* d = reinterpret_cast<unsigned long long*>(t.data_ptr());
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(d, dsa::g_trace, sizeof(unsigned long long) * n));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(d + n, dsa::g_trace2, sizeof(unsigned long long) * n));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(d + 2 * n, dsa::g_clk, sizeof(unsigned long long) * n));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(d + 3 * n, dsa::g_clk2, sizeof(unsigned long long) * n));
    return t;
#else
    return torch::empty({0}, torch::dtype(torch::kInt64));
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_sparse_attention", &dsa_sparse_attention, "DSA sparse attention (B200)");
    m.def("read_trace", &read_trace, "per-CTA phase timestamps (debug builds only)");
    m.def("read_smid", &read_smid, "per-tile SM id | blockIdx << 16 (debug builds only)");
}
