// MoE FP8 block-scale (DeepSeek-V3 routing) for Blackwell B200 — single persistent kernel.
//   consumer warps (16): routing (top-k, expert compaction, row copy) -> per-128-K fp32 promotion of
//                        tcgen05 accumulators -> SwiGLU/fp8 quant (GEMM1) or weighted scatter-add (GEMM2)
//   producer warp:       TMA loads (weights via 4D/2D boxes, activations, scales); L2 prefetch of the first
//                        weight tile overlaps the routing phase
//   MMA warp:            tcgen05.mma (M=128,N=256,K=32) into double-buffered TMEM temp accumulators
//   signaler warp:       publishes tile completion to other CTAs and converts finished output rows to bf16
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "common.cuh"

namespace {

constexpr int E_GLOBAL = 256, E_LOCAL = 32, HID = 7168, INTER = 2048, TOPK = 8, N_GROUP = 8, TOPK_GROUP = 4;
constexpr int KB1 = HID / 128;    // 56 k-blocks for GEMM1
constexpr int KB2 = INTER / 128;  // 16 k-blocks for GEMM2
constexpr int NT1 = INTER / 128;  // 16 n-tiles (128 gate + 128 up) for GEMM1
constexpr int NT2 = HID / 256;    // 28 n-tiles for GEMM2
constexpr int TILES1 = E_LOCAL * NT1;   // 512
constexpr int TILES2 = E_LOCAL * NT2;   // 896
#ifndef NSPLIT
#define NSPLIT 148
#endif
constexpr int TILES2_FULL = TILES2 - NSPLIT;          // GEMM2 tiles processed as one item
constexpr int QUEUE_LEN = TILES1 + TILES2_FULL + 2 * NSPLIT;
constexpr int TL_MAX = 400;
constexpr int TL_CH = 8;
#ifdef TIMELINE
#define TL_MARK(ch, g) { if ((g) < (uint32_t)TL_MAX) p.timeline[((size_t)blockIdx.x * TL_CH + (ch)) * TL_MAX + (g)] = clock64(); }
#else
#define TL_MARK(ch, g)
#endif
#ifndef QUEUE_LAG
#define QUEUE_LAG 32
#endif
#ifndef STAGES_OVERRIDE
constexpr int STAGES = 5;
#else
constexpr int STAGES = STAGES_OVERRIDE;
#endif
constexpr uint32_t STAGE_MASK_A = 0x15u;   // producer A: stages 0,2,4 ; producer B: stages 1,3
constexpr int PASS_ROWS = 80;          // token rows per pass (A tile rows); MMA rows beyond are garbage and discarded
constexpr int A_BYTES = PASS_ROWS * 128, B_BYTES = 256 * 128, STAGE_BYTES = A_BYTES + B_BYTES;
constexpr int AS_CHUNK = 4;            // k-steps per a-scale chunk load
constexpr int AS_COLS = 132;           // floats per k-step row (128 + 4 alignment slack)
constexpr int AS_RING = 4;             // chunks in flight
constexpr int AS_CHUNK_BYTES = AS_CHUNK * AS_COLS * 4;   // 2112 (TMA box bytes)
constexpr int AS_SLOT_FLOATS = (AS_CHUNK * AS_COLS + 31) & ~31;   // ring slot padded to a 128-byte multiple
constexpr int NUM_CONSUMER_WARPS = 16;
constexpr int NUM_CONSUMER_THREADS = NUM_CONSUMER_WARPS * 32;
constexpr int GEMM_THREADS = 32 * (NUM_CONSUMER_WARPS + 4);   // 4 consumer warpgroups + 1 producer warpgroup
constexpr int ROUTING_MAX_TPB = 16;    // max tokens per CTA in the routing phase (T <= 16 * num_SMs)
constexpr int NSLOT = 4;
constexpr int SIG_RING = 16;
constexpr int CONV_RING = 4;           // conversion lists in flight (signaler -> consumers)
#ifndef L2_PREFETCH_KSTEPS
#define L2_PREFETCH_KSTEPS 12
#endif // first-tile weight k-steps prefetched into L2 during routing

// counters layout (ints)
enum { C_ITEM = 0, C_DONE1 = 1, C_FINISHED = 33, C_BAR_COUNT = 34, C_BAR_GEN = 35, C_ZERO_DONE = 36, C_READY = 40, C_NUM = 128 };

struct ItemInfo { int gemm; int e; int j; int pass; int pair_base; int rows; int nk; int k0; };

struct SmemLayout {
    __align__(1024) float as_ring[AS_RING][AS_SLOT_FLOATS];   // [chunk][kstep-in-chunk][132] (+pad)
    __align__(128) float sB[2][128];   // [item parity][nb*nk + kb]  (TMA box per item)
    uint64_t full[STAGES];
    uint64_t empty[STAGES];
    uint64_t asfree[AS_RING];
    uint64_t mma_done[2];
    uint64_t tempfree[2];
    ItemInfo slot[NSLOT];
    ItemInfo sig_ring[SIG_RING];
    float amax[4][128];       // [column group][row]
    uint32_t tmem_base;
    int slots_written;        // producer -> others: number of ring slots written
    int items_done;           // consumers -> producer/signaler: number of (warp, item) completions
    int routing_done;         // consumers -> producer
    int copy_done;            // consumers -> producers: routing finished using the stage memory (row buffer / scratch)
    int npre;                 // producer A -> producer B: preloaded B tiles of the first item
    int prefetch_go;          // consumers -> producer: routing's latency-critical loads are done
    int lists_ready;          // signaler -> consumers: number of conversion lists published
    int lists_drained;        // consumers -> signaler: number of (warp, list) drains (16 per list)
    int drain_bcast;          // uniform broadcast of lists_ready to all consumer threads
    int conv_count[CONV_RING];
    int conv_j[CONV_RING];
    int conv_list[CONV_RING][128];
    int expert_off[E_LOCAL + 1];   // local copy of the expert prefix sums (computed by every CTA)
};
constexpr int SMEM_CTRL_OFF = STAGES * STAGE_BYTES;
constexpr int SMEM_TOTAL = SMEM_CTRL_OFF + sizeof(SmemLayout) + 1024;

// routing scratch: overlays the A regions of stages 0/1 (unused until the first loads)
struct RoutingScratch {
    int sel[ROUTING_MAX_TPB][TOPK];
    float selw[ROUTING_MAX_TPB][TOPK];
    int tot[16][E_LOCAL];
    int pre[16][E_LOCAL];
    int part[E_LOCAL];
    int base[E_LOCAL];
    int pair_t[ROUTING_MAX_TPB * TOPK];
    int pair_pos[ROUTING_MAX_TPB * TOPK];
    int pair_e[ROUTING_MAX_TPB * TOPK];
    int ready[E_LOCAL];
    int npairs;
    float scales[ROUTING_MAX_TPB][KB1];   // staged per-token scales
};
constexpr int ROWBUF_BYTES = 51200;   // staged token rows during routing (<= 7 tokens x 7168 B)
static_assert((int)sizeof(RoutingScratch) <= STAGES * A_BYTES, "routing scratch too large");
static_assert(ROWBUF_BYTES <= STAGES * B_BYTES, "row buffer too large");

struct GemmParams {
    const float* logits;        // [T][256]
    const __nv_bfloat16* bias;  // [256]
    const uint8_t* hs;          // [T][HID] fp8
    const float* hs_scale;      // [KB1][T]
    uint8_t* hs_sorted;         // [PC][HID]
    float* hss_sorted;          // [KB1][PC]
    const float* w1s;           // [E][32][56]
    const float* w2s;           // [E][56][16]
    uint8_t* C;                 // [PC][INTER] fp8
    float* C_scale;             // [KB2][PC]
    float* out_f32;             // [T][HID]
    __nv_bfloat16* output;      // [T][HID]
    int* tok_sorted;            // [PC]
    float* w_sorted;            // [PC]
    int* expert_off;            // [33]
    int* part_count;            // [grid][32]
    int* local_count;           // [T]
    int* cnt;                   // [T][NT2]
    int* counters;
    const int* queue;           // [QUEUE_LEN] work-queue order -> canonical tile id (shape-only, built on the host)
    long long* timeline;        // [grid][5][TL_MAX] (TIMELINE builds)
    unsigned long long* timing; // [grid][16] (TIMING builds)
    int T;
    int PC;
    int tpb;                    // tokens per CTA for routing
    int local_off;
    float rsf;
};

__device__ __forceinline__ void decode_tile_norows(int tile, ItemInfo& it) {
    it.pass = 0; it.pair_base = 0; it.rows = -1; it.k0 = 0;
    if (tile < TILES1) { it.gemm = 1; it.e = tile / NT1; it.j = tile % NT1; it.nk = KB1; }
    else {
        tile -= TILES1; it.gemm = 2;
        if (tile < TILES2_FULL) { it.e = tile / NT2; it.j = tile % NT2; it.nk = KB2; }
        else { tile -= TILES2_FULL; const int t2 = TILES2_FULL + (tile >> 1); it.e = t2 / NT2; it.j = t2 % NT2; it.nk = KB2 / 2; it.k0 = (tile & 1) * (KB2 / 2); }
    }
}
__device__ __forceinline__ void fill_rows(ItemInfo& it, const int* expert_off) {
    it.pair_base = expert_off[it.e];
    it.rows = expert_off[it.e + 1] - it.pair_base;
    if (it.rows <= 0) { it.rows = 0; it.nk = 0; }
}
__device__ __forceinline__ ItemInfo fetch_tile(const GemmParams& p, const int* expert_off) {
    ItemInfo it;
    while (true) {
        int tile = atomicAdd(p.counters + C_ITEM, 1);
        if (tile >= QUEUE_LEN) { it.gemm = -1; it.e = it.j = it.pass = it.pair_base = it.rows = it.nk = it.k0 = 0; break; }
        decode_tile_norows(__ldg(p.queue + tile), it);
        fill_rows(it, expert_off);
        if (it.rows > 0) break;
    }
    return it;
}
__device__ __forceinline__ ItemInfo next_subitem(const ItemInfo& prev, const GemmParams& p, const int* expert_off) {
    if (prev.gemm == -1) return prev;
    if ((prev.pass + 1) * PASS_ROWS < prev.rows) { ItemInfo it = prev; it.pass = prev.pass + 1; return it; }
    return fetch_tile(p, expert_off);
}
__device__ __forceinline__ int ld_volatile_smem(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ void st_volatile_smem(int* p, int v) { *reinterpret_cast<volatile int*>(p) = v; }
__device__ __forceinline__ int ld_acquire_cta_smem(const int* p) {
    int v; asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v;
}
__device__ __forceinline__ void red_release_cta_smem(int* p, int v) {
    asm volatile("red.release.cta.shared::cta.add.s32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ int atom_add_acqrel_gpu(int* p, int v) {
    int old; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void wait_slot(const SmemLayout& L, int idx) {
    WD_LOOP(ld_volatile_smem(&L.slots_written) < idx + 1, 2000 + idx, 4);
}
__device__ __forceinline__ void consumer_bar() { asm volatile("bar.sync 1, 512;" ::: "memory"); }

__device__ __forceinline__ void red_release_gpu(int* p, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
// grid barrier executed by the consumer warps of every CTA: one release-add per CTA, poll until `target`
// arrivals (the counter is reset by the last CTA at kernel end)
__device__ __forceinline__ void grid_barrier_consumers(int* counters, int tid, int target) {
    __threadfence();
    consumer_bar();
    if (tid == 0) {
        red_release_gpu(counters + C_BAR_COUNT, 1);
        WD_LOOP((int)ld_acquire_gpu(counters + C_BAR_COUNT) < target, 3000, 0);
        __threadfence();
    }
    consumer_bar();
}

#ifdef TIMING
#define TSTART(v) long long v = clock64()
#define TACC(acc, v) acc += clock64() - v
#else
#define TSTART(v)
#define TACC(acc, v)
#endif

// ------------------------------------------------------------------------------------------------
// Routing (executed by the 16 consumer warps of every CTA)
// ------------------------------------------------------------------------------------------------
#ifdef TIMING
#define RT_MARK(i) if (tid == 0) p.timing[(size_t)blockIdx.x * 32 + 16 + (i)] = clock64() - rt0
#else
#define RT_MARK(i)
#endif
__device__ void routing_phase(const GemmParams& p, RoutingScratch& R, SmemLayout& L, uint8_t* rowbuf, int tid, int warp, int lane,
                              float4 pl0, float4 pl1, uint4 pbraw) {
#ifdef TIMING
    const long long rt0 = clock64();
#endif
    const int T = p.T, tpb = p.tpb;
    const int t_begin = blockIdx.x * tpb;
    const int t_end = min(T, t_begin + tpb);
    if (tid < E_LOCAL) { R.part[tid] = 0; R.ready[tid] = 0; }
    if (tid == 0) R.npairs = 0;
    // stage this CTA's token rows and scales in smem with async copies (overlaps the top-k work below)
    for (int t = t_begin + warp; t < t_end; t += NUM_CONSUMER_WARPS) {
        const uint8_t* src = p.hs + (size_t)t * HID;
        const uint32_t dst = smem_u32(rowbuf + (size_t)(t - t_begin) * HID);
#pragma unroll
        for (int i = 0; i < HID / 16 / 32; ++i) cp_async_16(dst + (i * 32 + lane) * 16, src + (i * 32 + lane) * 16);
        if (lane < KB1 - 32) cp_async_4(smem_u32(&R.scales[t - t_begin][lane + 32]), p.hs_scale + (lane + 32) * T + t);
        cp_async_4(smem_u32(&R.scales[t - t_begin][lane]), p.hs_scale + lane * T + t);
    }
#ifdef PREFETCH_AT_ENTRY
    if (tid == 0) st_volatile_smem(&L.prefetch_go, 1);
#endif
    consumer_bar();
    RT_MARK(0);
    // ---- phase A: routing, one warp per token ----
    for (int t = t_begin + warp; t < t_end; t += NUM_CONSUMER_WARPS) {
        const int tl = t - t_begin;
        float4 l0 = pl0, l1 = pl1; uint4 braw = pbraw;
        if (t != t_begin + warp) {
            const float4* lp = reinterpret_cast<const float4*>(p.logits + (size_t)t * E_GLOBAL + lane * 8);
            l0 = lp[0]; l1 = lp[1];
        }
        float x[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const __nv_bfloat16* bb = reinterpret_cast<const __nv_bfloat16*>(&braw);
        float s[8], sb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i] = 1.0f / (1.0f + __expf(-x[i])); sb[i] = s[i] + __bfloat162float(bb[i]); }
        float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) { float v = sb[i]; if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v; }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            float o1 = __shfl_xor_sync(0xffffffffu, m1, off), o2 = __shfl_xor_sync(0xffffffffu, m2, off);
            float n1 = fmaxf(m1, o1), n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
            m1 = n1; m2 = n2;
        }
        float gs = m1 + m2;
        const int g = lane >> 2;
        int rank = 0;
#pragma unroll
        for (int gg = 0; gg < N_GROUP; ++gg) {
            float og = __shfl_sync(0xffffffffu, gs, gg * 4);
            rank += (og > gs) || (og == gs && gg < g);
        }
        const bool gsel = rank < TOPK_GROUP;
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = gsel ? sb[i] : -INFINITY;
        int sel_e[TOPK]; float sel_s[TOPK];
#pragma unroll
        for (int r = 0; r < TOPK; ++r) {
            float bv = -INFINITY; int bi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
            float cs = s[bi];
            int bidx = lane * 8 + bi;
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                int oi = __shfl_xor_sync(0xffffffffu, bidx, off);
                if (ov > bv || (ov == bv && oi < bidx)) { bv = ov; bidx = oi; }
            }
            float sv = __shfl_sync(0xffffffffu, cs, bidx >> 3);
            if (lane == (bidx >> 3)) {
#pragma unroll
                for (int i = 0; i < 8; ++i) if (i == (bidx & 7)) v[i] = -INFINITY;
            }
            sel_e[r] = bidx; sel_s[r] = sv;
        }
        float ssum = 0.f;
#pragma unroll
        for (int r = 0; r < TOPK; ++r) ssum += sel_s[r];
        const float inv = p.rsf / (ssum + 1e-20f);
        int lc = 0;
#pragma unroll
        for (int r = 0; r < TOPK; ++r) {
            int le = sel_e[r] - p.local_off;
            const bool is_local = le >= 0 && le < E_LOCAL;
            if (is_local) { lc++; if (lane == 0) atomicAdd(&R.part[le], 1); }
            if (lane == r) { R.sel[tl][r] = is_local ? le : -1; R.selw[tl][r] = sel_s[r] * inv; }
        }
        if (lane == 0) p.local_count[t] = lc;
        if (lc == 0) {
            uint4* orow = reinterpret_cast<uint4*>(p.output + (size_t)t * HID);
            for (int i = lane; i < HID * 2 / 16; i += 32) orow[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    consumer_bar();
    RT_MARK(1);
    if (tid < E_LOCAL) p.part_count[blockIdx.x * E_LOCAL + tid] = R.part[tid];
    cp_async_wait_all();   // staged rows/scales have landed (per thread); visibility to other threads via the barriers below
    grid_barrier_consumers(p.counters, tid, (int)gridDim.x);
    RT_MARK(2);
#ifndef PREFETCH_AT_ENTRY
    if (tid == 0) st_volatile_smem(&L.prefetch_go, 1);   // weight prefetch after the routing loads are done
#endif
    // ---- phase B: expert offsets (global prefix) and this CTA's prefix ----
    {
        const int ee = tid & 31, part = tid >> 5;   // 16 parts
        int tot = 0, pre = 0;
#pragma unroll 4
        for (int b = part; b < (int)gridDim.x; b += 16) { int c = __ldcg(p.part_count + b * E_LOCAL + ee); tot += c; pre += (b < (int)blockIdx.x) ? c : 0; }
        R.tot[part][ee] = tot; R.pre[part][ee] = pre;
    }
    consumer_bar();
    if (tid < E_LOCAL) {
        int tot = 0, pre = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) { tot += R.tot[i][tid]; pre += R.pre[i][tid]; }
        int v = tot;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) { int o = __shfl_up_sync(0xffffffffu, v, off); if (lane >= off) v += o; }
        R.base[tid] = (v - tot) + pre;
        L.expert_off[tid + 1] = v; if (tid == 0) L.expert_off[0] = 0;
        if (blockIdx.x == 0) { p.expert_off[tid + 1] = v; if (tid == 0) p.expert_off[0] = 0; }
    }
    consumer_bar();
    RT_MARK(3);
    if (tid == 0) { __threadfence_block(); st_volatile_smem(&L.routing_done, 1); }   // producer may start decoding tiles
    // ---- scatter: warp per token; lane r handles slot r; rank = earlier (token, slot) pairs with the same expert
    for (int t = t_begin + warp; t < t_end; t += NUM_CONSUMER_WARPS) {
        const int tl = t - t_begin;
        int le = -1; float w = 0.f;
        if (lane < TOPK) { le = R.sel[tl][lane]; w = R.selw[tl][lane]; }
        if (le >= 0) {
            int cnt = 0;
            for (int tt = 0; tt < tl; ++tt) {
#pragma unroll
                for (int r = 0; r < TOPK; ++r) cnt += (R.sel[tt][r] == le);
            }
            const int pos = R.base[le] + cnt;
            p.tok_sorted[pos] = t; p.w_sorted[pos] = w;
            int pi = atomicAdd(&R.npairs, 1);
            R.pair_t[pi] = t; R.pair_pos[pi] = pos; R.pair_e[pi] = le;
        }
    }
    consumer_bar();
    RT_MARK(4);
    // ---- copy: token rows (7168 B) and per-token scales into expert-sorted order ----
    const int npairs = R.npairs;
    for (int pi = warp; pi < npairs; pi += NUM_CONSUMER_WARPS) {
        const int t = R.pair_t[pi], pr = R.pair_pos[pi];
        const uint4* src = reinterpret_cast<const uint4*>(rowbuf + (size_t)(t - t_begin) * HID);
        uint4* dst = reinterpret_cast<uint4*>(p.hs_sorted) + (size_t)pr * (HID / 16);
#pragma unroll
        for (int i = 0; i < HID / 16 / 32; ++i) dst[i * 32 + lane] = src[i * 32 + lane];
        if (lane < KB1 - 32) p.hss_sorted[(lane + 32) * p.PC + pr] = R.scales[t - t_begin][lane + 32];
        p.hss_sorted[lane * p.PC + pr] = R.scales[t - t_begin][lane];
        if (lane == 0) atomicAdd(&R.ready[R.pair_e[pi]], 1);
    }
    __threadfence();
    consumer_bar();
    RT_MARK(5);
    // publish per-expert readiness (pairs copied) to all CTAs
    if (tid < E_LOCAL && R.ready[tid] > 0) atomicAdd(p.counters + C_READY + tid, R.ready[tid]);
    consumer_bar();
    if (tid == 0) { __threadfence_block(); st_volatile_smem(&L.copy_done, 1); }   // stage memory may now be overwritten
    RT_MARK(6);
}

// ------------------------------------------------------------------------------------------------
// Persistent kernel
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(GEMM_THREADS, 1) moe_kernel(
    const __grid_constant__ CUtensorMap map_w1, const __grid_constant__ CUtensorMap map_w2,
    const __grid_constant__ CUtensorMap map_c, const __grid_constant__ CUtensorMap map_a1,
    const __grid_constant__ CUtensorMap map_w1s, const __grid_constant__ CUtensorMap map_w2s,
    const __grid_constant__ CUtensorMap map_hss, const __grid_constant__ CUtensorMap map_cs, GemmParams p)
{
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    SmemLayout& L = *reinterpret_cast<SmemLayout*>(smem + SMEM_CTRL_OFF);
    RoutingScratch& R = *reinterpret_cast<RoutingScratch*>(smem + STAGES * B_BYTES);   // in the A-region area (unused during routing)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t smem_base = smem_u32(smem);
    const uint32_t full0 = smem_u32(&L.full[0]), empty0 = smem_u32(&L.empty[0]), asfree0 = smem_u32(&L.asfree[0]);
    const uint32_t mmadone0 = smem_u32(&L.mma_done[0]), tempfree0 = smem_u32(&L.tempfree[0]);
#ifdef TIMING
    long long t_wait = 0, t_total = 0, t_epi = 0, t_gather = 0, t_x1 = 0, t_x2 = 0, t_drain = 0;
    long long t_k0 = clock64();
#endif

#ifdef WATCHDOG
    g_watchdog = p.counters + 48;
#endif
    // consumer warps: issue the first token's routing loads immediately (latency overlaps the setup below)
    float4 pl0 = make_float4(0.f, 0.f, 0.f, 0.f), pl1 = pl0; uint4 pbraw = make_uint4(0u, 0u, 0u, 0u);
    if (warp < NUM_CONSUMER_WARPS) {
        const int t = blockIdx.x * p.tpb + warp;
        if (t < min(p.T, blockIdx.x * p.tpb + p.tpb)) {
            const float4* lp = reinterpret_cast<const float4*>(p.logits + (size_t)t * E_GLOBAL + lane * 8);
            pl0 = lp[0]; pl1 = lp[1];
        }
        pbraw = reinterpret_cast<const uint4*>(p.bias)[lane];
    }
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) { mbar_init(full0 + 8 * s, 1); mbar_init(empty0 + 8 * s, 1); }
        for (int s = 0; s < AS_RING; ++s) mbar_init(asfree0 + 8 * s, NUM_CONSUMER_WARPS);
        for (int b = 0; b < 2; ++b) { mbar_init(mmadone0 + 8 * b, 1); mbar_init(tempfree0 + 8 * b, NUM_CONSUMER_WARPS); }
        fence_barrier_init();
        tma_prefetch_desc(&map_w1); tma_prefetch_desc(&map_w2); tma_prefetch_desc(&map_c); tma_prefetch_desc(&map_a1);
        tma_prefetch_desc(&map_w1s); tma_prefetch_desc(&map_w2s); tma_prefetch_desc(&map_hss); tma_prefetch_desc(&map_cs);
        L.items_done = 0;
        L.slots_written = 0;
        L.routing_done = 0;
        L.copy_done = 0;
        L.prefetch_go = 0;
        L.lists_ready = 0;
        L.lists_drained = 0;
    }
    if (warp == NUM_CONSUMER_WARPS) {
        tmem_alloc(smem_u32(&L.tmem_base), 512);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = L.tmem_base;

    if (warp >= NUM_CONSUMER_WARPS) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
        if (warp == NUM_CONSUMER_WARPS) {
            // =========================== TMA producer ===========================
            if (lane == 0) {
#ifdef WEIGHTS_EVICT_NORMAL
                const uint64_t pol_w = policy_evict_normal();
#else
                const uint64_t pol_w = policy_evict_first();
#endif
                const uint64_t pol_c = policy_evict_last();
                // claim the first two tiles before routing is known; prefetch the first tile's weights into L2
                ItemInfo it0, it1;
                {
                    int t0 = atomicAdd(p.counters + C_ITEM, 1);
                    if (t0 >= QUEUE_LEN) { it0.gemm = -1; it0.e = it0.j = it0.pass = it0.pair_base = it0.rows = it0.nk = it0.k0 = 0; } else decode_tile_norows(__ldg(p.queue + t0), it0);
                }
                // weights of the first item do not depend on routing: load the first stages' B tiles now
                const int npre = 0;   // (no B preload: the routing row buffer lives in the B area)
                L.npre = npre;
                WD_LOOP(ld_volatile_smem(&L.prefetch_go) == 0, 3002, 1);
                if (it0.gemm == 1) {
                    for (int kk = npre; kk < npre + L2_PREFETCH_KSTEPS; ++kk) tma_prefetch_4d(&map_w1, kk * 128, it0.j * 128, 0, it0.e);
                }
                WD_LOOP(ld_volatile_smem(&L.routing_done) == 0, 3001, 1);
                __threadfence_block();
                if (it0.gemm != -1) fill_rows(it0, L.expert_off);
                it1 = next_subitem(it0, p, L.expert_off);
                L.slot[0] = it0; L.slot[1] = it1;
                L.sig_ring[0] = it0; L.sig_ring[1] = it1;
                __threadfence_block();
                st_volatile_smem(&L.slots_written, 2);
                WD_LOOP(ld_volatile_smem(&L.copy_done) == 0, 3003, 1);
                uint32_t g = 0;
                int n = 0;
                ItemInfo cur = it0;
                while (cur.gemm != -1) {
                    if (n >= 2) WD_LOOP(ld_volatile_smem(&L.items_done) < (n - 1) * NUM_CONSUMER_WARPS, 4000 + n, 1);
                    L.slot[(n + 2) % NSLOT] = next_subitem(L.slot[(n + 1) % NSLOT], p, L.expert_off);
                    L.sig_ring[(n + 2) % SIG_RING] = L.slot[(n + 2) % NSLOT];
                    __threadfence_block();
                    st_volatile_smem(&L.slots_written, n + 3);
                    if (cur.nk > 0) {
                        // rows of this expert copied by all CTAs (GEMM1 A / scales) and the output accumulator zeroed (GEMM2)
                        WD_LOOP((int)ld_acquire_gpu(p.counters + C_READY + cur.e) < cur.rows, 8000 + cur.e, 1);
                        if (cur.gemm == 2) {
                            WD_LOOP((int)ld_acquire_gpu(p.counters + C_DONE1 + cur.e) < NT1, 5000 + cur.e, 1);
                            WD_LOOP((int)ld_acquire_gpu(p.counters + C_ZERO_DONE) < (int)gridDim.x, 5500, 1);
                        }
                        fence_proxy_async_global();
                    }
                    const int row0 = cur.pair_base + cur.pass * PASS_ROWS;
                    const int row0a = row0 & ~3;   // 16-byte aligned start for the a-scale box
                    const int pp = n & 1;
                    for (int kk = 0; kk < cur.nk; ++kk, ++g) {
                        if (!((STAGE_MASK_A >> (g % STAGES)) & 1u)) continue;   // producer B owns the other stages
                        const uint32_t s = g % STAGES;
                        TSTART(tw);
                        TL_MARK(5, g);
                        if (g >= STAGES) mbar_wait(empty0 + 8 * s, ((g - STAGES) / STAGES) & 1, 1);
                        TL_MARK(6, g);
                        const uint32_t ch = (g / AS_CHUNK) % AS_RING;
                        const bool new_chunk = (kk % AS_CHUNK) == 0;
                        if (new_chunk && g >= AS_CHUNK * AS_RING) mbar_wait(asfree0 + 8 * ch, ((g / AS_CHUNK - AS_RING) / AS_RING) & 1, 1);
                        TL_MARK(7, g);
                        TACC(t_wait, tw);
                        const uint32_t sB = smem_base + s * B_BYTES, sA = smem_base + STAGES * B_BYTES + s * A_BYTES;
                        const uint32_t fb = full0 + 8 * s;
                        const bool b_preloaded = (n == 0) && (kk < npre);
                        uint32_t tx = A_BYTES + (b_preloaded ? 0u : B_BYTES) + (new_chunk ? AS_CHUNK_BYTES : 0);
                        const int kb = cur.k0 + kk;
                        if (cur.gemm == 1) {
                            if (kk == 0) tx += 2 * KB1 * 4;
                            mbar_expect_tx(fb, tx);
                            if (kk == 0) tma_load_4d(smem_u32(&L.sB[pp][0]), &map_w1s, fb, 0, cur.j, 0, cur.e);
                            if (new_chunk) tma_load_2d_plain(smem_u32(&L.as_ring[ch][0]), &map_hss, fb, row0a, kb);
                            tma_load_2d(sA, &map_a1, fb, kb * 128, row0, pol_c);
                            if (!b_preloaded) tma_load_4d(sB, &map_w1, fb, kb * 128, cur.j * 128, 0, cur.e, pol_w);
                            TL_MARK(0, g);
                        } else {
                            if (kk == 0) tx += 2 * KB2 * 4;
                            mbar_expect_tx(fb, tx);
                            if (kk == 0) tma_load_2d_plain(smem_u32(&L.sB[pp][0]), &map_w2s, fb, 0, cur.e * KB1 + 2 * cur.j);
                            if (new_chunk) tma_load_2d_plain(smem_u32(&L.as_ring[ch][0]), &map_cs, fb, row0a, kb);
                            tma_load_2d(sA, &map_c, fb, kb * 128, row0, pol_c);
                            tma_load_2d(sB, &map_w2, fb, kb * 128, cur.e * HID + cur.j * 256, pol_w);
                        }
                    }
                    ++n;
                    wait_slot(L, n);
                    cur = L.slot[n % NSLOT];
                }
            }
            __syncwarp();
        } else if (warp == NUM_CONSUMER_WARPS + 2) {
            // =========================== completion signaler ===========================
            // After all consumer warps finished an item (release/acquire on items_done): GPU-scope fence, then
            // publish GEMM1 tile completion, or (GEMM2) bump per-(token, n-tile) counters and convert finished
            // 256-column output slices to bf16.
            {
                float4* o4 = reinterpret_cast<float4*>(p.out_f32);
                const int n4 = p.T * HID / 4;
                const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int i = blockIdx.x * 32 + lane; i < n4; i += gridDim.x * 32) o4[i] = z;
                for (int i = blockIdx.x * 32 + lane; i < p.T * NT2; i += gridDim.x * 32) p.cnt[i] = 0;
                __threadfence();
                __syncwarp();
                if (lane == 0) atomicAdd(p.counters + C_ZERO_DONE, 1);
            }
            int n = 0, nlist = 0;
            while (true) {
                wait_slot(L, n);
                int gemm = 0, e = 0, j = 0, pass = 0, pair_base = 0, rows = 0, inc = 2;
                if (lane == 0) { const ItemInfo it = L.sig_ring[n % SIG_RING]; gemm = it.gemm; e = it.e; j = it.j; pass = it.pass; pair_base = it.pair_base; rows = it.rows; inc = (it.nk == KB2 / 2) ? 1 : 2; }
                gemm = __shfl_sync(0xffffffffu, gemm, 0); e = __shfl_sync(0xffffffffu, e, 0); j = __shfl_sync(0xffffffffu, j, 0);
                pass = __shfl_sync(0xffffffffu, pass, 0); pair_base = __shfl_sync(0xffffffffu, pair_base, 0); rows = __shfl_sync(0xffffffffu, rows, 0);
                inc = __shfl_sync(0xffffffffu, inc, 0);
                if (gemm == -1) break;
                if (lane == 0) WD_LOOP(ld_acquire_cta_smem(&L.items_done) < (n + 1) * NUM_CONSUMER_WARPS, 6000 + n, 3);
                __syncwarp();
                __threadfence();
                if (gemm == 1) {
                    if (lane == 0 && (pass + 1) * PASS_ROWS >= rows) atomicAdd(p.counters + C_DONE1 + e, 1);
                } else if (gemm == 2) {
                    // wait for a free list slot (consumers drain lists at item boundaries)
                    if (lane == 0) WD_LOOP(ld_acquire_cta_smem(&L.lists_drained) < (nlist - CONV_RING + 1) * NUM_CONSUMER_WARPS, 7000 + nlist, 3);
                    __syncwarp();
                    const int ls = nlist % CONV_RING;
                    const int row0 = pair_base + pass * PASS_ROWS;
                    int prows = rows - pass * PASS_ROWS; if (prows > PASS_ROWS) prows = PASS_ROWS;
                    int count = 0;
                    for (int r0 = 0; r0 < prows; r0 += 32) {
                        const int r = r0 + lane;
                        const bool valid = r < prows;
                        int tok = 0; bool last = false;
                        if (valid) {
                            tok = __ldcg(p.tok_sorted + row0 + r);
                            int old = atom_add_acqrel_gpu(p.cnt + tok * NT2 + j, inc);
                            last = old + inc == 2 * __ldcg(p.local_count + tok);
                        }
                        const unsigned mask = __ballot_sync(0xffffffffu, last);
                        if (last) L.conv_list[ls][count + __popc(mask & ((1u << lane) - 1u))] = tok;
                        count += __popc(mask);
                    }
                    if (lane == 0) { L.conv_count[ls] = count; L.conv_j[ls] = j; }
                    __syncwarp();
                    if (lane == 0) red_release_cta_smem(&L.lists_ready, 1);
                    ++nlist;
                }
                ++n;
            }
            __syncwarp();
        } else if (warp == NUM_CONSUMER_WARPS + 3) {
            // =========================== TMA producer B (odd k-steps) ===========================
            if (lane == 0) {
#ifdef WEIGHTS_EVICT_NORMAL
                const uint64_t pol_w = policy_evict_normal();
#else
                const uint64_t pol_w = policy_evict_first();
#endif
                const uint64_t pol_c = policy_evict_last();
                WD_LOOP(ld_volatile_smem(&L.routing_done) == 0, 3001, 1);
                wait_slot(L, 0);
                WD_LOOP(ld_volatile_smem(&L.copy_done) == 0, 3003, 1);
                const int npre = ld_volatile_smem(&L.npre);
                uint32_t g = 0;
                int n = 0;
                ItemInfo cur = L.slot[0];
                while (cur.gemm != -1) {
                    if (cur.nk > 0) {
                        WD_LOOP((int)ld_acquire_gpu(p.counters + C_READY + cur.e) < cur.rows, 8000 + cur.e, 1);
                        if (cur.gemm == 2) {
                            WD_LOOP((int)ld_acquire_gpu(p.counters + C_DONE1 + cur.e) < NT1, 5000 + cur.e, 1);
                            WD_LOOP((int)ld_acquire_gpu(p.counters + C_ZERO_DONE) < (int)gridDim.x, 5500, 1);
                        }
                        fence_proxy_async_global();
                    }
                    const int row0 = cur.pair_base + cur.pass * PASS_ROWS;
                    const int row0a = row0 & ~3;
                    const int pp = n & 1;
                    for (int kk = 0; kk < cur.nk; ++kk, ++g) {
                        if (((STAGE_MASK_A >> (g % STAGES)) & 1u)) continue;   // producer A owns the other stages
                        const uint32_t s = g % STAGES;
                        TSTART(tw);
                        TL_MARK(5, g);
                        if (g >= STAGES) mbar_wait(empty0 + 8 * s, ((g - STAGES) / STAGES) & 1, 1);
                        TL_MARK(6, g);
                        const uint32_t ch = (g / AS_CHUNK) % AS_RING;
                        const bool new_chunk = (kk % AS_CHUNK) == 0;
                        if (new_chunk && g >= AS_CHUNK * AS_RING) mbar_wait(asfree0 + 8 * ch, ((g / AS_CHUNK - AS_RING) / AS_RING) & 1, 1);
                        TL_MARK(7, g);
                        TACC(t_wait, tw);
                        const uint32_t sB = smem_base + s * B_BYTES, sA = smem_base + STAGES * B_BYTES + s * A_BYTES;
                        const uint32_t fb = full0 + 8 * s;
                        const bool b_preloaded = (n == 0) && (kk < npre);
                        uint32_t tx = A_BYTES + (b_preloaded ? 0u : B_BYTES) + (new_chunk ? AS_CHUNK_BYTES : 0);
                        const int kb = cur.k0 + kk;
                        if (cur.gemm == 1) {
                            if (kk == 0) tx += 2 * KB1 * 4;
                            mbar_expect_tx(fb, tx);
                            if (kk == 0) tma_load_4d(smem_u32(&L.sB[pp][0]), &map_w1s, fb, 0, cur.j, 0, cur.e);
                            if (new_chunk) tma_load_2d_plain(smem_u32(&L.as_ring[ch][0]), &map_hss, fb, row0a, kb);
                            tma_load_2d(sA, &map_a1, fb, kb * 128, row0, pol_c);
                            if (!b_preloaded) tma_load_4d(sB, &map_w1, fb, kb * 128, cur.j * 128, 0, cur.e, pol_w);
                            TL_MARK(0, g);
                        } else {
                            if (kk == 0) tx += 2 * KB2 * 4;
                            mbar_expect_tx(fb, tx);
                            if (kk == 0) tma_load_2d_plain(smem_u32(&L.sB[pp][0]), &map_w2s, fb, 0, cur.e * KB1 + 2 * cur.j);
                            if (new_chunk) tma_load_2d_plain(smem_u32(&L.as_ring[ch][0]), &map_cs, fb, row0a, kb);
                            tma_load_2d(sA, &map_c, fb, kb * 128, row0, pol_c);
                            tma_load_2d(sB, &map_w2, fb, kb * 128, cur.e * HID + cur.j * 256, pol_w);
                        }
                    }
                    ++n;
                    wait_slot(L, n);
                    cur = L.slot[n % NSLOT];
                }
            }
            __syncwarp();
        } else if (warp == NUM_CONSUMER_WARPS + 1) {
            // =========================== MMA issuer ===========================
            const uint32_t idesc = make_idesc(128, 256);
            uint32_t g = 0;
            int n = 0;
            wait_slot(L, 0);
            ItemInfo cur = L.slot[0];
            while (cur.gemm != -1) {
                for (int kk = 0; kk < cur.nk; ++kk, ++g) {
                    const uint32_t s = g % STAGES, b = g & 1;
                    TSTART(tw);
                    mbar_wait(full0 + 8 * s, (g / STAGES) & 1, 2);
                    TACC(t_wait, tw);
                    if (lane == 0) TL_MARK(1, g);
                    TSTART(tw2);
                    if (g >= 2) mbar_wait(tempfree0 + 8 * b, ((g - 2) >> 1) & 1, 2);
                    TACC(t_epi, tw2);
                    tc_fence_after();
                    if (lane == 0) {
                        const uint32_t sB = smem_base + s * B_BYTES, sA = smem_base + STAGES * B_BYTES + s * A_BYTES;
                        const uint32_t d = tmem + b * 256;
#ifndef SKIP_MMA
#pragma unroll
                        for (int ks = 0; ks < 4; ++ks)
                            tc_mma_f8(d, make_smem_desc(sA + ks * 32), make_smem_desc(sB + ks * 32), idesc, ks > 0);
#endif
                        tc_commit(mmadone0 + 8 * b);
                        tc_commit(empty0 + 8 * s);
                        TL_MARK(2, g);
                    }
                    __syncwarp();
                }
                ++n;
                wait_slot(L, n);
                cur = L.slot[n % NSLOT];
            }
        }
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" ::: "memory");
        // =========================== consumers: routing, then promotion + epilogue ===========================
        routing_phase(p, R, L, smem, tid, warp, lane, pl0, pl1, pbraw);   // row buffer in the B area
#ifdef TIMING
        long long t_rout = clock64() - t_k0;
#endif
        // 16 warps: lane quarter q = warp&3 (TMEM lanes 32q..32q+31), column group cg = warp>>2 owns
        // columns [32cg, 32cg+32) of both 128-column halves of the 256-wide tile.
        const int cg = warp >> 2;
        const int qrow = (warp & 3) * 32 + lane;
        const uint32_t tlane = (uint32_t)((warp & 3) * 32) << 16;
        const int c0 = cg * 32;
        float main_acc[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) main_acc[i] = 0.f;
        uint32_t g = 0;
        int n = 0;
        wait_slot(L, 1);
        ItemInfo cur = L.slot[0];
        ItemInfo nxt = L.slot[1];
        auto item_finished = [&]() {
            __syncwarp();
            if (lane == 0) red_release_cta_smem(&L.items_done, 1);
        };
        int drained = 0, my_gemm2 = 0;
        // convert the output slices listed by the signaler: each warp handles rows warp, warp+16, ... of a list
        auto drain_lists = [&](int target) {
            while (true) {
                int ready = 0;
                if (lane == 0) ready = ld_acquire_cta_smem(&L.lists_ready);
                ready = __shfl_sync(0xffffffffu, ready, 0);
                if (drained >= ready) { if (drained >= target) break; __nanosleep(128); continue; }
                const int ls = drained % CONV_RING;
                const int cnt = L.conv_count[ls], j = L.conv_j[ls];
                for (int r = warp; r < cnt; r += NUM_CONSUMER_WARPS) {
                    const int tk = L.conv_list[ls][r];
                    const float4* src = reinterpret_cast<const float4*>(p.out_f32 + (size_t)tk * HID + j * 256) + lane * 2;
                    const float4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
                    reinterpret_cast<uint4*>(p.output + (size_t)tk * HID + j * 256)[lane] =
                        make_uint4(f32x2_to_bf16x2(v0.x, v0.y), f32x2_to_bf16x2(v0.z, v0.w), f32x2_to_bf16x2(v1.x, v1.y), f32x2_to_bf16x2(v1.z, v1.w));
                }
                __syncwarp();
                if (lane == 0) red_release_cta_smem(&L.lists_drained, 1);
                ++drained;
            }
        };
        while (cur.gemm != -1) {
            const int pp = n & 1;
            const int row0 = cur.pair_base + cur.pass * PASS_ROWS;
            int prows = cur.rows - cur.pass * PASS_ROWS; if (prows > PASS_ROWS) prows = PASS_ROWS;   // rows in this pass
            if (prows <= 0) {   // empty item (expert without tokens, claimed before routing)
                item_finished();
                ++n; cur = nxt; wait_slot(L, n + 1); nxt = L.slot[(n + 1) % NSLOT];
                continue;
            }
            const bool active = (warp & 3) * 32 < prows;   // this warp's 32 rows contain real tokens
            const int sbstride = (cur.gemm == 1) ? KB1 : KB2;
            const uint32_t sB_addr = smem_u32(&L.sB[pp][0]) + 4u * (uint32_t)cur.k0;
            const int prow = row0 + (qrow < prows ? qrow : prows - 1);  // clamped pair row
            const float wgt = (cur.gemm == 2) ? p.w_sorted[prow] : 0.f;
            const int tok = (cur.gemm == 2) ? p.tok_sorted[prow] : 0;
            for (int kk = 0; kk < cur.nk; ++kk, ++g) {
                const uint32_t b = g & 1;
                TSTART(tw);
                mbar_wait(mmadone0 + 8 * b, (g >> 1) & 1, 0);
                TACC(t_wait, tw);
                if (warp == 0 && lane == 0) TL_MARK(3, g);
                TSTART(tx1);
                tc_fence_after();
                const float a_s = lds_f32(smem_u32(&L.as_ring[0][0]) + 4u * (uint32_t)(((g / AS_CHUNK) % AS_RING) * AS_SLOT_FLOATS + (g % AS_CHUNK) * AS_COLS + qrow + (row0 & 3)));
                if ((g % AS_CHUNK) == AS_CHUNK - 1) { __syncwarp(); if (lane == 0) mbar_arrive(asfree0 + 8 * ((g / AS_CHUNK) % AS_RING)); }
                const float sb0 = lds_f32(sB_addr + 4u * kk) * a_s, sb1 = lds_f32(sB_addr + 4u * (sbstride + kk)) * a_s;
                const uint32_t tbase = tmem + b * 256 + tlane + c0;
                TACC(t_x1, tx1);
#ifndef SKIP_PROMOTE
                TSTART(tp);
                if (active) {
                    float tmpa[16], tmpb[16];
                    tmem_ld16(tbase + 0, tmpa);
                    tmem_ld16(tbase + 16, tmpb);
                    tmem_ld_wait();
#pragma unroll
                    for (int i = 0; i < 16; ++i) main_acc[i] = fmaf(tmpa[i], sb0, main_acc[i]);
#pragma unroll
                    for (int i = 0; i < 16; ++i) main_acc[16 + i] = fmaf(tmpb[i], sb0, main_acc[16 + i]);
                    tmem_ld16(tbase + 128, tmpa);
                    tmem_ld16(tbase + 144, tmpb);
                    tmem_ld_wait();
                    // the temp accumulator is free once the last load completed: release it before the remaining FMAs
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) mbar_arrive(tempfree0 + 8 * b);
#pragma unroll
                    for (int i = 0; i < 16; ++i) main_acc[32 + i] = fmaf(tmpa[i], sb1, main_acc[32 + i]);
#pragma unroll
                    for (int i = 0; i < 16; ++i) main_acc[48 + i] = fmaf(tmpb[i], sb1, main_acc[48 + i]);
                } else {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) mbar_arrive(tempfree0 + 8 * b);
                }
                TACC(t_gather, tp);
#else
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(tempfree0 + 8 * b);
#endif
                if (warp == 0 && lane == 0) TL_MARK(4, g);
            }
            // ---------------- epilogue ----------------
            TSTART(te);
            const bool valid = qrow < prows;
            if (cur.gemm == 1) {
                float h[32];
                float amax = 0.f;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    float up = main_acc[32 + i], gate = main_acc[i];
                    float v = up / (1.0f + __expf(-up)) * gate;
                    h[i] = v; amax = fmaxf(amax, fabsf(v));
                }
                L.amax[cg][qrow] = amax;
                asm volatile("bar.sync %0, 128;" :: "r"(2 + (warp & 3)) : "memory");   // the 4 column-group warps of this quarter
                amax = fmaxf(fmaxf(L.amax[0][qrow], L.amax[1][qrow]), fmaxf(L.amax[2][qrow], L.amax[3][qrow]));
                const float scale = amax > 0.f ? amax * (1.0f / 448.0f) : 1.0f;
                const float inv = 1.0f / scale;
#ifndef SKIP_EPILOGUE
                if (valid) {
                    uint8_t* dst = p.C + (size_t)(row0 + qrow) * INTER + cur.j * 128 + c0;
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t w[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            uint16_t lo = f32x2_to_e4m3x2(h[q * 16 + u * 4 + 0] * inv, h[q * 16 + u * 4 + 1] * inv);
                            uint16_t hi = f32x2_to_e4m3x2(h[q * 16 + u * 4 + 2] * inv, h[q * 16 + u * 4 + 3] * inv);
                            w[u] = (uint32_t)lo | ((uint32_t)hi << 16);
                        }
                        reinterpret_cast<uint4*>(dst)[q] = make_uint4(w[0], w[1], w[2], w[3]);
                    }
                    if (cg == 0) p.C_scale[(size_t)cur.j * p.PC + row0 + qrow] = scale;
                }
#endif
            } else {
#ifndef SKIP_EPILOGUE
                if (valid) {
                    float* op = p.out_f32 + (size_t)tok * HID + cur.j * 256 + c0;
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                        for (int i = 0; i < 32; i += 4)
                            red_add_v4(op + hh * 128 + i, main_acc[hh * 32 + i] * wgt, main_acc[hh * 32 + i + 1] * wgt,
                                       main_acc[hh * 32 + i + 2] * wgt, main_acc[hh * 32 + i + 3] * wgt);
                    }
                }
#endif
            }
            TACC(t_epi, te);
#pragma unroll
            for (int i = 0; i < 64; ++i) main_acc[i] = 0.f;
            item_finished();
            if (cur.gemm == 2) ++my_gemm2;
            TSTART(td);
            drain_lists(0);
            TACC(t_drain, td);
            ++n;
            cur = nxt;
            wait_slot(L, n + 1);
            nxt = L.slot[(n + 1) % NSLOT];
        }
        drain_lists(my_gemm2);
#ifdef TIMING
        if (lane == 0 && warp == 0) { p.timing[(size_t)blockIdx.x * 32 + 23] = n; p.timing[(size_t)blockIdx.x * 32 + 24] = clock64() - t_k0; }
        if (lane == 0 && warp == 0) { p.timing[(size_t)blockIdx.x * 32 + 15] += t_rout; p.timing[(size_t)blockIdx.x * 32 + 12] += t_x1; p.timing[(size_t)blockIdx.x * 32 + 13] += t_x2; p.timing[(size_t)blockIdx.x * 32 + 14] += t_drain; }
#endif
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp == NUM_CONSUMER_WARPS) tmem_dealloc(tmem, 512);
#ifdef TIMING
    t_total = clock64() - t_k0;
    if (lane == 0 && (warp == 0 || warp == NUM_CONSUMER_WARPS || warp == NUM_CONSUMER_WARPS + 1)) {
        int role = warp == 0 ? 0 : (warp == NUM_CONSUMER_WARPS ? 1 : 2);
        unsigned long long* tp = p.timing + (size_t)blockIdx.x * 32 + role * 4;
        tp[0] += t_total; tp[1] += t_wait; tp[2] += t_epi; tp[3] += t_gather;
    }
#endif
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(p.counters + C_FINISHED, 1) == (int)gridDim.x - 1) {
            p.counters[C_ITEM] = 0;
            for (int i = 0; i < E_LOCAL; ++i) { p.counters[C_DONE1 + i] = 0; p.counters[C_READY + i] = 0; }
            p.counters[C_FINISHED] = 0;
            p.counters[C_ZERO_DONE] = 0;
            p.counters[C_BAR_COUNT] = 0;
            __threadfence();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------------------
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* ptr = nullptr;
        cudaDriverEntryPointQueryResult qr;
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &qr);
        TORCH_CHECK(ptr != nullptr, "cuTensorMapEncodeTiled not found");
        fn = (EncodeTiledFn)ptr;
    }
    return fn;
}
static void make_map_nd(CUtensorMap* map, const void* ptr, CUtensorMapDataType dt, int rank, const uint64_t* dims,
                        const uint64_t* strides_bytes /* rank-1 entries */, const uint32_t* box, CUtensorMapSwizzle swz) {
    cuuint64_t gdim[5], gstr[4]; cuuint32_t gbox[5], es[5];
    for (int i = 0; i < rank; ++i) { gdim[i] = dims[i]; gbox[i] = box[i]; es[i] = 1; }
    for (int i = 0; i < rank - 1; ++i) gstr[i] = strides_bytes[i];
    CUresult r = get_encode_fn()(map, dt, rank, const_cast<void*>(ptr), gdim, gstr, gbox, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
}
static void make_map(CUtensorMap* map, const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
    uint64_t dims[2] = {cols, rows}; uint64_t str[1] = {cols}; uint32_t box[2] = {128, box_rows};
    make_map_nd(map, ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B);
}

struct Workspace {
    int T = -1, PC = 0;
    torch::Tensor local_count, part_count, expert_off, tok_sorted, w_sorted, C, C_scale, out_f32, counters, timing,
                  hs_sorted, hss_sorted, cnt, queue, timeline;
};
static Workspace g_ws;
static int g_num_sms = 0;

static void ensure_workspace(int T, const torch::Device& dev) {
    if (g_ws.T == T) return;
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);
    int PC = T * TOPK + 256;
    g_ws.T = T; g_ws.PC = PC;
    g_ws.local_count = torch::empty({T}, i32);
    g_ws.part_count = torch::empty({g_num_sms, E_LOCAL}, i32);
    g_ws.expert_off = torch::zeros({E_LOCAL + 1}, i32);
    g_ws.tok_sorted = torch::empty({PC}, i32);
    g_ws.w_sorted = torch::empty({PC}, f32);
    g_ws.C = torch::zeros({PC, INTER}, u8);
    g_ws.hs_sorted = torch::zeros({PC, HID}, u8);
    g_ws.hss_sorted = torch::zeros({KB1, PC}, f32);
    g_ws.C_scale = torch::zeros({KB2, PC}, f32);
    g_ws.out_f32 = torch::empty({T, HID}, f32);
    g_ws.cnt = torch::empty({T, NT2}, i32);
    g_ws.counters = torch::zeros({C_NUM}, i32);
    g_ws.timing = torch::zeros({g_num_sms, 32}, i64);
    g_ws.timeline = torch::zeros({g_num_sms, TL_CH, TL_MAX}, i64);
    {   // work-queue order: GEMM1 tiles of expert e+QUEUE_LAG are followed by the GEMM2 items of expert e
        std::vector<int> q; q.reserve(QUEUE_LEN);
        auto push_g1 = [&](int e) { for (int j = 0; j < NT1; ++j) q.push_back(e * NT1 + j); };
        auto push_g2 = [&](int e) {
            for (int j = 0; j < NT2; ++j) {
                const int t = e * NT2 + j;   // canonical GEMM2 tile index
                if (t < TILES2_FULL) q.push_back(TILES1 + t);
                else { q.push_back(TILES1 + TILES2_FULL + 2 * (t - TILES2_FULL)); q.push_back(TILES1 + TILES2_FULL + 2 * (t - TILES2_FULL) + 1); }
            }
        };
        for (int e = 0; e < QUEUE_LAG && e < E_LOCAL; ++e) push_g1(e);
        for (int e = 0; e < E_LOCAL; ++e) { if (e + QUEUE_LAG < E_LOCAL) push_g1(e + QUEUE_LAG); push_g2(e); }
        TORCH_CHECK((int)q.size() == QUEUE_LEN, "queue size mismatch");
        g_ws.queue = torch::tensor(q, i32);
    }
}

torch::Tensor moe_run(torch::Tensor routing_logits, torch::Tensor routing_bias, torch::Tensor hidden_states,
                      torch::Tensor hidden_states_scale, torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                      torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale, int64_t local_expert_offset,
                      double routed_scaling_factor) {
    TORCH_CHECK(routing_logits.is_cuda() && routing_logits.is_contiguous() && routing_logits.dtype() == torch::kFloat32);
    TORCH_CHECK(hidden_states.is_contiguous() && hidden_states_scale.is_contiguous());
    TORCH_CHECK(gemm1_weights.is_contiguous() && gemm1_weights_scale.is_contiguous() && gemm2_weights.is_contiguous() && gemm2_weights_scale.is_contiguous());
    TORCH_CHECK(routing_bias.is_contiguous() && routing_bias.dtype() == torch::kBFloat16);
    const int T = routing_logits.size(0);
    TORCH_CHECK(routing_logits.size(1) == E_GLOBAL && hidden_states.size(0) == T && hidden_states.size(1) == HID);
    TORCH_CHECK(gemm1_weights.size(0) == E_LOCAL && gemm1_weights.size(1) == 2 * INTER && gemm1_weights.size(2) == HID);
    TORCH_CHECK(gemm2_weights.size(0) == E_LOCAL && gemm2_weights.size(1) == HID && gemm2_weights.size(2) == INTER);
    TORCH_CHECK(hidden_states_scale.size(0) == KB1 && hidden_states_scale.size(1) == T);
    const auto dev = routing_logits.device();
    c10::cuda::CUDAGuard guard(dev);
    if (g_num_sms == 0) {
        cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev.index());
        cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    }
    ensure_workspace(T, dev);
    auto output = torch::empty({T, HID}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    CUtensorMap map_w1, map_w2, map_c, map_a1, map_w1s, map_w2s, map_hss, map_cs;
    {   // W13 as [E][half][2048][7168]: one 4D box {128 k, 128 n, 2 halves, 1 e} = gate+up tile
        uint64_t dims[4] = {HID, INTER, 2, E_LOCAL};
        uint64_t str[3] = {HID, (uint64_t)INTER * HID, (uint64_t)2 * INTER * HID};
        uint32_t box[4] = {128, 128, 2, 1};
        make_map_nd(&map_w1, gemm1_weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B);
    }
    make_map(&map_w2, gemm2_weights.data_ptr(), (uint64_t)E_LOCAL * HID, INTER, 256);
    make_map(&map_c, g_ws.C.data_ptr(), (uint64_t)g_ws.PC, INTER, PASS_ROWS);
    make_map(&map_a1, g_ws.hs_sorted.data_ptr(), (uint64_t)g_ws.PC, HID, PASS_ROWS);
    {   // w1s as [E][half][16 j][56 kb] floats: box {56, 1, 2, 1}
        uint64_t dims[4] = {KB1, NT1, 2, E_LOCAL};
        uint64_t str[3] = {KB1 * 4, (uint64_t)NT1 * KB1 * 4, (uint64_t)2 * NT1 * KB1 * 4};
        uint32_t box[4] = {KB1, 1, 2, 1};
        make_map_nd(&map_w1s, gemm1_weights_scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, dims, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    {   // w2s as [E*56 nblocks][16 kb] floats: box {16, 2}
        uint64_t dims[2] = {KB2, (uint64_t)E_LOCAL * KB1};
        uint64_t str[1] = {KB2 * 4};
        uint32_t box[2] = {KB2, 2};
        make_map_nd(&map_w2s, gemm2_weights_scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, dims, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    {   // a-scales [KB][PC] floats: box {132, 8}
        uint64_t dims[2] = {(uint64_t)g_ws.PC, KB1};
        uint64_t str[1] = {(uint64_t)g_ws.PC * 4};
        uint32_t box[2] = {AS_COLS, AS_CHUNK};
        make_map_nd(&map_hss, g_ws.hss_sorted.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, dims, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
        uint64_t dims2[2] = {(uint64_t)g_ws.PC, KB2};
        make_map_nd(&map_cs, g_ws.C_scale.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, dims2, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    const int tpb = (T + g_num_sms - 1) / g_num_sms;
    TORCH_CHECK(tpb <= ROUTING_MAX_TPB && tpb * HID <= ROWBUF_BYTES, "seq_len too large for the routing phase");
    GemmParams gp;
    gp.logits = routing_logits.data_ptr<float>();
    gp.bias = reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
    gp.hs = reinterpret_cast<const uint8_t*>(hidden_states.data_ptr());
    gp.hs_scale = hidden_states_scale.data_ptr<float>();
    gp.hs_sorted = g_ws.hs_sorted.data_ptr<uint8_t>();
    gp.hss_sorted = g_ws.hss_sorted.data_ptr<float>();
    gp.w1s = gemm1_weights_scale.data_ptr<float>();
    gp.w2s = gemm2_weights_scale.data_ptr<float>();
    gp.C = g_ws.C.data_ptr<uint8_t>();
    gp.C_scale = g_ws.C_scale.data_ptr<float>();
    gp.out_f32 = g_ws.out_f32.data_ptr<float>();
    gp.output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    gp.tok_sorted = g_ws.tok_sorted.data_ptr<int>();
    gp.w_sorted = g_ws.w_sorted.data_ptr<float>();
    gp.expert_off = g_ws.expert_off.data_ptr<int>();
    gp.part_count = g_ws.part_count.data_ptr<int>();
    gp.local_count = g_ws.local_count.data_ptr<int>();
    gp.cnt = g_ws.cnt.data_ptr<int>();
    gp.counters = g_ws.counters.data_ptr<int>();
    gp.queue = g_ws.queue.data_ptr<int>();
    gp.timeline = reinterpret_cast<long long*>(g_ws.timeline.data_ptr<int64_t>());
    gp.timing = reinterpret_cast<unsigned long long*>(g_ws.timing.data_ptr<int64_t>());
    gp.T = T; gp.PC = g_ws.PC; gp.tpb = tpb;
    gp.local_off = (int)local_expert_offset;
    gp.rsf = (float)routed_scaling_factor;
    moe_kernel<<<g_num_sms, GEMM_THREADS, SMEM_TOTAL, stream>>>(map_w1, map_w2, map_c, map_a1, map_w1s, map_w2s, map_hss, map_cs, gp);
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &moe_run, "MoE FP8 block-scale forward");
    m.def("debug_state", []() {
        pybind11::dict d;
        d["local_count"] = g_ws.local_count;
        d["expert_off"] = g_ws.expert_off; d["tok_sorted"] = g_ws.tok_sorted; d["w_sorted"] = g_ws.w_sorted;
        d["C"] = g_ws.C; d["C_scale"] = g_ws.C_scale; d["out_f32"] = g_ws.out_f32; d["counters"] = g_ws.counters;
        d["timing"] = g_ws.timing; d["cnt"] = g_ws.cnt; d["timeline"] = g_ws.timeline;
        return d;
    });
}
