// Persistent fused MoE kernel (FP8 block-scale, DeepSeek-V3 routing) for NVIDIA B200, written from scratch.
//
// One launch does: routing (redundantly per CTA for T <= TMAX_SMALL; otherwise lists come from the route/scatter
// kernels) -> GEMM1 (split-K tiles, fp32 partials via red.add; the last finisher applies SwiGLU + fp8 requant of C)
// -> GEMM2 (split-K tiles; small T: weighted red.add into an fp32 scratch + bf16 finalize per n-slice; large T:
// per-pair bf16 rows combined by a separate kernel).
//
// Warp roles (NCW consumer warps + 4 service warps):
//   consumers : warp w owns TMEM lanes 32*(w%4).. (128 weight rows) and columns [(w/4)*CPW, +CPW) (tokens);
//               pulls MMA partials (tcgen05.ld), promotes with fp32 block scales, writes partial outputs.
//   MMA warp  : issues tcgen05.mma (fp8 e4m3, M=128 x N=NTOK x K=32) + commit per promotion unit.
//   TMA warp  : issues 2D TMA loads (weights: 128 rows x 128 B per stage; activations: NTOK rows x 128 B).
//   scheduler : grabs tiles (atomic work counter), decodes, waits for dependencies, gathers per-token scales
//               (and, for small T, the token rows into a per-CTA staging buffer), publishes descriptors.
//   epilogue  : fences, split-K counters, SwiGLU/requant, bf16 finalize.
// All hand-offs are mbarriers; stages/units are numbered globally so the weight stream never drains between tiles.
#pragma once
#include <cuda.h>
#include "moe_common.cuh"

namespace moe {

constexpr int TMAX_SMALL = 128;
constexpr int WSTAGE_BYTES = 128 * 128; // 128 rows x 128 B
constexpr int G1_ROWS = 64;             // gate rows per GEMM1 tile (plus 64 matching up rows)
constexpr int TILES1_PER_E = I / G1_ROWS;      // 32
constexpr int G2_ROWS = 128;            // rows per GEMM2 tile
constexpr int TILES2_PER_E = H / G2_ROWS;      // 56
constexpr int CS_BLOCK = 64;            // C (intermediate) scale block along I
constexpr int CS_PER_ROW = I / CS_BLOCK;       // 32

// counters layout (all zero on entry and on exit)
constexpr int CNT_FINISHED = 0;
constexpr int CNT_WORK = 1;
constexpr int CNT_DONE1 = 2;                        // [2, 34): completed GEMM1 (e, nt) epilogues per expert
constexpr int CNT_DONE2 = CNT_DONE1 + E_LOCAL;      // [34, 90): completed GEMM2 tiles per n-tile
constexpr int CNT_SPLIT1 = CNT_DONE2 + TILES2_PER_E;  // [90, 90+1024): completed K-splits per (e, nt)
constexpr int CNT_TOTAL = CNT_SPLIT1 + E_LOCAL * TILES1_PER_E;

#if defined(LT_PSPIN)
#define LT_PWAIT mbar_wait_spin
#elif defined(LT_PTW)
#define LT_PWAIT mbar_wait_tw
#else
#define LT_PWAIT mbar_wait
#endif
#if defined(LT_CSPIN)
#define LT_CWAIT mbar_wait_spin
#elif defined(LT_CTW)
#define LT_CWAIT mbar_wait_tw
#else
#define LT_CWAIT mbar_wait
#endif

#ifdef MOE_TRACE
__device__ __forceinline__ long long lt_gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define LT_ST(g, k) do { if (blockIdx.x == 0 && p.trace && (g) < 64) p.trace[20000 + (g) * 8 + (k)] = lt_gtimer(); } while (0)
#else
#define LT_ST(g, k) do { } while (0)
#endif

struct SmallParams {
    CUtensorMap tmap_w13;     // [32*4096 rows][7168 B], box [64][128]
    CUtensorMap tmap_w2;      // [32*7168 rows][2048 B], box [128][128]
    CUtensorMap tmap_act;     // GEMM1 activations: small T: per-CTA staging [grid*QN*NTOK][7168]; large T: Ag [pairs][7168]
    CUtensorMap tmap_c;       // Cbuf [T*8 rows][2048 B], box [NTOK][128]
    fp8_t* act_stage;         // small T staging buffer
    const float* logits;      // [T][256]
    const bf16_t* bias;       // [256]
    const fp8_t* A;           // [T][H]
    const float* A_scale;     // [56][T]
    const float* W13s;        // [32][32][56]
    const float* W2s;         // [32][56][16]
    int T;
    int local_offset;
    float scaling;
    int ks1, ks2;             // K-splits for GEMM1 / GEMM2 tiles
    float* P;                 // [T*8][N1] fp32 GEMM1 partials (zero on entry/exit)
    fp8_t* Cbuf;              // [T*8][I]
    float* Cs;                // [T*8][32]
    float* out_acc;           // [T][H] fp32 scratch (small T; zero on entry and exit)
    bf16_t* out;              // [T][H]
    bf16_t* obuf;             // [T*8][H] per-pair outputs (large T)
    int* counters;            // [CNT_TOTAL]
    long long* trace;
    // large-T (lists in global memory, produced by route/scatter kernels)
    int lists_global;
    const int* g_rcount;      // [32]
    const int* g_rstart;      // [32]
    const int* g_tok_list;    // [pairs]
    const float* g_w_list;    // [pairs]
    int* g_rzero;             // [64] counters to self-clean (rcount, rfill)
    const float* As_g;        // [56][pairs] gathered A scales (k-major, pairs contiguous)
    float* Cs_t;              // [32][pairs] C scales (k64-major) for large T
};

struct TileDesc {
    int kind;        // 0 = GEMM1, 1 = GEMM2, 2 = END
    int le, nt, ks;
    int kb0, nkb;    // k-chunk range of this tile
    int p0, cnt;     // pair list of the expert
    int tg, ntok;    // token group within the pair list
    int last_group;  // this descriptor is the last token group of the tile
    int n0;
    int y0, y1;      // TMA row coordinates (y1 < 0: single 128-row box)
};

#ifdef MOE_TRACE
#define TACC_DECL long long tacc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; long long tnow = clock64();
#define TACC(k) do { long long t_ = clock64(); tacc[(k)] += t_ - tnow; tnow = t_; } while (0)
#define TACC_STORE(w) do { if (blockIdx.x == 0 && p.trace && lane == 0) { for (int k_ = 0; k_ < 8; ++k_) p.trace[1800 + (w) * 8 + k_] = tacc[k_]; } } while (0)
#else
#define TACC_DECL
#define TACC(k) do {} while (0)
#define TACC_STORE(w) do {} while (0)
#endif

struct RoutingScratch {           // aliased into wstage[] (only used before streaming starts)
    float bias[E_GLOBAL];
    uint8_t ids[TMAX_SMALL][TOP_K];
    float w[TMAX_SMALL][TOP_K];
};

template <int NTOK, int NCW, int SCL_ROWS, int NSTAGE, int QN, int NBUF = 512 / NTOK, int MINB = 1, int CPS = 1, int NTEAM = 1>
struct SmallSmem {
    static constexpr int NB = NBUF;            // TMEM accumulator buffers
    static constexpr int XC = NCW == 4 ? 4 : ((NTOK / (NCW / 4)) < (NCW >= 16 ? 16 : 32) ? (NTOK / (NCW / 4)) : (NCW >= 16 ? 16 : 32));   // exchange columns per round
    alignas(16) float xbuf[NCW / 2][32][XC + 1];   // up-row partials handed to the gate warps (large-T SwiGLU epilogue); +1: no bank conflicts
    alignas(16) float cmax[NCW / 2][XC];       // per gate-warp column amax
    alignas(1024) uint8_t wstage[NSTAGE][CPS * WSTAGE_BYTES];
    alignas(1024) uint8_t aring[NSTAGE][CPS * NTOK * 128];
    alignas(16) float scl[QN][SCL_ROWS][NTOK]; // per queue slot (in-kernel-routing path only): GEMM1 ascale[s][tok]; GEMM2 cscale[2s+u][tok]
    alignas(16) float sclr[SCL_ROWS == 1 ? NSTAGE : 1][2][NTOK + 4];  // SCL_ROWS==1: per-stage scale rows bulk-copied with the stage (row u; +4 floats alignment slack)
    alignas(8) uint64_t full_bar[NSTAGE];      // TMA stage landed (weights + activations)
    alignas(8) uint64_t wfree_bar[NSTAGE];     // consumers finished the stage (slot reusable), count NCW
    alignas(8) uint64_t mma_bar[NB];           // MMA unit done
    alignas(8) uint64_t tfree_bar[NB];         // consumers finished reading TMEM buffer, count NCW
    alignas(8) uint64_t dfull_bar[QN];         // descriptor published (count 1)
    alignas(8) uint64_t dfree_bar[QN];         // descriptor consumed (count NCW)
    TileDesc desc[QN];
    int tile_tok[QN][NTOK];                    // activation row index (token for GEMM1, pair for GEMM2)
    float tile_w[QN][NTOK];
    uint32_t tmem_base;
    int count[E_LOCAL], start[E_LOCAL], fill[E_LOCAL], aexp[E_LOCAL];
    int na, flag;
    volatile int edone;                        // number of descriptors whose epilogue has completed
    int tok_list[TMAX_SMALL * TOP_K];
    float w_list[TMAX_SMALL * TOP_K];
};

// ---------------------------------------------------------------------------------------------
template <int NTOK, int NCW, int SCL_ROWS, int NSTAGE, int QN, int NBUF = 512 / NTOK, int MINB = 1, int CPS = 1, int NTEAM = 1>
__global__ void __launch_bounds__(NCW * 32 + 128, MINB) moe_small_kernel(const __grid_constant__ SmallParams p) {
    using Smem = SmallSmem<NTOK, NCW, SCL_ROWS, NSTAGE, QN, NBUF, MINB, CPS, NTEAM>;
    constexpr int TEAM_WARPS = NCW / NTEAM;          // consumer warps per team (each team covers all 128 rows)
    static_assert(NCW % NTEAM == 0 && TEAM_WARPS % 4 == 0, "team layout");
    constexpr int XC = Smem::XC;
    constexpr int SSTAGE_BYTES = CPS * WSTAGE_BYTES;            // weight bytes per stage
    constexpr int SASTAGE_BYTES = CPS * NTOK * 128;              // activation bytes per stage
    extern __shared__ __align__(16) uint8_t smem_raw[];
    Smem& S = *reinterpret_cast<Smem*>(
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023)));
    constexpr int ASTAGE_BYTES = NTOK * 128;
    constexpr int NB = Smem::NB;
    constexpr int CONS_THREADS = NCW * 32;
    constexpr int NTHREADS = CONS_THREADS + 128;
    constexpr int WARP_MMA = NCW, WARP_TMA = NCW + 1, WARP_SCHED = NCW + 2, WARP_EPI = NCW + 3;
    constexpr int CPW = NTOK / (TEAM_WARPS / 4);   // columns (tokens) per consumer thread
    constexpr bool NEED_TFREE = NB < 2 * NSTAGE * CPS;
    constexpr bool RING = SCL_ROWS == 1;             // scales travel with the stages (lists_global path only)
    static_assert(NCW % 4 == 0 && CPW % 16 == 0, "consumer layout");
    RoutingScratch& R = *reinterpret_cast<RoutingScratch*>(&S.wstage[0][0]);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = p.T;

    if (warp == 0) {
        tmem_alloc(smem_u32(&S.tmem_base), NB * NTOK);
        tmem_relinquish();
    }
    if (tid == 0) {
        for (int s = 0; s < NSTAGE; ++s) { mbar_init(smem_u32(&S.full_bar[s]), 1); mbar_init(smem_u32(&S.wfree_bar[s]), TEAM_WARPS); }
        for (int b = 0; b < NB; ++b) { mbar_init(smem_u32(&S.mma_bar[b]), 1); mbar_init(smem_u32(&S.tfree_bar[b]), TEAM_WARPS); }
        for (int q = 0; q < QN; ++q) { mbar_init(smem_u32(&S.dfull_bar[q]), 1); mbar_init(smem_u32(&S.dfree_bar[q]), NCW); }
        fence_proxy_async();
        tma_prefetch_desc(&p.tmap_w13);
        tma_prefetch_desc(&p.tmap_w2);
        tma_prefetch_desc(&p.tmap_act);
        tma_prefetch_desc(&p.tmap_c);
    }
    const int* tok_list = p.lists_global ? p.g_tok_list : S.tok_list;
    const float* w_list = p.lists_global ? p.g_w_list : S.w_list;
    if (!p.lists_global) {
        // ---------------- routing (redundant per CTA) ----------------
        for (int i = tid; i < E_GLOBAL; i += NTHREADS) R.bias[i] = __bfloat162float(p.bias[i]);
        if (tid < E_LOCAL) { S.count[tid] = 0; S.fill[tid] = 0; }
        __syncthreads();
        for (int t = warp; t < T; t += NTHREADS / 32) {
            int id; float w;
            route_token_warp(p.logits + (size_t)t * E_GLOBAL, R.bias, p.scaling, id, w);
            if (lane < TOP_K) { R.ids[t][lane] = (uint8_t)id; R.w[t][lane] = w; }
        }
        __syncthreads();
        for (int i = tid; i < T * TOP_K; i += NTHREADS) {
            const int le = (int)R.ids[i / TOP_K][i % TOP_K] - p.local_offset;
            if (le >= 0 && le < E_LOCAL) atomicAdd(&S.count[le], 1);
        }
        __syncthreads();
        if (warp == 0) {
            int c = S.count[lane];
            int incl = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int o = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off) incl += o;
            }
            S.start[lane] = incl - c;
            const unsigned m = __ballot_sync(0xffffffffu, c > 0);
            const int pos = __popc(m & ((1u << lane) - 1));
            if (c > 0) S.aexp[pos] = lane;
            if (lane == 0) S.na = __popc(m);
            __syncwarp();
            for (int t = 0; t < T; ++t) {
                if (lane < TOP_K) {
                    const int le = (int)R.ids[t][lane] - p.local_offset;
                    if (le >= 0 && le < E_LOCAL) {
                        const int pos2 = S.start[le] + S.fill[le];
                        S.tok_list[pos2] = t;
                        S.w_list[pos2] = R.w[t][lane];
                        S.fill[le] = S.fill[le] + 1;
                    }
                }
                __syncwarp();
            }
        }
    } else {
        if (warp == 0) {
            const int c = __ldcg(p.g_rcount + lane);
            S.count[lane] = c;
            S.start[lane] = __ldcg(p.g_rstart + lane);
            const unsigned m = __ballot_sync(0xffffffffu, c > 0);
            const int pos = __popc(m & ((1u << lane) - 1));
            if (c > 0) S.aexp[pos] = lane;
            if (lane == 0) S.na = __popc(m);
        }
    }
    __syncthreads();
    const int na = S.na;
    const int KS1 = p.ks1 > 0 ? p.ks1 : (p.lists_global ? 1 : (na <= 8 ? 4 : (na <= 14 ? 2 : 1)));
    const int KS2 = p.ks2 > 0 ? p.ks2 : (p.lists_global ? 1 : (na <= 8 ? 2 : 1));
    const int tiles1_e = TILES1_PER_E * KS1;
    const int tiles2_e = TILES2_PER_E * KS2;
    const int n1 = na * tiles1_e, n2 = na * tiles2_e;

    if (na == 0 && !p.lists_global) {
        uint4 z = make_uint4(0, 0, 0, 0);
        uint4* o = reinterpret_cast<uint4*>(p.out);
        const int n = T * H / 8;
        for (int i = blockIdx.x * NTHREADS + tid; i < n; i += gridDim.x * NTHREADS) o[i] = z;
    }

    const uint32_t wbase = smem_u32(&S.wstage[0][0]);
    const uint32_t abase = smem_u32(&S.aring[0][0]);
    const uint32_t fbar0 = smem_u32(&S.full_bar[0]);
    const uint32_t wfbar0 = smem_u32(&S.wfree_bar[0]);
    const uint32_t mbar0 = smem_u32(&S.mma_bar[0]);
    const uint32_t tbar0 = smem_u32(&S.tfree_bar[0]);
    const uint32_t dfbar0 = smem_u32(&S.dfull_bar[0]);
    const uint32_t drbar0 = smem_u32(&S.dfree_bar[0]);

    if (warp == WARP_SCHED) {
        // =============================== scheduler warp ===============================
        uint32_t qi = 0;
        if (lane == 0) S.edone = 0;
        __syncwarp();
        TACC_DECL
        for (;;) {
            int tile = 0;
            if (lane == 0) tile = atomicAdd(&p.counters[CNT_WORK], 1);
            tile = __shfl_sync(0xffffffffu, tile, 0);
            TACC(0);
            TileDesc nd;
            if (tile >= n1 + n2) {
                nd.kind = 2;
                const int q = qi % QN;
                while (qi >= QN && S.edone < (int)(qi - QN + 1)) { __nanosleep(100); }
                if (lane == 0) { S.desc[q] = nd; __threadfence_block(); mbar_arrive(dfbar0 + q * 8); }
                __syncwarp();
                break;
            }
            if (tile < n1) {
                nd.kind = 0;
                const int ae = tile / tiles1_e, rem = tile % tiles1_e;
                nd.nt = rem / KS1; nd.ks = rem % KS1;
                nd.le = S.aexp[ae];
                nd.n0 = nd.nt * G1_ROWS;
                nd.nkb = KB1 / KS1; nd.kb0 = nd.ks * nd.nkb;
                nd.y0 = nd.le * N1 + nd.n0; nd.y1 = nd.le * N1 + I + nd.n0;
            } else {
                nd.kind = 1;
                const int j = tile - n1;
                const int ae = j / tiles2_e, rem = j % tiles2_e;
                nd.nt = rem / KS2; nd.ks = rem % KS2;
                nd.le = S.aexp[ae];
                nd.n0 = nd.nt * G2_ROWS;
                nd.nkb = KB2 / KS2; nd.kb0 = nd.ks * nd.nkb;
                nd.y0 = nd.le * H + nd.n0; nd.y1 = -1;
                if (lane == 0) { while (ld_acquire_gpu(&p.counters[CNT_DONE1 + nd.le]) < TILES1_PER_E) __nanosleep(128); }
                __syncwarp();
            }
            TACC(1);
            nd.cnt = S.count[nd.le]; nd.p0 = S.start[nd.le];
            for (int tg = 0; tg < nd.cnt; tg += NTOK) {
                const int q = qi % QN;
                nd.tg = tg; nd.ntok = min(NTOK, nd.cnt - tg); nd.last_group = (tg + NTOK >= nd.cnt);
                while (qi >= QN && S.edone < (int)(qi - QN + 1)) { __nanosleep(100); }
                TACC(2);
                for (int t = lane; t < NTOK; t += 32) {
                    const bool v = t < nd.ntok;
                    const int tok = v ? tok_list[nd.p0 + tg + t] : 0;
                    S.tile_tok[q][t] = v ? (nd.kind == 0 ? tok : (nd.p0 + tg + t)) : 0;
                    S.tile_w[q][t] = v ? w_list[nd.p0 + tg + t] : 0.f;
                    if (p.lists_global) continue;
                    if (nd.kind == 0) {
                        const float* src = p.A_scale + tok;
                        for (int c0 = 0; c0 < nd.nkb; c0 += 14) {
                            float vv[14];
#pragma unroll
                            for (int i = 0; i < 14; ++i) { const int kb = nd.kb0 + c0 + i; vv[i] = (v && c0 + i < nd.nkb) ? src[(size_t)kb * T] : 0.f; }
#pragma unroll
                            for (int i = 0; i < 14; ++i) { if (c0 + i < nd.nkb) S.scl[q][c0 + i][t] = vv[i]; }
                        }
                    } else {
                        const float* src = p.Cs + (size_t)(nd.p0 + tg + (v ? t : 0)) * CS_PER_ROW + nd.kb0 * 2;
                        float vv[32];
#pragma unroll
                        for (int i = 0; i < 32; i += 4) {
                            float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
                            if (v && i < nd.nkb * 2) asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x.x), "=f"(x.y), "=f"(x.z), "=f"(x.w) : "l"(src + i) : "memory");
                            vv[i] = x.x; vv[i + 1] = x.y; vv[i + 2] = x.z; vv[i + 3] = x.w;
                        }
#pragma unroll
                        for (int i = 0; i < 32; ++i) { if (i < nd.nkb * 2 && i < SCL_ROWS) S.scl[q][i][t] = vv[i]; }
                    }
                }
                if (p.lists_global && SCL_ROWS >= 56) {
                    // (legacy) scales: pair-contiguous rows (As_g[kb][pair] or Cs_t[k64][pair]); NTOK/32 lanes-rounds per row
                    const int nrows = nd.kind == 0 ? nd.nkb : 2 * nd.nkb;
                    const float* base = nd.kind == 0 ? (p.As_g + (size_t)nd.kb0 * (size_t)(T * TOP_K)) : (p.Cs_t + (size_t)(2 * nd.kb0) * (size_t)(T * TOP_K));
                    const size_t rstride = (size_t)T * TOP_K;
                    constexpr int VPL = NTOK >= 32 ? NTOK / 32 : 1;  // values per lane per row
                    for (int rb = 0; rb < nrows; rb += 8) {
                        float vv[8][VPL];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
#pragma unroll
                            for (int j = 0; j < VPL; ++j) {
                                const int t = lane + 32 * j;
                                vv[i][j] = (rb + i < nrows && t < nd.ntok && t < NTOK) ? __ldcg(base + (size_t)(rb + i) * rstride + nd.p0 + tg + t) : 0.f;
                            }
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            if (rb + i < nrows && rb + i < SCL_ROWS) {
#pragma unroll
                                for (int j = 0; j < VPL; ++j) { if (lane + 32 * j < NTOK) S.scl[q][rb + i][lane + 32 * j] = vv[i][j]; }
                            }
                        }
                    }
                }
                __syncwarp();
                if (nd.kind == 0 && !p.lists_global) {
                    // gather the tile's token rows (K range of this tile) into per-CTA staging rows for TMA
                    const int nchunk = nd.nkb * 8;
                    const int total = nd.ntok * nchunk;
                    for (int cb = 0; cb < total; cb += 32 * 4) {
                        uint4 v4[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int c = cb + u * 32 + lane;
                            const int tt = (c < total) ? c / nchunk : 0, ch = (c < total) ? c % nchunk : 0;
                            const int row = S.tile_tok[q][tt];
                            v4[u] = (c < total) ? *reinterpret_cast<const uint4*>(p.A + (size_t)row * H + (size_t)nd.kb0 * 128 + ch * 16) : make_uint4(0, 0, 0, 0);
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int c = cb + u * 32 + lane;
                            if (c < total) {
                                const int tt = c / nchunk, ch = c % nchunk;
                                *reinterpret_cast<uint4*>(p.act_stage + (size_t)((blockIdx.x * QN + q) * NTOK + tt) * H + (size_t)nd.kb0 * 128 + ch * 16) = v4[u];
                            }
                        }
                    }
                    __threadfence();
                    fence_proxy_async_global();
                }
                if (lane == 0) S.desc[q] = nd;
                __syncwarp();
                if (lane == 0) { __threadfence_block(); mbar_arrive(dfbar0 + q * 8); }
                __syncwarp();
                TACC(3);
                ++qi;
            }
        }
        TACC_STORE(6);
    } else if (warp == WARP_TMA) {
        // =============================== TMA warp ===============================
        uint32_t g = 0, qi = 0;
        if (lane == 0) {
            TACC_DECL
            for (;;) {
                const int q = qi % QN;
                mbar_wait(dfbar0 + q * 8, (qi / QN) & 1);
                TACC(0);
                const TileDesc d = S.desc[q];
                if (d.kind == 2) break;
                const CUtensorMap* tmap = d.kind == 0 ? &p.tmap_w13 : &p.tmap_w2;
                const CUtensorMap* amap = d.kind == 0 ? &p.tmap_act : &p.tmap_c;
                const int ay = (d.kind == 0 && !p.lists_global) ? (int)(blockIdx.x * QN + q) * NTOK : (d.p0 + d.tg);
                // per-stage scale rows (lists_global): As_g[kb][pair] / Cs_t[2kb+u][pair], pair-contiguous; copied from the
                // 16B-aligned-down address, the consumer skips the first (p0+tg)&3 floats
                const size_t rstride = (size_t)T * TOP_K;
                const int soff = (d.p0 + d.tg) & 3;
                const uint32_t sbytes = (uint32_t)(((soff + d.ntok) * 4 + 15) & ~15);
                const float* sbase = d.kind == 0 ? (p.As_g + (size_t)d.kb0 * rstride) : (p.Cs_t + (size_t)(2 * d.kb0) * rstride);
                sbase += (d.p0 + d.tg) - soff;
                const int srows = (d.kind == 0 ? 1 : 2) * CPS;
                for (int s = 0; s < d.nkb; s += CPS, ++g) {
                    const uint32_t slot = g % NSTAGE;
                    if (g >= NSTAGE) LT_PWAIT(wfbar0 + slot * 8, ((g / NSTAGE) - 1) & 1);
                    TACC(1);
                    LT_ST(g, 0);
                    const uint32_t bar = fbar0 + slot * 8;
                    mbar_expect_tx(bar, SSTAGE_BYTES + SASTAGE_BYTES + (RING ? sbytes * srows : 0u));
                    if constexpr (RING) {
                        const float* src = sbase + (size_t)(s * srows) * rstride;
#pragma unroll
                        for (int u = 0; u < 2 * CPS; ++u)
                            if (u < srows) cp_async_bulk_g2s(smem_u32(&S.sclr[slot][u][0]), src + (size_t)u * rstride, sbytes, bar);
                    }
#pragma unroll
                    for (int c = 0; c < CPS; ++c) {
                        const uint32_t dst = wbase + slot * SSTAGE_BYTES + c * WSTAGE_BYTES;
                        const int x = (d.kb0 + s + c) * 128;
                        tma_load_2d(abase + slot * SASTAGE_BYTES + c * ASTAGE_BYTES, amap, x, ay, bar);
                        if (d.y1 < 0) {
                            tma_load_2d(dst, tmap, x, d.y0, bar);
                        } else {
                            tma_load_2d(dst, tmap, x, d.y0, bar);
                            tma_load_2d(dst + WSTAGE_BYTES / 2, tmap, x, d.y1, bar);
                        }
                    }
                    TACC(2);
                    LT_ST(g, 1);
                }
                ++qi;
            }
            TACC_STORE(5);
        }
        __syncwarp();
    } else if (warp == WARP_EPI) {
        // =============================== epilogue warp ===============================
        uint32_t qe = 0;
        TACC_DECL
        for (;;) {
            const int q = qe % QN;
            if (lane == 0) mbar_wait(dfbar0 + q * 8, (qe / QN) & 1);
            __syncwarp();
            TACC(0);
            const TileDesc d = S.desc[q];
            if (d.kind == 2) break;
            if (lane == 0) mbar_wait(drbar0 + q * 8, (qe / QN) & 1);
            __syncwarp();
            TACC(1);
            if (d.last_group) {
                if (d.kind == 0 && p.lists_global) {
                    // consumers already wrote C for this (expert, n-tile); publish completion
                    __threadfence();
                    fence_proxy_async_global();
                    __syncwarp();
                    if (lane == 0) { __threadfence(); atomicAdd(&p.counters[CNT_DONE1 + d.le], 1); }
                } else if (d.kind == 0) {
                    int last = 0;
                    if (lane == 0) {
                        __threadfence();
                        const int old = atomicAdd(&p.counters[CNT_SPLIT1 + d.le * TILES1_PER_E + d.nt], 1);
                        last = (old == KS1 - 1);
                    }
                    last = __shfl_sync(0xffffffffu, last, 0);
                    if (last) {
                        __threadfence();
                        const int cnt = d.cnt, p0 = d.p0, n0 = d.n0;
                        constexpr int U = 4;     // independent pairs-chunks in flight per lane
                        for (int ib = 0; ib < cnt * 16; ib += 32 * U) {
                            float4 g4[U], u4[U];
#pragma unroll
                            for (int u = 0; u < U; ++u) {
                                const int i = ib + u * 32 + lane;
                                const bool active = i < cnt * 16;
                                const size_t pair = (size_t)(p0 + (active ? (i >> 4) : 0));
                                const float* pg = p.P + pair * N1 + n0 + (i & 15) * 4;
                                g4[u] = make_float4(0.f, 0.f, 0.f, 0.f); u4[u] = g4[u];
                                if (active) {
                                    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(g4[u].x), "=f"(g4[u].y), "=f"(g4[u].z), "=f"(g4[u].w) : "l"(pg) : "memory");
                                    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(u4[u].x), "=f"(u4[u].y), "=f"(u4[u].z), "=f"(u4[u].w) : "l"(pg + I) : "memory");
                                }
                            }
#pragma unroll
                            for (int u = 0; u < U; ++u) {
                                const int i = ib + u * 32 + lane;
                                const bool active = i < cnt * 16;
                                const int n4 = (i & 15) * 4;
                                const size_t pair = (size_t)(p0 + (active ? (i >> 4) : 0));
                                float c[4] = {silu_f(u4[u].x) * g4[u].x, silu_f(u4[u].y) * g4[u].y, silu_f(u4[u].z) * g4[u].z, silu_f(u4[u].w) * g4[u].w};
                                float amax = fmaxf(fmaxf(fabsf(c[0]), fabsf(c[1])), fmaxf(fabsf(c[2]), fabsf(c[3])));
#pragma unroll
                                for (int off = 1; off < 16; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                                if (active) {
                                    const float scale = amax / 448.f;
                                    const float inv = amax > 0.f ? 448.f / amax : 0.f;
                                    const uint32_t packed = (uint32_t)pack_e4m3x2(c[0] * inv, c[1] * inv) |
                                                            ((uint32_t)pack_e4m3x2(c[2] * inv, c[3] * inv) << 16);
                                    *reinterpret_cast<uint32_t*>(p.Cbuf + pair * I + n0 + n4) = packed;
                                    if ((i & 15) == 0) p.Cs[pair * CS_PER_ROW + n0 / CS_BLOCK] = scale;
                                    float* pg = p.P + pair * N1 + n0 + n4;
                                    asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(pg), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f) : "memory");
                                    asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(pg + I), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f) : "memory");
                                }
                            }
                        }
                        __threadfence();
                        fence_proxy_async_global();
                        __syncwarp();
                        if (lane == 0) { __threadfence(); atomicAdd(&p.counters[CNT_DONE1 + d.le], 1); }
                    }
                } else if (!p.lists_global) {
                    int last = 0;
                    if (lane == 0) {
                        __threadfence();
                        const int old = atomicAdd(&p.counters[CNT_DONE2 + d.nt], 1);
                        last = (old == na * KS2 - 1);
                    }
                    last = __shfl_sync(0xffffffffu, last, 0);
                    if (last) {
                        __threadfence();
                        const int n0 = d.n0;
                        for (int i = lane; i < T * (G2_ROWS / 2); i += 32) {
                            const int t = i / (G2_ROWS / 2), c = (i % (G2_ROWS / 2)) * 2;
                            float* src = p.out_acc + (size_t)t * H + n0 + c;
                            const float v0 = ld_cg_f32(src), v1 = ld_cg_f32(src + 1);
                            *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)t * H + n0 + c) = __floats2bfloat162_rn(v0, v1);
                            asm volatile("st.global.cg.v2.f32 [%0], {%1,%2};" ::"l"(src), "f"(0.f), "f"(0.f) : "memory");
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) { __threadfence_block(); S.edone = (int)(qe + 1); }
            __syncwarp();
            TACC(2);
            ++qe;
        }
        TACC_STORE(7);
    } else if (warp == WARP_MMA) {
        // =============================== MMA warp ===============================
        uint32_t g = 0, ug = 0, qi = 0;
        const uint32_t tmem = S.tmem_base;
        const uint32_t idesc = make_idesc(128, NTOK);
        TACC_DECL
        for (;;) {
            const int q = qi % QN;
            if (lane == 0) mbar_wait(dfbar0 + q * 8, (qi / QN) & 1);
            __syncwarp();
            TACC(0);
            const TileDesc d = S.desc[q];
            if (d.kind == 2) break;
            const int nkb = d.nkb;
            if (d.kind == 0) {
                for (int s = 0; s < nkb; s += CPS, ++g) {
                    const uint32_t slot = g % NSTAGE;
                    const uint32_t par = (g / NSTAGE) & 1;
                    if (lane == 0) {
                        LT_PWAIT(fbar0 + slot * 8, par);
                        if (NEED_TFREE && ug >= (uint32_t)NB) {
#pragma unroll
                            for (int c = 0; c < CPS; ++c) mbar_wait(tbar0 + ((ug + c) % NB) * 8, (((ug + c) / NB) - 1) & 1);
                        }
                    }
                    __syncwarp();
                    TACC(1);
                    if (lane == 0) LT_ST(g, 2);
                    if (elect_one()) {
#pragma unroll
                        for (int c = 0; c < CPS; ++c) {
                            const uint64_t adesc = make_sdesc(wbase + slot * SSTAGE_BYTES + c * WSTAGE_BYTES);
                            const uint64_t bdesc = make_sdesc(abase + slot * SASTAGE_BYTES + c * ASTAGE_BYTES);
                            const uint32_t dd = tmem + ((ug + c) % NB) * NTOK;
                            umma_f8(dd, adesc, bdesc, idesc, 0u);
                            umma_f8(dd, adesc + 2, bdesc + 2, idesc, 1u);
                            umma_f8(dd, adesc + 4, bdesc + 4, idesc, 1u);
                            umma_f8(dd, adesc + 6, bdesc + 6, idesc, 1u);
                            umma_commit(mbar0 + ((ug + c) % NB) * 8);
                        }
                    }
                    ug += CPS;
                    __syncwarp();
                    TACC(2);
                    if (lane == 0) LT_ST(g, 3);
                }
            } else {
                for (int s = 0; s < nkb; s += CPS, ++g) {
                    const uint32_t slot = g % NSTAGE;
                    const uint32_t par = (g / NSTAGE) & 1;
                    if (lane == 0) {
                        LT_PWAIT(fbar0 + slot * 8, par);
                        if (NEED_TFREE && ug + 2 * CPS - 1 >= (uint32_t)NB) {
#pragma unroll
                            for (int c = 0; c < 2 * CPS; ++c) { if (ug + c >= (uint32_t)NB) mbar_wait(tbar0 + ((ug + c) % NB) * 8, (((ug + c) / NB) - 1) & 1); }
                        }
                    }
                    __syncwarp();
                    TACC(1);
                    if (elect_one()) {
#pragma unroll
                        for (int c = 0; c < CPS; ++c) {
                            const uint64_t adesc = make_sdesc(wbase + slot * SSTAGE_BYTES + c * WSTAGE_BYTES);
                            const uint64_t bdesc = make_sdesc(abase + slot * SASTAGE_BYTES + c * ASTAGE_BYTES);
                            const uint32_t u0 = ug + 2 * c;
                            const uint32_t d0 = tmem + (u0 % NB) * NTOK, d1 = tmem + ((u0 + 1) % NB) * NTOK;
                            umma_f8(d0, adesc, bdesc, idesc, 0u);
                            umma_f8(d0, adesc + 2, bdesc + 2, idesc, 1u);
                            umma_commit(mbar0 + (u0 % NB) * 8);
                            umma_f8(d1, adesc + 4, bdesc + 4, idesc, 0u);
                            umma_f8(d1, adesc + 6, bdesc + 6, idesc, 1u);
                            umma_commit(mbar0 + ((u0 + 1) % NB) * 8);
                        }
                    }
                    ug += 2 * CPS;
                    __syncwarp();
                    TACC(2);
                }
            }
            ++qi;
        }
        TACC_STORE(4);
    } else {
        // =============================== consumers ===============================
        const int team = warp / TEAM_WARPS;                  // teams consume alternate stages
        const int tw = warp % TEAM_WARPS;
        const int myrow = (tw & 3) * 32 + lane;              // TMEM lane / weight row within the 128-row stage
        const int col0 = (tw >> 2) * CPW;                    // first token column of this warp
        const uint32_t tmem = S.tmem_base;
        const uint32_t taddr_warp = ((uint32_t)((tw & 3) * 32) << 16) + (uint32_t)col0;
        uint32_t g = 0, ug = 0, qi = 0;
        TACC_DECL
        for (;;) {
            const int q = qi % QN;
            if (lane == 0) mbar_wait(dfbar0 + q * 8, (qi / QN) & 1);
            __syncwarp();
            TACC(0);
            const TileDesc d = S.desc[q];
            if (d.kind == 2) break;
            const int UPS = d.kind == 0 ? 1 : 2;
            const float* sw = d.kind == 0
                ? p.W13s + ((size_t)d.le * NB1 + ((myrow >> 6) & 1) * (I / 128) + d.n0 / 128) * KB1
                : p.W2s + ((size_t)d.le * NB2 + d.nt) * KB2;
            float acc[CPW];
#pragma unroll
            for (int n = 0; n < CPW; ++n) acc[n] = 0.f;
            float sw_next = __ldg(sw + d.kb0 + team);
            for (int s = team; s < d.nkb; s += NTEAM) {
                const float swk = sw_next;
                sw_next = __ldg(sw + d.kb0 + ((s + NTEAM < d.nkb) ? s + NTEAM : s));
                for (int uu = 0; uu < UPS; ++uu) {
                    const uint32_t ugs = ug + (uint32_t)(s * UPS + uu);
                    const uint32_t b = ugs % NB;
                    if (lane == 0) LT_CWAIT(mbar0 + b * 8, (ugs / NB) & 1);
                    __syncwarp();
                    TACC(1);
                    if (warp == 0 && lane == 0 && uu == 0) LT_ST(g + s / CPS, 4);
                    tc_fence_after();
                    const uint32_t ta = tmem + b * NTOK + taddr_warp;
                    const int srow = (UPS == 1) ? s : (2 * s + uu);
                    const float* sp = RING ? &S.sclr[(g + s / CPS) % NSTAGE][uu][((d.p0 + d.tg) & 3) + col0] : &S.scl[q][srow][col0];
                    if (col0 < d.ntok) {   // warps whose columns are all padding skip the promotion
                    uint32_t r[CPW];
#pragma unroll
                    for (int c = 0; c < CPW; c += 16) tmem_ld_x16(ta + c, r + c);
                    tmem_ld_wait();
#pragma unroll
                    for (int n = 0; n < CPW; n += 4) {
                        float4 sc;
                        if constexpr (RING) { sc.x = sp[n]; sc.y = sp[n + 1]; sc.z = sp[n + 2]; sc.w = sp[n + 3]; }   // ring row may be unaligned
                        else sc = *reinterpret_cast<const float4*>(sp + n);
                        acc[n + 0] += __uint_as_float(r[n + 0]) * (swk * sc.x);
                        acc[n + 1] += __uint_as_float(r[n + 1]) * (swk * sc.y);
                        acc[n + 2] += __uint_as_float(r[n + 2]) * (swk * sc.z);
                        acc[n + 3] += __uint_as_float(r[n + 3]) * (swk * sc.w);
                    }
                    }
                    tc_fence_before();
                    if (NEED_TFREE) { __syncwarp(); if (lane == 0) mbar_arrive_relaxed(tbar0 + b * 8); }
                    TACC(2);
                    if (warp == 0 && lane == 0) LT_ST(g + s / CPS, 5);
                }
                if ((s % CPS) == CPS - 1) {
                    __syncwarp();
                    if (lane == 0) mbar_arrive_relaxed(wfbar0 + ((g + s / CPS) % NSTAGE) * 8);    // stage consumed: slot reusable
                }
                TACC(3);
                if (warp == 0 && lane == 0) LT_ST(g + s / CPS, 6);
            }
            g += d.nkb / CPS;
            ug += (uint32_t)(d.nkb * UPS);
            // ---------------- epilogue ----------------
            if (d.kind == 0 && p.lists_global) {
                static_assert(NTEAM == 1 || NCW == 4 || true, "");
                // full-K tile: SwiGLU + fp8 requant here (requires NTEAM == 1). gate rows: warps with (warp&3) < 2; up rows: (warp&3) >= 2.
                const bool is_up = (warp & 3) >= 2;
                const int gw = (warp & 3) & 1;                       // gate-warp index within the quad (0,1)
                const int xq = ((warp >> 2) << 1) | gw;              // exchange buffer index: (column half, gate warp)
#pragma unroll
                for (int r0 = 0; r0 < CPW; r0 += XC) {
                    if (is_up) {
#pragma unroll
                        for (int n = 0; n < XC; ++n) S.xbuf[xq][lane][n] = acc[r0 + n];
                    }
                    named_bar_sync(1, CONS_THREADS);
                    if (r0 == 0) TACC(6);
                    if (!is_up) {
                        float c[XC];
                        float cm[XC];
#pragma unroll
                        for (int n = 0; n < XC; ++n) {
                            const float u = S.xbuf[xq][lane][n];
                            c[n] = (u / (1.0f + __expf(-u))) * acc[r0 + n];
                            cm[n] = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(fabsf(c[n]))));
                        }
                        if (lane == 0) {
#pragma unroll
                            for (int n = 0; n < XC; ++n) S.cmax[xq][n] = cm[n];
                        }
                        named_bar_sync(2, CONS_THREADS / 2);          // the 4 gate warps
                        TACC(5);
                        const int rowb = (warp & 3) * 32 + lane;      // gate row within 64
#pragma unroll
                        for (int n = 0; n < XC; ++n) {
                            const int col = col0 + r0 + n;
                            const float amax = fmaxf(S.cmax[xq & ~1][n], S.cmax[xq | 1][n]);
                            const float inv = amax > 0.f ? 448.f / amax : 0.f;
#ifdef MOE_NOEPISTORE
                            if (col < d.ntok && inv == 12345.f) {
#else
                            if (col < d.ntok) {
#endif
                                const size_t pair = (size_t)(d.p0 + d.tg + col);
                                p.Cbuf[pair * I + d.n0 + rowb] = pack_e4m3x2(c[n] * inv, 0.f) & 0xFF;
                            }
                        }
                        if ((warp & 3) == 0 && lane < XC) {   // per-column scales, one coalesced row per round
                            const int col = col0 + r0 + lane;
                            const float amax = fmaxf(S.cmax[xq & ~1][lane], S.cmax[xq | 1][lane]);
                            if (col < d.ntok) p.Cs_t[(size_t)(d.n0 / CS_BLOCK) * (size_t)(T * TOP_K) + (size_t)(d.p0 + d.tg + col)] = amax / 448.f;
                        }
                    }
                    TACC(7);
                    named_bar_sync(1, CONS_THREADS);
                    TACC(3);
                }
            } else if (d.kind == 0) {
                const int half = (myrow >> 6) & 1;
                const int wrow = half * I + d.n0 + (myrow & 63);
#pragma unroll
                for (int n = 0; n < CPW; ++n) {
                    if (col0 + n < d.ntok) red_add_f32(p.P + (size_t)(d.p0 + d.tg + col0 + n) * N1 + wrow, acc[n]);
                }
            } else if (!p.lists_global) {
                const int orow = d.n0 + myrow;
#pragma unroll
                for (int n = 0; n < CPW; ++n) {
                    if (col0 + n < d.ntok) {
                        const int tok = tok_list[d.p0 + d.tg + col0 + n];
                        red_add_f32(p.out_acc + (size_t)tok * H + orow, acc[n] * S.tile_w[q][col0 + n]);
                    }
                }
            } else {
                const int orow = d.n0 + myrow;
#pragma unroll
                for (int n = 0; n < CPW; ++n) {
#ifdef MOE_NOEPISTORE
                    if (col0 + n < d.ntok && acc[n] == 12345.f)
#else
                    if (col0 + n < d.ntok)
#endif
                        p.obuf[(size_t)(d.p0 + d.tg + col0 + n) * H + orow] = __float2bfloat16(acc[n] * S.tile_w[q][col0 + n]);
                }
            }
            if (d.kind == 0) TACC(5); else TACC(7);
            __syncwarp();
#ifdef MOE_CONS_FENCE
            if (lane == 0) { __threadfence(); mbar_arrive(drbar0 + q * 8); }   // descriptor slot consumed (one arrival per warp)
#else
            // stores are made visible by the epilogue warp's gpu-scope fence after it acquires this arrival
            if (p.lists_global) fence_proxy_async_global();
            if (lane == 0) mbar_arrive(drbar0 + q * 8);                        // descriptor slot consumed (one arrival per warp)
#endif
            TACC(4);
            ++qi;
        }
        if (warp == 0) TACC_STORE(0);
    }
    // ---------------- self-cleaning counters, TMEM release ----------------
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int old = atomicAdd(&p.counters[CNT_FINISHED], 1);
        S.flag = (old == (int)gridDim.x - 1);
    }
    __syncthreads();
    if (S.flag) {
        for (int i = tid; i < CNT_TOTAL; i += NTHREADS) p.counters[i] = 0;
        if (p.lists_global && tid < 64) p.g_rzero[tid] = 0;
        __threadfence();
    }
    if (warp == 0) {
        tc_fence_before();
        tmem_dealloc(S.tmem_base, NB * NTOK);
    }
}

template <int NTOK, int NCW, int SCL_ROWS, int NSTAGE, int QN, int NBUF = 512 / NTOK, int MINB = 1, int CPS = 1, int NTEAM = 1>
constexpr size_t small_smem_bytes() { return sizeof(SmallSmem<NTOK, NCW, SCL_ROWS, NSTAGE, QN, NBUF, MINB, CPS, NTEAM>) + 1024; }

// ---------------------------------------------------------------------------------------------
// Large-T helper kernels: gather activation rows by pair order, and combine per-pair outputs per token.
__global__ void gather_a_kernel(const fp8_t* __restrict__ A, const int* __restrict__ tok_list, const int* __restrict__ rcount, fp8_t* __restrict__ Ag,
                                const float* __restrict__ A_scale, int T, float* __restrict__ As_g) {
    // one warp per pair row: 7168 B = 448 x 16 B
    __shared__ int s_npairs;
    if (threadIdx.x < 32) {
        int c = __ldcg(rcount + threadIdx.x);
        c = __reduce_add_sync(0xffffffffu, c);
        if (threadIdx.x == 0) s_npairs = c;
    }
    __syncthreads();
    const int npairs = s_npairs;
    const int pr = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (pr >= npairs) return;
    const int tok = tok_list[pr];
    const uint4* src = reinterpret_cast<const uint4*>(A + (size_t)tok * H);
    uint4* dst = reinterpret_cast<uint4*>(Ag + (size_t)pr * H);
    for (int i = lane; i < H / 16; i += 32) dst[i] = src[i];
    const size_t rs = (size_t)T * TOP_K;
    for (int kb = lane; kb < KB1; kb += 32) As_g[(size_t)kb * rs + pr] = A_scale[(size_t)kb * T + tok];
}

// out[t] = sum over the token's local pairs of obuf[pair]  (bf16 rows, fp32 accumulate); zero if none
__global__ void combine_kernel(const bf16_t* __restrict__ obuf, const int* __restrict__ tok_pairs /*[T][8], -1 = none*/,
                               int T, bf16_t* __restrict__ out) {
    const int t = blockIdx.x * 4 + (threadIdx.x >> 6), sub = threadIdx.x & 63;   // 64 threads per token row
    if (t >= T) return;
    int prs[TOP_K];
    int np = 0;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) { const int pr = tok_pairs[t * TOP_K + k]; if (pr >= 0) prs[np++] = pr; }
    const int c0 = sub * (H / 64);          // 112 contiguous columns per thread = 14 x 8 bf16
    for (int c = c0; c < c0 + H / 64; c += 8) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int k = 0; k < np; ++k) {
            const uint4 v = *reinterpret_cast<const uint4*>(obuf + (size_t)prs[k] * H + c);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b2[j]); acc[2 * j] += f.x; acc[2 * j + 1] += f.y; }
        }
        uint4 o;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) o2[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        *reinterpret_cast<uint4*>(out + (size_t)t * H + c) = o;
    }
}

}  // namespace moe
