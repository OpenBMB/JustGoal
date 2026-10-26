// Persistent fused MoE kernel for small token counts (T <= TMAX_SMALL).
// One launch does: routing (redundantly per CTA) -> GEMM1 (split-K tiles, fp32 partials via red.add; the last
// finisher applies SwiGLU + fp8 requant of C) -> GEMM2 (split-K tiles, weighted red.add into fp32 scratch; the
// last finisher of an n-slice converts to bf16).
// Streaming core: weights arrive via 2D TMA (128B-swizzled [128 rows][128 B] stages, mbarrier pipeline);
// activations via a small cp.async ring; fragments via ldmatrix; fp8 -> fp16 conversion once per chunk;
// mma.sync.m16n8k16.f16 with fp32 accumulate and block-scale promotion.
#pragma once
#include <cuda.h>
#include "moe_common.cuh"

namespace moe3 {
using namespace moe;

constexpr int TMAX_SMALL = 128;
#ifndef V3_MT
#define V3_MT 2
#endif
constexpr int MT = V3_MT;               // 16-row m-subtiles per compute warp (rows per warp = 16*MT)
constexpr int NCOMP = 8 / MT;           // compute warps (128 rows per stage)
constexpr int FEEDER_WARP = NCOMP;      // TMA + activation copies
constexpr int SCHED_WARP = NCOMP + 1;   // tile scheduler / epilogues
constexpr int SMALL_THREADS = (NCOMP + 2) * 32;
#ifndef V3_WSTAGES
#define V3_WSTAGES 5
#endif
constexpr int WSTAGES = V3_WSTAGES;     // weight TMA ring depth
constexpr int WSTAGE_BYTES = 128 * 128; // 128 rows x 128 B
constexpr int ASTAGES = WSTAGES;        // activation ring shares the weight ring's slot index
#ifdef V3_NONG
#define V3_NG1 false
#else
#define V3_NG1 true
#endif
#if defined(V3_FSPIN)
#define V3_FWAIT mbar_wait_spin
#elif defined(V3_FTW)
#define V3_FWAIT mbar_wait_tw
#else
#define V3_FWAIT mbar_wait
#endif
#if defined(V3_SHINT)
#define V3_SWAIT mbar_wait
#elif defined(V3_SSPIN)
#define V3_SWAIT mbar_wait_spin
#else
#define V3_SWAIT mbar_wait_tw
#endif
#if defined(V3_SPINWAIT)
#define V3_WAIT mbar_wait_spin
#elif defined(V3_NSWAIT)
#define V3_WAIT mbar_wait_ns
#elif defined(V3_TWWAIT)
#define V3_WAIT mbar_wait_tw
#else
#define V3_WAIT mbar_wait
#endif
constexpr int ACT_LAG = 2;               // activation cp.async groups kept in flight before a stage is published
constexpr int G1_ROWS = 64;             // gate rows per GEMM1 tile (plus 64 matching up rows)
constexpr int TILES1_PER_E = I / G1_ROWS;      // 32
constexpr int G2_ROWS = 128;            // rows per GEMM2 tile
constexpr int TILES2_PER_E = H / G2_ROWS;      // 56
constexpr int CS_BLOCK = 32;            // C (intermediate) scale block along I
constexpr int CS_PER_ROW = I / CS_BLOCK;       // 64

// counters layout (all zero on entry and on exit)
constexpr int CNT_FINISHED = 0;
constexpr int CNT_WORK = 1;
constexpr int CNT_DONE1 = 2;                        // [2, 34): completed GEMM1 (e, nt) epilogues per expert
constexpr int CNT_DONE2 = CNT_DONE1 + E_LOCAL;      // [34, 90): completed GEMM2 tiles per n-tile
constexpr int CNT_SPLIT1 = CNT_DONE2 + TILES2_PER_E;  // [90, 90+1024): completed K-splits per (e, nt)
constexpr int CNT_ROUTE = CNT_SPLIT1 + E_LOCAL * TILES1_PER_E;  // grid barrier after distributed routing
constexpr int CNT_TOTAL = CNT_ROUTE + 1;

struct SmallParams {
    CUtensorMap tmap_w13;     // [32*4096 rows][7168 B], box [64][128]
    CUtensorMap tmap_w2;      // [32*7168 rows][2048 B], box [128][128]
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
    long long* trace;         // optional per-CTA timeline (globaltimer ns), 64 slots per CTA
    float* P;                 // [T*8][N1] fp32 GEMM1 partials (zero on entry/exit)
    fp8_t* Cbuf;              // [T*8][I]
    float* Cs;                // [T*8][64]
    float* out_acc;           // [T][H] fp32 scratch (zero on entry and exit)
    bf16_t* out;              // [T][H]
    int* counters;            // [CNT_TOTAL]
    uint8_t* r_ids;           // [T*8] routed expert ids (global exchange between CTAs)
    float* r_w;               // [T*8] routed weights
};

constexpr int ROUTE_CHUNKS = TMAX_SMALL * TOP_K / 32;
struct RoutingScratch {           // aliased into wstage[] (only used before streaming starts)
    float bias[E_GLOBAL];
    uint8_t ids[TMAX_SMALL][TOP_K];
    float w[TMAX_SMALL][TOP_K];
    int chist[ROUTE_CHUNKS][E_LOCAL];   // per 32-pair chunk, local pairs per expert
};

struct UnitDesc {                  // one work unit = (tile, 16-token group)
    int kind;                      // 0 = GEMM1, 1 = GEMM2, -1 = exit
    int le, cnt, p0, n0, nt, kb0, nkb, tg, ntok, ng, y0, y1, last, dep_ready;
    const float* sw;
};

template <int NTOK>
struct SmallSmem {
    alignas(1024) uint8_t wstage[WSTAGES][WSTAGE_BYTES];
    alignas(1024) uint8_t aring[ASTAGES][NTOK * 128];
    alignas(8) uint64_t full_bar[WSTAGES];     // TMA bytes + activation arrive (count 2)
    alignas(8) uint64_t free_bar[WSTAGES];     // compute warps done with the slot (count NCOMP)
    int count[E_LOCAL], start[E_LOCAL], fill[E_LOCAL], aexp[E_LOCAL];
    int na, flag;
    uint16_t tok_list[TMAX_SMALL * TOP_K];
    float w_list[TMAX_SMALL * TOP_K];
    alignas(16) float scl2[2][CS_PER_ROW][NTOK];       // per descriptor slot: GEMM1 ascale[56][NTOK] / GEMM2 cscale[64][NTOK]
    int dtok[2][NTOK];                                 // activation rows (GEMM1: token ids, GEMM2: pair rows of Cbuf)
    float dw[2][NTOK];                                 // GEMM2: routing weights per column
    UnitDesc desc[2];
    alignas(8) uint64_t dfull_bar[2];                  // scheduler -> feeder + compute (count 1)
    alignas(8) uint64_t dfree_bar[2];                  // feeder + compute warps done with the slot (count NCOMP+1)
    alignas(8) uint64_t fhalf_bar[2];                  // feeder has issued half of the unit's stages (count 1): grab the next tile
};

// ---------------------------------------------------------------------------------------------
// One tile's K loop. Each warp owns 16 rows of the weight stage (rows 16*warp .. +15).
//  GEMM1 (SCL32=false): 2 TMA boxes per stage (gate rows -> stage rows 0..63, up rows -> 64..127),
//                       promotion once per 128-chunk with sa[kb][tok] * sw[kb].
//  GEMM2 (SCL32=true):  1 box per stage; promotion every 32 k (2 mma steps) with sc[kb*4+i][tok] * sw[kb].
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

template <int NGRP, int NKB, bool SCL32, int NG = NGRP>
__device__ __forceinline__ void tile_k_loop(
    SmallSmem<NGRP * 8>& S, const CUtensorMap* tmap, int y0, int y1 /*row coords of the 2 boxes (y1<0: single 128-row box)*/,
    int kb_begin, const float* __restrict__ sw,
    const fp8_t* __restrict__ act, int act_stride, int ntok, int ng,
    uint32_t& it /*running stage counter*/, int tid, int warp, int lane,
    float (&acc)[MT][NGRP][4], long long* p_trace, const int* __restrict__ dtok, uint32_t sclb,
    const int* dep = nullptr, int dep_target = 0, const float* cs_src = nullptr, uint32_t fhalf = 0) {
    constexpr int NTOK = NGRP * 8;
    constexpr int ASTAGE_BYTES = NTOK * 128;
    const uint32_t wbase = smem_u32(&S.wstage[0][0]);
    const uint32_t abase = smem_u32(&S.aring[0][0]);
    const uint32_t fbar0 = smem_u32(&S.full_bar[0]);
    const uint32_t rbar0 = smem_u32(&S.free_bar[0]);

    if (warp == FEEDER_WARP) {
        // =============== feeder warp: weight TMA + activation cp.async per stage, no CTA barriers ===============
        auto issue_tma = [&](int j) {
            const uint32_t g = it + j;
            const uint32_t slot = g % WSTAGES;
            if (g >= WSTAGES) { if (lane == 0) V3_FWAIT(rbar0 + slot * 8, ((g / WSTAGES) - 1) & 1); __syncwarp(); }
#ifdef MOE_TRACE
#define V3ST(k, val) do { if (p_trace && lane == 0 && g < 64) p_trace[20000 + g * 8 + (k)] = (val); } while (0)
#else
#define V3ST(k, val) do { } while (0)
#endif
            V3ST(0, gtimer());
            if (lane == 0) {
#ifdef V3_NOTMA
                mbar_arrive(fbar0 + slot * 8);
#else
                const uint32_t dst = wbase + slot * WSTAGE_BYTES;
                mbar_expect_tx(fbar0 + slot * 8, WSTAGE_BYTES);
                const int x = (kb_begin + j) * 128;
                if (y1 < 0) {
                    tma_load_2d(dst, tmap, x, y0, fbar0 + slot * 8);
                } else {
                    tma_load_2d(dst, tmap, x, y0, fbar0 + slot * 8);
                    tma_load_2d(dst + WSTAGE_BYTES / 2, tmap, x, y1, fbar0 + slot * 8);
                }
#endif
            }
            V3ST(1, gtimer());
        };
        // weights first: they do not depend on the previous GEMM, so the ring fills while the dependency resolves
        // Only when this unit still waits for the previous GEMM: prefetch the weights first (they do not depend on
        // it), then wait, then gather the scales. Otherwise keep the interleaved order so that the first
        // activation copy is not held back by the last slots of the previous unit.
        const int PRE = dep ? (NKB < WSTAGES ? NKB : WSTAGES) : 0;
        if (dep) {
            for (int j = 0; j < PRE; ++j) issue_tma(j);
            if (lane == 0) { while (ld_acquire_gpu(dep) < dep_target) __nanosleep(64); }
            __syncwarp();
            // GEMM2 scale gather: scl[kb0*4 + k][t] = Cs[pair t][kb0*4 + k]; up to 32 loads in flight per lane
            const int nk = NKB * 4, n = nk * ntok;
            for (int i0 = 0; i0 < n; i0 += 32 * 32) {
                float v[32];
#pragma unroll
                for (int k = 0; k < 32; ++k) {
                    const int i = i0 + k * 32 + lane;
                    v[k] = (i < n) ? __ldcg(cs_src + (size_t)(i / nk) * CS_PER_ROW + kb_begin * 4 + i % nk) : 0.f;
                }
#pragma unroll
                for (int k = 0; k < 32; ++k) {
                    const int i = i0 + k * 32 + lane;
                    if (i < n) sts_f32(sclb + ((kb_begin * 4 + i % nk) * NTOK + i / nk) * 4, v[k]);
                }
            }
            __syncwarp();
        }
        for (int j = 0; j < NKB; ++j) {
            const uint32_t g = it + j;
            const uint32_t slot = g % WSTAGES;
            if (j >= PRE) issue_tma(j);
            if (j == NKB / 2 && lane == 0) mbar_arrive(fhalf);
#ifndef V3_NOACT
            {
                const int kb = kb_begin + j;
                for (int c = lane; c < ntok * 8; c += 32) {
                    const int t = c >> 3, ch = c & 7;
                    const int row = dtok[t];
                    cp_async_16(abase + slot * ASTAGE_BYTES + swz128(t, ch), act + (size_t)row * act_stride + kb * 128 + ch * 16, 16);
                }
                cp_async_commit();
            }
#endif
            V3ST(2, gtimer());
            if (j >= ACT_LAG) {      // activation copies of stage j-ACT_LAG have landed: publish that stage
                cp_async_wait<ACT_LAG>();
                __syncwarp();
                if (lane == 0) mbar_arrive(fbar0 + ((g - ACT_LAG) % WSTAGES) * 8);
            }
            V3ST(3, gtimer());
        }
        cp_async_wait<0>();
        __syncwarp();
        if (lane == 0)
            for (int k = (NKB > ACT_LAG ? NKB - ACT_LAG : 0); k < NKB; ++k) mbar_arrive(fbar0 + ((it + k) % WSTAGES) * 8);
    } else {
        // =============== compute warps (16*MT weight rows each) ===============
        const int arow0 = warp * (16 * MT) + (lane & 7) + ((lane >> 3) & 1) * 8;   // A: row within stage (m-subtile 0)
        const int achunk = (lane >> 4);                                      // A: + chunk offset (0/1)
        const int brow = (lane & 7);                                         // B: token row within group
        const int bchunk = (lane >> 3);                                      // B: chunk offset (0..3)
        const int tcol = (lane & 3) * 2;
        uint32_t slot = it % WSTAGES, phase = (it / WSTAGES) & 1;
#pragma unroll 1
        for (int j = 0; j < NKB; ++j) {
            const uint32_t g = it + j;
            if (lane == 0) V3_WAIT(fbar0 + slot * 8, phase);
            __syncwarp();
            if (warp == 0) V3ST(4, gtimer());
            if (warp == NCOMP - 1) V3ST(6, gtimer());
            const uint32_t ws = wbase + slot * WSTAGE_BYTES;
            const uint32_t as = abase + slot * ASTAGE_BYTES;
            const float swk = sw[kb_begin + j];
#ifdef V3_NOCOMPUTE
            if (true) { __syncwarp(); if (lane == 0) mbar_arrive_relaxed(rbar0 + slot * 8); continue; }
#endif
            float part[MT][NGRP][4];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int gg = 0; gg < NGRP; ++gg) { part[m][gg][0] = part[m][gg][1] = part[m][gg][2] = part[m][gg][3] = 0.f; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {          // 4 k16 steps per half
                uint32_t a[MT][4][4];              // [m][step][a0..a3] fp16x2
#pragma unroll
                for (int m = 0; m < MT; ++m) {
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        uint32_t r0, r1, r2, r3;
                        ldsm_x4(ws + swz128(arow0 + m * 16, 4 * h + 2 * cc + achunk), r0, r1, r2, r3);
                        a[m][2 * cc][0] = cvt_e4m3x2_lo(r0); a[m][2 * cc][1] = cvt_e4m3x2_lo(r1);
                        a[m][2 * cc][2] = cvt_e4m3x2_hi(r0); a[m][2 * cc][3] = cvt_e4m3x2_hi(r1);
                        a[m][2 * cc + 1][0] = cvt_e4m3x2_lo(r2); a[m][2 * cc + 1][1] = cvt_e4m3x2_lo(r3);
                        a[m][2 * cc + 1][2] = cvt_e4m3x2_hi(r2); a[m][2 * cc + 1][3] = cvt_e4m3x2_hi(r3);
                    }
                }
#pragma unroll
                for (int gg = 0; gg < NG; ++gg) {
                    uint32_t r[4], bl[4], bh[4];
                    ldsm_x4(as + swz128(gg * 8 + brow, 4 * h + bchunk), r[0], r[1], r[2], r[3]);
#pragma unroll
                    for (int s = 0; s < 4; ++s) { bl[s] = cvt_e4m3x2_lo(r[s]); bh[s] = cvt_e4m3x2_hi(r[s]); }   // B converted once per warp
                    if constexpr (!SCL32) {
#pragma unroll
                        for (int m = 0; m < MT; ++m)
#pragma unroll
                            for (int s = 0; s < 4; ++s)
                                mma_f16(part[m][gg], a[m][s][0], a[m][s][1], a[m][s][2], a[m][s][3], bl[s], bh[s]);
                    } else {
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const float2 sc = lds_f2(sclb + (((kb_begin + j) * 4 + 2 * h + i) * NTOK + gg * 8 + tcol) * 4);
                            const float sx = swk * sc.x, sy = swk * sc.y;
#pragma unroll
                            for (int m = 0; m < MT; ++m) {
                                float pp[4];
                                mma_f16_zero(pp, a[m][2 * i][0], a[m][2 * i][1], a[m][2 * i][2], a[m][2 * i][3], bl[2 * i], bh[2 * i]);
                                mma_f16(pp, a[m][2 * i + 1][0], a[m][2 * i + 1][1], a[m][2 * i + 1][2], a[m][2 * i + 1][3], bl[2 * i + 1], bh[2 * i + 1]);
                                acc[m][gg][0] += pp[0] * sx;
                                acc[m][gg][1] += pp[1] * sy;
                                acc[m][gg][2] += pp[2] * sx;
                                acc[m][gg][3] += pp[3] * sy;
                            }
                        }
                    }
                }
            }
            if constexpr (!SCL32) {
#pragma unroll
                for (int gg = 0; gg < NG; ++gg) {
                    const float2 sa = lds_f2(sclb + ((kb_begin + j) * NTOK + gg * 8 + tcol) * 4);
                    const float sx = swk * sa.x, sy = swk * sa.y;
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        acc[m][gg][0] += part[m][gg][0] * sx;
                        acc[m][gg][1] += part[m][gg][1] * sy;
                        acc[m][gg][2] += part[m][gg][2] * sx;
                        acc[m][gg][3] += part[m][gg][3] * sy;
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive_relaxed(rbar0 + slot * 8);     // slot consumed by this warp
            if (warp == 0) V3ST(5, gtimer());
            if (warp == NCOMP - 1) V3ST(7, gtimer());
            if (++slot == WSTAGES) { slot = 0; phase ^= 1u; }
        }
    }
    it += NKB;
}

// ---------------------------------------------------------------------------------------------

#ifdef MOE_TRACE
#define V3T(idx, val) do { if (p.trace && tid == 0 && blockIdx.x < 296) p.trace[blockIdx.x * 64 + (idx)] = (val); } while (0)
#else
#define V3T(idx, val) do { } while (0)
#endif
template <int NGRP>
__global__ void __launch_bounds__(SMALL_THREADS, 2) moe_small_kernel(const __grid_constant__ SmallParams p) {
    constexpr int NTOK = NGRP * 8;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    SmallSmem<NTOK>& S = *reinterpret_cast<SmallSmem<NTOK>*>(
        reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023)));
    RoutingScratch& R = *reinterpret_cast<RoutingScratch*>(&S.wstage[0][0]);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int T = p.T;

    if (tid == 0) {
        for (int s = 0; s < WSTAGES; ++s) { mbar_init(smem_u32(&S.full_bar[s]), 2); mbar_init(smem_u32(&S.free_bar[s]), NCOMP); }
        for (int s = 0; s < 2; ++s) { mbar_init(smem_u32(&S.dfull_bar[s]), 1); mbar_init(smem_u32(&S.dfree_bar[s]), NCOMP + 1); mbar_init(smem_u32(&S.fhalf_bar[s]), 1); }
        fence_proxy_async();
        tma_prefetch_desc(&p.tmap_w13);
        tma_prefetch_desc(&p.tmap_w2);
    }
    V3T(0, gtimer());
    // ---------------- routing: distributed over the grid (token t -> warp t / grid, CTA t % grid) ----------------
    const int nch = (T * TOP_K + 31) / 32;
    for (int i = tid; i < E_GLOBAL; i += SMALL_THREADS) R.bias[i] = __bfloat162float(p.bias[i]);
    for (int i = tid; i < nch * E_LOCAL; i += SMALL_THREADS) (&R.chist[0][0])[i] = 0;
    if (tid < E_LOCAL) { S.count[tid] = 0; S.fill[tid] = 0; }
    __syncthreads();
    {
        const int t = warp * (int)gridDim.x + (int)blockIdx.x;
        if (t < T) {
            int id; float w;
            route_token_warp(p.logits + (size_t)t * E_GLOBAL, R.bias, p.scaling, id, w);
            if (lane < TOP_K) { p.r_ids[t * TOP_K + lane] = (uint8_t)id; p.r_w[t * TOP_K + lane] = w; }
        }
    }
    __syncthreads();
    V3T(44, gtimer());
    if (tid == 0) {   // grid barrier (all CTAs are co-resident: grid = occupancy x SMs)
        __threadfence();
        atomicAdd(&p.counters[CNT_ROUTE], 1);
        while (ld_acquire_gpu(&p.counters[CNT_ROUTE]) < (int)gridDim.x) __nanosleep(64);
    }
    __syncthreads();
    V3T(45, gtimer());
    for (int i = tid; i < T * TOP_K; i += SMALL_THREADS) {
        R.ids[i / TOP_K][i % TOP_K] = __ldcg(p.r_ids + i);
        R.w[i / TOP_K][i % TOP_K] = __ldcg(p.r_w + i);
    }
    __syncthreads();
    // expert-grouped pair lists in pair-index order (every CTA must build the identical list):
    // pass 1: per 32-pair chunk histogram of local experts (match_any ranks equal-expert lanes)
    for (int c = warp; c < nch; c += SMALL_THREADS / 32) {
        const int i = c * 32 + lane;
        int le = -1;
        if (i < T * TOP_K) { le = (int)R.ids[i / TOP_K][i % TOP_K] - p.local_offset; if (le < 0 || le >= E_LOCAL) le = -1; }
        const unsigned m = __match_any_sync(0xffffffffu, le);
        if (le >= 0 && (m & ((1u << lane) - 1)) == 0) R.chist[c][le] = __popc(m);
    }
    __syncthreads();
    if (warp == 0) {
        int c = 0;
        for (int k = 0; k < nch; ++k) { const int h = R.chist[k][lane]; R.chist[k][lane] = c; c += h; }   // in place: exclusive prefix over chunks
        S.count[lane] = c;
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
    }
    __syncthreads();
    // pass 2: position = start[le] + pairs of le in earlier chunks + rank within the chunk
    for (int c = warp; c < nch; c += SMALL_THREADS / 32) {
        const int i = c * 32 + lane;
        int le = -1;
        if (i < T * TOP_K) { le = (int)R.ids[i / TOP_K][i % TOP_K] - p.local_offset; if (le < 0 || le >= E_LOCAL) le = -1; }
        const unsigned m = __match_any_sync(0xffffffffu, le);
        if (le >= 0) {
            const int off = S.start[le] + R.chist[c][le] + __popc(m & ((1u << lane) - 1));
            S.tok_list[off] = (uint16_t)(i / TOP_K);
            S.w_list[off] = R.w[i / TOP_K][i % TOP_K];
        }
    }
    __syncthreads();
    const int na = S.na;
    V3T(1, gtimer());
    // split-K policy from the number of active experts (few experts -> split for balance; many -> whole-K tiles)
    // split K until there are >= ~2 units per CTA (units are prefetched across boundaries, so small units are cheap)
    // GEMM1: ~4 units per CTA (measured sweet spot: deeper K splits balance the long GEMM1 phase);
    // GEMM2: ~2 units per CTA (its units are short already)
    const int want1 = 4 * (int)gridDim.x, want2 = 2 * (int)gridDim.x;
    const int ks1_max = na <= 4 ? 8 : 4;   // 7-stage units only pay off for very few experts
    int ks1 = 1; while (ks1 < ks1_max && na * TILES1_PER_E * ks1 < want1) ks1 <<= 1;
    int ks2 = 1; while (ks2 < 4 && na * TILES2_PER_E * ks2 < want2) ks2 <<= 1;
    const int KS1 = p.ks1 > 0 ? p.ks1 : ks1;
    const int KS2 = p.ks2 > 0 ? p.ks2 : ks2;
    const int tiles1_e = TILES1_PER_E * KS1;
    const int tiles2_e = TILES2_PER_E * KS2;
    const int n1 = na * tiles1_e, n2 = na * tiles2_e;

    if (na == 0) {
        uint4 z = make_uint4(0, 0, 0, 0);
        uint4* o = reinterpret_cast<uint4*>(p.out);
        const int n = T * H / 8;
        for (int i = blockIdx.x * SMALL_THREADS + tid; i < n; i += gridDim.x * SMALL_THREADS) o[i] = z;
    }

    uint32_t it = 0;   // running weight-stage counter (uniform within the feeder / compute roles)
    uint32_t unit = 0; // running work-unit counter; descriptor slot = unit & 1
#define TRP ((p.trace && blockIdx.x == 0) ? p.trace : nullptr)
    if (warp == SCHED_WARP) {
        // ======================= scheduler warp =======================
        // grabs tiles, resolves the GEMM1->GEMM2 dependency, gathers per-unit scales, publishes unit descriptors,
        // and runs the tile epilogues (split-K completion -> SwiGLU/requant, GEMM2 completion -> finalize unit)
        // off the compute warps' critical path.
        uint32_t reclaimed = 0;           // units whose completion has been consumed (in order)
        uint64_t pending_fin = 0;         // n-tiles whose fp32 accumulator is complete and must be converted
        auto wait_slot = [&](uint32_t u) -> UnitDesc {   // wait until unit u is fully consumed; returns its descriptor
            const int dd = u & 1;
            if (lane == 0) V3_SWAIT(smem_u32(&S.dfree_bar[dd]), (u >> 1) & 1);
            __syncwarp();
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296) { if (u < 8) p.trace[blockIdx.x * 64 + 16 + u] = gtimer(); p.trace[blockIdx.x * 64 + 51] = gtimer(); p.trace[blockIdx.x * 64 + 50] = u; }
#endif
            return S.desc[dd];
        };
        auto epilogue = [&](const UnitDesc& ud, uint32_t u) {
            if (ud.kind == 0 && ud.last) {
                int flag = 0;
#ifdef MOE_TRACE
#define RTR(k) do { if (p.trace && lane == 0 && blockIdx.x < 296 && u == 0) p.trace[blockIdx.x * 64 + 56 + (k)] = gtimer(); } while (0)
#else
#define RTR(k) do { } while (0)
#endif
                RTR(0);
                if (lane == 0) {
                    const int old = atom_add_acq_rel_gpu(&p.counters[CNT_SPLIT1 + ud.le * TILES1_PER_E + ud.nt], 1);
                    flag = (old == KS1 - 1);
                }
                flag = __shfl_sync(0xffffffffu, flag, 0);
                RTR(1);
                if (flag) {
                    RTR(2);
                    // SwiGLU + fp8 requant for all pairs of this expert, 16 lanes per pair (64 values = 2 scale blocks)
                    const int nitems = ud.cnt * 16;
                    for (int ib = 0; ib < nitems; ib += 32 * 4) {
                        float4 g4[4], u4[4];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int i = ib + k * 32 + lane;
                            g4[k] = make_float4(0.f, 0.f, 0.f, 0.f); u4[k] = g4[k];
                            if (i < nitems) {
                                const size_t pair = (size_t)(ud.p0 + (i >> 4));
                                const float* pg = p.P + pair * N1 + ud.n0 + (i & 15) * 4;
                                asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(g4[k].x), "=f"(g4[k].y), "=f"(g4[k].z), "=f"(g4[k].w) : "l"(pg) : "memory");
                                asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(u4[k].x), "=f"(u4[k].y), "=f"(u4[k].z), "=f"(u4[k].w) : "l"(pg + I) : "memory");
                            }
                        }
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const int i = ib + k * 32 + lane;
                            float c[4] = {silu_f(u4[k].x) * g4[k].x, silu_f(u4[k].y) * g4[k].y, silu_f(u4[k].z) * g4[k].z, silu_f(u4[k].w) * g4[k].w};
                            float amax = fmaxf(fmaxf(fabsf(c[0]), fabsf(c[1])), fmaxf(fabsf(c[2]), fabsf(c[3])));
#pragma unroll
                            for (int off = 1; off < 8; off <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
                            if (i < nitems) {
                                const size_t pair = (size_t)(ud.p0 + (i >> 4));
                                const int n4 = (i & 15) * 4;
                                float* pg = p.P + pair * N1 + ud.n0 + n4;
                                const float scale = amax / 448.f;
                                const float inv = amax > 0.f ? 448.f / amax : 0.f;
                                const uint32_t packed = (uint32_t)pack_e4m3x2(c[0] * inv, c[1] * inv) |
                                                        ((uint32_t)pack_e4m3x2(c[2] * inv, c[3] * inv) << 16);
                                *reinterpret_cast<uint32_t*>(p.Cbuf + pair * I + ud.n0 + n4) = packed;
                                if ((i & 7) == 0) p.Cs[pair * CS_PER_ROW + (ud.n0 + n4) / CS_BLOCK] = scale;
                                asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(pg), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f) : "memory");
                                asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(pg + I), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f) : "memory");
                            }
                        }
                    }
                    __syncwarp();
                    RTR(3);
                    if (lane == 0) {
#ifdef MOE_TRACE
                        const int o1 = atom_add_acq_rel_gpu(&p.counters[CNT_DONE1 + ud.le], 1);
                        RTR(4);
                        if (p.trace && o1 == TILES1_PER_E - 1) p.trace[30000 + ud.le] = gtimer();
                        if (p.trace && blockIdx.x < 296 && u < 8) p.trace[blockIdx.x * 64 + 24 + u] = gtimer();
#else
                        red_add_release_gpu(&p.counters[CNT_DONE1 + ud.le], 1);
#endif
                    }
                }
            } else if (ud.kind == 1 && ud.last) {
                int flag = 0;
#ifdef MOE_TRACE
                if (p.trace && lane == 0 && blockIdx.x < 296 && u + 1 == unit) p.trace[blockIdx.x * 64 + 57] = gtimer();
#endif
                if (lane == 0) {
                    const int old = atom_add_acq_rel_gpu(&p.counters[CNT_DONE2 + ud.nt], 1);
                    flag = (old == na * KS2 - 1);
                }
#ifdef MOE_TRACE
                if (p.trace && lane == 0 && blockIdx.x < 296 && u + 1 == unit) p.trace[blockIdx.x * 64 + 58] = gtimer();
#endif
                flag = __shfl_sync(0xffffffffu, flag, 0);
                if (flag) pending_fin |= (uint64_t)1 << ud.nt;   // finalize as a compute-warp unit (all compute threads)
            }
        };
        // publish unit `unit` into its slot (after the slot's previous unit has been reclaimed)
        auto reclaim = [&](uint32_t u) { const UnitDesc ud = wait_slot(u); epilogue(ud, u); };
        auto publish = [&](int kind, int le, int cnt, int p0, int n0, int nt, int kb0, int nkb, int tg, int y0, int y1, const float* sw) {
            // slot (unit & 1) was used by unit - 2: wait for it, publish first, run its epilogue afterwards (the
            // epilogue is a chain of global round trips that must not hold back the feeder)
            UnitDesc pend; pend.kind = -2;
            uint32_t pend_u = 0;
            while (reclaimed + 1 < unit) {
                if (pend.kind != -2) epilogue(pend, pend_u);
                pend_u = reclaimed++;
                pend = wait_slot(pend_u);
            }
            const int dd = unit & 1;
            const int ntok = kind <= 1 ? min(NTOK, cnt - tg) : 0;
            const int ng = (ntok + 7) >> 3;
            int dep_ready = 1;
#ifdef V3_FEEDDEP
            if (kind == 1) {
                if (lane == 0) dep_ready = ld_acquire_gpu(&p.counters[CNT_DONE1 + le]) >= TILES1_PER_E;
                dep_ready = __shfl_sync(0xffffffffu, dep_ready, 0);
            }
#endif
            if (lane == 0) {
                UnitDesc& u = S.desc[dd];
                u.kind = kind; u.le = le; u.cnt = cnt; u.p0 = p0; u.n0 = n0; u.nt = nt; u.kb0 = kb0; u.nkb = nkb;
                u.tg = tg; u.ntok = ntok; u.ng = ng; u.y0 = y0; u.y1 = y1; u.last = (tg + NTOK >= cnt); u.sw = sw;
                u.dep_ready = dep_ready;
            }
            if (kind <= 1) {
                if (lane < NTOK) {
                    S.dtok[dd][lane] = (lane < ntok) ? (kind == 0 ? (int)S.tok_list[p0 + tg + lane] : (p0 + tg + lane)) : 0;
                    S.dw[dd][lane] = (lane < ntok) ? S.w_list[p0 + tg + lane] : 0.f;
                }
                __syncwarp();
                // scale gather, 8 independent loads in flight per lane
                if (kind == 0) {
                    const int n = nkb * NTOK;
                    for (int i0 = 0; i0 < n; i0 += 32 * 8) {
                        float v[8];
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int i = i0 + k * 32 + lane;
                            const int kb = kb0 + i / NTOK, t = i % NTOK;
                            v[k] = (i < n && t < ntok) ? __ldg(p.A_scale + (size_t)kb * T + S.dtok[dd][t]) : 0.f;
                        }
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int i = i0 + k * 32 + lane;
                            if (i < n) S.scl2[dd][kb0 + i / NTOK][i % NTOK] = v[k];
                        }
                    }
                } else if (dep_ready) {
                    // GEMM1 of this expert already complete: gather here (otherwise the feeder does it after the wait)
                    const int nk = nkb * 4, n = nk * NTOK;      // i -> (t, k32) so that lanes read consecutive Cs entries
                    for (int i0 = 0; i0 < n; i0 += 32 * 8) {
                        float v[8];
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int i = i0 + k * 32 + lane;
                            const int t = i / nk, k32 = kb0 * 4 + i % nk;
                            v[k] = (i < n && t < ntok) ? __ldcg(p.Cs + (size_t)(p0 + tg + t) * CS_PER_ROW + k32) : 0.f;
                        }
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int i = i0 + k * 32 + lane;
                            if (i < n) S.scl2[dd][kb0 * 4 + i % nk][i / nk] = v[k];
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&S.dfull_bar[dd]));
            ++unit;
            if (pend.kind != -2) epilogue(pend, pend_u);
        };
        for (;;) {
#ifndef V3_NOGATE
            // grab the next tile only once the feeder is half-way through the last published unit (tail balance),
            // reclaiming finished units meanwhile
            if (unit >= 1) {
                const uint32_t lastu = unit - 1;
                for (;;) {
                    int ok = 0;
                    if (lane == 0) ok = mbar_test_wait(smem_u32(&S.fhalf_bar[lastu & 1]), (lastu >> 1) & 1);
                    ok = __shfl_sync(0xffffffffu, ok, 0);
                    if (ok) break;
                    if (reclaimed < unit) {
                        int done = 0;
                        if (lane == 0) done = mbar_test_wait(smem_u32(&S.dfree_bar[reclaimed & 1]), (reclaimed >> 1) & 1);
                        done = __shfl_sync(0xffffffffu, done, 0);
                        if (done) { reclaim(reclaimed++); continue; }
                    }
                    __nanosleep(100);
                }
            }
#endif
            while (pending_fin) {
                const int fnt = __ffsll((long long)pending_fin) - 1;
                pending_fin &= ~((uint64_t)1 << fnt);
                publish(2, 0, 0, 0, fnt * G2_ROWS, fnt, 0, 0, 0, 0, -1, nullptr);
            }
            int tile = 0;
            if (lane == 0) tile = atomicAdd(&p.counters[CNT_WORK], 1);
            tile = __shfl_sync(0xffffffffu, tile, 0);
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296 && unit < 14) p.trace[blockIdx.x * 64 + 32 + unit] = gtimer();
#endif
            if (tile >= n1 + n2) break;
            int kind, le, cnt, p0, n0, nt, kb0, nkb, y0, y1;
            const float* sw;
            int ae, rem;
            const bool is_g1 = tile < n1;
            if (is_g1) { ae = tile / tiles1_e; rem = tile % tiles1_e; }
            else { ae = (tile - n1) / tiles2_e; rem = (tile - n1) % tiles2_e; }
            if (is_g1) {
                nt = rem / KS1; const int ks = rem % KS1;
                le = S.aexp[ae]; cnt = S.count[le]; p0 = S.start[le];
                n0 = nt * G1_ROWS;
                sw = p.W13s + ((size_t)le * NB1 + n0 / 128) * KB1;   // gate block-row scales (up: + (I/128) * KB1)
                nkb = KB1 / KS1; kb0 = ks * nkb;
                y0 = le * N1 + n0; y1 = le * N1 + I + n0;
                kind = 0;
            } else {
                nt = rem / KS2; const int ks = rem % KS2;
                le = S.aexp[ae]; cnt = S.count[le]; p0 = S.start[le];
                n0 = nt * G2_ROWS;
                sw = p.W2s + ((size_t)le * NB2 + nt) * KB2;
                nkb = KB2 / KS2; kb0 = ks * nkb;
                y0 = le * H + n0; y1 = -1;
                kind = 1;
                // dependency: all GEMM1 n-tiles of this expert requantized. Keep reclaiming our own units meanwhile
                // (our own SwiGLU epilogues may be what this expert is waiting for).
#ifndef V3_FEEDDEP
                for (;;) {
                    int ok = 0;
                    if (lane == 0) ok = ld_acquire_gpu(&p.counters[CNT_DONE1 + le]) >= TILES1_PER_E;
                    ok = __shfl_sync(0xffffffffu, ok, 0);
                    if (ok) break;
                    if (reclaimed < unit) {
                        int done = 0;
                        if (lane == 0) done = mbar_test_wait(smem_u32(&S.dfree_bar[reclaimed & 1]), (reclaimed >> 1) & 1);
                        done = __shfl_sync(0xffffffffu, done, 0);
                        if (done) { reclaim(reclaimed++); continue; }
                    }
                    __nanosleep(128);
                }
#endif
            }
            for (int tg = 0; tg < cnt; tg += NTOK) publish(kind, le, cnt, p0, n0, nt, kb0, nkb, tg, y0, y1, sw);
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296 && unit < 14) p.trace[blockIdx.x * 64 + 46 + unit] = (gtimer() << 2) | kind;
#endif
        }
        // drain: reclaim everything (epilogues may enqueue finalize units), then publish the exit sentinel
#ifdef MOE_TRACE
        if (p.trace && lane == 0 && blockIdx.x < 296) { p.trace[blockIdx.x * 64 + 62] = gtimer(); p.trace[blockIdx.x * 64 + 61] = 0; }
#endif
        for (;;) {
            while (reclaimed < unit) reclaim(reclaimed++);
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296 && !p.trace[blockIdx.x * 64 + 61]) p.trace[blockIdx.x * 64 + 61] = gtimer();
#endif
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296) p.trace[blockIdx.x * 64 + 52] = gtimer();
#endif
            if (!pending_fin) break;
            while (pending_fin) {
                const int fnt = __ffsll((long long)pending_fin) - 1;
                pending_fin &= ~((uint64_t)1 << fnt);
                publish(2, 0, 0, 0, fnt * G2_ROWS, fnt, 0, 0, 0, 0, -1, nullptr);
            }
        }
        {
            const int dd = unit & 1;
            if (lane == 0) { S.desc[dd].kind = -1; mbar_arrive(smem_u32(&S.dfull_bar[dd])); }
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296) p.trace[blockIdx.x * 64 + 53] = gtimer();
#endif
        }
    } else if (warp == FEEDER_WARP) {
        // ======================= feeder warp: streams weights (TMA) + activations (cp.async) for each unit =======================
        for (;;) {
            const int dd = unit & 1;
            if (lane == 0) V3_SWAIT(smem_u32(&S.dfull_bar[dd]), (unit >> 1) & 1);
            __syncwarp();
            const UnitDesc u = S.desc[dd];
            if (u.kind < 0) break;
            float acc[MT][NGRP][4];
            const uint32_t sclb = smem_u32(&S.scl2[dd][0][0]);
            if (u.kind == 0) {
                if (KS1 == 1)      tile_k_loop<NGRP, 56, false>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, u.sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, nullptr, 0, nullptr, smem_u32(&S.fhalf_bar[dd]));
                else if (KS1 == 2) tile_k_loop<NGRP, 28, false>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, u.sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, nullptr, 0, nullptr, smem_u32(&S.fhalf_bar[dd]));
                else if (KS1 == 4) tile_k_loop<NGRP, 14, false>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, u.sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, nullptr, 0, nullptr, smem_u32(&S.fhalf_bar[dd]));
                else               tile_k_loop<NGRP, 7, false>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, u.sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, nullptr, 0, nullptr, smem_u32(&S.fhalf_bar[dd]));
            } else if (u.kind == 1) {
#ifdef V3_FEEDDEP
                const int* dep = u.dep_ready ? nullptr : (p.counters + CNT_DONE1 + u.le);   // unresolved: the feeder prefetches W2, then waits
#else
                const int* dep = nullptr;   // dependency resolved by the scheduler before publishing (compile-time: no feeder dep path)
#endif
                const float* cs = p.Cs + (size_t)(u.p0 + u.tg) * CS_PER_ROW;
                if (KS2 == 1)      tile_k_loop<NGRP, 16, true>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, dep, TILES1_PER_E, cs, smem_u32(&S.fhalf_bar[dd]));
                else if (KS2 == 2) tile_k_loop<NGRP, 8, true>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, dep, TILES1_PER_E, cs, smem_u32(&S.fhalf_bar[dd]));
                else               tile_k_loop<NGRP, 4, true>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb, dep, TILES1_PER_E, cs, smem_u32(&S.fhalf_bar[dd]));
            }
            __syncwarp();
#ifdef MOE_TRACE
            if (p.trace && lane == 0 && blockIdx.x < 296) p.trace[blockIdx.x * 64 + 45] = gtimer();
#endif
            if (lane == 0) mbar_arrive(smem_u32(&S.dfree_bar[dd]));
            ++unit;
        }
    } else {
        // ======================= compute warps: stream-consume units, fire-and-forget partial sums =======================
        for (;;) {
            const int dd = unit & 1;
            if (lane == 0) V3_SWAIT(smem_u32(&S.dfull_bar[dd]), (unit >> 1) & 1);
            __syncwarp();
            const UnitDesc u = S.desc[dd];
            if (u.kind < 0) break;
            if (unit < 14) V3T(2 + 2 * unit, gtimer());
            const uint32_t sclb = smem_u32(&S.scl2[dd][0][0]);
            float acc[MT][NGRP][4];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int g = 0; g < NGRP; ++g) { acc[m][g][0] = acc[m][g][1] = acc[m][g][2] = acc[m][g][3] = 0.f; }
            const int tcol = (lane & 3) * 2;
            if (u.kind == 0) {
                // ---- GEMM1 unit: (expert, 64 gate + 64 up rows, K-split) x 16 tokens ----
                const int half = (warp * MT) >> 2;                     // gate (0) / up (1) rows: 64 rows each
                const int rbase = half * I + u.n0 + ((warp * MT) & 3) * 16;   // row (within W13[le]) of this warp's first row
                const float* sw = u.sw + half * (I / 128) * KB1;      // gate / up block-row scales
                if (KS1 == 1)      { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 56, false, 1>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 56, false, 2>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                else if (KS1 == 2) { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 28, false, 1>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 28, false, 2>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                else if (KS1 == 4) { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 14, false, 1>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 14, false, 2>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                else               { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 7, false, 1>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 7, false, 2>(S, &p.tmap_w13, u.y0, u.y1, u.kb0, sw, p.A, H, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                // partial sums -> P[pair][row]
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int g = 0; g < NGRP; ++g) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int t = g * 8 + tcol + (i & 1);
                        if (t < u.ntok) {
                            const int row = rbase + m * 16 + (lane >> 2) + ((i >> 1) ? 8 : 0);
                            red_add_f32(p.P + (size_t)(u.p0 + u.tg + t) * N1 + row, acc[m][g][i]);
                        }
                    }
                }
            } else if (u.kind == 1) {
                // ---- GEMM2 unit: (expert, 128 rows, K-split) x 16 pairs ----
                const int r0 = u.n0 + warp * (16 * MT) + (lane >> 2);
                if (KS2 == 1)      { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 16, true, 1>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 16, true, 2>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                else if (KS2 == 2) { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 8, true, 1>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 8, true, 2>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
                else               { if (V3_NG1 && u.ng == 1) tile_k_loop<NGRP, 4, true, 1>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); else tile_k_loop<NGRP, 4, true, 2>(S, &p.tmap_w2, u.y0, -1, u.kb0, u.sw, p.Cbuf, I, u.ntok, u.ng, it, tid, warp, lane, acc, TRP, S.dtok[dd], sclb); }
#pragma unroll
                for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int g = 0; g < NGRP; ++g) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int t = g * 8 + tcol + (i & 1);
                        if (t < u.ntok) {
                            const int row = r0 + m * 16 + ((i >> 1) ? 8 : 0);
                            const int tok = (int)S.tok_list[u.p0 + u.tg + t];
                            red_add_f32(p.out_acc + (size_t)tok * H + row, acc[m][g][i] * S.dw[dd][t]);
                        }
                    }
                }
            } else {
                // ---- finalize unit: n-tile u.n0 of the fp32 accumulator is complete -> bf16 output, re-zero ----
                V3T(55, gtimer());
                constexpr int CT = NCOMP * 32;
                const int n = T * (G2_ROWS / 2);
                for (int i0 = 0; i0 < n; i0 += CT * 8) {
                    float2 v[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int i = i0 + k * CT + tid;
                        v[k] = make_float2(0.f, 0.f);
                        if (i < n) {
                            const float* src = p.out_acc + (size_t)(i / (G2_ROWS / 2)) * H + u.n0 + (i % (G2_ROWS / 2)) * 2;
                            asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v[k].x), "=f"(v[k].y) : "l"(src) : "memory");
                        }
                    }
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int i = i0 + k * CT + tid;
                        if (i < n) {
                            const int t = i / (G2_ROWS / 2), c = (i % (G2_ROWS / 2)) * 2;
                            float* src = p.out_acc + (size_t)t * H + u.n0 + c;
                            *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)t * H + u.n0 + c) = __floats2bfloat162_rn(v[k].x, v[k].y);
                            asm volatile("st.global.cg.v2.f32 [%0], {%1,%2};" ::"l"(src), "f"(0.f), "f"(0.f) : "memory");
                        }
                    }
                }
                V3T(54, gtimer());
#ifdef MOE_TRACE
                if (p.trace && lane == 0 && blockIdx.x < 296) p.trace[blockIdx.x * 64 + 40 + warp] = gtimer();
#endif
            }
            if (unit < 14) V3T(3 + 2 * unit, (gtimer() << 2) | u.kind);
            __syncwarp();
            if (u.kind == 2) { if (lane == 0) mbar_arrive_relaxed(smem_u32(&S.dfree_bar[dd])); }   // outputs are only read after the kernel
            else if (lane == 0) mbar_arrive(smem_u32(&S.dfree_bar[dd]));
            ++unit;
        }
    }
#undef TRP
    // ---------------- self-cleaning counters ----------------
    __syncthreads();
    V3T(56, gtimer());
    if (tid == 0) {
        // all CTAs' counter updates are atomics; acq_rel on the arrival counter orders the final zeroing after them
        const int old = atom_add_acq_rel_gpu(&p.counters[CNT_FINISHED], 1);
        V3T(63, gtimer());
        S.flag = (old == (int)gridDim.x - 1);
    }
    __syncthreads();
    if (S.flag) {
        for (int i = tid; i < CNT_TOTAL; i += SMALL_THREADS) p.counters[i] = 0;
    }
}

template <int NGRP>
constexpr size_t small_smem_bytes() { return sizeof(SmallSmem<NGRP * 8>) + 1024; }

}  // namespace moe3
