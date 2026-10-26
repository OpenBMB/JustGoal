"""Blackwell (sm_100a) persistent "tail" kernel for chunked GDN prefill: recurrent scan + output, fused.

One CTA per SM (576 threads, 165 KB smem, 512 TMEM columns) pulls tasks from a global queue:
  * tasks [0, num_seqs*8)         : scan items (sequence, v-head), longest sequences first.
                                    S^T_{c+1} = S^T_c M_c^T + B_c^T with the fp32 state resident in TMEM,
                                    the bf16 A operand copied inside TMEM by 8 readout warps, M_c tiles
                                    streamed through a 2-stage cp.async.bulk smem ring, B_c^T tiles through
                                    a 2-stage ring and preloaded into the destination TMEM region by 4
                                    preload warps, which also publish every chunk's bf16 state image to
                                    global memory (+ a ready flag) off the critical path.
  * tasks [num_seqs*8, +chunks*8) : output items (chunk, v-head) in readiness order (all local-0 chunks,
                                    then local-1, ...):  O = (scale e^{gc} Q) S_c + P (U - W S_c) with
                                    P = scale tril(Q K^T e^{gc_i - gc_j});  the CTA waits (acquire) on the
                                    chunk's ready flag before reading its state image.
Because every output item's dependency is a scan item that was pulled earlier by a running CTA, forward
progress is guaranteed with any number of resident CTAs.
Compiled with NVRTC at import time, launched through the CUDA driver API.
"""
import ctypes
import functools
import os

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from .intra_cuda import TmapCT, make_tmap_3d, _check

CHUNK = 128
MTILE_BYTES = 128 * 128 * 2
# Ready flags hold the launch epoch of the run that published them (no per-call reset needed); the task counter is reset by the
# last CTA to exit, so the sync buffer only has to be zero at allocation.
SMEM_BYTES = 6 * MTILE_BYTES + 4096 + 1024

KERNEL_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned short u16;

#define D 128
#define C 128
#define NH 8
#define HQ 4
#define MTILE 32768
#define N_MSTAGE TAIL_MSTAGE_C
#define N_REGION 3
#define B_STAGES TAIL_BSTAGE_C
#define A_COL 384            /* TMEM column of the packed-bf16 copy of S^T (MMA A operand) */
#define MAXSEQ 64

struct __align__(64) TmapT { u64 d[16]; };

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ bool mbar_test(u32 bar, u32 parity) {       // non-blocking phase test
    u32 ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) {
    asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_3d(u32 dst, const TmapT* tmap, int x, int y, int z, u32 bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const TmapT* tmap, u32 src, int x, int y, int z) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                 :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(src) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, u32 src, u32 bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_2() { asm volatile("cp.async.bulk.wait_group 2;" ::: "memory"); }
// Publish a chunk's ready flag. Called only after `cp.async.bulk.wait_group 0` returned for the image's bulk store, i.e. the
// TMA writes have been performed at L2 (the GPU's coherence point, bypassing L1) before the flag store is even issued; the
// consumer reads the flag with ld.acquire.gpu (L2) and the image with a TMA load (L2). A gpu-scope release store here would
// add a membar that waits behind this SM's in-flight bulk loads (~2300 cycles per step, measured), so a relaxed store is used.
__device__ __forceinline__ void publish_flag(u32* flag, u32 epoch) {
#if TAIL_PUBFENCE
    asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
#if TAIL_RELEASE_FLAG
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(flag), "r"(epoch) : "memory");
#else
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(flag), "r"(epoch) : "memory");
#endif
}
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
// wait for a ready flag (intra results or a published chunk state) to carry this launch's epoch
__device__ __forceinline__ void wait_flag(const u32* f, u32 epoch) {
    u32 v;
    while (true) {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
        if (v == epoch) break;
        __nanosleep(32);
    }
}
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(u32 bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
// K-major SWIZZLE_128B smem descriptor (bf16): SBO = 1024B, version 1
__device__ __forceinline__ u64 make_desc(u32 saddr, u32 lbo_bytes) {
    return (u64)((saddr >> 4) & 0x3FFF) | ((u64)((lbo_bytes >> 4) & 0x3FFF) << 16) | ((u64)64 << 32) | ((u64)1 << 46) | ((u64)2 << 61);
}
#define IDESC_KK  ((1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24))
#define IDESC_KMN ((1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | (16u << 17) | (8u << 24))
__device__ __forceinline__ void mma_ss(u32 d_tmem, u64 adesc, u64 bdesc, u32 idesc, u32 accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void mma_ts(u32 d_tmem, u32 a_tmem, u64 bdesc, u32 accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(IDESC_KK), "r"(accum) : "memory");
}
__device__ __forceinline__ void tmem_ld32(u32 taddr, u32* r) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
          "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
          "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
          "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
        : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(u32 taddr, const u32* r) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
           "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
           "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
           "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]),
           "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ u32 pack_bf16x2(float lo, float hi) {
    u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ u32 pack_bf16x2u(u32 lo, u32 hi) {
    u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "r"(hi), "r"(lo)); return r;
}
__device__ __forceinline__ float bf16_to_f32(u16 x) { return __uint_as_float(((u32)x) << 16); }
__device__ __forceinline__ u32 tile_off(int row, int col) {
    return (u32)((col >> 6) * 16384 + row * 128 + ((((col & 63) >> 3) ^ (row & 7)) << 4) + ((col & 7) << 1));
}
__device__ __forceinline__ uint4 pack8(const u32* r) {
    return make_uint4(pack_bf16x2(__uint_as_float(r[0]), __uint_as_float(r[1])), pack_bf16x2(__uint_as_float(r[2]), __uint_as_float(r[3])),
                      pack_bf16x2(__uint_as_float(r[4]), __uint_as_float(r[5])), pack_bf16x2(__uint_as_float(r[6]), __uint_as_float(r[7])));
}
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// store the 128 fp32 values of one TMEM lane (row) as the bf16 SW128 image row of a 32KB tile
__device__ __forceinline__ void store_bf16_row(unsigned char* tile, int row, const u32* r) {
    unsigned char* dst = tile + row * 128;
    #pragma unroll
    for (int g = 0; g < 16; ++g) {
        uint4 pk;
        pk.x = pack_bf16x2u(r[8 * g + 0], r[8 * g + 1]);
        pk.y = pack_bf16x2u(r[8 * g + 2], r[8 * g + 3]);
        pk.z = pack_bf16x2u(r[8 * g + 4], r[8 * g + 5]);
        pk.w = pack_bf16x2u(r[8 * g + 6], r[8 * g + 7]);
        *reinterpret_cast<uint4*>(dst + (g >> 3) * 16384 + (((g & 7) ^ (row & 7)) << 4)) = pk;
    }
}

struct ScanBars {
    u64 full_m[N_MSTAGE];
    u64 empty_m[N_MSTAGE];
    u64 full_b[B_STAGES];
    u64 empty_b[B_STAGES];
    u64 step_done[4];      // ring indexed by step & 3
    u64 bpre_done[4];
    u64 conv_done[2][4];   // packed-bf16 copy of S^T_c ready in the A region, per column half
    u64 region_free[N_REGION];
    u64 pub_read[N_REGION];
};
struct OutBars {
    u64 bar_load[2];
    u64 bar_s;
    u64 bar_qk;              // one completion per output item each: a shared barrier would run two phases ahead of a slow warp
    u64 bar_ws;              //   (the QS->WS commits have no CTA barrier between them), whose parity wait then never returns
    u64 bar_pu;
    u64 bar_b;               // single-chunk sequences (final state merged into the local-0 output item): B_0^T tile load
    u64 bar_fin;             //   and the S^T_0 M_0^T MMA
    float gc[2][128];        // cumulative log gates, slot = output item parity (computed by warp 17 for the next item when it is known)
    int gates_for[2];        // task whose gates slot [i] holds (written only by warp 17, for the other slot than the running item's)
};
struct Common {
    i64 cuseq[MAXSEQ + 1];
    int nch[MAXSEQ];
    int cbase[MAXSEQ];
    int seq_by_rank[MAXSEQ];
    short ord_seq[256];      // output items in readiness order: (sequence, local chunk)
    short ord_local[256];
    int task;
    int next_task;           // reserved next task (valid if have_next): output items reserve it at their start (warp 17) so that
    int have_next;           //   the next item's q/k, QK^T and gates can be prefetched; scan items reserve it at their flush
    int tq, tk, tW, tU, tqs; // smem tile roles of the current output item (rotate when the next item's q/k were prefetched)
    int pf_next;             // next item's q/k prefetched into tW/tqs: 1 = and its QK^T issued, 2 = loads only (partial chunk: the
    int pf_cur;              //   next item zeroes the k padding rows with all threads and issues QK^T itself); pf_cur: same for this item
    u64 wait_ns;
    int g_base;              // global step index of this CTA's next scan item (barrier phases persist across items)
    int out_n;               // output items processed by this CTA
    int fin_n;               // ... of which merged-final-state items
    u32 region_uses[N_REGION];   // completions of region_free[r] so far
    u32 pub_uses[N_REGION];      // completions of pub_read[r] so far
    u32 tmem_base;
    u32 epoch;
    u32 mk_n[18];            // per-warp event ring fill (lane 0 of each warp owns its slot)
};
// smem map: [0, 160KB) tiles (scan: 3 M stages + 2 B stages; output: t0..t3), then the small structs
#define OFF_BARS   (6 * MTILE)
#define OFF_OUT    (6 * MTILE + 512)
#define OFF_COMMON (6 * MTILE + 2048)

#define OUT_SYNC() asm volatile("bar.sync 2, 544;" ::: "memory")   /* warps 0..16 */

// ============================================================ scan item ============================================================
// warps: 0 producer (M and B rings) | 1 MMA | 2-5 preload (initial state, B_c^T -> TMEM) | 6-13 readout (bf16 A copy) | 14-17 publisher
__device__ __forceinline__ void scan_item(unsigned char* smem, ScanBars* bars, Common& cm, int seq, int h,
                                          const float* __restrict__ state_in, const u16* __restrict__ m_tiles,
                                          const u16* __restrict__ b_tiles, u16* __restrict__ state_chunks,
                                          float* __restrict__ state_out, u32* __restrict__ flags, u32* __restrict__ counter,
                                          const u32* __restrict__ ready, u64* __restrict__ dbg2, u64* __restrict__ dtask)
{
#define ST2(c, i) do { if (dbg2) dbg2[(c) * 16 + (i)] = clock64(); if (dtask) dtask[3] = 1000 + (c) * 16 + (i); } while (0)
    unsigned char* sM = smem;
    unsigned char* sB = smem + N_MSTAGE * MTILE;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nchunks = cm.nch[seq];
    const int chunk_base = cm.cbase[seq];
    const u32 tmem = cm.tmem_base;
    const int G = cm.g_base;                    // steps of this item are g = G + c; S^T_c lives in TMEM region g % 3
    if (dbg2 && threadIdx.x == 0) dbg2[(nchunks + 1) * 16 + 0] = clock64();
    if (dbg2 && lane == 0) {   // per-warp-class entry stamps
        if (warp == 0) dbg2[(nchunks + 1) * 16 + 9] = clock64();
        if (warp == 2) dbg2[(nchunks + 1) * 16 + 10] = clock64();
        if (warp == 14) dbg2[(nchunks + 1) * 16 + 11] = clock64();
        if (warp == 1) dbg2[(nchunks + 1) * 16 + 12] = clock64();
    }

    if (warp == 0) {
        // ---------------- bulk-copy producer (one thread): M tiles (N_MSTAGE ring) and B^T tiles (B_STAGES ring), polled ----------------
        if (lane == 0) {
            int cmi = 0, cbi = 0, ready_upto = -1;      // chunks <= ready_upto have their intra results in global memory
            while (cmi < nchunks || cbi < nchunks) {
                if (cmi < nchunks) {
                    const int g = G + cmi;
                    const int m = g % N_MSTAGE;
                    if (mbar_test(smem_u32(&bars->empty_m[m]), (u32)((g / N_MSTAGE) & 1) ^ 1)) {
                        if (ready && cmi > ready_upto) { wait_flag(ready + (size_t)(chunk_base + cmi) * NH + h, cm.epoch); ready_upto = cmi; }
                        const u16* src = m_tiles + ((size_t)(chunk_base + cmi) * NH + h) * (MTILE / 2);
                        mbar_expect_tx(smem_u32(&bars->full_m[m]), MTILE);
                        bulk_g2s(smem_u32(sM + m * MTILE), src, MTILE / 2, smem_u32(&bars->full_m[m]));
                        bulk_g2s(smem_u32(sM + m * MTILE) + MTILE / 2, src + MTILE / 4, MTILE / 2, smem_u32(&bars->full_m[m]));
                        if (dbg2 && cmi == 0) dbg2[(nchunks + 1) * 16 + 6] = clock64();
                        ++cmi;
                    }
                }
                if (cbi < nchunks) {
                    const int g = G + cbi;
                    const int bs = g % B_STAGES;
                    if (mbar_test(smem_u32(&bars->empty_b[bs]), (u32)((g / B_STAGES) & 1) ^ 1)) {
                        if (ready && cbi > ready_upto) { wait_flag(ready + (size_t)(chunk_base + cbi) * NH + h, cm.epoch); ready_upto = cbi; }
                        mbar_expect_tx(smem_u32(&bars->full_b[bs]), MTILE);
                        bulk_g2s(smem_u32(sB + bs * MTILE), b_tiles + ((size_t)(chunk_base + cbi) * NH + h) * (MTILE / 2), MTILE, smem_u32(&bars->full_b[bs]));
                        if (dbg2 && cbi == 0) dbg2[(nchunks + 1) * 16 + 2] = clock64();
                        ++cbi;
                    }
                }
            }
        }
        __syncwarp();
    } else if (warp == 1) {
        // ---------------- MMA issuer ----------------
        asm volatile("bar.sync 1, 544;" ::: "memory");   // warps 1..17: initial state in TMEM
        if (dbg2 && lane == 0) dbg2[(nchunks + 1) * 16 + 1] = clock64();
        tc_fence_after();
        if (lane == 0) {
            for (int c = 0; c < nchunks; ++c) {
                const int g = G + c;
                const int rd = (g + 1) % N_REGION;    // holds B_c^T, receives S^T_{c+1}
                const int m = g % N_MSTAGE;
                ST2(c, 0);
                mbar_wait(smem_u32(&bars->bpre_done[g & 3]), (u32)((g >> 2) & 1));
                ST2(c, 1);
                mbar_wait(smem_u32(&bars->full_m[m]), (u32)((g / N_MSTAGE) & 1));
                ST2(c, 2);
                if (c >= 1) mbar_wait(smem_u32(&bars->step_done[(g - 1) & 3]), (u32)(((g - 1) >> 2) & 1));
                if (c >= 1) mbar_wait(smem_u32(&bars->conv_done[0][(g - 1) & 3]), (u32)(((g - 1) >> 2) & 1));
                ST2(c, 3);
                tc_fence_after();
                const u32 sm = smem_u32(sM + m * MTILE);
                const u32 ta = tmem + (u32)A_COL;
                const u32 td = tmem + (u32)(rd * 128);
                #pragma unroll
                for (int s = 0; s < 4; ++s)
                    mma_ts(td, ta + (u32)(8 * s), make_desc(sm + (u32)((s >> 2) * 16384 + (s & 3) * 32), 16), 1u);
                if (c >= 1) mbar_wait(smem_u32(&bars->conv_done[1][(g - 1) & 3]), (u32)(((g - 1) >> 2) & 1));
                ST2(c, 4);
                tc_fence_after();
                #pragma unroll
                for (int s = 4; s < 8; ++s)
                    mma_ts(td, ta + (u32)(8 * s), make_desc(sm + (u32)((s >> 2) * 16384 + (s & 3) * 32), 16), 1u);
                tc_commit(smem_u32(&bars->step_done[g & 3]));
                tc_commit(smem_u32(&bars->empty_m[m]));
                ST2(c, 5);
            }
        }
        __syncwarp();
    } else if (warp < 6) {
        // ---------------- preload warps (2..5): initial state -> region 0 + A copy; then B_c^T -> region (c+1)%3 ----------------
        const int q = warp & 3;
        const int row = q * 32 + lane;
        const u32 lane_addr = (u32)((q * 32) << 16);
        u32 r[64];
        {
            const float* src = state_in + (((size_t)seq * NH + h) * D + row) * D;
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                #pragma unroll
                for (int i = 0; i < 64; i += 4) {
                    float4 v4 = *reinterpret_cast<const float4*>(src + hf * 64 + i);
                    r[i] = __float_as_uint(v4.x); r[i + 1] = __float_as_uint(v4.y);
                    r[i + 2] = __float_as_uint(v4.z); r[i + 3] = __float_as_uint(v4.w);
                }
                tmem_st32(tmem + lane_addr + (u32)((G % N_REGION) * 128 + hf * 64), r);
                tmem_st32(tmem + lane_addr + (u32)((G % N_REGION) * 128 + hf * 64 + 32), r + 32);
                u32 pk[32];
                #pragma unroll
                for (int i = 0; i < 32; ++i) pk[i] = pack_bf16x2u(r[2 * i], r[2 * i + 1]);
                tmem_st32(tmem + lane_addr + (u32)(A_COL + hf * 32), pk);
            }
            tmem_wait_st();
            tc_fence_before();
            asm volatile("bar.sync 1, 544;" ::: "memory");   // initial state visible to the MMA / publisher warps
        }
        for (int c = 0; c < nchunks; ++c) {
            const int g = G + c;
            const int rd = (g + 1) % N_REGION;
            const int bs = g % B_STAGES;
            mbar_wait(smem_u32(&bars->full_b[bs]), (u32)((g / B_STAGES) & 1));
            if (warp == 2 && lane == 0) ST2(c, 8);
            // region rd held S^T_{c-2} of this item (c >= 2): readers are the readout warps (A copy, step c-3) and the publisher
            // (image c-2 >= 1); their arrivals on region rd in this item are the (c/3 - 1)-th ones, after cm.*_uses[rd] earlier ones.
            if (c >= 3) {
                const u32 k = (u32)(c / 3 - 1);
                mbar_wait(smem_u32(&bars->region_free[rd]), (cm.region_uses[rd] + k) & 1);
                if (warp == 2 && lane == 0) ST2(c, 13);
                mbar_wait(smem_u32(&bars->pub_read[rd]), (cm.pub_uses[rd] + k) & 1);
            }
            if (warp == 2 && lane == 0) ST2(c, 9);
            tc_fence_after();
            const unsigned char* src = sB + bs * MTILE + row * 128;
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                #pragma unroll
                for (int gq = 0; gq < 8; ++gq) {
                    const int gg = hf * 8 + gq;
                    uint4 v4 = *reinterpret_cast<const uint4*>(src + (gg >> 3) * 16384 + (((gg & 7) ^ (row & 7)) << 4));
                    r[8 * gq + 0] = v4.x << 16; r[8 * gq + 1] = v4.x & 0xFFFF0000u;
                    r[8 * gq + 2] = v4.y << 16; r[8 * gq + 3] = v4.y & 0xFFFF0000u;
                    r[8 * gq + 4] = v4.z << 16; r[8 * gq + 5] = v4.z & 0xFFFF0000u;
                    r[8 * gq + 6] = v4.w << 16; r[8 * gq + 7] = v4.w & 0xFFFF0000u;
                }
                tmem_st32(tmem + lane_addr + (u32)(rd * 128 + hf * 64), r);
                tmem_st32(tmem + lane_addr + (u32)(rd * 128 + hf * 64 + 32), r + 32);
            }
            // the smem loads above must have returned before the stage is handed back to the async-proxy producer
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&bars->empty_b[bs]));
            tmem_wait_st();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&bars->bpre_done[g & 3]));
            if (warp == 2 && lane == 0) ST2(c, 10);
        }
    } else if (warp < 14) {
        // ---------------- readout warps (6..13): bf16 A copy of S^T_{c+1}; lane quarter q, column half rh ----------------
        const int q = warp & 3;
        const int rh = (warp - 6) >> 2;
        const u32 lane_addr = (u32)((q * 32) << 16);
        const u32 c0 = (u32)(rh * 64);
        asm volatile("bar.sync 1, 544;" ::: "memory");
        for (int c = 0; c < nchunks; ++c) {           // the last step's copy is unused but keeps one completion per step
            const int g = G + c;
            const int rd = (g + 1) % N_REGION;
            mbar_wait(smem_u32(&bars->step_done[g & 3]), (u32)((g >> 2) & 1));
            if (warp == 6 && lane == 0) ST2(c, 6);
            tc_fence_after();
            const u32 ta = tmem + lane_addr + (u32)(rd * 128) + c0;
            u32 r[32], pk[32];
            tmem_ld32(ta, r);
            tmem_wait_ld();
            #pragma unroll
            for (int i = 0; i < 16; ++i) pk[i] = pack_bf16x2u(r[2 * i], r[2 * i + 1]);
            tmem_ld32(ta + 32, r);
            tmem_wait_ld();
            #pragma unroll
            for (int i = 0; i < 16; ++i) pk[16 + i] = pack_bf16x2u(r[2 * i], r[2 * i + 1]);
            tmem_st32(tmem + lane_addr + (u32)A_COL + c0 / 2, pk);
            tmem_wait_st();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) { mbar_arrive(smem_u32(&bars->conv_done[rh][g & 3])); mbar_arrive(smem_u32(&bars->region_free[rd])); }
            if (warp == 6 && lane == 0) ST2(c, 7);
        }
    } else {
        // ---------------- publisher warps (14..17): S^T_j -> bf16 image (staging tile) -> bulk stores + ready flag (j < nchunks);
        // S^T_nchunks -> fp32 state_out. Two leader threads alternate (even/odd j), so a leader only ever waits on its own
        // bulk groups from two iterations back: the flag of image j is published at iteration j+2 after `wait_group 0`
        // (fence.proxy.async + st.release then see no pending async writes), and the staging tile is recycled after the
        // other leader's `wait_group.read 0`.
        const int q = warp & 3;
        const int row = q * 32 + lane;
        const u32 lane_addr = (u32)((q * 32) << 16);
        unsigned char* stg_base = smem + (N_MSTAGE + B_STAGES) * MTILE;   // two staging tiles, one per leader parity
        const bool lead_even = (warp == 14 && lane == 0);
        const bool lead_odd = (warp == 15 && lane == 0);
        asm volatile("bar.sync 1, 544;" ::: "memory");
        for (int j = 1; j <= nchunks; ++j) {        // S^T_0 is the input state: local-0 output items read it directly
            const int gj = G + j;
            mbar_wait(smem_u32(&bars->step_done[(gj - 1) & 3]), (u32)(((gj - 1) >> 2) & 1));
            if (warp == 14 && lane == 0 && j < nchunks) ST2(j, 11);
            tc_fence_after();
            const u32 ta = tmem + lane_addr + (u32)((gj % N_REGION) * 128);
            u32 r[32];
            if (j < nchunks) {
                const bool leader = (j & 1) ? lead_odd : lead_even;       // issues the stores of image j (tile j & 1)
                unsigned char* stg = stg_base + (j & 1) * MTILE;
                if (leader && j >= 3) {                                   // image j-2 (>= 1): same tile, same leader
                    if (dbg2 && warp == 14) dbg2[(nchunks + 2 + j) * 16 + 0] = clock64();
                    bulk_wait_all();
                    if (dbg2 && warp == 14) dbg2[(nchunks + 2 + j) * 16 + 1] = clock64();
                    publish_flag(flags + (size_t)(chunk_base + j - 2) * NH + h, cm.epoch);
                    if (dbg2 && warp == 14) dbg2[(nchunks + 2 + j) * 16 + 2] = clock64();
                }
                asm volatile("bar.sync 3, 128;" ::: "memory");
                if (warp == 14 && lane == 0) ST2(j, 14);
                unsigned char* dst = stg + row * 128;
                #pragma unroll
                for (int part = 0; part < 4; ++part) {
                    tmem_ld32(ta + (u32)(part * 32), r);
                    tmem_wait_ld();
                    #pragma unroll
                    for (int g = 0; g < 4; ++g) {
                        const int gg = part * 4 + g;
                        uint4 pk;
                        pk.x = pack_bf16x2u(r[8 * g + 0], r[8 * g + 1]);
                        pk.y = pack_bf16x2u(r[8 * g + 2], r[8 * g + 3]);
                        pk.z = pack_bf16x2u(r[8 * g + 4], r[8 * g + 5]);
                        pk.w = pack_bf16x2u(r[8 * g + 6], r[8 * g + 7]);
                        *reinterpret_cast<uint4*>(dst + (gg >> 3) * 16384 + (((gg & 7) ^ (row & 7)) << 4)) = pk;
                    }
                }
                if (warp == 14 && lane == 0) ST2(j, 15);
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(smem_u32(&bars->pub_read[gj % N_REGION]));
                fence_async_shared();
                asm volatile("bar.sync 3, 128;" ::: "memory");              // image complete in smem
                if (leader) {
#if TAIL_NO_PUB == 0
                    bulk_s2g(state_chunks + ((size_t)(chunk_base + j) * NH + h) * D * D, smem_u32(stg), MTILE);
#endif
                    bulk_commit();
                }
                if (warp == 14 && lane == 0) ST2(j, 12);
            } else {
                float* dst = state_out + (((size_t)seq * NH + h) * D + row) * D;
                #pragma unroll
                for (int part = 0; part < 4; ++part) {
                    tmem_ld32(ta + (u32)(part * 32), r);
                    tmem_wait_ld();
                    #pragma unroll
                    for (int i = 0; i < 32; i += 4)
                        *reinterpret_cast<float4*>(dst + part * 32 + i) =
                            make_float4(__uint_as_float(r[i]), __uint_as_float(r[i + 1]), __uint_as_float(r[i + 2]), __uint_as_float(r[i + 3]));
                }
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(smem_u32(&bars->pub_read[gj % N_REGION]));
            }
        }
        if (warp == 14 && lane == 0 && dbg2) dbg2[nchunks * 16 + 0] = clock64();
        // flush: each leader publishes the images it still owns (nchunks-1 and, if any, nchunks-2)
        if (lead_even || lead_odd) {
            bulk_wait_all();
            const int mine = lead_even ? 0 : 1;
            for (int j = nchunks - 2; j < nchunks; ++j)
                if (j >= 1 && (j & 1) == mine) publish_flag(flags + (size_t)(chunk_base + j) * NH + h, cm.epoch);
        }
        if (lead_even) { cm.next_task = (int)atomicAdd(counter, 1u); cm.have_next = 1; }   // next task: its latency hides behind the flush
        if (warp == 14 && lane == 0 && dbg2) dbg2[nchunks * 16 + 1] = clock64();
    }
}

// ============================================================ output item ============================================================
// executed by all 18 warps: warps 0..15 compute (row = tid & 127, column quarter cq = tid >> 7 of 32 columns); warp 16 is the
// helper (flag waits, W/U/S loads, QS and WS MMAs, O store); warp 17 computes the cumulative gates (named barrier 4).
// Chain: q,k -> QK^T -> P readout  ||  S,W -> QS, WS (helper)  ->  (U-WS) readout -> P(U-WS) -> O readout -> store.
__device__ __forceinline__ void output_item(unsigned char* smem, OutBars* ob, Common& cm, int seq, int local, int cglob, int h, int n_multi, int n_tasks,
                                            const TmapT* tm_q, const TmapT* tm_k, const TmapT* tm_w, const TmapT* tm_u, const TmapT* tm_o,
                                            const u16* __restrict__ a_raw, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                            const float* __restrict__ state_in, const u16* __restrict__ state_chunks,
                                            const u16* __restrict__ m_tiles, const u16* __restrict__ b_tiles, float* __restrict__ state_out,
                                            u16* __restrict__ o_ptr, float scale, const u32* __restrict__ flags, u32* __restrict__ counter,
                                            const u32* __restrict__ ready, u64* __restrict__ dbg3, u64* __restrict__ dtask, u64* __restrict__ dbgw)
{
#define ST3(i) do { if (dbg3 && threadIdx.x == 0) dbg3[i] = gtime(); if (dtask && threadIdx.x == 0) dtask[3] = (i); \
                    if (dbgw && (threadIdx.x & 31) == 0) dbgw[blockIdx.x * 18 + (threadIdx.x >> 5)] = (i); } while (0)
// per-warp event ring (hang diagnosis / timelines): entry = marker << 48 | globaltimer (ns, low 48 bits); 256 entries per warp
#define MK_RING 256
#define MARKW(i) do { if (dbgw && (threadIdx.x & 31) == 0) { const int w_ = threadIdx.x >> 5; const u32 n_ = cm.mk_n[w_]; \
        if (n_ < MK_RING) { dbgw[((size_t)blockIdx.x * 18 + w_) * MK_RING + n_] = ((u64)(i) << 48) | (gtime() & 0xFFFFFFFFFFFFull); cm.mk_n[w_] = n_ + 1; } } } while (0)
    ST3(0);
    // tile roles rotate between items so that the next item's q/k land in the tiles freed by WS/QS (see the main loop)
    unsigned char* t0 = smem + cm.tq * MTILE;  // q image -> (after QK^T) (U - WS) image
    unsigned char* t1 = smem + cm.tk * MTILE;  // k image -> P image
    unsigned char* t2 = smem + 2 * MTILE;      // S^T image (kept: A operand of the merged final-state MMA)
    unsigned char* t3 = smem + cm.tW * MTILE;  // W image -> B_0^T image (merged final state) | next item's q
    unsigned char* t4 = smem + cm.tU * MTILE;  // U image -> O image
    unsigned char* t5 = smem + cm.tqs * MTILE; // scale*gamma*q image -> M_0 image (merged final state) | next item's k
    const int pf_cur = cm.pf_cur;              // 1: q, k in t0/t1 and QK^T issued (bar_qk) by the previous item; 2: q, k only
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = tid & 127;
    const int cq = tid >> 7;
    const int hq = h / (NH / HQ);
    const u32 tmem = cm.tmem_base;
    const u32 R0 = tmem, R1 = tmem + 128, R2 = tmem + 256;   // R0: QK^T (then S_0 M_0^T); R1: O accumulator; R2: WS
    const u32 lane_addr = (u32)(((warp & 3) * 32) << 16);
    const u32 col0 = (u32)(cq * 32);
    const i64 seq_start = cm.cuseq[seq], seq_end = cm.cuseq[seq + 1];
    const int t0tok = (int)seq_start + local * C;
    const int nvalid = min(C, (int)(seq_end - t0tok));
    const u32 ph = (u32)(cm.out_n & 1);       // every OutBars barrier completes exactly once per output item
    const bool final_here = (local == 0 && cm.nch[seq] == 1);   // one-chunk sequence: its final state S_1^T = S_0^T M_0^T + B_0^T is computed here
    const u32 fph = (u32)(cm.fin_n & 1);
    const int gslot = cm.out_n & 1;
    float* gc = ob->gc[gslot];
    const bool gates_pf = (ob->gates_for[gslot] == (int)cm.task);   // computed by warp 17 during the previous item
    if (warp == 17) {
        // ---- reserve the next task (so that this item can prefetch for it), then the gates: this item's (unless prefetched,
        //      named barrier 4) and the next item's (slot gslot ^ 1, read only after the next item starts)
        if (lane == 0) { cm.next_task = (int)atomicAdd(counter, 1u); cm.have_next = 1; }
        __syncwarp();
        const int next = __shfl_sync(0xffffffffu, cm.next_task, 0);
        asm volatile("bar.arrive 5, 64;" ::: "memory");             // next_task visible to the helper warp
        for (int which = gates_pf ? 1 : 0; which < 2; ++which) {
            int gt = (which == 0) ? (int)cm.task : next;
            if (which == 1 && gt >= n_tasks) break;
            int gseq = seq, glocal = local, gh = h;
            if (which == 1) { const int b = (gt - n_multi) / NH; gh = gt % NH; gseq = cm.ord_seq[b]; glocal = cm.ord_local[b]; }
            const int gt0 = (int)cm.cuseq[gseq] + glocal * C;
            const int gnv = min(C, (int)(cm.cuseq[gseq + 1] - gt0));
            float* gdst = ob->gc[(which == 0) ? gslot : (gslot ^ 1)];
            const float h_Alog = __expf(A_log[gh]);
            const float h_dt = dt_bias[gh];
            float v[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = lane * 4 + e;
                const bool valid = t < gnv;
                float x = (valid ? bf16_to_f32(a_raw[(size_t)(gt0 + t) * NH + gh]) : 0.f) + h_dt;
                float sp = x > 20.f ? x : log1pf(__expf(x));
                v[e] = valid ? -h_Alog * sp : 0.f;
            }
            v[1] += v[0]; v[2] += v[1]; v[3] += v[2];
            float tot = v[3];
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { float n = __shfl_up_sync(0xffffffffu, tot, o); if (lane >= o) tot += n; }
            const float pre = tot - v[3];
            #pragma unroll
            for (int e = 0; e < 4; ++e) gdst[lane * 4 + e] = v[e] + pre;
            __syncwarp();
            if (which == 0) { MARKW(30); asm volatile("bar.arrive 4, 576;" ::: "memory"); MARKW(31); }
            else if (lane == 0) ob->gates_for[gslot ^ 1] = gt;
        }
        return;
    }
    // ---- loads: q, k (TMA) on bar_load[0] by tid 0 (unless prefetched); W, U on bar_load[1] by the helper (after the previous
    //      item's O store has left t4 and the chunk's intra flag is set); S^T image on bar_s (helper, after the scan's ready flag)
    if (tid == 512) {
        MARKW(10);
        bulk_wait_read();                    // previous item's O store (from t4) has finished reading smem
        MARKW(11);
        if (ready) wait_flag(ready + (size_t)cglob * NH + h, cm.epoch);   // W, U, M, B of this chunk written by the intra kernel
        MARKW(12);
        mbar_expect_tx(smem_u32(&ob->bar_load[1]), 2 * MTILE);
        tma_load_3d(smem_u32(t3), tm_w, 0, h, t0tok, smem_u32(&ob->bar_load[1]));
        tma_load_3d(smem_u32(t3) + 16384, tm_w, 64, h, t0tok, smem_u32(&ob->bar_load[1]));
        tma_load_3d(smem_u32(t4), tm_u, 0, h, t0tok, smem_u32(&ob->bar_load[1]));
        tma_load_3d(smem_u32(t4) + 16384, tm_u, 64, h, t0tok, smem_u32(&ob->bar_load[1]));
        if (local == 0) mbar_expect_tx(smem_u32(&ob->bar_s), 0);   // nothing in flight: the conversion below fills t2
    }
    if (tid == 0 && pf_cur == 0) {
        mbar_expect_tx(smem_u32(&ob->bar_load[0]), 2 * MTILE);
        tma_load_3d(smem_u32(t0), tm_q, 0, hq, t0tok, smem_u32(&ob->bar_load[0]));
        tma_load_3d(smem_u32(t0) + 16384, tm_q, 64, hq, t0tok, smem_u32(&ob->bar_load[0]));
        tma_load_3d(smem_u32(t1), tm_k, 0, hq, t0tok, smem_u32(&ob->bar_load[0]));
        tma_load_3d(smem_u32(t1) + 16384, tm_k, 64, hq, t0tok, smem_u32(&ob->bar_load[0]));
    }
    if (local == 0 && tid < 512) {
        // first chunk: S^T_0 = input state, fp32 [v][k] -> bf16 SW128 image in t2 (thread = row, 32 columns); the global
        // loads overlap the q/k TMA loads in flight
        const float* src = state_in + (((size_t)seq * NH + h) * D + row) * D + col0;
        float4 x[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = *reinterpret_cast<const float4*>(src + i * 4);
        #pragma unroll
        for (int ch = 0; ch < 4; ++ch)
            *reinterpret_cast<uint4*>(t2 + tile_off(row, (int)col0 + ch * 8)) =
                make_uint4(pack_bf16x2(x[2 * ch].x, x[2 * ch].y), pack_bf16x2(x[2 * ch].z, x[2 * ch].w),
                           pack_bf16x2(x[2 * ch + 1].x, x[2 * ch + 1].y), pack_bf16x2(x[2 * ch + 1].z, x[2 * ch + 1].w));
    }
    ST3(1);
    if (pf_cur != 1) {
        mbar_wait(smem_u32(&ob->bar_load[0]), ph);
        if (nvalid < C) {        // zero padding rows of k
            for (int idx = tid; idx < (C - nvalid) * 16; idx += 512) {
                int r = nvalid + idx / 16, ch = idx % 16;
                *reinterpret_cast<uint4*>(t1 + tile_off(r, ch * 8)) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    ST3(2);
    fence_async_shared();
    OUT_SYNC();              // k padding visible to the MMA; the previous O store has left the qs tile (helper's wait_read)
    // ---- QK^T -> R0 (unless issued by the previous item)
    if (tid == 0 && pf_cur != 1) {
        tc_fence_after();
        const u32 qb = smem_u32(t0), kb = smem_u32(t1);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R0, make_desc(qb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(kb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        tc_commit(smem_u32(&ob->bar_qk));
    }
    if (tid == 512 && local > 0) {
        // helper: wait for the scan's ready flag (acquire), then bulk-copy the S^T image
        const u64 tw0 = gtime();
        MARKW(13);
        wait_flag(flags + (size_t)cglob * NH + h, cm.epoch);
        MARKW(14);
        cm.wait_ns = gtime() - tw0;
#if TAIL_CONS_FENCE
        asm volatile("fence.proxy.async.global;" ::: "memory");
#endif
        mbar_expect_tx(smem_u32(&ob->bar_s), MTILE);
        bulk_g2s(smem_u32(t2), state_chunks + ((size_t)cglob * NH + h) * (MTILE / 2), MTILE, smem_u32(&ob->bar_s));
    }
    MARKW(20);
    if (!gates_pf) asm volatile("bar.sync 4, 576;" ::: "memory");   // gates ready (warp 17 arrived)
    // ---- scale*exp(gc_i) q -> t5 (q in t0 is being read by QK^T), while QK^T runs
    if (tid < 512) {
        const int i = row;
        const float sq = (i < nvalid) ? scale * __expf(gc[i]) : 0.f;
        #pragma unroll
        for (int ch = 0; ch < 4; ++ch) {
            const u32 off = tile_off(i, (int)col0 + ch * 8);
            uint4 x = *reinterpret_cast<uint4*>(t0 + off);
            u32 w[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                w[e] = pack_bf16x2(__uint_as_float(w[e] << 16) * sq, __uint_as_float(w[e] & 0xFFFF0000u) * sq);
            *reinterpret_cast<uint4*>(t5 + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    fence_async_shared();
    OUT_SYNC();
    ST3(3);
    if (tid == 512) {
        // helper: QS -> R1 and WS -> R2 as soon as S and W have landed; they overlap the P readout below. (QS is covered by
        // the WS commit: the tensor pipe runs this thread's MMAs in order.)
        MARKW(15);
        mbar_wait(smem_u32(&ob->bar_s), ph);
        MARKW(16);
        mbar_wait(smem_u32(&ob->bar_load[1]), ph);
        MARKW(17);
        tc_fence_after();
        const u32 qb = smem_u32(t5), sb = smem_u32(t2), wb = smem_u32(t3);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R1, make_desc(qb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(sb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R2, make_desc(wb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(sb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        tc_commit(smem_u32(&ob->bar_ws));
    }
    mbar_wait(smem_u32(&ob->bar_qk), ph);
    tc_fence_after();
    // ---- P = scale * tril(q k^T * exp(gc_i - gc_j)) -> bf16 image into t1 (k dead)
    if (tid < 512) {
        const int i = row;
        const float gi = gc[i];
        u32 r[32];
        if (cq > (warp & 3)) {                 // whole quarter above the diagonal: zeros, no exponentials
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) *reinterpret_cast<uint4*>(t1 + tile_off(i, (int)col0 + ch * 8)) = make_uint4(0, 0, 0, 0);
        } else {
            tmem_ld32(R0 + lane_addr + col0, r);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = (int)col0 + ch * 8 + e;
                    const float dg = fminf(gi - gc[j], 0.f);
                    float p = scale * __expf(dg) * __uint_as_float(r[ch * 8 + e]);
                    r[ch * 8 + e] = __float_as_uint((j <= i) ? p : 0.f);
                }
                *reinterpret_cast<uint4*>(t1 + tile_off(i, (int)col0 + ch * 8)) = pack8(r + ch * 8);
            }
        }
    }
    fence_async_shared();
    ST3(4);
    mbar_wait(smem_u32(&ob->bar_ws), ph);         // QS and WS done (S^T image and W/U landed)
    tc_fence_after();
    MARKW(50);
    int next = -1;
    if (warp == 16) {
        asm volatile("bar.sync 5, 64;" ::: "memory");                 // warp 17 has reserved next_task (every output item)
        // W (t3) and the scaled q (t5) are dead: fetch B_0^T and M_0 for a merged final state, or prefetch the next output
        // item's q and k into them (next item: roles q <- tW, k <- tqs)
        if (lane == 0 && final_here) {
            mbar_expect_tx(smem_u32(&ob->bar_b), 2 * MTILE);
            bulk_g2s(smem_u32(t3), b_tiles + ((size_t)cglob * NH + h) * (MTILE / 2), MTILE, smem_u32(&ob->bar_b));
            bulk_g2s(smem_u32(t5), m_tiles + ((size_t)cglob * NH + h) * (MTILE / 2), MTILE, smem_u32(&ob->bar_b));
        } else if (lane == 0) {
            next = cm.next_task;
            if (next < n_tasks) {
                const int b = (next - n_multi) / NH, nh = next % NH;
                const int nseq = cm.ord_seq[b], nlocal = cm.ord_local[b];
                const int nt0 = (int)cm.cuseq[nseq] + nlocal * C;
                const int nhq = nh / (NH / HQ);
                mbar_expect_tx(smem_u32(&ob->bar_load[0]), 2 * MTILE);   // completion for the next item (parity ph ^ 1)
                tma_load_3d(smem_u32(t3), tm_q, 0, nhq, nt0, smem_u32(&ob->bar_load[0]));
                tma_load_3d(smem_u32(t3) + 16384, tm_q, 64, nhq, nt0, smem_u32(&ob->bar_load[0]));
                tma_load_3d(smem_u32(t5), tm_k, 0, nhq, nt0, smem_u32(&ob->bar_load[0]));
                tma_load_3d(smem_u32(t5) + 16384, tm_k, 64, nhq, nt0, smem_u32(&ob->bar_load[0]));
                const int nnv = min(C, (int)(cm.cuseq[nseq + 1] - nt0));
                cm.pf_next = (nnv < C) ? 2 : 1;   // partial chunk: k padding rows must be zeroed by all threads of the next item
                if (nnv == C) next = next; else next = -1;
            }
        }
    }
    ST3(5); ST3(6); ST3(7); ST3(8);
    // ---- (U - WS) -> bf16 image into t0 (q dead), U read from t4 (thread = row j, column quarter)
    if (tid < 512) {
        const int j = row;
        u32 r[32];
        tmem_ld32(R2 + lane_addr + col0, r);
        tmem_wait_ld();
        MARKW(40);
        #pragma unroll
        for (int ch = 0; ch < 4; ++ch) {
            const u32 off = tile_off(j, (int)col0 + ch * 8);
            uint4 u = *reinterpret_cast<uint4*>(t4 + off);
            u32 uw[4] = {u.x, u.y, u.z, u.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float lo = __uint_as_float(uw[e] << 16) - __uint_as_float(r[ch * 8 + 2 * e]);
                float hi = __uint_as_float(uw[e] & 0xFFFF0000u) - __uint_as_float(r[ch * 8 + 2 * e + 1]);
                uw[e] = pack_bf16x2(lo, hi);
            }
            *reinterpret_cast<uint4*>(t0 + off) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
        }
    }
    fence_async_shared();
    MARKW(41);
    OUT_SYNC();                              // P image (t1) and (U-WS) image (t0) complete; R0 (QK^T) fully read
    ST3(9);
    if (tid == 512 && next >= 0 && next < n_tasks) {
        // next item's QK^T -> R0 as soon as its q/k have landed (full chunks only)
        MARKW(60);
        mbar_wait(smem_u32(&ob->bar_load[0]), ph ^ 1);
        MARKW(61);
        tc_fence_after();
        const u32 qb = smem_u32(t3), kb = smem_u32(t5);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R0, make_desc(qb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(kb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        tc_commit(smem_u32(&ob->bar_qk));    // completion for the next item (parity ph ^ 1)
        MARKW(62);
    }
    // ---- O += P (U - WS): A = P K-major [i][j] (t1), B = (U-WS) MN-major (rows j = K, cols v = N) (t0) -> R1
    if (tid == 0) {
        tc_fence_after();
        const u32 pb = smem_u32(t1), xb = smem_u32(t0);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R1, make_desc(pb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(xb + s * 2048, 16384), IDESC_KMN, 1u);
        tc_commit(smem_u32(&ob->bar_pu));
    }
    mbar_wait(smem_u32(&ob->bar_pu), ph);
    tc_fence_after();
    MARKW(70);
    ST3(10);
    // ---- readout O -> bf16 image into t4 (U dead) -> TMA store (or row stores for partial chunks)
    if (tid < 512) {
        const int i = row;
        u32 r[32];
        tmem_ld32(R1 + lane_addr + col0, r);
        tmem_wait_ld();
        #pragma unroll
        for (int ch = 0; ch < 4; ++ch)
            *reinterpret_cast<uint4*>(t4 + tile_off(i, (int)col0 + ch * 8)) = pack8(r + ch * 8);
    }
    fence_async_shared();
    OUT_SYNC();
    ST3(11);
    if (nvalid == C) {
        if (tid == 512) {
            MARKW(80);
            tma_store_3d(tm_o, smem_u32(t4), 0, h, t0tok);
            tma_store_3d(tm_o, smem_u32(t4) + 16384, 64, h, t0tok);
            bulk_commit();                   // completion is awaited before t4 is refilled (next item) / at kernel end
            MARKW(81);
        }
    } else {
        const int i = row;
        if (tid < 512 && i < nvalid) {
            uint4* od = reinterpret_cast<uint4*>(o_ptr + ((size_t)(t0tok + i) * NH + h) * D) + cq * 4;
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) od[ch] = *reinterpret_cast<const uint4*>(t4 + tile_off(i, (int)col0 + ch * 8));
        }
    }
    ST3(13);
    if (final_here) {
        // ---- final state of a one-chunk sequence: R0 = S_0^T M_0^T (A = S^T_0 image t2 [v][k], B = M_0 image t5 [k_out][k_in]),
        //      state_out[v][k_out] = R0 + B_0^T[v][k_out] (t3)
        mbar_wait(smem_u32(&ob->bar_b), fph);
        ST3(14);
        if (tid == 0) {
            tc_fence_after();
            const u32 ab = smem_u32(t2), bb = smem_u32(t5);
            #pragma unroll
            for (int s = 0; s < 8; ++s)
                mma_ss(R0, make_desc(ab + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(bb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
            tc_commit(smem_u32(&ob->bar_fin));
        }
        mbar_wait(smem_u32(&ob->bar_fin), fph);
        tc_fence_after();
        ST3(15);
        if (tid < 512) {
            u32 r[32];
            tmem_ld32(R0 + lane_addr + col0, r);
            tmem_wait_ld();
            float* dst = state_out + (((size_t)seq * NH + h) * D + row) * D + col0;
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                uint4 bv = *reinterpret_cast<const uint4*>(t3 + tile_off(row, (int)col0 + ch * 8));
                const u32 bw[4] = {bv.x, bv.y, bv.z, bv.w};
                float o[8];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[2 * e] = __uint_as_float(r[ch * 8 + 2 * e]) + __uint_as_float(bw[e] << 16);
                    o[2 * e + 1] = __uint_as_float(r[ch * 8 + 2 * e + 1]) + __uint_as_float(bw[e] & 0xFFFF0000u);
                }
                *reinterpret_cast<float4*>(dst + ch * 8) = make_float4(o[0], o[1], o[2], o[3]);
                *reinterpret_cast<float4*>(dst + ch * 8 + 4) = make_float4(o[4], o[5], o[6], o[7]);
            }
        }
    }
    ST3(12);   // smem reads of this item end before the next item's loads: the main loop's __syncthreads follows
}

// ============================================================ kernel ============================================================
extern "C" __global__ void __launch_bounds__(576, 1)
gdn_tail_kernel(const __grid_constant__ TmapT tm_q, const __grid_constant__ TmapT tm_k, const __grid_constant__ TmapT tm_w,
                const __grid_constant__ TmapT tm_u, const __grid_constant__ TmapT tm_o,
                const u16* __restrict__ a_raw, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                const i64* __restrict__ cu_seqlens, int num_seqs,
                const float* __restrict__ state_in, const u16* __restrict__ m_tiles, const u16* __restrict__ b_tiles,
                u16* __restrict__ state_chunks, float* __restrict__ state_out, u16* __restrict__ o_ptr, float scale,
                u32* __restrict__ flags, u32* __restrict__ counter, u32 epoch,            // counter[0]: task queue, counter[1]: exited CTAs
                const u32* __restrict__ ready,                                                // intra-kernel per-(chunk, head) ready flags (PDL overlap), or null
                u64* __restrict__ dbg,                                                        // dbg: [task][4] = block, t_start, t_end, wait_ns
                u64* __restrict__ dbg2,                                                       // dbg2: [chunk][16] clock64 stamps of the longest sequence, head 0
                u64* __restrict__ dbg3, int dbg3_task, int dbg2_task,                         // dbg3: [16] globaltimer stamps of output task dbg3_task
                u64* __restrict__ dbgw)                                                       // dbgw: [block][18] per-warp progress markers (hang diagnosis)
{
    const u64 t_entry = gtime();
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    u32 base = smem_u32(smem_raw);
    u32 pad = (1024u - (base & 1023u)) & 1023u;
    unsigned char* smem = smem_raw + pad;
    ScanBars* bars = reinterpret_cast<ScanBars*>(smem + OFF_BARS);
    OutBars* ob = reinterpret_cast<OutBars*>(smem + OFF_OUT);
    Common& cm = *reinterpret_cast<Common*>(smem + OFF_COMMON);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    if (tid <= num_seqs && tid <= MAXSEQ) cm.cuseq[tid] = cu_seqlens[tid];
    if (tid == 0) {
        cm.g_base = 0; cm.out_n = 0; cm.fin_n = 0; cm.epoch = epoch;
        cm.tq = 0; cm.tk = 1; cm.tW = 3; cm.tU = 4; cm.tqs = 5; cm.pf_next = 0; cm.pf_cur = 0;
        for (int i = 0; i < 18; ++i) cm.mk_n[i] = 0;
        ob->gates_for[0] = -1; ob->gates_for[1] = -1;
        for (int i = 0; i < N_REGION; ++i) { cm.region_uses[i] = 0; cm.pub_uses[i] = 0; }
        for (int i = 0; i < N_MSTAGE; ++i) { mbar_init(smem_u32(&bars->full_m[i]), 1); mbar_init(smem_u32(&bars->empty_m[i]), 1); }
        for (int i = 0; i < B_STAGES; ++i) { mbar_init(smem_u32(&bars->full_b[i]), 1); mbar_init(smem_u32(&bars->empty_b[i]), 4); }
        for (int i = 0; i < 4; ++i) {
            mbar_init(smem_u32(&bars->step_done[i]), 1); mbar_init(smem_u32(&bars->bpre_done[i]), 4);
            mbar_init(smem_u32(&bars->conv_done[0][i]), 4); mbar_init(smem_u32(&bars->conv_done[1][i]), 4);
        }
        for (int i = 0; i < N_REGION; ++i) { mbar_init(smem_u32(&bars->region_free[i]), 8); mbar_init(smem_u32(&bars->pub_read[i]), 4); }
        mbar_init(smem_u32(&ob->bar_load[0]), 1);
        mbar_init(smem_u32(&ob->bar_load[1]), 1);
        mbar_init(smem_u32(&ob->bar_s), 1);
        mbar_init(smem_u32(&ob->bar_qk), 1);
        mbar_init(smem_u32(&ob->bar_ws), 1);
        mbar_init(smem_u32(&ob->bar_pu), 1);
        mbar_init(smem_u32(&ob->bar_b), 1);
        mbar_init(smem_u32(&ob->bar_fin), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    const u64 t_p1 = gtime();
    if (warp == 1) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(&cm.tmem_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    const u64 t_p2 = gtime();
    if (tid < num_seqs) {
        const int n = (int)((cm.cuseq[tid + 1] - cm.cuseq[tid] + C - 1) / C);
        cm.nch[tid] = n;
    }
    __syncthreads();
    if (tid < num_seqs) {
        int rank = 0, base_c = 0;
        const int n = cm.nch[tid];
        for (int s = 0; s < num_seqs; ++s) {
            const int ns = cm.nch[s];
            if (ns > n || (ns == n && s < tid)) ++rank;
            if (s < tid) base_c += ns;
        }
        cm.cbase[tid] = base_c;
        cm.seq_by_rank[rank] = tid;
    }
    __syncthreads();
    int total_chunks = 0;
    for (int s = 0; s < num_seqs; ++s) total_chunks += cm.nch[s];
    if (tid < total_chunks && tid < 256) {
        // readiness order: level L (local chunk index) major, sequence order within a level. One thread per global chunk:
        // pos(s, L) = #pairs at levels < L + #pairs (s2 < s) at level L = sum_s2 min(nch[s2], L) + #{s2 < s : nch[s2] > L}
        // (a single-thread append loop here cost ~18 us per CTA)
        int s = 0;
        while (s + 1 < num_seqs && cm.cbase[s + 1] <= tid) ++s;
        const int L = tid - cm.cbase[s];
        int pos = 0;
        for (int s2 = 0; s2 < num_seqs; ++s2) { const int n2 = cm.nch[s2]; pos += min(n2, L) + ((s2 < s && n2 > L) ? 1 : 0); }
        if (pos < 256) { cm.ord_seq[pos] = (short)s; cm.ord_local[pos] = (short)L; }
    }
    __syncthreads();
    int n_multi_seqs = 0;
    for (int s = 0; s < num_seqs; ++s) n_multi_seqs += (cm.nch[s] >= 2) ? 1 : 0;
    const int n_multi = n_multi_seqs * NH;                 // ranks [0, n_multi_seqs) are the multi-chunk sequences
    const int n_out = min(total_chunks, 256) * NH;
    const int n_tasks = n_multi + n_out;
    // queue: multi-chunk scan items | output items (readiness order); one-chunk sequences get their final state inside
    // their local-0 output item

    // Tasks are pulled one at a time (a CTA never holds a task it is not running, so a long scan item cannot delay another
    // task); the atomic for the next task is issued near the end of each item so that its latency is hidden.
    const u64 t_p3 = gtime();
    if (tid == 0) { cm.task = (int)atomicAdd(counter, 1u); cm.have_next = 0; }
    u64 t_p4 = 0;
    while (true) {
        __syncthreads();
        if (t_p4 == 0) t_p4 = gtime();
        const int task = cm.task;
        if (task >= n_tasks) break;
        const u64 ts0 = gtime();
        if (tid == 0) cm.wait_ns = 0;
        u64* dtask = dbg ? dbg + (size_t)task * 4 : nullptr;           // [block, t_start, t_end, phase/wait_ns]
        if (dbgw && (tid & 31) == 0) { const int w_ = tid >> 5; const u32 n_ = cm.mk_n[w_]; if (n_ < MK_RING) { dbgw[((size_t)blockIdx.x * 18 + w_) * MK_RING + n_] = ((u64)100 << 48) | (u64)task; cm.mk_n[w_] = n_ + 1; } }
        if (dtask && tid == 0) { dtask[0] = blockIdx.x; dtask[1] = ts0; dtask[2] = 0; dtask[3] = 0; }
        if (task < n_multi) {
            scan_item(smem, bars, cm, cm.seq_by_rank[task / NH], task % NH, state_in, m_tiles, b_tiles, state_chunks, state_out, flags,
                      counter, ready, (task == dbg2_task) ? dbg2 : nullptr, dtask);
        } else {
            const int b = (task - n_multi) / NH;
            const int h = (task - n_multi) % NH;
            const int seq = cm.ord_seq[b], local = cm.ord_local[b];
            output_item(smem, ob, cm, seq, local, cm.cbase[seq] + local, h, n_multi, n_tasks, &tm_q, &tm_k, &tm_w, &tm_u, &tm_o,
                        a_raw, A_log, dt_bias, state_in, state_chunks, m_tiles, b_tiles, state_out, o_ptr, scale, flags, counter, ready,
                        (task == dbg3_task) ? dbg3 : nullptr, dtask, dbgw);
        }
        MARKW(90);
        __syncthreads();
        MARKW(91);
        if (tid == 0) {
            if (task < n_multi) {
                const int n = cm.nch[cm.seq_by_rank[task / NH]];
                for (int c = 0; c < n; ++c) cm.region_uses[(cm.g_base + c + 1) % N_REGION]++;   // one readout arrival per step
                for (int j = 1; j <= n; ++j) cm.pub_uses[(cm.g_base + j) % N_REGION]++;         // one publisher arrival per j
                cm.g_base += n;
                cm.pf_cur = 0;
            } else {
                const int b = (task - n_multi) / NH;
                if (cm.ord_local[b] == 0 && cm.nch[cm.ord_seq[b]] == 1) cm.fin_n++;
                cm.out_n++;
                if (cm.pf_next) {            // next item's q/k sit in tW/tqs: rotate the roles (q, k, W, U, qs) <- (W, qs, q, k, U)
                    const int q0 = cm.tq, k0 = cm.tk, W0 = cm.tW, U0 = cm.tU, qs0 = cm.tqs;
                    cm.tq = W0; cm.tk = qs0; cm.tW = q0; cm.tU = k0; cm.tqs = U0;
                }
                cm.pf_cur = cm.pf_next; cm.pf_next = 0;
            }
        }
        if (dbg && tid == 0) { dbg[task * 4] = blockIdx.x; dbg[task * 4 + 1] = ts0; dbg[task * 4 + 2] = gtime(); dbg[task * 4 + 3] = cm.wait_ns | ((u64)cm.pf_cur << 48); }
        if (tid == 0) { cm.task = cm.have_next ? cm.next_task : (int)atomicAdd(counter, 1u); cm.have_next = 0; }
    }
    if (tid == 512) bulk_wait_all();         // helper warp's O stores
    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(cm.tmem_base) : "memory");
    }
    if (tid == 0) {                          // the last CTA to exit resets the queue for the next launch (all CTAs' pulls are done)
        __threadfence();
        if (atomicAdd(counter + 1, 1u) == gridDim.x - 1) { counter[0] = 0; counter[1] = 0; __threadfence(); }
    }
    if (dbg && tid == 0) {   // per-CTA entry/exit + prologue stamps
        dbg[(size_t)n_tasks * 4 + blockIdx.x * 2] = t_entry; dbg[(size_t)n_tasks * 4 + blockIdx.x * 2 + 1] = gtime();
        u64* pp = dbg + (size_t)n_tasks * 4 + (size_t)gridDim.x * 2 + blockIdx.x * 4;
        pp[0] = t_p1; pp[1] = t_p2; pp[2] = t_p3; pp[3] = t_p4;
    }
}
"""


@functools.lru_cache(maxsize=None)
def _build():
    src = KERNEL_SRC
    src = src.replace("TAIL_MSTAGE_C", os.environ.get("TAIL_MSTAGE", "2")).replace("TAIL_BSTAGE_C", os.environ.get("TAIL_BSTAGE", "2"))
    src = "#define TAIL_NO_PUB " + os.environ.get("TAIL_NO_PUB", "0") + "\n#define TAIL_PUBFENCE " + os.environ.get("TAIL_PUBFENCE", "0") + "\n#define TAIL_RELEASE_FLAG " + os.environ.get("TAIL_RELEASE_FLAG", "0") + "\n#define TAIL_CONS_FENCE " + os.environ.get("TAIL_CONS_FENCE", "0") + "\n" + src
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"gdn_tail.cu", 0, [], [])
    _check(err, "nvrtcCreateProgram")
    opts = [b"--gpu-architecture=sm_100a", b"-default-device", b"--std=c++17", b"-lineinfo"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(res[0]) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log.decode())
    if os.environ.get("GDN_NVRTC_LOG"):
        print(log.decode())
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    nvrtc.nvrtcDestroyProgram(prog)
    if os.environ.get("TAIL_DUMP_CUBIN"):
        open(os.environ["TAIL_DUMP_CUBIN"], "wb").write(cubin)
    torch.empty(1, device="cuda")
    _check(cu.cuInit(0), "cuInit")
    err, module = cu.cuModuleLoadData(cubin)
    _check(err, "cuModuleLoadData")
    err, func = cu.cuModuleGetFunction(module, b"gdn_tail_kernel")
    _check(err, "cuModuleGetFunction")
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SMEM_BYTES),
           "cuFuncSetAttribute")
    return module, func


_NUM_SMS = None


def tail_launch(q, k, a, A_log, dt_bias, cu_seqlens, state, m_tiles, b_tiles, state_chunks, W, U, output, new_state, scale,
                flags, counter, epoch=1, ready=None, pdl=False, stream=None, grid=None, dbg=None, dbg2=None, dbg3=None,
                dbg3_task=-1, dbg2_task=0, dbgw=None):
    """flags: [max_chunks*8] uint32, counter: [>=2] uint32; both zero at allocation. epoch: distinct per call (ready-flag value);
    the kernel resets the counter itself at exit. ready: the intra kernel's flags (same epoch); with pdl=True the kernel is
    launched with programmatic stream serialization so that it overlaps the intra kernel's last wave."""
    global _NUM_SMS
    _, func = _build()
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_seqs = cu_seqlens.shape[0] - 1
    tm_q = make_tmap_3d(q, 4)
    tm_k = make_tmap_3d(k, 4)
    tm_w = make_tmap_3d(W, 8)
    tm_u = make_tmap_3d(U, 8)
    tm_o = make_tmap_3d(output, 8)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    cargs = [tm_q, tm_k, tm_w, tm_u, tm_o, ctypes.c_void_p(a.data_ptr()),
             ctypes.c_void_p(A_log.data_ptr()), ctypes.c_void_p(dt_bias.data_ptr()), ctypes.c_void_p(cu_seqlens.data_ptr()),
             ctypes.c_int(num_seqs), ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(m_tiles.data_ptr()),
             ctypes.c_void_p(b_tiles.data_ptr()), ctypes.c_void_p(state_chunks.data_ptr()), ctypes.c_void_p(new_state.data_ptr()),
             ctypes.c_void_p(output.data_ptr()), ctypes.c_float(scale),
             ctypes.c_void_p(flags.data_ptr()), ctypes.c_void_p(counter.data_ptr()), ctypes.c_uint32(int(epoch) & 0xFFFFFFFF),
             ctypes.c_void_p(ready.data_ptr() if ready is not None else 0),
             ctypes.c_void_p(dbg.data_ptr() if dbg is not None else 0), ctypes.c_void_p(dbg2.data_ptr() if dbg2 is not None else 0),
             ctypes.c_void_p(dbg3.data_ptr() if dbg3 is not None else 0), ctypes.c_int(dbg3_task), ctypes.c_int(dbg2_task),
             ctypes.c_void_p(dbgw.data_ptr() if dbgw is not None else 0)]
    ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(x) for x in cargs])
    g = grid or _NUM_SMS
    if not pdl:
        _check(cu.cuLaunchKernel(func, g, 1, 1, 576, 1, 1, SMEM_BYTES, stream, ctypes.addressof(ptrs), 0), "cuLaunchKernel")
        return
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = g, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 576, 1, 1
    cfg.sharedMemBytes = SMEM_BYTES
    cfg.hStream = cu.CUstream(stream)
    attr = cu.CUlaunchAttribute()
    attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    attr.value.programmaticStreamSerializationAllowed = 1
    cfg.attrs = [attr]
    cfg.numAttrs = 1
    _check(cu.cuLaunchKernelEx(cfg, func, ctypes.addressof(ptrs), 0), "cuLaunchKernelEx")
