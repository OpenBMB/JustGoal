"""Blackwell (sm_100a) recurrent-state scan kernel for chunked GDN prefill.

For every (sequence, v-head) the CTA walks the sequence's chunks and applies the
precomputed per-chunk affine map   S_{c+1} = M_c S_c + B_c   (S is K x V) with the
state held transposed (S^T, V x K) in Tensor Memory:

    S^T_{c+1} = S^T_c . M_c^T + B_c^T
    MMA: D[v][k] += sum_kk A[v][kk] * B[k][kk]   with A = a packed-bf16 copy of S^T_c
    read straight from TMEM (kind::f16) and B = M_c (bf16, N=k rows, K=kk) from smem.
    The fp32 state stays in TMEM; the readout warps produce the bf16 A copy each step.

B_c (bf16, [k/8][v][8] layout so that TMEM lane v reads 8 consecutive k with one 16B
load) is preloaded into the destination TMEM region by the "preload" warps
(global -> registers -> tcgen05.st), keeping it off the tensor pipe; per-chunk states
are written back as bf16 in the same [k/8][v][8] layout (one 16B store per 8 values);
M_c tiles are streamed into a 3-stage smem ring with cp.async.bulk from a global
buffer that already holds the canonical SW128 K-major smem image.

Warp roles (448 threads): warp0 = bulk-copy producer, warp1 = TMEM alloc + MMA
issuer, warps2-5 = preload (initial state + B^T into TMEM), warps6-13 = readout
(two warps per lane quarter, one per column half: packed-bf16 A copy, per-chunk
bf16 S^T images and the final fp32 state).
Compiled with NVRTC at import time, launched through the CUDA driver API.
"""
import ctypes
import functools

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

CHUNK = 128
TILE_BYTES = 128 * 128 * 4  # one fp32 128x128 tile
MTILE_BYTES = 128 * 128 * 2  # one bf16 128x128 tile

KERNEL_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned short u16;

#define D 128
#define NH 8
#define TILE_BYTES (128*128*4)
#define MTILE_BYTES (128*128*2)
#define N_MSTAGE 3
#define N_REGION 3
#define A_COL 384   /* TMEM column of the packed-bf16 copy of S^T used as the MMA A operand */

__device__ __forceinline__ u32 smem_u32(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(u32 bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(u32 bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 bar, u32 parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n"
        "}\n" :: "r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(u32 bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
// K-major SWIZZLE_128B smem descriptor (tf32/fp32 elements): LBO=1 (16B units), SBO=1024B, version 1.
__device__ __forceinline__ u64 make_desc(u32 saddr) {
    return (u64)((saddr >> 4) & 0x3FFF) | ((u64)1 << 16) | ((u64)64 << 32) | ((u64)1 << 46) | ((u64)2 << 61);
}
// instruction descriptor: bf16 x bf16 -> f32, M=128, N=128, A K-major (TMEM), B K-major
#define IDESC ((1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24))

__device__ __forceinline__ void mma_ts(u32 d_tmem, u32 a_tmem, u64 bdesc, u32 accum) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n"
        "}\n" :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(IDESC), "r"(accum) : "memory");
}
__device__ __forceinline__ void tmem_cp(u32 d_tmem, u64 sdesc) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(d_tmem), "l"(sdesc) : "memory");
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

// publish: all prior global stores of this warp -> visible at gpu scope before the counter increment
__device__ __forceinline__ void signal_ready(u32* flag, u32 inc) {
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(flag), "r"(inc) : "memory");
    }
}
__device__ __forceinline__ u16 f2bf(float f) {
    u32 u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}
// pack (lo, hi) fp32 -> bf16x2 with round-to-nearest-even (lo in the low 16 bits)
__device__ __forceinline__ u32 pack_bf16x2(u32 lo, u32 hi) {
    u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "r"(hi), "r"(lo));
    return r;
}

struct Bars {
    u64 full_m[N_MSTAGE];
    u64 empty_m[N_MSTAGE];
    u64 full_b[B_STAGES];
    u64 empty_b[B_STAGES];
    u64 step_done[4];      // ring indexed by step & 3 (no waiter can lag more than 3 steps)
    u64 bpre_done[4];
    u64 conv_done[2][4];   // packed-bf16 copy of S^T_c ready in the A region, per column half
    u64 region_free[N_REGION];
    u32 tmem_base;
};

extern "C" __global__ void __launch_bounds__(448, 1)
gdn_scan_kernel(const i64* __restrict__ cu_seqlens, int num_seqs,
                const float* __restrict__ state_in,      // [N][NH][D][D]  (S^T: [v][k])
                const u16* __restrict__ m_tiles,         // [chunk][NH][MTILE] bf16 SW128 K-major image of M_c (rows k, cols kk)
                const u16* __restrict__ b_tiles,         // [chunk][NH][MTILE] bf16 SW128 image of B_c^T (rows v, cols k)
                u16* __restrict__ state_chunks,          // [chunk][NH][MTILE] bf16 SW128 image of S^T_c (rows v, cols k) at the start of chunk c (local>=1)
                float* __restrict__ state_out,           // [N][NH][D][D]
                u32* __restrict__ flags,                 // [chunk][NH] ready counters: reaches FLAG_TARGET once the bf16 S^T_c image is in global memory
                i64* __restrict__ dbg)                   // optional [block][chunk][16] clock64 stamps
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    u32 base = smem_u32(smem_raw);
    u32 pad = (1024u - (base & 1023u)) & 1023u;
    unsigned char* smem = smem_raw + pad;
    unsigned char* sM = smem;                              // N_MSTAGE x 32KB
    unsigned char* sB = smem + N_MSTAGE * MTILE_BYTES;     // B_STAGES x 32KB B tile ring
    Bars* bars = (Bars*)(smem + (N_MSTAGE + B_STAGES) * MTILE_BYTES);

    const int seq = blockIdx.x / NH;
    const int h = blockIdx.x % NH;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const i64 seq_start = cu_seqlens[seq];
    const i64 seq_end = cu_seqlens[seq + 1];
    const int nchunks = (int)((seq_end - seq_start + CHUNK_SIZE - 1) / CHUNK_SIZE);
    int chunk_base = 0;
    for (int s = 0; s < seq; ++s) {
        i64 l = cu_seqlens[s + 1] - cu_seqlens[s];
        chunk_base += (int)((l + CHUNK_SIZE - 1) / CHUNK_SIZE);
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < N_MSTAGE; ++i) { mbar_init(smem_u32(&bars->full_m[i]), 1); mbar_init(smem_u32(&bars->empty_m[i]), 1); }
        for (int i = 0; i < B_STAGES; ++i) { mbar_init(smem_u32(&bars->full_b[i]), 1); mbar_init(smem_u32(&bars->empty_b[i]), 4); }
        for (int i = 0; i < 4; ++i) { mbar_init(smem_u32(&bars->step_done[i]), 1); mbar_init(smem_u32(&bars->bpre_done[i]), 4); mbar_init(smem_u32(&bars->conv_done[0][i]), 4); mbar_init(smem_u32(&bars->conv_done[1][i]), 4); }
        for (int i = 0; i < N_REGION; ++i) mbar_init(smem_u32(&bars->region_free[i]), 8);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) {
        u32 dst = smem_u32(&bars->tmem_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(dst) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    const u32 tmem = bars->tmem_base;

    if (warp == 0) {
        // ---------------- bulk-copy producer: M tiles, N_MSTAGE deep, 4 x 16KB copies per tile ----------------
        if (lane == 0) {
            for (int c = 0; c < nchunks; ++c) {
                const int m = c % N_MSTAGE;
                const u32 par = (u32)((c / N_MSTAGE) & 1);
                const u16* src = m_tiles + ((size_t)(chunk_base + c) * NH + h) * (MTILE_BYTES / 2);
                mbar_wait(smem_u32(&bars->empty_m[m]), par ^ 1);
                if (dbg) dbg[((size_t)blockIdx.x * nchunks + c) * 16 + 8] = clock64();
                mbar_expect_tx(smem_u32(&bars->full_m[m]), MTILE_BYTES);
                #pragma unroll
                for (int q = 0; q < 2; ++q)
                    bulk_g2s(smem_u32(sM + m * MTILE_BYTES + q * (MTILE_BYTES / 2)), src + q * (MTILE_BYTES / 4),
                             MTILE_BYTES / 2, smem_u32(&bars->full_m[m]));
                // B tile ring (released by the preload warps)
                {
                    const int bs = c % B_STAGES;
                    const u32 bpar = (u32)((c / B_STAGES) & 1);
                    mbar_wait(smem_u32(&bars->empty_b[bs]), bpar ^ 1);
                    mbar_expect_tx(smem_u32(&bars->full_b[bs]), MTILE_BYTES);
                    bulk_g2s(smem_u32(sB + bs * MTILE_BYTES), b_tiles + ((size_t)(chunk_base + c) * NH + h) * (MTILE_BYTES / 2), MTILE_BYTES, smem_u32(&bars->full_b[bs]));
                }
            }
        }
    } else if (warp == 1) {
        // ---------------- MMA issuer ----------------
        asm volatile("bar.sync 1, 416;" ::: "memory");   // warps 1..13: initial state + first preload done
        tc_fence_after();
        if (lane == 0) {
            for (int c = 0; c < nchunks; ++c) {
                const int rs = c % N_REGION;          // holds S^T_c
                const int rd = (c + 1) % N_REGION;    // holds B_c^T, receives S^T_{c+1}
                const int m = c % N_MSTAGE;
                i64* dbgc = dbg ? dbg + ((size_t)blockIdx.x * nchunks + c) * 16 : nullptr;
                if (dbgc) dbgc[0] = clock64();
                mbar_wait(smem_u32(&bars->bpre_done[c & 3]), (u32)((c >> 2) & 1));
                if (dbgc) dbgc[1] = clock64();
                mbar_wait(smem_u32(&bars->full_m[m]), (u32)((c / N_MSTAGE) & 1));
                if (dbgc) dbgc[2] = clock64();
                if (c >= 1) mbar_wait(smem_u32(&bars->step_done[(c - 1) & 3]), (u32)(((c - 1) >> 2) & 1));
                if (c >= 1) mbar_wait(smem_u32(&bars->conv_done[0][(c - 1) & 3]), (u32)(((c - 1) >> 2) & 1));
#if SPLIT_CONV == 0
                if (c >= 1) mbar_wait(smem_u32(&bars->conv_done[1][(c - 1) & 3]), (u32)(((c - 1) >> 2) & 1));
#endif
                if (dbgc) dbgc[3] = clock64();
                tc_fence_after();
                {
                    const u32 sm = smem_u32(sM + m * MTILE_BYTES);
                    const u32 ta = tmem + (u32)A_COL;
                    const u32 td = tmem + (u32)(rd * 128);
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const u32 saddr = sm + (u32)((s >> 2) * 16384 + (s & 3) * 32);
                        mma_ts(td, ta + (u32)(8 * s), make_desc(saddr), 1u);
                    }
#if SPLIT_CONV
                    if (c >= 1) mbar_wait(smem_u32(&bars->conv_done[1][(c - 1) & 3]), (u32)(((c - 1) >> 2) & 1));
                    tc_fence_after();
#endif
                    #pragma unroll
                    for (int s = 4; s < 8; ++s) {
                        const u32 saddr = sm + (u32)((s >> 2) * 16384 + (s & 3) * 32);
                        mma_ts(td, ta + (u32)(8 * s), make_desc(saddr), 1u);
                    }
                }
                tc_commit(smem_u32(&bars->step_done[c & 3]));
                tc_commit(smem_u32(&bars->empty_m[m]));
                if (dbgc) dbgc[4] = clock64();
            }
        }
        __syncwarp();
    } else if (warp < 6) {
        // ---------------- preload warps (2..5): initial state -> region 0, then B_c^T -> region (c+1)%3 ----------------
        const int q = warp & 3;
        const int row = q * 32 + lane;
        const u32 lane_addr = (u32)((q * 32) << 16);
        u32 r[128];
        {
            const float* src = state_in + (((size_t)seq * NH + h) * D + row) * D;
            #pragma unroll
            for (int i = 0; i < 128; i += 4) {
                float4 v4 = *reinterpret_cast<const float4*>(src + i);
                r[i] = __float_as_uint(v4.x); r[i + 1] = __float_as_uint(v4.y);
                r[i + 2] = __float_as_uint(v4.z); r[i + 3] = __float_as_uint(v4.w);
            }
            #pragma unroll
            for (int part = 0; part < 4; ++part) tmem_st32(tmem + lane_addr + (u32)(part * 32), r + part * 32);
            {
                u32 pk[64];
                #pragma unroll
                for (int i = 0; i < 64; ++i) pk[i] = pack_bf16x2(r[2 * i], r[2 * i + 1]);
                tmem_st32(tmem + lane_addr + (u32)A_COL, pk);
                tmem_st32(tmem + lane_addr + (u32)(A_COL + 32), pk + 32);
            }
            tmem_wait_st();
            tc_fence_before();
            // also publish the initial state as the bf16 S^T image of local chunk 0 (consumed by the output kernel)
            {
                unsigned char* dst = reinterpret_cast<unsigned char*>(state_chunks + ((size_t)chunk_base * NH + h) * D * D) + row * 128;
                #pragma unroll
                for (int g = 0; g < 16; ++g) {
                    uint4 pk;
                    pk.x = pack_bf16x2(r[8 * g + 0], r[8 * g + 1]);
                    pk.y = pack_bf16x2(r[8 * g + 2], r[8 * g + 3]);
                    pk.z = pack_bf16x2(r[8 * g + 4], r[8 * g + 5]);
                    pk.w = pack_bf16x2(r[8 * g + 6], r[8 * g + 7]);
                    *reinterpret_cast<uint4*>(dst + (g >> 3) * 16384 + (((g & 7) ^ (row & 7)) << 4)) = pk;
                }
                signal_ready(flags + (size_t)chunk_base * NH + h, 2u);
            }
            asm volatile("bar.sync 1, 416;" ::: "memory");   // initial state visible to the MMA warp
        }
        u32 region_uses[N_REGION] = {0, 0, 0};
        for (int c = 0; c < nchunks; ++c) {
            const int rd = (c + 1) % N_REGION;
            const int bs = c % B_STAGES;
            mbar_wait(smem_u32(&bars->full_b[bs]), (u32)((c / B_STAGES) & 1));
            {
                const unsigned char* src = sB + bs * MTILE_BYTES + row * 128;
                #pragma unroll
                for (int g = 0; g < 16; ++g) {
                    uint4 v4 = *reinterpret_cast<const uint4*>(src + (g >> 3) * 16384 + (((g & 7) ^ (row & 7)) << 4));
                    r[8 * g + 0] = v4.x << 16; r[8 * g + 1] = v4.x & 0xFFFF0000u;
                    r[8 * g + 2] = v4.y << 16; r[8 * g + 3] = v4.y & 0xFFFF0000u;
                    r[8 * g + 4] = v4.z << 16; r[8 * g + 5] = v4.z & 0xFFFF0000u;
                    r[8 * g + 6] = v4.w << 16; r[8 * g + 7] = v4.w & 0xFFFF0000u;
                }
            }
            // Release the B stage only once the loads above have actually returned: the mbarrier arrive alone
            // does not wait for in-flight generic loads, and the producer's next bulk copy (async proxy) would
            // otherwise overwrite the tail of the tile while it is still being read. A data dependency on every
            // loaded register plus a generic->async proxy fence makes the WAR ordering explicit.
            {
#if REL_MODE & 1
                u32 dep = 0;
                #pragma unroll
                for (int i = 0; i < 128; ++i) dep ^= r[i];
                asm volatile("" :: "r"(dep) : "memory");
#endif
#if REL_MODE & 2
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&bars->empty_b[bs]));
            if (c >= 2) {
                // region rd held S^T_{c-2}: wait until MMA step c-2 (reader of A) and the readout warps are done with it
                mbar_wait(smem_u32(&bars->step_done[(c - 2) & 3]), (u32)(((c - 2) >> 2) & 1));
            }
            if (c >= 3) {
                mbar_wait(smem_u32(&bars->region_free[rd]), region_uses[rd] & 1);
                region_uses[rd]++;
            }
            tc_fence_after();
            #pragma unroll
            for (int part = 0; part < 4; ++part) tmem_st32(tmem + lane_addr + (u32)(rd * 128 + part * 32), r + part * 32);
            tmem_wait_st();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&bars->bpre_done[c & 3]));
            if (dbg && lane == 0 && q == 0) dbg[((size_t)blockIdx.x * nchunks + c) * 16 + 9] = clock64();
        }
    } else {
        // ---------------- readout warps (6..13): lane quarter q = warp & 3, column half rh = (warp - 6) >> 2 ----------------
        const int q = warp & 3;
        const int rh = (warp - 6) >> 2;
        const int row = q * 32 + lane;
        const u32 lane_addr = (u32)((q * 32) << 16);
        const u32 c0 = (u32)(rh * 64);
        asm volatile("bar.sync 1, 416;" ::: "memory");
        for (int c = 0; c < nchunks; ++c) {
            const int rd = (c + 1) % N_REGION;
            mbar_wait(smem_u32(&bars->step_done[c & 3]), (u32)((c >> 2) & 1));
            tc_fence_after();
            const u32 ta = tmem + lane_addr + (u32)(rd * 128) + c0;
            u32 r[64];
            tmem_ld32(ta, r);
            tmem_ld32(ta + 32, r + 32);
            tmem_wait_ld();
            if (c + 1 < nchunks) {
                u32 pk[32];
                #pragma unroll
                for (int i = 0; i < 32; ++i) pk[i] = pack_bf16x2(r[2 * i], r[2 * i + 1]);
                tmem_st32(tmem + lane_addr + (u32)A_COL + c0 / 2, pk);
                tmem_wait_st();
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(smem_u32(&bars->conv_done[rh][c & 3]));
            }
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&bars->region_free[rd]));
            if (dbg && lane == 0 && q == 0 && rh == 0) dbg[((size_t)blockIdx.x * nchunks + c) * 16 + 11] = clock64();
            if (c + 1 < nchunks) {
                unsigned char* dst = reinterpret_cast<unsigned char*>(state_chunks + ((size_t)(chunk_base + c + 1) * NH + h) * D * D) + row * 128;
                #pragma unroll
                for (int g = 0; g < 8; ++g) {
                    uint4 pk;
                    pk.x = pack_bf16x2(r[8 * g + 0], r[8 * g + 1]);
                    pk.y = pack_bf16x2(r[8 * g + 2], r[8 * g + 3]);
                    pk.z = pack_bf16x2(r[8 * g + 4], r[8 * g + 5]);
                    pk.w = pack_bf16x2(r[8 * g + 6], r[8 * g + 7]);
                    const int gg = rh * 8 + g;
                    *reinterpret_cast<uint4*>(dst + (gg >> 3) * 16384 + (((gg & 7) ^ (row & 7)) << 4)) = pk;
                }
                signal_ready(flags + (size_t)(chunk_base + c + 1) * NH + h, 1u);
            } else {
                float* dst = state_out + (((size_t)seq * NH + h) * D + row) * D + c0;
                #pragma unroll
                for (int i = 0; i < 64; i += 4)
                    *reinterpret_cast<float4*>(dst + i) =
                        make_float4(__uint_as_float(r[i]), __uint_as_float(r[i + 1]), __uint_as_float(r[i + 2]), __uint_as_float(r[i + 3]));
            }
            if (dbg && lane == 0 && q == 0 && rh == 0) dbg[((size_t)blockIdx.x * nchunks + c) * 16 + 10] = clock64();
        }
    }
    __syncthreads();
    if (warp == 1) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem) : "memory");
    }
}
"""


def _check(err, what=""):
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"CUDA error {code} in {what}")


@functools.lru_cache(maxsize=None)
def _build(chunk_size: int):
    import os
    src = KERNEL_SRC.replace("CHUNK_SIZE", str(chunk_size))
    src = "#define READOUT_MODE " + os.environ.get("SCAN_READOUT_MODE", "0") + "\n#define SPLIT_CONV " + os.environ.get("SCAN_SPLIT_CONV", "1") + "\n#define B_STAGES " + os.environ.get("SCAN_B_STAGES", "2") + "\n#define REL_MODE " + os.environ.get("SCAN_REL_MODE", "2") + "\n" + src
    err, prog = nvrtc.nvrtcCreateProgram(src.encode(), b"gdn_scan.cu", 0, [], [])
    _check(err, "nvrtcCreateProgram")
    opts = [b"--gpu-architecture=sm_100a", b"-default-device", b"--std=c++17"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(res[0]) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log.decode())
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    _check(err, "nvrtcGetCUBINSize")
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    nvrtc.nvrtcDestroyProgram(prog)
    if os.environ.get("SCAN_DUMP_CUBIN"):
        open(os.environ["SCAN_DUMP_CUBIN"], "wb").write(cubin)
    torch.empty(1, device="cuda")  # make sure a CUDA context exists
    _check(cu.cuInit(0), "cuInit")
    err, module = cu.cuModuleLoadData(cubin)
    _check(err, "cuModuleLoadData")
    err, func = cu.cuModuleGetFunction(module, b"gdn_scan_kernel")
    _check(err, "cuModuleGetFunction")
    smem_bytes = (3 + int(os.environ.get("SCAN_B_STAGES", "2"))) * MTILE_BYTES + 1024 + 512
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes),
           "cuFuncSetAttribute")
    return module, func, smem_bytes


FLAG_TARGET = 8  # counter value of a published chunk state (4 preload warps x 2 for local chunk 0, 8 readout warps x 1 otherwise)


def scan_launch(cu_seqlens, state_in, m_tiles, b_tiles, state_chunks, state_out, chunk_size=CHUNK, dbg=None, flags=None, stream=None):
    """Launch the scan over all (sequence, head) pairs (on the current torch stream unless `stream` is given).
    `flags` ([max_chunks, 8] uint32, zeroed) receives FLAG_TARGET per (chunk, head) once its state image is published."""
    _, func, smem_bytes = _build(chunk_size)
    num_seqs = cu_seqlens.shape[0] - 1
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    if flags is None:
        flags = torch.zeros((m_tiles.shape[0], 8), dtype=torch.int32, device=m_tiles.device)
    args = (cu_seqlens.data_ptr(), num_seqs, state_in.data_ptr(), m_tiles.data_ptr(), b_tiles.data_ptr(),
            state_chunks.data_ptr(), state_out.data_ptr(), flags.data_ptr(), dbg.data_ptr() if dbg is not None else 0)
    types = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p)
    _check(cu.cuLaunchKernel(func, num_seqs * 8, 1, 1, 448, 1, 1, smem_bytes, stream, (args, types), 0), "cuLaunchKernel")


def swizzled_tile_index_bf16(device):
    """[128,128] int64: element offset (bf16 units) of (row, col) inside the 32KB SW128 K-major bf16 tile image."""
    r = torch.arange(128, device=device)[:, None]
    k = torch.arange(128, device=device)[None, :]
    byte = (k // 64) * 16384 + r * 128 + (((k % 64) // 8) ^ (r % 8)) * 16 + (k % 8) * 2
    return byte // 2


def swizzled_tile_index(device):
    """Return a [128,128] int64 tensor: element offset (in fp32 units) of (row, col) inside the 64KB tile image."""
    r = torch.arange(128, device=device)[:, None]
    k = torch.arange(128, device=device)[None, :]
    byte = (k // 32) * 16384 + r * 128 + (((k % 32) // 4) ^ (r % 8)) * 16 + (k % 4) * 4
    return byte // 4


def to_k8v8(x):
    """[..., K=128, V=128] -> [..., K/8, V, 8] contiguous (the lane-friendly scan layout)."""
    sh = x.shape[:-2]
    return x.reshape(*sh, 16, 8, 128).transpose(-1, -2).contiguous()


def from_k8v8(x):
    """[..., K/8, V, 8] -> [..., K, V]."""
    sh = x.shape[:-3]
    return x.reshape(*sh, 16, 128, 8).transpose(-1, -2).reshape(*sh, 128, 128)
