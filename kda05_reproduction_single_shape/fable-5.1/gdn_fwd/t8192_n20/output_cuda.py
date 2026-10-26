"""Blackwell (sm_100a) output kernel for chunked GDN prefill (C = 128 tokens).

One CTA (8 warps) per (chunk, v-head):
    P  = scale * tril(Q K^T * exp(gc_i - gc_j))                       (QK^T on tcgen05, mask/scale on CUDA cores)
    O  = (scale * exp(gc_i) Q) S_c + P (U - W S_c)                    (3 more MMAs; S_c = chunk start state)
S_c (K x V) is consumed as its transposed SW128 image [v][k] (the initial state for local chunk 0 is
converted on the fly, later chunks come straight from the scan kernel's per-chunk writes).
Shared memory: three 32KB tiles; TMEM: 256 columns (two CTAs per SM).
"""
import ctypes
import functools
import os

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

from .intra_cuda import TmapCT, make_tmap_3d, _check, MTILE_BYTES

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

struct __align__(64) TmapT { u64 d[16]; };

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 bar, u32 count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count)); }
__device__ __forceinline__ void mbar_expect_tx(u32 bar, u32 bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
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
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(u32 bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ u64 make_desc(u32 saddr, u32 lbo_bytes) {
    return (u64)((saddr >> 4) & 0x3FFF) | ((u64)((lbo_bytes >> 4) & 0x3FFF) << 16) | ((u64)64 << 32) | ((u64)1 << 46) | ((u64)2 << 61);
}
#define IDESC_KK  ((1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24))
#define IDESC_KMN ((1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | (16u << 17) | (8u << 24))
__device__ __forceinline__ void mma_ss(u32 d_tmem, u64 adesc, u64 bdesc, u32 idesc, u32 accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
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
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ u32 pack_bf16x2(float lo, float hi) {
    u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float bf16_to_f32(u16 x) { return __uint_as_float(((u32)x) << 16); }
__device__ __forceinline__ u32 tile_off(int row, int col) {
    return (u32)((col >> 6) * 16384 + row * 128 + ((((col & 63) >> 3) ^ (row & 7)) << 4) + ((col & 7) << 1));
}
__device__ __forceinline__ uint4 pack8(const u32* r) {
    return make_uint4(pack_bf16x2(__uint_as_float(r[0]), __uint_as_float(r[1])), pack_bf16x2(__uint_as_float(r[2]), __uint_as_float(r[3])),
                      pack_bf16x2(__uint_as_float(r[4]), __uint_as_float(r[5])), pack_bf16x2(__uint_as_float(r[6]), __uint_as_float(r[7])));
}

struct Smem {
    unsigned char t0[MTILE];     // q image -> q*gamma*scale -> (after QS) W image -> (after WS) U image -> (U - WS) image
    unsigned char t1[MTILE];     // k image -> P image -> O image
    unsigned char t2[MTILE];     // S^T image
    float gc[128];
    float wsum[8];
    i64 cuseq[64];
    u64 bar_load[2];
    u64 bar_mma;
    u32 tmem_base;
};

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_output_kernel(const __grid_constant__ TmapT tm_q, const __grid_constant__ TmapT tm_k,
                  const __grid_constant__ TmapT tm_w, const __grid_constant__ TmapT tm_u, const __grid_constant__ TmapT tm_o,
                  const u16* __restrict__ a_raw, const u16* __restrict__ b_raw,
                  const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                  const i64* __restrict__ cu_seqlens, int num_seqs,
                  const float* __restrict__ state_in, const u16* __restrict__ state_chunks,
                  u16* __restrict__ o_ptr, float scale,
                  const u32* __restrict__ flags, u32 flag_target,   // per-(chunk, head) ready counters written by the scan kernel (nullptr: no wait)
                  i64* __restrict__ dbg)
{
#define STAMP(n) do { if (dbg && tid == 0) dbg[blockIdx.x * 16 + (n)] = clock64(); } while (0)
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    u32 base = smem_u32(smem_raw);
    u32 pad = (1024u - (base & 1023u)) & 1023u;
    Smem& S = *reinterpret_cast<Smem*>(smem_raw + pad);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = tid & 127;
    const int half = tid >> 7;
    const int h = blockIdx.x % NH;
    const int hq = h / (NH / HQ);

    if (tid <= num_seqs && tid < 64) S.cuseq[tid] = cu_seqlens[tid];
    const float h_Alog = __expf(A_log[h]);
    const float h_dt = dt_bias[h];
    if (tid == 0) {
        mbar_init(smem_u32(&S.bar_load[0]), 1);
        mbar_init(smem_u32(&S.bar_load[1]), 1);
        mbar_init(smem_u32(&S.bar_mma), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(smem_u32(&S.tmem_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    int seq = -1, local = 0, cglob = 0; i64 seq_start = 0, seq_end = 0;
    {
        // block b -> the b-th (chunk) in readiness order: level L = local chunk index, sequences in order within a level
        int b = (int)(blockIdx.x / NH);
        for (int L = 0; seq < 0; ++L) {
            int cnt = 0, acc = 0, found = -1, fbase = 0;
            for (int s = 0; s < num_seqs; ++s) {
                i64 st = S.cuseq[s], en = S.cuseq[s + 1];
                int n = (int)((en - st + C - 1) / C);
                if (n > L) { if (cnt == b) { found = s; fbase = acc; } ++cnt; }
                acc += n;
            }
            if (cnt == 0) break;                       // b beyond the last chunk: padding block
            if (found >= 0) { seq = found; local = L; cglob = fbase + L; seq_start = S.cuseq[seq]; seq_end = S.cuseq[seq + 1]; }
            else b -= cnt;
        }
    }
    u16 a_val = 0;
    if (seq >= 0 && tid < 128) {
        const int t = (int)seq_start + local * C + tid;
        if (t < (int)seq_end) a_val = a_raw[(size_t)t * NH + h];
    }
    const u32 tmem = S.tmem_base;
    const u32 R0 = tmem, R1 = tmem + 128;     // R0: QK^T then WS ; R1: O accumulator
    const u32 lane_addr = (u32)(((warp & 3) * 32) << 16);
    const u32 col0 = (u32)(half * 64);
    u32 mma_phase = 0;
    if (seq < 0) {
        __syncthreads();
        if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem) : "memory");
        return;
    }
    STAMP(0);
    const int t0 = (int)seq_start + local * C;
    const int nvalid = min(C, (int)(seq_end - t0));

    // ---- loads: q, k (TMA) ; S^T image (bulk copy, or converted from the initial state) ; W, U (TMA, second barrier)
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load[0]), 3 * MTILE);
        tma_load_3d(smem_u32(S.t0), &tm_q, 0, hq, t0, smem_u32(&S.bar_load[0]));
        tma_load_3d(smem_u32(S.t0) + 16384, &tm_q, 64, hq, t0, smem_u32(&S.bar_load[0]));
        tma_load_3d(smem_u32(S.t1), &tm_k, 0, hq, t0, smem_u32(&S.bar_load[0]));
        tma_load_3d(smem_u32(S.t1) + 16384, &tm_k, 64, hq, t0, smem_u32(&S.bar_load[0]));
        if (flags) {
            // wait until the scan kernel has published this chunk's state image (acquire), then order the async-proxy read after it
            const u32* f = flags + (size_t)cglob * NH + h;
            u32 v; long spins = 0;
            while (true) {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
                if (v >= flag_target) break;
                __nanosleep(128);
                if (++spins > 20000000L) break;        // ~>2s: never expected; avoids a hard hang (result would be wrong)
            }
            asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        bulk_g2s(smem_u32(S.t2), state_chunks + ((size_t)cglob * NH + h) * (MTILE / 2), MTILE, smem_u32(&S.bar_load[0]));
    }
    // ---- gates (thread = token j, first 128 threads)
    if (tid < 128) {
        const bool valid = tid < nvalid;
        float ar = bf16_to_f32(a_val);
        float x = ar + h_dt;
        float sp = x > 20.f ? x : log1pf(__expf(x));
        float glog = valid ? -h_Alog * sp : 0.f;
        float v = glog;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { float n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
        if (lane == 31) S.wsum[warp] = v;
        S.gc[tid] = v;
    }
    __syncthreads();
    if (tid < 128) {
        float pre = 0.f;
        for (int w = 0; w < warp; ++w) pre += S.wsum[w];
        S.gc[tid] += pre;
    }
    __syncthreads();
    STAMP(1);
    mbar_wait(smem_u32(&S.bar_load[0]), 0);
    STAMP(2);
    // zero padding rows of k
    if (nvalid < C) {
        for (int idx = tid; idx < (C - nvalid) * 16; idx += 256) {
            int r = nvalid + idx / 16, ch = idx % 16;
            *reinterpret_cast<uint4*>(S.t1 + tile_off(r, ch * 8)) = make_uint4(0, 0, 0, 0);
        }
    }
    fence_async_shared();
    __syncthreads();

    // ---- QK^T -> R0
    if (tid == 0) {
        tc_fence_after();
        const u32 qb = smem_u32(S.t0), kb = smem_u32(S.t1);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            u64 ad = make_desc(qb + (s >> 2) * 16384 + (s & 3) * 32, 16);
            mma_ss(R0, ad, make_desc(kb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        }
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    STAMP(3);
    // ---- P = scale * tril(q k^T * exp(gc_i - gc_j)) -> bf16 image into t1 (k dead); q <- scale*exp(gc_i) q in place
    {
        const int i = row;
        const float gi = S.gc[i];
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 r[32];
            tmem_ld32(R0 + lane_addr + col0 + part * 32, r);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = (int)col0 + part * 32 + ch * 8 + e;
                    const float dg = fminf(gi - S.gc[j], 0.f);          // clamp keeps exp finite for j > i (masked anyway)
                    float p = scale * __expf(dg) * __uint_as_float(r[ch * 8 + e]);
                    r[ch * 8 + e] = __float_as_uint((j <= i) ? p : 0.f);
                }
                *reinterpret_cast<uint4*>(S.t1 + tile_off(i, (int)col0 + part * 32 + ch * 8)) = pack8(r + ch * 8);
            }
        }
        const float sq = (i < nvalid) ? scale * __expf(gi) : 0.f;
        #pragma unroll 2
        for (int ch = 0; ch < 8; ++ch) {
            const u32 off = tile_off(i, (int)col0 + ch * 8);
            uint4 x = *reinterpret_cast<uint4*>(S.t0 + off);
            u32 w[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                w[e] = pack_bf16x2(__uint_as_float(w[e] << 16) * sq, __uint_as_float(w[e] & 0xFFFF0000u) * sq);
            *reinterpret_cast<uint4*>(S.t0 + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    fence_async_shared();
    __syncthreads();
    // ---- QS -> R1 (O accumulator); then W tile into t0 once QS has consumed q
    if (tid == 0) {
        tc_fence_after();
        const u32 qb = smem_u32(S.t0), sb = smem_u32(S.t2);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            u64 ad = make_desc(qb + (s >> 2) * 16384 + (s & 3) * 32, 16);
            mma_ss(R1, ad, make_desc(sb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        }
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    // W tile into t0 (q dead after QS)
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load[0]), MTILE);
        tma_load_3d(smem_u32(S.t0), &tm_w, 0, h, t0, smem_u32(&S.bar_load[0]));
        tma_load_3d(smem_u32(S.t0) + 16384, &tm_w, 64, h, t0, smem_u32(&S.bar_load[0]));
    }
    mbar_wait(smem_u32(&S.bar_load[0]), 1);
    STAMP(4);
    // ---- WS = W S  (A = W K-major [i][k], B = S^T image K-major [v][k]) -> R0
    if (tid == 0) {
        tc_fence_after();
        const u32 wb = smem_u32(S.t0), sb = smem_u32(S.t2);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R0, make_desc(wb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(sb + (s >> 2) * 16384 + (s & 3) * 32, 16), IDESC_KK, s > 0 ? 1u : 0u);
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    // U tile into t0 (W dead after WS)
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load[1]), MTILE);
        tma_load_3d(smem_u32(S.t0), &tm_u, 0, h, t0, smem_u32(&S.bar_load[1]));
        tma_load_3d(smem_u32(S.t0) + 16384, &tm_u, 64, h, t0, smem_u32(&S.bar_load[1]));
    }
    mbar_wait(smem_u32(&S.bar_load[1]), 0);
    STAMP(5);
    // ---- (U - WS) -> bf16 image in place in t3 (thread = row j, half)
    {
        const int j = row;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 r[32];
            tmem_ld32(R0 + lane_addr + col0 + part * 32, r);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                const u32 off = tile_off(j, (int)col0 + part * 32 + ch * 8);
                uint4 u = *reinterpret_cast<uint4*>(S.t0 + off);
                u32 uw[4] = {u.x, u.y, u.z, u.w};
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float lo = __uint_as_float(uw[e] << 16) - __uint_as_float(r[ch * 8 + 2 * e]);
                    float hi = __uint_as_float(uw[e] & 0xFFFF0000u) - __uint_as_float(r[ch * 8 + 2 * e + 1]);
                    uw[e] = pack_bf16x2(lo, hi);
                }
                *reinterpret_cast<uint4*>(S.t0 + off) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
            }
        }
    }
    fence_async_shared();
    __syncthreads();
    // ---- O += P (U - WS): A = P K-major [i][j], B = (U-WS) MN-major (rows j = K, cols v = N) -> R1
    if (tid == 0) {
        tc_fence_after();
        const u32 pb = smem_u32(S.t1), xb = smem_u32(S.t0);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R1, make_desc(pb + (s >> 2) * 16384 + (s & 3) * 32, 16), make_desc(xb + s * 2048, 16384), IDESC_KMN, 1u);
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    STAMP(6);
    // ---- readout O -> bf16 image into t1 -> TMA store (or row stores for partial chunks)
    {
        const int i = row;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 r[32];
            tmem_ld32(R1 + lane_addr + col0 + part * 32, r);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch)
                *reinterpret_cast<uint4*>(S.t1 + tile_off(i, (int)col0 + part * 32 + ch * 8)) = pack8(r + ch * 8);
        }
    }
    fence_async_shared();
    __syncthreads();
    if (nvalid == C) {
        if (tid == 0) {
            tma_store_3d(&tm_o, smem_u32(S.t1), 0, h, t0);
            tma_store_3d(&tm_o, smem_u32(S.t1) + 16384, 64, h, t0);
            bulk_commit();
            bulk_wait_all();
        }
    } else {
        const int i = row;
        if (i < nvalid) {
            uint4* od = reinterpret_cast<uint4*>(o_ptr + ((size_t)(t0 + i) * NH + h) * D) + half * 8;
            #pragma unroll
            for (int ch = 0; ch < 8; ++ch) od[ch] = *reinterpret_cast<const uint4*>(S.t1 + tile_off(i, (int)col0 + ch * 8));
        }
    }
    __syncthreads();
    STAMP(7);
    if (warp == 0) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem) : "memory");
    }
}
"""


@functools.lru_cache(maxsize=None)
def _build():
    err, prog = nvrtc.nvrtcCreateProgram(KERNEL_SRC.encode(), b"gdn_output.cu", 0, [], [])
    _check(err, "nvrtcCreateProgram")
    opts = [b"--gpu-architecture=sm_100a", b"-default-device", b"--std=c++17", b"-lineinfo"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(res[0]) != 0:
        raise RuntimeError("NVRTC compile failed:\n" + log.decode())
    err, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * cubin_size
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
    nvrtc.nvrtcDestroyProgram(prog)
    torch.empty(1, device="cuda")
    _check(cu.cuInit(0), "cuInit")
    err, module = cu.cuModuleLoadData(cubin)
    _check(err, "cuModuleLoadData")
    err, func = cu.cuModuleGetFunction(module, b"gdn_output_kernel")
    _check(err, "cuModuleGetFunction")
    smem_bytes = 3 * MTILE_BYTES + 512 + 64 + 512 + 64 + 1024
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes),
           "cuFuncSetAttribute")
    return module, func, smem_bytes


def output_launch(q, k, a, b, A_log, dt_bias, cu_seqlens, state, state_chunks, W, U, output, scale, max_chunks, dbg=None,
                  flags=None, flag_target=8, stream=None):
    """flags: [max_chunks, 8] int32 ready counters from the scan kernel (None: state_chunks assumed complete)."""
    _, func, smem_bytes = _build()
    num_seqs = cu_seqlens.shape[0] - 1
    tm_q = make_tmap_3d(q, 4)
    tm_k = make_tmap_3d(k, 4)
    tm_w = make_tmap_3d(W, 8)
    tm_u = make_tmap_3d(U, 8)
    tm_o = make_tmap_3d(output, 8)
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    cargs = [tm_q, tm_k, tm_w, tm_u, tm_o, ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
             ctypes.c_void_p(A_log.data_ptr()), ctypes.c_void_p(dt_bias.data_ptr()), ctypes.c_void_p(cu_seqlens.data_ptr()),
             ctypes.c_int(num_seqs), ctypes.c_void_p(state.data_ptr()), ctypes.c_void_p(state_chunks.data_ptr()),
             ctypes.c_void_p(output.data_ptr()), ctypes.c_float(scale),
             ctypes.c_void_p(flags.data_ptr() if flags is not None else 0), ctypes.c_uint32(flag_target),
             ctypes.c_void_p(dbg.data_ptr() if dbg is not None else 0)]
    ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(x) for x in cargs])
    _check(cu.cuLaunchKernel(func, max_chunks * 8, 1, 1, 256, 1, 1, smem_bytes, stream, ctypes.addressof(ptrs), 0), "cuLaunchKernel")
