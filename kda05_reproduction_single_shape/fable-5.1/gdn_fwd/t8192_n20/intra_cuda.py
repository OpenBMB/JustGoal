"""Blackwell (sm_100a) intra-chunk kernel for chunked GDN prefill (C = 128 tokens).

One CTA (4 warps) per (chunk, v-head). Everything that does not depend on the chunk's
start state is computed here:
    A = tril_strict(beta_i exp(gc_i - gc_j) k_i.k_j),  T = (I + A)^{-1}
    W = T diag(beta gamma) K,  U = T diag(beta) V           -> stored bf16 [T,8,128] (TMA store)
    M = gamma_C I - K'^T W,    B = K'^T U,  K' = diag(gamma_C/gamma_j) K
        -> stored as bf16 SW128 K-major 32KB tile images for the scan kernel (bulk store)
Tensor-core work uses tcgen05 with fp32 accumulators in TMEM; the K/V tiles are loaded by
TMA (3D tensor maps, two 64-column SWIZZLE_128B boxes) and are used both as K-major and
MN-major MMA operands (the two interpretations share one smem image). The triangular
inverse: 16x16 diagonal blocks on CUDA cores (column-parallel), then 3 block-merge levels
T_2S = T_S - mask(T_S A) T_S with bf16 tcgen05 MMAs, the intermediate kept in TMEM as a
packed-bf16 A operand. Shared memory is 3 x 32KB tiles and TMEM 256 columns, so two CTAs
share an SM.
"""
import ctypes
import functools
import os

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

CHUNK = 128
MTILE_BYTES = 128 * 128 * 2

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
#define FTILE 65536

struct __align__(64) TmapT { u64 d[16]; };

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 bar, u32 count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
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
__device__ __forceinline__ void bulk_s2g(void* dst, u32 src, u32 bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(u32 bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
// SW128 smem descriptor. K-major: LBO unused(1), SBO = 1024B. MN-major: LBO = byte distance between
// 64-element (bf16) MN blocks, SBO = 1024B (8 K-rows).
__device__ __forceinline__ u64 make_desc(u32 saddr, u32 lbo_bytes) {
    return (u64)((saddr >> 4) & 0x3FFF) | ((u64)((lbo_bytes >> 4) & 0x3FFF) << 16) | ((u64)64 << 32) | ((u64)1 << 46) | ((u64)2 << 61);
}
// instruction descriptors: bf16 x bf16 -> f32, M=128, N=128
#define IDESC_KK  ((1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24))                 // A K-major, B K-major
#define IDESC_KMN ((1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | (16u << 17) | (8u << 24))    // A K-major, B MN-major
#define IDESC_MNMN ((1u << 4) | (1u << 7) | (1u << 10) | (1u << 15) | (1u << 16) | (16u << 17) | (8u << 24)) // both MN-major

__device__ __forceinline__ void mma_ss(u32 d_tmem, u64 adesc, u64 bdesc, u32 idesc, u32 accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
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
    u32 r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ float bf16_to_f32(u16 x) { return __uint_as_float(((u32)x) << 16); }

// bf16 SW128 tile image [128 rows][128 cols] (2 col-blocks of 64): byte offset of (row, col).
// The same image is a K-major operand (rows = M/N, cols = K) and an MN-major operand (rows = K, cols = M/N).
__device__ __forceinline__ u32 tile_off(int row, int col) {
    return (u32)((col >> 6) * 16384 + row * 128 + ((((col & 63) >> 3) ^ (row & 7)) << 4) + ((col & 7) << 1));
}
__device__ __forceinline__ void mma_ts(u32 d_tmem, u32 a_tmem, u64 bdesc, u32 idesc, u32 accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
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
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ uint4 pack8(const u32* r) {
    return make_uint4(pack_bf16x2(__uint_as_float(r[0]), __uint_as_float(r[1])), pack_bf16x2(__uint_as_float(r[2]), __uint_as_float(r[3])),
                      pack_bf16x2(__uint_as_float(r[4]), __uint_as_float(r[5])), pack_bf16x2(__uint_as_float(r[6]), __uint_as_float(r[7])));
}

struct Smem {
    unsigned char t0[MTILE];     // K image -> K'' (beta*gamma K) -> W image -> M image
    unsigned char t1[MTILE];     // A image -> V image -> V' -> U image -> B image
    unsigned char t2[MTILE];     // T image -> K reload -> K'
    float Adiag[8 * 16 * 16];    // fp32 diagonal 16x16 blocks of A
    float gc[128];
    float beta[128];
    float scal[128];             // gamma_C / gamma_j
    float wsum[8];
    i64 cuseq[64];
    int nch[64];
    int cbase[64];
    short ord[256];              // item index (level-major readiness order) -> global chunk
    u64 bar_load;
    u64 bar_mma;
    u32 tmem_base;
};

extern "C" __global__ void __launch_bounds__(256, 2)
gdn_intra_kernel(const __grid_constant__ TmapT tm_k, const __grid_constant__ TmapT tm_v,
                 const __grid_constant__ TmapT tm_w, const __grid_constant__ TmapT tm_u,
                 const u16* __restrict__ a_raw, const u16* __restrict__ b_raw,
                 const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                 const i64* __restrict__ cu_seqlens, int num_seqs,
                 u16* __restrict__ m_tiles, u16* __restrict__ b_tiles,
                 u16* __restrict__ w_ptr, u16* __restrict__ u_ptr, u32* __restrict__ ready, u32 epoch,
                 i64* __restrict__ dbg, float* __restrict__ dbgT)
{
    // Programmatic dependent launch: the tail kernel may be launched as soon as every CTA of this grid has started; it
    // consumes this CTA's results only after the per-(chunk, head) ready flag (value = epoch) published at the end.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#define STAMP(n) do { if (dbg && tid == 0) dbg[blockIdx.x * 16 + (n)] = clock64(); } while (0)
#define DUMP_TILE(ptr, slot) do { if (dbgT) { __syncthreads(); for (int e = threadIdx.x; e < 128 * 128; e += 256) { int r_ = e / 128, c_ = e % 128; \
        dbgT[((size_t)blockIdx.x * 6 + (slot)) * 16384 + e] = bf16_to_f32(*reinterpret_cast<const u16*>((ptr) + tile_off(r_, c_))); } __syncthreads(); } } while (0)
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    u32 base = smem_u32(smem_raw);
    u32 pad = (1024u - (base & 1023u)) & 1023u;
    Smem& S = *reinterpret_cast<Smem*>(smem_raw + pad);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = tid & 127;          // TMEM lane / matrix row handled by this thread
    const int half = tid >> 7;          // column half [64*half, 64*half+64)
    const int h = blockIdx.x % NH;
    const int item = blockIdx.x / NH;
    const int hq = h / (NH / HQ);

    if (tid <= num_seqs && tid < 64) S.cuseq[tid] = cu_seqlens[tid];
    const float h_Alog = __expf(A_log[h]);
    const float h_dt = dt_bias[h];
    if (tid == 0) {
        mbar_init(smem_u32(&S.bar_load), 1);
        mbar_init(smem_u32(&S.bar_mma), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(smem_u32(&S.tmem_base)) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    // CTA -> chunk: level-major readiness order (all local-0 chunks, then local-1, ...), the order in which the tail kernel
    // needs them (identity order if there are more than 256 chunks)
    if (tid < num_seqs) S.nch[tid] = (int)((S.cuseq[tid + 1] - S.cuseq[tid] + C - 1) / C);
    __syncthreads();
    int total_chunks = 0;
    for (int s2 = 0; s2 < num_seqs; ++s2) total_chunks += S.nch[s2];
    if (tid < num_seqs) { int b = 0; for (int s2 = 0; s2 < tid; ++s2) b += S.nch[s2]; S.cbase[tid] = b; }
    __syncthreads();
    if (tid < total_chunks && tid < 256) {
        int s2 = 0;
        while (s2 + 1 < num_seqs && S.cbase[s2 + 1] <= tid) ++s2;
        const int L = tid - S.cbase[s2];
        int pos = 0;
        for (int s3 = 0; s3 < num_seqs; ++s3) { const int n3 = S.nch[s3]; pos += min(n3, L) + ((s3 < s2 && n3 > L) ? 1 : 0); }
        if (pos < 256) S.ord[pos] = (short)tid;
    }
    __syncthreads();
    int seq = -1, local = 0, cglob = -1; i64 seq_start = 0, seq_end = 0;
    if (item < total_chunks) {
        cglob = (total_chunks <= 256) ? (int)S.ord[item] : item;
        int s2 = 0;
        while (s2 + 1 < num_seqs && S.cbase[s2 + 1] <= cglob) ++s2;
        seq = s2; local = cglob - S.cbase[s2]; seq_start = S.cuseq[s2]; seq_end = S.cuseq[s2 + 1];
    }
    // issue the gate-input loads early (their latency overlaps the TMA setup)
    u16 a_val = 0, b_val = 0;
    if (seq >= 0 && tid < 128) {
        const int t = (int)seq_start + local * C + tid;
        if (t < (int)seq_end) { a_val = a_raw[(size_t)t * NH + h]; b_val = b_raw[(size_t)t * NH + h]; }
    }
    const u32 tmem = S.tmem_base;
    const u32 R0 = tmem, R1 = tmem + 128;
    const u32 lane_addr = (u32)(((warp & 3) * 32) << 16);
    const u32 col0 = (u32)(half * 64);
    u32 mma_phase = 0, load_phase = 0;
    if (seq < 0) {
        __syncthreads();
        if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem) : "memory");
        return;
    }
    STAMP(0);
    const int t0 = (int)seq_start + local * C;
    const int nvalid = min(C, (int)(seq_end - t0));

    // ---- TMA load of the K tile into t0
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load), MTILE);
        tma_load_3d(smem_u32(S.t0), &tm_k, 0, hq, t0, smem_u32(&S.bar_load));
        tma_load_3d(smem_u32(S.t0) + 16384, &tm_k, 64, hq, t0, smem_u32(&S.bar_load));
    }
    // ---- gates (thread = token j, first 128 threads)
    if (tid < 128) {
        const bool valid = tid < nvalid;
        float ar = bf16_to_f32(a_val);
        float br = bf16_to_f32(b_val);
        float x = ar + h_dt;
        float sp = x > 20.f ? x : log1pf(__expf(x));
        float glog = valid ? -h_Alog * sp : 0.f;
        float bt = 1.f / (1.f + __expf(-br));
        float v = glog;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { float n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
        if (lane == 31) S.wsum[warp] = v;
        S.beta[tid] = bt;
        S.gc[tid] = v;
    }
    __syncthreads();
    if (tid < 128) {
        float pre = 0.f;
        for (int w = 0; w < warp; ++w) pre += S.wsum[w];
        S.gc[tid] += pre;
    }
    __syncthreads();
    if (tid < 128) S.scal[tid] = __expf(S.gc[C - 1] - S.gc[tid]);
    // zero the T tile (t2) while waiting for the load
    {
        uint4* z = reinterpret_cast<uint4*>(S.t2);
        #pragma unroll
        for (int i = tid; i < MTILE / 16; i += 256) z[i] = make_uint4(0, 0, 0, 0);
    }
    STAMP(1);
    mbar_wait(smem_u32(&S.bar_load), load_phase & 1); ++load_phase;
    STAMP(2);
    if (nvalid < C) {   // zero padding rows of K
        for (int idx = tid; idx < (C - nvalid) * 16; idx += 256) {
            int r = nvalid + idx / 16, ch = idx % 16;
            *reinterpret_cast<uint4*>(S.t0 + tile_off(r, ch * 8)) = make_uint4(0, 0, 0, 0);
        }
    }
    fence_async_shared();
    __syncthreads();

    // ---- KK^T -> R0
    if (tid == 0) {
        tc_fence_after();
        const u32 kb = smem_u32(S.t0);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            u64 dsc = make_desc(kb + (s >> 2) * 16384 + (s & 3) * 32, 16);
            mma_ss(R0, dsc, dsc, IDESC_KK, s > 0 ? 1u : 0u);
        }
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    STAMP(3);

    // ---- A = tril_strict(beta_i exp(gc_i - gc_j) KK[i][j]) as bf16 image -> t1 (thread = row i, half)
    {
        const int i = row;
        const float bi = S.beta[i], gi = S.gc[i];
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 r[32];
            tmem_ld32(R0 + lane_addr + col0 + part * 32, r);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                float a[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = (int)col0 + part * 32 + ch * 8 + e;
                    const float dg = fminf(gi - S.gc[j], 0.f);
                    const float val = bi * __expf(dg) * __uint_as_float(r[ch * 8 + e]);
                    a[e] = (j < i) ? val : 0.f;
                }
                const int j0 = (int)col0 + part * 32 + ch * 8;
                if ((j0 >> 4) == (i >> 4)) {   // this 8-chunk lies in row i's diagonal 16-block
                    float* ad = S.Adiag + (i >> 4) * 256 + (i & 15) * 16 + (j0 & 15);
                    *reinterpret_cast<float4*>(ad) = make_float4(a[0], a[1], a[2], a[3]);
                    *reinterpret_cast<float4*>(ad + 4) = make_float4(a[4], a[5], a[6], a[7]);
                }
                uint4 pa = make_uint4(pack_bf16x2(a[0], a[1]), pack_bf16x2(a[2], a[3]), pack_bf16x2(a[4], a[5]), pack_bf16x2(a[6], a[7]));
                *reinterpret_cast<uint4*>(S.t1 + tile_off(i, j0)) = pa;
            }
        }
    }
    __syncthreads();
    STAMP(4);

    // ---- T16: eight 16x16 unit-lower-triangular blocks (lane = column, 2 blocks per warp), fp32 Adiag
    if (tid < 128) {
        const int blk = tid >> 4;
        const int j = tid & 15;
        const int r0 = blk * 16;
        const float* Ab = S.Adiag + blk * 256;
        float tcol[16];
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            float acc = (s == j) ? 1.f : 0.f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                if (k < s) { float term = Ab[s * 16 + k] * tcol[k]; acc -= (k >= j) ? term : 0.f; }
            }
            tcol[s] = (s >= j) ? acc : 0.f;
        }
        #pragma unroll
        for (int s = 0; s < 16; ++s)
            *reinterpret_cast<u16*>(S.t2 + tile_off(r0 + s, r0 + j)) = (u16)(pack_bf16x2(tcol[s], 0.f) & 0xFFFFu);
    }
    fence_async_shared();
    __syncthreads();
    STAMP(5);
    DUMP_TILE(S.t1, 0);
    DUMP_TILE(S.t2, 1);

    // ---- merge levels: X = mask_S(T A) (TMEM, packed bf16), T_2S = T - X T
    #pragma unroll 1
    for (int lvl = 0; lvl < 3; ++lvl) {
        const int Sz = 16 << lvl;
        if (tid == 0) {
            tc_fence_after();
            const u32 tb = smem_u32(S.t2), ab = smem_u32(S.t1);
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                u64 ad = make_desc(tb + (s >> 2) * 16384 + (s & 3) * 32, 16);   // A = T K-major (rows i, K = l)
                u64 bd = make_desc(ab + s * 2048, 16384);                        // B = A MN-major (rows l = K, cols k = N)
                mma_ss(R0, ad, bd, IDESC_KMN, s > 0 ? 1u : 0u);
            }
            tc_commit(smem_u32(&S.bar_mma));
        }
        mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
        tc_fence_after();
        {   // masked readout of X[i][k]: keep i in odd S-blocks, k in the preceding S-block; pack bf16 into R1 cols [col0/2, +32)
            const int i = row;
            const int ib = i / Sz;
            const bool keep_row = (ib & 1) == 1;
            u32 pk[32];
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                u32 r[32];
                tmem_ld32(R0 + lane_addr + col0 + part * 32, r);
                tmem_wait_ld();
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int k = (int)col0 + part * 32 + 2 * e;
                    const bool keep = keep_row && ((k / Sz) == ib - 1);
                    const u32 v = pack_bf16x2(__uint_as_float(r[2 * e]), __uint_as_float(r[2 * e + 1]));
                    pk[part * 16 + e] = keep ? v : 0u;
                }
            }
            tmem_st32(R1 + lane_addr + col0 / 2, pk);
            tmem_wait_st();
        }
        tc_fence_before();
        __syncthreads();
        if (tid == 0) {
            tc_fence_after();
            const u32 tb = smem_u32(S.t2);
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                u64 bd = make_desc(tb + s * 2048, 16384);                        // B = T MN-major (rows k = K, cols j = N)
                mma_ts(R0, R1 + 8 * s, bd, IDESC_KMN, s > 0 ? 1u : 0u);         // A = X from TMEM (bf16 packed)
            }
            tc_commit(smem_u32(&S.bar_mma));
        }
        mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
        tc_fence_after();
        {   // T_new = T - D2 (own half-row), bf16 in place
            const int i = row;
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                u32 r[32];
                tmem_ld32(R0 + lane_addr + col0 + part * 32, r);
                tmem_wait_ld();
                #pragma unroll
                for (int ch = 0; ch < 4; ++ch) {
                    const int j0 = (int)col0 + part * 32 + ch * 8;
                    uint4* pt = reinterpret_cast<uint4*>(S.t2 + tile_off(i, j0));
                    uint4 t = *pt;
                    u32 tw[4] = {t.x, t.y, t.z, t.w};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float lo = __uint_as_float(tw[e] << 16) - __uint_as_float(r[ch * 8 + 2 * e]);
                        float hi = __uint_as_float(tw[e] & 0xFFFF0000u) - __uint_as_float(r[ch * 8 + 2 * e + 1]);
                        tw[e] = pack_bf16x2(lo, hi);
                    }
                    *pt = make_uint4(tw[0], tw[1], tw[2], tw[3]);
                }
            }
        }
        fence_async_shared();
        __syncthreads();
        if (lvl == 2) DUMP_TILE(S.t2, 5);
    }
    STAMP(6);

    // ---- V tile -> t1 (A dead); K'' = beta*gamma K in place (t0); V' = beta V in place
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load), MTILE);
        tma_load_3d(smem_u32(S.t1), &tm_v, 0, h, t0, smem_u32(&S.bar_load));
        tma_load_3d(smem_u32(S.t1) + 16384, &tm_v, 64, h, t0, smem_u32(&S.bar_load));
    }
    {
        const int j = row;
        const float sk = S.beta[j] * __expf(S.gc[j]);
        #pragma unroll 2
        for (int ch = 0; ch < 8; ++ch) {
            const u32 off = tile_off(j, (int)col0 + ch * 8);
            uint4 x = *reinterpret_cast<uint4*>(S.t0 + off);
            u32 w[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                w[e] = pack_bf16x2(__uint_as_float(w[e] << 16) * sk, __uint_as_float(w[e] & 0xFFFF0000u) * sk);
            *reinterpret_cast<uint4*>(S.t0 + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    mbar_wait(smem_u32(&S.bar_load), load_phase & 1); ++load_phase;
    {
        const int j = row;
        const float sv = (j < nvalid) ? S.beta[j] : 0.f;
        #pragma unroll 2
        for (int ch = 0; ch < 8; ++ch) {
            const u32 off = tile_off(j, (int)col0 + ch * 8);
            uint4 x = *reinterpret_cast<uint4*>(S.t1 + off);
            u32 w[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                w[e] = pack_bf16x2(__uint_as_float(w[e] << 16) * sv, __uint_as_float(w[e] & 0xFFFF0000u) * sv);
            *reinterpret_cast<uint4*>(S.t1 + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    fence_async_shared();
    __syncthreads();
    STAMP(7);

    // ---- W = T K'' -> R0 ; U = T V' -> R1   (A = T K-major, B = tile MN-major)
    if (tid == 0) {
        tc_fence_after();
        const u32 tb = smem_u32(S.t2), kb = smem_u32(S.t0), vb = smem_u32(S.t1);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            u64 ad = make_desc(tb + (s >> 2) * 16384 + (s & 3) * 32, 16);
            mma_ss(R0, ad, make_desc(kb + s * 2048, 16384), IDESC_KMN, s > 0 ? 1u : 0u);
        }
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            u64 ad = make_desc(tb + (s >> 2) * 16384 + (s & 3) * 32, 16);
            mma_ss(R1, ad, make_desc(vb + s * 2048, 16384), IDESC_KMN, s > 0 ? 1u : 0u);
        }
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    STAMP(8);
    // K reload into t2 (T dead now)
    if (tid == 0) {
        mbar_expect_tx(smem_u32(&S.bar_load), MTILE);
        tma_load_3d(smem_u32(S.t2), &tm_k, 0, hq, t0, smem_u32(&S.bar_load));
        tma_load_3d(smem_u32(S.t2) + 16384, &tm_k, 64, hq, t0, smem_u32(&S.bar_load));
    }
    // ---- readout W -> t0 image, U -> t1 image (thread = token row j, half)
    unsigned char* Wt = S.t0;
    unsigned char* Ut = S.t1;
    {
        const int j = row;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 rw[32], ru[32];
            tmem_ld32(R0 + lane_addr + col0 + part * 32, rw);
            tmem_ld32(R1 + lane_addr + col0 + part * 32, ru);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                const u32 off = tile_off(j, (int)col0 + part * 32 + ch * 8);
                *reinterpret_cast<uint4*>(Wt + off) = pack8(rw + ch * 8);
                *reinterpret_cast<uint4*>(Ut + off) = pack8(ru + ch * 8);
            }
        }
    }
    fence_async_shared();
    __syncthreads();
    if (tid == 0 && nvalid == C) {
        tma_store_3d(&tm_w, smem_u32(Wt), 0, h, t0);
        tma_store_3d(&tm_w, smem_u32(Wt) + 16384, 64, h, t0);
        tma_store_3d(&tm_u, smem_u32(Ut), 0, h, t0);
        tma_store_3d(&tm_u, smem_u32(Ut) + 16384, 64, h, t0);
        bulk_commit();
    }
    if (nvalid < C) {
        const int j = row;
        if (j < nvalid) {
            uint4* wd = reinterpret_cast<uint4*>(w_ptr + ((size_t)(t0 + j) * NH + h) * D) + half * 8;
            uint4* ud = reinterpret_cast<uint4*>(u_ptr + ((size_t)(t0 + j) * NH + h) * D) + half * 8;
            #pragma unroll
            for (int ch = 0; ch < 8; ++ch) {
                const u32 off = tile_off(j, (int)col0 + ch * 8);
                wd[ch] = *reinterpret_cast<const uint4*>(Wt + off);
                ud[ch] = *reinterpret_cast<const uint4*>(Ut + off);
            }
        }
    }
    mbar_wait(smem_u32(&S.bar_load), load_phase & 1); ++load_phase;
    {   // K' = (gamma_C/gamma_j) K in place (t2), padding rows zero
        const int j = row;
        const float sc = (j < nvalid) ? S.scal[j] : 0.f;
        #pragma unroll 2
        for (int ch = 0; ch < 8; ++ch) {
            const u32 off = tile_off(j, (int)col0 + ch * 8);
            uint4 x = *reinterpret_cast<uint4*>(S.t2 + off);
            u32 w[4] = {x.x, x.y, x.z, x.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                w[e] = pack_bf16x2(__uint_as_float(w[e] << 16) * sc, __uint_as_float(w[e] & 0xFFFF0000u) * sc);
            *reinterpret_cast<uint4*>(S.t2 + off) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    fence_async_shared();
    __syncthreads();
    STAMP(9);
    // ---- M = gamma_C I - K'^T W -> R0 ; B = K'^T U -> R1  (A = K' MN-major, B = W/U MN-major)
    if (tid == 0) {
        tc_fence_after();
        const u32 kb = smem_u32(S.t2), wb = smem_u32(Wt), ub = smem_u32(Ut);
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            mma_ss(R0, make_desc(kb + s * 2048, 16384), make_desc(wb + s * 2048, 16384), IDESC_MNMN, s > 0 ? 1u : 0u);
        #pragma unroll   // B^T = U^T K' : A = U image (MN-major, M = v), B = K' image (MN-major, N = k) -> lanes = v
        for (int s = 0; s < 8; ++s)
            mma_ss(R1, make_desc(ub + s * 2048, 16384), make_desc(kb + s * 2048, 16384), IDESC_MNMN, s > 0 ? 1u : 0u);
        tc_commit(smem_u32(&S.bar_mma));
    }
    mbar_wait(smem_u32(&S.bar_mma), mma_phase & 1); ++mma_phase;
    tc_fence_after();
    if (tid == 0) bulk_wait_read();   // TMA stores of W/U must have finished reading t0/t1 before they are overwritten
    __syncthreads();
    STAMP(10);
    // ---- readout M (R0, lane = k_out) -> t0 image [k_out][k_in], B^T (R1, lane = v) -> t1 image [v][k] -> bulk store
    {
        const int i = row;
        const float gC = __expf(S.gc[C - 1]);
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            u32 rm[32], rb[32];
            tmem_ld32(R0 + lane_addr + col0 + part * 32, rm);
            tmem_ld32(R1 + lane_addr + col0 + part * 32, rb);
            tmem_wait_ld();
            #pragma unroll
            for (int ch = 0; ch < 4; ++ch) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = (int)col0 + part * 32 + ch * 8 + e;
                    rm[ch * 8 + e] = __float_as_uint((i == j ? gC : 0.f) - __uint_as_float(rm[ch * 8 + e]));
                }
                const u32 off = tile_off(i, (int)col0 + part * 32 + ch * 8);
                *reinterpret_cast<uint4*>(S.t0 + off) = pack8(rm + ch * 8);
                *reinterpret_cast<uint4*>(S.t1 + off) = pack8(rb + ch * 8);
            }
        }
    }
    fence_async_shared();
    __syncthreads();
    if (tid == 0) {
        bulk_s2g(m_tiles + ((size_t)cglob * NH + h) * (MTILE / 2), smem_u32(S.t0), MTILE);
        bulk_s2g(b_tiles + ((size_t)cglob * NH + h) * (MTILE / 2), smem_u32(S.t1), MTILE);
        bulk_commit();
        bulk_wait_all();                 // M, B (and the W, U TMA stores) performed at L2
        if (ready) {
            if (nvalid < C) __threadfence();   // partial chunk: W/U were written with generic stores by all threads (ordered by the barriers above)
            asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ready + (size_t)cglob * NH + h), "r"(epoch) : "memory");
        }
    }
    __syncthreads();
    STAMP(11);
    if (warp == 0) {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(tmem) : "memory");
    }
}
"""


def _check(err, what=""):
    code = err[0] if isinstance(err, tuple) else err
    if int(code) != 0:
        raise RuntimeError(f"CUDA error {code} in {what}")


@functools.lru_cache(maxsize=None)
def _build():
    err, prog = nvrtc.nvrtcCreateProgram(KERNEL_SRC.encode(), b"gdn_intra.cu", 0, [], [])
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
    if os.environ.get("INTRA_DUMP_CUBIN"):
        open(os.environ["INTRA_DUMP_CUBIN"], "wb").write(cubin)
    torch.empty(1, device="cuda")
    _check(cu.cuInit(0), "cuInit")
    err, module = cu.cuModuleLoadData(cubin)
    _check(err, "cuModuleLoadData")
    err, func = cu.cuModuleGetFunction(module, b"gdn_intra_kernel")
    _check(err, "cuModuleGetFunction")
    smem_bytes = 3 * MTILE_BYTES + 8 * 1024 + 4 * 512 + 512 + 64 + 64 * 8 + 512 + 1024
    _check(cu.cuFuncSetAttribute(func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes),
           "cuFuncSetAttribute")
    return module, func, smem_bytes


class TmapCT(ctypes.Structure):
    _fields_ = [("d", ctypes.c_uint64 * 16)]


_TMAP_CACHE = {}


def make_tmap_3d(t, nheads):
    """3D bf16 tensor map over a [T][nheads][128] tensor: dims (128, nheads, T), box (64, 1, 128), SW128.
    Cached by (data pointer, shape): encoding costs several microseconds of host time per map."""
    key = (t.data_ptr(), tuple(t.shape), nheads)
    tm = _TMAP_CACHE.get(key)
    if tm is None:
        tm = _make_tmap_3d(t, nheads)
        if len(_TMAP_CACHE) > 256:
            _TMAP_CACHE.clear()
        _TMAP_CACHE[key] = tm
    return tm


def _make_tmap_3d(t, nheads):
    T = t.shape[0]
    dims = [cu.cuuint64_t(x) for x in (128, nheads, T)]
    strides = [cu.cuuint64_t(x) for x in (128 * 2, nheads * 128 * 2)]
    box = [cu.cuuint32_t(x) for x in (64, 1, 128)]
    estr = [cu.cuuint32_t(x) for x in (1, 1, 1)]
    err, tm = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), dims, strides, box, estr,
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B, cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    _check(err, "cuTensorMapEncodeTiled")
    return TmapCT.from_buffer_copy(ctypes.string_at(int(tm.getPtr()), 128))


def intra_launch(q, k, v, a, b, A_log, dt_bias, cu_seqlens, W, U, m_tiles, b_tiles, max_chunks, ready=None, epoch=1, dbg=None, dbgT=None):
    """ready: optional [max_chunks*8] uint32 flags set to `epoch` per (chunk, head) once its W/U/M/B are in global memory."""
    _, func, smem_bytes = _build()
    num_seqs = cu_seqlens.shape[0] - 1
    tm_k = make_tmap_3d(k, 4)
    tm_v = make_tmap_3d(v, 8)
    tm_w = make_tmap_3d(W, 8)
    tm_u = make_tmap_3d(U, 8)
    stream = torch.cuda.current_stream().cuda_stream
    cargs = [tm_k, tm_v, tm_w, tm_u, ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()),
             ctypes.c_void_p(A_log.data_ptr()), ctypes.c_void_p(dt_bias.data_ptr()), ctypes.c_void_p(cu_seqlens.data_ptr()),
             ctypes.c_int(num_seqs), ctypes.c_void_p(m_tiles.data_ptr()), ctypes.c_void_p(b_tiles.data_ptr()),
             ctypes.c_void_p(W.data_ptr()), ctypes.c_void_p(U.data_ptr()),
             ctypes.c_void_p(ready.data_ptr() if ready is not None else 0), ctypes.c_uint32(int(epoch) & 0xFFFFFFFF),
             ctypes.c_void_p(dbg.data_ptr() if dbg is not None else 0), ctypes.c_void_p(dbgT.data_ptr() if dbgT is not None else 0)]
    ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.addressof(x) for x in cargs])
    _check(cu.cuLaunchKernel(func, max_chunks * 8, 1, 1, 256, 1, 1, smem_bytes, stream,
                             ctypes.addressof(ptrs), 0), "cuLaunchKernel")
