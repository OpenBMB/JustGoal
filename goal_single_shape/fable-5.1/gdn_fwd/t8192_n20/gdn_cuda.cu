// GDN (Gated DeltaNet) prefill forward, varlen + GQA. Written from scratch (mma.sync / ldmatrix / cp.async / TMA bulk).
//
// Chunked WY formulation (chunk BT = 64 tokens). Per chunk and v-head:
//   L_ij = beta_i * exp(G_i - G_j) * (k_i . k_j)   (i > j),   solve (I + L) [W | U] = [beta*exp(G)*K | beta*V]
//   Aqk_ij = scale * exp(G_i - G_j) * (q_i . k_j)  (i >= j)
// Sequential over chunks (S is [K, V]; kernels work with S^T [V, K]):
//   R = U - W S0,  S1 = exp(G_last) S0 + (exp(G_last - G) K)^T R          (recurrence kernel, stores S0^T, R^T)
//   O = scale*exp(G)*Q S0 + Aqk R                                        (output kernel, fully parallel)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace {
#ifdef GDN_INTRA_NOSTORE
#define NOSTORE_GUARD(x) do { asm volatile("" :: "r"(0)); } while (0)
#else
#define NOSTORE_GUARD(x) x
#endif

constexpr int BT = 64;
constexpr int DK = 128;
constexpr int DV = 128;
constexpr int KPAD = DK + 8;   // bf16 row stride of padded K/Q/W/S tiles
constexpr int APAD = BT + 8;   // bf16 row stride of padded 64-col tiles
constexpr int XPAD = 264;      // fp32 row stride of X (=[W|U] solve buffer)
constexpr int LPAD = 68;       // fp32 row stride of L
constexpr int MAXSEQ = 32;
constexpr int VBR = 16;        // v-rows per recurrence CTA
constexpr int NVB = DV / VBR;  // v-blocks per head
constexpr int CL = 4;          // recurrence cluster size (CTAs per cluster); an item covers CL v-blocks of one head
constexpr int CROWS_ITEM = 16; // v-rows per recurrence CTA (== NGRP * VBR)
constexpr int NHALF = DV / (CL * CROWS_ITEM); // items per (sequence, head)
constexpr int UT_STRIDE = 72;  // bf16 row stride of stored U^T slices [VBR v][72]
constexpr int GT = 208;        // fp32 entries of the per-chunk gate tile: G[64], dtok[64], stok[64], gamma, pad

// ---- intra kernel smem layout (bytes); one CTA handles the two v-heads of one q/k head ----
constexpr int TPAD = 68;                              // fp32 row stride of L/T
constexpr int ASTG = 72;                              // bf16 row stride of the Aqk^T staging
constexpr int I_KS = 0;                               // [64][KPAD] bf16: K
constexpr int I_QS = I_KS + BT * KPAD * 2;            // [64][KPAD] bf16: Q (later V)
constexpr int I_LT = I_QS + BT * KPAD * 2;            // [64][TPAD] fp32: L, then T; later W / U^T staging (16KB, global tile layout)
constexpr int I_GG = I_LT + BT * TPAD * 4;            // g, G, beta, sw = beta*e^G, su = beta (5 x 64 fp32)
constexpr int I_TW = I_GG + 5 * BT * 4;               // [64][APAD] bf16: Tb_w = bf16(T diag(sw)); early: Aqk^T staging (8KB, global layout)
constexpr int I_TU = I_TW + BT * APAD * 2;            // [64][APAD] bf16: Tb_u = bf16(T diag(su)); early: doubling scratch (4 x 1KB) / level-1 P (2 x 2KB) / level-2 P (4KB)
constexpr int INTRA_SMEM = I_TU + BT * APAD * 2;      // 71936

__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void bulk_store_1d(void* gdst, uint32_t ssrc, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void proxy_fence_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// ---- mbarrier + TMA bulk (1D) ----
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(addr) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int count) { __syncwarp(); asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT;\n"
        "}\n" ::"r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ uint64_t l2_policy_evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p)); return p; }
__device__ __forceinline__ void tma_load_1d_hint(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar, uint64_t policy) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar), "l"(policy) : "memory");
}
__device__ __forceinline__ void tma_load_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}

__device__ __forceinline__ void tma_load_1d_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar, uint16_t mask) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar), "h"(mask) : "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync_all() {
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_addr) {   // cta-scope release (cheap): used for stage-consumed signals
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote_cluster(uint32_t cluster_addr) {   // cluster-scope release: used to publish data written to remote smem
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t addr, uint32_t parity) {   // cluster-scope acquire
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAITC:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAITC;\n"
        "}\n" ::"r"(addr), "r"(parity) : "memory");
}
__device__ __forceinline__ int ld_shared_cluster_s32(uint32_t cluster_addr) {
    int v; asm volatile("ld.shared::cluster.s32 %0, [%1];\n" : "=r"(v) : "r"(cluster_addr) : "memory"); return v;
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_relaxed(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void spin_wait_ge(const int* p, int target, int site = 0) {
    unsigned iters = 0;
    while (ld_acquire(p) < target) {
        __nanosleep(64);
        if (++iters > (1u << 24)) {   // ~1s: something is wrong, fail loudly instead of hanging
            printf("GDN spin timeout: site %d block %d thread %d target %d value %d\n", site, blockIdx.x, threadIdx.x, target, ld_acquire(p));
            __trap();
        }
    }
}

// Map a chunk position b (in "chunk-index-major" order: all sequences' chunk 0, then chunk 1, ...) to
// (global chunk id ci, token start t0, valid tokens). 256 threads; returns false when b is out of range.
// C[16][16] (+)= (sa*A + ia*I) * (B + ib*I) with tf32 mma; A,B,C fp32 row-major in smem. One warp. In-place safe.
__device__ __forceinline__ void mm16_tf32(const float* A, int lda, float sa, bool ia, const float* B, int ldb, bool ib,
                                          float* C, int ldc, bool accum, int g, int t4) {
    float acc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[nt][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
        const int k0 = kk * 8;
        float a0 = sa * A[g * lda + k0 + t4], a1 = sa * A[(g + 8) * lda + k0 + t4];
        float a2 = sa * A[g * lda + k0 + t4 + 4], a3 = sa * A[(g + 8) * lda + k0 + t4 + 4];
        if (ia) { a0 += (g == k0 + t4); a1 += (g + 8 == k0 + t4); a2 += (g == k0 + t4 + 4); a3 += (g + 8 == k0 + t4 + 4); }
        const uint32_t af[4] = {to_tf32(a0), to_tf32(a1), to_tf32(a2), to_tf32(a3)};
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const int n0 = nt * 8;
            float b0 = B[(k0 + t4) * ldb + n0 + g], b1 = B[(k0 + t4 + 4) * ldb + n0 + g];
            if (ib) { b0 += (k0 + t4 == n0 + g); b1 += (k0 + t4 + 4 == n0 + g); }
            const uint32_t bf[2] = {to_tf32(b0), to_tf32(b1)};
            mma_tf32(acc[nt], af, bf);
        }
    }
    if (accum) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const float2 lo = *reinterpret_cast<const float2*>(C + g * ldc + nt * 8 + 2 * t4);
            const float2 hi = *reinterpret_cast<const float2*>(C + (g + 8) * ldc + nt * 8 + 2 * t4);
            acc[nt][0] += lo.x; acc[nt][1] += lo.y; acc[nt][2] += hi.x; acc[nt][3] += hi.y;
        }
    }
    __syncwarp();
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        *reinterpret_cast<float2*>(C + g * ldc + nt * 8 + 2 * t4) = make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(C + (g + 8) * ldc + nt * 8 + 2 * t4) = make_float2(acc[nt][2], acc[nt][3]);
    }
    __syncwarp();
}

// swizzled 16B-chunk index (128B swizzle: chunk ^ (row & 7))
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ (row & 7); }
// element offset of (row, col) in a [rows][128 bf16] tile stored as two SW128 halves of [rows][64]
__device__ __forceinline__ int tile128_off(int rows, int row, int col) { return (col >> 6) * (rows * 64) + row * 64 + (((col >> 3) & 7) ^ (row & 7)) * 8 + (col & 7); }

// ============================================================================================
// Intra-chunk kernel: one CTA per (chunk, v-head), 256 threads.
// Outputs: W, -K~, Q~, -U^T, -Aqk^T tiles (global layouts, bulk-stored from smem staging), G tile, flag.
// ============================================================================================
__global__ void __launch_bounds__(256, 2)
gdn_intra_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                 const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ a_in,
                 const __nv_bfloat16* __restrict__ b_in, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, const int64_t* __restrict__ cu,
                 __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ kc_out, __nv_bfloat16* __restrict__ qc_out,
                 __nv_bfloat16* __restrict__ ut_out, __nv_bfloat16* __restrict__ aqk_out,
                 float* __restrict__ gc_out, int* __restrict__ flag, int epoch, float scale, int N, int HK, int HV, long long* __restrict__ dbg) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int meta[4];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem + I_KS);
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem + I_QS);
    float* LT = reinterpret_cast<float*>(smem + I_LT);
    __nv_bfloat16* Wst = reinterpret_cast<__nv_bfloat16*>(smem + I_LT);   // W / U^T staging (aliases LT)
    float* gg = reinterpret_cast<float*>(smem + I_GG);               // g | G | beta | sw | su
    __nv_bfloat16* Tw = reinterpret_cast<__nv_bfloat16*>(smem + I_TW);
    __nv_bfloat16* Tu = reinterpret_cast<__nv_bfloat16*>(smem + I_TU);
    __nv_bfloat16* Astg = reinterpret_cast<__nv_bfloat16*>(smem + I_TW);  // -Aqk^T staging (aliases Tw), global layout
    float* dbl = reinterpret_cast<float*>(smem + I_TU);              // diagonal-solve scratch (aliases Tu): block b -> 256 floats; level-1 P: warp w -> 512 floats
    float* Ps = reinterpret_cast<float*>(smem + I_TU);               // level-2 scratch (aliases Tu): [32][32]
#define DBG_I(i) if (dbg != nullptr && blockIdx.x == 0 && tid == 0) dbg[i] = clock64();

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int rep = HV / HK;               // == 2
    const int h = blockIdx.x % HV, hk = h / rep;
    DBG_I(0)
    if (dbg != nullptr && tid == 0 && blockIdx.x < 512) dbg[20000 + blockIdx.x * 2] = gtimer();
    if (warp == 0) {   // ---- chunk slot b (rank-major: longest sequence first, chunks ascending) -> (global chunk id, token start, valid tokens) ----
        const int b = blockIdx.x / HV;
        const int c_lo = (lane <= N) ? (int)cu[lane] : 0;
        const int c32 = (N >= 32) ? (int)cu[32] : 0;
        int c_hi = __shfl_down_sync(0xffffffffu, c_lo, 1);
        if (lane == 31) c_hi = c32;
        const int len_n = (lane < N) ? c_hi - c_lo : 0, nch_n = (len_n + BT - 1) / BT;   // lane = sequence n
        int rank = 0, cb = 0;
        for (int j = 0; j < N; j++) {
            const int lj = __shfl_sync(0xffffffffu, len_n, j);
            rank += (lj > len_n) || (lj == len_n && j < lane);
            if (j < lane) cb += (lj + BT - 1) / BT;
        }
        int n_r = 0;   // lane = rank r: the sequence with that rank
        for (int j = 0; j < N; j++) if (__shfl_sync(0xffffffffu, rank, j) == lane) n_r = j;
        const int nch_all = __shfl_sync(0xffffffffu, nch_n, n_r);
        const int nch_r = (lane < N) ? nch_all : 0;
        const int cb_r = __shfl_sync(0xffffffffu, cb, n_r), st_r = __shfl_sync(0xffffffffu, c_lo, n_r), len_r = __shfl_sync(0xffffffffu, len_n, n_r);
        int incl = nch_r;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += y; }
        const int excl = incl - nch_r, total = __shfl_sync(0xffffffffu, incl, 31);
        if (lane < N && excl <= b && b < incl) { const int c = b - excl; meta[0] = cb_r + c; meta[1] = st_r + c * BT; meta[2] = min(BT, len_r - c * BT); }
        if (lane == 0) meta[3] = (b < total);
    }
    __syncthreads();
    if (meta[3] == 0) return;
    const int ci = meta[0], t0 = meta[1], nvalid = meta[2];
    const size_t tile_id = (size_t)ci * HV + h;
    DBG_I(13)

    // ---- async loads of K, Q tiles (zero-fill invalid rows) ----
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int idx = tid + i * 256, row = idx >> 4, ch = idx & 15;
        const bool ok = row < nvalid;
        const size_t goff = ((size_t)(t0 + (ok ? row : 0)) * HK + hk) * DK + ch * 8;
        cp_async16(smem_u32(Ks + row * KPAD + ch * 8), k + goff, ok);
        cp_async16(smem_u32(Qs + row * KPAD + ch * 8), q + goff, ok);
    }
    cp_async_commit();
    __nv_bfloat16 a_raw = __float2bfloat16(0.f), b_raw = __float2bfloat16(0.f);
    float alog_h = 0.f, dtb_h = 0.f;
    if (tid < BT) {   // gate inputs: issue the loads now, consume after the K/Q wait
        const bool ok = tid < nvalid;
        const size_t idx = (size_t)(t0 + (ok ? tid : 0)) * HV + h;
        a_raw = a_in[idx]; b_raw = b_in[idx]; alog_h = A_log[h]; dtb_h = dt_bias[h];
    }
    DBG_I(14)
    for (int i = tid; i < BT * TPAD / 4; i += 256) reinterpret_cast<float4*>(LT)[i] = make_float4(0.f, 0.f, 0.f, 0.f);   // strictly-upper blocks of T must be zero
    cp_async_wait<0>();
    __syncthreads();
    DBG_I(15)
    if (tid < BT) {   // ---- gates ----
        float gv = 0.f, bv = 0.f;
        if (tid < nvalid) {
            gv = -expf(alog_h) * softplus_f(bf2f(a_raw) + dtb_h);
            bv = 1.f / (1.f + expf(-bf2f(b_raw)));
        }
        gg[tid] = gv;
        gg[2 * BT + tid] = bv;
    }
    __syncthreads();
    if (warp == 0) {   // inclusive scan of g -> G
        float x0 = gg[lane], x1 = gg[lane + 32];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float y0 = __shfl_up_sync(0xffffffffu, x0, o), y1 = __shfl_up_sync(0xffffffffu, x1, o);
            if (lane >= o) { x0 += y0; x1 += y1; }
        }
        const float tot0 = __shfl_sync(0xffffffffu, x0, 31);
        gg[BT + lane] = x0;
        gg[BT + lane + 32] = x1 + tot0;
    }
    DBG_I(16)
    __syncthreads();
    if (tid < BT) {   // column scales for W = T diag(sw) K and U = T diag(su) V
        gg[3 * BT + tid] = gg[2 * BT + tid] * __expf(gg[BT + tid]);
        gg[4 * BT + tid] = gg[2 * BT + tid];
    }
    DBG_I(1)

    // ---- KK^T (warps 0-3) and QK^T (warps 4-7); warp tile = 2 m-tiles x 4 n-tiles, ks-pipelined ----
    {
        const int prod = warp >> 2, mh = (warp >> 1) & 1, nh = warp & 1;
        const uint32_t As = smem_u32(prod ? Qs : Ks);
        const uint32_t Ksa = smem_u32(Ks);
        const uint32_t a_off = ((mh * 32) + (lane & 7) + 8 * ((lane >> 3) & 1)) * KPAD * 2 + 16 * (lane >> 4);
        const uint32_t b_off = ((nh * 32) + (lane & 7) + 8 * (lane >> 4)) * KPAD * 2 + 16 * ((lane >> 3) & 1);
        float acc[2][4][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[mt][j][e] = 0.f;
        uint32_t af[2][2][4], bf[2][2][4];
        ldsm_x4(af[0][0], As + a_off);
        ldsm_x4(af[0][1], As + a_off + 16 * KPAD * 2);
        ldsm_x4(bf[0][0], Ksa + b_off);
        ldsm_x4(bf[0][1], Ksa + b_off + 16 * KPAD * 2);
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int cur = ks & 1, nxt = cur ^ 1;
            if (ks < 7) {
                ldsm_x4(af[nxt][0], As + a_off + (ks + 1) * 32);
                ldsm_x4(af[nxt][1], As + a_off + 16 * KPAD * 2 + (ks + 1) * 32);
                ldsm_x4(bf[nxt][0], Ksa + b_off + (ks + 1) * 32);
                ldsm_x4(bf[nxt][1], Ksa + b_off + 16 * KPAD * 2 + (ks + 1) * 32);
            }
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                mma_bf16(acc[mt][0], af[cur][mt], bf[cur][0]);
                mma_bf16(acc[mt][1], af[cur][mt], bf[cur][0] + 2);
                mma_bf16(acc[mt][2], af[cur][mt], bf[cur][1]);
                mma_bf16(acc[mt][3], af[cur][mt], bf[cur][1] + 2);
            }
        }
        DBG_I(17)
        const float* Gs = gg + BT;
        const float* betas = gg + 2 * BT;
        float gi[4], bi[4], gj[4][2];   // gate values for this thread's rows / column pairs (registers: no smem reloads between stores)
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int half = 0; half < 2; half++) { const int i = (mh * 2 + mt) * 16 + g + 8 * half; gi[mt * 2 + half] = Gs[i]; bi[mt * 2 + half] = betas[i]; }
#pragma unroll
        for (int jn = 0; jn < 4; jn++) { const float2 gp = *reinterpret_cast<const float2*>(Gs + (nh * 4 + jn) * 8 + 2 * t4); gj[jn][0] = gp.x; gj[jn][1] = gp.y; }
        if (warp < 4) {   // L_ij = beta_i e^{G_i - G_j} (k_i . k_j), i > j
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int jn = 0; jn < 4; jn++)
#pragma unroll
                    for (int half = 0; half < 2; half++) {
                        const int i = (mh * 2 + mt) * 16 + g + 8 * half;
                        const int j0 = (nh * 4 + jn) * 8 + 2 * t4;
                        const float v0 = (i > j0) ? bi[mt * 2 + half] * __expf(gi[mt * 2 + half] - gj[jn][0]) * acc[mt][jn][2 * half] : 0.f;
                        const float v1 = (i > j0 + 1) ? bi[mt * 2 + half] * __expf(gi[mt * 2 + half] - gj[jn][1]) * acc[mt][jn][2 * half + 1] : 0.f;
                        *reinterpret_cast<float2*>(LT + i * TPAD + j0) = make_float2(v0, v1);
                    }
        } else {          // -Aqk^T[j][i] = -scale e^{G_i - G_j} (q_i . k_j), i >= j  (global tile layout)
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int jn = 0; jn < 4; jn++)
#pragma unroll
                    for (int half = 0; half < 2; half++) {
                        const int i = (mh * 2 + mt) * 16 + g + 8 * half;
                        const int j0 = (nh * 4 + jn) * 8 + 2 * t4;
                        const bool iv = i < nvalid;
                        const float v0 = (iv && i >= j0) ? scale * __expf(gi[mt * 2 + half] - gj[jn][0]) * acc[mt][jn][2 * half] : 0.f;
                        const float v1 = (iv && i >= j0 + 1) ? scale * __expf(gi[mt * 2 + half] - gj[jn][1]) * acc[mt][jn][2 * half + 1] : 0.f;
                        Astg[j0 * BT + swz(j0, i >> 3) * 8 + (i & 7)] = __float2bfloat16(-v0);
                        Astg[(j0 + 1) * BT + swz(j0 + 1, i >> 3) * 8 + (i & 7)] = __float2bfloat16(-v1);
                    }
        }
    }
    proxy_fence_smem();
    __syncthreads();
    DBG_I(18)
    if (tid == 0) { bulk_store_1d(aqk_out + tile_id * (BT * BT), smem_u32(Astg), BT * BT * 2); bulk_commit(); }
    {   // ---- -K~ = -e^{G_last - G} K and Q~ = scale e^{G} Q tiles (coalesced 128B rows) ----
        const float* Gh = gg + BT;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = tid + i * 256, half = idx >> 9, row = (idx >> 3) & 63, ch = idx & 7;
            const uint4 kval = *reinterpret_cast<const uint4*>(Ks + row * KPAD + half * 64 + ch * 8);
            const uint4 qval = *reinterpret_cast<const uint4*>(Qs + row * KPAD + half * 64 + ch * 8);
            const int toff = half * (BT * 64) + row * 64 + swz(row, ch) * 8;
            const float dk = (row < nvalid) ? -__expf(Gh[BT - 1] - Gh[row]) : 0.f;   // stored negated (the recurrence works with -R)
            const float st = (row < nvalid) ? scale * __expf(Gh[row]) : 0.f;
            uint32_t kv[4] = {kval.x, kval.y, kval.z, kval.w}, qv[4] = {qval.x, qval.y, qval.z, qval.w};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const __nv_bfloat162 pk = *reinterpret_cast<const __nv_bfloat162*>(&kv[e]), pq = *reinterpret_cast<const __nv_bfloat162*>(&qv[e]);
                kv[e] = pack_bf16(dk * __low2float(pk), dk * __high2float(pk));
                qv[e] = pack_bf16(st * __low2float(pq), st * __high2float(pq));
            }
            *reinterpret_cast<uint4*>(kc_out + tile_id * (BT * DK) + toff) = make_uint4(kv[0], kv[1], kv[2], kv[3]);
            *reinterpret_cast<uint4*>(qc_out + tile_id * (BT * DK) + toff) = make_uint4(qv[0], qv[1], qv[2], qv[3]);
        }
    }
    DBG_I(19)
    __syncthreads();   // Q tile consumed -> V may land in Qs
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int idx = tid + i * 256, row = idx >> 4, ch = idx & 15;
        const bool ok = row < nvalid;
        cp_async16(smem_u32(Qs + row * KPAD + ch * 8), v + ((size_t)(t0 + (ok ? row : 0)) * HV + h) * DV + ch * 8, ok);
    }
    cp_async_commit();
    DBG_I(2)
    // ---- diagonal blocks: T_II = (I + L_II)^-1 by column forward substitution (16 threads per block) ----
    if (tid < 64) {
        const int blk = tid >> 4, j = tid & 15;
        float* Lb = LT + (blk * 16) * TPAD + blk * 16;   // L_II, becomes T_II
        float* Tsc = dbl + blk * 256;                     // [16][16] scratch for this block
        float col[16];
#pragma unroll
        for (int kk = 0; kk < 16; kk++) col[kk] = (kk == j) ? 1.f : 0.f;
#pragma unroll
        for (int i = 1; i < 16; i++) {   // T[i][j] = -sum_{k<i} L[i][k] col[k]   (col[k] = 0 for k < j, 1 for k == j)
            float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
            for (int qq = 0; qq < (i + 3) / 4; qq++) {
                const float4 r4 = *reinterpret_cast<const float4*>(Lb + i * TPAD + 4 * qq);
                const float rr[4] = {r4.x, r4.y, r4.z, r4.w};
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int kk = 4 * qq + e;
                    if (kk < i) { if (e & 1) acc1 = fmaf(rr[e], col[kk], acc1); else acc0 = fmaf(rr[e], col[kk], acc0); }
                }
            }
            col[i] = (i == j) ? 1.f : -(acc0 + acc1);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) Tsc[i * 16 + j] = col[i];
        __syncwarp();
#pragma unroll
        for (int qq = 0; qq < 4; qq++) *reinterpret_cast<float4*>(Lb + j * TPAD + 4 * qq) = *reinterpret_cast<const float4*>(Tsc + j * 16 + 4 * qq);
    }
    __syncthreads();
    DBG_I(3)
    // ---- level 1 (2 tasks): T10 = -T11 L10 T00 and T32 = -T33 L32 T22 ----
    if (warp < 2) {
        const int b0 = 2 * warp, b1 = b0 + 1;
        float* P = dbl + warp * 512;
        mm16_tf32(LT + (b1 * 16) * TPAD + b0 * 16, TPAD, 1.f, false, LT + (b0 * 16) * TPAD + b0 * 16, TPAD, false, P, 16, false, g, t4);
        mm16_tf32(LT + (b1 * 16) * TPAD + b1 * 16, TPAD, -1.f, false, P, 16, false, LT + (b1 * 16) * TPAD + b0 * 16, TPAD, false, g, t4);
    }
    __syncthreads();
    DBG_I(4)
    // ---- level 2a (4 tasks): P = L[32:64][0:32] T[0:32][0:32] ----
    if (warp < 4) {
        const int mb = (warp >> 1) & 1, nb = warp & 1;
#pragma unroll
        for (int kb = 0; kb < 2; kb++)
            mm16_tf32(LT + (32 + 16 * mb) * TPAD + 16 * kb, TPAD, 1.f, false, LT + (16 * kb) * TPAD + 16 * nb, TPAD, false,
                      Ps + (16 * mb) * 32 + 16 * nb, 32, kb > 0, g, t4);
    }
    __syncthreads();
    DBG_I(5)
    // ---- level 2b (4 tasks): T[32:64][0:32] = -T[32:64][32:64] P ----
    if (warp < 4) {
        const int mb = (warp >> 1) & 1, nb = warp & 1;
#pragma unroll
        for (int kb = 0; kb < 2; kb++)
            mm16_tf32(LT + (32 + 16 * mb) * TPAD + 32 + 16 * kb, TPAD, -1.f, false, Ps + (16 * kb) * 32 + 16 * nb, 32, false,
                      LT + (32 + 16 * mb) * TPAD + 16 * nb, TPAD, kb > 0, g, t4);
    }
    if (tid == 0) bulk_wait_read();   // -Aqk^T staging (Tw region) drained
    __syncthreads();
    DBG_I(6)
    // ---- Tb_w = bf16(T diag(sw)), Tb_u = bf16(T diag(su)) ----
    auto write_tb = [&](__nv_bfloat16* Tb, int which) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int idx = tid + i * 256, row = idx >> 3, c8 = (idx & 7) * 8;
            const float* scp = gg + which * BT + c8;
            const float* tp = LT + row * TPAD + c8;
            const float4 sa = *reinterpret_cast<const float4*>(scp), sb2 = *reinterpret_cast<const float4*>(scp + 4);
            const float sc[8] = {sa.x, sa.y, sa.z, sa.w, sb2.x, sb2.y, sb2.z, sb2.w};
            const float4 p0 = *reinterpret_cast<const float4*>(tp), p1 = *reinterpret_cast<const float4*>(tp + 4);
            uint4 o;
            o.x = pack_bf16(p0.x * sc[0], p0.y * sc[1]); o.y = pack_bf16(p0.z * sc[2], p0.w * sc[3]);
            o.z = pack_bf16(p1.x * sc[4], p1.y * sc[5]); o.w = pack_bf16(p1.z * sc[6], p1.w * sc[7]);
            *reinterpret_cast<uint4*>(Tb + row * APAD + c8) = o;
        }
    };
    write_tb(Tw, 3);
    write_tb(Tu, 4);
    cp_async_wait<0>();   // V landed
    __syncthreads();      // T (fp32) no longer needed: LT becomes the W staging
    DBG_I(7)
    // ---- W = Tb_w K : warp w -> rows 16(w&3).., columns 64(w>>2).. -> staging (global tile layout) ----
    const uint32_t ksa = smem_u32(Ks), qsa = smem_u32(Qs), twa = smem_u32(Tw), tua = smem_u32(Tu);
    const uint32_t ta_off = ((lane & 7) + 8 * ((lane >> 3) & 1)) * APAD * 2 + 16 * (lane >> 4);      // A-frag lane offset in Tb
    const uint32_t tb_off = ((lane & 7) + 8 * (lane >> 4)) * APAD * 2 + 16 * ((lane >> 3) & 1);      // B-frag lane offset in Tb ([n][k])
    const uint32_t kt_off = ((lane & 7) + 8 * ((lane >> 3) & 1)) * KPAD * 2 + 16 * (lane >> 4);      // trans B-frag offset in K/V tile ([k][n])
    const uint32_t vt_off = ((lane & 7) + 8 * (lane >> 4)) * KPAD * 2 + 16 * ((lane >> 3) & 1);      // trans A-frag offset in V tile
    {
        const int m = warp & 3, nq = warp >> 2;
        uint32_t af[4][4];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) ldsm_x4(af[ks], twa + (16 * m) * APAD * 2 + ta_off + ks * 32);
#pragma unroll
        for (int npi = 0; npi < 4; npi++) {
            const int np = nq * 4 + npi;
            float acc[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[nt][e] = 0.f;
            uint32_t bf[4][4];
#pragma unroll
            for (int ks = 0; ks < 4; ks++) ldsm_x4_t(bf[ks], ksa + kt_off + (16 * ks) * KPAD * 2 + (16 * np) * 2);
#pragma unroll
            for (int ks = 0; ks < 4; ks++) { mma_bf16(acc[0], af[ks], bf[ks]); mma_bf16(acc[1], af[ks], bf[ks] + 2); }
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int tok = 16 * m + g + 8 * half, col = 16 * np + 8 * nt + 2 * t4;
                    *reinterpret_cast<uint32_t*>(Wst + tile128_off(BT, tok, col)) = pack_bf16(acc[nt][2 * half], acc[nt][2 * half + 1]);
                }
        }
    }
    proxy_fence_smem();
    __syncthreads();   // W staged; K tile no longer needed
    DBG_I(8)
    if (tid == 0) { bulk_store_1d(w_out + tile_id * (BT * DK), smem_u32(Wst), BT * DK * 2); bulk_commit(); }
    DBG_I(9)
    // ---- -U^T = -V^T Tb_u^T : warp w -> rows v 16w.., all 4 token pairs; staged in LT after the W store drained ----
    {
        const int m = warp;
        uint32_t bf[4][4][4];   // [np][ks]
#pragma unroll
        for (int np = 0; np < 4; np++)
#pragma unroll
            for (int ks = 0; ks < 4; ks++) ldsm_x4(bf[np][ks], tua + (16 * np) * APAD * 2 + tb_off + ks * 32);
        uint32_t af[4][4];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) ldsm_x4_t(af[ks], qsa + vt_off + (16 * ks) * KPAD * 2 + (16 * m) * 2);
        float acc[4][2][4];
#pragma unroll
        for (int np = 0; np < 4; np++) {
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[np][nt][e] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) { mma_bf16(acc[np][0], af[ks], bf[np][ks]); mma_bf16(acc[np][1], af[ks], bf[np][ks] + 2); }
        }
        if (tid == 0) bulk_wait_read();   // W staging drained
        __syncthreads();
        DBG_I(10)
#pragma unroll
        for (int np = 0; np < 4; np++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int vr = 16 * m + g + 8 * half, tok = 16 * np + 8 * nt + 2 * t4;
                    *reinterpret_cast<uint32_t*>(Wst + vr * BT + swz(vr, tok >> 3) * 8 + (tok & 7)) = pack_bf16(-acc[np][nt][2 * half], -acc[np][nt][2 * half + 1]);
                }
    }
    if (tid < BT) {   // ---- G tile: G, dtok = e^{G_last - G}, stok = scale e^{G}, gamma = e^{G_last} ----
        const float* Gs = gg + BT;
        float* gt = gc_out + tile_id * GT;
        const float glast = Gs[BT - 1];
        gt[tid] = Gs[tid];
        gt[BT + tid] = __expf(glast - Gs[tid]);
        gt[2 * BT + tid] = scale * __expf(Gs[tid]);
        if (tid == 0) gt[3 * BT] = __expf(glast);
    }
    proxy_fence_smem();
    __syncthreads();
    if (tid == 0) {
        bulk_store_1d(ut_out + tile_id * (DV * BT), smem_u32(Wst), DV * BT * 2); bulk_commit();
        bulk_wait_all();   // all tiles of this chunk are in global memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
    __syncthreads();
    if (tid == 0) st_release(flag + tile_id, epoch);
    if (dbg != nullptr && tid == 0 && blockIdx.x < 512) dbg[20000 + blockIdx.x * 2 + 1] = gtimer();
    DBG_I(11)
}

// ============================================================================================
// Recurrence kernel (tcgen05): one CTA per (sequence, v-head), 4 compute warps + 1 TMA producer warp.
// Accumulators live in TMEM; all MMA operands come from shared memory (SW128 tiles):
//   P^T [128 v x 64 tok] = S0^T W^T            A = S0^T tile (smem), B = W tile
//   O   [64 tok x 128 v] = Q~ S0 + Aqk R       A = Q~ / Aqk tiles, B = S0^T / R^T tiles
//   S1^T = gamma S0^T + Rt^T K                 A = Rt^T tile, B = K tile (MN-major), D = T_S (accumulate)
// ============================================================================================
constexpr int NSTAGE = 3;
constexpr int R_B1 = 0;                             // PO operand B, K-major SW128: [k 0-63: Q~ rows | W rows][k 64-127: Q~ rows | W rows] (32KB)
constexpr int R_B2 = R_B1 + 2 * BT * DK * 2;         // S/O2 operand B, MN-major SW128: [K~ k 0-63][K~ k 64-127][Aqk^T] (24KB)
constexpr int R_U = R_B2 + BT * DK * 2 + BT * BT * 2; // [128 v][64 tok] bf16 (128B rows, chunk ^ (v&7))
constexpr int R_G = R_U + DV * BT * 2;               // [GT] fp32
constexpr int R_STAGE_BYTES = R_G + GT * 4;          // 74560 bytes transferred per stage
constexpr int R_STAGE = ((R_STAGE_BYTES + 1023) / 1024) * 1024;   // 74752
constexpr int R_MBAR = NSTAGE * R_STAGE;            // fullB[NSTAGE] (B1+B2+G), fullU[NSTAGE] (U), emptyB[NSTAGE], emptyU[NSTAGE], mmaPO, mmaS, mmaO, ofree[2], state
constexpr int R_BBYTES = 2 * BT * DK * 2 + BT * DK * 2 + BT * BT * 2 + GT * 4;   // bytes of the B1 + B2 + G part of a stage (58176)
constexpr int RECUR_SMEM = R_MBAR + (4 * NSTAGE + 6) * 8 + 1024;   // +1024 for runtime alignment
constexpr int RECUR_THREADS = 480;   // 8 compute warps (warp w and w+4 share TMEM lanes, split columns) + TMA producer + 2 MMA issuers + 4 output (epilogue) warps
constexpr int W_PROD = 8, W_MMA = 9, W_EPI = 10, W_MMA2 = 14;   // epilogue warps 10..13 own TMEM lane quarters (warp & 3); warp 14 issues the O2 MMAs
constexpr int NCOMP = 256;
constexpr int BAR_R = 2, BAR_SB = 3;   // named barriers compute(arrive) -> issuer(sync): R^T ready (both issuers) / S^T bf16 operand ready
constexpr int BAR_EPI = 4;             // epilogue warps only (128 threads)
constexpr int BAR_ST = 5, BAR_FIN = 6;  // compute(arrive) -> producer(sync): initial state consumed from smem / final state left the staging
constexpr int BAR_S2 = 7;              // issuer 1 (arrive, after issuing S) -> issuer 2 (sync): keeps O2 behind S in the tensor pipe
constexpr uint32_t T_S = 0, T_SB = 128, T_R = 192, T_OP = 256;   // TMEM columns: S^T fp32 [128] | S^T bf16 [64] | R^T bf16 [32] | per chunk parity: O^T [64] P^T [64]

__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3fff);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3fff) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3fff) << 32;
    d |= (uint64_t)1 << 46;      // descriptor version (Blackwell)
    d |= (uint64_t)2 << 61;      // SWIZZLE_128B
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(int M, int N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= 1u << 4; d |= 1u << 7; d |= 1u << 10;    // D = F32, A = BF16, B = BF16
    d |= (uint32_t)a_major << 15; d |= (uint32_t)b_major << 16;
    d |= (uint32_t)(N >> 3) << 17; d |= (uint32_t)(M >> 4) << 24;
    return d;
}
__device__ __forceinline__ void umma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(mbar) : "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};\n"
                 :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                    "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr) : "memory");
}
// 16x256b.x8: 16 lanes x 64 columns; thread T gets, per 8-column group g (regs 4g..4g+3): (lane T/4, col 8g+2(T%4)), (.., +1), (lane T/4+8, ..), (.., +1)
__device__ __forceinline__ void tmem_ld_16x256b_x8(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(taddr));
}
__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t f0, uint32_t f1, uint32_t f2, uint32_t f3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(f0), "r"(f1), "r"(f2), "r"(f3) : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ int ld_volatile_s32(const int* p) { int v; asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v; }
__device__ __forceinline__ void st_volatile_s32(int* p, int v) { asm volatile("st.volatile.shared.s32 [%0], %1;\n" ::"r"(smem_u32(p)), "r"(v) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int count) { __syncwarp(); asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void tma_load_2d(uint32_t sdst, const CUtensorMap* tm, int x, int y, uint32_t mbar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
                 ::"r"(sdst), "l"(tm), "r"(mbar), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* tm, uint32_t ssrc, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n" ::"l"(tm), "r"(ssrc), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes) { asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(src), "r"(bytes) : "memory"); }

// ============================================================================================
// Recurrence kernel (tcgen05): one CTA per (sequence, v-head); 8 compute warps + TMA producer warp + MMA issuer warp.
// Accumulators and the A operands live in TMEM, B operands are TMA-loaded SW128 smem tiles:
//   [O1^T | P^T] [128 v x 128]   = S0^T [Q~^T | W^T]      A = S0^T bf16 (TMEM), B = stage tile B1 (K-major)
//   S1^T          [128 v x 128 k] = gamma S0^T + R^T K~     A = R^T bf16 (TMEM),  B = K~ (MN-major), D = T_S (accumulate)
//   O^T          [128 v x 64 tok] += R^T Aqk^T              A = R^T bf16 (TMEM),  B = Aqk^T (MN-major)
// ============================================================================================
__global__ void __launch_bounds__(RECUR_THREADS, 1)
gdn_recur_kernel(const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ kc,
                 const __nv_bfloat16* __restrict__ qc, const __nv_bfloat16* __restrict__ aqk,
                 const __nv_bfloat16* __restrict__ ut, const float* __restrict__ gc,
                 const float* __restrict__ state, float* __restrict__ new_state, __nv_bfloat16* __restrict__ out, const __grid_constant__ CUtensorMap tm_out,
                 const __grid_constant__ CUtensorMap tm_state, const __grid_constant__ CUtensorMap tm_nstate,
                 const int64_t* __restrict__ cu, const int* __restrict__ intra_flag,
                 int item_begin, int item_end, int epoch, int wait_flags, int N, int HK, int HV, long long* __restrict__ dbg) {
    extern __shared__ __align__(1024) unsigned char smem[];
    __shared__ int cus[MAXSEQ + 1];
    __shared__ int seqtab[MAXSEQ][5];   // by length rank: n, start, len, nchunks, cbase
    __shared__ uint32_t tmem_base_s;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint32_t sbase = (smem_u32(smem) + 1023u) & ~1023u;
    unsigned char* const smem_al = smem + (sbase - smem_u32(smem));
    const uint32_t mbar0 = sbase + R_MBAR;                 // fullB[s]: B1 (Q~/W), B2 (K~/Aqk^T), G landed
    const uint32_t mbarFU = mbar0 + NSTAGE * 8;            // fullU[s]: -U^T landed
    const uint32_t mbarE = mbar0 + 2 * NSTAGE * 8;         // emptyB[s]: B1/B2/G free (MMAs done and the output staged in B1 has left)
    const uint32_t mbarEU = mbar0 + 3 * NSTAGE * 8;        // emptyU[s]: U free (read by the compute warps for the D init; 8 arrivals, one per warp)
    const uint32_t mbarP = mbar0 + 4 * NSTAGE * 8, mbarS = mbarP + 8, mbarO = mbarP + 16, mbarOF = mbarP + 24, mbarST = mbarP + 40;   // ofree[par] = mbarOF + par * 8
    if (tid <= N) cus[tid] = (int)cu[tid];
    if (tid == 0) {
        for (int s = 0; s < 4 * NSTAGE + 3; s++) mbar_init(mbar0 + s * 8, (s >= 3 * NSTAGE && s < 4 * NSTAGE) ? 8 : 1);
        mbar_init(mbarOF, 4); mbar_init(mbarOF + 8, 4);   // one arrival per epilogue warp
        mbar_init(mbarST, 1);
        mbar_fence_init();
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(&tmem_base_s)), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    __syncthreads();
    if (tid < N) {
        const int st_i = cus[tid], len_i = cus[tid + 1] - st_i;
        int rank = 0, cb = 0;
        for (int j = 0; j < N; j++) {
            const int lj = cus[j + 1] - cus[j];
            rank += (lj > len_i) || (lj == len_i && j < tid);
            if (j < tid) cb += (lj + BT - 1) / BT;
        }
        seqtab[rank][0] = tid; seqtab[rank][1] = st_i; seqtab[rank][2] = len_i; seqtab[rank][3] = (len_i + BT - 1) / BT; seqtab[rank][4] = cb;
    }
    __syncthreads();
    tc_fence_after();
    const uint32_t tb = tmem_base_s;
    const uint32_t tlane = tb + ((uint32_t)(warp & 3) * 32u << 16);
    int gchunk = 0;

    if (warp == W_PROD) {
        // =============================== TMA producer warp ===============================
        // B parts (Q~/W, K~/Aqk^T, G) are loaded as soon as the MMAs of the stage's previous chunk finished; the U part
        // (whose region doubles as the output staging) once that output left the stage.
        int itcount = 0;
        for (int item = item_begin + blockIdx.x; item < item_end; item += gridDim.x, itcount++) {
            const int h = item % HV, r = item / HV;
            const int nchunks = seqtab[r][3], cbase = seqtab[r][4], n = seqtab[r][0];
            const __nv_bfloat16* wsrc = w + ((size_t)(cbase * HV + h)) * (BT * DK);
            const __nv_bfloat16* ksrc = kc + ((size_t)(cbase * HV + h)) * (BT * DK);
            const __nv_bfloat16* qsrc = qc + ((size_t)(cbase * HV + h)) * (BT * DK);
            const __nv_bfloat16* asrc = aqk + ((size_t)(cbase * HV + h)) * (BT * BT);
            const __nv_bfloat16* usrc = ut + ((size_t)(cbase * HV + h)) * (DV * BT);
            const float* gsrc = gc + (size_t)(cbase * HV + h) * GT;
            const int* fsrc = intra_flag + (cbase * HV + h);
            if (itcount > 0) bar_sync(BAR_FIN, NCOMP + 32);   // previous item's final state left stage gchunk%3
            const uint64_t pol = l2_policy_evict_first();
            if (lane == 0) {   // initial state S0^T [128 v][128 k] fp32 -> 4 SW128 boxes (32 cols each) in the stage of chunk 2 (chunks 0/1 stream in meanwhile)
                const int gs = gchunk + 2, s2 = gs % NSTAGE;
                if (gs >= NSTAGE) { mbar_wait(mbarE + s2 * 8, ((gs / NSTAGE) - 1) & 1); mbar_wait(mbarEU + s2 * 8, ((gs / NSTAGE) - 1) & 1); }
                const uint32_t so = sbase + s2 * R_STAGE;
                mbar_expect_tx(mbarST, DV * DK * 4);
                tma_load_1d(so, state + ((size_t)(n * HV + h)) * (DV * DK), DV * DK * 4, mbarST);   // natural [128 v][128 k] fp32 layout (512B rows)
            }
            for (int c = 0; c < nchunks; c++) {
                const int gcx = gchunk + c, s = gcx % NSTAGE;
                if (c == 2) bar_sync(BAR_ST, NCOMP + 32);   // compute warps consumed the state: stage s2 may be filled
                if (lane == 0) {
                    const uint32_t so = sbase + s * R_STAGE;
                    if (dbg != nullptr && blockIdx.x == 0 && gcx < 60) dbg[100000 + gcx * 4] = clock64();
                    if (gcx >= NSTAGE) mbar_wait(mbarEU + s * 8, ((gcx / NSTAGE) - 1) & 1);   // U region consumed (D init of chunk gcx-3)
                    if (dbg != nullptr && blockIdx.x == 0 && gcx < 60) dbg[100000 + gcx * 4 + 1] = clock64();
                    if (wait_flags) spin_wait_ge(fsrc + c * HV, epoch, 2);
                    if (dbg != nullptr && blockIdx.x == 0 && gcx < 60) dbg[100000 + gcx * 4 + 2] = clock64();
                    asm volatile("fence.proxy.async.global;\n" ::: "memory");
                    mbar_expect_tx(mbarFU + s * 8, DV * BT * 2);
                    tma_load_1d_hint(so + R_U, usrc + (size_t)c * (HV * DV * BT), DV * BT * 2, mbarFU + s * 8, pol);
                    if (c + 3 < nchunks) {   // warm L2 three chunks ahead
                        const int cp = c + 3;
                        l2_prefetch(wsrc + (size_t)cp * (HV * BT * DK), BT * DK * 2);
                        l2_prefetch(qsrc + (size_t)cp * (HV * BT * DK), BT * DK * 2);
                        l2_prefetch(ksrc + (size_t)cp * (HV * BT * DK), BT * DK * 2);
                        l2_prefetch(asrc + (size_t)cp * (HV * BT * BT), BT * BT * 2);
                        l2_prefetch(usrc + (size_t)cp * (HV * DV * BT), DV * BT * 2);
                    }
                    if (gcx >= NSTAGE) mbar_wait(mbarE + s * 8, ((gcx / NSTAGE) - 1) & 1);   // B1/B2/G free (chunk gcx-3 done, its output left the staging)
                    const uint32_t mb = mbar0 + s * 8;
                    mbar_expect_tx(mb, R_BBYTES);
                    const __nv_bfloat16* wq = wsrc + (size_t)c * (HV * BT * DK);
                    const __nv_bfloat16* qq = qsrc + (size_t)c * (HV * BT * DK);
                    tma_load_1d_hint(so + R_B1, qq, BT * 64 * 2, mb, pol);
                    tma_load_1d_hint(so + R_B1 + 8192, wq, BT * 64 * 2, mb, pol);
                    tma_load_1d_hint(so + R_B1 + 16384, qq + BT * 64, BT * 64 * 2, mb, pol);
                    tma_load_1d_hint(so + R_B1 + 24576, wq + BT * 64, BT * 64 * 2, mb, pol);
                    tma_load_1d_hint(so + R_B2, ksrc + (size_t)c * (HV * BT * DK), BT * DK * 2, mb, pol);
                    tma_load_1d_hint(so + R_B2 + 16384, asrc + (size_t)c * (HV * BT * BT), BT * BT * 2, mb, pol);
                    tma_load_1d_hint(so + R_G, gsrc + (size_t)c * (HV * GT), GT * 4, mb, pol);
                    if (dbg != nullptr && blockIdx.x == 0 && gcx < 60) dbg[100000 + gcx * 4 + 3] = clock64();
                }
            }
            if (nchunks < 3) bar_sync(BAR_ST, NCOMP + 32);   // (the compute warps arrive once per item)
            __syncwarp();
            gchunk += nchunks;
        }
        return;
    }
    if (warp == W_MMA) {
        // =============================== MMA issuer warp (lane 0 issues; issue blocks while the tensor core is busy) ===============================
        const uint32_t idesc_PO = make_idesc(128, 128, 0, 0), idesc_S = make_idesc(128, 128, 0, 1);
        const uint64_t dB1_0 = make_sdesc(sbase + R_B1, 16, 1024), dB2_0 = make_sdesc(sbase + R_B2, 8192, 1024);
        const uint64_t dstage = (uint64_t)(R_STAGE >> 4);
        for (int item = item_begin + blockIdx.x; item < item_end; item += gridDim.x) {
            const int nchunks = seqtab[item / HV][3];
            for (int c = 0; c < nchunks; c++) {
                const int gcx = gchunk + c, s = gcx % NSTAGE, par = gcx & 1;
                const uint32_t tO = T_OP + par * 128;
                const uint64_t dB1s = dB1_0 + s * dstage, dB2s = dB2_0 + s * dstage;
#define DBG_M(i) if (dbg != nullptr && blockIdx.x == 0 && lane == 0 && gcx < 60) dbg[110000 + gcx * 8 + (i)] = clock64();
                DBG_M(0)
                mbar_wait(mbar0 + s * 8, (gcx / NSTAGE) & 1);   // B1/B2/G landed
                DBG_M(1)
                DBG_M(2)
                bar_sync(BAR_SB, NCOMP + 32);   // S0^T bf16 operand written
                tc_fence_after();
                DBG_M(3)
                if (lane == 0) {
#pragma unroll
                    for (int ks = 0; ks < 8; ks++) {   // [O1^T | P^T] = S0^T [Q~^T | W^T]
                        const uint32_t hs = ((ks >> 2) * 16384 + (ks & 3) * 32) >> 4;
                        umma_ts(tb + tO, tb + T_SB + ks * 8, dB1s + hs, idesc_PO, 1);   // D pre-initialized to [0 | -U^T] -> yields [O1^T | -R^T]
                    }
                    umma_commit(mbarP);
                }
                DBG_M(4)
                bar_sync(BAR_R, NCOMP + 64);    // R^T operand and the fp32 S^T accumulator written
                tc_fence_after();
                DBG_M(5)
                if (lane == 0) {
#pragma unroll
                    for (int ks = 0; ks < 4; ks++)     // S1^T = gamma S0^T + R^T K~
                        umma_ts(tb + T_S, tb + T_R + ks * 8, dB2s + ks * 128, idesc_S, 1);
                    umma_commit(mbarS);
                    DBG_M(6)
                }
                bar_arrive(BAR_S2, 64);        // O2 may be queued now (behind S)
            }
            gchunk += nchunks;
        }
        return;
    }

    if (warp == W_MMA2) {
        // =============================== second MMA issuer: O^T += R^T Aqk^T (queued right after S so its issue never delays the next PO) ===============================
        const uint32_t idesc_O2 = make_idesc(128, 64, 0, 1);
        const uint64_t dB2_0 = make_sdesc(sbase + R_B2, 8192, 1024);
        const uint64_t dstage = (uint64_t)(R_STAGE >> 4);
        for (int item = item_begin + blockIdx.x; item < item_end; item += gridDim.x) {
            const int nchunks = seqtab[item / HV][3];
            for (int c = 0; c < nchunks; c++) {
                const int gcx = gchunk + c, s = gcx % NSTAGE, par = gcx & 1;
                const uint64_t dB2s = dB2_0 + s * dstage;
                mbar_wait(mbar0 + s * 8, (gcx / NSTAGE) & 1);   // Aqk^T landed (normally long ago)
                bar_sync(BAR_R, NCOMP + 64);
                tc_fence_after();
                bar_sync(BAR_S2, 64);
                if (lane == 0) {
#pragma unroll
                    for (int ks = 0; ks < 4; ks++)
                        umma_ts(tb + T_OP + par * 128, tb + T_R + ks * 8, dB2s + 1024 + ks * 128, idesc_O2, 1);
                    umma_commit(mbarO);
                }
            }
            gchunk += nchunks;
        }
        return;
    }
    if (warp >= W_EPI && warp < W_EPI + 4) {
        // =============================== output (epilogue) warps: O^T [128 v][64 tok] (TMEM) -> stmatrix.trans -> [tok][v] SW128 boxes -> out ===============================
        const int q = warp & 3, m = lane >> 3, t7 = lane & 7;
        const uint32_t tl0 = tb + ((uint32_t)(q * 32) << 16), tl1 = tb + ((uint32_t)(q * 32 + 16) << 16);
        const int etid = tid - W_EPI * 32;
        for (int item = item_begin + blockIdx.x; item < item_end; item += gridDim.x) {
            const int h = item % HV, r = item / HV;
            const int seq_start = seqtab[r][1], seq_len = seqtab[r][2], nchunks = seqtab[r][3];
            for (int c = 0; c < nchunks; c++) {
                const int gcx = gchunk + c, s = gcx % NSTAGE, par = gcx & 1;
                const int t0 = seq_start + c * BT, nvalid = seq_len - c * BT;
                mbar_wait(mbarO, gcx & 1);   // O^T complete; stage s no longer read by MMAs or compute warps
                tc_fence_after();
                uint32_t oa[32], ob[32];
                tmem_ld_16x256b_x8(tl0 + T_OP + par * 128, oa);
                tmem_ld_16x256b_x8(tl1 + T_OP + par * 128, ob);
                tmem_wait_ld();
                tc_fence_before();
                __syncwarp();
                if (lane == 0) mbar_arrive(mbarOF + par * 8);   // O^T[par] may be overwritten
                unsigned char* const os = smem_al + s * R_STAGE + R_B1 + (q >> 1) * 8192;   // SW128 box of this warp's v half (staged in the B1 region)
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const uint32_t* rr = hh ? ob : oa;
                    const int vchunk = 4 * (q & 1) + 2 * hh + (m & 1);   // 16B chunk index (8 v) of this lane's matrix rows
#pragma unroll
                    for (int gp = 0; gp < 4; gp++) {   // matrices: (cols 16gp..+7, v lo/hi), (cols 16gp+8..+15, v lo/hi)
                        const uint32_t f0 = pack_bf16(__uint_as_float(rr[8 * gp]), __uint_as_float(rr[8 * gp + 1]));
                        const uint32_t f1 = pack_bf16(__uint_as_float(rr[8 * gp + 2]), __uint_as_float(rr[8 * gp + 3]));
                        const uint32_t f2 = pack_bf16(__uint_as_float(rr[8 * gp + 4]), __uint_as_float(rr[8 * gp + 5]));
                        const uint32_t f3 = pack_bf16(__uint_as_float(rr[8 * gp + 6]), __uint_as_float(rr[8 * gp + 7]));
                        const int tok = 16 * gp + 8 * (m >> 1) + t7;
                        stmatrix_x4_trans(smem_u32(os) + tok * 128 + ((vchunk ^ (tok & 7)) * 16), f0, f1, f2, f3);
                    }
                }
                proxy_fence_smem();
                bar_sync(BAR_EPI, 128);
                if (nvalid >= BT) {   // full chunk: TMA 2D stores
                    if (warp == W_EPI && lane == 0) {
                        const uint32_t so = sbase + s * R_STAGE + R_B1;
                        tma_store_2d(&tm_out, so, h * DV, t0);
                        tma_store_2d(&tm_out, so + 8192, h * DV + 64, t0);
                        bulk_commit();
                        bulk_wait_read();
                        mbar_arrive(mbarE + s * 8);   // B1/B2/G of stage s free (MMAs done, output staging read)
                    }
                } else {   // partial (last) chunk: plain stores of the valid rows
                    const unsigned char* const stg = smem_al + s * R_STAGE + R_B1;
                    __nv_bfloat16* const obase = out + ((size_t)t0 * HV + h) * DV;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const int idx = etid + i * 128, row = idx >> 4, ch = idx & 15;
                        if (row < nvalid) {
                            const uint4 val = *reinterpret_cast<const uint4*>(stg + (ch >> 3) * 8192 + row * 128 + (((ch & 7) ^ (row & 7)) * 16));
                            *reinterpret_cast<uint4*>(obase + (size_t)row * (HV * DV) + ch * 8) = val;
                        }
                    }
                    bar_sync(BAR_EPI, 128);
                    if (warp == W_EPI && lane == 0) mbar_arrive(mbarE + s * 8);   // B1/B2/G of stage s free
                }
            }
            gchunk += nchunks;
        }
        if (warp == W_EPI && lane == 0) bulk_wait_all();
        return;
    }

    // =============================== compute warps: thread <-> (v-row, column half) ===============================
    const int v = tid & 127, chalf = tid >> 7;   // warps 0-3: columns [0,64) of each row; warps 4-7: columns [64,128)
    // O epilogue for a finished chunk: O^T [128 v][64 tok] in TMEM (this thread: 32 tokens of row v) -> transposed smem rows -> bulk stores
    // D init for chunk g (global index): P^T half <- -U^T (bf16 tile of its stage) so the PO MMA yields -R^T directly; O^T half <- 0
    auto init_du = [&](int g) {
        const int sg = g % NSTAGE, pg = g & 1;
        mbar_wait(mbarFU + sg * 8, (g / NSTAGE) & 1);                 // -U^T tile landed
        const __nv_bfloat16* Us = reinterpret_cast<const __nv_bfloat16*>(smem_al + sg * R_STAGE + R_U) + v * BT;
        uint32_t uw[16];
#pragma unroll
        for (int ch = 0; ch < 4; ch++) { const uint4 u = *reinterpret_cast<const uint4*>(Us + ((chalf * 4 + ch) ^ (v & 7)) * 8); uw[4 * ch] = u.x; uw[4 * ch + 1] = u.y; uw[4 * ch + 2] = u.z; uw[4 * ch + 3] = u.w; }
        __syncwarp();
        if (lane == 0) mbar_arrive(mbarEU + sg * 8);   // this warp's U reads are done
        uint32_t f[2][16];
#pragma unroll
        for (int q = 0; q < 16; q++) { f[q >> 3][2 * (q & 7)] = uw[q] << 16; f[q >> 3][2 * (q & 7) + 1] = uw[q] & 0xffff0000u; }
        const uint32_t tO = T_OP + pg * 128;
        tmem_st16(tlane + tO + 64 + chalf * 32, f[0]);
        tmem_st16(tlane + tO + 64 + chalf * 32 + 16, f[1]);
    };
    auto init_dz = [&](int g) {
        const int pg = g & 1;
        if (g >= 2) mbar_wait(mbarOF + pg * 8, ((g >> 1) - 1) & 1);   // O^T[pg] of chunk g-2 read out by the epilogue warps
        uint32_t z[16];
#pragma unroll
        for (int q = 0; q < 16; q++) z[q] = 0u;
        tmem_st16(tlane + T_OP + pg * 128 + chalf * 32, z);
        tmem_st16(tlane + T_OP + pg * 128 + chalf * 32 + 16, z);
    };
    // pack 64 fp32 (this thread's S^T row half) into bf16 pairs and store them as the TMEM A operand (32 columns)
    auto store_sb = [&](const uint32_t (&s32)[4][16]) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint32_t pk[16];
#pragma unroll
            for (int q = 0; q < 8; q++) { pk[q] = pack_bf16(__uint_as_float(s32[2 * j][2 * q]), __uint_as_float(s32[2 * j][2 * q + 1])); pk[8 + q] = pack_bf16(__uint_as_float(s32[2 * j + 1][2 * q]), __uint_as_float(s32[2 * j + 1][2 * q + 1])); }
            tmem_st16(tlane + T_SB + chalf * 32 + j * 16, pk);
        }
    };

    int itcount = 0;
    for (int item = item_begin + blockIdx.x; item < item_end; item += gridDim.x, itcount++) {
        const int h = item % HV, r = item / HV;
        const int n = seqtab[r][0], seq_start = seqtab[r][1], seq_len = seqtab[r][2], nchunks = seqtab[r][3];
        const int islot = (item - item_begin) / gridDim.x;
        if (dbg != nullptr && tid == 0 && item < 512) { long long* tr = dbg + 40000 + (size_t)item * 8; tr[0] = item; tr[1] = nchunks; tr[2] = gtimer(); }
        {   // ---- initial state (TMA-staged SW128 boxes in the stage of chunk 0) -> fp32 accumulator (T_S) and bf16 operand (T_SB) ----
            mbar_wait(mbarST, itcount & 1);
            if (dbg != nullptr && tid == 0 && item < 512) dbg[40000 + (size_t)item * 8 + 4] = gtimer();
            const unsigned char* sst = smem_al + ((gchunk + 2) % NSTAGE) * R_STAGE + v * 512 + chalf * 256;   // this thread's 64 floats (16 x 16B chunks)
            uint32_t s32[4][16];
            {   // conflict-free: lane l reads chunk (i + l) & 15 at step i, then rotates the registers back into chunk order
                const int kr = lane & 15;
                uint4 rr[16];
#pragma unroll
                for (int i = 0; i < 16; i++) rr[i] = *reinterpret_cast<const uint4*>(sst + (((i + kr) & 15) * 16));
                bar_arrive(BAR_ST, NCOMP + 32);   // staging consumed
#pragma unroll
                for (int b = 1; b < 16; b <<= 1) {   // rr[c] currently holds chunk (c + kr); rotate so that rr[c] holds chunk c
                    const bool doit = (kr & b) != 0;
                    uint4 tmp[16];
#pragma unroll
                    for (int c = 0; c < 16; c++) tmp[c] = rr[(c + 16 - b) & 15];
#pragma unroll
                    for (int c = 0; c < 16; c++) rr[c] = doit ? tmp[c] : rr[c];
                }
#pragma unroll
                for (int c = 0; c < 16; c++) { s32[c >> 2][4 * (c & 3)] = rr[c].x; s32[c >> 2][4 * (c & 3) + 1] = rr[c].y; s32[c >> 2][4 * (c & 3) + 2] = rr[c].z; s32[c >> 2][4 * (c & 3) + 3] = rr[c].w; }
            }
#pragma unroll
            for (int j = 0; j < 4; j++) tmem_st16(tlane + T_S + chalf * 64 + j * 16, s32[j]);
            if (nchunks > 0) {
                store_sb(s32);
                init_du(gchunk);
                init_dz(gchunk);
                tmem_wait_st();
                tc_fence_before();
                bar_arrive(BAR_SB, NCOMP + 32);
            } else tmem_wait_st();
            if (dbg != nullptr && tid == 0 && item < 512) dbg[40000 + (size_t)item * 8 + 5] = gtimer();
        }

        for (int c = 0; c < nchunks; c++) {
            const int gcx = gchunk + c, s = gcx % NSTAGE, par = gcx & 1;
            const uint32_t tP = T_OP + par * 128 + 64;   // this chunk's P^T columns
            const int t0 = seq_start + c * BT, nvalid = seq_len - c * BT;
#define DBG_R(i) if (dbg != nullptr && blockIdx.x == 0 && tid == 0 && islot == 0 && c < 60) dbg[64 + c * 16 + (i)] = clock64();
            DBG_R(0)
            mbar_wait(mbar0 + s * 8, (gcx / NSTAGE) & 1);   // this chunk's B1/B2/G tiles (normally landed long ago)
            const float gamma = reinterpret_cast<const float*>(smem_al + s * R_STAGE + R_G)[3 * BT];
            DBG_R(10)
            {   // ---- fp32 S^T x gamma (in place, runs under the PO MMAs) ----
                uint32_t s32[4][16];
#pragma unroll
                for (int j = 0; j < 4; j++) tmem_ld16(tlane + T_S + chalf * 64 + j * 16, s32[j]);
                tmem_wait_ld();
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int q = 0; q < 16; q++) s32[j][q] = __float_as_uint(__uint_as_float(s32[j][q]) * gamma);
                    tmem_st16(tlane + T_S + chalf * 64 + j * 16, s32[j]);
                }
            }
            tmem_wait_st();
            DBG_R(9)
            DBG_R(1)
            mbar_wait(mbarP, gcx & 1);
            tc_fence_after();
            DBG_R(2)
            {   // ---- -R^T = P^T - U^T (PO output) -> bf16 TMEM A operand (this thread: 32 tokens) ----
                uint32_t p16[2][16];
#pragma unroll
                for (int j = 0; j < 2; j++) tmem_ld16(tlane + tP + chalf * 32 + j * 16, p16[j]);
                tmem_wait_ld();
                DBG_R(14)
                uint32_t rr[16];
#pragma unroll
                for (int q = 0; q < 16; q++) rr[q] = pack_bf16(__uint_as_float(p16[q >> 3][2 * (q & 7)]), __uint_as_float(p16[q >> 3][2 * (q & 7) + 1]));
                tmem_st16(tlane + T_R + chalf * 16, rr);
                tmem_wait_st();
            }
            tc_fence_before();
            bar_arrive(BAR_R, NCOMP + 64);
            DBG_R(3)
            if (c + 1 < nchunks) { init_dz(gcx + 1); init_du(gcx + 1); tmem_wait_st(); }   // next chunk's D init (runs under the S MMAs)
            DBG_R(15)
            mbar_wait(mbarS, gcx & 1);
            tc_fence_after();
            DBG_R(4)
            if (c + 1 < nchunks) {   // ---- S1^T -> bf16 TMEM operand for the next chunk ----
                uint32_t s32[4][16];
#pragma unroll
                for (int j = 0; j < 4; j++) tmem_ld16(tlane + T_S + chalf * 64 + j * 16, s32[j]);
                tmem_wait_ld();
                store_sb(s32);
                tmem_wait_st();
                tc_fence_before();
                bar_arrive(BAR_SB, NCOMP + 32);
            }
            DBG_R(12)
            DBG_R(5)
        }
        {   // ---- final state: TMEM -> SW128 boxes in the stage after the last chunk's (free: its chunk was consumed 3 chunks ago) -> TMA 2D stores ----
            const int gf = gchunk + (nchunks > 0 ? nchunks : 0), sf = gf % NSTAGE;   // stage that would hold chunk gf
            if (dbg != nullptr && tid == 0 && item < 512) dbg[40000 + (size_t)item * 8 + 6] = gtimer();
            if (nchunks > 0) { mbar_wait(mbarO, (gf - 1) & 1); tc_fence_after(); }
            if (gf >= NSTAGE) { mbar_wait(mbarE + sf * 8, ((gf / NSTAGE) - 1) & 1); mbar_wait(mbarEU + sf * 8, ((gf / NSTAGE) - 1) & 1); }   // chunk gf-3 fully consumed
            if (dbg != nullptr && tid == 0 && item < 512) dbg[40000 + (size_t)item * 8 + 7] = gtimer();
            unsigned char* const sst = smem_al + sf * R_STAGE + v * 512 + chalf * 256;   // natural [128 v][128 k] fp32 layout
            {
                uint4 rr[16];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t r16[16];
                    tmem_ld16(tlane + T_S + chalf * 64 + j * 16, r16);
                    tmem_wait_ld();
#pragma unroll
                    for (int q = 0; q < 4; q++) rr[4 * j + q] = make_uint4(r16[4 * q], r16[4 * q + 1], r16[4 * q + 2], r16[4 * q + 3]);
                }
                const int kr = lane & 15;
#pragma unroll
                for (int b = 1; b < 16; b <<= 1) {   // rotate so that rr[i] holds chunk (i + kr): store i writes chunk (i + kr) & 15 (conflict-free)
                    const bool doit = (kr & b) != 0;
                    uint4 tmp[16];
#pragma unroll
                    for (int c = 0; c < 16; c++) tmp[c] = rr[(c + b) & 15];
#pragma unroll
                    for (int c = 0; c < 16; c++) rr[c] = doit ? tmp[c] : rr[c];
                }
#pragma unroll
                for (int i = 0; i < 16; i++) *reinterpret_cast<uint4*>(sst + (((i + kr) & 15) * 16)) = rr[i];
            }
            bar_sync(1, NCOMP);
            {   // rows -> global with full-row coalescing (each warp store instruction writes one 512B row); drains asynchronously
                const unsigned char* const stg = smem_al + sf * R_STAGE;
                float* const np = new_state + ((size_t)(n * HV + h)) * (DV * DK);
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const int idx = tid + i * NCOMP, row = idx >> 5, ch = idx & 31;
                    *reinterpret_cast<uint4*>(np + (size_t)row * DK + ch * 4) = *reinterpret_cast<const uint4*>(stg + row * 512 + ch * 16);
                }
            }
            bar_sync(1, NCOMP);
            if (item + (int)gridDim.x < item_end) bar_arrive(BAR_FIN, NCOMP + 32);   // producer may reuse the stage for the next item
        }
        gchunk += nchunks;
        if (dbg != nullptr && tid == 0 && item < 512) dbg[40000 + (size_t)item * 8 + 3] = gtimer();
        bar_sync(1, NCOMP);
    }
    if (tid == 0) bulk_wait_all();
    tc_fence_before();
    bar_sync(1, NCOMP);
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tb), "r"(512) : "memory");
}

}  // namespace

namespace { struct ScratchView { torch::Tensor w, kc, qc, aqk, gc, sc, rt, iflag, rcnt, wctr, octr; }; }
static void* g_scratch = nullptr;
static std::vector<torch::Tensor>* g_keep = nullptr;
std::vector<torch::Tensor> gdn_scratch() { return g_keep ? *g_keep : std::vector<torch::Tensor>{}; }
std::vector<torch::Tensor> gdn_prefill_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
                                            torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
                                            torch::Tensor cu_seqlens, double scale, torch::Tensor dbg, int64_t mode) {
    const int T = q.size(0), HK = q.size(1), HV = v.size(1), N = state.size(0);
    TORCH_CHECK(q.size(2) == DK && v.size(2) == DV && state.size(2) == DV && state.size(3) == DK, "head dims must be 128");
    TORCH_CHECK(N <= MAXSEQ, "at most 32 sequences");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && state.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64);
    const int maxch = T / BT + N;
    auto bopts = q.options();
    struct ScratchLocal {
        int maxch = -1, HV = -1, HK = -1;
        torch::Tensor w, kc, qc, ut, aqk, gc, sc, rt, iflag, rcnt, wctr, octr;
        int num_sms = 148, recur_grid = 148;
        int epoch = 0;
        cudaStream_t side = nullptr;
        cudaEvent_t ev_start = nullptr, ev_end = nullptr;
    };
    static ScratchLocal S;
    g_scratch = &S;
    if (S.maxch != maxch || S.HV != HV || S.HK != HK) {
        S.maxch = maxch; S.HV = HV; S.HK = HK;
        S.w = torch::empty({maxch, HV, BT, DK}, bopts);
        S.kc = torch::empty({maxch, HV, BT, DK}, bopts);
        S.qc = torch::empty({maxch, HV, BT, DK}, bopts);
        S.ut = torch::empty({maxch, HV, DV, BT}, bopts);
        S.aqk = torch::empty({maxch, HV, BT, BT}, bopts);
        S.gc = torch::empty({maxch, HV, GT}, state.options());
        S.sc = torch::empty({maxch, HV, DV, DK}, bopts);
        S.rt = torch::empty({maxch, HV, DV, BT}, bopts);
        S.iflag = torch::zeros({maxch * HV}, q.options().dtype(torch::kInt32));
        S.rcnt = torch::zeros({maxch * HV}, q.options().dtype(torch::kInt32));
        S.wctr = torch::zeros({1}, q.options().dtype(torch::kInt32));
        S.octr = torch::zeros({1}, q.options().dtype(torch::kInt32));
        S.epoch = 0;
        if (S.side == nullptr) {
            { int lo = 0, hi = 0; cudaDeviceGetStreamPriorityRange(&lo, &hi); cudaStreamCreateWithPriority(&S.side, cudaStreamNonBlocking, hi); }   // highest priority: its CTAs are dispatched before the intra kernel's
            cudaEventCreateWithFlags(&S.ev_start, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&S.ev_end, cudaEventDisableTiming);
        }
        cudaFuncSetAttribute(gdn_intra_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, INTRA_SMEM);
        cudaFuncSetAttribute(gdn_recur_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RECUR_SMEM);
        cudaFuncSetAttribute(gdn_intra_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(gdn_recur_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaDeviceGetAttribute(&S.num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
        S.recur_grid = S.num_sms;
        if (const char* e = getenv("GDN_RECUR_GRID")) S.num_sms = std::min(S.num_sms, atoi(e));
        if (getenv("GDN_VERBOSE")) {
            int nb = 0; size_t smpm = 0; cudaDeviceGetAttribute((int*)&smpm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, q.get_device());
            cudaFuncAttributes fa;
            cudaFuncGetAttributes(&fa, gdn_recur_kernel); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gdn_recur_kernel, RECUR_THREADS, RECUR_SMEM);
            printf("smem/SM=%zu recur: regs=%d static=%zu dyn=%d occ=%d\n", smpm, fa.numRegs, fa.sharedSizeBytes, RECUR_SMEM, nb);
            cudaFuncGetAttributes(&fa, gdn_intra_kernel); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gdn_intra_kernel, 256, INTRA_SMEM);
            printf("intra: regs=%d static=%zu dyn=%d occ=%d\n", fa.numRegs, fa.sharedSizeBytes, INTRA_SMEM, nb);
        }
        cudaDeviceSynchronize();
    }
    { static std::vector<torch::Tensor> keep; keep = {S.w, S.kc, S.qc, S.aqk, S.gc, S.ut, S.iflag}; g_keep = &keep; }
    S.epoch += 1;
    const int epoch = S.epoch;
    auto out = torch::empty({T, HV, DV}, bopts);
    auto new_state = torch::empty_like(state);
    typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                                      CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static EncodeTiledFn encode_tiled = nullptr;
    if (encode_tiled == nullptr) {
        cudaDriverEntryPointQueryResult qres;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode_tiled), cudaEnableDefault, &qres) == cudaSuccess && encode_tiled != nullptr, "cuTensorMapEncodeTiled unavailable");
    }
    CUtensorMap tm_out;
    {
        const cuuint64_t gdim[2] = {(cuuint64_t)HV * DV, (cuuint64_t)T};
        const cuuint64_t gstride[1] = {(cuuint64_t)HV * DV * 2};
        const cuuint32_t box[2] = {64, 64}, estr[2] = {1, 1};
        const CUresult cr = encode_tiled(&tm_out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, out.data_ptr(), gdim, gstride, box, estr,
                                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)cr);
    }
    CUtensorMap tm_state, tm_nstate;
    {
        const cuuint64_t gdim[2] = {(cuuint64_t)DK, (cuuint64_t)N * HV * DV};
        const cuuint64_t gstride[1] = {(cuuint64_t)DK * 4};
        const cuuint32_t box[2] = {32, 128}, estr[2] = {1, 1};
        CUresult cr = encode_tiled(&tm_state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, state.data_ptr(), gdim, gstride, box, estr,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled(state) failed: ", (int)cr);
        cr = encode_tiled(&tm_nstate, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, new_state.data_ptr(), gdim, gstride, box, estr,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(cr == CUDA_SUCCESS, "cuTensorMapEncodeTiled(new_state) failed: ", (int)cr);
    }
    long long* dbgp = dbg.numel() ? reinterpret_cast<long long*>(dbg.data_ptr<int64_t>()) : nullptr;
    cudaStream_t sA = at::cuda::getCurrentCUDAStream();
    cudaStream_t sB = S.side;
    auto bp = [](const torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
    const int total_items = N * HV;   // (sequence, head) items
    const bool overlap = (mode == 0);
    const int nA = overlap ? std::min(total_items, 3 * HV) : 0;   // longest 3 sequence ranks run concurrently with the intra kernel
    auto launch_recur = [&](int ib, int ie, int wflags, cudaStream_t st) {
        if (ie <= ib) return;
        gdn_recur_kernel<<<std::min(ie - ib, S.num_sms), RECUR_THREADS, RECUR_SMEM, st>>>(
            bp(S.w), bp(S.kc), bp(S.qc), bp(S.aqk), bp(S.ut), S.gc.data_ptr<float>(), state.data_ptr<float>(), new_state.data_ptr<float>(), bp(out), tm_out, tm_state, tm_nstate,
            cu_seqlens.data_ptr<int64_t>(), S.iflag.data_ptr<int>(), ib, ie, epoch, wflags, N, HK, HV, dbgp);
    };
    if (overlap) { cudaEventRecord(S.ev_start, sA); cudaStreamWaitEvent(sB, S.ev_start, 0); launch_recur(0, nA, 1, sB); }
    gdn_intra_kernel<<<maxch * HV, 256, INTRA_SMEM, sA>>>(
        bp(q), bp(k), bp(v), bp(a), bp(b), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), cu_seqlens.data_ptr<int64_t>(),
        bp(S.w), bp(S.kc), bp(S.qc), bp(S.ut), bp(S.aqk), S.gc.data_ptr<float>(), S.iflag.data_ptr<int>(), epoch, (float)scale, N, HK, HV, dbgp);
    launch_recur(nA, total_items, 0, sA);
    if (overlap) { cudaEventRecord(S.ev_end, sB); cudaStreamWaitEvent(sA, S.ev_end, 0); }
    return {out, new_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_prefill", &gdn_prefill_cuda, "GDN prefill forward (CUDA)"); m.def("gdn_scratch", &gdn_scratch, "scratch tensors (w, kc, qc, aqk, gc, ut)"); }
