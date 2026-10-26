// Common device helpers for the FP8 block-scale MoE kernels (written from scratch).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

namespace moe {

constexpr int H = 7168;          // hidden size
constexpr int I = 2048;          // intermediate size
constexpr int N1 = 2 * I;        // gemm1 out size (gate+up)
constexpr int E_LOCAL = 32;
constexpr int E_GLOBAL = 256;
constexpr int KB1 = H / 128;     // 56 k-blocks of GEMM1
constexpr int KB2 = I / 128;     // 16 k-blocks of GEMM2
constexpr int NB1 = N1 / 128;    // 32 n-blocks of W13
constexpr int NB2 = H / 128;     // 56 n-blocks of W2
constexpr int TOP_K = 8;

typedef uint8_t fp8_t;
typedef __nv_bfloat16 bf16_t;

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint4 ld_nc_16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ld_nc_8(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 lds_8(uint32_t saddr) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(saddr));
    return v;
}
__device__ __forceinline__ uint4 lds_16(uint32_t saddr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(saddr));
    return v;
}
__device__ __forceinline__ int atom_add_release_gpu(int* p, int v) {   // release-ordered add (prior writes visible before it)
#ifdef V3_RELAXED_EXPERIMENT
    int old; asm volatile("atom.add.relaxed.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
#else
    int old; asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
#endif
}
__device__ __forceinline__ int atom_add_acq_rel_gpu(int* p, int v) {   // release prior writes + acquire for what follows
    int old; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void red_add_release_gpu(int* p, int v) {      // fire-and-forget release increment
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void sts_f32(uint32_t addr, float v) { asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory"); }
__device__ __forceinline__ float2 lds_f2(uint32_t saddr) {
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(saddr));
    return v;
}
__device__ __forceinline__ float lds_f(uint32_t saddr) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(saddr));
    return v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async_16(uint32_t saddr, const void* gptr, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr), "l"(gptr), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void named_bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// D = A(16x32, e4m3, row) * B(32x8, e4m3, col) + C ; fp32 accumulate
__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// same with zero C input
__device__ __forceinline__ void mma_fp8_zero(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

// pack 2 floats to 2 x e4m3 (round to nearest, saturate)
__device__ __forceinline__ uint16_t pack_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void red_add_f32(float* p, float v) {
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- mbarrier / TMA
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t phase) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n }"
        : "=r"(ok) : "r"(bar), "r"(phase) : "memory");
    return ok != 0;
}
__device__ __forceinline__ bool mbar_test_wait(uint32_t bar, uint32_t phase) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n"
        "mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n }"
        : "=r"(ok) : "r"(bar), "r"(phase) : "memory");
    return ok != 0;
}
__device__ __forceinline__ bool mbar_try_wait_hint(uint32_t bar, uint32_t phase) {
    uint32_t ok;
    asm volatile(
        "{\n .reg .pred p;\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%1], %2, %3;\n"
        "selp.u32 %0, 1, 0, p;\n }"
        : "=r"(ok) : "r"(bar), "r"(phase), "r"(0x989680u) : "memory");
    return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
    if (mbar_test_wait(bar, phase)) return;
    while (!mbar_try_wait_hint(bar, phase)) {}
}
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, uint32_t phase) {   // no suspend: ~50-cycle poll, no wake-up latency
    while (!mbar_test_wait(bar, phase)) {}
}
__device__ __forceinline__ void mbar_wait_ns(uint32_t bar, uint32_t phase) {     // poll with short sleeps
    while (!mbar_test_wait(bar, phase)) __nanosleep(40);
}
__device__ __forceinline__ void mbar_wait_tw(uint32_t bar, uint32_t phase) {     // try_wait without a suspend hint
    while (!mbar_try_wait(bar, phase)) {}
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_relaxed(uint32_t bar) {
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t elect_one() {
    uint32_t pred;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n }" : "=r"(pred));
    return pred;
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
// 2D TMA tile load: coordinates (x = inner/bytes, y = row)
__device__ __forceinline__ void cp_async_bulk_g2s(uint32_t smem_dst, const void* gsrc, uint32_t bytes, uint32_t bar) {  // 16B-aligned, bytes % 16 == 0
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_dst), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t smem_dst, const void* tmap, int x, int y, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_prefetch_desc(const void* tmap) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

// ---------------------------------------------------------------- ldmatrix / fp16 mma
__device__ __forceinline__ void ldsm_x4(uint32_t saddr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr));
}
// 2 x e4m3 (packed in the low/high 16 bits of a register) -> f16x2
__device__ __forceinline__ uint32_t cvt_e4m3x2_lo(uint32_t v) {
    uint32_t r;
    asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n }" : "=r"(r) : "r"(v));
    return r;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_hi(uint32_t v) {
    uint32_t r;
    asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, hi;\n }" : "=r"(r) : "r"(v));
    return r;
}
__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16_zero(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
// ---------------------------------------------------------------- tcgen05 (5th-gen tensor core) helpers
__device__ __forceinline__ void tmem_alloc(uint32_t smem_dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_dst), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// D[tmem] (+)= A[smem desc] * B[smem desc], kind::f8f6f4 (e4m3 x e4m3 -> f32), single-thread issue
__device__ __forceinline__ void umma_f8(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n }"
        ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tmem_ld_x16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// UMMA shared-memory matrix descriptor: K-major, 128B swizzle, 8-row core groups 1024 B apart
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr) {
    return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
// UMMA instruction descriptor: e4m3 x e4m3 -> f32, K-major A and B, M x N
__device__ __forceinline__ uint32_t make_idesc(int M, int N) {
    return (1u << 4) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// swizzled byte offset of (row, 16B-chunk) in a [rows][128B] tile with 128B swizzle (base 1024B aligned)
__device__ __forceinline__ uint32_t swz128(int row, int chunk) { return (uint32_t)(row * 128 + ((chunk ^ (row & 7)) << 4)); }

// ---------------------------------------------------------------- routing
// One warp routes one token. lane l holds experts [8l, 8l+8).
// Writes 8 global expert ids and 8 weights (lane i < 8 holds entry i).
// Mimics the reference: sigmoid, +bias (bf16->f32), 8 groups of 32, group score = top2 sum, keep 4 groups,
// top-8 over kept, weights = s / sum(s) * scaling.
__device__ __forceinline__ void route_token_warp(const float* __restrict__ logits_row, const float* __restrict__ bias_smem,
                                                 float scaling, int& out_id, float& out_w, long long* tr = nullptr) {
    const int lane = threadIdx.x & 31;
    float s[8], v[8];
    if (tr && lane == 0) tr[0] = clock64();
    {
        float4 x0 = *reinterpret_cast<const float4*>(logits_row + lane * 8);
        float4 x1 = *reinterpret_cast<const float4*>(logits_row + lane * 8 + 4);
        float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s[i] = 1.0f / (1.0f + expf(-xs[i]));
            v[i] = s[i] + bias_smem[lane * 8 + i];
        }
    }
    if (tr && lane == 0) { tr[1] = clock64(); tr[7] = (long long)__float_as_int(v[0]); }
    // local top-2
    float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (v[i] > m1) { m2 = m1; m1 = v[i]; }
        else if (v[i] > m2) { m2 = v[i]; }
    }
    // merge within group (4 lanes)
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        float o1 = __shfl_xor_sync(0xffffffffu, m1, off);
        float o2 = __shfl_xor_sync(0xffffffffu, m2, off);
        // top2 of {m1,m2,o1,o2} (m1>=m2, o1>=o2)
        float n1, n2;
        if (m1 >= o1) { n1 = m1; n2 = fmaxf(m2, o1); }
        else { n1 = o1; n2 = fmaxf(o2, m1); }
        m1 = n1; m2 = n2;
    }
    float gscore = m1 + m2;
    const int g = lane >> 2;
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float gs = __shfl_sync(0xffffffffu, gscore, j * 4);
        rank += (gs > gscore) || (gs == gscore && j < g);
    }
    const bool keep = rank < 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = keep ? v[i] : -INFINITY;

    if (tr && lane == 0) tr[2] = clock64();
    int my_id = -1; float my_s = 0.f;
#pragma unroll
    for (int r = 0; r < TOP_K; ++r) {
        // local argmax (value desc, index asc)
        float bv = v[0]; int bi = 0;
#pragma unroll
        for (int i = 1; i < 8; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
        // warp argmax: order-preserving integer key + hardware max-reduce, lowest lane among the maxima
        const uint32_t kb = __float_as_uint(bv);
        const uint32_t key = (kb & 0x80000000u) ? ~kb : (kb | 0x80000000u);
        const uint32_t kmax = __reduce_max_sync(0xffffffffu, key);
        const unsigned win = __ballot_sync(0xffffffffu, key == kmax);
        const int wl = __ffs(win) - 1;
        const int bidx = __shfl_sync(0xffffffffu, wl * 8 + bi, wl);
        if (lane == r) { my_id = bidx; }
        if (wl == lane) {
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = (i == bi) ? -INFINITY : v[i];   // keep v[] in registers (no dynamic indexing)
        }
    }
    if (tr && lane == 0) tr[3] = clock64();
    // weights: s of selected experts
    {
        int owner = my_id >> 3, sub = my_id & 7;
        // fetch s[sub] from lane owner: do via shuffles over the 8 selected
        float sel = 0.f;
#pragma unroll
        for (int r = 0; r < TOP_K; ++r) {
            int id = __shfl_sync(0xffffffffu, my_id, r);
            float val = 0.f;
            int o = id >> 3, sb = id & 7;
            // lane o picks s[sb]
            float pick = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) pick = (sb == i) ? s[i] : pick;
            val = __shfl_sync(0xffffffffu, pick, o);
            if (lane == r) sel = val;
        }
        my_s = sel;
        (void)owner; (void)sub;
    }
    float sum = (lane < TOP_K) ? my_s : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    out_id = my_id;
    out_w = my_s / (sum + 1e-20f) * scaling;
    if (tr && lane == 0) tr[4] = clock64();
}


// 4 tokens per warp, 8 lanes per token, one lane per expert group (32 experts): same semantics/tie rules as
// route_token_warp, with 3 shuffle steps per selection round instead of 5 and 4x the tokens per round.
// Lane (tok = lane>>3, g = lane&7) ends with out_id/out_w = selected expert #g of token t0+tok (if t0+tok < T).
__device__ __forceinline__ void route_tokens_warp4(const float* __restrict__ logits, int T, int t0,
                                                   const float* __restrict__ bias_smem, float scaling,
                                                   int& out_id, float& out_w) {
    const int lane = threadIdx.x & 31;
    const int tok = lane >> 3, g = lane & 7;
    const int t = t0 + tok;
    const bool active = t < T;
    float s[32], v[32];
    {
        const float4* src = reinterpret_cast<const float4*>(logits + (size_t)(active ? t : 0) * E_GLOBAL + g * 32);
        const float4* bb = reinterpret_cast<const float4*>(bias_smem + g * 32);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float4 x = src[q];
            const float4 b = bb[q];
            s[4 * q + 0] = 1.0f / (1.0f + expf(-x.x)); v[4 * q + 0] = s[4 * q + 0] + b.x;
            s[4 * q + 1] = 1.0f / (1.0f + expf(-x.y)); v[4 * q + 1] = s[4 * q + 1] + b.y;
            s[4 * q + 2] = 1.0f / (1.0f + expf(-x.z)); v[4 * q + 2] = s[4 * q + 2] + b.z;
            s[4 * q + 3] = 1.0f / (1.0f + expf(-x.w)); v[4 * q + 3] = s[4 * q + 3] + b.w;
        }
    }
    // group score = top-2 sum within this lane's 32 experts
    float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (v[i] > m1) { m2 = m1; m1 = v[i]; }
        else if (v[i] > m2) { m2 = v[i]; }
    }
    const float gscore = m1 + m2;
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float gs = __shfl_sync(0xffffffffu, gscore, j, 8);
        rank += (gs > gscore) || (gs == gscore && j < g);
    }
    const bool keep = rank < 4;
#pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = keep ? v[i] : -INFINITY;

    int my_id = -1;
#pragma unroll
    for (int r = 0; r < TOP_K; ++r) {
        // lane-local argmax (value desc, index asc) as a tree for ILP
        float bv[32]; int bi[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) { bv[i] = v[i]; bi[i] = i; }
#pragma unroll
        for (int w = 16; w > 0; w >>= 1) {
#pragma unroll
            for (int i = 0; i < w; ++i) {
                const bool take = bv[i + w] > bv[i];
                bv[i] = take ? bv[i + w] : bv[i];
                bi[i] = take ? bi[i + w] : bi[i];
            }
        }
        float best = bv[0];
        int bidx = g * 32 + bi[0];
#pragma unroll
        for (int off = 1; off <= 4; off <<= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bidx, off);
            if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
        }
        if (g == r) my_id = bidx;
        if ((bidx >> 5) == g) {
            const int kill = bidx & 31;
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = (i == kill) ? -INFINITY : v[i];
        }
    }
    // weights: s of the selected experts, normalized over the 8 picks
    float sel = 0.f;
#pragma unroll
    for (int r = 0; r < TOP_K; ++r) {
        const int id = __shfl_sync(0xffffffffu, my_id, r, 8);
        const int o = id >> 5, sb = id & 31;
        float pick = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) pick = (sb == i) ? s[i] : pick;
        const float val = __shfl_sync(0xffffffffu, pick, o, 8);
        if (g == r) sel = val;
    }
    float sum = sel;
#pragma unroll
    for (int off = 1; off <= 4; off <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    out_id = my_id;
    out_w = sel / (sum + 1e-20f) * scaling;
}

}  // namespace moe
