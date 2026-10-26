// Hand-written PTX wrappers for sm_100a: mbarrier, TMA, TMEM, tcgen05.mma
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------- mbarrier ----------------
DEVI void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
// arrive on a barrier in another CTA of the cluster (addr from mapa)
DEVI void mbar_arrive_cluster(uint32_t bar_cluster_addr) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar_cluster_addr) : "memory");
}
DEVI void mbar_arrive_cluster_relaxed(uint32_t bar_cluster_addr) {
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar_cluster_addr) : "memory");
}
DEVI void mbar_arrive_relaxed(uint32_t bar) {
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
DEVI uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
DEVI bool mbar_try_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
DEVI bool mbar_test_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
DEVI bool mbar_try_wait_cluster(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) {} }
DEVI bool mbar_test_wait_cluster(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
DEVI bool mbar_try_wait_cluster_hint(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2, %3;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(hint_ns) : "memory");
    return ok != 0;
}
DEVI bool mbar_try_wait_hint(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(hint_ns) : "memory");
    return ok != 0;
}
#ifdef WAIT_HINT
DEVI void mbar_wait_cluster(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_cluster_hint(bar, parity, WAIT_HINT)) {} }
DEVI void mbar_wait_hint(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_hint(bar, parity, WAIT_HINT)) {} }
#else
DEVI void mbar_wait_cluster(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_cluster(bar, parity)) {} }
DEVI void mbar_wait_hint(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) {} }
#endif
// sleep-based wait for helper warps (issuers/relays): poll with nanosleep backoff
DEVI void mbar_wait_cluster_sleep(uint32_t bar, uint32_t parity, uint32_t ns) { while (!mbar_try_wait_cluster(bar, parity)) { __nanosleep(ns); } }
DEVI void mbar_wait_sleep(uint32_t bar, uint32_t parity, uint32_t ns) { while (!mbar_try_wait(bar, parity)) { __nanosleep(ns); } }

// ---------------- cluster ----------------
DEVI uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

// ---------------- TMA ----------------
DEVI void tma_prefetch_desc(const void* map) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
DEVI void tma_load_3d(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_load_3d_hint(uint32_t dst, const void* map, uint32_t bar, int c0, int c1, int c2, uint64_t policy) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(policy) : "memory");
}
DEVI void tma_store_3d(const void* map, uint32_t src, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
                 :: "l"(reinterpret_cast<uint64_t>(map)), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void tma_store_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void tma_store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DEVI void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ---------------- TMEM ----------------
DEVI void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

#define R32(i) "=r"(r[i])
DEVI void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : R32(0),R32(1),R32(2),R32(3),R32(4),R32(5),R32(6),R32(7),R32(8),R32(9),R32(10),R32(11),R32(12),R32(13),R32(14),R32(15),
          R32(16),R32(17),R32(18),R32(19),R32(20),R32(21),R32(22),R32(23),R32(24),R32(25),R32(26),R32(27),R32(28),R32(29),R32(30),R32(31)
        : "r"(taddr) : "memory");
}
DEVI void tmem_ld_32x32b_x16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : R32(0),R32(1),R32(2),R32(3),R32(4),R32(5),R32(6),R32(7),R32(8),R32(9),R32(10),R32(11),R32(12),R32(13),R32(14),R32(15)
        : "r"(taddr) : "memory");
}
#undef R32
#define W32(i) "r"(r[i])
DEVI void tmem_st_32x32b_x32(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
        :: W32(0),W32(1),W32(2),W32(3),W32(4),W32(5),W32(6),W32(7),W32(8),W32(9),W32(10),W32(11),W32(12),W32(13),W32(14),W32(15),
           W32(16),W32(17),W32(18),W32(19),W32(20),W32(21),W32(22),W32(23),W32(24),W32(25),W32(26),W32(27),W32(28),W32(29),W32(30),W32(31),
           "r"(taddr) : "memory");
}
DEVI void tmem_st_32x32b_x16(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
        :: W32(0),W32(1),W32(2),W32(3),W32(4),W32(5),W32(6),W32(7),W32(8),W32(9),W32(10),W32(11),W32(12),W32(13),W32(14),W32(15),
           "r"(taddr) : "memory");
}
DEVI void tmem_st_32x32b_x8(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
        :: W32(0),W32(1),W32(2),W32(3),W32(4),W32(5),W32(6),W32(7), "r"(taddr) : "memory");
}
#undef W32

// ---------------- tcgen05.mma ----------------
// smem matrix descriptor
DEVI uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle_code, uint32_t const_bits = 1, uint32_t base_off = 0) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)(const_bits & 7) << 46;
    d |= (uint64_t)(base_off & 7) << 49;
    d |= (uint64_t)(swizzle_code & 7) << 61;
    return d;
}
// instruction descriptor for kind::f16
DEVI uint32_t make_idesc_f16(uint32_t M, uint32_t N, bool a_bf16, bool b_bf16, bool a_mn, bool b_mn, bool d_f32) {
    uint32_t d = 0;
    d |= (d_f32 ? 1u : 0u) << 4;
    d |= (a_bf16 ? 1u : 0u) << 7;
    d |= (b_bf16 ? 1u : 0u) << 10;
    d |= (a_mn ? 1u : 0u) << 15;
    d |= (b_mn ? 1u : 0u) << 16;
    d |= (N >> 3) << 17;
    d |= (M >> 4) << 24;
    return d;
}
DEVI void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma2_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
                 :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma2_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\ttcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}"
                 :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
DEVI void mma2_commit_mc(uint32_t bar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(bar), "h"(mask) : "memory");
}
DEVI void mma2_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}

// ---------------- host: tensor map ----------------
typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PFN_encodeTiled get_encode_fn() {
    static PFN_encodeTiled fn = nullptr;
    if (!fn) {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        fn = (PFN_encodeTiled)p;
    }
    return fn;
}
// 3D map over a (S, H, D) bf16 tensor (row = s*H + h): dims {D, H, S}
static int make_map_bf16_3d(CUtensorMap* map, void* ptr, uint64_t D, uint64_t H, uint64_t S, uint32_t box_d, uint32_t box_s, int swizzle128 = 1) {
    cuuint64_t dims[3] = {D, H, S};
    cuuint64_t strides[2] = {D * 2, H * D * 2};
    cuuint32_t box[3] = {box_d, 1, box_s};
    cuuint32_t es[3] = {1, 1, 1};
    CUresult r = get_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return (int)r;
}

// cta_group::2 TMEM management
DEVI void tmem_alloc2(uint32_t dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}

// fast waits: poll with non-suspending test_wait first (cheap when already complete), then fall back to try_wait
DEVI bool mbar_test_wait_cluster_acq(uint32_t bar, uint32_t parity) { return mbar_test_wait_cluster(bar, parity); }
DEVI void mbar_wait_fast(uint32_t bar, uint32_t parity) {
    #pragma unroll 1
    for (int i = 0; i < 8; i++) if (mbar_test_wait(bar, parity)) return;
    while (!mbar_try_wait(bar, parity)) {}
}
DEVI void mbar_wait_cluster_fast(uint32_t bar, uint32_t parity) {
    #pragma unroll 1
    for (int i = 0; i < 8; i++) if (mbar_test_wait_cluster(bar, parity)) return;
    while (!mbar_try_wait_cluster(bar, parity)) {}
}
