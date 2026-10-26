// Low-level primitives for sm_100a: mbarrier, TMA, TMEM, tcgen05 MMA.  Written from scratch.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------- mbarrier ----------------
DEVI void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
DEVI void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEVI uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
    while (!mbar_try_wait(bar, parity)) {}
}

// ---------------- TMA ----------------
DEVI void tma_load_3d(const CUtensorMap* map, uint32_t smem_dst, uint32_t bar, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                 ::"r"(smem_dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_prefetch_desc(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}

// ---------------- TMEM ----------------
DEVI void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

#define R8(b)  "=r"(r[b+0]), "=r"(r[b+1]), "=r"(r[b+2]), "=r"(r[b+3]), "=r"(r[b+4]), "=r"(r[b+5]), "=r"(r[b+6]), "=r"(r[b+7])
#define W8(b)  "r"(r[b+0]), "r"(r[b+1]), "r"(r[b+2]), "r"(r[b+3]), "r"(r[b+4]), "r"(r[b+5]), "r"(r[b+6]), "r"(r[b+7])

DEVI void tmem_ld_x16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : R8(0), R8(8) : "r"(taddr));
}
DEVI void tmem_ld_x32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : R8(0), R8(8), R8(16), R8(24) : "r"(taddr));
}
DEVI void tmem_ld_x64(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                 "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
                 "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
                 "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
                 : R8(0), R8(8), R8(16), R8(24), R8(32), R8(40), R8(48), R8(56) : "r"(taddr));
}
DEVI void tmem_st_x16(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
                 ::"r"(taddr), W8(0), W8(8) : "memory");
}
DEVI void tmem_st_x32(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
                 "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
                 ::"r"(taddr), W8(0), W8(8), W8(16), W8(24) : "memory");
}

// ---------------- descriptors ----------------
// Shared-memory matrix descriptor. swizzle: 0 none, 2 = 128B, 4 = 64B, 6 = 32B
DEVI uint64_t make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle) {
    uint64_t d = 0;
    d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle << 61;
    return d;
}
// Instruction descriptor for kind::f16, D = f32.
DEVI uint32_t make_idesc_f16(int M, int N, bool a_bf16, bool b_bf16, bool a_mn_major, bool b_mn_major) {
    uint32_t d = 0;
    d |= 1u << 4;
    d |= (a_bf16 ? 1u : 0u) << 7;
    d |= (b_bf16 ? 1u : 0u) << 10;
    d |= (a_mn_major ? 1u : 0u) << 15;
    d |= (b_mn_major ? 1u : 0u) << 16;
    d |= (uint32_t)(N >> 3) << 17;
    d |= (uint32_t)(M >> 4) << 24;
    return d;
}

// ---------------- MMA ----------------
DEVI void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

// ---------------- host helpers ----------------
typedef CUresult (*PFN_encodeTiled)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                    const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static PFN_encodeTiled get_encode_fn() {
    static PFN_encodeTiled fn = nullptr;
    if (!fn) {
        cudaDriverEntryPointQueryResult q;
        void* p = nullptr;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        fn = (PFN_encodeTiled)p;
    }
    return fn;
}
// 3D bf16 map for a (S, H, D) tensor: dim0 = D, dim1 = H, dim2 = S.  box = {64, 1, rows}
static int make_map_shd(CUtensorMap* map, const void* ptr, uint64_t S, uint64_t H, uint64_t D, uint32_t rows, uint32_t d_box = 64) {
    cuuint64_t dims[3] = {D, H, S};
    cuuint64_t strides[2] = {D * 2, H * D * 2};
    cuuint32_t box[3] = {d_box, 1, rows};
    cuuint32_t estr[3] = {1, 1, 1};
    CUresult r = get_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return (int)r;
}

// ---- additions ----
DEVI void tma_load_4d(const CUtensorMap* map, uint32_t smem_dst, uint32_t bar, int c0, int c1, int c2, int c3) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
                 ::"r"(smem_dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tmem_st_x64(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
                 "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,"
                 "%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,"
                 "%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64};"
                 ::"r"(taddr), W8(0), W8(8), W8(16), W8(24), W8(32), W8(40), W8(48), W8(56) : "memory");
}
DEVI float fmax3(float a, float b, float c) {
    float d;
    asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
DEVI float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
DEVI uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}
template <int N> DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N> DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }

// 4D bf16 map for a (B, S, H, D) tensor: dim0 = D, dim1 = H, dim2 = S, dim3 = B.  box = {64, 1, rows, 1}
static int make_map_bshd(CUtensorMap* map, const void* ptr, uint64_t B, uint64_t S, uint64_t H, uint64_t D, uint32_t rows) {
    cuuint64_t dims[4] = {D, H, S, B};
    cuuint64_t strides[3] = {D * 2, H * D * 2, S * H * D * 2};
    cuuint32_t box[4] = {64, 1, rows, 1};
    cuuint32_t estr[4] = {1, 1, 1, 1};
    CUresult r = get_encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(ptr), dims, strides, box, estr,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return (int)r;
}

// ---- fast wait: test_wait first (8 clk), fall back to try_wait (suspends) ----
DEVI uint32_t mbar_test_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
DEVI void mbar_wait_fast(uint32_t bar, uint32_t parity) {
    if (mbar_test_wait(bar, parity)) return;
    while (!mbar_try_wait(bar, parity)) {}
}
DEVI void named_bar_sync(int id, int nthreads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}

// ---- cluster / multicast primitives ----
DEVI uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI void cluster_sync_all() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEVI void tma_load_4d_mc(const CUtensorMap* map, uint32_t smem_dst, uint32_t bar, int c0, int c1, int c2, int c3, uint16_t mask) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                 ::"r"(smem_dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "h"(mask) : "memory");
}
DEVI void mma_commit_mc(uint32_t bar, uint16_t mask) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"(mask) : "memory");
}

// ---- single-thread-elected tcgen05 ops usable from convergent (all-lane) code: compiler emits predicated uniform instrs ----
DEVI void mma_ss_e(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p, q;\nelect.sync _|q, 0xffffffff;\nsetp.ne.b32 p, %4, 0;\n"
                 "@q tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_ts_e(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p, q;\nelect.sync _|q, 0xffffffff;\nsetp.ne.b32 p, %4, 0;\n"
                 "@q tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_commit_e(uint32_t bar) {
    asm volatile("{\n.reg .pred q;\nelect.sync _|q, 0xffffffff;\n@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n" ::"r"(bar) : "memory");
}
DEVI void mma_commit_mc_e(uint32_t bar, uint16_t mask) {
    asm volatile("{\n.reg .pred q;\nelect.sync _|q, 0xffffffff;\n@q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n}\n" ::"r"(bar), "h"(mask) : "memory");
}
DEVI uint32_t mbar_try_wait_hint(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 1000000;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
DEVI void mbar_wait_fast2(uint32_t bar, uint32_t parity) {
    if (mbar_test_wait(bar, parity)) return;
    while (!mbar_try_wait_hint(bar, parity)) {}
}

// ---- pinned waits: tie the asynchronously loaded/stored registers to the wait so the compiler cannot
//      consume them (ld) or overwrite them (st) before the wait completes ----
#define P8(b)  "+r"(r[b+0]), "+r"(r[b+1]), "+r"(r[b+2]), "+r"(r[b+3]), "+r"(r[b+4]), "+r"(r[b+5]), "+r"(r[b+6]), "+r"(r[b+7])
DEVI void tmem_ld_wait_pin16(uint32_t* r) { asm volatile("tcgen05.wait::ld.sync.aligned;" : P8(0), P8(8) :: "memory"); }
DEVI void tmem_ld_wait_pin32(uint32_t* r) { asm volatile("tcgen05.wait::ld.sync.aligned;" : P8(0), P8(8), P8(16), P8(24) :: "memory"); }
DEVI void tmem_ld_wait_pin64(uint32_t* r) { asm volatile("tcgen05.wait::ld.sync.aligned;" : P8(0), P8(8), P8(16), P8(24), P8(32), P8(40), P8(48), P8(56) :: "memory"); }
DEVI void tmem_st_wait_pin16(uint32_t* r) { asm volatile("tcgen05.wait::st.sync.aligned;" : P8(0), P8(8) :: "memory"); }
DEVI void tmem_st_wait_pin32(uint32_t* r) { asm volatile("tcgen05.wait::st.sync.aligned;" : P8(0), P8(8), P8(16), P8(24) :: "memory"); }

// ---- cta_group::2 (2-SM) MMA primitives ----
DEVI void mma2_ss_e(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p, q;\nelect.sync _|q, 0xffffffff;\nsetp.ne.b32 p, %4, 0;\n"
                 "@q tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma2_ts_e(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p, q;\nelect.sync _|q, 0xffffffff;\nsetp.ne.b32 p, %4, 0;\n"
                 "@q tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma2_commit_mc_e(uint32_t bar, uint16_t mask) {
    asm volatile("{\n.reg .pred q;\nelect.sync _|q, 0xffffffff;\n@q tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n}\n" ::"r"(bar), "h"(mask) : "memory");
}
DEVI void tmem_alloc2(uint32_t dst_smem, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory"); }
// map a local shared address to the same offset in another CTA of the cluster
DEVI uint32_t mapa_shared(uint32_t local_addr, uint32_t cta_rank) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_addr), "r"(cta_rank)); return r;
}
DEVI void mbar_arrive_remote(uint32_t cluster_addr) {
#ifdef REMOTE_ARRIVE_CLUSTER_RELEASE
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
#else
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");   // default sem/scope (.release.cta), as CUTLASS's ClusterBarrier
#endif
}
DEVI void tma_load_4d_remote_bar(const CUtensorMap* map, uint32_t smem_dst, uint32_t bar_cluster_addr, int c0, int c1, int c2, int c3) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
                 ::"r"(smem_dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar_cluster_addr), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tma_load_3d_bar(const CUtensorMap* map, uint32_t smem_dst, uint32_t bar_cluster_addr, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
                 ::"r"(smem_dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(bar_cluster_addr), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
