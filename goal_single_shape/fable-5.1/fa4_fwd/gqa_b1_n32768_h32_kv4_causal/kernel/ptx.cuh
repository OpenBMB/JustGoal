// Minimal PTX wrappers for sm_100a: mbarrier, TMA, tcgen05 (TMEM + UMMA). Written from the PTX ISA.
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
DEVI void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(tx) : "memory");
}
DEVI void mbar_expect_tx(uint32_t bar, uint32_t tx) {  // no arrival, only tx-count
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(tx) : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
// arrive on a barrier in a (possibly remote) CTA of the cluster; addr from mapa()
DEVI void mbar_arrive_cluster(uint32_t bar_cluster) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar_cluster) : "memory");
}
// async 4-byte store to a peer CTA's smem that performs complete_tx(4) on the peer's mbarrier (fire-and-forget signal)
DEVI void st_async_signal(uint32_t remote_addr, uint32_t value, uint32_t remote_bar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" :: "r"(remote_addr), "r"(value), "r"(remote_bar) : "memory");
}
DEVI uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p;}"
               : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  return ok;
}
DEVI uint32_t mbar_try_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2, %3; selp.u32 %0, 1, 0, p;}"
               : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  return ok;
}
DEVI uint32_t mbar_try_wait_nohint(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
DEVI uint32_t mbar_try_wait_cluster_nohint(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
DEVI void mbar_wait_nohint(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_nohint(bar, parity)) {} }
DEVI uint32_t mbar_test_wait(uint32_t bar, uint32_t parity) {   // non-blocking probe
  uint32_t ok;
  asm volatile("{.reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) {} }
DEVI void mbar_wait_cluster(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_cluster(bar, parity)) {} }

// ---------------- cluster ----------------
DEVI uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI void cluster_sync() { cluster_arrive(); cluster_wait(); }

// ---------------- TMA ----------------
DEVI void tma_prefetch_desc(const void* tm) { asm volatile("prefetch.tensormap [%0];" :: "l"(tm) : "memory"); }
DEVI void tma_load_2d(uint32_t dst, const void* tm, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(tm), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_load_3d(uint32_t dst, const void* tm, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(tm), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
// Same, but the mbarrier may live in the peer CTA of a cta_group::2 pair (bar is a shared::cluster address).
DEVI void tma_load_3d_cg2(uint32_t dst, const void* tm, uint32_t bar_cluster, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(tm), "r"(bar_cluster), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_load_2d_cg2(uint32_t dst, const void* tm, uint32_t bar_cluster, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(tm), "r"(bar_cluster), "r"(c0), "r"(c1) : "memory");
}
// multicast to the CTAs in ctaMask; complete_tx lands on the barrier at the same offset in the pair leader of each destination
DEVI void tma_load_3d_mc_cg2(uint32_t dst, const void* tm, uint32_t bar_cluster, int c0, int c1, int c2, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(tm), "r"(bar_cluster), "r"(c0), "r"(c1), "r"(c2), "h"(mask) : "memory");
}
DEVI void tma_store_3d(const void* tm, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
               :: "l"(tm), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_store_2d(const void* tm, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(tm), "r"(src), "r"(c0), "r"(c1) : "memory");
}
DEVI void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void tma_store_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void tma_store_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DEVI void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ---------------- tcgen05: TMEM alloc ----------------
DEVI void tmem_alloc1(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish1() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc1(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tmem_alloc2(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after()  { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// ---------------- tcgen05: UMMA ----------------
// SMEM matrix descriptor (sm100). layout: 0 none, 2 = SWIZZLE_128B, 4 = 64B, 6 = 32B
DEVI uint64_t make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;                 // descriptor version (sm100)
  d |= (uint64_t)(layout & 7) << 61;
  return d;
}
// Instruction descriptor: kind::f16 with BF16 A/B, F32 accumulate.
__host__ __device__ constexpr uint32_t make_idesc_bf16(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16)
       | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}"
               :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void umma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;}"
               :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void umma2_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}"
               :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void umma2_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;}"
               :: "r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void umma_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
// cta_group::2 commit, multicast to the barrier at the same smem offset in the CTAs of ctamask
DEVI void umma2_commit_mc(uint32_t bar, uint16_t ctamask) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               :: "r"(bar), "h"(ctamask) : "memory");
}
DEVI void umma2_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}

// ---------------- tcgen05: TMEM ld/st (32x32b shape: lane = thread) ----------------
#define R8(i) "=r"(r[i+0]), "=r"(r[i+1]), "=r"(r[i+2]), "=r"(r[i+3]), "=r"(r[i+4]), "=r"(r[i+5]), "=r"(r[i+6]), "=r"(r[i+7])
#define I8(i) "r"(r[i+0]), "r"(r[i+1]), "r"(r[i+2]), "r"(r[i+3]), "r"(r[i+4]), "r"(r[i+5]), "r"(r[i+6]), "r"(r[i+7])
DEVI void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : R8(0), R8(8), R8(16), R8(24) : "r"(taddr));
}
DEVI void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : R8(0), R8(8) : "r"(taddr));
}
DEVI void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(taddr), I8(0), I8(8), I8(16), I8(24) : "memory");
}
DEVI void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(taddr), I8(0), I8(8) : "memory");
}
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ---------------- release/acquire shared flags ----------------
DEVI void st_release_u32(uint32_t addr, uint32_t v) { asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
DEVI uint32_t ld_acquire_u32(uint32_t addr) { uint32_t v; asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory"); return v; }
// ---------------- misc ----------------
DEVI uint32_t elect_one() {  // returns 1 for exactly one active lane of the warp
  uint32_t pred = 0;
  asm volatile("{.reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p;}" : "=r"(pred));
  return pred;
}
DEVI void named_bar_sync(uint32_t id, uint32_t nthreads) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory"); }
DEVI void named_bar_arrive(uint32_t id, uint32_t nthreads) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(nthreads) : "memory"); }
DEVI void setmaxnreg_inc(int n) = delete;
#define SETMAXNREG_INC(n) asm volatile("setmaxnreg.inc.sync.aligned.u32 " #n ";")
#define SETMAXNREG_DEC(n) asm volatile("setmaxnreg.dec.sync.aligned.u32 " #n ";")

// ---------------- host: tensor map creation via driver entry point ----------------
#include <cstdio>
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
// rank-r bf16 tiled map, dims[0] innermost. strides in bytes for dims 1..r-1. box in elements. swizzle 128B.
static int make_tmap_bf16(CUtensorMap* tm, void* base, int rank, const uint64_t* dims, const uint64_t* strides_bytes,
                          const uint32_t* box, CUtensorMapSwizzle sw = CU_TENSOR_MAP_SWIZZLE_128B) {
  uint32_t estr[5] = {1, 1, 1, 1, 1};
  CUresult r = get_encode_fn()(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, base, dims, strides_bytes, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)r); return -1; }
  return 0;
}
