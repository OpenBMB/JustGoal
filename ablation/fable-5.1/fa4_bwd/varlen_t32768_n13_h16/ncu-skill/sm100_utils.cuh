// Minimal sm_100a (Blackwell) primitives: mbarrier, TMA, tcgen05 (TMEM / UMMA), named barriers.
// Written from the PTX ISA; CuTe is only used for UMMA smem descriptor construction and layouts.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

namespace k {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t lane_id() { uint32_t r; asm volatile("mov.u32 %0, %%laneid;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t warp_id() { return threadIdx.x >> 5; }
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
  return pred != 0;
}

// ---------------- mbarrier ----------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
#ifdef WAIT_HINT
__device__ __forceinline__ bool mbar_try_wait_hint(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase), "r"((uint32_t)WAIT_HINT) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  while (!mbar_try_wait_hint(bar, phase)) {}
}
#else
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  while (!mbar_try_wait(bar, phase)) {}
}
#endif
// pure polling (test_wait never suspends the thread): lowest wake-up latency for hops on the critical chain
__device__ __forceinline__ void mbar_wait_poll(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  } while (!ok);
}

// ---------------- TMA ----------------
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map, uint64_t* bar, void* smem_dst, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(smem_dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
// L2 prefetch of a 3D tile (no smem destination, no completion)
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* map, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global [%0, {%1, %2, %3}];"
               :: "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map, uint64_t* bar, void* smem_dst, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(smem_u32(smem_dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_reduce_add_2d(const CUtensorMap* map, const void* smem_src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, const void* smem_src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map, const void* smem_src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
               :: "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void bulk_commit_group() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_group_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_group() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

__device__ __forceinline__ void sts_b32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t lds_b32(uint32_t addr) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }
// ---------------- explicit shared-memory accesses (avoid generic LD/ST) ----------------
__device__ __forceinline__ float lds_f32(uint32_t addr) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr)); return v; }
__device__ __forceinline__ uint4 lds_v4(uint32_t addr) {
  uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr)); return v;
}
__device__ __forceinline__ void sts_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ int lds_s32(uint32_t addr) { int v; asm volatile("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }

// ---------------- register reallocation (whole warpgroup must execute) ----------------
template <int R> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(R)); }
template <int R> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(R)); }

__device__ __forceinline__ bool mbar_test_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ int atom_add_acqrel_gpu(int* ptr, int v) {
  int old; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ float ldcg_f32(const float* ptr) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(ptr)); return v; }

__device__ __forceinline__ void red_add_f16x2(void* ptr, uint32_t h2) {
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" :: "l"(ptr), "r"(h2));
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi)); return r;   // low half = lo
}
__device__ __forceinline__ void red_add_v4(float* ptr, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

// ---------------- named barriers ----------------
// producer side of a hardware named barrier (non-blocking, whole warp); consumer uses named_bar_sync with the same count
__device__ __forceinline__ void named_bar_arrive_all(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}

// ---------------- TMEM ----------------
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_dst, uint32_t ncols) {   // whole warp
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(smem_dst)), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {      // whole warp
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after()  { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// tcgen05.ld 32x32b.xN : thread i of the warp gets lane (warp%4)*32+i, N consecutive columns
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                 "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                 "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(taddr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                  "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]) : "memory");
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(taddr), "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]),
                  "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),
                  "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
                  "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]) : "memory");
}

// ---------------- UMMA ----------------
// idesc: bf16 x bf16 -> f32 (or f16 accumulator), M x N, A/B majors.
template <int M, int N, cute::UMMA::Major AM, cute::UMMA::Major BM, class TC = float>
__device__ __forceinline__ uint32_t make_idesc() {
  return static_cast<uint32_t>(cute::UMMA::make_instr_desc<cute::bfloat16_t, cute::bfloat16_t, TC, M, N, AM, BM>());
}
__device__ __forceinline__ void unpack_f16x2(uint32_t h, float& lo, float& hi) {
  asm("{\n .reg .f16 l, u;\n mov.b32 {l, u}, %2;\n cvt.f32.f16 %0, l;\n cvt.f32.f16 %1, u;\n}" : "=f"(lo), "=f"(hi) : "r"(h));
}
// D[tmem] (+)= A[smem desc] * B[smem desc]
__device__ __forceinline__ void umma_ss(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate) : "memory");
}
// D[tmem] (+)= A[tmem] * B[smem desc]
__device__ __forceinline__ void umma_ts(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}"
               :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(bar)) : "memory");
}
// advance a smem descriptor start address by `bytes` (must be multiple of 16)
__device__ __forceinline__ uint64_t desc_advance(uint64_t desc, uint32_t bytes) {
  return desc + static_cast<uint64_t>(bytes >> 4);
}

// Canonical smem layouts (CuTe): rows of 128B, 8-row swizzle period == TMA SWIZZLE_128B boxes
using cute::bfloat16_t;
// K-major tile (MN rows, KK cols, K contiguous).  Boxes of 64 K-elements stacked along MN first.
template <int MN, int KK>
using SmemLayoutK = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<bfloat16_t>{}, cute::Shape<cute::Int<MN>, cute::Int<KK>>{}));
// K-major tile with 32 B rows (one 16-element k-step per row): SW32 swizzle (TMA SWIZZLE_32B boxes of 16 bf16)
template <int MN, int KK>
using SmemLayoutK32 = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW32_Atom<bfloat16_t>{}, cute::Shape<cute::Int<MN>, cute::Int<KK>>{}));
// MN-major tile (MN contiguous): same memory as a K-major (KK rows of MN) tile loaded by TMA with 64-elem boxes.
template <int MN, int KK>
using SmemLayoutMN = decltype(cute::tile_to_shape(cute::UMMA::Layout_MN_SW128_Atom<bfloat16_t>{}, cute::Shape<cute::Int<MN>, cute::Int<KK>>{}, cute::Step<cute::_2, cute::_1>{}));

// Descriptor for the first (MN x 16) K-slice of the tile; advance with desc_advance for other slices.
template <cute::UMMA::Major MJ, class Layout>
__device__ __forceinline__ uint64_t make_desc(const void* smem, Layout const& l) {
  auto t = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<const bfloat16_t*>(smem)), l);
  auto t16 = cute::local_tile(t, cute::Shape<cute::Int<cute::size<0>(Layout{})>, cute::_16>{}, cute::make_coord(0, 0));
  return static_cast<uint64_t>(cute::UMMA::make_umma_desc<MJ>(t16));
}

} // namespace k

// ---------------- cluster / DSMEM ----------------
namespace k {
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
__device__ __forceinline__ void st_cluster_b32(uint32_t raddr, uint32_t v) {
  asm volatile("st.shared::cluster.b32 [%0], %1;" :: "r"(raddr), "r"(v) : "memory");
}
__device__ __forceinline__ void st_cluster_s32(uint32_t raddr, int v) {
  asm volatile("st.shared::cluster.s32 [%0], %1;" :: "r"(raddr), "r"(v) : "memory");
}
// arrive (release, cluster scope) on an mbarrier given by its cluster-mapped shared address
__device__ __forceinline__ void mbar_arrive_remote(uint32_t raddr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(raddr) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait_acq_cluster(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
} // namespace k
