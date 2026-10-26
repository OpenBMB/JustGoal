// Minimal Blackwell (sm_100a) tcgen05 / mbarrier / cluster helpers. Written from the PTX ISA + CUTLASS descriptor
// field definitions; no attention-kernel code was consulted.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ------------------------------------------------------------------ mbarrier
DEVI void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEVI void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEVI void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEVI void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}
DEVI void mbar_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}
DEVI uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok;
}
DEVI uint32_t mbar_test_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
DEVI void mbar_wait_poll(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) { } }
DEVI void mbar_wait_sleep(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) { } }
DEVI void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}
// cluster helpers
DEVI uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DEVI uint32_t mapa_shared(uint32_t addr, uint32_t cta) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
  return r;
}
DEVI void mbar_arrive_remote(uint32_t remote_bar) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote_bar) : "memory");
}
DEVI void mbar_arrive_generic(uint32_t addr, uint32_t rank) { if (rank == 0) mbar_arrive(addr); else mbar_arrive_remote(addr); }
DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
DEVI void cluster_sync() { cluster_arrive(); cluster_wait(); }

// ------------------------------------------------------------------ cp.async (16B) with mbarrier completion
DEVI void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEVI void cp_async_16_zfill(uint32_t dst, const void* src, bool valid) {
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(sz) : "memory");
}
DEVI void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// ------------------------------------------------------------------ TMEM
template <int CG = 1>
DEVI void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols) : "memory");
  else
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols) : "memory");
}
template <int CG = 1>
DEVI void tmem_relinquish() {
  if constexpr (CG == 1) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  else asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
template <int CG = 1>
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
  else
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ------------------------------------------------------------------ descriptors
// SMEM matrix descriptor. layout: 0 none, 2 = 128B swizzle.
DEVI uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;  // version = 1 (sm100)
  d |= (uint64_t)layout << 61;
  return d;
}
// Instruction descriptor for kind::f16, bf16 inputs, f32 accum.
__host__ __device__ constexpr uint32_t make_idesc_bf16(int M, int N, bool a_mn, bool b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((a_mn ? 1u : 0u) << 15) | ((b_mn ? 1u : 0u) << 16) |
         ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

// ------------------------------------------------------------------ MMA issue (single thread)
template <int CG>
DEVI void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  if constexpr (CG == 1)
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
        ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
  else
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}"
        ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
template <int CG>
DEVI void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  if constexpr (CG == 1)
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}"
        ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum));
  else
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        " tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}"
        ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum));
}
template <int CG>
DEVI void mma_commit(uint32_t bar) {
  if constexpr (CG == 1)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
  else
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        ::"r"(bar), "h"((uint16_t)3) : "memory");
}

DEVI void mma_commit_mask(uint32_t bar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar), "h"(mask) : "memory");
}
// ------------------------------------------------------------------ tcgen05.ld / st  (32x32b shape)
#define R8(i) "=r"(r[i]), "=r"(r[i+1]), "=r"(r[i+2]), "=r"(r[i+3]), "=r"(r[i+4]), "=r"(r[i+5]), "=r"(r[i+6]), "=r"(r[i+7])
#define W8(i) "r"(r[i]), "r"(r[i+1]), "r"(r[i+2]), "r"(r[i+3]), "r"(r[i+4]), "r"(r[i+5]), "r"(r[i+6]), "r"(r[i+7])
DEVI void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : R8(0), R8(8), R8(16), R8(24)
      : "r"(taddr) : "memory");
}
DEVI void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : R8(0), R8(8)
      : "r"(taddr) : "memory");
}
DEVI void tmem_ld8(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : R8(0) : "r"(taddr) : "memory");
}
DEVI void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};" :: W8(0), "r"(taddr) : "memory");
}
DEVI void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
      :: W8(0), W8(8), W8(16), W8(24), "r"(taddr) : "memory");
}
DEVI void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
      :: W8(0), W8(8), "r"(taddr) : "memory");
}
#undef R8
#undef W8

// 128B-swizzle byte offset inside a [rows][64 bf16] block: row r, 16B-chunk c (0..7)
DEVI uint32_t sw128(uint32_t r, uint32_t c) { return r * 128u + ((c ^ (r & 7u)) << 4); }

// ---- cluster-scope release/acquire variants (generic-proxy SMEM writes handed to a peer CTA)
DEVI void mbar_arrive_remote_release(uint32_t remote_bar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote_bar) : "memory");
}
DEVI uint32_t mbar_try_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
DEVI void mbar_wait_cluster(uint32_t bar, uint32_t parity) { while (!mbar_try_wait_cluster(bar, parity)) { } }
DEVI uint32_t mbar_test_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
