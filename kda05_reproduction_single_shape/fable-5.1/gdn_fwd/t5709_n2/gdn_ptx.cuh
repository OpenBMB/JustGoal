// Raw PTX helpers for Blackwell (sm_100a): mbarrier, TMA, tcgen05 (UMMA/TMEM).
// Self-contained; no CUTLASS dependency.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t r; asm volatile("mov.u32 %0, %%laneid;" : "=r"(r)); return r;
}
__device__ __forceinline__ uint32_t warp_id() {
  uint32_t r; asm volatile("mov.u32 %0, %%warpid;" : "=r"(r)); return r;
}
__device__ __forceinline__ bool elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "elect.sync _|p, 0xffffffff;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(pred));
  return pred != 0;
}

// ----------------------------------------------------------------------------
// mbarrier
// ----------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.shared::cta.b64 st, [%0];\n\t}" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n\t}" ::"r"(smem_u32(bar)), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  while (!mbar_try_wait(bar, phase)) {}
}
// One warp polls the mbarrier, the rest of the CTA sleeps on the hardware barrier
// (spinning warps steal issue slots from the single-thread MMA/TMA issue paths).
__device__ __forceinline__ void mbar_wait_cta(uint64_t* bar, uint32_t phase) {
  if (threadIdx.x < 32) { while (!mbar_try_wait(bar, phase)) {} }
  __syncthreads();
}

// ----------------------------------------------------------------------------
// fences
// ----------------------------------------------------------------------------
// Make generic-proxy smem writes visible to the async proxy (TMA, UMMA).
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ----------------------------------------------------------------------------
// TMA (2D tiled)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void tma_prefetch_desc(const void* desc) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(desc)) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const void* desc, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* smem_dst, const void* desc, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
      ::"r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void tma_store_2d(const void* desc, const void* smem_src, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      ::"l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void tma_store_3d(const void* desc, const void* smem_src, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      ::"l"(reinterpret_cast<uint64_t>(desc)), "r"(smem_u32(smem_src)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

// ----------------------------------------------------------------------------
// TMEM allocation (must be called by one full warp)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void tmem_alloc(uint32_t* smem_slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(smem_slot)), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

// ----------------------------------------------------------------------------
// UMMA descriptors
// ----------------------------------------------------------------------------
enum : uint8_t { SW_NONE = 0, SW_128B = 2, SW_64B = 4, SW_32B = 6 };

// smem matrix descriptor. lbo/sbo in bytes. Blackwell version bit set.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint8_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;  // version = 1 (Blackwell)
  d |= (uint64_t)layout << 61;
  return d;
}

// Instruction descriptor for kind::f16 with bf16 inputs and fp32 accumulate.
// a_mn / b_mn: 1 if the operand is MN-major in smem, 0 if K-major.
__device__ __forceinline__ uint32_t make_idesc_bf16(int M, int N, int a_mn, int b_mn, int a_neg = 0, int b_neg = 0) {
  uint32_t d = 0;
  d |= (1u << 4);              // c_format = F32
  d |= (1u << 7);              // a_format = BF16
  d |= (1u << 10);             // b_format = BF16
  d |= ((uint32_t)a_neg << 13);
  d |= ((uint32_t)b_neg << 14);
  d |= ((uint32_t)a_mn << 15);
  d |= ((uint32_t)b_mn << 16);
  d |= ((uint32_t)(N >> 3) << 17);
  d |= ((uint32_t)(M >> 4) << 24);
  return d;
}

// D[tmem] (+)= A[smem] * B[smem]; one thread issues.
__device__ __forceinline__ void umma_bf16_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
      ::"r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}
// D[tmem] (+)= A[tmem] * B[smem]
__device__ __forceinline__ void umma_bf16_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
      ::"r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}
// Arrive on mbarrier when all previously issued tcgen05 ops of this thread complete.
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

// ----------------------------------------------------------------------------
// TMEM <-> registers. 32x32b shape: warp (w%4) covers lanes 32*(w%4)..+31, each
// thread reads N consecutive 32-bit columns of its lane starting at `taddr`'s column.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

#define GDN_R8(i) "=r"(r[i]), "=r"(r[i+1]), "=r"(r[i+2]), "=r"(r[i+3]), "=r"(r[i+4]), "=r"(r[i+5]), "=r"(r[i+6]), "=r"(r[i+7])
#define GDN_I8(i) "r"(r[i]), "r"(r[i+1]), "r"(r[i+2]), "r"(r[i+3]), "r"(r[i+4]), "r"(r[i+5]), "r"(r[i+6]), "r"(r[i+7])

__device__ __forceinline__ void tmem_ld_x8(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : GDN_R8(0) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st_x8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};" :: GDN_I8(0), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_ld_x16(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : GDN_R8(0), GDN_R8(8) : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : GDN_R8(0), GDN_R8(8), GDN_R8(16), GDN_R8(24) : "r"(taddr));
}
__device__ __forceinline__ void tmem_st_x16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
      :: GDN_I8(0), GDN_I8(8), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_st_x32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
      :: GDN_I8(0), GDN_I8(8), GDN_I8(16), GDN_I8(24), "r"(taddr) : "memory");
}
#undef GDN_R8
#undef GDN_I8

// ----------------------------------------------------------------------------
// 128B-swizzle addressing for bf16 tiles with 128-byte rows (64 elements):
// byte offset of (row r, elem c) inside a tile whose base is 1024B aligned.
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t sw128_off(uint32_t row, uint32_t elem) {
  uint32_t byte_in_row = elem * 2;
  uint32_t chunk = (byte_in_row >> 4) ^ (row & 7);
  return row * 128 + (chunk << 4) + (byte_in_row & 15);
}

// ----------------------------------------------------------------------------
// warp-level tf32 MMA (m16n8k8, row.col, fp32 accumulate)
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t f32_to_tf32(float f) {
  uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f)); return r;
}
__device__ __forceinline__ void mma_tf32_16x8x8(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ----------------------------------------------------------------------------
// misc math
// ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}

// ----------------------------------------------------------------------------
// named barriers (subsets of the CTA) and inter-CTA dependency flags
// ----------------------------------------------------------------------------
__device__ __forceinline__ void named_sync(uint32_t id, uint32_t nthreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory");
}
// One warp polls the mbarrier, the other compute warps wait on the named barrier.
__device__ __forceinline__ void mbar_wait_cta_n(uint64_t* bar, uint32_t phase, uint32_t nthreads) {
  if (threadIdx.x < 32) { while (!mbar_try_wait(bar, phase)) {} }
  named_sync(1, nthreads);
}
// Dependency flags: 64-bit words = (epoch << 32) | payload (float bits), written with release
// semantics after all prior writes of the producing thread (generic and completed bulk-async).
__device__ __forceinline__ void flag_set(uint64_t* f, uint32_t epoch, float payload = 0.f) {
  const uint64_t v = ((uint64_t)epoch << 32) | (uint64_t)__float_as_uint(payload);
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("st.release.gpu.global.b64 [%0], %1;" ::"l"(f), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t flag_load(const uint64_t* f) {
  uint64_t x;
  asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(x) : "l"(f) : "memory");
  return x;
}
__device__ __forceinline__ void flag_fence() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
// Consumer: spin until the flag carries `epoch`, order the subsequent async-proxy (TMA) reads, return the payload.
__device__ __forceinline__ float flag_wait(const uint64_t* f, uint32_t epoch) {
  uint64_t x = flag_load(f);
  while ((uint32_t)(x >> 32) != epoch) { __nanosleep(64); x = flag_load(f); }
  flag_fence();
  return __uint_as_float((uint32_t)x);
}
// Two flags polled concurrently (one round trip when both are ready); returns the payload of the first.
__device__ __forceinline__ float flag_wait2(const uint64_t* f1, const uint64_t* f2, uint32_t epoch) {
  uint64_t x1 = flag_load(f1), x2 = flag_load(f2);
  while ((uint32_t)(x1 >> 32) != epoch || (uint32_t)(x2 >> 32) != epoch) {
    __nanosleep(64); x1 = flag_load(f1); x2 = flag_load(f2);
  }
  flag_fence();
  return __uint_as_float((uint32_t)x1);
}

__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // namespace gdn
