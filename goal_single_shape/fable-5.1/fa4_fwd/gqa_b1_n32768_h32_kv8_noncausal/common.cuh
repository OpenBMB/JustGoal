#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------- mbarrier ----------------
DEVI void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count));
}
DEVI void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}
DEVI bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
#ifdef NO_SUSPEND_HINT
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
#else
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
               "selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity), "r"(1000000u) : "memory");
#endif
  return ok != 0;
}
DEVI bool mbar_try_wait_ns(uint64_t* bar, uint32_t parity, uint32_t ns) {   // blocks at most ~ns
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
               "selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity), "r"(ns) : "memory");
  return ok != 0;
}
DEVI bool mbar_test_wait(uint64_t* bar, uint32_t parity) {   // non-blocking
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok != 0;
}
DEVI void mbar_wait(uint64_t* bar, uint32_t parity) {
#ifdef LIGHT_SLEEP
  while (!mbar_try_wait(bar, parity)) { __nanosleep(LIGHT_SLEEP); }
#else
  while (!mbar_try_wait(bar, parity)) {}
#endif
}

// ---------------- TMA ----------------
DEVI void tma_load_4d(void* dst, const CUtensorMap* tmap, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
               :: "r"(smem_u32(dst)), "l"((uint64_t)tmap), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tma_prefetch_desc(const CUtensorMap* tmap) {
  asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)tmap) : "memory");
}
DEVI void tma_store_4d(const CUtensorMap* tmap, const void* src, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
               :: "l"((uint64_t)tmap), "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void tma_store_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void tma_store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ---------------- TMEM ----------------
DEVI void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(dst_smem)), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ---------------- MMA ----------------
// smem descriptor: swizzle 0=none, 1=128B_base32B, 2=128B, 4=64B, 6=32B
DEVI uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)swizzle << 61;
  return d;
}
// instruction descriptor kind::f16: D=f32, A/B bf16
// formats: c_fmt 0=F16 1=F32 ; a_fmt/b_fmt 0=F16 1=BF16
DEVI uint32_t make_idesc_fmt(uint32_t M, uint32_t N, uint32_t a_mn_major, uint32_t b_mn_major, uint32_t c_fmt, uint32_t a_fmt, uint32_t b_fmt) {
  uint32_t d = 0;
  d |= (c_fmt << 4);
  d |= (a_fmt << 7);
  d |= (b_fmt << 10);
  d |= (a_mn_major << 15);
  d |= (b_mn_major << 16);
  d |= ((N >> 3) << 17);
  d |= ((M >> 4) << 24);
  return d;
}
DEVI uint32_t make_idesc(uint32_t M, uint32_t N, uint32_t a_mn_major, uint32_t b_mn_major) {
  uint32_t d = 0;
  d |= (1u << 4);         // c_format = F32
  d |= (1u << 7);         // a_format = BF16
  d |= (1u << 10);        // b_format = BF16
  d |= (a_mn_major << 15);
  d |= (b_mn_major << 16);
  d |= ((N >> 3) << 17);
  d |= ((M >> 4) << 24);
  return d;
}
DEVI void mma_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void mma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_u32(bar)) : "memory");
}

// ---------------- host: tensor map ----------------
static inline CUtensorMap make_tmap_4d_bf16(void* base, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3,
                                            uint64_t s1, uint64_t s2, uint64_t s3,  // strides in ELEMENTS
                                            uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, bool swz128) {
  CUtensorMap tm;
  cuuint64_t gdim[4] = {d0, d1, d2, d3};
  cuuint64_t gstr[3] = {s1 * 2, s2 * 2, s3 * 2};
  cuuint32_t box[4] = {b0, b1, b2, b3};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, base, gdim, gstr, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      swz128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed %d\n", (int)r); }
  return tm;
}
