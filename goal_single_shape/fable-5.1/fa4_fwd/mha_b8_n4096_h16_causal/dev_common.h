#pragma once
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tmem_ops.h"

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// ---------------- mbarrier ----------------
__device__ __forceinline__ void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t addr) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t addr, uint32_t tx) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(addr), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, uint32_t parity) {
  asm volatile("{\n"
    ".reg .pred p;\n"
    "WAIT_LOOP:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "@p bra WAIT_DONE;\n"
    "bra WAIT_LOOP;\n"
    "WAIT_DONE:\n"
    "}" :: "r"(addr), "r"(parity) : "memory");
}

// ---------------- TMA ----------------
__device__ __forceinline__ void tma_load_3d(uint32_t smem_dst, const CUtensorMap* map, uint32_t mbar,
                                            int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
    :: "r"(smem_dst), "l"((uint64_t)map), "r"(mbar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* map, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global [%0, {%1, %2, %3}];"
    :: "l"((uint64_t)map), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_prefetch_desc(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)map) : "memory");
}

// ---------------- tcgen05 ----------------
__device__ __forceinline__ void tmem_alloc(uint32_t smem_slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_slot), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after()  { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}

// Shared-memory matrix descriptor (SWIZZLE_128B). lbo/sbo in bytes.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFFu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFFu) << 32;
  d |= (uint64_t)1 << 46;   // descriptor version (sm100)
  d |= (uint64_t)2 << 61;   // SWIZZLE_128B
  return d;
}
// Instruction descriptor: kind::f16, bf16 inputs, f32 accum.
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, bool a_mn_major, bool b_mn_major) {
  uint32_t d = 0;
  d |= 1u << 4;            // D = F32
  d |= 1u << 7;            // A = BF16
  d |= 1u << 10;           // B = BF16
  d |= (a_mn_major ? 1u : 0u) << 15;
  d |= (b_mn_major ? 1u : 0u) << 16;
  d |= (N >> 3) << 17;
  d |= (M >> 4) << 24;
  return d;
}
// D[tmem] (+)= A[smem] * B[smem]
__device__ __forceinline__ void mma_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accum) : "memory");
}
// D[tmem] (+)= A[tmem] * B[smem]
__device__ __forceinline__ void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}"
    :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

// ---------------- host: tensor map ----------------
static inline CUtensorMap make_tmap_bshd(const void* ptr, int B, int S, int H, int D) {
  // dims: (D, H, B*S), box (64, 1, 128), 128B swizzle
  CUtensorMap m;
  cuuint64_t gdim[3]    = {(cuuint64_t)D, (cuuint64_t)H, (cuuint64_t)B * S};
  cuuint64_t gstride[2] = {(cuuint64_t)D * 2, (cuuint64_t)H * D * 2};
  cuuint32_t box[3]     = {64, 1, 128};
  cuuint32_t estr[3]    = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)ptr, gdim, gstride, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)r); }
  return m;
}
