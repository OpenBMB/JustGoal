// Shared PTX wrappers for tcgen05 / TMA / mbarrier (sm_100a).
#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- mbarrier ----
DEVI void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count));
}
DEVI void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void mbar_arrive(uint32_t mbar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}" :: "r"(mbar) : "memory");
}
DEVI void mbar_arrive_expect_tx(uint32_t mbar, uint32_t tx) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(mbar), "r"(tx) : "memory");
}
DEVI bool mbar_try_wait(uint32_t mbar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(mbar), "r"(parity) : "memory");
  return ok != 0;
}
DEVI bool mbar_test_wait(uint32_t mbar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(mbar), "r"(parity) : "memory");
  return ok != 0;
}
// Fast path: a non-blocking test (16 cycles) before the potentially-suspending try_wait (~128 cycles even when complete).
DEVI void mbar_wait(uint32_t mbar, uint32_t parity) {
  if (mbar_test_wait(mbar, parity)) return;
#ifdef DBG_SPIN_WAIT
  while (!mbar_test_wait(mbar, parity)) {}
#else
  while (!mbar_try_wait(mbar, parity)) {}
#endif
}

// Remote arrive without a release fence: used when the signalled condition is established by an mbarrier the arriving thread
// has already observed (no memory operations of this thread need ordering).
DEVI void mbar_arrive_remote_relaxed(uint32_t raddr) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" :: "r"(raddr) : "memory");
}
DEVI bool mbar_try_wait_hint(uint32_t mbar, uint32_t parity, uint32_t hint_ns) {
  uint32_t ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(mbar), "r"(parity), "r"(hint_ns) : "memory");
  return ok != 0;
}
template <uint32_t HINT> DEVI void mbar_wait_hint(uint32_t mbar, uint32_t parity) {
  if (mbar_test_wait(mbar, parity)) return;
  while (!mbar_try_wait_hint(mbar, parity, HINT)) {}
}
DEVI void mbar_wait_spin(uint32_t mbar, uint32_t parity) { while (!mbar_test_wait(mbar, parity)) {} }
// One lane of the (converged) warp is elected; the compiler treats code guarded by it as single-lane, which lets tcgen05
// operands go to uniform registers without per-instruction election loops.
DEVI bool elect_one_sync() {
  uint32_t pred = 0;
  asm volatile("{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
  return pred != 0;
}
// ---- TMA ----
DEVI void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
DEVI void tma_load_4d(uint32_t dst, const CUtensorMap* map, int c0, int c1, int c2, int c3, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
               :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar) : "memory");
}
// Prefetch one box of a tensor map into L2 (no SMEM destination, no completion tracking).
DEVI void tma_prefetch_4d(const CUtensorMap* map, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
DEVI void tma_prefetch_desc(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

// ---- tcgen05 ----
DEVI void tmem_alloc(uint32_t smem_dst, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_dst), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// SMEM matrix descriptor. swz: 0 none, 2 = 128B swizzle.
DEVI uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)swz << 61;
  return d;
}
// Instruction descriptor for kind::f16. a_fmt/b_fmt: 0 = F16, 1 = BF16. d_f32: 1 = F32 accum, 0 = F16.
DEVI uint32_t make_idesc(int M, int N, int a_mn, int b_mn, int a_fmt, int b_fmt, int d_f32) {
  uint32_t d = 0;
  d |= (uint32_t)d_f32 << 4;
  d |= (uint32_t)a_fmt << 7;
  d |= (uint32_t)b_fmt << 10;
  d |= (uint32_t)a_mn << 15;
  d |= (uint32_t)b_mn << 16;
  d |= (uint32_t)(N >> 3) << 17;
  d |= (uint32_t)(M >> 4) << 24;
  return d;
}
DEVI void mma_ss(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
DEVI void mma_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
// Arrives on the mbarrier at the same CTA-relative offset in every CTA of the cluster mask.
DEVI void mma_commit_mc(uint32_t mbar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mbar), "h"(mask) : "memory");
}
// 32 lanes x 32 columns of 32-bit, one warp. taddr encodes lane base (bits 31:16) and column (bits 15:0).
DEVI void tmem_ld32(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// Load + wait in ONE asm statement so the compiler cannot use the destination registers before the wait completes.
DEVI void tmem_ld32_sync(uint32_t taddr, uint32_t (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               "tcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr) : "memory");
}
// Two x32 loads (64 columns) followed by one wait, all in one asm statement.
// Load 32 columns, wait, fence, then (lane 0) arrive on an mbarrier -- all inside one asm block so that no computation on the
// loaded values can be scheduled before the arrival (which would delay the release of the TMEM region).
DEVI void tmem_ld32_sync_arrive(uint32_t taddr, uint32_t (&r)[32], uint32_t mbar) {
  asm volatile("{\n.reg .pred p;\n"
               "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               "tcgen05.wait::ld.sync.aligned;\n"
               "tcgen05.fence::before_thread_sync;\n"
               "bar.warp.sync 0xffffffff;\n"
               "setp.eq.u32 p, %34, 0;\n"
               "@p mbarrier.arrive.shared::cta.b64 _, [%33];\n}"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr), "r"(mbar), "r"(threadIdx.x & 31) : "memory");
}
DEVI void tmem_ld64_sync(uint32_t taddr, uint32_t (&r)[64]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%64];\n"
               "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%65];\n"
               "tcgen05.wait::ld.sync.aligned;"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
                 "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
                 "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
                 "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
                 "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(taddr), "r"(taddr + 32) : "memory");
}

// ---- host: tensor map creation via driver entry point ----
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
// 2D bf16 tensor map: global [rows][cols] row-major (cols contiguous), box [box_rows][box_cols], 128B swizzle.
static CUtensorMap make_map_2d_bf16(void* ptr, uint64_t rows, uint64_t cols, uint64_t row_stride_bytes, uint32_t box_rows, uint32_t box_cols, bool swizzle) {
  CUtensorMap m;
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {row_stride_bytes};
  cuuint32_t box[2] = {box_cols, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gdim, gstride, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed %d\n", (int)r); exit(1); }
  return m;
}
