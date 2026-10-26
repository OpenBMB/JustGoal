// From-scratch PTX wrappers for sm_100a (Blackwell): mbarrier, TMA, tcgen05 (TMEM + UMMA).
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------- mbarrier ----------------
DEVI void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
DEVI void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
#ifndef MBAR_HINT
#define MBAR_HINT 0
#endif
DEVI uint32_t mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
#if MBAR_HINT > 0
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity), "r"((uint32_t)MBAR_HINT) : "memory");
#else
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
#endif
  return ok;
}
DEVI uint32_t mbar_test_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok;
}
DEVI void mbar_wait(uint64_t* bar, uint32_t parity) {
#ifndef NO_TEST_WAIT
  if (mbar_test_wait(bar, parity)) return;
#endif
  while (!mbar_try_wait(bar, parity)) {}
}
// wait with nanosleep backoff (for helper warps that should not steal issue slots)
DEVI void mbar_wait_sleep(uint64_t* bar, uint32_t parity) {
  if (mbar_test_wait(bar, parity)) return;
  while (!mbar_try_wait(bar, parity)) {
#ifdef AUX_SLEEP
    asm volatile("nanosleep.u32 %0;" :: "r"((uint32_t)AUX_SLEEP));
#endif
  }
}

// ---------------- TMA ----------------
DEVI void tma_load_2d(void* dst, const CUtensorMap* tm, uint64_t* bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(tm), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
DEVI void tma_load_3d(void* dst, const CUtensorMap* tm, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(tm), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
DEVI void tma_load_4d(void* dst, const CUtensorMap* tm, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(tm), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
}
DEVI void tma_prefetch_desc(const CUtensorMap* tm) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tm) : "memory");
}

// ---------------- TMEM ----------------
DEVI void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols)
               : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// 32x32b.x16 : each thread gets 16 consecutive columns of its lane.
DEVI void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr) : "memory");
}
DEVI void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr) : "memory");
}
DEVI void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])
      : "memory");
}
DEVI void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
      : "memory");
}

// ---------------- UMMA (tcgen05.mma) ----------------
// SMEM matrix descriptor, 128B swizzle.
DEVI uint64_t make_desc_sw128(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;   // descriptor version (sm100)
  d |= (uint64_t)2 << 61;   // SWIZZLE_128B
  return d;
}
// Instruction descriptor: bf16 inputs, fp32 accumulate.
DEVI uint32_t make_idesc_bf16(int M, int N, bool a_mn_major, bool b_mn_major) {
  uint32_t d = 0;
  d |= 1u << 4;                             // D format: F32
  d |= 1u << 7;                             // A: BF16
  d |= 1u << 10;                            // B: BF16
  d |= (a_mn_major ? 1u : 0u) << 15;        // A major-ness (0 = K-major)
  d |= (b_mn_major ? 1u : 0u) << 16;        // B major-ness
  d |= ((uint32_t)(N >> 3) & 0x3F) << 17;
  d |= ((uint32_t)(M >> 4) & 0x1F) << 24;
  return d;
}
DEVI void umma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum)
      : "memory");
}
DEVI void umma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(d_tmem),
      "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(accum)
      : "memory");
}
DEVI void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}

// ---------------- misc ----------------
DEVI uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DEVI uint32_t elect_one() {
  uint32_t pred = 0;
  asm volatile(
      "{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(pred));
  return pred;
}

DEVI float fmax3(float a, float b, float c) {
#ifdef USE_FMAX3
  float d;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
#else
  return fmaxf(fmaxf(a, b), c);
#endif
}

// 8 back-to-back MMAs (K=128 in 8 steps of 16), A and B from SMEM. First MMA has accumulate=acc0 (0/1), rest 1.
#define UMMA8_SS_BODY(PRED0) \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %2, %10, %1, " PRED0 ";\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %3, %11, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %12, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %5, %13, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %14, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %7, %15, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %16, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], %9, %17, %1, pt;\n"
DEVI void umma8_ss_noacc(uint32_t d_tmem, uint32_t idesc, const uint64_t* a, const uint64_t* b) {
  asm volatile("{\n .reg .pred pt, pf;\n setp.ne.u32 pt, %0, 0xffffffff;\n setp.eq.u32 pf, %0, 0xffffffff;\n"
               UMMA8_SS_BODY("pf") "}\n"
               :: "r"(d_tmem), "r"(idesc), "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(a[4]), "l"(a[5]), "l"(a[6]), "l"(a[7]),
                  "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]) : "memory");
}
DEVI void umma8_ss_acc(uint32_t d_tmem, uint32_t idesc, const uint64_t* a, const uint64_t* b) {
  asm volatile("{\n .reg .pred pt;\n setp.ne.u32 pt, %0, 0xffffffff;\n"
               UMMA8_SS_BODY("pt") "}\n"
               :: "r"(d_tmem), "r"(idesc), "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(a[4]), "l"(a[5]), "l"(a[6]), "l"(a[7]),
                  "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]) : "memory");
}
// 8 back-to-back MMAs, A from TMEM (a_tmem + 8*k columns), B from SMEM descriptors.
#define UMMA8_TS_BODY(PRED0) \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], %3, %1, " PRED0 ";\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%11], %4, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%12], %5, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%13], %6, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%14], %7, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%15], %8, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%16], %9, %1, pt;\n" \
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%17], %10, %1, pt;\n"
DEVI void umma8_ts_noacc(uint32_t d_tmem, uint32_t idesc, uint32_t a_tmem, const uint64_t* b) {
  asm volatile("{\n .reg .pred pt, pf;\n setp.ne.u32 pt, %0, 0xffffffff;\n setp.eq.u32 pf, %0, 0xffffffff;\n"
               UMMA8_TS_BODY("pf") "}\n"
               :: "r"(d_tmem), "r"(idesc), "r"(a_tmem), "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]),
                  "r"(a_tmem + 8), "r"(a_tmem + 16), "r"(a_tmem + 24), "r"(a_tmem + 32), "r"(a_tmem + 40), "r"(a_tmem + 48), "r"(a_tmem + 56) : "memory");
}
DEVI void umma8_ts_acc(uint32_t d_tmem, uint32_t idesc, uint32_t a_tmem, const uint64_t* b) {
  asm volatile("{\n .reg .pred pt;\n setp.ne.u32 pt, %0, 0xffffffff;\n"
               UMMA8_TS_BODY("pt") "}\n"
               :: "r"(d_tmem), "r"(idesc), "r"(a_tmem), "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]), "l"(b[4]), "l"(b[5]), "l"(b[6]), "l"(b[7]),
                  "r"(a_tmem + 8), "r"(a_tmem + 16), "r"(a_tmem + 24), "r"(a_tmem + 32), "r"(a_tmem + 40), "r"(a_tmem + 48), "r"(a_tmem + 56) : "memory");
}

DEVI void tmem_ld64(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]) : "r"(taddr) : "memory");
}
DEVI void fma_f32x2(float& d0, float& d1, float a0, float a1, float b, float c) {
  asm("{\n .reg .b64 ra, rb, rc, rd;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %4};\n mov.b64 rc, {%5, %5};\n"
      " fma.rn.f32x2 rd, ra, rb, rc;\n mov.b64 {%0, %1}, rd;\n}" : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b), "f"(c));
}

DEVI void add_f32x2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n .reg .b64 ra, rb, rd;\n mov.b64 ra, {%2, %3};\n mov.b64 rb, {%4, %5};\n"
      " add.rn.f32x2 rd, ra, rb;\n mov.b64 {%0, %1}, rd;\n}"
      : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}

DEVI uint32_t cvt_f16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t fma_f16x2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
DEVI uint32_t ex2_f16x2(uint32_t x) { uint32_t y; asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x)); return y; }
DEVI uint32_t add_f16x2(uint32_t a, uint32_t b) { uint32_t r; asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t min_f16x2(uint32_t a, uint32_t b) { uint32_t r; asm("min.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI float f16lo(uint32_t v) { float f; asm("{.reg .b16 h, g; mov.b32 {h, g}, %1; cvt.f32.f16 %0, h;}" : "=f"(f) : "r"(v)); return f; }
DEVI float f16hi(uint32_t v) { float f; asm("{.reg .b16 h, g; mov.b32 {h, g}, %1; cvt.f32.f16 %0, g;}" : "=f"(f) : "r"(v)); return f; }
DEVI uint32_t make_idesc_f16(int M, int N, bool a_mn_major, bool b_mn_major) {
  uint32_t d = 0;
  d |= 1u << 4;                             // D format: F32
  // A, B format F16 = 0
  d |= (a_mn_major ? 1u : 0u) << 15;
  d |= (b_mn_major ? 1u : 0u) << 16;
  d |= ((uint32_t)(N >> 3) & 0x3F) << 17;
  d |= ((uint32_t)(M >> 4) & 0x1F) << 24;
  return d;
}

DEVI void tmem_st4(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
DEVI void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(taddr),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}

DEVI uint32_t make_idesc_f16_acc16(int M, int N, bool a_mn_major, bool b_mn_major) {
  uint32_t d = 0;   // D format F16 (0), A/B F16 (0)
  d |= (a_mn_major ? 1u : 0u) << 15;
  d |= (b_mn_major ? 1u : 0u) << 16;
  d |= ((uint32_t)(N >> 3) & 0x3F) << 17;
  d |= ((uint32_t)(M >> 4) & 0x1F) << 24;
  return d;
}

// ---- fp16-accumulator helpers: packed loads (two adjacent 16-bit columns -> one register) and ALU-port-clean math ----
DEVI void tmem_ld16_pack(uint32_t taddr, uint32_t* r) {   // 16 columns of 16-bit values (x8 with pack) -> 8 packed registers
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(taddr) : "memory");
}
DEVI void tmem_ld32_pack(uint32_t taddr, uint32_t* r) {   // 32 columns (x16 with pack) -> 16 packed registers
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr) : "memory");
}
DEVI void tmem_ld64_pack(uint32_t taddr, uint32_t* r) {   // 64 columns (x32 with pack) -> 32 packed registers
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr) : "memory");
}
// scalar exps of the two halves of z (negated), scalar f16 sums into sl/sh (low/high halves), packed result pk
DEVI void ex2_neg_scalar2(uint32_t z, uint32_t& pk, uint32_t& sl, uint32_t& sh) {
  asm("{ .reg .f16 zl, zh, pl, ph, al, ah;\n"
      "  mov.b32 {zl, zh}, %3; neg.f16 zl, zl; neg.f16 zh, zh;\n"
      "  ex2.approx.f16 pl, zl; ex2.approx.f16 ph, zh;\n"
      "  mov.b32 {al, ah}, %4; add.rn.f16 al, al, pl; mov.b32 %1, {al, ah};\n"
      "  mov.b32 {al, ah}, %5; add.rn.f16 ah, ah, ph; mov.b32 %2, {al, ah};\n"
      "  mov.b32 %0, {pl, ph}; }"
      : "=r"(pk), "=r"(sl), "=r"(sh) : "r"(z), "r"(sl), "r"(sh));
}
DEVI uint32_t fma_relu_f16x2(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("fma.rn.relu.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
// 2^(-z): the negation folds into the MUFU operand (no separate ALU op)
DEVI uint32_t ex2_neg_f16x2_v(uint32_t z) { uint32_t y; asm volatile("{ .reg .b32 t; neg.f16x2 t, %1; ex2.approx.f16x2 %0, t; }" : "=r"(y) : "r"(z)); return y; }
DEVI uint32_t ex2_neg_f16x2(uint32_t z) { uint32_t y; asm("{ .reg .b32 t; neg.f16x2 t, %1; ex2.approx.f16x2 %0, t; }" : "=r"(y) : "r"(z)); return y; }

// ---- FMA/ALU-pipe exp for packed f16x2 (no MUFU): p = 2^(-z), z >= 0 finite. |rel err| < 8e-4, subnormals correct, 0 for z > 25 ----
DEVI uint32_t sub_f16x2(uint32_t a, uint32_t b) { uint32_t r; asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t mul_f16x2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t min_s16x2(uint32_t a, uint32_t b) { uint32_t r; asm("min.s16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t ex2_neg_poly_f16x2(uint32_t z) {
  uint32_t zc = min_f16x2(z, 0x4E804E80u);               // min(z, 26)
  uint32_t t = add_f16x2(zc, 0x64006400u);               // zc + 1024: integer-valued (round to nearest)
  uint32_t u = add_f16x2(t, 0xE400E400u);                // n = t - 1024 (exact)
  uint32_t fr = sub_f16x2(u, zc);                        // n - z in [-0.5, 0.5] (exact)
  uint32_t q = fma_f16x2(fr, 0x2AFD2AFDu, 0x33BE33BEu);  // 2^fr, degree 3
  q = fma_f16x2(q, fr, 0x398C398Cu);
  q = fma_f16x2(q, fr, 0x3C003C00u);
  uint32_t n = t - 0x64006400u;                          // packed integer n per half (0..26)
  uint32_t n1 = min_s16x2(n, 0x000D000Du);               // min(n, 13)
  uint32_t n2 = n - n1;                                  // 0..13
  uint32_t p1 = q - (n1 << 10);                          // q * 2^-n1 by exponent arithmetic (stays normal: q >= 0.7)
  uint32_t s2 = 0x3C003C00u - (n2 << 10);                // 2^-n2 as f16
  return mul_f16x2(p1, s2);                              // may be subnormal
}
DEVI uint32_t prmt_lo16(uint32_t a, uint32_t b) { uint32_t r; asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI uint32_t max_f16x2(uint32_t a, uint32_t b) { uint32_t r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
