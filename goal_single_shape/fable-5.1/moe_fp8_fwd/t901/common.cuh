// Common PTX wrappers for TMA / mbarrier / tcgen05 (Blackwell sm_100a).
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
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_expect_tx_noarrive(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
DEVI bool mbar_try_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
DEVI bool mbar_test_wait(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}
#ifdef WATCHDOG
__device__ int* g_watchdog = nullptr;   // set by the kernel: [0]=code of first timed-out wait
#define WD_LOOP(cond, code, role) { long long _wd = 0; while (cond) { if (++_wd > (1ll << 24)) { if (g_watchdog) atomicCAS(g_watchdog + (role), 0, (code)); break; } __nanosleep(32); } }
#else
#define WD_LOOP(cond, code, role) { while (cond) { __nanosleep(20); } }
#endif
#ifdef WATCHDOG
DEVI void mbar_wait(uint32_t bar, uint32_t parity, int role = 0) {
    WD_LOOP(!mbar_try_wait(bar, parity), 1000 + (int)((bar >> 3) & 1023), role);
}
#else
#if defined(MBAR_TESTWAIT)
DEVI void mbar_wait(uint32_t bar, uint32_t parity, int role = 0) {
    while (!mbar_test_wait(bar, parity)) { }
}
#elif defined(MBAR_HINT)
DEVI bool mbar_try_wait_hint(uint32_t bar, uint32_t parity, uint32_t hint_ns) {
    uint32_t ok;
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(hint_ns) : "memory");
    return ok != 0;
}
DEVI void mbar_wait(uint32_t bar, uint32_t parity, int role = 0) {
    while (!mbar_try_wait_hint(bar, parity, MBAR_HINT)) { }
}
#else
DEVI void mbar_wait(uint32_t bar, uint32_t parity, int role = 0) {
    while (!mbar_try_wait(bar, parity)) { }
}
#endif
#endif
DEVI void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------- TMA ----------------
DEVI void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int x, int y, uint64_t policy) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(x), "r"(y), "l"(policy) : "memory");
}
DEVI void tma_load_2d_plain(uint32_t dst, const CUtensorMap* map, uint32_t bar, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(x), "r"(y) : "memory");
}
DEVI void tma_load_4d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int x, int y, int z, int w, uint64_t policy) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(x), "r"(y), "r"(z), "r"(w), "l"(policy) : "memory");
}
DEVI void tma_load_4d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int x, int y, int z, int w) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
                 :: "r"(dst), "l"(map), "r"(bar), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
DEVI void tma_prefetch_4d(const CUtensorMap* map, int x, int y, int z, int w) {
    asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];"
                 :: "l"(map), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
DEVI void tma_load_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEVI uint64_t policy_evict_first() {
    uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
DEVI uint64_t policy_evict_normal() {
    uint64_t p; asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p)); return p;
}
DEVI uint64_t policy_evict_last() {
    uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
DEVI void tma_prefetch_desc(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

// ---------------- cp.async ----------------
DEVI void cp_async_16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
DEVI void cp_async_4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src) : "memory");
}
DEVI void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
DEVI void cp_async_mbar_arrive_noinc(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}

// ---------------- tcgen05 ----------------
DEVI void tmem_alloc(uint32_t smem_dst, uint32_t ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_dst), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// smem matrix descriptor: K-major, 128B swizzle, 8-row atoms of 1024B.
DEVI uint64_t make_smem_desc(uint32_t saddr) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr & 0x3FFFF) >> 4);          // start address [0,14)
    d |= (uint64_t)(16 >> 4) << 16;                    // leading byte offset [16,30) (unused for swizzled K-major)
    d |= (uint64_t)(1024 >> 4) << 32;                  // stride byte offset [32,46)
    d |= (uint64_t)1 << 46;                            // version = 1 (sm100)
    d |= (uint64_t)2 << 61;                            // layout: SWIZZLE_128B
    return d;
}
// instruction descriptor: kind::f8f6f4, D=f32, A=B=e4m3, K-major, M, N
DEVI constexpr uint32_t make_idesc(int M, int N) {
    return (1u << 4) | (0u << 7) | (0u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
DEVI void tc_mma_f8(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t enable_d) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                 " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}"
                 :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d) : "memory");
}
DEVI void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}
// load 32 columns (32 lanes x 32b) from TMEM into 32 registers
DEVI void tmem_ld32(uint32_t taddr, float* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),
                   "=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]),
                   "=f"(r[16]),"=f"(r[17]),"=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),
                   "=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),"=f"(r[30]),"=f"(r[31])
                 : "r"(taddr) : "memory");
}
DEVI void tmem_ld16(uint32_t taddr, float* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                 : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]),
                   "=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),"=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15])
                 : "r"(taddr) : "memory");
}
DEVI void tmem_ld8(uint32_t taddr, float* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),"=f"(r[6]),"=f"(r[7]) : "r"(taddr) : "memory");
}
DEVI void tmem_ld4(uint32_t taddr, float* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]) : "r"(taddr) : "memory");
}
template <int NC> DEVI void tmem_ldn(uint32_t taddr, float* r);
template <> DEVI void tmem_ldn<4>(uint32_t taddr, float* r) { tmem_ld4(taddr, r); }
template <> DEVI void tmem_ldn<8>(uint32_t taddr, float* r) { tmem_ld8(taddr, r); }
template <> DEVI void tmem_ldn<16>(uint32_t taddr, float* r) { tmem_ld16(taddr, r); }
template <> DEVI void tmem_ldn<32>(uint32_t taddr, float* r) { tmem_ld32(taddr, r); }
DEVI void red_add_f32(float* p, float v) {
    asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}
DEVI uint8_t f32_to_e4m3(float v) {
    uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.f), "f"(v)); return (uint8_t)(r & 0xFF);
}
DEVI void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// ---------------- explicit shared-memory loads (avoid generic LD.E when the address space is not inferable) ----------------
DEVI float lds_f32(uint32_t addr) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr)); return v; }
DEVI float4 lds_f32x4(uint32_t addr) {
    float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr)); return v;
}
DEVI int lds_s32(uint32_t addr) { int v; asm volatile("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(addr)); return v; }
// ---------------- misc ----------------
DEVI void red_add_v4(float* p, float a, float b, float c, float d) {
    asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
DEVI uint32_t ld_acquire_gpu(const int* p) {
    uint32_t v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI uint16_t f32_to_bf16_bits(float f) {
    uint16_t r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r;
}
DEVI uint32_t f32x2_to_bf16x2(float lo, float hi) {
    uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI uint16_t f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
