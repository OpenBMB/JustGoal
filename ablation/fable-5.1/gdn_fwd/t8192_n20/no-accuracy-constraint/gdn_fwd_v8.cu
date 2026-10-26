// GDN (gated delta net) varlen prefill forward, written from scratch for B200 (sm_100a).  v6: persistent megakernel, tcgen05 GEMMs, pipelined chain.
// Layout: q/k [T,4,128] bf16, v [T,8,128] bf16, a/b [T,8] bf16, A_log/dt_bias [8] f32,
//         state [N,8,128(V),128(K)] f32 (k-last), cu_seqlens [N+1] i64.
// Output: out [T,8,128] bf16, new_state [N,8,128,128] f32.
//
// Algorithm (chunk size 64, per v-head h, k-head h/2):
//   logg_i = -exp(A_log)*softplus(a_i+dt_bias), beta_i = sigmoid(b_i), cl_i = cumsum(logg) (within chunk)
//   A_ij = beta_i * exp(cl_i-cl_j) * (k_i.k_j)  (j<i),  T = (I+A)^-1
//   U = T diag(beta) V,  W = T diag(beta*lam) K          (lam_i = exp(cl_i))          [item PREP, parallel]
//   Delta = U - W S0,  S1 = lam_C S0 + K^T diag(lam_C/lam_j) Delta                     [item CHAIN, sequential per seq]
//   O_i = scale*(lam_i q_i S0 + sum_{j<=i} exp(cl_i-cl_j)(q_i.k_j) Delta_j)            [item OUT, parallel]
// All three item types run inside one persistent kernel; a global work queue orders items so that every
// dependency points to an earlier item (deadlock free), and per-item flags (epoch-tagged) publish results.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef GDN_DBG
#define GDN_DBGON (P.dbg)
#else
#define GDN_DBGON false
#endif
namespace {
constexpr int C = 64, D = 128, HQ = 4, HV = 8, NT = 256;
constexpr int NSPLIT = 8;  // max v-splits per (seq, head): stride of chain items / progress flags
constexpr int NSPLIT_LONG = 8, NSPLIT_SHORT = 4;  // long sequences (nc >= CHAIN_INPLACE_NC) use VW=16, short ones VW=32
#ifndef STAGES_DEF
#define STAGES_DEF 2
#endif
#ifndef MINB
#define MINB 2
#endif
constexpr int STAGES = STAGES_DEF;
constexpr int KI = 36;
constexpr uint32_t TMEM_COLS = 128;  // TMEM columns allocated per block (2 blocks/SM -> 256 of 512)
constexpr int CHAIN_INPLACE_NC = 16;  // sequences with fewer chunks get their chain items queued after all prep items  // chain items are queued after the first KI chunks' prep items of their sequence
typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// ---- mbarrier / TMA
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_inval(uint64_t* bar) {
  asm volatile("mbarrier.inval.shared.b64 [%0];\n" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar), done;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(addr), "r"(parity) : "memory");
  } while (!done);
}
#ifndef SUSPEND_NS
#define SUSPEND_NS 0x2000
#endif
// wait with a suspend-time hint: the warp is descheduled until the phase completes (or the hint expires) instead of spinning
__device__ __forceinline__ void mbar_wait_sleep(uint64_t* bar, uint32_t parity) {
#ifndef USE_SLEEP
  mbar_wait(bar, parity); return;
#endif
  uint32_t addr = smem_u32(bar), done;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(addr), "r"(parity), "n"(SUSPEND_NS) : "memory");
  } while (!done);
}
__device__ __forceinline__ void mbar_wait_test(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar), done;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(addr), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* tm, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(smem_u32(smem)), "l"(tm), "r"(c0), "r"(c1), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* tm, int c0, int c1, const void* smem) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
               ::"l"(tm), "r"(c0), "r"(c1), "r"(smem_u32(smem)) : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* smem, const CUtensorMap* tm, int row, int cb, uint64_t* bar) {  // box {inner, rows, panels}: coords {0, row, panel}
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
               ::"r"(smem_u32(smem)), "l"(tm), "r"(0), "r"(row), "r"(cb), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* tm, int row, int cb, const void* smem) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];\n"
               ::"l"(tm), "r"(0), "r"(row), "r"(cb), "r"(smem_u32(smem)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(smem)), "l"(gmem), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
// ---- flags
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long ld_acquire64(const unsigned long long* p) {
  unsigned long long v; asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ int ld_relaxed(const int* p) {
  int v; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v; asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void st_relaxed64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory"); }
#ifndef PUB_EVERY
#define PUB_EVERY 4
#endif
// relaxed polling (no per-iteration L1 invalidate) followed by one acquire fence
__device__ __forceinline__ void wait_flag(const int* p, int target) {
  while (ld_relaxed(p) < target) __nanosleep(40);
}
__device__ __forceinline__ void wait_flag64_nofence(const unsigned long long* p, unsigned long long target) {
  while (ld_relaxed64(p) < target) __nanosleep(40);
}
__device__ __forceinline__ void wait_flag64_acq(const unsigned long long* p, unsigned long long target) {
  while (ld_acquire64(p) < target) __nanosleep(40);
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}
__device__ __forceinline__ float fexp(float x) { return __expf(x); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// ---- tcgen05 / UMMA (5th-gen tensor core: operands from smem via descriptors, accumulator in TMEM)
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {  // one warp
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(dst_smem)), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {  // one warp
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(ncols) : "memory");
}
// smem matrix descriptor: start | LBO | SBO | fixed bit 46 | swizzle (0 none, 2 SW128, 4 SW64, 6 SW32)
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= (uint64_t)((lbo & 0x3FFFFu) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFFu) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)swz << 61;
  return d;
}
// instruction descriptor kind::f16: D f32, A/B bf16, a_mn/b_mn: operand is MN-major
__device__ __forceinline__ uint32_t make_idesc(int M, int N, int a_mn, int b_mn) {
  uint32_t d = 0;
  d |= 1u << 4; d |= 1u << 7; d |= 1u << 10;
  d |= (uint32_t)a_mn << 15; d |= (uint32_t)b_mn << 16;
  d |= (uint32_t)(N >> 3) << 17; d |= (uint32_t)(M >> 4) << 24;
  return d;
}
__device__ __forceinline__ void umma(uint32_t dtmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, int accum) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               ::"r"(dtmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n"
               ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                 "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]) : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }

// ---- swizzled smem tile addressing (byte offsets)
template <int ROWS>
__device__ __forceinline__ uint32_t sw128(int r, int c) {
  return (uint32_t)((c >> 6) * (ROWS * 128) + r * 128 + ((((c >> 3) & 7) ^ (r & 7)) << 4) + ((c & 7) << 1));
}
__device__ __forceinline__ uint32_t sw64(int r, int c) {
  return (uint32_t)(r * 64 + ((((c >> 3) & 3) ^ ((r >> 1) & 3)) << 4) + ((c & 7) << 1));
}
__device__ __forceinline__ uint32_t sw32(int r, int c) {  // 32-byte rows, TMA SWIZZLE_32B
  return (uint32_t)(r * 32 + ((((c >> 3) & 1) ^ ((r >> 2) & 1)) << 4) + ((c & 7) << 1));
}
template <int VW_> __device__ __forceinline__ uint32_t swv(int r, int c) { if constexpr (VW_ == 32) return sw64(r, c); else return sw32(r, c); }
template <int ROWS>
__device__ __forceinline__ uint32_t sw128f(int r, int c) {
  return (uint32_t)((c >> 5) * (ROWS * 128) + r * 128 + ((((c >> 2) & 7) ^ (r & 7)) << 4) + ((c & 3) << 2));
}
__device__ __forceinline__ void frag_a(int lane, int m0, int k0, int& r, int& c) {
  r = m0 + (lane & 7) + 8 * ((lane >> 3) & 1); c = k0 + 8 * (lane >> 4);
}
__device__ __forceinline__ void frag_at(int lane, int m0, int k0, int& r, int& c) {
  r = k0 + (lane & 7) + 8 * (lane >> 4); c = m0 + 8 * ((lane >> 3) & 1);
}
__device__ __forceinline__ void frag_bn(int lane, int n0, int k0, int& r, int& c) {
  r = n0 + (lane & 7) + 8 * (lane >> 4); c = k0 + 8 * ((lane >> 3) & 1);
}
__device__ __forceinline__ void frag_bt(int lane, int n0, int k0, int& r, int& c) {
  r = k0 + (lane & 7) + 8 * ((lane >> 3) & 1); c = n0 + 8 * (lane >> 4);
}

// ============================================================================================
// global table / flags layout
// ============================================================================================
struct Params {
  CUtensorMap tmQ, tmK, tmV, tmW, tmU, tmU32, tmD, tmD32, tmS0, tmS032, tmOut, tmState, tmNewState;
  CUtensorMap tmU16, tmD16, tmS016, tmState16, tmNewState16;  // VW=16 variants (box width 16)
  CUtensorMap tmW2, tmK2;  // 2D panel maps for the chain
  CUtensorMap tmG;               // G = W_c K_{c-1}^T tiles [ncm*HV*64][64] bf16
  const bf16* a; const bf16* b; const float* A_log; const float* dt_bias; const int64_t* cu;
  float* clbuf; bf16* out;
  bf16* Dg; bf16* S0g; float* nsg;  // raw pointers for direct stores from the chain
  int* tbl;                      // table ints
  int* flags;                    // [0] table_ready, [1..1024] queue counters, [1025..] prep_done[ncm*HV]
  unsigned long long* prog;      // chain progress [N*HV*NSPLIT]
  int T, N, ncm, epoch; float scale;
  unsigned long long* dbg; int dbg_seq;
};
// table offsets (ints)
struct Tbl {
  int* hdr;      // [0]=nchunk [1]=nitems [2]=maxnc
  int* cseq; int* ct0; int* cL;          // per chunk
  int* sc0; int* snc; int* stb; int* slen; int* order; int* rank;  // per seq
  int* items;
  __device__ __host__ Tbl(int* base, int ncm, int N) {
    hdr = base; cseq = base + 4; ct0 = cseq + ncm; cL = ct0 + ncm;
    sc0 = cL + ncm; snc = sc0 + N; stb = snc + N; slen = stb + N; order = slen + N; rank = order + N;
    items = rank + N;
  }
};
__host__ __device__ inline int tbl_ints(int ncm, int N) { return 4 + 3 * ncm + 6 * N + 2 * ncm * HV + N * HV * NSPLIT; }
// build_table scratch: 7N+1+260+8+N ints

// ============================================================================================
// shared memory layouts
// ============================================================================================
struct __align__(1024) PrepSmem {
  bf16 Ks[C * D];           // 16384  sw128<64>
  bf16 Vs[C * D];           // 16384  sw128<64>
  bf16 Kp[C * D];           // 16384  sw128<64>  K of the previous chunk (for G)
  bf16 Abf[C * C];          // 8192 (later: G staging)
  bf16 Tb[C * C];           // 8192
  bf16 Tbl_[C * C];         // 8192
  bf16 Td[4 * 16 * 24];     // 3072
  bf16 tmpb[16 * 56];       // 1792
  float Ad[4 * 16 * 17];    // 4352
  float cl[C], bet[C], lam[C];
  uint64_t bar;
};
template <int VW_>
struct __align__(1024) ChainSmemT {
  static constexpr int WST = (VW_ == 16) ? 3 : 2;  // W stages (3 lets W[c+2] be prefetched at step start)
  bf16 Ws[WST][C * D];      // sw128<64>
  bf16 Ks[2][C * D];        // sw128<64>
  bf16 Us[WST][C * VW_];    // swv
  bf16 Sb[2][D * VW_];      // swv
  bf16 Db[C * VW_];         // swv (scaled delta)
  bf16 Du[2][C * VW_];      // swv (unscaled delta, TMA stored)
  float cls[WST][C];
  uint64_t barW[WST];
  uint64_t barK[2];
  uint64_t bar_state;
  uint64_t bar_mma;
};
#ifndef GDN_WSLOTS
#define GDN_WSLOTS 2
#endif
struct __align__(1024) ChainSmemP {  // v8 warp-specialized mma.sync chain, VW = 16
  bf16 Ws[GDN_WSLOTS][C * D];      // 32768/49152 sw128<64>
  bf16 Ks[2][C * D];      // 32768 sw128<64>
  bf16 Gs[3][C * C];      // 24576 sw128<64> ; Gs[2] doubles as the fp32 state staging in the prologue
  bf16 Us[2][C * 16];     // 4096  sw32
  bf16 Sbw[2][D * 16];    // 8192  sw32: bf16(lamC_{c-1} S_{c-1}) (GEMMw operand), ring by chunk parity
  bf16 Db[2][C * 16];     // 4096  sw32: scaled delta (GEMM operand), ring by chunk parity
  float cls[2][C];        // 512
  float clr[4][4];        // 64: cl[60..63] of chunk (ring of 4) for the S-warps
#if GDN_WSLOTS == 2
  float XX[4 * 32 * 8];   // 4096: Xx_{c+1} = G_{c+1} Db_c in mma accumulator lane layout [warp q][lane][8], S-warps -> E-warps
#endif
  uint64_t barW[3], barK[2], barG[3], barC[4], barSbw[2], barD[2], barK2done[2], bar_state, barX, barU[2], barWfree[2];
  int wiss;
};
struct __align__(1024) OutSmem {
  bf16 Qs[C * D];     // sw128<64>
  bf16 Ks[C * D];     // sw128<64>
  bf16 S0s[D * D];    // sw128<128>
  bf16 Ds[C * D];     // sw128<64>
  bf16 Ps[C * C];     // sw128<64>
  float cl[C];
  uint64_t bar, bar2;
};
constexpr size_t cmax3(size_t a, size_t b, size_t c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }
struct PrepSmem2; constexpr size_t PREP2_BYTES = 6 * 16384 + 8704 + 3584 + 1536 + 8;  // == sizeof(PrepSmem2) (checked below)
constexpr size_t UNION_BYTES = (cmax3(cmax3(sizeof(PrepSmem), sizeof(ChainSmemP), sizeof(ChainSmemT<32>)), sizeof(OutSmem), PREP2_BYTES) + 1023) & ~size_t(1023);
constexpr int SMEM_BYTES = (int)(UNION_BYTES + 1024 + 128);
// ctl words live right after the smem union: [0] item, [1] tmem base, [2] item code, [3] pending prep flag (chunk*HV+h) or -1
__device__ __forceinline__ int* ctl_of(void* sm) { return reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(sm) + UNION_BYTES); }
// tid 0 only: publish the previous prep item's flag once its TMA stores have landed (called after the current item's loads were issued)
__device__ __forceinline__ void flush_pending(const Params& P, int* ctl) {
  const int pf = ctl[3];
  if (pf >= 0) { bulk_wait0(); fence_proxy_async_global(); const int f = pf & 0x3fffffff; st_release(P.flags + 1025 + f, P.epoch); if (!(pf & (1 << 30))) st_release(P.flags + 1025 + f + 1, P.epoch); ctl[3] = -1; }  // 2-head prep items unless bit 30
}
// one thread of a non-chain item, after the item's first CTA barrier (the loop-top reads of ctl are then complete): pop the next item and fetch its code
// so that both round trips overlap the item's loads instead of the inter-item gap
// before the first read of the global table (fast path: only block 0 wrote it): wait once for its ready flag (acquire), then remember in ctl[6]
__device__ __forceinline__ void tbl_ready(const Params& P, int* ctl) {
  if (ctl[6] == 0) { wait_flag(P.flags, P.epoch); fence_acq_rel_gpu(); ctl[6] = 1; }
}
__device__ __forceinline__ void pop_next(const Params& P, int* ctl) {
  int* counter = P.flags + 1 + (P.epoch & 1023);
  const int nx = atomicAdd(counter, 1) + (int)gridDim.x;
  const Tbl tb(P.tbl, P.ncm, P.N);
  tbl_ready(P, ctl);
  ctl[0] = nx; ctl[2] = (nx < ctl[5]) ? __ldcg(tb.items + nx) : 0; ctl[4] = 1;
}

// ============================================================================================
// item PREP: per (chunk, head) -> W, U (bf16 [64][128]) and cl (f32 [64]); publishes prep_done
// ============================================================================================
__device__ __forceinline__ void dev_prep(const Params& P, const Tbl& tb, PrepSmem& sm, int ci, int h) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t sKs = smem_u32(sm.Ks), sVs = smem_u32(sm.Vs), sAbf = smem_u32(sm.Abf), sTb = smem_u32(sm.Tb), sTbl = smem_u32(sm.Tbl_);
  const int t0 = __ldcg(tb.ct0 + ci), L = __ldcg(tb.cL + ci);
  const bool hasprev = ((__ldcg(tb.cseq + ci) >> 8) & 0xfff) > 0;  // not the first chunk of its sequence -> compute G
  auto pst = [&](int j) { if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(2500 + ci * HV + h) * 8 + j] = t; } };
  pst(0);
  if (tid == 0) { mbar_inval(&sm.bar); mbar_init(&sm.bar, 1); }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_tx(&sm.bar, (hasprev ? 6 : 4) * 8192);
    tma_load_3d(sm.Ks, &P.tmK, t0, ((h >> 1) * D) >> 6, &sm.bar);
    tma_load_3d(sm.Vs, &P.tmV, t0, (h * D) >> 6, &sm.bar);
    if (hasprev) {
      tma_load_3d(sm.Kp, &P.tmK, t0 - C, ((h >> 1) * D) >> 6, &sm.bar);
    }
    flush_pending(P, ctl_of(&sm)); pop_next(P, ctl_of(&sm));
  }
  if (w == 7) {
    float lg[2], bt[2];
    const float eA = __expf(P.A_log[h]);
    const float db = P.dt_bias[h];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      int i = lane + 32 * e;
      if (i < L) {
        float x = __bfloat162float(P.a[(size_t)(t0 + i) * HV + h]) + db;
        lg[e] = -eA * softplus_f(x);
        bt[e] = sigmoid_f(__bfloat162float(P.b[(size_t)(t0 + i) * HV + h]));
      } else { lg[e] = 0.f; bt[e] = 0.f; }
    }
    float v0 = lg[0], v1 = lg[1];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float u0 = __shfl_up_sync(0xffffffffu, v0, o);
      float u1 = __shfl_up_sync(0xffffffffu, v1, o);
      if (lane >= o) { v0 += u0; v1 += u1; }
    }
    float tot0 = __shfl_sync(0xffffffffu, v0, 31);
    v1 += tot0;
    sm.cl[lane] = v0; sm.cl[lane + 32] = v1;
    sm.bet[lane] = bt[0]; sm.bet[lane + 32] = bt[1];
    sm.lam[lane] = fexp(v0); sm.lam[lane + 32] = fexp(v1);
  }
  for (int idx = tid; idx < 6 * 128; idx += NT) {
    int blkid = idx >> 7, r = (idx >> 3) & 15, c2 = (idx & 7) * 2;
    int I = blkid < 3 ? 0 : (blkid < 5 ? 1 : 2);
    int J = blkid < 3 ? blkid + 1 : (blkid < 5 ? blkid - 1 : 3);
    uint32_t off = sw128<C>(16 * I + r, 16 * J + c2);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tb) + off) = 0u;
    *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tbl_) + off) = 0u;
  }
  mbar_wait(&sm.bar, 0);
  __syncthreads();
  pst(1);

  {
    const int m0 = 16 * (w & 3), n0 = 32 * (w >> 2);
    uint32_t af[8][4], bfr[8][2][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      int r, c;
      frag_a(lane, m0, ks * 16, r, c);       ldsm_x4(af[ks], sKs + sw128<C>(r, c));
      frag_bn(lane, n0, ks * 16, r, c);      ldsm_x4(bfr[ks][0], sKs + sw128<C>(r, c));
      frag_bn(lane, n0 + 16, ks * 16, r, c); ldsm_x4(bfr[ks][1], sKs + sw128<C>(r, c));
    }
    float acc[4][4] = {};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
        mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
      }
    const int i0 = m0 + g, i1 = i0 + 8;
    const float cl0 = sm.cl[i0], cl1 = sm.cl[i1];
    const float b0 = sm.bet[i0], b1 = sm.bet[i1];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      int j = n0 + 8 * nt + 2 * t;
      float clj0 = sm.cl[j], clj1 = sm.cl[j + 1];
      float v00 = (j < i0) ? b0 * fexp(cl0 - clj0) * acc[nt][0] : 0.f;
      float v01 = (j + 1 < i0) ? b0 * fexp(cl0 - clj1) * acc[nt][1] : 0.f;
      float v10 = (j < i1) ? b1 * fexp(cl1 - clj0) * acc[nt][2] : 0.f;
      float v11 = (j + 1 < i1) ? b1 * fexp(cl1 - clj1) * acc[nt][3] : 0.f;
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Abf) + sw128<C>(i0, j)) = pack2(v00, v01);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Abf) + sw128<C>(i1, j)) = pack2(v10, v11);
      if ((j >> 4) == (i0 >> 4)) {
        float* Ad = sm.Ad + (i0 >> 4) * 16 * 17;
        Ad[(i0 & 15) * 17 + (j & 15)] = v00; Ad[(i0 & 15) * 17 + (j & 15) + 1] = v01;
        Ad[(i1 & 15) * 17 + (j & 15)] = v10; Ad[(i1 & 15) * 17 + (j & 15) + 1] = v11;
      }
    }
  }
  __syncthreads();
  pst(2);

  {
    const int col = tid >> 2, p = tid & 3, blk = col >> 4, cc = col & 15;
    const float* Ad = sm.Ad + blk * 16 * 17;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float part = 0.f;
#pragma unroll
      for (int j = 0; j < i; ++j)
        if ((j & 3) == p) part += Ad[i * 17 + j] * x[j];
      part += __shfl_xor_sync(0xffffffffu, part, 1);
      part += __shfl_xor_sync(0xffffffffu, part, 2);
      x[i] = ((i == cc) ? 1.f : 0.f) - part;
    }
    if (p == 0) {
      const float bt = sm.bet[col], lm = sm.lam[col];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        sm.Td[blk * 16 * 24 + i * 24 + cc] = __float2bfloat16(x[i]);
        uint32_t off = sw128<C>(16 * blk + i, col);
        *reinterpret_cast<bf16*>(reinterpret_cast<unsigned char*>(sm.Tb) + off) = __float2bfloat16(x[i] * bt);
        *reinterpret_cast<bf16*>(reinterpret_cast<unsigned char*>(sm.Tbl_) + off) = __float2bfloat16(x[i] * bt * lm);
      }
    }
  }
  __syncthreads();
  pst(3);
  const uint32_t sTd = smem_u32(sm.Td), sTmp = smem_u32(sm.tmpb);
  for (int I = 1; I < 4; ++I) {
    if (w < 2 * I) {
      const int n0 = 8 * w;
      float acc[4] = {};
      for (int K = 0; K < I; ++K) {
        uint32_t af[4], bfr[2];
        int r, c;
        frag_a(lane, 16 * I, 16 * K, r, c); ldsm_x4(af, sAbf + sw128<C>(r, c));
        int rb = 16 * K + (lane & 7) + 8 * ((lane >> 3) & 1);
        ldsm_x2_t(bfr, sTb + sw128<C>(rb, n0));
        mma16816(acc, af, bfr[0], bfr[1]);
      }
      *reinterpret_cast<uint32_t*>(sm.tmpb + g * 56 + n0 + 2 * t) = pack2(acc[0], acc[1]);
      *reinterpret_cast<uint32_t*>(sm.tmpb + (g + 8) * 56 + n0 + 2 * t) = pack2(acc[2], acc[3]);
    }
    __syncthreads();
    if (w < 2 * I) {
      const int n0 = 8 * w;
      float acc[4] = {};
      uint32_t af[4], bfr[2];
      ldsm_x4(af, sTd + (uint32_t)(I * 16 * 24 * 2) + (uint32_t)(((lane & 7) + 8 * ((lane >> 3) & 1)) * 48 + 16 * (lane >> 4)));
      ldsm_x2_t(bfr, sTmp + (uint32_t)((((lane & 7) + 8 * ((lane >> 3) & 1)) * 56 + n0) * 2));
      mma16816(acc, af, bfr[0], bfr[1]);
      int j = n0 + 2 * t;
      float l0 = sm.lam[j], l1 = sm.lam[j + 1];
      uint32_t o0 = sw128<C>(16 * I + g, j), o1 = sw128<C>(16 * I + g + 8, j);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tb) + o0) = pack2(-acc[0], -acc[1]);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tb) + o1) = pack2(-acc[2], -acc[3]);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tbl_) + o0) = pack2(-acc[0] * l0, -acc[1] * l1);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Tbl_) + o1) = pack2(-acc[2] * l0, -acc[3] * l1);
    }
    __syncthreads();
  }

  pst(4);
  {
    const int m0 = 16 * (w & 3), n0 = 64 * (w >> 2);
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const uint32_t sTa = which == 0 ? sTb : sTbl;
      const uint32_t sB = which == 0 ? sVs : sKs;
      unsigned char* Bs = reinterpret_cast<unsigned char*>(which == 0 ? sm.Vs : sm.Ks);
      uint32_t af[4][4], bfr[4][4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        int r, c;
        frag_a(lane, m0, ks * 16, r, c); ldsm_x4(af[ks], sTa + sw128<C>(r, c));
#pragma unroll
        for (int np = 0; np < 4; ++np) { frag_bt(lane, n0 + 16 * np, ks * 16, r, c); ldsm_x4_t(bfr[ks][np], sB + sw128<C>(r, c)); }
      }
      float acc[8][4] = {};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
        }
      __syncthreads();
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        int col = n0 + 8 * nt + 2 * t;
        *reinterpret_cast<uint32_t*>(Bs + sw128<C>(m0 + g, col)) = pack2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<uint32_t*>(Bs + sw128<C>(m0 + g + 8, col)) = pack2(acc[nt][2], acc[nt][3]);
      }
    }
    fence_proxy_async();
    __syncthreads();
    if (tid == 0) {
      const int row0 = ((int)ci * HV + h) * C;
      tma_store_3d(&P.tmU, row0, (0) >> 6, sm.Vs);
      tma_store_3d(&P.tmW, row0, (0) >> 6, sm.Ks);
      bulk_commit();
    }
  }
  if (hasprev) {  // G = W K_{c-1}^T (64 x 64, K = 128): W (bf16) is in sm.Ks, K_{c-1} in sm.Kp -> bf16 K-major tile in Abf -> TMA store
    const int m0 = 16 * (w & 3), n0 = 32 * (w >> 2);
    const uint32_t sKp = smem_u32(sm.Kp);
    uint32_t af[8][4], bfr[8][2][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      int r, c;
      frag_a(lane, m0, ks * 16, r, c);       ldsm_x4(af[ks], sKs + sw128<C>(r, c));
      frag_bn(lane, n0, ks * 16, r, c);      ldsm_x4(bfr[ks][0], sKp + sw128<C>(r, c));
      frag_bn(lane, n0 + 16, ks * 16, r, c); ldsm_x4(bfr[ks][1], sKp + sw128<C>(r, c));
    }
    float acc[4][4] = {};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
        mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
      }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int j = n0 + 8 * nt + 2 * t;
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Abf) + sw128<C>(m0 + g, j)) = pack2(acc[nt][0], acc[nt][1]);
      *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.Abf) + sw128<C>(m0 + g + 8, j)) = pack2(acc[nt][2], acc[nt][3]);
    }
    fence_proxy_async();
    __syncthreads();
    if (tid == 0) { tma_store_2d(&P.tmG, 0, ((int)ci * HV + h) * C, sm.Abf); bulk_commit(); }
  }
  pst(5);
  if (tid < C) P.clbuf[((size_t)ci * HV + h) * C + tid] = sm.cl[tid];
  __syncthreads();
  if (tid == 0) { bulk_wait_read0(); ctl_of(&sm)[3] = (ci * HV + h) | (1 << 30); }  // smem reusable now; the flag is published by flush_pending in the next item (bit 30: single head)
  pst(6);
}

// ============================================================================================
// item PREP2: one item per (chunk, head pair): the two v-heads 2hp, 2hp+1 share the k-head hp -> K / K_{c-1} loads and the K K^T GEMM
// are done once; everything else (gates, T = (I+A)^-1, U = T diag(b) V, W = T diag(b lam) K, G = W K_{c-1}^T) per head.
// ============================================================================================
constexpr int HP = HV / 2;  // prep items per chunk
constexpr int PREP1_K = 2;  // chunks k < PREP1_K of the long sequences use 1-head prep items (item type 3)
struct __align__(1024) PrepSmem2 {
  bf16 Ks[C * D];            // 16384  K_c (sw128<64>); after the W GEMMs: G staging [2][64 x 64] (sw128<64>)
  bf16 Kp[C * D];            // 16384  K_{c-1}
  bf16 Vs[2][C * D];         // 32768  V per head -> U in place
  bf16 R[2][2 * C * C];      // 32768  per head: [Abf: A (later its diagonal blocks hold T_II) | Tb: T diag(b)] -> W in place (sw128<64>, 16 KB)
  float Ad[2][4 * 16 * 17];  // 8704   fp32 diagonal blocks of A
  bf16 tmpb[2][16 * 56];     // 3584   merge scratch
  float cl[2][C], bet[2][C], lam[2][C];  // 1536
  uint64_t bar;
};
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 r = __hmul2(x, y);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void dev_prep2(const Params& P, const Tbl& tb, PrepSmem2& sm, int ci, int hp) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int h0 = 2 * hp;
  const uint32_t sKs = smem_u32(sm.Ks), sKp = smem_u32(sm.Kp);
  const int t0 = __ldcg(tb.ct0 + ci), L = __ldcg(tb.cL + ci);
  const bool hasprev = ((__ldcg(tb.cseq + ci) >> 8) & 0xfff) > 0;  // not the first chunk of its sequence -> compute G
  auto pst = [&](int j) { if (GDN_DBGON && tid == 0) { unsigned long long tt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt)); P.dbg[(2500 + ci * HV + h0) * 8 + j] = tt; } };
  pst(0);
  if (tid == 0) { mbar_inval(&sm.bar); mbar_init(&sm.bar, 1); }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_tx(&sm.bar, (hasprev ? 8 : 6) * 8192);
    tma_load_3d(sm.Ks, &P.tmK, t0, (hp * D) >> 6, &sm.bar);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      tma_load_3d(sm.Vs[hh], &P.tmV, t0, ((h0 + hh) * D) >> 6, &sm.bar);
    }
    if (hasprev) {
      tma_load_3d(sm.Kp, &P.tmK, t0 - C, (hp * D) >> 6, &sm.bar);
    }
    flush_pending(P, ctl_of(&sm)); pop_next(P, ctl_of(&sm));
  }
  if (w >= 6) {  // gates: warp 6 -> head h0, warp 7 -> head h0+1
    const int hh = w - 6, h = h0 + hh;
    float lg[2], bt[2];
    const float eA = __expf(P.A_log[h]);
    const float db = P.dt_bias[h];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      int i = lane + 32 * e;
      if (i < L) {
        float x = __bfloat162float(P.a[(size_t)(t0 + i) * HV + h]) + db;
        lg[e] = -eA * softplus_f(x);
        bt[e] = sigmoid_f(__bfloat162float(P.b[(size_t)(t0 + i) * HV + h]));
      } else { lg[e] = 0.f; bt[e] = 0.f; }
    }
    float v0 = lg[0], v1 = lg[1];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float u0 = __shfl_up_sync(0xffffffffu, v0, o);
      float u1 = __shfl_up_sync(0xffffffffu, v1, o);
      if (lane >= o) { v0 += u0; v1 += u1; }
    }
    float tot0 = __shfl_sync(0xffffffffu, v0, 31);
    v1 += tot0;
    sm.cl[hh][lane] = v0; sm.cl[hh][lane + 32] = v1;
    sm.bet[hh][lane] = bt[0]; sm.bet[hh][lane + 32] = bt[1];
    sm.lam[hh][lane] = fexp(v0); sm.lam[hh][lane + 32] = fexp(v1);
  }
  for (int idx = tid; idx < 2 * 6 * 128; idx += NT) {  // zero the strictly upper block triangle of Tb (both heads)
    const int hh = idx / 768, id2 = idx - hh * 768;
    int blkid = id2 >> 7, r = (id2 >> 3) & 15, c2 = (id2 & 7) * 2;
    int I = blkid < 3 ? 0 : (blkid < 5 ? 1 : 2);
    int J = blkid < 3 ? blkid + 1 : (blkid < 5 ? blkid - 1 : 3);
    *reinterpret_cast<uint32_t*>(reinterpret_cast<unsigned char*>(sm.R[hh] + C * C) + sw128<C>(16 * I + r, 16 * J + c2)) = 0u;
  }
  mbar_wait(&sm.bar, 0);
  __syncthreads();
  pst(1);

  {  // K K^T (64 x 64, K = 128) once; per-head gating epilogue -> Abf_h (bf16, masked j < i) and Ad_h (fp32 diagonal blocks)
    const int m0 = 16 * (w & 3), n0 = 32 * (w >> 2);
    uint32_t af[8][4], bfr[8][2][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      int r, c;
      frag_a(lane, m0, ks * 16, r, c);       ldsm_x4(af[ks], sKs + sw128<C>(r, c));
      frag_bn(lane, n0, ks * 16, r, c);      ldsm_x4(bfr[ks][0], sKs + sw128<C>(r, c));
      frag_bn(lane, n0 + 16, ks * 16, r, c); ldsm_x4(bfr[ks][1], sKs + sw128<C>(r, c));
    }
    float acc[4][4] = {};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
        mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
      }
    const int i0 = m0 + g, i1 = i0 + 8;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float cl0 = sm.cl[hh][i0], cl1 = sm.cl[hh][i1];
      const float b0 = sm.bet[hh][i0], b1 = sm.bet[hh][i1];
      unsigned char* Abf = reinterpret_cast<unsigned char*>(sm.R[hh]);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        int j = n0 + 8 * nt + 2 * t;
        float clj0 = sm.cl[hh][j], clj1 = sm.cl[hh][j + 1];
        float v00 = (j < i0) ? b0 * fexp(cl0 - clj0) * acc[nt][0] : 0.f;
        float v01 = (j + 1 < i0) ? b0 * fexp(cl0 - clj1) * acc[nt][1] : 0.f;
        float v10 = (j < i1) ? b1 * fexp(cl1 - clj0) * acc[nt][2] : 0.f;
        float v11 = (j + 1 < i1) ? b1 * fexp(cl1 - clj1) * acc[nt][3] : 0.f;
        *reinterpret_cast<uint32_t*>(Abf + sw128<C>(i0, j)) = pack2(v00, v01);
        *reinterpret_cast<uint32_t*>(Abf + sw128<C>(i1, j)) = pack2(v10, v11);
        if ((j >> 4) == (i0 >> 4)) {
          float* Ad = sm.Ad[hh] + (i0 >> 4) * 16 * 17;
          Ad[(i0 & 15) * 17 + (j & 15)] = v00; Ad[(i0 & 15) * 17 + (j & 15) + 1] = v01;
          Ad[(i1 & 15) * 17 + (j & 15)] = v10; Ad[(i1 & 15) * 17 + (j & 15) + 1] = v11;
        }
      }
    }
  }
  __syncthreads();
  pst(2);

  {  // 16 x 16 unit-lower-triangular inverses: thread pair (p = j parity) per (head, column); T_II -> Abf diagonal blocks (bf16), T_II diag(b) -> Tb
    const int hh = tid >> 7, rest = tid & 127, col = rest >> 1, p = rest & 1, blk = col >> 4, cc = col & 15;
    const float* Ad = sm.Ad[hh] + blk * 16 * 17;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float part = 0.f;
#pragma unroll
      for (int j = 0; j < i; ++j)
        if ((j & 1) == p) part += Ad[i * 17 + j] * x[j];
      part += __shfl_xor_sync(0xffffffffu, part, 1);
      x[i] = ((i == cc) ? 1.f : 0.f) - part;
    }
    if (p == 0) {
      const float bt = sm.bet[hh][col];
      unsigned char* Abf = reinterpret_cast<unsigned char*>(sm.R[hh]);
      unsigned char* Tb = Abf + C * C * 2;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        *reinterpret_cast<bf16*>(Abf + sw128<C>(16 * blk + i, 16 * blk + cc)) = __float2bfloat16(x[i]);
        *reinterpret_cast<bf16*>(Tb + sw128<C>(16 * blk + i, col)) = __float2bfloat16(x[i] * bt);
      }
    }
  }
  __syncthreads();
  pst(3);
  {  // block forward substitution: Tb[I rows, < 16I cols] = -T_II * sum_{K<I} A_IK Tb[K rows]; warps 0-3 head 0, 4-7 head 1
    const int hh = w >> 2, ww = w & 3;
    const uint32_t sAbf = smem_u32(sm.R[hh]), sTb = sAbf + C * C * 2, sTmp = smem_u32(sm.tmpb[hh]);
    for (int I = 1; I < 4; ++I) {
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int nt = ww + 4 * tt;
        if (nt < 2 * I) {
          const int n0 = 8 * nt;
          float acc[4] = {};
          for (int K = 0; K < I; ++K) {
            uint32_t af[4], bfr[2];
            int r, c;
            frag_a(lane, 16 * I, 16 * K, r, c); ldsm_x4(af, sAbf + sw128<C>(r, c));
            int rb = 16 * K + (lane & 7) + 8 * ((lane >> 3) & 1);
            ldsm_x2_t(bfr, sTb + sw128<C>(rb, n0));
            mma16816(acc, af, bfr[0], bfr[1]);
          }
          *reinterpret_cast<uint32_t*>(sm.tmpb[hh] + g * 56 + n0 + 2 * t) = pack2(acc[0], acc[1]);
          *reinterpret_cast<uint32_t*>(sm.tmpb[hh] + (g + 8) * 56 + n0 + 2 * t) = pack2(acc[2], acc[3]);
        }
      }
      __syncthreads();
#pragma unroll
      for (int tt = 0; tt < 2; ++tt) {
        const int nt = ww + 4 * tt;
        if (nt < 2 * I) {
          const int n0 = 8 * nt;
          float acc[4] = {};
          uint32_t af[4], bfr[2];
          int r, c;
          frag_a(lane, 16 * I, 16 * I, r, c); ldsm_x4(af, sAbf + sw128<C>(r, c));  // T_II (diagonal block of Abf)
          ldsm_x2_t(bfr, sTmp + (uint32_t)((((lane & 7) + 8 * ((lane >> 3) & 1)) * 56 + n0) * 2));
          mma16816(acc, af, bfr[0], bfr[1]);
          const int j = n0 + 2 * t;
          unsigned char* Tb = reinterpret_cast<unsigned char*>(sm.R[hh]) + C * C * 2;
          *reinterpret_cast<uint32_t*>(Tb + sw128<C>(16 * I + g, j)) = pack2(-acc[0], -acc[1]);
          *reinterpret_cast<uint32_t*>(Tb + sw128<C>(16 * I + g + 8, j)) = pack2(-acc[2], -acc[3]);
        }
      }
      __syncthreads();
    }
  }
  pst(4);
  {  // per head: U = Tb V (in place into Vs), W = (Tb diag(lam)) K (in place into R) ; TMA stores
    const int m0 = 16 * (w & 3), n0 = 64 * (w >> 2);
#pragma unroll 1
    for (int hh = 0; hh < 2; ++hh) {
      const uint32_t sTb = smem_u32(sm.R[hh]) + C * C * 2, sV = smem_u32(sm.Vs[hh]);
      uint32_t af[4][4], bfr[4][4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        int r, c;
        frag_a(lane, m0, ks * 16, r, c); ldsm_x4(af[ks], sTb + sw128<C>(r, c));
#pragma unroll
        for (int np = 0; np < 4; ++np) { frag_bt(lane, n0 + 16 * np, ks * 16, r, c); ldsm_x4_t(bfr[ks][np], sV + sw128<C>(r, c)); }
      }
      float acc[8][4] = {};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
        }
      __syncthreads();  // all warps done reading V_h and Tb_h fragments
      {
        unsigned char* Us = reinterpret_cast<unsigned char*>(sm.Vs[hh]);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          int col = n0 + 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t*>(Us + sw128<C>(m0 + g, col)) = pack2(acc[nt][0], acc[nt][1]);
          *reinterpret_cast<uint32_t*>(Us + sw128<C>(m0 + g + 8, col)) = pack2(acc[nt][2], acc[nt][3]);
        }
      }
      // W: scale the A fragments (k = token) by lam_k, B = K
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16 + 2 * t;
        const uint32_t lp0 = pack2(sm.lam[hh][k0], sm.lam[hh][k0 + 1]), lp1 = pack2(sm.lam[hh][k0 + 8], sm.lam[hh][k0 + 9]);
        af[ks][0] = bmul2(af[ks][0], lp0); af[ks][1] = bmul2(af[ks][1], lp0);
        af[ks][2] = bmul2(af[ks][2], lp1); af[ks][3] = bmul2(af[ks][3], lp1);
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int np = 0; np < 4; ++np) { int r, c; frag_bt(lane, n0 + 16 * np, ks * 16, r, c); ldsm_x4_t(bfr[ks][np], sKs + sw128<C>(r, c)); }
#pragma unroll
      for (int e = 0; e < 8; ++e) { acc[e][0] = 0.f; acc[e][1] = 0.f; acc[e][2] = 0.f; acc[e][3] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
        }
      {
        unsigned char* Ws = reinterpret_cast<unsigned char*>(sm.R[hh]);  // Tb_h reads of all warps precede the barrier above
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          int col = n0 + 8 * nt + 2 * t;
          *reinterpret_cast<uint32_t*>(Ws + sw128<C>(m0 + g, col)) = pack2(acc[nt][0], acc[nt][1]);
          *reinterpret_cast<uint32_t*>(Ws + sw128<C>(m0 + g + 8, col)) = pack2(acc[nt][2], acc[nt][3]);
        }
      }
      fence_proxy_async();
      __syncthreads();
      if (tid == 0) {
        const int row0 = ((int)ci * HV + h0 + hh) * C;
        tma_store_3d(&P.tmU, row0, (0) >> 6, sm.Vs[hh]);
        tma_store_3d(&P.tmW, row0, (0) >> 6, sm.R[hh]);
        bulk_commit();
      }
    }
  }
  if (hasprev) {  // G_h = W_h K_{c-1}^T (64 x 64, K = 128) -> staging in Ks (dead: both W GEMMs are past the barrier) -> TMA store
    const int m0 = 16 * (w & 3), n0 = 32 * (w >> 2);
    uint32_t bfr[8][2][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      int r, c;
      frag_bn(lane, n0, ks * 16, r, c);      ldsm_x4(bfr[ks][0], sKp + sw128<C>(r, c));
      frag_bn(lane, n0 + 16, ks * 16, r, c); ldsm_x4(bfr[ks][1], sKp + sw128<C>(r, c));
    }
#pragma unroll 1
    for (int hh = 0; hh < 2; ++hh) {
      const uint32_t sW = smem_u32(sm.R[hh]);
      uint32_t af[8][4];
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) { int r, c; frag_a(lane, m0, ks * 16, r, c); ldsm_x4(af[ks], sW + sw128<C>(r, c)); }
      float acc[4][4] = {};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
        }
      unsigned char* Gst = reinterpret_cast<unsigned char*>(sm.Ks) + hh * (C * C * 2);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int j = n0 + 8 * nt + 2 * t;
        *reinterpret_cast<uint32_t*>(Gst + sw128<C>(m0 + g, j)) = pack2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<uint32_t*>(Gst + sw128<C>(m0 + g + 8, j)) = pack2(acc[nt][2], acc[nt][3]);
      }
    }
    fence_proxy_async();
    __syncthreads();
    if (tid == 0) {
      tma_store_2d(&P.tmG, 0, ((int)ci * HV + h0) * C, sm.Ks);
      tma_store_2d(&P.tmG, 0, ((int)ci * HV + h0 + 1) * C, sm.Ks + C * C);
      bulk_commit();
    }
  }
  pst(5);
  if (tid < 2 * C) P.clbuf[((size_t)ci * HV + h0 + (tid >> 6)) * C + (tid & 63)] = sm.cl[tid >> 6][tid & 63];
  __syncthreads();
  if (tid == 0) { bulk_wait_read0(); ctl_of(&sm)[3] = ci * HV + h0; }  // smem reusable now; both heads' flags are published by flush_pending in the next item
  pst(6);
}

// ============================================================================================
// item CHAIN: sequential over the chunks of (seq, head, v-slice of VW_); publishes progress.
// GEMMs on tcgen05 (UMMA), state S resident in TMEM (fp32, 128 keys x VW_ values).
// ============================================================================================
template <int VW_>
__device__ __forceinline__ void dev_chain(const Params& P, const Tbl& tb, ChainSmemT<VW_>& sm, uint32_t tmem, int s, int h, int sl) {
  constexpr int WST = ChainSmemT<VW_>::WST;
  constexpr int HC = VW_ / 2;                       // values per thread (S slice: one key x HC values)
  constexpr int SWZ = (VW_ == 32) ? 4 : 6;          // UMMA swizzle code of the VW_-wide tiles (SW64 / SW32)
  constexpr int ROWB = VW_ * 2;                     // bytes per row of a VW_-wide bf16 tile
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, q = w & 3, hf = w >> 2;
  const int v0 = sl * VW_;
  const int c0 = __ldcg(tb.sc0 + s), nc = __ldcg(tb.snc + s), tbase = __ldcg(tb.stb + s);
  const int* pflag = P.flags + 1025;
  unsigned long long* prog = P.prog + ((size_t)(s * HV + h) * NSPLIT + sl) * 4;
  const size_t prog2off = (size_t)P.N * HV * NSPLIT * 4;
  const unsigned long long ep64 = (unsigned long long)P.epoch << 32;
  if (tid < WST) { mbar_inval(&sm.barW[tid]); mbar_init(&sm.barW[tid], 1); }
  if (tid < 2) { mbar_inval(&sm.barK[tid]); mbar_init(&sm.barK[tid], 1); }
  if (tid == 0) { mbar_inval(&sm.bar_state); mbar_init(&sm.bar_state, 1); mbar_inval(&sm.bar_mma); mbar_init(&sm.bar_mma, 1); }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncthreads();
  if (tid == 0) flush_pending(P, ctl_of(&sm));
  if (tid == NT - 1) pop_next(P, ctl_of(&sm));  // short chains depend only on the short preps queued before them
  const uint32_t sSb0 = smem_u32(sm.Sb[0]), sDb = smem_u32(sm.Db);
  const uint32_t tS = tmem, tDl = tmem + VW_;       // TMEM columns: S [0,VW_), Delta [VW_,2VW_)
  const uint32_t tq = (uint32_t)(32 * q) << 16;     // this warp's TMEM lane quadrant
  const int key = 32 * q + lane, cs = hf * HC;      // this thread's S slice
  const uint32_t idesc1 = make_idesc(64, VW_, 0, 1), idesc2 = make_idesc(128, VW_, 1, 1);
  constexpr int PT = 7 * 32;  // control thread (warp 7, lane 0): K prefetch, TMA stores, publishing
  constexpr int WT = 5 * 32;  // control thread (warp 5, lane 0): prep-flag polling + W/U/cl prefetch
  const int cend = c0 + nc;
  const CUtensorMap* mU = (VW_ == 16) ? &P.tmU16 : &P.tmU32;
  const CUtensorMap* mD = (VW_ == 16) ? &P.tmD16 : &P.tmD32;
  const CUtensorMap* mS0 = (VW_ == 16) ? &P.tmS016 : &P.tmS032;
  const CUtensorMap* mSt = (VW_ == 16) ? &P.tmState16 : &P.tmState;
  const CUtensorMap* mNSt = (VW_ == 16) ? &P.tmNewState16 : &P.tmNewState;
  uint32_t mph = 0;  // bar_mma phase

  auto prefetch_W = [&](int c, int stg) {  // needs prep flag of chunk c (+ fence.proxy.async by the caller)
    uint64_t* bar = &sm.barW[stg];
    const int row0 = (c * HV + h) * C;
    mbar_arrive_tx(bar, 2 * 8192 + C * VW_ * 2 + C * 4);
    tma_load_3d(sm.Ws[stg], &P.tmW, row0, (0) >> 6, bar);
    tma_load_2d(sm.Us[stg], mU, v0, row0, bar);
    bulk_g2s(sm.cls[stg], P.clbuf + (size_t)row0, C * 4, bar);
  };
  auto prefetch_K = [&](int c, int stg) {  // raw input, no flag needed
    uint64_t* bar = &sm.barK[stg];
    const int t0 = tbase + (c - c0) * C;
    mbar_arrive_tx(bar, 2 * 8192);
    tma_load_3d(sm.Ks[stg], &P.tmK, t0, ((h >> 1) * D) >> 6, bar);
  };
  // TMEM slice access (S: lane = key, cols cs..cs+HC-1 ; Delta: M=64 layout, lane 32q+i <-> row 16q+i)
  auto ld_slice = [&](uint32_t taddr, float* v) {
    if constexpr (HC == 8) tmem_ld8(taddr, reinterpret_cast<uint32_t*>(v)); else tmem_ld16(taddr, reinterpret_cast<uint32_t*>(v));
    tmem_wait_ld();
  };
  auto st_slice = [&](uint32_t taddr, const float* v) {
    if constexpr (HC == 8) tmem_st8(taddr, reinterpret_cast<const uint32_t*>(v)); else tmem_st16(taddr, reinterpret_cast<const uint32_t*>(v));
    tmem_wait_st();
  };
  auto write_Sb = [&](int buf, const float* v) {
    unsigned char* Sb = reinterpret_cast<unsigned char*>(sm.Sb[buf]);
#pragma unroll
    for (int j = 0; j < HC; j += 8) {
      uint4 pk = make_uint4(pack2(v[j], v[j + 1]), pack2(v[j + 2], v[j + 3]), pack2(v[j + 4], v[j + 5]), pack2(v[j + 6], v[j + 7]));
      *reinterpret_cast<uint4*>(Sb + swv<VW_>(key, cs + j)) = pk;
    }
  };
  float* stg_f = reinterpret_cast<float*>(sm.Ws[WST - 1]);  // fp32 state staging (last W stage is free until its first prefetch)
  int pf_flag = 0;
  if (tid == PT) {
    mbar_arrive_tx(&sm.bar_state, 4 * VW_ * 32 * 4);
#pragma unroll
    tma_load_3d(stg_f, mSt, (s * HV + h) * D + v0, 0, &sm.bar_state);
    if (nc > 0) {
      prefetch_K(c0, 0);
      if (nc > 1) prefetch_K(c0 + 1, 1);
    }
  }
  if (tid == WT && nc > 0) {
    wait_flag(pflag + c0 * HV + h, P.epoch);
    fence_proxy_async_global();
    prefetch_W(c0, 0);
    if (nc > 1) pf_flag = ld_relaxed(pflag + (c0 + 1) * HV + h);
    if constexpr (WST == 3) {  // stage 1 is free right away (stage 2 is the state staging buffer)
      if (nc > 1) {
        while (pf_flag < P.epoch) { __nanosleep(40); pf_flag = ld_relaxed(pflag + (c0 + 1) * HV + h); }
        fence_proxy_async_global();
        prefetch_W(c0 + 1, 1);
        if (nc > 2) pf_flag = ld_relaxed(pflag + (c0 + 2) * HV + h);
      }
    }
  }
  mbar_wait(&sm.bar_state, 0);
  float Sv[HC];
  {
    const unsigned char* sf = reinterpret_cast<const unsigned char*>(stg_f);
#pragma unroll
    for (int j = 0; j < HC; ++j) Sv[j] = *reinterpret_cast<const float*>(sf + sw128f<VW_>(cs + j, key));
  }
  __syncthreads();  // staging buffer consumed
  if constexpr (WST == 2) {  // stage 1 (staging buffer) may now receive W[c0+1]
    if (tid == WT && nc > 1) {
      while (pf_flag < P.epoch) { __nanosleep(40); pf_flag = ld_relaxed(pflag + (c0 + 1) * HV + h); }
      fence_proxy_async_global();
      prefetch_W(c0 + 1, 1);
      if (nc > 2) pf_flag = ld_relaxed(pflag + (c0 + 2) * HV + h);
    }
  }
  if (nc > 0) {
    write_Sb(0, Sv);
    mbar_wait(&sm.barW[0], 0);
    const float lam0 = fexp(sm.cls[0][C - 1]);
#pragma unroll
    for (int j = 0; j < HC; ++j) Sv[j] *= lam0;
    st_slice(tS + tq + cs, Sv);  // TMEM S = lam_C(c0) * S0
    fence_proxy_async();
    tc_fence_before();
    __syncthreads();
    if (tid == PT) { tma_store_2d(mS0, v0, (c0 * HV + h) * D, sm.Sb[0]); bulk_commit(); }
  }

  const bool dbgsel = GDN_DBGON && s == P.dbg_seq && h == 0 && sl == 0;
  const bool dbgit = dbgsel && tid == 128;
  auto stamp = [&](int it, int j) { if (dbgit) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(8192 + it) * 8 + j] = t; } };

  for (int it = 0; it < nc; ++it) {
    const int c = c0 + it, wstg = it % WST, kstg = it & 1, buf = it & 1;
    const uint32_t kpar = (it >> 1) & 1;
    const uint32_t sSb = sSb0 + buf * (D * VW_ * 2);
    stamp(it, 0);
    // W[c], U[c], cls[c] (barW[wstg]) and Sb[buf] (= bf16 S_c) are complete: waited/written before the last barrier
    if (tid == 0) {  // GEMM1: Delta_raw = W S  (64 x VW_, K = 128); A = W K-major SW128, B = Sb MN-major
      tc_fence_after();
      const uint32_t sW = smem_u32(sm.Ws[wstg]);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int k0 = ks * 16;
        umma(tDl, make_desc(sW + (k0 >> 6) * 8192 + (k0 & 63) * 2, 8192, 1024, 2), make_desc(sSb + k0 * ROWB, 2 * ROWB * 8, ROWB * 8, SWZ), idesc1, ks > 0);
      }
      umma_commit(&sm.bar_mma);
    }
    if (tid == PT) {  // control work, overlapped with GEMM1
      if (it >= 1 && c + 1 < cend) prefetch_K(c + 1, kstg ^ 1);  // Ks[kstg^1] was consumed by GEMM2 of step it-1 (completed)
      if (it >= 2 && (it % PUB_EVERY) == 0) {  // publish: chunks < it-1 fully in global (TMA writes complete)
        bulk_wait<2>();
        fence_proxy_async_global();
#pragma unroll
        for (int z = 0; z < 4; ++z) { st_relaxed64(prog + z, ep64 | (unsigned long long)(it - 1)); st_relaxed64(prog + prog2off + z, ep64 | (unsigned long long)(it - 1)); }
      }
    }
    if (tid == WT && c + 2 < cend) {
      const int nf = (c + 3 < cend) ? ld_relaxed(pflag + (c + 3) * HV + h) : 0;
      while (pf_flag < P.epoch) { __nanosleep(32); pf_flag = ld_relaxed(pflag + (c + 2) * HV + h); }
      if constexpr (WST == 3) {  // stage (it+2)%3 was consumed by step it-1 (barrier since)
        fence_proxy_async_global();
        prefetch_W(c + 2, (it + 2) % WST);
        pf_flag = nf;
      }
    }
    const float clC = sm.cls[wstg][C - 1];
    mbar_wait(&sm.bar_mma, mph & 1); ++mph;
    tc_fence_after();
    stamp(it, 1);
    {  // epilogue: Delta = U - W S ; Du = bf16(Delta) (TMA stored), Db = bf16(Delta * exp(clC - cl_i))
      float dv[HC];
      ld_slice(tDl + tq + cs, dv);
      if (lane < 16) {
        const int m = 16 * q + lane;
        const float rs = fexp(clC - sm.cls[wstg][m]);
        const unsigned char* Us = reinterpret_cast<const unsigned char*>(sm.Us[wstg]);
        unsigned char* Db = reinterpret_cast<unsigned char*>(sm.Db);
        unsigned char* Du = reinterpret_cast<unsigned char*>(sm.Du[buf]);
#pragma unroll
        for (int j = 0; j < HC; j += 8) {
          const uint32_t o = swv<VW_>(m, cs + j);
          const uint4 u = *reinterpret_cast<const uint4*>(Us + o);
          const float2 u0 = unpack2(u.x), u1 = unpack2(u.y), u2 = unpack2(u.z), u3 = unpack2(u.w);
          const float d0 = u0.x - dv[j], d1 = u0.y - dv[j + 1], d2 = u1.x - dv[j + 2], d3 = u1.y - dv[j + 3];
          const float d4 = u2.x - dv[j + 4], d5 = u2.y - dv[j + 5], d6 = u3.x - dv[j + 6], d7 = u3.y - dv[j + 7];
          *reinterpret_cast<uint4*>(Du + o) = make_uint4(pack2(d0, d1), pack2(d2, d3), pack2(d4, d5), pack2(d6, d7));
          *reinterpret_cast<uint4*>(Db + o) = make_uint4(pack2(d0 * rs, d1 * rs), pack2(d2 * rs, d3 * rs), pack2(d4 * rs, d5 * rs), pack2(d6 * rs, d7 * rs));
        }
      }
    }
    fence_proxy_async();
    tc_fence_before();
    if (tid == PT) bulk_wait_read<2>();  // pending: Du[c-1], S0[c] at most -> S0[c-1] done reading Sb[buf^1] (rewritten below)
    __syncthreads();
    stamp(it, 2);
    if (tid == PT) { tma_store_2d(mD, v0, (c * HV + h) * C, sm.Du[buf]); bulk_commit(); }
    if constexpr (WST == 2) {  // Ws[wstg]/Us[wstg]/cls[wstg] consumed by this step
      if (tid == WT && c + 2 < cend) {
        fence_proxy_async_global();
        prefetch_W(c + 2, wstg);
        if (c + 3 < cend) pf_flag = ld_relaxed(pflag + (c + 3) * HV + h);
      }
    }
    if (tid == 0) {  // GEMM2: S += K^T Db  (128 x VW_, K = 64); A = K MN-major SW128 (2 atoms, LBO 8192), B = Db MN-major
      mbar_wait(&sm.barK[kstg], kpar);
      tc_fence_after();
      const uint32_t sK = smem_u32(sm.Ks[kstg]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16;
        umma(tS, make_desc(sK + k0 * 128, 8192, 1024, 2), make_desc(sDb + k0 * ROWB, 2 * ROWB * 8, ROWB * 8, SWZ), idesc2, 1);
      }
      umma_commit(&sm.bar_mma);
    }
    mbar_wait(&sm.bar_mma, mph & 1); ++mph;
    tc_fence_after();
    stamp(it, 3);
    if (it + 1 < nc) {  // S_{c+1} -> Sb[buf^1] (bf16, GEMM1 operand + S0 store), TMEM S *= lam_C(c+1)
      const int nx = it + 1, nw = nx % WST;
      mbar_wait(&sm.barW[nw], (nx / WST) & 1);
      const float lamN = fexp(sm.cls[nw][C - 1]);
      ld_slice(tS + tq + cs, Sv);
      write_Sb(buf ^ 1, Sv);
#pragma unroll
      for (int j = 0; j < HC; ++j) Sv[j] *= lamN;
      st_slice(tS + tq + cs, Sv);
      fence_proxy_async();
      tc_fence_before();
      if (tid == PT) bulk_wait_read<2>();  // pending: Du[c], S0[c] at most -> Du[buf^1] reusable next step
      __syncthreads();
      if (tid == PT) { tma_store_2d(mS0, v0, ((c + 1) * HV + h) * D, sm.Sb[buf ^ 1]); bulk_commit(); }
    }
    stamp(it, 4);
  }
  if (tid == PT) bulk_wait0();
  __syncthreads();
  {  // new state: TMEM S (unscaled after the last step) -> fp32 staging -> TMA store
    ld_slice(tS + tq + cs, Sv);
    unsigned char* sf = reinterpret_cast<unsigned char*>(sm.Ws[0]);
#pragma unroll
    for (int j = 0; j < HC; ++j) *reinterpret_cast<float*>(sf + sw128f<VW_>(cs + j, key)) = Sv[j];
  }
  fence_proxy_async();
  __syncthreads();
  if (tid == PT) {
    float* f = reinterpret_cast<float*>(sm.Ws[0]);
#pragma unroll
    tma_store_3d(mNSt, (s * HV + h) * D + v0, 0, f);
    bulk_commit();
    bulk_wait0();
    fence_proxy_async_global();
    fence_acq_rel_gpu();
#pragma unroll
    for (int z = 0; z < 4; ++z) { st_relaxed64(prog + z, ep64 | (unsigned long long)nc); st_relaxed64(prog + prog2off + z, ep64 | (unsigned long long)nc); }
  }
}

// ============================================================================================
// item CHAIN (long sequences, VW = 16), software pipelined:
//   X_c = W_c bf16(lamC_{c-1} S_{c-1}) + G_c Db_{c-1}   (G_c = W_c K_{c-1}^T from prep)
//   Delta_c = U_c - X_c ;  S_c = lamC_{c-1} S_{c-1} + K_{c-1}^T Db_{c-1}
// The only GEMM on the serial path is the 4-instruction G_c Db_{c-1} accumulate; the W GEMM (8) and the
// state update (4) of the previous chunk overlap with it.  S and X live in TMEM.
// ============================================================================================
// ============================================================================================
// item CHAIN16 (v8): warp-specialized chain with mma.sync, state in registers, VW = 16.
//   E-warps (0-3, rows m = 16q..16q+15):  Xw = W_c Sbw_c ; Delta_c = U_c - lam Xw - Xx_c ; D (global), Db (smem)
//   S-warps (4-7, rows d = 32q..32q+31):  Sb0_c = bf16(S_c) (global) ; Xx_{c+1} = G_{c+1} Db_c (rows 16q.., -> XX smem for E) ; S = lamC_c S_c + K_c^T Db_c (registers) ; Sbw_{c+2} = bf16(S_{c+1})
//   Sbw_c = bf16(lamC_{c-1} S_{c-1}) and the correction term G_c Db_{c-1} (G_c = W_c K_{c-1}^T, from prep) give X_c = W_c S_c.
// ============================================================================================
__device__ __forceinline__ void dev_chain16(const Params& P, const Tbl& tb, ChainSmemP& sm, uint32_t, int s, int h, int sl) {
  constexpr int VW_ = 16;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, q = w & 3, g = lane >> 2, t4 = lane & 3;
  const int v0 = sl * VW_;
  const int c0 = __ldcg(tb.sc0 + s), nc = __ldcg(tb.snc + s), tbase = __ldcg(tb.stb + s);
  const int cend = c0 + nc;
  const int* pflag = P.flags + 1025;
  unsigned long long* prog = P.prog + ((size_t)(s * HV + h) * NSPLIT + sl) * 4;
  unsigned long long* prog2 = prog + (size_t)P.N * HV * NSPLIT * 4;
  const unsigned long long ep64 = (unsigned long long)P.epoch << 32;
  if (tid < 2) {
    mbar_inval(&sm.barW[tid]); mbar_init(&sm.barW[tid], 1);
    mbar_inval(&sm.barU[tid]); mbar_init(&sm.barU[tid], 1);
    mbar_inval(&sm.barWfree[tid]); mbar_init(&sm.barWfree[tid], 4);  // all E-warps finished GEMMw(c): Ws slot reusable
    mbar_inval(&sm.barK[tid]); mbar_init(&sm.barK[tid], 1);
    mbar_inval(&sm.barSbw[tid]); mbar_init(&sm.barSbw[tid], 4);
    mbar_inval(&sm.barD[tid]); mbar_init(&sm.barD[tid], 4);
    mbar_inval(&sm.barK2done[tid]); mbar_init(&sm.barK2done[tid], 4);  // ring by chunk parity: a fast S-warp may finish GEMM2(j+1) before a slow one finishes GEMM2(j)
  }
  if (tid < 3) { mbar_inval(&sm.barG[tid]); mbar_init(&sm.barG[tid], 1); }
  if (tid == 0) {
    mbar_inval(&sm.bar_state); mbar_init(&sm.bar_state, 1);
    sm.wiss = c0 + 2;  // W chunks < wiss have been issued by S-warp 6 (prep flag observed): gates the U issue on S-warp 4
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncthreads();
  if (tid == 0) flush_pending(P, ctl_of(&sm));  // a pending prep flag must not wait until this chain ends

  const bool dbgsel = GDN_DBGON && s == P.dbg_seq && h == 0 && sl == 0;
  auto stamp = [&](int it, int j) { if (dbgsel && it < 512) { P.dbg[(8192 + it) * 8 + j] = (unsigned long long)clock64(); } };
  auto calib = [&](int j) { if (dbgsel) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(8192 + 1024) * 8 + j] = t; P.dbg[(8192 + 1024) * 8 + j + 2] = (unsigned long long)clock64(); } };
  auto xstamp = [&](int it, int j) { if (dbgsel && it < 512) { P.dbg[(8192 + 512 + it) * 8 + j] = (unsigned long long)clock64(); } };
  auto count = [&](int j) { if (dbgsel) atomicAdd((unsigned long long*)P.dbg + (8192 + 1025) * 8 + j, 1ull); };

  auto prefetch_W = [&](int c, int stg) {  // W_c -> stage stg (issued as soon as GEMMw(c-2) is done on all E-warps)
    const int row0 = (c * HV + h) * C;
    mbar_arrive_tx(&sm.barW[stg], 2 * 8192);
    tma_load_3d(sm.Ws[stg], &P.tmW, row0, (0) >> 6, &sm.barW[stg]);
  };
  auto prefetch_U = [&](int c, int stg) {  // U_c, cl_c -> stage stg (the slot is busy until the E-warps' epilogue(c-2): barD)
    const int row0 = (c * HV + h) * C;
    mbar_arrive_tx(&sm.barU[stg], C * VW_ * 2 + C * 4);
    tma_load_2d(sm.Us[stg], &P.tmU16, v0, row0, &sm.barU[stg]);
    bulk_g2s(sm.cls[stg], P.clbuf + (size_t)row0, C * 4, &sm.barU[stg]);
  };
  auto prefetch_K = [&](int c, int stg) {
    mbar_arrive_tx(&sm.barK[stg], 2 * 8192);
    const int t0 = tbase + (c - c0) * C;
    tma_load_3d(sm.Ks[stg], &P.tmK, t0, ((h >> 1) * D) >> 6, &sm.barK[stg]);
  };
  auto prefetch_G = [&](int c, int stg) {
    mbar_arrive_tx(&sm.barG[stg], 8192);
    tma_load_2d(sm.Gs[stg], &P.tmG, 0, (c * HV + h) * C, &sm.barG[stg]);
  };
  auto flag_wait = [&](int c) { wait_flag(pflag + c * HV + h, P.epoch); fence_proxy_async_global(); };
  // ldmatrix lane geometry
  const int rA = (lane & 7) + 8 * ((lane >> 3) & 1), cA = 8 * (lane >> 4);   // A [m][k] (non-trans) and B [k][n] (trans)
  const int rAT = (lane & 7) + 8 * (lane >> 4), cAT = 8 * ((lane >> 3) & 1);  // A^T from [k][m] storage (trans)

  // ---- prologue loads
  float* stg_f = reinterpret_cast<float*>(sm.Gs[2]);  // fp32 state staging [16 v][128 d] (G slot 2 is loaded after the staging is consumed)
  if (tid == 128) {
    mbar_arrive_tx(&sm.bar_state, 4 * VW_ * 32 * 4);
#pragma unroll
    tma_load_3d(stg_f, &P.tmState16, (s * HV + h) * D + v0, 0, &sm.bar_state);
    prefetch_K(c0, 0);
    prefetch_K(c0 + 1, 1);
    flag_wait(c0);
    prefetch_W(c0, 0); prefetch_U(c0, 0);
  }
  if (tid == 64) { flag_wait(c0 + 1); prefetch_W(c0 + 1, 1); prefetch_U(c0 + 1, 1); prefetch_G(c0 + 1, 1); }

  if (w < 4) {
    // =============================== E-warps ===============================
    const int m0 = 16 * q, r0 = m0 + g, r1 = r0 + 8;
    float lam_prev = 1.f;  // exp(clC_{c-1}): Sbw_c = bf16(S_{c-1}) is unscaled, X_c = lam_prev * W_c Sbw_c
    float xx[8];  // Xx_c = G_c Db_{c-1} (GEMMx of the previous iteration, same accumulator lane layout); 0 for the first chunk
#pragma unroll
    for (int e = 0; e < 8; ++e) xx[e] = 0.f;
    for (int i = 0; i < nc; ++i) {
      const int c = c0 + i, stg = i & 1;
      mbar_wait(&sm.barSbw[stg], (i >> 1) & 1);  // Sbw_c = bf16(S_{c-1}) from the S-warps (packed after GEMM2(c-2))
      mbar_wait(&sm.barW[stg], (i >> 1) & 1);
      mbar_wait(&sm.barU[stg], (i >> 1) & 1);
      if (tid == 0) { stamp(i, 0); if (i == 0 || i == nc - 1) calib(i == 0 ? 0 : 1); }
      const float clC = sm.cls[stg][C - 1], lamC = fexp(clC);
      if (tid == 0) sm.clr[stg][0] = lamC;  // -> S-warps (read after barD(c)): S = lamC S + K^T Db
      const float rs0 = fexp(clC - sm.cls[stg][r0]), rs1 = fexp(clC - sm.cls[stg][r1]);
      const unsigned char* Ub = reinterpret_cast<const unsigned char*>(sm.Us[stg]);
      uint32_t uraw[4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        uraw[nt * 2] = *reinterpret_cast<const uint32_t*>(Ub + sw32(r0, nt * 8 + 2 * t4));
        uraw[nt * 2 + 1] = *reinterpret_cast<const uint32_t*>(Ub + sw32(r1, nt * 8 + 2 * t4));
      }
      // GEMMw: xw = W_c Sbw_c   (16 x 16 per warp, K = 128)
      float xw[8], xw1[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) { xw[e] = 0.f; xw1[e] = 0.f; }
      {
        const uint32_t sW = smem_u32(sm.Ws[stg]), sB = smem_u32(sm.Sbw[stg]);
#ifdef HOIST_W
        uint32_t a[8][4], b[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          ldsm_x4(a[ks], sW + sw128<64>(m0 + rA, ks * 16 + cA));
          ldsm_x4_t(b[ks], sB + sw32(ks * 16 + rA, cA));
        }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          float* acc = (ks & 1) ? xw1 : xw;
          mma16816(acc, a[ks], b[ks][0], b[ks][1]);
          mma16816(acc + 4, a[ks], b[ks][2], b[ks][3]);
        }
#else
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          uint32_t a[4][4], b[4][4];
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            const int ks = half * 4 + kk;
            ldsm_x4(a[kk], sW + sw128<64>(m0 + rA, ks * 16 + cA));
            ldsm_x4_t(b[kk], sB + sw32(ks * 16 + rA, cA));
          }
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            float* acc = (kk & 1) ? xw1 : xw;
            mma16816(acc, a[kk], b[kk][0], b[kk][1]);
            mma16816(acc + 4, a[kk], b[kk][2], b[kk][3]);
          }
        }
#endif
#pragma unroll
        for (int e = 0; e < 8; ++e) xw[e] = (xw[e] + xw1[e]) * lam_prev;
      }
      if (lane == 0) mbar_arrive(&sm.barWfree[stg]);  // Ws[stg] reads done (ldmatrix results consumed by the mma above)
      lam_prev = lamC;
      if (tid == 0) stamp(i, 1);
      if (tid == 0) stamp(i, 2);
      // epilogue: D_c = U_c - X_c - G_c Db_{c-1} -> global (bf16) ; Db_c = bf16(D_c * exp(clC - cl_i)) -> smem
      unsigned char* Dbb = reinterpret_cast<unsigned char*>(sm.Db[stg]);
      bf16* Dg = P.Dg + ((size_t)(c * HV + h) * C) * D + v0;
      unsigned char* Ust = reinterpret_cast<unsigned char*>(sm.Us[stg]);  // D_c staging: each thread overwrites exactly the U_c elements it read
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = nt * 8 + 2 * t4;
        const float2 u0 = unpack2(uraw[nt * 2]), u1 = unpack2(uraw[nt * 2 + 1]);
        const float d00 = u0.x - xw[nt * 4 + 0] - xx[nt * 4 + 0], d01 = u0.y - xw[nt * 4 + 1] - xx[nt * 4 + 1];
        const float d10 = u1.x - xw[nt * 4 + 2] - xx[nt * 4 + 2], d11 = u1.y - xw[nt * 4 + 3] - xx[nt * 4 + 3];
        *reinterpret_cast<uint32_t*>(Ust + sw32(r0, col)) = pack2(d00, d01);
        *reinterpret_cast<uint32_t*>(Ust + sw32(r1, col)) = pack2(d10, d11);
        *reinterpret_cast<uint32_t*>(Dbb + sw32(r0, col)) = pack2(d00 * rs0, d01 * rs0);
        *reinterpret_cast<uint32_t*>(Dbb + sw32(r1, col)) = pack2(d10 * rs1, d11 * rs1);
      }
      __syncwarp();
      const int drow = m0 + (lane >> 1), dcol = (lane & 1) * 8;  // coalesced read-back of this warp's 16 rows x 32 B (own warp's writes only)
      const uint4 dv = *reinterpret_cast<const uint4*>(Ust + sw32(drow, dcol));
      if (tid == 0) stamp(i, 3);
      if (tid == 32 || tid == 96) xstamp(i, 6 + (tid >> 6));
      asm volatile("bar.sync 1, 128;\n" ::: "memory");    // all E-warps: Db_c complete, W_c/U_c/cls_c/XX reads done
      if (lane == 0) mbar_arrive(&sm.barD[stg]);  // -> S-warps (GEMMx / GEMM2 may start)
      if (tid == 0) stamp(i, 4);
      *reinterpret_cast<uint4*>(Dg + (size_t)drow * D + dcol) = dv;  // D_c -> global: one 16 B store per lane, full sectors
      if (i + 1 < nc) {  // GEMMx: xx = G_{c+1} Db_c  (16 x 16 per warp, K = 64); Db_c complete (bar.sync 1 above)
        const int gs = (i + 1) % 3;
        mbar_wait(&sm.barG[gs], (i / 3) & 1);  // chunk m = i+1 -> phase (m-1)/3
        float xx1[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { xx[e] = 0.f; xx1[e] = 0.f; }
        const uint32_t sG = smem_u32(sm.Gs[gs]), sB = smem_u32(sm.Db[stg]);
        uint32_t a[4][4], b[4][4];
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          ldsm_x4(a[ks], sG + sw128<64>(m0 + rA, ks * 16 + cA));
          ldsm_x4_t(b[ks], sB + sw32(ks * 16 + rA, cA));
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          float* acc = (ks & 1) ? xx1 : xx;
          mma16816(acc, a[ks], b[ks][0], b[ks][1]);
          mma16816(acc + 4, a[ks], b[ks][2], b[ks][3]);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) xx[e] += xx1[e];
      }
      if (tid == 0) stamp(i, 5);
    }
    __syncwarp();
  } else {
    // =============================== S-warps ===============================
    const int d0 = 32 * q;
    float S[16];  // S[mt*8 + nt*4 + e]: row d0 + mt*16 + g + 8*(e>>1), col nt*8 + 2*t4 + (e&1)
    mbar_wait(&sm.bar_state, 0);
    {
      const unsigned char* sf = reinterpret_cast<const unsigned char*>(stg_f);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            S[mt * 8 + nt * 4 + e] = *reinterpret_cast<const float*>(sf + sw128f<VW_>(nt * 8 + 2 * t4 + (e & 1), d0 + mt * 16 + g + 8 * (e >> 1)));
    }
    uint32_t sp[8];  // bf16(S) packed pairs along v: sp[mt*4 + nt*2 + hh] = rows d0+mt*16+g+8*hh, cols nt*8+2*t4..+1
    auto pack_S = [&]() {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          sp[mt * 4 + nt * 2 + 0] = pack2(S[mt * 8 + nt * 4 + 0], S[mt * 8 + nt * 4 + 1]);
          sp[mt * 4 + nt * 2 + 1] = pack2(S[mt * 8 + nt * 4 + 2], S[mt * 8 + nt * 4 + 3]);
        }
    };
    auto store_tile = [&](bf16* tile) {  // sp -> sw32 tile [128 d][16 v]
      unsigned char* b = reinterpret_cast<unsigned char*>(tile);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int dd = d0 + mt * 16 + g, col = nt * 8 + 2 * t4;
          *reinterpret_cast<uint32_t*>(b + sw32(dd, col)) = sp[mt * 4 + nt * 2 + 0];
          *reinterpret_cast<uint32_t*>(b + sw32(dd + 8, col)) = sp[mt * 4 + nt * 2 + 1];
        }
    };
    auto store_Sb0 = [&](const bf16* tile, int c) {  // bf16(S_c) tile (this warp's 32 rows x 32 B) -> S0g, 16 B per lane
      const unsigned char* b = reinterpret_cast<const unsigned char*>(tile);
      bf16* Sg = P.S0g + ((size_t)(c * HV + h) * D) * D + v0;
      const int rr = d0 + (lane >> 1), cc = (lane & 1) * 8;
      const uint4 v0v = *reinterpret_cast<const uint4*>(b + sw32(rr, cc)), v1v = *reinterpret_cast<const uint4*>(b + sw32(rr + 16, cc));
      *reinterpret_cast<uint4*>(Sg + (size_t)rr * D + cc) = v0v;
      *reinterpret_cast<uint4*>(Sg + (size_t)(rr + 16) * D + cc) = v1v;
    };
    int gnext = c0 + 2, fval = -1;  // tid 224: next G chunk to fetch and the pipelined flag value loaded for it
    int wnext = c0 + 2, fw = -1, unext = c0 + 2;    // tid 192: next W chunk to fetch (and its pipelined flag value); tid 128: next U/cl chunk
    auto gblock = [&](int gmax) {  // tid 224: G_{gnext} -> slot (gnext-c0)%3 while gnext <= gmax; spin on the prep flag only for the chunk needed next (GEMMx(c) needs G_{c+1})
      for (int rep = 0; rep < 2 && gnext < cend && gnext <= gmax; ++rep) {
        if (fval < P.epoch) {
          if (gnext != gmax - 1) { count(0); break; }
          count(1); wait_flag(pflag + gnext * HV + h, P.epoch); fence_proxy_async_global();
        }
        prefetch_G(gnext, (gnext - c0) % 3);
        xstamp(gnext - c0, 0);
        ++gnext;
        fval = (gnext < cend) ? ld_relaxed(pflag + gnext * HV + h) : -1;
      }
    };
    if ((tid == 192 || tid == 224) && gnext < cend) { fval = ld_relaxed(pflag + gnext * HV + h); fw = fval; }
    pack_S(); store_tile(sm.Sbw[0]); store_tile(sm.Sbw[1]);  // Sbw_{c0} = Sbw_{c0+1} = bf16(S0)  (E scales Sbw_{c0+1} by lamC_{c0})
    __syncwarp();
    if (lane == 0) { mbar_arrive(&sm.barSbw[0]); mbar_arrive(&sm.barSbw[1]); }
    store_Sb0(sm.Sbw[0], c0);
    for (int j = 0; j < nc; ++j) {
      const int c = c0 + j, stg = j & 1;
      // (Sb0_c = bf16(S_c) was stored right after GEMM2(c-1); Sbw_{c+1} = bf16(S_c) was handed to E then)
      if (tid == 128) xstamp(j, 3);
      if (tid == 192) {
        // W_{wnext} -> Ws[(wnext-c0)&1] once all E-warps finished GEMMw(wnext-2) (barWfree) and the prep flag (loaded one iteration ahead) is set;
        // only spin on the flag when the chunk is needed next.
        for (int rep = 0; rep < 2 && wnext < cend && wnext <= c + 2; ++rep) {
          if (fw < P.epoch) {
            if (wnext != c + 1) { count(2); break; }
            count(3); wait_flag(pflag + wnext * HV + h, P.epoch); fence_proxy_async_global();
          }
          mbar_wait(&sm.barWfree[(wnext - c0) & 1], ((wnext - c0 - 2) >> 1) & 1);
          prefetch_W(wnext, (wnext - c0) & 1);
          xstamp(wnext - c0, 1);
          ++wnext;
          fw = (wnext < cend) ? ld_relaxed(pflag + wnext * HV + h) : -1;
          *reinterpret_cast<volatile int*>(&sm.wiss) = wnext;
        }
      }
      if (tid == 128) xstamp(j, 5);
      mbar_wait(&sm.barD[stg], (j >> 1) & 1);  // D_c / Db_c (and lamC_c) from the E-warps
      const float lam = sm.clr[stg][0];
#pragma unroll
      for (int e = 0; e < 16; ++e) S[e] *= lam;
      uint32_t b[4][4];  // Db_c fragments (B operand of both GEMMx and GEMM2)
      const uint32_t sB = smem_u32(sm.Db[stg]);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) ldsm_x4_t(b[ks], sB + sw32(ks * 16 + rA, cA));
      if (tid == 128) {
        // U_{unext}/cl -> slot (unext-c0)&1: the slot held U_c (D_c staging), all E-warps are past epilogue(c) (barD(c) waited above);
        // W_{unext} must have been issued by S-warp 6 (prep flag seen, wiss): only wait for it when U_{c+1} is needed next
        while (unext < cend && unext <= c + 2) {
          int wi = *reinterpret_cast<volatile int*>(&sm.wiss);
          if (wi <= unext) { if (unext != c + 1) break; do { wi = *reinterpret_cast<volatile int*>(&sm.wiss); } while (wi <= unext); }
          prefetch_U(unext, (unext - c0) & 1); ++unext;
        }
      }
      if (tid == 224) {
        // G_{gnext} -> slot (gnext-c0)%3 once gnext <= c+3: the slot held G_c, read by the E-warps in GEMMx(c-1) before their bar.sync(c) / barD(c) (waited above);
        // at j == 0 slot 2 was the state staging, consumed by all S-warps before Sbw_{c0} (waited by E before barD(c0)).
        for (int rep = 0; rep < 2 && gnext < cend && gnext <= c + 3; ++rep) {
          if (fval < P.epoch) {
            if (gnext != c + 2) { count(0); break; }
            count(1); wait_flag(pflag + gnext * HV + h, P.epoch); fence_proxy_async_global();
          }
          prefetch_G(gnext, (gnext - c0) % 3);
          xstamp(gnext - c0, 0);
          ++gnext;
          fval = (gnext < cend) ? ld_relaxed(pflag + gnext * HV + h) : -1;
        }
      }
      mbar_wait(&sm.barK[stg], (j >> 1) & 1);
      if (tid == 128) stamp(j, 6);
      {  // GEMM2: S += K_c^T Db_c  (32 x 16 per warp, K = 64), all fragments hoisted
        const uint32_t sK = smem_u32(sm.Ks[stg]);
        uint32_t a[4][2][4];
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          ldsm_x4_t(a[ks][0], sK + sw128<64>(ks * 16 + rAT, d0 + cAT));
          ldsm_x4_t(a[ks][1], sK + sw128<64>(ks * 16 + rAT, d0 + 16 + cAT));
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(S + mt * 8, a[ks][mt], b[ks][0], b[ks][1]);
            mma16816(S + mt * 8 + 4, a[ks][mt], b[ks][2], b[ks][3]);
          }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&sm.barK2done[stg]);
      if (tid == 128) stamp(j, 7);
      if (tid == 160 && j >= 1 && (j % PUB_EVERY) == 0) {
        // single publisher: D_{<c} (E-warps, via barD(j)) and Sb0_{<c} (S-warps, via barK2done(j-1)); placed where this warp has no young global stores (the release drains them)
        mbar_wait(&sm.barK2done[(j - 1) & 1], ((j - 1) >> 1) & 1);
        st_release64(prog, ep64 | (unsigned long long)j);
      }
      // no fence.proxy.async between the (control-dependent) flag observation and the TMA issue: TMA reads L2, where the prep's bulk stores were complete before its flag release
      pack_S();  // sp = bf16(S_{c+1}): Sbw_{c+2} for E (E applies lamC_{c+1}) and Sb0_{c+1} (global, coalesced via the tile)
      if (j + 1 < nc) {  // slot stg: E finished GEMMw(c) before Db_c existed (barD(c) precedes this GEMM2); unused by E when j+2 >= nc
        store_tile(sm.Sbw[stg]);
        __syncwarp();
        if (j + 2 < nc && lane == 0) mbar_arrive(&sm.barSbw[stg]);
        store_Sb0(sm.Sbw[stg], c + 1);  // (c+1 < cend: chunk cend's Sb0 belongs to the next sequence)
      }
      if (tid == 160) { mbar_wait(&sm.barK2done[stg], (j >> 1) & 1); if (c + 2 < cend) { prefetch_K(c + 2, stg); xstamp(j + 2, 2); } }
    }
    if (tid == NT - 1) pop_next(P, ctl_of(&sm));  // next item: both round trips overlap the state store / the E-warps' last epilogue
    {  // final state S_nc -> new_state[s][h][v][d] fp32
      float* ns = P.nsg + ((size_t)(s * HV + h) * D + v0) * D;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            ns[(size_t)(nt * 8 + 2 * t4 + (e & 1)) * D + d0 + mt * 16 + g + 8 * (e >> 1)] = S[mt * 8 + nt * 4 + e];
    }
    if (tid == 160) { mbar_wait(&sm.barK2done[(nc - 1) & 1], ((nc - 1) >> 1) & 1); st_release64(prog, ep64 | (unsigned long long)nc); }
    __syncwarp();
  }
  __syncthreads();
}

// ============================================================================================
// item OUT: output per (chunk, head)
// ============================================================================================
__device__ __forceinline__ void dev_out(const Params& P, const Tbl& tb, OutSmem& sm, int ci, int h) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t sQs = smem_u32(sm.Qs), sKs = smem_u32(sm.Ks), sS0 = smem_u32(sm.S0s), sDs = smem_u32(sm.Ds), sPs = smem_u32(sm.Ps);
  const int cinfo = __ldcg(tb.cseq + ci), t0 = __ldcg(tb.ct0 + ci), L = __ldcg(tb.cL + ci);  // cseq: seq | k << 8 | long << 20
  const int s = cinfo & 0xff, k = (cinfo >> 8) & 0xfff;
  const bool longseq = (cinfo >> 20) & 1;
  auto ost = [&](int j) { if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4000 + ci * HV + h) * 8 + j] = t; } };
  ost(0);
  if (tid == 0) { mbar_inval(&sm.bar); mbar_init(&sm.bar, 1); mbar_inval(&sm.bar2); mbar_init(&sm.bar2, 1); }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_tx(&sm.bar, 4 * 8192);
    tma_load_3d(sm.Qs, &P.tmQ, t0, ((h >> 1) * D) >> 6, &sm.bar);
    tma_load_3d(sm.Ks, &P.tmK, t0, ((h >> 1) * D) >> 6, &sm.bar);
    flush_pending(P, ctl_of(&sm));
  }
  if (tid == 64) pop_next(P, ctl_of(&sm));
  if (tid < 64) {  // warp 0 polls the D progress counters (slice, producer warp), warp 1 the S0 counters
    const int nsplit = longseq ? NSPLIT_LONG : NSPLIT_SHORT;
    const unsigned long long target = ((unsigned long long)P.epoch << 32) | (unsigned long long)(k + 1);
    const int slp = (tid & 31) >> 2;
    if (slp < nsplit) wait_flag64_acq(P.prog + ((size_t)(s * HV + h) * NSPLIT + slp) * 4, target);
  }
  asm volatile("bar.sync 1, 64;\n" ::: "memory");
  ost(1);
  if (tid == 0) {
    const int chh = ci * HV + h;
    fence_proxy_async_global();
    mbar_arrive_tx(&sm.bar2, 2 * 8192 + 2 * 16384 + C * 4);
    tma_load_3d(sm.S0s, &P.tmS0, chh * D, (0) >> 6, &sm.bar2);
    tma_load_3d(sm.Ds, &P.tmD, chh * C, (0) >> 6, &sm.bar2);
    bulk_g2s(sm.cl, P.clbuf + (size_t)chh * C, C * 4, &sm.bar2);
  }
  mbar_wait(&sm.bar, 0);  // Q, K
  ost(2);

  {
    const int m0 = 16 * (w & 3), n0 = 32 * (w >> 2);
    uint32_t af[8][4], bfr[8][2][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      int r, c;
      frag_a(lane, m0, ks * 16, r, c);       ldsm_x4(af[ks], sQs + sw128<C>(r, c));
      frag_bn(lane, n0, ks * 16, r, c);      ldsm_x4(bfr[ks][0], sKs + sw128<C>(r, c));
      frag_bn(lane, n0 + 16, ks * 16, r, c); ldsm_x4(bfr[ks][1], sKs + sw128<C>(r, c));
    }
    float acc[4][4] = {};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
        mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
      }
    const int i0 = m0 + g, i1 = i0 + 8;
    mbar_wait(&sm.bar2, 0);  // S0, D, cl (overlapped with the QK^T GEMM above)
    const float cl0 = sm.cl[i0], cl1 = sm.cl[i1];
    unsigned char* Ps = reinterpret_cast<unsigned char*>(sm.Ps);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      int j = n0 + 8 * nt + 2 * t;
      float clj0 = sm.cl[j], clj1 = sm.cl[j + 1];
      float v00 = (j <= i0) ? fexp(cl0 - clj0) * acc[nt][0] : 0.f;
      float v01 = (j + 1 <= i0) ? fexp(cl0 - clj1) * acc[nt][1] : 0.f;
      float v10 = (j <= i1) ? fexp(cl1 - clj0) * acc[nt][2] : 0.f;
      float v11 = (j + 1 <= i1) ? fexp(cl1 - clj1) * acc[nt][3] : 0.f;
      *reinterpret_cast<uint32_t*>(Ps + sw128<C>(i0, j)) = pack2(v00, v01);
      *reinterpret_cast<uint32_t*>(Ps + sw128<C>(i1, j)) = pack2(v10, v11);
    }
  }
  __syncthreads();
  ost(3);

  {
    const int m0 = 16 * (w & 3), n0 = 64 * (w >> 2);
    float acc[8][4] = {};
    {
      uint32_t af[8][4];
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) { int r, c; frag_a(lane, m0, ks * 16, r, c); ldsm_x4(af[ks], sQs + sw128<C>(r, c)); }
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t bfr[4][4];
#pragma unroll
        for (int np = 0; np < 4; ++np) { int r, c; frag_bt(lane, n0 + 16 * np, ks * 16, r, c); ldsm_x4_t(bfr[np], sS0 + sw128<D>(r, c)); }
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[np][0], bfr[np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[np][2], bfr[np][3]);
        }
      }
    }
    const int i0 = m0 + g, i1 = m0 + g + 8;
    const float l0 = fexp(sm.cl[i0]), l1 = fexp(sm.cl[i1]);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { acc[nt][0] *= l0; acc[nt][1] *= l0; acc[nt][2] *= l1; acc[nt][3] *= l1; }
    {
      uint32_t af[4][4], bfr[4][4][4];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        int r, c;
        frag_a(lane, m0, ks * 16, r, c); ldsm_x4(af[ks], sPs + sw128<C>(r, c));
#pragma unroll
        for (int np = 0; np < 4; ++np) { frag_bt(lane, n0 + 16 * np, ks * 16, r, c); ldsm_x4_t(bfr[ks][np], sDs + sw128<C>(r, c)); }
      }
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          mma16816(acc[2 * np], af[ks], bfr[ks][np][0], bfr[ks][np][1]);
          mma16816(acc[2 * np + 1], af[ks], bfr[ks][np][2], bfr[ks][np][3]);
        }
    }
    if (GDN_DBGON && tid == 0) { unsigned d; asm volatile("mov.b32 %0, %1;" : "=r"(d) : "f"(acc[7][3])); unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4000 + ci * HV + h) * 8 + 4] = t ^ (d & 1); }
    if (L == C) {
      __syncthreads();
      unsigned char* Qs = reinterpret_cast<unsigned char*>(sm.Qs);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        int col = n0 + 8 * nt + 2 * t;
        *reinterpret_cast<uint32_t*>(Qs + sw128<C>(i0, col)) = pack2(P.scale * acc[nt][0], P.scale * acc[nt][1]);
        *reinterpret_cast<uint32_t*>(Qs + sw128<C>(i1, col)) = pack2(P.scale * acc[nt][2], P.scale * acc[nt][3]);
      }
      fence_proxy_async();
      __syncthreads();
      if (tid == 0) {
        tma_store_3d(&P.tmOut, t0, (h * D) >> 6, sm.Qs);
        bulk_commit();
        bulk_wait_read0();
      }
    } else {
      bf16* o0 = P.out + (size_t)(t0 + i0) * HV * D + h * D;
      bf16* o1 = P.out + (size_t)(t0 + i1) * HV * D + h * D;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        int col = n0 + 8 * nt + 2 * t;
        if (i0 < L) *reinterpret_cast<uint32_t*>(o0 + col) = pack2(P.scale * acc[nt][0], P.scale * acc[nt][1]);
        if (i1 < L) *reinterpret_cast<uint32_t*>(o1 + col) = pack2(P.scale * acc[nt][2], P.scale * acc[nt][3]);
      }
    }
  }
  ost(5);
}

// ============================================================================================
// table construction (block 0)
// ============================================================================================
__device__ __forceinline__ void bstamp(const Params& P, int j) { if (GDN_DBGON && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(8190) * 8 + j] = t; P.dbg[(8191) * 8 + j] = (unsigned long long)clock64(); } }
// in-place exclusive scan of arr[0..n) by the whole block (any n); returns the total. Uses wsum[NT/32+1] smem scratch.
__device__ int block_exscan(int* arr, int n, int* wsum) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  int carry = 0;
  for (int base = 0; base < n; base += NT) {
    const int v = (base + tid < n) ? arr[base + tid] : 0;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) wsum[w] = x;
    __syncthreads();
    if (tid == 0) { int p = 0; for (int i = 0; i < NT / 32; ++i) { int c = wsum[i]; wsum[i] = p; p += c; } wsum[NT / 32] = p; }
    __syncthreads();
    if (base + tid < n) arr[base + tid] = x - v + wsum[w] + carry;
    carry += wsum[NT / 32];
    __syncthreads();
  }
  return carry;
}
__device__ void build_table(const Params& P, const Tbl& tb, int* sc) {
  const int tid = threadIdx.x, N = P.N;
  bstamp(P, 0);
  // smem scratch layout
  int* slen = sc; int* snc = slen + N; int* sc0 = snc + N; int* stb = sc0 + N; int* rank = stb + N; int* order = rank + N;
  int* baseA = order + N;      // [N+1]  part-A start per rank (prep items + in-place chain items)
  int* PB = baseA + N + 1;     // [maxnc+1]
  int* hdr = PB + 260;         // [0]=nchunk [1]=nitems [2]=maxnc [3]=offC (deferred chains) [4]=offB (out items)
  int* cbase = hdr + 8;        // [N] deferred chain item base per rank (or -1)
  int* wsum = cbase + N;       // [NT/32+1] scan scratch
  int* tmpA = wsum + NT / 32 + 1;  // [N]
  for (int i = tid; i < N; i += NT) {
    int len = (int)(P.cu[i + 1] - P.cu[i]);
    slen[i] = len; snc[i] = (len + C - 1) / C; sc0[i] = snc[i]; stb[i] = len;
  }
  __syncthreads();
  for (int i = tid; i < N; i += NT) {
    int li = slen[i], r = 0;
    for (int j = 0; j < N; ++j) { int lj = slen[j]; r += (lj > li) || (lj == li && j < i); }
    rank[i] = r; order[r] = i;
  }
  __syncthreads();
  bstamp(P, 1);
  {
    const int nchunk = block_exscan(sc0, N, wsum);   // sc0 = exclusive scan of snc
    block_exscan(stb, N, wsum);                      // stb = exclusive scan of slen
    // maxnc: block max reduction (order[0] is the longest sequence)
    for (int r = tid; r < N; r += NT) {
      const int sq = order[r], nc = snc[sq];
      baseA[r] = nc * HP + (nc >= CHAIN_INPLACE_NC ? HV * NSPLIT_LONG : 0);
      tmpA[r] = (nc >= CHAIN_INPLACE_NC) ? 0 : HV * NSPLIT_SHORT;
    }
    __syncthreads();
    const int offC = block_exscan(baseA, N, wsum);
    const int sizeC = block_exscan(tmpA, N, wsum);
    for (int r = tid; r < N; r += NT) { const int nc = snc[order[r]]; cbase[r] = (nc >= CHAIN_INPLACE_NC) ? -1 : offC + tmpA[r]; }
    if (tid == 0) {
      const int maxnc = snc[order[0]];
      baseA[N] = offC; hdr[0] = nchunk; hdr[2] = maxnc; hdr[3] = offC; hdr[4] = offC + sizeC; hdr[1] = offC + sizeC + nchunk * HV;
    }
    __syncthreads();
    const int maxnc = hdr[2];
    for (int k = tid; k < maxnc; k += NT) { int cnt = 0; for (int sq = 0; sq < N; ++sq) cnt += snc[sq] > k; PB[k] = cnt; }
    __syncthreads();
    const int totB = block_exscan(PB, maxnc, wsum);
    if (tid == 0) PB[maxnc] = totB;
  }
  __syncthreads();
  bstamp(P, 2);
  const int nchunk = hdr[0], offB = hdr[4];
  for (int i = tid; i < N; i += NT) { tb.slen[i] = slen[i]; tb.snc[i] = snc[i]; tb.sc0[i] = sc0[i]; tb.stb[i] = stb[i]; tb.rank[i] = rank[i]; tb.order[i] = order[i]; }
  if (tid < 3) tb.hdr[tid] = hdr[tid];
  for (int ci = tid; ci < nchunk; ci += NT) {
    int sq = 0;
    while (sq + 1 < N && sc0[sq + 1] <= ci) ++sq;
    const int k = ci - sc0[sq], t0 = stb[sq] + k * C, L = min(C, slen[sq] - k * C);
    const int r = rank[sq], nc = snc[sq];
    tb.cseq[ci] = sq | (k << 8) | ((nc >= CHAIN_INPLACE_NC) << 20); tb.ct0[ci] = t0; tb.cL[ci] = L;
    const bool inplace = nc >= CHAIN_INPLACE_NC;
    const int ki = inplace ? min(nc, KI) : nc;
    const int posA = baseA[r] + (k < ki ? k * HP : ki * HP + HV * NSPLIT_LONG + (k - ki) * HP);
    int before = 0;
    for (int rr = 0; rr < r; ++rr) before += snc[order[rr]] > k;
    const int posB = offB + (PB[k] + before) * HV;
    for (int h = 0; h < HV; ++h) tb.items[posB + h] = (2 << 28) | (ci * HV + h);
    for (int h = 0; h < HP; ++h) tb.items[posA + h] = (0 << 28) | (ci * HP + h);
  }
  for (int idx = tid; idx < N * HV * NSPLIT; idx += NT) {
    const int sq = idx / (HV * NSPLIT), rem = idx % (HV * NSPLIT), hh = rem / NSPLIT, sl = rem % NSPLIT;
    const int r = rank[sq], nc = snc[sq];
    const bool inplace = nc >= CHAIN_INPLACE_NC;
    if (sl >= (inplace ? NSPLIT_LONG : NSPLIT_SHORT)) continue;
    const int pos = inplace ? baseA[r] + min(nc, KI) * HP + rem : cbase[r] + hh * NSPLIT_SHORT + sl;
    tb.items[pos] = (1 << 28) | idx;
  }
  __syncthreads();
  bstamp(P, 3);
  __threadfence();
  __syncthreads();
  bstamp(P, 4);
  if (tid == 0) st_release(P.flags, P.epoch);
}

// fast table build for N <= 32 and maxnc <= 128, executed by EVERY block: all blocks derive the full queue layout (so nobody waits for the table
// before its first item), but only block 0 writes the global table (concurrent identical writes from all blocks contend in L2). Other blocks write
// just the chunk / sequence info their own first item reads and consult the table-ready flag once before their first pop (tbl_ready).
// This code runs once per launch and is latency-bound, so the work is spread over the 8 warps with short dependency chains.
// Queue: [long-seq preps, chunk-major over long seqs, k < KI (k < PREP1_K as 1-head items)][long chains][long preps k >= KI, chunk-major][short preps, rank/chunk order]
//        [short chains][long outs k < KA][long outs KA <= k < KB][short outs, chunk-major][long outs k >= KB]
// Returns nitems; the block's own first item code (position blockIdx.x) is written to ctl[2] while the queue is filled.
__device__ int build_table_fast(const Params& P, const Tbl& tb, int* sc, int* ctl) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, N = P.N;
  const bool blk0 = blockIdx.x == 0, writer = blk0;
  auto bst1 = [&](int j) { if (GDN_DBGON && blockIdx.x == 1 && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[8188 * 8 + j] = t; } };
  if (blk0) bstamp(P, 0); bst1(0);
  int* s_sc0 = sc; int* s_stb = s_sc0 + 32; int* s_snc = s_stb + 32; int* s_slen = s_snc + 32; int* s_rank = s_slen + 32; int* s_order = s_rank + 32;
  int* s_baseA = s_order + 32; int* s_cbase = s_baseA + 32; int* s_PB = s_cbase + 32; int* s_hdr = s_PB + 128;
  int* s_PBL = s_hdr + 16;   // [129]: #long-seq chunks with index < k
  int* s_ncL = s_PBL + 132;  // [32]: long nc by rank (0 for short / inactive)
  int* s_ncR = s_ncL + 32;   // [32]: nc by rank (0 for inactive)
  int4* s_part = reinterpret_cast<int4*>(s_ncR + 32);  // [8][32] partial (rank, sc0, stb, nchunk) of sequence `lane` over sequences [4w, 4w+4)
  // ---- phase A (all warps): per-sequence lengths and partial rank / prefix sums
  const bool act = lane < N;
  long long cuj0 = 0, cuj1 = 0;
  if (act) { cuj0 = P.cu[lane]; cuj1 = P.cu[lane + 1]; }
  long long cui[5];
#pragma unroll
  for (int e = 0; e < 5; ++e) cui[e] = (4 * w + e <= N) ? P.cu[4 * w + e] : 0;
  const int len = act ? (int)(cuj1 - cuj0) : 0, nc = act ? (len + C - 1) / C : 0;
  int rp = 0, sc0p = 0, stbp = 0, nchp = 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int i = 4 * w + e;
    const bool acti = i < N;
    const int li = acti ? (int)(cui[e + 1] - cui[e]) : 0, ni = acti ? (li + C - 1) / C : 0;
    rp += (acti && act && ((li > len) || (li == len && i < lane))) ? 1 : 0;  // rank: longer first, ties by index
    sc0p += (i < lane) ? ni : 0; stbp += (i < lane) ? li : 0; nchp += ni;
  }
  s_part[w * 32 + lane] = make_int4(rp, sc0p, stbp, nchp);
  if (w == 0) { s_slen[lane] = act ? len : -1; s_snc[lane] = nc; }
  __syncthreads();
  if (blk0) bstamp(P, 1); bst1(1);
  // ---- phase B (warp 0): reduce the partials, rank order, scans over ranks
  if (w == 0) {
    int4 acc = s_part[lane];
#pragma unroll
    for (int p = 1; p < 8; ++p) { const int4 v = s_part[p * 32 + lane]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    const int r = acc.x, sc0 = acc.y, stb = acc.z, nchunk = acc.w;
    if (act) s_order[r] = lane;
    __syncwarp();
    const int sq_r = act ? s_order[lane] : 0;  // sequence of rank `lane`
    const int nc_r = act ? s_snc[sq_r] : 0;
    const bool longr = act && nc_r >= CHAIN_INPLACE_NC;
    const int nL = __popc(__ballot_sync(0xffffffffu, longr));  // long sequences are ranks 0..nL-1
    const int ncL = longr ? nc_r : 0;
    const int bsz = longr ? 0 : nc_r * HP;                       // short prep item count (rank-indexed)
    const int tsz = longr ? 0 : (act ? HV * NSPLIT_SHORT : 0);   // short chain item count
    int bin = bsz, tin = tsz;                                    // inclusive scans over ranks
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int bb = __shfl_up_sync(0xffffffffu, bin, o), tt = __shfl_up_sync(0xffffffffu, tin, o);
      if (lane >= o) { bin += bb; tin += tt; }
    }
    const int sumS = __shfl_sync(0xffffffffu, bin, 31), sizeC = __shfl_sync(0xffffffffu, tin, 31);
    const int sumLong = __reduce_add_sync(0xffffffffu, ncL);
    const int maxnc = __shfl_sync(0xffffffffu, nc_r, 0);
    s_ncL[lane] = ncL; s_ncR[lane] = nc_r;
    s_sc0[lane] = act ? sc0 : 0x7fffffff;  // sentinel: never <= a chunk index
    s_baseA[lane] = longr ? 0 : bin - bsz; s_cbase[lane] = longr ? -1 : tin - tsz;  // rank-indexed exclusive prefixes (offShort / offC added at use)
    if (act) {
      s_stb[lane] = stb; s_rank[lane] = r;
      if (writer) { tb.slen[lane] = len; tb.snc[lane] = nc; tb.sc0[lane] = sc0; tb.stb[lane] = stb; tb.rank[lane] = r; tb.order[lane] = sq_r; }
    }
    if (lane == 0) { s_hdr[0] = nchunk; s_hdr[2] = maxnc; s_hdr[12] = sumS; s_hdr[13] = sizeC; s_hdr[14] = nL; s_hdr[15] = sumLong; }
  }
  __syncthreads();
  if (blk0) bstamp(P, 2); bst1(2);
  // ---- phase C (thread k): s_PB[k] = #chunks with index < k over all sequences = sum_j min(nc_j, k) ; s_PBL: long sequences only
  const int maxnc = s_hdr[2];
  if (tid <= maxnc) {
    const int k = tid;
    int4 R4[8], L4[8];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) { R4[jj] = reinterpret_cast<const int4*>(s_ncR)[jj]; L4[jj] = reinterpret_cast<const int4*>(s_ncL)[jj]; }
    const int* Rp = reinterpret_cast<const int*>(R4); const int* Lp = reinterpret_cast<const int*>(L4);
    int ex0 = 0, ex1 = 0, xl0 = 0, xl1 = 0;
#pragma unroll
    for (int j = 0; j < 32; j += 2) { ex0 += min(Rp[j], k); ex1 += min(Rp[j + 1], k); xl0 += min(Lp[j], k); xl1 += min(Lp[j + 1], k); }
    if (k < 128) s_PB[k] = ex0 + ex1;
    s_PBL[k] = xl0 + xl1;
  }
  __syncthreads();
  // ---- queue offsets (uniform, recomputed by every thread from smem)
  const int nchunk = s_hdr[0], sumS = s_hdr[12], sizeC = s_hdr[13], nL = s_hdr[14], sumLong = s_hdr[15];
  const int kx = min(KI, maxnc), k1 = min(PREP1_K, kx);
  const int offCh = HV * s_PBL[k1] + HP * (s_PBL[kx] - s_PBL[k1]);  // long chains
  const int offRest = offCh + nL * HV * NSPLIT_LONG;                 // long preps k >= KI
  const int offShort = offRest + HP * (sumLong - s_PBL[kx]);         // short preps
  const int offC = offShort + sumS;                                  // short chains (no dependencies: filler while the long chains advance)
  const int offO1 = offC + sizeC;                                    // long outs k < KA
  const int KA = maxnc / 2, KB = (3 * maxnc) / 4;
  const int offO2 = offO1 + HV * s_PBL[KA];                          // long outs KA <= k < KB (contiguous with the first band)
  const int offOS = offO2 + HV * (s_PBL[KB] - s_PBL[KA]);            // short outs
  const int offO3 = offOS + HV * (nchunk - sumLong);                 // long outs k >= KB
  const int nitems = offO3 + HV * (sumLong - s_PBL[KB]);
  if (tid == 0 && writer) { tb.hdr[0] = nchunk; tb.hdr[1] = nitems; tb.hdr[2] = maxnc; }
  if (blk0) bstamp(P, 3); bst1(3);
  // ---- phase D (all threads): per-chunk info + queue fill; the block's own first item is captured on the way
  const int myit = blockIdx.x;
  auto put4 = [&](int pos, int4 v) -> bool {
    if (writer) *reinterpret_cast<int4*>(tb.items + pos) = v;
    const int dlt = myit - pos;
    if (dlt >= 0 && dlt < 4) { ctl[2] = dlt == 0 ? v.x : (dlt == 1 ? v.y : (dlt == 2 ? v.z : v.w)); return true; }
    return false;
  };
  for (int ci = tid; ci < nchunk; ci += NT) {
    int sq = -1;
    {
      int4 S4[8];
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) S4[jj] = reinterpret_cast<const int4*>(s_sc0)[jj];
      const int* Sp = reinterpret_cast<const int*>(S4);
#pragma unroll
      for (int j = 0; j < 32; ++j) sq += (Sp[j] <= ci);
    }
    const int k = ci - s_sc0[sq], t0 = s_stb[sq] + k * C, L = min(C, s_slen[sq] - k * C);
    const int r = s_rank[sq], nc = s_snc[sq];
    const bool inplace = nc >= CHAIN_INPLACE_NC;
    bool mine = false;
    // long sequences: chunk-major; the r higher-ranked long sequences all have chunk k (ranks sorted by nc descending)
    const int b0 = ci * HV, a0 = ci * HP;
    if (inplace && k < k1) {  // the first chunks of the long sequences gate the chain start: 1-head prep items (shorter latency)
      const int pa = HV * (s_PBL[k] + r);
      mine |= put4(pa, make_int4((3 << 28) | b0, (3 << 28) | (b0 + 1), (3 << 28) | (b0 + 2), (3 << 28) | (b0 + 3)));
      mine |= put4(pa + 4, make_int4((3 << 28) | (b0 + 4), (3 << 28) | (b0 + 5), (3 << 28) | (b0 + 6), (3 << 28) | (b0 + 7)));
    } else {
      const int posA = inplace ? (k < kx ? HV * s_PBL[k1] + HP * (s_PBL[k] - s_PBL[k1] + r) : offRest + HP * (s_PBL[k] - s_PBL[kx] + r)) : offShort + s_baseA[r] + k * HP;
      mine |= put4(posA, make_int4(a0, a0 + 1, a0 + 2, a0 + 3));
    }
    // outs: long sequences chunk-major in three bands (interleaved with the short chains / short outs so that they are popped roughly when the chain reaches them)
    const int posB = inplace ? (k < KA ? offO1 + HV * (s_PBL[k] + r) : (k < KB ? offO2 + HV * (s_PBL[k] - s_PBL[KA] + r) : offO3 + HV * (s_PBL[k] - s_PBL[KB] + r)))
                             : offOS + HV * (s_PB[k] - s_PBL[k] + r - nL);  // short: chunk-major over the short sequences
    mine |= put4(posB, make_int4((2 << 28) | b0, (2 << 28) | (b0 + 1), (2 << 28) | (b0 + 2), (2 << 28) | (b0 + 3)));
    mine |= put4(posB + 4, make_int4((2 << 28) | (b0 + 4), (2 << 28) | (b0 + 5), (2 << 28) | (b0 + 6), (2 << 28) | (b0 + 7)));
    if (writer || mine) { tb.cseq[ci] = sq | (k << 8) | ((int)inplace << 20); tb.ct0[ci] = t0; tb.cL[ci] = L; }  // packed: one load gives the out item its poll target
  }
  for (int idx = tid; idx < N * HV * NSPLIT; idx += NT) {
    const int sq = idx / (HV * NSPLIT), rem = idx % (HV * NSPLIT), hh = rem / NSPLIT, sl = rem % NSPLIT;
    const int r = s_rank[sq], nc = s_snc[sq];
    const bool inplace = nc >= CHAIN_INPLACE_NC;
    if (sl >= (inplace ? NSPLIT_LONG : NSPLIT_SHORT)) continue;
    const int pos = inplace ? offCh + r * HV * NSPLIT_LONG + rem : offC + s_cbase[r] + hh * NSPLIT_SHORT + sl;
    if (writer) tb.items[pos] = (1 << 28) | idx;
    if (pos == myit) {
      ctl[2] = (1 << 28) | idx;
      if (!writer) { tb.sc0[sq] = s_sc0[sq]; tb.snc[sq] = nc; tb.stb[sq] = s_stb[sq]; }  // what the chain item reads
    }
  }
  if (tid == 0) ctl[6] = writer ? 1 : 0;  // table-ready seen (writer: its own writes are visible after the barrier below)
  __syncthreads();  // this block's own table writes are visible to its threads (CTA barrier)
  if (writer && tid == 0) { __threadfence(); st_release(P.flags, P.epoch); }  // release is cumulative over the CTA barrier
  if (blk0) bstamp(P, 4); bst1(4);
  return nitems;
}

// ============================================================================================
// the persistent kernel
// ============================================================================================
__global__ void __launch_bounds__(NT, MINB) k_mega(const __grid_constant__ Params P) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* base = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);  // keeps the shared address space visible to the compiler
  int* ctl = reinterpret_cast<int*>(base + UNION_BYTES);  // [0] item, [1..] scratch for table build
  const Tbl tb(P.tbl, P.ncm, P.N);
  const int tid = threadIdx.x;
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(ctl) + 1;  // TMEM base address (ctl[1] is otherwise unused)
  if (tid < 32) { tmem_alloc(s_tmem, TMEM_COLS); tc_fence_before(); }
  if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4096 + blockIdx.x) * 4 + 0] = t; }
  int nitems;
  if (P.N <= 32 && P.T <= 128 * C) {  // every block builds the (identical) table itself: no cross-block wait
    if (tid == 0) ctl[2] = 0;
    nitems = build_table_fast(P, tb, reinterpret_cast<int*>(base), ctl);
    if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4096 + blockIdx.x) * 4 + 2] = t; }
  } else {
    if (blockIdx.x == 0) {
      build_table(P, tb, reinterpret_cast<int*>(base));  // union region as scratch (plenty)
      if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4096 + blockIdx.x) * 4 + 2] = t; }
    } else {
      if (tid == 0) wait_flag(P.flags, P.epoch);
    }
    __syncthreads();
    nitems = __ldcg(tb.hdr + 1);
    if (tid == 0) { ctl[2] = (blockIdx.x < nitems) ? __ldcg(tb.items + blockIdx.x) : 0; ctl[6] = 1; }
  }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *s_tmem;
  int* counter = P.flags + 1 + (P.epoch & 1023);
  if (tid == 0) { ctl[3] = -1; ctl[4] = 0; ctl[5] = nitems; ctl[0] = blockIdx.x; }  // first item without the atomic storm; later pops are offset by gridDim.x
  __syncthreads();
  for (;;) {
    const int item = ctl[0], code = ctl[2];
    int nxt = nitems;
    // next pop in flight during this item -- but never while running a chain item: it may wait on prep items queued behind it
    if (tid == 0) ctl[4] = 0;  // set to 1 by pop_next inside the item (non-chain items)
    if (item >= nitems) {
      if (tid == 0) flush_pending(P, ctl);
      if (GDN_DBGON && tid == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4096 + blockIdx.x) * 4 + 1] = t; }
      break;
    }
    const int type = code >> 28, idx = code & 0x0fffffff;
    unsigned long long t_beg = 0;
    if (GDN_DBGON && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_beg));
    if (type == 0) dev_prep2(P, tb, *reinterpret_cast<PrepSmem2*>(base), idx / HP, idx % HP);
    else if (type == 3) dev_prep(P, tb, *reinterpret_cast<PrepSmem*>(base), idx / HV, idx % HV);
    else if (type == 1) {
      const int s = idx / (HV * NSPLIT);
      if (__ldcg(tb.snc + s) >= CHAIN_INPLACE_NC) dev_chain16(P, tb, *reinterpret_cast<ChainSmemP*>(base), tmem, s, (idx / NSPLIT) % HV, idx % NSPLIT);
      else dev_chain<32>(P, tb, *reinterpret_cast<ChainSmemT<32>*>(base), tmem, s, (idx / NSPLIT) % HV, idx % NSPLIT);
    }
    else dev_out(P, tb, *reinterpret_cast<OutSmem*>(base), idx / HV, idx % HV);
    __syncthreads();
    if (GDN_DBGON && tid == 0) {
      unsigned long long t_end; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
      P.dbg[item * 4 + 0] = t_beg; P.dbg[item * 4 + 1] = t_end; P.dbg[item * 4 + 2] = (unsigned long long)code; P.dbg[item * 4 + 3] = blockIdx.x;
    }
    if (tid == 0 && ctl[4] == 0) {  // items that did not pop themselves (long chains pop only near their end: they wait on prep items queued behind them)
      nxt = atomicAdd(counter, 1) + (int)gridDim.x; tbl_ready(P, ctl); ctl[0] = nxt; ctl[2] = (nxt < nitems) ? __ldcg(tb.items + nxt) : 0;
    }
    __syncthreads();
  }
  tc_fence_before();
  __syncthreads();
  if (tid < 32) tmem_dealloc(tmem, TMEM_COLS);
}

__global__ void __launch_bounds__(NT, 1) k_tbl(const __grid_constant__ Params P) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* base = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const Tbl tb(P.tbl, P.ncm, P.N);
  if (GDN_DBGON && threadIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.dbg[(4096) * 4 + 0] = t; }
  int* ctl = reinterpret_cast<int*>(base + UNION_BYTES);
  build_table_fast(P, tb, reinterpret_cast<int*>(base), ctl);
}
// ============================================================================================
// host side
// ============================================================================================
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn g_encode = nullptr;

bool make_map(CUtensorMap* tm, const void* ptr, CUtensorMapDataType dt, int esize, uint64_t cols, uint64_t rows, uint64_t pitch_bytes,
              uint32_t box_c, uint32_t box_r, CUtensorMapSwizzle sw) {
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {pitch_bytes};
  cuuint32_t box[2] = {box_c, box_r};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = g_encode(tm, dt, 2, const_cast<void*>(ptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled failed %d (cols=%llu rows=%llu box=%u,%u)\n", (int)r,
                                   (unsigned long long)cols, (unsigned long long)rows, box_c, box_r); return false; }
  return true;
}
// 3D map over a row-major [rows][cols] tensor viewed as [panel][row][inner]: dims {inner, rows, cols/inner}, strides {pitch, inner*esize}
bool make_map3(CUtensorMap* tm, const void* ptr, CUtensorMapDataType dt, int esize, uint64_t cols, uint64_t rows, uint64_t pitch_bytes,
               uint32_t inner, uint32_t box_r, uint32_t npan, CUtensorMapSwizzle sw) {
  cuuint64_t dims[3] = {inner, rows, cols / inner};
  cuuint64_t strides[2] = {pitch_bytes, (cuuint64_t)inner * esize};
  cuuint32_t box[3] = {inner, box_r, npan};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = g_encode(tm, dt, 3, const_cast<void*>(ptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { fprintf(stderr, "cuTensorMapEncodeTiled(3d) failed %d (cols=%llu rows=%llu box=%u,%u,%u)\n", (int)r,
                                   (unsigned long long)cols, (unsigned long long)rows, inner, box_r, npan); return false; }
  return true;
}
int g_blocks = 0;
unsigned long long* g_dbg = nullptr;
int g_dbg_seq = -1;
}  // namespace

extern "C" {
int gdn_tbl_test(const void* cu, int T, int N, void* tbl, void* flags, int epoch, cudaStream_t stream) {
  Params P; memset(&P, 0, sizeof(P));
  P.cu = (const int64_t*)cu; P.tbl = (int*)tbl; P.flags = (int*)flags; P.T = T; P.N = N; P.ncm = (T + C - 1) / C + N; P.epoch = epoch; P.dbg = g_dbg; P.dbg_seq = g_dbg_seq;
  cudaFuncSetAttribute(k_tbl, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  k_tbl<<<1, NT, SMEM_BYTES, stream>>>(P);
  return (int)cudaGetLastError();
}
// sizes (in bytes) the caller must provide: tbl: tbl_ints(ncm,N)*4 ; flags: (1025 + ncm*HV)*4 (zeroed) ; prog: N*HV*NSPLIT*8 (zeroed)
int gdn_ws_sizes(int T, int N, long long* out3) {
  const int ncm = (T + C - 1) / C + N;
  out3[0] = (long long)tbl_ints(ncm, N) * 4; out3[1] = (long long)(1025 + ncm * HV) * 4; out3[2] = (long long)N * HV * NSPLIT * 8 * 8;
  return 0;
}
int gdn_fwd(const void* q, const void* k, const void* v, const void* state, const void* A_log, const void* a,
            const void* dt_bias, const void* b, const void* cu, int T, int N, float scale, void* out, void* new_state,
            void* Wbuf, void* Ubuf, void* clbuf, void* S0buf, void* Dbuf, void* Gbuf, void* tbl, void* flags, void* prog, int epoch,
            cudaStream_t stream) {
  const int ncm = (T + C - 1) / C + N;
  static bool init = false;
  if (!init) {
    cudaDriverEntryPointQueryResult qr;
    void* fn = nullptr;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &qr) != cudaSuccess || fn == nullptr) {
      fprintf(stderr, "cannot get cuTensorMapEncodeTiled\n"); return -2;
    }
    g_encode = (EncodeFn)fn;
    cudaFuncSetAttribute(k_mega, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    int dev = 0, nsm = 0, per = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_mega, NT, SMEM_BYTES);
    if (per < 1) { fprintf(stderr, "k_mega cannot be resident (smem %d)\n", SMEM_BYTES); return -4; }
    g_blocks = nsm * per;
    if (const char* e = getenv("GDN_PER_SM")) g_blocks = nsm * atoi(e);  // override (the occupancy API is conservative for tcgen05 kernels)
    init = true;
  }
  Params P;
  const uint64_t rowsW = (uint64_t)ncm * HV * C, rowsS0 = (uint64_t)ncm * HV * D;
  const auto BF = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const auto SW128 = CU_TENSOR_MAP_SWIZZLE_128B, SW64 = CU_TENSOR_MAP_SWIZZLE_64B;
  bool ok = true;
  ok &= make_map3(&P.tmQ, q, BF, 2, HQ * D, T, HQ * D * 2, 64, 64, 2, SW128);
  ok &= make_map3(&P.tmK, k, BF, 2, HQ * D, T, HQ * D * 2, 64, 64, 2, SW128);
  ok &= make_map3(&P.tmV, v, BF, 2, HV * D, T, HV * D * 2, 64, 64, 2, SW128);
  ok &= make_map3(&P.tmW, Wbuf, BF, 2, D, rowsW, D * 2, 64, 64, 2, SW128);
  ok &= make_map(&P.tmW2, Wbuf, BF, 2, D, rowsW, D * 2, 64, 64, SW128);
  ok &= make_map(&P.tmK2, k, BF, 2, HQ * D, T, HQ * D * 2, 64, 64, SW128);
  ok &= make_map3(&P.tmU, Ubuf, BF, 2, D, rowsW, D * 2, 64, 64, 2, SW128);
  ok &= make_map(&P.tmU32, Ubuf, BF, 2, D, rowsW, D * 2, 32, 64, SW64);
  ok &= make_map3(&P.tmD, Dbuf, BF, 2, D, rowsW, D * 2, 64, 64, 2, SW128);
  ok &= make_map(&P.tmD32, Dbuf, BF, 2, D, rowsW, D * 2, 32, 64, SW64);
  ok &= make_map3(&P.tmS0, S0buf, BF, 2, D, rowsS0, D * 2, 64, 128, 2, SW128);
  ok &= make_map(&P.tmS032, S0buf, BF, 2, D, rowsS0, D * 2, 32, 128, SW64);
  ok &= make_map3(&P.tmOut, out, BF, 2, HV * D, T, HV * D * 2, 64, 64, 2, SW128);
  ok &= make_map3(&P.tmState, state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, D, (uint64_t)N * HV * D, D * 4, 32, 32, 4, SW128);
  ok &= make_map3(&P.tmNewState, new_state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, D, (uint64_t)N * HV * D, D * 4, 32, 32, 4, SW128);
  const auto SW32 = CU_TENSOR_MAP_SWIZZLE_32B;
  ok &= make_map(&P.tmU16, Ubuf, BF, 2, D, rowsW, D * 2, 16, 64, SW32);
  ok &= make_map(&P.tmD16, Dbuf, BF, 2, D, rowsW, D * 2, 16, 64, SW32);
  ok &= make_map(&P.tmS016, S0buf, BF, 2, D, rowsS0, D * 2, 16, 128, SW32);
  ok &= make_map3(&P.tmState16, state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, D, (uint64_t)N * HV * D, D * 4, 32, 16, 4, SW128);
  ok &= make_map3(&P.tmNewState16, new_state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, D, (uint64_t)N * HV * D, D * 4, 32, 16, 4, SW128);
  ok &= make_map(&P.tmG, Gbuf, BF, 2, C, rowsW, C * 2, 64, 64, SW128);
  if (!ok) return -3;
  P.a = (const bf16*)a; P.b = (const bf16*)b; P.A_log = (const float*)A_log; P.dt_bias = (const float*)dt_bias;
  P.cu = (const int64_t*)cu; P.clbuf = (float*)clbuf; P.out = (bf16*)out; P.Dg = (bf16*)Dbuf; P.S0g = (bf16*)S0buf; P.nsg = (float*)new_state;
  P.tbl = (int*)tbl; P.flags = (int*)flags; P.prog = (unsigned long long*)prog;
  P.T = T; P.N = N; P.ncm = ncm; P.epoch = epoch; P.scale = scale; P.dbg = g_dbg; P.dbg_seq = g_dbg_seq;
  const int nitems_max = 2 * ncm * HV + N * HV * NSPLIT;
  const int grid = g_blocks < nitems_max ? g_blocks : nitems_max;
  k_mega<<<grid, NT, SMEM_BYTES, stream>>>(P);
  return (int)cudaGetLastError();
}
int gdn_set_dbg(void* p, int seq) { g_dbg = (unsigned long long*)p; g_dbg_seq = seq; return 0; }
int gdn_info(int* out) {
  static_assert(sizeof(PrepSmem2) <= UNION_BYTES, "PrepSmem2 too large");
  out[0] = SMEM_BYTES; out[1] = g_blocks; out[2] = (int)sizeof(ChainSmemP); out[3] = (int)sizeof(PrepSmem2); out[4] = (int)sizeof(OutSmem);
  cudaFuncAttributes at; cudaFuncGetAttributes(&at, k_mega);
  out[5] = at.numRegs; out[6] = (int)at.sharedSizeBytes; out[7] = (int)at.localSizeBytes; out[8] = at.maxThreadsPerBlock; out[9] = at.maxDynamicSharedSizeBytes;
  int per = -1; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_mega, NT, SMEM_BYTES); out[10] = per;
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0); out[11] = (int)pr.sharedMemPerMultiprocessor; out[12] = (int)pr.sharedMemPerBlockOptin; out[13] = (int)pr.reservedSharedMemPerBlock;
  return 0; }
}
