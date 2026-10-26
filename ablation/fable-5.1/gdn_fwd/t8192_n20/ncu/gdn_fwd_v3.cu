// GDN (gated delta net) varlen prefill forward, from scratch, for NVIDIA B200 (sm_100a).
// qk heads = 4, v heads = 8, head_dim = 128, bf16 inputs, fp32 k-last state.
//
// Math (per v-head; k-head = vhead/2):
//   S_t = g_t S_{t-1} + beta_t k_t^T (v_t - g_t k_t S_{t-1}),   o_t = scale * q_t S_t     (S is [K,V])
// Chunked WY form (chunk C = 64 tokens, Gam = cumsum(log g) inside the chunk):
//   A[t][i] = beta_t exp(Gam_t - Gam_i) (k_t.k_i)  (i<t),   T = (I + A)^-1  (unit lower triangular)
//   Wt = T diag(beta exp(Gam)) K,   Ut = T diag(beta) V                   (64 x 128 each)
//   per chunk with entry state S0:   Ub = Ut - Wt S0
//        S_end = exp(Gam_C) S0 + (K * exp(Gam_C - Gam_t))^T Ub
//        O     = scale exp(Gam_t) Q S0 + P Ub,   P[t][i] = scale exp(Gam_t - Gam_i) (q_t.k_i) (i<=t)
// Kernel A ("prologue") computes A, T (exact fp32), Wt, Ut, P and the gate vectors for every (chunk, vhead).
// Kernel B ("recurrence") walks the chunks of one sequence for one (vhead, 32-wide V block) keeping the
// fp32 state in registers; the two chained GEMMs per chunk run on tcgen05 (TMEM accumulators, TMA loads).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <type_traits>

#define DEVI __device__ __forceinline__
#define PRO_SMEM (65536 + 2 * 17408 + 9216 + 1024)
#define AS 68   // padded row stride (floats) of the 64x64 fp32 A/T matrices
#define YS 36   // padded row stride of the 32x32 scratch
typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------
// small PTX helpers
// ------------------------------------------------------------------------------------------------
DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

DEVI uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);  // .x = lo (low 16 bits), .y = hi
  return *reinterpret_cast<uint32_t*>(&v);
}
DEVI float bf16lo(uint32_t u) { return __bfloat162float(__ushort_as_bfloat16((unsigned short)(u & 0xffff))); }
DEVI float bf16hi(uint32_t u) { return __bfloat162float(__ushort_as_bfloat16((unsigned short)(u >> 16))); }

DEVI void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
DEVI void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

DEVI void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D(16x8,f32) += A(16x16,bf16) * B(16x8,bf16)
DEVI void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mbarrier
DEVI void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count)); }
DEVI void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
DEVI bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
#ifdef WATCHDOG
__device__ int* g_wdp;   // pinned host memory (readable after a sticky error)
DEVI void wd_trap(int kind, int info) {
  volatile int* w = g_wdp;
  if (atomicCAS((int*)w + 3, 0, 1) == 0) { w[0] = blockIdx.x; w[1] = threadIdx.x; w[2] = info; w[4] = kind; __threadfence_system(); }
  __trap();
}
#ifdef WD_LITE   // progress words only (timing close to the plain build); no spin counters / traps
#define WD_SPIN(cond, kind, info) { while (cond) {} }
#else
#define WD_SPIN(cond, kind, info) { long long _n = 0; while (cond) { if (++_n > 400000000LL) wd_trap(kind, info); } }
#endif
#define WDP(k, v) { ((volatile int*)g_wdp)[1024 + blockIdx.x * 16 + (k)] = (v); }
static int* h_wd = nullptr;
static cudaStream_t g_wd_stream = 0; static int g_wd_nsms = 0, g_wd_epoch = 0;
// WATCHDOG, GDN_WD_SYNC=0 mode: after a batch of back-to-back launches, wait up to timeout_s for the stream and dump the
// per-CTA progress words of the hung launch (no host memset between launches, so untouched words may be stale).
extern "C" int gdn_wd_check(double timeout_s) {
  auto t0 = std::chrono::steady_clock::now(); bool done = false;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < timeout_s) { if (cudaStreamQuery(g_wd_stream) == cudaSuccess) { done = true; break; } }
  if (done) return 0;
  fprintf(stderr, "WATCHDOG(nosync): kernel hung. last epoch %d. CTAs not done:\n", g_wd_epoch);
  h_wd[512] = 1;   // ask the stuck CTAs' flag warps to dump their barrier state
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  static const char* bn[18] = {"K0","K1","K2","K3","W0","W1","W2","W3","Q0","Q1","U0","U1","U2","U3","Chain","Epi","Mneg","KU"};
  for (int b = 0; b < g_wd_nsms; b++) { const int* w = h_wd + 1024 + b * 16; if (w[0] != 3) {
    fprintf(stderr, "  bid %3d phase %d | p1 items lane0 %4d lane1 %4d pstage %2d | rec item %4d  Mci %3d  Lci %3d  Sci %3d | A %d.%d  B %d.%d  fa %d qn %d prog %d %d  M %d.%d  S %d.%d\n",
            b, w[0], w[1], w[5], w[3], w[2], w[4], w[6], w[7], w[8] >> 3, w[8] & 7, w[9] >> 3, w[9] & 7, w[10], w[11], w[12], w[13], w[14] >> 2, w[14] & 3, w[15] >> 2, w[15] & 3);
    const int* d = h_wd + 16384 + b * 64;
    if (d[0] == 1) { fprintf(stderr, "    dump: prog %d %d sfa %d skw %d\n", d[37], d[38], d[39], d[40]);
      for (int i = 0; i < 18; i++) fprintf(stderr, "    %-5s raw %08x %08x  par0done %d\n", bn[i], d[1 + 2 * i], d[2 + 2 * i], d[41 + i]); }
    else fprintf(stderr, "    (no dump)\n"); } }
  h_wd[512] = 0;
  fprintf(stderr, "  trap info: bid %d tid %d info %d kind %d\n", h_wd[0], h_wd[1], h_wd[2], h_wd[4]);
  return 1;
}
#ifdef WD_LITE
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) {} }
#else
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { WD_SPIN(!mbar_try_wait(bar, parity), 1, (int)bar); }
#endif
#else
#define WD_SPIN(cond, kind, info) { while (cond) {} }
#define WDP(k, v)
DEVI void mbar_wait(uint32_t bar, uint32_t parity) { while (!mbar_try_wait(bar, parity)) {} }
#endif

// TMA
DEVI void tma_load_2d(uint32_t dst, const void* tmap, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DEVI void tma_load_3d(uint32_t dst, const void* tmap, int c0, int c1, int c2, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               ::"r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// tcgen05
DEVI void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols) : "memory");
}
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tc_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}
DEVI void tc_mma(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t accum) {
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum) : "memory");
}
DEVI void tc_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// shared memory matrix descriptor (sm100 UMMA)
// swz: 0 none, 2 = 128B, 4 = 64B, 6 = 32B
DEVI void tc_ld32_nw(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
DEVI void tc_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)swz << 61;
  return d;
}
// instruction descriptor: kind::f16, bf16 x bf16 -> f32
DEVI uint32_t make_idesc(uint32_t M, uint32_t N, bool a_mn_major, bool b_mn_major) {
  uint32_t d = 0;
  d |= 1u << 4;             // D = f32
  d |= 1u << 7;             // A = bf16
  d |= 1u << 10;            // B = bf16
  d |= (a_mn_major ? 1u : 0u) << 15;
  d |= (b_mn_major ? 1u : 0u) << 16;
  d |= (N >> 3) << 17;
  d |= (M >> 4) << 24;
  return d;
}

DEVI void bar_named_sync(uint32_t id, uint32_t n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEVI void bar_named_arrive(uint32_t id, uint32_t n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }

DEVI void bulk_load_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
DEVI void bulk_store_1d(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
DEVI void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
DEVI void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
DEVI void mbar_inval(uint32_t bar) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bar) : "memory"); }
DEVI int ld_relaxed_gpu(const int* p) { int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEVI void st_release_gpu(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEVI void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
DEVI void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEVI uint4 ldcg16(const void* p) { return __ldcg((const uint4*)p); }

// tile [rows][128 B] with 128B swizzle (16B chunk index ^= row & 7) -- the layout TMA SWIZZLE_128B produces and
// the layout the UMMA SW128 descriptors expect (base must be 1024B aligned)
DEVI uint32_t sw128(int row, int chunk16) { return row * 128 + ((chunk16 ^ (row & 7)) << 4); }

// ------------------------------------------------------------------------------------------------
// Blob produced by the prologue for every (chunk, vhead): 64 KB, exactly the recurrence smem layouts.
//   [0     , 16K): gdWt  = bf16(gd_t * Wt[t])         rows t, dk halves at +8192  (sw128 rows)
//   [16K   , 32K): gdUt  = bf16(gd_t * Ut[t])         [vhalf][t][64 dv]  (8 KB per v half)
//   [32K   , 48K): Qhat  = bf16(ge_t Q[t] - (P Wt)[t]) rows t, dk halves at +8192
//   [48K   , 64K): Ohat  = bf16((P Ut)[t])            [vhalf][t][64 dv]
// ------------------------------------------------------------------------------------------------
#define BLOB_BYTES 65536
#define BL_W 0
#define BL_U 16384
#define BL_Q 32768
#define BL_O 49152

DEVI float softplus_f(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#define LOG2E 1.4426950408889634f

// all per-launch tables travel as kernel parameters (constant bank; uniform loads): no HtoD copies
#define MAXC 640
#define MAX_SEQ 255
struct GdnTabs {
  int seq[MAXC];               // [nc] sequence of each chunk
  int t0[MAXC];                // [nc] first token (global index)
  int len[MAXC];               // [nc] valid tokens (1..64)
  int order[MAX_SEQ];          // [N] sequence ids sorted by length desc
  int cu[MAX_SEQ + 1];         // [N+1] token offsets
  int cstart[MAX_SEQ + 1];     // [N+1] first chunk index of each sequence
  int border[MAXC];            // [nc] prologue work order (chunk ids); item i -> chunk border[i>>3], head i&7
  int nA;                      // fused: number of statically assigned recurrence CTAs (K longest seqs x 16)
  int nrec;                    // fused: N*16 recurrence items
  int nitems;                  // fused: nc*8 prologue items
  int nsolo;                   // fused: leading prologue items run alone on a CTA (one lane) for minimum latency
  int pool0;                   // fused: first dynamically claimed prologue item
  int epoch;                   // fused: flag value marking "blob ready" for this launch
};

// ------------------------------------------------------------------------------------------------
// Kernel A: prologue.  grid = (NC, 8 v-heads), 128 threads, 2 CTAs / SM.
// smem (bytes):
#define P_KQ   0        // [dk half 0: K rows 0-63 | Q rows 64-127][dk half 1: K | Q]   32 KB  (A operand [K;Q], M=128)
#define P_V    32768    // V [dv half 0: 64 rows][dv half 1]                          16 KB
#define P_A    49152    // fp32 A / T, 64 x AS                                        17 KB
#define P_P    66560    // P bf16 [64][64] sw128 (as an M=128 A operand its rows 64-127 read T')  8 KB
#define P_TA   74752    // [T' rows 0-63 ; (P T') rows 64-127] bf16 sw128             16 KB
#define P_TB   91136    // [T''        ; (P T'')]                                     16 KB
#define P_TM   (P_TA + 8192)   // bf16 mirror of T for the mma.sync merges (dead before P T' is written)
#define P_G    107520   // gates: G2[64] (Gam*log2e), beta[64], gd[64], ge[64], bg[64] 2 KB
#define P_BAR  109568   // mbarriers + tmem slot
#define P_SMEM (P_BAR + 1024)


// one (chunk, v-head) prologue item, executed by 128 threads (tid 0..127) that share the smem window `smem`
// (P_SMEM bytes) and 256 TMEM columns at `tmem`; BARID = named barrier used for the 128-thread syncs.
template <int BARID>
DEVI void prologue_item(unsigned char* smem, const uint32_t tmem, const int c, const int h, const int tid,
                        const CUtensorMap* tmQ, const CUtensorMap* tmK, const CUtensorMap* tmV,
                        const bf16* __restrict__ a, const bf16* __restrict__ b,
                        const float* __restrict__ A_log, const float* __restrict__ dt_bias, const float scale,
                        const GdnTabs& tabs, unsigned char* __restrict__ blob, float* __restrict__ gC,
                        long long* ptacc, const bool reinit = false, int* prev_flag = nullptr, const int epoch = 0)
{
#ifdef TIMING
  long long ptlast = clock64();
#define PSTAMP(i) { if (tid == 0) { long long tn = clock64(); ptacc[i] += tn - ptlast; ptlast = tn; } }
#elif defined(WATCHDOG) && !defined(WD_LITE)
#define PSTAMP(i) { if (tid == 0) WDP(3, (i)); }
#else
#define PSTAMP(i)
#endif
#define PSYNC() bar_named_sync(BARID, 128)
  const int kh = h >> 1;
  const int warp = tid >> 5, lane = tid & 31;
  const int t0 = tabs.t0[c], L = tabs.len[c];
  const uint32_t sb = smem_u32(smem);
  float* sG2 = (float*)(smem + P_G);    // Gam_t * log2(e)  (inclusive cumsum of log g)
  float* sBeta = sG2 + 64;
  float* sGd = sG2 + 128;               // exp(GamC - Gam_t)
  float* sGe = sG2 + 192;               // scale * exp(Gam_t)
  float* sBg = sG2 + 256;               // beta_t * exp(Gam_t)
  const uint32_t bar_ld = sb + P_BAR, bar_m0 = sb + P_BAR + 8, bar_m1 = sb + P_BAR + 16, bar_m2 = sb + P_BAR + 24, bar_v = sb + P_BAR + 32;
  // gate inputs first: their latency overlaps the barrier setup and the TMA loads
  float x_a = 0.f, x_b = 0.f;
  if (tid < L) { const size_t idx = (size_t)(t0 + tid) * 8 + h; x_a = __bfloat162float(a[idx]); x_b = __bfloat162float(b[idx]); }
  const float Alog = A_log[h], dtb = dt_bias[h];

  if (reinit) PSYNC();   // all threads finished the previous item (no one still waits on the barriers below)
  if (tid == 0) {
    if (reinit) { mbar_inval(bar_ld); mbar_inval(bar_m0); mbar_inval(bar_m1); mbar_inval(bar_m2); mbar_inval(bar_v); }
    mbar_init(bar_ld, 1); mbar_init(bar_m0, 1); mbar_init(bar_m1, 1); mbar_init(bar_m2, 1); mbar_init(bar_v, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  PSYNC();
  PSTAMP(12);
  if (tid == 0) {
    mbar_expect_tx(bar_ld, 32768);
    tma_load_3d(sb + P_KQ + 0, tmK, 0, kh, t0, bar_ld);
    tma_load_3d(sb + P_KQ + 8192, tmQ, 0, kh, t0, bar_ld);
    tma_load_3d(sb + P_KQ + 16384, tmK, 64, kh, t0, bar_ld);
    tma_load_3d(sb + P_KQ + 24576, tmQ, 64, kh, t0, bar_ld);
  }
  PSTAMP(13);
  // ---- gates (token t = tid < 64): Gam = inclusive cumsum(log g), beta; padded tokens: log g = 0, beta = 0 ----
  if (tid < 64) {
    const int t = tid;
    float lg = 0.f, be = 0.f;
    if (t < L) {
      lg = -__expf(Alog) * softplus_f(x_a + dtb);
      be = 1.f / (1.f + __expf(-x_b));
    }
    PSTAMP(15);
    float incl = lg;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float n = __shfl_up_sync(0xffffffff, incl, o);
      if (lane >= o) incl += n;
    }
    sG2[t] = incl * LOG2E; sBeta[t] = be;
  }
  PSYNC();
  PSTAMP(16);
  if (tid < 64) {
    const int t = tid;
    float G = sG2[t] + (t >= 32 ? sG2[31] : 0.f);
    __syncwarp();
    sG2[t] = G;
  }
  PSYNC();
  if (tid < 64) {
    const int t = tid;
    const float G2C = sG2[63], G2t = sG2[t];
    const float eg = ex2f(G2t);
    sGd[t] = ex2f(G2C - G2t);
    sGe[t] = scale * eg;
    sBg[t] = sBeta[t] * eg;
    if (t == 0) gC[(size_t)c * 8 + h] = ex2f(G2C);
  }
  PSYNC();
  const uint32_t D0 = tmem;          // [K;Q] K^T : 128 x 64
  const uint32_t D1 = tmem + 128;    // P [T'|T''] : 128 x 128 (rows 0-63 valid)
  const uint32_t D2 = tmem + 128;    // [T';PT'] K  : 128 x 128
  const uint32_t D3 = tmem;          // [T'';PT''] V : 128 x 128
  PSTAMP(0);
  // ---- GEMM0: D0 = [K;Q] (K-major, 2 k-blocks of 16 KB) x K^T (B = K rows, K-major) ----
  mbar_wait(bar_ld, 0);
  PSTAMP(1);
  if (tid == 0) {
    tc_fence_after();
    const uint32_t idesc = make_idesc(128, 64, false, false);
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      uint32_t off = (ks >> 2) * 16384 + (ks & 3) * 32;
      tc_mma(D0, make_desc(sb + P_KQ + off, 16, 1024, 2), make_desc(sb + P_KQ + off, 16, 1024, 2), idesc, ks > 0 ? 1u : 0u);
    }
    tc_commit(bar_m0);
  }
  mbar_wait(bar_m0, 0);
  tc_fence_after();
  if (lane == 0) { bulk_wait0(); fence_proxy_async_global(); }   // previous item's stores are complete (staging free)
  PSYNC();
  if (tid == 0) {
    if (prev_flag) st_release_gpu(prev_flag, epoch);
    mbar_expect_tx(bar_v, 16384);
    tma_load_3d(sb + P_V + 0, tmV, 0, h, t0, bar_v);
    tma_load_3d(sb + P_V + 8192, tmV, 64, h, t0, bar_v);
  }
  PSTAMP(2);
  // ---- A (fp32 + bf16 mirror) from rows 0-63, P (bf16) from rows 64-127; masked entries via ex2(-inf) = 0 ----
  {
    float* sA = (float*)(smem + P_A);
    const int t = tid & 63;   // row
    const float g2t = sG2[t], bt = sBeta[t];
    uint32_t r[64];
    tc_ld32_nw(D0 + ((32 * warp) << 16), r);
    tc_ld32_nw(D0 + ((32 * warp) << 16) + 32, r + 32);
    tc_wait_ld();
    PSTAMP(17);
    if (warp < 2) {
      float vp[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int q = 0; q < 16; q++) {
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int i = 4 * q + e;
          const float arg = (i < t) ? g2t - sG2[i] : -INFINITY;
          v[e] = bt * __uint_as_float(r[i]) * ex2f(arg);
        }
        *(float4*)(sA + t * AS + 4 * q) = make_float4(v[0], v[1], v[2], v[3]);
        if (q & 1) {
          *(uint4*)(smem + P_TM + sw128(t, q >> 1)) =
              make_uint4(pack_bf16(vp[0], vp[1]), pack_bf16(vp[2], vp[3]), pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]));
        } else { vp[0] = v[0]; vp[1] = v[1]; vp[2] = v[2]; vp[3] = v[3]; }
      }
    } else {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int i = 8 * q + 2 * e;
          const float a0 = (i <= t) ? g2t - sG2[i] : -INFINITY;
          const float a1 = (i + 1 <= t) ? g2t - sG2[i + 1] : -INFINITY;
          w[e] = pack_bf16(scale * __uint_as_float(r[i]) * ex2f(a0), scale * __uint_as_float(r[i + 1]) * ex2f(a1));
        }
        *(uint4*)(smem + P_P + sw128(t, q)) = make_uint4(w[0], w[1], w[2], w[3]);
      }
    }
  }
  PSYNC();
  PSTAMP(3);
  // ---- T = (I + A)^-1.  (i) 16x16 diagonal blocks, one column per thread (64 threads) ----
  {
    float* sA = (float*)(smem + P_A);
    const int bidx = (tid >> 4) & 3, j = tid & 15, r0 = 16 * bidx;
    float col[16];
    if (tid < 64) {
      col[0] = (j == 0) ? 1.f : 0.f;
#pragma unroll
      for (int i = 1; i < 16; i++) {
        float s = 0.f;
#pragma unroll
        for (int m = 0; m < i; m++) s -= sA[(r0 + i) * AS + r0 + m] * col[m];
        col[i] = (i == j) ? 1.f : s;
      }
    }
    PSTAMP(22);
    PSYNC();
    if (tid < 64) {
#pragma unroll
      for (int i = 0; i < 16; i++) sA[(r0 + i) * AS + r0 + j] = col[i];
    }
    PSYNC();
    PSTAMP(23);
    // bf16 mirror of the diagonal blocks (row t = tid&63, 16 cols of its block)
    {
      const int t = tid & 63, bb = t >> 4;
      if (tid < 64) {
        float4 x0 = *(float4*)(sA + t * AS + 16 * bb), x1 = *(float4*)(sA + t * AS + 16 * bb + 4);
        float4 x2 = *(float4*)(sA + t * AS + 16 * bb + 8), x3 = *(float4*)(sA + t * AS + 16 * bb + 12);
        *(uint4*)(smem + P_TM + sw128(t, 2 * bb)) = make_uint4(pack_bf16(x0.x, x0.y), pack_bf16(x0.z, x0.w), pack_bf16(x1.x, x1.y), pack_bf16(x1.z, x1.w));
        *(uint4*)(smem + P_TM + sw128(t, 2 * bb + 1)) = make_uint4(pack_bf16(x2.x, x2.y), pack_bf16(x2.z, x2.w), pack_bf16(x3.x, x3.y), pack_bf16(x3.z, x3.w));
      }
    }
  }
  PSYNC();
  PSTAMP(4);
  // helpers for the bf16 mma.sync merges on the mirror tile (row-major 64x64 bf16, sw128)
  const uint32_t sTM = sb + P_TM;
  // A fragment (16x16 at rows r0.., k cols k0..): rows r0 + (lane&15), chunk k0/8 + (lane>>4)
  auto ldA = [&](uint32_t* af, int r0, int k0) {
    ldmatrix_x4(af[0], af[1], af[2], af[3], sTM + sw128(r0 + (lane & 15), (k0 >> 3) + (lane >> 4)));
  };
  // B fragments for B[k][n] stored row-major (rows = k): k rows k0..k0+15, n cols n0..n0+15 (two n-tiles)
  auto ldBt = [&](uint32_t* bb, int k0, int n0) {
    ldmatrix_x4_trans(bb[0], bb[1], bb[2], bb[3], sTM + sw128(k0 + (lane & 7) + (((lane >> 3) & 1) << 3), (n0 >> 3) + (lane >> 4)));
  };
  // C (two n-tiles, 16x16 fp32) -> A fragment (bf16), optionally negated
  auto c2a = [&](const float (*cc)[4], uint32_t* af, float sgn) {
    af[0] = pack_bf16(sgn * cc[0][0], sgn * cc[0][1]); af[1] = pack_bf16(sgn * cc[0][2], sgn * cc[0][3]);
    af[2] = pack_bf16(sgn * cc[1][0], sgn * cc[1][1]); af[3] = pack_bf16(sgn * cc[1][2], sgn * cc[1][3]);
  };
  // write a 16x16 fp32 C block (two n-tiles) to sA and its bf16 to the mirror (in place: block rows R.., cols C..)
  auto storeC = [&](const float (*cc)[4], int R, int C) {
    float* sA = (float*)(smem + P_A);
    const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int n = 0; n < 2; n++) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        int rr = R + gid + 8 * half, col = C + 8 * n + 2 * tig;
        *(float2*)(sA + rr * AS + col) = make_float2(cc[n][2 * half], cc[n][2 * half + 1]);
      }
    }
  };
  // (ii) merge 16 -> 32: warp p (0,1): T21 = -(T22 A21) T11 for blocks (R2 = 32p+16, R1 = 32p)
  if (warp < 2) {
    const int R1 = 32 * warp, R2 = 32 * warp + 16;
    float x[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    uint32_t af[4], bb[4];
    ldA(af, R2, R2);            // T22
    ldBt(bb, R2, R1);           // A21 rows k = R2.., cols n = R1..
    mma_bf16(x[0], af[0], af[1], af[2], af[3], bb[0], bb[1]);
    mma_bf16(x[1], af[0], af[1], af[2], af[3], bb[2], bb[3]);
    uint32_t ax[4];
    c2a(x, ax, -1.f);
    float z[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    ldBt(bb, R1, R1);           // T11
    mma_bf16(z[0], ax[0], ax[1], ax[2], ax[3], bb[0], bb[1]);
    mma_bf16(z[1], ax[0], ax[1], ax[2], ax[3], bb[2], bb[3]);
    __syncwarp();
    storeC(z, R2, R1);
  }
  PSYNC();
  // bf16 mirror of the new blocks (rows 16-31 cols 0-15, rows 48-63 cols 32-47): thread -> (row, 8-col chunk)
  if (tid < 64) {
    const int p = tid >> 5, rr = (tid & 31) >> 1, ch = tid & 1;
    const int row = 32 * p + 16 + rr, col = 32 * p + 8 * ch;
    const float* sA = (float*)(smem + P_A);
    float4 x0 = *(const float4*)(sA + row * AS + col), x1 = *(const float4*)(sA + row * AS + col + 4);
    *(uint4*)(smem + P_TM + sw128(row, col >> 3)) = make_uint4(pack_bf16(x0.x, x0.y), pack_bf16(x0.z, x0.w), pack_bf16(x1.x, x1.y), pack_bf16(x1.z, x1.w));
  }
  PSYNC();
  PSTAMP(5);
  // (iii) merge 32 -> 64: T(32-63, 0-31) = -(T22 A21) T11 with 32x32 blocks.
  // warp w: m-tile mt = w>>1 (rows 32+16mt), computes X row block (16 x 32) fully, then n-tiles 2(w&1),+1.
  {
    const int mt = warp >> 1, nh = warp & 1;
    const int R = 32 + 16 * mt;
    float x[4][4];
#pragma unroll
    for (int n = 0; n < 4; n++) { x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f; }
    uint32_t af[4], bb[4];
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      ldA(af, R, 32 + 16 * ks);                 // T22 rows R.., k cols 32+16ks..
#pragma unroll
      for (int np = 0; np < 2; np++) {
        ldBt(bb, 32 + 16 * ks, 16 * np);        // A21 rows k = 32+16ks.., n cols 16np..
        mma_bf16(x[2 * np], af[0], af[1], af[2], af[3], bb[0], bb[1]);
        mma_bf16(x[2 * np + 1], af[0], af[1], af[2], af[3], bb[2], bb[3]);
      }
    }
    // Z = -X T11 : n-tiles 2nh, 2nh+1 (cols 16nh..16nh+15), K = 32
    float z[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      uint32_t ax[4];
      c2a(x + 2 * ks, ax, -1.f);
      ldBt(bb, 16 * ks, 16 * nh);               // T11 rows k = 16ks.., n cols 16nh..
      mma_bf16(z[0], ax[0], ax[1], ax[2], ax[3], bb[0], bb[1]);
      mma_bf16(z[1], ax[0], ax[1], ax[2], ax[3], bb[2], bb[3]);
    }
    PSYNC();   // everyone finished reading A21 / T blocks before the in-place write
    storeC(z, R, 16 * nh);
  }
  PSYNC();
  PSTAMP(6);
  // ---- T' = T diag(beta exp(Gam)) -> P_TA rows 0-63,  T'' = T diag(beta) -> P_TB rows 0-63 (bf16 sw128) ----
  {
    const float* sA = (const float*)(smem + P_A);
    const int t = tid >> 1, c0 = 32 * (tid & 1);
#pragma unroll
    for (int ch = 0; ch < 4; ch++) {
      uint32_t w1[4], w2[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int i = c0 + 8 * ch + 2 * e;
        float2 tv = *(const float2*)(sA + t * AS + i);
        w1[e] = pack_bf16(tv.x * sBg[i], tv.y * sBg[i + 1]);
        w2[e] = pack_bf16(tv.x * sBeta[i], tv.y * sBeta[i + 1]);
      }
      uint32_t off = sw128(t, (c0 >> 3) + ch);
      *(uint4*)(smem + P_TA + off) = make_uint4(w1[0], w1[1], w1[2], w1[3]);
      *(uint4*)(smem + P_TB + off) = make_uint4(w2[0], w2[1], w2[2], w2[3]);
    }
  }
  fence_proxy_async();
  PSYNC();
  PSTAMP(7);
  // ---- GEMM1: D1 = P (A K-major, M=128: rows 64+ garbage) x [T' | T''] (B MN-major, N = 128, LBO = 16 KB) ----
  if (tid == 0) {
    tc_fence_after();
    const uint32_t idesc = make_idesc(128, 128, false, true);
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
      tc_mma(D1, make_desc(sb + P_P + ks * 32, 16, 1024, 2), make_desc(sb + P_TA + ks * 2048, 16384, 1024, 2), idesc, ks > 0 ? 1u : 0u);
    tc_commit(bar_m1);
  }
  mbar_wait(bar_m1, 0);
  tc_fence_after();
  // readout rows 0-63 (warps 0,1): cols 0-63 -> P_TA rows 64+t, cols 64-127 -> P_TB rows 64+t
  if (warp < 2) {
    const int t = tid;
    uint32_t r[32];
#pragma unroll
    for (int piece = 0; piece < 4; piece++) {
      tc_ld32(D1 + ((32 * warp) << 16) + 32 * piece, r);
      unsigned char* dst = smem + ((piece < 2) ? P_TA : P_TB);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint4 w = make_uint4(pack_bf16(__uint_as_float(r[8 * q]), __uint_as_float(r[8 * q + 1])),
                             pack_bf16(__uint_as_float(r[8 * q + 2]), __uint_as_float(r[8 * q + 3])),
                             pack_bf16(__uint_as_float(r[8 * q + 4]), __uint_as_float(r[8 * q + 5])),
                             pack_bf16(__uint_as_float(r[8 * q + 6]), __uint_as_float(r[8 * q + 7])));
        *(uint4*)(dst + sw128(64 + t, 4 * (piece & 1) + q)) = w;
      }
    }
    fence_proxy_async();
  }
  tc_fence_before();
  PSYNC();
  PSTAMP(8);
  // ---- GEMM2: D2 = [T';PT'] x K (B MN-major: N = dk, blocks at +16 KB), GEMM3: D3 = [T'';PT''] x V (blocks +8 KB) ----
  if (tid == 0) {
    tc_fence_after();
    const uint32_t idesc = make_idesc(128, 128, false, true);
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
      tc_mma(D2, make_desc(sb + P_TA + ks * 32, 16, 1024, 2), make_desc(sb + P_KQ + ks * 2048, 16384, 1024, 2), idesc, ks > 0 ? 1u : 0u);
    mbar_wait(bar_v, 0);
    tc_fence_after();
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
      tc_mma(D3, make_desc(sb + P_TB + ks * 32, 16, 1024, 2), make_desc(sb + P_V + ks * 2048, 8192, 1024, 2), idesc, ks > 0 ? 1u : 0u);
    tc_commit(bar_m2);
  }
  mbar_wait(bar_m2, 0);
  tc_fence_after();
  PSTAMP(9);
  // ---- outputs: TMEM -> registers -> bf16 -> smem staging (blob layout) -> per-warp 4 KB bulk stores ----
  // rows 0-63 of D2 = Wt, rows 64-127 = P Wt ; rows 0-63 of D3 = Ut, rows 64-127 = P Ut
  // staging: W -> P_TA, Q -> P_TB, U -> P_A, O -> P_V (all dead once GEMM1-3 completed); drain waited by the caller
  {
    unsigned char* g = blob + ((size_t)c * 8 + h) * BLOB_BYTES;
    const int t = tid & 63;
    const float gdt = sGd[t], get = sGe[t];
    const uint32_t trow = (32 * warp) << 16;
    const int wofs = (warp & 1) * 4096;
#pragma unroll
    for (int pp = 0; pp < 2; pp++) {     // D2 columns 64pp .. 64pp+63 (dk half pp)
      uint32_t r[64];
      tc_ld32_nw(D2 + trow + 64 * pp, r);
      tc_ld32_nw(D2 + trow + 64 * pp + 32, r + 32);
      tc_wait_ld();
      if (warp < 2) {
        unsigned char* st = smem + P_TA + pp * 8192;
#pragma unroll
        for (int q = 0; q < 8; q++) {
          uint4 w = make_uint4(pack_bf16(gdt * __uint_as_float(r[8 * q]), gdt * __uint_as_float(r[8 * q + 1])),
                               pack_bf16(gdt * __uint_as_float(r[8 * q + 2]), gdt * __uint_as_float(r[8 * q + 3])),
                               pack_bf16(gdt * __uint_as_float(r[8 * q + 4]), gdt * __uint_as_float(r[8 * q + 5])),
                               pack_bf16(gdt * __uint_as_float(r[8 * q + 6]), gdt * __uint_as_float(r[8 * q + 7])));
          *(uint4*)(st + sw128(t, q)) = w;
        }
        __syncwarp();
        fence_proxy_async();
        if (lane == 0) { bulk_store_1d(g + BL_W + pp * 8192 + wofs, sb + P_TA + pp * 8192 + wofs, 4096); bulk_commit(); }
      } else {
        // Qhat = ge * Q - (P Wt);  Q row t lives at P_KQ + pp*16384 + 8192 (rows 64-127 of the [K;Q] tile)
        unsigned char* st = smem + P_TB + pp * 8192;
#pragma unroll
        for (int q = 0; q < 8; q++) {
          uint4 qq = *(const uint4*)(smem + P_KQ + pp * 16384 + 8192 + sw128(t, q));
          uint32_t qs[4] = {qq.x, qq.y, qq.z, qq.w}, w[4];
#pragma unroll
          for (int e = 0; e < 4; e++)
            w[e] = pack_bf16(get * bf16lo(qs[e]) - __uint_as_float(r[8 * q + 2 * e]), get * bf16hi(qs[e]) - __uint_as_float(r[8 * q + 2 * e + 1]));
          *(uint4*)(st + sw128(t, q)) = make_uint4(w[0], w[1], w[2], w[3]);
        }
        __syncwarp();
        fence_proxy_async();
        if (lane == 0) { bulk_store_1d(g + BL_Q + pp * 8192 + wofs, sb + P_TB + pp * 8192 + wofs, 4096); bulk_commit(); }
      }
    }
    PSTAMP(19);
#pragma unroll
    for (int pp = 0; pp < 2; pp++) {     // D3 columns 64pp .. (dv half pp)
      uint32_t r[64];
      tc_ld32_nw(D3 + trow + 64 * pp, r);
      tc_ld32_nw(D3 + trow + 64 * pp + 32, r + 32);
      tc_wait_ld();
      const float sc = (warp < 2) ? gdt : 1.f;
      const uint32_t stg = ((warp < 2) ? P_A : P_V) + pp * 8192;
      unsigned char* st = smem + stg;
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint4 w = make_uint4(pack_bf16(sc * __uint_as_float(r[8 * q]), sc * __uint_as_float(r[8 * q + 1])),
                             pack_bf16(sc * __uint_as_float(r[8 * q + 2]), sc * __uint_as_float(r[8 * q + 3])),
                             pack_bf16(sc * __uint_as_float(r[8 * q + 4]), sc * __uint_as_float(r[8 * q + 5])),
                             pack_bf16(sc * __uint_as_float(r[8 * q + 6]), sc * __uint_as_float(r[8 * q + 7])));
        *(uint4*)(st + sw128(t, q)) = w;
      }
      __syncwarp();
      fence_proxy_async();
      if (lane == 0) { bulk_store_1d(g + ((warp < 2) ? BL_U : BL_O) + pp * 8192 + wofs, sb + stg + wofs, 4096); bulk_commit(); }
    }
  }
  tc_fence_before();
  PSTAMP(10);
#undef PSYNC
#undef PSTAMP
}

__global__ void __launch_bounds__(128, 2)
gdn_prologue_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                    const __grid_constant__ CUtensorMap tmV,
                    const bf16* __restrict__ a, const bf16* __restrict__ b,
                    const float* __restrict__ A_log, const float* __restrict__ dt_bias, float scale,
                    const __grid_constant__ GdnTabs tabs, unsigned char* __restrict__ blob, float* __restrict__ gC,
                    long long* __restrict__ pdbg)
{
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5;
  uint32_t* tmem_slot = (uint32_t*)(smem + P_BAR + 64);
  if (warp == 1) { tmem_alloc(smem_u32(tmem_slot), 256); tmem_relinquish(); }
  __syncthreads();
  const uint32_t tmem = *tmem_slot;
  long long ptacc[24] = {};
#ifdef TIMING
  ptacc[20] = clock64();
#endif
  prologue_item<0>(smem, tmem, blockIdx.x, blockIdx.y, tid, &tmQ, &tmK, &tmV, a, b, A_log, dt_bias, scale, tabs, blob, gC, ptacc);
  if ((tid & 31) == 0) bulk_wait_read0();
#ifdef TIMING
  if (tid == 0) { long long tn = clock64(); ptacc[20] = tn - ptacc[20]; }
#endif
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
#ifdef TIMING
  if (tid == 0 && pdbg) { for (int i = 0; i < 24; i++) pdbg[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + i] = ptacc[i]; }
#endif
}

// ------------------------------------------------------------------------------------------------
// Kernel B: chunk recurrence.  One CTA per (sequence, v-head, 64-wide v half).  288 threads:
//   warps 0-3 own the fp32 state (thread = dk row, 64 v columns),
//   warps 4-7 convert Mneg (TMEM fp32 -> bf16 TMEM A operand) and write O,
//   warp 8 (issuer A) issues the critical-path MMAs (chain, epi), warp 9 (issuer B) the off-path ones (KU, Mneg).
//   A single thread can only issue one tcgen05.mma every ~60-70 cycles (ELECT loop) while the pipe finishes an
//   M128 N64 K16 MMA in ~36, so two issuing threads are needed to keep the tensor pipe busy.
// Each role runs its own loop (so the state registers are only live in the S warps); the roles are
// coupled through mbarriers (tensor-core completion) and two named barriers:
//   bar 1: S warps -> issuer A     "S_ci tile is in smem" (also: S warps are done reading TM_C[(ci-1)&1])
//   bar 2: M warps -> issuers A,B  "Mneg_{ci} converted (TM_A[ci&1] valid), TM_M is free"
//   bar 3: issuer A -> issuer B    "chain_ci has been issued" (keeps the chain ahead of KU/Mneg in the pipe)
//   barKU (mbarrier, issuer B commits): KU_ci complete -> A may accumulate chain_ci onto it
// Tensor pipe work per chunk c (issue order):
//   chain_c : Dc[c&1] += Mneg_c (TMEM A) . bf16(S_c)        8 MMAs  -> barChain
//   epi_c   : Depi[c&1] = Qhat_c . bf16(S_c)                8 MMAs  -> barEpi
//   KU_{c+1}: Dc[(c+1)&1] = K_{c+1}^T gdUt_{c+1}            4 MMAs  (off the critical path)
//   Mneg_{c+2} = K^T gdWt (fp32, TM_M)                      4 MMAs  -> barMneg
//   S_{c+1} = gC S_c + Dc[c&1]  (fp32 registers, S warps), O_c = Depi + Ohat (M warps)
// ------------------------------------------------------------------------------------------------
#define R_K     0          // 4 stages x 16 KB: K tile [64 t][128 dk] (two TMA boxes, dk halves at +8192)
#define R_W     65536      // 4 stages x 16 KB: gdWt [64 t][128 dk'] (dk' halves at +8192)
#define R_Q     131072     // 2 stages x 16 KB: Qhat [64 t][128 dk] K-major (dk halves at +8192)
#define R_U     163840     // 4 stages x 8 KB : gdUt half [64 t][64 v]
#define R_S     196608     // 2 x 16 KB: S bf16 [128 dk][64 v] sw128
#define R_BAR   229376
#define R_SMEM  (R_BAR + 1024)


DEVI void tc_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
               :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr) : "memory");
}
DEVI void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// MMA helpers: the operand adds (base + immediate offset) are done inside the asm block so that neither
// LLVM nor ptxas hoists dozens of loop-invariant descriptors into vector registers (which then cost a pair of
// R2UR moves per MMA).  ACC is a compile-time accumulate flag.
template <int ACC, int DOFF, int AOFF, int BOFF>
DEVI void mma_ss(uint32_t d_base, uint64_t a_base, uint64_t b_base, uint32_t idesc) {
  if (ACC) asm volatile("{\n .reg .pred p;\n .reg .b32 d;\n .reg .b64 a, b;\n"
                        " add.u32 d, %0, %4;\n add.u64 a, %1, %5;\n add.u64 b, %2, %6;\n"
                        " setp.eq.u32 p, %3, %3;\n tcgen05.mma.cta_group::1.kind::f16 [d], a, b, %3, p;\n}"
                        ::"r"(d_base), "l"(a_base), "l"(b_base), "r"(idesc), "n"(DOFF), "n"(AOFF), "n"(BOFF) : "memory");
  else asm volatile("{\n .reg .pred p;\n .reg .b32 d;\n .reg .b64 a, b;\n"
                    " add.u32 d, %0, %4;\n add.u64 a, %1, %5;\n add.u64 b, %2, %6;\n"
                    " setp.ne.u32 p, %3, %3;\n tcgen05.mma.cta_group::1.kind::f16 [d], a, b, %3, p;\n}"
                    ::"r"(d_base), "l"(a_base), "l"(b_base), "r"(idesc), "n"(DOFF), "n"(AOFF), "n"(BOFF) : "memory");
}
// A operand from TMEM ([lane = m][32-bit column j = bf16 pair k=2j,2j+1]; + 8 columns per K=16 slab)
template <int ACC, int DOFF, int AOFF, int BOFF>
DEVI void mma_ts(uint32_t d_base, uint32_t a_base, uint64_t b_base, uint32_t idesc) {
  if (ACC) asm volatile("{\n .reg .pred p;\n .reg .b32 d, a;\n .reg .b64 b;\n"
                        " add.u32 d, %0, %4;\n add.u32 a, %1, %5;\n add.u64 b, %2, %6;\n"
                        " setp.eq.u32 p, %3, %3;\n tcgen05.mma.cta_group::1.kind::f16 [d], [a], b, %3, p;\n}"
                        ::"r"(d_base), "r"(a_base), "l"(b_base), "r"(idesc), "n"(DOFF), "n"(AOFF), "n"(BOFF) : "memory");
  else asm volatile("{\n .reg .pred p;\n .reg .b32 d, a;\n .reg .b64 b;\n"
                    " add.u32 d, %0, %4;\n add.u32 a, %1, %5;\n add.u64 b, %2, %6;\n"
                    " setp.ne.u32 p, %3, %3;\n tcgen05.mma.cta_group::1.kind::f16 [d], [a], b, %3, p;\n}"
                    ::"r"(d_base), "r"(a_base), "l"(b_base), "r"(idesc), "n"(DOFF), "n"(AOFF), "n"(BOFF) : "memory");
}
DEVI uint4 ldg16(const void* p) { return __ldg((const uint4*)p); }

template <bool FUSED>
DEVI void recur_item(unsigned char* smem, const uint32_t tmem_base, const int bid,
                     const CUtensorMap* tmK, const unsigned char* __restrict__ blob,
                     const float* __restrict__ gC,
                     const float* __restrict__ state_in,  // [N][8][128 v][128 k] (may be null)
                     float* __restrict__ state_out,       // [N][8][128][128]
                     bf16* __restrict__ out,              // [T][8][128]
                     const GdnTabs& tabs, const int* __restrict__ flags, long long* __restrict__ dbg, const bool reinit)
{
#ifdef TIMING
  long long tacc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; long long tlast = 0;
#define STAMP(i) { long long tn = clock64(); tacc[i] += tn - tlast; tlast = tn; }
#define TRACE(idx, val) { if (dbg && (blockIdx.x & 15) == 0 && (idx) < 1024) dbg[640 * 32 + (blockIdx.x >> 4) * 1024 + (idx)] = (val); }
#else
#define STAMP(i)
#define TRACE(idx, val)
#endif
#ifdef WATCHDOG
#define WDA(i) { if (lane == 0) WDP(8, ci * 8 + (i)); }
#define WDB(i) { if (lane == 0) WDP(9, ci * 8 + (i)); }
#else
#define WDA(i)
#define WDB(i)
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int s = tabs.order[bid >> 4];
  const int h = (bid >> 1) & 7;
  const int hv = bid & 1;
  const int kh = h >> 1;
  const int v0 = hv * 64;
  const int tok0 = tabs.cu[s], tok1 = tabs.cu[s + 1];
  const int c0 = tabs.cstart[s], c1 = tabs.cstart[s + 1];
  const int nc = c1 - c0;

  const uint32_t sbase = smem_u32(smem);
  const uint32_t barK0 = sbase + R_BAR;          // 4
  const uint32_t barW0 = sbase + R_BAR + 32;     // 4
  const uint32_t barQ0 = sbase + R_BAR + 64;     // 2
  const uint32_t barU0 = sbase + R_BAR + 80;     // 4
  const uint32_t barChain = sbase + R_BAR + 112, barEpi = sbase + R_BAR + 120, barMneg = sbase + R_BAR + 128, barKU = sbase + R_BAR + 136;
  // Epi uses two alternating barriers (barEpi for even chunks, barEpi1 for odd): Epi_{c+1} may be issued and complete before the
  // M warps have observed Epi_c (their bar-2 arrival precedes the Epi wait), so a single parity-tracked barrier could flip twice
  // (Epi_c, Epi_{c+1}) while an M warp is suspended in try_wait -> the waiter would then alias onto the wrong phase and hang.
  // Epi_{c+2} needs bar-2 gen c+2, i.e. every M warp's arrival from iteration c+1, which follows its Epi_c wait -> no aliasing.
  const uint32_t barEpi1 = sbase + R_BAR + 368;
  volatile int* prog = (volatile int*)(smem + R_BAR + 176);   // [0]: #Mneg observed complete by M warps, [1]: #Epi
  volatile float* sgC = (volatile float*)(smem + R_BAR + 192);  // [8] gC ring: written by the loader (after the flag acquire) before the
                                                                 // W-stage expect_tx arrive, read by the S warps after the chain wait (causally after)
  volatile float* sgv = (volatile float*)(smem + R_BAR + 224);  // [32] gC staging ring: flag warp -> loader
  volatile int* sfa = (volatile int*)(smem + R_BAR + 352);      // #chunks whose flag has been acquired (flag warp -> loader)
  volatile int* skw = (volatile int*)(smem + R_BAR + 356);
  volatile int* sdone = (volatile int*)(smem + R_BAR + 360);   // WATCHDOG: M warps finished (releases the flag warp's dump poll)      // #chunks whose K/W/U loads were issued (loader -> flag warp, ring backpressure)
  const int n4 = nc < 4 ? nc : 4;
  const int epoch = tabs.epoch;

  if (nc == 0) {   // state passthrough
    if (warp < 4) {
      const float* sp = state_in ? state_in + (((size_t)s * 8 + h) * 128 + v0) * 128 + tid : nullptr;
      float* dp = state_out + (((size_t)s * 8 + h) * 128 + v0) * 128 + tid;
#pragma unroll 8
      for (int j = 0; j < 64; j++) dp[(size_t)j * 128] = sp ? sp[(size_t)j * 128] : 0.f;
    }
    return;
  }
  if (tid == 0) {
    for (int i = 0; i < 19; i++) { const uint32_t bb = i < 18 ? sbase + R_BAR + 8 * i : barEpi1; if (reinit) mbar_inval(bb); mbar_init(bb, 1); }
    prog[0] = 0; prog[1] = 0; *sfa = 0; *skw = 0; *sdone = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // blob-ready flag (fused mode): spin until the prologue lane published chunk ci for this head, then order the
  // following async-proxy loads / generic loads after it.  `pre` is an optional earlier relaxed read of the flag.
  auto flagp = [&](int ci) { return flags + ((size_t)(c0 + ci) * 8 + h); };
  auto wait_flag = [&](int ci) {
    if (FUSED) {
      const int* f = flagp(ci);
#if defined(WATCHDOG) && !defined(WD_LITE)
      { long long _n = 0; while (ld_relaxed_gpu(f) != epoch) { if (++_n > 400000000LL) { volatile int* w = g_wdp; int k = atomicAdd((int*)w + 5, 1); if (k < 64) { w[8192 + k * 4] = blockIdx.x; w[8192 + k * 4 + 1] = c0 + ci; w[8192 + k * 4 + 2] = h; w[8192 + k * 4 + 3] = nc; } break; } } }
#else
      WD_SPIN(ld_relaxed_gpu(f) != epoch, 2, ci);
#endif
      fence_acq_rel_gpu();
      fence_proxy_async_global();
    }
  };
  const uint32_t TM_C0 = tmem_base, TM_E0 = tmem_base + 128, TM_M = tmem_base + 256, TM_A0 = tmem_base + 384;   // TMEM column map

  const unsigned char* blobh = blob + ((size_t)c0 * 8 + h) * BLOB_BYTES;  // chunk ci -> blobh + ci*8*BLOB_BYTES
  // ---- single-thread load issue helpers ----
  auto issue_K = [&](int ci, int e) {
    const uint32_t bar = barK0 + 8 * e, dst = sbase + R_K + e * 16384;
    mbar_expect_tx(bar, 16384);
    tma_load_3d(dst, tmK, 0, kh, tok0 + ci * 64, bar);
    tma_load_3d(dst + 8192, tmK, 64, kh, tok0 + ci * 64, bar);
    TRACE(ci * 16 + 0, clock64());
  };
  auto issue_W = [&](int ci, int e) {
    const uint32_t bar = barW0 + 8 * e, dst = sbase + R_W + e * 16384;
    const unsigned char* g = blobh + (size_t)ci * 8 * BLOB_BYTES + BL_W;
    mbar_expect_tx(bar, 16384);
    bulk_load_1d(dst, g, 8192, bar);
    bulk_load_1d(dst + 8192, g + 8192, 8192, bar);
    TRACE(ci * 16 + 2, clock64());
  };
  auto issue_Q = [&](int ci, int e) {
    const uint32_t bar = barQ0 + 8 * e, dst = sbase + R_Q + e * 16384;
    const unsigned char* g = blobh + (size_t)ci * 8 * BLOB_BYTES + BL_Q;
    mbar_expect_tx(bar, 16384);
    bulk_load_1d(dst, g, 8192, bar);
    bulk_load_1d(dst + 8192, g + 8192, 8192, bar);
      TRACE(ci * 16 + 14, clock64());
  };
  auto issue_U = [&](int ci, int e) {
    const uint32_t bar = barU0 + 8 * e, dst = sbase + R_U + e * 8192;
    const unsigned char* g = blobh + (size_t)ci * 8 * BLOB_BYTES + BL_U + hv * 8192;
    mbar_expect_tx(bar, 8192);
    bulk_load_1d(dst, g, 8192, bar);
      TRACE(ci * 16 + 13, clock64());
  };

#ifdef WATCHDOG
  auto wd_dump = [&]() {   // on host request: dump this CTA's barrier words + control words (called by the flag warp lane 0)
    volatile int* hw = (volatile int*)g_wdp;
    if (hw[512] == 0) return;
    volatile int* d = hw + 16384 + blockIdx.x * 64;
    if (d[0] == 1) return;
    for (int i = 0; i < 18; i++) { d[1 + 2 * i] = *(volatile int*)(smem + R_BAR + 8 * i); d[2 + 2 * i] = *(volatile int*)(smem + R_BAR + 8 * i + 4);
      d[41 + i] = mbar_try_wait(sbase + R_BAR + 8 * i, 0) ? 1 : 0; }
    d[37] = prog[0]; d[38] = prog[1]; d[39] = *sfa; d[40] = *skw;
    __threadfence_system(); d[0] = 1;
  };
#endif
  if (warp == 11) {
    // =============================== flag warp: acquires the prologue-ready flags in chunk order ===============================
    // Blocking global polls live here so that the loader's issue loop is smem-only.  Up to 16 consecutive flags are polled per
    // round (one per lane); each lane that saw its flag does the acquire + proxy fences itself (same thread as the load), then
    // the warp publishes the count of consecutively acquired chunks in sfa and their gC values in the sgv ring.
    int fa = 0;
    long long spins = 0;
    while (fa < nc) {
      int lim = nc - fa; if (lim > 16) lim = 16;
      { const int room = *skw + 32 - fa; if (room < lim) lim = room; }   // staging ring backpressure
      // The loader advances skw concurrently; keep every lane on one cursor.
      lim = __shfl_sync(0xffffffffu, lim, 0);
      bool mine = false;
      if (lim > 0 && lane < lim) mine = !FUSED || ld_relaxed_gpu(flagp(fa + lane)) == epoch;
      const unsigned m = __ballot_sync(0xffffffffu, mine);
      int k = __ffs(~m) - 1;   // # consecutive lanes (from 0) whose flag is set
      if (k > lim) k = lim;
      if (k > 0) {
        if (lane < k) {
          if (FUSED) { fence_acq_rel_gpu(); fence_proxy_async_global(); }
          sgv[(fa + lane) & 31] = __ldcg(gC + (size_t)(c0 + fa + lane) * 8 + h);
        }
        __syncwarp();
        if (lane == 0) { __threadfence_block(); *sfa = fa + k; WDP(10, fa + k); }
        fa += k;
        spins = 0;
      }
#ifdef WATCHDOG
      else if ((++spins & 4095) == 0) { if (lane == 0) wd_dump(); __syncwarp();
#ifndef WD_LITE
        if (spins > 400000000LL) wd_trap(2, fa);
#endif
      }
#endif
    }
#ifdef WATCHDOG
    if (lane == 0) { long long n = 0; while (*sdone == 0) { if ((++n & 4095) == 0) wd_dump(); } }
#endif
  } else if (warp == 10) {
    // =============================== loader warp: all smem ring loads ===============================
    if (lane == 0) {
      for (int ci = 0; ci < n4; ci++) issue_K(ci, ci);   // K does not depend on the prologue
      // Independent cursors: kw = next chunk whose (K/)W/U loads are issued, qn = next chunk whose Q load is issued,
      // fa = #flags acquired (mirrored from the flag warp).  Neither cursor blocks the other.
      int kw = 0, qn = 0, fa = 0;
      long long spins = 0;
      WDP(6, kw);
      while (kw < nc || qn < nc) {
        { const int f = *sfa; if (f != fa) { fa = f; __threadfence_block(); fence_proxy_async_global(); } }
        if (qn < nc && fa > qn && (qn < 2 || prog[1] >= qn - 1)) { issue_Q(qn, qn & 1); qn++; WDP(11, qn); }   // Epi_{qn-2} done -> Q stage free
        if (kw < nc && fa > kw && (kw < 4 || prog[0] >= kw - 3)) {                                 // Mneg_{kw-3} done -> K/W/U stage free
          sgC[kw & 7] = sgv[kw & 31];                                                               // released by issue_W's expect_tx arrive
          if (kw >= 4) issue_K(kw, kw & 3);
          issue_W(kw, kw & 3); issue_U(kw, kw & 3); kw++;
          *skw = kw;
          WDP(6, kw);
        }
#ifdef WATCHDOG
        if (++spins > 400000000LL) wd_trap(3, kw);
#endif
      }
    }
  } else if (warp >= 8) {
    // =============================== issuer warps (8 = A, 9 = B) ===============================
    const uint32_t idescMM = make_idesc(128, 128, true, true);   // K^T (MN) x gdWt (MN)
    const uint32_t idescC1 = make_idesc(128, 64, true, true);    // K^T (MN) x gdUt (MN)
    const uint32_t idescC2 = make_idesc(128, 64, false, true);   // Mneg(TMEM) / Qhat (K-major) x S (MN)
    // descriptor bases (byte offset -> +offset>>4)
    const uint64_t dK0 = make_desc(sbase + R_K, 8192, 1024, 2);
    const uint64_t dW0 = make_desc(sbase + R_W, 8192, 1024, 2);
    const uint64_t dU0 = make_desc(sbase + R_U, 8192, 1024, 2);
    const uint64_t dS0 = make_desc(sbase + R_S, 8192, 1024, 2);
    const uint64_t dQ0 = make_desc(sbase + R_Q, 16, 1024, 2);
    if (warp == 9) {
      // ---------------- issuer B: KU_{c+1}, Mneg_{c+2} ----------------
      auto issue_Mneg = [&](auto kec, auto wec) {       // TM_M = K^T gdWt  (4 MMAs, N=128) -> barMneg
        constexpr int KO = decltype(kec)::value * 1024, WO = decltype(wec)::value * 1024;
#if defined(EXPT) && (EXPT & 1)
        tc_commit(barMneg); return;
#endif
        mma_ss<0, 256, KO + 0, WO + 0>(tmem_base, dK0, dW0, idescMM);
        mma_ss<1, 256, KO + 128, WO + 128>(tmem_base, dK0, dW0, idescMM);
        mma_ss<1, 256, KO + 256, WO + 256>(tmem_base, dK0, dW0, idescMM);
        mma_ss<1, 256, KO + 384, WO + 384>(tmem_base, dK0, dW0, idescMM);
        tc_commit(barMneg);
      };
      auto issue_KU = [&](auto kec, auto uec, auto cbc) {   // TM_C[cb] = K^T gdUt (4 MMAs, N=64) -> barKU
        constexpr int KO = decltype(kec)::value * 1024, UO = decltype(uec)::value * 512, DO = decltype(cbc)::value * 64;
#if defined(EXPT) && (EXPT & 1)
        tc_commit(barKU); return;
#endif
        mma_ss<0, DO, KO + 0, UO + 0>(tmem_base, dK0, dU0, idescC1);
        mma_ss<1, DO, KO + 128, UO + 128>(tmem_base, dK0, dU0, idescC1);
        mma_ss<1, DO, KO + 256, UO + 256>(tmem_base, dK0, dU0, idescC1);
        mma_ss<1, DO, KO + 384, UO + 384>(tmem_base, dK0, dU0, idescC1);
        tc_commit(barKU);
      };
      // pre-loop: Mneg_0, KU_0, [Mneg_0 converted] Mneg_1
      mbar_wait(barK0, 0); mbar_wait(barW0, 0);
      tc_fence_after();
      if (lane == 0) { TRACE(1, clock64()); issue_Mneg(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}); TRACE(3, clock64()); }
      mbar_wait(barU0, 0);
      tc_fence_after();
      if (lane == 0) issue_KU(std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
      __syncwarp();
      bar_named_sync(2, 192);                           // gen 0: Mneg_0 converted -> TM_M free
      if (lane == 0) WDP(9, -1);
      if (nc > 1) {
        mbar_wait(barK0 + 8, 0); mbar_wait(barW0 + 8, 0);
        tc_fence_after();
        if (lane == 0) { TRACE(17, clock64()); issue_Mneg(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}); TRACE(19, clock64()); }
        __syncwarp();
      }
#ifdef TIMING
      tlast = clock64();
#endif
      auto stepB = [&](auto phc, const int ci) {
        constexpr int PH = decltype(phc)::value;
        bar_named_sync(3, 64);                          // chain_ci issued (implies S warps are done with TM_C[(ci+1)&1])
        STAMP(0); WDB(1);
        if (ci + 1 < nc) {
          mbar_wait(barK0 + 8 * ((PH + 1) & 3), ((ci + 1) >> 2) & 1);
          mbar_wait(barU0 + 8 * ((PH + 1) & 3), ((ci + 1) >> 2) & 1);
          tc_fence_after();
          STAMP(1); WDB(2);
          if (lane == 0) TRACE((ci + 1) * 16 + 12, clock64());
          if (lane == 0) issue_KU(std::integral_constant<int, (PH + 1) & 3>{}, std::integral_constant<int, (PH + 1) & 3>{}, std::integral_constant<int, (PH + 1) & 1>{});
          __syncwarp();
          STAMP(2); WDB(3);
          bar_named_sync(2, 192);                       // gen ci+1: Mneg_{ci+1} converted -> TM_M free
          STAMP(3); WDB(4);
          if (ci + 2 < nc) {
            mbar_wait(barK0 + 8 * ((PH + 2) & 3), ((ci + 2) >> 2) & 1);
            mbar_wait(barW0 + 8 * ((PH + 2) & 3), ((ci + 2) >> 2) & 1);
            tc_fence_after();
            STAMP(4); WDB(5);
            if (lane == 0) { TRACE((ci + 2) * 16 + 1, tlast); issue_Mneg(std::integral_constant<int, (PH + 2) & 3>{}, std::integral_constant<int, (PH + 2) & 3>{}); TRACE((ci + 2) * 16 + 3, clock64()); }
            __syncwarp();
          }
          STAMP(5); WDB(6);
        }
      };
      for (int ci = 0; ci < nc; ci += 4) {
        stepB(std::integral_constant<int, 0>{}, ci);
        if (ci + 1 < nc) stepB(std::integral_constant<int, 1>{}, ci + 1);
        if (ci + 2 < nc) stepB(std::integral_constant<int, 2>{}, ci + 2);
        if (ci + 3 < nc) stepB(std::integral_constant<int, 3>{}, ci + 3);
      }
#ifdef TIMING
      if (dbg && lane == 0) { for (int i = 0; i < 6; i++) dbg[bid * 32 + 8 + i] = tacc[i]; }
#endif
    } else {
      // ---------------- issuer A: chain_c, epi_c ----------------
      bar_named_sync(2, 192);                           // gen 0: Mneg_0 converted -> TM_A[0] valid
      if (lane == 0) WDP(8, -1);
#ifdef TIMING
      tlast = clock64();
#endif
      auto step = [&](auto phc, const int ci) {
        constexpr int PH = decltype(phc)::value;
        constexpr int sb = PH & 1;
        constexpr int CO = 64 * sb, EO = 128 + 64 * sb, AO = 384 + 64 * sb, SO = sb * 1024, QO = sb * 1024;
        bar_named_sync(1, 160);                         // S_ci tile ready
        STAMP(3); WDA(1);
        if (lane == 0) TRACE(ci * 16 + 7, tlast);
        mbar_wait(barQ0 + 8 * sb, (ci >> 1) & 1);        // Qhat_ci
        mbar_wait(barKU, ci & 1);                        // KU_ci landed in TM_C[sb]
        tc_fence_after();
        STAMP(4); WDA(2);
        if (lane == 0) TRACE(ci * 16 + 15, clock64());
        if (lane == 0) {
#if defined(EXPT) && (EXPT & 8)
          if (0)
#endif
          {
          mma_ts<1, CO, AO + 0, SO + 0>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 8, SO + 128>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 16, SO + 256>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 24, SO + 384>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 32, SO + 512>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 40, SO + 640>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 48, SO + 768>(tmem_base, tmem_base, dS0, idescC2);
          mma_ts<1, CO, AO + 56, SO + 896>(tmem_base, tmem_base, dS0, idescC2);
          }
          tc_commit(barChain);
          TRACE(ci * 16 + 6, clock64());
        }
        __syncwarp(); WDA(3);
#if defined(EXPT) && (EXPT & 4)
        mbar_wait(barChain, ci & 1);
        STAMP(2);
#endif
        bar_named_arrive(3, 64);                        // let issuer B queue KU/Mneg behind the chain
        STAMP(5); WDA(4);
        if (lane == 0) {
#if defined(EXPT) && (EXPT & 2)
          if (0)
#endif
          {
          mma_ss<0, EO, QO + 0, SO + 0>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 2, SO + 128>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 4, SO + 256>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 6, SO + 384>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 512, SO + 512>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 514, SO + 640>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 516, SO + 768>(tmem_base, dQ0, dS0, idescC2);
          mma_ss<1, EO, QO + 518, SO + 896>(tmem_base, dQ0, dS0, idescC2);
          }
          tc_commit(sb ? barEpi1 : barEpi);
        }
        __syncwarp();
        STAMP(6); WDA(5);
        if (ci + 1 < nc) {
          bar_named_sync(2, 192);                       // gen ci+1: Mneg_{ci+1} converted -> TM_A[(ci+1)&1] valid
          STAMP(7); WDA(6);
        }
      };
      for (int ci = 0; ci < nc; ci += 4) {
        step(std::integral_constant<int, 0>{}, ci);
        if (ci + 1 < nc) step(std::integral_constant<int, 1>{}, ci + 1);
        if (ci + 2 < nc) step(std::integral_constant<int, 2>{}, ci + 2);
        if (ci + 3 < nc) step(std::integral_constant<int, 3>{}, ci + 3);
      }
#ifdef TIMING
      if (dbg && lane == 0) { for (int i = 3; i < 8; i++) dbg[bid * 32 + i] = tacc[i]; dbg[bid * 32 + 14] = tacc[2]; }
#endif
    }
  } else if (warp < 4) {
    // =============================== S warps ===============================
    const int dk = tid;
    const uint32_t lrow = (uint32_t)(32 * warp) << 16;
    float S[64];
    if (state_in) {
      const float* sp = state_in + (((size_t)s * 8 + h) * 128 + v0) * 128 + dk;
#pragma unroll
      for (int j = 0; j < 64; j++) S[j] = sp[(size_t)j * 128];
    } else {
#pragma unroll
      for (int j = 0; j < 64; j++) S[j] = 0.f;
    }
    // S tile (bf16, [dk][v] sw128 rows) into buffer b
    auto store_S_tile = [&](int b) {
      unsigned char* sS = smem + R_S + b * 16384;
#pragma unroll
      for (int ch = 0; ch < 8; ch++) {
        uint4 w;
        w.x = pack_bf16(S[8 * ch + 0], S[8 * ch + 1]);
        w.y = pack_bf16(S[8 * ch + 2], S[8 * ch + 3]);
        w.z = pack_bf16(S[8 * ch + 4], S[8 * ch + 5]);
        w.w = pack_bf16(S[8 * ch + 6], S[8 * ch + 7]);
        *(uint4*)(sS + sw128(dk, ch)) = w;
      }
    };
    store_S_tile(0);
    fence_proxy_async();
    tc_fence_before();
    bar_named_arrive(1, 160);
#ifdef TIMING
    tlast = clock64();
    if (tid == 0) TRACE(4, tlast);
#endif
    for (int ci = 0; ci < nc; ci++) {
      if (tid == 0) WDP(7, ci);
      const uint32_t TM_C = TM_C0 + 64 * (ci & 1) + lrow;
      mbar_wait(barChain, ci & 1);       // chain_ci complete => W_ci landed => the loader's sgC[ci&7] store is visible
      tc_fence_after();
      STAMP(0);
      if (tid == 0) TRACE(ci * 16 + 5, tlast);
      if (tid == 0) WDP(15, ci * 4 + 1);
      const float gCc = sgC[ci & 7];
      uint32_t r[64];
      tc_ld32_nw(TM_C, r);
      tc_ld32_nw(TM_C + 32, r + 32);
      tc_wait_ld();
#pragma unroll
      for (int j = 0; j < 64; j++) S[j] = fmaf(S[j], gCc, __uint_as_float(r[j]));
#if defined(EXPT) && (EXPT & 16)
      if (0)
#endif
      { store_S_tile((ci + 1) & 1);
      fence_proxy_async(); }
      tc_fence_before();
      STAMP(1);
      if (ci + 1 < nc) bar_named_arrive(1, 160);
      if (tid == 0) WDP(15, ci * 4 + 2);
      if (tid == 0) TRACE((ci + 1) * 16 + 4, clock64());
      if (lane == 0) TRACE((ci + 1) * 16 + 8 + warp, clock64());
    }
#ifdef TIMING
    if (dbg && tid == 0) { for (int i = 0; i < 3; i++) dbg[bid * 32 + i] = tacc[i]; }
#endif
    float* dp = state_out + (((size_t)s * 8 + h) * 128 + v0) * 128 + dk;
#pragma unroll
    for (int j = 0; j < 64; j++) dp[(size_t)j * 128] = S[j];
  } else {
    // =============================== M warps (4..7) ===============================
    const int t = tid - 128;                          // 0..127; rows 0..63 are real tokens
    const uint32_t lrow = (uint32_t)(32 * (warp & 3)) << 16;
    // Mneg conversion: TM_M fp32 [dk][dk'] -> -bf16 pairs in TM_A[b]   (32 columns per piece)
    auto convert_Mneg = [&](int b) {
      const uint32_t dstA = TM_A0 + 64 * b + lrow;
#pragma unroll
      for (int p = 0; p < 4; p++) {
        uint32_t r[32], pk[16];
#if defined(EXPT) && (EXPT & 256)
        for (int j = 0; j < 32; j++) r[j] = j + p;
#else
        tc_ld32_nw(TM_M + lrow + 32 * p, r);
        tc_wait_ld();
#endif
#pragma unroll
        for (int j = 0; j < 16; j++) pk[j] = pack_bf16(-__uint_as_float(r[2 * j]), -__uint_as_float(r[2 * j + 1]));
        tc_st16(dstA + 16 * p, pk);
      }
      tc_wait_st();
      tc_fence_before();
    };
    mbar_wait(barMneg, 0);
    tc_fence_after();
    convert_Mneg(0);
    bar_named_arrive(2, 192);
#ifdef TIMING
    tlast = clock64();
#endif
    for (int ci = 0; ci < nc; ci++) {
      const int tok = tok0 + ci * 64 + t;
      if (warp == 4 && lane == 0) WDP(4, ci);
      uint4 oh[8];
      if (warp < 6) {
        const unsigned char* g = blobh + (size_t)ci * 8 * BLOB_BYTES + BL_O + hv * 8192 + t * 128;
#if defined(EXPT) && (EXPT & 32)
        for (int ch = 0; ch < 8; ch++) oh[ch] = make_uint4(t, ch, 0, 0);
#else
#pragma unroll
        for (int ch = 0; ch < 8; ch++) oh[ch] = ldg16(g + ((ch ^ (t & 7)) << 4));
#endif
      }
      STAMP(0);
      if (ci + 1 < nc) {
        mbar_wait(barMneg, (ci + 1) & 1);               // Mneg_{ci+1} done => (issuer B order) KU_ci, Mneg_ci done too
        tc_fence_after();
        STAMP(1);
        if (warp == 4 && lane == 0) prog[0] = ci + 1;
        if (warp == 4 && lane == 0) { WDP(14, ci * 4 + 1); WDP(12, ci + 1); }   // -> loader: K/W/U stage ci&3 free
        convert_Mneg((ci + 1) & 1);
        bar_named_arrive(2, 192);
        STAMP(2);
        if (warp == 4 && lane == 0) WDP(14, ci * 4 + 2);
      }
      mbar_wait((ci & 1) ? barEpi1 : barEpi, (ci >> 1) & 1);
      tc_fence_after();
      STAMP(3);
      if (warp == 4 && lane == 0) prog[1] = ci + 1;
      if (warp == 4 && lane == 0) { WDP(14, ci * 4 + 3); WDP(13, ci + 1); }   // -> loader: Q stage ci&1 free
      if (warp < 6) {
        const uint32_t TM_E = TM_E0 + 64 * (ci & 1) + lrow;
        uint32_t r[64];
#if defined(EXPT) && (EXPT & 128)
        for (int j = 0; j < 64; j++) r[j] = j + t;
#else
        tc_ld32_nw(TM_E, r);
        tc_ld32_nw(TM_E + 32, r + 32);
        tc_wait_ld();
#endif
#if defined(EXPT) && (EXPT & 64)
        if (tok < tok1 && r[0] == 0x7fc00001u) {
#else
        if (tok < tok1) {
#endif
          uint4* op = (uint4*)(out + ((size_t)tok * 8 + h) * 128 + v0);
#pragma unroll
          for (int ch = 0; ch < 8; ch++) {
            const uint32_t* rr = r + 8 * ch;
            uint32_t oo[4] = {oh[ch].x, oh[ch].y, oh[ch].z, oh[ch].w}, w[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; e2++)
              w[e2] = pack_bf16(bf16lo(oo[e2]) + __uint_as_float(rr[2 * e2]), bf16hi(oo[e2]) + __uint_as_float(rr[2 * e2 + 1]));
            op[ch] = make_uint4(w[0], w[1], w[2], w[3]);
          }
        }
      }
      STAMP(4);
    }
    if (warp == 4 && lane == 0) *sdone = 1;
#ifdef TIMING
    if (dbg && t == 0) { for (int i = 0; i < 5; i++) dbg[bid * 32 + 16 + i] = tacc[i]; }
#endif
  }
  tc_fence_before();
  __syncthreads();
}

__global__ void __launch_bounds__(384, 1)
gdn_recur_kernel(const __grid_constant__ CUtensorMap tmK, const unsigned char* __restrict__ blob,
                 const float* __restrict__ gC, const float* __restrict__ state_in, float* __restrict__ state_out,
                 bf16* __restrict__ out, uint32_t tmem_base, const __grid_constant__ GdnTabs tabs,
                 long long* __restrict__ dbg)
{
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  uint32_t* tmem_slot = (uint32_t*)(smem + R_BAR + 144);
  if (warp == 0) { tmem_alloc(smem_u32(tmem_slot), 512); tmem_relinquish(); }
  __syncthreads();
  // the CTA is alone on its SM (smem > half) and allocates the whole TMEM -> base is 0 == tmem_base
  if (*tmem_slot != tmem_base) { if (dbg) dbg[0] = -1; __trap(); }
  recur_item<false>(smem, tmem_base, blockIdx.x, &tmK, blob, gC, state_in, state_out, out, tabs, nullptr, dbg, false);
  if (warp == 0) tmem_dealloc(tmem_base, 512);
}

// ------------------------------------------------------------------------------------------------
// Fused persistent kernel: grid = #SMs, 320 threads, whole smem + TMEM per CTA.
//   phase 1: warps 0-7 form two independent 128-thread prologue lanes (smem windows of P_SMEM bytes, 256 TMEM
//            columns each) and process (chunk, head) items: CTAs [0, nA) take items 2*bid, 2*bid+1 (one each),
//            the others pull items from an atomic counter until all are claimed.  Each finished item publishes a
//            flag (= epoch) once its bulk stores have completed.
//   phase 2: CTAs [0, nA) run the recurrence of the nA longest-sequence items; the others pull the remaining
//            recurrence items from a second counter.  Recurrence loads of blob data wait on the flags.
//   Deadlock-free: prologue items never wait; recurrence items only wait for flags of items that are claimed
//   (phase 2 of a non-A CTA starts only once every item has been claimed, and nA < grid).
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(384, 1)
gdn_fused_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
                 const __grid_constant__ CUtensorMap tmV,
                 const bf16* __restrict__ a, const bf16* __restrict__ b,
                 const float* __restrict__ A_log, const float* __restrict__ dt_bias, float scale,
                 const __grid_constant__ GdnTabs tabs, unsigned char* __restrict__ blob, float* __restrict__ gC,
                 const float* __restrict__ state_in, float* __restrict__ state_out, bf16* __restrict__ out,
                 int* __restrict__ ctl, int* __restrict__ flags, uint32_t tmem_base, long long* __restrict__ dbg)
{
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int bid = blockIdx.x;
  uint32_t* tmem_slot = (uint32_t*)(smem + R_BAR + 144);
  volatile int* sitem = (int*)(smem + R_BAR + 160);   // [2 lanes][2 slots]: per-lane next item id (2-deep ring: a slow warp may still be reading the previous slot)
  if (warp == 0) { tmem_alloc(smem_u32(tmem_slot), 512); tmem_relinquish(); }
  __syncthreads();
  if (*tmem_slot != tmem_base) { if (dbg) dbg[0] = -1; __trap(); }
  const int epoch = tabs.epoch;
#ifdef TIMING
  long long* ft = dbg ? dbg + 640 * 32 + 20 * 1024 + MAXC * 8 * 24 + bid * 32 : nullptr;
  auto gtime = []() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  if (ft && tid == 0) ft[0] = gtime();
#define FT(i, v) { if (ft) ft[i] = (v); }
#else
#define FT(i, v)
#endif

  // ---------------- phase 1: prologue items ----------------
  if (tid == 0) WDP(0, 1);
  if (warp < 8) {
    const int pl = warp >> 2;
    unsigned char* lsm = smem + pl * P_SMEM;
    const uint32_t ltm = tmem_base + pl * 256;
    const int ltid = tid - pl * 128;
    const bool statik = bid < tabs.nA;
    // static CTAs: the 16 recurrence CTAs of the longest sequence run no prologue item (they start the recurrence
    // as soon as its chunk 0 is published); the next nsolo CTAs each run one leading item on lane 0 only (a solo item
    // finishes ~2x sooner than two lanes sharing the SM); the remaining static CTAs run two items each.
    int item;
    if (!statik) item = 0;
    else if (bid < 16) item = tabs.nitems;
    else if (bid < 16 + tabs.nsolo) item = pl == 0 ? bid - 16 : tabs.nitems;
    else item = tabs.nsolo + 2 * (bid - 16 - tabs.nsolo) + pl;
    if (!statik) {
      if (ltid == 0) sitem[2 * pl] = tabs.pool0 + atomicAdd(ctl, 1);
      bar_named_sync(1 + pl, 128);
      item = sitem[2 * pl];
    }
    int slot = 1;
    bool first = true;
    int* prev = nullptr;
    long long ptacc[24] = {};
    while (item < tabs.nitems) {
      const int c = tabs.border[item >> 3], hh = item & 7;
      // claim the next item early (its latency hides behind this item)
      if (ltid == 0) sitem[2 * pl + slot] = statik ? tabs.nitems : tabs.pool0 + atomicAdd(ctl, 1);
#ifdef WATCHDOG
      if (ltid == 0) WDP(1 + 4 * pl, item);
#endif
      if (pl == 0) prologue_item<1>(lsm, ltm, c, hh, ltid, &tmQ, &tmK, &tmV, a, b, A_log, dt_bias, scale, tabs, blob, gC, ptacc, !first, prev, epoch);
      else        prologue_item<2>(lsm, ltm, c, hh, ltid, &tmQ, &tmK, &tmV, a, b, A_log, dt_bias, scale, tabs, blob, gC, ptacc, !first, prev, epoch);
      prev = flags + (size_t)c * 8 + hh;
      item = sitem[2 * pl + slot];
      slot ^= 1;
      first = false;
#ifdef TIMING
      if (ltid == 0) { ft[2 + pl] += 1; ft[4 + pl] = gtime(); }
#endif
    }
    if (lane == 0) { bulk_wait0(); fence_proxy_async_global(); }
    bar_named_sync(1 + pl, 128);
    if (ltid == 0 && prev) st_release_gpu(prev, epoch);
    if (ltid == 0) WDP(1 + 4 * pl, -1);
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) FT(1, gtime());

  // ---------------- phase 2: recurrence items ----------------
  int r;
  if (bid < tabs.nA) r = bid;
  else {
    if (tid == 0) sitem[0] = tabs.nA + atomicAdd(ctl + 1, 1);
    __syncthreads();
    r = sitem[0];
  }
  bool reinit = false;
  if (tid == 0) WDP(0, 2);
  while (r < tabs.nrec) {
#ifdef TIMING
    if (tid == 0) { int k = (int)ft[6]; if (k < 8) { ft[8 + 3 * k] = r; ft[9 + 3 * k] = gtime(); } }
#endif
    if (tid == 0) WDP(2, r);
    recur_item<true>(smem, tmem_base, r, &tmK, blob, gC, state_in, state_out, out, tabs, flags, dbg, reinit);
#ifdef TIMING
    if (tid == 0) { int k = (int)ft[6]; if (k < 8) ft[10 + 3 * k] = gtime(); ft[6] = k + 1; }
#endif
    reinit = true;
    if (tid == 0) sitem[0] = tabs.nA + atomicAdd(ctl + 1, 1);
    __syncthreads();
    r = sitem[0];
  }
  // last CTA out resets the counters for the next launch
  if (tid == 0) {
    __threadfence();
    const int d = atomicAdd(ctl + 2, 1);
    if (d == (int)gridDim.x - 1) { ctl[0] = 0; ctl[1] = 0; __threadfence(); ctl[2] = 0; }
  }
  tc_fence_before();
  __syncthreads();
  if (tid == 0) WDP(0, 3);
  if (warp == 0) tmem_dealloc(tmem_base, 512);
}
// ------------------------------------------------------------------------------------------------
// host side
// ------------------------------------------------------------------------------------------------
static int make_tmap(CUtensorMap* tm, void* ptr, int rank, const cuuint64_t* dims, const cuuint64_t* strides_bytes,
                     const cuuint32_t* box, CUtensorMapSwizzle swz) {
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, ptr, dims, strides_bytes + 1, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

static int g_split = -1;
extern "C" void gdn_set_split(int v) { g_split = v; }
extern "C" int gdn_fwd_launch(const void* q, const void* k, const void* v, const void* state, const void* A_log,
                              const void* a, const void* dt_bias, const void* b, const int64_t* cu_host, int N, int T,
                              float scale, void* out, void* new_state,
                              void* ws_ptr, int64_t ws_bytes, cudaStream_t stream, int* nc_out, void* dbg_ptr) {
  std::vector<int> cseq, ct0, clen, cstart(N + 1), cu32(N + 1), order(N);
  int nc = 0;
  for (int s = 0; s < N; s++) {
    cstart[s] = nc;
    cu32[s] = (int)cu_host[s];
    int len = (int)(cu_host[s + 1] - cu_host[s]);
    for (int t = 0; t < len; t += 64) { cseq.push_back(s); ct0.push_back((int)cu_host[s] + t); clen.push_back(std::min(64, len - t)); nc++; }
  }
  cstart[N] = nc; cu32[N] = (int)cu_host[N];
  for (int s = 0; s < N; s++) order[s] = s;
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return (cu_host[x + 1] - cu_host[x]) > (cu_host[y + 1] - cu_host[y]); });
  *nc_out = nc;
  unsigned char* wp = (unsigned char*)ws_ptr;
  size_t off = 0;
  auto carve = [&](size_t bytes) { void* p = wp + off; off += (bytes + 1023) & ~(size_t)1023; return p; };
  unsigned char* blob = (unsigned char*)carve((size_t)nc * 8 * BLOB_BYTES);
  float* gC = (float*)carve((size_t)nc * 8 * 4);
  int* ctl = (int*)carve(1024);
  int* flags = (int*)carve((size_t)nc * 8 * 4);
  if ((int64_t)off > ws_bytes) return -100;
  if (N > MAX_SEQ) return -101;
  if (nc > MAXC) return -102;
  static GdnTabs tabs;
  for (int i = 0; i < nc; i++) { tabs.seq[i] = cseq[i]; tabs.t0[i] = ct0[i]; tabs.len[i] = clen[i]; }
  for (int i = 0; i < N; i++) tabs.order[i] = order[i];
  for (int i = 0; i <= N; i++) { tabs.cu[i] = cu32[i]; tabs.cstart[i] = cstart[i]; }
  // fused schedule: K longest sequences (>= 8 chunks, at most 4) get static recurrence CTAs; their chunks come
  // first in the prologue order (interleaved by chunk index), then the remaining chunks by sequence length desc.
  int K = 0;
  while (K < N && K < 4 && cstart[order[K] + 1] - cstart[order[K]] >= 8) K++;
  {
    // earliest-deadline-first: chunk ci of a sequence with ncs chunks gets key ci + (ncmax - ncs), i.e. every
    // sequence's recurrence is scheduled to finish together with the longest one; ties -> longer sequence first.
    const int ncmax = cstart[order[0] + 1] - cstart[order[0]];
    int nb = 0;
    for (int key = 0; key < ncmax; key++)
      for (int i = 0; i < N; i++) {
        const int sq = order[i], ncs = cstart[sq + 1] - cstart[sq], ci = key - (ncmax - ncs);
        if (ci >= 0 && ci < ncs) tabs.border[nb++] = cstart[sq] + ci;
      }
  }
  tabs.nA = K * 16; tabs.nrec = N * 16; tabs.nitems = nc * 8;
  {
    static int nsolo_env = getenv("GDN_NSOLO") ? atoi(getenv("GDN_NSOLO")) : 12;   // swept 0/4/8/12/16/24/32/48 on the workload: 8-16 fastest (0.0883-0.0884 ms vs 0.0895 at 32)
    int ns = tabs.nA > 16 ? tabs.nA - 16 : 0;          // static CTAs available for solo items
    if (ns > nsolo_env) ns = nsolo_env;
    if (ns > tabs.nitems) ns = tabs.nitems;
    tabs.nsolo = ns;
    tabs.pool0 = tabs.nA > 16 ? ns + 2 * (tabs.nA - 16 - ns) : 0;
    if (tabs.pool0 > tabs.nitems) tabs.pool0 = tabs.nitems;
  }
  static int epoch = 0; epoch++; tabs.epoch = epoch;
  static void* last_ws = nullptr; static int last_nc = -1;
  if (ws_ptr != last_ws || nc != last_nc) {
    cudaMemsetAsync(ctl, 0, 1024 + (((size_t)nc * 8 * 4 + 1023) & ~(size_t)1023), stream);
    last_ws = ws_ptr; last_nc = nc;
  }
  static int num_sms = 0;
  if (!num_sms) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev); }

  CUtensorMap tmQ, tmK, tmV;
  {
    cuuint64_t dims[3] = {128, 4, (cuuint64_t)T};
    cuuint64_t str[3] = {2, 256, 1024};
    cuuint32_t box[3] = {64, 1, 64};
    if (make_tmap(&tmQ, (void*)q, 3, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B)) return -2;
    if (make_tmap(&tmK, (void*)k, 3, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B)) return -3;
  }
  {
    cuuint64_t dims[3] = {128, 8, (cuuint64_t)T};
    cuuint64_t str[3] = {2, 256, 2048};
    cuuint32_t box[3] = {64, 1, 64};
    if (make_tmap(&tmV, (void*)v, 3, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B)) return -4;
  }
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_prologue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, P_SMEM);
    cudaFuncSetAttribute(gdn_recur_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R_SMEM);
    cudaFuncSetAttribute(gdn_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R_SMEM);
    attr_set = true;
  }
  if (g_split < 0) g_split = getenv("GDN_SPLIT") ? atoi(getenv("GDN_SPLIT")) : 0;
  const int split = g_split;
  if (!split) {
#ifdef WATCHDOG
    if (!h_wd) { cudaHostAlloc((void**)&h_wd, 262144, cudaHostAllocMapped); int* dp; cudaHostGetDevicePointer((void**)&dp, h_wd, 0); cudaMemcpyToSymbol(g_wdp, &dp, sizeof(dp)); }
    static int wd_sync = getenv("GDN_WD_SYNC") ? atoi(getenv("GDN_WD_SYNC")) : 1;
    if (wd_sync) memset(h_wd, 0, 262144);
    g_wd_stream = stream; g_wd_nsms = num_sms; g_wd_epoch = tabs.epoch;
#endif
    gdn_fused_kernel<<<dim3(num_sms), 384, R_SMEM, stream>>>(tmQ, tmK, tmV, (const bf16*)a, (const bf16*)b,
        (const float*)A_log, (const float*)dt_bias, scale, tabs, blob, gC, (const float*)state, (float*)new_state,
        (bf16*)out, ctl, flags, 0u, (long long*)dbg_ptr);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "launch error: %s\n", cudaGetErrorString(e)); return -10; }
#ifdef WATCHDOG
    if (!split && wd_sync) {
      auto t0 = std::chrono::steady_clock::now(); bool done = false;
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 3.0) { if (cudaStreamQuery(stream) == cudaSuccess) { done = true; break; } }
      if (!done) {
        fprintf(stderr, "WATCHDOG: kernel hung. epoch %d nA %d nitems %d nrec %d. CTAs not done:\n", tabs.epoch, tabs.nA, tabs.nitems, tabs.nrec);
        for (int b = 0; b < num_sms; b++) { const int* w = h_wd + 1024 + b * 16; if (w[0] != 3)
          fprintf(stderr, "  bid %3d phase %d | p1 items lane0 %4d lane1 %4d pstage %2d | rec item %4d  Mci %3d  Lci %3d  Sci %3d\n", b, w[0], w[1], w[5], w[3], w[2], w[4], w[6], w[7]); }
        std::vector<int> hf(nc * 8); if (cudaMemcpyAsync(hf.data(), flags, nc * 8 * 4, cudaMemcpyDeviceToHost, 0) == cudaSuccess) {}
        exit(1);
      }
    }
    if (!wd_sync) return 0;
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "WATCHDOG: %s  bid %d tid %d info %d kind %d\n", cudaGetErrorString(e), h_wd[0], h_wd[1], h_wd[2], h_wd[4]); return -11; }
    if (!split) {
      std::vector<int> hf(nc * 8); cudaMemcpy(hf.data(), flags, nc * 8 * 4, cudaMemcpyDeviceToHost);
      int miss = 0;
      for (int i = 0; i < nc * 8; i++) if (hf[i] != tabs.epoch) {
        // find item with border[item>>3] == chunk i/8 and head i%8
        int it = -1; for (int j = 0; j < tabs.nitems; j++) if (tabs.border[j >> 3] == i / 8 && (j & 7) == (i % 8)) { it = j; break; }
        if (miss < 24) fprintf(stderr, "  missing flag chunk %d head %d (val %d, epoch %d): item %d rec bid %d pl %d stage %d prevflag %d\n", i / 8, i % 8, hf[i], tabs.epoch, it, h_wd[16 + it * 4], h_wd[16 + it * 4 + 1], h_wd[16 + it * 4 + 2], h_wd[16 + it * 4 + 3]);
        miss++;
      }
      fprintf(stderr, "WATCHDOG report: %d missing flags, %d flag give-ups\n", miss, h_wd[5]);
      for (int k = 0; k < h_wd[5] && k < 16; k++) fprintf(stderr, "  give-up: bid %d chunk %d head %d nc %d\n", h_wd[8192 + k * 4], h_wd[8192 + k * 4 + 1], h_wd[8192 + k * 4 + 2], h_wd[8192 + k * 4 + 3]);
    }
#endif
    return 0;
  }
#ifdef TIMING
  static int pro_only = getenv("GDN_PRO_ONLY") ? atoi(getenv("GDN_PRO_ONLY")) : 0;
  if (pro_only) {
    gdn_prologue_kernel<<<dim3(pro_only, 8), 128, P_SMEM, stream>>>(tmQ, tmK, tmV, (const bf16*)a, (const bf16*)b,
      (const float*)A_log, (const float*)dt_bias, scale, tabs, blob, gC, dbg_ptr ? (long long*)dbg_ptr + 640 * 32 + 20 * 1024 : nullptr);
    return 0;
  }
#endif
  gdn_prologue_kernel<<<dim3(nc, 8), 128, P_SMEM, stream>>>(tmQ, tmK, tmV, (const bf16*)a, (const bf16*)b,
      (const float*)A_log, (const float*)dt_bias, scale, tabs, blob, gC, dbg_ptr ? (long long*)dbg_ptr + 640 * 32 + 20 * 1024 : nullptr);
  gdn_recur_kernel<<<dim3(N * 16), 384, R_SMEM, stream>>>(tmK, blob, gC, (const float*)state, (float*)new_state,
      (bf16*)out, 0u, tabs, (long long*)dbg_ptr);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "launch error: %s\n", cudaGetErrorString(e)); return -10; }
  return 0;
}

extern "C" int64_t gdn_fwd_workspace_bytes(const int64_t* cu_host, int N) {
  int nc = 0;
  for (int s = 0; s < N; s++) { int len = (int)(cu_host[s + 1] - cu_host[s]); nc += (len + 63) / 64; }
  size_t b = 0;
  auto add = [&](size_t x) { b += (x + 1023) & ~(size_t)1023; };
  add((size_t)nc * 8 * BLOB_BYTES); add((size_t)nc * 8 * 4); add(1024); add((size_t)nc * 8 * 4);
  return (int64_t)b + 4096;
}
