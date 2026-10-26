// GDN (gated delta net) prefill, chunked WY form, written from scratch for B200 (sm_100a). v6.
// Layout: q/k [T][4][128] bf16, v [T][8][128] bf16, a/b [T][8] bf16, A_log/dt_bias [8] f32,
// state/new_state [N][8][128(v)][128(k)] f32 (k-last), cu_seqlens [N+1] int64, out [T][8][128] bf16.
//
// Math per chunk (C=64 tokens, gamma = in-chunk cumulative log decay, beta = sigmoid(b)):
//   A_ij = beta_i e^{gamma_i-gamma_j} (k_i.k_j)  (j<i)   ;  (I+A) W = diag(beta e^gamma) K ;  (I+A) U = diag(beta) V
//   Vnew = U - W S ; O = scale e^gamma Q S + tril(scale QK^T e^{gamma_i-gamma_j}) Vnew ; S' = e^{gamma_C} S + (K e^{gamma_C-gamma})^T Vnew
//
// v20: fused chain step. With M = -K'^T W (K' = K e^{gamma_C-gamma}) and Q'' = scale e^gamma Q - Att W, both precomputed per
// chunk by the prep kernel, the recurrence and the output become one matmul each with no dependent elementwise step:
//   S' = e^{gamma_C} S + [M | K'^T] [S ; U]        (M=128 d, N=64 v, K=128+64)
//   O  = [Q'' | Att] [S ; U]                        (M=64 tok, N=64 v, K=128+64)
// Kernel 1 (prep): one CTA per (chunk, q-head), 256 threads, 2 CTAs/SM. Gates, KQ = [K;Q]K^T (UMMA) -> A, Att; T = (I+A)^-1
//   per v-head (diagonal 16x16 blocks scalar fp32, off-diagonal mma.sync); U = (T diag(beta)) V, W = (T diag(beta e^gamma)) K^T,
//   M = -K'^T W, Q'' = scale e^gamma Q - Att W (all UMMA). Writes M, K'^T, U, Q'', Att, gam images per (chunk, v-head).
// Kernel 2 (recurrence): one CTA per (sequence rank, v-head, 64-column V half), 352 threads: warps 0-7 compute (lane 0 of
//   warps 0-3 issue the chain UMMA), warp 8 producer (bulk loads, 2-deep chain ring + 3-deep output ring), warps 9-10 issue
//   the output UMMA. State fp32 in registers, bf16 copy double-buffered in smem as the B operand.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define DEVI __device__ __forceinline__

namespace {

constexpr int HQ = 4, HV = 8, D = 128, CH = 64;
constexpr int NV = 64;                    // V columns per recurrence CTA (half of a v-head)
constexpr int P64 = 144;                  // byte pitch of [rows][64] bf16 tiles (A_bf16 in prep, U halves)
constexpr int P32 = 80;                   // byte pitch of [rows][32] bf16 tiles (prep M staging)
constexpr int TD_PITCH = 48;              // [64][16] bf16 diagonal-inverse tile
// [rows][128] bf16 tile layouts (rows*256 B):
//   L_SW: two halves (cols 0-63 / 64-127) of [rows][128 B], 16 B chunk XOR (row & 7)  == TMA SWIZZLE_128B image
//   L_MW: [16][32] bf16, 64 B rows, 16 B chunk XOR ((row >> 1) & 3)   (per-warp staging, 1 KB)
constexpr int L_SW = -1, L_MW = -3;
// per (chunk, v-head) images produced by prep (all SWIZZLE_128B, 1 KB aligned):
constexpr int M_BYTES = 2 * D * 128;              // 32768: M = -K'^T W, [2 k-atoms (d' 0-63 / 64-127)][128 d][128 B]  (K-major A operand)
constexpr int KT_BYTES = D * 128;                 // 16384: K'^T = K^T e^{gamma_C-gamma_t}, [128 d][128 B = 64 tokens]   (K-major A operand)
constexpr int U_BYTES = 2 * CH * 128;             // 16384: U, [2 halves (v 0-63 / 64-127)][64 tok][128 B]            (MN-major B operand)
constexpr int Q2_BYTES = CH * 256;                // 16384: Q'' = scale e^gamma Q - Att W, [2 k-atoms (d)][64 tok][128 B] (K-major A operand)
constexpr int ATT_BYTES = CH * 128;               // 8192 : Att = scale QK^T e^{gamma_i-gamma_j} (i>=j), [64][128 B]  (K-major A operand)
constexpr int GAM_BYTES = CH * 4;                 // 256  : in-chunk cumulative log decay per token

struct GdnParams {
  const __nv_bfloat16* q; const __nv_bfloat16* k; const __nv_bfloat16* v;
  const float* state;  // may be null
  const float* A_log; const __nv_bfloat16* a; const float* dt_bias; const __nv_bfloat16* b;
  const long long* cu; float scale;
  __nv_bfloat16* out; float* new_state;
  unsigned char* M; unsigned char* KT; unsigned char* U; unsigned char* Q2; unsigned char* Att; float* gam; int* order;
  int N; int T; int maxC;
  long long* dbg;
};
struct TmaMaps { CUtensorMap q, k, v, o; };
DEVI long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#ifdef PHASE_TIMERS
#define PH_T(cond, slot) do { if (phd && (cond)) { long long _t = clock64(); phacc[(slot) & 15] += _t - _ph_prev; _ph_prev = _t; } } while (0)
#else
#define PH_T(cond, slot) do { } while (0)
#endif
DEVI int smid() { int r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
DEVI void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
// ---- mbarrier + bulk async copies
DEVI void mbar_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(count) : "memory");
}
DEVI void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
DEVI void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes) : "memory");
}
DEVI void mbar_arrive(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(addr) : "memory");
}
DEVI uint32_t mbar_try_wait(uint32_t addr, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred P;\n mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.u32 %0, 1, 0, P;\n}\n"
               : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
  return ok;
}
DEVI uint32_t mbar_test_wait(uint32_t addr, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred P;\n mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.u32 %0, 1, 0, P;\n}\n"
               : "=r"(ok) : "r"(addr), "r"(parity) : "memory");
  return ok;
}
#ifdef MBAR_DEBUG
DEVI void mbar_wait(uint32_t addr, uint32_t parity) {
  long long n = 0;
  while (!mbar_try_wait(addr, parity)) { if (++n > (1ll << 24)) { printf("MBAR TIMEOUT block %d warp %d lane %d addr 0x%x parity %u\n", blockIdx.x, threadIdx.x >> 5, threadIdx.x & 31, addr, parity); __trap(); } }
}
#else
DEVI void mbar_wait(uint32_t addr, uint32_t parity) { while (!mbar_try_wait(addr, parity)) {} }
#endif
// non-suspending spin (lower wake-up latency; used on the recurrence chain)
DEVI void mbar_spin(uint32_t addr, uint32_t parity) { while (!mbar_test_wait(addr, parity)) {} }
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
DEVI void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(dst), "r"(src), "r"(bytes) : "memory");
}
DEVI uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEVI uint32_t mapa_own(uint32_t addr, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r; }
DEVI void st_async16(uint32_t dst, const uint4& v, uint32_t mbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(mbar) : "memory");
}
DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
DEVI void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
DEVI void bulk_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory"); }
DEVI void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
DEVI void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t mbar, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
      ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(mbar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
DEVI void tma_store_3d(const CUtensorMap* map, uint32_t src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];\n"
               ::"l"(reinterpret_cast<uint64_t>(map)), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
// full [64][128] tile of head `head`, rows tok0..tok0+63: half 0 at dst, half 1 at dst + half_stride
DEVI void tma_load_tile(uint32_t dst, const CUtensorMap* map, uint32_t mbar, int head, int tok0, uint32_t half_stride = 8192) {
  tma_load_3d(dst, map, mbar, 0, head, tok0);
  tma_load_3d(dst + half_stride, map, mbar, 64, head, tok0);
}
// ---- tcgen05
DEVI void tmem_alloc(uint32_t dst_smem, uint32_t ncols) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory"); }
DEVI void tmem_relinquish() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
DEVI void tmem_dealloc(uint32_t taddr, uint32_t ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory"); }
DEVI void umma_f16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t enable_d) {
  asm volatile("{\n\t.reg .pred P;\n\tsetp.ne.b32 P, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t}"
               :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d) : "memory");
}
DEVI void umma_commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
DEVI void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DEVI void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
DEVI void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                 "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(taddr));
}
DEVI void tmem_st32(uint32_t taddr, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                  "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
                  "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
}
DEVI void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DEVI void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// UMMA shared-memory descriptor: SWIZZLE_128B canonical layouts (8-row groups 1024 B apart)
DEVI uint64_t mkdesc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)(16 >> 4) << 16;         // LBO (unused for single-atom / K-major swizzled layouts)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 46;                 // version
  d |= (uint64_t)2 << 61;                 // SWIZZLE_128B
  return d;
}
// instruction descriptor: bf16 inputs, f32 accumulate, a_mn/b_mn = 1 for MN-major (N/M contiguous) operands
DEVI constexpr uint32_t mkidesc(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

DEVI uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
DEVI float2 unpack_bf16(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}
DEVI float softplus_f(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
DEVI float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }

// byte offset of element (row, col) in a tile with layout L (pitch in bytes, or L_SW / L_MW)
template <int L>
DEVI int toff(int row, int col) {
  if (L == L_SW) return ((col >> 6) << 13) + row * 128 + (((((col >> 3) & 7) ^ (row & 7))) << 4) + ((col & 7) << 1);
  if (L == L_MW) return row * 64 + ((((col >> 3) ^ ((row >> 1) & 3))) << 4) + ((col & 7) << 1);
  return row * L + col * 2;
}
// swizzled [rows][128 B] offset (one 64-element-wide half): 16 B chunk XOR (row & 7)
DEVI int offsw(int row, int col) { return row * 128 + (((((col >> 3) & 7) ^ (row & 7))) << 4) + ((col & 7) << 1); }

template <int L>
DEVI void load_a(uint32_t* a, const unsigned char* tile, int m0, int k0, int lane) {
  int row = m0 + (lane & 7) + ((lane >> 3) & 1) * 8;
  int col = k0 + (lane >> 4) * 8;
  ldsm_x4(a, smem_u32(tile + toff<L>(row, col)));
}
template <int L>
DEVI void load_b_t(uint32_t* b, const unsigned char* tile, int n0, int k0, int lane) {
  int mi = lane >> 3;
  int krow = k0 + (mi & 1) * 8 + (lane & 7);
  int ncol = n0 + (mi >> 1) * 8;
  ldsm_x4_t(b, smem_u32(tile + toff<L>(krow, ncol)));
}
template <int L>
DEVI void load_b_n(uint32_t* b, const unsigned char* tile, int n0, int k0, int lane) {
  int mi = lane >> 3;
  int nrow = n0 + (mi >> 1) * 8 + (lane & 7);
  int kcol = k0 + (mi & 1) * 8;
  ldsm_x4(b, smem_u32(tile + toff<L>(nrow, kcol)));
}

// ----------------------------------------------------------------------------------------------
// Kernel 1 (v20): per-chunk preparation on UMMA. One CTA per (global chunk, q-head), both v-heads. 256 threads, 2 CTAs/SM.
// ----------------------------------------------------------------------------------------------
constexpr int PP_OFF_AKQ = 0;        // 32768: [K h0 | Q h0 | K h1 | Q h1] (stacked A tile [K;Q]); K slots (+0,+16384) -> V1 halves -> W1 image halves
constexpr int PP_OFF_V = 32768;      // 16384: V0 (TMA halves +0/+8192) -> W0 image halves
constexpr int PP_OFF_KT = 49152;     // 16384: K^T unscaled [128 d][128 B] (B operand of W) -> Att0 | Att1 (after W)
constexpr int PP_OFF_AB1 = 65536;    // 9216 : A bf16 v-head 1 (pitch 144); prologue cu cache   } -> K'^T_1 image (16384 @ 65536, after W)
constexpr int PP_OFF_AB0 = 74752;    // 9216 : A bf16 v-head 0                                   }
constexpr int PP_OFF_T = 83968;      // 16384: T stack [128 rows][128 B] (rows 0-63 vh0, 64-127 vh1): T -> T diag(beta) -> T' ; -> K'^T_0 image (after W)
constexpr int PP_OFF_AD = 100352;    // 8192 : Ad f32 [2][4][16][16] ; later per-warp M staging (8 x 1 KB)
constexpr int PP_OFF_G = 108544;     // 3072 : gam, bet, gexp, feg, fkt, fq  [2][64] each
constexpr int PP_OFF_MB = PP_OFF_G + 6 * 512;   // 111616: mbarriers: tma(+0) kq(+8) u[4](+16..) w(+48) m(+56) aw(+64) ; tmem slot (+72)
constexpr int PREP_SMEM = PP_OFF_MB + 128;      // 111744
constexpr int PP_V1 = PP_OFF_AKQ;               // V1 halves land in the K slots (+0, +16384) after the KQ UMMA
constexpr uint32_t IDESC_KQ = mkidesc(128, 64, 0, 0);   // [K;Q] . K^T : A K-major, B K-major (K tile rows = tokens)
constexpr uint32_t IDESC_U = mkidesc(64, 64, 0, 1);     // T . V (per 64-col half) : A K-major, B MN-major ; also Att . W
constexpr uint32_t IDESC_W = mkidesc(128, 128, 0, 0);   // T' . K^T : both K-major
constexpr uint32_t IDESC_M = mkidesc(128, 64, 0, 1);    // K'^T . W (per 64-col half of d') : A K-major, B MN-major

__global__ void __launch_bounds__(256, 2) gdn_prep_kernel(const __grid_constant__ GdnParams p,
                                                          const __grid_constant__ TmaMaps maps) {
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* AKQ = smem + PP_OFF_AKQ;
  unsigned char* Vb = smem + PP_OFF_V;
  unsigned char* KT = smem + PP_OFF_KT;
  unsigned char* Ab0 = smem + PP_OFF_AB0;
  unsigned char* Ab1 = smem + PP_OFF_AB1;
  unsigned char* Tt = smem + PP_OFF_T;
  float* Ad = reinterpret_cast<float*>(smem + PP_OFF_AD);
  unsigned char* Mst = smem + PP_OFF_AD;
  float* gam = reinterpret_cast<float*>(smem + PP_OFF_G);   // [2][64]
  float* bet = gam + 128;                                   // [2][64]
  float* gexp = gam + 256;                                  // [2][64] e^{logg_t}
  float* feg = gam + 384;                                   // [2][64] e^{gamma_t}
  float* fkt = gam + 512;                                   // [2][64] e^{gamma_C - gamma_t}
  float* fq = gam + 640;                                    // [2][64] scale e^{gamma_t}
  const uint32_t mb_tma = smem_u32(smem + PP_OFF_MB), mb_kq = mb_tma + 8, mb_u0 = mb_tma + 16, mb_w = mb_tma + 48, mb_m = mb_tma + 56, mb_aw = mb_tma + 64;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + PP_OFF_MB + 72);
  long long* cu_s = reinterpret_cast<long long*>(Ab1);   // temporary cu_seqlens cache (up to 1152 entries)

  const long long* __restrict__ cu = p.cu;
  const int N = p.N;
  const float scale = p.scale;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const int gc = bid >> 2, qh = bid & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int wq = warp & 3, ch = warp >> 2;
  const int L = 32 * wq + lane;

  long long* dbg = p.dbg;
  long long* phd = (p.dbg && bid == 4) ? p.dbg + 4096 * 12 : nullptr;
#ifdef PHASE_TIMERS
  long long phacc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) phacc[i] = 0;
  long long _ph_prev = clock64();
#endif

  const bool cu_cached = (N + 1) <= 1152;
  if (cu_cached) for (int i = tid; i <= N; i += 256) cu_s[i] = cu[i];
  PH_T(tid == 0, 12);  // cu cache
  if (bid == 0) {  // sequence order by length (descending), stable
    int* order = p.order;
    for (int i = tid; i < N; i += 256) {
      long long li = cu[i + 1] - cu[i];
      int r = 0;
      for (int j = 0; j < N; j++) {
        long long lj = cu[j + 1] - cu[j];
        r += (lj > li) || (lj == li && j < i);
      }
      order[r] = i;
    }
  }
  if (tid == 0) {
    if (smem_u32(smem) & 1023) __trap();
    mbar_init(mb_tma, 1); mbar_init(mb_kq, 1); mbar_init(mb_w, 1); mbar_init(mb_m, 1); mbar_init(mb_aw, 1);
    for (int i = 0; i < 4; i++) mbar_init(mb_u0 + 8 * i, 1);
    fence_mbar_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(tmem_slot), 256); tmem_relinquish(); }
  __syncthreads();
  tc_fence_after();
  PH_T(tid == 0, 13);  // mbar init + tmem alloc
  const uint32_t taddr = *tmem_slot;
  int c0 = 0, n = 0, found = 0;
  {
    int base = 0;
    for (int s2 = 0; s2 < N; s2++) {
      long long st = cu_cached ? cu_s[s2] : cu[s2], en = cu_cached ? cu_s[s2 + 1] : cu[s2 + 1];
      int len = (int)(en - st);
      int cnt = len > 0 ? (len + CH - 1) / CH : 0;
      if (gc < base + cnt) { c0 = (int)st + (gc - base) * CH; n = min(CH, (int)en - c0); found = 1; break; }
      base += cnt;
    }
  }
  if (!found) {
    tc_fence_before();
    __syncthreads();
    if (warp == 0) tmem_dealloc(taddr, 256);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // recurrence CTAs may become resident as SMs free up
  if (tid == 0) {
    mbar_expect_tx(mb_tma, 3 * 16384);
    tma_load_tile(smem_u32(AKQ), &maps.k, mb_tma, qh, c0, 16384);           // K halves -> +0, +16384
    tma_load_tile(smem_u32(AKQ) + 8192, &maps.q, mb_tma, qh, c0, 16384);    // Q halves -> +8192, +24576
    tma_load_tile(smem_u32(Vb), &maps.v, mb_tma, 2 * qh, c0);               // V0 halves -> +0, +8192
  }
  // gate inputs (global latency overlaps the TMA loads)
  float ga = 0.f, gbv = 0.f, alog = 0.f, dtb = 0.f;
  if (tid < 128) {
    int vl = tid >> 6, t = tid & 63, vh = 2 * qh + vl;
    if (t < n) {
      ga = __bfloat162float(p.a[(size_t)(c0 + t) * HV + vh]);
      gbv = __bfloat162float(p.b[(size_t)(c0 + t) * HV + vh]);
    }
    alog = p.A_log[vh]; dtb = p.dt_bias[vh];
  }
  PH_T(tid == 0, 14);  // chunk search + TMA issue + gate loads issued
  __syncthreads();   // cu_s reads done (Ab1 region reused below)
  PH_T(tid == 0, 15);  // gate loads landed
  if (dbg && tid == 0) { dbg[bid * 4 + 0] = gtimer(); dbg[bid * 4 + 2] = smid(); }
  // ---- gates for both v-heads
  float lg_raw = 0.f;
  if (tid < 128) {
    int vl = tid >> 6, t = tid & 63;
    float lg = 0.f, bt = 0.f;
    if (t < n) {
      lg = -__expf(alog) * softplus_f(ga + dtb);
      bt = sigmoid_f(gbv);
    }
    lg_raw = lg;
    bet[vl * 64 + t] = bt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float tt = __shfl_up_sync(0xffffffffu, lg, o);
      if (lane >= o) lg += tt;
    }
    gam[vl * 64 + t] = lg;
  }
  __syncthreads();
  if (tid < 128 && (tid & 63) >= 32) gam[tid] += gam[(tid >> 6) * 64 + 31];
  __syncthreads();
  if (tid < 128) {
    const int vl = tid >> 6;
    const float gt = gam[tid], gC = gam[vl * 64 + 63];
    gexp[tid] = __expf(lg_raw);
    feg[tid] = __expf(gt);
    fkt[tid] = __expf(gC - gt);
    fq[tid] = scale * __expf(gt);
  }
  // zero the T stack (needed later) while the tiles are in flight
  for (int e = tid; e < 1024; e += 256) *reinterpret_cast<uint4*>(Tt + e * 16) = make_uint4(0, 0, 0, 0);
  PH_T(tid == 0, 0);  // prologue + gates
  mbar_wait(mb_tma, 0);
  __syncthreads();
  PH_T(tid == 0, 1);  // tile wait
  auto offK = [&](int t, int d) { return ((d >> 6) << 14) + offsw(t, d & 63); };
  // ---- K^T (unscaled) into smem via ldmatrix.trans of 8x8 blocks: warp w handles token block tb = w
  {
    const int tb = warp;
#pragma unroll
    for (int q4 = 0; q4 < 4; q4++) {
      const int db0 = q4 * 4;
      uint32_t r[4];
      ldsm_x4_t(r, smem_u32(AKQ + offK(8 * tb + (lane & 7), 8 * (db0 + (lane >> 3)))));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int d = 8 * (db0 + j) + (lane >> 2);
        const int t = 8 * tb + 2 * (lane & 3);
        *reinterpret_cast<uint32_t*>(KT + offsw(d, t)) = r[j];
      }
    }
  }
  fence_proxy_async();
  __syncthreads();
  PH_T(tid == 0, 2);  // K^T image
  // ---- KQ = [K;Q] K^T  (M=128, N=64, K=128) -> TMEM cols 0-63
  if (tid == 0) {
    const uint64_t dA = mkdesc(smem_u32(AKQ));
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      const uint64_t dd = dA + 1024 * (ks >> 2) + 2 * (ks & 3);
      umma_f16(taddr, dd, dd, IDESC_KQ, ks > 0 ? 1u : 0u);
    }
    umma_commit(mb_kq);
  }
  mbar_wait(mb_kq, 0);
  tc_fence_after();
  __syncthreads();
  if (tid == 0) {   // V1 -> the (now free) K slots
    mbar_expect_tx(mb_tma, 16384);
    tma_load_tile(smem_u32(smem + PP_V1), &maps.v, mb_tma, 2 * qh + 1, c0, 16384);
  }
  PH_T(tid == 0, 3);  // KQ UMMA
  // ---- elementwise from TMEM rows: lanes 0-63 = KK rows i, lanes 64-127 = QK rows (token i = L-64); cols 32ch..32ch+31
  float kq[32];
  {
    uint32_t r[32];
    const uint32_t tl = taddr + ((uint32_t)(32 * wq) << 16) + 32 * ch;
    tmem_ld32(tl, r);
    tmem_wait_ld();
#pragma unroll
    for (int i = 0; i < 32; i++) kq[i] = __uint_as_float(r[i]);
  }
  const bool kkrow = (L < 64);
  const int irow = kkrow ? L : L - 64;
  const int jbase = 32 * ch;
  // decay factors dec[j] = e^{gamma_i - gamma_j} for this thread's 32 columns: 4 running products of 8 (one exp each)
  auto make_dec = [&](int vl, float* dec) {
    const float* gm = gam + vl * 64;
    float ge[32];
#pragma unroll
    for (int q = 0; q < 8; q++) {
      float4 g4 = *reinterpret_cast<const float4*>(gexp + vl * 64 + jbase + 4 * q);
      ge[4 * q] = g4.x; ge[4 * q + 1] = g4.y; ge[4 * q + 2] = g4.z; ge[4 * q + 3] = g4.w;
    }
    const float gi = gm[irow];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int j0 = jbase + 8 * q;
      const int jtop = min(irow, j0 + 7);                  // highest column with j <= irow in this block
      float d = (jtop >= j0) ? __expf(gi - gm[jtop]) : 0.f;
#pragma unroll
      for (int e = 7; e >= 0; e--) {
        const int j = j0 + e;
        dec[8 * q + e] = (j <= jtop) ? d : 0.f;
        if (j <= jtop) d *= ge[8 * q + e];                 // dec(i, j-1) = dec(i, j) * g_j
      }
    }
  };
  auto write_att = [&](int vl) {
    if (!kkrow) {
      float dec[32];
      make_dec(vl, dec);
      const float sc = (irow < n) ? scale : 0.f;
      unsigned char* Attg = Tt + vl * ATT_BYTES;   // staged in the T region (unused until Tdiag), bulk-stored below
      uint4 pk[4];
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint32_t* pp = reinterpret_cast<uint32_t*>(&pk[c4]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          pp[e] = pack_bf16(sc * dec[8 * c4 + 2 * e] * kq[8 * c4 + 2 * e], sc * dec[8 * c4 + 2 * e + 1] * kq[8 * c4 + 2 * e + 1]);
        *reinterpret_cast<uint4*>(Attg + offsw(irow, jbase + 8 * c4)) = pk[c4];
      }
    }
  };
  auto write_ab = [&](int vl, unsigned char* Abt) {
    if (kkrow) {
      float dec[32];
      make_dec(vl, dec);
      const float bi = bet[vl * 64 + irow];
      // strictly lower: zero the diagonal element (j == irow)
      if (irow >= jbase && irow < jbase + 32) {
#pragma unroll
        for (int e = 0; e < 32; e++) if (jbase + e == irow) dec[e] = 0.f;
      }
      uint4 pk[4];
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint32_t* pp = reinterpret_cast<uint32_t*>(&pk[c4]);
#pragma unroll
        for (int e = 0; e < 4; e++)
          pp[e] = pack_bf16(bi * dec[8 * c4 + 2 * e] * kq[8 * c4 + 2 * e], bi * dec[8 * c4 + 2 * e + 1] * kq[8 * c4 + 2 * e + 1]);
        *reinterpret_cast<uint4*>(Abt + irow * P64 + (jbase + 8 * c4) * 2) = pk[c4];
      }
      const int blk = irow >> 4;
      if ((blk >> 1) == ch) {
        float* Adv = Ad + (vl * 4 + blk) * 256 + (irow & 15) * 16;
        if (blk & 1) {
#pragma unroll
          for (int e = 0; e < 16; e++) Adv[e] = bi * dec[16 + e] * kq[16 + e];
        } else {
#pragma unroll
          for (int e = 0; e < 16; e++) Adv[e] = bi * dec[e] * kq[e];
        }
      }
    }
  };
  write_att(0);
  write_att(1);
  fence_proxy_async();
  write_ab(0, Ab0);
  write_ab(1, Ab1);
  __syncthreads();
  if (tid == 0) {
    bulk_s2g(p.Att + ((size_t)gc * HV + 2 * qh) * ATT_BYTES, smem_u32(Tt), 2 * ATT_BYTES);
    bulk_commit();
    bulk_wait_read0();   // T region reused by Tdiag right after this barrier
  }
  __syncthreads();
  PH_T(tid == 0, 4);  // elementwise + Att store
  // re-zero the T stack (Att staging used it); then T_II = (I + A_II)^-1 (threads 0..127) -> bf16 into the T stack
  for (int e = tid; e < 1024; e += 256) *reinterpret_cast<uint4*>(Tt + e * 16) = make_uint4(0, 0, 0, 0);
  __syncthreads();
  if (tid < 128) {
    int vl = tid >> 6, blk = (tid >> 4) & 3, c = tid & 15;
    const float* Ablk = Ad + (vl * 4 + blk) * 256;
    float col[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float sum = (i == c) ? 1.f : 0.f;
#pragma unroll
      for (int j = 0; j < i; j++) sum -= Ablk[i * 16 + j] * col[j];
      col[i] = sum;
    }
#pragma unroll
    for (int i = 0; i < 16; i++)
      *reinterpret_cast<__nv_bfloat16*>(Tt + offsw(64 * vl + blk * 16 + i, blk * 16 + c)) = __float2bfloat16(col[i]);
  }
  __syncthreads();
  PH_T(tid == 0, 5);  // Tdiag
  // ---- off-diagonal blocks: T_IJ = -T_II sum_{L=J}^{I-1} A_IL T_LJ ; warps 0-2 -> v-head 0, warps 4-6 -> v-head 1 (J = wq)
  {
    const int vl = ch;
    const unsigned char* Abt = vl ? Ab1 : Ab0;
    unsigned char* Tv = Tt + 64 * vl * 128;
    unsigned char* Mw = Mst + warp * 1024;
#pragma unroll 1
    for (int I = 1; I < 4; I++) {
      const int J = wq;
      if (J < I) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int Lb = J; Lb < I; Lb++) {
          uint32_t a[4], b[4];
          load_a<P64>(a, Abt, 16 * I, 16 * Lb, lane);
          load_b_t<L_SW>(b, Tv, 16 * J, 16 * Lb, lane);
          mma_bf16(acc[0], a, b); mma_bf16(acc[1], a, b + 2);
        }
#pragma unroll
        for (int j = 0; j < 2; j++) {
          *reinterpret_cast<uint32_t*>(Mw + toff<L_MW>(g, 8 * j + 2 * t4)) = pack_bf16(acc[j][0], acc[j][1]);
          *reinterpret_cast<uint32_t*>(Mw + toff<L_MW>(g + 8, 8 * j + 2 * t4)) = pack_bf16(acc[j][2], acc[j][3]);
        }
        __syncwarp();
        uint32_t a[4], b[4];
        load_a<L_SW>(a, Tv, 16 * I, 16 * I, lane);
        load_b_t<L_MW>(b, Mw, 0, 0, lane);
        float r2[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        mma_bf16(r2[0], a, b); mma_bf16(r2[1], a, b + 2);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          *reinterpret_cast<uint32_t*>(Tv + offsw(16 * I + g, 16 * J + 8 * j + 2 * t4)) = pack_bf16(-r2[j][0], -r2[j][1]);
          *reinterpret_cast<uint32_t*>(Tv + offsw(16 * I + g + 8, 16 * J + 8 * j + 2 * t4)) = pack_bf16(-r2[j][2], -r2[j][3]);
        }
      }
      __syncthreads();
    }
  }
  PH_T(tid == 0, 6);  // T off-diagonal
  // ---- T <- T diag(beta) in place (column j scaled by beta_j), both v-heads
  auto scale_T_cols = [&](const float* fac) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = tid + k * 256;
      const int row = e >> 3, cc = (e & 7) * 8;
      const float* fe = fac + (row >> 6) * 64;
      uint4 v4 = *reinterpret_cast<uint4*>(Tt + offsw(row, cc));
      uint32_t* vv = reinterpret_cast<uint32_t*>(&v4);
#pragma unroll
      for (int q = 0; q < 4; q++) { float2 x = unpack_bf16(vv[q]); vv[q] = pack_bf16(x.x * fe[cc + 2 * q], x.y * fe[cc + 2 * q + 1]); }
      *reinterpret_cast<uint4*>(Tt + offsw(row, cc)) = v4;
    }
  };
  scale_T_cols(bet);
  mbar_wait(mb_tma, 1);   // V1
  fence_proxy_async();
  __syncthreads();
  PH_T(tid == 0, 7);  // T beta scaling + V1 wait
  // ---- U_vh,h = (T diag(beta))_vh V_vh,h  (M=64, N=64, K=64): 4 sequences -> TMEM cols 64*useq, issued by lane 0 of warps 0-3
  if (lane == 0 && warp < 4) {
    const int useq = warp, vl = useq >> 1, h = useq & 1;
    const uint64_t dA = mkdesc(smem_u32(Tt + 64 * vl * 128));
    const uint64_t dB = mkdesc(smem_u32(vl ? (AKQ + h * 16384) : (Vb + h * 8192)));
#pragma unroll
    for (int ks = 0; ks < 4; ks++) umma_f16(taddr + 64 * useq, dA + 2 * ks, dB + 128 * ks, IDESC_U, ks > 0 ? 1u : 0u);
    umma_commit(mb_u0 + 8 * useq);
  }
  {
    // consume: stage each [64 tok][128 B] half image in smem (useq 0 -> AB1, 1 -> AB0, 2/3 -> V0 halves: dead once useq 0/1 are done)
    const uint32_t tl = taddr + ((uint32_t)(32 * wq) << 16) + 32 * ch;
#pragma unroll 1
    for (int useq = 0; useq < 4; useq++) {
      const int vl = useq >> 1, h = useq & 1;
      unsigned char* Ust = (useq == 0) ? Ab1 : (useq == 1) ? Ab0 : (Vb + (useq - 2) * 8192);
      mbar_wait(mb_u0 + 8 * useq, 0);
      tc_fence_after();
      uint32_t r[32];
      tmem_ld32(tl + 64 * useq, r);
      tmem_wait_ld();
      if (lane < 16) {   // M=64 layout: rows 16 wq + lane
        const int row = 16 * wq + lane;
        uint4 pk[4];
#pragma unroll
        for (int i = 0; i < 16; i++) reinterpret_cast<uint32_t*>(pk)[i] = pack_bf16(__uint_as_float(r[2 * i]), __uint_as_float(r[2 * i + 1]));
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++) *reinterpret_cast<uint4*>(Ust + offsw(row, 32 * ch + 8 * c4)) = pk[c4];
      }
      fence_proxy_async();
      __syncthreads();
      if (tid == 0) {
        bulk_s2g(p.U + ((size_t)gc * HV + 2 * qh + vl) * U_BYTES + h * 8192, smem_u32(Ust), 8192);
        bulk_commit();
      }
    }
  }
  tc_fence_before();
  __syncthreads();   // all U UMMAs done reading T (T' overwrites it below)
  tc_fence_after();
  PH_T(tid == 0, 8);  // U
  // ---- T' = T diag(beta e^gamma) in place, then W = [T'_0;T'_1] K^T (M=128, N=128, K=64) -> TMEM cols 0-127
  scale_T_cols(feg);
  fence_proxy_async();
  __syncthreads();
  if (tid == 0) {
    const uint64_t dA = mkdesc(smem_u32(Tt)), dB = mkdesc(smem_u32(KT));
#pragma unroll
    for (int ks = 0; ks < 4; ks++) umma_f16(taddr, dA + 2 * ks, dB + 2 * ks, IDESC_W, ks > 0 ? 1u : 0u);
    umma_commit(mb_w);
  }
  if (tid < 128) p.gam[((size_t)gc * HV + 2 * qh + (tid >> 6)) * CH + (tid & 63)] = gam[tid];
  // Att images back from global (written above by this CTA) -> registers, parked until the KT region is free
  uint4 attr[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int e = tid + k * 256;
    attr[k] = *reinterpret_cast<const uint4*>(p.Att + ((size_t)gc * HV + 2 * qh + (e >> 9)) * ATT_BYTES + (e & 511) * 16);
  }
  if (tid == 0) bulk_wait_read0();   // U staging buffers (AB0/AB1/V0) free again
  mbar_wait(mb_w, 0);
  tc_fence_after();
  __syncthreads();
  {
    // D_W rows: lane L -> v-head L>>6, token L&63 ; cols: warps ch handle d = 64ch..64ch+63 (two x32 loads)
    // W_0 image -> V0 region (halves +0/+8192), W_1 image -> K slots of the stack (halves +0/+16384)
    const uint32_t tl = taddr + ((uint32_t)(32 * wq) << 16) + 64 * ch;
    const int vl = L >> 6, i = L & 63;
    unsigned char* Wimg = (vl ? AKQ : Vb) + ch * (vl ? 16384 : 8192);
#pragma unroll
    for (int half = 0; half < 2; half++) {
      uint32_t r[32];
      tmem_ld32(tl + 32 * half, r);
      tmem_wait_ld();
      uint4 pk[4];
#pragma unroll
      for (int q = 0; q < 16; q++) reinterpret_cast<uint32_t*>(pk)[q] = pack_bf16(__uint_as_float(r[2 * q]), __uint_as_float(r[2 * q + 1]));
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) *reinterpret_cast<uint4*>(Wimg + offsw(i, 32 * half + 8 * c4)) = pk[c4];
    }
  }
  // ---- scaled K'^T images: K'^T_vh[d][t] = KT[d][t] e^{gamma_C - gamma_t} -> smem (vh0: T region, vh1: AB region), bulk-stored
#pragma unroll
  for (int vl = 0; vl < 2; vl++) {
    const float* fk = fkt + vl * 64;
    unsigned char* KTs = vl ? (smem + PP_OFF_AB1) : Tt;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = tid + k * 256;
      const int d = e >> 3, cks = e & 7;
      const int tc = 8 * (cks ^ (d & 7));
      const int off = d * 128 + (cks << 4);
      uint4 v4 = *reinterpret_cast<uint4*>(KT + off);
      uint32_t* vv = reinterpret_cast<uint32_t*>(&v4);
#pragma unroll
      for (int q = 0; q < 4; q++) { float2 x = unpack_bf16(vv[q]); vv[q] = pack_bf16(x.x * fk[tc + 2 * q], x.y * fk[tc + 2 * q + 1]); }
      *reinterpret_cast<uint4*>(KTs + off) = v4;
    }
  }
  tc_fence_before();
  fence_proxy_async();
  __syncthreads();   // KT reads done; W / K'^T images complete; W TMEM reads complete
  if (tid == 0) {
    bulk_s2g(p.KT + ((size_t)gc * HV + 2 * qh) * KT_BYTES, smem_u32(Tt), KT_BYTES);
    bulk_s2g(p.KT + ((size_t)gc * HV + 2 * qh + 1) * KT_BYTES, smem_u32(smem + PP_OFF_AB1), KT_BYTES);
    bulk_commit();
  }
  // Att images -> smem (KT region: vh0 at +0, vh1 at +8192)
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int e = tid + k * 256;
    *reinterpret_cast<uint4*>(KT + (e >> 9) * 8192 + (e & 511) * 16) = attr[k];
  }
  fence_proxy_async();
  __syncthreads();
  PH_T(tid == 0, 9);  // W + K'T + Att images
  // ---- M_vh = K'^T_vh W_vh (M=128 d, N=64 per d' half, K=64 tok) -> TMEM cols 128 vl + 64 nh ; negated when stored
  if (tid == 0) {
#pragma unroll
    for (int vl = 0; vl < 2; vl++) {
      const uint64_t dA = mkdesc(smem_u32(vl ? (smem + PP_OFF_AB1) : Tt));
#pragma unroll
      for (int nh = 0; nh < 2; nh++) {
        const uint64_t dB = mkdesc(smem_u32((vl ? AKQ : Vb) + nh * (vl ? 16384 : 8192)));
#pragma unroll
        for (int ks = 0; ks < 4; ks++) umma_f16(taddr + 128 * vl + 64 * nh, dA + 2 * ks, dB + 128 * ks, IDESC_M, ks > 0 ? 1u : 0u);
      }
    }
    umma_commit(mb_m);
  }
  mbar_wait(mb_m, 0);
  tc_fence_after();
  {
    // rows d = L ; this warp's cols: 64 ch + 32 h -> d' = 64 ch + 32 h + i -> k-atom ch of the M image.
    // Staged per v-head into the K'^T regions (k-atom 0 -> T region, k-atom 1 -> AB region), then bulk-stored.
#pragma unroll 1
    for (int vl = 0; vl < 2; vl++) {
      if (tid == 0) bulk_wait_read0();   // previous bulk stores (K'^T, M_0) done reading the staging regions
      __syncthreads();
      unsigned char* Mst2 = (ch ? (smem + PP_OFF_AB1) : Tt);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t r[32];
        tmem_ld32(taddr + ((uint32_t)(32 * wq) << 16) + 128 * vl + 64 * ch + 32 * h, r);
        tmem_wait_ld();
        uint4 pk[4];
#pragma unroll
        for (int q = 0; q < 16; q++) reinterpret_cast<uint32_t*>(pk)[q] = pack_bf16(-__uint_as_float(r[2 * q]), -__uint_as_float(r[2 * q + 1]));
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++) *reinterpret_cast<uint4*>(Mst2 + offsw(L, 32 * h + 8 * c4)) = pk[c4];
      }
      fence_proxy_async();
      __syncthreads();
      if (tid == 0) {
        unsigned char* Mg = p.M + ((size_t)gc * HV + 2 * qh + vl) * M_BYTES;
        bulk_s2g(Mg, smem_u32(Tt), 16384);
        bulk_s2g(Mg + 16384, smem_u32(smem + PP_OFF_AB1), 16384);
        bulk_commit();
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  PH_T(tid == 0, 10);  // M
  // ---- AttW_vh = Att_vh W_vh (M=64 tok, N=64 per d' half, K=64) -> TMEM cols 128 vl + 64 nh ; Q'' = fq Q - AttW
  if (tid == 0) {
#pragma unroll
    for (int vl = 0; vl < 2; vl++) {
      const uint64_t dA = mkdesc(smem_u32(KT + vl * 8192));
#pragma unroll
      for (int nh = 0; nh < 2; nh++) {
        const uint64_t dB = mkdesc(smem_u32((vl ? AKQ : Vb) + nh * (vl ? 16384 : 8192)));
#pragma unroll
        for (int ks = 0; ks < 4; ks++) umma_f16(taddr + 128 * vl + 64 * nh, dA + 2 * ks, dB + 128 * ks, IDESC_U, ks > 0 ? 1u : 0u);
      }
    }
    umma_commit(mb_aw);
  }
  mbar_wait(mb_aw, 0);
  tc_fence_after();
  PH_T(tid == 0, 11);  // AttW wait
  {
    const int tok = 16 * wq + lane;
#pragma unroll
    for (int vl = 0; vl < 2; vl++) {
      const float f = fq[vl * 64 + (tok & 63)];
      unsigned char* Qg = p.Q2 + ((size_t)gc * HV + 2 * qh + vl) * Q2_BYTES + ch * 8192;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t r[32];
        tmem_ld32(taddr + ((uint32_t)(32 * wq) << 16) + 128 * vl + 64 * ch + 32 * h, r);
        tmem_wait_ld();
        if (lane < 16) {
#pragma unroll
          for (int c4 = 0; c4 < 4; c4++) {
            uint4 q4 = *reinterpret_cast<const uint4*>(AKQ + 8192 + ch * 16384 + offsw(tok, 32 * h + 8 * c4));
            const uint32_t* qq = reinterpret_cast<const uint32_t*>(&q4);
            uint4 pk;
            uint32_t* pp = reinterpret_cast<uint32_t*>(&pk);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              float2 x = unpack_bf16(qq[i]);
              pp[i] = pack_bf16(f * x.x - __uint_as_float(r[8 * c4 + 2 * i]), f * x.y - __uint_as_float(r[8 * c4 + 2 * i + 1]));
            }
            *reinterpret_cast<uint4*>(Qg + offsw(tok, 32 * h + 8 * c4)) = pk;   // direct global store: no smem staging, no drain wait
          }
        }
      }
    }
  }
  PH_T(tid == 0, 12);  // Q'' epilogue (direct stores)
  if (tid == 0) bulk_wait_read0();   // M_1 bulk stores done reading smem (issued long before)
  PH_T(tid == 0, 11);  // Q''
  if (dbg && tid == 0) dbg[bid * 4 + 1] = gtimer();
#ifdef PHASE_TIMERS
  if (phd && tid == 0) { for (int i = 0; i < 16; i++) phd[48 + i] = phacc[i]; }
#endif
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(taddr, 256);
}

// ----------------------------------------------------------------------------------------------
// Kernel 2 (v21): fused chain recurrence + inline output on UMMA. One CTA per (sequence rank, v-head, 64-col V half).
// Per chunk c (b = c&1, r = c%3):  D[b] = K'^T U (4 k-steps, issued early) ; D[b] += M_c S_c (8 k-steps, A = M from TMEM)
//                                  S_{c+1} = e^{gamma_C} S_c + D[b] (fp32 registers; bf16 image -> smem ring of 3)
//                                  D_o[b] = Att U (4, early) ; D_o[b] += Q'' S_c (8) ; O = D_o[b] -> bf16 -> TMA store
// Warps: 0-7 S update (TMEM lane quarter warp&3, columns 32*(warp>>2)), 8 producer (bulk loads), 9 chain issuer, 10 output
//        issuer, 11 output store, 12-19 output epilogue (group (warp-12)>>2 handles chunks of parity group), 20-23 M loaders.
// ----------------------------------------------------------------------------------------------
constexpr int RS_STAGE = 49152;                       // smem ring stage: K'^T 16384 | U half 8192 | Q'' 16384 | Att 8192
constexpr int RS_KT = 0, RS_U = 16384, RS_Q = 24576, RS_ATT = 40960;
constexpr int RS_NST = 3;
constexpr int R_OFF_G = RS_NST * RS_STAGE;            // 147456: gam [3][64] f32
constexpr int R_OFF_S = R_OFF_G + 1024;               // 148480: S bf16 [3][128 d][128 B] (1 KB aligned: SWIZZLE_128B operand)
constexpr int R_NPIECE = 2;                           // M piece ring depth per loader warp
constexpr int R_OFF_MP = R_OFF_S + 3 * 16384;         // 197632: M pieces [4 loader warps][R_NPIECE][4096]
constexpr int R_OFF_MB = R_OFF_MP + 4 * R_NPIECE * 4096;   // 230400
constexpr int R_SMEM = R_OFF_MB + 512;                // 230912
constexpr int R_THREADS = 768;                        // 24 warps
constexpr int TM_D = 0, TM_O = 128, TM_M = 256;       // TMEM: D[2] (M=128 rows = d, 64 cols) ; D_o[2] (M=64 rows = tok) ; M slots [3] (128 lanes x 64 cols bf16 pairs)
constexpr uint32_t IDESC_CH = mkidesc(128, 64, 0, 1); // (K'^T | M) x (U | S) : A K-major (or TMEM), B MN-major
constexpr uint32_t IDESC_O = mkidesc(64, 64, 0, 1);   // (Att | Q'') x (U | S)
DEVI void umma_f16_ta(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, uint32_t enable_d) {
  asm volatile("{\n\t.reg .pred P;\n\tsetp.ne.b32 P, %4, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, P;\n\t}"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(enable_d) : "memory");
}

__global__ void __launch_bounds__(R_THREADS, 1) gdn_recur_kernel(const __grid_constant__ GdnParams p,
                                                                const __grid_constant__ TmaMaps maps) {
  extern __shared__ __align__(1024) unsigned char smem[];
  // mbarriers (8 B each). Every waiter targets a phase whose successor depends on the waiter's own progress (no parity aliasing).
  const uint32_t mb_full = smem_u32(smem + R_OFF_MB);          // [3] stage r input part (K'^T, U, Q'', gam) loaded (tx)   phase c/3
  const uint32_t mb_empty = mb_full + 24;                      // [3] stage r input part released (count 136: 8 S-update lane0 + 128 epilogue)
  const uint32_t mb_mready = mb_full + 48;                     // [3] M slot r written (count 4 loader warps)    phase c/3
  const uint32_t mb_mfree = mb_full + 72;                      // [3] M slot r consumed (count 8, after D done)  phase c/3
  const uint32_t mb_s = mb_full + 96;                          // S_c written (count 256)                        phase c
  const uint32_t mb_d = mb_full + 104;                         // chain UMMA c done (commit)                     phase c
  const uint32_t mb_ci = mb_full + 112;                        // [4] chain UMMA c issued (count 1)              phase c/4
  const uint32_t mb_ob = mb_full + 144;                        // [2] output UMMA c done (commit)                phase c/2
  const uint32_t mb_sfree = mb_full + 160;                     // [3] output UMMA c done (commit): S buffer c%3 free   phase c/3
  const uint32_t mb_od = mb_full + 184;                        // [2] D_o[b] read by the epilogue (count 128)    phase c/2
  const uint32_t mb_ost = mb_full + 200;                       // [3] O tile staged in stage r (count 128)       phase c/3
  const uint32_t mb_piece = mb_full + 272;                     // [4 loader warps][3] M piece landed (tx)        phase i/3
  const uint32_t mb_fulla = mb_full + 368;                     // [3] stage r Att loaded (tx)                     phase c/3
  const uint32_t mb_emptya = mb_full + 392;                    // [3] stage r Att region released (count 1: store warp after the O store)
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + R_OFF_MB + 256);
  int* seq_slot = reinterpret_cast<int*>(smem + R_OFF_MB + 260);
  unsigned char* Sb0 = smem + R_OFF_S;

  const long long* __restrict__ cu = p.cu;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x;
  const int rank = bid / (HV * 2);
  const int vh = (bid / 2) % HV;
  const int hh = bid & 1;                 // V half
  {
    const int N = p.N;
    for (int i = tid; i < N; i += R_THREADS) {
      long long li = cu[i + 1] - cu[i];
      int r = 0;
      for (int j = 0; j < N; j++) { long long lj = cu[j + 1] - cu[j]; r += (lj > li) || (lj == li && j < i); }
      if (r == rank) *seq_slot = i;
    }
    __syncthreads();
  }
  const int seq = *seq_slot;
  const long long st = cu[seq], en = cu[seq + 1];
  const int len = (int)(en - st);
  float* nsg = p.new_state + ((size_t)seq * HV + vh) * (D * D) + (size_t)hh * NV * D;   // rows v (64), cols d
  int gc_base = 0;
  for (int s = 0; s < seq; s++) {
    int l = (int)(cu[s + 1] - cu[s]);
    gc_base += l > 0 ? (l + CH - 1) / CH : 0;
  }
  const int nch = len > 0 ? (len + CH - 1) / CH : 0;
  long long* dbg = p.dbg ? p.dbg + 4096 * 4 + bid * 8 : nullptr;
  long long* phd = (p.dbg && bid == 0) ? p.dbg + 4096 * 12 : nullptr;
#ifdef PHASE_TIMERS
  long long phacc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) phacc[i] = 0;
  long long _ph_prev = clock64();
#endif
  if (dbg && tid == 0) { dbg[0] = gtimer(); dbg[3] = smid(); dbg[4] = nch; dbg[5] = seq; }
  if (tid == 0) {
    if (smem_u32(smem) & 1023) __trap();
    for (int i = 0; i < 3; i++) {
      mbar_init(mb_full + 8 * i, 1); mbar_init(mb_empty + 8 * i, 136); mbar_init(mb_mready + 8 * i, 4); mbar_init(mb_mfree + 8 * i, 8);
      mbar_init(mb_fulla + 8 * i, 1); mbar_init(mb_emptya + 8 * i, 1);
      mbar_init(mb_ost + 8 * i, 128);
    }
    mbar_init(mb_sfree, 1); mbar_init(mb_sfree + 8, 1); mbar_init(mb_sfree + 16, 1);
    for (int i = 0; i < 4; i++) mbar_init(mb_ci + 8 * i, 1);
    for (int i = 0; i < 4 * R_NPIECE; i++) mbar_init(mb_piece + 8 * i, 1);
    mbar_init(mb_s, 256); mbar_init(mb_d, 1);
    mbar_init(mb_ob, 1); mbar_init(mb_ob + 8, 1); mbar_init(mb_od, 128); mbar_init(mb_od + 8, 128);
    fence_mbar_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(tmem_slot), 512); tmem_relinquish(); }
  __syncthreads();
  tc_fence_after();
  const uint32_t taddr = *tmem_slot;
  if (len <= 0) {
    for (int e = tid; e < NV * D; e += R_THREADS) nsg[e] = 0.f;
  } else if (warp == 8) {
    // =============================== producer: smem ring ===============================
    if (lane == 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");   // prep outputs complete and visible
      for (int c = 0; c < nch; c++) {
        const int r = c % 3;
        const size_t ci = (size_t)(gc_base + c) * HV + vh;
        PH_T(true, 10);
        if (c >= RS_NST) mbar_wait(mb_empty + r * 8, ((c - RS_NST) / RS_NST) & 1);
        PH_T(true, 11);
        const uint32_t base = smem_u32(smem + r * RS_STAGE);
        const uint32_t mb = mb_full + r * 8, mba = mb_fulla + r * 8;
#ifdef NO_STAGE_LOADS
        if (c >= 3) { mbar_expect_tx(mb, 0); mbar_expect_tx(mba, 0); continue; }
#endif
        mbar_expect_tx(mb, KT_BYTES + 8192 + Q2_BYTES + GAM_BYTES);
        bulk_g2s(base + RS_KT, p.KT + ci * KT_BYTES, KT_BYTES, mb);
        bulk_g2s(base + RS_U, p.U + ci * U_BYTES + hh * 8192, 8192, mb);
        bulk_g2s(base + RS_Q, p.Q2 + ci * Q2_BYTES, Q2_BYTES, mb);
        bulk_g2s(smem_u32(smem + R_OFF_G + r * 256), p.gam + ci * CH, GAM_BYTES, mb);
        if (c >= RS_NST) mbar_wait(mb_emptya + r * 8, ((c - RS_NST) / RS_NST) & 1);
        mbar_expect_tx(mba, ATT_BYTES);
        bulk_g2s(base + RS_ATT, p.Att + ci * ATT_BYTES, ATT_BYTES, mba);
#ifdef PHASE_TIMERS
        if (phd && c < 64) phd[128 + c] = clock64();
#endif
        PH_T(true, 12);
      }
#ifdef PHASE_TIMERS
      if (phd) { phd[40] = phacc[11]; phd[41] = phacc[12]; phd[42] = 0; phd[43] = phacc[10]; }
#endif
    }
  } else if (warp >= 20) {
    // =============================== M loaders (warps 20-23): 4 KB pieces (rows 32q..32q+31 of k-atom ka) via bulk copies into a
    // 3-deep per-warp piece ring, then smem -> registers -> TMEM slot c%3 ===============================
    const int q = warp & 3;
    const uint32_t tz = taddr + ((uint32_t)(32 * q) << 16) + TM_M;
    const uint32_t pb = smem_u32(smem + R_OFF_MP + q * R_NPIECE * 4096);
    const uint32_t mbp = mb_piece + q * 8 * R_NPIECE;
    const int npieces = 2 * nch;
    auto issue_piece = [&](int i) {   // lane 0
      const int c = i >> 1, ka = i & 1;
      const unsigned char* src = p.M + ((size_t)(gc_base + c) * HV + vh) * M_BYTES + ka * 16384 + 32 * q * 128;
      mbar_expect_tx(mbp + 8 * (i % R_NPIECE), 4096);
      bulk_g2s(pb + (i % R_NPIECE) * 4096, src, 4096, mbp + 8 * (i % R_NPIECE));
    };
    if (lane == 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < R_NPIECE && i < npieces; i++) issue_piece(i);
    }
    __syncwarp();
    for (int c = 0; c < nch; c++) {
      const int r = c % 3;
      if (c >= 3) { if (lane == 0) mbar_wait(mb_mfree + r * 8, ((c - 3) / 3) & 1); __syncwarp(); }
      tc_fence_after();
#pragma unroll
      for (int ka = 0; ka < 2; ka++) {
        const int i = 2 * c + ka;
        if (lane == 0) mbar_wait(mbp + 8 * (i % R_NPIECE), (i / R_NPIECE) & 1);
        __syncwarp();
        const unsigned char* piece = smem + R_OFF_MP + (q * R_NPIECE + (i % R_NPIECE)) * 4096 + lane * 128;
        uint32_t rr[32];
#pragma unroll
        for (int jl = 0; jl < 8; jl++) {
          const uint4 a0 = *reinterpret_cast<const uint4*>(piece + ((jl ^ (lane & 7)) << 4));
          rr[4 * jl] = a0.x; rr[4 * jl + 1] = a0.y; rr[4 * jl + 2] = a0.z; rr[4 * jl + 3] = a0.w;
        }
        tmem_st32(tz + 64 * r + 32 * ka, rr);          // consumes rr (piece read complete) before the buffer is reused
        __syncwarp();
        if (lane == 0 && i + R_NPIECE < npieces) issue_piece(i + R_NPIECE);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(mb_mready + r * 8);
    }
  } else if (warp == 9) {
    // =============================== chain UMMA issuer ===============================
    if (lane == 0) {
      #ifdef PHASE_TIMERS
      long long tI[4] = {0, 0, 0, 0};
#endif
      for (int c = 0; c < nch; c++) {
        const int r = c % 3, b = c & 1;
        const uint32_t base = smem_u32(smem + r * RS_STAGE);
        const uint64_t dKT = mkdesc(base + RS_KT), dU = mkdesc(base + RS_U);
        const uint64_t dS = mkdesc(smem_u32(Sb0 + r * 16384));
        const uint32_t td = taddr + TM_D + 64 * b, ta = taddr + TM_M + 64 * r;
        mbar_wait(mb_full + r * 8, (c / 3) & 1);
        tc_fence_after();
        // D[b] = K'^T U  (D[b] was read in chunk c-2: implied by mb_s(c-1), waited in the previous iteration)
        umma_f16(td, dKT, dU, IDESC_CH, 0u);
        umma_f16(td, dKT + 2, dU + 128, IDESC_CH, 1u);
        umma_f16(td, dKT + 4, dU + 256, IDESC_CH, 1u);
        umma_f16(td, dKT + 6, dU + 384, IDESC_CH, 1u);
        mbar_wait(mb_mready + r * 8, (c / 3) & 1);
#ifdef PHASE_TIMERS
        long long t0 = clock64();
#endif
        mbar_wait(mb_s, c & 1);
#ifdef PHASE_TIMERS
        long long t1 = clock64();
#endif
        fence_proxy_async();   // cumulative: orders the S-update warps' generic smem writes (acquired via mb_s) before the UMMA reads
        tc_fence_after();
        umma_f16_ta(td, ta, dS, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 8, dS + 128, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 16, dS + 256, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 24, dS + 384, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 32, dS + 512, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 40, dS + 640, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 48, dS + 768, IDESC_CH, 1u);
        umma_f16_ta(td, ta + 56, dS + 896, IDESC_CH, 1u);
        umma_commit(mb_d);
        mbar_arrive(mb_ci + 8 * (c & 3));
#ifdef PHASE_TIMERS
        long long t2 = clock64();
#endif
#ifdef PHASE_TIMERS
        if (phd) { mbar_wait(mb_d, c & 1); long long t3 = clock64(); tI[0] += t1 - t0; tI[1] += t2 - t1; tI[2] += t3 - t2; }
#endif
      }
#ifdef PHASE_TIMERS
      if (phd) { phd[60] = tI[0]; phd[61] = tI[1]; phd[62] = tI[2]; }
#endif
    }
  } else if (warp == 10) {
    // =============================== output UMMA issuer ===============================
    if (lane == 0) {
      #ifdef PHASE_TIMERS
      long long tO[4] = {0, 0, 0, 0};
#endif
      for (int c = 0; c < nch; c++) {
        const int r = c % 3, b = c & 1;
        const uint32_t base = smem_u32(smem + r * RS_STAGE);
        const uint64_t dAtt = mkdesc(base + RS_ATT), dU = mkdesc(base + RS_U), dQ = mkdesc(base + RS_Q);
        const uint64_t dS = mkdesc(smem_u32(Sb0 + r * 16384));
        const uint32_t to = taddr + TM_O + 64 * b;
#ifdef PHASE_TIMERS
        long long to0 = clock64();
#endif
        mbar_wait(mb_full + r * 8, (c / 3) & 1);
        mbar_wait(mb_fulla + r * 8, (c / 3) & 1);
        if (c >= 2) mbar_wait(mb_od + 8 * b, ((c - 2) >> 1) & 1);   // D_o[b] read by the epilogue of chunk c-2
#ifdef PHASE_TIMERS
        long long to1 = clock64();
#endif
        tc_fence_after();
        umma_f16(to, dAtt, dU, IDESC_O, 0u);
        umma_f16(to, dAtt + 2, dU + 128, IDESC_O, 1u);
        umma_f16(to, dAtt + 4, dU + 256, IDESC_O, 1u);
        umma_f16(to, dAtt + 6, dU + 384, IDESC_O, 1u);
#ifdef PHASE_TIMERS
        long long to2 = clock64();
#endif
        mbar_wait(mb_ci + 8 * (c & 3), (c >> 2) & 1);                 // chain UMMA c queued first
#ifdef PHASE_TIMERS
        long long to3 = clock64();
#endif
        tc_fence_after();
        umma_f16(to, dQ, dS, IDESC_O, 1u);
        umma_f16(to, dQ + 2, dS + 128, IDESC_O, 1u);
        umma_f16(to, dQ + 4, dS + 256, IDESC_O, 1u);
        umma_f16(to, dQ + 6, dS + 384, IDESC_O, 1u);
        umma_f16(to, dQ + 512, dS + 512, IDESC_O, 1u);
        umma_f16(to, dQ + 514, dS + 640, IDESC_O, 1u);
        umma_f16(to, dQ + 516, dS + 768, IDESC_O, 1u);
        umma_f16(to, dQ + 518, dS + 896, IDESC_O, 1u);
        umma_commit(mb_ob + 8 * b);
        umma_commit(mb_sfree + 8 * r);
#ifdef PHASE_TIMERS
        long long to4 = clock64();
        tO[0] += to1 - to0; tO[1] += to2 - to1; tO[2] += to3 - to2; tO[3] += to4 - to3;
#endif
      }
#ifdef PHASE_TIMERS
      if (phd) { phd[72] = tO[0]; phd[73] = tO[1]; phd[74] = tO[2]; phd[75] = tO[3]; }
#endif
    }
  } else if (warp == 11) {
    // =============================== output store warp ===============================
    if (lane == 0) {
      #ifdef PHASE_TIMERS
      long long tS[3] = {0, 0, 0};
#endif
      for (int c = 0; c < nch; c++) {
        const int r = c % 3;
        const int c0 = (int)st + c * CH;
        const int n = min(CH, (int)en - c0);
#ifdef PHASE_TIMERS
        long long ts0 = clock64();
#endif
        mbar_wait(mb_ost + r * 8, (c / 3) & 1);
#ifdef PHASE_TIMERS
        long long ts1 = clock64();
#endif
        if (n == CH) {
          tma_store_3d(&maps.o, smem_u32(smem + r * RS_STAGE + RS_ATT), hh * NV, vh, c0);
          bulk_commit();
          bulk_wait_read0();
        }
        mbar_arrive(mb_emptya + r * 8);   // Att region (O staging) released
#ifdef PHASE_TIMERS
        long long ts2 = clock64();
        tS[0] += ts1 - ts0; tS[1] += ts2 - ts1;
#endif
      }
      bulk_wait0();
#ifdef PHASE_TIMERS
      if (phd) { phd[68] = tS[0]; phd[69] = tS[1]; }
#endif
    }
  } else if (warp >= 12) {
    // =============================== output epilogue warps 12-19: group g = (warp-12)>>2 handles chunks c with (c&1) == g ===============================
    const int q = warp & 3, g = (warp - 12) >> 2;
    const int tok = 16 * q + lane;
    const uint32_t tlo = taddr + ((uint32_t)(32 * q) << 16) + TM_O + 64 * g;
    #ifdef PHASE_TIMERS
      long long tE[3] = {0, 0, 0};
#endif
    for (int c = g; c < nch; c += 2) {
      const int r = c % 3;
      unsigned char* Ost = smem + r * RS_STAGE + RS_ATT;   // Att region of the stage: output staging after the UMMA
      const int c0 = (int)st + c * CH;
      const int n = min(CH, (int)en - c0);
#ifdef PHASE_TIMERS
      long long te0 = clock64();
#endif
      mbar_wait(mb_ob + 8 * g, (c >> 1) & 1);
#ifdef PHASE_TIMERS
      long long te1 = clock64();
#endif
      tc_fence_after();
      uint32_t rr[64];
      tmem_ld32(tlo, rr);
      tmem_ld32(tlo + 32, rr + 32);
      tmem_wait_ld();
      tc_fence_before();
      mbar_arrive(mb_od + 8 * g);   // D_o[g] read
      mbar_arrive(mb_empty + r * 8);   // stage r input part consumed (output UMMA done)
#pragma unroll
      for (int i = 0; i < 32; i++) rr[i] = pack_bf16(__uint_as_float(rr[2 * i]), __uint_as_float(rr[2 * i + 1]));
      if (n == CH) {
        if (lane < 16) {
#pragma unroll
          for (int c4 = 0; c4 < 8; c4++) *reinterpret_cast<uint4*>(Ost + offsw(tok, 8 * c4)) = make_uint4(rr[4 * c4], rr[4 * c4 + 1], rr[4 * c4 + 2], rr[4 * c4 + 3]);
        }
        fence_proxy_async();
      } else if (lane < 16 && tok < n) {
        __nv_bfloat16* og = p.out + ((size_t)(c0 + tok) * HV + vh) * D + hh * NV;
#pragma unroll
        for (int c4 = 0; c4 < 8; c4++) *reinterpret_cast<uint4*>(og + 8 * c4) = make_uint4(rr[4 * c4], rr[4 * c4 + 1], rr[4 * c4 + 2], rr[4 * c4 + 3]);
      }
      mbar_arrive(mb_ost + r * 8);
#ifdef PHASE_TIMERS
      long long te2 = clock64();
      tE[0] += te1 - te0; tE[1] += te2 - te1;
#endif
    }
#ifdef PHASE_TIMERS
    if (phd && warp == 12 && lane == 0) { phd[64] = tE[0]; phd[65] = tE[1]; }
#endif
  } else {
    // =============================== S-update warps 0-7 ===============================
    const int wq = warp & 3;
    const int ch = warp >> 2;
    const int L = 32 * wq + lane;                            // TMEM lane = d row
    const uint32_t tl = taddr + ((uint32_t)(32 * wq) << 16) + 32 * ch;
    float sreg[32];
    {
      const float* sg = p.state ? p.state + ((size_t)seq * HV + vh) * (D * D) + ((size_t)hh * NV + 32 * ch) * D + L : nullptr;
#pragma unroll
      for (int i = 0; i < 32; i++) sreg[i] = sg ? sg[(size_t)i * D] : 0.f;
      uint4 pk[4];
#pragma unroll
      for (int i = 0; i < 16; i++) reinterpret_cast<uint32_t*>(pk)[i] = pack_bf16(sreg[2 * i], sreg[2 * i + 1]);
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) *reinterpret_cast<uint4*>(Sb0 + offsw(L, 32 * ch + 8 * c4)) = pk[c4];
    }
    mbar_arrive(mb_s);   // S_0 ready (phase 0)
    for (int c = 0; c < nch; c++) {
      const int r = c % 3, b = c & 1, rn = (c + 1) % 3;
      const float* Gb = reinterpret_cast<const float*>(smem + R_OFF_G + r * 256);
      PH_T(tid == 0, 0);
      mbar_wait(mb_full + r * 8, (c / 3) & 1);
#ifdef PHASE_TIMERS
      if (phd && tid == 0 && c < 64) phd[192 + c] = clock64();
#endif
      PH_T(tid == 0, 1);  // full wait
      const float eC = __expf(Gb[CH - 1]);
      if (lane == 0) mbar_arrive(mb_empty + r * 8);     // gam read (stage release, S-update side)
      mbar_wait(mb_d, c & 1);
      PH_T(tid == 0, 2);  // D wait
      tc_fence_after();
      if (lane == 0) mbar_arrive(mb_mfree + r * 8);     // M slot r consumed
      {
        uint32_t rr[32];
        tmem_ld32(tl + TM_D + 64 * b, rr);
        tmem_wait_ld();
        PH_T(tid == 0, 4);  // tmem ld
#pragma unroll
        for (int i = 0; i < 32; i++) sreg[i] = sreg[i] * eC + __uint_as_float(rr[i]);
        if (c >= 2) mbar_wait(mb_sfree + rn * 8, ((c - 2) / 3) & 1);   // output UMMA of chunk c-2 finished reading S buffer rn
        PH_T(tid == 0, 5);  // sfree wait
        unsigned char* Sbn = Sb0 + 16384 * rn;
#pragma unroll
        for (int c4 = 0; c4 < 4; c4++) {
          uint4 pk;
          uint32_t* pp = reinterpret_cast<uint32_t*>(&pk);
#pragma unroll
          for (int i = 0; i < 4; i++) pp[i] = pack_bf16(sreg[8 * c4 + 2 * i], sreg[8 * c4 + 2 * i + 1]);
          *reinterpret_cast<uint4*>(Sbn + offsw(L, 32 * ch + 8 * c4)) = pk;
        }
        tc_fence_before();
        mbar_arrive(mb_s);
      }
      PH_T(tid == 0, 3);  // S update
    }
#pragma unroll
    for (int i = 0; i < 32; i++) nsg[(size_t)(32 * ch + i) * D + L] = sreg[i];
    if (dbg && tid == 0) { dbg[1] = gtimer(); }
#ifdef PHASE_TIMERS
    if (phd && tid == 0) { for (int i = 0; i < 16; i++) phd[16 + i] = phacc[i]; }
#endif
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(taddr, 512);
}

}  // namespace

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qr;
    void* f = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &qr);
    fn = (EncodeTiledFn)f;
  }
  return fn;
}
static int make_map(CUtensorMap* m, const void* ptr, int T, int heads) {
  cuuint64_t dims[3] = {128, (cuuint64_t)heads, (cuuint64_t)T};
  cuuint64_t strides[2] = {256, (cuuint64_t)heads * 256};
  cuuint32_t box[3] = {64, 1, 64};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode_fn()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : (int)r;
}

extern "C" {

size_t gdn_scratch_bytes(int T, int N) {
  size_t maxC = (size_t)(T / CH) + N;
  return maxC * HV * (M_BYTES + KT_BYTES + U_BYTES + Q2_BYTES + ATT_BYTES + GAM_BYTES) + (size_t)N * 4 + 4096;
}

int gdn_forward(const void* q, const void* k, const void* v, const void* state, const void* A_log, const void* a,
                const void* dt_bias, const void* b, const void* cu, float scale, void* out, void* new_state,
                void* scratch, int T, int N, cudaStream_t stream, void* dbg) {
  GdnParams p;
  p.dbg = (long long*)dbg;
  p.q = (const __nv_bfloat16*)q; p.k = (const __nv_bfloat16*)k; p.v = (const __nv_bfloat16*)v;
  p.state = (const float*)state; p.A_log = (const float*)A_log; p.a = (const __nv_bfloat16*)a;
  p.dt_bias = (const float*)dt_bias; p.b = (const __nv_bfloat16*)b; p.cu = (const long long*)cu;
  p.scale = scale; p.out = (__nv_bfloat16*)out; p.new_state = (float*)new_state;
  p.N = N; p.T = T; p.maxC = T / CH + N;
  unsigned char* s = (unsigned char*)scratch;
  size_t maxC = (size_t)p.maxC;
  p.M = s; s += maxC * HV * M_BYTES;
  p.KT = s; s += maxC * HV * KT_BYTES;
  p.U = s; s += maxC * HV * U_BYTES;
  p.Q2 = s; s += maxC * HV * Q2_BYTES;
  p.Att = s; s += maxC * HV * ATT_BYTES;
  p.gam = (float*)s; s += maxC * HV * GAM_BYTES;
  p.order = (int*)s;

  TmaMaps maps;
  int rc = make_map(&maps.q, q, T, HQ);
  rc |= make_map(&maps.k, k, T, HQ);
  rc |= make_map(&maps.v, v, T, HV);
  rc |= make_map(&maps.o, out, T, HV);
  if (rc) { fprintf(stderr, "gdn_forward: cuTensorMapEncodeTiled failed (%d)\n", rc); return 2; }

  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, PREP_SMEM);
    cudaFuncSetAttribute(gdn_recur_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, R_SMEM);
    attr_set = true;
  }
  gdn_prep_kernel<<<p.maxC * HQ, 256, PREP_SMEM, stream>>>(p, maps);
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N * HV * 2); cfg.blockDim = dim3(R_THREADS); cfg.dynamicSmemBytes = R_SMEM; cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gdn_recur_kernel, p, maps);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "gdn_forward launch error: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}

}  // extern "C"
