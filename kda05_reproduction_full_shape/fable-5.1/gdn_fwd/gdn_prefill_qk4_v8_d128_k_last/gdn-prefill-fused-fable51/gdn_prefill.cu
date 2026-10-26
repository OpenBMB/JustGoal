// Fused Gated Delta Net prefill kernel (qk4_v8_d128, k-last state layout) for NVIDIA B200.
//
// v5.0 design (Phase 4: v3.13 + fp32 column-substitution inverse with transposed T storage): one CTA per (sequence, value head, 32-column slice of V) -> grid (32, N),
// 8 warps in lock-step. Chunk size 64, chunked WY formulation (FLA chunk_gated_delta_rule math):
//
//   gamma_i = cumsum(lg)_i (chunk local),  A[i,j] = beta_i (k_i.k_j) e^{gamma_i-gamma_j} (j<i)
//   (I + A) [W | U] = [beta e^gamma K | beta V]     solved blockwise (16x16) on tensor cores
//   Delta^T = U^T - S0^T W^T
//   o^T     = e^{gamma_i} (S0^T q_i^T) + Delta^T P^T,   P[i,j] = (q_i.k_j) e^{gamma_i-gamma_j} (j<=i)
//   S^T     = e^{gamma_C} S^T + (Delta e^{gamma_C-gamma})^T K
//
// State slice S^T [32 v][128 k] (k-last layout) lives in registers as mma.sync accumulator
// fragments (2 m-tiles x 16 n-tiles spread over 8 warps) plus a replicated bf16 A-operand copy of
// the warp's m-tile. Raw q/k/v tiles and gate rows of the next chunk are prefetched with cp.async;
// the next chunk's gates are computed by otherwise idle warps while the diagonal blocks are inverted.
// Work on padded tokens (beyond the sequence end) is skipped at 16-token block granularity.
// Single kernel launch; cu_seqlens is read on the device.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

namespace gdn {

#ifdef GDN_TIMING
__device__ unsigned long long* g_dbg = nullptr;
#define GDN_TS_INIT() unsigned long long* const dbgp = \
    (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) ? g_dbg : nullptr
#ifdef GDN_TIMING_REP
#define GDN_NREP 2
#else
#define GDN_NREP 1
#endif
#define GDN_TS(slot) do { if (dbgp && ch < 2 && rep == GDN_NREP - 1) dbgp[ch * 16 + (slot)] = clock64(); } while (0)
#define GDN_TS_ABS(slot) do { if (dbgp && rep == GDN_NREP - 1) dbgp[slot] = clock64(); } while (0)
#else
#define GDN_NREP 1
#define GDN_TS_INIT() do {} while (0)
#define GDN_TS(slot) do {} while (0)
#define GDN_TS_ABS(slot) do {} while (0)
#endif

constexpr int D = 128;       // head size (K and V)
constexpr int C = 64;        // chunk size (tokens)
constexpr int HV = 8;        // value heads
constexpr int HQ = 4;        // q/k heads
constexpr int NT = 256;      // threads per CTA
constexpr int BS = 16;       // solve block size
constexpr int NB = C / BS;   // 4 blocks per chunk
constexpr int LDK = D + 8;   // bf16 [C][D] tile row stride
constexpr int LDS = C + 8;   // [C][C] tile row stride (fp32 or bf16)
constexpr int LDT = BS + 8;  // 16x16 bf16 block row stride (24 el = 48 B)
constexpr int NG = NB * (NB - 1) / 2;  // 6 off-diagonal block pairs
constexpr int NUNITS = 20;   // 10 lower-triangular 16x16 blocks x 2 n8-tiles for KK/QK

template <int VS, int NPB>   // NPB = 2: chunk-parity buffers for the state-independent quantities (PRE kernels)
struct __align__(16) SmemT {
  static constexpr int VW = D / VS;    // v columns per CTA
  static constexpr int LDV = VW + 8;   // bf16 [C][VW] tile row stride
  static constexpr int LDU = VW + 4;   // fp32 U tile row stride
  static constexpr int NMT = VW / 16;  // state m-tiles
  __nv_bfloat16 K[2][C * LDK];    // raw k tile (head hq)          [token][dim]   (double buffered)
  __nv_bfloat16 Q[2][C * LDK];    // raw q tile (head hq)          [token][dim]
  __nv_bfloat16 V[2][C * LDV];    // raw v slice (head h, 32 cols) [token][v]
  uint4 AB[2][C][2];              // raw a[t][0..7], b[t][0..7] (bf16 x 8 each)
  static constexpr int LDD = C + 8;    // bf16 [v][token] tile row stride (Delta tiles, v-major)
  union {
    struct {
  __nv_bfloat16 W[C * LDK];       // -W (bf16)                     [token][dim]
  __nv_bfloat16 XU[C * LDV];      // U (bf16, solve operand)       [token][v]
  float U32[C * LDU];             // U (fp32, Delta init)          [token][v]
    } old;   // v10.0 solve path (OLDS)
    struct {
  __nv_bfloat16 Rh[VW * LDD];     // R^T = beta (V^T - e^gamma S^T K^T), bf16 hi   [v][token]
  __nv_bfloat16 Rl[VW * LDD];     // low bf16 part of R^T
  __nv_bfloat16 Dlo[VW * LDD];    // low bf16 part of Delta (chain operand)
  float O32[VW * LDD];            // o_inter^T = e^gamma S^T Q^T (fp32)      [v][token]
    } nw;    // reassociated path
  };
  float A32[C * LDS];             // scratch (k-split KK/QK reduction for <= 16-token chunks)
  // state-independent per-chunk quantities, indexed by chunk parity (NPB = 2 when chunk c+1's copies are built
  // during chunk c, Phase 8 D7); T_rr and its column-scaled variants are stored TRANSPOSED:
  // X[r][j * LDT + i] = T_rr[i][j] (consumers load them with ldmatrix.trans as [k = j][m = i] tiles)
  __nv_bfloat16 A16[NPB][C * LDS];     // A (bf16)                      [i][j]
  __nv_bfloat16 P16[NPB][C * LDS];     // P (causal, bf16)              [i][j]
  __nv_bfloat16 T[NPB][NB][BS * LDT];    // T_rr^T
  __nv_bfloat16 Tlo[NPB][NB][BS * LDT];  // low bf16 parts (x - bf16(x)) of T (transposed)
  float ATf[NPB][BS * BS];               // fp32 transposed diagonal block 0: ATf[j * 16 + i] = A_00[i][j] (one-block chunks)
  __nv_bfloat16 AT16[NPB][NB][BS * LDT]; // bf16 transposed diagonal blocks: AT16[r][j * LDT + i] = A_rr[i][j]
  __nv_bfloat16 G16[NPB][NG][BS * LDT];  // G_rs = T_rr A_rs (bf16, row-major) for the 6 lower block pairs
  __nv_bfloat16 TK[NB][BS * LDT];   // (T_rr * diag(beta e^gamma))^T      (v10.0 solve path only)
  __nv_bfloat16 TV[NB][BS * LDT];   // (T_rr * diag(beta))^T
  __nv_bfloat16 TKlo[NB][BS * LDT];
  __nv_bfloat16 TVlo[NB][BS * LDT];
  __nv_bfloat16 Dl[VW * LDD];     // Delta (bf16)                  [v][token]
  __nv_bfloat16 Ds[VW * LDD];     // Delta e^{gamma_C-gamma}       [v][token]  (bf16 hi part)
  __nv_bfloat16 Dslo[VW * LDD];   // low bf16 part of Delta e^{gamma_C-gamma}
  __nv_bfloat16 O[C * LDV];       // output staging                [token][v]
  uint32_t Sb[NMT][8][32][4];     // state bf16 A-fragment exchange [mt][ks][lane][4]
  float gamma[2][C];              // gate buffers, indexed by chunk parity
  float beta[2][C];
  float lg[2][C];      // per-token log gate before the chunk scan
  float gC[2];
  static constexpr bool EXPT = (VS != 4);   // e^gamma tables (reassociated path only)
  float eg[EXPT ? 2 : 1][C];     // e^{gamma_t}
  float ec[EXPT ? 2 : 1][C];     // e^{gamma_C - gamma_t}
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float bf16_residual(float x) {   // x - bf16(x)
  return x - __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool valid) {
  const int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(smem)), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// Early global loads whose issue order must not be sunk below the long cp.async issue sequence.
__device__ __forceinline__ float2 ldg_f2_early(const float* p) {
  float2 r;
  asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];\n" : "=f"(r.x), "=f"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ float ldg_bf16_early(const __nv_bfloat16* p) {
  unsigned short u;
  asm volatile("ld.global.nc.u16 %0, [%1];\n" : "=h"(u) : "l"(p));
  return __uint_as_float(static_cast<uint32_t>(u) << 16);
}
__device__ __forceinline__ int64_t ldg_s64_early(const int64_t* p) {
  int64_t r;
  asm volatile("ld.global.nc.s64 %0, [%1];\n" : "=l"(r) : "l"(p));
  return r;
}
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// ---- mbarrier + TMA bulk copy (1D, async proxy) --------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_copy_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra LAB_WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count));
}
__device__ __forceinline__ void named_bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count));
}

// ---- fragment loaders (lane = threadIdx.x & 31) ------------------------------------------------
// A fragment (m16 x k16) from row-major [m][k] tile.
__device__ __forceinline__ void ldA_rm(uint32_t* a, const __nv_bfloat16* base, int m0, int k0, int ld, int lane) {
  const int row = m0 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int col = k0 + (lane >> 4) * 8;
  ldsm_x4(a, smem_u32(base + row * ld + col));
}
// A fragment (m16 x k16) from k-major [k][m] tile (transposed load).
__device__ __forceinline__ void ldA_km(uint32_t* a, const __nv_bfloat16* base, int m0, int k0, int ld, int lane) {
  const int row = k0 + (lane & 7) + (lane >> 4) * 8;
  const int col = m0 + ((lane >> 3) & 1) * 8;
  ldsm_x4_t(a, smem_u32(base + row * ld + col));
}
// B fragments for two n8 tiles (n0, n0+8) x k16 from n-major [n][k] tile. b[0..1] tile n0, b[2..3] n0+8.
__device__ __forceinline__ void ldB_nm(uint32_t* b, const __nv_bfloat16* base, int n0, int k0, int ld, int lane) {
  const int row = n0 + (lane & 7) + (lane >> 4) * 8;
  const int col = k0 + ((lane >> 3) & 1) * 8;
  ldsm_x4(b, smem_u32(base + row * ld + col));
}
// B fragment for one n8 tile x k16 from n-major [n][k] tile.
__device__ __forceinline__ void ldB_nm1(uint32_t* b, const __nv_bfloat16* base, int n0, int k0, int ld, int lane) {
  const int row = n0 + (lane & 7);
  const int col = k0 + ((lane >> 3) & 1) * 8;
  ldsm_x2(b, smem_u32(base + row * ld + col));
}
// B fragment for one n8 tile x k16 from k-major [k][n] tile (transposed load).
__device__ __forceinline__ void ldB_km1(uint32_t* b, const __nv_bfloat16* base, int n0, int k0, int ld, int lane) {
  const int row = k0 + (lane & 7) + ((lane >> 3) & 1) * 8;
  ldsm_x2_t(b, smem_u32(base + row * ld + n0));
}
// B fragments for two n8 tiles (n0, n0+8) x k16 from k-major [k][n] tile (transposed load).
__device__ __forceinline__ void ldB_km(uint32_t* b, const __nv_bfloat16* base, int n0, int k0, int ld, int lane) {
  const int row = k0 + (lane & 7) + ((lane >> 3) & 1) * 8;
  const int col = n0 + (lane >> 4) * 8;
  ldsm_x4_t(b, smem_u32(base + row * ld + col));
}

__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}

// ---- TMA / UMMA / TMEM helpers ---------------------------------------------------------------------
constexpr int SWROWS = 128;   // rows of a stacked [K;Q] SW128 half tile (K rows 0-63, Q rows 64-127)
constexpr int SWHALF = SWROWS * 64;   // elements per half tile
// SW128 K-major element offset within a [rows][64] bf16 half tile (128 B rows): 16 B chunk ^= (row & 7)
__device__ __forceinline__ int sw128(int row, int col) {
  return row * 64 + ((((col >> 3) ^ (row & 7)) & 7) << 3) + (col & 7);
}
__device__ __forceinline__ void tma_load_3d(void* dst, const CUtensorMap* tmap, int c0, int c1, int c2, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
      ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
// UMMA shared-memory descriptor (version 1). lbo/sbo in bytes; layout 2 = SWIZZLE_128B.
__device__ __forceinline__ uint64_t umma_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((saddr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;      // version (Blackwell)
  d |= static_cast<uint64_t>(2) << 61;      // SWIZZLE_128B
  return d;
}
// Instruction descriptor: bf16 x bf16 -> f32, M, N, operand majors (0 = K-major, 1 = MN-major)
__device__ __forceinline__ uint32_t umma_idesc(int M, int N, int a_mn, int b_mn) {
  uint32_t d = 0;
  d |= 1u << 4;    // c_format F32
  d |= 1u << 7;    // a_format BF16
  d |= 1u << 10;   // b_format BF16
  d |= static_cast<uint32_t>(a_mn & 1) << 15;
  d |= static_cast<uint32_t>(b_mn & 1) << 16;
  d |= static_cast<uint32_t>((N >> 3) & 0x3F) << 17;
  d |= static_cast<uint32_t>((M >> 4) & 0x1F) << 24;
  return d;
}
__device__ __forceinline__ void umma(uint32_t d_tmem, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
      ::"r"(d_tmem), "l"(da), "l"(db), "r"(idesc), "r"(accumulate));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* slot, int ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(slot)), "r"(ncols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
}
__device__ __forceinline__ void tmem_dealloc(uint32_t base, int ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(base), "r"(ncols));
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
      "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n"
      ::"r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
        "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }

// ---- mma.sync fragment loaders from a stacked SW128 [K;Q] tile (two halves of 64 dims) ------------
// B fragments (two n8 tiles over dims, k16 over tokens) from [token][dim] rows (transposed load)
__device__ __forceinline__ void ldB_tile_t(uint32_t* b, const __nv_bfloat16* tile2, int n0, int k0, int lane) {
  const int row = k0 + (lane & 7) + ((lane >> 3) & 1) * 8;   // token
  const int col = n0 + (lane >> 4) * 8;                      // dim
  const __nv_bfloat16* half = tile2 + (col >> 6) * SWHALF;
  ldsm_x4_t(b, smem_u32(half + sw128(row, col & 63)));
}
// B fragments (two n8 tiles over tokens, k16 over dims) from [token][dim] rows (non-transposed load)
__device__ __forceinline__ void ldB_tile_nm(uint32_t* b, const __nv_bfloat16* tile2, int n0, int k0, int lane) {
  const int row = n0 + (lane & 7) + (lane >> 4) * 8;         // token
  const int col = k0 + ((lane >> 3) & 1) * 8;                // dim
  const __nv_bfloat16* half = tile2 + (col >> 6) * SWHALF;
  ldsm_x4(b, smem_u32(half + sw128(row, col & 63)));
}

__device__ __forceinline__ float softplus_fast(float x) {
  // branch-free: max(x, 0) + log(1 + e^{-|x|}) (matches the thresholded form to ~1e-7 absolute; the
  // thresholded version compiled to divergent branches around the MUFU chain)
  return fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x)));
}
__device__ __forceinline__ float sigmoid_fast(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
// e^x for x <= 0 via the raw ex2 unit (denormals flushed to zero): no range-check predication
__device__ __forceinline__ float exp_neg_fast(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950408889634f));
  return r;
}

// Gates of one chunk in two parts (so the MUFU/shuffle latency runs on otherwise idle warps):
//   part 1: lg = -e^{A_log} softplus(a + dt_bias), beta = sigmoid(b) for token t -> smem
//   scan:   chunk-local inclusive cumsum of lg -> gamma, gamma_C (one warp, lane -> tokens 2*lane, 2*lane+1)
// e^{gamma} and e^{gamma_C - gamma} are computed on the fly by their consumers.
template <class Smem>
__device__ __forceinline__ void gates_token(Smem& sm, int gb, int ab, int Lc, int h, float neg_expA, float dtb, int t) {
  float lg = 0.f, bt = 0.f;
  if (t < Lc) {
    const __nv_bfloat16* ar = reinterpret_cast<const __nv_bfloat16*>(&sm.AB[ab][t][0]);
    const __nv_bfloat16* br = reinterpret_cast<const __nv_bfloat16*>(&sm.AB[ab][t][1]);
    lg = neg_expA * softplus_fast(__bfloat162float(ar[h]) + dtb);
    bt = sigmoid_fast(__bfloat162float(br[h]));
  }
  sm.lg[gb][t] = lg;
  sm.beta[gb][t] = bt;
}
// chunk scan from registers: lane holds lg of tokens 2*lane (l0) and 2*lane+1 (l1)
template <class Smem>
__device__ __forceinline__ void gates_scan_reg(Smem& sm, int gb, int lane, float l0, float l1) {
  const int t0 = 2 * lane;
  float x = l0 + l1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  *reinterpret_cast<float2*>(&sm.gamma[gb][t0]) = make_float2(x - l1, x);
  if (lane == 31) sm.gC[gb] = x;
  if constexpr (Smem::EXPT) {
    const float gc = __shfl_sync(0xffffffffu, x, 31);
    const float ga = x - l1;
    *reinterpret_cast<float2*>(&sm.eg[gb][t0]) = make_float2(__expf(ga), __expf(x));
    *reinterpret_cast<float2*>(&sm.ec[gb][t0]) = make_float2(__expf(gc - ga), __expf(gc - x));
  }
}
template <class Smem>
__device__ __forceinline__ void gates_scan(Smem& sm, int gb, int lane) {
  const float2 l2 = *reinterpret_cast<const float2*>(&sm.lg[gb][2 * lane]);
  gates_scan_reg(sm, gb, lane, l2.x, l2.y);
}
// Chunk-0 gates on one warp (Phase 8 D2): the raw a[t][h] / b[t][h] of tokens 2*lane, 2*lane+1 are loaded
// straight from global memory with the state loads (gate_loads), the per-token math, the scan and the e^gamma
// tables run in registers while the K/Q/V tiles are in flight (gates_chunk0): no a/b cp.async group, no CTA
// barrier and no scan on the KK/QK path.
// ONE (one-block chunks): one token per lane on lanes 0-15, 4-step scan.
template <bool ONE>
__device__ __forceinline__ void gate_loads(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                           int64_t tb, int Lc, int h, int lane, float& a0, float& b0, float& a1,
                                           float& b1) {
  a0 = 0.f; b0 = 0.f; a1 = 0.f; b1 = 0.f;
  if constexpr (ONE) {
    if (lane < Lc && lane < BS) { a0 = ldg_bf16_early(a + (tb + lane) * HV + h); b0 = ldg_bf16_early(b + (tb + lane) * HV + h); }
  } else {
    const int t0 = 2 * lane;
    if (t0 < Lc) { a0 = ldg_bf16_early(a + (tb + t0) * HV + h); b0 = ldg_bf16_early(b + (tb + t0) * HV + h); }
    if (t0 + 1 < Lc) { a1 = ldg_bf16_early(a + (tb + t0 + 1) * HV + h); b1 = ldg_bf16_early(b + (tb + t0 + 1) * HV + h); }
  }
}
template <bool ONE, class Smem>
__device__ __forceinline__ void gates_chunk0(Smem& sm, int Lc, float neg_expA, float dtb, int lane, float a0, float b0,
                                             float a1, float b1) {
  if constexpr (ONE) {
    float l0 = 0.f, be0 = 0.f;
    if (lane < Lc && lane < BS) { l0 = neg_expA * softplus_fast(a0 + dtb); be0 = sigmoid_fast(b0); }
    float x = l0;
#pragma unroll
    for (int o = 1; o < BS; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    const float gc = __shfl_sync(0xffffffffu, x, BS - 1);
    if (lane < BS) {
      sm.beta[0][lane] = be0;
      sm.gamma[0][lane] = x;
      if constexpr (Smem::EXPT) { sm.eg[0][lane] = __expf(x); sm.ec[0][lane] = __expf(gc - x); }
    }
    if (lane == 0) sm.gC[0] = gc;
  } else {
    const int t0 = 2 * lane;
    float l0 = 0.f, l1 = 0.f, be0 = 0.f, be1 = 0.f;
    if (t0 < Lc) { l0 = neg_expA * softplus_fast(a0 + dtb); be0 = sigmoid_fast(b0); }
    if (t0 + 1 < Lc) { l1 = neg_expA * softplus_fast(a1 + dtb); be1 = sigmoid_fast(b1); }
    *reinterpret_cast<float2*>(&sm.beta[0][t0]) = make_float2(be0, be1);
    gates_scan_reg(sm, 0, lane, l0, l1);
  }
}

// cp.async loads of one chunk's tiles into buffer `buf`. Only rows of valid 16-token blocks are loaded
// (rows >= round16(Lc) are never read by the consumers); rows in [Lc, round16(Lc)) are zero-filled.
// a/b rows: 64 tokens x 2 x 16 B (threads tid < 2*rows of any 256-thread call), one commit group.
template <class Smem>
__device__ __forceinline__ void prefetch_ab(Smem& sm, int buf, const __nv_bfloat16* __restrict__ a,
                                            const __nv_bfloat16* __restrict__ b, int64_t tbase, int Lc, int tid) {
  const int rows = ((Lc + BS - 1) / BS) * BS;
  if (tid < 2 * rows) {
    const int t = tid >> 1;
    const bool ok = t < Lc;
    const int64_t tg = tbase + (ok ? t : 0);
    const __nv_bfloat16* src = (tid & 1) ? (b + tg * HV) : (a + tg * HV);
    cp_async_16(&sm.AB[buf][t][tid & 1], src, ok);
  }
  cp_async_commit();
}
// K, Q tiles (64 x 128 bf16) and the V slice, issued by NTH threads (tid in [0, NTH)), one commit group.
template <int VS, int NTH, class Smem>
__device__ __forceinline__ void prefetch_kqv(Smem& sm, int buf, const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v, int64_t tbase, int Lc,
                                             int hq, int h, int v0, int tid) {
  const int rows = ((Lc + BS - 1) / BS) * BS;
  constexpr int ROWS_PER_IT = NTH / 16;   // 16 x 16 B chunks per row
#pragma unroll
  for (int it = 0; it < (C + ROWS_PER_IT - 1) / ROWS_PER_IT; ++it) {
    if (it * ROWS_PER_IT < rows) {
      const int idx = it * NTH + tid;
      const int t = idx >> 4;
      const int cc = (idx & 15) * 8;
      const bool ok = t < Lc;
      const int64_t tg = tbase + (ok ? t : 0);
      if (t < C) {   // NTH not a divisor of C * 16: the last iteration runs past the tile
        cp_async_16(sm.K[buf] + t * LDK + cc, k + (tg * HQ + hq) * D + cc, ok);
        cp_async_16(sm.Q[buf] + t * LDK + cc, q + (tg * HQ + hq) * D + cc, ok);
      }
    }
  }
  {
    constexpr int CPR = Smem::VW / 8;   // 16 B chunks per V row
#pragma unroll
    for (int it = 0; it < (C * CPR + NTH - 1) / NTH; ++it) {
      const int idx = it * NTH + tid;
      const int t = idx / CPR;
      const int cc = (idx % CPR) * 8;
      if (t < rows) {
        const bool ok = t < Lc;
        const int64_t tg = tbase + (ok ? t : 0);
        cp_async_16(sm.V[buf] + t * Smem::LDV + cc, v + (tg * HV + h) * D + v0 + cc, ok);
      }
    }
  }
  cp_async_commit();
}
template <int VS, class Smem>
__device__ __forceinline__ void prefetch_chunk(Smem& sm, int buf, const __nv_bfloat16* __restrict__ q,
                                               const __nv_bfloat16* __restrict__ k,
                                               const __nv_bfloat16* __restrict__ v,
                                               const __nv_bfloat16* __restrict__ a,
                                               const __nv_bfloat16* __restrict__ b, int64_t tbase, int Lc,
                                               int hq, int h, int v0, int tid) {
  prefetch_ab(sm, buf, a, b, tbase, Lc, tid);
  prefetch_kqv<VS, NT>(sm, buf, q, k, v, tbase, Lc, hq, h, v0, tid);
}

// KK^T / QK^T units of one warp: block row BI (rows I0 = 16*BI .. +15) against NU n8 column tiles
// J0A, J0B, J0C (-1 = none), K = 128 in 8 k-steps (kk_mma); then A = beta_i (k_i.k_j) e^{gamma_i - gamma_j}
// (j < i), P = (q_i.k_j) e^{gamma_i - gamma_j} (j <= i) are written in bf16, diagonal tiles also transposed
// into AT16 (kk_scale). Everything is compile-time so that loads and the independent MMA chains interleave.
template <int J0A, int J0B, int J0C>
struct KKUnits {
  static constexpr int NU = (J0C >= 0) ? 3 : (J0B >= 0 ? 2 : 1);
  float KK[NU][4], QK[NU][4];
};
template <int BI, int J0A, int J0B, int J0C>
__device__ __forceinline__ void kk_mma(const __nv_bfloat16* __restrict__ Kt, const __nv_bfloat16* __restrict__ Qt,
                                       int Lc, int lane, KKUnits<J0A, J0B, J0C>& acc) {
  constexpr int I0 = BI * 16;
  constexpr int NU = KKUnits<J0A, J0B, J0C>::NU;
  constexpr int J0[3] = {J0A, J0B, J0C};
#pragma unroll
  for (int u = 0; u < NU; ++u)
#pragma unroll
    for (int e = 0; e < 4; ++e) { acc.KK[u][e] = 0.f; acc.QK[u][e] = 0.f; }
  if (I0 < Lc) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint32_t aK[4], aQ[4], bb[NU][2];
      ldA_rm(aK, Kt, I0, ks * 16, LDK, lane);
      ldA_rm(aQ, Qt, I0, ks * 16, LDK, lane);
#pragma unroll
      for (int u = 0; u < NU; ++u) ldB_nm1(bb[u], Kt, J0[u], ks * 16, LDK, lane);
#pragma unroll
      for (int u = 0; u < NU; ++u) {
        mma16816(acc.KK[u], aK, bb[u]);
        mma16816(acc.QK[u], aQ, bb[u]);
      }
    }
  }
}
template <int BI, int J0A, int J0B, int J0C, class Smem>
__device__ __forceinline__ void kk_scale(Smem& sm, __nv_bfloat16* A16d, __nv_bfloat16* P16d, int Lc, const float* gam,
                                         const float* bet, int lane, const KKUnits<J0A, J0B, J0C>& acc) {
  constexpr int I0 = BI * 16;
  constexpr int NU = KKUnits<J0A, J0B, J0C>::NU;
  constexpr int J0[3] = {J0A, J0B, J0C};
  if (I0 < Lc) {
#pragma unroll
    for (int u = 0; u < NU; ++u) {
      const int j = J0[u] + (lane & 3) * 2;
      const float gj0 = gam[j], gj1 = gam[j + 1];
      const bool diag = (J0[u] >= I0);   // n8 tile inside the diagonal 16x16 block (resolved after unrolling)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = I0 + (lane >> 2) + half * 8;
        const float gi = gam[i], bi_ = bet[i];
        const float d0 = __expf(gi - gj0), d1 = __expf(gi - gj1);
        const float a0 = (j < i) ? bi_ * acc.KK[u][half * 2] * d0 : 0.f;
        const float a1 = (j + 1 < i) ? bi_ * acc.KK[u][half * 2 + 1] * d1 : 0.f;
        *reinterpret_cast<uint32_t*>(A16d + i * LDS + j) = pack_bf16(a0, a1);
        if (diag) {   // bf16 transposed copy for the inverse: AT16[BI][jl][il]
          const int il = i - I0, jl = j - I0;
          sm.AT16[BI][jl * LDT + il] = __float2bfloat16(a0);
          sm.AT16[BI][(jl + 1) * LDT + il] = __float2bfloat16(a1);
        }
        const float p0 = (j <= i) ? acc.QK[u][half * 2] * d0 : 0.f;
        const float p1 = (j + 1 <= i) ? acc.QK[u][half * 2 + 1] * d1 : 0.f;
        *reinterpret_cast<uint32_t*>(P16d + i * LDS + j) = pack_bf16(p0, p1);
      }
    }
  }
}
// one unit set per warp (lock-step kernel): MMAs, optional chunk-0 gate scan + barrier, scaling
template <int BI, int J0A, int J0B, int J0C, class Smem>
__device__ __forceinline__ void kk_units(Smem& sm, const __nv_bfloat16* __restrict__ Kt,
                                         const __nv_bfloat16* __restrict__ Qt, __nv_bfloat16* P16d, int Lc,
                                         const float* gam, const float* bet, int lane, bool chunk0, bool scan_warp,
                                         int buf) {
  KKUnits<J0A, J0B, J0C> acc;
  kk_mma<BI, J0A, J0B, J0C>(Kt, Qt, Lc, lane, acc);
  if (chunk0) {   // chunk 0: the scan warp produces the gates while the MMAs drain; everyone needs them
    if (scan_warp) gates_scan(sm, buf, lane);
    named_bar_sync(2, NT);
  }
  kk_scale<BI, J0A, J0B, J0C>(sm, sm.A16, P16d, Lc, gam, bet, lane, acc);
}
// two or three unit sets per warp (warp-specialised kernel, 4 WY warps): all MMAs first, then the scalings
template <int BI1, int A1, int B1, int C1, int BI2, int A2, int B2, int C2, int BI3, int A3, int B3, int C3, class Smem>
__device__ __forceinline__ void kk_units3(Smem& sm, const __nv_bfloat16* __restrict__ Kt,
                                          const __nv_bfloat16* __restrict__ Qt, __nv_bfloat16* A16d, __nv_bfloat16* P16d,
                                          int Lc, const float* gam, const float* bet, int lane) {
  KKUnits<A1, B1, C1> u1;
  KKUnits<A2, B2, C2> u2;
  kk_mma<BI1, A1, B1, C1>(Kt, Qt, Lc, lane, u1);
  kk_mma<BI2, A2, B2, C2>(Kt, Qt, Lc, lane, u2);
  if constexpr (A3 >= 0) {
    KKUnits<A3, B3, C3> u3;
    kk_mma<BI3, A3, B3, C3>(Kt, Qt, Lc, lane, u3);
    kk_scale<BI1, A1, B1, C1>(sm, A16d, P16d, Lc, gam, bet, lane, u1);
    kk_scale<BI2, A2, B2, C2>(sm, A16d, P16d, Lc, gam, bet, lane, u2);
    kk_scale<BI3, A3, B3, C3>(sm, A16d, P16d, Lc, gam, bet, lane, u3);
  } else {
    kk_scale<BI1, A1, B1, C1>(sm, A16d, P16d, Lc, gam, bet, lane, u1);
    kk_scale<BI2, A2, B2, C2>(sm, A16d, P16d, Lc, gam, bet, lane, u2);
  }
}

// T_rr = (I + A_rr)^-1 (unit lower triangular, 16x16) by fp32 column substitution; lane c (lanes 16-31
// replicate lanes 0-15) owns column c. With t_j = T[j][c] finalised in order j = 0..15:
//   P[i] += A[i][j] t_j (i > j),  t_j = (j == c) ? 1 : -P[j].
// Writes T, T*diag(beta e^gamma), T*diag(beta) as bf16 hi/lo pairs, transposed (row c = column c of T).
template <class Smem>
__device__ __forceinline__ void inverse_block(Smem& sm, int r, int nblk, const float* gam, const float* bet, int lane) {
  const int c = lane & 15;
  float P[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) P[i] = 0.f;
  float Tc[16];
  if (nblk == 1) {
    // fp32 rows (one block: a single warp, latency-bound), loaded LA rows ahead
    const float* At = sm.ATf;
    constexpr int LA = 4;
    float4 Q[15][4];
    auto ld_row = [&](int j) {
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4)
        if (q4 * 4 + 3 > j) {
          const uint32_t addr = smem_u32(At + j * BS + q4 * 4);
          asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                       : "=f"(Q[j][q4].x), "=f"(Q[j][q4].y), "=f"(Q[j][q4].z), "=f"(Q[j][q4].w) : "r"(addr));
        }
    };
#pragma unroll
    for (int j = 0; j < LA; ++j) ld_row(j);
#pragma unroll
    for (int j = 0; j < 15; ++j) {
      if (j + LA < 15) ld_row(j + LA);
      const float tj = (j == c) ? 1.f : -P[j];
      Tc[j] = tj;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        if (q4 * 4 + 3 > j) {
          const float av[4] = {Q[j][q4].x, Q[j][q4].y, Q[j][q4].z, Q[j][q4].w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = q4 * 4 + e;
            if (i > j) P[i] = fmaf(av[e], tj, P[i]);
          }
        }
      }
    }
  } else {
    // bf16 rows (several warps at once: half the smem traffic of fp32 rows), all loaded up front
    const __nv_bfloat16* At = sm.AT16[r];
    uint4 R[15][2];
#pragma unroll
    for (int j = 0; j < 15; ++j) {
      if (j < 7) R[j][0] = *reinterpret_cast<const uint4*>(At + j * LDT);   // i = 1..7 live only for j < 7
      else R[j][0] = make_uint4(0u, 0u, 0u, 0u);
      R[j][1] = *reinterpret_cast<const uint4*>(At + j * LDT + 8);
    }
#pragma unroll
    for (int j = 0; j < 15; ++j) {
      const float tj = (j == c) ? 1.f : -P[j];
      Tc[j] = tj;
      const uint32_t w[8] = {R[j][0].x, R[j][0].y, R[j][0].z, R[j][0].w, R[j][1].x, R[j][1].y, R[j][1].z, R[j][1].w};
#pragma unroll
      for (int i = j + 1; i < 16; ++i) {
        const uint32_t u = w[i >> 1];
        const float av = __uint_as_float((i & 1) ? (u & 0xffff0000u) : (u << 16));
        P[i] = fmaf(av, tj, P[i]);
      }
    }
  }
  Tc[15] = (c == 15) ? 1.f : -P[15];
  {   // transposed writes: lane c writes row c of T^T (= column c of T); lanes 0-15 take rows
      // i = 0..7 of the column, lanes 16-31 rows 8..15 (8 elements each, all six outputs, 16 B stores)
    const float cV = bet[r * BS + c];
    const float cK = cV * __expf(gam[r * BS + c]);
    const bool up = lane >= 16;
    auto split = [](float a, float b, uint32_t& hi_, uint32_t& lo_) {   // bf16 hi/lo pair split
      hi_ = pack_bf16(a, b);
      const float ha = __uint_as_float(hi_ << 16), hb = __uint_as_float(hi_ & 0xffff0000u);
      lo_ = pack_bf16(a - ha, b - hb);
    };
    uint32_t t[4], tl[4], k[4], kl[4], v[4], vl[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float ta = up ? Tc[8 + 2 * e] : Tc[2 * e];
      const float tb = up ? Tc[9 + 2 * e] : Tc[2 * e + 1];
      split(ta, tb, t[e], tl[e]);
      split(ta * cK, tb * cK, k[e], kl[e]);
      split(ta * cV, tb * cV, v[e], vl[e]);
    }
    const int off = c * LDT + (up ? 8 : 0);   // 16 B aligned (LDT = 24 -> 48 B rows)
    *reinterpret_cast<uint4*>(sm.T[r] + off) = make_uint4(t[0], t[1], t[2], t[3]);
    *reinterpret_cast<uint4*>(sm.Tlo[r] + off) = make_uint4(tl[0], tl[1], tl[2], tl[3]);
    *reinterpret_cast<uint4*>(sm.TK[r] + off) = make_uint4(k[0], k[1], k[2], k[3]);
    *reinterpret_cast<uint4*>(sm.TKlo[r] + off) = make_uint4(kl[0], kl[1], kl[2], kl[3]);
    *reinterpret_cast<uint4*>(sm.TV[r] + off) = make_uint4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<uint4*>(sm.TVlo[r] + off) = make_uint4(vl[0], vl[1], vl[2], vl[3]);
  }
}

// G_rs = T_rr A_rs (bf16, 16x16) for the warp's pairs pi = warp, warp + 4 of the six lower block pairs
// (0:(1,0) 1:(2,0) 2:(2,1) 3:(3,0) 4:(3,1) 5:(3,2)), shared through smem.
template <class Smem>
__device__ __forceinline__ void g_pairs(Smem& sm, const __nv_bfloat16* A16d, int warp, int nblk, int lane) {
#pragma unroll
  for (int q2 = 0; q2 < 2; ++q2) {
    const int pi = warp + 4 * q2;
    if (pi < NG) {
      const int r = (pi == 0) ? 1 : (pi <= 2 ? 2 : 3);
      const int s_ = (pi == 0) ? 0 : (pi <= 2 ? pi - 1 : pi - 3);
      if (r < nblk) {
        float g[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        uint32_t at[4], ba[4];
        ldB_km(ba, A16d, s_ * BS, r * BS, LDS, lane);   // B[k=l][n=j] = A[r*16+l][s*16+j]
        ldA_km(at, sm.T[r], 0, 0, LDT, lane);
        mma16816(g[0], at, ba);
        mma16816(g[1], at, ba + 2);
        ldA_km(at, sm.Tlo[r], 0, 0, LDT, lane);
        mma16816(g[0], at, ba);
        mma16816(g[1], at, ba + 2);
        const int gi = lane >> 2, gj = (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(sm.G16[pi] + gi * LDT + gj) = pack_bf16(g[0][0], g[0][1]);
        *reinterpret_cast<uint32_t*>(sm.G16[pi] + (gi + 8) * LDT + gj) = pack_bf16(g[0][2], g[0][3]);
        *reinterpret_cast<uint32_t*>(sm.G16[pi] + gi * LDT + gj + 8) = pack_bf16(g[1][0], g[1][1]);
        *reinterpret_cast<uint32_t*>(sm.G16[pi] + (gi + 8) * LDT + gj + 8) = pack_bf16(g[1][2], g[1][3]);
      }
    }
  }
}

// A fragments of all six G_rs (zeros for block rows beyond nblk).
template <class Smem>
__device__ __forceinline__ void load_gf(Smem& sm, int nblk, int lane, uint32_t (&gf)[NG][4]) {
#pragma unroll
  for (int pi = 0; pi < NG; ++pi) {
    const int r = (pi == 0) ? 1 : (pi <= 2 ? 2 : 3);
    if (r < nblk) ldA_rm(gf[pi], sm.G16[pi], 0, 0, LDT, lane);
    else { gf[pi][0] = gf[pi][1] = gf[pi][2] = gf[pi][3] = 0u; }
  }
}

// Block forward substitution X_r = T_rr (B_r) - sum_{s<r} G_rs X_s for the warp's 32 W columns
// (wc0 .. wc0+31) and, if UW, its U columns 8*warp .. 8*warp+7 (V slice columns). Only the warp's own
// columns of X_s are needed, so no CTA barrier is required between block rows.
template <bool UW, bool SWZ, class Smem>
__device__ __forceinline__ void solve_wy(Smem& sm, const __nv_bfloat16* Kt, const __nv_bfloat16* Vt,
                                         __nv_bfloat16* Wd, float* U32d, int warp, int nblk, int lane,
                                         const uint32_t (&gf)[NG][4]) {
  constexpr int LDV = Smem::LDV, LDU = Smem::LDU;
  const int wc0 = warp * 32;   // first W column of this warp
#pragma unroll
  for (int r = 0; r < NB; ++r) {
    if (r >= nblk) break;
    const int r0 = r * BS;
    float xw[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u) { xw[u][0] = xw[u][1] = xw[u][2] = xw[u][3] = 0.f; }
    float xu[4] = {0.f, 0.f, 0.f, 0.f};
    {
      uint32_t aa[4], al[4], bbf[4];
      ldA_km(aa, sm.TK[r], 0, 0, LDT, lane);
      ldA_km(al, sm.TKlo[r], 0, 0, LDT, lane);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        if constexpr (SWZ) ldB_tile_t(bbf, Kt, wc0 + hf * 16, r0, lane);   // B[k=token][n=dim], two n-tiles
        else ldB_km(bbf, Kt, wc0 + hf * 16, r0, LDK, lane);
        mma16816(xw[2 * hf], aa, bbf);
        mma16816(xw[2 * hf + 1], aa, bbf + 2);
        mma16816(xw[2 * hf], al, bbf);
        mma16816(xw[2 * hf + 1], al, bbf + 2);
      }
      if constexpr (UW) {
        ldB_km1(bbf, Vt, warp * 8, r0, LDV, lane);
        ldA_km(aa, sm.TV[r], 0, 0, LDT, lane);
        mma16816(xu, aa, bbf);
        ldA_km(aa, sm.TVlo[r], 0, 0, LDT, lane);
        mma16816(xu, aa, bbf);
      }
    }
#pragma unroll
    for (int s_ = 0; s_ < NB - 1; ++s_) {
      if (s_ < r) {
        const int pi = r * (r - 1) / 2 + s_;   // compile-time after unrolling
        uint32_t bbf[4];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          ldB_km(bbf, Wd, wc0 + hf * 16, s_ * BS, LDK, lane);  // B = -X_s (W part) -> accumulates -G X_s
          mma16816(xw[2 * hf], gf[pi], bbf);
          mma16816(xw[2 * hf + 1], gf[pi], bbf + 2);
        }
        if constexpr (UW) {
          uint32_t gn[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) gn[e] = gf[pi][e] ^ 0x80008000u;   // -G
          ldB_km1(bbf, sm.XU, warp * 8, s_ * BS, LDV, lane);  // B = +X_s (U part), A = -G
          mma16816(xu, gn, bbf);
        }
      }
    }
    // store -W (bf16), U (bf16 + fp32) for this warp's columns
    {
      const int rr = r0 + (lane >> 2);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int cc = wc0 + nt * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(Wd + rr * LDK + cc) = pack_bf16(-xw[nt][0], -xw[nt][1]);
        *reinterpret_cast<uint32_t*>(Wd + (rr + 8) * LDK + cc) = pack_bf16(-xw[nt][2], -xw[nt][3]);
      }
      if constexpr (UW) {
        const int cc = warp * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(sm.XU + rr * LDV + cc) = pack_bf16(xu[0], xu[1]);
        *reinterpret_cast<uint32_t*>(sm.XU + (rr + 8) * LDV + cc) = pack_bf16(xu[2], xu[3]);
        *reinterpret_cast<float2*>(U32d + rr * LDU + cc) = make_float2(xu[0], xu[1]);
        *reinterpret_cast<float2*>(U32d + (rr + 8) * LDU + cc) = make_float2(xu[2], xu[3]);
      }
    }
    __syncwarp();
  }
}

// NBLK > 0: compile-time number of valid 16-token blocks (single-chunk sequences whose length is
// known on the host); NBLK < 0: compile-time upper bound -NBLK on the block count (N == 2, T <= 64:
// each sequence has at most T tokens; the runtime count is min(-NBLK, ceil(L / 16))); SINGLE: every
// sequence fits in one chunk (no loop, no prefetch of a next chunk).
// Phase 9 (D9-1): longest-first sequence order. CTA row y processes the sequence whose length rank is y (ties by
// index), so the dominant sequence's CTAs launch first (CTAs launch in blockIdx order; every long batch of the
// workload set is dominated by one sequence). Every thread of the CTA must call it (two CTA barriers); the
// identity order is kept for N < 3 or N > LPT_MAXN.
// The two static arrays total exactly 1024 B so that the dynamic smem base keeps its 1024 B alignment (the TMA
// 128B-swizzled tiles of the ws7 kernel need it; 1544 B of static scratch gave "misaligned address").
constexpr int LPT_MAXN = 127;
__shared__ int lpt_cs[LPT_MAXN + 1];       // smem copy of cu_seqlens (the kernel reads its sequence bounds from it)
__shared__ int lpt_perm[LPT_MAXN + 1];
__device__ __forceinline__ bool lpt_active(int N) { return N >= 3 && N <= LPT_MAXN; }
__device__ __forceinline__ int lpt_sequence(const int64_t* __restrict__ cu_seqlens, int N, int tid) {
  if (!lpt_active(N)) return static_cast<int>(blockIdx.y);
  for (int s = tid; s <= N; s += NT) lpt_cs[s] = static_cast<int>(cu_seqlens[s]);
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31;
  auto lpt_len = [&](int i) { return lpt_cs[i + 1] - lpt_cs[i]; };
#ifdef GDN_LPT_BALLOT   // one sequence per warp, 32 comparisons per ballot: measured 10-20 us SLOWER than the serial
                        // form on the 8192 batches with 38-57 sequences (iteration 2 A/B, docs/draft.md Phase 9)
  for (int s = warp; s < N; s += NT / 32) {
    const int len = lpt_len(s);
    int rank = 0;
    for (int t0 = 0; t0 < N; t0 += 32) {
      const int t = t0 + lane;
      bool before = false;
      if (t < N) { const int lt = lpt_len(t); before = (lt > len) || (lt == len && t < s); }
      rank += __popc(__ballot_sync(0xffffffffu, before));
    }
    if (lane == 0) lpt_perm[rank] = s;
  }
#else                   // one sequence per thread, serial count (unrolled so the shared loads pipeline)
  (void)warp; (void)lane;
#ifndef GDN_LPT_NOLOOP   // timing experiment: prologue without the ranking (use with GDN_LPT_IDENTITY)
  for (int s = tid; s < N; s += NT) {
    const int len = lpt_len(s);
    int rank = 0;
#ifndef GDN_LPT_NOUNROLL
#pragma unroll 8
#endif
    for (int t = 0; t < N; ++t) {
      const int lt = lpt_len(t);
      rank += (lt > len || (lt == len && t < s)) ? 1 : 0;
    }
    lpt_perm[rank] = s;
  }
#endif
#endif
#ifdef GDN_LPT_IDENTITY   // timing experiment: same prologue, identity order
  __syncthreads();
  for (int s = tid; s < N; s += NT) lpt_perm[s] = s;
#endif
  __syncthreads();
  return lpt_perm[blockIdx.y];
}

template <int VS, int NBLK, bool SINGLE, bool LPT = false>
__global__ void __launch_bounds__(NT, 1)
gdn_prefill_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                   const __nv_bfloat16* __restrict__ v, const float* __restrict__ state,
                   const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ a,
                   const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ b,
                   const int64_t* __restrict__ cu_seqlens, float scale, int T_total,
                   __nv_bfloat16* __restrict__ out, float* __restrict__ new_state) {
  constexpr int NPB = (!SINGLE && VS != 4 && NBLK > 0 && NBLK <= 2) ? 2 : 1;   // parity buffers (PRE kernels)
  using Smem = SmemT<VS, NPB>;
  constexpr int VW = Smem::VW, LDV = Smem::LDV, LDU = Smem::LDU, NMT = Smem::NMT, LDD = Smem::LDD;
  constexpr int NBMAX = NBLK < 0 ? -NBLK : NBLK;   // compile-time block count (exact or upper bound), 0 = runtime
  // TWO: exactly two chunks (N == 1, 64 < T <= 128): chunk 0 has NB blocks, chunk 1 has NBLK blocks; the chunk
  // loop is unrolled so every per-chunk quantity is compile-time in each copy.
  constexpr bool TWO = (!SINGLE && NBLK > 0);
#ifdef GDN_OLD_SOLVE
  constexpr bool OLDS = true;
#else
  constexpr bool OLDS = (VS == 4);   // the reassociated path loses on the VS = 4 (N >= 3) multi-chunk kernel
#endif
  // EARLY (chunk 0 of the reassociated-path kernels, Phase 8 D3/D4): warp 7 issues no chunk-0 tile loads and runs
  // the gates while warps 0-6 wait for the tiles on named barrier 3 and start the KK/QK MMAs; if warp 7 has KK/QK
  // units (3-4 block tables) it joins through named barrier 4 after the gates; the gates are consumed after the
  // CTA barrier (id 2 / __syncthreads) that precedes the A/P epilogue. ONE: one-block kernels (16-lane gates).
#ifdef GDN_EARLY_SMALL_ONLY   // v12.1 behaviour: only the single-chunk kernels with <= 2 blocks
  constexpr bool EARLY = SINGLE && (NBMAX >= 1 && NBMAX <= 2);
#else
  constexpr bool EARLY = !OLDS;
#endif
  constexpr bool ONE = SINGLE && (NBMAX == 1);
  constexpr int PF_NTH = EARLY ? 7 * 32 : NT;   // threads issuing the chunk-0 K/Q/V tile loads
  // PRE (two-chunk kernels whose second chunk has <= 2 blocks, Phase 8 D7): chunk 1's state-independent quantities
  // (A, P, T, G) are built by warps 0-3 during chunk 0's phase 3a (after their inverse) from chunk 1's tiles (issued
  // by warps 0-3 at chunk 0's start) and gates (warps 4-7, handed over through named barrier 5); chunk 1 then skips
  // its KK/QK phase, inverse and G pairs, and all 8 warps run its units. (The same scheme for 3-4 block second
  // chunks and for the runtime-chunk-count kernel measured 0.2-0.6 us slower: the serial unit chains on 4 warps
  // exceed the idle time of phase 3a.)
#ifdef GDN_NO_PRE
  constexpr bool PRE = false;
#else
  constexpr bool PRE = !OLDS && TWO && (NBLK <= 2);
#endif
  static_assert(!PRE || NPB == 2, "PRE needs parity buffers");
  constexpr int WG = 8 / NMT;      // warps per m-tile
  constexpr int NKT = 16 / WG;     // state k n-tiles per warp
  constexpr int NTT = 8 / WG;      // token n-tiles per warp (o_inter / Delta / o)
  constexpr int NUT = VW / 8;      // U n-tiles (warps 0..NUT-1 own one each in the solve)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

  const int h = blockIdx.x / VS;          // value head
  const int vs = blockIdx.x % VS;         // v slice
  const int v0 = vs * VW;
  const int tid = threadIdx.x;
  static_assert(!LPT || NBLK == 0, "LPT order only in the runtime-chunk-count kernels");
  const int n = LPT ? lpt_sequence(cu_seqlens, static_cast<int>(gridDim.y), tid) : static_cast<int>(blockIdx.y);   // sequence
  const int hq = h / (HV / HQ);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int mt = warp % NMT;              // state m-tile (v rows 16*mt..+15 of the slice)
  const int wq = warp / NMT;              // 0..WG-1: state k n-tile group / token group
  GDN_TS_INIT();
  { const int rep = GDN_NREP - 1; GDN_TS_ABS(32); }

  // S^T slice accumulator: warp owns m-tile mt (v rows), n-tiles 4*wq..4*wq+3 (k cols).
  // Issued first so that the HBM latency overlaps the cu_seqlens / gate / tile prologue.
  float S[NKT][4];
  {
    const float* st_in = state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
    const int r = mt * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (wq * NKT + j) * 8 + (lane & 3) * 2;
#ifdef GDN_EXP_NO_STATE_LOAD
      S[j][0] = S[j][1] = S[j][2] = S[j][3] = 0.f;   // timing experiment only (wrong results)
#else
      const float2 lo = ldg_f2_early(st_in + r * D + c);
      const float2 hi = ldg_f2_early(st_in + (r + 8) * D + c);
      S[j][0] = lo.x; S[j][1] = lo.y; S[j][2] = hi.x; S[j][3] = hi.y;
#endif
    }
  }
  const float neg_expA = -__expf(A_log[h]);
  const float dtb = dt_bias[h];
  // host-known sequence length (N == 1 single-chunk kernels): no cu_seqlens round trip at all
  const bool lpt_on = LPT && lpt_active(static_cast<int>(gridDim.y));
  const int64_t s0 = (NBLK > 0) ? 0 : (lpt_on ? static_cast<int64_t>(lpt_cs[n]) : ldg_s64_early(cu_seqlens + n));
  const int64_t s1 = (NBLK > 0) ? T_total : (lpt_on ? static_cast<int64_t>(lpt_cs[n + 1]) : ldg_s64_early(cu_seqlens + n + 1));
  // Sequence 0 starts at token 0 (cu_seqlens[0] == 0 by construction): issue its chunk-0 tile loads
  // before cu_seqlens arrives, so the two DRAM round trips overlap. Verified below; if the assumption
  // fails the loads are drained and re-issued from the true start. Rows beyond the sequence end then
  // hold other tokens instead of zeros; every consumer masks them through beta = 0 / block skipping.
  float ga0, gb0, ga1, gb1;   // warp 7: raw gate inputs of tokens 2*lane, 2*lane+1 of chunk 0
#ifndef GDN_EXP_NO_LOADS   // timing experiment only (wrong results): no tile / gate-input loads at all
  if (n == 0) {
    if (tid < PF_NTH) prefetch_kqv<VS, PF_NTH>(sm, 0, q, k, v, 0, min(C, T_total), hq, h, v0, tid);
    if (warp == 7) gate_loads<ONE>(a, b, 0, min(C, T_total), h, lane, ga0, gb0, ga1, gb1);
  }
#else
  ga0 = gb0 = ga1 = gb1 = 0.f;
#endif
  const int L = static_cast<int>(s1 - s0);
  const bool spec = (n == 0);
  { const int rep = GDN_NREP - 1; GDN_TS_ABS(35); }

  float* st_out = new_state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
  if (L <= 0) {
    cp_async_wait<0>();
    for (int i = tid * 4; i < VW * D; i += NT * 4)
      *reinterpret_cast<float4*>(st_out + i) = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  const int nchunks = SINGLE ? 1 : (TWO ? 2 : (L + C - 1) / C);

  for (int rep = 0; rep < GDN_NREP; ++rep) {   // GDN_NREP > 1 only in timing experiments
  if (rep > 0) {
    __syncthreads();
    if (spec) {
      if (tid < PF_NTH) prefetch_kqv<VS, PF_NTH>(sm, 0, q, k, v, 0, min(C, T_total), hq, h, v0, tid);
      if (warp == 7) gate_loads<ONE>(a, b, 0, min(C, T_total), h, lane, ga0, gb0, ga1, gb1);
    }
  }
  // ---- prologue: chunk 0 tiles in flight, gates of chunk 0 on warp 7 ------------------------------
#ifndef GDN_EXP_NO_LOADS
  if (!spec) {
    if (tid < PF_NTH) prefetch_kqv<VS, PF_NTH>(sm, 0, q, k, v, s0, min(C, L), hq, h, v0, tid);
    if (warp == 7) gate_loads<ONE>(a, b, s0, min(C, L), h, lane, ga0, gb0, ga1, gb1);
  } else if (s0 != 0) {   // speculation failed (never for packed batches): drain and re-issue
    cp_async_wait<0>();
    __syncthreads();
    if (tid < PF_NTH) prefetch_kqv<VS, PF_NTH>(sm, 0, q, k, v, s0, min(C, L), hq, h, v0, tid);
    if (warp == 7) gate_loads<ONE>(a, b, s0, min(C, L), h, lane, ga0, gb0, ga1, gb1);
  }
#endif
  GDN_TS_ABS(36);

  uint32_t Sb[8][4];  // bf16 A fragments of S^T m-tile mt (M = v, K = k), all 8 k-steps
  auto exchange_S = [&]() {
#pragma unroll
    for (int kp = 0; kp < NKT / 2; ++kp) {   // k-step = pair of this warp's n-tiles
      const uint4 pk = make_uint4(pack_bf16(S[2 * kp][0], S[2 * kp][1]), pack_bf16(S[2 * kp][2], S[2 * kp][3]),
                                  pack_bf16(S[2 * kp + 1][0], S[2 * kp + 1][1]), pack_bf16(S[2 * kp + 1][2], S[2 * kp + 1][3]));
      *reinterpret_cast<uint4*>(sm.Sb[mt][wq * (NKT / 2) + kp][lane]) = pk;
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint4 x = *reinterpret_cast<const uint4*>(sm.Sb[mt][ks][lane]);
      Sb[ks][0] = x.x; Sb[ks][1] = x.y; Sb[ks][2] = x.z; Sb[ks][3] = x.w;
    }
  };
  // state fragment exchange (its global-load wait overlaps the gate-input / tile DRAM round trips), then the
  // chunk-0 gates on warp 7 while the other warps wait for the tiles (the tile-wait barrier orders the gates)
  exchange_S();
  GDN_TS_ABS(33);
#ifndef GDN_EXP_NO_GATES
  if (warp == 7) gates_chunk0<ONE>(sm, min(C, L), neg_expA, dtb, lane, ga0, gb0, ga1, gb1);
#endif
  GDN_TS_ABS(38);

#ifdef GDN_TWO_NOUNROLL
#pragma unroll 1
#else
#pragma unroll (TWO ? 2 : 1)
#endif
  for (int ch = 0; ch < nchunks; ++ch) {
    const int buf = ch & 1;
    const int pb = (NPB == 2) ? (ch & 1) : 0;   // parity buffer of this chunk's state-independent quantities
    const int c0 = ch * C;
    const int Lc = SINGLE ? L : min(C, L - c0);
    // valid 16-token blocks (NBLK > 0: exact; NBLK < 0: compile-time upper bound; 0: runtime; TWO: per chunk)
    const int nblk = TWO ? (ch == 0 ? NB : NBLK)
                         : (NBLK > 0 ? NBLK : (NBLK < 0 ? min(NBMAX, (Lc + BS - 1) / BS) : (Lc + BS - 1) / BS));
    // one-block chunk: k-split KK/QK + fp32 inverse (compile-time decision in the NBMAX / TWO kernels)
    const bool KSPLIT = (NBMAX > 0 && !TWO) ? (NBMAX == 1) : (nblk == 1);
    // KK/QK warp table selector (compile-time block count; 0 / 1 / 4 -> the 4-block table). The NBLK < 0
    // (N == 2) kernels keep the upper bound's table: selecting the table / k-split from the per-sequence
    // runtime block count measured 0.2-0.5 us slower on the N == 2 workloads (Phase 8 D1).
    const int tsel = TWO ? nblk : NBMAX;
    const int64_t tbase = s0 + c0;
    const bool has_next = SINGLE ? false : (ch + 1 < nchunks);
    GDN_TS(0);

    // ---- prefetch next chunk into the other buffer, wait for this chunk's tiles ----------------
    // a/b rows of the next chunk go out now (its gates are computed during this chunk's inverse);
    // its K/Q/V tiles are issued by warps 4-7 during the solve (off the critical path).
    const bool pre_done = PRE && (ch > 0);   // this chunk's A / P / T / G were built during the previous chunk
    if (has_next) {
      prefetch_ab(sm, buf ^ 1, a, b, tbase + C, min(C, L - c0 - C), tid);
      if (PRE && tid < 128) prefetch_kqv<VS, 128>(sm, buf ^ 1, q, k, v, tbase + C, min(C, L - c0 - C), hq, h, v0, tid);
      GDN_TS(1);
      if (PRE && tid < 128) cp_async_wait<2>(); else cp_async_wait<1>();   // this chunk's tiles (older groups) have landed
    } else {
      GDN_TS(1);
      cp_async_wait<0>();
    }
    // chunk 0 of EARLY kernels: tiles visible to warps 0-6 (the only issuers) through barrier 3 while warp 7 is
    // still in the gates; if warp 7 has KK/QK units it waits for the tiles on barrier 4 after the gates
    const bool early = EARLY && (ch == 0);
    const bool w7units = !KSPLIT && tsel != 2;   // CTA-uniform: the 3- and 4-block tables give warp 7 units
    if (early) {
      if (warp < 7) { named_bar_sync(3, PF_NTH); if (w7units) named_bar_arrive(4, NT); }
      else if (w7units) named_bar_sync(4, NT);
    } else {
      __syncthreads();
    }
    GDN_TS(2);
    const float* gam = sm.gamma[buf];
    const float* bet = sm.beta[buf];

    // ---- phase 2: KK^T / QK^T for lower-triangular 16x16 blocks, 20 (block, n8) units -------------
#ifndef GDN_EXP_NO_P2
    if (!pre_done) {
      // zero P upper blocks (bj > bi): 6 blocks x 16 rows x 16 bf16 = 192 x 16 B
      if (tid < 192) {
        const int ub = tid >> 5;             // 0..5 -> (bi,bj) in {(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}
        const int rr = tid & 31;             // row (0..15) x half (0..1)
        const int ubi = (ub < 3) ? 0 : (ub < 5 ? 1 : 2);
        const int ubj = (ub < 3) ? ub + 1 : (ub < 5 ? ub - 1 : 3);
        const int i = ubi * 16 + (rr >> 1);
        const int j = ubj * 16 + (rr & 1) * 8;
        *reinterpret_cast<uint4*>(sm.P16[pb] + i * LDS + j) = make_uint4(0u, 0u, 0u, 0u);
      }
      if (KSPLIT) {
        // <= 16 valid tokens: only block (0,0). Split the K=128 contraction (8 k-steps) over warps 0-6
        // (warp 6 takes two), reduce the 16x16 partials through smem (A32 is free scratch), then build
        // A/P. In chunk 0 warp 7 runs the gate scan meanwhile.
        float* red = sm.A32;   // [8 k-steps][2 kinds][16][16] fp32 = 16 KB scratch
        if (warp < 7) {
          float KK[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          float QK[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
          for (int kk2 = 0; kk2 < 2; ++kk2) {
            const int ks = (kk2 == 0) ? warp : 7;
            if (kk2 == 0 || warp == 6) {
              uint32_t aK[4], aQ[4], bb[4];
              ldA_rm(aK, sm.K[buf], 0, ks * 16, LDK, lane);
              ldA_rm(aQ, sm.Q[buf], 0, ks * 16, LDK, lane);
              ldB_nm(bb, sm.K[buf], 0, ks * 16, LDK, lane);
              mma16816(KK[0], aK, bb);
              mma16816(KK[1], aK, bb + 2);
              mma16816(QK[0], aQ, bb);
              mma16816(QK[1], aQ, bb + 2);
            }
          }
          GDN_TS(13);
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const int j = nt * 8 + (lane & 3) * 2;
            const int i = lane >> 2;
            *reinterpret_cast<float2*>(red + (warp * 2 + 0) * 256 + i * 16 + j) = make_float2(KK[nt][0], KK[nt][1]);
            *reinterpret_cast<float2*>(red + (warp * 2 + 0) * 256 + (i + 8) * 16 + j) = make_float2(KK[nt][2], KK[nt][3]);
            *reinterpret_cast<float2*>(red + (warp * 2 + 1) * 256 + i * 16 + j) = make_float2(QK[nt][0], QK[nt][1]);
            *reinterpret_cast<float2*>(red + (warp * 2 + 1) * 256 + (i + 8) * 16 + j) = make_float2(QK[nt][2], QK[nt][3]);
          }
        }
        __syncthreads();
        GDN_TS(14);
        if constexpr (OLDS) {
        if (tid >= 128) {   // warps 4-7: zero-fill the solve outputs of padded block rows 16..63
          const int rows = (NB - 1) * BS, t4 = tid - 128;
          for (int idx = t4; idx < rows * (D / 8); idx += 128) {
            const int t = BS + idx / (D / 8);
            const int cc = (idx % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.old.W + t * LDK + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = t4; idx < rows * (VW / 8); idx += 128) {
            const int t = BS + idx / (VW / 8);
            const int cc = (idx % (VW / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.old.XU + t * LDV + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = t4; idx < rows * (VW / 4); idx += 128) {
            const int t = BS + idx / (VW / 4);
            const int cc = (idx % (VW / 4)) * 4;
            *reinterpret_cast<float4*>(sm.old.U32 + t * LDU + cc) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        }
        if (tid < 128) {
          // thread -> (row i, column pair j, j+1) of the 16x16 block: 32-bit A/P writes
          const int i = tid >> 3, j = (tid & 7) * 2;
          float2 kk = make_float2(0.f, 0.f), qk = make_float2(0.f, 0.f);
#pragma unroll
          for (int w = 0; w < 7; ++w) {
            const float2 a = *reinterpret_cast<const float2*>(red + (w * 2 + 0) * 256 + i * 16 + j);
            const float2 b = *reinterpret_cast<const float2*>(red + (w * 2 + 1) * 256 + i * 16 + j);
            kk.x += a.x; kk.y += a.y; qk.x += b.x; qk.y += b.y;
          }
          const float gi = gam[i], bi_ = bet[i];
          const float d0 = __expf(gi - gam[j]), d1 = __expf(gi - gam[j + 1]);
          const float a0 = (j < i) ? bi_ * kk.x * d0 : 0.f;
          const float a1 = (j + 1 < i) ? bi_ * kk.y * d1 : 0.f;
          const float p0 = (j <= i) ? qk.x * d0 : 0.f;
          const float p1 = (j + 1 <= i) ? qk.y * d1 : 0.f;
          *reinterpret_cast<uint32_t*>(sm.A16[pb] + i * LDS + j) = pack_bf16(a0, a1);
          *reinterpret_cast<uint32_t*>(sm.P16[pb] + i * LDS + j) = pack_bf16(p0, p1);
          sm.ATf[pb][j * BS + i] = a0;
          sm.ATf[pb][(j + 1) * BS + i] = a1;
        }
      } else {
        // warp -> (block row bi, up to 3 n8 column tiles j0) of the lower-triangular units; the table
        // depends on the number of valid blocks (one code path: the table is a few selects, so the
        // kernel does not grow; inlined per-case copies cost more in cold instruction fetch than they save).
        // 4 blocks (20 units): w0: bi0 {0,8} | w1: bi1 {0,8,16} | w2: bi1 {24} | w3: bi2 {0,8,16}
        //   | w4: bi2 {24,32,40} | w5: bi3 {0,8,16} | w6: bi3 {24,32,40} | w7: bi3 {48,56}
        // 3 blocks (12 units, 3 per SMSP as warps w and w+4 share one): w0: bi0 {0,8} | w1: bi1 {0,8}
        //   | w2: bi1 {16,24} | w3: bi2 {0,8} | w4: bi2 {40} | w5: bi2 {32} | w6: bi2 {24} | w7: bi2 {16}
        // 2 blocks (6 units): w0: bi0 {0} | w1: bi0 {8} | w2..w5: bi1 {0,8,16,24} | w6, w7: none
        // (every warp carries chains of 8 dependent MMAs per unit: the phase is latency-bound, so all
        // eight warps take part; fewer warps with more units each was measured 2x slower)
        int bi, j0a, j0b, j0c;
        // (compile-time NBLK kernels only: runtime-selected tables in the multi-chunk kernels measured
        // 0.1-0.3 us slower on 49-132 token workloads although their chunk-0 table is unchanged)
        if (tsel == 2) {
          bi = (warp < 2) ? 0 : (warp < 6 ? 1 : 3);
          j0a = (warp < 2) ? warp * 8 : (warp - 2) * 8;
          j0b = -1; j0c = -1;
        } else if (tsel == 3) {
          bi = (warp == 0) ? 0 : (warp <= 2 ? 1 : 2);
          j0a = (warp == 2) ? 16 : (warp <= 3 ? 0 : (7 - warp) * 8 + 16);
          j0b = (warp <= 3) ? j0a + 8 : -1;
          j0c = -1;
        } else {
          bi = (warp == 0) ? 0 : (warp <= 2 ? 1 : (warp <= 4 ? 2 : 3));
          const bool first = (warp == 0 || warp == 1 || warp == 3 || warp == 5);
          j0a = first ? 0 : (warp == 7 ? 48 : 24);
          j0b = first ? 8 : (warp == 2 ? -1 : (warp == 7 ? 56 : 32));
          j0c = (warp == 1 || warp == 3 || warp == 5) ? 16 : ((warp == 4 || warp == 6) ? 40 : -1);
        }
        const int i0 = bi * 16;
        float KK[3][4], QK[3][4];
        if (i0 < Lc) {
#pragma unroll
          for (int u = 0; u < 3; ++u)
#pragma unroll
            for (int e = 0; e < 4; ++e) { KK[u][e] = 0.f; QK[u][e] = 0.f; }
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t aK[4], aQ[4], b0[2], b1[2], b2[2];
            ldA_rm(aK, sm.K[buf], i0, ks * 16, LDK, lane);
            ldA_rm(aQ, sm.Q[buf], i0, ks * 16, LDK, lane);
            ldB_nm1(b0, sm.K[buf], j0a, ks * 16, LDK, lane);
            if (j0b >= 0) ldB_nm1(b1, sm.K[buf], j0b, ks * 16, LDK, lane);
            if (j0c >= 0) ldB_nm1(b2, sm.K[buf], j0c, ks * 16, LDK, lane);
            mma16816(KK[0], aK, b0);
            mma16816(QK[0], aQ, b0);
            if (j0b >= 0) { mma16816(KK[1], aK, b1); mma16816(QK[1], aQ, b1); }
            if (j0c >= 0) { mma16816(KK[2], aK, b2); mma16816(QK[2], aQ, b2); }
          }
        }
        if (early) named_bar_sync(2, NT);   // chunk-0 gates (warp 7) visible before the A/P epilogue
        if (i0 < Lc) {
#pragma unroll
          for (int u = 0; u < 3; ++u) {
            const int j0 = (u == 0) ? j0a : (u == 1 ? j0b : j0c);
            if (j0 >= 0) {
              const int j = j0 + (lane & 3) * 2;
              const float gj0 = gam[j], gj1 = gam[j + 1];
              const bool diag = (j0 >= i0);   // n8 tile inside the diagonal 16x16 block
#pragma unroll
              for (int half = 0; half < 2; ++half) {
                const int i = i0 + (lane >> 2) + half * 8;
                const float gi = gam[i], bi_ = bet[i];
                const float d0 = __expf(gi - gj0), d1 = __expf(gi - gj1);
                const float a0 = (j < i) ? bi_ * KK[u][half * 2] * d0 : 0.f;
                const float a1 = (j + 1 < i) ? bi_ * KK[u][half * 2 + 1] * d1 : 0.f;
                *reinterpret_cast<uint32_t*>(sm.A16[pb] + i * LDS + j) = pack_bf16(a0, a1);
                if (diag) {   // bf16 transposed copy for the inverse: AT16[bi][jl][il]
                  const int il = i - i0, jl = j - i0;
                  sm.AT16[pb][bi][jl * LDT + il] = __float2bfloat16(a0);
                  sm.AT16[pb][bi][(jl + 1) * LDT + il] = __float2bfloat16(a1);
                }
                const float p0 = (j <= i) ? QK[u][half * 2] * d0 : 0.f;
                const float p1 = (j + 1 <= i) ? QK[u][half * 2 + 1] * d1 : 0.f;
                *reinterpret_cast<uint32_t*>(sm.P16[pb] + i * LDS + j) = pack_bf16(p0, p1);
              }
            }
          }
        }
        if constexpr (OLDS) {
        // zero-fill solve outputs for padded block rows (never computed below)
        if (nblk < NB) {
          const int rows = (NB - nblk) * BS;       // padded rows: nblk*16 .. 63
          for (int idx = tid; idx < rows * (D / 8); idx += NT) {  // W: 16 B per 8 bf16
            const int t = nblk * BS + idx / (D / 8);
            const int cc = (idx % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.old.W + t * LDK + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = tid; idx < rows * (VW / 8); idx += NT) {
            const int t = nblk * BS + idx / (VW / 8);
            const int cc = (idx % (VW / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.old.XU + t * LDV + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = tid; idx < rows * (VW / 4); idx += NT) {
            const int t = nblk * BS + idx / (VW / 4);
            const int cc = (idx % (VW / 4)) * 4;
            *reinterpret_cast<float4*>(sm.old.U32 + t * LDU + cc) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        }
      }
    }
#endif
    GDN_TS(12);
    if (has_next) cp_async_wait<0>();   // next chunk's a/b rows have landed
    __syncthreads();
    GDN_TS(3);

    // Reassociated Delta (Phase 7): Delta = T (beta (V - e^gamma (K S))) instead of U - W S, so the per-CTA
    // 64x128 W solve disappears. Phase 3a: the warps idle during the inverse compute o_inter^T = S^T Q^T
    // (fp32 -> O32) and R^T = beta (V^T - e^gamma S^T K^T) (bf16 hi/lo -> Rh / Rl) in (block, product, token
    // half) units. Phase 3b/4: block forward substitution X_r = R_r^T T_rr^T - sum_{s<r} X_s G_rs^T, stage r on
    // the 2*NMT warps of stage slot r % NS (one (m-tile, token half) tile each); stage 0 runs beside the G
    // pairs, later stages hand over through named barriers (arrive / sync). Phase 5 = o += Delta^T P^T.
    constexpr int NS = 8 / (2 * NMT);        // stage slots
    const int sti = warp % (2 * NMT);        // tile index inside a stage: m-tile sti % NMT (== mt), token half
    const int shalf = sti / NMT;
    const int sslot = warp / (2 * NMT);
    // phase-4a units of this warp's m-tile: u -> block u / 2, product u & 1 (0: o_inter^T = e^gamma S^T Q^T -> O32,
    // 1: R^T = beta (V^T - e^gamma S^T K^T) -> Rh / Rl), both token halves (ldmatrix x4); units u0 + uidx + i*ucnt,
    // two per iteration so that four MMA chains interleave
    auto unit_ep1 = [&](const int r, const int prod, const int h2, const float* acc) {
      const int vr = mt * 16 + (lane >> 2);
      const int tk = r * BS + h2 * 8 + (lane & 3) * 2;
      const float2 e2 = *reinterpret_cast<const float2*>(&sm.eg[buf][tk]);
      if (prod == 0) {
        *reinterpret_cast<float2*>(sm.nw.O32 + vr * LDD + tk) = make_float2(acc[0] * e2.x, acc[1] * e2.y);
        *reinterpret_cast<float2*>(sm.nw.O32 + (vr + 8) * LDD + tk) = make_float2(acc[2] * e2.x, acc[3] * e2.y);
      } else {
        const float2 b2 = *reinterpret_cast<const float2*>(&sm.beta[buf][tk]);
        const float r0 = b2.x * (__bfloat162float(sm.V[buf][tk * LDV + vr]) - e2.x * acc[0]);
        const float r1 = b2.y * (__bfloat162float(sm.V[buf][(tk + 1) * LDV + vr]) - e2.y * acc[1]);
        const float r2 = b2.x * (__bfloat162float(sm.V[buf][tk * LDV + vr + 8]) - e2.x * acc[2]);
        const float r3 = b2.y * (__bfloat162float(sm.V[buf][(tk + 1) * LDV + vr + 8]) - e2.y * acc[3]);
        *reinterpret_cast<uint32_t*>(sm.nw.Rh + vr * LDD + tk) = pack_bf16(r0, r1);
        *reinterpret_cast<uint32_t*>(sm.nw.Rh + (vr + 8) * LDD + tk) = pack_bf16(r2, r3);
        *reinterpret_cast<uint32_t*>(sm.nw.Rl + vr * LDD + tk) = pack_bf16(bf16_residual(r0), bf16_residual(r1));
        *reinterpret_cast<uint32_t*>(sm.nw.Rl + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(r2), bf16_residual(r3));
      }
    };
    // full units (block, product): both token halves from one ldmatrix x4 per k-step; all B fragments of the
    // two units are loaded before the 32 MMAs (asm volatile keeps program order, so loads must be hoisted by hand)
    auto run_units = [&](const int uidx, const int ucnt, const int u0) {
      const int nu = 2 * nblk;
      for (int u = u0 + uidx; u < nu; u += 2 * ucnt) {
        const int u2 = u + ucnt;
        const bool has2 = (u2 < nu);
        const int ub = has2 ? u2 : u;
        const __nv_bfloat16* ta = (u & 1) ? sm.K[buf] : sm.Q[buf];
        const __nv_bfloat16* tb = (ub & 1) ? sm.K[buf] : sm.Q[buf];
        const int na = (u >> 1) * BS, nb = (ub >> 1) * BS;
        uint32_t ba[8][4], bb[8][4];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          ldB_nm(ba[ks], ta, na, ks * 16, LDK, lane);
          ldB_nm(bb[ks], tb, nb, ks * 16, LDK, lane);
        }
        float acca[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}, accb[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          mma16816(acca[0], Sb[ks], ba[ks]);
          mma16816(accb[0], Sb[ks], bb[ks]);
          mma16816(acca[1], Sb[ks], ba[ks] + 2);
          mma16816(accb[1], Sb[ks], bb[ks] + 2);
        }
        unit_ep1(u >> 1, u & 1, 0, acca[0]);
        unit_ep1(u >> 1, u & 1, 1, acca[1]);
        if (has2) { unit_ep1(ub >> 1, ub & 1, 0, accb[0]); unit_ep1(ub >> 1, ub & 1, 1, accb[1]); }
      }
    };
    // half units (block, product, token half) for chunks with <= 2 blocks: 4 nblk units of 8 MMAs over the
    // 6-7 idle warps
    auto run_half_units = [&](const int uidx, const int ucnt) {
      const int nu = 4 * nblk;
      for (int u = uidx; u < nu; u += 2 * ucnt) {
        const int u2 = u + ucnt;
        const bool has2 = (u2 < nu);
        const int ub = has2 ? u2 : u;
        const __nv_bfloat16* ta = ((u >> 1) & 1) ? sm.K[buf] : sm.Q[buf];
        const __nv_bfloat16* tb = ((ub >> 1) & 1) ? sm.K[buf] : sm.Q[buf];
        const int na = (u >> 2) * BS + (u & 1) * 8, nb = (ub >> 2) * BS + (ub & 1) * 8;
        uint32_t ba[8][2], bb[8][2];
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          ldB_nm1(ba[ks], ta, na, ks * 16, LDK, lane);
          ldB_nm1(bb[ks], tb, nb, ks * 16, LDK, lane);
        }
        float acca[4] = {0.f, 0.f, 0.f, 0.f}, accb[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          mma16816(acca, Sb[ks], ba[ks]);
          mma16816(accb, Sb[ks], bb[ks]);
        }
        unit_ep1(u >> 2, (u >> 1) & 1, u & 1, acca);
        if (has2) unit_ep1(ub >> 2, (ub >> 1) & 1, ub & 1, accb);
      }
    };
    // k-split units for chunks with <= 2 blocks (Phase 8 D3c, REJECTED: slower than the half units on every <= 2-block
    // workload; kept under -DGDN_KSPLIT_UNITS): the 4 (prod, half) tiles of each block are accumulated
    // over disjoint k-step sets by the idle warps (nblk = 1: 7 warps, k-steps j, j + 7; nblk = 2: 3 warps per
    // block, k-steps j, j + 3, j + 6): no dependent MMA chains, no duplicated loads; the fp32 partials are reduced
    // through A32 after a named barrier among the idle warps, one tile per warp, straight into the unit epilogue.
    auto run_ksplit_units = [&]() {
      const int nw = 8 - nblk;               // idle warps nblk .. 7
      const int iw = warp - nblk;
      const int wpb = (nblk == 1) ? 7 : 3;   // warps per block
      const int b = (nblk == 1) ? 0 : (iw >= 3 ? 1 : 0);
      const int j = iw - b * wpb;
      float* red = sm.A32;                   // [block * wpb + j][4 tiles][32 lanes][4] fp32 (<= 14 KB)
      {
        float acc[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) { acc[t][0] = 0.f; acc[t][1] = 0.f; acc[t][2] = 0.f; acc[t][3] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          if (ks % wpb == j) {
            uint32_t bq[4], bk[4];
            ldB_nm(bq, sm.Q[buf], b * BS, ks * 16, LDK, lane);
            ldB_nm(bk, sm.K[buf], b * BS, ks * 16, LDK, lane);
            mma16816(acc[0], Sb[ks], bq);
            mma16816(acc[1], Sb[ks], bq + 2);
            mma16816(acc[2], Sb[ks], bk);
            mma16816(acc[3], Sb[ks], bk + 2);
          }
        }
        float* dst = red + ((b * wpb + j) * 4) * 128 + lane * 4;
#pragma unroll
        for (int t = 0; t < 4; ++t)
          *reinterpret_cast<float4*>(dst + t * 128) = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
      }
      named_bar_sync(3, nw * 32);
      for (int ti = iw; ti < 4 * nblk; ti += nw) {
        const int tb = ti >> 2, tp = (ti >> 1) & 1, th = ti & 1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int jj = 0; jj < wpb; ++jj) {
          const float4 x = *reinterpret_cast<const float4*>(red + ((tb * wpb + jj) * 4 + tp * 2 + th) * 128 + lane * 4);
          acc[0] += x.x; acc[1] += x.y; acc[2] += x.z; acc[3] += x.w;
        }
        unit_ep1(tb, tp, th, acc);
      }
    };
    // T_rr^T (and Tlo) of block r from the parity buffer pbx: fp32 column substitution (ks: fp32 rows from ATf for
    // one-block chunks, else bf16 rows from AT16)
    auto do_inverse = [&](const int pbx, const int r, const bool ks) {
      // fp32 column substitution of the 16x16 unit-lower-triangular (I + A_rr): lane c (lanes 16-31
      // replicate lanes 0-15) owns column c of T. With t_j = T[j][c] finalised in order j = 0..15:
      //   P[i] += A[i][j] t_j (i > j),  t_j = (j == c) ? 1 : -P[j].
      // A_rr is read transposed (ATf[j][i], fp32, broadcast loads), no smem round trips, no __syncwarp.
      const int c = lane & 15;
      float P[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) P[i] = 0.f;
      float Tc[16];
      if (ks) {
        // fp32 rows (one block: a single warp, latency-bound), loaded LA rows ahead
        const float* At = sm.ATf[pbx];
        constexpr int LA = 4;
        float4 Q[15][4];
        auto ld_row = [&](int j) {
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4)
            if (q4 * 4 + 3 > j) {
              const uint32_t addr = smem_u32(At + j * BS + q4 * 4);
              asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                           : "=f"(Q[j][q4].x), "=f"(Q[j][q4].y), "=f"(Q[j][q4].z), "=f"(Q[j][q4].w) : "r"(addr));
            }
        };
#pragma unroll
        for (int j = 0; j < LA; ++j) ld_row(j);
#pragma unroll
        for (int j = 0; j < 15; ++j) {
          if (j + LA < 15) ld_row(j + LA);
          const float tj = (j == c) ? 1.f : -P[j];
          Tc[j] = tj;
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            if (q4 * 4 + 3 > j) {
              const float av[4] = {Q[j][q4].x, Q[j][q4].y, Q[j][q4].z, Q[j][q4].w};
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int i = q4 * 4 + e;
                if (i > j) P[i] = fmaf(av[e], tj, P[i]);
              }
            }
          }
        }
      } else {
        // bf16 rows (several warps at once: half the smem traffic of fp32 rows), all loaded up front
        const __nv_bfloat16* At = sm.AT16[pbx][r];
        uint4 R[15][2];
#pragma unroll
        for (int j = 0; j < 15; ++j) {
          if (j < 7) R[j][0] = *reinterpret_cast<const uint4*>(At + j * LDT);   // i = 1..7 live only for j < 7
          else R[j][0] = make_uint4(0u, 0u, 0u, 0u);
          R[j][1] = *reinterpret_cast<const uint4*>(At + j * LDT + 8);
        }
#pragma unroll
        for (int j = 0; j < 15; ++j) {
          const float tj = (j == c) ? 1.f : -P[j];
          Tc[j] = tj;
          const uint32_t w[8] = {R[j][0].x, R[j][0].y, R[j][0].z, R[j][0].w, R[j][1].x, R[j][1].y, R[j][1].z, R[j][1].w};
#pragma unroll
          for (int i = j + 1; i < 16; ++i) {
            const uint32_t u = w[i >> 1];
            const float av = __uint_as_float((i & 1) ? (u & 0xffff0000u) : (u << 16));
            P[i] = fmaf(av, tj, P[i]);
          }
        }
      }
      Tc[15] = (c == 15) ? 1.f : -P[15];
      GDN_TS(10);
      {   // transposed writes: lane c writes row c of T^T (= column c of T); lanes 0-15 take rows
          // i = 0..7 of the column, lanes 16-31 rows 8..15 (8 elements each, 16 B stores). The column-scaled
          // variants TK / TV (and their low parts) exist only for the v10.0 solve (OLDS).
        const bool up = lane >= 16;
        auto split = [](float a, float b, uint32_t& hi_, uint32_t& lo_) {   // bf16 hi/lo pair split
          hi_ = pack_bf16(a, b);
          const float ha = __uint_as_float(hi_ << 16), hb = __uint_as_float(hi_ & 0xffff0000u);
          lo_ = pack_bf16(a - ha, b - hb);
        };
        const int off = c * LDT + (up ? 8 : 0);   // 16 B aligned (LDT = 24 -> 48 B rows)
        uint32_t t[4], tl[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float ta = up ? Tc[8 + 2 * e] : Tc[2 * e];
          const float tb = up ? Tc[9 + 2 * e] : Tc[2 * e + 1];
          split(ta, tb, t[e], tl[e]);
        }
        *reinterpret_cast<uint4*>(sm.T[pbx][r] + off) = make_uint4(t[0], t[1], t[2], t[3]);
        *reinterpret_cast<uint4*>(sm.Tlo[pbx][r] + off) = make_uint4(tl[0], tl[1], tl[2], tl[3]);
        if constexpr (OLDS) {
          const float cV = bet[r * BS + c];
          const float cK = cV * __expf(gam[r * BS + c]);
          uint32_t k[4], kl[4], v[4], vl[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float ta = up ? Tc[8 + 2 * e] : Tc[2 * e];
            const float tb = up ? Tc[9 + 2 * e] : Tc[2 * e + 1];
            split(ta * cK, tb * cK, k[e], kl[e]);
            split(ta * cV, tb * cV, v[e], vl[e]);
          }
          *reinterpret_cast<uint4*>(sm.TK[r] + off) = make_uint4(k[0], k[1], k[2], k[3]);
          *reinterpret_cast<uint4*>(sm.TKlo[r] + off) = make_uint4(kl[0], kl[1], kl[2], kl[3]);
          *reinterpret_cast<uint4*>(sm.TV[r] + off) = make_uint4(v[0], v[1], v[2], v[3]);
          *reinterpret_cast<uint4*>(sm.TVlo[r] + off) = make_uint4(vl[0], vl[1], vl[2], vl[3]);
        }
      }
      GDN_TS(11);
    };
    // G_rs = T_rr A_rs (bf16) for block pair pi of the parity buffer pbx (nb valid blocks)
    auto g_pair = [&](const int pbx, const int pi, const int nb) {
      const int r = (pi == 0) ? 1 : (pi <= 2 ? 2 : 3);
      const int s_ = (pi == 0) ? 0 : (pi <= 2 ? pi - 1 : pi - 3);
      if (r < nb) {
        float g[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        uint32_t at[4], ba[4];
        ldB_km(ba, sm.A16[pbx], s_ * BS, r * BS, LDS, lane);   // B[k=l][n=j] = A[r*16+l][s*16+j]
        ldA_km(at, sm.T[pbx][r], 0, 0, LDT, lane);
        mma16816(g[0], at, ba);
        mma16816(g[1], at, ba + 2);
        ldA_km(at, sm.Tlo[pbx][r], 0, 0, LDT, lane);
        mma16816(g[0], at, ba);
        mma16816(g[1], at, ba + 2);
        const int gi = lane >> 2, gj = (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(sm.G16[pbx][pi] + gi * LDT + gj) = pack_bf16(g[0][0], g[0][1]);
        *reinterpret_cast<uint32_t*>(sm.G16[pbx][pi] + (gi + 8) * LDT + gj) = pack_bf16(g[0][2], g[0][3]);
        *reinterpret_cast<uint32_t*>(sm.G16[pbx][pi] + gi * LDT + gj + 8) = pack_bf16(g[1][0], g[1][1]);
        *reinterpret_cast<uint32_t*>(sm.G16[pbx][pi] + (gi + 8) * LDT + gj + 8) = pack_bf16(g[1][2], g[1][3]);
      }
    };
    // D7: chunk c+1's A / P / AT16 (KK^T, QK^T units over the lower triangle, round-robin over warps 0-3), T / Tlo
    // (inverse per block) and G pairs into the other parity buffer, from its tiles (issued by warps 0-3 at this
    // chunk's start) and gates (warps 4-7 arrive on barrier 5 after the scan)
    auto pre_next_chunk = [&]() {
      const int Ln = min(C, L - c0 - C);
      const int nbn = TWO ? NBLK : (Ln + BS - 1) / BS;
      const int pn = pb ^ 1, bn = buf ^ 1;
      const float* gamn = sm.gamma[bn];
      const float* betn = sm.beta[bn];
      cp_async_wait<0>();          // next chunk's tiles (this thread's copies)
      named_bar_sync(5, NT);       // gates of the next chunk (warps 4-7 arrive) + tiles visible across warps 0-3
      // (block row bi, n8 tile j0) units of the lower triangle: NBLK = 1 -> (0,0) (0,8); NBLK = 2 -> + (1,0) .. (1,24);
      // warp w takes units w and w + 4 as two interleaved MMA chains
      auto kk_epi = [&](const int bi, const int j0, const float* KK, const float* QK) {
        const int i0 = bi * BS;
        const int j = j0 + (lane & 3) * 2;
        const float gj0 = gamn[j], gj1 = gamn[j + 1];
        const bool diag = (j0 >= i0);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = i0 + (lane >> 2) + half * 8;
          const float gi = gamn[i], bi_ = betn[i];
          const float d0 = __expf(gi - gj0), d1 = __expf(gi - gj1);
          const float a0 = (j < i) ? bi_ * KK[half * 2] * d0 : 0.f;
          const float a1 = (j + 1 < i) ? bi_ * KK[half * 2 + 1] * d1 : 0.f;
          *reinterpret_cast<uint32_t*>(sm.A16[pn] + i * LDS + j) = pack_bf16(a0, a1);
          if (diag) {
            const int il = i - i0, jl = j - i0;
            sm.AT16[pn][bi][jl * LDT + il] = __float2bfloat16(a0);
            sm.AT16[pn][bi][(jl + 1) * LDT + il] = __float2bfloat16(a1);
          }
          const float p0 = (j <= i) ? QK[half * 2] * d0 : 0.f;
          const float p1 = (j + 1 <= i) ? QK[half * 2 + 1] * d1 : 0.f;
          *reinterpret_cast<uint32_t*>(sm.P16[pn] + i * LDS + j) = pack_bf16(p0, p1);
        }
      };
      {
        constexpr int NU = NBMAX * (NBMAX + 1);   // 2 or 6 units
        const int u1 = warp, u2 = warp + 4;
        const bool h2 = (u2 < NU);
        if (u1 < NU) {
          const int bi1 = (u1 < 2) ? 0 : 1, j01 = (u1 < 2) ? u1 * 8 : (u1 - 2) * 8;
          const int bi2 = (u2 < 2) ? 0 : 1, j02 = (u2 < 2) ? u2 * 8 : (u2 - 2) * 8;
          float KK1[4] = {0.f, 0.f, 0.f, 0.f}, QK1[4] = {0.f, 0.f, 0.f, 0.f};
          float KK2[4] = {0.f, 0.f, 0.f, 0.f}, QK2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t aK1[4], aQ1[4], b1[2], aK2[4], aQ2[4], b2[2];
            ldA_rm(aK1, sm.K[bn], bi1 * BS, ks * 16, LDK, lane);
            ldA_rm(aQ1, sm.Q[bn], bi1 * BS, ks * 16, LDK, lane);
            ldB_nm1(b1, sm.K[bn], j01, ks * 16, LDK, lane);
            if (h2) {
              ldA_rm(aK2, sm.K[bn], bi2 * BS, ks * 16, LDK, lane);
              ldA_rm(aQ2, sm.Q[bn], bi2 * BS, ks * 16, LDK, lane);
              ldB_nm1(b2, sm.K[bn], j02, ks * 16, LDK, lane);
            }
            mma16816(KK1, aK1, b1);
            mma16816(QK1, aQ1, b1);
            if (h2) { mma16816(KK2, aK2, b2); mma16816(QK2, aQ2, b2); }
          }
          kk_epi(bi1, j01, KK1, QK1);
          if (h2) kk_epi(bi2, j02, KK2, QK2);
        }
      }
      named_bar_sync(5, 128);      // A / AT16 of the next chunk visible among warps 0-3
      if (warp < nbn) do_inverse(pn, warp, false);
      named_bar_sync(5, 128);      // T / Tlo visible
      for (int pi = warp; pi < NG; pi += 4) g_pair(pn, pi, nbn);
    };
    // ---- phase 3a: T_rr = (I + A_rr)^-1 for valid diagonal blocks (warp r < nblk, fp32 column
    //      substitution) ------------------------------------------------------------------------------
#ifdef GDN_EXP_NO_INV
    if (false) {
#else
    if (!pre_done && warp < (OLDS ? NB : nblk)) {
#endif
      const int r = warp;
      if (!OLDS || r < nblk) {
        do_inverse(pb, r, KSPLIT);
        if constexpr (PRE) { if (has_next && warp < 4) pre_next_chunk(); }
#ifndef GDN_EXP_NO_4A
        if constexpr (!OLDS) {
        {   // units beyond two per idle warp (VS = 4 only): the inverse warps take them after their block
          int ni, uidx2, ucnt2;
          if constexpr (NMT == 1) { ni = 8 - nblk; uidx2 = warp; ucnt2 = nblk; }
          else { ni = (8 - nblk + (((nblk % NMT) == mt) ? 1 : 0)) / NMT; uidx2 = warp / NMT; ucnt2 = (nblk + (NMT - 1 - mt)) / NMT; }
          if (2 * ni < 2 * nblk) run_units(uidx2, ucnt2, 2 * ni);
        }
        }
#endif
      }
    } else {
      if (warp >= 4 && has_next) {
        // warps 4-7: gates of the next chunk (its a/b rows landed before the barrier above):
        // part 1 on 16 lanes per warp, named barrier among the four warps, scan on warp 7
        if (lane < 16) gates_token(sm, buf ^ 1, buf ^ 1, min(C, L - c0 - C), h, neg_expA, dtb, (warp - 4) * 16 + lane);
        named_bar_sync(1, 128);
        if (warp == 7) gates_scan(sm, buf ^ 1, lane);
        if constexpr (PRE) {
          named_bar_arrive(5, NT);   // hand the next chunk's gates to warps 0-3 (pre_next_chunk)
        } else if constexpr (!OLDS) {
          // K/Q/V tiles of the next chunk (off the critical path: the inverse is longer than this issue)
          prefetch_kqv<VS, 128>(sm, buf ^ 1, q, k, v, tbase + C, min(C, L - c0 - C), hq, h, v0, tid - 128);
        }
      }
      if constexpr (!OLDS) {
      {
        // ---- phase 4a on the warps idle during the inverse (nblk .. 7, round-robin per m-tile) ----------
        int uidx, ucnt;
        if constexpr (NMT == 1) { uidx = pre_done ? warp : warp - nblk; ucnt = pre_done ? 8 : 8 - nblk; }
        else { uidx = (warp - nblk) / NMT; ucnt = (8 - nblk + (((nblk % NMT) == mt) ? 1 : 0)) / NMT; }
#ifndef GDN_EXP_NO_4A
#ifdef GDN_KSPLIT_UNITS   // rejected (Phase 8 D3c): +0.1 .. +0.5 us on every <= 2-block workload
        if (nblk <= 2) run_ksplit_units(); else run_units(uidx, ucnt, 0);
#else
        if (nblk <= 2) run_half_units(uidx, ucnt); else run_units(uidx, ucnt, 0);
#endif
        if constexpr (PRE) { if (pre_done && has_next && warp < 4) pre_next_chunk(); }
#endif
      }
      }
    }
    __syncthreads();
    GDN_TS(4);
    GDN_TS(5);
    // ---- phase 3b (warps 0-3): block forward substitution X_r = T_rr (B_r) - sum_{s<r} G_rs X_s with
    //      G_rs = T_rr A_rs computed once per pair (bf16, shared through smem). Warp w owns W columns
    //      32w..32w+31 (4 n-tiles); warps < NUT also own U columns 8w..8w+7. Only the warp's own columns
    //      of X_s are needed, so no CTA barrier is required between block rows.
    //      Warps 4-7 issue the next chunk's K/Q/V tile loads meanwhile. -----------------------------
    const float gC = sm.gC[buf];
    const float eC = __expf(gC);
    if constexpr (OLDS) {
    if (warp < 4) {
      // G pairs pi = warp, warp + 4 (pairs (r, s): 0:(1,0) 1:(2,0) 2:(2,1) 3:(3,0) 4:(3,1) 5:(3,2))
#pragma unroll
      for (int q2 = 0; q2 < 2; ++q2) {
        const int pi = warp + 4 * q2;
        if (pi < NG) {
          const int r = (pi == 0) ? 1 : (pi <= 2 ? 2 : 3);
          const int s_ = (pi == 0) ? 0 : (pi <= 2 ? pi - 1 : pi - 3);
          if (r < nblk) {
            float g[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            uint32_t at[4], ba[4];
            ldB_km(ba, sm.A16[pb], s_ * BS, r * BS, LDS, lane);   // B[k=l][n=j] = A[r*16+l][s*16+j]
            ldA_km(at, sm.T[pb][r], 0, 0, LDT, lane);
            mma16816(g[0], at, ba);
            mma16816(g[1], at, ba + 2);
            ldA_km(at, sm.Tlo[pb][r], 0, 0, LDT, lane);
            mma16816(g[0], at, ba);
            mma16816(g[1], at, ba + 2);
            const int gi = lane >> 2, gj = (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(sm.G16[pb][pi] + gi * LDT + gj) = pack_bf16(g[0][0], g[0][1]);
            *reinterpret_cast<uint32_t*>(sm.G16[pb][pi] + (gi + 8) * LDT + gj) = pack_bf16(g[0][2], g[0][3]);
            *reinterpret_cast<uint32_t*>(sm.G16[pb][pi] + gi * LDT + gj + 8) = pack_bf16(g[1][0], g[1][1]);
            *reinterpret_cast<uint32_t*>(sm.G16[pb][pi] + (gi + 8) * LDT + gj + 8) = pack_bf16(g[1][2], g[1][3]);
          }
        }
      }
      named_bar_sync(3, 128);
      GDN_TS(5);
      uint32_t gf[NG][4];
#pragma unroll
      for (int pi = 0; pi < NG; ++pi) {
        const int r = (pi == 0) ? 1 : (pi <= 2 ? 2 : 3);
        if (r < nblk) ldA_rm(gf[pi], sm.G16[pb][pi], 0, 0, LDT, lane);
        else { gf[pi][0] = gf[pi][1] = gf[pi][2] = gf[pi][3] = 0u; }
      }
      const int wc0 = warp * 32;   // first W column of this warp
#pragma unroll
      for (int r = 0; r < NB; ++r) {
        if (r >= nblk) break;
        const int r0 = r * BS;
        float xw[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { xw[u][0] = xw[u][1] = xw[u][2] = xw[u][3] = 0.f; }
        float xu[4] = {0.f, 0.f, 0.f, 0.f};
        {
          uint32_t aa[4], al[4], bbf[4];
          ldA_km(aa, sm.TK[r], 0, 0, LDT, lane);
          ldA_km(al, sm.TKlo[r], 0, 0, LDT, lane);
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            ldB_km(bbf, sm.K[buf], wc0 + hf * 16, r0, LDK, lane);  // B[k=token][n=dim], two n-tiles
            mma16816(xw[2 * hf], aa, bbf);
            mma16816(xw[2 * hf + 1], aa, bbf + 2);
            mma16816(xw[2 * hf], al, bbf);
            mma16816(xw[2 * hf + 1], al, bbf + 2);
          }
          if (warp < NUT) {
            ldB_km1(bbf, sm.V[buf], warp * 8, r0, LDV, lane);
            ldA_km(aa, sm.TV[r], 0, 0, LDT, lane);
            mma16816(xu, aa, bbf);
            ldA_km(aa, sm.TVlo[r], 0, 0, LDT, lane);
            mma16816(xu, aa, bbf);
          }
        }
#pragma unroll
        for (int s_ = 0; s_ < NB - 1; ++s_) {
          if (s_ < r) {
            const int pi = r * (r - 1) / 2 + s_;   // compile-time after unrolling
            uint32_t bbf[4];
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              ldB_km(bbf, sm.old.W, wc0 + hf * 16, s_ * BS, LDK, lane);  // B = -X_s (W part) -> accumulates -G X_s
              mma16816(xw[2 * hf], gf[pi], bbf);
              mma16816(xw[2 * hf + 1], gf[pi], bbf + 2);
            }
            if (warp < NUT) {
              uint32_t gn[4];
#pragma unroll
              for (int e = 0; e < 4; ++e) gn[e] = gf[pi][e] ^ 0x80008000u;   // -G
              ldB_km1(bbf, sm.old.XU, warp * 8, s_ * BS, LDV, lane);  // B = +X_s (U part), A = -G
              mma16816(xu, gn, bbf);
            }
          }
        }
        // store -W (bf16), U (bf16 + fp32) for this warp's columns
        {
          const int rr = r0 + (lane >> 2);
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int cc = wc0 + nt * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(sm.old.W + rr * LDK + cc) = pack_bf16(-xw[nt][0], -xw[nt][1]);
            *reinterpret_cast<uint32_t*>(sm.old.W + (rr + 8) * LDK + cc) = pack_bf16(-xw[nt][2], -xw[nt][3]);
          }
          if (warp < NUT) {
            const int cc = warp * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(sm.old.XU + rr * LDV + cc) = pack_bf16(xu[0], xu[1]);
            *reinterpret_cast<uint32_t*>(sm.old.XU + (rr + 8) * LDV + cc) = pack_bf16(xu[2], xu[3]);
            *reinterpret_cast<float2*>(sm.old.U32 + rr * LDU + cc) = make_float2(xu[0], xu[1]);
            *reinterpret_cast<float2*>(sm.old.U32 + (rr + 8) * LDU + cc) = make_float2(xu[2], xu[3]);
          }
        }
        __syncwarp();
      }
    } else if (has_next) {
      // warps 4-7: K/Q/V tiles of the next chunk (its a/b rows went out at the chunk start)
      prefetch_kqv<VS, 128>(sm, buf ^ 1, q, k, v, tbase + C, min(C, L - c0 - C), hq, h, v0, tid - 128);
    }
    }
    // ---- phase 3b: G pairs on the warps outside stage slot 0 (pi = warp - 2 NMT, step 8 - 2 NMT); stage 0
    //      of the chain on slot 0 ----------------------------------------------------------------------
    auto chain_stage = [&](const int r) {
      const int vr = mt * 16 + (lane >> 2);
      const int tk = r * BS + shalf * 8 + (lane & 3) * 2;
      float xh[4] = {0.f, 0.f, 0.f, 0.f}, xl[4] = {0.f, 0.f, 0.f, 0.f};
      {
        uint32_t ah[4], al[4], bt[2], btl[2];
        ldA_rm(ah, sm.nw.Rh, mt * 16, r * BS, LDD, lane);          // A[m = v][k = j] = R^T[v][j]
        ldA_rm(al, sm.nw.Rl, mt * 16, r * BS, LDD, lane);
        ldB_km1(bt, sm.T[pb][r], shalf * 8, 0, LDT, lane);          // B[k = j][n = i] = T_rr[i][j]
        ldB_km1(btl, sm.Tlo[pb][r], shalf * 8, 0, LDT, lane);
        mma16816(xh, ah, bt);
        mma16816(xl, al, bt);
        mma16816(xl, ah, btl);
      }
      if (r >= 2) named_bar_sync(4 + r, 2 * 32 * 2 * NMT);    // X_{r-1} (and, transitively, all earlier) stored
      uint32_t ash[NB][4];
#pragma unroll
      for (int s_ = 0; s_ < NB - 1; ++s_) {
        if (s_ < r) {
          const int pi = r * (r - 1) / 2 + s_;
          uint32_t asl[4], bg[2];
          ldA_rm(ash[s_], sm.Dl, mt * 16, s_ * BS, LDD, lane);
          ldA_rm(asl, sm.nw.Dlo, mt * 16, s_ * BS, LDD, lane);
          ldB_nm1(bg, sm.G16[pb][pi], shalf * 8, 0, LDT, lane);      // B[k = j][n = i] = G_rs[i][j]
          bg[0] ^= 0x80008000u; bg[1] ^= 0x80008000u;            // -G
          mma16816(xh, ash[s_], bg);
          mma16816(xl, asl, bg);
        }
      }
      const float d0 = xh[0] + xl[0], d1 = xh[1] + xl[1], d2 = xh[2] + xl[2], d3 = xh[3] + xl[3];
      const uint32_t p01 = pack_bf16(d0, d1), p23 = pack_bf16(d2, d3);
      *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = p01;
      *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = p23;
      *reinterpret_cast<uint32_t*>(sm.nw.Dlo + vr * LDD + tk) = pack_bf16(bf16_residual(d0), bf16_residual(d1));
      *reinterpret_cast<uint32_t*>(sm.nw.Dlo + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(d2), bf16_residual(d3));
      if (r >= 1 && r + 1 < nblk) named_bar_arrive(5 + r, 2 * 32 * 2 * NMT);   // release stage r + 1
      {   // state-update operand Delta e^{gamma_C - gamma} (hi / lo)
        const float2 c2 = *reinterpret_cast<const float2*>(&sm.ec[buf][tk]);
        const float s0v = d0 * c2.x, s1v = d1 * c2.y, s2v = d2 * c2.x, s3v = d3 * c2.y;
        *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = pack_bf16(s0v, s1v);
        *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = pack_bf16(s2v, s3v);
        *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = pack_bf16(bf16_residual(s0v), bf16_residual(s1v));
        *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(s2v), bf16_residual(s3v));
      }
      // o^T tile of this (block, half) = O32 + sum_{s <= r} Delta_s^T P_rs^T (P is block lower-triangular); the
      // own block's Delta^T (both halves) is read back after a barrier among the stage's warps
      named_bar_sync(8 + r, 32 * 2 * NMT);
      float o[4];
      {
        const float2 lo = *reinterpret_cast<const float2*>(sm.nw.O32 + vr * LDD + tk);
        const float2 hi = *reinterpret_cast<const float2*>(sm.nw.O32 + (vr + 8) * LDD + tk);
        o[0] = lo.x; o[1] = lo.y; o[2] = hi.x; o[3] = hi.y;
      }
#pragma unroll
      for (int s_ = 0; s_ < NB; ++s_) {
        if (s_ <= r) {
          uint32_t bp[2];
          ldB_nm1(bp, sm.P16[pb], r * BS + shalf * 8, s_ * BS, LDS, lane);   // B[k = j][n = i] = P[i][j]
          if (s_ < r) mma16816(o, ash[s_], bp);
          else { uint32_t ar[4]; ldA_rm(ar, sm.Dl, mt * 16, r * BS, LDD, lane); mma16816(o, ar, bp); }
        }
      }
      sm.O[tk * LDV + vr] = __float2bfloat16(o[0] * scale);
      sm.O[(tk + 1) * LDV + vr] = __float2bfloat16(o[1] * scale);
      sm.O[tk * LDV + vr + 8] = __float2bfloat16(o[2] * scale);
      sm.O[(tk + 1) * LDV + vr + 8] = __float2bfloat16(o[3] * scale);
    };
    if constexpr (!OLDS) {
#ifndef GDN_EXP_NO_3B
    if (sslot != 0) {
      if (!pre_done)
        for (int pi = warp - 2 * NMT; pi < NG; pi += 8 - 2 * NMT) g_pair(pb, pi, nblk);
    } else {
      chain_stage(0);
    }
#endif
    }
    __syncthreads();
    GDN_TS(6);

    if constexpr (OLDS) {
    // ---- phase 4: o_inter^T = S^T Q^T (then scaled by e^gamma_i); Delta^T = U^T - S^T W^T -------
    // warp: m-tile mt (v rows), token n-tiles wq*NTT .. wq*NTT+NTT-1
    const int tn0 = wq * (8 * NTT);
    float oT[NTT][4], dT[NTT][4];
    const bool tok_valid = (tn0 < Lc);   // warp-uniform: this warp's token tiles hold valid tokens
    if (!tok_valid) {
      // nothing to compute; the Delta columns of these tokens must still be finite (they are multiplied
      // by zero P entries / zero-filled K rows in later phases)
      const int vr = mt * 16 + (lane >> 2);
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const int tk = tn0 + nt * 8 + (lane & 3) * 2;
        *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = 0u;
        *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = 0u;
        *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = 0u;
        *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = 0u;
        *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = 0u;
        *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = 0u;
      }
    } else {
      const int vr = mt * 16 + (lane >> 2);
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const int tk = tn0 + nt * 8 + (lane & 3) * 2;
        oT[nt][0] = oT[nt][1] = oT[nt][2] = oT[nt][3] = 0.f;
        dT[nt][0] = sm.old.U32[tk * LDU + vr];
        dT[nt][1] = sm.old.U32[(tk + 1) * LDU + vr];
        dT[nt][2] = sm.old.U32[tk * LDU + vr + 8];
        dT[nt][3] = sm.old.U32[(tk + 1) * LDU + vr + 8];
      }
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t bq[4], bw[4];
        if constexpr (NTT > 1) {
          ldB_nm(bq, sm.Q[buf], tn0, ks * 16, LDK, lane);
          ldB_nm(bw, sm.old.W, tn0, ks * 16, LDK, lane);
        } else {   // one token n-tile per warp: load only it (halves the smem traffic)
          ldB_nm1(bq, sm.Q[buf], tn0, ks * 16, LDK, lane);
          ldB_nm1(bw, sm.old.W, tn0, ks * 16, LDK, lane);
        }
        mma16816(oT[0], Sb[ks], bq);
        mma16816(dT[0], Sb[ks], bw);
        if constexpr (NTT > 1) {
          mma16816(oT[1], Sb[ks], bq + 2);
          mma16816(dT[1], Sb[ks], bw + 2);
        }
      }
      // scale o_inter columns by e^{gamma_i}; write Delta and Delta*e^{gamma_C-gamma} as [token][v]
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const int tk = tn0 + nt * 8 + (lane & 3) * 2;
        const float g0 = gam[tk], g1 = gam[tk + 1];
        const float e0 = __expf(g0), e1 = __expf(g1);
        oT[nt][0] *= e0; oT[nt][1] *= e1; oT[nt][2] *= e0; oT[nt][3] *= e1;
        const float c0 = __expf(gC - g0), c1 = __expf(gC - g1);
        // v-major tiles: the accumulator pair (tk, tk+1) of one v row is one 32-bit store
        *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = pack_bf16(dT[nt][0], dT[nt][1]);
        *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = pack_bf16(dT[nt][2], dT[nt][3]);
        {
          const float s0v = dT[nt][0] * c0, s1v = dT[nt][1] * c1, s2v = dT[nt][2] * c0, s3v = dT[nt][3] * c1;
          *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = pack_bf16(s0v, s1v);
          *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = pack_bf16(s2v, s3v);
          *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = pack_bf16(bf16_residual(s0v), bf16_residual(s1v));
          *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(s2v), bf16_residual(s3v));
        }
      }
    }
    __syncthreads();
    GDN_TS(7);

    // ---- phase 5: o^T += Delta^T P^T ; scale ; stage [token][v] ; coalesced store ---------------
    if (tok_valid) {
      {
        uint32_t aa[2][4], bbf[2][4];
        ldA_rm(aa[0], sm.Dl, mt * 16, 0, LDD, lane);          // A[m=v][k=j] from Dl[v][j]
        if constexpr (NTT > 1) ldB_nm(bbf[0], sm.P16[pb], tn0, 0, LDS, lane);   // B[k=j][n=i] = P[i][j]
        else ldB_nm1(bbf[0], sm.P16[pb], tn0, 0, LDS, lane);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          if (ks * 16 < Lc) {
            if (ks + 1 < 4 && (ks + 1) * 16 < Lc) {
              ldA_rm(aa[(ks + 1) & 1], sm.Dl, mt * 16, (ks + 1) * 16, LDD, lane);
              if constexpr (NTT > 1) ldB_nm(bbf[(ks + 1) & 1], sm.P16[pb], tn0, (ks + 1) * 16, LDS, lane);
              else ldB_nm1(bbf[(ks + 1) & 1], sm.P16[pb], tn0, (ks + 1) * 16, LDS, lane);
            }
            mma16816(oT[0], aa[ks & 1], bbf[ks & 1]);
            if constexpr (NTT > 1) mma16816(oT[1], aa[ks & 1], bbf[ks & 1] + 2);
          }
        }
      }
      const int vr = mt * 16 + (lane >> 2);
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const int tk = tn0 + nt * 8 + (lane & 3) * 2;
        sm.O[tk * LDV + vr] = __float2bfloat16(oT[nt][0] * scale);
        sm.O[(tk + 1) * LDV + vr] = __float2bfloat16(oT[nt][1] * scale);
        sm.O[tk * LDV + vr + 8] = __float2bfloat16(oT[nt][2] * scale);
        sm.O[(tk + 1) * LDV + vr + 8] = __float2bfloat16(oT[nt][3] * scale);
      }
    }
    __syncthreads();
    } else {
    // ---- phase 4: chain stages 1 .. nblk-1 (stage r on slot r % NS, named barriers 6, 7 between stages) --
#pragma unroll
    for (int r = 1; r < NB; ++r) {
      if (r >= nblk) break;
#ifndef GDN_EXP_NO_STAGES
      if (sslot == (r % NS)) chain_stage(r);
#endif
    }
    __syncthreads();
    GDN_TS(7);
    }
    {
      constexpr int CPR = VW / 8;
      const int t = tid / CPR;
      const int cc = (tid % CPR) * 8;
      if (t < Lc) {
        const uint4 val = *reinterpret_cast<const uint4*>(sm.O + t * LDV + cc);
        *reinterpret_cast<uint4*>(out + ((tbase + t) * HV + h) * D + v0 + cc) = val;
      }
    }
    GDN_TS(8);

    // ---- phase 6: S^T = e^{gamma_C} S^T + Ds^T K  (warp: m-tile mt, k n-tiles 4*wq..+3) ------------
#pragma unroll
    for (int j = 0; j < NKT; ++j) { S[j][0] *= eC; S[j][1] *= eC; S[j][2] *= eC; S[j][3] *= eC; }
#ifndef GDN_EXP_NO_P6
    {
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (ks * 16 < Lc) {   // token blocks beyond the sequence end contribute nothing (warp-uniform)
          uint32_t aa[4], al[4], bb[NKT / 2][4];
          ldA_rm(aa, sm.Ds, mt * 16, ks * 16, LDD, lane);             // A[m=v][k=token] from Ds[v][token]
          ldA_rm(al, sm.Dslo, mt * 16, ks * 16, LDD, lane);
#pragma unroll
          for (int kp = 0; kp < NKT / 2; ++kp) {
            ldB_km(bb[kp], sm.K[buf], (wq * NKT + 2 * kp) * 8, ks * 16, LDK, lane);   // B[k=token][n=dim]
            mma16816(S[2 * kp], aa, bb[kp]);
            mma16816(S[2 * kp + 1], aa, bb[kp] + 2);
            mma16816(S[2 * kp], al, bb[kp]);
            mma16816(S[2 * kp + 1], al, bb[kp] + 2);
          }
        }
      }
    }
#endif
    if (has_next) exchange_S();  // includes a __syncthreads that also orders this chunk's smem reads before the next
    GDN_TS(9);
  }
  }  // rep

  // ---- store final state slice S^T -> new_state[n][h][v0+v][k] -----------------------------------
  { const int rep = GDN_NREP - 1; GDN_TS_ABS(34); }
#ifdef GDN_EXP_NO_STATE_STORE
  if (S[0][0] == 12345.f) st_out[tid] = S[1][1];   // timing experiment only (wrong results): keep S live
#else
  {
    const int r = mt * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (wq * NKT + j) * 8 + (lane & 3) * 2;
      *reinterpret_cast<float2*>(st_out + r * D + c) = make_float2(S[j][0], S[j][1]);
      *reinterpret_cast<float2*>(st_out + (r + 8) * D + c) = make_float2(S[j][2], S[j][3]);
    }
  }
#endif
}


// ---------------------------------------------------------------------------------------------------
// Warp-specialised multi-chunk kernel, all products on mma.sync (best for 2-3 chunk sequences, N <= 2)
template <int VS>
struct __align__(16) SmemWS6 {
  static constexpr int VW = D / VS, LDV = VW + 8, LDU = VW + 4, NMT = VW / 16, LDD = C + 8;
  __nv_bfloat16 K[3][C * LDK];
  __nv_bfloat16 Q[3][C * LDK];
  __nv_bfloat16 V[3][C * LDV];
  uint4 AB[2][C][2];
  __nv_bfloat16 W[2][C * LDK];      // -W (bf16); also the k-split scratch of the chunk being built
  float U32[2][C * LDU];
  __nv_bfloat16 P16[2][C * LDS];
  __nv_bfloat16 XU[C * LDV];
  // A (bf16, 64x64) lives in the upper 9 KB of the W buffer under construction (W[pb] + 4096 elements);
  // the k-split scratch uses its lower 8 KB. Both are dead before W is written by the solve.
  __nv_bfloat16 T[NB][BS * LDT];
  __nv_bfloat16 TK[NB][BS * LDT];
  __nv_bfloat16 TV[NB][BS * LDT];
  __nv_bfloat16 Tlo[NB][BS * LDT];
  __nv_bfloat16 TKlo[NB][BS * LDT];
  __nv_bfloat16 TVlo[NB][BS * LDT];
  __nv_bfloat16 AT16[NB][BS * LDT];
  union {
    float ATf[BS * BS];               // fp32 transposed diagonal block (one-block chunks): read by the inverse
    __nv_bfloat16 G16[NG][BS * LDT];  // G_rs, written after the inverse
  };
  __nv_bfloat16 Dl[VW * LDD];
  __nv_bfloat16 Ds[VW * LDD];
  __nv_bfloat16 Dslo[VW * LDD];
  __nv_bfloat16 O[C * LDV];
  uint32_t Sb[NMT][8][32][4];
  float gamma[2][C];
  float beta[2][C];
  float lg[2][C];
  float gC[2];
  static constexpr bool EXPT = false;
  uint64_t tiles_full[3];
};

template <int VS>
__global__ void __launch_bounds__(NT, 1)
gdn_prefill_ws6_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                      const __nv_bfloat16* __restrict__ v, const float* __restrict__ state,
                      const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ a,
                      const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ b,
                      const int64_t* __restrict__ cu_seqlens, float scale, int T_total,
                      __nv_bfloat16* __restrict__ out, float* __restrict__ new_state) {
  using Smem = SmemWS6<VS>;
  constexpr int VW = Smem::VW, LDV = Smem::LDV, LDU = Smem::LDU, LDD = Smem::LDD;
  static_assert(Smem::NMT == 1, "warp-specialised kernel: one state m-tile (VS = 8)");
  constexpr int NKT = 4;        // state k n-tiles per ST warp (4 warps x 4 = 16)
  constexpr int NTT = 2;        // token n-tiles per ST warp (4 warps x 2 = 8)
  constexpr int NUT = VW / 8;   // U n-tiles (WY warps 0..NUT-1)
  constexpr int CPR = VW / 8;   // 16 B chunks per output row
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

  const int h = blockIdx.x / VS;
  const int vs = blockIdx.x % VS;
  const int v0 = vs * VW;
  const int n = blockIdx.y;
  const int hq = h / (HV / HQ);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_st = warp >= 4;
  const int stw = warp - 4;       // ST warp: k n-tiles 4*stw.., token tiles 2*stw, 2*stw+1
  const int wyw = warp;           // WY warp: inverse block, 32 W columns, G pairs
  const int t128 = tid & 127;
#ifdef GDN_TIMING
  unsigned long long* const dbgw = (blockIdx.x == 0 && blockIdx.y == 0 && (tid == 0 || tid == 128)) ? g_dbg : nullptr;
#define WTS(slot) do { if (dbgw && p < 2) dbgw[p * 16 + (slot) + (is_st ? 8 : 0)] = clock64(); } while (0)
#else
#define WTS(slot) do {} while (0)
#endif

  // ST warps own the state slice: warp stw holds k n-tiles 4*stw .. 4*stw+3 of the 16 v rows.
  float S[NKT][4];
  if (is_st) {
    const float* st_in = state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
    const int r = lane >> 2;
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (stw * NKT + j) * 8 + (lane & 3) * 2;
      const float2 lo = ldg_f2_early(st_in + r * D + c);
      const float2 hi = ldg_f2_early(st_in + (r + 8) * D + c);
      S[j][0] = lo.x; S[j][1] = lo.y; S[j][2] = hi.x; S[j][3] = hi.y;
    }
  }
  const float neg_expA = -__expf(A_log[h]);
  const float dtb = dt_bias[h];
  const int64_t s0 = ldg_s64_early(cu_seqlens + n);
  const int64_t s1 = ldg_s64_early(cu_seqlens + n + 1);
  const int L = static_cast<int>(s1 - s0);
  float* st_out = new_state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
  if (tid == 0) {
    mbar_init(&sm.tiles_full[0], 128);
    mbar_init(&sm.tiles_full[1], 128);
    mbar_init(&sm.tiles_full[2], 128);
    fence_barrier_init();
  }
  __syncthreads();
  if (L <= 0) {
    for (int i = tid * 4; i < VW * D; i += NT * 4)
      *reinterpret_cast<float4*>(st_out + i) = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  const int nchunks = (L + C - 1) / C;

  // ST: a/b rows and K/Q/V tiles of chunk c (buffers c & 1 / c % 3), completion on tiles_full[c % 3]
  auto issue_chunk = [&](int c) {
    const int Lc = min(C, L - c * C);
    prefetch_ab(sm, c & 1, a, b, s0 + c * C, Lc, t128);
    prefetch_kqv<VS, 128>(sm, c % 3, q, k, v, s0 + c * C, Lc, hq, h, v0, t128);
    cp_async_mbar_arrive(&sm.tiles_full[c % 3]);
  };
  // ST: gates of chunk c from its (landed) a/b rows -> gamma[c & 1], beta[c & 1], gC[c & 1]
  auto gates_chunk = [&](int c) {
    if (lane < 16) gates_token(sm, c & 1, c & 1, min(C, L - c * C), h, neg_expA, dtb, stw * 16 + lane);
    named_bar_sync(5, 128);
    if (warp == 7) gates_scan(sm, c & 1, lane);
  };
  uint32_t Sb[8][4];   // ST: bf16 A fragments of S^T (M = v, K = k), all 8 k-steps
  auto write_Sb = [&]() {
#pragma unroll
    for (int kp = 0; kp < NKT / 2; ++kp) {
      const uint4 pk = make_uint4(pack_bf16(S[2 * kp][0], S[2 * kp][1]), pack_bf16(S[2 * kp][2], S[2 * kp][3]),
                                  pack_bf16(S[2 * kp + 1][0], S[2 * kp + 1][1]), pack_bf16(S[2 * kp + 1][2], S[2 * kp + 1][3]));
      *reinterpret_cast<uint4*>(sm.Sb[0][stw * (NKT / 2) + kp][lane]) = pk;
    }
  };
  auto read_Sb = [&]() {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint4 x = *reinterpret_cast<const uint4*>(sm.Sb[0][ks][lane]);
      Sb[ks][0] = x.x; Sb[ks][1] = x.y; Sb[ks][2] = x.z; Sb[ks][3] = x.w;
    }
  };

  // ---- prologue (ST): chunks 0 and 1 in flight, gates of chunk 0, initial Sb ----------------------
  if (is_st) {
    issue_chunk(0);
    if (nchunks > 1) issue_chunk(1);
    if (nchunks > 1) cp_async_wait<3>(); else cp_async_wait<1>();   // chunk 0's a/b rows (oldest group)
    named_bar_sync(5, 128);
    gates_chunk(0);
    write_Sb();
    named_bar_sync(5, 128);
    read_Sb();
  }
  __syncthreads();

  for (int p = 0; p <= nchunks; ++p) {
    if (!is_st) {
      // =================== WY: chunk p ===================
      if (p < nchunks) {
        const int c = p, tb = c % 3, pb = c & 1;
        const int c0 = c * C;
        const int Lc = min(C, L - c0);
        const int nblk = (Lc + BS - 1) / BS;
        const float* gam = sm.gamma[pb];
        const float* bet = sm.beta[pb];
        __nv_bfloat16* P16d = sm.P16[pb];
        __nv_bfloat16* Wd = sm.W[pb];
        __nv_bfloat16* A16d = Wd + 4096;   // A aliased onto the upper part of this chunk's W buffer
        float* U32d = sm.U32[pb];
        WTS(0);
        mbar_wait(&sm.tiles_full[tb], (c / 3) & 1);
        WTS(1);
        // zero the upper P blocks (bj > bi): 6 blocks x 16 rows x 16 bf16 = 192 x 16 B
        for (int idx = t128; idx < 192; idx += 128) {
          const int ub = idx >> 5, rr = idx & 31;
          const int ubi = (ub < 3) ? 0 : (ub < 5 ? 1 : 2);
          const int ubj = (ub < 3) ? ub + 1 : (ub < 5 ? ub - 1 : 3);
          *reinterpret_cast<uint4*>(P16d + (ubi * 16 + (rr >> 1)) * LDS + ubj * 16 + (rr & 1) * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (nblk == 1) {
          // <= 16 valid tokens: k-split of the single block over the 4 WY warps (2 k-steps each)
          float* red = reinterpret_cast<float*>(Wd);   // [4 warps][2 kinds][16][16] fp32 scratch (W written later)
          float KK[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          float QK[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
          for (int kk2 = 0; kk2 < 2; ++kk2) {
            const int ks = 2 * wyw + kk2;
            uint32_t aK[4], aQ[4], bb[4];
            ldA_rm(aK, sm.K[tb], 0, ks * 16, LDK, lane);
            ldA_rm(aQ, sm.Q[tb], 0, ks * 16, LDK, lane);
            ldB_nm(bb, sm.K[tb], 0, ks * 16, LDK, lane);
            mma16816(KK[0], aK, bb);
            mma16816(KK[1], aK, bb + 2);
            mma16816(QK[0], aQ, bb);
            mma16816(QK[1], aQ, bb + 2);
          }
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const int j = nt * 8 + (lane & 3) * 2;
            const int i = lane >> 2;
            *reinterpret_cast<float2*>(red + (wyw * 2 + 0) * 256 + i * 16 + j) = make_float2(KK[nt][0], KK[nt][1]);
            *reinterpret_cast<float2*>(red + (wyw * 2 + 0) * 256 + (i + 8) * 16 + j) = make_float2(KK[nt][2], KK[nt][3]);
            *reinterpret_cast<float2*>(red + (wyw * 2 + 1) * 256 + i * 16 + j) = make_float2(QK[nt][0], QK[nt][1]);
            *reinterpret_cast<float2*>(red + (wyw * 2 + 1) * 256 + (i + 8) * 16 + j) = make_float2(QK[nt][2], QK[nt][3]);
          }
          named_bar_sync(4, 128);
          {
            const int i = t128 >> 3, j = (t128 & 7) * 2;
            float2 kk = make_float2(0.f, 0.f), qk = make_float2(0.f, 0.f);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
              const float2 x = *reinterpret_cast<const float2*>(red + (w * 2 + 0) * 256 + i * 16 + j);
              const float2 y = *reinterpret_cast<const float2*>(red + (w * 2 + 1) * 256 + i * 16 + j);
              kk.x += x.x; kk.y += x.y; qk.x += y.x; qk.y += y.y;
            }
            const float gi = gam[i], bi_ = bet[i];
            const float d0 = __expf(gi - gam[j]), d1 = __expf(gi - gam[j + 1]);
            const float a0 = (j < i) ? bi_ * kk.x * d0 : 0.f;
            const float a1 = (j + 1 < i) ? bi_ * kk.y * d1 : 0.f;
            const float p0 = (j <= i) ? qk.x * d0 : 0.f;
            const float p1 = (j + 1 <= i) ? qk.y * d1 : 0.f;
            *reinterpret_cast<uint32_t*>(A16d + i * LDS + j) = pack_bf16(a0, a1);
            *reinterpret_cast<uint32_t*>(P16d + i * LDS + j) = pack_bf16(p0, p1);
            sm.ATf[j * BS + i] = a0;
            sm.ATf[(j + 1) * BS + i] = a1;
          }
        } else {
          // 20 lower-triangular units over the 4 WY warps (compile-time per warp, 2-3 unit sets each):
          // w0: bi0 {0,8} + bi3 {48,56} | w1: bi1 {0,8,16} + bi1 {24} + bi3 {40} | w2: bi2 {0,8,16} + bi2 {24,32,40}
          // w3: bi3 {0,8,16} + bi3 {24,32}
          switch (wyw) {
            case 0: kk_units3<0, 0, 8, -1, 3, 48, 56, -1, 0, -1, -1, -1>(sm, sm.K[tb], sm.Q[tb], A16d, P16d, Lc, gam, bet, lane); break;
            case 1: kk_units3<1, 0, 8, 16, 1, 24, -1, -1, 3, 40, -1, -1>(sm, sm.K[tb], sm.Q[tb], A16d, P16d, Lc, gam, bet, lane); break;
            case 2: kk_units3<2, 0, 8, 16, 2, 24, 32, 40, 0, -1, -1, -1>(sm, sm.K[tb], sm.Q[tb], A16d, P16d, Lc, gam, bet, lane); break;
            default: kk_units3<3, 0, 8, 16, 3, 24, 32, -1, 0, -1, -1, -1>(sm, sm.K[tb], sm.Q[tb], A16d, P16d, Lc, gam, bet, lane); break;
          }
        }
        // zero-fill solve outputs for padded block rows (W rows after g_pairs: they overlap A)
        if (nblk < NB) {
          const int rows = (NB - nblk) * BS;
          for (int idx = t128; idx < rows * (VW / 8); idx += 128) {
            const int t = nblk * BS + idx / (VW / 8);
            const int cc = (idx % (VW / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.XU + t * LDV + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = t128; idx < rows * (VW / 4); idx += 128) {
            const int t = nblk * BS + idx / (VW / 4);
            const int cc = (idx % (VW / 4)) * 4;
            *reinterpret_cast<float4*>(U32d + t * LDU + cc) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        named_bar_sync(4, 128);
        WTS(2);
        if (wyw < nblk) inverse_block(sm, wyw, nblk, gam, bet, lane);
        named_bar_sync(4, 128);
        WTS(3);
        g_pairs(sm, A16d, wyw, nblk, lane);
        named_bar_sync(4, 128);   // A (aliased onto W) is dead from here on
        if (nblk < NB) {
          const int rows = (NB - nblk) * BS;
          for (int idx = t128; idx < rows * (D / 8); idx += 128) {
            const int t = nblk * BS + idx / (D / 8);
            const int cc = (idx % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(Wd + t * LDK + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
        WTS(4);
        uint32_t gf[NG][4];
        load_gf(sm, nblk, lane, gf);
        if (wyw < NUT) solve_wy<true, false>(sm, sm.K[tb], sm.V[tb], Wd, U32d, wyw, nblk, lane, gf);
        else solve_wy<false, false>(sm, sm.K[tb], sm.V[tb], Wd, U32d, wyw, nblk, lane, gf);
        WTS(5);
      }
    } else {
      // =================== ST: chunk p - 1 ===================
      WTS(0);
      if (p >= 1 && p + 1 < nchunks) issue_chunk(p + 1);   // into the buffers freed by chunk p - 2
      if (p >= 1) {
        const int c = p - 1, tb = c % 3, pb = c & 1;
        const int c0 = c * C;
        const int Lc = min(C, L - c0);
        const int64_t tbase = s0 + c0;
        const float* gam = sm.gamma[pb];
        const float gC = sm.gC[pb];
        const float eC = __expf(gC);
        const int tn0 = stw * 16;
        const bool tok_valid = (tn0 < Lc);
        // ---- o_inter^T = S^T Q^T (scaled by e^gamma_i); Delta^T = U^T - S^T W^T ------------------
        float oT[NTT][4], dT[NTT][4];
        const int vr = lane >> 2;
        if (!tok_valid) {
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = 0u;
          }
        } else {
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            oT[nt][0] = oT[nt][1] = oT[nt][2] = oT[nt][3] = 0.f;
            dT[nt][0] = sm.U32[pb][tk * LDU + vr];
            dT[nt][1] = sm.U32[pb][(tk + 1) * LDU + vr];
            dT[nt][2] = sm.U32[pb][tk * LDU + vr + 8];
            dT[nt][3] = sm.U32[pb][(tk + 1) * LDU + vr + 8];
          }
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t bq[4], bw[4];
            ldB_nm(bq, sm.Q[tb], tn0, ks * 16, LDK, lane);
            ldB_nm(bw, sm.W[pb], tn0, ks * 16, LDK, lane);
            mma16816(oT[0], Sb[ks], bq);
            mma16816(dT[0], Sb[ks], bw);
            mma16816(oT[1], Sb[ks], bq + 2);
            mma16816(dT[1], Sb[ks], bw + 2);
          }
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            const float g0 = gam[tk], g1 = gam[tk + 1];
            const float e0 = __expf(g0), e1 = __expf(g1);
            oT[nt][0] *= e0; oT[nt][1] *= e1; oT[nt][2] *= e0; oT[nt][3] *= e1;
            const float c0v = __expf(gC - g0), c1v = __expf(gC - g1);
            *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = pack_bf16(dT[nt][0], dT[nt][1]);
            *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = pack_bf16(dT[nt][2], dT[nt][3]);
            const float s0v = dT[nt][0] * c0v, s1v = dT[nt][1] * c1v, s2v = dT[nt][2] * c0v, s3v = dT[nt][3] * c1v;
            *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = pack_bf16(s0v, s1v);
            *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = pack_bf16(s2v, s3v);
            *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = pack_bf16(bf16_residual(s0v), bf16_residual(s1v));
            *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(s2v), bf16_residual(s3v));
          }
        }
        named_bar_sync(5, 128);
        WTS(1);
        // ---- o^T += Delta^T P^T ; scale ; stage [token][v] -------------------------------------------
        if (tok_valid) {
          uint32_t aa[2][4], bbf[2][4];
          ldA_rm(aa[0], sm.Dl, 0, 0, LDD, lane);
          ldB_nm(bbf[0], sm.P16[pb], tn0, 0, LDS, lane);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            if (ks * 16 < Lc) {
              if (ks + 1 < 4 && (ks + 1) * 16 < Lc) {
                ldA_rm(aa[(ks + 1) & 1], sm.Dl, 0, (ks + 1) * 16, LDD, lane);
                ldB_nm(bbf[(ks + 1) & 1], sm.P16[pb], tn0, (ks + 1) * 16, LDS, lane);
              }
              mma16816(oT[0], aa[ks & 1], bbf[ks & 1]);
              mma16816(oT[1], aa[ks & 1], bbf[ks & 1] + 2);
            }
          }
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            sm.O[tk * LDV + vr] = __float2bfloat16(oT[nt][0] * scale);
            sm.O[(tk + 1) * LDV + vr] = __float2bfloat16(oT[nt][1] * scale);
            sm.O[tk * LDV + vr + 8] = __float2bfloat16(oT[nt][2] * scale);
            sm.O[(tk + 1) * LDV + vr + 8] = __float2bfloat16(oT[nt][3] * scale);
          }
        }
        // ---- S^T = e^{gamma_C} S^T + Ds^T K (warp: k n-tiles 4*stw .. 4*stw+3) ----------------------
#pragma unroll
        for (int j = 0; j < NKT; ++j) { S[j][0] *= eC; S[j][1] *= eC; S[j][2] *= eC; S[j][3] *= eC; }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          if (ks * 16 < Lc) {
            uint32_t aa[4], al[4], bb[NKT / 2][4];
            ldA_rm(aa, sm.Ds, 0, ks * 16, LDD, lane);
            ldA_rm(al, sm.Dslo, 0, ks * 16, LDD, lane);
#pragma unroll
            for (int kp = 0; kp < NKT / 2; ++kp) {
              ldB_km(bb[kp], sm.K[tb], (stw * NKT + 2 * kp) * 8, ks * 16, LDK, lane);
              mma16816(S[2 * kp], aa, bb[kp]);
              mma16816(S[2 * kp + 1], aa, bb[kp] + 2);
              mma16816(S[2 * kp], al, bb[kp]);
              mma16816(S[2 * kp + 1], al, bb[kp] + 2);
            }
          }
        }
        write_Sb();
        named_bar_sync(5, 128);   // O staged by all warps; Sb written
        WTS(2);
        {   // coalesced output store: 64 rows x CPR chunks over the 128 ST threads
          const int t = t128 / CPR, cc = (t128 % CPR) * 8;
          if (t < Lc) {
            const uint4 val = *reinterpret_cast<const uint4*>(sm.O + t * LDV + cc);
            *reinterpret_cast<uint4*>(out + ((tbase + t) * HV + h) * D + v0 + cc) = val;
          }
        }
        read_Sb();
      }
      if (p + 1 < nchunks) {   // gates of chunk p + 1 (its a/b rows: second-newest cp.async group)
        cp_async_wait<1>();
        named_bar_sync(5, 128);
        gates_chunk(p + 1);
      }
      WTS(3);
    }
    __syncthreads();
  }

  // ---- store final state slice S^T -> new_state[n][h][v0+v][k] (ST warps) --------------------------
  if (is_st) {
    const int r = lane >> 2;
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (stw * NKT + j) * 8 + (lane & 3) * 2;
      *reinterpret_cast<float2*>(st_out + r * D + c) = make_float2(S[j][0], S[j][1]);
      *reinterpret_cast<float2*>(st_out + (r + 8) * D + c) = make_float2(S[j][2], S[j][3]);
    }
  }
#undef WTS
}


// ---------------------------------------------------------------------------------------------------
// Warp-specialised multi-chunk kernel (N <= 2, VS = 8): warps 0-3 ("WY") compute A, T, G, W, U, P of
// chunk p while warps 4-7 ("ST") run o_inter/Delta, o + P and the state update of chunk p-1. Tiles are
// triple-buffered (ST still reads K/Q of chunk p-1 while WY reads chunk p and chunk p+1 lands); W, U, P
// and the gate arrays are double-buffered by chunk parity. ST issues every prefetch (cp.async, completion
// tracked per tile buffer by an mbarrier) and computes the gates of chunk p+1 at the end of its period.

constexpr int TM_KKQK = 0;    // TMEM columns 0-63 / 64-127 (chunk parity): [K;Q] K^T (rows 0-63 = KK, 64-127 = QK)
constexpr int TM_COLS = 128;
template <int VS>
struct __align__(1024) SmemWS {
  static constexpr int VW = D / VS, LDV = VW + 8, LDU = VW + 4, NMT = VW / 16, LDD = C + 8;
  __nv_bfloat16 KQ[3][2][SWHALF];   // stacked [K;Q] tiles, SW128 K-major halves (dims 0-63 | 64-127), via TMA
  __nv_bfloat16 V[3][C * LDV];
  uint4 AB[2][C][2];
  __nv_bfloat16 W[2][C * LDK];      // -W (bf16); also the k-split scratch of the chunk being built
  float U32[2][C * LDU];
  __nv_bfloat16 P16[2][C * LDS];
  __nv_bfloat16 XU[C * LDV];
  // A (bf16, 64x64) lives in the upper 9 KB of the W buffer under construction (W[pb] + 4096 elements);
  // the k-split scratch uses its lower 8 KB. Both are dead before W is written by the solve.
  __nv_bfloat16 T[NB][BS * LDT];
  __nv_bfloat16 TK[NB][BS * LDT];
  __nv_bfloat16 TV[NB][BS * LDT];
  __nv_bfloat16 Tlo[NB][BS * LDT];
  __nv_bfloat16 TKlo[NB][BS * LDT];
  __nv_bfloat16 TVlo[NB][BS * LDT];
  __nv_bfloat16 AT16[NB][BS * LDT];
  union {
    float ATf[BS * BS];               // fp32 transposed diagonal block (one-block chunks): read by the inverse
    __nv_bfloat16 G16[NG][BS * LDT];  // G_rs, written after the inverse
  };
  __nv_bfloat16 Dl[VW * LDD];
  __nv_bfloat16 Ds[VW * LDD];
  __nv_bfloat16 Dslo[VW * LDD];
  __nv_bfloat16 O[C * LDV];
  uint32_t Sb[NMT][8][32][4];
  float gamma[2][C];
  float beta[2][C];
  float lg[2][C];
  float gC[2];
  static constexpr bool EXPT = false;
  uint64_t tiles_full[3];   // V slice + a/b rows (cp.async by the 128 ST threads)
  uint64_t bar_tma[3];      // K/Q tiles (TMA, expect_tx)
  uint64_t bar_kkqk;        // UMMA completion
  uint32_t tmem_base;
};

template <int VS, bool LPT = false>
__global__ void __launch_bounds__(NT, 1)
gdn_prefill_ws_kernel(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k,
                      const __nv_bfloat16* __restrict__ v, const float* __restrict__ state,
                      const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ a,
                      const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ b,
                      const int64_t* __restrict__ cu_seqlens, float scale, int T_total,
                      __nv_bfloat16* __restrict__ out, float* __restrict__ new_state) {
  using Smem = SmemWS<VS>;
  constexpr int VW = Smem::VW, LDV = Smem::LDV, LDU = Smem::LDU, LDD = Smem::LDD;
  static_assert(Smem::NMT == 1, "warp-specialised kernel: one state m-tile (VS = 8)");
  constexpr int NKT = 4;        // state k n-tiles per ST warp (4 warps x 4 = 16)
  constexpr int NTT = 2;        // token n-tiles per ST warp (4 warps x 2 = 8)
  constexpr int NUT = VW / 8;   // U n-tiles (WY warps 0..NUT-1)
  constexpr int CPR = VW / 8;   // 16 B chunks per output row
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

  const int h = blockIdx.x / VS;
  const int vs = blockIdx.x % VS;
  const int v0 = vs * VW;
  const int tid = threadIdx.x;
  const int n = LPT ? lpt_sequence(cu_seqlens, static_cast<int>(gridDim.y), tid) : static_cast<int>(blockIdx.y);
  const int hq = h / (HV / HQ);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_st = warp >= 4;
  const int stw = warp - 4;       // ST warp: k n-tiles 4*stw.., token tiles 2*stw, 2*stw+1
  const int wyw = warp;           // WY warp: inverse block, 32 W columns, G pairs
  const int t128 = tid & 127;
#ifdef GDN_TIMING
  unsigned long long* const dbgw = (blockIdx.x == 0 && blockIdx.y == 0 && (tid == 0 || tid == 128)) ? g_dbg : nullptr;
#define WTS(slot) do { if (dbgw && p < 2) dbgw[p * 16 + (slot) + (is_st ? 8 : 0)] = clock64(); } while (0)
#else
#define WTS(slot) do {} while (0)
#endif

  // ST warps own the state slice: warp stw holds k n-tiles 4*stw .. 4*stw+3 of the 16 v rows.
  float S[NKT][4];
  if (is_st) {
    const float* st_in = state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
    const int r = lane >> 2;
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (stw * NKT + j) * 8 + (lane & 3) * 2;
      const float2 lo = ldg_f2_early(st_in + r * D + c);
      const float2 hi = ldg_f2_early(st_in + (r + 8) * D + c);
      S[j][0] = lo.x; S[j][1] = lo.y; S[j][2] = hi.x; S[j][3] = hi.y;
    }
  }
  const float neg_expA = -__expf(A_log[h]);
  const float dtb = dt_bias[h];
  const bool lpt_on = LPT && lpt_active(static_cast<int>(gridDim.y));
  const int64_t s0 = lpt_on ? static_cast<int64_t>(lpt_cs[n]) : ldg_s64_early(cu_seqlens + n);
  const int64_t s1 = lpt_on ? static_cast<int64_t>(lpt_cs[n + 1]) : ldg_s64_early(cu_seqlens + n + 1);
  const int L = static_cast<int>(s1 - s0);
  float* st_out = new_state + ((static_cast<size_t>(n) * HV + h) * D + v0) * D;
  if (tid == 0) {
    mbar_init(&sm.tiles_full[0], 128);
    mbar_init(&sm.tiles_full[1], 128);
    mbar_init(&sm.tiles_full[2], 128);
    mbar_init(&sm.bar_tma[0], 1);
    mbar_init(&sm.bar_tma[1], 1);
    mbar_init(&sm.bar_tma[2], 1);
    mbar_init(&sm.bar_kkqk, 1);
    fence_barrier_init();
  }
  if (warp == 0) tmem_alloc(&sm.tmem_base, TM_COLS);
  __syncthreads();
  const uint32_t tmem = sm.tmem_base;
  if (L <= 0) {
    for (int i = tid * 4; i < VW * D; i += NT * 4)
      *reinterpret_cast<float4*>(st_out + i) = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
    if (warp == 0) tmem_dealloc(tmem, TM_COLS);
    return;
  }
  const int nchunks = (L + C - 1) / C;

  // ST: K/Q tiles of chunk c by TMA (thread 0 of the group; bar_tma[c % 3]), V slice and a/b rows by
  // cp.async (all 128 ST threads; tiles_full[c % 3])
  auto issue_chunk = [&](int c) {
    const int Lc = min(C, L - c * C);
    const int tb = c % 3;
    if (t128 == 0) {
      mbar_expect_tx(&sm.bar_tma[tb], 4u * 64 * 64 * 2);
      const int tok = static_cast<int>(s0 + c * C);
      tma_load_3d(sm.KQ[tb][0], &tmap_k, 0, hq, tok, &sm.bar_tma[tb]);
      tma_load_3d(sm.KQ[tb][1], &tmap_k, 64, hq, tok, &sm.bar_tma[tb]);
      tma_load_3d(sm.KQ[tb][0] + 64 * 64, &tmap_q, 0, hq, tok, &sm.bar_tma[tb]);
      tma_load_3d(sm.KQ[tb][1] + 64 * 64, &tmap_q, 64, hq, tok, &sm.bar_tma[tb]);
    }
    prefetch_ab(sm, c & 1, a, b, s0 + c * C, Lc, t128);
    {   // V slice: 64 rows x (VW/8) chunks of 16 B over the 128 ST threads
      const int rows = ((Lc + BS - 1) / BS) * BS;
      constexpr int CPRV = VW / 8;
#pragma unroll
      for (int it = 0; it < (C * CPRV + 127) / 128; ++it) {
        const int idx = it * 128 + t128;
        const int t = idx / CPRV;
        const int cc = (idx % CPRV) * 8;
        if (t < rows) {
          const bool ok = t < Lc;
          const int64_t tg = s0 + c * C + (ok ? t : 0);
          cp_async_16(sm.V[tb] + t * LDV + cc, v + (tg * HV + h) * D + v0 + cc, ok);
        }
      }
      cp_async_commit();
    }
    cp_async_mbar_arrive(&sm.tiles_full[tb]);
  };
  // ST: gates of chunk c from its (landed) a/b rows -> gamma[c & 1], beta[c & 1], gC[c & 1]
  auto gates_chunk = [&](int c) {
    if (lane < 16) gates_token(sm, c & 1, c & 1, min(C, L - c * C), h, neg_expA, dtb, stw * 16 + lane);
    named_bar_sync(5, 128);
    if (warp == 7) gates_scan(sm, c & 1, lane);
  };
  // UMMA [K;Q] K^T of chunk cc -> TMEM (parity buffer) rows 0-63 = KK, 64-127 = QK (M=128, N=64, K=128),
  // issued by thread 128 (ST) once the chunk's tiles landed; completion on bar_kkqk (one phase per chunk)
  auto issue_kkqk = [&](int cc) {
    const int tbb = cc % 3;
    mbar_wait(&sm.bar_tma[tbb], (cc / 3) & 1);
    mbar_wait(&sm.tiles_full[tbb], (cc / 3) & 1);
    tc_fence_after();
    const uint32_t id128x64 = umma_idesc(128, 64, 0, 0);
    const uint64_t d0 = umma_desc(smem_u32(sm.KQ[tbb][0]), 16, 1024);
    const uint64_t d1 = umma_desc(smem_u32(sm.KQ[tbb][1]), 16, 1024);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const uint64_t d = ((kk < 4) ? d0 : d1) + static_cast<uint64_t>((kk & 3) * 2);   // +32 B per k-step
      umma(tmem + TM_KKQK + (cc & 1) * 64, d, d, id128x64, kk > 0);
    }
    umma_commit(&sm.bar_kkqk);
  };
  uint32_t Sb[8][4];   // ST: bf16 A fragments of S^T (M = v, K = k), all 8 k-steps
  auto write_Sb = [&]() {
#pragma unroll
    for (int kp = 0; kp < NKT / 2; ++kp) {
      const uint4 pk = make_uint4(pack_bf16(S[2 * kp][0], S[2 * kp][1]), pack_bf16(S[2 * kp][2], S[2 * kp][3]),
                                  pack_bf16(S[2 * kp + 1][0], S[2 * kp + 1][1]), pack_bf16(S[2 * kp + 1][2], S[2 * kp + 1][3]));
      *reinterpret_cast<uint4*>(sm.Sb[0][stw * (NKT / 2) + kp][lane]) = pk;
    }
  };
  auto read_Sb = [&]() {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint4 x = *reinterpret_cast<const uint4*>(sm.Sb[0][ks][lane]);
      Sb[ks][0] = x.x; Sb[ks][1] = x.y; Sb[ks][2] = x.z; Sb[ks][3] = x.w;
    }
  };

  // ---- prologue (ST): chunks 0 and 1 in flight, gates of chunk 0, initial Sb ----------------------
  if (is_st) {
    issue_chunk(0);
    if (nchunks > 1) issue_chunk(1);
    if (nchunks > 1) cp_async_wait<3>(); else cp_async_wait<1>();   // chunk 0's a/b rows (oldest group)
    named_bar_sync(5, 128);
    gates_chunk(0);
    write_Sb();
    named_bar_sync(5, 128);
    read_Sb();
    if (t128 == 0) issue_kkqk(0);
  }
  __syncthreads();

  for (int p = 0; p <= nchunks; ++p) {
    if (!is_st) {
      // =================== WY: chunk p ===================
      if (p < nchunks) {
        const int c = p, tb = c % 3, pb = c & 1;
        const int c0 = c * C;
        const int Lc = min(C, L - c0);
        const int nblk = (Lc + BS - 1) / BS;
        const float* gam = sm.gamma[pb];
        const float* bet = sm.beta[pb];
        __nv_bfloat16* P16d = sm.P16[pb];
        __nv_bfloat16* Wd = sm.W[pb];
        __nv_bfloat16* A16d = Wd + 4096;   // A aliased onto the upper part of this chunk's W buffer
        float* U32d = sm.U32[pb];
        WTS(0);
        // (the KK/QK UMMA of this chunk was issued by the ST group; its tiles landed before that)
        WTS(1);
        const __nv_bfloat16* KQt = sm.KQ[tb][0];   // stacked tile (half 0); half 1 at + SWHALF
        // zero the upper P blocks (bj > bi): 6 blocks x 16 rows x 16 bf16 = 192 x 16 B
        for (int idx = t128; idx < 192; idx += 128) {
          const int ub = idx >> 5, rr = idx & 31;
          const int ubi = (ub < 3) ? 0 : (ub < 5 ? 1 : 2);
          const int ubj = (ub < 3) ? ub + 1 : (ub < 5 ? ub - 1 : 3);
          *reinterpret_cast<uint4*>(P16d + (ubi * 16 + (rr >> 1)) * LDS + ubj * 16 + (rr & 1) * 8) = make_uint4(0u, 0u, 0u, 0u);
        }
        mbar_wait(&sm.bar_kkqk, c & 1);
        tc_fence_after();
        // ---- A / P epilogue straight from TMEM: thread -> row i = 32*(wyw&1) + lane of KK (warps 0,1) or
        //      QK (warps 2,3); 64 columns in two halves. A[i][j] = beta_i KK e^{g_i - g_j} (j < i),
        //      P[i][j] = QK e^{g_i - g_j} (j <= i); rows / columns beyond Lc are zero. The loop body is
        //      branch-free (straight-line after unrolling) so that the 32 independent exp/select/pack
        //      chains overlap; warps holding rows 0-31 skip the all-upper second half. --------------
        {
          const int i = (wyw & 1) * 32 + lane;
          const bool is_kk = (wyw < 2);
          const uint32_t trow = tmem + (static_cast<uint32_t>(wyw * 32) << 16) + TM_KKQK + (c & 1) * 64;
          const float gi = gam[i];
          // rows i >= Lc: beta_i = 0 zeroes A; P rows beyond Lc only feed outputs that are never stored.
          // columns j >= Lc with j < i imply i >= Lc. So only the triangular mask is needed.
          const float bi_ = is_kk ? bet[i] : 1.f;
          const int jlim = is_kk ? i : i + 1;     // columns j < jlim contribute
          __nv_bfloat16* const drow = (is_kk ? A16d : P16d) + i * LDS;
#pragma unroll
          for (int c0c = 0; c0c < 64; c0c += 32) {
            uint4* dst4 = reinterpret_cast<uint4*>(drow + c0c);
            if (c0c == 32 && (wyw & 1) == 0) {   // rows 0-31: columns 32-63 are strictly upper -> zeros
              dst4[0] = dst4[1] = dst4[2] = dst4[3] = make_uint4(0u, 0u, 0u, 0u);
            } else {
              float g[32];
              const float4* g4 = reinterpret_cast<const float4*>(gam + c0c);
#pragma unroll
              for (int q4 = 0; q4 < 8; ++q4) { const float4 t4 = g4[q4]; g[4 * q4] = t4.x; g[4 * q4 + 1] = t4.y; g[4 * q4 + 2] = t4.z; g[4 * q4 + 3] = t4.w; }
              uint32_t acc[32];
              tmem_ld32(trow + c0c, acc);
              tmem_wait_ld();
              uint32_t pk[16];
#pragma unroll
              for (int jj = 0; jj < 32; jj += 2) {
                const int j = c0c + jj;
                // gamma is non-increasing, so g_i - g_j <= 0 for j <= i; clamping at 0 keeps the masked
                // (upper) entries finite and leaves the kept ones unchanged -> no predication
                const float d0 = exp_neg_fast(fminf(gi - g[jj], 0.f));
                const float d1 = exp_neg_fast(fminf(gi - g[jj + 1], 0.f));
                const float m0 = (j < jlim) ? bi_ : 0.f;
                const float m1 = (j + 1 < jlim) ? bi_ : 0.f;
                const float x0 = __uint_as_float(acc[jj]) * d0 * m0;
                const float x1 = __uint_as_float(acc[jj + 1]) * d1 * m1;
                pk[jj >> 1] = pack_bf16(x0, x1);
              }
              const uint4* pk4 = reinterpret_cast<const uint4*>(pk);
#pragma unroll
              for (int q4 = 0; q4 < 4; ++q4) dst4[q4] = pk4[q4];
            }
          }
        }
        // transposed copies of the diagonal blocks for the inverse: AT16[r][jl][il] = A[16r+il][16r+jl]
        // by the KK warp that wrote those rows (warp 0: blocks 0,1; warp 1: blocks 2,3), 16 per lane
        if (wyw < 2) {
          __syncwarp();
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            const int flat = lane * 16 + e;           // 0..511 over the two blocks
            const int r = wyw * 2 + (flat >> 8), il = (flat >> 4) & 15, jl = flat & 15;
            const __nv_bfloat16 av = A16d[(16 * r + il) * LDS + 16 * r + jl];
            sm.AT16[r][jl * LDT + il] = av;
            if (nblk == 1 && r == 0) sm.ATf[jl * BS + il] = __bfloat162float(av);
          }
        }
        // zero-fill solve outputs for padded block rows (W rows after g_pairs: they overlap A)
        if (nblk < NB) {
          const int rows = (NB - nblk) * BS;
          for (int idx = t128; idx < rows * (VW / 8); idx += 128) {
            const int t = nblk * BS + idx / (VW / 8);
            const int cc = (idx % (VW / 8)) * 8;
            *reinterpret_cast<uint4*>(sm.XU + t * LDV + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
          for (int idx = t128; idx < rows * (VW / 4); idx += 128) {
            const int t = nblk * BS + idx / (VW / 4);
            const int cc = (idx % (VW / 4)) * 4;
            *reinterpret_cast<float4*>(U32d + t * LDU + cc) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        named_bar_sync(4, 128);
        WTS(2);
        if (wyw < nblk) inverse_block(sm, wyw, nblk, gam, bet, lane);
        named_bar_sync(4, 128);
        WTS(3);
        g_pairs(sm, A16d, wyw, nblk, lane);
        named_bar_sync(4, 128);   // A (aliased onto W) is dead from here on
        if (nblk < NB) {
          const int rows = (NB - nblk) * BS;
          for (int idx = t128; idx < rows * (D / 8); idx += 128) {
            const int t = nblk * BS + idx / (D / 8);
            const int cc = (idx % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(Wd + t * LDK + cc) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
        WTS(4);
        uint32_t gf[NG][4];
        load_gf(sm, nblk, lane, gf);
        if (wyw < NUT) solve_wy<true, true>(sm, sm.KQ[tb][0], sm.V[tb], Wd, U32d, wyw, nblk, lane, gf);
        else solve_wy<false, true>(sm, sm.KQ[tb][0], sm.V[tb], Wd, U32d, wyw, nblk, lane, gf);
        WTS(5);
      }
    } else {
      // =================== ST: chunk p - 1 ===================
      WTS(0);
      if (p >= 1 && p + 1 < nchunks) issue_chunk(p + 1);   // into the buffers freed by chunk p - 2
      if (p >= 1) {
        const int c = p - 1, tb = c % 3, pb = c & 1;
        const int c0 = c * C;
        const int Lc = min(C, L - c0);
        const int64_t tbase = s0 + c0;
        const float* gam = sm.gamma[pb];
        const float gC = sm.gC[pb];
        const float eC = __expf(gC);
        const int tn0 = stw * 16;
        const bool tok_valid = (tn0 < Lc);
        // ---- o_inter^T = S^T Q^T (scaled by e^gamma_i); Delta^T = U^T - S^T W^T ------------------
        float oT[NTT][4], dT[NTT][4];
        const int vr = lane >> 2;
        if (!tok_valid) {
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = 0u;
            *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = 0u;
          }
        } else {
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            oT[nt][0] = oT[nt][1] = oT[nt][2] = oT[nt][3] = 0.f;
            dT[nt][0] = sm.U32[pb][tk * LDU + vr];
            dT[nt][1] = sm.U32[pb][(tk + 1) * LDU + vr];
            dT[nt][2] = sm.U32[pb][tk * LDU + vr + 8];
            dT[nt][3] = sm.U32[pb][(tk + 1) * LDU + vr + 8];
          }
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            uint32_t bq[4], bw[4];
            ldB_tile_nm(bq, sm.KQ[tb][0] + 64 * 64, tn0, ks * 16, lane);   // Q rows 64-127 of the stacked tile
            ldB_nm(bw, sm.W[pb], tn0, ks * 16, LDK, lane);
            mma16816(oT[0], Sb[ks], bq);
            mma16816(dT[0], Sb[ks], bw);
            mma16816(oT[1], Sb[ks], bq + 2);
            mma16816(dT[1], Sb[ks], bw + 2);
          }
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            const float g0 = gam[tk], g1 = gam[tk + 1];
            const float e0 = __expf(g0), e1 = __expf(g1);
            oT[nt][0] *= e0; oT[nt][1] *= e1; oT[nt][2] *= e0; oT[nt][3] *= e1;
            const float c0v = __expf(gC - g0), c1v = __expf(gC - g1);
            *reinterpret_cast<uint32_t*>(sm.Dl + vr * LDD + tk) = pack_bf16(dT[nt][0], dT[nt][1]);
            *reinterpret_cast<uint32_t*>(sm.Dl + (vr + 8) * LDD + tk) = pack_bf16(dT[nt][2], dT[nt][3]);
            const float s0v = dT[nt][0] * c0v, s1v = dT[nt][1] * c1v, s2v = dT[nt][2] * c0v, s3v = dT[nt][3] * c1v;
            *reinterpret_cast<uint32_t*>(sm.Ds + vr * LDD + tk) = pack_bf16(s0v, s1v);
            *reinterpret_cast<uint32_t*>(sm.Ds + (vr + 8) * LDD + tk) = pack_bf16(s2v, s3v);
            *reinterpret_cast<uint32_t*>(sm.Dslo + vr * LDD + tk) = pack_bf16(bf16_residual(s0v), bf16_residual(s1v));
            *reinterpret_cast<uint32_t*>(sm.Dslo + (vr + 8) * LDD + tk) = pack_bf16(bf16_residual(s2v), bf16_residual(s3v));
          }
        }
        named_bar_sync(5, 128);
        WTS(1);
        // ---- o^T += Delta^T P^T ; scale ; stage [token][v] -------------------------------------------
        if (tok_valid) {
          uint32_t aa[2][4], bbf[2][4];
          ldA_rm(aa[0], sm.Dl, 0, 0, LDD, lane);
          ldB_nm(bbf[0], sm.P16[pb], tn0, 0, LDS, lane);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            if (ks * 16 < Lc) {
              if (ks + 1 < 4 && (ks + 1) * 16 < Lc) {
                ldA_rm(aa[(ks + 1) & 1], sm.Dl, 0, (ks + 1) * 16, LDD, lane);
                ldB_nm(bbf[(ks + 1) & 1], sm.P16[pb], tn0, (ks + 1) * 16, LDS, lane);
              }
              mma16816(oT[0], aa[ks & 1], bbf[ks & 1]);
              mma16816(oT[1], aa[ks & 1], bbf[ks & 1] + 2);
            }
          }
#pragma unroll
          for (int nt = 0; nt < NTT; ++nt) {
            const int tk = tn0 + nt * 8 + (lane & 3) * 2;
            sm.O[tk * LDV + vr] = __float2bfloat16(oT[nt][0] * scale);
            sm.O[(tk + 1) * LDV + vr] = __float2bfloat16(oT[nt][1] * scale);
            sm.O[tk * LDV + vr + 8] = __float2bfloat16(oT[nt][2] * scale);
            sm.O[(tk + 1) * LDV + vr + 8] = __float2bfloat16(oT[nt][3] * scale);
          }
        }
        // ---- S^T = e^{gamma_C} S^T + Ds^T K (warp: k n-tiles 4*stw .. 4*stw+3) ----------------------
#pragma unroll
        for (int j = 0; j < NKT; ++j) { S[j][0] *= eC; S[j][1] *= eC; S[j][2] *= eC; S[j][3] *= eC; }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          if (ks * 16 < Lc) {
            uint32_t aa[4], al[4], bb[NKT / 2][4];
            ldA_rm(aa, sm.Ds, 0, ks * 16, LDD, lane);
            ldA_rm(al, sm.Dslo, 0, ks * 16, LDD, lane);
#pragma unroll
            for (int kp = 0; kp < NKT / 2; ++kp) {
              ldB_tile_t(bb[kp], sm.KQ[tb][0], (stw * NKT + 2 * kp) * 8, ks * 16, lane);
              mma16816(S[2 * kp], aa, bb[kp]);
              mma16816(S[2 * kp + 1], aa, bb[kp] + 2);
              mma16816(S[2 * kp], al, bb[kp]);
              mma16816(S[2 * kp + 1], al, bb[kp] + 2);
            }
          }
        }
        write_Sb();
        named_bar_sync(5, 128);   // O staged by all warps; Sb written
        WTS(2);
        {   // coalesced output store: 64 rows x CPR chunks over the 128 ST threads
          const int t = t128 / CPR, cc = (t128 % CPR) * 8;
          if (t < Lc) {
            const uint4 val = *reinterpret_cast<const uint4*>(sm.O + t * LDV + cc);
            *reinterpret_cast<uint4*>(out + ((tbase + t) * HV + h) * D + v0 + cc) = val;
          }
        }
        read_Sb();
      }
      if (p + 1 < nchunks) {   // gates of chunk p + 1 (its a/b rows: second-newest cp.async group)
        cp_async_wait<1>();
        named_bar_sync(5, 128);
        gates_chunk(p + 1);
        if (t128 == 0) issue_kkqk(p + 1);   // its tiles landed (waited inside); WY(p+1) only waits for the result
      }
      WTS(3);
    }
    __syncthreads();
  }

  // ---- store final state slice S^T -> new_state[n][h][v0+v][k] (ST warps) --------------------------
  if (is_st) {
    const int r = lane >> 2;
#pragma unroll
    for (int j = 0; j < NKT; ++j) {
      const int c = (stw * NKT + j) * 8 + (lane & 3) * 2;
      *reinterpret_cast<float2*>(st_out + r * D + c) = make_float2(S[j][0], S[j][1]);
      *reinterpret_cast<float2*>(st_out + (r + 8) * D + c) = make_float2(S[j][2], S[j][3]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tmem, TM_COLS);
#undef WTS
}

}  // namespace gdn

void gdn_set_debug(torch::Tensor buf) {
#ifdef GDN_TIMING
  unsigned long long* p = reinterpret_cast<unsigned long long*>(buf.data_ptr());
  cudaMemcpyToSymbol(gdn::g_dbg, &p, sizeof(p));
#endif
}

template <int VS, int NBLK, bool SINGLE, bool LPT = false>
static void launch(int N, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                   const torch::Tensor& state, const torch::Tensor& A_log, const torch::Tensor& a,
                   const torch::Tensor& dt_bias, const torch::Tensor& b, const torch::Tensor& cu_seqlens,
                   float scale, torch::Tensor& output, torch::Tensor& new_state) {
  const size_t smem = sizeof(gdn::SmemT<VS, (!SINGLE && VS != 4 && NBLK > 0 && NBLK <= 2) ? 2 : 1>);
  static bool attr_set = false;
  if (!attr_set) {
    const cudaError_t e = cudaFuncSetAttribute(gdn::gdn_prefill_kernel<VS, NBLK, SINGLE, LPT>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(e));
    attr_set = true;
  }
#ifdef GDN_EXP_SEQ0_ONLY
  dim3 grid(gdn::HV * VS, 1);   // timing experiment only: sequence 0's CTAs alone (wrong results)
#else
  dim3 grid(gdn::HV * VS, N);
#endif
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn::gdn_prefill_kernel<VS, NBLK, SINGLE, LPT><<<grid, gdn::NT, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), state.data_ptr<float>(),
      A_log.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      dt_bias.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      cu_seqlens.data_ptr<int64_t>(), scale, static_cast<int>(q.size(0)),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), new_state.data_ptr<float>());
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&fn), cudaEnableDefault, &qres);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not available");
  }
  return fn;
}
// [T, 4, 128] bf16 -> 3D map, box {64 dims, 1 head, 64 tokens}, 128B swizzle, OOB zero fill
static CUtensorMap make_tmap_qk(const torch::Tensor& t) {
  TORCH_CHECK(t.is_contiguous(), "q/k must be contiguous");
  CUtensorMap m;
  cuuint64_t gdim[3] = {static_cast<cuuint64_t>(gdn::D), static_cast<cuuint64_t>(gdn::HQ), static_cast<cuuint64_t>(t.size(0))};
  cuuint64_t gstride[2] = {static_cast<cuuint64_t>(gdn::D * 2), static_cast<cuuint64_t>(gdn::HQ * gdn::D * 2)};
  cuuint32_t box[3] = {64, 1, 64};
  cuuint32_t estr[3] = {1, 1, 1};
  const CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), gdim, gstride, box, estr,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

template <int VS, bool LPT = false>
static void launch_ws(int N, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                      const torch::Tensor& state, const torch::Tensor& A_log, const torch::Tensor& a,
                      const torch::Tensor& dt_bias, const torch::Tensor& b, const torch::Tensor& cu_seqlens,
                      float scale, torch::Tensor& output, torch::Tensor& new_state) {
  const size_t smem = sizeof(gdn::SmemWS<VS>);
  static bool attr_set = false;
  if (!attr_set) {
    const cudaError_t e = cudaFuncSetAttribute(gdn::gdn_prefill_ws_kernel<VS, LPT>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute (ws) failed: ", cudaGetErrorString(e));
    attr_set = true;
  }
  dim3 grid(gdn::HV * VS, N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const CUtensorMap tq = make_tmap_qk(q);
  const CUtensorMap tk = make_tmap_qk(k);
  gdn::gdn_prefill_ws_kernel<VS, LPT><<<grid, gdn::NT, smem, stream>>>(
      tq, tk,
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), state.data_ptr<float>(),
      A_log.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      dt_bias.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      cu_seqlens.data_ptr<int64_t>(), scale, static_cast<int>(q.size(0)),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), new_state.data_ptr<float>());
}

template <int VS>
static void launch_ws6(int N, const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                      const torch::Tensor& state, const torch::Tensor& A_log, const torch::Tensor& a,
                      const torch::Tensor& dt_bias, const torch::Tensor& b, const torch::Tensor& cu_seqlens,
                      float scale, torch::Tensor& output, torch::Tensor& new_state) {
  const size_t smem = sizeof(gdn::SmemWS6<VS>);
  static bool attr_set = false;
  if (!attr_set) {
    const cudaError_t e = cudaFuncSetAttribute(gdn::gdn_prefill_ws6_kernel<VS>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute (ws6) failed: ", cudaGetErrorString(e));
    attr_set = true;
  }
  dim3 grid(gdn::HV * VS, N);
  auto stream = at::cuda::getCurrentCUDAStream();
  gdn::gdn_prefill_ws6_kernel<VS><<<grid, gdn::NT, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), state.data_ptr<float>(),
      A_log.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      dt_bias.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      cu_seqlens.data_ptr<int64_t>(), scale, static_cast<int>(q.size(0)),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), new_state.data_ptr<float>());
}

void gdn_prefill_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
                     torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
                     torch::Tensor cu_seqlens, double scale, torch::Tensor output,
                     torch::Tensor new_state) {
  const int N = static_cast<int>(cu_seqlens.size(0)) - 1;
  if (N <= 0) return;
  const int T = static_cast<int>(q.size(0));
  const float sc = static_cast<float>(scale);
  // Dispatch on host-known shape only (no device reads): N == 1 fixes the sequence length (= T), so
  // single-chunk sequences get a compile-time block count; N == 2 with T <= 64 is single-chunk with a
  // runtime block count; few sequences use 8 V slices per head (64 CTAs per sequence), else 4.
  // N <= 2 and T <= 64: single-chunk kernels with a compile-time block count; for N == 2 it is an upper
  // bound (each sequence has <= T tokens; warps beyond a sequence's end skip their units). The
  // runtime-block-count single-chunk kernel measured 0.4-0.85 us slower than these on the same lengths.
#ifdef GDN_NO_LPT
  constexpr bool LPT = false;
#else
  constexpr bool LPT = true;
#endif
  if (N == 1 && T <= 16)      launch<8, 1, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 32) launch<8, 2, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 48) launch<8, 3, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 64) launch<8, 4, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 2 && T <= 16) launch<8, -1, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 2 && T <= 32) launch<8, -2, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 2 && T <= 48) launch<8, -3, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 2 && T <= 64) launch<8, -4, true>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
#ifndef GDN_NO_TWO
  // N == 1, two chunks: chunk 0 full, chunk 1 with a compile-time block count (TWO mode of the lock-step kernel)
  else if (N == 1 && T <= 80)  launch<8, 1, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 96)  launch<8, 2, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 112) launch<8, 3, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
  else if (N == 1 && T <= 128) launch<8, 4, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
#endif
  else if (N <= 2) {
    // Few multi-chunk sequences. The warp-specialised kernels only pay off when a sequence spans >= 3
    // chunks (official full-100 measurements): N == 1 from 129 tokens (all-mma.sync variant below 160,
    // the UMMA KK^T/QK^T variant above), N == 2 from ~300 tokens (the shorter of the two sequences must
    // still be long); everything else keeps the lock-step kernel with 8 V slices per head.
#ifdef GDN_NO_WS
    launch<8, 0, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
#else
    if (N == 1 && T >= 160)      launch_ws<8>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
    else if (N == 1 && T >= 129) launch_ws6<8>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
    else if (N == 2 && T >= 300) launch_ws<8>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
    else                         launch<8, 0, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
#endif
  }
  // Phase 9: N >= 3. Every such batch of the workload set is dominated by one long sequence, so (D9-1) the
  // kernels process the sequences in longest-first order (LPT template parameter, device-side ranking of
  // cu_seqlens) and (D9-2) batches of 3-5 sequences use the VS = 8 kernels (lower per-chunk latency; the short
  // sequences' CTAs fill the SMs the dominant sequence does not use) from 600 tokens (ws7 beats the VS = 4
  // lock-step kernel only from ~7 chunks; the VS = 8 lock-step kernel's 192+ CTAs need a second wave: +4 us).
  // Batches of <= 5 sequences on VS = 4 fit one CTA wave, so they keep the identity order (the ranking prologue
  // costs 0.3-1.2 us there); batches of > 55 sequences (> 12 CTAs per SM) are throughput-bound and the prologue
  // (cu_seqlens staging + two barriers before the state / tile loads, ~0.7 us per CTA) only costs there.
#ifndef GDN_NO_N35
  else if (N <= 5 && T >= 600) launch_ws<8, LPT>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
#endif
  else if (N <= 5 || N > 55)  launch<4, 0, false>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);   // one CTA wave / > 12 CTAs per SM: no reordering
  else                        launch<4, 0, false, LPT>(N, q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, sc, output, new_state);
}
