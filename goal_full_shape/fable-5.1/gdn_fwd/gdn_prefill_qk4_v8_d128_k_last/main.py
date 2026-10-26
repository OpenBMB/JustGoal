"""GDN prefill (gdn_prefill_qk4_v8_d128_k_last) - written from scratch for B200 (tcgen05 + mma.sync)."""
import os, torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r'''
// GDN (gated delta net) prefill, written from scratch.  v3: tcgen05 (5th-gen tensor core) version.
// Single kernel, two CTA roles chosen by a dispatch ticket:
//   prep  CTA (chunk c, v-head h): chunk-local WY quantities
//         W = T*(beta*gam*K), U~ = T*(beta*V), P = scale*(Q K^T (.) decay), gam, gamc
//         with T = (I + A)^-1, A_ij = beta_i exp(G_i-G_j) k_i.k_j (j<i).  KK^T/QK^T on tcgen05, the
//         16x16 diagonal blocks of T by a warp recurrence, the rest by blocked forward substitution (mma.sync).
//   chain CTA (seq, v-head, v-slice): walks the chunks of one sequence; the 128 x VS state slice lives in
//         tensor memory:  S <- gamC*S + B + M' S ;  O = O_intra + Q' S   (one tcgen05 phase per chunk).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef GDN_ABL
#define GDN_ABL 0
#endif
#ifndef DBG_TID
#define DBG_TID 0
#endif
namespace gdn {

constexpr int C = 64, D = 128, HQ = 4, HV = 8;
constexpr int NTHREADS = 256, NWARPS = 8;
constexpr int REG64 = 64 * 128;    // bytes of one 64-row x 64-col bf16 swizzled region
constexpr int REG128 = 128 * 128;  // 128-row region

// Chunk record in the workspace (all matrices in the tensor-core smem layouts, copied verbatim).
struct __align__(1024) ChunkRec {
  unsigned char Mp[2 * REG128];  // M' = -K'^T W  [dim m][dim k] bf16, K-major 128B-swizzled (2 regions of 128 rows)
  unsigned char Qp[2 * REG64];   // Q' = scale*gam*Q - P W [token][dim] bf16, K-major (2 regions of 64 rows)
  __nv_bfloat16 Bc[D * D];       // B = K'^T U~ [dim][v] bf16 row-major
  __nv_bfloat16 Oi[C * D];       // O_intra = P U~ [token][v] bf16 row-major
  float gam[C];
  float gamc[C];
  float pad[128];                // 32768+16384+32768+16384+512+512 = 99328 (multiple of 1024)
};
static_assert(sizeof(ChunkRec) % 1024 == 0, "rec alignment");

struct Params {
  const __nv_bfloat16* q; const __nv_bfloat16* k; const __nv_bfloat16* v;
  const float* state; const float* A_log; const __nv_bfloat16* a; const float* dt_bias; const __nv_bfloat16* b;
  const int64_t* cu; float scale;
  __nv_bfloat16* out; float* new_state;
  ChunkRec* recs; int* flags; unsigned* counters;
  unsigned epoch; int T; int N; int num_prep; int num_chain; int debug; int64_t* dbg; int flags_ready;
};

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(unsigned saddr, const void* gmem, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(saddr), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
  __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&t);
}
__device__ __forceinline__ float2 unpack_bf16(unsigned u) {
  __nv_bfloat162 t = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(t);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ unsigned ld_acquire(const int* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v;
}
__device__ __forceinline__ void st_release(int* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v));
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(unsigned mbar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
}
__device__ __forceinline__ void mbar_wait(unsigned mbar, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}" ::"r"(mbar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tmem_alloc(unsigned dst_smem, unsigned ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc(unsigned taddr, unsigned ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols));
}
__device__ __forceinline__ uint64_t make_sdesc(unsigned saddr, unsigned lbo, unsigned sbo, unsigned swz_mode) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)swz_mode << 61;
  return d;
}
__device__ __forceinline__ unsigned make_idesc(int M, int N, int tA, int tB) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((unsigned)tA << 15) | ((unsigned)tB << 16) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24);
}
__device__ __forceinline__ void tc_mma(unsigned d_tmem, uint64_t adesc, uint64_t bdesc, unsigned idesc, int accum) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
               ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tc_mma_mask(unsigned d_tmem, uint64_t adesc, uint64_t bdesc, unsigned idesc, int accum,
                                            unsigned m0, unsigned m1, unsigned m2, unsigned m3) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n}"
               ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum), "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}
__device__ __forceinline__ void tc_commit(unsigned mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
}
__device__ __forceinline__ void tmem_ld32(unsigned taddr, unsigned (&r)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                 "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st32(unsigned taddr, const unsigned (&r)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
               ::"r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                 "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]),
                 "r"(taddr));
}
__device__ __forceinline__ void tmem_ld16(unsigned taddr, unsigned (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr));
}
__device__ __forceinline__ void tmem_st16(unsigned taddr, const unsigned (&r)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
               ::"r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ void bulk_s2g(void* gdst, unsigned ssrc, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_g2s(unsigned sdst, const void* gsrc, unsigned bytes, unsigned mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sdst), "l"(gsrc), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned mbar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}
// swizzled chunk index inside a 256B row of the B / O_intra records (16B chunks 0..15)
__device__ __forceinline__ int rowswz16(int r, int ch) { return (ch ^ (r & 15)) & 15; }
#define TL(i) do { if (p.dbg && threadIdx.x == 0) p.dbg[(size_t)blockIdx.x * 32 + (i)] = clock64(); } while (0)
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// swizzled 128B-row layouts: byte offset of 16B chunk `ch` (0..7) of row r
__device__ __forceinline__ unsigned swz128(int r, int ch) { return (unsigned)(r * 128 + ((ch ^ (r & 7)) << 4)); }
// element (r, k) of a [rows][128] bf16 matrix stored as 2 K-regions of `rows` rows (region stride rows*128)
__device__ __forceinline__ unsigned swz_elem(int rows, int r, int k) { return (unsigned)((k >> 6) * rows * 128) + swz128(r, (k & 63) >> 3) + (unsigned)((k & 7) * 2); }

// ------------------------------------------------------------------ prep role (chunk c, v-head h)
constexpr int P_KQ = 0;                        // 2 regions x 128 rows: [K rows 0-63 ; Q rows 64-127]
constexpr int P_V = P_KQ + 2 * REG128;         // V (2 x 64-row regions)
constexpr int P_W = P_V + 2 * REG64;           // W out
constexpr int P_U = P_V;                       // U~ out (in place over V: each warp overwrites only the rows/cols it consumed)
constexpr int P_P = P_W + 2 * REG64;           // P out (one 64-row region)
constexpr int LDC = 72, LDT = 24, LDY = 32;
constexpr int P_AHI = P_P + REG64;             // bf16 (-A) hi [64][72]
constexpr int P_ALO = P_AHI + C * LDC * 2;
constexpr int P_THI = P_ALO + C * LDC * 2;     // diag T blocks [64][24]
constexpr int P_TLO = P_THI + C * LDT * 2;
constexpr int P_AFD = P_TLO + C * LDT * 2;     // f32 diag blocks of A [4][16][17]
constexpr int P_YS = P_AFD + 4 * 16 * 17 * 4;  // per-warp Y scratch [8][16][40] bf16
constexpr int P_G = P_YS + 8 * 16 * LDY * 2;
constexpr int P_BETA = P_G + C * 4;
constexpr int P_GAMC = P_BETA + C * 4;
constexpr int P_WSUM = P_GAMC + C * 4;
constexpr int P_MBAR = P_WSUM + 64;
constexpr int P_TMEM = P_MBAR + 8;
constexpr int PREP_SMEM = P_TMEM + 8;
constexpr int PREP_SMEM_ALIGNED = ((PREP_SMEM + 1023) / 1024) * 1024;
constexpr int F_WQ = ((PREP_SMEM + 1023) / 1024) * 1024;   // fused: [W rows 0..63 ; Q rows 64..127] per dims-region (2 x REG128)
constexpr int F_SB = F_WQ + 2 * REG128;                     // fused: S bf16 [128 k][128 v] N-major (2 regions)
constexpr int F_UB = F_SB + 2 * REG128;                     // fused: U [64][128] N-major (2 x REG64)
constexpr int F_X = F_UB + 2 * REG64;                       // fused: extra 16 KB so that [F_SB, F_SB+64K) can land the input state
constexpr int FUSED_SMEM = F_X + 2 * REG64;

struct PrepCtx { int t0, L, seq, c, h; };
// FUSED: no record, no flag; W is written into the fused [W;Q] buffer (128-row regions); TMEM/mbarrier are owned by the caller.
template <bool FUSED>
__device__ void prep_core(const Params& p, const PrepCtx& cx, unsigned char* smem, unsigned tmem_in) {
  const int tid = threadIdx.x & 255, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int h = cx.h, c = cx.c, kh = h >> 1;
  const int half = threadIdx.x >> 8;
  auto sync_half = [&]() { asm volatile("bar.sync %0, 256;" ::"r"(half + 1) : "memory"); };
  const int t0 = cx.t0, L = cx.L;
  TL(1);
  ChunkRec& rec = p.recs[(size_t)(FUSED ? 0 : c) * HV + h];
  const unsigned sbase = smem_u32(smem);
  unsigned char* KQ = smem + P_KQ;
  unsigned char* Vs = smem + P_V;
  unsigned char* Wb = smem + P_W;
  unsigned char* Ub = smem + P_U;
  unsigned char* Pb = smem + P_P;
  __nv_bfloat16* Ahi = (__nv_bfloat16*)(smem + P_AHI);
  __nv_bfloat16* Alo = (__nv_bfloat16*)(smem + P_ALO);
  __nv_bfloat16* Thi = (__nv_bfloat16*)(smem + P_THI);
  __nv_bfloat16* Tlo = (__nv_bfloat16*)(smem + P_TLO);
  float* Afd = (float*)(smem + P_AFD);
  float* Gs = (float*)(smem + P_G);
  float* Bs = (float*)(smem + P_BETA);
  float* GCs = (float*)(smem + P_GAMC);
  float* wsum = (float*)(smem + P_WSUM);
  const unsigned mbar = sbase + P_MBAR;
  unsigned* tmem_slot = (unsigned*)(smem + P_TMEM);

  if (!FUSED) {
    if ((threadIdx.x >> 5) == 0) tmem_alloc(sbase + P_TMEM, (blockDim.x > 256) ? 512 : 256);
    if (tid == 0) { mbar_init(mbar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const unsigned tmem = FUSED ? tmem_in : (*(unsigned*)(smem - (threadIdx.x >> 8) * PREP_SMEM_ALIGNED + P_TMEM) + (threadIdx.x >> 8) * 256);

  // gate inputs first (their latency overlaps the cp.async issue below)
  float av = 0.f, bv = 0.f, alog = 0.f, dtb = 0.f;
  if (tid < C && tid < L) {
    av = __bfloat162float(p.a[(size_t)(t0 + tid) * HV + h]);
    bv = __bfloat162float(p.b[(size_t)(t0 + tid) * HV + h]);
    alog = p.A_log[h]; dtb = p.dt_bias[h];
  }
  // ---- async loads of K, Q (stacked), V into swizzled regions (rows >= L zero-filled)
  for (int idx = tid; idx < C * 16; idx += NTHREADS) {
    const int r = idx >> 4, ch = idx & 15, reg = ch >> 3, c8 = ch & 7;
    const bool valid = r < L;
    const int row = t0 + (valid ? r : 0);
    cp_async16(sbase + P_KQ + reg * REG128 + swz128(r, c8), p.k + ((size_t)row * HQ + kh) * D + ch * 8, valid);
    cp_async16(sbase + P_KQ + reg * REG128 + swz128(64 + r, c8), p.q + ((size_t)row * HQ + kh) * D + ch * 8, valid);
    cp_async16(sbase + P_V + reg * REG64 + swz128(r, c8), p.v + ((size_t)row * HV + h) * D + ch * 8, valid);
  }
  cp_async_commit();

  // ---- gates (warps 0,1): log g, inclusive cumulative G, beta
  if (tid < C) {
    const int i = tid;
    const bool valid = i < L;
    float lg = 0.f, beta = 0.f;
    if (valid) {
      lg = -expf(alog) * softplus_f(av + dtb);
      beta = 1.f / (1.f + expf(-bv));
    }
    float G = lg;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const float t = __shfl_up_sync(0xffffffffu, G, o); if (lane >= o) G += t; }
    if (lane == 31) wsum[warp] = G;
    Bs[i] = beta;
    Gs[i] = G;
  }
  sync_half();
  if (tid < C) {
    float G = Gs[tid];
    if (warp == 1) G += wsum[0];
    Gs[tid] = G;
  }
  sync_half();
  if (tid < C) {
    const float G = Gs[tid], G63 = Gs[C - 1];
    const float gc = expf(G63 - G), gm = expf(G);
    GCs[tid] = gc;
    if (!FUSED) { rec.gam[tid] = gm; rec.gamc[tid] = gc; }
  }
  TL(2);
  cp_async_wait<0>();
  fence_async_smem();
  sync_half();
  TL(3);

  // ---- [KK^T ; QK^T] = KQ * K^T on tcgen05:  M=128, N=64, K=128
  if (tid == 0) {
    tc_fence_after();
    const unsigned idesc = make_idesc(128, 64, 0, 0);
    for (int ks = 0; ks < 8; ks++) {
      const unsigned a0 = sbase + P_KQ + (ks >> 2) * REG128 + (ks & 3) * 32;
      tc_mma(tmem, make_sdesc(a0, 16, 1024, 2), make_sdesc(a0, 16, 1024, 2), idesc, ks > 0);
    }
    tc_commit(mbar);
  }
  mbar_wait(mbar, FUSED ? (unsigned)(cx.c & 1) : 0u);
  tc_fence_after();
  TL(4);
  // ---- A (f32 diag blocks + bf16 hi/lo of -A off-diagonal) and P (bf16, swizzled) from the TMEM rows.
  //      warp w: TMEM lanes 32*(w&3).. (A rows for w&3 < 2, P rows otherwise), columns 32*(w>>2)..
  {
    const int q = warp & 3, half = warp >> 2;
    unsigned rr[32];
    tmem_ld32(tmem + ((q * 32) << 16) + half * 32, rr);
    tmem_wait_ld();
    TL(21);
    const int i = (q & 1) * 32 + lane;  // token row
    const bool isA = q < 2;
    const float Gi = Gs[i];
    const float Bi = isA ? Bs[i] : p.scale;
    float Gv[32];
#pragma unroll
    for (int qq = 0; qq < 8; qq++) {
      const float4 g4 = *reinterpret_cast<const float4*>(Gs + half * 32 + 4 * qq);
      Gv[4 * qq] = g4.x; Gv[4 * qq + 1] = g4.y; Gv[4 * qq + 2] = g4.z; Gv[4 * qq + 3] = g4.w;
    }
    float a[32];
#pragma unroll
    for (int jj = 0; jj < 32; jj++) {
      const int j = half * 32 + jj;
      const bool keep = isA ? (j < i) : (j <= i);
      a[jj] = (keep ? Bi : 0.f) * __expf(fminf(Gi - Gv[jj], 0.f)) * __uint_as_float(rr[jj]);  // clamp: masked entries would overflow
    }
    if (isA) {
#pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        uint4 wh, wl;
        unsigned* ph = reinterpret_cast<unsigned*>(&wh);
        unsigned* pl = reinterpret_cast<unsigned*>(&wl);
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float v0 = -a[ch * 8 + 2 * e], v1 = -a[ch * 8 + 2 * e + 1];
          const float h0 = bf16_round(v0), h1 = bf16_round(v1);
          ph[e] = pack_bf16(h0, h1);
          pl[e] = pack_bf16(v0 - h0, v1 - h1);
        }
        *reinterpret_cast<uint4*>(Ahi + i * LDC + half * 32 + ch * 8) = wh;
        *reinterpret_cast<uint4*>(Alo + i * LDC + half * 32 + ch * 8) = wl;
      }
      if (half == (q & 1)) {  // the diagonal 16x16 block of row i lies in this column half
        float* dst = Afd + ((i >> 4) * 16 + (i & 15)) * 17;
#pragma unroll
        for (int jj = 0; jj < 16; jj++) dst[jj] = (lane < 16) ? a[jj] : a[16 + jj];
      }
    } else {
#pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        uint4 w;
        unsigned* pw = reinterpret_cast<unsigned*>(&w);
#pragma unroll
        for (int e = 0; e < 4; e++) pw[e] = pack_bf16(a[ch * 8 + 2 * e], a[ch * 8 + 2 * e + 1]);
        *reinterpret_cast<uint4*>(Pb + swz128(i, half * 4 + ch)) = w;
      }
    }
    TL(22);
  }
  sync_half();
  TL(5);
  // ---- warps 0..3: invert the 16x16 unit-lower-triangular diagonal blocks (I + A_bb)
  //      lane = (column l = lane&15, row half hf = lane>>4): rows 8hf..8hf+7 of column l.
  //      warps 4..7: copy P to the record meanwhile.
  if (warp < 4) {
    const int bb = warp, l = lane & 15, hf = lane >> 4;
    const float* A = Afd + bb * 16 * 17 + (8 * hf) * 17;
    float rest[8], acur[8], anext[8];
#pragma unroll
    for (int ii = 0; ii < 8; ii++) { rest[ii] = 0.f; acur[ii] = A[ii * 17]; }
#pragma unroll
    for (int j = 0; j < 16; j++) {
      if (j + 1 < 16) {
#pragma unroll
        for (int ii = 0; ii < 8; ii++) anext[ii] = A[ii * 17 + j + 1];
      }
      const float rj = __shfl_sync(0xffffffffu, rest[j & 7], l + 16 * (j >> 3));
      const float tj = (j == l) ? 1.f : -rj;
#pragma unroll
      for (int ii = 0; ii < 8; ii++) rest[ii] = fmaf(acur[ii], tj, rest[ii]);  // A[i][j] = 0 for j >= i
#pragma unroll
      for (int ii = 0; ii < 8; ii++) acur[ii] = anext[ii];
    }
    TL(23);
#pragma unroll
    for (int ii = 0; ii < 8; ii++) {
      const int i = 8 * hf + ii;
      const float t = (i == l) ? 1.f : ((i < l) ? 0.f : -rest[ii]);
      const float th = bf16_round(t);
      Thi[(16 * bb + i) * LDT + l] = __float2bfloat16(th);
      Tlo[(16 * bb + i) * LDT + l] = __float2bfloat16(t - th);
    }
    TL(24);
  }
  sync_half();
  TL(6);
  // ---- blocked forward substitution (8 warps, 4 n-tiles each over the 256-wide [W | U~]):
  //      Y_r = X_r - sum_{m<r} A_rm W_m ; W_r = T_rr Y_r ;   X = [beta*gam*K | beta*V]
  {
    const int n0 = 32 * warp;                       // first column of this warp in [W|U~]
    const bool isW = n0 < D;
    const unsigned char* Xsrc = isW ? KQ : Vs;      // source rows (K rows of KQ / V)
    const int xrows = isW ? 128 : 64;
    unsigned char* Out = isW ? (FUSED ? smem + F_WQ : Wb) : Ub;
    const int orows = (isW && FUSED) ? 128 : 64;
    const int xc = isW ? n0 : n0 - D;
    __nv_bfloat16* Ysw = (__nv_bfloat16*)(smem + P_YS) + warp * 16 * LDY;
#pragma unroll 1
    for (int r = 0; r < 4; r++) {
      float acc[4][4];
      const int i0 = 16 * r + g, i1 = i0 + 8;
      const float s0 = isW ? Bs[i0] * expf(Gs[i0]) : Bs[i0];
      const float s1 = isW ? Bs[i1] * expf(Gs[i1]) : Bs[i1];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const int col = xc + 8 * nt + 2 * tq;
        const float2 x0 = unpack_bf16(*reinterpret_cast<const unsigned*>(Xsrc + swz_elem(xrows, i0, col)));
        const float2 x1 = unpack_bf16(*reinterpret_cast<const unsigned*>(Xsrc + swz_elem(xrows, i1, col)));
        acc[nt][0] = x0.x * s0; acc[nt][1] = x0.y * s0; acc[nt][2] = x1.x * s1; acc[nt][3] = x1.y * s1;
      }
      for (int m = 0; m < r; m++) {
        unsigned ah[4], al[4];
        const int arow = 16 * r + (lane & 7) + 8 * ((lane >> 3) & 1), acol = 16 * m + 8 * (lane >> 4);
        ldsm_x4(ah, Ahi + arow * LDC + acol);
        ldsm_x4(al, Alo + arow * LDC + acol);
        const int brow = 16 * m + (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
        for (int np = 0; np < 2; np++) {
          unsigned b[4];
          const int bcol = xc + 16 * np + 8 * (lane >> 4);
          ldsm_x4_t(b, smem_u32(Out) + swz_elem(orows, brow, bcol));
          mma_bf16(acc[2 * np], ah, b[0], b[1]);
          mma_bf16(acc[2 * np + 1], ah, b[2], b[3]);
          mma_bf16(acc[2 * np], al, b[0], b[1]);
          mma_bf16(acc[2 * np + 1], al, b[2], b[3]);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        *reinterpret_cast<unsigned*>(Ysw + g * LDY + 8 * nt + 2 * tq) = pack_bf16(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<unsigned*>(Ysw + (g + 8) * LDY + 8 * nt + 2 * tq) = pack_bf16(acc[nt][2], acc[nt][3]);
      }
      __syncwarp();
      unsigned th[4], tl[4];
      ldsm_x4(th, Thi + (16 * r + (lane & 7) + 8 * ((lane >> 3) & 1)) * LDT + 8 * (lane >> 4));
      ldsm_x4(tl, Tlo + (16 * r + (lane & 7) + 8 * ((lane >> 3) & 1)) * LDT + 8 * (lane >> 4));
      float w[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) w[nt][e] = 0.f;
#pragma unroll
      for (int np = 0; np < 2; np++) {
        unsigned by[4];
        ldsm_x4_t(by, smem_u32(Ysw) + ((lane & 7) + 8 * ((lane >> 3) & 1)) * LDY * 2 + (16 * np + 8 * (lane >> 4)) * 2);
        mma_bf16(w[2 * np], th, by[0], by[1]);
        mma_bf16(w[2 * np + 1], th, by[2], by[3]);
        mma_bf16(w[2 * np], tl, by[0], by[1]);
        mma_bf16(w[2 * np + 1], tl, by[2], by[3]);
      }
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
        const int col = xc + 8 * nt + 2 * tq;
        const unsigned o0 = swz_elem(orows, i0, col), o1 = swz_elem(orows, i1, col);
        const unsigned v0 = pack_bf16(w[nt][0], w[nt][1]), v1 = pack_bf16(w[nt][2], w[nt][3]);
        *reinterpret_cast<unsigned*>(Out + o0) = v0;
        *reinterpret_cast<unsigned*>(Out + o1) = v1;
      }
      __syncwarp();
    }
  }
  sync_half();
  TL(7);
  // ---- K' = gamc (.) K in place (K rows 0..63 of the KQ regions)
  if (!(p.debug & 32768)) for (int idx = tid; idx < C * 16; idx += NTHREADS) {
    const int rr = idx >> 4, ch = idx & 15, reg = ch >> 3, c8 = ch & 7;
    const float s = GCs[rr];
    uint4* ptr = (uint4*)(KQ + reg * REG128 + swz128(rr, c8));
    uint4 v = *ptr;
    unsigned* w = reinterpret_cast<unsigned*>(&v);
#pragma unroll
    for (int e2 = 0; e2 < 4; e2++) { float2 f = unpack_bf16(w[e2]); w[e2] = pack_bf16(f.x * s, f.y * s); }
    *ptr = v;
  }
  fence_async_smem();
  tc_fence_before();
  sync_half();
  TL(25);
  if (FUSED) return;
  // ---- M' = -K'^T W (128x128), B = K'^T U~ (128x128), [PW | PU~] = P [W | U~] (M=128 rows, only 0..63 meaningful; N=256)
  if (tid == 0) {
    tc_fence_after();
    const unsigned idA = make_idesc(128, 128, 1, 1), idAn = idA | (1u << 13);
    for (int ks = 0; ks < 4; ks++) {
      const uint64_t ak = make_sdesc(sbase + P_KQ + ks * 2048, REG128, 1024, 2);   // K'^T: MN-major, 64-dim blocks REG128 apart
      const uint64_t bw = make_sdesc(sbase + P_W + ks * 2048, REG64, 1024, 2);     // W: N-major, 64-dim blocks REG64 apart
      const uint64_t bu = make_sdesc(sbase + P_U + ks * 2048, REG64, 1024, 2);     // U~: N-major
      tc_mma(tmem + 0, ak, bw, idAn, ks > 0);
      tc_mma(tmem + 128, ak, bu, idA, ks > 0);
    }
    tc_commit(mbar);
  }
  // ---- meanwhile (all warps): [PW | PU~] = P [W | U~] with mma.sync; warp w -> columns 32w..32w+31 of the 256-wide result
  {
    const int n0 = 32 * warp;
    const bool isW = n0 < D;
    const unsigned char* Bsrc = isW ? Wb : Ub;
    const int xc = isW ? n0 : n0 - D;
    float acc[4][4][4];  // [m-tile][n-tile][4]
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e2 = 0; e2 < 4; e2++) acc[mt][nt][e2] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      unsigned b[2][4];
#pragma unroll
      for (int np = 0; np < 2; np++) {
        const int brow = 16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1), bcol = xc + 16 * np + 8 * (lane >> 4);
        ldsm_x4_t(b[np], smem_u32(Bsrc) + swz_elem(64, brow, bcol));
      }
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        unsigned a[4];
        const int arow = 16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1), acol = 16 * ks + 8 * (lane >> 4);
        ldsm_x4(a, Pb + swz128(arow, acol >> 3));
#pragma unroll
        for (int np = 0; np < 2; np++) {
          mma_bf16(acc[mt][2 * np], a, b[np][0], b[np][1]);
          mma_bf16(acc[mt][2 * np + 1], a, b[np][2], b[np][3]);
        }
      }
    }
    TL(10);
    // Q' = scale*gam*Q - PW (W part) -> stageQ (record layout, 2 x 64-row regions) ; O_intra = PU~ -> stageO (row-major, chunk-swizzled rows)
    unsigned char* stageQ = smem + P_AHI;
    unsigned char* stageO = smem + P_AHI + 2 * REG64;
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int i = 16 * mt + g + 8 * half;
        const float sg = p.scale * expf(Gs[i]);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          const int col = xc + 8 * nt + 2 * tq;
          float v0 = acc[mt][nt][2 * half], v1 = acc[mt][nt][2 * half + 1];
          if (isW) {
            const float2 qf = unpack_bf16(*reinterpret_cast<const unsigned*>(KQ + swz_elem(128, 64 + i, col)));
            *reinterpret_cast<unsigned*>(stageQ + swz_elem(64, i, col)) = pack_bf16(sg * qf.x - v0, sg * qf.y - v1);
          } else {
            *reinterpret_cast<unsigned*>(stageO + i * 256 + rowswz16(i, col >> 3) * 16 + (col & 7) * 2) = pack_bf16(v0, v1);
          }
        }
      }
    }
  }
  TL(11);
  TL(26);
  mbar_wait(mbar, 1);
  tc_fence_after();
  sync_half();  // all warps are done reading KQ/W/U~ (mma.sync block) before they become the staging area
  TL(8);
  // ---- conversions -> staging (record layouts).  M'/B: warp w handles rows 32*(w&3)+lane, columns 64*(w>>2)..+63
  {
    const int q = warp & 3, hcol = warp >> 2, row = q * 32 + lane;
    unsigned ra[32], rb[32];
    unsigned char* stageM = smem + P_KQ;                // M': 2 K-major regions of 128 rows (32 KB)
    unsigned char* stageB = smem + P_KQ + 2 * REG128;   // B: row-major 256B rows, chunk-swizzled (32 KB)
    tmem_ld32(tmem + ((q * 32) << 16) + hcol * 64, ra);
    tmem_ld32(tmem + ((q * 32) << 16) + hcol * 64 + 32, rb);
    tmem_wait_ld();
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
      const unsigned* r8 = (ch < 4) ? ra : rb;
      uint4 w;
      w.x = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 0]), __uint_as_float(r8[(ch & 3) * 8 + 1]));
      w.y = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 2]), __uint_as_float(r8[(ch & 3) * 8 + 3]));
      w.z = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 4]), __uint_as_float(r8[(ch & 3) * 8 + 5]));
      w.w = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 6]), __uint_as_float(r8[(ch & 3) * 8 + 7]));
      *reinterpret_cast<uint4*>(stageM + hcol * REG128 + swz128(row, ch)) = w;
    }
    tmem_ld32(tmem + 128 + ((q * 32) << 16) + hcol * 64, ra);
    tmem_ld32(tmem + 128 + ((q * 32) << 16) + hcol * 64 + 32, rb);
    tmem_wait_ld();
#pragma unroll
    for (int ch = 0; ch < 8; ch++) {
      const unsigned* r8 = (ch < 4) ? ra : rb;
      uint4 w;
      w.x = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 0]), __uint_as_float(r8[(ch & 3) * 8 + 1]));
      w.y = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 2]), __uint_as_float(r8[(ch & 3) * 8 + 3]));
      w.z = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 4]), __uint_as_float(r8[(ch & 3) * 8 + 5]));
      w.w = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 6]), __uint_as_float(r8[(ch & 3) * 8 + 7]));
      *reinterpret_cast<uint4*>(stageB + row * 256 + rowswz16(row, hcol * 8 + ch) * 16) = w;
    }
  }
  fence_async_smem();
  sync_half();
  if (tid == 0) {
    bulk_s2g(rec.Mp, sbase + P_KQ, 2 * REG128);
    bulk_s2g(rec.Bc, sbase + P_KQ + 2 * REG128, 2 * REG128);
    bulk_s2g(rec.Qp, sbase + P_AHI, 2 * REG64);
    bulk_s2g(rec.Oi, sbase + P_AHI + 2 * REG64, 2 * REG64);
    bulk_commit();
    bulk_wait_all();
  }
  __threadfence();
  tc_fence_before();
  sync_half();
  if (tid == 0) st_release(p.flags + c * HV + h, p.epoch);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tmem_dealloc(tmem, (blockDim.x > 256) ? 512 : 256);
  TL(9);
}

// record-producing prep CTA (chunk c, v-head h)
__device__ void prep_role(const Params& p, int c, int h, unsigned char* smem) {
  // smem/tmem of this half: the caller passes the half's smem base; TMEM columns 0..255 / 256..511 are used by halves 0 / 1
  PrepCtx cx; cx.c = c; cx.h = h; cx.seq = -1; cx.t0 = 0; cx.L = 0;
  {
    int acc = 0;
    for (int s = 0; s < p.N; s++) {
      int st = (int)p.cu[s], en = (int)p.cu[s + 1];
      int len = en - st; if (len < 0) len = 0;
      int nc = (len + C - 1) / C;
      if (c < acc + nc) { int lc = c - acc; cx.seq = s; cx.t0 = st + lc * C; cx.L = min(C, len - lc * C); break; }
      acc += nc;
    }
  }
  if (cx.seq < 0) {  // chunk slot beyond the real chunks: run a dummy pass on chunk 0 (keeps all halves in lockstep; its record/flag are unused)
    cx.seq = 0; cx.t0 = (int)p.cu[0]; cx.L = min(C, (int)(p.cu[1] - p.cu[0]));
    if (cx.L <= 0) cx.L = 1;
  }
  prep_core<false>(p, cx, smem, 0u);
}

// ------------------------------------------------------------------ fused CTA (seq, v-head h): all chunks, whole head, no records
__device__ void fused_role(const Params& p, int seq, int h, unsigned char* smem) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const unsigned sbase = smem_u32(smem);
  const int sstart = (int)p.cu[seq];
  int len = (int)(p.cu[seq + 1] - p.cu[seq]); if (len < 0) len = 0;
  const int nch = (len + C - 1) / C;
  float* new_state = p.new_state + (size_t)(seq * HV + h) * D * D;  // [v][k]
  if (len == 0) {
    for (int i = tid; i < D * D / 4; i += NTHREADS) reinterpret_cast<float4*>(new_state)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  const unsigned mbar = sbase + P_MBAR;
  unsigned* tmem_slot = (unsigned*)(smem + P_TMEM);
  if (warp == 0) tmem_alloc(sbase + P_TMEM, 512);
  if (tid == 0) { mbar_init(mbar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const unsigned tmem = *tmem_slot;
  const unsigned tS = tmem + 256, tUO = tmem + 384;  // columns 0..63 are the prep's KK/QK accumulator
  for (int i = tid; i < REG64 / 16; i += NTHREADS) reinterpret_cast<uint4*>(smem + P_P - REG64)[i] = make_uint4(0u, 0u, 0u, 0u);  // junk rows of [junk;P] -> zeros
  unsigned char* KQ = smem + P_KQ;
  unsigned char* WQ = smem + F_WQ;
  unsigned char* SB = smem + F_SB;
  unsigned char* UB = smem + F_UB;
  const float* Gs = (const float*)(smem + P_G);
  const float* GCs = (const float*)(smem + P_GAMC);
  // input state [128 v][128 k] f32 -> landing zone [F_SB, F_SB + 64 KB) via cp.async (overlaps the whole prep of chunk 0)
  {
    const float* st = p.state ? p.state + (size_t)(seq * HV + h) * D * D : nullptr;
    for (int i = tid; i < D * D / 4; i += NTHREADS) cp_async16(sbase + F_SB + i * 16, st ? (const void*)(st + i * 4) : (const void*)p.cu, st != nullptr);
    cp_async_commit();
  }
  const unsigned idUO = make_idesc(128, 128, 0, 1);
  const unsigned idS = make_idesc(128, 128, 1, 1);
  for (int c = 0; c < nch; c++) {
    PrepCtx cx; cx.c = c; cx.h = h; cx.seq = seq; cx.t0 = sstart + c * C; cx.L = min(C, len - c * C);
    if (c > 0) { tc_fence_before(); __syncthreads(); }
    prep_core<true>(p, cx, smem, tmem);           // W -> WQ rows 0..63, U~ -> V buffer, P -> Pb, K' in KQ rows 0..63
    const int t0 = cx.t0, Lc = cx.L;
    // Q' = scale*gam*Q rows (KQ rows 64..127, region stride REG128) -> WQ rows 64..127
    for (int idx = tid; idx < 2 * 64 * 8; idx += NTHREADS) {
      const int reg = idx >> 9, rr = (idx & 511) >> 3, c8 = idx & 7;
      const float s = p.scale * expf(Gs[rr]);
      uint4 v = *reinterpret_cast<const uint4*>(KQ + reg * REG128 + swz128(64 + rr, c8));
      unsigned* w = reinterpret_cast<unsigned*>(&v);
#pragma unroll
      for (int e2 = 0; e2 < 4; e2++) { float2 f = unpack_bf16(w[e2]); w[e2] = pack_bf16(f.x * s, f.y * s); }
      *reinterpret_cast<uint4*>(WQ + reg * REG128 + swz128(64 + rr, c8)) = v;
    }
    // S_tmem <- gamC * S  (c > 0: S is the previous chunk's result; c == 0: from the landing zone)
    {
      const int q = warp & 3, vh = warp >> 2, k = q * 32 + lane;
      const float gC = expf(Gs[C - 1]);
      unsigned a[32], b[32];
      if (c == 0) {
        const float* zone = (const float*)(smem + F_SB);
#pragma unroll
        for (int j = 0; j < 32; j++) { a[j] = __float_as_uint(zone[(vh * 64 + j) * D + k]); b[j] = __float_as_uint(zone[(vh * 64 + 32 + j) * D + k]); }
        __syncthreads();  // everyone has read the zone before S_bf16 overwrites it
      } else {
        tc_fence_after();
        tmem_ld32(tS + ((q * 32) << 16) + vh * 64, a);
        tmem_ld32(tS + ((q * 32) << 16) + vh * 64 + 32, b);
        tmem_wait_ld();
      }
      {
#pragma unroll
        for (int ch = 0; ch < 8; ch++) {
          const unsigned* r8 = (ch < 4) ? a : b;
          uint4 w;
          w.x = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 0]), __uint_as_float(r8[(ch & 3) * 8 + 1]));
          w.y = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 2]), __uint_as_float(r8[(ch & 3) * 8 + 3]));
          w.z = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 4]), __uint_as_float(r8[(ch & 3) * 8 + 5]));
          w.w = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 6]), __uint_as_float(r8[(ch & 3) * 8 + 7]));
          *reinterpret_cast<uint4*>(SB + vh * REG128 + swz128(k, ch)) = w;
        }
      }
#pragma unroll
      for (int j = 0; j < 32; j++) { a[j] = __float_as_uint(__uint_as_float(a[j]) * gC); b[j] = __float_as_uint(__uint_as_float(b[j]) * gC); }
      tmem_st32(tS + ((q * 32) << 16) + vh * 64, a);
      tmem_st32(tS + ((q * 32) << 16) + vh * 64 + 32, b);
      tmem_wait_st();
    }
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    // ---- [U_pre ; O_pre] = [W ; Q] S_bf16   (M=128, N=128, K=128)
    if (tid == 0) {
      tc_fence_after();
      for (int ks = 0; ks < 8; ks++) {
        const uint64_t ad = make_sdesc(sbase + F_WQ + (ks >> 2) * REG128 + (ks & 3) * 32, 16, 1024, 2);
        const uint64_t bd = make_sdesc(sbase + F_SB + ks * 2048, REG128, 1024, 2);
        tc_mma(tUO, ad, bd, idUO, ks > 0);
      }
      tc_commit(mbar);
    }
    mbar_wait(mbar, (unsigned)((c & 1) ^ 1));  // commits per chunk: KKQK (3c), UO (3c+1), PU/S (3c+2)
    tc_fence_after();
    // ---- U = U~ - U_pre (rows = tokens at lanes 0..63: warps 0,1 cols 0..63 ; warps 4,5 cols 64..127) ; U' = gamc*U
    if ((warp & 3) < 2) {
      const int q = warp & 3, vh = warp >> 2, i = q * 32 + lane;
      unsigned a[32], b[32];
      tmem_ld32(tUO + ((q * 32) << 16) + vh * 64, a);
      tmem_ld32(tUO + ((q * 32) << 16) + vh * 64 + 32, b);
      tmem_wait_ld();
#pragma unroll
      for (int ch = 0; ch < 8; ch++) {
        const unsigned* r8 = (ch < 4) ? a : b;
        const int vcol = vh * 64 + ch * 8;
        const uint4 uu = *reinterpret_cast<const uint4*>(smem + P_U + swz_elem(64, i, vcol));
        const unsigned* uw = reinterpret_cast<const unsigned*>(&uu);
        uint4 w;
        unsigned* ww = reinterpret_cast<unsigned*>(&w);
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float2 ut = unpack_bf16(uw[e]);
          const float u0 = ut.x - __uint_as_float(r8[(ch & 3) * 8 + 2 * e]), u1 = ut.y - __uint_as_float(r8[(ch & 3) * 8 + 2 * e + 1]);
          ww[e] = pack_bf16(u0, u1);
        }
        *reinterpret_cast<uint4*>(UB + vh * REG64 + swz128(i, ch)) = w;
      }
    }
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    // ---- O_pre += P U (lanes 64..127 via A = [junk ; P]) ; S = gamC S + K'^T U'
    if (tid == 0) {
      tc_fence_after();
      for (int ks = 0; ks < 4; ks++) {
        const uint64_t ap = make_sdesc(sbase + P_P - REG64 + ks * 32, 16, 1024, 2);   // rows 0..63 junk (W buffer tail), rows 64..127 = P
        const uint64_t bu = make_sdesc(sbase + F_UB + ks * 2048, REG64, 1024, 2);
        tc_mma_mask(tUO, ap, bu, idUO, 1, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u);
        const uint64_t ak = make_sdesc(sbase + P_KQ + ks * 2048, REG128, 1024, 2);
        tc_mma(tS, ak, bu, idS, 1);   // K' = gamc*K already carries the decay factor
      }
      tc_commit(mbar);
    }
    mbar_wait(mbar, (unsigned)(c & 1));
    tc_fence_after();
    // ---- O rows (lanes 64..127): warps 2,3 cols 0..63 ; warps 6,7 cols 64..127
    if ((warp & 3) >= 2) {
      const int q = warp & 3, vh = warp >> 2, i = (q - 2) * 32 + lane;
      unsigned a[32], b[32];
      tmem_ld32(tUO + ((q * 32) << 16) + vh * 64, a);
      tmem_ld32(tUO + ((q * 32) << 16) + vh * 64 + 32, b);
      tmem_wait_ld();
      if (i < Lc) {
        __nv_bfloat16* op = p.out + ((size_t)(t0 + i) * HV + h) * D + vh * 64;
#pragma unroll
        for (int ch = 0; ch < 8; ch++) {
          const unsigned* r8 = (ch < 4) ? a : b;
          uint4 w;
          w.x = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 0]), __uint_as_float(r8[(ch & 3) * 8 + 1]));
          w.y = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 2]), __uint_as_float(r8[(ch & 3) * 8 + 3]));
          w.z = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 4]), __uint_as_float(r8[(ch & 3) * 8 + 5]));
          w.w = pack_bf16(__uint_as_float(r8[(ch & 3) * 8 + 6]), __uint_as_float(r8[(ch & 3) * 8 + 7]));
          *reinterpret_cast<uint4*>(op + ch * 8) = w;
        }
      }
    }
  }
  // ---- final state: S_tmem [k lanes][v cols] -> new_state[v][k] (coalesced: warp writes 128 B per v)
  {
    const int q = warp & 3, vh = warp >> 2, k = q * 32 + lane;
    unsigned a[32], b[32];
    tc_fence_after();
    tmem_ld32(tS + ((q * 32) << 16) + vh * 64, a);
    tmem_ld32(tS + ((q * 32) << 16) + vh * 64 + 32, b);
    tmem_wait_ld();
#pragma unroll
    for (int j = 0; j < 32; j++) { new_state[(size_t)(vh * 64 + j) * D + k] = __uint_as_float(a[j]); new_state[(size_t)(vh * 64 + 32 + j) * D + k] = __uint_as_float(b[j]); }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tmem, 512);
}


// ------------------------------------------------------------------ chain role (seq, v-head h, v-slice js)
// stage layout (bytes): M' 2 x REG128 | Q' 2 x REG64 | B slice [128][VS] bf16 | O_intra slice [64][VS] bf16 | gam
template <int VS, int NSTAGES>
struct ChainSmem {
  static constexpr int RB = 2 * VS;                        // row bytes of the N-major B operand / slices
  static constexpr int S_MP = 0, S_QP = 2 * REG128, S_BC = S_QP + 2 * REG64, S_OI = S_BC + D * RB, S_G = S_OI + C * RB;
  static constexpr int STAGE = ((S_G + C * 4 + 1023) / 1024) * 1024;
  static constexpr int NST = NSTAGES;                      // prefetch depth
  static constexpr int SB = NST * STAGE;                   // S (bf16) [128 dims][VS]
  static constexpr int MISC = SB + D * RB;                 // misc ints / mbarriers / tmem slot / stage mbarriers (at +64)
  static constexpr int BYTES = MISC + 128;
  static constexpr int TCOLS = (VS == 32) ? 128 : 256;     // TMEM: S | O[0] | O[1]
};

template <int VS, int NSTAGES>
__device__ void chain_role(const Params& p, int r, unsigned char* smem) {
  using CS = ChainSmem<VS, NSTAGES>;
  constexpr int NSPLIT = D / VS, RB = CS::RB, NCH = VS / 8, STAGE = CS::STAGE, NST = CS::NST;
  constexpr int SWZB = (VS == 32) ? 2 : 3;
  constexpr unsigned SWZMODE = (VS == 32) ? 4u : 2u;
  constexpr int NB32 = VS / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = HV * NSPLIT, slot = r / per, rem = r % per, h = rem / NSPLIT, js = rem % NSPLIT;
  const unsigned sbase = smem_u32(smem);
  int* misc = (int*)(smem + CS::MISC);
  const unsigned mbar0 = sbase + CS::MISC + 16;
  unsigned* tmem_slot = (unsigned*)(smem + CS::MISC + 32);
  if (tid == 0) misc[0] = -1;
  __syncthreads();
  for (int x = tid; x < p.N; x += NTHREADS) {  // sequence with rank `slot` by descending length
    const int lx = (int)(p.cu[x + 1] - p.cu[x]);
    int rank = 0;
    for (int y = 0; y < p.N; y++) { const int ly = (int)(p.cu[y + 1] - p.cu[y]); rank += (ly > lx) || (ly == lx && y < x); }
    if (rank == slot) misc[0] = x;
  }
  __syncthreads();
  const int seq = misc[0];
  if (seq < 0) return;
  TL(1);
  if (p.debug & (512 | 8)) return;
  const int sstart = (int)p.cu[seq];
  int len = (int)(p.cu[seq + 1] - p.cu[seq]); if (len < 0) len = 0;
  const int nch = (len + C - 1) / C;
  int cb = 0;
  for (int s = 0; s < seq; s++) { const int l = (int)(p.cu[s + 1] - p.cu[s]); if (l > 0) cb += (l + C - 1) / C; }
  float* new_state = p.new_state + ((size_t)(seq * HV + h) * D + VS * js) * D;  // [v][k] rows
  if (len == 0) {
    for (int i = tid; i < VS * D / 4; i += NTHREADS) reinterpret_cast<float4*>(new_state)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  if (warp == 0) tmem_alloc(sbase + CS::MISC + 32, CS::TCOLS);
  if (tid == 0) { mbar_init(mbar0, 1); for (int s2 = 0; s2 < NST; s2++) mbar_init(sbase + CS::MISC + 64 + 8 * s2, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

  // stage loads: warps 0..3 copy M'/Q' (common parts), warps 4..7 copy the B/O_intra slices + gam (after their output emission,
  // which reads the O_intra slice of the stage being refilled) -> no block barrier needed between emission and prefetch.
  auto issue_stage = [&](int c, int buf) {
    const unsigned sb = sbase + buf * STAGE;
    const ChunkRec& rec = p.recs[(size_t)(cb + c) * HV + h];
    if (warp < 4) {
      if (tid == 0) {  // M' and Q' are adjacent in the record and in the stage: one 48 KB bulk copy
        const unsigned mb = sbase + CS::MISC + 64 + 8 * buf;
        mbar_expect_tx(mb, 2 * REG128 + 2 * REG64);
        bulk_g2s(sb + CS::S_MP, rec.Mp, 2 * REG128 + 2 * REG64, mb);
      }
    } else {
      const int it = tid - 128;
      for (int idx = it; idx < D * NCH; idx += 128) {
        const int rr = idx / NCH, ch = idx % NCH;
        cp_async16(sb + CS::S_BC + rr * RB + ch * 16, rec.Bc + rr * D + rowswz16(rr, VS * js / 8 + ch) * 8, true);
      }
      for (int idx = it; idx < C * NCH; idx += 128) {
        const int rr = idx / NCH, ch = idx % NCH;
        cp_async16(sb + CS::S_OI + rr * RB + ch * 16, rec.Oi + rr * D + rowswz16(rr, VS * js / 8 + ch) * 8, true);
      }
      if (it < 16) cp_async16(sb + CS::S_G + it * 16, rec.gam + it * 4, true);
    }
    cp_async_commit();
  };
  auto wait_flag = [&](int c) {
    if (p.flags_ready) return;  // chain kernel launched after the prep kernel: all records are complete
    const int* f = p.flags + (cb + c) * HV + h;
    if (lane == 0 && !(p.debug & 1)) { while (ld_acquire(f) != p.epoch) { } }
    __syncwarp();
  };
  // ---- initial state slice -> smem scratch (last stage buffer area, [VS][128] f32; it is first prefetched into after the S handling of chunk 0)
  float* scratch = (float*)(smem + (NST - 1) * STAGE);
  {
    const float* st = p.state ? p.state + ((size_t)(seq * HV + h) * D + VS * js) * D : nullptr;
    for (int i = tid; i < VS * D / 4; i += NTHREADS)
      reinterpret_cast<float4*>(scratch)[i] = st ? reinterpret_cast<const float4*>(st)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const unsigned tmem = *tmem_slot;
  const unsigned tS = tmem, tO0 = tmem + VS, tO1 = tmem + 2 * VS;
  TL(2);
  wait_flag(0);
  TL(3);
  issue_stage(0, 0);
  if (NST == 3 && nch > 1) { wait_flag(1); issue_stage(1, 1); }
  const unsigned idescS = make_idesc(128, VS, 0, 1);
  const unsigned idescO = make_idesc(64, VS, 0, 1);

  // O output of chunk c (rows at lanes (m/16)*32 + m%16): warps 4..7 -> quadrant q = warp&3 holds rows 16q + (lane&15)
  auto emit_output = [&](int c) {
    const int q = warp & 3, i = 16 * q + (lane & 15);
    const unsigned tO = (c & 1) ? tO1 : tO0;
    const unsigned char* oi = smem + (c % NST) * STAGE + CS::S_OI;
    const int t0 = sstart + c * C, Lc = min(C, len - c * C);
#pragma unroll
    for (int cb32 = 0; cb32 < NB32; cb32++) {
      unsigned ra[32];
      tmem_ld32(tO + ((q * 32) << 16) + cb32 * 32, ra);
      tmem_wait_ld();
      if (lane < 16 && i < Lc) {
        __nv_bfloat16* op = p.out + ((size_t)(t0 + i) * HV + h) * D + VS * js + cb32 * 32;
#pragma unroll
        for (int ch = 0; ch < 4; ch++) {
          const uint4 ov = *reinterpret_cast<const uint4*>(oi + i * RB + (cb32 * 4 + ch) * 16);
          const unsigned* ow = reinterpret_cast<const unsigned*>(&ov);
          uint4 w;
          unsigned* ww = reinterpret_cast<unsigned*>(&w);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 of = unpack_bf16(ow[e]);
            ww[e] = pack_bf16(__uint_as_float(ra[ch * 8 + 2 * e]) + of.x, __uint_as_float(ra[ch * 8 + 2 * e + 1]) + of.y);
          }
          *reinterpret_cast<uint4*>(op + ch * 8) = w;
        }
      }
    }
  };

  for (int c = 0; c < nch; c++) {
    const int buf = c % NST;
    const unsigned sb = sbase + buf * STAGE;
    const float* gs = (const float*)(smem + buf * STAGE + CS::S_G);
    if (NST == 3 && c + 1 < nch) cp_async_wait<1>(); else cp_async_wait<0>();
    mbar_wait(sbase + CS::MISC + 64 + 8 * buf, (unsigned)((c / NST) & 1));
    fence_async_smem();
    __syncthreads();
    if (c == 0) TL(4); if (c == 1) TL(14); if (c == nch - 1 && c > 1) TL(21);
    // ---- S handling (all warps; warp w: lanes 32*(w&3) = dims, columns 16*(w>>2) + 32*cb32 .. +15):
    //      S_bf16 <- S_c ; S_tmem <- gamC_c * S_c + B_c
    {
      const int q = warp & 3, hh = warp >> 2, d = q * 32 + lane;
      const float gC = gs[C - 1];
      if (c > 0) tc_fence_after();
#pragma unroll
      for (int cb32 = 0; cb32 < NB32; cb32++) {
        const int col0 = cb32 * 32 + hh * 16;
        float sv[16];
        if (c == 0) {
#pragma unroll
          for (int v = 0; v < 16; v++) sv[v] = scratch[(col0 + v) * D + d];
        } else {
          unsigned ra[16];
          tmem_ld16(tS + ((q * 32) << 16) + col0, ra);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 16; j++) sv[j] = __uint_as_float(ra[j]);
        }
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
          uint4 w;
          w.x = pack_bf16(sv[ch * 8 + 0], sv[ch * 8 + 1]);
          w.y = pack_bf16(sv[ch * 8 + 2], sv[ch * 8 + 3]);
          w.z = pack_bf16(sv[ch * 8 + 4], sv[ch * 8 + 5]);
          w.w = pack_bf16(sv[ch * 8 + 6], sv[ch * 8 + 7]);
          const int chg = col0 / 8 + ch;  // 16B chunk index within the row
          const unsigned off = d * RB + ((chg ^ ((d * RB >> 7) & ((1 << SWZB) - 1))) << 4);
          *reinterpret_cast<uint4*>(smem + CS::SB + off) = w;
        }
        const unsigned char* bc = smem + buf * STAGE + CS::S_BC + d * RB + col0 * 2;
        unsigned rr[16];
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
          const uint4 bv = *reinterpret_cast<const uint4*>(bc + ch * 16);
          const unsigned* bw = reinterpret_cast<const unsigned*>(&bv);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 bf = unpack_bf16(bw[e]);
            rr[ch * 8 + 2 * e] = __float_as_uint(sv[ch * 8 + 2 * e] * gC + bf.x);
            rr[ch * 8 + 2 * e + 1] = __float_as_uint(sv[ch * 8 + 2 * e + 1] * gC + bf.y);
          }
        }
        tmem_st16(tS + ((q * 32) << 16) + col0, rr);
      }
      tmem_wait_st();
    }
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    if (c == 0) TL(5); if (c == 1) TL(15); if (c == nch - 1 && c > 1) TL(22);
    // ---- S += M' S_bf16 ; O[c&1] = Q' S_bf16     (K = 128 dims, 8 steps each)
    if (tid == 0 && !(p.debug & (1 << 21))) {
      tc_fence_after();
      const unsigned tO = (c & 1) ? tO1 : tO0;  // O accumulators alternate by chunk parity (independent of the stage index)
      for (int ks = 0; ks < 8; ks++) {
        const uint64_t am = make_sdesc(sb + CS::S_MP + (ks >> 2) * REG128 + (ks & 3) * 32, 16, 1024, 2);
        const uint64_t aq = make_sdesc(sb + CS::S_QP + (ks >> 2) * REG64 + (ks & 3) * 32, 16, 1024, 2);
        const uint64_t bd = make_sdesc(sbase + CS::SB + ks * 16 * RB, 16, 8 * RB, SWZMODE);
        tc_mma(tS, am, bd, idescS, 1);
        tc_mma(tO, aq, bd, idescO, ks > 0);
      }
      tc_commit(mbar0);
    }
    // ---- while the MMAs run: emit the previous chunk's output (warps 4..7) and prefetch the next stage
    if (c > 0 && warp >= 4) emit_output(c - 1);
    if (!(p.debug & (1 << 20))) {
      if (NST == 3) { if (c + 2 < nch) { wait_flag(c + 2); issue_stage(c + 2, (c + 2) % NST); } }
      else { if (c + 1 < nch) { wait_flag(c + 1); issue_stage(c + 1, (c + 1) % NST); } }
    }
    if (!(p.debug & (1 << 21))) mbar_wait(mbar0, c & 1);
    tc_fence_after();
    if (c == 0) TL(6); if (c == 1) TL(16); if (c == nch - 1 && c > 1) TL(23);
  }
  if (warp >= 4) emit_output(nch - 1);
  // ---- final state -> scratch (stage 0 area is free now; stage 1 may hold O_intra of the last chunk if nch is even... use stage of the other parity)
  float* fin = (float*)(smem + (((nch - 1) % NST + 1) % NST) * STAGE);
  if (warp < 4) {
    tc_fence_after();
    const int d = warp * 32 + lane;
#pragma unroll
    for (int cb32 = 0; cb32 < NB32; cb32++) {
      unsigned ra[32];
      tmem_ld32(tS + ((warp * 32) << 16) + cb32 * 32, ra);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 32; j++) fin[(cb32 * 32 + j) * D + d] = __uint_as_float(ra[j]);
    }
  }
  tc_fence_before();
  __syncthreads();
  for (int i = tid; i < VS * D / 4; i += NTHREADS) reinterpret_cast<float4*>(new_state)[i] = reinterpret_cast<const float4*>(fin)[i];
  if (warp == 0) tmem_dealloc(tmem, CS::TCOLS);
  TL(20);
}

__global__ void __launch_bounds__(2 * NTHREADS, 1) gdn_prep_kernel(Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  TL(0);
  if (p.dbg && threadIdx.x == 0) { unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 28] = (int64_t)gt; }
  const int half = threadIdx.x >> 8;
  const int t = min(2 * blockIdx.x + half, p.num_prep - 1);  // both halves must run the same barrier sequence
  prep_role(p, t / HV, t % HV, smem + half * PREP_SMEM_ALIGNED);
  TL(30);
  if (p.dbg && threadIdx.x == 0) { p.dbg[(size_t)blockIdx.x * 32 + 31] = (int64_t)blockIdx.x; unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 29] = (int64_t)gt; }
}
// ------------------------------------------------------------------ recurrent path for short sequences (CUDA cores)
// CTA = (seq, v-head h, v-slice of RVS columns); thread = (dg = tid % LPC -> DPT dims, vl = tid / LPC -> one column).
// Per token: c_t = k_t.S_{t-1} (per column), u_t = beta_t (v_t - g_t c_t), S_t = g_t S_{t-1} + k_t^T u_t, o_t = scale q_t.S_t.
// DPT (dims per thread) = 4 / 8 / 16 -> RVS = NT/LPC columns per CTA, grid = 8 N D / RVS CTAs: small DPT spreads the
// per-token work over more SMs (the token loop is bound by the shared-memory/shuffle pipe and issue rate).
// Lookahead (LA = 3): the cross-lane reduction is taken off the per-token critical path by expanding the recurrence
//   c_t = gp_t e_t + c1_t u_{t-1} + c2_t u_{t-2} + c3_t u_{t-3},   e_t = k_t . S_{max(t-4,-1)}
// with per-block scalar tables gp_t = prod_{j=max(t-3,0)}^{t-1} g_j, ci_t = (k_t.k_{t-i}) prod_{j=t-i+1}^{t-1} g_j (0 if t<i),
// so e_t is a full dot product issued 4 tokens early and the dependent chain per token is two FMAs.
// Each k row is read from shared memory once (for e_{t+4}) and kept in a 4-deep register ring for the S update.
template <int DPT, int NT> struct RecurCfg {
  static constexpr int LPC = D / DPT, RVS = NT / LPC;
  static constexpr int R_K = 0, R_Q = R_K + C * 128 * 2, R_V = R_Q + C * 128 * 2, R_G = R_V + C * RVS * 2, R_B = R_G + C * 4;
  static constexpr int R_CF = R_B + C * 4, R_GB = R_CF + C * 16, R_O = R_GB + C * 8;
  static constexpr int SMEM = R_O + C * RVS * 2;
};

template <int DPT, int NT>
__global__ void __launch_bounds__(NT, 1) gdn_recur_kernel(Params p) {
  using Cfg = RecurCfg<DPT, NT>;
  constexpr int LPC = Cfg::LPC, RVS = Cfg::RVS, KW = DPT / 2;
  extern __shared__ __align__(16) unsigned char smem[];
  TL(0);
  const int tid = threadIdx.x, lane = tid & 31, dg = tid % LPC, vl = tid / LPC;
  const int js = blockIdx.x % (D / RVS), h = (blockIdx.x / (D / RVS)) % HV, seq = blockIdx.x / (D / RVS * HV), kh = h >> 1;
  const int v = RVS * js + vl;
  float* new_state = p.new_state + (size_t)(seq * HV + h) * D * D;  // [v][k]
  // state slice: S[DPT*dg..+DPT-1][v]  <- state[v][DPT*dg..]   (issued before the cu_seqlens round trip)
  float s[DPT];
  {
    const float* st = p.state ? p.state + (size_t)(seq * HV + h) * D * D + (size_t)v * D + DPT * dg : nullptr;
#pragma unroll
    for (int i = 0; i < DPT / 4; i++) {
      const float4 x = st ? reinterpret_cast<const float4*>(st)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
      s[4 * i] = x.x; s[4 * i + 1] = x.y; s[4 * i + 2] = x.z; s[4 * i + 3] = x.w;
    }
  }
  const float alog = expf(p.A_log[h]), dtb = p.dt_bias[h];
  const int sstart = (int)p.cu[seq];
  int len = (int)(p.cu[seq + 1] - p.cu[seq]); if (len < 0) len = 0;
  if (len == 0) {
    for (int i = tid; i < RVS * D / 4; i += NT) reinterpret_cast<float4*>(new_state + (size_t)RVS * js * D)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  __nv_bfloat16* Ks = (__nv_bfloat16*)(smem + Cfg::R_K);
  __nv_bfloat16* Qs = (__nv_bfloat16*)(smem + Cfg::R_Q);
  __nv_bfloat16* Vs = (__nv_bfloat16*)(smem + Cfg::R_V);
  float* gs = (float*)(smem + Cfg::R_G);
  float* bs = (float*)(smem + Cfg::R_B);
  float4* cf4 = (float4*)(smem + Cfg::R_CF);
  float2* gb2 = (float2*)(smem + Cfg::R_GB);
  __nv_bfloat16* Os = (__nv_bfloat16*)(smem + Cfg::R_O);
  TL(1);
  auto load_row = [&](const __nv_bfloat16* row, float* f) {  // this thread's DPT dims of a bf16 row -> fp32
    const unsigned* w = reinterpret_cast<const unsigned*>(row + DPT * dg);
    if (DPT == 4) {
      const uint2 a = *reinterpret_cast<const uint2*>(w);
      const float2 x = unpack_bf16(a.x), y = unpack_bf16(a.y);
      f[0] = x.x; f[1] = x.y; f[2] = y.x; f[3] = y.y;
    } else {
#pragma unroll
      for (int i = 0; i < DPT / 8; i++) {
        const uint4 a = reinterpret_cast<const uint4*>(w)[i];
        const float2 x = unpack_bf16(a.x), y = unpack_bf16(a.y), z = unpack_bf16(a.z), q = unpack_bf16(a.w);
        f[8 * i] = x.x; f[8 * i + 1] = x.y; f[8 * i + 2] = y.x; f[8 * i + 3] = y.y;
        f[8 * i + 4] = z.x; f[8 * i + 5] = z.y; f[8 * i + 6] = q.x; f[8 * i + 7] = q.y;
      }
    }
  };
  for (int t0 = 0; t0 < len; t0 += C) {
    const int L = min(C, len - t0);
    // loads: K, Q rows (256 B each), V slice rows (RVS*2 B), gates
    for (int idx = tid; idx < C * 16; idx += NT) {
      const int r = idx >> 4, ch = idx & 15;
      const bool valid = r < L;
      const int row = sstart + t0 + (valid ? r : 0);
      cp_async16(smem_u32(Ks + r * 128 + ch * 8), p.k + ((size_t)row * HQ + kh) * D + ch * 8, valid);
      cp_async16(smem_u32(Qs + r * 128 + ch * 8), p.q + ((size_t)row * HQ + kh) * D + ch * 8, valid);
    }
    for (int idx = tid; idx < C * (RVS / 8); idx += NT) {
      const int r = idx / (RVS / 8), ch = idx % (RVS / 8);
      const bool valid = r < L;
      const int row = sstart + t0 + (valid ? r : 0);
      cp_async16(smem_u32(Vs + r * RVS + ch * 8), p.v + ((size_t)row * HV + h) * D + RVS * js + ch * 8, valid);
    }
    cp_async_commit();
    if (tid < C) {
      float g = 1.f, b = 0.f;
      if (tid < L) {
        const int row = sstart + t0 + tid;
        const float av = __bfloat162float(p.a[(size_t)row * HV + h]), bv = __bfloat162float(p.b[(size_t)row * HV + h]);
        g = expf(-alog * softplus_f(av + dtb));
        b = 1.f / (1.f + expf(-bv));
      }
      gs[tid] = g; bs[tid] = b;
    }
    cp_async_wait<0>();
    __syncthreads();
    TL(2);
    // lookahead coefficient tables: thread (row = tid/8 (+NT/8 per pass), part = tid%8 -> dims {8 part + 64 i}), conflict-free
    {
      constexpr int RPP = NT / 8;  // rows per pass
      const int part = tid & 7;
#pragma unroll
      for (int pass = 0; pass < C / RPP; pass++) {
        const int t = (tid >> 3) + pass * RPP;
        float a[16];
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const uint4 x = *reinterpret_cast<const uint4*>(Ks + t * 128 + part * 8 + 64 * i);
          const float2 f0 = unpack_bf16(x.x), f1 = unpack_bf16(x.y), f2 = unpack_bf16(x.z), f3 = unpack_bf16(x.w);
          a[8 * i] = f0.x; a[8 * i + 1] = f0.y; a[8 * i + 2] = f1.x; a[8 * i + 3] = f1.y; a[8 * i + 4] = f2.x; a[8 * i + 5] = f2.y; a[8 * i + 6] = f3.x; a[8 * i + 7] = f3.y;
        }
        float G[3];
#pragma unroll
        for (int d = 1; d <= 3; d++) {
          const int tr = t - d >= 0 ? t - d : t;
          float acc = 0.f;
#pragma unroll
          for (int i = 0; i < 2; i++) {
            const uint4 x = *reinterpret_cast<const uint4*>(Ks + tr * 128 + part * 8 + 64 * i);
            const float2 f0 = unpack_bf16(x.x), f1 = unpack_bf16(x.y), f2 = unpack_bf16(x.z), f3 = unpack_bf16(x.w);
            acc += a[8 * i] * f0.x + a[8 * i + 1] * f0.y + a[8 * i + 2] * f1.x + a[8 * i + 3] * f1.y + a[8 * i + 4] * f2.x + a[8 * i + 5] * f2.y + a[8 * i + 6] * f3.x + a[8 * i + 7] * f3.y;
          }
          G[d - 1] = (t - d >= 0) ? acc : 0.f;
        }
#pragma unroll
        for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
          for (int i = 0; i < 3; i++) G[i] += __shfl_xor_sync(0xffffffffu, G[i], m);
        }
        if (part == 0) {
          const float g1 = t >= 1 ? gs[t - 1] : 1.f, g2 = t >= 2 ? gs[t - 2] : 1.f, g3 = t >= 3 ? gs[t - 3] : 1.f;
          cf4[t] = make_float4(g1 * g2 * g3, G[0], g1 * G[1], g1 * g2 * G[2]);
          gb2[t] = make_float2(gs[t], bs[t]);
        }
      }
    }
    __syncthreads();
    TL(3);
    // prologue: k_0..k_3 into the ring, e_j = k_j . S_{-1} for j = 0..3
    float kr[4][DPT], e[4], u1 = 0.f, u2 = 0.f, u3 = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      load_row(Ks + j * 128, kr[j]);
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < DPT; d++) acc += kr[j][d] * s[d];
      e[j] = acc;
    }
#pragma unroll
    for (int m = 1; m < LPC; m <<= 1) {
#pragma unroll
      for (int j = 0; j < 4; j++) e[j] += __shfl_xor_sync(0xffffffffu, e[j], m);
    }
    const int Lr = (L + 3) & ~3;  // padded rows: k = q = 0, g = 1, b = 0 -> u = 0, S unchanged
    for (int tb = 0; tb < Lr; tb += 4) {
      float op[4], en[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int t = tb + j;
#if (GDN_ABL & 4)
        const float4 cf = make_float4(0.9f, 0.1f, 0.1f, 0.1f);
        const float2 gb = make_float2(0.99f, 0.5f);
        const float vt = (float)t;
#else
        const float4 cf = cf4[t];
        const float2 gb = gb2[t];
        const float vt = __bfloat162float(Vs[t * RVS + vl]);
#endif
        float c = cf.x * e[j] + cf.z * u2 + cf.w * u3;
        c = fmaf(cf.y, u1, c);
        const float u = gb.y * fmaf(-gb.x, c, vt);
        const float g = gb.x;
        float q[DPT], kn[DPT];
#if (GDN_ABL & 2)
#pragma unroll
        for (int d = 0; d < DPT; d++) { q[d] = kr[j][d] * 0.5f; kn[d] = kr[j][d]; }
        if (false) {}
#else
        load_row(Qs + t * 128, q);
        if (t + 4 < C) load_row(Ks + (t + 4) * 128, kn);  // k_{t+4}
#endif
        else {
#pragma unroll
          for (int d = 0; d < DPT; d++) kn[d] = 0.f;
        }
        float o = 0.f, ee = 0.f;
#pragma unroll
        for (int d = 0; d < DPT; d++) {
          s[d] = g * s[d] + kr[j][d] * u;
          o += q[d] * s[d];
          ee += kn[d] * s[d];
        }
#pragma unroll
        for (int d = 0; d < DPT; d++) kr[j][d] = kn[d];
        op[j] = o; en[j] = ee;
        u3 = u2; u2 = u1; u1 = u;
      }
      // o_t = scale q_t . S_t and e_{t+4} = k_{t+4} . S_t for the 4 tokens: 8 values reduced over the LPC lanes of the
      // column by a transpose-reduce (each lane ends with the full sum of value idx = top 3 bits of its lane index).
#if (GDN_ABL & 1)
      {
#pragma unroll
        for (int j = 0; j < 4; j++) { e[j] = en[j]; if (dg == 0) Os[(tb + j) * RVS + vl] = __float2bfloat16(p.scale * op[j]); }
      }
#else
      {
        float val[8];
#pragma unroll
        for (int j = 0; j < 4; j++) { val[j] = op[j]; val[4 + j] = en[j]; }
        constexpr int LOG = (LPC == 32) ? 5 : (LPC == 16 ? 4 : 3);
#pragma unroll
        for (int r = 0; r < 3; r++) {
          const int M = LPC >> (r + 1), NV = 8 >> r;
          const bool hi = (lane & M) != 0;
#pragma unroll
          for (int i = 0; i < NV / 2; i++) {
            const float send = hi ? val[i] : val[i + NV / 2];
            const float keep = hi ? val[i + NV / 2] : val[i];
            val[i] = keep + __shfl_xor_sync(0xffffffffu, send, M);
          }
        }
#pragma unroll
        for (int M = LPC >> 4; M >= 1; M >>= 1) val[0] += __shfl_xor_sync(0xffffffffu, val[0], M);
        const int lp = lane % LPC, idx = lp >> (LOG - 3);
        if (idx < 4 && (lp & ((LPC >> 3) - 1)) == 0) Os[(tb + idx) * RVS + vl] = __float2bfloat16(p.scale * val[0]);
        const int base = lane & ~(LPC - 1);
#pragma unroll
        for (int j = 0; j < 4; j++) e[j] = __shfl_sync(0xffffffffu, val[0], base + ((4 + j) << (LOG - 3)));
      }
#endif
    }
    TL(4);
    __syncthreads();
    // outputs for this block of tokens
    for (int idx = tid; idx < L * (RVS / 8); idx += NT) {
      const int r = idx / (RVS / 8), ch = idx % (RVS / 8);
      *reinterpret_cast<uint4*>(p.out + ((size_t)(sstart + t0 + r) * HV + h) * D + RVS * js + ch * 8) = *reinterpret_cast<const uint4*>(Os + r * RVS + ch * 8);
    }
    __syncthreads();  // smem buffers reused by the next block
    TL(5);
  }
  // final state: S[DPT*dg..][v] -> new_state[v][DPT*dg..]
  {
    float4* dst = reinterpret_cast<float4*>(new_state + (size_t)v * D + DPT * dg);
#pragma unroll
    for (int i = 0; i < DPT / 4; i++) dst[i] = make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
  }
  TL(6);
}

// ------------------------------------------------------------------ tensor-core recurrent path (8-token groups)
// CTA = (seq, v-head h, 8 v-columns); 8 warps, warp w owns dims [16w, 16w+16) of S for the 8 columns as an mma C fragment.
// Per group of 8 tokens (state S = S at group start, gates g_m, m = 0..7 within the group):
//   E = [K_g; Q_g] . S  (16 x 8, mma over the dims split across warps),  G = [K_g; Q_g] . K_g^T  (16 x 8),
//   c_t = gp_t E_k[t] + sum_{i<t} cfK[t][i] u_i,  u_t = beta_t (v_t - g_t c_t)   (8 threads, one per column),
//   o_t = gpq_t E_q[t] + sum_{i<=t} cfQ[t][i] u_i,   S' = Gam S + K_g^T (w_t u_t)  (mma, hi/lo bf16 split of S and u).
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned (&r)[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void split_bf16x2(float x0, float x1, unsigned& hi, unsigned& lo) {
  hi = pack_bf16(x0, x1);
  const float2 h = unpack_bf16(hi);
  lo = pack_bf16(x0 - h.x, x1 - h.y);
}
struct RecurTC {
  static constexpr int SLD = 132;  // padded column stride of the fp32 S copy
  static constexpr int K_ = 0, Q_ = K_ + C * 128 * 2, V_ = Q_ + C * 128 * 2, G_ = V_ + C * 8 * 2, B_ = G_ + C * 4;
  static constexpr int S_ = B_ + C * 4, EP_ = S_ + 8 * SLD * 4, GP_ = EP_ + 8 * 16 * 8 * 4, E_ = GP_ + 8 * 16 * 8 * 4;
  static constexpr int CF_ = E_ + 8 * 16 * 4, U_ = CF_ + 16 * 8 * 4, UP_ = U_ + 8 * 8 * 4, VB_ = UP_ + 8 * 8 * 4, WG_ = VB_ + 8 * 8 * 4;
  static constexpr int PR_ = WG_ + 16 * 4 + 64, VBT_ = PR_ + 8 * 81 * 4, WGT_ = VBT_ + 512 * 4, O_ = WGT_ + 8 * 20 * 4, SMEM = O_ + C * 8 * 2;
};

__global__ void __launch_bounds__(256, 2) gdn_recur_tc_kernel(Params p) {
  using Cfg = RecurTC;
  constexpr int SLD = Cfg::SLD;
  extern __shared__ __align__(16) unsigned char smem[];
  TL(0);
  // K/Q rows in shared memory: 16-B chunk ch of row r lives at chunk (ch ^ (r & 7)) so ldmatrix over 8 consecutive rows is conflict-free
  auto kq_off = [](int r, int ch) -> int { return r * 128 + ((ch ^ (r & 7)) << 3); };
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
  const int js = blockIdx.x % 16, h = (blockIdx.x / 16) % HV, seq = blockIdx.x / (16 * HV), kh = h >> 1;
  const int v0 = 8 * js;
  float* new_state = p.new_state + (size_t)(seq * HV + h) * D * D;  // [v][k]
  // S fragment: s[0] = S[16w+g][2c], s[1] = S[16w+g][2c+1], s[2] = S[16w+g+8][2c], s[3] = S[16w+g+8][2c+1]
  float s[4] = {0.f, 0.f, 0.f, 0.f};
  if (p.state) {
    const float* st = p.state + (size_t)(seq * HV + h) * D * D;
    s[0] = st[(size_t)(v0 + 2 * c) * D + 16 * w + g]; s[1] = st[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g];
    s[2] = st[(size_t)(v0 + 2 * c) * D + 16 * w + g + 8]; s[3] = st[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g + 8];
  }
  const float alog = expf(p.A_log[h]), dtb = p.dt_bias[h];
  __nv_bfloat16* Ks = (__nv_bfloat16*)(smem + Cfg::K_);
  __nv_bfloat16* Qs = (__nv_bfloat16*)(smem + Cfg::Q_);
  __nv_bfloat16* Vs = (__nv_bfloat16*)(smem + Cfg::V_);
  // Issue the K/Q/V/a/b loads for rows r0.. before the cu_seqlens round trip (rows beyond L are zero-filled / zeroed)
  auto issue_loads = [&](int r0, int L) {
    for (int idx = tid; idx < C * 16; idx += 256) {
      const int r = idx >> 4, ch = idx & 15;
      const bool valid = r < L;
      const int row = r0 + (valid ? r : 0);
      cp_async16(smem_u32(Ks + kq_off(r, ch)), p.k + ((size_t)row * HQ + kh) * D + ch * 8, valid);
      cp_async16(smem_u32(Qs + kq_off(r, ch)), p.q + ((size_t)row * HQ + kh) * D + ch * 8, valid);
    }
    if (tid < C) {
      const bool valid = tid < L;
      const int row = r0 + (valid ? tid : 0);
      cp_async16(smem_u32(Vs + tid * 8), p.v + ((size_t)row * HV + h) * D + v0, valid);
    }
    cp_async_commit();
  };
  // speculation for sequence 0: its first block starts at row 0 (cu_seqlens[0] == 0, verified after the cu load)
  const bool spec = (seq == 0);
  const int Lspec = min(C, p.T);
  const int cu0 = (int)p.cu[seq], cu1 = (int)p.cu[seq + 1];  // issued first; consumed after the speculative loads are in flight
  float av = 0.f, bv = 0.f;
  if (spec) {
    issue_loads(0, Lspec);
    if (tid < Lspec) { av = __bfloat162float(p.a[(size_t)tid * HV + h]); bv = __bfloat162float(p.b[(size_t)tid * HV + h]); }
  }
  const int sstart = cu0;
  int len = cu1 - cu0; if (len < 0) len = 0;
  if (len == 0) {
    if (spec) cp_async_wait<0>();
    for (int i = tid; i < 8 * D / 4; i += 256) reinterpret_cast<float4*>(new_state + (size_t)v0 * D)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  float* gs = (float*)(smem + Cfg::G_);
  float* bs = (float*)(smem + Cfg::B_);
  float* Ss = (float*)(smem + Cfg::S_);      // [8 cols][SLD]
  float* Epart = (float*)(smem + Cfg::EP_);  // [8 warps][16][8]
  float* Gpart = (float*)(smem + Cfg::GP_);  // [8 warps][16][8]
  float* Es = (float*)(smem + Cfg::E_);      // [8 cols][16]  (scaled E: rows 0-7 k-part, 8-15 q-part)
  float* CFs = (float*)(smem + Cfg::CF_);    // [16][8]  cfK rows 0-7, cfQ rows 8-15
  float* Us = (float*)(smem + Cfg::U_);      // [8 cols][8 t]
  float* Ups = (float*)(smem + Cfg::UP_);    // [8 t][8 cols]  (w_t u_t)
  float* VBs = (float*)(smem + Cfg::VB_);    // [8 cols][8 t]  (beta_t v_t)
  float* WGs = (float*)(smem + Cfg::WG_);    // [8 t] {beta_t g_t, w_t} + [16] Gam
  float* PRt = (float*)(smem + Cfg::PR_);     // [8 groups][9 a][9 b]: prod_{m=a}^{b-1} g_m (within the group; 1 if b <= a)
  float* VBt = (float*)(smem + Cfg::VBT_);    // [8 grp][8 cols][8 t]: beta_t v_t
  float* WGt = (float*)(smem + Cfg::WGT_);    // [8 grp][20]: {beta_t g_t, w_t} x 8, Gam
  __nv_bfloat16* Os = (__nv_bfloat16*)(smem + Cfg::O_);
  auto store_S = [&]() {
    Ss[(2 * c) * SLD + 16 * w + g] = s[0]; Ss[(2 * c + 1) * SLD + 16 * w + g] = s[1];
    Ss[(2 * c) * SLD + 16 * w + g + 8] = s[2]; Ss[(2 * c + 1) * SLD + 16 * w + g + 8] = s[3];
  };
  store_S();
  __syncwarp();
  TL(1);
  for (int t0 = 0; t0 < len; t0 += C) {
    const int L = min(C, len - t0);
    const bool use_spec = spec && t0 == 0 && sstart == 0;  // rows [0, L) are already in flight (L <= Lspec)
    if (!use_spec) {
      if (spec && t0 == 0) cp_async_wait<0>();  // mis-speculated (cu_seqlens[0] != 0): drain, then load the right rows
      issue_loads(sstart + t0, L);
      if (tid < L) { const int row = sstart + t0 + tid; av = __bfloat162float(p.a[(size_t)row * HV + h]); bv = __bfloat162float(p.b[(size_t)row * HV + h]); }
    }
    if (tid < C) {
      float gg = 1.f, b = 0.f;
      if (tid < L) {
        gg = expf(-alog * softplus_f(av + dtb));
        b = 1.f / (1.f + expf(-bv));
      }
      gs[tid] = gg; bs[tid] = b;
    }
    __syncthreads();  // gates visible
    if (tid < 72) {  // thread (grp, a): running products prod_{m=a}^{b-1} g_m for b = 0..8, and the per-group gate tables
      const int grp = tid / 9, a = tid - grp * 9;
      float gg[8];
      {
        const float4* g4 = reinterpret_cast<const float4*>(gs + 8 * grp);
        const float4 x0 = g4[0], x1 = g4[1];
        gg[0] = x0.x; gg[1] = x0.y; gg[2] = x0.z; gg[3] = x0.w; gg[4] = x1.x; gg[5] = x1.y; gg[6] = x1.z; gg[7] = x1.w;
      }
      float* row = PRt + grp * 81 + a * 9;
      float pr = 1.f;
#pragma unroll
      for (int b = 0; b < 9; b++) {
        row[b] = pr;
        if (b < 8) pr *= (b >= a) ? gg[b] : 1.f;
      }
      if (a < 8) {
        float wt = 1.f;
#pragma unroll
        for (int m = 0; m < 8; m++) wt *= (m > a) ? gg[m] : 1.f;
        WGt[grp * 20 + 2 * a] = bs[8 * grp + a] * gg[a]; WGt[grp * 20 + 2 * a + 1] = wt;
      } else {
        float G = 1.f;
#pragma unroll
        for (int m = 0; m < 8; m++) G *= gg[m];
        WGt[grp * 20 + 16] = G;
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    if (use_spec && L < Lspec) {  // speculatively loaded rows of the next sequence -> zero (padded tokens: k = q = v = 0)
      for (int idx = tid; idx < (Lspec - L) * 33; idx += 256) {
        const int r = L + idx / 33, j = idx % 33;
        uint4* dst = (j < 16) ? reinterpret_cast<uint4*>(Ks + r * 128 + j * 8) : (j < 32 ? reinterpret_cast<uint4*>(Qs + r * 128 + (j - 16) * 8) : reinterpret_cast<uint4*>(Vs + r * 8));
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
      __syncthreads();
    }
    for (int idx = tid; idx < 512; idx += 256) {  // VBt[grp][col][t] = beta_t v_t
      const int grp = idx >> 6, col = (idx >> 3) & 7, t = idx & 7, tok = 8 * grp + t;
      VBt[idx] = bs[tok] * __bfloat162float(Vs[tok * 8 + col]);
    }
    TL(2);
    const int ngroups = (L + 7) >> 3;
    long long tacc[4] = {0, 0, 0, 0};
    for (int grp = 0; grp < ngroups; grp++) {
      const int tb = 8 * grp;
      long long tp0 = p.dbg ? clock64() : 0;
      // ---- phase A: E and G partials (this warp's 16-dim slice)
      {
        unsigned a[4];
        {
          const int r = lane & 7, half = (lane >> 3) & 1;
          ldsm_x4(a, (half ? Qs : Ks) + kq_off(tb + r, 2 * w + (lane >> 4)));
        }
        const float2 s01 = *reinterpret_cast<const float2*>(Ss + g * SLD + 16 * w + 2 * c);
        const float2 s89 = *reinterpret_cast<const float2*>(Ss + g * SLD + 16 * w + 8 + 2 * c);
        unsigned b0h, b0l, b1h, b1l;
        split_bf16x2(s01.x, s01.y, b0h, b0l);
        split_bf16x2(s89.x, s89.y, b1h, b1l);
        float ce[4] = {0.f, 0.f, 0.f, 0.f};
        mma_bf16(ce, a, b0h, b1h);
        mma_bf16(ce, a, b0l, b1l);
        unsigned bk[2];
        ldsm_x2(bk, smem_u32(Ks + kq_off(tb + (lane & 7), 2 * w + ((lane >> 3) & 1))));
        float cg[4] = {0.f, 0.f, 0.f, 0.f};
        mma_bf16(cg, a, bk[0], bk[1]);
        float* ep = Epart + w * 128;
        float* gp = Gpart + w * 128;
        *reinterpret_cast<float2*>(ep + g * 8 + 2 * c) = make_float2(ce[0], ce[1]);
        *reinterpret_cast<float2*>(ep + (g + 8) * 8 + 2 * c) = make_float2(ce[2], ce[3]);
        *reinterpret_cast<float2*>(gp + g * 8 + 2 * c) = make_float2(cg[0], cg[1]);
        *reinterpret_cast<float2*>(gp + (g + 8) * 8 + 2 * c) = make_float2(cg[2], cg[3]);
      }
      __syncthreads();
      if (p.dbg) { const long long t = clock64(); tacc[0] += t - tp0; tp0 = t; }
      // ---- phase B1: all threads reduce the partials (thread (r, i)); gate products from the PR table
      {
        const float* PR = PRt + grp * 81;  // PR[a*9+b] = prod_{m=a}^{b-1} g_m
        const int r = tid >> 3, i = tid & 7;
        if (r < 16) {
          float e = 0.f;
#pragma unroll
          for (int ww = 0; ww < 8; ww++) e += Epart[ww * 128 + r * 8 + i];
          Es[i * 16 + r] = e * ((r < 8) ? PR[r] : PR[r - 7]);
        } else {
          const int rr = r - 16;
          float G = 0.f;
#pragma unroll
          for (int ww = 0; ww < 8; ww++) G += Gpart[ww * 128 + rr * 8 + i];
          float cf;
          if (rr < 8) cf = (i < rr) ? G * PR[(i + 1) * 9 + rr] : 0.f;
          else cf = (i <= rr - 8) ? G * PR[(i + 1) * 9 + rr - 7] : 0.f;
          CFs[rr * 8 + i] = cf;
        }
      }
      // solve inputs that do not depend on this group's E/G (preloaded before the barrier)
      float vb[8], wg[16];
      if (tid < 8) {
        const float4* v4 = reinterpret_cast<const float4*>(VBt + grp * 64 + tid * 8);
        const float4 x0 = v4[0], x1 = v4[1];
        vb[0] = x0.x; vb[1] = x0.y; vb[2] = x0.z; vb[3] = x0.w; vb[4] = x1.x; vb[5] = x1.y; vb[6] = x1.z; vb[7] = x1.w;
        const float4* w4 = reinterpret_cast<const float4*>(WGt + grp * 20);
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) { const float4 x = w4[q4]; wg[4 * q4] = x.x; wg[4 * q4 + 1] = x.y; wg[4 * q4 + 2] = x.z; wg[4 * q4 + 3] = x.w; }
      }
      __syncthreads();
      if (p.dbg) { const long long t = clock64(); tacc[1] += t - tp0; tp0 = t; }
      // ---- phase B2: per-column forward substitution (8 threads)
      if (tid < 8) {
        const int col = tid;
        float acc[8], cf[8][8];
        {
          const float4* e4 = reinterpret_cast<const float4*>(Es + col * 16);
          const float4 e0 = e4[0], e1 = e4[1];
          acc[0] = e0.x; acc[1] = e0.y; acc[2] = e0.z; acc[3] = e0.w; acc[4] = e1.x; acc[5] = e1.y; acc[6] = e1.z; acc[7] = e1.w;
#pragma unroll
          for (int t = 1; t < 8; t++) {
            const float4* c4 = reinterpret_cast<const float4*>(CFs + t * 8);
            const float4 y0 = c4[0], y1 = c4[1];
            cf[t][0] = y0.x; cf[t][1] = y0.y; cf[t][2] = y0.z; cf[t][3] = y0.w; cf[t][4] = y1.x; cf[t][5] = y1.y; cf[t][6] = y1.z; cf[t][7] = y1.w;
          }
        }
#pragma unroll
        for (int t = 0; t < 8; t++) {
          const float u = fmaf(-wg[2 * t], acc[t], vb[t]);  // beta (v - g c)
          Us[col * 8 + t] = u;
          Ups[t * 8 + col] = u * wg[2 * t + 1];
#pragma unroll
          for (int t2 = t + 1; t2 < 8; t2++) acc[t2] = fmaf(cf[t2][t], u, acc[t2]);
        }
      }
      __syncthreads();
      if (p.dbg) { const long long t = clock64(); tacc[2] += t - tp0; tp0 = t; }
      // ---- phase C: outputs (64 threads) and the rank-8 state update (all warps)
      if (tid < 64) {
        const int t = tid >> 3, col = tid & 7;
        float o = Es[col * 16 + 8 + t];
        const float4* c4 = reinterpret_cast<const float4*>(CFs + (8 + t) * 8);
        const float4* u4 = reinterpret_cast<const float4*>(Us + col * 8);
        const float4 y0 = c4[0], y1 = c4[1], x0 = u4[0], x1 = u4[1];  // cfQ[t][i] = 0 for i > t
        o = fmaf(y0.x, x0.x, o); o = fmaf(y0.y, x0.y, o); o = fmaf(y0.z, x0.z, o); o = fmaf(y0.w, x0.w, o);
        o = fmaf(y1.x, x1.x, o); o = fmaf(y1.y, x1.y, o); o = fmaf(y1.z, x1.z, o); o = fmaf(y1.w, x1.w, o);
        Os[(tb + t) * 8 + col] = __float2bfloat16(p.scale * o);
      }
      {
        const float G = WGt[grp * 20 + 16];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i] *= G;
        unsigned ak2[2];
        ldsm_x2_t(ak2, smem_u32(Ks + kq_off(tb + (lane & 7), 2 * w + ((lane >> 3) & 1))));
        const unsigned ak[4] = {ak2[0], ak2[1], 0u, 0u};
        const float u0 = Ups[(2 * c) * 8 + g], u1 = Ups[(2 * c + 1) * 8 + g];
        unsigned bh, bl;
        split_bf16x2(u0, u1, bh, bl);
        mma_bf16(s, ak, bh, 0u);
        mma_bf16(s, ak, bl, 0u);
        store_S();
      }
      __syncwarp();  // the next group's E operand (this warp's S slice) comes from this warp's own store
      if (p.dbg) { const long long t = clock64(); tacc[3] += t - tp0; tp0 = t; }
    }
    __syncthreads();
    if (p.dbg && threadIdx.x == 0) { for (int i = 0; i < 4; i++) p.dbg[(size_t)blockIdx.x * 32 + 10 + i] = tacc[i]; }
    TL(3);
    // outputs for this block of tokens (16 B per row)
    if (tid < L) *reinterpret_cast<uint4*>(p.out + ((size_t)(sstart + t0 + tid) * HV + h) * D + v0) = *reinterpret_cast<const uint4*>(Os + tid * 8);
    __syncthreads();  // smem buffers reused by the next block
    TL(4);
  }
  new_state[(size_t)(v0 + 2 * c) * D + 16 * w + g] = s[0]; new_state[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g] = s[1];
  new_state[(size_t)(v0 + 2 * c) * D + 16 * w + g + 8] = s[2]; new_state[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g + 8] = s[3];
  TL(5);
}

// ------------------------------------------------------------------ tensor-core recurrent path, pipelined (9 warps)
// Warps 0-7 ("updaters") own dims [16w, 16w+16) of S for the CTA's 8 columns (mma C fragment); warp 8 ("solver") runs the
// per-column 8-token forward substitution, which is the only serial dependency between groups.
// Group g (tokens tb..tb+7 of the 64-token block), S_g = state at the group start:
//   E_g = [K_g; Q_g] . S_g = Gam_{g-1} E~_g + Gram'_{g-1} U'_{g-1},   E~_g = [K_g; Q_g] . S_{g-1}  (computed one group early)
//   Gram'_g = [K_{g+1}; Q_{g+1}] . K_g^T,  G_g = [K_g; Q_g] . K_g^T  (all precomputed per block with mma)
//   c_t = gp_t E_g[t] + sum_{i<t} cfK u_i,  u_t = beta_t (v_t - g_t c_t),  U'_t = w_t u_t,  o_t = gpq_t E_g[8+t] + sum_{i<=t} cfQ u_i
//   S_{g+1} = Gam_g S_g + K_g^T U'_g   (mma, bf16 hi/lo splits of S and U').
struct RecurTC2 {
  static constexpr int NT = 288, SLD = 132;
  static constexpr int K_ = 0, Q_ = K_ + C * 128 * 2, V_ = Q_ + C * 128 * 2, G_ = V_ + C * 8 * 2, B_ = G_ + C * 4;
  static constexpr int S_ = B_ + C * 4, EP_ = S_ + 8 * SLD * 4, CF_ = EP_ + 2 * 8 * 128 * 4, GP_ = CF_ + 8 * 128 * 4;
  static constexpr int PR_ = GP_ + 8 * 128 * 4, VB_ = PR_ + 8 * 81 * 4, WG_ = VB_ + 512 * 4, E_ = WG_ + 8 * 20 * 4;
  static constexpr int U_ = E_ + 2 * 128 * 4, UP_ = U_ + 64 * 4, O_ = UP_ + 2 * 64 * 4, SMEM = O_ + C * 8 * 2;
};

__global__ void __launch_bounds__(288, 2) gdn_recur_tc2_kernel(Params p) {
  using Cfg = RecurTC2;
  constexpr int SLD = Cfg::SLD, NT = Cfg::NT;
  extern __shared__ __align__(16) unsigned char smem[];
  TL(0);
  auto kq_off = [](int r, int ch) -> int { return r * 128 + ((ch ^ (r & 7)) << 3); };
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, c = lane & 3;
  const int js = blockIdx.x % 16, h = (blockIdx.x / 16) % HV, seq = blockIdx.x / (16 * HV), kh = h >> 1;
  const int v0 = 8 * js;
  const bool upd = w < 8;
  float* new_state = p.new_state + (size_t)(seq * HV + h) * D * D;  // [v][k]
  float s[4] = {0.f, 0.f, 0.f, 0.f};
  if (p.state && upd) {
    const float* st = p.state + (size_t)(seq * HV + h) * D * D;
    s[0] = st[(size_t)(v0 + 2 * c) * D + 16 * w + g]; s[1] = st[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g];
    s[2] = st[(size_t)(v0 + 2 * c) * D + 16 * w + g + 8]; s[3] = st[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g + 8];
  }
  const float alog = expf(p.A_log[h]), dtb = p.dt_bias[h];
  const int sstart = (int)p.cu[seq];
  int len = (int)(p.cu[seq + 1] - p.cu[seq]); if (len < 0) len = 0;
  if (len == 0) {
    for (int i = tid; i < 8 * D / 4; i += NT) reinterpret_cast<float4*>(new_state + (size_t)v0 * D)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  __nv_bfloat16* Ks = (__nv_bfloat16*)(smem + Cfg::K_);
  __nv_bfloat16* Qs = (__nv_bfloat16*)(smem + Cfg::Q_);
  __nv_bfloat16* Vs = (__nv_bfloat16*)(smem + Cfg::V_);
  float* gs = (float*)(smem + Cfg::G_);
  float* bs = (float*)(smem + Cfg::B_);
  float* Ss = (float*)(smem + Cfg::S_);      // [8 cols][SLD]
  float* Epart = (float*)(smem + Cfg::EP_);  // [2 parity][8 warps][16][8]
  float* CFt = (float*)(smem + Cfg::CF_);    // [8 grp][16][8]: cfK rows 0-7, cfQ rows 8-15 (gate products applied)
  float* GPt = (float*)(smem + Cfg::GP_);    // [8 grp][16][8]: Gram'_g
  float* PRt = (float*)(smem + Cfg::PR_);    // [8 grp][9][9]: prod_{m=a}^{b-1} g_m
  float* VBt = (float*)(smem + Cfg::VB_);    // [8 grp][8 cols][8 t]: beta_t v_t
  float* WGt = (float*)(smem + Cfg::WG_);    // [8 grp][20]: {beta_t g_t, w_t} x 8, Gam (stride 20 keeps float4 alignment)
  float* Es = (float*)(smem + Cfg::E_);      // [2 parity][8 cols][16]
  float* Us = (float*)(smem + Cfg::U_);      // [8 cols][8 t]
  float* Ups = (float*)(smem + Cfg::UP_);    // [2 parity][8 t][8 cols]
  __nv_bfloat16* Os = (__nv_bfloat16*)(smem + Cfg::O_);
  auto store_S = [&]() {
    Ss[(2 * c) * SLD + 16 * w + g] = s[0]; Ss[(2 * c + 1) * SLD + 16 * w + g] = s[1];
    Ss[(2 * c) * SLD + 16 * w + g + 8] = s[2]; Ss[(2 * c + 1) * SLD + 16 * w + g + 8] = s[3];
  };
  // E~ partials for group ag against the current Ss (this warp's dim slice) -> Epart[par][w]
  auto e_partials = [&](int ag, int par) {
    const int tb = 8 * ag;
    unsigned a[4];
    {
      const int r = lane & 7, half = (lane >> 3) & 1;
      ldsm_x4(a, (half ? Qs : Ks) + kq_off(tb + r, 2 * w + (lane >> 4)));
    }
    const float2 s01 = *reinterpret_cast<const float2*>(Ss + g * SLD + 16 * w + 2 * c);
    const float2 s89 = *reinterpret_cast<const float2*>(Ss + g * SLD + 16 * w + 8 + 2 * c);
    unsigned b0h, b0l, b1h, b1l;
    split_bf16x2(s01.x, s01.y, b0h, b0l);
    split_bf16x2(s89.x, s89.y, b1h, b1l);
    float ce[4] = {0.f, 0.f, 0.f, 0.f};
    mma_bf16(ce, a, b0h, b1h);
    mma_bf16(ce, a, b0l, b1l);
    float* ep = Epart + (par * 8 + w) * 128;
    *reinterpret_cast<float2*>(ep + g * 8 + 2 * c) = make_float2(ce[0], ce[1]);
    *reinterpret_cast<float2*>(ep + (g + 8) * 8 + 2 * c) = make_float2(ce[2], ce[3]);
  };
  if (upd) { store_S(); __syncwarp(); }
  TL(1);
  for (int t0 = 0; t0 < len; t0 += C) {
    const int L = min(C, len - t0);
    for (int idx = tid; idx < C * 16; idx += NT) {
      const int r = idx >> 4, ch = idx & 15;
      const bool valid = r < L;
      const int row = sstart + t0 + (valid ? r : 0);
      cp_async16(smem_u32(Ks + kq_off(r, ch)), p.k + ((size_t)row * HQ + kh) * D + ch * 8, valid);
      cp_async16(smem_u32(Qs + kq_off(r, ch)), p.q + ((size_t)row * HQ + kh) * D + ch * 8, valid);
    }
    if (tid < C) {
      const bool valid = tid < L;
      const int row = sstart + t0 + (valid ? tid : 0);
      cp_async16(smem_u32(Vs + tid * 8), p.v + ((size_t)row * HV + h) * D + v0, valid);
    }
    cp_async_commit();
    if (tid < C) {
      float gg = 1.f, b = 0.f;
      if (tid < L) {
        const int row = sstart + t0 + tid;
        const float av = __bfloat162float(p.a[(size_t)row * HV + h]), bv = __bfloat162float(p.b[(size_t)row * HV + h]);
        gg = expf(-alog * softplus_f(av + dtb));
        b = 1.f / (1.f + expf(-bv));
      }
      gs[tid] = gg; bs[tid] = b;
    }
    __syncthreads();  // gates visible
    const int ngroups = (L + 7) >> 3;
    if (tid < 72) {  // PR rows and the per-group gate tables: thread (grp, a)
      const int grp = tid / 9, a = tid - grp * 9;
      float gg[8];
      {
        const float4* g4 = reinterpret_cast<const float4*>(gs + 8 * grp);
        const float4 x0 = g4[0], x1 = g4[1];
        gg[0] = x0.x; gg[1] = x0.y; gg[2] = x0.z; gg[3] = x0.w; gg[4] = x1.x; gg[5] = x1.y; gg[6] = x1.z; gg[7] = x1.w;
      }
      float* row = PRt + grp * 81 + a * 9;
      float pr = 1.f;
#pragma unroll
      for (int b = 0; b < 9; b++) {
        row[b] = pr;
        if (b < 8) pr *= (b >= a) ? gg[b] : 1.f;
      }
      if (a < 8) {
        float wt = 1.f;
#pragma unroll
        for (int m = 0; m < 8; m++) wt *= (m > a) ? gg[m] : 1.f;
        WGt[grp * 20 + 2 * a] = bs[8 * grp + a] * gg[a]; WGt[grp * 20 + 2 * a + 1] = wt;
      } else {
        float G = 1.f;
#pragma unroll
        for (int m = 0; m < 8; m++) G *= gg[m];
        WGt[grp * 20 + 16] = G;
      }
    }
    cp_async_wait<0>();
    __syncthreads();  // K/Q/V landed
    for (int idx = tid; idx < 512; idx += NT) {  // VBt[grp][col][t] = beta_t v_t
      const int grp = idx >> 6, col = (idx >> 3) & 7, t = idx & 7, tok = 8 * grp + t;
      VBt[idx] = bs[tok] * __bfloat162float(Vs[tok * 8 + col]);
    }
    // Gram tiles with mma (full 128-dim contraction per warp): tile tI: even -> G_{tI/2} (cfK/cfQ), odd -> Gram'_{tI/2}
    for (int tI = w; tI < 2 * ngroups - 1; tI += 9) {
      const int grp = tI >> 1, prime = tI & 1, ag = grp + prime;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < 8; ks++) {
        unsigned a[4], b[2];
        const int r = lane & 7, half = (lane >> 3) & 1;
        ldsm_x4(a, (half ? Qs : Ks) + kq_off(8 * ag + r, 2 * ks + (lane >> 4)));
        ldsm_x2(b, smem_u32(Ks + kq_off(8 * grp + r, 2 * ks + half)));
        mma_bf16(acc, a, b[0], b[1]);
      }
      // acc: rows g (k-part t=g) and g+8 (q-part t=g), cols 2c, 2c+1 (= token index i)
      if (!prime) {
        const float* PR = PRt + grp * 81;
        const int t = g, i0 = 2 * c, i1 = 2 * c + 1;
        const float k0 = (i0 < t) ? acc[0] * PR[(i0 + 1) * 9 + t] : 0.f, k1 = (i1 < t) ? acc[1] * PR[(i1 + 1) * 9 + t] : 0.f;
        const float q0 = (i0 <= t) ? acc[2] * PR[(i0 + 1) * 9 + t + 1] : 0.f, q1 = (i1 <= t) ? acc[3] * PR[(i1 + 1) * 9 + t + 1] : 0.f;
        *reinterpret_cast<float2*>(CFt + grp * 128 + t * 8 + i0) = make_float2(k0, k1);
        *reinterpret_cast<float2*>(CFt + grp * 128 + (8 + t) * 8 + i0) = make_float2(q0, q1);
      } else {
        *reinterpret_cast<float2*>(GPt + grp * 128 + g * 8 + 2 * c) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(GPt + grp * 128 + (g + 8) * 8 + 2 * c) = make_float2(acc[2], acc[3]);
      }
    }
    if (upd) e_partials(0, 0);  // E~_0 = [K_0; Q_0] . S_0
    __syncthreads();
    // E assembly for group ag (warps 2-5, one (row r, col i) per lane): E = Gam E~ + Gram' U' (no fixup for ag == 0),
    // then the gate scaling gp_t / gpq_t; written to Es[par]
    auto e_assemble = [&](int ag, int par) {
      const int idx = tid - 64, r = idx >> 3, i = idx & 7;
      float e = 0.f;
#pragma unroll
      for (int ww = 0; ww < 8; ww++) e += Epart[(par * 8 + ww) * 128 + r * 8 + i];
      if (ag > 0) {
        const int pg = ag - 1;
        e *= WGt[pg * 20 + 16];
        const float* gpr = GPt + pg * 128 + r * 8;
        const float* up = Ups + (pg & 1) * 64;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) e = fmaf(gpr[kk], up[kk * 8 + i], e);
      }
      const float* PR = PRt + ag * 81;
      e *= (r < 8) ? PR[r] : PR[r - 7];
      Es[par * 128 + i * 16 + r] = e;
    };
    if (w >= 2 && w < 6) e_assemble(0, 0);
    __syncthreads();
    TL(2);
    long long tacc[4] = {0, 0, 0, 0};
    for (int grp = 0; grp < ngroups; grp++) {
      const int tb = 8 * grp, par = grp & 1;
      long long tp0 = p.dbg ? clock64() : 0;
      // ---- phase 1: solver solves group grp; updaters compute E~_{grp+1} partials against S_grp
      if (upd) {
        if (grp + 1 < ngroups) e_partials(grp + 1, par ^ 1);
      } else if (lane < 8) {
        const int col = lane;
        float acc[8], vb[8], wg[16], cf[8][8];
        {
          const float4* e4 = reinterpret_cast<const float4*>(Es + par * 128 + col * 16);
          const float4 e0 = e4[0], e1 = e4[1];
          acc[0] = e0.x; acc[1] = e0.y; acc[2] = e0.z; acc[3] = e0.w; acc[4] = e1.x; acc[5] = e1.y; acc[6] = e1.z; acc[7] = e1.w;
          const float4* v4 = reinterpret_cast<const float4*>(VBt + grp * 64 + col * 8);
          const float4 x0 = v4[0], x1 = v4[1];
          vb[0] = x0.x; vb[1] = x0.y; vb[2] = x0.z; vb[3] = x0.w; vb[4] = x1.x; vb[5] = x1.y; vb[6] = x1.z; vb[7] = x1.w;
          const float4* w4 = reinterpret_cast<const float4*>(WGt + grp * 20);
#pragma unroll
          for (int q4 = 0; q4 < 4; q4++) { const float4 x = w4[q4]; wg[4 * q4] = x.x; wg[4 * q4 + 1] = x.y; wg[4 * q4 + 2] = x.z; wg[4 * q4 + 3] = x.w; }
#pragma unroll
          for (int t2 = 1; t2 < 8; t2++) {
            const float4* c4 = reinterpret_cast<const float4*>(CFt + grp * 128 + t2 * 8);
            const float4 y0 = c4[0], y1 = c4[1];
            cf[t2][0] = y0.x; cf[t2][1] = y0.y; cf[t2][2] = y0.z; cf[t2][3] = y0.w; cf[t2][4] = y1.x; cf[t2][5] = y1.y; cf[t2][6] = y1.z; cf[t2][7] = y1.w;
          }
        }
        float* upw = Ups + par * 64;
#pragma unroll
        for (int t2 = 0; t2 < 8; t2++) {
          const float u = fmaf(-wg[2 * t2], acc[t2], vb[t2]);  // beta (v - g c)
          Us[col * 8 + t2] = u;
          upw[t2 * 8 + col] = u * wg[2 * t2 + 1];
#pragma unroll
          for (int t3 = t2 + 1; t3 < 8; t3++) acc[t3] = fmaf(cf[t3][t2], u, acc[t3]);
        }
      }
      __syncthreads();
      if (p.dbg) { const long long tt = clock64(); tacc[0] += tt - tp0; tp0 = tt; }
      // ---- phase 2: updaters: S_{grp+1} = Gam S_grp + K_grp^T U'_grp; warps 0-1: outputs; warps 2-5: E_{grp+1} assembly
      if (upd) {
        const float Gm = WGt[grp * 20 + 16];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i] *= Gm;
        unsigned ak2[2];
        ldsm_x2_t(ak2, smem_u32(Ks + kq_off(tb + (lane & 7), 2 * w + ((lane >> 3) & 1))));
        const unsigned ak[4] = {ak2[0], ak2[1], 0u, 0u};
        const float* upr = Ups + par * 64;
        const float u0 = upr[(2 * c) * 8 + g], u1 = upr[(2 * c + 1) * 8 + g];
        unsigned bh, bl;
        split_bf16x2(u0, u1, bh, bl);
        mma_bf16(s, ak, bh, 0u);
        mma_bf16(s, ak, bl, 0u);
        store_S();
        __syncwarp();
        if (w < 2) {
          // outputs of group grp: lane -> (t = idx>>3, col = idx&7)
          const int t = tid >> 3, col = tid & 7;
          const float4* c4 = reinterpret_cast<const float4*>(CFt + grp * 128 + (8 + t) * 8);
          const float4* u4 = reinterpret_cast<const float4*>(Us + col * 8);
          const float4 y0 = c4[0], y1 = c4[1], x0 = u4[0], x1 = u4[1];
          float v = Es[par * 128 + col * 16 + 8 + t];
          v = fmaf(y0.x, x0.x, v); v = fmaf(y0.y, x0.y, v); v = fmaf(y0.z, x0.z, v); v = fmaf(y0.w, x0.w, v);
          v = fmaf(y1.x, x1.x, v); v = fmaf(y1.y, x1.y, v); v = fmaf(y1.z, x1.z, v); v = fmaf(y1.w, x1.w, v);
          Os[(tb + t) * 8 + col] = __float2bfloat16(p.scale * v);
        } else if (w < 6) {
          if (grp + 1 < ngroups) e_assemble(grp + 1, par ^ 1);
        }
      }
      __syncthreads();
      if (p.dbg) { const long long tt = clock64(); tacc[1] += tt - tp0; }
    }
    __syncthreads();
    if (p.dbg && threadIdx.x == DBG_TID) { for (int i = 0; i < 4; i++) p.dbg[(size_t)blockIdx.x * 32 + 10 + i] = tacc[i]; }
    TL(3);
    if (tid < L) *reinterpret_cast<uint4*>(p.out + ((size_t)(sstart + t0 + tid) * HV + h) * D + v0) = *reinterpret_cast<const uint4*>(Os + tid * 8);
    __syncthreads();  // smem buffers reused by the next block
    TL(4);
  }
  if (upd) {
    new_state[(size_t)(v0 + 2 * c) * D + 16 * w + g] = s[0]; new_state[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g] = s[1];
    new_state[(size_t)(v0 + 2 * c) * D + 16 * w + g + 8] = s[2]; new_state[(size_t)(v0 + 2 * c + 1) * D + 16 * w + g + 8] = s[3];
  }
  TL(5);
}

template <int DPT, int NT>
static void launch_recur(const Params& p, cudaStream_t stream) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(gdn_recur_kernel<DPT, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, RecurCfg<DPT, NT>::SMEM); attr = true; }
  gdn_recur_kernel<DPT, NT><<<p.N * HV * (D / RecurCfg<DPT, NT>::RVS), NT, RecurCfg<DPT, NT>::SMEM, stream>>>(p);
}

__global__ void __launch_bounds__(NTHREADS, 1) gdn_fused_kernel(Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  TL(0);
  if (p.dbg && threadIdx.x == 0) { unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 28] = (int64_t)gt; }
  fused_role(p, blockIdx.x / HV, blockIdx.x % HV, smem);
  TL(30);
  if (p.dbg && threadIdx.x == 0) { p.dbg[(size_t)blockIdx.x * 32 + 31] = (int64_t)blockIdx.x; unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 29] = (int64_t)gt; }
}
template <int VS, int NSTAGES>
__global__ void __launch_bounds__(NTHREADS, 1) gdn_chain_kernel(Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  Params q = p; if (q.dbg) q.dbg += (size_t)p.num_prep * 32;  // debug rows after the prep rows
  { const Params& p = q; TL(0);
    if (p.dbg && threadIdx.x == 0) { unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 28] = (int64_t)gt; } }
  chain_role<VS, NSTAGES>(q, blockIdx.x, smem);
  { const Params& p = q; TL(30);
    if (p.dbg && threadIdx.x == 0) { p.dbg[(size_t)blockIdx.x * 32 + 31] = (int64_t)blockIdx.x; unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 29] = (int64_t)gt; } }
}
template <int VS, int NSTAGES>
__global__ void __launch_bounds__(NTHREADS, 1) gdn_kernel(Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ unsigned s_ticket;
  TL(0);
  if (p.dbg && threadIdx.x == 0) { unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 28] = (int64_t)gt; }
  if (threadIdx.x == 0) s_ticket = atomicAdd(&p.counters[0], 1u);
  __syncthreads();
  const unsigned t = s_ticket;
  if (t < (unsigned)p.num_prep) { if (!(p.debug & 2)) prep_role(p, t / HV, t % HV, smem); }
  else chain_role<VS, NSTAGES>(p, t - p.num_prep, smem);
  __syncthreads();
  TL(30);
  if (p.dbg && threadIdx.x == 0) { p.dbg[(size_t)blockIdx.x * 32 + 31] = (int64_t)s_ticket; unsigned long long gt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt)); p.dbg[(size_t)blockIdx.x * 32 + 29] = (int64_t)gt; }
  if (threadIdx.x == 0) {
    const unsigned d = atomicAdd(&p.counters[1], 1u);
    if (d == gridDim.x - 1) { p.counters[0] = 0; p.counters[1] = 0; __threadfence(); }
  }
}

// ------------------------------------------------------------------ host side
struct Workspace {
  void* buf = nullptr; size_t bytes = 0; int max_chunks = 0;
  ChunkRec* recs = nullptr; int* flags = nullptr; unsigned* counters = nullptr;
};
static Workspace g_ws;
static unsigned g_epoch = 0;
static torch::Tensor g_dbg;
torch::Tensor get_dbg() { return g_dbg; }

static void ensure_workspace(int max_chunks, cudaStream_t stream) {
  if (max_chunks <= g_ws.max_chunks) return;
  const int mc = std::max(max_chunks, 256);
  const size_t rec_bytes = (size_t)mc * HV * sizeof(ChunkRec);
  const size_t flag_bytes = (size_t)mc * HV * sizeof(int);
  const size_t total = rec_bytes + flag_bytes + 256;
  if (g_ws.buf) { cudaStreamSynchronize(stream); cudaFree(g_ws.buf); }
  cudaMalloc(&g_ws.buf, total);
  cudaMemset(g_ws.buf, 0, total);
  g_ws.recs = (ChunkRec*)g_ws.buf;
  g_ws.flags = (int*)((char*)g_ws.buf + rec_bytes);
  g_ws.counters = (unsigned*)((char*)g_ws.buf + rec_bytes + flag_bytes);
  g_ws.max_chunks = mc; g_ws.bytes = total;
  g_epoch = 0;
}

template <int VS, int NSTAGES>
static void launch(const Params& p, int grid, cudaStream_t stream) {
  const int smem = std::max(PREP_SMEM, ChainSmem<VS, NSTAGES>::BYTES);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(gdn_kernel<VS, NSTAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(gdn_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * PREP_SMEM_ALIGNED);
    cudaFuncSetAttribute(gdn_prep_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(gdn_kernel<VS, NSTAGES>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(gdn_chain_kernel<VS, NSTAGES>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(gdn_chain_kernel<VS, NSTAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, ChainSmem<VS, NSTAGES>::BYTES);
    attr_set = true;
    if (getenv("GDN_OCC")) {
      int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, gdn_prep_kernel, 2 * NTHREADS, 2 * PREP_SMEM_ALIGNED);
      int nc = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nc, gdn_chain_kernel<VS, NSTAGES>, NTHREADS, ChainSmem<VS, NSTAGES>::BYTES);
      cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, gdn_prep_kernel);
      printf("occupancy: prep %d blocks/SM (smem %d B, regs %d, local %zu, static %zu, maxdyn %d), chain<%d> %d blocks/SM (smem %d)\n", nb, PREP_SMEM, fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes, VS, nc, ChainSmem<VS, NSTAGES>::BYTES);
    }
  }
  static bool fattr = false;
  if (!fattr) { cudaFuncSetAttribute(gdn_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FUSED_SMEM); cudaFuncSetAttribute(gdn_fused_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); fattr = true; }
  // recurrent path (tensor-core 8-token groups) when sequences are short: T <= 60 N + 80 (measured crossover vs fused/chunked)
  const char* er = getenv("GDN_RECUR_AVG");
  const int recur_avg = er ? atoi(er) : 60;
  const char* eo = getenv("GDN_RECUR_OFF");
  const int recur_off = eo ? atoi(eo) : 80;
  if (p.T <= recur_avg * p.N + recur_off && p.T <= 512) {
    // config: 1 = tensor-core groups (default), 2 = pipelined 9-warp variant, DPT*1000+NT = CUDA-core variants
    const char* ec = getenv("GDN_RCFG");
    int cfg = ec ? atoi(ec) : 1;
    if (cfg == 2) {
      static bool t2attr = false;
      if (!t2attr) { cudaFuncSetAttribute(gdn_recur_tc2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RecurTC2::SMEM); t2attr = true; }
      gdn_recur_tc2_kernel<<<p.N * HV * 16, RecurTC2::NT, RecurTC2::SMEM, stream>>>(p);
      return;
    }
    if (cfg == 1) {
      static bool tattr = false;
      if (!tattr) { cudaFuncSetAttribute(gdn_recur_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RecurTC::SMEM); tattr = true; }
      gdn_recur_tc_kernel<<<p.N * HV * 16, 256, RecurTC::SMEM, stream>>>(p);
      return;
    }
    switch (cfg) {
      case 4128: launch_recur<4, 128>(p, stream); break;
      case 4256: launch_recur<4, 256>(p, stream); break;
      case 8128: launch_recur<8, 128>(p, stream); break;
      case 8256: launch_recur<8, 256>(p, stream); break;
      case 16128: launch_recur<16, 128>(p, stream); break;
      default: launch_recur<16, 256>(p, stream); break;
    }
    return;
  }
  const char* ef = getenv("GDN_FUSED_T");
  const int fused_T = ef ? atoi(ef) : 320;
  const char* ea = getenv("GDN_FUSED_AVG");
  const int fused_avg = ea ? atoi(ea) : 80;
  if (p.T <= fused_T && p.T <= fused_avg * p.N) {
    gdn_fused_kernel<<<p.N * HV, NTHREADS, FUSED_SMEM, stream>>>(p);
    return;
  }
  const char* e = getenv("GDN_SPLIT");
  const int split_thresh = e ? atoi(e) : 148;
  if (grid > split_thresh) {
    static const bool ktime = getenv("GDN_KTIME") != nullptr;
    cudaEvent_t e0, e1, e2;
    if (ktime) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventRecord(e0, stream); }
    gdn_prep_kernel<<<(p.num_prep + 1) / 2, 2 * NTHREADS, 2 * PREP_SMEM_ALIGNED, stream>>>(p);
    if (ktime) cudaEventRecord(e1, stream);
    Params pc = p; pc.flags_ready = 1;
    gdn_chain_kernel<VS, NSTAGES><<<p.num_chain, NTHREADS, ChainSmem<VS, NSTAGES>::BYTES, stream>>>(pc);
    if (ktime) { cudaEventRecord(e2, stream); cudaEventSynchronize(e2); float a, b; cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); printf("ktime: prep %.1f us (%d blocks x 2), chain %.1f us (%d blocks, VS=%d)\n", a * 1000, (p.num_prep + 1) / 2, b * 1000, p.num_chain, VS); }
  } else {
    gdn_kernel<VS, NSTAGES><<<grid, NTHREADS, smem, stream>>>(p);
  }
}

std::vector<torch::Tensor> run(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> state,
                               torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
                               torch::Tensor cu_seqlens, double scale, int64_t nsplit) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && a.is_contiguous() && b.is_contiguous());
  const int T = q.size(0), N = cu_seqlens.size(0) - 1;
  TORCH_CHECK(q.size(1) == HQ && k.size(1) == HQ && v.size(1) == HV && q.size(2) == D);
  auto cu = cu_seqlens;
  if (cu.dtype() != torch::kInt64) cu = cu.to(torch::kInt64);
  cu = cu.contiguous();
  auto A_logf = A_log.to(torch::kFloat32).contiguous();
  auto dtb = dt_bias.to(torch::kFloat32).contiguous();
  torch::Tensor st;
  if (state.has_value() && state->defined()) { st = state->to(torch::kFloat32).contiguous(); TORCH_CHECK(st.size(0) == N); }
  auto out = torch::empty({T, HV, D}, q.options());
  auto new_state = torch::empty({N, HV, D, D}, q.options().dtype(torch::kFloat32));
  if (N == 0) return {out, new_state};
  if (scale == 0.0) scale = 1.0 / std::sqrt((double)D);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_chunks = (T + 63 * N) / C + 1;
  ensure_workspace(max_chunks, stream);
  g_epoch++;
  if (g_epoch == 0) { cudaMemsetAsync(g_ws.flags, 0, (size_t)g_ws.max_chunks * HV * sizeof(int), stream); g_epoch = 1; }
  Params p;
  p.q = (const __nv_bfloat16*)q.data_ptr(); p.k = (const __nv_bfloat16*)k.data_ptr(); p.v = (const __nv_bfloat16*)v.data_ptr();
  p.state = st.defined() ? st.data_ptr<float>() : nullptr;
  p.A_log = A_logf.data_ptr<float>(); p.a = (const __nv_bfloat16*)a.data_ptr(); p.dt_bias = dtb.data_ptr<float>();
  p.b = (const __nv_bfloat16*)b.data_ptr(); p.cu = cu.data_ptr<int64_t>(); p.scale = (float)scale;
  p.out = (__nv_bfloat16*)out.data_ptr(); p.new_state = new_state.data_ptr<float>();
  p.recs = g_ws.recs; p.flags = g_ws.flags; p.counters = g_ws.counters; p.epoch = g_epoch; p.T = T; p.N = N;
  p.num_prep = max_chunks * HV;
  int nst = 3;
  const char* envst = getenv("GDN_NST");
  if (nsplit <= 0) { nsplit = (N >= 50) ? 2 : 4; nst = (nsplit == 4) ? 3 : 2; }  // 3-stage prefetch for every 32-column chain
  if (envst) nst = atoi(envst);
  const int VS = (int)(D / nsplit);
  p.num_chain = N * HV * (int)nsplit;
  { const char* e = getenv("GDN_DEBUG"); p.debug = e ? atoi(e) : 0; }
  p.flags_ready = 0;
  static torch::Tensor dbg_t; p.dbg = nullptr;
  if (p.debug & 256) { const int ndbg = std::max(p.num_prep + p.num_chain, 128 * p.N); if (!dbg_t.defined() || dbg_t.numel() < ndbg * 32) dbg_t = torch::zeros({ndbg * 32}, q.options().dtype(torch::kInt64)); dbg_t.zero_(); p.dbg = dbg_t.data_ptr<int64_t>(); g_dbg = dbg_t; }
  const int grid = p.num_prep + p.num_chain;
  if (VS == 32 && nst == 3) launch<32, 3>(p, grid, stream);
  else if (VS == 32) launch<32, 2>(p, grid, stream);
  else if (VS == 64) launch<64, 2>(p, grid, stream);
  else TORCH_CHECK(false, "nsplit must be 2 or 4");
  return {out, new_state};
}

}  // namespace gdn


std::vector<torch::Tensor> run(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> state, torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b, torch::Tensor cu_seqlens, double scale, int64_t nsplit) { return gdn::run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, nsplit); }

'''
_CPP_DECL = "std::vector<torch::Tensor> run(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> state, torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b, torch::Tensor cu_seqlens, double scale, int64_t nsplit);"

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(name="gdn_prefill_b200_v11", cpp_sources=[_CPP_DECL], cuda_sources=[_CUDA_SRC], functions=["run"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-ftz=true"],
                           extra_cflags=["-std=c++17"], verbose=False)
    return _ext

def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    ext = _get_ext()
    if scale is None:
        scale = 0.0
    if torch.is_tensor(scale):
        scale = float(scale.item())
    out, new_state = ext.run(q.contiguous(), k.contiguous(), v.contiguous(), state, A_log, a.contiguous(), dt_bias, b.contiguous(), cu_seqlens, float(scale), 0)
    return out, new_state
