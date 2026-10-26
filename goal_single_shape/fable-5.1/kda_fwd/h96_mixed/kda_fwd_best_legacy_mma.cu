// KDA forward v3: warp-specialized. 8 consumer warps own the fp32 state S^T[v][k] (16 v-rows each) and run the
// state MMAs; 4 producer warps compute per-chunk gate/decay, L2 norms, operand tiles (K~, Q~, K^), the intra-chunk
// A/B matrices and the triangular solve matrix T_beta. Producer/consumer handshake via named barriers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int C = 16;
constexpr int NCONS = 128, NGATE = 128, NLOAD = 128, NT = 384;  // warps 0-3 consumers (32 v-rows each), 4-7 gate, 8-11 loaders
constexpr int NB = 3;
constexpr int NSTAGE = 6;
constexpr int ROWB = D * 2;
constexpr uint32_t FULL = 0xffffffffu;
// named barrier ids
constexpr int BAR_FULL0 = 1, BAR_EMPTY0 = 4, BAR_TILES0 = 7, BAR_STAGE0 = 10, BAR_GATE = 13, BAR_LD = 14;
constexpr int CNT_FULL = NT, CNT_EMPTY = NCONS + NGATE, CNT_TILES = NGATE + NLOAD, CNT_STAGE = NGATE + NLOAD;
constexpr int W_GATE = 4, W_LOAD = 8;

struct __align__(16) Smem {
  uint8_t stage[NSTAGE][4][C * ROWB];  // raw q,k,v,g
  uint8_t kt[NB][C * ROWB];
  uint8_t qt[NB][C * ROWB];
  uint8_t kb[NB][C * ROWB];
  uint8_t vt[NB][C * ROWB];
  __nv_bfloat16 tt[NB][C][C];
  __nv_bfloat16 bt[NB][C][C];
  float egp[NB][4][32];   // e^{gamma_last} permuted: [lane&3][2*nt+j] = e[8nt + 2(lane&3) + j]
  float at[C][C];
  float tw[2][8][8];
  float mw[8][8];
  float betas[4][C];
  float rn[4][2][C];      // [chunk&3][0=rnqs,1=rnk][tok]
  float tot[2][2][D];     // [chunk parity][half][d]
  uint8_t opatch[4][C * 64];
  uint64_t mb[5][NB];
};

__device__ __forceinline__ uint32_t swz_off(int tok, int chunk) {
  return tok * ROWB + ((chunk ^ (tok & 7)) << 4);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void bar_sync(int id, int cnt) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(cnt)); }
__device__ __forceinline__ void bar_arrive(int id, int cnt) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(cnt)); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void stsm_x4_t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16(uint32_t u) {
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}
__device__ __forceinline__ float tanh_approx(float x) {
  float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float ex2_approx(float x) {
  float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float sigmoid_acc(float x) { return 1.0f / (1.0f + __expf(-x)); }
constexpr float LOG2E = 1.4426950408889634f;
__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(cnt)); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("{.reg .b64 st; mbarrier.arrive.shared.b64 st, [%0];}\n" ::"r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t phase) {
  asm volatile("{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra LAB_WAIT;\n}\n" ::"r"(a), "r"(phase) : "memory");
}
enum { MB_FULL = 0, MB_EMPTY = 1, MB_TILES = 2, MB_STAGE = 3, MB_LD = 4 };


template <int N> __device__ __forceinline__ void reg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N> __device__ __forceinline__ void reg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }



__global__ void __launch_bounds__(NT, 1)
kda_fwd_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
               const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ g,
               const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log,
               const float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ state,
               const int64_t* __restrict__ cu_seqlens, const int* __restrict__ seq_order,
               __nv_bfloat16* __restrict__ out, int H, float scale, float lower_bound) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.x % H;
  const int n = seq_order[blockIdx.x / H];
  const int64_t start = cu_seqlens[n], end = cu_seqlens[n + 1];
  const int len = (int)(end - start);
  const int nchunks = (len + C - 1) / C;
  const size_t tokstride = (size_t)H * D;
  const int tile = lane >> 3, r = lane & 7;
  const uint32_t mb0 = smem_u32(&sm.mb[0][0]);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      mbar_init(mb0 + 8 * (MB_FULL * NB + i), NGATE + NLOAD);
      mbar_init(mb0 + 8 * (MB_EMPTY * NB + i), NCONS);
      mbar_init(mb0 + 8 * (MB_TILES * NB + i), NGATE);
      mbar_init(mb0 + 8 * (MB_STAGE * NB + i), NLOAD);
      mbar_init(mb0 + 8 * (MB_LD * NB + i), NLOAD);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
#define MBA(kind, i) (mb0 + 8 * ((kind) * NB + (i)))

  if (warp >= W_LOAD) {
    // ===================== LOADERS (warps 8-11): cp.async, norms (MMA), beta, A/B MMAs, T (warp 8) =====================
    reg_dec<112>();
    const int p = tid - NCONS - NGATE;  // 0..127
    const int lw = warp - W_LOAD;       // 0..3
    const float eps = 1e-6f;
    const __nv_bfloat16* src0[8];
    uint32_t doff[8];
    int rtok[8];
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      int lin = p + NLOAD * rr, tsel = lin >> 8, rem = lin & 255, tok = rem >> 4, chunk = rem & 15;
      const __nv_bfloat16* base = tsel == 0 ? q : tsel == 1 ? k : tsel == 2 ? v : g;
      src0[rr] = base + (start + tok) * tokstride + (size_t)h * D + chunk * 8;
      doff[rr] = tsel * (C * ROWB) + swz_off(tok, chunk);
      rtok[rr] = tok;
    }
    auto load_chunk = [&](int c, int stg) {
      const size_t coff = (size_t)c * C * tokstride;
      const uint32_t sbase = smem_u32(&sm.stage[stg][0][0]);
      const int nv = len - c * C;
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        bool valid = rtok[rr] < nv;
        cp_async16(sbase + doff[rr], valid ? (src0[rr] + coff) : src0[rr], valid);
      }
    };
    const int a_tok = ((tile & 1) << 3) + r;
    const uint32_t a_row = a_tok * ROWB;
    const int a_x = (tile >> 1) ^ (a_tok & 7);
    auto norms_and_beta = [&](int c, int stg) {
      if (lw >= 2) {
        const int wq = lw - 2;  // 0: q (scaled), 1: k
        const uint32_t src = smem_u32(wq == 0 ? sm.stage[stg][0] : sm.stage[stg][1]) + a_row;
        float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
        uint32_t a[2][4];
        ldsm_x4(a[0][0], a[0][1], a[0][2], a[0][3], src + (a_x << 4));
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const int cur = ks & 1, nxt = cur ^ 1;
          if (ks + 1 < 8) ldsm_x4(a[nxt][0], a[nxt][1], a[nxt][2], a[nxt][3], src + ((a_x ^ (2 * (ks + 1))) << 4));
          mma_bf16(c0, a[cur], a[cur][0], a[cur][2]);
          mma_bf16(c1, a[cur], a[cur][1], a[cur][3]);
        }
        const int i0 = lane >> 2, jq = 2 * (lane & 3);
        float* dst = sm.rn[c & 3][wq];
        const float mul = wq == 0 ? scale : 1.f;
        if (i0 == jq) { dst[i0] = rsqrtf(c0[0] + eps) * mul; dst[i0 + 8] = rsqrtf(c1[2] + eps) * mul; }
        if (i0 == jq + 1) { dst[i0] = rsqrtf(c0[1] + eps) * mul; dst[i0 + 8] = rsqrtf(c1[3] + eps) * mul; }
        if (lw == 2 && lane < C) {
          int lt = c * C + lane;
          float bl = 0.f;
          if (lt < len) bl = sigmoid_acc(__bfloat162float(beta[(start + lt) * H + h]));
          sm.betas[c & 3][lane] = bl;
        }
      }
    };
    int stg_next = 0, ldk = 0;
    auto ld_sync = [&]() { mbar_arrive(MBA(MB_LD, 0)); mbar_wait(MBA(MB_LD, 0), ldk & 1); ldk++; };
#pragma unroll
    for (int i = 0; i < NSTAGE - 1; ++i) {
      if (i < nchunks) load_chunk(i, i);
      cp_async_commit();
    }
    stg_next = NSTAGE - 1;
    if (nchunks > 0) { cp_async_wait<NSTAGE - 2>(); ld_sync(); norms_and_beta(0, 0); mbar_arrive(MBA(MB_STAGE, 0)); }
    if (nchunks > 1) { cp_async_wait<NSTAGE - 3>(); ld_sync(); norms_and_beta(1, 1); mbar_arrive(MBA(MB_STAGE, 1)); }
    if (nchunks > 2) { cp_async_wait<NSTAGE - 4>(); ld_sync(); norms_and_beta(2, 2); mbar_arrive(MBA(MB_STAGE, 2)); }
    const int bn_tok = ((tile >> 1) << 3) + r;   // B frags (non-trans): tiles (tok0-7,lo),(tok0-7,hi),(tok8-15,lo),(tok8-15,hi)
    const uint32_t b_row = bn_tok * ROWB;
    const int b_x = (tile & 1) ^ (bn_tok & 7);
    int stg_c3 = 3, b = 0, par = 0;
    for (int c = 0; c < nchunks; ++c) {
      mbar_wait(MBA(MB_TILES, b), par);  // gate pass c done
      if (lw < 2) {
        const uint32_t abase = smem_u32(lw == 0 ? sm.kt[b] : sm.qt[b]) + a_row;
        const uint32_t bbase = smem_u32(sm.kb[b]) + b_row;
        float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};   // A (or B) n-tiles 0/1
        float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};   // A^T = K^ K~^T (warp 8 only)
        uint32_t af[2][4], bf[2][4];
        ldsm_x4(af[0][0], af[0][1], af[0][2], af[0][3], abase + (a_x << 4));
        ldsm_x4(bf[0][0], bf[0][1], bf[0][2], bf[0][3], bbase + (b_x << 4));
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const int cur = ks & 1, nxt = cur ^ 1;
          if (ks + 1 < 8) {
            ldsm_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3], abase + ((a_x ^ (2 * (ks + 1))) << 4));
            ldsm_x4(bf[nxt][0], bf[nxt][1], bf[nxt][2], bf[nxt][3], bbase + ((b_x ^ (2 * (ks + 1))) << 4));
          }
          mma_bf16(c0, af[cur], bf[cur][0], bf[cur][1]);
          mma_bf16(c1, af[cur], bf[cur][2], bf[cur][3]);
          if (lw == 0) {
            // A^T: A-operand = K^ tile, B-operand = K~ tile. Tile order differs: A frag = {t0,t2,t1,t3} of B-order and vice versa.
            uint32_t ta[4] = {bf[cur][0], bf[cur][2], bf[cur][1], bf[cur][3]};
            mma_bf16(d0, ta, af[cur][0], af[cur][2]);
            mma_bf16(d1, ta, af[cur][1], af[cur][3]);
          }
        }
        const int i0 = lane >> 2, j0 = 2 * (lane & 3);
        if (lw == 0) {
          // L = diag(beta) strict_lower(A) ; Lt = L^T. C-layout: x[nt][e]: row i0 (+8 for e>=2), col 8nt + j0 + (e&1)
          const float* betas = sm.betas[c & 3];
          const float bi0 = betas[i0], bi1 = betas[i0 + 8];
          const float bj[4] = {betas[j0], betas[j0 + 1], betas[j0 + 8], betas[j0 + 9]};
          float X[2][4], Xt[2][4];
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int ii = i0 + ((e >> 1) << 3), jj = 8 * nt + j0 + (e & 1);
              const float av = nt ? c1[e] : c0[e], atv = nt ? d1[e] : d0[e];
              X[nt][e] = (jj < ii) ? av * ((e >> 1) ? bi1 : bi0) : 0.f;
              Xt[nt][e] = (jj > ii) ? atv * bj[2 * nt + (e & 1)] : 0.f;
            }
          }
          const float dg0 = (i0 == j0) ? 1.f : 0.f, dg1 = (i0 == j0 + 1) ? 1.f : 0.f;
          // product helper: P = X*Y given X (C-layout) and Yt (C-layout)
#define KDA_PROD(P, Xm, Ytm) do { \
            uint32_t pa[4] = {pack_bf16(Xm[0][0], Xm[0][1]), pack_bf16(Xm[0][2], Xm[0][3]), pack_bf16(Xm[1][0], Xm[1][1]), pack_bf16(Xm[1][2], Xm[1][3])}; \
            P[0][0] = P[0][1] = P[0][2] = P[0][3] = P[1][0] = P[1][1] = P[1][2] = P[1][3] = 0.f; \
            mma_bf16(P[0], pa, pack_bf16(Ytm[0][0], Ytm[0][1]), pack_bf16(Ytm[1][0], Ytm[1][1])); \
            mma_bf16(P[1], pa, pack_bf16(Ytm[0][2], Ytm[0][3]), pack_bf16(Ytm[1][2], Ytm[1][3])); } while (0)
#define KDA_ADDI(Z, Xm, sgn) do { \
            Z[0][0] = sgn * Xm[0][0] + dg0; Z[0][1] = sgn * Xm[0][1] + dg1; Z[0][2] = sgn * Xm[0][2]; Z[0][3] = sgn * Xm[0][3]; \
            Z[1][0] = sgn * Xm[1][0]; Z[1][1] = sgn * Xm[1][1]; Z[1][2] = sgn * Xm[1][2] + dg0; Z[1][3] = sgn * Xm[1][3] + dg1; } while (0)
          float L2[2][4], L2t[2][4], L4[2][4], L4t[2][4], L8[2][4], L8t[2][4];
          KDA_PROD(L2, X, Xt);   KDA_PROD(L2t, Xt, X);
          KDA_PROD(L4, L2, L2t); KDA_PROD(L4t, L2t, L2);
          KDA_PROD(L8, L4, L4t); KDA_PROD(L8t, L4t, L4);
          float F1[2][4], G2t[2][4], G4[2][4], G4t[2][4], G8t[2][4];
          KDA_ADDI(F1, X, -1.f);      // I - L
          KDA_ADDI(G2t, L2t, 1.f);    // (I + L^2)^T
          KDA_ADDI(G4, L4, 1.f);
          KDA_ADDI(G4t, L4t, 1.f);
          KDA_ADDI(G8t, L8t, 1.f);
          float P1[2][4], P2t[2][4], TT[2][4];
          KDA_PROD(P1, F1, G2t);      // (I-L)(I+L^2)
          KDA_PROD(P2t, G8t, G4);     // ((I+L^4)(I+L^8))^T = (I+L^8)^T (I+L^4)^T ; B-operand needs ((I+L^4)^T)^T = G4
          KDA_PROD(TT, P1, P2t);      // T
          (void)G4t;
          // T_beta = T diag(beta): scale columns, store bf16 pairs tt[i][j..j+1]
          *reinterpret_cast<uint32_t*>(&sm.tt[b][i0][j0]) = pack_bf16(TT[0][0] * bj[0], TT[0][1] * bj[1]);
          *reinterpret_cast<uint32_t*>(&sm.tt[b][i0 + 8][j0]) = pack_bf16(TT[0][2] * bj[0], TT[0][3] * bj[1]);
          *reinterpret_cast<uint32_t*>(&sm.tt[b][i0][j0 + 8]) = pack_bf16(TT[1][0] * bj[2], TT[1][1] * bj[3]);
          *reinterpret_cast<uint32_t*>(&sm.tt[b][i0 + 8][j0 + 8]) = pack_bf16(TT[1][2] * bj[2], TT[1][3] * bj[3]);
        } else {
          *reinterpret_cast<uint32_t*>(&sm.bt[b][i0][j0]) = pack_bf16(j0 <= i0 ? c0[0] : 0.f, j0 + 1 <= i0 ? c0[1] : 0.f);
          *reinterpret_cast<uint32_t*>(&sm.bt[b][i0 + 8][j0]) = pack_bf16(c0[2], j0 + 1 <= i0 + 8 ? c0[3] : 0.f);
          *reinterpret_cast<uint32_t*>(&sm.bt[b][i0][j0 + 8]) = pack_bf16(j0 + 8 <= i0 ? c1[0] : 0.f, j0 + 9 <= i0 ? c1[1] : 0.f);
          *reinterpret_cast<uint32_t*>(&sm.bt[b][i0 + 8][j0 + 8]) =
              pack_bf16(j0 + 8 <= i0 + 8 ? c1[2] : 0.f, j0 + 9 <= i0 + 8 ? c1[3] : 0.f);
        }
      }
      mbar_arrive(MBA(MB_FULL, b));
      if (c + NSTAGE - 1 < nchunks) load_chunk(c + NSTAGE - 1, stg_next);
      cp_async_commit();
      stg_next = (stg_next + 1 == NSTAGE) ? 0 : stg_next + 1;
      cp_async_wait<NSTAGE - 4>();
      ld_sync();
      if (c + 3 < nchunks) {
        norms_and_beta(c + 3, stg_c3);
        mbar_arrive(MBA(MB_STAGE, b));
      }
      stg_c3 = (stg_c3 + 1 == NSTAGE) ? 0 : stg_c3 + 1;
      if (b + 1 == NB) { b = 0; par ^= 1; } else { b = b + 1; }
    }
    cp_async_wait<0>();
    return;
  }

  if (warp >= W_GATE) {
    // ===================== GATE (warps 4-7): per-token gate/decay + operand tiles =====================
    const int p = tid - NCONS;  // 0..127
    const float Aexp = __expf(A_log[h]);
    const int half = p & 1, pm = p >> 1;
    const int cp = pm;
    const float adt0 = Aexp * dt_bias[h * D + 2 * cp], adt1 = Aexp * dt_bias[h * D + 2 * cp + 1];
    const float lbh = 0.5f * lower_bound;
    uint32_t offs[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) offs[t] = swz_off(8 * half + t, cp >> 2) + (cp & 3) * 4;
    const int egi = (cp & 3), egj = 2 * (cp >> 2);
    int stg = 0, b = 0, par = 0;
    for (int c = 0; c < nchunks; ++c) {
      mbar_wait(MBA(MB_STAGE, b), par);
      if (c >= NB) mbar_wait(MBA(MB_EMPTY, b), par ^ 1);
      const uint8_t* sq = sm.stage[stg][0];
      const uint8_t* sk = sm.stage[stg][1];
      const uint8_t* sv = sm.stage[stg][2];
      const uint8_t* sg = sm.stage[stg][3];
      uint32_t graw[8], kraw[8], qraw[8];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        graw[t] = *reinterpret_cast<const uint32_t*>(sg + offs[t]);
        kraw[t] = *reinterpret_cast<const uint32_t*>(sk + offs[t]);
        qraw[t] = *reinterpret_cast<const uint32_t*>(sq + offs[t]);
      }
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        int lin = p + NGATE * rr, tok = lin >> 4, chunk = lin & 15;
        uint32_t o = swz_off(tok, chunk);
        *reinterpret_cast<uint4*>(sm.vt[b] + o) = *reinterpret_cast<const uint4*>(sv + o);
      }
      const int nv = len - c * C;
      float gam[8][2];
      float p0 = 0.f, p1 = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int tok = 8 * half + t;
        float2 x = unpack_bf16(graw[t]);
        bool valid = tok < nv;
        float t0 = tanh_approx(0.5f * fmaf(Aexp, x.x, adt0));
        float t1 = tanh_approx(0.5f * fmaf(Aexp, x.y, adt1));
        p0 += valid ? fmaf(lbh, t0, lbh) : 0.f;
        p1 += valid ? fmaf(lbh, t1, lbh) : 0.f;
        gam[t][0] = p0;
        gam[t][1] = p1;
      }
      const float o0 = __shfl_xor_sync(FULL, p0, 1), o1 = __shfl_xor_sync(FULL, p1, 1);
      const float base0 = half ? o0 : 0.f, base1 = half ? o1 : 0.f;
      if (!half) {
        const float g15_0 = p0 + o0, g15_1 = p1 + o1;
        *reinterpret_cast<float2*>(&sm.egp[b][egi][egj]) =
            make_float2(ex2_approx(fmaxf(g15_0, -80.f) * LOG2E), ex2_approx(fmaxf(g15_1, -80.f) * LOG2E));
      }
      const float* rnqs = sm.rn[c & 3][0];
      const float* rnk = sm.rn[c & 3][1];
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int tok = 8 * half + t;
        float e0 = ex2_approx(fmaxf(gam[t][0] + base0, -80.f) * LOG2E);
        float e1 = ex2_approx(fmaxf(gam[t][1] + base1, -80.f) * LOG2E);
        float r0 = rcp_approx(e0), r1 = rcp_approx(e1);
        float2 kk = unpack_bf16(kraw[t]);
        float2 qq = unpack_bf16(qraw[t]);
        float rk = rnk[tok], rq = rnqs[tok];
        float kn0 = kk.x * rk, kn1 = kk.y * rk, qn0 = qq.x * rq, qn1 = qq.y * rq;
        *reinterpret_cast<uint32_t*>(sm.kt[b] + offs[t]) = pack_bf16(kn0 * e0, kn1 * e1);
        *reinterpret_cast<uint32_t*>(sm.qt[b] + offs[t]) = pack_bf16(qn0 * e0, qn1 * e1);
        *reinterpret_cast<uint32_t*>(sm.kb[b] + offs[t]) = pack_bf16(kn0 * r0, kn1 * r1);
      }
      mbar_arrive(MBA(MB_TILES, b));
      mbar_arrive(MBA(MB_FULL, b));
      stg = (stg + 1 == NSTAGE) ? 0 : stg + 1;
      if (b + 1 == NB) { b = 0; par ^= 1; } else { b = b + 1; }
    }
    return;
  }

  // ===================== CONSUMERS (warps 0-3): 32 v-rows each, fp32 state in mma accumulators =====================
  reg_inc<224>();
  float s[2][16][4];
  __nv_bfloat16* sptr = state + ((size_t)n * H + h) * D * D;
  const int kq = 2 * (lane & 3);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int v0 = warp * 32 + m * 16 + (lane >> 2);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      float2 a = unpack_bf16(*reinterpret_cast<const uint32_t*>(sptr + v0 * D + nt * 8 + kq));
      float2 bb = unpack_bf16(*reinterpret_cast<const uint32_t*>(sptr + (v0 + 8) * D + nt * 8 + kq));
      s[m][nt][0] = a.x; s[m][nt][1] = a.y; s[m][nt][2] = bb.x; s[m][nt][3] = bb.y;
    }
  }
  const int btok = ((tile >> 1) << 3) + r, bch = tile & 1;
  const int ttok = ((tile & 1) << 3) + r, tch = tile >> 1;
  const uint32_t nt_row = btok * ROWB, tr_row = ttok * ROWB;
  const int nt_x = bch ^ (btok & 7), tr_x = tch ^ (ttok & 7);
  const uint32_t v_off0 = swz_off(btok, 4 * warp + bch), v_off1 = swz_off(btok, 4 * warp + 2 + bch);
  const uint32_t tb_off = (((tile >> 1) << 3) + r) * 32 + (tile & 1) * 16;
  // output patch [tok][32 v] bf16: 64B rows; 16B chunk index (0..3) swizzled by (tok>>1)&3
  const int prow = ((tile >> 1) << 3) + r;
  const uint32_t patch_w0 = smem_u32(&sm.opatch[warp][0]) + prow * 64 + (((tile & 1) ^ ((prow >> 1) & 3)) << 4);
  const uint32_t patch_w1 = smem_u32(&sm.opatch[warp][0]) + prow * 64 + (((2 + (tile & 1)) ^ ((prow >> 1) & 3)) << 4);
  const int orow = lane >> 1;
  const uint8_t* patch_r0 = &sm.opatch[warp][0] + orow * 64 + ((((lane & 1) * 2) ^ ((orow >> 1) & 3)) << 4);
  const uint8_t* patch_r1 = &sm.opatch[warp][0] + orow * 64 + ((((lane & 1) * 2 + 1) ^ ((orow >> 1) & 3)) << 4);
  __nv_bfloat16* obase = out + (size_t)h * D + warp * 32 + (lane & 1) * 16;

  int b = 0, par = 0;
  for (int c = 0; c < nchunks; ++c) {
    mbar_wait(MBA(MB_FULL, b), par);
    const uint32_t kt_base = smem_u32(sm.kt[b]) + nt_row, qt_base = smem_u32(sm.qt[b]) + nt_row;
    const uint32_t kb_base = smem_u32(sm.kb[b]) + tr_row;
    float tmp[2][2][4], o[2][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) { tmp[m][i][j] = 0.f; o[m][i][j] = 0.f; }
    uint32_t bk[2][4], bq[2][4], vr[2][4], bt_[4], bb_[4];
    ldsm_x4(bk[0][0], bk[0][1], bk[0][2], bk[0][3], kt_base + (nt_x << 4));
    ldsm_x4(bq[0][0], bq[0][1], bq[0][2], bq[0][3], qt_base + (nt_x << 4));
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int cur = ks & 1, nxt = cur ^ 1;
      if (ks + 1 < 8) {
        ldsm_x4(bk[nxt][0], bk[nxt][1], bk[nxt][2], bk[nxt][3], kt_base + ((nt_x ^ (2 * (ks + 1))) << 4));
        ldsm_x4(bq[nxt][0], bq[nxt][1], bq[nxt][2], bq[nxt][3], qt_base + ((nt_x ^ (2 * (ks + 1))) << 4));
      } else {
        ldsm_x4_t(vr[0][0], vr[0][1], vr[0][2], vr[0][3], smem_u32(sm.vt[b]) + v_off0);
        ldsm_x4_t(vr[1][0], vr[1][1], vr[1][2], vr[1][3], smem_u32(sm.vt[b]) + v_off1);
        ldsm_x4(bt_[0], bt_[1], bt_[2], bt_[3], smem_u32(&sm.tt[b][0][0]) + tb_off);
        ldsm_x4(bb_[0], bb_[1], bb_[2], bb_[3], smem_u32(&sm.bt[b][0][0]) + tb_off);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        uint32_t a[4] = {pack_bf16(s[m][2 * ks][0], s[m][2 * ks][1]), pack_bf16(s[m][2 * ks][2], s[m][2 * ks][3]),
                         pack_bf16(s[m][2 * ks + 1][0], s[m][2 * ks + 1][1]), pack_bf16(s[m][2 * ks + 1][2], s[m][2 * ks + 1][3])};
        mma_bf16(tmp[m][0], a, bk[cur][0], bk[cur][1]);
        mma_bf16(tmp[m][1], a, bk[cur][2], bk[cur][3]);
        mma_bf16(o[m][0], a, bq[cur][0], bq[cur][1]);
        mma_bf16(o[m][1], a, bq[cur][2], bq[cur][3]);
      }
    }
    uint32_t au[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      uint32_t ar[4];
      float2 x;
      x = unpack_bf16(vr[m][0]); ar[0] = pack_bf16(x.x - tmp[m][0][0], x.y - tmp[m][0][1]);
      x = unpack_bf16(vr[m][1]); ar[1] = pack_bf16(x.x - tmp[m][0][2], x.y - tmp[m][0][3]);
      x = unpack_bf16(vr[m][2]); ar[2] = pack_bf16(x.x - tmp[m][1][0], x.y - tmp[m][1][1]);
      x = unpack_bf16(vr[m][3]); ar[3] = pack_bf16(x.x - tmp[m][1][2], x.y - tmp[m][1][3]);
      float u[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      mma_bf16(u[0], ar, bt_[0], bt_[1]);
      mma_bf16(u[1], ar, bt_[2], bt_[3]);
      au[m][0] = pack_bf16(u[0][0], u[0][1]); au[m][1] = pack_bf16(u[0][2], u[0][3]);
      au[m][2] = pack_bf16(u[1][0], u[1][1]); au[m][3] = pack_bf16(u[1][2], u[1][3]);
      mma_bf16(o[m][0], au[m], bb_[0], bb_[1]);
      mma_bf16(o[m][1], au[m], bb_[2], bb_[3]);
    }
    // stage the output tile in the per-warp patch (transposed); store after the update MMAs
    stsm_x4_t(patch_w0, pack_bf16(o[0][0][0], o[0][0][1]), pack_bf16(o[0][0][2], o[0][0][3]), pack_bf16(o[0][1][0], o[0][1][1]),
              pack_bf16(o[0][1][2], o[0][1][3]));
    stsm_x4_t(patch_w1, pack_bf16(o[1][0][0], o[1][0][1]), pack_bf16(o[1][0][2], o[1][0][3]), pack_bf16(o[1][1][0], o[1][1][1]),
              pack_bf16(o[1][1][2], o[1][1][3]));
    // state update S^T += U^T K^ with the decay interleaved (n-tile pairs two steps behind)
    const float* eg = sm.egp[b][lane & 3];
    uint32_t bu[2][4];
    ldsm_x4_t(bu[0][0], bu[0][1], bu[0][2], bu[0][3], kb_base + (tr_x << 4));
    float4 eg4[2];
    eg4[0] = *reinterpret_cast<const float4*>(eg);
#pragma unroll
    for (int np = 0; np < 8; ++np) {
      const int cur = np & 1, nxt = cur ^ 1;
      if (np + 1 < 8) ldsm_x4_t(bu[nxt][0], bu[nxt][1], bu[nxt][2], bu[nxt][3], kb_base + ((tr_x ^ (2 * (np + 1))) << 4));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        mma_bf16(s[m][2 * np], au[m], bu[cur][0], bu[cur][1]);
        mma_bf16(s[m][2 * np + 1], au[m], bu[cur][2], bu[cur][3]);
      }
      if (np == 7) mbar_arrive(MBA(MB_EMPTY, b));
      if (np == 0) {
        __syncwarp();
      }
      if (np >= 1) {
        // decay n-tiles of step np-1 (their MMAs were issued one step ago)
        const int pp = np - 1;
        const float4 e = eg4[pp & 1];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          s[m][2 * pp][0] *= e.x; s[m][2 * pp][1] *= e.y; s[m][2 * pp][2] *= e.x; s[m][2 * pp][3] *= e.y;
          s[m][2 * pp + 1][0] *= e.z; s[m][2 * pp + 1][1] *= e.w; s[m][2 * pp + 1][2] *= e.z; s[m][2 * pp + 1][3] *= e.w;
        }
      }
      if (np + 1 < 8) eg4[nxt] = *reinterpret_cast<const float4*>(eg + 4 * (np + 1));
      if (np == 2) {
        int lt = c * C + orow;
        uint4 val0 = *reinterpret_cast<const uint4*>(patch_r0);
        uint4 val1 = *reinterpret_cast<const uint4*>(patch_r1);
        if (lt < len) {
          uint4* dst = reinterpret_cast<uint4*>(obase + (start + lt) * tokstride);
          dst[0] = val0; dst[1] = val1;
        }
      }
    }
    {
      const float4 e = eg4[1];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        s[m][14][0] *= e.x; s[m][14][1] *= e.y; s[m][14][2] *= e.x; s[m][14][3] *= e.y;
        s[m][15][0] *= e.z; s[m][15][1] *= e.w; s[m][15][2] *= e.z; s[m][15][3] *= e.w;
      }
    }
    __syncwarp();
    if (b + 1 == NB) { b = 0; par ^= 1; } else { b = b + 1; }
  }
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int v0 = warp * 32 + m * 16 + (lane >> 2);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      *reinterpret_cast<uint32_t*>(sptr + v0 * D + nt * 8 + kq) = pack_bf16(s[m][nt][0], s[m][nt][1]);
      *reinterpret_cast<uint32_t*>(sptr + (v0 + 8) * D + nt * 8 + kq) = pack_bf16(s[m][nt][2], s[m][nt][3]);
    }
  }
}

}  // namespace

torch::Tensor kda_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
                      torch::Tensor A_log, torch::Tensor dt_bias, torch::Tensor state, torch::Tensor cu_seqlens,
                      torch::Tensor seq_order, torch::Tensor out, double scale, double lower_bound) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == D);
  const int H = q.size(2);
  const int N = cu_seqlens.numel() - 1;
  TORCH_CHECK(state.dim() == 4 && state.size(0) == N && state.size(1) == H && state.dtype() == torch::kBFloat16);
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64 && seq_order.dtype() == torch::kInt32 && seq_order.numel() == N);
  const size_t smem = sizeof(Smem);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(kda_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  kda_fwd_kernel<<<N * H, NT, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(state.data_ptr()), cu_seqlens.data_ptr<int64_t>(), seq_order.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), H, (float)scale, (float)lower_bound);
  return out;
}

int smem_bytes() { return (int)sizeof(Smem); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_fwd", &kda_fwd);
  m.def("smem_bytes", &smem_bytes);
}
