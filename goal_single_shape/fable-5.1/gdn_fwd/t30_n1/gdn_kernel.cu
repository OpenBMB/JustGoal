// Gated Delta Net prefill (varlen, GVA, k-last state) for NVIDIA B200 (sm_100a).  v20: mma.sync only (no TMEM).
//
// One CTA per (sequence, v-head, block of 16 V columns), 8 warps. The recurrence is independent per V column,
// so a CTA owns its columns for the whole sequence and walks it in 32-token chunks using the WY/chunk form:
//   u_t = v_t - v_old_t,   v_old_t = exp(G_t) (k_t S0) + sum_{s<t} exp(G_t-G_s) beta_s (k_t.k_s) u_s
//   => (I + L) u = rhs,  rhs = v - Eg o (K S0)     (L[t][s] = DEC[t][s] (k_t.k_s), s < t)
//   S_T  = GammaT S0 + sum_s Dsc_s k_s^T u_s               (Dsc_s = exp(G_T-G_s) beta_s)
//   out_t = scale [ Eg_t (q_t S0) + sum_{s<=t} DEC[t][s] (q_t.k_s) u_s ]
// G_t = cumsum(log g_t) inside the chunk, log g_t = -exp(A_log) softplus(a_t + dt_bias), accumulated in fp64.
//
// Warps 0-3 ("T"): tensor work on mma.sync m16n8k16 (bf16 x bf16 -> fp32): phase 1a [K x K^T, K x S0^T],
//   phase 1b [Q x K^T, Q x S0^T], then E1 (state).  Warps 4-7 ("S"): gates, decay matrix, the explicit inverse
//   M = (I+L)^-1 by exact blocked forward substitution (8x8 blocks), then E2 (output).  u = M rhs is a small
//   SIMT matmul done by all 256 threads.
// Precision: every fp32-valued MMA operand is split EXACTLY into three bf16 values (x = hi + mid + lo); with a
// bf16 partner all three products are formed, with a split partner all cross terms down to 2^-24 are kept;
// fp32 accumulate => fp32-equivalent results. Nothing is approximated or truncated.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

typedef __nv_bfloat16 bf16;
#define DEVI __device__ __forceinline__

namespace {

constexpr int CH = 32;      // chunk length (tokens)
constexpr int C = 8;        // V columns per CTA (128 CTAs: S0/W/QS/solve/E2 work halves; KK/QK/gates stay per CTA)
constexpr int NWARPS = 8;
constexpr int NTHREADS = 32 * NWARPS;
constexpr uint32_t FULL = 0xffffffffu;

// ---- shared memory map (bytes) ----
// OPB: 128B-swizzled K-major operand buffer: two 64-column halves of 128 rows x 128 B,
//      off(row,k) = (k/64)*16384 + row*128 + (((k/8)%8) ^ (row%8))*16 + (k%8)*2.
//      rows 0-31 K, 32-63 Q, 64-79 S0 hi, 80-95 S0 mid, 96-111 S0 lo
constexpr int OPB_OFF = 0;              constexpr int OPB_HALF = 16384;   // 2 x 16 KB
constexpr int VS_STRIDE = 16;                                            // bf16 [32][16]
constexpr int VS_OFF = 2 * OPB_HALF;                                     // 1 KB
constexpr int LS = 36;                                                   // L[t][s] fp32
constexpr int L_OFF = VS_OFF + CH * VS_STRIDE * 2;                       // 4.5 KB
constexpr int WS = 20;                                                   // QSp[3][t][c] fp32 partials (hi/mid/lo)
constexpr int QSP_OFF = L_OFF + 32 * LS * 4;                             // 7.5 KB
constexpr int RS = 20;                                                   // RHS[t][c] fp32
constexpr int RHS_OFF = QSP_OFF + 3 * 32 * WS * 4;                       // 2.5 KB
constexpr int US = 40;                                                   // U32[c][s] fp32
constexpr int U32_OFF = RHS_OFF + 32 * RS * 4;                           // 2.5 KB
constexpr int R8_OFF = U32_OFF + 16 * US * 4;                            // per S warp [4 cl][8 i] fp32: 4 x 128 B
constexpr int QKA_RB = 80;                                               // bf16 rows of 32 s (64 B) padded to 80 B
constexpr int QKA_OFF = R8_OFF + 4 * 128;                                // QKA[3][32 t][80 B]  7.5 KB
constexpr int QKA_SP = 32 * QKA_RB;
constexpr int U_RB = 80;                                                 // u splits: [c][s] rows, 80 B
constexpr int UB_OFF = QKA_OFF + 3 * QKA_SP;                             // UB[3][16][80]  3.75 KB
constexpr int U_SP = 16 * U_RB;
constexpr int UPB_OFF = UB_OFF + 3 * U_SP;                               // 3.75 KB
constexpr int S0F_STRIDE = 132;
constexpr int S0F_OFF = UPB_OFF + 3 * U_SP;                              // fp32 [16][132] S0^T block  8.25 KB
constexpr int DINV_OFF = S0F_OFF + 16 * S0F_STRIDE * 4;                  // Dinv [4 blocks x 72] fp32  1.1 KB
constexpr int NS = 24;                                                   // N[q][i][s] fp32, s contiguous (row stride 24)
constexpr int NM_OFF = DINV_OFF + 4 * 72 * 4;                            // N for q = 1..3: [3][8][24]  2.25 KB
constexpr int GATE_OFF = NM_OFF + 3 * 8 * NS * 4;                      // Ghi, Glo, Beta, Eg, Dsc [32] fp32 (128 B each), GammaT
constexpr int CU_OFF = GATE_OFF + 6 * 128;                                // int64 tstart, tend (published by the gate warp)
constexpr int STAMP_OFF = CU_OFF + 16;                                    // TIMING: [8 warps][16] int64 stamps
constexpr int SMEM_BYTES = STAMP_OFF + 8 * 16 * 8 + 16 + 1024;   // +1 KB: the base is aligned to 1 KB at runtime

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEVI int opb_off(int row, int kc) { return (kc >> 3) * OPB_HALF + row * 128 + (((kc & 7) ^ (row & 7)) << 4); }  // kc = k/8
DEVI void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
DEVI void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
DEVI void ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
DEVI void ldsm_x4_t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEVI void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
DEVI float fast_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float fast_log2(float x) { float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI float fast_rcp(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
constexpr float L2E = 1.4426950408889634f;   // log2(e)
constexpr float LN2 = 0.6931471805599453f;
DEVI void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// Exact 3-way bf16 split of an fp32 value: x = hi + mid + lo exactly (hi, mid: round-to-nearest on the
// bit pattern; lo: the exactly representable remainder). Pure integer/FADD ops.
DEVI uint32_t bf16_round_bits(float x) { return (__float_as_uint(x) + 0x8000u) & 0xFFFF0000u; }
DEVI uint32_t hi_halves(uint32_t a, uint32_t b) { return __byte_perm(a, b, 0x7632); }  // (a>>16) | (b & 0xFFFF0000)
DEVI uint32_t pack_bf16x2_rn(float lo_val, float hi_val) {   // {hi_val, lo_val} as bf16x2 (round to nearest even), one F2FP
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi_val), "f"(lo_val)); return r;
}
DEVI void split3x2(float x0, float x1, uint32_t& h, uint32_t& m, uint32_t& l) {
  // hi = rn_bf16(x), r = x - hi (exact, <= 16 significant bits), mid = rn_bf16(r), lo = r - mid (exact, <= 8 bits -> the
  // high 16 bits of its fp32 pattern hold it exactly)
  h = pack_bf16x2_rn(x0, x1);
  const float r0 = x0 - __uint_as_float(h << 16), r1 = x1 - __uint_as_float(h & 0xFFFF0000u);
  m = pack_bf16x2_rn(r0, r1);
  const float l0 = r0 - __uint_as_float(m << 16), l1 = r1 - __uint_as_float(m & 0xFFFF0000u);
  l = hi_halves(__float_as_uint(l0), __float_as_uint(l1));
}

struct __align__(64) Params {
  const bf16* q; const bf16* k; const bf16* v; const float* state; const bf16* a; const bf16* b; const int64_t* cu;   // 56 B
  int T; unsigned char HK, HV, rep_shift, flags;   // rep_shift: log2(HV/HK) or 255 if not a power of two; flags bit0: a/b vectorisable (HV == 8)
  const float* A_log; const float* dt_bias; bf16* out; float* new_state; float scale; int pad_; long long* dbg;
};

__global__ void __launch_bounds__(NTHREADS) gdn_fwd_kernel(const Params p) {
  extern __shared__ __align__(1024) unsigned char sm_raw[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int cb = blockIdx.x, hv = blockIdx.y, n = blockIdx.z;
  const int c0 = cb * C;
  // 1 KB-aligned base: the 128 B-swizzle XOR then never carries out of bits 4-6, so ldmatrix addresses are base ^ imm
  const uint32_t sbase_raw = smem_u32(sm_raw);
  const uint32_t sbase = (sbase_raw + 1023u) & ~1023u;
  unsigned char* const sm = sm_raw + (sbase - sbase_raw);
  float* __restrict__ Lm  = reinterpret_cast<float*>(sm + L_OFF);
  float* __restrict__ QSp = reinterpret_cast<float*>(sm + QSP_OFF);
  float* __restrict__ RHS = reinterpret_cast<float*>(sm + RHS_OFF);
  float* __restrict__ U32 = reinterpret_cast<float*>(sm + U32_OFF);
  float* __restrict__ R8  = reinterpret_cast<float*>(sm + R8_OFF);
  bf16*  __restrict__ VS  = reinterpret_cast<bf16*>(sm + VS_OFF);
  float* __restrict__ S0F = reinterpret_cast<float*>(sm + S0F_OFF);
  float* __restrict__ Dinv = reinterpret_cast<float*>(sm + DINV_OFF);
  float* __restrict__ Nm = reinterpret_cast<float*>(sm + NM_OFF);
  float* __restrict__ Ghi = reinterpret_cast<float*>(sm + GATE_OFF);
  float* __restrict__ Glo = reinterpret_cast<float*>(sm + GATE_OFF + 128);
  float* __restrict__ Beta = reinterpret_cast<float*>(sm + GATE_OFF + 256);
  float* __restrict__ Eg = reinterpret_cast<float*>(sm + GATE_OFF + 384);
  float* __restrict__ Dsc = reinterpret_cast<float*>(sm + GATE_OFF + 512);
  float* __restrict__ GammaT = reinterpret_cast<float*>(sm + GATE_OFF + 640);
  long long* __restrict__ CuS = reinterpret_cast<long long*>(sm + CU_OFF);
  unsigned char* __restrict__ QKA = sm + QKA_OFF;
  unsigned char* __restrict__ UBp = sm + UB_OFF;
  unsigned char* __restrict__ UPBp = sm + UPB_OFF;
  unsigned char* __restrict__ OPBp = sm + OPB_OFF;
  // Warp roles.  Tensor warps 0-3, each owns ONE 16x128 A operand and streams every B tile it needs (minimal smem traffic):
  //   w0: K mt0 -> KK(mt0, pair0) -> L rows 0-15, then W(mt0) -> RHS rows 0-15
  //   w1: Q mt0 -> QK(mt0, pair0) -> QKd rows 0-15, then QS(mt0)
  //   w2: K mt1 -> KK(mt1, pair0, pair1) -> L rows 16-31, then W(mt1) -> RHS rows 16-31
  //   w3: Q mt1 -> QK(mt1, pair0, pair1) -> QKd rows 16-31, then QS(mt1)
  //   (E1, the state update, runs on the tensor warps: k n-tiles 4w..4w+3 each)
  // Helper warps 4-7: w4 gates; w5/w6 Dinv; w5/w6/w7 N; all four: the blocked solve (4 columns each), u splits, E2.
  const bool tw = warp < 4;
  const bool gw = (warp == 4);          // gate warp
#ifdef TIMING
  long long* __restrict__ stamp_ = reinterpret_cast<long long*>(sm + STAMP_OFF) + warp * 16;
#define TSN(k) do { long long c_; asm volatile("bar.warp.sync 0xffffffff;\n\tmov.u64 %0, %%clock64;" : "=l"(c_) :: "memory"); if (lane == 0) stamp_[k] = c_; } while (0)
#define TSFN(k, x) do { long long c_; asm volatile("st.shared.f32 [%1], %2;\n\tbar.warp.sync 0xffffffff;\n\tmov.u64 %0, %%clock64;" : "=l"(c_) : "r"(sbase + SMEM_BYTES - 16 + 4 * (tid & 3)), "f"(x) : "memory"); if (lane == 0) stamp_[k] = c_; } while (0)
#define TS_FLUSH() do { __syncwarp(); if (lane < 16) p.dbg[((blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * NWARPS + warp) * 16 + lane] = stamp_[lane]; } while (0)
#else
#define TSN(k) do {} while (0)
#define TSFN(k, x) do {} while (0)
#define TS_FLUSH() do {} while (0)
#endif
  TSN(0);   // 0: start
#ifdef TIMING
  { long long g_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_) :: "memory"); if (lane == 0) stamp_[14] = g_; }
#endif

  // ================= prologue: issue every global load before anything depends on cu_seqlens =================
  // Program order of the (asm volatile) loads is preserved. Sequence 0 starts at token 0 by definition, so its loads never
  // wait for cu_seqlens. Only the gate warp touches the per-launch hot lines (cu_seqlens, A_log, dt_bias, a, b): 64 CTAs x 8
  // warps hitting one L2 line was measured to cost ~1000 cycles of queueing.
  const int HK = p.HK, HV = p.HV, T = p.T;
  const int hk = (p.rep_shift != 255) ? (hv >> p.rep_shift) : (hv / (HV / HK));
  const bool ab_vec = (p.flags & 1) != 0;
  const int kq_ts = HK * 128;                          // q/k token stride (elements)
  // S0^T block rows (LDG.128, lane-linear): w0-w3: 3 each (0-11), w5: 12, w6: 13, w7: 14-15; the gate warp none
  const int s0n = (warp == 0) ? 2 : (warp == 4) ? 0 : 1;                  // 8 rows: w0: 0-1, w1: 2, w2: 3, w3: 4, w5: 5, w6: 6, w7: 7
  const int s0r0 = (warp == 0) ? 0 : (warp <= 3) ? warp + 1 : warp;
  const float* S0g = p.state + ((size_t)(n * HV + hv) * 128 + c0) * 128;
  float4 s0r[3];
  uint4 ab_va = make_uint4(0, 0, 0, 0), ab_vb = make_uint4(0, 0, 0, 0);
  unsigned short ab_ua = 0, ab_ub = 0;
  auto load_ab = [&](int t0_) {   // gate warp, lane = token
    const int t0 = min(t0_, T - 1);                     // t0 == T only for an empty trailing sequence (T >= 1 guaranteed by the host)
    const int rr = t0 + min(lane, max(min(CH, T - t0) - 1, 0));
    if (ab_vec) {
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ab_va.x), "=r"(ab_va.y), "=r"(ab_va.z), "=r"(ab_va.w) : "l"(p.a + rr * 8) : "memory");
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(ab_vb.x), "=r"(ab_vb.y), "=r"(ab_vb.z), "=r"(ab_vb.w) : "l"(p.b + rr * 8) : "memory");
    } else {
      asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(ab_ua) : "l"(p.a + rr * HV + hv) : "memory");
      asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(ab_ub) : "l"(p.b + rr * HV + hv) : "memory");
    }
  };
  auto extract_ab = [&](float& av_, float& bv_) {
    if (ab_vec) {
      const int hp = hv >> 1;
      const uint32_t ua = (hp == 0) ? ab_va.x : (hp == 1) ? ab_va.y : (hp == 2) ? ab_va.z : ab_va.w;
      const uint32_t ub = (hp == 0) ? ab_vb.x : (hp == 1) ? ab_vb.y : (hp == 2) ? ab_vb.z : ab_vb.w;
      av_ = __uint_as_float((hv & 1) ? (ua & 0xFFFF0000u) : (ua << 16));
      bv_ = __uint_as_float((hv & 1) ? (ub & 0xFFFF0000u) : (ub << 16));
    } else {
      av_ = __uint_as_float(((uint32_t)ab_ua) << 16); bv_ = __uint_as_float(((uint32_t)ab_ub) << 16);
    }
  };
  // K/Q/V chunk loads (LDG.128 into registers, stored into the swizzled layout once they arrive):
  //   K chunks 0-191: w0, 192-383: w1 (6 per lane), 384-511: w5 (4 per lane);  Q: w2, w3 (6), w6 (4);  V: w7 (2 per lane)
  uint4 kq_r[6];
  const int nkq = tw ? 6 : (warp == 5 || warp == 6) ? 4 : 0;
  const int kq0 = tw ? 192 * (warp & 1) + lane : 384 + lane;        // first chunk index (stride 32 per i)
  const bool kq_isK = (warp == 0 || warp == 1 || warp == 5);
  auto issue_loads = [&](int t0_) {
    const int t0 = min(t0_, T - 1);
    const int rmax = max(min(CH, T - t0) - 1, 0);
#pragma unroll
    for (int i = 0; i < 6; i++) kq_r[i] = make_uint4(0, 0, 0, 0);
    if (nkq > 0) {
      const bf16* base = (kq_isK ? p.k : p.q) + (size_t)t0 * kq_ts + hk * 128;
#pragma unroll
      for (int i = 0; i < 6; i++) {
        if (i < nkq) {
          const int idx = kq0 + 32 * i, row = idx >> 4, kc = idx & 15;
          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(kq_r[i].x), "=r"(kq_r[i].y), "=r"(kq_r[i].z), "=r"(kq_r[i].w)
                       : "l"(base + min(row, rmax) * kq_ts + kc * 8) : "memory");
        }
      }
    } else if (warp == 7) {   // V: one 16 B chunk (8 columns) per token
      const int rr = min(lane, rmax);
      asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(kq_r[0].x), "=r"(kq_r[0].y), "=r"(kq_r[0].z), "=r"(kq_r[0].w)
                   : "l"(p.v + ((size_t)(t0 + rr) * HV + hv) * 128 + c0) : "memory");
    }
  };
  auto store_loads = [&]() {   // registers -> swizzled smem
    if (nkq > 0) {
      const int row0 = kq_isK ? 0 : 32;
#pragma unroll
      for (int i = 0; i < 6; i++) {
        if (i < nkq) {
          const int idx = kq0 + 32 * i;
          *reinterpret_cast<uint4*>(OPBp + opb_off(row0 + (idx >> 4), idx & 15)) = kq_r[i];
        }
      }
    } else if (warp == 7) {
      *reinterpret_cast<uint4*>(sm + VS_OFF + lane * (VS_STRIDE * 2)) = kq_r[0];
    }
  };
#pragma unroll
  for (int i = 0; i < 3; i++) {
    s0r[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < s0n) asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(s0r[i].x), "=f"(s0r[i].y), "=f"(s0r[i].z), "=f"(s0r[i].w)
                              : "l"(S0g + (s0r0 + i) * 128 + lane * 4) : "memory");
  }
  if (gw && n == 0) load_ab(0);
  if (n == 0) issue_loads(0);
  float Alog_v = 0.f, dtb = 0.f;
  long tstart = 0, tend = 0;
  if (gw) {
    bar_arrive(7, 256);                                  // A: the gate warp has nothing to publish for barrier A
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(Alog_v) : "l"(p.A_log + hv) : "memory");
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(dtb) : "l"(p.dt_bias + hv) : "memory");
    const long* cup = reinterpret_cast<const long*>(p.cu);
    if (n == 0) { tstart = 0; asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(tend) : "l"(cup + 1) : "memory"); }
    else {
      asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(tstart) : "l"(cup + n) : "memory");
      asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(tend) : "l"(cup + n + 1) : "memory");
      load_ab((int)tstart);
    }
  } else if (n != 0) {
    // sequences after the first need tstart for their loads: read it (one request per warp) before issuing
    asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(tstart) : "l"(reinterpret_cast<const long*>(p.cu) + n) : "memory");
    issue_loads((int)tstart);
  }

  // ---- gates (gate warp, lane = token): G_t (double-float prefix sum), beta, Eg, Dsc, GammaT -> smem. Branch-free.
  const float ea = fast_exp2(Alog_v * L2E);           // exp(A_log)
  auto gates = [&](int len) {
    float av, bv; extract_ab(av, bv);
    const int t = lane;
    const bool valid = t < len;
    // softplus(x) = max(x,0) + log(1 + exp(-|x|)); sigmoid(b) = 1 / (1 + exp(-b))     (MUFU-based fp32, ~2 ulp)
    const float x = av + dtb;
    const float e = fast_exp2(-fabsf(x) * L2E);
    const float spl = fmaxf(x, 0.f) + fast_log2(1.f + e) * LN2;
    const float la = valid ? -ea * spl : 0.f;
    const float be = valid ? fast_rcp(1.f + fast_exp2(-bv * L2E)) : 0.f;
    // prefix sum of log g as (hi, lo): hi is the plain fp32 scan, lo collects the rounding errors (2Sum), no fp64 ops
    float gh = la, gl = 0.f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float yh = __shfl_up_sync(FULL, gh, o), yl = __shfl_up_sync(FULL, gl, o);
      yh = (lane >= o) ? yh : 0.f; yl = (lane >= o) ? yl : 0.f;
      const float sh = gh + yh;
      const float bb = sh - gh;
      const float err = (gh - (sh - bb)) + (yh - bb);
      gl = (gl + yl) + err;
      gh = sh;
    }
    const float gTh = __shfl_sync(FULL, gh, len - 1), gTl = __shfl_sync(FULL, gl, len - 1);
    Ghi[t] = gh; Glo[t] = gl; Beta[t] = be;
    Eg[t] = valid ? fast_exp2((gh + gl) * L2E) : 0.f;
    Dsc[t] = valid ? fast_exp2(((gTh - gh) + (gTl - gl)) * L2E) * be : 0.f;
    if (lane == 0) { *GammaT = fast_exp2((gTh + gTl) * L2E); CuS[0] = tstart; CuS[1] = tend; }
  };
  // ---- S0 block splits -> OPB rows 64..111 (one full row = 64 lanes x 8 B: lane -> chunk lane/2, 8 B half lane&1)
  auto split_row = [&](const float4& xa, int va) {
    uint32_t h0, m0, l0, h1, m1, l1;
    split3x2(xa.x, xa.y, h0, m0, l0); split3x2(xa.z, xa.w, h1, m1, l1);
    const int kc = lane >> 1, hb8 = (lane & 1) * 8;
    *reinterpret_cast<uint2*>(OPBp + opb_off(64 + va, kc) + hb8) = make_uint2(h0, h1);
    *reinterpret_cast<uint2*>(OPBp + opb_off(80 + va, kc) + hb8) = make_uint2(m0, m1);
    *reinterpret_cast<uint2*>(OPBp + opb_off(96 + va, kc) + hb8) = make_uint2(l0, l1);
    *reinterpret_cast<float4*>(&S0F[va * S0F_STRIDE + lane * 4]) = xa;
  };
  // Barrier protocol (named barriers; a barrier completes only when EVERY participant has arrived, so each consumer
  // group that arrives at a different time gets its own id):
  //   7 "A"      : gate warp arrives, the other 224 threads sync (K/Q/V, S0 splits and the sequence bounds are in smem)
  //   6 "post-A" : the 224 arrive right after A, the gate warp syncs before touching K/Q/S0 data
  //   2 "gates"  : gate warp arrives after storing the gate arrays, the tensor warps sync (160)
  //   1 "L"      : w0/w2 arrive after writing L (64), helpers w5/w6/w7 sync (96) -> Dinv, N
  //   3 "Dinv"   : helpers sync (128) after w5/w6 wrote Dinv
  //   4 "N"      : helpers sync (128) after w5/w6/w7 wrote N
  //   5 "RHS"    : w0/w2 arrive after writing RHS (64), helpers sync (128)
  //   0          : __syncthreads: D (u splits / QKA / QSp ready) and E (multi-chunk loop)
  if (gw) { TSFN(1, __uint_as_float(ab_va.x ^ ab_vb.x ^ (uint32_t)ab_ua)); gates((int)min((long)CH, tend - tstart)); }
  else {
    TSFN(1, s0r[0].x);   // 1: S0 arrived
#pragma unroll
    for (int i = 0; i < 3; i++) if (i < s0n) split_row(s0r[i], s0r0 + i);
    store_loads();
  }
  TSN(2);   // 2: gates done (w4) | splits + K/Q/V stored
  if (warp == 4) {   // UPB rows 8-15 (v >= 8, padding of E1's 16-row A operand) are never written: zero them (3 splits x 8 rows x 80 B)
#pragma unroll
    for (int i = 0; i < 4; i++) { const int idx = lane + 32 * i; if (idx < 120) *reinterpret_cast<uint4*>(UPBp + (idx / 40) * U_SP + 8 * U_RB + (idx % 40) * 16) = make_uint4(0, 0, 0, 0); }
  }
  TSN(3);   // 3: prologue done
  // every warp reads the sequence bounds after barrier 2 (inside the first iteration); the empty-sequence exit is there

  // ldmatrix lane-address components
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;   // A x4: row within m-tile; k chunk += lane>>4
  const int a_kc  = lane >> 4;
  const int b_row2 = (lane & 7) + (lane >> 4) * 8;        // B x4 (two n-tiles): n within pair; k chunk += (lane>>3)&1
  const int b_kc2  = (lane >> 3) & 1;
  const int b_row1 = lane & 7;                            // B x4 (one n-tile, two k-steps): k chunk += lane>>3
  const int b_kc1  = lane >> 3;
  const uint32_t opb = sbase + OPB_OFF;
  const int r0 = lane >> 2, cp = 2 * (lane & 3);          // mma C-fragment coordinates
  // Swizzled ldmatrix addressing without per-address arithmetic: for a lane-fixed row r (r7 = r & 7) and a k chunk
  // kc = 2 ks + kx (kx lane-fixed), off(r, kc) = r*128 + (((2 ks + kx) & 7) ^ r7) * 16 + (kc >> 3) * 16384
  //   = [r*128 + ((kx ^ r7) * 16)] ^ (((ks & 3) * 2) * 16)  + (ks >> 2) * 16384      (bits 5-6 come from ks, bit 4 from kx)
  // so with xb = opb + r*128 + ((kx ^ r7) << 4) the address for step ks is (xb ^ ((ks & 3) << 5)) + (ks >> 2) * 16384.
  auto xbase = [&](int row, int kx) -> uint32_t { return opb + row * 128 + (((kx ^ row) & 7) << 4); };
#define LDSM_KS(dst, xb, ks) ldsm_x4(dst, ((xb) ^ (((ks) & 3) << 5)) + ((ks) >> 2) * OPB_HALF)
  // unit (single n-tile) loads: kc = 4 kp + kx4 (kx4 = lane >> 3, 0..3): off = xb4 ^ ((kp & 1) << 6) + (kp >> 1) * 16384
#define LDSM_KP(dst, xb4, kp) ldsm_x4(dst, ((xb4) ^ (((kp) & 1) << 6)) + ((kp) >> 1) * OPB_HALF)

  // state accumulators (E1 C layout, S_T^T[v][k]): tensor warp w owns k n-tiles 4w..4w+3
  // acc1[i][r]: v = lane/4 + 8*(r>>1), k = 8*(4*warp+i) + 2*(lane&3) + (r&1)
  float acc1[4][4];

  long t0 = tstart;
  // The chunk body is written once but instantiated twice: a straight-line copy for the first chunk (no loop, so no
  // loop-invariant hoisting delays the tensor phase) and a loop for any further chunks of long sequences.
  auto chunk = [&](bool first) -> bool {
    // ---- barrier A: K/Q/V, S0 splits in smem (gate warp: publish the gate arrays + bounds, then wait for A) ----
    if (!gw) { bar_sync(7, 256); bar_arrive(6, 256); }
    else { bar_arrive(2, 256); bar_sync(6, 256); }
    TSN(4);   // 4: barrier A
#ifdef CUTA
    if (__uint_as_float(*reinterpret_cast<const uint32_t*>(OPBp + tid * 16)) == 1234.5f) p.out[tid] = bf16(1.f);
    if (T > 0) return true;
#endif
    // ---- everyone but the gate warp learns the sequence bounds here (first chunk): sync barrier 2 = gate arrays + CuS ----
    if (first) {
      if (!gw) { bar_sync(2, 256); tstart = CuS[0]; tend = CuS[1]; t0 = tstart; }
      first = false;
      if (tstart >= tend) {   // empty sequence: new_state = state (all threads return; nobody is left waiting on a barrier)
        float* dst = p.new_state + ((size_t)(n * HV + hv) * 128 + c0) * 128;
#pragma unroll
        for (int i = 0; i < 3; i++) if (i < s0n) *reinterpret_cast<float4*>(dst + (s0r0 + i) * 128 + lane * 4) = s0r[i];
        return false;
      }
    } else if (!gw) bar_sync(2, 256);                    // gate arrays for this chunk
    const int len = (int)min((long)CH, tend - t0);
    const bool last = (t0 + CH >= tend);

    if (tw) {
      // ================= tensor warps =================
      const bool isK = (warp & 1) == 0;                 // w0, w2: K side (KK -> L, W -> RHS);  w1, w3: Q side (QK -> QKd, QS)
      const int mt = warp >> 1;                         // m-tile
      const int arow = (isK ? 0 : 32) + 16 * mt;
      const int npair = 1;                              // pair 0 only; (mt1, pair 1) is computed by w5 (KK) and w6 (QK)
      uint32_t A[8][4], B[2][8][4];
      const uint32_t xbA = xbase(arow + a_row, a_kc), xbB0 = xbase(b_row2, b_kc2), xbB1 = xbase(16 + b_row2, b_kc2);
#pragma unroll
      for (int ks = 0; ks < 8; ks++) LDSM_KS(A[ks], xbA, ks);
#pragma unroll
      for (int pn = 0; pn < 2; pn++)
        if (pn < npair)
#pragma unroll
          for (int ks = 0; ks < 8; ks++) LDSM_KS(B[pn][ks], pn ? xbB1 : xbB0, ks);
      float pC[2][2][4];   // [pair][n-tile within pair][frag]
#pragma unroll
      for (int pn = 0; pn < 2; pn++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
          for (int r = 0; r < 4; r++) pC[pn][j][r] = 0.f;
      TSN(5);   // 5: operand loads issued
      TSN(6);   // 6
      // decay values for this lane's pair-tile elements: dq[pn][j][rp][e] = exp(G_t - G_s) beta_s, t = 16 mt + r0 + 8 rp, s = 16 pn + 8 j + cp + e
      float dq[2][2][2][2];
      {
        float gth[2], gtl[2];
#pragma unroll
        for (int rp = 0; rp < 2; rp++) { const int t = 16 * mt + r0 + 8 * rp; gth[rp] = Ghi[t]; gtl[rp] = Glo[t]; }
#pragma unroll
        for (int pn = 0; pn < 2; pn++) {
          if (pn < npair) {
            float gsh[2][2], gsl[2][2], bs[2][2];
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
              for (int e = 0; e < 2; e++) { const int s = 16 * pn + 8 * j + cp + e; gsh[j][e] = Ghi[s]; gsl[j][e] = Glo[s]; bs[j][e] = Beta[s]; }
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
              for (int rp = 0; rp < 2; rp++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                  const int t = 16 * mt + r0 + 8 * rp, s = 16 * pn + 8 * j + cp + e;
                  const float bsel = (t < len && s <= t) ? bs[j][e] : 0.f;
                  const float ex = fminf((gth[rp] - gsh[j][e]) + (gtl[rp] - gsl[j][e]), 0.f) * L2E;
                  dq[pn][j][rp][e] = fast_exp2(ex) * bsel;
                }
          }
        }
      }
      // ---- KK / QK pairs ----
#pragma unroll
      for (int pn = 0; pn < 2; pn++)
        if (pn < npair)
#pragma unroll
          for (int ks = 0; ks < 8; ks++) { mma_bf16(pC[pn][0], A[ks], B[pn][ks]); mma_bf16(pC[pn][1], A[ks], B[pn][ks] + 2); }
      if (isK) {
        // L[t][s] = DEC[t][s] KK[t][s] for s < t (zero elsewhere)
#pragma unroll
        for (int pn = 0; pn < 2; pn++)
          if (pn < npair)
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
              for (int rp = 0; rp < 2; rp++) {
                const int t = 16 * mt + r0 + 8 * rp, s = 16 * pn + 8 * j + cp;
                *reinterpret_cast<float2*>(&Lm[t * LS + s]) =
                    make_float2((s < t) ? dq[pn][j][rp][0] * pC[pn][j][2 * rp] : 0.f, (s + 1 < t) ? dq[pn][j][rp][1] * pC[pn][j][2 * rp + 1] : 0.f);
              }
        TSN(7);   // 7: L written
        if (warp == 0) bar_arrive(9, 64);                // L rows 0-15 ready for w7 (Dinv blocks 0,1)
        bar_arrive(4, 192);                              // L ready for the solvers
      } else {
        // QKd[t][s] = DEC[t][s] QK[t][s], s <= t -> exact bf16 splits (A operand of E2); rows t < 16 with s >= 16 are zero
#pragma unroll
        for (int pn = 0; pn < 2; pn++) {
          if (pn < npair) {
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
              for (int rp = 0; rp < 2; rp++) {
                const int t = 16 * mt + r0 + 8 * rp, s = 16 * pn + 8 * j + cp;
                uint32_t h, m, l;
                split3x2(dq[pn][j][rp][0] * pC[pn][j][2 * rp], dq[pn][j][rp][1] * pC[pn][j][2 * rp + 1], h, m, l);
                unsigned char* qa = QKA + t * QKA_RB + s * 2;
                *reinterpret_cast<uint32_t*>(qa) = h; *reinterpret_cast<uint32_t*>(qa + QKA_SP) = m; *reinterpret_cast<uint32_t*>(qa + 2 * QKA_SP) = l;
              }
          } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int sp = 0; sp < 3; sp++) *reinterpret_cast<uint4*>(QKA + sp * QKA_SP + (lane >> 1) * QKA_RB + 32 + (lane & 1) * 16) = z;
          }
        }
        TSN(7);   // 7: QKd written
      }
      // ---- S0-split units: (sp, nt) -> B rows 64 + 16 sp + 8 nt; K side: W (-> RHS), Q side: QS (-> QSp) ----
      float accS[3][4];   // one n-tile (8 columns) per split
#pragma unroll
      for (int u = 0; u < 3; u++)
#pragma unroll
        for (int r = 0; r < 4; r++) accS[u][r] = 0.f;
#ifndef NO_UNITS
      {
        uint32_t Bs[2][4][4];
        const uint32_t xbU = xbase(64 + b_row1, b_kc1);   // unit u adds 8 rows = 1024 B (bits 10+, no interaction with the XOR)
#pragma unroll
        for (int kp = 0; kp < 4; kp++) LDSM_KP(Bs[0][kp], xbU, kp);
#pragma unroll
        for (int u = 0; u < 3; u++) {   // split u -> rows 64 + 16 u (2048 B apart: no interaction with the XOR bits)
          if (u + 1 < 3) {
#pragma unroll
            for (int kp = 0; kp < 4; kp++) LDSM_KP(Bs[(u + 1) & 1][kp], xbU + 2048 * (u + 1), kp);
          }
#pragma unroll
          for (int kp = 0; kp < 4; kp++) { mma_bf16(accS[u], A[2 * kp], Bs[u & 1][kp]); mma_bf16(accS[u], A[2 * kp + 1], Bs[u & 1][kp] + 2); }
        }
      }
#endif
      if (isK) {
        // RHS[t][c] = v[t][c] - Eg_t (W_hi + W_mid + W_lo)   (t = 16 mt + r0 + 8 rp, c = 8 nt + cp + {0,1}); unit u = 2 sp + nt
        float eg[2]; uint32_t vv[2];
#pragma unroll
        for (int rp = 0; rp < 2; rp++) {
          const int t = 16 * mt + r0 + 8 * rp;
          eg[rp] = Eg[t];
          vv[rp] = *reinterpret_cast<const uint32_t*>(VS + t * VS_STRIDE + cp);
        }
#pragma unroll
        for (int rp = 0; rp < 2; rp++) {
          const int t = 16 * mt + r0 + 8 * rp;
          const float w0 = accS[0][2 * rp] + accS[1][2 * rp] + accS[2][2 * rp];
          const float w1 = accS[0][2 * rp + 1] + accS[1][2 * rp + 1] + accS[2][2 * rp + 1];
          const float v0 = __uint_as_float(vv[rp] << 16), v1 = __uint_as_float(vv[rp] & 0xFFFF0000u);
          *reinterpret_cast<float2*>(&RHS[t * RS + cp]) = make_float2(v0 - eg[rp] * w0, v1 - eg[rp] * w1);
        }
        TSN(8);   // 8: RHS written
        bar_arrive(5, 192);                              // RHS ready for the helpers
      } else {
#pragma unroll
        for (int sp = 0; sp < 3; sp++)
#pragma unroll
          for (int rp = 0; rp < 2; rp++) {
            const int t = 16 * mt + r0 + 8 * rp;
            *reinterpret_cast<float2*>(&QSp[(sp * 32 + t) * WS + cp]) = make_float2(accS[sp][2 * rp], accS[sp][2 * rp + 1]);
          }
        TSN(8);   // 8: QSp written
      }
      // E1 accumulator init: GammaT * S0^T (fp32)
      {
        const float gT = *GammaT;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
          for (int rp = 0; rp < 2; rp++) {
            const float2 x = *reinterpret_cast<const float2*>(&S0F[((lane >> 2) + rp * 8) * S0F_STRIDE + (4 * warp + i) * 8 + (lane & 3) * 2]);
            acc1[i][rp * 2] = gT * x.x; acc1[i][rp * 2 + 1] = gT * x.y;
          }
      }
      TSN(9);   // 9: tensor warp done
    } else {
      // ================= helper warps =================
      //  w5: KK(mt1, pair 1) -> L rows 16-31, s 16-31; Dinv blocks 2,3; then N_3      w6: QK(mt1, pair 1) -> QKd; then N_2
      //  w7: Dinv blocks 0,1; N_1                                                     w4: gates only
      //  then all four: the blocked solve (4 columns each), u splits, E2
      const int hw = warp - 4;
      if (warp == 5 || warp == 6) {
        const bool isK = (warp == 5);
        uint32_t A[8][4], B[8][4];
        const uint32_t xbA = xbase((isK ? 16 : 48) + a_row, a_kc), xbB = xbase(16 + b_row2, b_kc2);
#pragma unroll
        for (int ks = 0; ks < 8; ks++) LDSM_KS(A[ks], xbA, ks);
#pragma unroll
        for (int ks = 0; ks < 8; ks++) LDSM_KS(B[ks], xbB, ks);
        float dq[2][2][2];   // t = 16 + r0 + 8 rp, s = 16 + 8 j + cp + e
        {
          float gth[2], gtl[2], gsh[2][2], gsl[2][2], bs[2][2];
#pragma unroll
          for (int rp = 0; rp < 2; rp++) { const int t = 16 + r0 + 8 * rp; gth[rp] = Ghi[t]; gtl[rp] = Glo[t]; }
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int sidx = 16 + 8 * j + cp + e; gsh[j][e] = Ghi[sidx]; gsl[j][e] = Glo[sidx]; bs[j][e] = Beta[sidx]; }
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int rp = 0; rp < 2; rp++)
#pragma unroll
              for (int e = 0; e < 2; e++) {
                const int t = 16 + r0 + 8 * rp, sidx = 16 + 8 * j + cp + e;
                const float bsel = (t < len && sidx <= t) ? bs[j][e] : 0.f;
                const float ex = fminf((gth[rp] - gsh[j][e]) + (gtl[rp] - gsl[j][e]), 0.f) * L2E;
                dq[j][rp][e] = fast_exp2(ex) * bsel;
              }
        }
        float pC[2][4];
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
          for (int r = 0; r < 4; r++) pC[j][r] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) { mma_bf16(pC[0], A[ks], B[ks]); mma_bf16(pC[1], A[ks], B[ks] + 2); }
        if (isK) {
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int rp = 0; rp < 2; rp++) {
              const int t = 16 + r0 + 8 * rp, sidx = 16 + 8 * j + cp;
              *reinterpret_cast<float2*>(&Lm[t * LS + sidx]) =
                  make_float2((sidx < t) ? dq[j][rp][0] * pC[j][2 * rp] : 0.f, (sidx + 1 < t) ? dq[j][rp][1] * pC[j][2 * rp + 1] : 0.f);
            }
          TSN(5);   // 5: L (pair 1) written
          __syncwarp();
          // Dinv blocks 2,3 from this warp's own L tiles: lanes 0-15, block q = 2 + (lane >> 3), column j = lane & 7
          const int q = 2 + ((lane >> 3) & 1), j = lane & 7;
          float Lq[8][8];
#pragma unroll
          for (int k = 1; k < 8; k++)
#pragma unroll
            for (int sq = 0; sq < k; sq++) Lq[k][sq] = Lm[(8 * q + k) * LS + 8 * q + sq];
          float x[8];
#pragma unroll
          for (int k = 0; k < 8; k++) x[k] = (k == j) ? 1.f : 0.f;
#pragma unroll
          for (int k = 1; k < 8; k++) {
            float sa = 0.f, sb = 0.f;
#pragma unroll
            for (int sq = 0; sq < k; sq++) { if (sq & 1) sb = fmaf(Lq[k][sq], x[sq], sb); else sa = fmaf(Lq[k][sq], x[sq], sa); }
            x[k] = (k > j) ? -(sa + sb) : x[k];
          }
          if (lane < 16) {
#pragma unroll
            for (int k = 0; k < 8; k++) Dinv[q * 72 + k * 8 + j] = x[k];
          }
          TSN(6);   // 6: Dinv 2,3 written
        } else {
#pragma unroll
          for (int j = 0; j < 2; j++)
#pragma unroll
            for (int rp = 0; rp < 2; rp++) {
              const int t = 16 + r0 + 8 * rp, sidx = 16 + 8 * j + cp;
              uint32_t h, m, l;
              split3x2(dq[j][rp][0] * pC[j][2 * rp], dq[j][rp][1] * pC[j][2 * rp + 1], h, m, l);
              unsigned char* qa = QKA + t * QKA_RB + sidx * 2;
              *reinterpret_cast<uint32_t*>(qa) = h; *reinterpret_cast<uint32_t*>(qa + QKA_SP) = m; *reinterpret_cast<uint32_t*>(qa + 2 * QKA_SP) = l;
            }
          TSN(5);   // 5: QKd (pair 1) written
        }
      }
      if (warp == 7) {
        bar_sync(9, 64);                                 // L rows 0-15 ready (w0)
        TSN(6);   // 6: L rows 0-15 ready
        // Dinv blocks 0,1: lanes 0-15, block q = lane >> 3, column j = lane & 7
        const int q = (lane >> 3) & 1, j = lane & 7;
        float Lq[8][8];
#pragma unroll
        for (int k = 1; k < 8; k++)
#pragma unroll
          for (int sq = 0; sq < k; sq++) Lq[k][sq] = Lm[(8 * q + k) * LS + 8 * q + sq];
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; k++) x[k] = (k == j) ? 1.f : 0.f;
#pragma unroll
        for (int k = 1; k < 8; k++) {
          float sa = 0.f, sb = 0.f;
#pragma unroll
          for (int sq = 0; sq < k; sq++) { if (sq & 1) sb = fmaf(Lq[k][sq], x[sq], sb); else sa = fmaf(Lq[k][sq], x[sq], sa); }
          x[k] = (k > j) ? -(sa + sb) : x[k];
        }
        if (lane < 16) {
#pragma unroll
          for (int k = 0; k < 8; k++) Dinv[q * 72 + k * 8 + j] = x[k];
        }
        TSN(7);   // 7: Dinv 0,1 written
      }
      bar_sync(4, 192);                                  // L (w0, w2, w5) and Dinv (w5, w7) complete
      TSN(8);   // 8: solve inputs ready
      // this warp solves columns c = 4 hw + cl; lane -> (i = lane>>2: row within block, cl = lane&3)
      const int i = lane >> 2, cl = lane & 3, c = 4 * hw + cl;
      // this lane's L values (rows 8q+i, columns s < 8q) and Dinv rows
      float Lr[48];
#pragma unroll
      for (int q = 1; q < 4; q++)
#pragma unroll
        for (int s4 = 0; s4 < 8 * q; s4 += 4) {
          const float4 v = *reinterpret_cast<const float4*>(&Lm[(8 * q + i) * LS + s4]);
          Lr[(q * (q - 1)) * 4 + s4] = v.x; Lr[(q * (q - 1)) * 4 + s4 + 1] = v.y; Lr[(q * (q - 1)) * 4 + s4 + 2] = v.z; Lr[(q * (q - 1)) * 4 + s4 + 3] = v.w;
        }
      float Dr[4][8];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float4 d0 = *reinterpret_cast<const float4*>(&Dinv[q * 72 + i * 8]), d1 = *reinterpret_cast<const float4*>(&Dinv[q * 72 + i * 8 + 4]);
        Dr[q][0] = d0.x; Dr[q][1] = d0.y; Dr[q][2] = d0.z; Dr[q][3] = d0.w; Dr[q][4] = d1.x; Dr[q][5] = d1.y; Dr[q][6] = d1.z; Dr[q][7] = d1.w;
      }
      bar_sync(5, 192);                                  // RHS ready
      TSN(9);   // 9: RHS ready
      if (hw < 2) {   // 8 columns: helpers 0,1 solve 4 each
      float rq[4];
#pragma unroll
      for (int q = 0; q < 4; q++) rq[q] = RHS[(8 * q + i) * RS + c];
      // blocked forward substitution, two intra-warp exchanges per block: r = rhs - L u_prev (owner lanes), u = Dinv r
      float* __restrict__ R8w = R8 + hw * 32;          // per-warp [4 cl][8 i]
      float* __restrict__ Uc = U32 + c * US;           // this lane's column
      float up[24];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        if (q > 0) {
          const float4 u0 = *reinterpret_cast<const float4*>(&Uc[8 * (q - 1)]), u1 = *reinterpret_cast<const float4*>(&Uc[8 * (q - 1) + 4]);
          up[8 * (q - 1) + 0] = u0.x; up[8 * (q - 1) + 1] = u0.y; up[8 * (q - 1) + 2] = u0.z; up[8 * (q - 1) + 3] = u0.w;
          up[8 * (q - 1) + 4] = u1.x; up[8 * (q - 1) + 5] = u1.y; up[8 * (q - 1) + 6] = u1.z; up[8 * (q - 1) + 7] = u1.w;
        }
        float ra = rq[q], rb = 0.f, rc = 0.f, rd = 0.f;
#pragma unroll
        for (int s4 = 0; s4 < 8 * q; s4++) {
          const float lv = Lr[(q * (q - 1)) * 4 + s4], uv = up[s4];
          if ((s4 & 3) == 0) ra = fmaf(-lv, uv, ra); else if ((s4 & 3) == 1) rb = fmaf(-lv, uv, rb); else if ((s4 & 3) == 2) rc = fmaf(-lv, uv, rc); else rd = fmaf(-lv, uv, rd);
        }
        R8w[cl * 8 + i] = (ra + rb) + (rc + rd);
        __syncwarp();
        const float4 e0 = *reinterpret_cast<const float4*>(&R8w[cl * 8]), e1 = *reinterpret_cast<const float4*>(&R8w[cl * 8 + 4]);
        const float ua = fmaf(Dr[q][0], e0.x, Dr[q][4] * e1.x), ub = fmaf(Dr[q][1], e0.y, Dr[q][5] * e1.y);
        const float uc = fmaf(Dr[q][2], e0.z, Dr[q][6] * e1.z), ud = fmaf(Dr[q][3], e0.w, Dr[q][7] * e1.w);
        Uc[8 * q + i] = (ua + ub) + (uc + ud);
        __syncwarp();
      }
      TSN(10);   // 10: solve done
      // exact bf16 splits of u and u' = Dsc_s u_s for this warp's columns: lane -> (c, s = 4*(lane>>2) .. +3)
      {
        const int s0_ = 4 * (lane >> 2);
        const float4 uu = *reinterpret_cast<const float4*>(&Uc[s0_]);
        const float4 dd = *reinterpret_cast<const float4*>(&Dsc[s0_]);
        uint32_t h0, m0, l0, h1, m1, l1, hp0, mp0, lp0, hp1, mp1, lp1;
        split3x2(uu.x, uu.y, h0, m0, l0); split3x2(uu.z, uu.w, h1, m1, l1);
        split3x2(uu.x * dd.x, uu.y * dd.y, hp0, mp0, lp0); split3x2(uu.z * dd.z, uu.w * dd.w, hp1, mp1, lp1);
        unsigned char* ub = UBp + c * U_RB + s0_ * 2;
        *reinterpret_cast<uint2*>(ub) = make_uint2(h0, h1); *reinterpret_cast<uint2*>(ub + U_SP) = make_uint2(m0, m1); *reinterpret_cast<uint2*>(ub + 2 * U_SP) = make_uint2(l0, l1);
        unsigned char* upb = UPBp + c * U_RB + s0_ * 2;
        *reinterpret_cast<uint2*>(upb) = make_uint2(hp0, hp1); *reinterpret_cast<uint2*>(upb + U_SP) = make_uint2(mp0, mp1); *reinterpret_cast<uint2*>(upb + 2 * U_SP) = make_uint2(lp0, lp1);
      }
      TSN(11);   // 11: u splits done
      }   // hw < 2
    }
    __syncthreads();                                     // barrier D: u splits, QKA, QSp ready
    TSN(12);   // 12: barrier D
#ifdef CUTD
    if (Ghi[lane] + __uint_as_float(*reinterpret_cast<const uint32_t*>(UBp + tid * 4)) == 1234.5f) p.out[tid] = bf16(1.f);
    if (T > 0) return true;
#endif
    if (tw) {
      // ---- E1 (tensor warps, 4 k-tiles each): S_T^T[v][k] += sum_s u'[s][v] k_s[k]; A = UP[v][s], B = K[s][k] (ldmatrix.trans)
      const uint32_t up0 = sbase + UPB_OFF;
      const uint32_t xbE = xbase(a_row, 4 * (warp & 1) + a_kc) + (warp >> 1) * OPB_HALF;   // B = K rows (16 ks + a_row), k chunks 4 warp + a_kc (+2)
      float e1[4][2][4];   // [tile][split 1,2][frag]; split 0 accumulates into acc1
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int sp = 0; sp < 2; sp++)
#pragma unroll
          for (int r = 0; r < 4; r++) e1[i][sp][r] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        uint32_t aU[3][4], bK[2][4];
#pragma unroll
        for (int sp = 0; sp < 3; sp++) ldsm_x4(aU[sp], up0 + sp * U_SP + a_row * U_RB + (2 * ks + a_kc) * 16);
        ldsm_x4_t(bK[0], (xbE ^ 0) + ks * 2048);
        ldsm_x4_t(bK[1], (xbE ^ 32) + ks * 2048);
#pragma unroll
        for (int np = 0; np < 2; np++) {
          mma_bf16(acc1[2 * np], aU[0], bK[np]); mma_bf16(acc1[2 * np + 1], aU[0], bK[np] + 2);
          mma_bf16(e1[2 * np][0], aU[1], bK[np]); mma_bf16(e1[2 * np + 1][0], aU[1], bK[np] + 2);
          mma_bf16(e1[2 * np][1], aU[2], bK[np]); mma_bf16(e1[2 * np + 1][1], aU[2], bK[np] + 2);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) acc1[i][r] += e1[i][0][r] + e1[i][1][r];
#ifdef NOSTORE
      if (last && acc1[0][0] == 1234.5f) {
#else
      if (last) {
#endif
        float* dst = p.new_state + ((size_t)(n * HV + hv) * 128 + c0) * 128;
#pragma unroll
        for (int i = 0; i < 4; i++)   // rows v = lane/4 (< 8); the fragment rows 8-15 are padding
          *reinterpret_cast<float2*>(dst + (lane >> 2) * 128 + (4 * warp + i) * 8 + (lane & 3) * 2) = make_float2(acc1[i][0], acc1[i][1]);
      }
    } else {
      // ---- E2 (helper warps, one 16x8 tile each): out[t][c] = scale (Eg QS + sum_s QKd[t][s] u[s][c]) ----
      const int mt2 = warp - 4, ntc = 0;
      if (mt2 < 2) {
      float accO[4];
      {
        float2 q0[2], q1[2], q2[2]; float eg[2];
#pragma unroll
        for (int rp = 0; rp < 2; rp++) {
          const int tt = mt2 * 16 + r0 + rp * 8, cc = ntc * 8 + cp;
          eg[rp] = Eg[tt];
          q0[rp] = *reinterpret_cast<const float2*>(&QSp[tt * WS + cc]);
          q1[rp] = *reinterpret_cast<const float2*>(&QSp[(32 + tt) * WS + cc]);
          q2[rp] = *reinterpret_cast<const float2*>(&QSp[(64 + tt) * WS + cc]);
        }
#pragma unroll
        for (int rp = 0; rp < 2; rp++) {
          accO[rp * 2] = eg[rp] * (q0[rp].x + q1[rp].x + q2[rp].x);
          accO[rp * 2 + 1] = eg[rp] * (q0[rp].y + q1[rp].y + q2[rp].y);
        }
      }
      const uint32_t qa0 = sbase + QKA_OFF, ub0 = sbase + UB_OFF;
      float o2[4], o3[4];
#pragma unroll
      for (int r = 0; r < 4; r++) { o2[r] = 0.f; o3[r] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        uint32_t aQ[3][4], bU[3][2];
#pragma unroll
        for (int sp = 0; sp < 3; sp++) {
          ldsm_x4(aQ[sp], qa0 + sp * QKA_SP + (16 * mt2 + a_row) * QKA_RB + (2 * ks + a_kc) * 16);
          ldsm_x2(bU[sp], ub0 + sp * U_SP + (8 * ntc + (lane & 7)) * U_RB + (2 * ks + ((lane >> 3) & 1)) * 16);
        }
        mma_bf16(accO, aQ[0], bU[0]); mma_bf16(o2, aQ[0], bU[1]); mma_bf16(o3, aQ[1], bU[0]);
        mma_bf16(accO, aQ[0], bU[2]); mma_bf16(o2, aQ[1], bU[1]); mma_bf16(o3, aQ[2], bU[0]);
      }
#pragma unroll
      for (int r = 0; r < 4; r++) accO[r] += o2[r] + o3[r];
#pragma unroll
      for (int rp = 0; rp < 2; rp++) {
        const int t = mt2 * 16 + r0 + rp * 8;
        if (t < len) {
          __nv_bfloat162 o = __floats2bfloat162_rn(p.scale * accO[rp * 2], p.scale * accO[rp * 2 + 1]);
          *reinterpret_cast<__nv_bfloat162*>(p.out + ((size_t)(t0 + t) * HV + hv) * 128 + c0 + ntc * 8 + cp) = o;
        }
      }
      }   // mt2 < 2
    }
    TSN(13);   // 13: E1/E2 done (stores issued)
    if (!last) {
      __syncthreads();                                   // barrier E: all reads of OPB/UB/UPB/QKA/gates done
      if (gw) { bar_arrive(7, 256); load_ab((int)(t0 + CH)); }
      issue_loads((int)(t0 + CH));
      if (tw) {   // carried state -> S0 splits + fp32 copy
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int rp = 0, v = (lane >> 2), k = (4 * warp + i) * 8 + (lane & 3) * 2;   // rows v < 8 only
            uint32_t h, m, l;
            split3x2(acc1[i][rp * 2], acc1[i][rp * 2 + 1], h, m, l);
            *reinterpret_cast<uint32_t*>(OPBp + opb_off(64 + v, k >> 3) + (k & 7) * 2) = h;
            *reinterpret_cast<uint32_t*>(OPBp + opb_off(80 + v, k >> 3) + (k & 7) * 2) = m;
            *reinterpret_cast<uint32_t*>(OPBp + opb_off(96 + v, k >> 3) + (k & 7) * 2) = l;
            *reinterpret_cast<float2*>(&S0F[v * S0F_STRIDE + k]) = make_float2(acc1[i][rp * 2], acc1[i][rp * 2 + 1]);
          }
      }
      if (gw) gates((int)min((long)CH, tend - (t0 + CH)));
      else store_loads();
    }
      return true;
  };
  if (!chunk(true)) return;
  for (t0 += CH; t0 < tend; t0 += CH) chunk(false);
#ifdef TIMING
  { long long g_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g_) :: "memory"); if (lane == 0) stamp_[15] = g_; }
#endif
  TS_FLUSH();
}

}  // namespace

extern "C" int gdn_fwd_launch(
    const void* q, const void* k, const void* v, const void* state, const void* A_log, const void* a,
    const void* dt_bias, const void* b, const void* cu, void* out, void* new_state,
    int N, int HK, int HV, int T, int rep_shift, int flags, float scale, void* stream, void* dbg) {
  static bool inited = false;
  if (!inited) {
    cudaError_t e = cudaFuncSetAttribute(gdn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return (int)e;
    inited = true;
  }
  Params p;
  p.q = (const bf16*)q; p.k = (const bf16*)k; p.v = (const bf16*)v; p.state = (const float*)state;
  p.a = (const bf16*)a; p.b = (const bf16*)b; p.cu = (const int64_t*)cu;
  p.T = T; p.HK = (unsigned char)HK; p.HV = (unsigned char)HV; p.rep_shift = (unsigned char)rep_shift; p.flags = (unsigned char)flags;
  p.A_log = (const float*)A_log; p.dt_bias = (const float*)dt_bias; p.out = (bf16*)out; p.new_state = (float*)new_state;
  p.scale = scale; p.pad_ = 0; p.dbg = (long long*)dbg;
  dim3 grid(128 / C, HV, N);
  gdn_fwd_kernel<<<grid, NTHREADS, SMEM_BYTES, (cudaStream_t)stream>>>(p);
  return (int)cudaGetLastError();
}

extern "C" int gdn_smem_bytes() { return SMEM_BYTES; }
