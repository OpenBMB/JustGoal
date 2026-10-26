// ---------------------------------------------------------------------------
// Kimi-Delta-Attention forward for B200 (sm_100a) -- two-kernel design.
//
// The chunk-parallel delta rule splits cleanly into
//   (1) per-chunk work that does NOT depend on the recurrent state
//       (l2 norm, gate/cumsum, Kbar/Qhat/Khat, W, Ah and the UT-transform
//        inverse).  This is embarrassingly parallel over 256*64 chunk-heads,
//        so it runs as its own kernel with very high occupancy, where its long
//        dependency chains are hidden by other blocks.
//   (2) the sequential state scan, which only has 64 independent chains.  It
//       runs as a persistent kernel with the value dimension split over NV=2
//       CTAs (-> 128 CTAs) and keeps the state, Yt and Ut entirely in
//       registers: an mma accumulator already has exactly the layout of an mma
//       A-fragment, so M1 -> M3 -> {M4, M5} chain without ever touching shared
//       memory.  Only the B operands come from smem.
//
// Math (per head, chunk of C steps, b_t = inclusive cumulative log decay,
// symmetric per-channel offset c = b_{C-1}/2 so no exponent overflows):
//   Kbar = k e^{b-c}  Khat = k e^{c-b}  Qhat = q e^{b-c} scale  S' = Diag(e^c) S
//   W = strict_tril(Kbar Khat^T)         Ah = tril(Qhat Khat^T)
//   U = (I + Diag(beta) W)^{-1} Diag(beta) (V - Kbar S')
//   O = Qhat S' + Ah U                   S_next = Diag(e^c)(S' + Khat^T U)
// everything transposed: St[v][d] = S[d][v].
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef LGMIN
#define LGMIN (-5.0f)   // floor on per-step log decay; bounds the chunk span to 32*5=160
#endif
#ifndef HALF_SHARED
#define HALF_SHARED 0
#endif
#ifndef TPB
#define TPB 256
#endif
#ifndef NGROUP
#define NGROUP 1
#endif

#define DEV __device__ __forceinline__

static constexpr int D   = 128;
static constexpr int DVV = 128;
static constexpr int C   = 32;
#ifndef NVAL
#define NVAL 2
#endif
static constexpr int NV  = NVAL;
static constexpr int VP  = DVV / NV;
static constexpr int NMW = VP / 16;      // one math warp per 16 value rows
#ifndef LD128V
#define LD128V 136
#endif
static constexpr int LD128 = LD128V;   // row padding; sets the ldmatrix bank stride
static constexpr int LD64  = 72;
static constexpr int LD32  = 40;
static constexpr int LDF32 = 36;

using bf16 = __nv_bfloat16;

// ------------------------------- ptx helpers -------------------------------
DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void cp_async16(void* d_, const void* s_) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(d_)), "l"(s_));
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> DEV void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
DEV void bar_sync(int id, int cnt) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt)); }

DEV void ldm_a(uint32_t (&r)[4], const bf16* t, int ld, int m0, int k0, int lane) {
  const bf16* p = t + (m0 + (lane & 15)) * ld + k0 + ((lane >> 4) << 3);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
DEV void ldm_b(uint32_t (&r)[2], const bf16* t, int ld, int n0, int k0, int lane) {
  const bf16* p = t + (n0 + (lane & 7)) * ld + k0 + (((lane >> 3) & 1) << 3);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
DEV void ldm_bt(uint32_t (&r)[2], const bf16* t, int ld, int n0, int k0, int lane) {
  const bf16* p = t + (k0 + (lane & 15)) * ld + n0;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
// Two B n-tiles per instruction (halves the ldmatrix issue count).
// non-transposed (N x K source):  b[nt] = {r0, r2},  b[nt+1] = {r1, r3}
DEV void ldm_b4(uint32_t (&r)[4], const bf16* t, int ld, int n0, int k0, int lane) {
  const bf16* p = t + (n0 + (lane & 15)) * ld + k0 + ((lane >> 4) << 3);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
// transposed (K x N source):      b[nt] = {r0, r1},  b[nt+1] = {r2, r3}
DEV void ldm_bt4(uint32_t (&r)[4], const bf16* t, int ld, int n0, int k0, int lane) {
  const bf16* p = t + (k0 + (lane & 15)) * ld + n0 + ((lane >> 4) << 3);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
DEV void mma2(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Store four 8x8 bf16 fragments to shared memory *transposed*, one instruction.
// Turns the 16 scattered 2-byte transposing stores of the output stage into 2.
DEV void stm_trans4(bf16* t, int ld, int row, int col, const uint32_t (&x)[4]) {
  const bf16* p = t + row * ld + col;
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :: "r"(smem_u32(p)), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}
DEV void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// An mma accumulator for output columns [16*ks,16*ks+16) is *bit-for-bit* the
// A-fragment of that 16x16 block -- no shuffles, no shared memory.
DEV void acc_to_a(uint32_t (&a)[4], const float (&lo)[4], const float (&hi)[4]) {
  a[0] = pack2(lo[0], lo[1]);
  a[1] = pack2(lo[2], lo[3]);
  a[2] = pack2(hi[0], hi[1]);
  a[3] = pack2(hi[2], hi[3]);
}
DEV float softplus_fast(float x) { return fmaxf(x, 0.f) + __logf(1.f + __expf(-fabsf(x))); }
DEV int acc_row(int lane, int e) { return (lane >> 2) + ((e >> 1) << 3); }
DEV int acc_col(int lane, int e) { return ((lane & 3) << 1) + (e & 1); }

// ===========================================================================
// Kernel 1: state-independent per-chunk preparation.
// grid = (NC, H), block = 128 threads (4 warps).
// ===========================================================================
struct SmemP {
  // KQ rows [0,C) hold k then Kbar; rows [C,2C) hold q then Qhat;
  // Kh holds g then Khat.  Every rewrite is in place and per-thread, so the
  // input tiles and the outputs can share storage (52 KB -> 4 blocks/SM).
  bf16  KQ[2 * C][LD128];
  bf16  Kh[C][LD128];
  // Bcum/Cof are dead by the time W is computed, so they share storage with
  // the W / X workspace (keeps the block at 4 CTAs per SM).
  union {
    struct { float Wf[2 * C][LDF32]; float Xi[C][LDF32]; } inv;
  } u;
#ifndef SCR_LD
#define SCR_LD LD32
#endif
  // SCR_LD < LD32 is a TIMING PROBE ONLY (rows alias; results are wrong) used
  // to price what extra prep CTAs/SM would buy before refactoring for real.
  bf16  Xb[C][SCR_LD], Pb[C][SCR_LD], Zb[C][SCR_LD], Lb[C][SCR_LD], Gb[C][SCR_LD];
  float Bt[C + 8], Rq[C], Rk[C], Ecs[D];
  float2 Psum[2][D / 2];
#ifdef PREP_PAD
  char pad[PREP_PAD];              // occupancy probe: unused, shifts CTAs/SM
#endif
};

// --- producer/consumer handshake between the prep and scan kernels ---------
// prep publishes one 32-bit epoch per (chunk, head) once its global stores are
// visible device-wide; the scan acquires it before issuing that chunk's loads.
// With prep dispatched chunk-major the two kernels stay a few chunks apart, so
// the intermediate tiles are still in L2 when the scan reads them.
__device__ __forceinline__ void pipe_publish(unsigned* f, unsigned epoch) {
  asm volatile("fence.release.gpu;\n" ::: "memory");
  asm volatile("st.global.relaxed.gpu.u32 [%0], %1;\n" :: "l"(f), "r"(epoch) : "memory");
}
__device__ __forceinline__ void pipe_wait(const unsigned* f, unsigned epoch) {
  unsigned got;
  do {
    asm volatile("ld.global.relaxed.gpu.u32 %0, [%1];\n" : "=r"(got) : "l"(f) : "memory");
    if (got == epoch) break;
    __nanosleep(48);
  } while (true);
  asm volatile("fence.acquire.gpu;\n" ::: "memory");
}

__global__ __launch_bounds__(128, 4) void kda_prep_kernel(
    const bf16* __restrict__ gq, const bf16* __restrict__ gk, const bf16* __restrict__ gg,
    const bf16* __restrict__ gbeta, const float* __restrict__ gAlog,
    const float* __restrict__ gdtb,
    bf16* __restrict__ oKQ, bf16* __restrict__ oKh, bf16* __restrict__ oTm,
    bf16* __restrict__ oAh, float* __restrict__ oEc,
    unsigned* __restrict__ gprod, unsigned epoch,
    int T, int H, float scale, int chunk0) {
  extern __shared__ __align__(1024) char raw[];
  SmemP& sm = *reinterpret_cast<SmemP*>(raw);
#ifdef PIPE
  const int chunk = chunk0 + blockIdx.y, ih = blockIdx.x;   // chunk-major dispatch
#else
  const int chunk = chunk0 + blockIdx.x, ih = blockIdx.y;
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long base = ((long)chunk * C) * (long)H * D + (long)ih * D;

  // ---- load q,k,g tiles (128 threads * 16B * 2 = one 8KB tile) ------------
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    int idx = p * 128 + tid, t = idx >> 4, d = (idx & 15) << 3;
    const long off = base + (long)t * H * D + d;
    cp_async16(&sm.KQ[C + t][d], gq + off);
    cp_async16(&sm.KQ[t][d], gk + off);
  }
  cp_commit();
  if (tid < C)
    sm.Bt[tid] = __bfloat162float(gbeta[((long)chunk * C + tid) * H + ih]);
  const float Ahead = __expf(gAlog[ih]);
  const float2 dtb2 = *(const float2*)(gdtb + ih * D + ((tid & 63) << 1));
  cp_wait<0>();
  __syncthreads();

#ifndef PA_NOL2
  // ---- l2 norms (4 threads per row) --------------------------------------
  {
    int row = tid >> 2, sub = tid & 3;
    float sq = 0.f, sk = 0.f;
#pragma unroll
    for (int h8 = 0; h8 < 4; ++h8) {
      int d = sub * 32 + h8 * 8;
      uint4 qa = *(const uint4*)&sm.KQ[C + row][d];
      uint4 ka = *(const uint4*)&sm.KQ[row][d];
      const __nv_bfloat162* qp = (const __nv_bfloat162*)&qa;
      const __nv_bfloat162* kp = (const __nv_bfloat162*)&ka;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float2 a = __bfloat1622float2(qp[u]), b = __bfloat1622float2(kp[u]);
        sq += a.x * a.x + a.y * a.y;
        sk += b.x * b.x + b.y * b.y;
      }
    }
#pragma unroll
    for (int m = 1; m < 4; m <<= 1) {
      sq += __shfl_xor_sync(0xffffffff, sq, m);
      sk += __shfl_xor_sync(0xffffffff, sk, m);
    }
    if (sub == 0) { sm.Rq[row] = rsqrtf(fmaxf(sq, 1e-12f)); sm.Rk[row] = rsqrtf(fmaxf(sk, 1e-12f)); }
  }
#else
  if (tid < C) { sm.Rq[tid] = 1.f; sm.Rk[tid] = 1.f; }
#endif
  __syncthreads();

  // ---- gate: one channel per thread, cumsum along t is thread-local -------
  // Each thread owns *two adjacent channels* and half of the timesteps, so the
  // cumulative sum stays thread-local while every shared access is 4B wide
  // (a 2B access only fills half a wavefront).
  {
    const int cp = tid & 63, d = cp << 1, half = tid >> 6, t0 = half * (C / 2);
    float2 b[C / 2];
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < C / 2; ++j) {
      float2 gv = __bfloat1622float2(
          *(const __nv_bfloat162*)(gg + base + (long)(t0 + j) * H * D + d));
      // Match the bounded sigmoid gate used by the mathematical reference.
#ifdef PA_NOGATE1
      acc.x += -0.05f; acc.y += -0.05f;
#else
      acc.x += LGMIN / (1.f + expf(-Ahead * (gv.x + dtb2.x)));
      acc.y += LGMIN / (1.f + expf(-Ahead * (gv.y + dtb2.y)));
#endif
      b[j] = acc;
    }
    sm.Psum[half][cp] = acc;
    __syncthreads();
    const float2 p0 = sm.Psum[0][cp], p1 = sm.Psum[1][cp];
    const float2 cof = make_float2(0.5f * (p0.x + p1.x), 0.5f * (p0.y + p1.y));
    const float2 off = half ? p0 : make_float2(0.f, 0.f);
    if (half == 0) *(float2*)&sm.Ecs[d] = make_float2(__expf(cof.x), __expf(cof.y));
#ifndef PA_NOGATE2
#pragma unroll
    for (int j = 0; j < C / 2; ++j) {
      const int t = t0 + j;
      float bx = b[j].x + off.x - cof.x, by = b[j].y + off.y - cof.y;
      float epx = __expf(bx), epy = __expf(by);
      float enx = __expf(-bx), eny = __expf(-by);
      float rk = sm.Rk[t], rq = sm.Rq[t] * scale;
      float2 kv = __bfloat1622float2(*(const __nv_bfloat162*)&sm.KQ[t][d]);
      float2 qv = __bfloat1622float2(*(const __nv_bfloat162*)&sm.KQ[C + t][d]);
      // smem copies stay in the symmetric-offset form that W / Ah need ...
      *(__nv_bfloat162*)&sm.KQ[t][d]     = __floats2bfloat162_rn(kv.x * rk * epx, kv.y * rk * epy);
      *(__nv_bfloat162*)&sm.KQ[C + t][d] = __floats2bfloat162_rn(qv.x * rq * epx, qv.y * rq * epy);
      *(__nv_bfloat162*)&sm.Kh[t][d]     = __floats2bfloat162_rn(kv.x * rk * enx, kv.y * rk * eny);
      // ... while the copies the scan consumes get e^c folded in (and Kbar
      // negated) and go straight to global, avoiding an smem round trip.
      float ecx = __expf(cof.x), ecy = __expf(cof.y);
      const long go_ = ((long)chunk * H + ih);
#ifndef PA_NOWRITE
      *(__nv_bfloat162*)(oKQ + go_ * (2 * C * D) + (long)t * D + d) =
          __floats2bfloat162_rn(-kv.x * rk * epx * ecx, -kv.y * rk * epy * ecy);
      *(__nv_bfloat162*)(oKQ + go_ * (2 * C * D) + (long)(C + t) * D + d) =
          __floats2bfloat162_rn(qv.x * rq * epx * ecx, qv.y * rq * epy * ecy);
#endif
#ifndef PA_NOKHW
      *(__nv_bfloat162*)(oKh + go_ * (C * D) + (long)t * D + d) =
          __floats2bfloat162_rn(kv.x * rk * enx * ecx, kv.y * rk * eny * ecy);
#endif
    }
#endif
  }
  __syncthreads();

#ifndef PA_NOM2
  // ---- [W ; Ah] = KQ * Kh^T  (M=2C, N=C, K=D), warp tile 16 x 32 ---------
  {
    float acc[4][4] = {};
    uint32_t af[4], bfr[2][4];
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
      ldm_a(af, &sm.KQ[0][0], LD128, warp * 16, ks * 16, lane);
#pragma unroll
      for (int gp = 0; gp < 2; ++gp)
        ldm_b4(bfr[gp], &sm.Kh[0][0], LD128, gp * 16, ks * 16, lane);
#pragma unroll
      for (int gp = 0; gp < 2; ++gp) {
        mma2(acc[2 * gp],     af, bfr[gp][0], bfr[gp][2]);
        mma2(acc[2 * gp + 1], af, bfr[gp][1], bfr[gp][3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        sm.u.inv.Wf[warp * 16 + acc_row(lane, e)][nt * 8 + acc_col(lane, e)] = acc[nt][e];
  }
  __syncthreads();

#endif
  // ---- L = diag(beta) strict_tril(W),  X = I - L,  Ah masked -------------
  {
    const int r = tid >> 2, c0 = (tid & 3) * 8;
    const float bt = sm.Bt[r];
    float4 w0 = *(const float4*)&sm.u.inv.Wf[r][c0];
    float4 w1 = *(const float4*)&sm.u.inv.Wf[r][c0 + 4];
    const float* wp0 = (const float*)&w0; const float* wp1 = (const float*)&w1;
    float xi[8]; __nv_bfloat162 lb[4], xb[4];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      int c = c0 + u;
      float Lv = (c < r) ? bt * ((u < 4) ? wp0[u] : wp1[u - 4]) : 0.f;
      xi[u] = (c == r) ? 1.f : -Lv;
      if (u & 1) {
        int cm = c - 1;
        float Lm = (cm < r) ? bt * ((u - 1 < 4) ? wp0[u - 1] : wp1[u - 5]) : 0.f;
        lb[u >> 1] = __floats2bfloat162_rn(Lm, Lv);
        xb[u >> 1] = __floats2bfloat162_rn(xi[u - 1], xi[u]);
      }
    }
    *(uint4*)&sm.Lb[r][c0] = *(const uint4*)lb;
    *(uint4*)&sm.Pb[r][c0] = *(const uint4*)lb;
    *(uint4*)&sm.Xb[r][c0] = *(const uint4*)xb;
    *(float4*)&sm.u.inv.Xi[r][c0] = *(const float4*)xi;
    *(float4*)&sm.u.inv.Xi[r][c0 + 4] = *(const float4*)(xi + 4);
  }
  __syncthreads();

#ifndef PA_NOINV
  // ---- (I+L)^-1 : 16x16 diagonal blocks by Neumann doubling, tensor cores -
  {
    const int o0 = warp * 16;
    const bool act = (warp < 2);
    auto blk = [&](const bf16* A, const bf16* Bk, int am, int ak, int bn, int bk,
                   float (&acc)[2][4]) {
      uint32_t af[4], bfr[2][2];
      ldm_a(af, A, LD32, am, ak, lane);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) ldm_bt(bfr[nt], Bk, LD32, bn + nt * 8, bk, lane);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) mma16816(acc[nt], af, bfr[nt]);
    };
#pragma unroll 1
    for (int it = 0; it < 3; ++it) {
      if (act) {
        float acc[2][4] = {};
        blk(&sm.Pb[0][0], &sm.Pb[0][0], o0, o0, o0, o0, acc);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            sm.Zb[o0 + acc_row(lane, e)][o0 + nt * 8 + acc_col(lane, e)] =
                __float2bfloat16(acc[nt][e]);
      }
      __syncthreads();
      if (act) {
        float acc[2][4] = {};
        blk(&sm.Xb[0][0], &sm.Zb[0][0], o0, o0, o0, o0, acc);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            int r = o0 + acc_row(lane, e), c = o0 + nt * 8 + acc_col(lane, e);
            float x = sm.u.inv.Xi[r][c] + acc[nt][e];
            sm.u.inv.Xi[r][c] = x;
            sm.Xb[r][c] = __float2bfloat16(x);
            sm.Pb[r][c] = sm.Zb[r][c];
          }
      }
      __syncthreads();
    }
    if (warp == 0) {                       // G = X22 * L21
      float acc[2][4] = {};
      blk(&sm.Xb[0][0], &sm.Lb[0][0], 16, 16, 0, 16, acc);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          sm.Gb[acc_row(lane, e)][nt * 8 + acc_col(lane, e)] = __float2bfloat16(acc[nt][e]);
    }
    __syncthreads();
    if (warp == 0) {                       // X21 = -G * X11
      float acc[2][4] = {};
      blk(&sm.Gb[0][0], &sm.Xb[0][0], 0, 0, 0, 0, acc);
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          sm.u.inv.Xi[16 + acc_row(lane, e)][nt * 8 + acc_col(lane, e)] = -acc[nt][e];
    }
    __syncthreads();
  }

  // ---- write results -----------------------------------------------------
  const long ch = (long)chunk * H + ih;
#endif
  {  // Tm = X diag(beta)  and  Ah = tril(W rows C..2C-1)
    const int r = tid >> 2, c0 = (tid & 3) * 8;
    float4 x0 = *(const float4*)&sm.u.inv.Xi[r][c0], x1 = *(const float4*)&sm.u.inv.Xi[r][c0 + 4];
    float4 a0 = *(const float4*)&sm.u.inv.Wf[C + r][c0], a1 = *(const float4*)&sm.u.inv.Wf[C + r][c0 + 4];
    float4 b0 = *(const float4*)&sm.Bt[c0], b1 = *(const float4*)&sm.Bt[c0 + 4];
    const float* xp = (const float*)&x0; const float* xq = (const float*)&x1;
    const float* ap = (const float*)&a0; const float* aq = (const float*)&a1;
    const float* bp = (const float*)&b0; const float* bq = (const float*)&b1;
    bf16 tv[8], av[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      int c = c0 + u;
      float xx = (u < 4) ? xp[u] : xq[u - 4];
      float aa = (u < 4) ? ap[u] : aq[u - 4];
      float bb = (u < 4) ? bp[u] : bq[u - 4];
      tv[u] = __float2bfloat16(c <= r ? xx * bb : 0.f);
      av[u] = __float2bfloat16(c <= r ? aa : 0.f);
    }
    *(uint4*)(oTm + ch * (C * C) + r * C + c0) = *(const uint4*)tv;
    *(uint4*)(oAh + ch * (C * C) + r * C + c0) = *(const uint4*)av;
  }
  if (tid < D) oEc[ch * D + tid] = sm.Ecs[tid] * sm.Ecs[tid];   // e^{2c}
#ifdef PIPE
  __threadfence();                 // make this block's stores device-visible
  __syncthreads();                 // ... for every thread, before we publish
  if (tid == 0) pipe_publish(gprod + (long)chunk * H + ih, epoch);
#endif

}

// ===========================================================================
// Kernel 2: sequential state scan.  grid = (NV, H), block = 256 threads.
// Warps 0-3 do all the math with the state / Yt / Ut held in registers.
// ===========================================================================
#ifndef STG
#define STG 4
#endif
#ifndef PFD
#define PFD 2            // prefetch distance; needs STG >= PFD + 2
#endif
struct SmemM {
  bf16  KQ[STG][2 * C][LD128];
  bf16  Kh[STG][C][LD128];
  bf16  Tm[STG][C][LD32];
  bf16  Ah[STG][C][LD32];
  float Ec[STG][D];
  bf16  v [STG][C][VP + 8];
  bf16  Ost[C][VP + 8];
};

__global__ __launch_bounds__(TPB, 1) void kda_scan_kernel(
    const bf16* __restrict__ gKQ, const bf16* __restrict__ gKh, const bf16* __restrict__ gTm,
    const bf16* __restrict__ gAh, const float* __restrict__ gEc, const bf16* __restrict__ gv,
    const float* __restrict__ gh0, bf16* __restrict__ go, float* __restrict__ ght,
    const unsigned* __restrict__ gprod, unsigned epoch,
    int T, int H, int NC, int chunk0, int nch) {
  extern __shared__ __align__(1024) char raw[];
  SmemM& sm = *reinterpret_cast<SmemM*>(raw);
  const int iv = blockIdx.x, ih = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool math = warp < NMW;
  const long v_base = (long)ih * DVV + iv * VP;

  auto load = [&](int st, int ch) {
    if (ch >= nch) return;
    ch += chunk0;
    const long cbase = (long)ch * H + ih;
#ifndef SKIP_SHARED
    for (int i = tid + (HALF_SHARED ? iv * (C * (D / 8)) : 0);
         i < (HALF_SHARED ? (iv + 1) * (C * (D / 8)) : 2 * C * (D / 8)); i += TPB) {
      int r = i / (D / 8), c = (i % (D / 8)) * 8;
      cp_async16(&sm.KQ[st][r][c], gKQ + cbase * (2 * C * D) + r * D + c);
    }
    for (int i = tid; i < C * (D / 8); i += TPB) {         // Kh: 256
      int r = i / (D / 8), c = (i % (D / 8)) * 8;
      cp_async16(&sm.Kh[st][r][c], gKh + cbase * (C * D) + r * D + c);
    }
    if (tid < C * (C / 8)) {                               // Tm, Ah: 128 each
      int r = tid / (C / 8), c = (tid % (C / 8)) * 8;
      cp_async16(&sm.Tm[st][r][c], gTm + cbase * (C * C) + r * C + c);
      cp_async16(&sm.Ah[st][r][c], gAh + cbase * (C * C) + r * C + c);
    }
    if (tid < D / 4) {
      cp_async16(&sm.Ec[st][tid * 4], gEc + cbase * D + tid * 4);
    }
#endif
    if (tid < C * (VP / 8)) {                              // v tile
      int r = tid / (VP / 8), c = (tid % (VP / 8)) * 8;
      cp_async16(&sm.v[st][r][c],
                 gv + ((long)ch * C + r) * (long)H * DVV + v_base + c);
    }
  };

  // O^T of the previous chunk, held so its stores can overlap the next
  // chunk's mma pipeline instead of trailing it.
  float oprev[4][4];
  auto emit_out = [&](const float (&o)[4][4], int ch) {
#pragma unroll
    for (int vb = 0; vb < 2; ++vb) {
      uint32_t xr[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) xr[m] = pack2(o[m][2 * vb], o[m][2 * vb + 1]);
      stm_trans4(&sm.Ost[0][0], VP + 8, lane, warp * 16 + vb * 8, xr);
    }
    const long row0 = (long)ch * C;
    const int c0 = warp * 16;
    bf16* src = &sm.Ost[lane][c0];
    bf16* dst = go + ((row0 + lane) * (long)H * DVV) + v_base + c0;
    *(uint4*)(dst)     = *(const uint4*)(src);
    *(uint4*)(dst + 8) = *(const uint4*)(src + 8);
  };

  // state St[v][d]: warp `warp` owns rows [16*warp, +16), all 128 channels
  float St[16][4];
  {
    const long h0b = ((long)ih * D) * DVV + iv * VP;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int vr = warp * 16 + acc_row(lane, e), dc = nt * 8 + acc_col(lane, e);
        St[nt][e] = math ? gh0[h0b + (long)dc * DVV + vr] : 0.f;
      }
  }

#pragma unroll 1
  for (int j = 0; j < PFD; ++j) {
#ifdef PIPE
    if (tid == 0) pipe_wait(gprod + (long)(chunk0 + j) * H + ih, epoch);
    __syncthreads();
#endif
    load(j, j); cp_commit();
  }

  for (int i = 0; i < nch; ++i) {
#if STG < 4
    // With fewer than 4 stages the slot we are about to fill, (i+2)%STG, is the
    // one the PREVIOUS iteration's math warps read, so they must be done first.
    __syncthreads();
#endif
#ifndef NOLOAD
    if (i + PFD < nch) {
#ifdef PIPE
      if (tid == 0) pipe_wait(gprod + (long)(chunk0 + i + PFD) * H + ih, epoch);
      __syncthreads();
#endif
      load((i + PFD) % STG, i + PFD);              // PFD chunks of slack
    }
    cp_commit();
#ifndef NOWAIT
    cp_wait<PFD - 1>();
    __syncthreads();
#endif
#endif
    const int st = i % STG;

    if (math) {
      const float* ec = sm.Ec[st];
      // ---- M1: acc = S * KQ^T   (M=16 rows of VP, N=2C, K=D).
      // Kbar arrives negated, so seeding the first four n-tiles with V^T makes
      // the accumulator come out as Y^T directly.
      float a1[8][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          a1[nt][e] = __bfloat162float(
              sm.v[st][nt * 8 + acc_col(lane, e)][warp * 16 + acc_row(lane, e)]);
#pragma unroll
      for (int nt = 4; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) a1[nt][e] = 0.f;
      {
#ifdef M1X4
        uint32_t af[4], bfr[2][4][4];                      // x4: half the ldmatrix
#pragma unroll
        for (int np = 0; np < 4; ++np)
          ldm_b4(bfr[0][np], &sm.KQ[st][0][0], LD128, np * 16, 0, lane);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          if (ks + 1 < D / 16) {
#pragma unroll
            for (int np = 0; np < 4; ++np)
              ldm_b4(bfr[(ks + 1) & 1][np], &sm.KQ[st][0][0], LD128, np * 16, (ks + 1) * 16, lane);
          }
          acc_to_a(af, St[2 * ks], St[2 * ks + 1]);
#pragma unroll
          for (int np = 0; np < 4; ++np) {
            mma2(a1[2 * np],     af, bfr[ks & 1][np][0], bfr[ks & 1][np][2]);
            mma2(a1[2 * np + 1], af, bfr[ks & 1][np][1], bfr[ks & 1][np][3]);
          }
        }
#else
        uint32_t af[4], bfr[2][8][2];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
          ldm_b(bfr[0][nt], &sm.KQ[st][0][0], LD128, nt * 8, 0, lane);
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
          if (ks + 1 < D / 16) {                           // prefetch next K step
#pragma unroll
            for (int nt = 0; nt < 8; ++nt)
              ldm_b(bfr[(ks + 1) & 1][nt], &sm.KQ[st][0][0], LD128, nt * 8, (ks + 1) * 16, lane);
          }
          acc_to_a(af, St[2 * ks], St[2 * ks + 1]);        // A straight from registers
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) mma16816(a1[nt], af, bfr[ks & 1][nt]);
        }
#endif
      }
#ifdef DEFER_OUT
      if (i > 0) emit_out(oprev, chunk0 + i - 1);   // overlaps the mma below
#endif
#ifndef SKIP_TAIL
      // ---- M3: Ut = Yt * Tm^T   (K=C) -----------------------------------
      float u3[4][4] = {};
      {
        uint32_t af[4], bfr[2][4];
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          acc_to_a(af, a1[2 * ks], a1[2 * ks + 1]);
#pragma unroll
          for (int gp = 0; gp < 2; ++gp)
            ldm_b4(bfr[gp], &sm.Tm[st][0][0], LD32, gp * 16, ks * 16, lane);
#pragma unroll
          for (int gp = 0; gp < 2; ++gp) {
            mma2(u3[2 * gp],     af, bfr[gp][0], bfr[gp][2]);
            mma2(u3[2 * gp + 1], af, bfr[gp][1], bfr[gp][3]);
          }
        }
      }
      // ---- M4: Ot += Ut * Ah^T   (into a1[4..7]) -------------------------
      {
        uint32_t af[4], bfr[2][4];
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          acc_to_a(af, u3[2 * ks], u3[2 * ks + 1]);
#pragma unroll
          for (int gp = 0; gp < 2; ++gp)
            ldm_b4(bfr[gp], &sm.Ah[st][0][0], LD32, gp * 16, ks * 16, lane);
#pragma unroll
          for (int gp = 0; gp < 2; ++gp) {
            mma2(a1[4 + 2 * gp],     af, bfr[gp][0], bfr[gp][2]);
            mma2(a1[4 + 2 * gp + 1], af, bfr[gp][1], bfr[gp][3]);
          }
        }
      }
      // ---- M5: St = Diag(e^{2c}) St + Ut * Khat^T  (one rescale per chunk) --
      {
        const int ec0 = (lane & 3) << 1;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          float2 e2 = *(const float2*)&ec[nt * 8 + ec0];
          St[nt][0] *= e2.x; St[nt][1] *= e2.y;
          St[nt][2] *= e2.x; St[nt][3] *= e2.y;
        }
#ifdef M5X4
        uint32_t af[4], bfr[2][8][4];                      // x4.trans: half the ldmatrix
#pragma unroll
        for (int np = 0; np < 8; ++np)
          ldm_bt4(bfr[0][np], &sm.Kh[st][0][0], LD128, np * 16, 0, lane);
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          if (ks + 1 < C / 16) {
#pragma unroll
            for (int np = 0; np < 8; ++np)
              ldm_bt4(bfr[(ks + 1) & 1][np], &sm.Kh[st][0][0], LD128, np * 16, (ks + 1) * 16, lane);
          }
          acc_to_a(af, u3[2 * ks], u3[2 * ks + 1]);
#pragma unroll
          for (int np = 0; np < 8; ++np) {
            mma2(St[2 * np],     af, bfr[ks & 1][np][0], bfr[ks & 1][np][1]);
            mma2(St[2 * np + 1], af, bfr[ks & 1][np][2], bfr[ks & 1][np][3]);
          }
        }
#else
#ifdef M5_SINGLE
        // single-buffered B: 32 fewer live registers, at the cost of the
        // k-step prefetch (16 n-tiles still give plenty of ILP).
        uint32_t af[4], bfr[16][2];
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
#pragma unroll
          for (int nt = 0; nt < 16; ++nt)
            ldm_bt(bfr[nt], &sm.Kh[st][0][0], LD128, nt * 8, ks * 16, lane);
          acc_to_a(af, u3[2 * ks], u3[2 * ks + 1]);
#pragma unroll
          for (int nt = 0; nt < 16; ++nt) mma16816(St[nt], af, bfr[nt]);
        }
#else
        uint32_t af[4], bfr[2][16][2];
#pragma unroll
        for (int nt = 0; nt < 16; ++nt)
          ldm_bt(bfr[0][nt], &sm.Kh[st][0][0], LD128, nt * 8, 0, lane);
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          if (ks + 1 < C / 16) {
#pragma unroll
            for (int nt = 0; nt < 16; ++nt)
              ldm_bt(bfr[(ks + 1) & 1][nt], &sm.Kh[st][0][0], LD128, nt * 8, (ks + 1) * 16, lane);
          }
          acc_to_a(af, u3[2 * ks], u3[2 * ks + 1]);
#pragma unroll
          for (int nt = 0; nt < 16; ++nt) mma16816(St[nt], af, bfr[ks & 1][nt]);
        }
#endif
#endif
      }
#endif
      // ---- stage O and write ---------------------------------------------
      // O^T[v][t] -> Ost[t][v] via two transposing matrix stores
#pragma unroll
#ifdef DEFER_OUT
#pragma unroll
      for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int e = 0; e < 4; ++e) oprev[m][e] = a1[4 + m][e];
#else
      for (int vb = 0; vb < 2; ++vb) {
        uint32_t xr[4];
#pragma unroll
        for (int m = 0; m < 4; ++m)
          xr[m] = pack2(a1[4 + m][2 * vb], a1[4 + m][2 * vb + 1]);
        stm_trans4(&sm.Ost[0][0], VP + 8, lane, warp * 16 + vb * 8, xr);
      }
      // each warp reads back exactly the columns it just wrote, so no barrier
      {
        const long row0 = (long)(chunk0 + i) * C;
        const int c0 = warp * 16;
        bf16* src = &sm.Ost[lane][c0];
        bf16* dst = go + ((row0 + lane) * (long)H * DVV) + v_base + c0;
        *(uint4*)(dst)     = *(const uint4*)(src);
        *(uint4*)(dst + 8) = *(const uint4*)(src + 8);
      }
#endif
    }
  }

#ifdef DEFER_OUT
  if (math && nch > 0) emit_out(oprev, chunk0 + nch - 1);
#endif
  if (math) {
    const long h0b = ((long)ih * D) * DVV + iv * VP;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        ght[h0b + (long)(nt * 8 + acc_col(lane, e)) * DVV + warp * 16 + acc_row(lane, e)] =
            St[nt][e];
  }
}

// ---------------------------------------------------------------------------
extern "C" int kda2_launch(const void* q, const void* k, const void* v, const void* g,
                           const void* beta, const void* A_log, const void* dt_bias,
                           const void* h0, void* o, void* ht,
                           void* wKQ, void* wKh, void* wTm, void* wAh, void* wEc,
                           void* wProd,
                           int B, int T, int H, float scale, void* stream) {
  if (B != 1) return -1;
  const int NC = T / C;
  cudaStream_t s = (cudaStream_t)stream;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(kda_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)sizeof(SmemP)) != cudaSuccess) return -2;
    if (cudaFuncSetAttribute(kda_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)sizeof(SmemM)) != cudaSuccess) return -3;
    once = true;
  }
    static cudaStream_t sp = nullptr, ss = nullptr;
  static cudaEvent_t ev[16], evs[16];
  if (!sp) {
    cudaStreamCreateWithFlags(&sp, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking);
    for (int i = 0; i < 16; ++i) {
      cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&evs[i], cudaEventDisableTiming);
    }
  }
  static unsigned epoch = 0; ++epoch;   // flags need no re-zeroing between calls
#if NGROUP == 1 && !defined(OVERLAP_TEST) && !defined(PIPE)
  // One group: no cross-stream handshake is needed, so issue both kernels on
  // the caller's stream and skip the event round-trip entirely.
  kda_prep_kernel<<<dim3(NC, H), 128, sizeof(SmemP), s>>>(
      (const bf16*)q, (const bf16*)k, (const bf16*)g, (const bf16*)beta,
      (const float*)A_log, (const float*)dt_bias,
      (bf16*)wKQ, (bf16*)wKh, (bf16*)wTm, (bf16*)wAh, (float*)wEc,
      (unsigned*)wProd, epoch, T, H, scale, 0);
  kda_scan_kernel<<<dim3(NV, H), TPB, sizeof(SmemM), s>>>(
      (const bf16*)wKQ, (const bf16*)wKh, (const bf16*)wTm, (const bf16*)wAh,
      (const float*)wEc, (const bf16*)v, (const float*)h0, (bf16*)o, (float*)ht,
      (const unsigned*)wProd, epoch, T, H, NC, 0, NC);
  return 0;
#endif
  int G = NGROUP; if (G > 16) G = 16;
  while (NC % G) --G;
  const int gs = NC / G;
  cudaEvent_t start; cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
  cudaEventRecord(start, s);
  cudaStreamWaitEvent(sp, start, 0);
  cudaStreamWaitEvent(ss, start, 0);
#ifdef PIPE
  // The scan is launched FIRST so its 128 CTAs are resident before prep's
  // 16384 CTAs arrive; otherwise prep saturates every SM and the scan cannot
  // start until prep has fully drained.  prep then fills the leftover capacity
  // and the two run as a producer/consumer pipeline over the chunk index.
  kda_scan_kernel<<<dim3(NV, H), TPB, sizeof(SmemM), ss>>>(
      (const bf16*)wKQ, (const bf16*)wKh, (const bf16*)wTm, (const bf16*)wAh,
      (const float*)wEc, (const bf16*)v, (const float*)h0, (bf16*)o, (float*)ht,
      (const unsigned*)wProd, epoch, T, H, NC, 0, NC);
  kda_prep_kernel<<<dim3(H, NC), 128, sizeof(SmemP), sp>>>(
      (const bf16*)q, (const bf16*)k, (const bf16*)g, (const bf16*)beta,
      (const float*)A_log, (const float*)dt_bias,
      (bf16*)wKQ, (bf16*)wKh, (bf16*)wTm, (bf16*)wAh, (float*)wEc,
      (unsigned*)wProd, epoch, T, H, scale, 0);
  cudaEventRecord(ev[0], sp);  cudaEventRecord(evs[0], ss);
  cudaStreamWaitEvent(s, ev[0], 0);  cudaStreamWaitEvent(s, evs[0], 0);
  cudaEventDestroy(start);
  return 0;
#endif
#ifdef OVERLAP_TEST
  // TIMING PROBE ONLY: no producer->consumer dependency, so the scan consumes
  // garbage.  Measures how fast the two kernels can run concurrently.
  kda_scan_kernel<<<dim3(NV, H), TPB, sizeof(SmemM), ss>>>(
      (const bf16*)wKQ, (const bf16*)wKh, (const bf16*)wTm, (const bf16*)wAh,
      (const float*)wEc, (const bf16*)v, (const float*)h0, (bf16*)o, (float*)ht,
      (const unsigned*)wProd, epoch, T, H, NC, 0, NC);
  kda_prep_kernel<<<dim3(NC, H), 128, sizeof(SmemP), sp>>>(
      (const bf16*)q, (const bf16*)k, (const bf16*)g, (const bf16*)beta,
      (const float*)A_log, (const float*)dt_bias,
      (bf16*)wKQ, (bf16*)wKh, (bf16*)wTm, (bf16*)wAh, (float*)wEc,
      (unsigned*)wProd, epoch, T, H, scale, 0);
  cudaEventRecord(ev[0], sp);  cudaEventRecord(evs[0], ss);
  cudaStreamWaitEvent(s, ev[0], 0);  cudaStreamWaitEvent(s, evs[0], 0);
  cudaEventDestroy(start);
  return 0;
#endif
  for (int gi = 0; gi < G; ++gi) {
    kda_prep_kernel<<<dim3(gs, H), 128, sizeof(SmemP), sp>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)g, (const bf16*)beta,
        (const float*)A_log, (const float*)dt_bias,
        (bf16*)wKQ, (bf16*)wKh, (bf16*)wTm, (bf16*)wAh, (float*)wEc,
      (unsigned*)wProd, epoch, T, H, scale, gi * gs);
    cudaEventRecord(ev[gi], sp);
    cudaStreamWaitEvent(ss, ev[gi], 0);
    kda_scan_kernel<<<dim3(NV, H), TPB, sizeof(SmemM), ss>>>(
        (const bf16*)wKQ, (const bf16*)wKh, (const bf16*)wTm, (const bf16*)wAh,
        (const float*)wEc, (const bf16*)v,
        (const float*)(gi == 0 ? h0 : ht), (bf16*)o, (float*)ht,
        (const unsigned*)wProd, epoch, T, H, NC, gi * gs, gs);
    cudaEventRecord(evs[gi], ss);
  }
  cudaStreamWaitEvent(s, evs[G - 1], 0);
  cudaEventDestroy(start);
  return 0;
}
extern "C" size_t kda2_smem_prep() { return sizeof(SmemP); }
extern "C" size_t kda2_smem_scan() { return sizeof(SmemM); }
