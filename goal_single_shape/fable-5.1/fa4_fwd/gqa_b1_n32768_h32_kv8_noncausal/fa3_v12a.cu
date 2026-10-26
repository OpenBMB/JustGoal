// FlashAttention-style forward for Blackwell (sm_100a), written from scratch. v12: three softmax warpgroups.
// Layout (B, S, H, D) bf16, head_dim = 128, GQA, non-causal. Persistent CTAs, one 128-row q tile per work unit.
// Warps: 0 = K/Q TMA producer, 1 = QK MMA issuer, 2 = PV MMA issuer, 3 = V TMA producer, 4-15 = softmax WGs 0..2.
// Softmax WG w handles global kv tiles g == w (mod 3). TMEM: per WG one 128-column buffer holding S (fp32) and, aliased
// in its first 64 columns, the packed bf16 P; O (fp32, 128 columns) at column 384.
// Max handling: a shared per-row reference max m_ref (hint); every tile's P is relative to the max its WG used, published
// in used_m. Before tile g signals P-ready, its WG waits for tile g-1's P-ready and reconciles: if the previous max is
// larger, this tile's P is scaled down to it; if smaller, O is scaled (after PV(g-1) completed) to this tile's max.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstddef>
#include "common.cuh"
#include "tmem_ops.cuh"

#ifndef EMUL_FRAC_NUM   // fraction of exp2 done on the FMA pipe = EMUL_FRAC_NUM / 32 (even)
#define EMUL_FRAC_NUM 8
#endif
#ifndef NK_STAGES
#define NK_STAGES 3
#endif
#ifndef NV_STAGES
#define NV_STAGES 2
#endif

namespace fa {

constexpr int D = 128;
constexpr int BM = 128;     // q rows per tile
constexpr int BN = 128;     // kv rows per tile
constexpr int NK = NK_STAGES;
constexpr int NV = NV_STAGES;
constexpr int NWG = 3;      // softmax warpgroups (= TMEM S/P buffers)
constexpr int K_STAGE_BYTES = BN * D * 2;   // 32 KB
constexpr int V_STAGE_BYTES = BN * D * 2;   // 32 KB
constexpr int Q_BYTES = BM * D * 2;         // 32 KB
constexpr int NTHREADS = 512;               // 16 warps
constexpr int CPT = BN;                     // columns per softmax thread per tile (full row)
constexpr int HPT = 32;                     // columns per chunk
constexpr int PPT = CPT / 2;                // packed bf16x2 regs per thread per tile
constexpr int NEMUL = (HPT * EMUL_FRAC_NUM) / 32;   // emulated elements per chunk (even)
constexpr int NCH = CPT / HPT;              // chunks per tile
static_assert(NEMUL % 2 == 0, "NEMUL must be even");

constexpr int O_COL = 384;    // O accumulator columns [384, 512)
constexpr float PSUM_LIMIT = 4096.f;   // chunk row-sum above this -> refresh the reference max

struct __align__(1024) Smem {
  uint8_t k[NK][K_STAGE_BYTES];
  uint8_t v[NV][V_STAGE_BYTES];
  uint8_t q[Q_BYTES];
  uint64_t k_full[NK], k_empty[NK];
  uint64_t v_full[NV], v_empty[NV];
  uint64_t s_full[NWG][2];   // per buffer, per 64-column half: QK MMAs complete
  uint64_t buf_empty[NWG];   // per buffer: PV of its tile complete (P consumed)
  uint64_t p_full[NWG];      // per buffer: P stored (4 warps)
  uint64_t q_smem_full, q_smem_empty;
  uint64_t o_full, o_empty, l_done, m_init;
  float m_ref[2][BM];        // shared reference max per row (raw S units), per unit parity
  float used_m[NWG][BM];     // per buffer: max its most recent P is relative to
  float l_part[NWG][BM];     // per WG: partial row sum at unit end (relative to m_part)
  float m_part[NWG][BM];
  uint32_t tmem_base;
};

struct Params {
  int B, S, HQ, HKV, G, NQT, NT;
  int total_units;
  float scale_log2;
  __nv_bfloat16* out;
  unsigned long long* prof;
  int* wd;                          // [grid][16 warps][2] : (checkpoint id, tile) ; mapped host memory (WATCHDOG)
};
#ifdef PROF
#define PROF_DECL unsigned long long prof_acc[8] = {0,0,0,0,0,0,0,0}; unsigned long long prof_t0 = clock64(); unsigned long long prof_last = clock64();
#define PROF_WAIT(idx, stmt) { unsigned long long _t = ((threadIdx.x & 31) == 0) ? clock64() : 0; stmt; if ((threadIdx.x & 31) == 0) prof_acc[idx] += clock64() - _t; }
#define PROF_END(slot) { prof_acc[7] = clock64() - prof_t0; if (prm.prof) for (int _i = 0; _i < 8; ++_i) prm.prof[(blockIdx.x * 16 + (slot)) * 8 + _i] = prof_acc[_i]; }
#define PROF_MARK(idx) { if ((threadIdx.x & 31) == 0) { unsigned long long _t = clock64(); prof_acc[idx] += _t - prof_last; prof_last = _t; } }
#define PROF_RESET { if ((threadIdx.x & 31) == 0) prof_last = clock64(); }
#else
#define PROF_DECL
#define PROF_WAIT(idx, stmt) stmt
#define PROF_END(slot) {}
#define PROF_MARK(idx) {}
#define PROF_RESET {}
#endif
#ifdef WATCHDOG
#define WD(id, tile) { if (lane == 0 && prm.wd) { volatile int* _p = prm.wd + (blockIdx.x * 16 + warp) * 2; _p[0] = (id); _p[1] = (int)(tile); } }
#else
#define WD(id, tile) {}
#endif
#define MBW(bar, par) mbar_wait(bar, par)

// ---- helpers ----
DEVI uint32_t cvt_bf16x2(float hi, float lo) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEVI uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
DEVI float fmax3(float a, float b, float c) { float r; asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
DEVI float fast_exp2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
DEVI void ffma2(float& a0, float& a1, float b0, float b1, float c0, float c1) {   // (a0,a1) = (a0,a1)*(b0,b1)+(c0,c1)
  asm("{\n.reg .b64 pa, pb, pc;\n mov.b64 pa, {%0, %1};\n mov.b64 pb, {%2, %3};\n mov.b64 pc, {%4, %5};\n fma.rn.f32x2 pa, pa, pb, pc;\n mov.b64 {%0, %1}, pa;\n}"
      : "+f"(a0), "+f"(a1) : "f"(b0), "f"(b1), "f"(c0), "f"(c1));
}
DEVI void fadd2(float& a0, float& a1, float b0, float b1) {
  asm("{\n.reg .b64 pa, pb;\n mov.b64 pa, {%0, %1};\n mov.b64 pb, {%2, %3};\n add.rn.f32x2 pa, pa, pb;\n mov.b64 {%0, %1}, pa;\n}"
      : "+f"(a0), "+f"(a1) : "f"(b0), "f"(b1));
}
// exp2 of two values via a cubic polynomial on the FMA pipe (inputs <= 0)
DEVI void exp2_emul2(float& x0, float& x1) {
  x0 = fmaxf(x0, -120.f); x1 = fmaxf(x1, -120.f);
  float t0 = x0, t1 = x1; fadd2(t0, t1, 12582912.f, 12582912.f);
  float n0 = t0, n1 = t1; fadd2(n0, n1, -12582912.f, -12582912.f);
  float f0 = x0, f1 = x1; fadd2(f0, f1, -n0, -n1);
  float p0 = 0.0551711f, p1 = 0.0551711f;
  ffma2(p0, p1, f0, f1, 0.2426113f, 0.2426113f);
  ffma2(p0, p1, f0, f1, 0.6932611f, 0.6932611f);
  ffma2(p0, p1, f0, f1, 0.9999281f, 0.9999281f);
  x0 = __int_as_float(__float_as_int(p0) + (__float_as_int(t0) << 23));
  x1 = __int_as_float(__float_as_int(p1) + (__float_as_int(t1) << 23));
}
// warp-collective: lane 0 arrives after all lanes fenced
DEVI void warp_arrive(uint64_t* bar) {
  tc_fence_before();
  __syncwarp();
  if ((threadIdx.x & 31) == 0) mbar_arrive(bar);
}
#ifdef FAKE_MMA   // ablation: no tensor-core work; commits become plain arrives
#define mma_ss(...) {}
#define mma_ts(...) {}
#define mma_commit(b) mbar_arrive(b)
#endif

__global__ void __launch_bounds__(NTHREADS, 1)
fa_fwd_kernel(const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK,
              const __grid_constant__ CUtensorMap tmV, Params prm) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  if ((smem_u32(smem_raw) & 1023) != 0) { if (threadIdx.x == 0) printf("smem base not 1024-aligned: %u\n", smem_u32(smem_raw)); __trap(); }
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NK; ++i) { mbar_init(&sm.k_full[i], 1); mbar_init(&sm.k_empty[i], 1); }
    for (int i = 0; i < NV; ++i) { mbar_init(&sm.v_full[i], 1); mbar_init(&sm.v_empty[i], 1); }
    for (int i = 0; i < NWG; ++i) {
      mbar_init(&sm.s_full[i][0], 1); mbar_init(&sm.s_full[i][1], 1);
      mbar_init(&sm.buf_empty[i], 1); mbar_init(&sm.p_full[i], 4);
    }
    mbar_init(&sm.q_smem_full, 1); mbar_init(&sm.q_smem_empty, 1);
    mbar_init(&sm.o_full, 1); mbar_init(&sm.o_empty, 4); mbar_init(&sm.l_done, 8); mbar_init(&sm.m_init, 4);
    mbar_fence_init();
    tma_prefetch_desc(&tmQ); tma_prefetch_desc(&tmK); tma_prefetch_desc(&tmV);
  }
  __syncthreads();
  if (warp == 0) { tmem_alloc(&sm.tmem_base, 512); tmem_relinquish(); }
  __syncthreads();
  const uint32_t tbase = sm.tmem_base;
  const int NT = prm.NT;

  auto unit_coords = [&](int u, int& b, int& qh, int& kvh, int& q0) {
    int per_b = prm.HQ * prm.NQT;
    b = u / per_b;
    int rem = u % per_b;
    int per_kv = prm.G * prm.NQT;
    kvh = rem / per_kv;
    int rem2 = rem % per_kv;
    qh = kvh * prm.G + (rem2 % prm.G);
    q0 = (rem2 / prm.G) * BM;
  };

  if (warp == 0) {
    // ---------------- K + Q producer ----------------
    if (lane == 0) {
      PROF_DECL
      uint32_t g = 0; int ul = 0;
      for (int u = blockIdx.x; u < prm.total_units; u += gridDim.x, ++ul) {
        int b, qh, kvh, q0; unit_coords(u, b, qh, kvh, q0);
        WD(1, ul); PROF_WAIT(0, MBW(&sm.q_smem_empty, (ul & 1) ^ 1)); WD(2, ul);
        mbar_expect_tx(&sm.q_smem_full, Q_BYTES);
        tma_load_4d(sm.q, &tmQ, &sm.q_smem_full, 0, qh, q0, b);
        tma_load_4d(sm.q + Q_BYTES / 2, &tmQ, &sm.q_smem_full, 64, qh, q0, b);
        for (int j = 0; j < NT; ++j, ++g) {
          const int ks = g % NK; const uint32_t ph = (g / NK) & 1;
          WD(3, g); PROF_WAIT(1, MBW(&sm.k_empty[ks], ph ^ 1)); WD(4, g);
          mbar_expect_tx(&sm.k_full[ks], K_STAGE_BYTES);
          tma_load_4d(sm.k[ks], &tmK, &sm.k_full[ks], 0, kvh, j * BN, b);
          tma_load_4d(sm.k[ks] + K_STAGE_BYTES / 2, &tmK, &sm.k_full[ks], 64, kvh, j * BN, b);
        }
      }
      PROF_END(0)
    }
  } else if (warp == 3) {
    if (lane == 0) {
      // ---------------- V producer ----------------
      PROF_DECL
      uint32_t g = 0;
      for (int u = blockIdx.x; u < prm.total_units; u += gridDim.x) {
        int b, qh, kvh, q0; unit_coords(u, b, qh, kvh, q0);
        for (int j = 0; j < NT; ++j, ++g) {
          const int vs = g % NV; const uint32_t ph = (g / NV) & 1;
          WD(5, g); PROF_WAIT(1, MBW(&sm.v_empty[vs], ph ^ 1)); WD(6, g);
          mbar_expect_tx(&sm.v_full[vs], V_STAGE_BYTES);
          tma_load_4d(sm.v[vs], &tmV, &sm.v_full[vs], 0, kvh, j * BN, b);
          tma_load_4d(sm.v[vs] + V_STAGE_BYTES / 2, &tmV, &sm.v_full[vs], 64, kvh, j * BN, b);
        }
      }
      PROF_END(3)
    }
  } else if (warp == 1) {
    // ---------------- QK MMA issuer: tile g -> buffer g % 3, two 64-column halves ----------------
    if (lane == 0) {
      const uint32_t idesc = make_idesc_fmt(BM, 64, 0, 0, 1, 1, 1);
      PROF_DECL
      uint32_t g = 0; int ul = 0;
      const uint32_t qbase = smem_u32(sm.q);
      const uint32_t kbase0 = smem_u32(sm.k[0]);
      for (int u = blockIdx.x; u < prm.total_units; u += gridDim.x, ++ul) {
        WD(10, ul); PROF_WAIT(0, MBW(&sm.q_smem_full, ul & 1)); WD(11, ul);
        tc_fence_after();
        for (int j = 0; j < NT; ++j, ++g) {
          const int ks = g % NK; const uint32_t kph = (g / NK) & 1;
          const int bf = g % NWG; const uint32_t bph = (g / NWG) & 1;
          WD(12, g); PROF_WAIT(2, MBW(&sm.k_full[ks], kph)); WD(13, g);
          WD(14, g); PROF_WAIT(1, MBW(&sm.buf_empty[bf], bph ^ 1)); WD(16, g);
          tc_fence_after();
          const uint32_t kbase = kbase0 + ks * K_STAGE_BYTES;
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const uint32_t d_tmem = tbase + bf * BN + hf * 64;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
              uint64_t adesc = make_smem_desc(qbase + (kk / 4) * (Q_BYTES / 2) + (kk % 4) * 32, 16, 1024, 2);
              uint64_t bdesc = make_smem_desc(kbase + (kk / 4) * (K_STAGE_BYTES / 2) + hf * 8192 + (kk % 4) * 32, 16, 1024, 2);
              mma_ss(d_tmem, adesc, bdesc, idesc, kk > 0);
            }
            mma_commit(&sm.s_full[bf][hf]);
          }
          mma_commit(&sm.k_empty[ks]);
        }
        mma_commit(&sm.q_smem_empty);
      }
      PROF_END(1)
    }
  } else if (warp == 2) {
    // ---------------- PV MMA issuer (in order) ----------------
    if (lane == 0) {
      const uint32_t idesc = make_idesc_fmt(BM, D, 0, 1, 1, 1, 1);   // A bf16 (TMEM), B bf16 MN-major, f32 accum
      PROF_DECL
      uint32_t g = 0; int ul = 0;
      const uint32_t vbase0 = smem_u32(sm.v[0]);
      const uint32_t d_tmem = tbase + O_COL;
      for (int u = blockIdx.x; u < prm.total_units; u += gridDim.x, ++ul) {
        PROF_WAIT(0, MBW(&sm.o_empty, (ul & 1) ^ 1));
        for (int j = 0; j < NT; ++j, ++g) {
          const int vs = g % NV; const uint32_t vph = (g / NV) & 1;
          const int bf = g % NWG; const uint32_t bph = (g / NWG) & 1;
          WD(20, g); PROF_WAIT(1, MBW(&sm.p_full[bf], bph)); WD(21, g);
          PROF_WAIT(2, MBW(&sm.v_full[vs], vph)); WD(22, g);
          tc_fence_after();
          const uint32_t a_tmem = tbase + bf * BN;
          const uint32_t vbase = vbase0 + vs * V_STAGE_BYTES;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            uint64_t bdesc = make_smem_desc(vbase + kk * 2048, V_STAGE_BYTES / 2, 1024, 2);
            mma_ts(d_tmem, a_tmem + kk * 8, bdesc, idesc, j > 0 || kk > 0);
          }
          mma_commit(&sm.buf_empty[bf]);
          mma_commit(&sm.v_empty[vs]);
        }
        mma_commit(&sm.o_full);
      }
      PROF_END(2)
    }
  } else {
    // ---------------- softmax warpgroups: WG w handles global tiles g == w (mod 3), full rows ----------------
    const int w = (warp - 4) / 4;
    const int wq = warp % 4;
    const int r = wq * 32 + lane;
    const uint32_t lane_base = (uint32_t)(wq * 32) << 16;
    const float c = prm.scale_log2;
    const uint32_t sp_tmem = tbase + w * BN + lane_base;   // S chunks at + h*32, P halves at + hf*32
    const uint32_t o_tmem = tbase + O_COL + lane_base;
    PROF_DECL
    float m_seen = -INFINITY;   // reference max this thread's l is relative to
    float l = 0.f;
    // helper: rescale a 32-column packed-bf16 P block in TMEM by a per-row factor
    auto rescale_tmem_p32 = [&](uint32_t taddr, float f) {
      tmem_st_wait();   // our own earlier P store to this buffer may still be in flight
      const uint32_t f2 = cvt_bf16x2(f, f);
#pragma unroll 1
      for (int q = 0; q < 2; ++q) {
        uint32_t po[16];
        tmem_ld16(taddr + q * 16, po);
        tmem_ld_wait16(po);
#pragma unroll
        for (int i = 0; i < 16; ++i) po[i] = bmul2(po[i], f2);
        tmem_st16(taddr + q * 16, po);
        tmem_st_wait16(po);
      }
    };
    for (uint32_t g = (uint32_t)w; ; g += NWG) {
      const int ui = (int)(g / (uint32_t)NT);     // local unit index
      const int j = (int)(g % (uint32_t)NT);      // tile within the unit
      const int u = blockIdx.x + ui * gridDim.x;
      if (u >= prm.total_units) break;
      const int par = ui & 1;
      float* m_ref = sm.m_ref[par];
      if (j < NWG) { m_seen = -INFINITY; l = 0.f; }   // first tile of this WG in the unit
      PROF_RESET
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      // ---- reference max for this tile (optimistic read; refreshed per chunk on overflow) ----
      float m_use;
      if (j == 0) {
        m_use = -INFINITY;   // first tile: the first chunk overflows and installs the exact chunk max
      } else {
        if (j < NWG) { WD(80, j); MBW(&sm.m_init, par); WD(81, j); }
        m_use = fmaxf(m_ref[r], m_seen);
      }
      if (m_use != m_seen) { l *= fast_exp2((m_seen - m_use) * c); m_seen = m_use; }
      PROF_MARK(1)
      float m_stored = m_seen;   // reference max the already-stored half (if any) is relative to
      uint32_t sb[2][HPT];       // S chunk double buffer: chunk h lives in sb[h & 1], loaded one chunk ahead
      bool half1_loaded = false;
      WD(30, j); PROF_WAIT(0, MBW(&sm.s_full[w][0], (g / NWG) & 1)); WD(32, j);
      tc_fence_after();
      tmem_ld<HPT>(sp_tmem, sb[0]);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t pk[PPT / 2];   // packed P of this half (64 columns)
        if (hf == 1 && !half1_loaded) {   // warp-uniform
          WD(31, j); PROF_WAIT(0, MBW(&sm.s_full[w][1], (g / NWG) & 1)); WD(33, j);
          tc_fence_after();
          tmem_ld<HPT>(sp_tmem + (NCH / 2) * HPT, sb[(NCH / 2) & 1]);
        }
#pragma unroll
        for (int hc = 0; hc < NCH / 2; ++hc) {
          const int h = hf * (NCH / 2) + hc;
          uint32_t* s = sb[h & 1];
          PROF_WAIT(6, tmem_ld_wait_r<HPT>(s));
          if (hc + 1 < NCH / 2) {
            tmem_ld<HPT>(sp_tmem + (h + 1) * HPT, sb[(h + 1) & 1]);   // prefetch the next chunk of this half
          } else if (hf == 0) {
            // prefetch the first chunk of half 1 if its S is already complete (non-blocking test, warp-uniform)
            uint32_t rdy = (lane == 0) ? (uint32_t)mbar_test_wait(&sm.s_full[w][1], (g / NWG) & 1) : 0u;
            rdy = __shfl_sync(0xffffffffu, rdy, 0);
            if (rdy) { tc_fence_after(); tmem_ld<HPT>(sp_tmem + (h + 1) * HPT, sb[(h + 1) & 1]); half1_loaded = true; }
          }
          auto chunk_p = [&](float mref, float& csum) {
            const float negmc = -mref * c;
            float ca[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < HPT / 2; ++i) {
              float x0 = __uint_as_float(s[2 * i]), x1 = __uint_as_float(s[2 * i + 1]);
              ffma2(x0, x1, c, c, negmc, negmc);
              if (2 * i < HPT - NEMUL) { x0 = fast_exp2(x0); x1 = fast_exp2(x1); } else { exp2_emul2(x0, x1); }
              fadd2(ca[(2 * i) & 2], ca[((2 * i) & 2) + 1], x0, x1);
              pk[hc * (HPT / 2) + i] = cvt_bf16x2(x1, x0);
            }
            csum = (ca[0] + ca[1]) + (ca[2] + ca[3]);
          };
          float csum;
          chunk_p(m_use, csum);
#ifdef NO_REFRESH
          const bool of = false;
#else
          const bool of = !(csum <= PSUM_LIMIT);
#endif
          if (__any_sync(0xffffffffu, of)) {
            // ---- refresh the reference max from this chunk; rescale this tile's earlier P (registers / stored half) ----
            float mx = fmax3(__uint_as_float(s[0]), __uint_as_float(s[1]), __uint_as_float(s[2]));
#pragma unroll
            for (int i = 3; i + 1 < HPT; i += 2) mx = fmax3(mx, __uint_as_float(s[i]), __uint_as_float(s[i + 1]));
            mx = fmaxf(mx, __uint_as_float(s[HPT - 1]));
            const float m_cur = (j == 0) ? m_use : fmaxf(m_ref[r], m_seen);
            const float m_new = of ? fmaxf(m_cur, mx) : m_cur;
            if (__any_sync(0xffffffffu, m_new != m_cur)) m_ref[r] = m_new;
            const bool chg = (m_new != m_seen);
            if (__any_sync(0xffffffffu, chg)) {   // warp-uniform block (contains warp-collective ops)
              const float a1 = chg ? fast_exp2((m_seen - m_new) * c) : 1.f;
              const uint32_t a12 = cvt_bf16x2(a1, a1);
#pragma unroll
              for (int i = 0; i < PPT / 2; ++i) if (i < hc * (HPT / 2)) pk[i] = bmul2(pk[i], a12);
              if (hf == 1) {
                const float a0 = chg ? fast_exp2((m_stored - m_new) * c) : 1.f;
                if (__any_sync(0xffffffffu, a0 != 1.f)) { tc_fence_after(); rescale_tmem_p32(sp_tmem, a0); tmem_st_wait(); tc_fence_before(); }
                if (chg) m_stored = m_new;
              }
#pragma unroll
              for (int a = 0; a < 4; ++a) acc[a] *= a1;
              l *= a1;
              if (chg) m_seen = m_new;
            }
            m_use = m_new;
            chunk_p(m_use, csum);
          }
          acc[h & 3] += csum;
        }
        // ---- store this half of P (aliases S columns already consumed) ----
        tmem_st32(sp_tmem + hf * 32, pk);
        tmem_st_wait32(pk);   // registers must stay intact until the asynchronous store completes
        if (hf == 0) m_stored = m_seen;
      }
      float psum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      PROF_MARK(2)
      // ---- reconcile with the previous tile's max before publishing P ----
      if (j > 0) {
        const int pb = (int)((g - 1) % NWG); const uint32_t pph = ((g - 1) / NWG) & 1;
        WD(50, j); PROF_WAIT(3, MBW(&sm.p_full[pb], pph)); WD(51, j);
        tc_fence_after();
        const float um = sm.used_m[pb][r];
        if (__any_sync(0xffffffffu, um != m_seen)) {
          // previous max larger: scale this tile's P (both halves), its sum and l to it
          const float fP = (um > m_seen) ? fast_exp2((m_seen - um) * c) : 1.f;
          if (__any_sync(0xffffffffu, fP != 1.f)) {
            rescale_tmem_p32(sp_tmem, fP);
            rescale_tmem_p32(sp_tmem + 32, fP);
            psum *= fP; l *= fP;
          }
          // previous max smaller: scale O (all PVs < g complete once PV(g-1) is) to this tile's max
          const float fO = (um < m_seen) ? fast_exp2((um - m_seen) * c) : 1.f;
          if (__any_sync(0xffffffffu, fO != 1.f)) {
            WD(52, j); MBW(&sm.buf_empty[pb], pph); WD(53, j);
            tc_fence_after();
#pragma unroll 1
            for (int q8 = 0; q8 < 8; ++q8) {
              uint32_t o[16];
              tmem_ld16(o_tmem + q8 * 16, o);
              tmem_ld_wait16(o);
#pragma unroll
              for (int i = 0; i < 16; ++i) o[i] = __float_as_uint(__uint_as_float(o[i]) * fO);
              tmem_st16(o_tmem + q8 * 16, o);
              tmem_st_wait16(o);
            }
          }
          m_seen = fmaxf(m_seen, um);
        }
      }
      if (j == 0) { m_ref[r] = m_seen; }
      sm.used_m[w][r] = m_seen;
      l += psum;
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { __threadfence_block(); mbar_arrive(&sm.p_full[w]); }
      if (j == 0) warp_arrive(&sm.m_init);
      WD(60, j);
      PROF_MARK(4)
      // ---- unit end ----
      if (j >= NT - NWG) {
        if (j == NT - 1) {
          int b, qh, kvh, q0; unit_coords(u, b, qh, kvh, q0);
          WD(70, ui); PROF_WAIT(5, MBW(&sm.l_done, par)); WD(71, ui);
          float lt = l;   // O is relative to m_seen (the largest max published)
#pragma unroll
          for (int ow = 1; ow < NWG; ++ow) {
            const int w2 = (w + ow) % NWG;
            lt += sm.l_part[w2][r] * fast_exp2((sm.m_part[w2][r] - m_seen) * c);
          }
          const float inv = 1.f / lt;
          WD(72, ui); PROF_WAIT(5, MBW(&sm.o_full, par)); WD(73, ui);
          tc_fence_after();
          __nv_bfloat16* orow = prm.out + (((size_t)b * prm.S + q0 + r) * prm.HQ + qh) * D;
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            uint32_t o[64];
            tmem_ld64(o_tmem + half * 64, o);
            tmem_ld_wait64(o);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              uint4 pkk;
              pkk.x = cvt_bf16x2(__uint_as_float(o[8 * i + 1]) * inv, __uint_as_float(o[8 * i + 0]) * inv);
              pkk.y = cvt_bf16x2(__uint_as_float(o[8 * i + 3]) * inv, __uint_as_float(o[8 * i + 2]) * inv);
              pkk.z = cvt_bf16x2(__uint_as_float(o[8 * i + 5]) * inv, __uint_as_float(o[8 * i + 4]) * inv);
              pkk.w = cvt_bf16x2(__uint_as_float(o[8 * i + 7]) * inv, __uint_as_float(o[8 * i + 6]) * inv);
              *reinterpret_cast<uint4*>(orow + half * 64 + i * 8) = pkk;
            }
          }
          warp_arrive(&sm.o_empty);
        } else {
          sm.l_part[w][r] = l; sm.m_part[w][r] = m_seen;
          __syncwarp();
          if (lane == 0) { __threadfence_block(); mbar_arrive(&sm.l_done); }
        }
      }
    }
    if (lane == 0) PROF_END(warp)
  }
  __syncthreads();
  if (warp == 0) tmem_dealloc(tbase, 512);
}

}  // namespace fa

// ---------------- host ----------------
torch::Tensor fa_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, double softmax_scale, int64_t num_ctas, torch::Tensor prof, int64_t wd_ptr, int64_t dbg_ptr = 0, int64_t ev_ptr = 0) {
  using namespace fa;
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 && v.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
  int B = q.size(0), S = q.size(1), HQ = q.size(2), Dd = q.size(3);
  int HKV = k.size(2);
  TORCH_CHECK(Dd == D && S % 512 == 0 && HQ % HKV == 0 && k.size(1) == S);
  auto out = torch::empty_like(q);
  Params prm;
  prm.B = B; prm.S = S; prm.HQ = HQ; prm.HKV = HKV; prm.G = HQ / HKV; prm.NQT = S / BM; prm.NT = S / BN;
  prm.total_units = B * HQ * prm.NQT;
  prm.scale_log2 = (float)(softmax_scale * 1.4426950408889634);
  prm.out = (__nv_bfloat16*)out.data_ptr();
  prm.prof = prof.numel() > 0 ? (unsigned long long*)prof.data_ptr() : nullptr;
  prm.wd = (int*)wd_ptr;

  CUtensorMap tmQ = make_tmap_4d_bf16(q.data_ptr(), D, HQ, S, B, D, (uint64_t)HQ * D, (uint64_t)S * HQ * D, 64, 1, BM, 1, true);
  CUtensorMap tmK = make_tmap_4d_bf16(k.data_ptr(), D, HKV, S, B, D, (uint64_t)HKV * D, (uint64_t)S * HKV * D, 64, 1, BN, 1, true);
  CUtensorMap tmV = make_tmap_4d_bf16(v.data_ptr(), D, HKV, S, B, D, (uint64_t)HKV * D, (uint64_t)S * HKV * D, 64, 1, BN, 1, true);
  size_t smem = offsetof(Smem, tmem_base) + sizeof(uint32_t);
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t ea = cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(ea == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(ea), " smem=", smem);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, fa_fwd_kernel);
    printf("kernel attrs: numRegs=%d maxThreads=%d sharedSizeBytes=%zu maxDynamicShared=%d smem_req=%zu\n", fa.numRegs, fa.maxThreadsPerBlock, fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes, smem);
    attr_set = true;
  }
  int grid = (int)num_ctas;
  if (grid <= 0) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&grid, cudaDevAttrMultiProcessorCount, dev); }
  if (grid > prm.total_units) grid = prm.total_units;
  auto stream = at::cuda::getCurrentCUDAStream();
  fa_fwd_kernel<<<grid, NTHREADS, smem, stream>>>(tmQ, tmK, tmV, prm);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
  return out;
}

// mapped host buffer for the watchdog: returns (host_ptr, device_ptr)
std::vector<int64_t> wd_alloc(int64_t n) {
  void* h = nullptr; void* d = nullptr;
  cudaHostAlloc(&h, n * sizeof(int), cudaHostAllocMapped);
  cudaHostGetDevicePointer(&d, h, 0);
  memset(h, 0, n * sizeof(int));
  return {(int64_t)h, (int64_t)d};
}
std::vector<int64_t> wd_read(int64_t h, int64_t n) {
  std::vector<int64_t> r(n); const int* p = (const int*)h; for (int64_t i = 0; i < n; ++i) r[i] = p[i]; return r;
}
int64_t smem_bytes() { return offsetof(fa::Smem, tmem_base) + sizeof(uint32_t); }
int64_t nthreads() { return fa::NTHREADS; }

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_fwd", &fa_fwd, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("softmax_scale"), py::arg("num_ctas"), py::arg("prof"),
        py::arg("wd_ptr") = 0, py::arg("dbg_ptr") = 0, py::arg("ev_ptr") = 0);
  m.def("wd_alloc", &wd_alloc);
  m.def("wd_read", &wd_read);
  m.def("smem_bytes", &smem_bytes);
  m.def("nthreads", &nthreads);
}
