// From-scratch FlashAttention forward (causal, bf16 in/out, hdim 128) for Blackwell sm_100a.  v18: fp16-accumulated S
// (fp16 Q/K), packed TMEM loads (.pack::16b) and an ALU-port-clean softmax loop (HFMA2.relu + MUFU(neg) + HADD2 only),
// so the softmax warps no longer steal issue slots from the tcgen05.mma issuers on their SMSPs.
// v24: Q is converted bf16->fp16 in SMEM by the idle lanes 1-31 of the V-producer warp (one tile ahead).  v9
// 384 threads: warp0 = TMA Q/K producer, warp1 = S-GEMM issuer, warp2 = TMA V producer, warp3 = PV-GEMM issuer,
// warps 4-7 = softmax warpgroup 0 (even blocks), warps 8-11 = softmax warpgroup 1 (odd blocks).
// Each softmax thread owns one row of its block (all 128 columns, streamed in two 64-column halves).
// V is pre-converted to fp16; P is fp16; O fp32 in TMEM. TMEM: S x2, P x2 (one per warpgroup), O.
// SMEM: Q x2, K x3, V x2 (32KB tiles, 128B swizzle).
#include "ptx.cuh"
#include "tmap.cuh"
#include <math.h>

#ifndef NPOLY
#define NPOLY 4
#endif

namespace {
constexpr int BM = 128, BN = 128, HD = 128;
constexpr int QSTAGES = 2, KSTAGES = 2, VSTAGES = 2;   // v25: K x2 -> 2-step S issuer rotation
constexpr uint32_t T_S0 = 0, T_P0 = 256, T_O = 384;
constexpr int TILE_BYTES = BM * HD * 2;
constexpr int NUM_THREADS = 640;
constexpr float SUM_GUARD = 1024.f;
constexpr float X_CLAMP = 11.f;

struct __align__(1024) Smem {
  __nv_bfloat16 q[QSTAGES][BM * HD];
  __nv_bfloat16 k[KSTAGES][BN * HD];
  __nv_bfloat16 v[VSTAGES][BN * HD];
  float m_pub[128];        // m_used published by the diagonal-block warpgroup
  float xch_epi[4][2][2][32];   // [wl][bpar][half][lane]: diagonal-block max exchange
  uint64_t bar_q_full[QSTAGES], bar_q_empty[QSTAGES], bar_q_tma[QSTAGES];
  uint64_t bar_k_full[KSTAGES], bar_k_empty[KSTAGES];
  uint64_t bar_v_full[VSTAGES], bar_v_empty[VSTAGES];
  uint64_t bar_s_full[2], bar_s_read[2];
  uint64_t bar_p_full[2], bar_pv_done[2];
  uint64_t bar_m_ready;
  uint64_t bar_epi;
  uint32_t tmem_base;
};

struct Params {
  unsigned long long* prof;
  int B, S, HQ, HKV;
  int q_tiles;
  int num_tiles;
  float scale_log2;
  __nv_bfloat16* out;
};

DEVI long long clk64() { long long c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)); return c; }
#ifdef PROF
#define PROF_T(v) long long v = clk64()
#define PROF_ACC(acc, t0) acc += clk64() - (t0)
#else
#define PROF_T(v)
#define PROF_ACC(acc, t0)
#endif

DEVI void decode_tile(const Params& p, int i, int& b, int& h, int& qt) {
  int bh = p.B * p.HQ;
  qt = p.q_tiles - 1 - i / bh;
  int rem = i % bh;
  b = rem / p.HQ;
  h = rem % p.HQ;
}
DEVI int tile_index(int k, int G, int bx) { return k * G + ((k & 1) ? (G - 1 - bx) : bx); }
DEVI constexpr bool use_poly(int idx16) { return (((idx16 + 1) * NPOLY) / 16) != ((idx16 * NPOLY) / 16); }
DEVI float poly_exp2(float x) {
  x = fminf(fmaxf(x, -126.f), X_CLAMP);
  const float magic = 12582912.0f;
  float t = x + magic;
  float fr = x - (t - magic);
  float pp = fmaf(fr, 0.0551601394f, 0.242582748f);
  pp = fmaf(pp, fr, 0.693260557f);
  pp = fmaf(pp, fr, 0.999930224f);
  return __int_as_float(__float_as_int(pp) + (__float_as_int(t) << 23));
}
DEVI void named_bar_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(nthreads) : "memory"); }

// Keep scores and normalization in FP32. The diagonal maximum is a shift,
// not an upper bound: earlier causal keys can have larger scores.
DEVI void exp8p(const uint32_t* r, uint32_t* pk, float scale, float maximum,
               float& sum_lo, float& sum_hi) {
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    float lo = exp2f(__uint_as_float(r[2*x]) * scale - maximum - X_CLAMP);
    float hi = exp2f(__uint_as_float(r[2*x+1]) * scale - maximum - X_CLAMP);
    pk[x] = cvt_f16x2(lo, hi);
    sum_lo += f16lo(pk[x]);
    sum_hi += f16hi(pk[x]);
  }
}

__global__ void __launch_bounds__(NUM_THREADS, 1)
fa_fwd_kernel(const __grid_constant__ CUtensorMap tq, const __grid_constant__ CUtensorMap tk,
              const __grid_constant__ CUtensorMap tv, const Params p) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if ((smem_u32(smem_raw) & 1023u) != 0) { if (tid == 0) p.prof[16383] = 0xBAD; return; }

  if (tid == 0) {
    for (int i = 0; i < QSTAGES; ++i) { mbar_init(&sm.bar_q_full[i], 1); mbar_init(&sm.bar_q_empty[i], 1); mbar_init(&sm.bar_q_tma[i], 1); }
    for (int i = 0; i < KSTAGES; ++i) { mbar_init(&sm.bar_k_full[i], 1); mbar_init(&sm.bar_k_empty[i], 1); }
    for (int i = 0; i < VSTAGES; ++i) { mbar_init(&sm.bar_v_full[i], 1); mbar_init(&sm.bar_v_empty[i], 1); }
    for (int i = 0; i < 2; ++i) { mbar_init(&sm.bar_s_full[i], 1); mbar_init(&sm.bar_s_read[i], 8); mbar_init(&sm.bar_p_full[i], 8); mbar_init(&sm.bar_pv_done[i], 1); }
    mbar_init(&sm.bar_m_ready, 8);
    mbar_init(&sm.bar_epi, 16);
    fence_barrier_init();
  }
  if (warp == 0) {
    tmem_alloc(&sm.tmem_base, 512);
    tmem_relinquish();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem_base = sm.tmem_base;
  const uint32_t t_o = tmem_base + T_O;
  const int G = gridDim.x;
  const int hq_per_kv = p.HQ / p.HKV;

  if (warp == 0) {
    // ===================== TMA producer: Q + K =====================
    if (lane == 0) {
      tma_prefetch_desc(&tq); tma_prefetch_desc(&tk);
      uint32_t kv_iter = 0, tile_count = 0;
      for (int k = 0;; ++k) {
        int i = tile_index(k, G, blockIdx.x);
        if (i >= p.num_tiles) break;
        int b, h, qt; decode_tile(p, i, b, h, qt);
        int hk = h / hq_per_kv;
        int row0 = b * p.S + qt * BM;
        int nb = qt + 1;
        for (int j = 0; j < nb; ++j) {
          int krow = b * p.S + (qt - j) * BN;
          int ks = kv_iter % KSTAGES, kp = (kv_iter / KSTAGES) & 1;
          mbar_wait_sleep(&sm.bar_k_empty[ks], kp ^ 1);
          mbar_expect_tx(&sm.bar_k_full[ks], TILE_BYTES);
          tma_load_3d(sm.k[ks], &tk, &sm.bar_k_full[ks], 0, hk, krow);
          tma_load_3d(sm.k[ks] + 64 * BN, &tk, &sm.bar_k_full[ks], 64, hk, krow);
          ++kv_iter;
        }
        ++tile_count;
      }
    }
  } else if (warp == 2) {
    // ===================== warp 2: lane 0 = TMA producer V; lanes 1-31 = Q loader + in-place bf16->fp16 converter =====================
    if (lane == 0) {
      tma_prefetch_desc(&tv);
      uint32_t kv_iter = 0;
      for (int k = 0;; ++k) {
        int i = tile_index(k, G, blockIdx.x);
        if (i >= p.num_tiles) break;
        int b, h, qt; decode_tile(p, i, b, h, qt);
        int hk = h / hq_per_kv;
        int nb = qt + 1;
        for (int j = 0; j < nb; ++j) {
          int krow = b * p.S + (qt - j) * BN;
          int vs = kv_iter % VSTAGES, vp = (kv_iter / VSTAGES) & 1;
          mbar_wait_sleep(&sm.bar_v_empty[vs], vp ^ 1);
          mbar_expect_tx(&sm.bar_v_full[vs], TILE_BYTES);
          tma_load_3d(sm.v[vs], &tv, &sm.bar_v_full[vs], 0, hk, krow);
          tma_load_3d(sm.v[vs] + 64 * BN, &tv, &sm.bar_v_full[vs], 64, hk, krow);
          ++kv_iter;
        }
      }
    } else {
      const unsigned CMASK = 0xFFFFFFFEu;   // lanes 1-31
      if (lane == 1) tma_prefetch_desc(&tq);
      uint32_t tile_count = 0;
      for (int k = 0;; ++k) {
        int i = tile_index(k, G, blockIdx.x);
        if (i >= p.num_tiles) break;
        int b, h, qt; decode_tile(p, i, b, h, qt);
        int row0 = b * p.S + qt * BM;
        int qs = tile_count % QSTAGES, qp = (tile_count / QSTAGES) & 1;
        if (lane == 1) {
          mbar_wait_sleep(&sm.bar_q_empty[qs], qp ^ 1);
          mbar_expect_tx(&sm.bar_q_tma[qs], TILE_BYTES);
          tma_load_3d(sm.q[qs], &tq, &sm.bar_q_tma[qs], 0, h, row0);
          tma_load_3d(sm.q[qs] + 64 * BM, &tq, &sm.bar_q_tma[qs], 64, h, row0);
        }
        __syncwarp(CMASK);
        mbar_wait_sleep(&sm.bar_q_tma[qs], qp);
        uint4* tile = reinterpret_cast<uint4*>(sm.q[qs]);
#pragma unroll 2
        for (int e = lane - 1; e < TILE_BYTES / 16; e += 31) {
          uint4 v = tile[e];
          uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            float lo = __uint_as_float(w[t] << 16), hi = __uint_as_float(w[t] & 0xFFFF0000u);
            lo = fminf(fmaxf(lo, -65504.f), 65504.f); hi = fminf(fmaxf(hi, -65504.f), 65504.f);
            w[t] = cvt_f16x2(lo, hi);
          }
          tile[e] = make_uint4(w[0], w[1], w[2], w[3]);
        }
        fence_proxy_async();
        __syncwarp(CMASK);
        if (lane == 1) mbar_arrive(&sm.bar_q_full[qs]);
        ++tile_count;
      }
    }
  } else if (warp == 1) {
    // ===================== S-GEMM issuer =====================
    if (lane == 0) {
      const uint32_t idesc_ss = make_idesc_f16(BM, BN, false, false);
      uint64_t da[QSTAGES][8], db[KSTAGES][8];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        uint64_t off = (uint64_t)(((kk / 4) * 16384 + (kk % 4) * 32) >> 4);
#pragma unroll
        for (int st = 0; st < QSTAGES; ++st) da[st][kk] = make_desc_sw128(smem_u32(sm.q[st]), 16, 1024) + off;
#pragma unroll
        for (int st = 0; st < KSTAGES; ++st) db[st][kk] = make_desc_sw128(smem_u32(sm.k[st]), 16, 1024) + off;
      }
      uint32_t s_iter = 0, tile_count = 0;
      long long acc_wq = 0, acc_wk = 0, acc_wsr = 0, acc_is = 0; PROF_T(t_start);
      for (int k = 0;; ++k) {
        int i = tile_index(k, G, blockIdx.x);
        if (i >= p.num_tiles) break;
        int b, h, qt; decode_tile(p, i, b, h, qt);
        int nb = qt + 1;
        int qs = tile_count % QSTAGES;
        PROF_T(t0);
        mbar_wait(&sm.bar_q_full[qs], (tile_count / QSTAGES) & 1);
        PROF_ACC(acc_wq, t0);
        int j = 0;
        int u = s_iter & 1;
        while (j < nb) {
          switch (u) {
#define S_STEP(U) \
            case U: { \
              constexpr int ks = U % KSTAGES, sb = U % 2; \
              uint32_t kp = (s_iter / KSTAGES) & 1; \
              PROF_T(t1); mbar_wait(&sm.bar_k_full[ks], kp); PROF_ACC(acc_wk, t1); \
              PROF_T(t2); if (s_iter >= 2) mbar_wait(&sm.bar_s_read[sb], ((s_iter - 2) / 2) & 1); PROF_ACC(acc_wsr, t2); \
              PROF_T(t3); \
              if (qs == 0) umma8_ss_noacc(tmem_base + T_S0 + sb * 128, idesc_ss, da[0], db[ks]); \
              else umma8_ss_noacc(tmem_base + T_S0 + sb * 128, idesc_ss, da[1], db[ks]); \
              umma_commit(&sm.bar_s_full[sb]); umma_commit(&sm.bar_k_empty[ks]); \
              PROF_ACC(acc_is, t3); \
              ++s_iter; ++j; u = (U + 1) & 1; \
            } break;
            S_STEP(0) S_STEP(1)
#undef S_STEP
          }
        }
        umma_commit(&sm.bar_q_empty[qs]);
        ++tile_count;
      }
#ifdef PROF
      p.prof[blockIdx.x * 16 + 0] = acc_wq; p.prof[blockIdx.x * 16 + 1] = acc_wk; p.prof[blockIdx.x * 16 + 2] = acc_wsr;
      p.prof[blockIdx.x * 16 + 3] = acc_is; p.prof[blockIdx.x * 16 + 4] = clk64() - t_start;
#endif
    }
  } else if (warp == 3) {
    // ===================== PV-GEMM issuer =====================
    if (lane == 0) {
      const uint32_t idesc_ts = make_idesc_f16(BM, HD, false, true);
      uint64_t dv[VSTAGES][8];
#pragma unroll
      for (int st = 0; st < VSTAGES; ++st) {
        uint64_t vdesc = make_desc_sw128(smem_u32(sm.v[st]), 16384, 1024);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) dv[st][kk] = vdesc + (uint64_t)((kk * 2048) >> 4);
      }
      uint32_t p_iter = 0;
      long long acc_wp = 0, acc_wv = 0, acc_ipv = 0; PROF_T(t_start);
      for (int k = 0;; ++k) {
        int i = tile_index(k, G, blockIdx.x);
        if (i >= p.num_tiles) break;
        int b, h, qt; decode_tile(p, i, b, h, qt);
        int nb = qt + 1;
        int j = 0;
        int u = p_iter & 1;
        while (j < nb) {
          switch (u) {
#define PV_STEP(U) \
            case U: { \
              constexpr int vs = U, pb = U; \
              uint32_t par = (p_iter >> 1) & 1; \
              PROF_T(t0); mbar_wait(&sm.bar_p_full[pb], par); PROF_ACC(acc_wp, t0); \
              tc_fence_after(); \
              PROF_T(t1); mbar_wait(&sm.bar_v_full[vs], par); PROF_ACC(acc_wv, t1); \
              PROF_T(t2); \
              if (j == 0) umma8_ts_noacc(t_o, idesc_ts, tmem_base + T_P0 + pb * 64, dv[vs]); \
              else umma8_ts_acc(t_o, idesc_ts, tmem_base + T_P0 + pb * 64, dv[vs]); \
              umma_commit(&sm.bar_pv_done[pb]); umma_commit(&sm.bar_v_empty[vs]); \
              PROF_ACC(acc_ipv, t2); \
              ++p_iter; ++j; u = (U + 1) & 1; \
            } break;
            PV_STEP(0) PV_STEP(1)
#undef PV_STEP
          }
        }
      }
#ifdef PROF
      p.prof[blockIdx.x * 16 + 5] = acc_wp; p.prof[blockIdx.x * 16 + 6] = acc_wv; p.prof[blockIdx.x * 16 + 7] = acc_ipv;
      p.prof[blockIdx.x * 16 + 8] = clk64() - t_start;
#endif
    }
  } else {
    // ===================== softmax: 4 warpgroups. wg&1 = block parity handled, wg>>1 = column half =====================
    const int wg = (warp - 4) / 4;
    const int bpar = wg & 1;
    const int half = wg >> 1;
    const int wl = warp & 3;
    const int rloc = wl * 32 + lane;
    const uint32_t lane_off = (uint32_t)(wl * 32) << 16;
    const uint32_t t_s = tmem_base + T_S0 + bpar * 128 + lane_off + half * 64;
    const uint32_t t_p = tmem_base + T_P0 + bpar * 64 + lane_off + half * 32;
    const float c = p.scale_log2;
    const int pair_bar = 1 + wl + 4 * bpar;     // named barrier ids 1..8: the two column-half warps of the same rows/parity
    uint32_t kv_iter = 0;
    uint32_t r[64];
    uint32_t pk[16];
    float m_used = 0.f;
    long long acc_ws = 0, acc_cmp = 0, acc_st = 0, n_slow = 0, acc_rs = 0, acc_wpv = 0; PROF_T(t_start);
    for (int k = 0;; ++k) {
      int i = tile_index(k, G, blockIdx.x);
      if (i >= p.num_tiles) break;
      int b, h, qt; decode_tile(p, i, b, h, qt);
      int nb = qt + 1;
      float l0 = 0.f, l1 = 0.f;
      const int diag_par = (int)(kv_iter & 1);
      const int j0 = (bpar == diag_par) ? 0 : 1;
      for (int j = j0; j < nb; j += 2) {
        const uint32_t kv = kv_iter + j;
        const uint32_t n = kv >> 1;                 // index of this block among its parity
        const uint32_t par = n & 1;
        PROF_T(t0);
        mbar_wait(&sm.bar_s_full[bpar], par);
        PROF_ACC(acc_ws, t0);
        tc_fence_after();
        PROF_T(t2);
        float b0 = 0.f, b1 = 0.f;
        tmem_ld32(t_s, r);
        tmem_ld32(t_s + 32, r + 32);
        tmem_ld_wait();
        tc_fence_before();
        if (lane == 0) mbar_arrive(&sm.bar_s_read[bpar]);   // S buffer consumed: the issuer may overwrite it
        if (j == 0) {
          // diagonal block: mask, my 64 columns' row max, exchange with the other half, publish m
          float mx = -INFINITY;
#pragma unroll
          for (int cidx = 0; cidx < 64; ++cidx) {
            int col = half * 64 + cidx;
            if (col > rloc) r[cidx] = __float_as_uint(-INFINITY);
            mx = fmaxf(mx, __uint_as_float(r[cidx]));
          }
          sm.xch_epi[wl][bpar][half][lane] = mx;
          named_bar_sync(pair_bar, 64);
          mx = fmaxf(mx, sm.xch_epi[wl][bpar][half ^ 1][lane]);
          named_bar_sync(pair_bar, 64);
          m_used = mx * c;
          if (half == 0) sm.m_pub[rloc] = m_used;
          __syncwarp();
          if (lane == 0) mbar_arrive(&sm.bar_m_ready);
          l0 = 0.f; l1 = 0.f;
        } else if (j == 1) {
          mbar_wait(&sm.bar_m_ready, (uint32_t)k & 1);
          m_used = sm.m_pub[rloc];
          l0 = 0.f; l1 = 0.f;
        }
        exp8p(r, pk, c, m_used, b0, b1);
        exp8p(r + 16, pk + 8, c, m_used, b0, b1);
        if (n >= 1) { PROF_T(t0b); mbar_wait(&sm.bar_pv_done[bpar], (n - 1) & 1); PROF_ACC(acc_wpv, t0b); tc_fence_after(); }
        tmem_st16(t_p, pk);
        exp8p(r + 32, pk, c, m_used, b0, b1);
        exp8p(r + 48, pk + 8, c, m_used, b0, b1);
        tmem_st16(t_p + 16, pk);
        PROF_ACC(acc_cmp, t2);
        PROF_T(t3);
        add_f32x2(l0, l1, l0, l1, b0, b1);
        tmem_st_wait();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.bar_p_full[bpar]);
        PROF_ACC(acc_st, t3);
      }
      // ---- epilogue: all 16 softmax warps ----
      {
        uint32_t last_kv = kv_iter + nb - 1;
        int last_par = (int)(last_kv & 1);
        uint32_t last_n = last_kv >> 1;
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.bar_epi);
        mbar_wait(&sm.bar_epi, (uint32_t)k & 1);
        mbar_wait(&sm.bar_pv_done[last_par], last_n & 1);
        tc_fence_after();
        float* l_x = &sm.xch_epi[0][0][0][0];   // aliases the diagonal-block exchange buffer (disjoint in time)
        l_x[wg * 128 + rloc] = l0 + l1;
        named_bar_sync(9, 512);
        float ltot = l_x[rloc] + l_x[128 + rloc] + l_x[256 + rloc] + l_x[384 + rloc];
        uint32_t to = t_o + lane_off + wg * 32;
        tmem_ld32(to, r);
        tmem_ld_wait();
        float inv = 1.f / ltot;
        size_t row = ((size_t)(b * p.S + qt * BM + rloc) * p.HQ + h);
        uint4* dst = reinterpret_cast<uint4*>(p.out + row * HD + wg * 32);
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          uint4 val;
          val.x = pack_bf16x2(__uint_as_float(r[8 * x + 0]) * inv, __uint_as_float(r[8 * x + 1]) * inv);
          val.y = pack_bf16x2(__uint_as_float(r[8 * x + 2]) * inv, __uint_as_float(r[8 * x + 3]) * inv);
          val.z = pack_bf16x2(__uint_as_float(r[8 * x + 4]) * inv, __uint_as_float(r[8 * x + 5]) * inv);
          val.w = pack_bf16x2(__uint_as_float(r[8 * x + 6]) * inv, __uint_as_float(r[8 * x + 7]) * inv);
          dst[x] = val;
        }
        tc_fence_before();
        named_bar_sync(9, 512);   // O fully read before the next tile's first PV (accumulate=0)
      }
      kv_iter += nb;
    }
#ifdef PROF
    if (lane == 0 && (warp == 4 || warp == 5 || warp == 6 || warp == 7)) {
      int base = 9 + (warp - 4) * 2;   // slots 9..16 -> we only have up to 15: use 9,11,13 for warps 4,5,6 ; warp7 -> skip
      if (warp < 7) { p.prof[blockIdx.x * 16 + base] = acc_cmp; p.prof[blockIdx.x * 16 + base + 1] = acc_ws; }
      if (warp == 4) p.prof[blockIdx.x * 16 + 15] = kv_iter;
    }
    if (lane == 0 && warp == 4) { p.prof[(512 + blockIdx.x) * 16 + 0] = acc_st; p.prof[(512 + blockIdx.x) * 16 + 1] = n_slow; p.prof[(512 + blockIdx.x) * 16 + 2] = clk64() - t_start; p.prof[(512 + blockIdx.x) * 16 + 3] = acc_wpv; }
    if (lane == 0 && warp == 12) { p.prof[(512 + blockIdx.x) * 16 + 4] = acc_wpv; p.prof[(512 + blockIdx.x) * 16 + 5] = clk64() - t_start; p.prof[(512 + blockIdx.x) * 16 + 6] = acc_cmp; p.prof[(512 + blockIdx.x) * 16 + 7] = acc_ws; }
#endif
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tmem_base, 512);
}

__global__ void cvt_bf16_to_f16_kernel(const uint4* __restrict__ in, uint4* __restrict__ out, size_t n8) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n8; i += stride) {
    uint4 v = in[i];
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float lo = __uint_as_float(w[k] << 16), hi = __uint_as_float(w[k] & 0xFFFF0000u);
      lo = fminf(fmaxf(lo, -65504.f), 65504.f); hi = fminf(fmaxf(hi, -65504.f), 65504.f);
      w[k] = cvt_f16x2(lo, hi);
    }
    out[i] = make_uint4(w[0], w[1], w[2], w[3]);
  }
}

}  // namespace

static unsigned long long* g_prof = nullptr;
static void* g_v16 = nullptr; static size_t g_v16_bytes = 0;
static void* g_k16 = nullptr; static size_t g_k16_bytes = 0;
extern "C" unsigned long long* fa_prof_buf() { return g_prof; }
extern "C" int fa_fwd(void* q, void* k, void* v, void* o, int B, int S, int HQ, int HKV, float softmax_scale,
                      cudaStream_t stream, int grid_override) {
  if (S % BM != 0 || HQ % HKV != 0) return -3;
  size_t vbytes = (size_t)B * S * HKV * HD * 2;
  if (g_v16_bytes < vbytes) { if (g_v16) cudaFree(g_v16); cudaMalloc(&g_v16, vbytes); g_v16_bytes = vbytes; }
  if (g_k16_bytes < vbytes) { if (g_k16) cudaFree(g_k16); cudaMalloc(&g_k16, vbytes); g_k16_bytes = vbytes; }
  {
    size_t n8 = vbytes / 16;
    int blocks = (int)((n8 + 255) / 256); if (blocks > 4096) blocks = 4096;
    cvt_bf16_to_f16_kernel<<<blocks, 256, 0, stream>>>((const uint4*)v, (uint4*)g_v16, n8);
    cvt_bf16_to_f16_kernel<<<blocks, 256, 0, stream>>>((const uint4*)k, (uint4*)g_k16, n8);
  }
  CUtensorMap tq, tk, tv;
  {
    cuuint64_t dims[3] = {(cuuint64_t)HD, (cuuint64_t)HQ, (cuuint64_t)B * S};
    cuuint64_t strides[2] = {(cuuint64_t)HD * 2, (cuuint64_t)HQ * HD * 2};
    cuuint32_t box[3] = {64, 1, BM};
    if (make_tmap_bf16_ptr(&tq, q, 3, dims, strides, box)) return -1;
  }
  {
    cuuint64_t dims[3] = {(cuuint64_t)HD, (cuuint64_t)HKV, (cuuint64_t)B * S};
    cuuint64_t strides[2] = {(cuuint64_t)HD * 2, (cuuint64_t)HKV * HD * 2};
    cuuint32_t box[3] = {64, 1, BN};
    if (make_tmap_bf16_ptr(&tk, g_k16, 3, dims, strides, box)) return -1;
    if (make_tmap_bf16_ptr(&tv, g_v16, 3, dims, strides, box)) return -1;
  }
  Params p;
  p.B = B; p.S = S; p.HQ = HQ; p.HKV = HKV;
  p.q_tiles = S / BM;
  p.num_tiles = B * HQ * p.q_tiles;
  p.scale_log2 = softmax_scale * 1.4426950408889634f;
  p.out = reinterpret_cast<__nv_bfloat16*>(o);
  if (!g_prof) { cudaMalloc(&g_prof, 1 << 20); cudaMemset(g_prof, 0, 1 << 20); }
  p.prof = g_prof;
  static int smem_set = 0;
  size_t smem = sizeof(Smem);
  if (!smem_set) {
    cudaError_t e = cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) { fprintf(stderr, "smem attr err %s (%zu)\n", cudaGetErrorString(e), smem); return -4; }
    smem_set = 1;
  }
  int dev; cudaGetDevice(&dev);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = grid_override > 0 ? grid_override : nsm;
  if (grid > p.num_tiles) grid = p.num_tiles;
  static int checked = 0;
  if (!checked) { cudaMemset(g_prof + 16383, 0, 8); }
  fa_fwd_kernel<<<grid, NUM_THREADS, smem, stream>>>(tq, tk, tv, p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "launch err %s\n", cudaGetErrorString(e)); return -2; }
  if (!checked) {
    unsigned long long flag = 0; cudaStreamSynchronize(stream); cudaMemcpy(&flag, g_prof + 16383, 8, cudaMemcpyDeviceToHost);
    if (flag == 0xBAD) { fprintf(stderr, "dynamic smem base not 1024-aligned\n"); return -5; }
    checked = 1;
  }
  return 0;
}
