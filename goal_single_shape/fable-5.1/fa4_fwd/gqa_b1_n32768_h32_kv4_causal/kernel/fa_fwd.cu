// FlashAttention-style forward (bf16, hdim 128, causal, GQA) for sm_100a, written from scratch.
// v9: cluster of 2 CTAs, 128 query rows each; 2-CTA UMMA (M=256, N=128) S = Q K^T into 3 TMEM S buffers;
// P (bf16) overwrites its S buffer and feeds O += P V from TMEM. Each 128-key block's softmax is split into two
// 64-column halves handled by two of three softmax warpgroups (units u = 2j + h, warpgroup u % 3), so a
// warpgroup's dependency chain is one 64-element unit. Running max is published at block start by the
// half-0 warpgroup and updated lazily (deferred) when a block's row sums get large.
#include "ptx.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>

namespace cfg {
constexpr int BN = 128;
constexpr int NK = 4, NV = 4;          // K/V ring depth (16KB per stage per CTA each)
constexpr int NSB = 3;                 // S buffers in TMEM (128 cols each)
constexpr int O_COL = 384;
constexpr float M_THRESH = 8.0f;
constexpr int NSW = 2;                 // softmax warpgroups
constexpr int NTHREADS = 128 * (1 + NSW);
constexpr int PV_LAG = NSB - 1;
#ifndef NPP_DEF
#define NPP_DEF 3
#endif
constexpr int NPP = NPP_DEF;           // poly pairs out of every 16 (interleaved)
}
DEVI constexpr bool is_poly_pair(int i) { return ((i * 5) & 15) < cfg::NPP; }

struct __align__(1024) Smem {
  uint8_t q[2][16384];              // Q boxes (d halves): 128 rows x 128B, SW128
  uint8_t k[cfg::NK][2][8192];      // K stage: 2 d-half boxes of 64 kv rows x 128B (this CTA's kv half)
  uint8_t v[cfg::NV][16384];        // V stage: 128 kv rows x 64 d (this CTA's d half)
  uint8_t o[2][16384];              // O staging (d halves)
  uint64_t bar_q, q_empty;
  uint64_t kfull[cfg::NK], kempty[cfg::NK], vfull[cfg::NV], vempty[cfg::NV];
  uint64_t s_full[cfg::NSB], p_ready[cfg::NSB], pv_done[cfg::NSB], all_done;
  float m_sh[2][128];
  float l_sh[cfg::NSW][128];
  uint64_t bar_m[2][4];             // m published for block j (slot j&1), per warp-row-group
  uint64_t gate_qk[8], gate_pv[8];  // gatekeeper -> issuer (ring of 8, phase per use)
  uint32_t dummy[cfg::NSB][4];      // landing slots for peer st.async signals
  uint32_t tmem_base;
};

struct Params {
  CUtensorMap tmQ, tmK, tmV, tmO;
  unsigned long long* dbg;
  int B, S, HQ, HKV, num_pairs;
  float scale_log2;
};

#ifdef DEBUG_TIMING
DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TRACE(slot, j) do { if (blockIdx.x < 2) P.dbg[(1u << 20) + (size_t)blockIdx.x * 65536 + (size_t)(j) * 16 + (slot)] = gtimer(); } while (0)
#else
#define TRACE(slot, j)
#endif

DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEVI uint32_t pack_bf16x2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI uint32_t mad_u32(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
DEVI void fma2(float& d0, float& d1, float a0, float a1, float b0, float b1, float c0, float c1) {
  asm("{.reg .b64 a, b, c, d; mov.b64 a, {%2,%3}; mov.b64 b, {%4,%5}; mov.b64 c, {%6,%7}; fma.rn.f32x2 d, a, b, c; mov.b64 {%0,%1}, d;}"
      : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
}
DEVI void add2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{.reg .b64 a, b, d; mov.b64 a, {%2,%3}; mov.b64 b, {%4,%5}; add.rn.f32x2 d, a, b; mov.b64 {%0,%1}, d;}"
      : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
DEVI void sub2(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{.reg .b64 a, b, d; mov.b64 a, {%2,%3}; mov.b64 b, {%4,%5}; sub.rn.f32x2 d, a, b; mov.b64 {%0,%1}, d;}"
      : "=f"(d0), "=f"(d1) : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
// packed-math poly: x already computed (pair)
DEVI void exp2_poly_pair2(float& r0, float& r1, float x0, float x1) {
  constexpr float MAGIC = 12582912.0f;
  constexpr float C1 = 0.70291467f, C2 = 0.23984623f;
  x0 = fmaxf(x0, -126.f); x1 = fmaxf(x1, -126.f);
  float t0, t1, n0, n1, f0, f1, p0, p1;
  add2(t0, t1, x0, x1, MAGIC, MAGIC);
  sub2(n0, n1, t0, t1, MAGIC, MAGIC);
  sub2(f0, f1, x0, x1, n0, n1);
  fma2(p0, p1, f0, f1, C2, C2, C1, C1);
  fma2(p0, p1, p0, p1, f0, f1, 1.f, 1.f);
  r0 = __uint_as_float(mad_u32(__float_as_uint(t0), 1u << 23, __float_as_uint(p0)));
  r1 = __uint_as_float(mad_u32(__float_as_uint(t1), 1u << 23, __float_as_uint(p1)));
}
DEVI void exp2_poly_pair(float& r0, float& r1, float x0, float x1) {
  constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23
  constexpr float C1 = 0.70291467f, C2 = 0.23984623f;
  x0 = fmaxf(x0, -126.f); x1 = fmaxf(x1, -126.f);
  const float t0 = x0 + MAGIC, t1 = x1 + MAGIC;
  const float n0 = t0 - MAGIC, n1 = t1 - MAGIC;
  const float f0 = x0 - n0, f1 = x1 - n1;
  float p0 = fmaf(f0, C2, C1), p1 = fmaf(f1, C2, C1);
  p0 = fmaf(p0, f0, 1.f); p1 = fmaf(p1, f1, 1.f);
  r0 = __uint_as_float(mad_u32(__float_as_uint(t0), 1u << 23, __float_as_uint(p0)));
  r1 = __uint_as_float(mad_u32(__float_as_uint(t1), 1u << 23, __float_as_uint(p1)));
}
#define DEP32(u) asm volatile("" : "+r"((u)[0]), "+r"((u)[1]), "+r"((u)[2]), "+r"((u)[3]), "+r"((u)[4]), "+r"((u)[5]), "+r"((u)[6]), "+r"((u)[7]), \
  "+r"((u)[8]), "+r"((u)[9]), "+r"((u)[10]), "+r"((u)[11]), "+r"((u)[12]), "+r"((u)[13]), "+r"((u)[14]), "+r"((u)[15]), \
  "+r"((u)[16]), "+r"((u)[17]), "+r"((u)[18]), "+r"((u)[19]), "+r"((u)[20]), "+r"((u)[21]), "+r"((u)[22]), "+r"((u)[23]), \
  "+r"((u)[24]), "+r"((u)[25]), "+r"((u)[26]), "+r"((u)[27]), "+r"((u)[28]), "+r"((u)[29]), "+r"((u)[30]), "+r"((u)[31]))

// exp2 + pack + row-sum for 32 elements (16 pairs, poly/MUFU interleaved), packed f32x2 math
DEVI void exp_block32(const float* sv, float c, float nm, uint32_t* pk, float& rs, float& maxpart) {
  float ra[4], rb[4];
#pragma unroll
  for (int i = 0; i < 4; i++) { ra[i] = 0.f; rb[i] = 0.f; }
#pragma unroll
  for (int i = 0; i < 16; i++) {
    float x0, x1, p0, p1;
    fma2(x0, x1, sv[2 * i], sv[2 * i + 1], c, c, nm, nm);
    if (is_poly_pair(i)) exp2_poly_pair2(p0, p1, x0, x1);
    else { p0 = ex2f(x0); p1 = ex2f(x1); }
    add2(ra[i & 3], rb[i & 3], ra[i & 3], rb[i & 3], p0, p1);
    pk[i] = pack_bf16x2(p0, p1);
  }
  float t[4];
#pragma unroll
  for (int i = 0; i < 4; i++) t[i] = ra[i] + rb[i];
  rs = (t[0] + t[1]) + (t[2] + t[3]);
  maxpart = fmaxf(fmaxf(t[0], t[1]), fmaxf(t[2], t[3]));
}
// exp2 + pack + row-sum for 64 elements (32 pairs, poly/MUFU interleaved)
DEVI void exp_block64(const float* sv, float c, float nm, uint32_t* pk, float& rs, float& maxpart) {
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    const float x0 = fmaf(sv[2 * i], c, nm), x1 = fmaf(sv[2 * i + 1], c, nm);
    float p0, p1;
    if (is_poly_pair(i)) exp2_poly_pair(p0, p1, x0, x1);
    else { p0 = ex2f(x0); p1 = ex2f(x1); }
    acc[i & 7] += p0 + p1;
    pk[i] = pack_bf16x2(p0, p1);
  }
  rs = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  maxpart = fmaxf(fmaxf(fmaxf(acc[0], acc[1]), fmaxf(acc[2], acc[3])), fmaxf(fmaxf(acc[4], acc[5]), fmaxf(acc[6], acc[7])));
}

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(cfg::NTHREADS, 1)
fa_fwd_kernel(const __grid_constant__ Params P) {
  extern __shared__ uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>((((uintptr_t)smem_raw) + 1023) & ~(uintptr_t)1023);
  const uint32_t rank = cluster_ctarank();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

  const int num_clusters = gridDim.x >> 1;
  const int cluster_id = blockIdx.x >> 1;
  const int BH = P.B * P.HQ;
  const int num_tiles = BH * P.num_pairs;
  // tile t (descending work order) -> (pair, batch, head). Snake assignment: round r, cluster c -> t = r*NC + (r even ? c : NC-1-c)
  auto tile_of = [&](int r) { return r * num_clusters + ((r & 1) ? (num_clusters - 1 - cluster_id) : cluster_id); };
  const int num_rounds = (num_tiles + num_clusters - 1) / num_clusters;
  int my_rounds = 0;
  for (int r = 0; r < num_rounds; r++) if (tile_of(r) < num_tiles) my_rounds++;

  if (threadIdx.x == 0) {
    mbar_init(smem_u32(&sm.bar_q), 1); mbar_init(smem_u32(&sm.q_empty), 1);
    for (int s = 0; s < cfg::NK; s++) { mbar_init(smem_u32(&sm.kfull[s]), 1); mbar_init(smem_u32(&sm.kempty[s]), 1); }
    for (int s = 0; s < cfg::NV; s++) { mbar_init(smem_u32(&sm.vfull[s]), 1); mbar_init(smem_u32(&sm.vempty[s]), 1); }
    for (int i = 0; i < cfg::NSB; i++) { mbar_init(smem_u32(&sm.s_full[i]), 1); mbar_init(smem_u32(&sm.p_ready[i]), 4); mbar_init(smem_u32(&sm.pv_done[i]), 1); }
    mbar_init(smem_u32(&sm.all_done), 1);
    for (int i = 0; i < 8; i++) mbar_init(smem_u32(&sm.bar_m[i >> 2][i & 3]), 32);
    for (int i = 0; i < 8; i++) { mbar_init(smem_u32(&sm.gate_qk[i]), 1); mbar_init(smem_u32(&sm.gate_pv[i]), 1); }
    fence_mbar_init();
  }
  if (warp == 2) { tmem_alloc2(smem_u32(&sm.tmem_base), 512); tmem_relinquish2(); }
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = sm.tmem_base;
  cluster_sync();

  struct TileInfo { int h, hk, qrow0, kvbase, nblk, jdiag; };
  auto decode = [&](int t) {
    TileInfo ti;
    const int bh = t % BH, pr = t / BH;
    const int pidx = P.num_pairs - 1 - pr;
    const int b = bh / P.HQ; ti.h = bh % P.HQ;
    ti.hk = ti.h / (P.HQ / P.HKV);
    ti.qrow0 = b * P.S + pidx * 256 + 128 * (int)rank;
    ti.kvbase = b * P.S;
    ti.nblk = 2 * pidx + 2;
    ti.jdiag = 2 * pidx + (int)rank;
    return ti;
  };

  if (warp < 4) {
    SETMAXNREG_DEC(56);
    const uint32_t tmem_u = __shfl_sync(0xffffffffu, tmem, 0);
    if (warp == 0) {
      // ================= TMA producer =================
      if (elect_one()) { tma_prefetch_desc(&P.tmQ); tma_prefetch_desc(&P.tmK); tma_prefetch_desc(&P.tmV); }
      const uint32_t bq = smem_u32(&sm.bar_q);
      const uint32_t bq_l = mapa_shared(bq, 0);
      int g0 = 0;
      for (int r = 0; r < my_rounds; r++) {
        const TileInfo ti = decode(tile_of(r));
        if (r > 0) mbar_wait(smem_u32(&sm.q_empty), (r - 1) & 1);   // previous tile's QKs all done with the Q buffer
        if (elect_one()) {
          if (rank == 0) mbar_arrive_expect_tx(bq, 65536);
          tma_load_3d_cg2(smem_u32(sm.q[0]), &P.tmQ, bq_l, 0, ti.h, ti.qrow0);
          tma_load_3d_cg2(smem_u32(sm.q[1]), &P.tmQ, bq_l, 64, ti.h, ti.qrow0);
        }
        for (int j = 0; j < ti.nblk + cfg::PV_LAG; j++) {
          if (j < ti.nblk) {
            const int gj = g0 + j, s = gj % cfg::NK, rr = gj / cfg::NK;
            mbar_wait(smem_u32(&sm.kempty[s]), (rr & 1) ^ 1);
            const uint32_t bf = smem_u32(&sm.kfull[s]);
            const uint32_t bf_l = mapa_shared(bf, 0);
            const int kvrow = ti.kvbase + j * cfg::BN + 64 * (int)rank;
            if (elect_one()) {
#ifdef SKIP_TMA
              if (rank == 0) mbar_arrive_expect_tx(bf, 0);
#else
              if (rank == 0) mbar_arrive_expect_tx(bf, 32768);
              tma_load_3d_cg2(smem_u32(sm.k[s][0]), &P.tmK, bf_l, 0, ti.hk, kvrow);
              tma_load_3d_cg2(smem_u32(sm.k[s][1]), &P.tmK, bf_l, 64, ti.hk, kvrow);
#endif
            }
          }
          if (j >= cfg::PV_LAG) {
            const int i = j - cfg::PV_LAG, gi = g0 + i, s = gi % cfg::NV, rr = gi / cfg::NV;
            mbar_wait(smem_u32(&sm.vempty[s]), (rr & 1) ^ 1);
            const uint32_t bf = smem_u32(&sm.vfull[s]);
            const uint32_t bf_l = mapa_shared(bf, 0);
            if (elect_one()) {
#ifdef SKIP_TMA
              if (rank == 0) mbar_arrive_expect_tx(bf, 0);
#else
              if (rank == 0) mbar_arrive_expect_tx(bf, 32768);
              tma_load_3d_cg2(smem_u32(sm.v[s]), &P.tmV, bf_l, 64 * (int)rank, ti.hk, ti.kvbase + i * cfg::BN);
#endif
            }
          }
        }
        g0 += ti.nblk;
      }
    } else if (warp == 2 && rank == 0) {
      // ================= gatekeeper =================
      int g0 = 0;
      for (int r = 0; r < my_rounds; r++) {
        const TileInfo ti = decode(tile_of(r));
        mbar_wait(smem_u32(&sm.bar_q), r & 1);
        for (int j = 0; j < ti.nblk; j++) {
          const int gj = g0 + j;
          mbar_wait(smem_u32(&sm.kfull[gj % cfg::NK]), (gj / cfg::NK) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_u32(&sm.gate_qk[gj & 7]));
          if (j >= cfg::PV_LAG) {
            const int gi = gj - cfg::PV_LAG;
            mbar_wait_cluster(smem_u32(&sm.p_ready[gi % cfg::NSB]), (gi / cfg::NSB) & 1);
            mbar_wait(smem_u32(&sm.vfull[gi % cfg::NV]), (gi / cfg::NV) & 1);
            __syncwarp();
            if (lane == 0) mbar_arrive(smem_u32(&sm.gate_pv[gi & 7]));
          }
        }
        for (int i = ti.nblk - cfg::PV_LAG; i < ti.nblk; i++) {
          const int gi = g0 + i;
          mbar_wait_cluster(smem_u32(&sm.p_ready[gi % cfg::NSB]), (gi / cfg::NSB) & 1);
          mbar_wait(smem_u32(&sm.vfull[gi % cfg::NV]), (gi / cfg::NV) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_u32(&sm.gate_pv[gi & 7]));
        }
        g0 += ti.nblk;
      }
    } else if (warp == 1 && rank == 0) {
      // ================= MMA issuer (leader) =================
      const uint32_t idesc_qk = make_idesc_bf16(256, 128, 0, 0);
      const uint32_t idesc_pv = make_idesc_bf16(256, 128, 0, 1);
      const uint32_t gq0 = smem_u32(&sm.gate_qk[0]), gp0 = smem_u32(&sm.gate_pv[0]);
      const uint32_t kbase0 = smem_u32(sm.k[0][0]), vbase0 = smem_u32(sm.v[0]);
      const uint32_t sfull0 = smem_u32(&sm.s_full[0]), pvd0 = smem_u32(&sm.pv_done[0]);
      const uint32_t kempty0 = smem_u32(&sm.kempty[0]), vempty0 = smem_u32(&sm.vempty[0]);
      const uint64_t qdesc = make_smem_desc(smem_u32(sm.q[0]), 16, 1024, 2);
      const uint32_t o_t = tmem_u + cfg::O_COL;
#ifdef DEBUG_TIMING
      const unsigned long long t_clk0 = clock64(), t_ns0 = gtimer();
      unsigned long long t_gq = 0, t_iq = 0, t_gp = 0, t_ip = 0, tot_blk = 0;
#define IT0() unsigned long long _t0 = clock64()
#define IT1(acc) do { unsigned long long _t1 = clock64(); acc += _t1 - _t0; _t0 = _t1; } while (0)
#else
#define IT0()
#define IT1(acc)
#endif
      int buf_q = 0, st_k = 0, buf_p = 0, st_v = 0;
      int gq = 0, gp = 0;   // global block counters for QK / PV
      for (int r = 0; r < my_rounds; r++) {
        const TileInfo ti = decode(tile_of(r));
        mbar_wait(smem_u32(&sm.bar_q), r & 1);
        tc_fence_after();
        auto issue_pv = [&](int i) {
          IT0();
          mbar_wait(gp0 + (gp & 7) * 8, (gp >> 3) & 1);
          IT1(t_gp);
          tc_fence_after();
          const uint32_t p_t = tmem_u + buf_p * 128;
          const uint64_t vdesc = make_smem_desc(vbase0 + st_v * 16384, 16384, 1024, 2);
          const uint32_t pvd = pvd0 + buf_p * 8, vemp = vempty0 + st_v * 8;
          const uint32_t acc0 = (i > 0) ? 1u : 0u;
          if (elect_one()) {
#ifndef SKIP_MMA
            umma2_ts(o_t, p_t, vdesc, idesc_pv, acc0);
#pragma unroll
            for (int ks = 1; ks < 8; ks++) umma2_ts(o_t, p_t + ks * 8, vdesc + (uint64_t)((ks * 2048) >> 4), idesc_pv, 1u);
#endif
            umma2_commit_mc(pvd, 0x3);
            umma2_commit_mc(vemp, 0x3);
          }
          __syncwarp();
          IT1(t_ip);
          gp++;
          buf_p = (buf_p == cfg::NSB - 1) ? 0 : buf_p + 1;
          st_v = (st_v == cfg::NV - 1) ? 0 : st_v + 1;
        };
        for (int j = 0; j < ti.nblk; j++) {
          IT0();
          mbar_wait(gq0 + (gq & 7) * 8, (gq >> 3) & 1);
          IT1(t_gq);
          tc_fence_after();
          {
            const uint32_t d_t = tmem_u + buf_q * 128;
            const uint64_t kdesc = make_smem_desc(kbase0 + st_k * 16384, 16, 1024, 2);
            const uint32_t sfull = sfull0 + buf_q * 8, kemp = kempty0 + st_k * 8;
            if (elect_one()) {
#ifndef SKIP_MMA
#pragma unroll
              for (int ks = 0; ks < 8; ks++) {
                const uint64_t off = (uint64_t)((ks >> 2) * 16384 + (ks & 3) * 32) >> 4;
                const uint64_t koff = (uint64_t)((ks >> 2) * 8192 + (ks & 3) * 32) >> 4;
                umma2_ss(d_t, qdesc + off, kdesc + koff, idesc_qk, ks > 0);
              }
#endif
              umma2_commit_mc(sfull, 0x3);
              umma2_commit_mc(kemp, 0x3);
              if (j == ti.nblk - 1) umma2_commit_mc(smem_u32(&sm.q_empty), 0x3);
            }
            __syncwarp();
            IT1(t_iq);
          }
          gq++;
          buf_q = (buf_q == cfg::NSB - 1) ? 0 : buf_q + 1;
          st_k = (st_k == cfg::NK - 1) ? 0 : st_k + 1;
          if (j >= cfg::PV_LAG) issue_pv(j - cfg::PV_LAG);
        }
        for (int i = ti.nblk - cfg::PV_LAG; i < ti.nblk; i++) issue_pv(i);
        if (elect_one()) umma2_commit_mc(smem_u32(&sm.all_done), 0x3);
        __syncwarp();
#ifdef DEBUG_TIMING
        tot_blk += ti.nblk;
#endif
      }
#ifdef DEBUG_TIMING
      mbar_wait(smem_u32(&sm.all_done), (my_rounds - 1) & 1);
      if (lane == 0) { unsigned long long* d = P.dbg + (size_t)blockIdx.x * 16; d[6] = clock64() - t_clk0; d[7] = gtimer() - t_ns0; d[5] = tot_blk; d[1] = t_gq; d[2] = t_iq; d[3] = t_gp; d[4] = t_ip; }
#endif
    }
  } else {
    SETMAXNREG_INC(224);
    // ================= softmax warpgroups: unit u = 2*j + h (block j, column half h), warpgroup u % NSW =================
    const int wg = (warp >> 2) - 1;
    const int wq = warp & 3;
    const int row = wq * 32 + lane;
    const uint32_t lane_base = (uint32_t)(wq * 32) << 16;
    const float c_log2 = P.scale_log2;
    float m_ref = -INFINITY, l_acc = 0.f;
    float m_pending = -INFINITY;
    const uint32_t p_ready_l = mapa_shared(smem_u32(&sm.p_ready[0]), 0);
    const uint32_t dummy_l = mapa_shared(smem_u32(&sm.dummy[0][0]), 0);

    auto publish = [&](int buf, int hf) {
      tmem_st_wait();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) {
        if (rank == 0) { if (wq == 0) mbar_arrive_expect_tx(smem_u32(&sm.p_ready[buf]), 16); else mbar_arrive(smem_u32(&sm.p_ready[buf])); }
        else st_async_signal(dummy_l + (buf * 4 + wq) * 4, 1u, p_ready_l + buf * 8);
      }
    };
    auto load_s = [&](uint32_t s_t, float* sv) {
      uint32_t u[64];
      tmem_ld32(s_t, u); tmem_ld32(s_t + 32, u + 32);
      tmem_ld_wait();
      DEP32(u); DEP32(u + 32);
#pragma unroll
      for (int i = 0; i < 64; i++) sv[i] = __uint_as_float(u[i]);
    };
    auto wait_m = [&](uint32_t ver) {   // returns m of block (ver-1) once published
      const int jj = (int)ver - 1;
      mbar_wait(smem_u32(&sm.bar_m[jj & 1][wq]), (jj >> 1) & 1);
      return sm.m_sh[jj & 1][row];
    };
    auto publish_m = [&](int j, float m) {
      sm.m_sh[j & 1][row] = m;
      mbar_arrive(smem_u32(&sm.bar_m[j & 1][wq]));
    };
    auto rescale_O = [&](int j, float m_prev, float m_new) {
      mbar_wait(smem_u32(&sm.pv_done[(j - 1) % cfg::NSB]), ((j - 1) / cfg::NSB) & 1);
      tc_fence_after();
      const float alpha = ex2f(m_prev - m_new);
      const uint32_t o_t = tmem + lane_base + cfg::O_COL;
#pragma unroll
      for (int c4 = 0; c4 < 4; c4++) {
        uint32_t r[32];
        tmem_ld32(o_t + c4 * 32, r);
        tmem_ld_wait();
        DEP32(r);
#pragma unroll
        for (int i = 0; i < 32; i++) r[i] = __float_as_uint(__uint_as_float(r[i]) * alpha);
        tmem_st32(o_t + c4 * 32, r);
      }
      tmem_st_wait();
    };

    int g0 = 0;
    for (int r = 0; r < my_rounds; r++) {
      const TileInfo ti = decode(tile_of(r));
      const int nblk = ti.nblk, jdiag = ti.jdiag;
      m_ref = -INFINITY; l_acc = 0.f; m_pending = -INFINITY;
    for (int j = wg; j < nblk; j += cfg::NSW) {
      const int gj = g0 + j;
      const int buf = gj % cfg::NSB;
      const uint32_t s_t = tmem + lane_base + buf * 128;
      mbar_wait(smem_u32(&sm.s_full[buf]), (gj / cfg::NSB) & 1);
      tc_fence_after();
      if (j > jdiag) {
        const float m_prev = wait_m((uint32_t)gj); publish_m(gj, m_prev);
        uint32_t pk[32];
#pragma unroll
        for (int i = 0; i < 32; i++) pk[i] = 0u;
        tmem_st32(s_t, pk);
        tmem_st32(s_t + 32, pk);
        publish(buf, 0);
        continue;
      }
      float m_use = 0.f;
      float rs = 0.f, mp = 0.f;
#pragma unroll
      for (int qtr = 0; qtr < 4; qtr++) {
        float sv[32];
        {
          uint32_t u[32];
          tmem_ld32(s_t + qtr * 32, u);
          tmem_ld_wait();
          DEP32(u);
#pragma unroll
          for (int i = 0; i < 32; i++) sv[i] = __uint_as_float(u[i]);
        }
        if (j == jdiag) {
          const int lim = row - qtr * 32;   // valid iff c <= lim
#pragma unroll
          for (int c = 0; c < 32; c++) sv[c] = (c > lim) ? -INFINITY : sv[c];
        }
        if (qtr == 0) {
          float m_prev;
          if (j == 0) {
            float mx[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mx[i] = sv[i];
#pragma unroll
            for (int c = 8; c < 32; c += 8)
#pragma unroll
              for (int i = 0; i < 8; i++) mx[i] = fmaxf(mx[i], sv[c + i]);
            m_prev = fmaxf(fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3])), fmaxf(fmaxf(mx[4], mx[5]), fmaxf(mx[6], mx[7]))) * c_log2;
          } else {
            m_prev = wait_m((uint32_t)gj);
          }
          m_use = fmaxf(m_prev, m_pending);
          m_pending = -INFINITY;
          publish_m(gj, m_use);
          if (j > 0 && m_use != m_prev) rescale_O(gj, m_prev, m_use);
          if (m_ref != m_use) { l_acc *= ex2f(m_ref - m_use); m_ref = m_use; }
        }
        uint32_t pk[16]; float rsq, mpq;
#ifdef SKIP_SOFTMAX
#pragma unroll
        for (int i = 0; i < 16; i++) pk[i] = 0x3c003c00u; rsq = 32.f; mpq = 8.f;
#else
        exp_block32(sv, c_log2, -m_use, pk, rsq, mpq);
#endif
        tmem_st16(s_t + qtr * 16, pk);
        rs += rsq; mp = fmaxf(mp, mpq);
      }
      publish(buf, 0);
      l_acc += rs;
      constexpr float RS_MAX = 16777216.f;   // 2^24
      if (!(rs <= RS_MAX)) {
        float lg; asm("lg2.approx.f32 %0, %1;" : "=f"(lg) : "f"(mp));
        m_pending = (lg >= 0.f && lg < 100.f) ? m_use + lg : m_use + 64.f;
      }
    }

    // ================= epilogue (per tile) =================
    {
      const float m_fin = wait_m((uint32_t)(g0 + nblk));
      if (m_ref != m_fin) l_acc *= ex2f(m_ref - m_fin);
      sm.l_sh[wg][row] = l_acc;
      named_bar_sync(1, 128 * cfg::NSW);
      float l_tot = 0.f;
#pragma unroll
      for (int g = 0; g < cfg::NSW; g++) l_tot += sm.l_sh[g][row];
      const float inv_l = 1.f / l_tot;
      mbar_wait(smem_u32(&sm.all_done), r & 1);
      tc_fence_after();
      const uint32_t o_t = tmem + lane_base + cfg::O_COL + 64 * wg;
      uint32_t rr[64];
      tmem_ld32(o_t, rr);
      tmem_ld32(o_t + 32, rr + 32);
      tmem_ld_wait();
      DEP32(rr); DEP32(rr + 32);
      if (wq == 0 && lane == 0 && r > 0) tma_store_wait_read0();   // previous tile's O store has read the staging buffer
      named_bar_sync(2 + wg, 128);
      uint8_t* dst = sm.o[wg] + row * 128;
#pragma unroll
      for (int c = 0; c < 8; c++) {
        uint4 w;
        w.x = pack_bf16x2(__uint_as_float(rr[c * 8 + 0]) * inv_l, __uint_as_float(rr[c * 8 + 1]) * inv_l);
        w.y = pack_bf16x2(__uint_as_float(rr[c * 8 + 2]) * inv_l, __uint_as_float(rr[c * 8 + 3]) * inv_l);
        w.z = pack_bf16x2(__uint_as_float(rr[c * 8 + 4]) * inv_l, __uint_as_float(rr[c * 8 + 5]) * inv_l);
        w.w = pack_bf16x2(__uint_as_float(rr[c * 8 + 6]) * inv_l, __uint_as_float(rr[c * 8 + 7]) * inv_l);
        *reinterpret_cast<uint4*>(dst + ((c ^ (row & 7)) << 4)) = w;
      }
      fence_proxy_async_smem();
      named_bar_sync(2 + wg, 128);
      if (wq == 0 && lane == 0) {
        tma_store_3d(&P.tmO, smem_u32(sm.o[wg]), 64 * wg, ti.h, ti.qrow0);
        tma_store_commit();
      }
      // the O accumulator may be overwritten by the next tile's first PV only after every thread finished reading it
      tc_fence_before();
      named_bar_sync(1, 128 * cfg::NSW);
    }
    g0 += nblk;
    }
    if (wq == 0 && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc2(tmem, 512);
  cluster_sync();
}

// ------------------------------------------------------------------------------------------------
static unsigned long long* g_dbg = nullptr;
extern "C" int fa_max_clusters(int cluster_size) {
  cudaLaunchConfig_t cfgl = {};
  cfgl.gridDim = dim3(cluster_size * 64, 1, 1); cfgl.blockDim = dim3(cfg::NTHREADS, 1, 1); cfgl.dynamicSmemBytes = sizeof(Smem) + 1024;
  cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim.x = cluster_size; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
  cfgl.attrs = &attr; cfgl.numAttrs = 1;
  cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(sizeof(Smem) + 1024));
  if (cluster_size != 2) cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  int n = 0; cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)fa_fwd_kernel, &cfgl);
  return e == cudaSuccess ? n : -(int)e;
}
extern "C" void fa_dbg_read(unsigned long long* host, int n) { if (g_dbg) cudaMemcpy(host, g_dbg, (size_t)n * 8, cudaMemcpyDeviceToHost); }
extern "C" int fa_fwd(const void* q, const void* k, const void* v, void* o, int B, int S, int HQ, int HKV,
                      float softmax_scale, cudaStream_t stream) {
  if (S % 256 != 0 || HQ % HKV != 0) return -2;
  Params p;
  const uint64_t rows = (uint64_t)B * S;
  {
    uint64_t dims[3] = {128, (uint64_t)HQ, rows}, str[2] = {256, (uint64_t)HQ * 256};
    uint32_t box[3] = {64, 1, 128};
    if (make_tmap_bf16(&p.tmQ, (void*)q, 3, dims, str, box)) return -1;
    if (make_tmap_bf16(&p.tmO, o, 3, dims, str, box)) return -1;
  }
  {
    uint64_t dims[3] = {128, (uint64_t)HKV, rows}, str[2] = {256, (uint64_t)HKV * 256};
    uint32_t boxk[3] = {64, 1, 64}, boxv[3] = {64, 1, 128};
    if (make_tmap_bf16(&p.tmK, (void*)k, 3, dims, str, boxk)) return -1;
    if (make_tmap_bf16(&p.tmV, (void*)v, 3, dims, str, boxv)) return -1;
  }
  p.B = B; p.S = S; p.HQ = HQ; p.HKV = HKV; p.num_pairs = S / 256;
  p.scale_log2 = softmax_scale * 1.4426950408889634f;
#ifdef DEBUG_TIMING
  if (!g_dbg) { cudaMalloc(&g_dbg, 16 * 8 * 1000000); cudaMemset(g_dbg, 0, 16 * 8 * 1000000); }
#endif
  p.dbg = g_dbg;
  static bool attr_set = false;
  const size_t smem = sizeof(Smem) + 1024;
  if (!attr_set) {
    if (cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) return -3;
    attr_set = true;
  }
  static int num_sms = 0;
  if (!num_sms) cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  const int num_tiles = B * HQ * p.num_pairs;
  int clusters = num_sms / 2; if (clusters > num_tiles) clusters = num_tiles;
  { const char* ev = getenv("FA_CLUSTERS"); if (ev) { int c = atoi(ev); if (c > 0 && c <= clusters) clusters = c; } }
  const int grid = 2 * clusters;
  fa_fwd_kernel<<<grid, cfg::NTHREADS, smem, stream>>>(p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); return -4; }
  return 0;
}
