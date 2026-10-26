// From-scratch FlashAttention forward (causal, bf16, D=128) for Blackwell sm_100a.
// Warp-specialized: warp0 = TMA producer, warp1 = MMA issuer, warp2 = TMEM alloc,
// warps 4-7 / 8-11 = two softmax warpgroups (128 Q rows each). CTA = 256 Q rows.
#include <stdio.h>
#include <math.h>
#include "dev_common.h"

#define D_HEAD 128
#define BM 256          // Q rows per CTA
#define BN 128          // KV rows per tile
#define NSTAGE 2
#define THR 8.0f        // lazy-rescale threshold in log2 units

// smem layout (bytes)
#define SQ_OFF 0                       // 4 boxes x 16KB  (wg*2 + dhalf)
#define SK_OFF 65536                   // NSTAGE x 32KB (2 boxes each)
#define SV_OFF (SK_OFF + NSTAGE*32768)
#define BAR_OFF (SV_OFF + NSTAGE*32768)
#define SMEM_BYTES (BAR_OFF + 1024)

struct Bars {
  uint64_t q_full, q_empty;
  uint64_t k_full[NSTAGE], k_empty[NSTAGE];
  uint64_t v_full[NSTAGE], v_empty[NSTAGE];
  uint64_t s_full[2], p_full[2], pv_done[2];
  uint64_t work_full[2], work_empty[2];
  uint32_t tmem_slot;
  int work_item[2];
};

struct Params {
  __nv_bfloat16* O;
  int* ws;
  int total_items, items_per_chunk, G, NT, B, S, H;
  float scale_log2;
};

__device__ __forceinline__ void decode_item(const Params& p, int item, int& b, int& h, int& t) {
  int chunk = item / p.items_per_chunk; int r = item - chunk * p.items_per_chunk;
  t = p.NT - 1 - r / p.G;
  int head = chunk * p.G + (r % p.G);
  b = head / p.H; h = head - b * p.H;
}

// QK^T: D[tmem] = Q_wg[smem] * K[smem]^T   (both K-major)
__device__ __forceinline__ void issue_qk(uint32_t d_tmem, uint32_t sQ, uint32_t sK, uint32_t idesc) {
  #pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    uint32_t off = (kk >> 2) * 16384 + (kk & 3) * 32;
    mma_ss(d_tmem, make_smem_desc(sQ + off, 16, 1024), make_smem_desc(sK + off, 16, 1024), idesc, kk > 0);
  }
}
// PV: D[tmem] (+)= P[tmem] * V[smem]  (V MN-major)
__device__ __forceinline__ void issue_pv(uint32_t d_tmem, uint32_t p_tmem, uint32_t sV, uint32_t idesc, uint32_t accum) {
  #pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    mma_ts(d_tmem, p_tmem + kk * 8, make_smem_desc(sV + kk * 2048, 16384, 1024), idesc, (kk > 0) | accum);
  }
}

__global__ void __launch_bounds__(384, 1) fa_fwd_kernel(const __grid_constant__ CUtensorMap tmQ,
                                                       const __grid_constant__ CUtensorMap tmK,
                                                       const __grid_constant__ CUtensorMap tmV,
                                                       Params p) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  Bars* bars = (Bars*)(smem + BAR_OFF);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t sQ = smem_u32(smem + SQ_OFF), sK = smem_u32(smem + SK_OFF), sV = smem_u32(smem + SV_OFF);

  if (tid == 0) {
    mbar_init(smem_u32(&bars->q_full), 1); mbar_init(smem_u32(&bars->q_empty), 1);
    for (int i = 0; i < NSTAGE; i++) {
      mbar_init(smem_u32(&bars->k_full[i]), 1); mbar_init(smem_u32(&bars->k_empty[i]), 1);
      mbar_init(smem_u32(&bars->v_full[i]), 1); mbar_init(smem_u32(&bars->v_empty[i]), 1);
    }
    for (int i = 0; i < 2; i++) {
      mbar_init(smem_u32(&bars->s_full[i]), 1); mbar_init(smem_u32(&bars->p_full[i]), 4);
      mbar_init(smem_u32(&bars->pv_done[i]), 1);
      mbar_init(smem_u32(&bars->work_full[i]), 1); mbar_init(smem_u32(&bars->work_empty[i]), 3);
    }
    mbar_fence_init();
  }
  if (warp == 0 && lane == 0) { tma_prefetch_desc(&tmQ); tma_prefetch_desc(&tmK); tma_prefetch_desc(&tmV); }
  __syncthreads();
  if (warp == 2) { tmem_alloc(smem_u32(&bars->tmem_slot), 512); tmem_relinquish(); }
  __syncthreads();
  const uint32_t tmem = *(volatile uint32_t*)&bars->tmem_slot;
  const uint32_t S_COL0 = 0, O_COL0 = 256;   // per-wg: S at wg*128, O at 256+wg*128, P at wg*128

  if (warp < 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == 0 && lane == 0) {
      // ================= TMA producer =================
      uint32_t item_cnt = 0, kv_cnt = 0;
      int item = atomicAdd(&p.ws[0], 1); if (item >= p.total_items) item = -1;
      *(volatile int*)&bars->work_item[0] = item;
      mbar_arrive(smem_u32(&bars->work_full[0]));
      while (item >= 0) {
        int b, h, t; decode_item(p, item, b, h, t);
        const int q0 = t * BM, J = 2 * t + 2, rowQ = b * p.S + q0, rowKV0 = b * p.S;
        mbar_wait(smem_u32(&bars->q_empty), (item_cnt & 1) ^ 1);
        mbar_arrive_expect_tx(smem_u32(&bars->q_full), 65536);
        #pragma unroll
        for (int i = 0; i < 4; i++)
          tma_load_3d(sQ + i * 16384, &tmQ, smem_u32(&bars->q_full), (i & 1) * 64, h, rowQ + (i >> 1) * 128);
        // grab next item, publish it, prefetch its Q into L2
        {
          uint32_t nslot = (item_cnt + 1) & 1, nph = ((item_cnt + 1) >> 1) & 1;
          mbar_wait(smem_u32(&bars->work_empty[nslot]), nph ^ 1);
          int nitem = atomicAdd(&p.ws[0], 1); if (nitem >= p.total_items) nitem = -1;
          *(volatile int*)&bars->work_item[nslot] = nitem;
          mbar_arrive(smem_u32(&bars->work_full[nslot]));
          if (nitem >= 0) {
            int nb, nh, nt; decode_item(p, nitem, nb, nh, nt);
            int nrow = nb * p.S + nt * BM;
            #pragma unroll
            for (int i = 0; i < 4; i++) tma_prefetch_3d(&tmQ, (i & 1) * 64, nh, nrow + (i >> 1) * 128);
          }
          item = nitem;
        }
        for (int j = 0; j < J; j++) {
          uint32_t s = kv_cnt & 1, ph = (kv_cnt >> 1) & 1;
          mbar_wait(smem_u32(&bars->k_empty[s]), ph ^ 1);
          mbar_arrive_expect_tx(smem_u32(&bars->k_full[s]), 32768);
          tma_load_3d(sK + s * 32768,         &tmK, smem_u32(&bars->k_full[s]), 0,  h, rowKV0 + j * BN);
          tma_load_3d(sK + s * 32768 + 16384, &tmK, smem_u32(&bars->k_full[s]), 64, h, rowKV0 + j * BN);
          mbar_wait(smem_u32(&bars->v_empty[s]), ph ^ 1);
          mbar_arrive_expect_tx(smem_u32(&bars->v_full[s]), 32768);
          tma_load_3d(sV + s * 32768,         &tmV, smem_u32(&bars->v_full[s]), 0,  h, rowKV0 + j * BN);
          tma_load_3d(sV + s * 32768 + 16384, &tmV, smem_u32(&bars->v_full[s]), 64, h, rowKV0 + j * BN);
          kv_cnt++;
        }
        item_cnt++;
      }
    } else if (warp == 1 && lane == 0) {
      // ================= MMA issuer =================
      uint32_t item_cnt = 0, kv_cnt = 0, p_cnt0 = 0, p_cnt1 = 0;
      const uint32_t idesc_qk = make_idesc(128, 128, false, false);
      const uint32_t idesc_pv = make_idesc(128, 128, false, true);
      while (true) {
        uint32_t slot = item_cnt & 1, ph = (item_cnt >> 1) & 1;
        mbar_wait(smem_u32(&bars->work_full[slot]), ph);
        int item = *(volatile int*)&bars->work_item[slot];
        mbar_arrive(smem_u32(&bars->work_empty[slot]));
        if (item < 0) break;
        int b, h, t; decode_item(p, item, b, h, t);
        const int J = 2 * t + 2, J1 = J - 1;
        mbar_wait(smem_u32(&bars->q_full), item_cnt & 1);
        tc_fence_after();
        {
          uint32_t s0 = kv_cnt & 1, ph0 = (kv_cnt >> 1) & 1;
          mbar_wait(smem_u32(&bars->k_full[s0]), ph0);
          tc_fence_after();
          issue_qk(tmem + S_COL0 + 0,   sQ + 0,     sK + s0 * 32768, idesc_qk); tc_commit(smem_u32(&bars->s_full[0]));
          issue_qk(tmem + S_COL0 + 128, sQ + 32768, sK + s0 * 32768, idesc_qk); tc_commit(smem_u32(&bars->s_full[1]));
          tc_commit(smem_u32(&bars->k_empty[s0]));
        }
        for (int j = 0; j < J; j++) {
          uint32_t kvj = kv_cnt + j, sj = kvj & 1, phj = (kvj >> 1) & 1;
          mbar_wait(smem_u32(&bars->v_full[sj]), phj);
          tc_fence_after();
          if (j < J1) {
            mbar_wait(smem_u32(&bars->p_full[0]), p_cnt0 & 1); p_cnt0++;
            tc_fence_after();
            issue_pv(tmem + O_COL0 + 0, tmem + S_COL0 + 0, sV + sj * 32768, idesc_pv, j > 0);
            tc_commit(smem_u32(&bars->pv_done[0]));
          }
          uint32_t sn = 0;
          if (j + 1 < J) {
            uint32_t kvn = kvj + 1; sn = kvn & 1; uint32_t phn = (kvn >> 1) & 1;
            mbar_wait(smem_u32(&bars->k_full[sn]), phn);
            tc_fence_after();
            if (j + 1 < J1) { issue_qk(tmem + S_COL0 + 0, sQ + 0, sK + sn * 32768, idesc_qk); tc_commit(smem_u32(&bars->s_full[0])); }
          }
          mbar_wait(smem_u32(&bars->p_full[1]), p_cnt1 & 1); p_cnt1++;
          tc_fence_after();
          issue_pv(tmem + O_COL0 + 128, tmem + S_COL0 + 128, sV + sj * 32768, idesc_pv, j > 0);
          tc_commit(smem_u32(&bars->pv_done[1]));
          tc_commit(smem_u32(&bars->v_empty[sj]));
          if (j + 1 < J) {
            issue_qk(tmem + S_COL0 + 128, sQ + 32768, sK + sn * 32768, idesc_qk); tc_commit(smem_u32(&bars->s_full[1]));
            tc_commit(smem_u32(&bars->k_empty[sn]));
            if (j + 1 == J - 1) tc_commit(smem_u32(&bars->q_empty));
          }
        }
        kv_cnt += J; item_cnt++;
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
    // ================= softmax warpgroups =================
    const int wg = (warp >> 2) - 1, wq = warp & 3, row_in_wg = wq * 32 + lane;
    const uint32_t taddr_lane = tmem + ((uint32_t)(wq * 32) << 16);
    const uint32_t S_COL = S_COL0 + wg * 128, P_COL = wg * 128, O_COL = O_COL0 + wg * 128;
    const uint32_t bar_s = smem_u32(&bars->s_full[wg]), bar_p = smem_u32(&bars->p_full[wg]), bar_pv = smem_u32(&bars->pv_done[wg]);
    const float scale_log2 = p.scale_log2;
    uint32_t item_cnt = 0, s_cnt = 0, pv_cnt = 0;
    while (true) {
      uint32_t slot = item_cnt & 1, ph = (item_cnt >> 1) & 1;
      mbar_wait(smem_u32(&bars->work_full[slot]), ph);
      int item = *(volatile int*)&bars->work_item[slot];
      if (row_in_wg == 0) mbar_arrive(smem_u32(&bars->work_empty[slot]));
      if (item < 0) break;
      int b, h, t; decode_item(p, item, b, h, t);
      const int J = 2 * t + 2, Jw = (wg == 0) ? J - 1 : J, q0 = t * BM;
      float m_used = -INFINITY, l = 0.f;
      for (int j = 0; j < Jw; j++) {
        mbar_wait(bar_s, s_cnt & 1); s_cnt++;
        tc_fence_after();
        uint32_t s[128];
        tmem_ld_x128(taddr_lane + S_COL, s);
        tmem_ld_wait();
        float mt;
        {
          float m[64];
          if (j == Jw - 1) {
            #pragma unroll
            for (int c = 0; c < 128; c++) if (c > row_in_wg) s[c] = __float_as_uint(-INFINITY);
          }
          #pragma unroll
          for (int c = 0; c < 64; c++) m[c] = fmaxf(__uint_as_float(s[2*c]), __uint_as_float(s[2*c+1]));
          #pragma unroll
          for (int w = 32; w >= 1; w >>= 1) {
            #pragma unroll
            for (int c = 0; c < w; c++) m[c] = fmaxf(m[c], m[c + w]);
          }
          mt = m[0] * scale_log2;
        }
        bool need = mt > m_used + THR;
        if (__any_sync(0xffffffffu, need)) {
          float m_new = need ? mt : m_used;
          float alpha = ex2_approx(m_used - m_new);
          l *= alpha;
          if (j > 0) {
            mbar_wait(bar_pv, (pv_cnt - 1) & 1);
            tc_fence_after();
            #pragma unroll
            for (int ch = 0; ch < 4; ch++) {
              uint32_t o[32];
              tmem_ld_x32(taddr_lane + O_COL + ch * 32, o);
              tmem_ld_wait();
              #pragma unroll
              for (int i = 0; i < 32; i++) o[i] = __float_as_uint(__uint_as_float(o[i]) * alpha);
              tmem_st_x32(taddr_lane + O_COL + ch * 32, o);
            }
            tmem_st_wait();
          }
          m_used = m_new;
        }
        const float neg_m = -m_used;
        uint32_t pk[64];
        float ls[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) ls[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < 64; c++) {
          float x0 = fmaf(__uint_as_float(s[2*c]),   scale_log2, neg_m);
          float x1 = fmaf(__uint_as_float(s[2*c+1]), scale_log2, neg_m);
          float p0 = ex2_approx(x0), p1 = ex2_approx(x1);
          ls[c & 7] += p0 + p1;
          pk[c] = pack_bf16x2(p0, p1);
        }
        #pragma unroll
        for (int w = 4; w >= 1; w >>= 1) {
          #pragma unroll
          for (int i = 0; i < w; i++) ls[i] += ls[i + w];
        }
        l += ls[0];
        tmem_st_x64(taddr_lane + P_COL, pk);
        tmem_st_wait();
        tc_fence_before();
        if (lane == 0) mbar_arrive(bar_p);
        pv_cnt++;
      }
      // ---- epilogue ----
      mbar_wait(bar_pv, (pv_cnt - 1) & 1);
      tc_fence_after();
      const float inv_l = 1.f / l;
      const int q_idx = q0 + wg * 128 + row_in_wg;
      uint4* orow = (uint4*)(p.O + ((size_t)(b * p.S + q_idx) * p.H + h) * D_HEAD);
      #pragma unroll
      for (int ch = 0; ch < 4; ch++) {
        uint32_t o[32];
        tmem_ld_x32(taddr_lane + O_COL + ch * 32, o);
        tmem_ld_wait();
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 v;
          v.x = pack_bf16x2(__uint_as_float(o[8*i+0]) * inv_l, __uint_as_float(o[8*i+1]) * inv_l);
          v.y = pack_bf16x2(__uint_as_float(o[8*i+2]) * inv_l, __uint_as_float(o[8*i+3]) * inv_l);
          v.z = pack_bf16x2(__uint_as_float(o[8*i+4]) * inv_l, __uint_as_float(o[8*i+5]) * inv_l);
          v.w = pack_bf16x2(__uint_as_float(o[8*i+6]) * inv_l, __uint_as_float(o[8*i+7]) * inv_l);
          orow[ch * 4 + i] = v;
        }
      }
      item_cnt++;
    }
  }
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
  if (tid == 0) {
    __threadfence();
    int done = atomicAdd(&p.ws[1], 1);
    if (done == (int)gridDim.x - 1) { p.ws[0] = 0; p.ws[1] = 0; __threadfence(); }
  }
}

extern "C" int fa_fwd(const void* q, const void* k, const void* v, void* o, int* ws,
                      int B, int S, int H, int G, int grid, cudaStream_t stream) {
  static bool inited = false; static int num_sms = 0;
  if (!inited) {
    cudaFuncSetAttribute(fa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES + 1024);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    inited = true;
  }
  CUtensorMap tq = make_tmap_bshd(q, B, S, H, D_HEAD), tk = make_tmap_bshd(k, B, S, H, D_HEAD), tv = make_tmap_bshd(v, B, S, H, D_HEAD);
  Params p;
  p.O = (__nv_bfloat16*)o; p.ws = ws; p.NT = S / BM; p.G = G; p.items_per_chunk = G * p.NT;
  p.total_items = B * H * p.NT; p.B = B; p.S = S; p.H = H;
  p.scale_log2 = (1.0f / sqrtf((float)D_HEAD)) * 1.4426950408889634f;
  if (grid <= 0) grid = num_sms;
  fa_fwd_kernel<<<grid, 384, SMEM_BYTES + 1024, stream>>>(tq, tk, tv, p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "launch error: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
