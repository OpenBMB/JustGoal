// MLA paged decode (DeepSeek-V3 TP8 shape): H=16, ckv=512, kpe=64, page_size=1, bf16 in, bf16 out, lse(base2) f32.
// Written from scratch. v5: cp.async gather pipeline continuous across work items, L2 bulk prefetch ahead,
// 8 warps, Q held in registers (K-split across 4 warps), partial-S exchange through smem,
// dynamic work queue (full chunks first, tails last), split-KV + combine kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef TILE
#define TILE 64          // tokens per pipeline stage
#endif
#ifndef STAGES
#define STAGES 2
#endif
#ifndef CHUNK
#define CHUNK 256        // tokens per work item
#endif
#ifndef PD
#define PD 3             // L2 prefetch distance (tiles)
#endif
#define NWARPS 8

namespace {
constexpr int H = 16, D = 512, DP = 64, DQK = 576;
constexpr int ROW_BYTES = DQK * 2;            // 1152
constexpr int ROW_STRIDE = ROW_BYTES + 16;    // 1168: conflict-free ldmatrix
constexpr int NTHREADS = NWARPS * 32;
constexpr int G = NWARPS / 4;                 // token groups for QK
constexpr int TG = TILE / G;                  // tokens per group
constexpr int NT = TG / 8;                    // n-tiles per warp in QK
constexpr int KSL = DQK / 4;                  // 144 dims per K-slice
constexpr int KT_QK = KSL / 16;               // 9 k-tiles per slice
constexpr int STAGE_BYTES = TILE * ROW_STRIDE;
constexpr int Q_BYTES = H * ROW_STRIDE;
constexpr int P_STRIDE = TILE * 2 + 16;
constexpr int P_BYTES = H * P_STRIDE;
constexpr int SS_STRIDE = TG * 4 + 32;        // bytes per row of a partial-S slab
constexpr int SS_SLAB = H * SS_STRIDE;
constexpr int SS_BYTES = G * 4 * SS_SLAB;
constexpr int RED_BYTES = NWARPS * H * 4;
constexpr int MAXB = 512;
constexpr int PREFIX_BYTES = 3 * (MAXB + 1) * 4 + 64;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + 2 * Q_BYTES + P_BYTES + SS_BYTES + RED_BYTES + PREFIX_BYTES;
constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;           // 72
constexpr int DW = D / NWARPS;                           // 64 output dims per warp in PV
constexpr int NT_PV = DW / 8;                            // 8
constexpr int KT_PV = TILE / 16;
constexpr int NSOFT = TILE / 8;                          // warps doing softmax
static_assert(TILE % 16 == 0 && TG % 16 == 0, "tile shape");
static_assert(NSOFT <= NWARPS, "softmax warps");
static_assert(2 * TILE <= NTHREADS, "prefetch threads");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

struct Params {
  const __nv_bfloat16* q_nope;   // [B,16,512]
  const __nv_bfloat16* q_pe;     // [B,16,64]
  const __nv_bfloat16* ckv;      // [P,512]
  const __nv_bfloat16* kpe;      // [P,64]
  const int* kv_indptr;          // [B+1]
  const int* kv_indices;         // [total]
  __nv_bfloat16* out;            // [B,16,512]
  float* lse;                    // [B,16]
  float* part_o;                 // [max_items,16,512]
  float* part_m;                 // [max_items,16]
  float* part_l;                 // [max_items,16]
  int* counter;                  // [1] dynamic work counter (reset by combine kernel)
  int* g_prefix;                 // [B+1] chunk prefix (ceil), written by block 0
  float scale_log2;              // sm_scale * log2(e)
  int B;
};
struct Work { int b, c, n_tok, n_tiles, kv_off, slot, nchunks; bool valid; };

__global__ void __launch_bounds__(NTHREADS, 1) mla_decode_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sK = smem;
  uint8_t* sQ = sK + STAGES * STAGE_BYTES;      // 2 buffers
  uint8_t* sP = sQ + 2 * Q_BYTES;
  uint8_t* sS = sP + P_BYTES;
  float* red = reinterpret_cast<float*>(sS + SS_BYTES);
  int* prefix = reinterpret_cast<int*>(red + NWARPS * H);   // ceil chunk prefix  [B+1]
  int* prefixF = prefix + (MAXB + 1);                         // full-chunk prefix  [B+1]
  int* prefixT = prefixF + (MAXB + 1);                        // tail-chunk prefix  [B+1]
  int* s_next = prefixT + (MAXB + 1);                         // [3] fetched work indices

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int grp = warp >> 2, ks = warp & 3;
  const int B = p.B;

  if (tid == 0) {
    int acc = 0, accF = 0, accT = 0;
    prefix[0] = 0; prefixF[0] = 0; prefixT[0] = 0;
    for (int b = 0; b < B; ++b) {
      int len = p.kv_indptr[b + 1] - p.kv_indptr[b];
      acc += (len + CHUNK - 1) / CHUNK;
      accF += len / CHUNK;
      accT += (len % CHUNK) ? 1 : 0;
      prefix[b + 1] = acc; prefixF[b + 1] = accF; prefixT[b + 1] = accT;
    }
    s_next[0] = atomicAdd(p.counter, 1);
    s_next[1] = atomicAdd(p.counter, 1);
    if (blockIdx.x == 0) {
      for (int b = 0; b <= B; ++b) p.g_prefix[b] = prefix[b];
    }
  }
  __syncthreads();
  const int total_items = prefix[B];
  const int F = prefixF[B];
  const uint32_t sK_u = smem_u32(sK), sQ_u = smem_u32(sQ), sP_u = smem_u32(sP);
  float* sS_f = reinterpret_cast<float*>(sS);

  auto decode = [&](int witem) {
    Work w; w.valid = witem >= 0 && witem < total_items;
    if (!w.valid) { w.b = w.c = w.n_tok = w.n_tiles = w.kv_off = w.slot = w.nchunks = 0; return w; }
    const int* pf; int key;
    if (witem < F) { pf = prefixF; key = witem; } else { pf = prefixT; key = witem - F; }
    int lo = 0, hi = B - 1;
    while (lo < hi) {
      int mid = (lo + hi + 1) >> 1;
      if (pf[mid] <= key) lo = mid; else hi = mid - 1;
    }
    w.b = lo;
    const int len_b = p.kv_indptr[w.b + 1] - p.kv_indptr[w.b];
    if (witem < F) { w.c = key - prefixF[w.b]; w.n_tok = CHUNK; }
    else { w.c = len_b / CHUNK; w.n_tok = len_b - w.c * CHUNK; }
    w.n_tiles = (w.n_tok + TILE - 1) / TILE;
    w.kv_off = p.kv_indptr[w.b] + w.c * CHUNK;
    w.slot = prefix[w.b] + w.c;
    w.nchunks = prefix[w.b + 1] - prefix[w.b];
    return w;
  };
  auto load_q = [&](const Work& w, int qb) {
    const uint32_t dst = sQ_u + qb * Q_BYTES;
    const __nv_bfloat16* qn = p.q_nope + (size_t)w.b * H * D;
    const __nv_bfloat16* qp = p.q_pe + (size_t)w.b * H * DP;
#pragma unroll
    for (int j = tid; j < H * CHUNKS_PER_ROW; j += NTHREADS) {
      int r = j / CHUNKS_PER_ROW, col = j % CHUNKS_PER_ROW;
      const __nv_bfloat16* src = (col < 64) ? (qn + r * D + col * 8) : (qp + r * DP + (col - 64) * 8);
      cp_async16(dst + r * ROW_STRIDE + col * 16, src, 16);
    }
  };
  auto load_tile = [&](const Work& w, int ti, int stage) {
    const uint32_t base = sK_u + stage * STAGE_BYTES;
    const int tok0 = ti * TILE;
    const int* idx = p.kv_indices + w.kv_off;
    constexpr int NL = (TILE * CHUNKS_PER_ROW + NTHREADS - 1) / NTHREADS;
    int pages[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      int jj = tid + k * NTHREADS;
      int r = jj / CHUNKS_PER_ROW;
      int tok = tok0 + r;
      pages[k] = (jj < TILE * CHUNKS_PER_ROW && tok < w.n_tok) ? __ldg(idx + tok) : -1;
    }
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      int jj = tid + k * NTHREADS;
      if (jj < TILE * CHUNKS_PER_ROW) {
        int r = jj / CHUNKS_PER_ROW, col = jj % CHUNKS_PER_ROW;
        bool valid = pages[k] >= 0;
        int page = valid ? pages[k] : 0;
        const __nv_bfloat16* src = (col < 64) ? (p.ckv + (size_t)page * D + col * 8)
                                              : (p.kpe + (size_t)page * DP + (col - 64) * 8);
#ifndef NO_LOAD
        cp_async16(base + r * ROW_STRIDE + col * 16, src, valid ? 16 : 0);
#endif
      }
    }
  };
  auto prefetch_tile = [&](const Work& w, int ti) {
#if PD > 0
    const int tok0 = ti * TILE;
    const int* idx = p.kv_indices + w.kv_off;
#ifdef PF_BULK
    if (tid < TILE) {
      int tok = tok0 + tid;
      if (tok < w.n_tok) prefetch_l2(p.ckv + (size_t)__ldg(idx + tok) * D, D * 2);
    } else if (tid < 2 * TILE) {
      int tok = tok0 + tid - TILE;
      if (tok < w.n_tok) prefetch_l2(p.kpe + (size_t)__ldg(idx + tok) * DP, DP * 2);
    }
#else
    // one prefetch.global.L2 per 128B line: 8 lines ckv + 1 line kpe per token
#pragma unroll
    for (int j = tid; j < TILE * 9; j += NTHREADS) {
      int r = j / 9, li = j % 9;
      int tok = tok0 + r;
      if (tok < w.n_tok) {
        int page = __ldg(idx + tok);
        const void* addr = (li < 8) ? (const void*)(p.ckv + (size_t)page * D + li * 64) : (const void*)(p.kpe + (size_t)page * DP);
        asm volatile("prefetch.global.L2 [%0];\n" ::"l"(addr));
      }
    }
#endif
#endif
  };

  // ---- work queue: ring of 3 fetched indices; items k (cur) and k+1 (nxt) are known
  Work cur = decode(s_next[0]);
  if (!cur.valid) return;
  Work nxt = decode(s_next[1]);
  int k = 0;                         // item sequence number
  uint32_t tload = 0, tcons = 0;     // tiles loaded / consumed (stage = counter % STAGES)
  int qbuf = 0;
  int pf_off = 0, pf_tile = 1;       // prefetch cursor: (item offset 0=cur/1=nxt, tile)
  auto advance_pf = [&]() {
    if (pf_off == 0) {
      if (pf_tile < cur.n_tiles) { prefetch_tile(cur, pf_tile++); return; }
      pf_off = 1; pf_tile = 0;
    }
    if (nxt.valid && pf_tile < nxt.n_tiles) { prefetch_tile(nxt, pf_tile++); }
  };
  load_q(cur, qbuf);
  load_tile(cur, 0, tload % STAGES); ++tload;
  cp_commit();
  for (int i = 0; i < PD; ++i) advance_pf();

  for (;;) {
    uint32_t qf[KT_QK][4];
    float o[NT_PV][4];
#pragma unroll
    for (int i = 0; i < NT_PV; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    const int n_tok = cur.n_tok;
    const bool soft = warp < NSOFT;
    const int sgrp = warp / NT, snl = warp % NT;

    for (int ti = 0; ti < cur.n_tiles; ++ti) {
      cp_wait<0>();
      __syncthreads();   // tile ti ready (and Q if ti==0); previous stage free
      if (ti == 0 && tid == 0) s_next[(k + 2) % 3] = atomicAdd(p.counter, 1);
      const int stage = tcons % STAGES; ++tcons;
      // loader: next tile of cur, or first tile (+Q) of nxt
      if (ti + 1 < cur.n_tiles) {
        load_tile(cur, ti + 1, tload % STAGES); ++tload;
      } else if (nxt.valid) {
        load_q(nxt, qbuf ^ 1);
        load_tile(nxt, 0, tload % STAGES); ++tload;
      }
      cp_commit();
      advance_pf();
#ifdef NO_COMPUTE
      continue;
#endif
      const uint32_t kbase = sK_u + stage * STAGE_BYTES;
      if (ti == 0) {
        const uint32_t qb = sQ_u + qbuf * Q_BYTES;
#pragma unroll
        for (int kt = 0; kt < KT_QK; ++kt)
          ldsm_x4(qf[kt], qb + (lane & 15) * ROW_STRIDE + (ks * KSL + kt * 16 + (lane >> 4) * 8) * 2);
      }
      // ---- partial S over this warp's K-slice for its group's TG tokens
      float s[NT][4];
#pragma unroll
      for (int n = 0; n < NT; ++n) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
      {
        const uint32_t krow = kbase + (grp * TG + (lane & 7) + (lane >> 4) * 8) * ROW_STRIDE + (ks * KSL + ((lane >> 3) & 1) * 8) * 2;
        uint32_t kf[2][4];
        ldsm_x4(kf[0], krow);
#pragma unroll
        for (int i = 0; i < NT / 2 * KT_QK; ++i) {
          const int np = i / KT_QK, kt = i % KT_QK;
          if (i + 1 < NT / 2 * KT_QK) {
            const int np1 = (i + 1) / KT_QK, kt1 = (i + 1) % KT_QK;
            ldsm_x4(kf[(i + 1) & 1], krow + np1 * 16 * ROW_STRIDE + kt1 * 32);
          }
#ifndef NO_MMA
          mma_bf16(s[2 * np], qf[kt], kf[i & 1]);
          mma_bf16(s[2 * np + 1], qf[kt], kf[i & 1] + 2);
#endif
        }
      }
      {
        float* slab = sS_f + ((grp * 4 + ks) * SS_SLAB) / 4;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          int col = n * 8 + 2 * t;
          *reinterpret_cast<float2*>(slab + (g * SS_STRIDE) / 4 + col) = make_float2(s[n][0], s[n][1]);
          *reinterpret_cast<float2*>(slab + ((g + 8) * SS_STRIDE) / 4 + col) = make_float2(s[n][2], s[n][3]);
        }
      }
      __syncthreads();
      // ---- softmax warps: sum 4 partials, mask, scale, row max
      float sv[4] = {0.f, 0.f, 0.f, 0.f};
      float mx0 = -1e30f, mx1 = -1e30f;
      if (soft) {
        const int col = snl * 8 + 2 * t;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const float* slab = sS_f + ((sgrp * 4 + kk) * SS_SLAB) / 4;
          float2 v0 = *reinterpret_cast<const float2*>(slab + (g * SS_STRIDE) / 4 + col);
          float2 v1 = *reinterpret_cast<const float2*>(slab + ((g + 8) * SS_STRIDE) / 4 + col);
          sv[0] += v0.x; sv[1] += v0.y; sv[2] += v1.x; sv[3] += v1.y;
        }
        const int tk = ti * TILE + warp * 8 + 2 * t;
        sv[0] = (tk < n_tok) ? sv[0] * p.scale_log2 : -1e30f;
        sv[1] = (tk + 1 < n_tok) ? sv[1] * p.scale_log2 : -1e30f;
        sv[2] = (tk < n_tok) ? sv[2] * p.scale_log2 : -1e30f;
        sv[3] = (tk + 1 < n_tok) ? sv[3] * p.scale_log2 : -1e30f;
        mx0 = fmaxf(sv[0], sv[1]);
        mx1 = fmaxf(sv[2], sv[3]);
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
      }
      if (t == 0) { red[warp * H + g] = mx0; red[warp * H + g + 8] = mx1; }
      __syncthreads();
#pragma unroll
      for (int w = 0; w < NSOFT; ++w) {
        mx0 = fmaxf(mx0, red[w * H + g]);
        mx1 = fmaxf(mx1, red[w * H + g + 8]);
      }
      const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
      const float alpha0 = exp2f(m0 - mn0), alpha1 = exp2f(m1 - mn1);
      m0 = mn0; m1 = mn1;
      l0 *= alpha0; l1 *= alpha1;
      if (soft) {
        float p0 = exp2f(sv[0] - mn0), p1 = exp2f(sv[1] - mn0);
        float p2 = exp2f(sv[2] - mn1), p3 = exp2f(sv[3] - mn1);
        l0 += p0 + p1; l1 += p2 + p3;
        const int col = warp * 8 + 2 * t;
        *reinterpret_cast<uint32_t*>(sP + g * P_STRIDE + col * 2) = pack_bf16(p0, p1);
        *reinterpret_cast<uint32_t*>(sP + (g + 8) * P_STRIDE + col * 2) = pack_bf16(p2, p3);
      }
#pragma unroll
      for (int i = 0; i < NT_PV; ++i) { o[i][0] *= alpha0; o[i][1] *= alpha0; o[i][2] *= alpha1; o[i][3] *= alpha1; }
      __syncthreads();
      // ---- O += P V (this warp's 64 dims)
      {
        const uint32_t vrow = kbase + (lane & 15) * ROW_STRIDE + (warp * DW + (lane >> 4) * 8) * 2;
        uint32_t vf[2][4];
        ldsm_x4_t(vf[0], vrow);
#pragma unroll
        for (int kt = 0; kt < KT_PV; ++kt) {
          uint32_t a[4];
          ldsm_x4(a, sP_u + (lane & 15) * P_STRIDE + (kt * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
          for (int np = 0; np < NT_PV / 2; ++np) {
            const int i = kt * (NT_PV / 2) + np;
            if (i + 1 < KT_PV * (NT_PV / 2)) {
              const int kt1 = (i + 1) / (NT_PV / 2), np1 = (i + 1) % (NT_PV / 2);
              ldsm_x4_t(vf[(i + 1) & 1], vrow + kt1 * 16 * ROW_STRIDE + np1 * 32);
            }
#ifndef NO_MMA
            mma_bf16(o[2 * np], a, vf[i & 1]);
            mma_bf16(o[2 * np + 1], a, vf[i & 1] + 2);
#endif
          }
        }
      }
    }
    // ---- epilogue for cur
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    __syncthreads();
    if (t == 0) { red[warp * H + g] = l0; red[warp * H + g + 8] = l1; }
    __syncthreads();
    float L0 = 0.f, L1 = 0.f;
#pragma unroll
    for (int w = 0; w < NSOFT; ++w) { L0 += red[w * H + g]; L1 += red[w * H + g + 8]; }

    if (cur.nchunks == 1) {
      __nv_bfloat16* ob = p.out + ((size_t)cur.b * H) * D;
      const float inv0 = 1.f / L0, inv1 = 1.f / L1;
#pragma unroll
      for (int i = 0; i < NT_PV; ++i) {
        int col = warp * DW + i * 8 + 2 * t;
        *reinterpret_cast<uint32_t*>(ob + g * D + col) = pack_bf16(o[i][0] * inv0, o[i][1] * inv0);
        *reinterpret_cast<uint32_t*>(ob + (g + 8) * D + col) = pack_bf16(o[i][2] * inv1, o[i][3] * inv1);
      }
      if (warp == 0 && t == 0) {
        p.lse[cur.b * H + g] = m0 + log2f(L0);
        p.lse[cur.b * H + g + 8] = m1 + log2f(L1);
      }
    } else {
      float* po = p.part_o + (size_t)cur.slot * H * D;
#pragma unroll
      for (int i = 0; i < NT_PV; ++i) {
        int col = warp * DW + i * 8 + 2 * t;
        *reinterpret_cast<float2*>(po + g * D + col) = make_float2(o[i][0], o[i][1]);
        *reinterpret_cast<float2*>(po + (g + 8) * D + col) = make_float2(o[i][2], o[i][3]);
      }
      if (warp == 0 && t == 0) {
        p.part_m[cur.slot * H + g] = m0; p.part_m[cur.slot * H + g + 8] = m1;
        p.part_l[cur.slot * H + g] = L0; p.part_l[cur.slot * H + g + 8] = L1;
      }
    }
    __syncthreads();   // red reuse + s_next visibility
    // ---- rotate to next item
    ++k;
    cur = nxt;
    if (!cur.valid) break;
    qbuf ^= 1;
    nxt = decode(s_next[(k + 1) % 3]);
    pf_off = 0;   // cursor was in nxt (now cur); keep pf_tile
  }
  cp_wait<0>();
}

// combine: one block per (b, h); CG chunk-groups x 64 threads x 8 dims
constexpr int CG = 4;
__global__ void __launch_bounds__(CG * 64) mla_combine_kernel(Params p) {
  const int b = blockIdx.x / H, h = blockIdx.x % H;
  __shared__ float s_m[256];
  __shared__ float s_red[CG][64][10];
  const int tid = threadIdx.x;
  if (blockIdx.x == 0 && tid == 0) *p.counter = 0;   // reset work counter for the next launch
  const int base = p.g_prefix[b];
  const int n = p.g_prefix[b + 1] - base;
  if (n <= 1) return;
  for (int c = tid; c < n; c += blockDim.x) s_m[c] = p.part_m[(base + c) * H + h];
  __syncthreads();
  float M = -1e30f;
  for (int c = 0; c < n; ++c) M = fmaxf(M, s_m[c]);
  const int grp = tid >> 6, dt = tid & 63;
  const int d0 = dt * 8;
  float L = 0.f;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll 4
  for (int c = grp; c < n; c += CG) {
    float w = exp2f(s_m[c] - M);
    L += w * p.part_l[(base + c) * H + h];
    const float4* src = reinterpret_cast<const float4*>(p.part_o + ((size_t)(base + c) * H + h) * D + d0);
    float4 v0 = src[0], v1 = src[1];
    acc[0] += w * v0.x; acc[1] += w * v0.y; acc[2] += w * v0.z; acc[3] += w * v0.w;
    acc[4] += w * v1.x; acc[5] += w * v1.y; acc[6] += w * v1.z; acc[7] += w * v1.w;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) s_red[grp][dt][j] = acc[j];
  s_red[grp][dt][8] = L;
  __syncthreads();
  if (grp == 0) {
#pragma unroll
    for (int gg = 1; gg < CG; ++gg) {
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += s_red[gg][dt][j];
      L += s_red[gg][dt][8];
    }
    const float inv = 1.f / L;
    __nv_bfloat16* ob = p.out + ((size_t)b * H + h) * D + d0;
    uint4 packed = make_uint4(pack_bf16(acc[0] * inv, acc[1] * inv), pack_bf16(acc[2] * inv, acc[3] * inv),
                              pack_bf16(acc[4] * inv, acc[5] * inv), pack_bf16(acc[6] * inv, acc[7] * inv));
    *reinterpret_cast<uint4*>(ob) = packed;
    if (dt == 0) p.lse[b * H + h] = M + log2f(L);
  }
}
}  // namespace

extern "C" int mla_decode_run(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe,
                              const int* kv_indptr, const int* kv_indices, void* out, float* lse,
                              float* part_o, float* part_m, float* part_l, int* counter, int* g_prefix,
                              float sm_scale, int B, int grid, cudaStream_t stream) {
  static bool inited = false;
  static int max_blocks = 0;
  if (!inited) {
    cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    int dev; cudaGetDevice(&dev);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mla_decode_kernel, NTHREADS, SMEM_BYTES);
    max_blocks = nsm * occ;
    inited = true;
    fprintf(stderr, "[mla] TILE=%d STAGES=%d NWARPS=%d CHUNK=%d PD=%d smem=%d occ=%d grid=%d\n", TILE, STAGES, NWARPS,
            CHUNK, PD, SMEM_BYTES, occ, max_blocks);
  }
  Params p;
  p.q_nope = (const __nv_bfloat16*)q_nope; p.q_pe = (const __nv_bfloat16*)q_pe;
  p.ckv = (const __nv_bfloat16*)ckv; p.kpe = (const __nv_bfloat16*)kpe;
  p.kv_indptr = kv_indptr; p.kv_indices = kv_indices;
  p.out = (__nv_bfloat16*)out; p.lse = lse;
  p.part_o = part_o; p.part_m = part_m; p.part_l = part_l; p.counter = counter; p.g_prefix = g_prefix;
  p.scale_log2 = sm_scale * 1.4426950408889634f; p.B = B;
  if (grid <= 0) grid = max_blocks;
  mla_decode_kernel<<<grid, NTHREADS, SMEM_BYTES, stream>>>(p);
  mla_combine_kernel<<<B * H, CG * 64, 0, stream>>>(p);
  return (int)cudaGetLastError();
}
