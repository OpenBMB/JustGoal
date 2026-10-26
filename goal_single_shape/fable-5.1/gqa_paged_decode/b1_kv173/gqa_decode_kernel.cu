// GQA paged decode (page_size = 1) for Blackwell (sm_100a).
// Written from scratch. One launch. Grid: (SPLITS * num_kv_heads, batch).
// A thread-block cluster of SPLITS blocks handles one (batch, kv_head):
//   - each block computes partial softmax stats + unnormalised P·V for a
//     contiguous chunk of the kv sequence (4 q heads of the group at once),
//   - partials are merged across the cluster through distributed shared memory.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cmath>

namespace cg = cooperative_groups;

#ifndef GQA_SPLITS
#define GQA_SPLITS 8
#endif
#ifndef GQA_THREADS
#define GQA_THREADS 128
#endif
#ifndef GQA_ROUNDS
#define GQA_ROUNDS 2
#endif

namespace gqa {

constexpr int D = 128;           // head dim
constexpr int G = 4;             // q heads per kv head (GQA group)
constexpr int LANES_PER_KEY = 8; // 8 lanes x 16 dims = 128
constexpr int DIMS_PER_LANE = D / LANES_PER_KEY; // 16

template <int SPLITS, int NTHREADS, int ROUNDS>
struct Cfg {
  static constexpr int KEYS_PER_ROUND = NTHREADS / LANES_PER_KEY;
  static constexpr int KEYS_MAX = KEYS_PER_ROUND * ROUNDS;   // keys per super-round per block
  static constexpr int NWARPS = NTHREADS / 32;
};

__device__ __forceinline__ float bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <int SPLITS, int NTHREADS, int ROUNDS>
__global__ void __launch_bounds__(NTHREADS, 1)
gqa_paged_decode_kernel(const __nv_bfloat16* __restrict__ q,        // (B, HQ, D)
                        const __nv_bfloat16* __restrict__ k_cache,  // (P, 1, HKV, D)
                        const __nv_bfloat16* __restrict__ v_cache,  // (P, 1, HKV, D)
                        const int32_t* __restrict__ kv_indptr,      // (B+1)
                        const int32_t* __restrict__ kv_indices,     // (nnz)
                        __nv_bfloat16* __restrict__ out,            // (B, HQ, D)
                        float* __restrict__ lse,                    // (B, HQ)
                        int num_kv_heads, int nnz, float scale_log2, // nnz = kv_indices.numel(), scale = sm_scale*log2(e)
                        unsigned long long* __restrict__ tbuf)      // optional phase timestamps
{
#ifdef GQA_TIMING
  unsigned long long ts[10];
  int nts = 0;
#define TSTAMP() do { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t) :: "memory"); ts[nts++] = _t; } while (0)
#else
#define TSTAMP() do {} while (0)
#endif
  TSTAMP();
  using C = Cfg<SPLITS, NTHREADS, ROUNDS>;
  constexpr int KPR = C::KEYS_PER_ROUND;
  constexpr int KMAX = C::KEYS_MAX;
  constexpr int NW = C::NWARPS;
  static_assert(NW >= G, "need at least 4 warps");
  static_assert(KMAX <= 128, "softmax lane mapping supports <= 128 keys per super-round");

  __shared__ __align__(16) __nv_bfloat16 v_s[KMAX][D];   // V rows of this super-round
  __shared__ float s_s[G][KMAX];                          // scores (log2 domain)
  __shared__ __align__(16) float p_s[KMAX][G];            // probabilities, one float4 per key
  __shared__ float m_s[G];                                // running max (log2 domain)
  __shared__ float l_s[G];                                // running sum
  __shared__ float alpha_s[G];                            // rescale factor for the current super-round
  // partial slots: only rank 0 of the cluster uses them (other blocks push into rank 0 via DSMEM)
  __shared__ __align__(16) float part_acc[SPLITS][G][D];  // unnormalised P.V per split
  __shared__ float part_m[SPLITS][G], part_l[SPLITS][G];  // per-split max / sum (log2 domain)
  __shared__ float scale_s[SPLITS][G], Lfin_s[G], Mfin_s[G];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int slot = tid / LANES_PER_KEY;          // key slot within a round
  const int sub  = tid % LANES_PER_KEY;          // 16-dim slice index
  const int b = blockIdx.y;
  const int kvh = blockIdx.x / SPLITS;
  const int split = blockIdx.x % SPLITS;

  const int HQ = num_kv_heads * G;
  // Interleaved key assignment: this block owns keys j = split + SPLITS*i. Their positions in
  // kv_indices do not depend on kv_len, so the page-id loads are issued speculatively (assuming
  // kv_indptr[b] == 0, bounded by nnz) in parallel with the kv_indptr load; a fallback reload
  // handles kv_indptr[b] != 0 (batch > 1).
  const int kv_begin = __ldg(kv_indptr + b);
  const int kv_end = __ldg(kv_indptr + b + 1);

  // ---- Q slice: 4 heads x 16 dims, pre-scaled, fp32 in registers (independent of indices) ----
  float qreg[G][DIMS_PER_LANE];
  {
    const __nv_bfloat16* qp = q + ((size_t)b * HQ + (size_t)kvh * G) * D + sub * DIMS_PER_LANE;
#pragma unroll
    for (int h = 0; h < G; ++h) {
      uint4 a = ldg_nc(qp + h * D);
      uint4 c = ldg_nc(qp + h * D + 8);
      uint32_t w[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qreg[h][2 * i] = bf16lo(w[i]) * scale_log2;
        qreg[h][2 * i + 1] = bf16hi(w[i]) * scale_log2;
      }
    }
  }

  if (tid < G) { m_s[tid] = -INFINITY; l_s[tid] = 0.f; }
  // PV accumulators: thread owns dim pair dp for 2 heads (h0, h0+1): 4 heads x 64 pairs = 128 tasks
  static_assert(NTHREADS == 128, "PV / merge mapping assumes 128 threads");
  constexpr int DPAIRS = D / 2;                              // 64
  const int dp = tid % DPAIRS;
  const int h0 = (tid / DPAIRS) * 2;
  float acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i) { acc[i][0] = acc[i][1] = 0.f; }

  const size_t row_stride = (size_t)num_kv_heads * D;      // elements per page
  const __nv_bfloat16* kbase = k_cache + (size_t)kvh * D + sub * DIMS_PER_LANE;
  const __nv_bfloat16* vbase = v_cache + (size_t)kvh * D + sub * DIMS_PER_LANE;

  // speculative page ids for the first super-round (issued before kv_indptr is known)
  int spec_page[ROUNDS];
#pragma unroll
  for (int r = 0; r < ROUNDS; ++r) {
    const int j = split + SPLITS * (r * KPR + slot);
    spec_page[r] = (j < nnz) ? __ldg(kv_indices + j) : 0;
  }
  const int kv_len = kv_end - kv_begin;
  const int my_len = (kv_len > split) ? (kv_len - split + SPLITS - 1) / SPLITS : 0;
  TSTAMP();

  for (int base = 0; base < my_len; base += KMAX) {
    // ---- gather page ids, issue all K/V loads for this super-round ----
    uint4 kr[ROUNDS][2], vr[ROUNDS][2];
    bool valid[ROUNDS];
#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
      const int i = base + r * KPR + slot;          // local key index
      const int j = split + SPLITS * i;             // key index within the sequence
      valid[r] = i < my_len;
      int page = 0;
      if (base == 0 && kv_begin == 0) page = spec_page[r];
      else if (valid[r]) page = __ldg(kv_indices + kv_begin + j);
      const size_t off = (size_t)page * row_stride;
      if (valid[r]) {
        kr[r][0] = ldg_nc(kbase + off);
        kr[r][1] = ldg_nc(kbase + off + 8);
        vr[r][0] = ldg_nc(vbase + off);
        vr[r][1] = ldg_nc(vbase + off + 8);
      } else {
        kr[r][0] = kr[r][1] = vr[r][0] = vr[r][1] = make_uint4(0, 0, 0, 0);
      }
    }
    if (base > 0) __syncthreads();  // previous super-round finished reading v_s / p_s
#ifdef GQA_TIMING
    if (valid[0]) { asm volatile("" :: "r"(kr[0][0].x), "r"(vr[0][0].x) : "memory"); }
    TSTAMP();  // K/V of round 0 in registers
#endif

    // ---- scores: partial dot over 16 dims, reduce across the 8 lanes of a key ----
#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
      uint32_t w[8] = {kr[r][0].x, kr[r][0].y, kr[r][0].z, kr[r][0].w,
                       kr[r][1].x, kr[r][1].y, kr[r][1].z, kr[r][1].w};
      float s[G];
#pragma unroll
      for (int h = 0; h < G; ++h) {
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          a = fmaf(qreg[h][2 * i], bf16lo(w[i]), a);
          a = fmaf(qreg[h][2 * i + 1], bf16hi(w[i]), a);
        }
        s[h] = a;
      }
#pragma unroll
      for (int h = 0; h < G; ++h) {
        s[h] += __shfl_xor_sync(0xffffffffu, s[h], 1);
        s[h] += __shfl_xor_sync(0xffffffffu, s[h], 2);
        s[h] += __shfl_xor_sync(0xffffffffu, s[h], 4);
      }
      const int jj = r * KPR + slot;  // local key index in super-round
      if (sub == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) s_s[h][jj] = valid[r] ? s[h] : -INFINITY;
      }
      // stash V row slice into smem (row jj, dims sub*16 .. +16)
      *reinterpret_cast<uint4*>(&v_s[jj][sub * DIMS_PER_LANE]) = vr[r][0];
      *reinterpret_cast<uint4*>(&v_s[jj][sub * DIMS_PER_LANE + 8]) = vr[r][1];
    }
    __syncthreads();

    TSTAMP();  // scores done
    // ---- softmax over this super-round: warp h handles head h ----
    if (warp < G) {
      const int h = warp;
      float sv[(KMAX + 31) / 32];
      float mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < (KMAX + 31) / 32; ++i) {
        const int j = i * 32 + lane;
        sv[i] = (j < KMAX) ? s_s[h][j] : -INFINITY;
        mx = fmaxf(mx, sv[i]);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      const float m_old = m_s[h];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < (KMAX + 31) / 32; ++i) {
        const int j = i * 32 + lane;
        const float p = (j < KMAX) ? exp2f(sv[i] - m_new) : 0.f;   // -inf -> 0
        if (j < KMAX) p_s[j][h] = p;
        sum += p;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
      if (lane == 0) {
        const float alpha = (m_old == -INFINITY) ? 0.f : exp2f(m_old - m_new);
        alpha_s[h] = alpha;
        l_s[h] = l_s[h] * alpha + sum;
        m_s[h] = m_new;
      }
    }
    __syncthreads();

    TSTAMP();  // softmax done
    // ---- P·V for this thread's (2 heads, dim pair) ----
    {
      const int nkeys = min(KMAX, my_len - base);
      const float a0 = alpha_s[h0], a1 = alpha_s[h0 + 1];
      acc[0][0] *= a0; acc[0][1] *= a0; acc[1][0] *= a1; acc[1][1] *= a1;
      const uint32_t* vrow = reinterpret_cast<const uint32_t*>(&v_s[0][0]) + dp;
      const float2* prow = reinterpret_cast<const float2*>(&p_s[0][0]) + (h0 >> 1);
      for (int j = 0; j < nkeys; ++j) {
        const uint32_t vv = vrow[j * (D / 2)];
        const float2 p = prow[j * (G / 2)];
        const float v0 = bf16lo(vv), v1 = bf16hi(vv);
        acc[0][0] = fmaf(p.x, v0, acc[0][0]); acc[0][1] = fmaf(p.x, v1, acc[0][1]);
        acc[1][0] = fmaf(p.y, v0, acc[1][0]); acc[1][1] = fmaf(p.y, v1, acc[1][1]);
      }
    }
  }

  TSTAMP();  // PV done
  const float LN2 = 0.6931471805599453f;
  if constexpr (SPLITS == 1) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
      *reinterpret_cast<float2*>(&part_acc[0][h0 + i][2 * dp]) = make_float2(acc[i][0], acc[i][1]);
    __syncthreads();
    for (int o = tid; o < G * D; o += NTHREADS) {
      const int h = o / D, d = o % D;
      const float L = l_s[h];
      out[((size_t)b * HQ + kvh * G + h) * D + d] = __float2bfloat16((L > 0.f) ? part_acc[0][h][d] / L : 0.f);
    }
    if (tid < G) {
      const float M = m_s[tid], L = l_s[tid];
      lse[(size_t)b * HQ + kvh * G + tid] = (L > 0.f) ? (M + log2f(L)) * LN2 : -INFINITY;
    }
  } else {
    cg::cluster_group cluster = cg::this_cluster();
    // ---- push this block's partial into rank 0's shared memory ----
    float* dst_acc = cluster.map_shared_rank(&part_acc[0][0][0], 0) + (size_t)split * (G * D);
#pragma unroll
    for (int i = 0; i < 2; ++i)
      *reinterpret_cast<float2*>(dst_acc + (h0 + i) * D + 2 * dp) = make_float2(acc[i][0], acc[i][1]);
    if (tid < G) {
      cluster.map_shared_rank(&part_m[0][0], 0)[split * G + tid] = m_s[tid];
      cluster.map_shared_rank(&part_l[0][0], 0)[split * G + tid] = l_s[tid];
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    if (split != 0) {         // non-merging blocks are done (exited threads count as arrived)
#ifdef GQA_TIMING
      TSTAMP();  // after arrive
      if (tbuf != nullptr && tid == 0) {
        for (int i = 0; i < nts; ++i) tbuf[blockIdx.x * 16 + i] = ts[i];
        tbuf[blockIdx.x * 16 + 15] = nts;
      }
#endif
      return;
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    TSTAMP();  // all partials landed
    // ---- merge: per (split, head) scale factors ----
    if (tid < SPLITS * G) {
      const int s = tid / G, h = tid % G;
      float M = -INFINITY;
#pragma unroll
      for (int t = 0; t < SPLITS; ++t) M = fmaxf(M, part_m[t][h]);
      const float mv = part_m[s][h];
      const float sc = (mv == -INFINITY) ? 0.f : exp2f(mv - M);
      scale_s[s][h] = sc;
      if (s == 0) {
        float L = 0.f;
#pragma unroll
        for (int t = 0; t < SPLITS; ++t) {
          const float mt = part_m[t][h];
          L = fmaf(part_l[t][h], (mt == -INFINITY) ? 0.f : exp2f(mt - M), L);
        }
        Lfin_s[h] = L; Mfin_s[h] = M;
        lse[(size_t)b * HQ + kvh * G + h] = (L > 0.f) ? (M + log2f(L)) * LN2 : -INFINITY;
      }
    }
    __syncthreads();
    // ---- weighted sum of partials, normalise, store ----
    {
      // same mapping as PV: 2 heads x dim pair per thread
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int h = h0 + i;
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
          const float sc = scale_s[s][h];
          const float2 v = *reinterpret_cast<const float2*>(&part_acc[s][h][2 * dp]);
          a0 = fmaf(v.x, sc, a0); a1 = fmaf(v.y, sc, a1);
        }
        const float L = Lfin_s[h];
        const float inv = (L > 0.f) ? 1.f / L : 0.f;
        *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)b * HQ + kvh * G + h) * D + 2 * dp) =
            __floats2bfloat162_rn(a0 * inv, a1 * inv);
      }
    }
    TSTAMP();  // merge done
  }
#ifdef GQA_TIMING
  if (tbuf != nullptr && tid == 0) {
    for (int i = 0; i < nts; ++i) tbuf[blockIdx.x * 16 + i] = ts[i];
    tbuf[blockIdx.x * 16 + 15] = nts;
  }
#endif
}

}  // namespace gqa

// ---- host launcher -----------------------------------------------------------
extern "C" cudaError_t gqa_paged_decode_launch(const void* q, const void* k_cache, const void* v_cache,
                                               const int32_t* kv_indptr, const int32_t* kv_indices,
                                               void* out, float* lse, int batch, int num_kv_heads, int nnz,
                                               float sm_scale, cudaStream_t stream, unsigned long long* tbuf) {
  constexpr int SPLITS = GQA_SPLITS, NT = GQA_THREADS, R = GQA_ROUNDS;
  auto kern = gqa::gqa_paged_decode_kernel<SPLITS, NT, R>;
  const float scale_log2 = sm_scale * 1.4426950408889634f;
  if (SPLITS > 8) {
    static bool attr_set = false;
    if (!attr_set) {
      cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      if (e != cudaSuccess) return e;
      attr_set = true;
    }
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(SPLITS * num_kv_heads, batch, 1);
  cfg.blockDim = dim3(NT, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = SPLITS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (SPLITS > 1) ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kern,
                            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache,
                            kv_indptr, kv_indices, (__nv_bfloat16*)out, lse, num_kv_heads, nnz, scale_log2, tbuf);
}
