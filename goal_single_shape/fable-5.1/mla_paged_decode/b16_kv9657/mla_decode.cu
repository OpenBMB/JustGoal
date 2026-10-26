// MLA paged decode (DeepSeek-V3 TP8 shape): H=16, d_ckv=512, d_kpe=64, page_size=1, bf16 in, bf16 out, fp32 lse (log2).
// Written from scratch. Design:
//   * one kernel launch, thread-block clusters of CL CTAs; each cluster works on one batch element
//     (a long batch may span several clusters, the per-CTA token count is balanced inside a batch)
//   * each CTA gathers up to T_MAX kv rows (ckv 1024B + kpe 128B) with 16B cp.async into padded smem rows
//   * S = Q K^T and O = P V on tensor cores (mma.sync m16n8k16 bf16 -> fp32); the 16 heads are the mma M dimension
//   * partial (O, m, l) of the CL CTAs combined through distributed shared memory (push model, 2 cluster barriers)
//   * batches spanning several clusters: each rank owns a DC/CL slice of the output; partial slices go to global
//     scratch and the last-arriving rank for that slice (release/acquire atomic counter) merges them
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <math.h>

namespace cg = cooperative_groups;

#ifndef T_MAX
#define T_MAX 64
#endif
#ifndef CL
#define CL 8
#endif
#define NTHREADS 128
#define NWARPS 4
#define HD 16
#define DC 512
#define DP 64
#define DK 576
#define ROW_BYTES 1152
#define ROW_STRIDE 1168           // 1152 + 16 pad: conflict-free ldmatrix
#define CHUNKS_PER_ROW 72         // 16B chunks per kv row (64 ckv + 8 kpe)
#define P_STRIDE (T_MAX * 2 + 16)
#define NT_TILES (T_MAX / 8)
#define CPT (T_MAX / 8)           // softmax columns per thread (8 threads per head row)
#define DPR (DC / CL)             // output dims owned by each cluster rank
#define VEC (DPR / 8)             // dims per thread in the combine (128 threads = 16 heads x 8)
#define TILES_PER_RANK (DPR / 8)  // mma n-tiles (8 dims) per rank slice

static_assert(T_MAX % 16 == 0 && T_MAX <= 128, "T_MAX");
static_assert(NT_TILES % NWARPS == 0, "tiles per warp");
static_assert(T_MAX * ROW_STRIDE >= HD * DC * 4, "O staging buffer must fit in K region");
static_assert(VEC % 4 == 0 && VEC * 8 * CL == DC, "combine split");

struct __align__(1024) Smem {
  uint8_t k[T_MAX * ROW_STRIDE];   // K tile [T][576] bf16 (padded rows); reused as red[CL][16][DPR] fp32
  uint8_t q[HD * ROW_STRIDE];      // Q [16][576] bf16 (padded rows)
  float s[HD][T_MAX];              // logits fp32
  uint8_t p[HD * P_STRIDE];        // P bf16 [16][T_MAX] (padded rows)
  float m[HD];                     // row max (log2 domain) of this CTA
  float l[HD];                     // row sum of this CTA
  int idx[T_MAX];
  int flag;
  int meta[8];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ int atom_add_acq_rel_gpu(int* p, int v) {
  int old;
  asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ float4 ld_cg_f4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ float ld_cg_f(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}

extern "C" __global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(NTHREADS)
mla_decode_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                  const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
                  const int* __restrict__ indptr, const int* __restrict__ indices, float scale_log2, int B,
                  int max_part, __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
                  float* __restrict__ part_o, float* __restrict__ part_ml, int* __restrict__ counters,
                  unsigned long long* __restrict__ prof) {
#ifdef PROF
  unsigned long long pt[12]; int npt = 0;
#define STAMP() do { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); pt[npt++] = t_; } while (0)
#else
#define STAMP() do {} while (0)
#endif
  STAMP();
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = (int)cluster.block_rank();
  const int cid = blockIdx.x / CL;

  // ---- work assignment: cluster id -> (batch b, cluster index j within batch) ----
  if (warp == 0) {
    int fb = -1, fj = 0, fncl = 0, fbeg = 0, fL = 0;
    int run = 0;
    for (int base = 0; base < B; base += 32) {
      const int b = base + lane;
      int beg = 0, end = 0;
      if (b < B) { beg = __ldg(indptr + b); end = __ldg(indptr + b + 1); }
      const int L = max(end - beg, 0);
      const int nchunk = (L + T_MAX - 1) / T_MAX;
      const int ncl = (b < B) ? max((nchunk + CL - 1) / CL, 1) : 0;
      int incl = ncl;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
      }
      const int excl = incl - ncl;
      const int total = __shfl_sync(0xffffffffu, incl, 31);
      if (b < B && cid >= run + excl && cid < run + incl) { fb = b; fj = cid - run - excl; fncl = ncl; fbeg = beg; fL = L; }
      run += total;
    }
    const unsigned msk = __ballot_sync(0xffffffffu, fb >= 0);
    const int src = msk ? (__ffs(msk) - 1) : 0;
    fb = __shfl_sync(0xffffffffu, fb, src);
    fj = __shfl_sync(0xffffffffu, fj, src);
    fncl = __shfl_sync(0xffffffffu, fncl, src);
    fbeg = __shfl_sync(0xffffffffu, fbeg, src);
    fL = __shfl_sync(0xffffffffu, fL, src);
    if (lane == 0) { sm.meta[0] = fb; sm.meta[1] = fj; sm.meta[2] = fncl; sm.meta[3] = fbeg; sm.meta[4] = fL; }
  }
  __syncthreads();
  STAMP();
  const int b = sm.meta[0];
  if (b < 0) return;  // whole cluster is idle
  const int j = sm.meta[1], ncl = sm.meta[2], beg = sm.meta[3], L = sm.meta[4];
  const int nct = ncl * CL;
  const int Tb = (L + nct - 1) / nct;  // tokens per CTA, <= T_MAX
  const int c = j * CL + rank;
  const int tok0 = c * Tb;
  const int valid = min(max(L - tok0, 0), Tb);
  const int n16 = (valid + 15) >> 4;

  // ---- gather (16B cp.async): Q rows first (independent of the index load), then the kv rows ----
  const uint32_t kbase = smem_u32(sm.k);
  const uint32_t qbase = smem_u32(sm.q);
  if (tid < valid) sm.idx[tid] = __ldg(indices + beg + tok0 + tid);
  {
    const int r = tid >> 3, part = tid & 7;
    const uint32_t dst = qbase + r * ROW_STRIDE + part * 16;
    const __nv_bfloat16* sc = q_nope + ((size_t)b * HD + r) * DC + part * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) cp_async16(dst + i * 128, sc + i * 64);
    cp_async16(dst + 1024, q_pe + ((size_t)b * HD + r) * DP + part * 8);
  }
  __syncthreads();
  {
    // 16 rows per pass, 8 threads per row, chunk = i*8 + part: the 8 lanes of a row cover 128 contiguous bytes
    constexpr int NPASS = T_MAX / 16;
    const int rl = tid >> 3, part = tid & 7;
#pragma unroll
    for (int ps = 0; ps < NPASS; ++ps) {
      const int r = ps * 16 + rl;
      if (r < valid) {
        const size_t page = (size_t)sm.idx[r];
        const uint32_t dst = kbase + r * ROW_STRIDE + part * 16;
        const __nv_bfloat16* sc = ckv + page * DC + part * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) cp_async16(dst + i * 128, sc + i * 64);
        cp_async16(dst + 1024, kpe + page * DP + part * 8);
      }
    }
    cp_async_commit();
  }
  {  // zero the ckv part of tail rows [valid, n16*16) so P(=0) * V(garbage) can not produce NaN
    const int nrows = n16 * 16 - valid;
    for (int e = tid; e < nrows * 64; e += NTHREADS) {
      const int r = valid + (e >> 6), cc = e & 63;
      *reinterpret_cast<uint4*>(sm.k + r * ROW_STRIDE + cc * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  STAMP();
  cp_async_wait_all();
  __syncthreads();
  STAMP();

  // ---- S = Q K^T : each warp owns n-tiles {warp, warp+4, ...}; even/odd k-steps use separate accumulators ----
  if (valid > 0) {
    const uint32_t a_addr = qbase + (lane & 15) * ROW_STRIDE + (lane >> 4) * 16;
    constexpr int TPW = NT_TILES / NWARPS;  // tiles per warp
    float acc_e[TPW][4], acc_o[TPW][4];
    uint32_t b_addr[TPW];
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      acc_e[t][0] = acc_e[t][1] = acc_e[t][2] = acc_e[t][3] = 0.f;
      acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.f;
      b_addr[t] = kbase + ((warp + t * NWARPS) * 8 + (lane & 7)) * ROW_STRIDE + (lane >> 3) * 16;
    }
    const int ntiles = min(TPW, (valid - warp * 8 + NWARPS * 8 - 1) / (NWARPS * 8));  // tiles with any valid token
    if (ntiles == TPW) {
#pragma unroll 3
      for (int ks = 0; ks < DK / 32; ++ks) {
        uint32_t a0[4], a1[4];
        ldsm_x4(a0, a_addr + ks * 64);
        ldsm_x4(a1, a_addr + ks * 64 + 32);
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
          uint32_t bb[4];
          ldsm_x4(bb, b_addr[t] + ks * 64);
          mma_bf16(acc_e[t], a0, bb[0], bb[1]);
          mma_bf16(acc_o[t], a1, bb[2], bb[3]);
        }
      }
    } else {
#pragma unroll 3
      for (int ks = 0; ks < DK / 32; ++ks) {
        uint32_t a0[4], a1[4];
        ldsm_x4(a0, a_addr + ks * 64);
        ldsm_x4(a1, a_addr + ks * 64 + 32);
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
          if (t < ntiles) {
            uint32_t bb[4];
            ldsm_x4(bb, b_addr[t] + ks * 64);
            mma_bf16(acc_e[t], a0, bb[0], bb[1]);
            mma_bf16(acc_o[t], a1, bb[2], bb[3]);
          }
        }
      }
    }
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int t = 0; t < TPW; ++t) {
      if (t < ntiles) {
        const int col = (warp + t * NWARPS) * 8 + tq * 2;
        *reinterpret_cast<float2*>(&sm.s[g][col]) = make_float2(acc_e[t][0] + acc_o[t][0], acc_e[t][1] + acc_o[t][1]);
        *reinterpret_cast<float2*>(&sm.s[g + 8][col]) = make_float2(acc_e[t][2] + acc_o[t][2], acc_e[t][3] + acc_o[t][3]);
      }
    }
  }
  __syncthreads();
  STAMP();

  // ---- softmax (8 threads per head row) ----
  {
    const int h = tid >> 3, sub = tid & 7;
    if (valid > 0) {
      float sv[CPT];
      float mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < CPT; ++i) {
        const int col = sub * CPT + i;
        const float v = (col < valid) ? sm.s[h][col] * scale_log2 : -INFINITY;
        sv[i] = v;
        mx = fmaxf(mx, v);
      }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      float sum = 0.f;
      uint32_t pk[CPT / 2];
#pragma unroll
      for (int i = 0; i < CPT; i += 2) {
        const int col = sub * CPT + i;
        const float p0 = (col < valid) ? ex2(sv[i] - mx) : 0.f;
        const float p1 = (col + 1 < valid) ? ex2(sv[i + 1] - mx) : 0.f;
        sum += p0 + p1;
        pk[i / 2] = pack_bf16(p0, p1);
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      uint32_t* pdst = reinterpret_cast<uint32_t*>(sm.p + h * P_STRIDE + sub * CPT * 2);
#pragma unroll
      for (int i = 0; i < CPT / 2; ++i) pdst[i] = pk[i];
      if (sub == 0) { sm.m[h] = mx; sm.l[h] = sum; }
    } else if (sub == 0) {
      sm.m[h] = -INFINITY; sm.l[h] = 0.f;
    }
  }
  __syncthreads();
  STAMP();

  // ---- O = P V (per warp: 128 dims = 16 n-tiles) ----
  float o[16][4];
#pragma unroll
  for (int jj = 0; jj < 16; ++jj) { o[jj][0] = o[jj][1] = o[jj][2] = o[jj][3] = 0.f; }
  if (valid > 0) {
    const uint32_t pbase = smem_u32(sm.p) + (lane & 15) * P_STRIDE + (lane >> 4) * 16;
    const uint32_t vb = kbase + ((lane & 7) + ((lane >> 3) & 1) * 8) * ROW_STRIDE + warp * 256 + (lane >> 4) * 16;
    for (int ks = 0; ks < n16; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, pbase + ks * 32);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        uint32_t bb[4];
        ldsm_x4_t(bb, vb + ks * 16 * ROW_STRIDE + jj * 32);
        mma_bf16(o[2 * jj], a, bb[0], bb[1]);
        mma_bf16(o[2 * jj + 1], a, bb[2], bb[3]);
      }
    }
  }
  STAMP();
  __syncthreads();  // this CTA is done with its K tile -> stage O there as fp32 [16][512]
  float* stage = reinterpret_cast<float*>(sm.k);
  {
    const int g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) {
      const int d = warp * 128 + jj * 8 + tq * 2;
      *reinterpret_cast<float2*>(stage + g * DC + d) = make_float2(o[jj][0], o[jj][1]);
      *reinterpret_cast<float2*>(stage + (g + 8) * DC + d) = make_float2(o[jj][2], o[jj][3]);
    }
  }
  cluster.sync();  // every CTA of the cluster has staged (O, m, l)
  STAMP();

  // ---- combine inside the cluster (DSMEM pull, all loads issued up front) ----
  // rank r owns dims [r*DPR, (r+1)*DPR); thread (hh, dg) owns head hh and VEC dims split in VEC/4 groups of 4:
  // group q -> dims r*DPR + q*(DPR/(VEC/4)) + dg*4 + {0..3}   (each group is 128B contiguous across the 8 dg lanes)
  constexpr int NG = VEC / 4;          // float4 groups per thread
  constexpr int GSTRIDE = DPR / NG;    // dims between groups
  const int hh = tid >> 3, dg = tid & 7;
  const int dbase = rank * DPR + dg * 4;
  float mi[CL], li[CL];
  float4 xi[CL][NG];
#pragma unroll
  for (int i = 0; i < CL; ++i) {
    const float* pm = cluster.map_shared_rank(sm.m, i);
    const float* pl = cluster.map_shared_rank(sm.l, i);
    const float* ps = cluster.map_shared_rank(stage, i) + hh * DC + dbase;
    mi[i] = pm[hh];
    li[i] = pl[hh];
#pragma unroll
    for (int q = 0; q < NG; ++q) xi[i][q] = *reinterpret_cast<const float4*>(ps + q * GSTRIDE);
  }
  cluster.barrier_arrive();  // my pulls from the peers are complete (values are in registers)
  float M = -INFINITY;
#pragma unroll
  for (int i = 0; i < CL; ++i) M = fmaxf(M, mi[i]);
  float O[VEC];
#pragma unroll
  for (int v = 0; v < VEC; ++v) O[v] = 0.f;
  float Lsum = 0.f;
  if (M > -INFINITY) {
#pragma unroll
    for (int i = 0; i < CL; ++i) {
      const float w = ex2(mi[i] - M);
      Lsum += w * li[i];
#pragma unroll
      for (int q = 0; q < NG; ++q) {
        O[q * 4] += w * xi[i][q].x; O[q * 4 + 1] += w * xi[i][q].y;
        O[q * 4 + 2] += w * xi[i][q].z; O[q * 4 + 3] += w * xi[i][q].w;
      }
    }
  }
  STAMP();

  auto write_final = [&](float Mf, float Lf, const float* Of) {
    __nv_bfloat16* dst = out + ((size_t)b * HD + hh) * DC + dbase;
    if (Mf > -INFINITY && Lf > 0.f) {
      const float inv = 1.f / Lf;
#pragma unroll
      for (int q = 0; q < NG; ++q) {
        uint2 u;
        u.x = pack_bf16(Of[q * 4] * inv, Of[q * 4 + 1] * inv);
        u.y = pack_bf16(Of[q * 4 + 2] * inv, Of[q * 4 + 3] * inv);
        *reinterpret_cast<uint2*>(dst + q * GSTRIDE) = u;
      }
    } else {
#pragma unroll
      for (int q = 0; q < NG; ++q) *reinterpret_cast<uint2*>(dst + q * GSTRIDE) = make_uint2(0u, 0u);
    }
    if (rank == 0 && dg == 0) lse[b * HD + hh] = (Mf > -INFINITY && Lf > 0.f) ? (Mf + log2f(Lf)) : -INFINITY;
  };

  if (ncl == 1) {
    write_final(M, Lsum, O);
  } else {
    // partial slice of this rank -> global scratch; the last rank (over the clusters of this batch) to arrive
    // for this slice merges. counters layout: [B][CL]
    const size_t pidx = (size_t)b * max_part + j;
    float* po = part_o + (pidx * HD + hh) * DC + dbase;
#pragma unroll
    for (int q = 0; q < NG; ++q)
      *reinterpret_cast<float4*>(po + q * GSTRIDE) = make_float4(O[q * 4], O[q * 4 + 1], O[q * 4 + 2], O[q * 4 + 3]);
    if (dg == 0) { part_ml[(pidx * CL + rank) * 32 + hh] = M; part_ml[(pidx * CL + rank) * 32 + HD + hh] = Lsum; }
    __syncthreads();
    if (tid == 0) {
      const int old = atom_add_acq_rel_gpu(counters + b * CL + rank, 1);
      sm.flag = (old == ncl - 1) ? 1 : 0;
    }
    __syncthreads();
    if (sm.flag) {
      float M2 = -INFINITY;
      float mj[8];
      for (int jj = 0; jj < ncl; ++jj) {
        const float v = ld_cg_f(part_ml + (((size_t)b * max_part + jj) * CL + rank) * 32 + hh);
        if (jj < 8) mj[jj] = v;
        M2 = fmaxf(M2, v);
      }
      float O2[VEC];
#pragma unroll
      for (int v = 0; v < VEC; ++v) O2[v] = 0.f;
      float L2 = 0.f;
      if (M2 > -INFINITY) {
        for (int jj = 0; jj < ncl; ++jj) {
          const size_t pj = (size_t)b * max_part + jj;
          const float mjj = (jj < 8) ? mj[jj] : ld_cg_f(part_ml + (pj * CL + rank) * 32 + hh);
          const float w = ex2(mjj - M2);
          L2 += w * ld_cg_f(part_ml + (pj * CL + rank) * 32 + HD + hh);
          const float* pp = part_o + (pj * HD + hh) * DC + dbase;
#pragma unroll
          for (int q = 0; q < NG; ++q) {
            const float4 x = ld_cg_f4(pp + q * GSTRIDE);
            O2[q * 4] += w * x.x; O2[q * 4 + 1] += w * x.y; O2[q * 4 + 2] += w * x.z; O2[q * 4 + 3] += w * x.w;
          }
        }
      }
      write_final(M2, L2, O2);
      if (tid == 0) counters[b * CL + rank] = 0;  // reset for the next launch
    }
  }
  STAMP();
  cluster.barrier_wait();  // keep the staged smem alive until every peer has pulled from it
  STAMP();
#ifdef PROF
  if (tid == 0) { for (int i = 0; i < npt; ++i) prof[blockIdx.x * 16 + i] = pt[i]; prof[blockIdx.x * 16 + 15] = valid; }
#endif
}

extern "C" int mla_decode_launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe, const int* indptr,
                                 const int* indices, float sm_scale, int B, int n_indices, void* out, float* lse,
                                 float* part_o, float* part_ml, int* counters, int max_part, cudaStream_t stream,
                                 unsigned long long* prof) {
  static bool inited = false;
  if (!inited) {
    cudaError_t e = cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
    if (e != cudaSuccess) return (int)e;
    if (CL > 8) {
      e = cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      if (e != cudaSuccess) return (int)e;
    }
    inited = true;
  }
  const int max_clusters = n_indices / (T_MAX * CL) + B;
  const float scale_log2 = sm_scale * 1.4426950408889634f;
  mla_decode_kernel<<<dim3(max_clusters * CL), NTHREADS, sizeof(Smem), stream>>>(
      (const __nv_bfloat16*)q_nope, (const __nv_bfloat16*)q_pe, (const __nv_bfloat16*)ckv, (const __nv_bfloat16*)kpe,
      indptr, indices, scale_log2, B, max_part, (__nv_bfloat16*)out, lse, part_o, part_ml, counters, prof);
  return (int)cudaGetLastError();
}

extern "C" int mla_decode_smem_bytes() { return (int)sizeof(Smem); }
extern "C" int mla_decode_max_part(int n_indices) { return n_indices / (T_MAX * CL) + 1; }
extern "C" int mla_decode_cluster_size() { return CL; }
