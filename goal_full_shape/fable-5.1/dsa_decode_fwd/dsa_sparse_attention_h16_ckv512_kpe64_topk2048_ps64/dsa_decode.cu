// DSA sparse MLA decode kernel for B200 (sm_100a) — v5.
//
// Work unit = (token t, ROWS = 32*NB consecutive *raw* positions of the token's top-k list).
// grid = T * NSPLIT CTAs of 256 threads.  A CTA first probes only its own ROWS indices; if none is
// valid it exits at once (or, when all CTAs of a token are co-resident, becomes a combine helper).
// Workers load their K rows straight into mma fragments, overlap the read of the full index list
// (needed only to learn which units are workers) with the QK/PV math, and either write the output
// directly (single worker) or publish a partial (bf16 O, fp32 m/l) and reduce cooperatively.
// Synchronisation between the workers of a token uses epoch-stamped flags (no counters to reset).
// When the grid may exceed the number of co-resident CTAs, CTAs take dynamic tickets (token order
// == start order), which makes the flag barrier deadlock-free.
//
// Data path: no shared-memory staging of K/V.  Warp w owns ckv dims [64w, 64w+64) and kpe dims
// [8w, 8w+8).  Each thread loads 16-byte pieces of its rows straight into mma.sync m16n8k16 B
// fragments (a fixed permutation of the dims, applied identically to Q, makes that legal); the 8
// warps' partial logits are summed through smem by warp 0 which also does the softmax; the same K
// registers are turned into V fragments with movmatrix.trans for the PV product.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define NTHREADS 256
#define MAX_TOPK 2048
#define MAX_SPLIT 64
#define NSLICE 64          // output slices of 8 dims
#define PSTRIDE_BYTES(rows) ((rows) * 2 + 16)

// smem layout (bytes)
#define OFF_XCH 0                        // xch [NB*4][8][32] float4 (<= 32 KB)
#define OFF_BUF (OFF_XCH + 32768)        // combine run buffer overflow area: xch+buf = [8][9][32] float4 = 36864
#define OFF_P (OFF_BUF + 8192)           // P [16][ROWS+8] bf16 (<= 2304)
#define OFF_STATS (OFF_P + 2304)         // m[16], l[16] | M[16], inv[16]
#define OFF_W (OFF_STATS + 128)          // W [64][16] f32 combine weights
#define OFF_RED (OFF_W + 4096)           // [16][16] f32 x2
#define OFF_WU (OFF_RED + 2048)          // worker unit ids [64] int
#define OFF_SCAN (OFF_WU + 256)          // misc ints [32] + unit flags [64]
#define SMEM_TOTAL (OFF_SCAN + 512)      // 50304

#if defined(EXP_PLAINLD)
#define LD_QUAL "ld.global"
#else
#define LD_QUAL "ld.global.nc.L1::no_allocate"
#endif
__device__ __forceinline__ uint4 ld16(const void* p) {
  uint4 r;
  asm volatile(LD_QUAL ".v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld4(const void* p) {
  uint32_t r;
  asm volatile(LD_QUAL ".u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ int4 ld16i(const void* p) {
  int4 r;
  asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ uint32_t movtrans(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16(uint32_t x) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&x);
  return __bfloat1622float2(v);
}
__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void st_release_gpu(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// loads of partials after the acquire barrier: weak loads are legal (L1 was invalidated by the acquire).
// asm volatile keeps the compiler from sinking them to their uses (which would serialise the round trips).
__device__ __forceinline__ uint2 ldp_u2(const void* p) {
  uint2 r;
  asm volatile("ld.global.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ float ldp_f(const float* p) {
  float r;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

struct Workspace {
  unsigned int* ticket;  // monotonically increasing ticket counter
  int* flags;            // [cap][MAX_SPLIT] epoch of the last published partial
  float* part_ml;        // [cap][MAX_SPLIT][32]
  uint16_t* part_o;      // [cap][MAX_SPLIT][16][512] bf16
};

template <int NB, bool USE_TICKET, int MINB>
__global__ void __launch_bounds__(NTHREADS, MINB)
    dsa_decode_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                      const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
                      const int* __restrict__ sidx, __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
                      int topk, int num_kv, float scale_log2, int nsplit, unsigned int ticket_base, int epoch,
                      Workspace ws, long long* __restrict__ dbg) {
  constexpr int ROWS = 32 * NB;
  constexpr int PSTRIDE = PSTRIDE_BYTES(ROWS);
#ifdef DSA_DEBUG
#define TS(k) do { if (threadIdx.x == 0) { long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); dbg[bid * 16 + (k)] = _t; } } while (0)
#else
#define TS(k) do { } while (0)
#endif
  __shared__ __align__(128) uint8_t smem[SMEM_TOTAL];
  float4* xch = reinterpret_cast<float4*>(smem + OFF_XCH);
  float* stats = reinterpret_cast<float*>(smem + OFF_STATS);
  float* W = reinterpret_cast<float*>(smem + OFF_W);
  float* red = reinterpret_cast<float*>(smem + OFF_RED);
  int* wu = reinterpret_cast<int*>(smem + OFF_WU);
  int* scan_s = reinterpret_cast<int*>(smem + OFF_SCAN);
  const uint32_t p_base = (uint32_t)__cvta_generic_to_shared(smem + OFF_P);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
  int bid = blockIdx.x;
  if (USE_TICKET) {
    if (tid == 0) scan_s[24] = (int)(atomicAdd(ws.ticket, 1u) - ticket_base);
    __syncthreads();
    bid = scan_s[24];
  }
  TS(0);
  const int u = bid % nsplit;
  const int t = bid / nsplit;
  const int* tidx = sidx + (size_t)t * topk;
  const int row0 = u * ROWS;

  // ---- probe: this unit's own row indices.  Tile row g of tile n (batch b) is unit entry
  //      32b + 4g + n, so a lane's four indices of a batch are one 16-byte load. ----
  int ridx[NB][4];
  bool anyv = false;
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const int p0 = row0 + 32 * b + 4 * g;
    int4 q4;
    if (p0 + 4 <= topk) q4 = ld16i(tidx + p0);
    else { q4.x = (p0 < topk) ? tidx[p0] : -1; q4.y = (p0 + 1 < topk) ? tidx[p0 + 1] : -1; q4.z = (p0 + 2 < topk) ? tidx[p0 + 2] : -1; q4.w = (p0 + 3 < topk) ? tidx[p0 + 3] : -1; }
    ridx[b][0] = (q4.x < 0 || q4.x >= num_kv) ? -1 : q4.x;
    ridx[b][1] = (q4.y < 0 || q4.y >= num_kv) ? -1 : q4.y;
    ridx[b][2] = (q4.z < 0 || q4.z >= num_kv) ? -1 : q4.z;
    ridx[b][3] = (q4.w < 0 || q4.w >= num_kv) ? -1 : q4.w;
    anyv |= (ridx[b][0] >= 0) | (ridx[b][1] >= 0) | (ridx[b][2] >= 0) | (ridx[b][3] >= 0);
  }
  anyv = __any_sync(0xffffffffu, anyv);  // identical in every warp (all warps probe the same entries)
  TS(1);
  const bool worker = anyv;
  if (!worker && USE_TICKET && u != 0) return;  // idle, nobody will need us
  // ---- full index list (8 consecutive entries per thread): which units have valid rows? ----
  int v8[8];
  auto load_list = [&]() {
    const int base = tid * 8;
    const int* src = tidx + base;
    if (base + 8 <= topk && (((size_t)src) & 15) == 0) {
      int4 a = ld16i(src);
      int4 b = ld16i(src + 4);
      v8[0] = a.x; v8[1] = a.y; v8[2] = a.z; v8[3] = a.w; v8[4] = b.x; v8[5] = b.y; v8[6] = b.z; v8[7] = b.w;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) v8[i] = (base + i < topk) ? src[i] : -1;
    }
  };
  if (!USE_TICKET) load_list();

  // ---- Q fragments first (they do not depend on the probe), then the K row loads (workers only) ----
  uint32_t kr[NB][4][2][4], kp[NB][4];
  uint32_t qa[2][4], qb[2][4], qpa, qpb;
  if (worker) {
    // Q fragments (permuted dims)
    const __nv_bfloat16* qn0 = q_nope + ((size_t)t * 16 + g) * 512 + 64 * warp + 8 * c;
    const __nv_bfloat16* qn1 = qn0 + 8 * 512;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint4 a = ld16(qn0 + 32 * i), b = ld16(qn1 + 32 * i);
      qa[i][0] = a.x; qa[i][1] = a.y; qa[i][2] = a.z; qa[i][3] = a.w;
      qb[i][0] = b.x; qb[i][1] = b.y; qb[i][2] = b.z; qb[i][3] = b.w;
    }
    const __nv_bfloat16* qp0 = q_pe + ((size_t)t * 16 + g) * 64 + 8 * warp + 2 * c;
    qpa = ld4(qp0);
    qpb = ld4(qp0 + 8 * 64);
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int v = ridx[b][n];
        if (v >= 0) {
          const __nv_bfloat16* kc = ckv + (size_t)v * 512 + 64 * warp + 8 * c;
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            uint4 a = ld16(kc + 32 * i);
            kr[b][n][i][0] = a.x; kr[b][n][i][1] = a.y; kr[b][n][i][2] = a.z; kr[b][n][i][3] = a.w;
          }
          kp[b][n] = ld4(kpe + (size_t)v * 64 + 8 * warp + 2 * c);
        } else {
#pragma unroll
          for (int i = 0; i < 2; ++i) { kr[b][n][i][0] = 0; kr[b][n][i][1] = 0; kr[b][n][i][2] = 0; kr[b][n][i][3] = 0; }
          kp[b][n] = 0;
        }
      }
  }
  if (USE_TICKET) load_list();
  TS(2);
  // per-unit "any valid" flags over the full list (thread tid holds raw entries [8 tid, 8 tid + 8))
  {
    int myany = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) myany |= (v8[i] >= 0 && v8[i] < num_kv && tid * 8 + i < topk);
    const unsigned bal = __ballot_sync(0xffffffffu, myany);
    constexpr int TPU = ROWS / 8;           // threads per unit
    constexpr int UPW = 32 / TPU;           // units per warp (8 or 4)
    unsigned fl = 0u;
#pragma unroll
    for (int k = 0; k < UPW; ++k) fl |= (((bal >> (k * TPU)) & ((1u << TPU) - 1u)) != 0u) ? (1u << k) : 0u;
    if (lane == 0) scan_s[8 + warp] = (int)fl;   // unit flags of units [UPW*warp, UPW*warp+UPW)
  }

  float o[2][4][4];
  if (worker) {
    // ---- QK: partial logits over this warp's dims for all rows ----
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 2; ++j)
            mma_bf16(s, qa[i][2 * j], qb[i][2 * j], qa[i][2 * j + 1], qb[i][2 * j + 1], kr[b][n][i][2 * j], kr[b][n][i][2 * j + 1]);
        mma_bf16(s, qpa, qpb, 0u, 0u, kp[b][n], 0u);
        xch[((b * 4 + n) * 8 + warp) * 32 + lane] = make_float4(s[0], s[1], s[2], s[3]);
      }
    __syncthreads();
    TS(3);
    // ---- softmax: warp w < 4 handles row tiles w, w+4, ... (lane holds heads g, g+8; cols 2c, 2c+1) ----
    float* wmax = reinterpret_cast<float*>(smem + OFF_W);        // [4][16] (combine W region, free here)
    float* wsum = wmax + 64;                                     // [4][16]
    constexpr int TPW = NB;  // tiles per warp
    float sv[TPW][4];
    if (warp < 4) {
      float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
      for (int q = 0; q < TPW; ++q) {
        const int nt = warp + 4 * q;
        float4 acc = xch[(nt * 8) * 32 + lane];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
          const float4 v = xch[(nt * 8 + w) * 32 + lane];
          acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        // column validity: row index of column (nt, 2c+e) is ridx[nt/4][nt%4] held by lane with g == 2c+e
        // ridx[b][warp] without dynamic register indexing
        const int bq = (q == 0) ? 0 : (NB - 1);
        const int rsel = (warp == 0) ? ridx[bq][0] : (warp == 1) ? ridx[bq][1] : (warp == 2) ? ridx[bq][2] : ridx[bq][3];
        const int vlo = __shfl_sync(0xffffffffu, rsel, (2 * c) * 4);
        const int vhi = __shfl_sync(0xffffffffu, rsel, (2 * c + 1) * 4);
        sv[q][0] = (vlo >= 0) ? acc.x * scale_log2 : -INFINITY;
        sv[q][1] = (vhi >= 0) ? acc.y * scale_log2 : -INFINITY;
        sv[q][2] = (vlo >= 0) ? acc.z * scale_log2 : -INFINITY;
        sv[q][3] = (vhi >= 0) ? acc.w * scale_log2 : -INFINITY;
        mx0 = fmaxf(mx0, fmaxf(sv[q][0], sv[q][1]));
        mx1 = fmaxf(mx1, fmaxf(sv[q][2], sv[q][3]));
      }
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      if (c == 0) { wmax[warp * 16 + g] = mx0; wmax[warp * 16 + g + 8] = mx1; }
    }
    __syncthreads();
    {
      const float m0 = fmaxf(fmaxf(wmax[g], wmax[16 + g]), fmaxf(wmax[32 + g], wmax[48 + g]));
      const float m1 = fmaxf(fmaxf(wmax[g + 8], wmax[16 + g + 8]), fmaxf(wmax[32 + g + 8], wmax[48 + g + 8]));
      if (warp < 4) {
        float sum0 = 0.f, sum1 = 0.f;
        uint8_t* pb = smem + OFF_P;
#pragma unroll
        for (int q = 0; q < TPW; ++q) {
          const int nt = warp + 4 * q;
          const float p0 = fast_exp2(sv[q][0] - m0), p1 = fast_exp2(sv[q][1] - m0);
          const float p2 = fast_exp2(sv[q][2] - m1), p3 = fast_exp2(sv[q][3] - m1);
          sum0 += p0 + p1;
          sum1 += p2 + p3;
          const int col = 8 * nt + 2 * c;
          *reinterpret_cast<uint32_t*>(pb + g * PSTRIDE + col * 2) = pack_bf16(p0, p1);
          *reinterpret_cast<uint32_t*>(pb + (g + 8) * PSTRIDE + col * 2) = pack_bf16(p2, p3);
        }
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        if (c == 0) { wsum[warp * 16 + g] = sum0; wsum[warp * 16 + g + 8] = sum1; }
      }
      if (warp == 0 && c == 0) { stats[g] = m0; stats[g + 8] = m1; }
    }
    __syncthreads();
    if (warp == 0 && c == 0) {
      stats[16 + g] = wsum[g] + wsum[16 + g] + wsum[32 + g] + wsum[48 + g];
      stats[16 + g + 8] = wsum[g + 8] + wsum[16 + g + 8] + wsum[32 + g + 8] + wsum[48 + g + 8];
    }
    __syncthreads();
    TS(9);
    // ---- PV: O[16 heads][this warp's 64 ckv dims] over all rows ----
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) { o[i][jj][0] = 0.f; o[i][jj][1] = 0.f; o[i][jj][2] = 0.f; o[i][jj][3] = 0.f; }
#pragma unroll
    for (int ks = 0; ks < 2 * NB; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldsm_x4(p_base + (lane & 15) * PSTRIDE + (16 * ks + 8 * (lane >> 4)) * 2, a0, a1, a2, a3);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const uint32_t b0 = movtrans(kr[ks >> 1][(ks & 1) * 2][i][jj]);
          const uint32_t b1 = movtrans(kr[ks >> 1][(ks & 1) * 2 + 1][i][jj]);
          mma_bf16(o[i][jj], a0, a1, a2, a3, b0, b1);
        }
    }
    TS(4);
  }

  // ---- worker units of this token (flags were written by every thread before the barriers above) ----
  if (!worker) __syncthreads();  // non-workers skipped the compute barriers
  unsigned long long umask = 0ull;
#pragma unroll
  for (int w = 0; w < 8; ++w) umask |= ((unsigned long long)(unsigned)scan_s[8 + w]) << (w * (32 / (ROWS / 8)));
  const int nwork = __popcll(umask);
  const int myrank = __popcll(umask & ((1ull << u) - 1ull));
  if (tid < nsplit && ((umask >> tid) & 1ull)) wu[__popcll(umask & ((1ull << tid) - 1ull))] = tid;
  if (nwork >= 2) __syncthreads();  // the worker list is only read on the combine path
  TS(5);
  int* tflags = ws.flags + (size_t)t * MAX_SPLIT;
  const int ncomb = USE_TICKET ? nwork : nsplit;  // CTAs taking part in the combine
  if (!worker) {
    if (nwork == 0 && u == 0) {
      uint4 z = make_uint4(0, 0, 0, 0);
      uint4* op = reinterpret_cast<uint4*>(out + (size_t)t * 16 * 512);
      for (int i = tid; i < 16 * 512 / 8; i += NTHREADS) op[i] = z;
      if (tid < 16) lse[t * 16 + tid] = -INFINITY;
    }
    if (USE_TICKET || nwork <= 1) return;
  } else {
    const float m0 = stats[g], m1 = stats[g + 8], l0 = stats[16 + g], l1 = stats[16 + g + 8];
    if (nwork == 1) {
      const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int d = 64 * warp + 32 * i + 8 * c;
        uint4 pk0, pk1;
        pk0.x = pack_bf16(o[i][0][0] * inv0, o[i][0][1] * inv0); pk0.y = pack_bf16(o[i][1][0] * inv0, o[i][1][1] * inv0);
        pk0.z = pack_bf16(o[i][2][0] * inv0, o[i][2][1] * inv0); pk0.w = pack_bf16(o[i][3][0] * inv0, o[i][3][1] * inv0);
        pk1.x = pack_bf16(o[i][0][2] * inv1, o[i][0][3] * inv1); pk1.y = pack_bf16(o[i][1][2] * inv1, o[i][1][3] * inv1);
        pk1.z = pack_bf16(o[i][2][2] * inv1, o[i][2][3] * inv1); pk1.w = pack_bf16(o[i][3][2] * inv1, o[i][3][3] * inv1);
        *reinterpret_cast<uint4*>(out + ((size_t)t * 16 + g) * 512 + d) = pk0;
        *reinterpret_cast<uint4*>(out + ((size_t)t * 16 + g + 8) * 512 + d) = pk1;
      }
      if (warp == 0 && c == 0) {
        lse[t * 16 + g] = m0 + log2f(l0);
        lse[t * 16 + g + 8] = m1 + log2f(l1);
      }
      TS(6);
      return;
    }
    // ---- publish the partial (bf16 O, fp32 m/l) under slot u ----
    uint16_t* my_o = ws.part_o + ((size_t)t * MAX_SPLIT + u) * (16 * 512);
    float* my_ml = ws.part_ml + ((size_t)t * MAX_SPLIT + u) * 32;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int d = 64 * warp + 32 * i + 8 * c;
      uint4 pk0, pk1;
      pk0.x = pack_bf16(o[i][0][0], o[i][0][1]); pk0.y = pack_bf16(o[i][1][0], o[i][1][1]);
      pk0.z = pack_bf16(o[i][2][0], o[i][2][1]); pk0.w = pack_bf16(o[i][3][0], o[i][3][1]);
      pk1.x = pack_bf16(o[i][0][2], o[i][0][3]); pk1.y = pack_bf16(o[i][1][2], o[i][1][3]);
      pk1.z = pack_bf16(o[i][2][2], o[i][2][3]); pk1.w = pack_bf16(o[i][3][2], o[i][3][3]);
      *reinterpret_cast<uint4*>(my_o + g * 512 + d) = pk0;
      *reinterpret_cast<uint4*>(my_o + (g + 8) * 512 + d) = pk1;
    }
    if (warp == 0 && c == 0) {
      my_ml[g] = m0; my_ml[g + 8] = m1; my_ml[16 + g] = l0; my_ml[16 + g + 8] = l1;
    }
    __syncthreads();
    TS(14);
    if (tid == 0) st_release_gpu(tflags + u, epoch);
    TS(15);
  }
  // ---- wait until every worker of the token has published (warp 0 polls, others wait at the barrier) ----
  if (warp == 0) {
    int iters = 0;
    while (true) {
      ++iters;
      bool mine = true;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int r = lane + 32 * k;
#ifdef EXP_RELAXED
        if (r < nwork) mine &= (__ldcg(tflags + wu[r]) == epoch);
#else
        if (r < nwork) mine &= (ld_acquire_gpu(tflags + wu[r]) == epoch);
#endif
      }
      if (__all_sync(0xffffffffu, mine)) break;
#ifdef DSA_DEBUG
      if (iters > 100000) break;
#endif
#ifdef EXP_SLEEP
      __nanosleep(EXP_SLEEP);
#endif
    }
#ifdef EXP_RELAXED
    __threadfence();  // acquire side: order the relaxed observations before the partial reads
#endif
#ifdef DSA_DEBUG
    if (lane == 0) dbg[bid * 16 + 1] = iters;  // overwrite the probe stamp slot with the poll count
#endif
  }
  __syncthreads();
  TS(7);
  // ---- combine: every participating CTA finalises its slices ----
  {
    const float* tok_ml = ws.part_ml + (size_t)t * MAX_SPLIT * 32;
    const uint16_t* tok_o = ws.part_o + (size_t)t * MAX_SPLIT * (16 * 512);
    const int cidx = USE_TICKET ? myrank : u;   // this CTA's index among the ncomb participants
    const int rg = tid >> 5, l32 = tid & 31, hh2 = l32 >> 1, half = l32 & 1;
    // pass 0: issue every global load of the combine at once
    //   m/l: thread (hh = tid & 15, gi = tid >> 4) loads ranks gi + 16k
    const int hh = tid & 15, gi = tid >> 4;
    float mr[4], lr[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = gi + 16 * k;
      const int slot = (r < nwork) ? wu[r] : 0;
      mr[k] = ldp_f(tok_ml + slot * 32 + hh);        // slot 0 when r >= nwork (harmless), masked below
      lr[k] = ldp_f(tok_ml + slot * 32 + 16 + hh);
      if (r >= nwork) { mr[k] = -INFINITY; lr[k] = 0.f; }
    }
    // per group: first pair p0 = 8 rg -> (si0, r0) with ONE division; the rest is incremental
    const int si0 = (8 * rg) / nwork, r0 = 8 * rg - si0 * nwork;
    const int SG = max(1, 64 / nwork);             // slices per iteration (pairs <= 64)
    constexpr int RPG = 9;  // max runs per group: 8 pairs span at most ceil(8/nwork)+1 <= 9 slices
    float4* rbuf = xch;
    uint2 raw[8];
    int pr[8];
    auto issue_raw = [&](int s0) {  // the 8 partial-slice loads of this thread for slice group s0, back to back
      const int nsl = min(SG, (NSLICE - 1 - s0) / ncomb + 1);
      const int npairs = nsl * nwork;
      int si = si0, r = r0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int p = 8 * rg + k;
        pr[k] = r;
        const int sl = (p < npairs) ? (s0 + si * ncomb) : s0;
        raw[k] = ldp_u2(tok_o + (size_t)wu[r] * (16 * 512) + hh2 * 512 + 8 * sl + 4 * half);
        if (++r == nwork) { r = 0; ++si; }
      }
    };
    issue_raw(cidx);  // same round trip as the m/l loads above
    if (l32 == 0) scan_s[16 + rg] = si0;   // first slice-local index of each group (for pass 3)
    // pass 1: M, L, W
    float M = fmaxf(fmaxf(mr[0], mr[1]), fmaxf(mr[2], mr[3]));
    red[gi * 16 + hh] = M;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) M = fmaxf(M, red[k * 16 + hh]);
    float L = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = gi + 16 * k;
      const float w = (mr[k] == -INFINITY) ? 0.f : fast_exp2(mr[k] - M);
      if (r < nwork) W[r * 16 + hh] = w;
      L += w * lr[k];
    }
    red[256 + gi * 16 + hh] = L;
    __syncthreads();
    L = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) L += red[256 + k * 16 + hh];
    if (gi == 0) {
      stats[hh] = M;
      stats[16 + hh] = (L > 0.f) ? 1.f / L : 0.f;
      if (cidx == 0) lse[t * 16 + hh] = (L > 0.f) ? (M + log2f(L)) : -INFINITY;
    }
    int gsi0[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) gsi0[k] = scan_s[16 + k];
    __syncthreads();
    TS(11);
    for (int s0 = cidx; s0 < NSLICE; s0 += ncomb * SG) {
      const int nsl = min(SG, (NSLICE - 1 - s0) / ncomb + 1);
      const int npairs = nsl * nwork;
      if (s0 != cidx) issue_raw(s0);
      {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        int si = si0, r = r0, j = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int p = 8 * rg + k;
          if (p < npairs) {
            const float w = W[pr[k] * 16 + hh2];
            const float2 a = unpack_bf16(raw[k].x), b = unpack_bf16(raw[k].y);
            acc.x += w * a.x; acc.y += w * a.y; acc.z += w * b.x; acc.w += w * b.y;
          }
          if (++r == nwork) {  // end of a run (slice boundary): flush
            r = 0;
            if (p < npairs) rbuf[(rg * RPG + j) * 32 + l32] = acc;
            acc = make_float4(0.f, 0.f, 0.f, 0.f);
            ++j; ++si;
          }
        }
        if (r != 0 && 8 * rg < npairs) rbuf[(rg * RPG + j) * 32 + l32] = acc;  // partial last run
      }
      __syncthreads();
      for (int si = rg; si < nsl; si += 8) {
        const int rg_lo = (si * nwork) >> 3, rg_hi = ((si + 1) * nwork - 1) >> 3;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r2 = rg_lo; r2 <= rg_hi; ++r2) {
          const int j = si - gsi0[r2];
          const float4 v = rbuf[(r2 * RPG + j) * 32 + l32];
          acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        const float inv = stats[16 + hh2];
        const int sl = s0 + si * ncomb;
        const int d = 8 * sl + 4 * half;
        uint2 pk;
        pk.x = pack_bf16(acc.x * inv, acc.y * inv);
        pk.y = pack_bf16(acc.z * inv, acc.w * inv);
        *reinterpret_cast<uint2*>(out + ((size_t)t * 16 + hh2) * 512 + d) = pk;
      }
      __syncthreads();
    }
  }
  TS(8);
}

// workspace layout for a capacity of `cap` tokens (depends only on cap):
//   ticket [64] uint | flags [cap][64] int | part_ml [cap][64][32] f32 | part_o [cap][64][16][512] bf16
static inline size_t ws_ctrl_bytes() { return 256; }
static inline size_t ws_flag_bytes(int cap) { return (size_t)cap * MAX_SPLIT * sizeof(int); }
extern "C" size_t dsa_decode_workspace_bytes(int cap) {
  return ws_ctrl_bytes() + ws_flag_bytes(cap) + (size_t)cap * MAX_SPLIT * 32 * sizeof(float) +
         (size_t)cap * MAX_SPLIT * 16 * 512 * sizeof(uint16_t);
}

template <int NB, bool USE_TICKET, int MINB>
static cudaError_t launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe, const int* sidx,
                          void* out, float* lse, int topk, int num_kv, float scale_log2, int nsplit, int T,
                          unsigned int ticket_base, int epoch, Workspace ws, cudaStream_t stream, long long* dbg) {
  dsa_decode_kernel<NB, USE_TICKET, MINB><<<T * nsplit, NTHREADS, 0, stream>>>(
      (const __nv_bfloat16*)q_nope, (const __nv_bfloat16*)q_pe, (const __nv_bfloat16*)ckv, (const __nv_bfloat16*)kpe,
      sidx, (__nv_bfloat16*)out, lse, topk, num_kv, scale_log2, nsplit, ticket_base, epoch, ws, dbg);
  return cudaGetLastError();
}

// `state` (host-side, per workspace): [0] epoch, [1] ticket base (cumulative grid size of ticket launches)
extern "C" int dsa_decode_launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe,
                                 const int* sidx, void* out, float* lse, int T, int topk, int num_kv,
                                 float sm_scale, void* stream, void* workspace, int ws_cap, int num_sms,
                                 int force_nb, unsigned int* state, long long* dbg) {
  if (T <= 0) return 0;
  if (topk > MAX_TOPK || T > ws_cap) return -2;
  const float scale_log2 = sm_scale * 1.4426950408889634f;
  cudaStream_t st = (cudaStream_t)stream;
  Workspace ws;
  ws.ticket = (unsigned int*)workspace;
  ws.flags = (int*)((char*)workspace + ws_ctrl_bytes());
  ws.part_ml = (float*)((char*)workspace + ws_ctrl_bytes() + ws_flag_bytes(ws_cap));
  ws.part_o = (uint16_t*)(ws.part_ml + (size_t)ws_cap * MAX_SPLIT * 32);
  // 64-row units.  Three builds of the same kernel:
  //   grid <= #SMs        : 1 CTA/SM (176 registers), every CTA of a token co-resident, combine helpers
  //   grid <= 2 #SMs      : 2 CTAs/SM (register cap 128), still fully co-resident, no tickets
  //   larger grids        : 1 CTA/SM with dynamic tickets (token order == start order)
  const int nb = 2;
  const int rows = 32 * nb;
  const int nsplit = (topk + rows - 1) / rows;
  const int grid = T * nsplit;
  state[0] += 1u;
  if (state[0] == 0u) state[0] = 1u;  // epoch 0 is the "never published" value
  const int epoch = (int)state[0];
  cudaError_t e;
  if (grid <= num_sms) {
    e = launch<2, false, 1>(q_nope, q_pe, ckv, kpe, sidx, out, lse, topk, num_kv, scale_log2, nsplit, T, 0u, epoch, ws, st, dbg);
  } else if (grid <= 2 * num_sms) {
    e = launch<2, false, 2>(q_nope, q_pe, ckv, kpe, sidx, out, lse, topk, num_kv, scale_log2, nsplit, T, 0u, epoch, ws, st, dbg);
  } else {
    const unsigned int ticket_base = state[1];
    state[1] += (unsigned int)grid;
    e = launch<2, true, 1>(q_nope, q_pe, ckv, kpe, sidx, out, lse, topk, num_kv, scale_log2, nsplit, T, ticket_base, epoch, ws, st, dbg);
  }
  return (int)e;
}
