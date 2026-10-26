// Gated Delta Net prefill (qk4_v8_d128, k-last state) for Blackwell sm_100a.
// Chunked WY formulation; three kernels: prep (parallel), chain (sequential over
// chunks), output (parallel). All MMAs on tcgen05 with TMEM accumulators.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "gdn_ptx.cuh"

using namespace gdn;
typedef __nv_bfloat16 bf16;

namespace {

constexpr int C = 64;    // chunk length (tokens)
constexpr int D = 128;   // head dim
constexpr int HQ = 4;
constexpr int HV = 8;
constexpr int NTHREADS = 128;

struct Params {
  const bf16* q; const bf16* k; const bf16* v;
  const float* state;      // [N][HV][D][D] or null
  const float* A_log; const bf16* a; const float* dt_bias; const bf16* b;
  const int64_t* cu_seqlens;
  int num_seqs; int total; float scale; int dbg; unsigned long long* ts;
  bf16* out; float* new_state;
  bf16* Tk; bf16* Tv; bf16* NT; bf16* BT; bf16* Sc; float* gamma;
  bf16* NP; bf16* BP; float* gamma_p;
  bf16* Mp; bf16* PUp;     // K3a1 outputs: M, PU (same layout)
  bf16* Up; float* Gs;     // K1 outputs for K3: U = Tv V (bf16 [chunk][head][C][D]), G cumsum (fp32 [chunk][head][C])
  uint32_t epoch; uint64_t* fK1; uint64_t* fNP; uint64_t* fBP; uint64_t* fS; uint64_t* fM;   // dependency flags (epoch<<32 | payload)
};

// ---------------------------------------------------------------------------
// cu_seqlens cache: copied to smem (int32) once per CTA so index decoding never
// waits on dependent global loads.  Falls back to global for > MAX_CU-1 sequences.
// ---------------------------------------------------------------------------
constexpr int MAX_CU = 257;
struct CuView {
  const int* s;        // smem copy (valid if use_smem)
  const int64_t* g;    // global
  bool use_smem;
  __device__ __forceinline__ int operator[](int i) const { return use_smem ? s[i] : (int)g[i]; }
};
__device__ __forceinline__ CuView load_cu(int* s_cu, const int64_t* g, int num_seqs) {
  CuView v; v.s = s_cu; v.g = g; v.use_smem = (num_seqs + 1 <= MAX_CU);
  if (v.use_smem) {
    for (int i = threadIdx.x; i <= num_seqs; i += blockDim.x) s_cu[i] = (int)g[i];
  }
  __syncthreads();
  return v;
}

// ---------------------------------------------------------------------------
// chunk decoding (no host sync): global chunk id -> (seq, t0, len, first, last)
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool decode_chunk(const CuView& cu, int num_seqs, int c, int& seq, int& t0, int& len,
                                             bool& first, bool& last) {
  int acc = 0;
  for (int s = 0; s < num_seqs; ++s) {
    int st = cu[s], en = cu[s + 1];
    int L = en - st;
    int nc = (L + C - 1) / C;
    if (c < acc + nc) {
      int local = c - acc;
      seq = s; t0 = st + local * C; len = min(C, L - local * C);
      first = (local == 0); last = (local == nc - 1);
      return true;
    }
    acc += nc;
  }
  return false;
}

// gates for one chunk of head h: sg (log gate), sbeta, sG (inclusive cumsum). 128 threads.
// sa / sb: the chunk's rows of a and b ([64][HV] bf16, TMA-loaded into smem).
__device__ __forceinline__ void compute_gates(const Params& p, int h, int len, const bf16* sa, const bf16* sb, float* sg, float* sbeta, float* sG) {
  const int tid = threadIdx.x;
  if (tid < C) {
    float g = 0.f, beta = 0.f;
    if (tid < len) {
      float x = __bfloat162float(sa[tid * HV + h]) + p.dt_bias[h];
      float sp = x > 20.f ? x : log1pf(__expf(x));
      g = -__expf(p.A_log[h]) * sp;
      float bb = __bfloat162float(sb[tid * HV + h]);
      beta = 1.f / (1.f + __expf(-bb));
    }
    sg[tid] = g; sbeta[tid] = beta;
  }
  __syncthreads();
  if (tid < 32) {
    float v0 = sg[2 * tid], v1 = v0 + sg[2 * tid + 1];
    float x = v1;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float y = __shfl_up_sync(0xffffffff, x, o);
      if (tid >= o) x += y;
    }
    float excl = x - v1;
    sG[2 * tid] = excl + v0;
    sG[2 * tid + 1] = excl + v1;
  }
  __syncthreads();
}

// Write 8 bf16 (packed in 4 uint32) to a SW128 tile row: tile base (1024B aligned), row r, element column e (multiple of 8)
__device__ __forceinline__ void st_sw128_16B(uint8_t* tile, int r, int e, uint4 v) {
  uint32_t chunk = (e >> 3) ^ (r & 7);
  *reinterpret_cast<uint4*>(tile + r * 128 + (chunk << 4)) = v;
}
__device__ __forceinline__ uint4 ld_sw128_16B(const uint8_t* tile, int r, int e) {
  uint32_t chunk = (e >> 3) ^ (r & 7);
  return *reinterpret_cast<const uint4*>(tile + r * 128 + (chunk << 4));
}

// K-major SW128 descriptor for a tile stored as 64-column blocks of `rows` rows (block stride rows*128 B), k-step ks (16 elems)
__device__ __forceinline__ uint64_t desc_kmajor(uint32_t base, int rows, int ks) {
  return make_smem_desc(base + (ks >> 2) * rows * 128 + (ks & 3) * 32, 16, 1024, SW_128B);
}
// MN-major SW128 descriptor: tile stored [k' rows][MN cols] as 64-col blocks with block stride `kp_rows*128` bytes
__device__ __forceinline__ uint64_t desc_mnmajor(uint32_t base, int kp_rows, int ks) {
  return make_smem_desc(base + ks * 16 * 128, kp_rows * 128, 1024, SW_128B);
}

// ===========================================================================
// K1: per (chunk, head) prep.  Produces W, U (bf16), N^T (bf16), B^T (fp32), gamma.
// ===========================================================================
// smem: K 16K | V 16K | A fp32 [64][65] 16.6K (later: Tk, Tv bf16 overlay; later: Ud overlay) | T fp32 [64][65] 16.6K (later: Wd overlay) | P 4K | gates 1K
constexpr int K1_LD = 68;   // padded fp32 row stride (16B aligned rows, 2-way bank spread)
constexpr int K1_A_BYTES = 64 * K1_LD * 4;   // 17408
constexpr int K1_A_OFF = 32768, K1_T_OFF = K1_A_OFF + K1_A_BYTES, K1_P_OFF = K1_T_OFF + K1_A_BYTES, K1_G_OFF = K1_P_OFF + 4096;
constexpr int K1_SMEM = K1_G_OFF + 2048 + 1024 /*align*/;

constexpr int K1_STORE_GROUPS = 6;   // bulk async-groups committed per item: U, W, N^T x2 halves, B^T x2 halves
__global__ void __launch_bounds__(NTHREADS, 3) k1_prep(const Params p, const __grid_constant__ CUtensorMap tm_k,
                                                        const __grid_constant__ CUtensorMap tm_v,
                                                        const __grid_constant__ CUtensorMap tm_tk,
                                                        const __grid_constant__ CUtensorMap tm_up,
                                                        const __grid_constant__ CUtensorMap tm_nt,
                                                        const __grid_constant__ CUtensorMap tm_bt,
                                                        const __grid_constant__ CUtensorMap tm_a,
                                                        const __grid_constant__ CUtensorMap tm_b, const int n_items) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sK = smem;                 // 2 blocks [64][64] bf16
  uint8_t* sV = sK + 16384;
  float* sA = reinterpret_cast<float*>(smem + K1_A_OFF);   // [64][65] (padded rows)
  float* sT = reinterpret_cast<float*>(smem + K1_T_OFF);   // [64][65]
  float* sP = reinterpret_cast<float*>(smem + K1_P_OFF);   // [32][32] solve scratch; before the solve: a / b rows of the chunk
  const bf16* sAa = reinterpret_cast<const bf16*>(smem + K1_P_OFF);
  const bf16* sAb = reinterpret_cast<const bf16*>(smem + K1_P_OFF + 1024);
  uint8_t* sTk = smem + K1_A_OFF;     // overlays A (dead after T10)
  uint8_t* sTv = sTk + 8192;
  uint8_t* sUd = smem + K1_T_OFF;     // decayed U: overlays T (dead after Tk/Tv)  [64 t][128 n] as 2 blocks
  uint8_t* sWd = smem + K1_A_OFF;     // decayed W: overlays Tk/Tv (dead after both W/U MMAs)
  float* sg = reinterpret_cast<float*>(smem + K1_G_OFF);
  float* sbeta = sg + 64;
  float* sG = sbeta + 64;
  float* sSk = sG + 64;      // beta_j * exp(G_j)
  float* sSv = sSk + 64;     // beta_j
  __shared__ __align__(8) uint64_t bar_tma;
  __shared__ __align__(8) uint64_t bar_ab;
  __shared__ __align__(8) uint64_t bar_mma;
  __shared__ uint32_t tmem_slot;

  __shared__ int s_cu[MAX_CU];
  if (p.ts && threadIdx.x == 0 && blockIdx.x == 0) p.ts[2046] = globaltimer_ns();   // ~K1 start
  if (threadIdx.x == 0) {
    tma_prefetch_desc(&tm_k); tma_prefetch_desc(&tm_v); tma_prefetch_desc(&tm_tk); tma_prefetch_desc(&tm_up); tma_prefetch_desc(&tm_nt); tma_prefetch_desc(&tm_bt);
    tma_prefetch_desc(&tm_a); tma_prefetch_desc(&tm_b);
  }
  const CuView cu = load_cu(s_cu, p.cu_seqlens, p.num_seqs);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) { mbar_init(&bar_tma, 1); mbar_init(&bar_ab, 1); mbar_init(&bar_mma, 1); fence_mbar_init(); }
  auto issue_ab = [&](int t0_) {   // a / b rows of a chunk -> P scratch (bar_ab)
    mbar_arrive_expect_tx(&bar_ab, 2048);
    tma_load_2d(smem + K1_P_OFF, &tm_a, &bar_ab, 0, t0_);
    tma_load_2d(smem + K1_P_OFF + 1024, &tm_b, &bar_ab, 0, t0_);
  };
  auto next_item = [&](int item, int& t0_, int& h_) {   // this CTA's next valid item (t0, head), or false
    for (int nx = item + gridDim.x; nx < n_items; nx += gridDim.x) {
      int c_ = nx / HV, seq_, len_; bool f_, l_;
      if (decode_chunk(cu, p.num_seqs, c_, seq_, t0_, len_, f_, l_)) { h_ = nx % HV; return true; }
    }
    return false;
  };
  auto next_t0 = [&](int item) { int t0_, h_; return next_item(item, t0_, h_) ? t0_ : -1; };
  if (warp == 0) { tmem_alloc(&tmem_slot, 128); tmem_relinquish(); }
  __syncthreads();
  // (no griddepcontrol.wait: K2 of this call is launched first in plain stream order, after the previous call completed)
  griddep_launch_dependents();
  const uint32_t tmem = tmem_slot;
  // all accumulators share the 128 columns; each is read back before the next MMA is issued
  const uint32_t tmem_A = tmem, tmem_W = tmem, tmem_U = tmem, tmem_N = tmem, tmem_B = tmem;
  int mma_phase = 0, tma_phase = 0, ab_phase = 0;
  // Persistent over (chunk, head) items (head fastest).  The flag of an item is set one item later, after its bulk
  // stores have completed (wait_group), by a thread that issued no bulk stores (a release fence drains the issuing
  // thread's in-flight stores).  The a / b gate rows of the next item are prefetched after the solve.
  int prev_flag = -1; float prev_gamma = 0.f;
  {
    int c_ = blockIdx.x / HV, seq_, t0_, len_; bool f_, l_;
    if (tid == 0 && blockIdx.x < n_items && decode_chunk(cu, p.num_seqs, c_, seq_, t0_, len_, f_, l_)) issue_ab(t0_);
  }
  for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
  const int c = item / HV, h = item % HV, hq = h / (HV / HQ);
  int seq, t0, len; bool first, last;
  if (!decode_chunk(cu, p.num_seqs, c, seq, t0, len, first, last)) continue;

  unsigned long long k1t[12]; int k1n = 0;
#define K1_MARK() do { if (tid == 0) k1t[k1n] = clock64(); k1n++; } while (0)
  K1_MARK();
  if (tid == 0 && prev_flag < 0) {   // first item: K, V now (later items: V and K are prefetched during the previous item)
    mbar_arrive_expect_tx(&bar_tma, 32768);
    for (int kb = 0; kb < 2; ++kb) {
      tma_load_2d(sK + kb * 8192, &tm_k, &bar_tma, hq * D + kb * 64, t0);
      tma_load_2d(sV + kb * 8192, &tm_v, &bar_tma, h * D + kb * 64, t0);
    }
  }
  mbar_wait_cta(&bar_ab, ab_phase & 1); ab_phase++;
  compute_gates(p, h, len, sAa, sAb, sg, sbeta, sG);   // overlaps the K/V TMA loads
  if (tid < C) p.Gs[(size_t)(c * HV + h) * C + tid] = sG[tid];   // for K3 (ordered before this item's flag by the barriers)
  K1_MARK();
  mbar_wait_cta(&bar_tma, tma_phase & 1); tma_phase++;
  K1_MARK();

  // ---- A = K K^T  (M=64, N=64, K'=128)
  if (tid == 0) {
    const uint32_t idesc = make_idesc_bf16(64, 64, 0, 0);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks)
      umma_bf16_ss(tmem_A, desc_kmajor(smem_u32(sK), 64, ks), desc_kmajor(smem_u32(sK), 64, ks), idesc, ks > 0);
    umma_commit(&bar_mma);
  }
  mbar_wait_cta(&bar_mma, mma_phase & 1); mma_phase++;
  tcgen05_fence_after();
  K1_MARK();
  {
    // M=64 layout: row i = 16*warp + lane (lane < 16)
    uint32_t rA[64];
    const int i = 16 * warp + lane;
    const uint32_t ta = tmem_A + ((uint32_t)(32 * warp) << 16);
    tmem_ld_x32(ta, rA);
    tmem_ld_x32(ta + 32, rA + 32);
    tmem_ld_wait();
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const uint32_t* r = rA + 32 * q;
      if (lane < 16) {
        const float bi = sbeta[i], Gi = sG[i];
#pragma unroll
        for (int j4 = 0; j4 < 8; ++j4) {
          const int j = 32 * q + 4 * j4;
          float4 g = *reinterpret_cast<const float4*>(sG + j);
          float4 o;
          o.x = (j < i) ? bi * __expf(Gi - g.x) * __uint_as_float(r[4 * j4]) : 0.f;
          o.y = (j + 1 < i) ? bi * __expf(Gi - g.y) * __uint_as_float(r[4 * j4 + 1]) : 0.f;
          o.z = (j + 2 < i) ? bi * __expf(Gi - g.z) * __uint_as_float(r[4 * j4 + 2]) : 0.f;
          o.w = (j + 3 < i) ? bi * __expf(Gi - g.w) * __uint_as_float(r[4 * j4 + 3]) : 0.f;
          *reinterpret_cast<float4*>(sA + i * K1_LD + j) = o;
        }
      }
    }
  }
  __syncthreads();
  K1_MARK();

  // ---- T = (I + A)^{-1} on 16x16 blocks: diagonal blocks by forward substitution (one warp each),
  // off-diagonal blocks T_ij = -T_ii (sum_{j<=k<i} A_ik T_kj) with tf32 mma.sync (levels d = i-j = 1, 2, 3)
  {
    const int b = warp, cc = lane;
    if (lane < 16) {
      const float* Ab = sA + (16 * b) * K1_LD + 16 * b;
      float t[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float sres = (i == cc) ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < i; ++j) sres -= Ab[i * K1_LD + j] * t[j];
        t[i] = sres;
        sT[(16 * b + i) * K1_LD + 16 * b + cc] = sres;
      }
    } else {
      // zero the 6 upper blocks (i < j): 6 x 16 rows x 16 cols = 96 float4 rows of 4 -> 64 threads x 6 float4
      const int idx = tid - 16 * (warp + 1);   // 0..63 across the four warps' upper lanes: warp*16 + (lane-16)
      const int u = warp * 16 + (lane - 16);
#pragma unroll
      for (int m = 0; m < 6; ++m) {
        const int f = u + 64 * m;            // 0..383 float4 index over the 6 blocks (16 rows x 4 float4 each)
        const int blk = f >> 6, rem = f & 63;
        const int bi = (blk == 0) ? 0 : (blk == 1) ? 0 : (blk == 2) ? 1 : (blk == 3) ? 0 : (blk == 4) ? 1 : 2;
        const int bj = (blk == 0) ? 1 : (blk == 1) ? 2 : (blk == 2) ? 2 : (blk == 3) ? 3 : (blk == 4) ? 3 : 3;
        const int row = rem >> 2, c4 = rem & 3;
        *reinterpret_cast<float4*>(sT + (16 * bi + row) * K1_LD + 16 * bj + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      (void)idx;
      if (lane < 32) {
        const int e = u;   // 0..63: column scales
        sSk[e] = sbeta[e] * __expf(sG[e]);
        sSv[e] = sbeta[e];
      }
    }
  }
  __syncthreads();
  K1_MARK();
  {
    const int g = lane >> 2, t4 = lane & 3;
    float* scratch = sP + warp * 256;   // 16x16 fp32 per warp
    // P(acc) += X[16x16] * Y[16x16] (tf32 mma, fp32 accumulate)
    auto blockmm = [&](float (&acc)[2][4], const float* X, const float* Y) {
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t af[4];
        af[0] = f32_to_tf32(X[g * K1_LD + 8 * kk + t4]);
        af[1] = f32_to_tf32(X[(g + 8) * K1_LD + 8 * kk + t4]);
        af[2] = f32_to_tf32(X[g * K1_LD + 8 * kk + t4 + 4]);
        af[3] = f32_to_tf32(X[(g + 8) * K1_LD + 8 * kk + t4 + 4]);
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
          uint32_t bf[2];
          bf[0] = f32_to_tf32(Y[(8 * kk + t4) * K1_LD + 8 * nn + g]);
          bf[1] = f32_to_tf32(Y[(8 * kk + t4 + 4) * K1_LD + 8 * nn + g]);
          mma_tf32_16x8x8(acc[nn], af, bf);
        }
      }
    };
    auto store_acc = [&](const float (&acc)[2][4], float* Yout, int ld, float sign) {
#pragma unroll
      for (int nn = 0; nn < 2; ++nn) {
        Yout[g * ld + 8 * nn + 2 * t4] = sign * acc[nn][0];
        Yout[g * ld + 8 * nn + 2 * t4 + 1] = sign * acc[nn][1];
        Yout[(g + 8) * ld + 8 * nn + 2 * t4] = sign * acc[nn][2];
        Yout[(g + 8) * ld + 8 * nn + 2 * t4 + 1] = sign * acc[nn][3];
      }
    };
    auto Ablk = [&](int i, int j) { return sA + (16 * i) * K1_LD + 16 * j; };
    auto Tblk = [&](int i, int j) { return sT + (16 * i) * K1_LD + 16 * j; };
    // level d: warps 0..3-d handle (i, j) = (w + d, w)
#pragma unroll 1
    for (int d = 1; d <= 3; ++d) {
      if (warp + d < 4) {
        const int i = warp + d, j = warp;
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int k = j; k < i; ++k) blockmm(acc, Ablk(i, k), Tblk(k, j));
        store_acc(acc, scratch, 16, 1.f);
        __syncwarp();
        float acc2[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        // Q = T_ii * P  (P in scratch with ld 16)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          uint32_t af[4];
          const float* X = Tblk(i, i);
          af[0] = f32_to_tf32(X[g * K1_LD + 8 * kk + t4]);
          af[1] = f32_to_tf32(X[(g + 8) * K1_LD + 8 * kk + t4]);
          af[2] = f32_to_tf32(X[g * K1_LD + 8 * kk + t4 + 4]);
          af[3] = f32_to_tf32(X[(g + 8) * K1_LD + 8 * kk + t4 + 4]);
#pragma unroll
          for (int nn = 0; nn < 2; ++nn) {
            uint32_t bf[2];
            bf[0] = f32_to_tf32(scratch[(8 * kk + t4) * 16 + 8 * nn + g]);
            bf[1] = f32_to_tf32(scratch[(8 * kk + t4 + 4) * 16 + 8 * nn + g]);
            mma_tf32_16x8x8(acc2[nn], af, bf);
          }
        }
        store_acc(acc2, Tblk(i, j), K1_LD, -1.f);
      }
      __syncthreads();
    }
  }
  K1_MARK();
  if (tid == 0) { const int nt0 = next_t0(item); if (nt0 >= 0) issue_ab(nt0); }   // solve scratch is free: prefetch a / b
  // Tk = T diag(beta e^G), Tv = T diag(beta) -> bf16 K-major SW128 [64][64]
  {
    const int i = tid >> 1, half = tid & 1;
#pragma unroll
    for (int e8 = 0; e8 < 4; ++e8) {
      int j0 = half * 32 + e8 * 8;
      float4 t0 = *reinterpret_cast<const float4*>(sT + i * K1_LD + j0);
      float4 t1 = *reinterpret_cast<const float4*>(sT + i * K1_LD + j0 + 4);
      float4 k0 = *reinterpret_cast<const float4*>(sSk + j0), k1 = *reinterpret_cast<const float4*>(sSk + j0 + 4);
      float4 v0 = *reinterpret_cast<const float4*>(sSv + j0), v1 = *reinterpret_cast<const float4*>(sSv + j0 + 4);
      uint4 wk, wv;
      wk.x = pack_bf16x2(t0.x * k0.x, t0.y * k0.y); wk.y = pack_bf16x2(t0.z * k0.z, t0.w * k0.w);
      wk.z = pack_bf16x2(t1.x * k1.x, t1.y * k1.y); wk.w = pack_bf16x2(t1.z * k1.z, t1.w * k1.w);
      wv.x = pack_bf16x2(t0.x * v0.x, t0.y * v0.y); wv.y = pack_bf16x2(t0.z * v0.z, t0.w * v0.w);
      wv.z = pack_bf16x2(t1.x * v1.x, t1.y * v1.y); wv.w = pack_bf16x2(t1.z * v1.z, t1.w * v1.w);
      st_sw128_16B(sTk, i, j0, wk);
      st_sw128_16B(sTv, i, j0, wv);
    }
  }
  fence_proxy_async_smem();
  __syncthreads();
  if (tid == 0) { tma_store_2d(&tm_tk, sTk, 0, (c * HV + h) * C); tma_store_commit(); }   // Tk for K3 (W = Tk K there)
  K1_MARK();

  // ---- U = Tv V (first: V is then dead), W = Tk K  (M=64, N=128, K'=64); B operands MN-major (token rows)
  // Each readback writes the decayed bf16 copy (sUd over T, sWd over Tk/Tv) for the N/B MMAs and the plain bf16 copy
  // into the dead V tile, from where it is TMA-stored for K3 (W, U workspace).
  {
    const int t = 16 * warp + lane;
    const float dec = (lane < 16) ? __expf(sG[C - 1] - sG[t]) : 0.f;
    const uint32_t idesc = make_idesc_bf16(64, 128, 0, 1);
    for (int which = 0; which < 2; ++which) {
      if (tid == 0) {
        const uint32_t a_base = smem_u32(which == 0 ? sTv : sTk);
        const uint32_t b_base = smem_u32(which == 0 ? sV : sK);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          umma_bf16_ss(tmem_W, desc_kmajor(a_base, 64, ks), desc_mnmajor(b_base, 64, ks), idesc, ks > 0);
        umma_commit(&bar_mma);
      }
      mbar_wait_cta(&bar_mma, mma_phase & 1); mma_phase++;
      tcgen05_fence_after();
      if (which == 1) {
        if (tid == 0) tma_store_wait<1>();   // the Tk store has finished reading its tile (Wd overlays Tk); the U store may still read V
        __syncthreads();
      }
      uint8_t* sXd = which == 0 ? sUd : sWd;
      const uint32_t tw = tmem_W + ((uint32_t)(32 * warp) << 16);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t rw[32];
        tmem_ld_x32(tw + 32 * q, rw);
        tmem_ld_wait();
        if (lane < 16) {
          uint8_t* bXd = sXd + (q >> 1) * 8192;
#pragma unroll
          for (int e8 = 0; e8 < 4; ++e8) {
            uint32_t pwd[4], pw[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              float w0 = __uint_as_float(rw[e8 * 8 + 2 * m]), w1 = __uint_as_float(rw[e8 * 8 + 2 * m + 1]);
              pwd[m] = pack_bf16x2(w0 * dec, w1 * dec);
              pw[m] = pack_bf16x2(w0, w1);
            }
            int n0 = 32 * q + e8 * 8;
            st_sw128_16B(bXd, t, n0 & 63, make_uint4(pwd[0], pwd[1], pwd[2], pwd[3]));
            if (which == 0) st_sw128_16B(sV + (q >> 1) * 8192, t, n0 & 63, make_uint4(pw[0], pw[1], pw[2], pw[3]));   // plain U -> V tile
          }
        }
      }
      fence_proxy_async_smem();
      tcgen05_fence_before();
      __syncthreads();   // TMEM columns free for the next MMA
      if (tid == 0 && which == 0) {
        for (int kb = 0; kb < 2; ++kb) tma_store_2d(&tm_up, sV + kb * 8192, kb * 64, (c * HV + h) * C);   // U for K3
        tma_store_commit();
      }
      K1_MARK();
    }
  }
  fence_proxy_async_smem();
  __syncthreads();

  // ---- N^T = -Wd^T K, B^T = Ud^T K   (M=128, N=128, K'=64); A, B MN-major
  for (int which = 0; which < 2; ++which) {
    if (tid == 0) {
      const uint32_t idesc = make_idesc_bf16(128, 128, 1, 1, which == 0 ? 1 : 0, 0);
      const uint32_t a_base = smem_u32(which == 0 ? sWd : sUd);
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
        umma_bf16_ss(tmem_N, desc_mnmajor(a_base, 64, ks), desc_mnmajor(smem_u32(sK), 64, ks), idesc, ks > 0);
      umma_commit(&bar_mma);
    }
    mbar_wait_cta(&bar_mma, mma_phase & 1); mma_phase++;
    tcgen05_fence_after();
    {
      const int n = 32 * warp + lane;  // M=128 layout
      const uint32_t tn = tmem_N + ((uint32_t)(32 * warp) << 16);
      const CUtensorMap* tm = (which == 0) ? &tm_nt : &tm_bt;
      uint8_t* stg = sWd;  // [128 rows][64 cols] bf16 SW128 staging (decayed W is dead after the N MMA)
      if (which == 1 && tid == 0) {   // Kd consumed by both MMAs: prefetch the next item's K
        int nt0, nh;
        if (next_item(item, nt0, nh)) { const int nhq = nh / (HV / HQ); for (int kb = 0; kb < 2; ++kb) tma_load_2d(sK + kb * 8192, &tm_k, &bar_tma, nhq * D + kb * 64, nt0); }
      }
      for (int half = 0; half < 2; ++half) {
        if (tid == 0) {
          tma_store_wait<0>();   // previous half's store done reading the staging tile (and, at the first wait, the U store: V is free)
          if (which == 0 && half == 0) {   // prefetch the next item's V into the free V tile (K follows after the B MMA)
            int nt0, nh;
            if (next_item(item, nt0, nh)) {
              mbar_arrive_expect_tx(&bar_tma, 32768);
              for (int kb = 0; kb < 2; ++kb) tma_load_2d(sV + kb * 8192, &tm_v, &bar_tma, nh * D + kb * 64, nt0);
            }
          }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          uint32_t rn[32];
          tmem_ld_x32(tn + 64 * half + 32 * q, rn);
          tmem_ld_wait();
#pragma unroll
          for (int e8 = 0; e8 < 4; ++e8) {
            uint32_t pn[4];
#pragma unroll
            for (int m = 0; m < 4; ++m)
              pn[m] = pack_bf16x2(__uint_as_float(rn[e8 * 8 + 2 * m]), __uint_as_float(rn[e8 * 8 + 2 * m + 1]));
            st_sw128_16B(stg, n, 32 * q + e8 * 8, make_uint4(pn[0], pn[1], pn[2], pn[3]));
          }
        }
        fence_proxy_async_smem();
        __syncthreads();
        if (tid == 0) {
          tma_store_2d(tm, stg, 64 * half, (c * HV + h) * D);
          tma_store_commit();
        }
      }
    }
    tcgen05_fence_before();
    __syncthreads();
    K1_MARK();
  }
  if (tid == 0) {
    p.gamma[c * HV + h] = __expf(sG[C - 1]);
    if (prev_flag >= 0) tma_store_wait_all<K1_STORE_GROUPS>();   // the previous item's stores are complete
  }
  __syncthreads();
  if (tid == 32 && prev_flag >= 0) flag_set(&p.fK1[prev_flag], p.epoch, prev_gamma);   // previous item's Tk, U, N^T, B^T ready
  prev_flag = c * HV + h; prev_gamma = __expf(sG[C - 1]);   // (all threads: thread 0 uses it for the loads/waits)
  if (p.ts && tid == 0) {
    for (int i = 1; i < k1n; ++i) atomicAdd(&p.ts[16 + i - 1], k1t[i] - k1t[i - 1]);
    atomicAdd(&p.ts[31], 1ull);
    atomicMax(&p.ts[2047], globaltimer_ns());   // K1 end
  }
#undef K1_MARK
  }
  if (tid == 0) tma_store_wait_all<0>();
  __syncthreads();
  if (tid == 32 && prev_flag >= 0) flag_set(&p.fK1[prev_flag], p.epoch, prev_gamma);
  tcgen05_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tmem, 128);
}

// ===========================================================================
// Chunk groups: the chain advances over groups of G chunks using prefix maps.
// ===========================================================================
constexpr int G = 12;

__device__ __forceinline__ int seq_chunk_base(const CuView& cu, int s) {
  int acc = 0;
  for (int i = 0; i < s; ++i) {
    int L = cu[i + 1] - cu[i];
    acc += (L + C - 1) / C;
  }
  return acc;
}
// group id -> (seq, first chunk c0 (global), m = chunks in group, first/last group of seq)
__device__ __forceinline__ bool decode_group(const CuView& cu, int num_seqs, int gi, int& seq, int& c0, int& m, bool& first, bool& last) {
  int acc_g = 0, acc_c = 0;
  for (int s = 0; s < num_seqs; ++s) {
    int L = cu[s + 1] - cu[s];
    int nc = (L + C - 1) / C;
    int ng = (nc + G - 1) / G;
    if (gi < acc_g + ng) {
      int lg = gi - acc_g;
      seq = s; c0 = acc_c + lg * G; m = min(G, nc - lg * G);
      first = (lg == 0); last = (lg == ng - 1);
      return true;
    }
    acc_g += ng; acc_c += nc;
  }
  return false;
}

// ===========================================================================
// K15: compose prefix maps inside each group.  PM[c] = map_c o ... o map_c0.  One CTA per (group, head) and part:
//   part 0:  N'^T = NP^T Nc^T + gc NP^T + gp Nc^T      (A = NP^T prefix in TMEM, B = Nc^T slot)
//   part 1:  B'^T = BP^T Nc^T + gc BP^T + Bc^T          (A = BP^T prefix in TMEM, B = Nc^T slot)
// The running prefix lives in TMEM as the bf16 A operand (64 columns) next to the fp32 accumulator (128): TMEM 256.
// The epilogue writes the new prefix to TMEM and in place over the Y tile (Nc^T slot for part 0, Bc^T buffer for
// part 1), from where the storer warp TMA-stores it (prefix of chunk r >= 1); a flagger warp sets the flags once the
// storer reports completion (a release fence by the storing thread would drain its in-flight bulk stores).
// smem: part 0: 2 Nc slots (65 KB); part 1: 2 Nc slots + 2 Bc buffers (129 KB).  <= 53 registers: a CTA shares its SM
// with two K1 CTAs or with K3a1 CTAs.
// ===========================================================================
// part 0 (N chain): 3 Nc slots (97 KB): one 68 KB K3a1 CTA fits next to it (two would stretch the compositions);
// part 1 (B chain): 2 Nc slots + 2 Bc buffers (129 KB): likewise one K3a1 CTA.
constexpr int K15_NSLOT_N = 2, K15_NSLOT_B = 2;
constexpr int K15_SMEM_N = K15_NSLOT_N * 32768 + 1024;
constexpr int K15_SMEM_B = K15_NSLOT_B * 65536 + 1024;
constexpr int K15_SMEM = K15_SMEM_B;

constexpr int K15_CTHREADS = 512;                 // compute threads: 16 warps (4 column groups x 4 lane quadrants)
constexpr int K15_THREADS = K15_CTHREADS + 96;    // + loader warp + storer warp + flagger warp
template <int PART>
__global__ void __launch_bounds__(K15_THREADS, 2) k15_compose(const Params p, const __grid_constant__ CUtensorMap tm_nt,
                                                            const __grid_constant__ CUtensorMap tm_bt,
                                                            const __grid_constant__ CUtensorMap tm_np,
                                                            const __grid_constant__ CUtensorMap tm_bp, const int n_items) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  constexpr bool BP = (PART == 1);
  constexpr int K15_NSLOT = BP ? K15_NSLOT_B : K15_NSLOT_N;
  uint8_t* sNcB = smem;                          // Nc slots
  uint8_t* sBcB = smem + K15_NSLOT * 32768;      // Bc buffers (part 1)
  __shared__ __align__(8) uint64_t bar_full[K15_NSLOT];    // loader -> compute: Nc^T of the chunk landed
  __shared__ __align__(8) uint64_t bar_empty[K15_NSLOT];   // -> loader: Nc slot free
  __shared__ __align__(8) uint64_t bar_bfull[K15_NSLOT];   // loader -> compute: Bc^T landed (part 1)
  __shared__ __align__(8) uint64_t bar_bempty[K15_NSLOT];  // -> loader: Bc buffer free (part 1)
  __shared__ __align__(8) uint64_t bar_ready[K15_NSLOT];   // compute -> storer: prefix r complete in slot/buffer r%2
  __shared__ __align__(8) uint64_t bar_mma;
  __shared__ uint32_t tmem_slot;
  __shared__ int s_nflag;            // storer -> flagger: number of chunks whose prefix stores are complete
  __shared__ float sGc[K15_NSLOT];   // gamma of the chunk in each slot
  __shared__ float sGp[K15_NSLOT];   // prefix gamma of the prefix in each slot (flag payload)
  __shared__ float sGpr[4];          // storer -> flagger: payload ring (indexed by r & 3)
  __shared__ int s_cu[MAX_CU];

  griddep_launch_dependents();   // dependencies are tracked with flags, dependents may launch now
  if (threadIdx.x == 0) { tma_prefetch_desc(&tm_nt); tma_prefetch_desc(&tm_bt); tma_prefetch_desc(&tm_np); tma_prefetch_desc(&tm_bp); }
  if (p.ts && threadIdx.x == 0 && blockIdx.x < 128) p.ts[2048 + 2 * (PART * 128 + blockIdx.x)] = globaltimer_ns();   // [2048, 2560)
  const CuView cu = load_cu(s_cu, p.cu_seqlens, p.num_seqs);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  if (tid == 0) {
    for (int i = 0; i < K15_NSLOT; ++i) {
      mbar_init(&bar_full[i], 1); mbar_init(&bar_empty[i], 1); mbar_init(&bar_bfull[i], 1); mbar_init(&bar_bempty[i], 1);
      mbar_init(&bar_ready[i], 1);
    }
    mbar_init(&bar_mma, 1);
    s_nflag = 0;
    fence_mbar_init();
  }
  if (warp == 0) { tmem_alloc(&tmem_slot, 256); tmem_relinquish(); }
  __syncthreads();
  const uint32_t tmem = tmem_slot;
  const uint32_t tA = tmem, tD = tmem + 64;

  auto decode_item = [&](int item, int& gi, int& h, int& c0, int& m) {
    gi = item / HV; h = item % HV;
    int seq; bool f, l;
    return decode_group(cu, p.num_seqs, gi, seq, c0, m, f, l);
  };
  uint64_t* fX = BP ? p.fBP : p.fNP;

  if (warp == K15_CTHREADS / 32 + 2) {
    // ---- flagger warp
    if (lane == 0) {
      int n_flag = 0;
      for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        int gi, h, c0, m;
        if (!decode_item(item, gi, h, c0, m)) continue;
        for (int r = 1; r < m; ++r) {
          n_flag++;
          while (*reinterpret_cast<volatile int*>(&s_nflag) < n_flag) __nanosleep(100);
          __threadfence_block();
          flag_set(&fX[(c0 + r) * HV + h], p.epoch, sGpr[r & 3]);
        }
      }
    }
    return;
  }
  if (warp == K15_CTHREADS / 32 + 1) {
    // ---- storer warp: prefix of chunk r (r >= 1) -> workspace; release the slot/buffer; report completion (r-2)
    if (lane == 0) {
      int n_ready[K15_NSLOT] = {};
      const CUtensorMap* tm_out = BP ? &tm_bp : &tm_np;
      for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        int gi, h, c0, m;
        if (!decode_item(item, gi, h, c0, m)) continue;
        float gpr[4] = {};
        for (int r = 1; r < m; ++r) {
          const int c = c0 + r, sl = r % K15_NSLOT;
          mbar_wait(&bar_ready[sl], n_ready[sl] & 1); n_ready[sl]++;
          if (p.ts && item == 24) p.ts[1024 + r * 8 + 4] = clock64();
          gpr[r & 3] = sGp[sl];
          const uint8_t* src = (BP ? sBcB : sNcB) + sl * 32768;
          for (int kb = 0; kb < 2; ++kb) tma_store_2d(tm_out, src + kb * 16384, kb * 64, (c * HV + h) * D);
          tma_store_commit();
          tma_store_wait<0>();   // store r done reading its tile -> release
          if (p.ts && item == 24) p.ts[1024 + r * 8 + 6] = clock64();
          mbar_arrive(BP ? &bar_bempty[sl] : &bar_empty[sl]);
          if (r >= 3) {   // stores of r-2 complete (at most two groups pending) -> flagger
            tma_store_wait_all<2>();
            sGpr[(r - 2) & 3] = gpr[(r - 2) & 3];
            __threadfence_block();
            atomicAdd(&s_nflag, 1);
          }
        }
        if (m > 1) {
          tma_store_wait_all<0>();
          for (int r = max(1, m - 2); r < m; ++r) { sGpr[r & 3] = gpr[r & 3]; __threadfence_block(); atomicAdd(&s_nflag, 1); }
        }
      }
    }
    return;
  }
  if (warp == K15_CTHREADS / 32) {
    // ---- loader warp: Nc^T -> slot r%2 (part 1: chunk 0's Nc is not needed), Bc^T -> buffer r%2 (part 1)
    if (lane == 0) {
      int n_load[K15_NSLOT] = {}, n_bload[K15_NSLOT] = {};
      for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        int gi, h, c0, m;
        if (!decode_item(item, gi, h, c0, m)) continue;
        for (int r = 0; r < m; ++r) {
          const int c = c0 + r, sl = r % K15_NSLOT;
          const float gc = flag_wait(&p.fK1[c * HV + h], p.epoch);   // payload = gamma of chunk c
          if (!(BP && r == 0)) {
            if (n_load[sl] > 0) mbar_wait(&bar_empty[sl], (n_load[sl] - 1) & 1);
            n_load[sl]++;
            if (p.ts && item == 24 && r < 16) p.ts[1200 + r * 2] = clock64();
            sGc[sl] = gc;
            mbar_arrive_expect_tx(&bar_full[sl], 32768);
            for (int kb = 0; kb < 2; ++kb) tma_load_2d(sNcB + sl * 32768 + kb * 16384, &tm_nt, &bar_full[sl], kb * 64, (c * HV + h) * D);
            if (p.ts && item == 24 && r < 16) p.ts[1200 + r * 2 + 1] = clock64();
          }
          if (BP) {
            if (n_bload[sl] > 0) mbar_wait(&bar_bempty[sl], (n_bload[sl] - 1) & 1);
            n_bload[sl]++;
            mbar_arrive_expect_tx(&bar_bfull[sl], 32768);
            for (int kb = 0; kb < 2; ++kb) tma_load_2d(sBcB + sl * 32768 + kb * 16384, &tm_bt, &bar_bfull[sl], kb * 64, (c * HV + h) * D);
          }
        }
      }
    }
    return;
  }

  // ---- compute warps (named barrier 1, 512 threads): thread = row v (lane quadrant q4), columns [32cg, 32cg+32)
  const int q4 = warp & 3, cg = warp >> 2;
  const int v = 32 * q4 + lane;
  const uint32_t lane_q = (uint32_t)(32 * q4) << 16;
  const uint32_t tDq = tD + lane_q + 32 * cg;   // this thread's 32 accumulator columns
  const uint32_t tAq = tA + lane_q + 16 * cg;   // this thread's 16 A cells (bf16 pairs)
  int mma_phase = 0;
  int n_full[K15_NSLOT] = {}, n_bfull[K15_NSLOT] = {};
  for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
    int gi, h, c0, m;
    if (!decode_item(item, gi, h, c0, m)) continue;
    const bool trace = p.ts && tid == 0 && item == 24 && BP;
#define K15_MARK(i) do { if (trace) p.ts[1024 + r * 8 + (i)] = clock64(); } while (0)
    if (trace) { p.ts[1023] = clock64(); p.ts[1022] = globaltimer_ns(); }
    float gp;   // prefix gamma (part 0: scaling of Nc and flag payload)
    // ---- chunk 0: prefix = Y_0 -> TMEM A (bf16 pairs are the raw 32-bit words of the bf16 tile)
    {
      const uint8_t* Y;
      if (BP) { mbar_wait_cta_n(&bar_bfull[0], n_bfull[0] & 1, K15_CTHREADS); n_bfull[0]++; Y = sBcB; gp = 1.f; }
      else { mbar_wait_cta_n(&bar_full[0], n_full[0] & 1, K15_CTHREADS); n_full[0]++; Y = sNcB; gp = sGc[0]; }
      const uint8_t* blk = Y + (cg >> 1) * 16384;
      uint32_t w[16];
#pragma unroll
      for (int e8 = 0; e8 < 4; ++e8) {
        uint4 y = ld_sw128_16B(blk, v, (32 * cg + e8 * 8) & 63);
        w[4 * e8] = y.x; w[4 * e8 + 1] = y.y; w[4 * e8 + 2] = y.z; w[4 * e8 + 3] = y.w;
      }
      tmem_st_x16(tAq, w);
      tmem_st_wait();
      tcgen05_fence_before();
      named_sync(1, K15_CTHREADS);
      if (tid == 0) mbar_arrive(BP ? &bar_bempty[0] : &bar_empty[0]);   // chunk 0's tile is never stored
    }
    for (int r = 1; r < m; ++r) {
      const int sl = r % K15_NSLOT;
      uint8_t* sNc = sNcB + sl * 32768;
      uint8_t* Yb = BP ? sBcB + sl * 32768 : sNc;
      K15_MARK(0);
      mbar_wait_cta_n(&bar_full[sl], n_full[sl] & 1, K15_CTHREADS); n_full[sl]++;
      K15_MARK(1);
      if (trace && r == 1) p.ts[1021] = globaltimer_ns();
      const float gc = sGc[sl];
      if (tid == 0) {
        tcgen05_fence_after();
        const uint32_t idesc = make_idesc_bf16(128, 128, 0, 1);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) umma_bf16_ts(tD, tA + ks * 8, desc_mnmajor(smem_u32(sNc), 128, ks), idesc, ks > 0);
        umma_commit(&bar_mma);
      }
      if (BP) { mbar_wait_cta_n(&bar_bfull[sl], n_bfull[sl] & 1, K15_CTHREADS); n_bfull[sl]++; }
      mbar_wait_cta_n(&bar_mma, mma_phase & 1, K15_CTHREADS); mma_phase++;
      tcgen05_fence_after();
      if (BP && tid == 0) mbar_arrive(&bar_empty[sl]);   // part 1: Nc_r consumed by the MMA (never stored)
      K15_MARK(2);
      // ---- epilogue in two 16-column halves: X' = D + gc*Xprev + f*Y -> TMEM A (bf16) and Y tile in place
      const float f = BP ? 1.f : gp;
      uint8_t* blk = Yb + (cg >> 1) * 16384;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint32_t rr[16], pa[8], o[8];
        tmem_ld_x16(tDq + 16 * hh, rr);
        tmem_ld_x8(tAq + 8 * hh, pa);
        tmem_ld_wait();
#pragma unroll
        for (int e8 = 0; e8 < 2; ++e8) {
          const int e = (32 * cg + 16 * hh + e8 * 8) & 63;
          uint4 y = ld_sw128_16B(blk, v, e);
          const uint32_t yu[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
          for (int mm = 0; mm < 4; ++mm) {
            const int i = e8 * 4 + mm;
            const float2 fy = unpack_bf16x2(yu[mm]), fp = unpack_bf16x2(pa[i]);
            o[i] = pack_bf16x2(__uint_as_float(rr[2 * i]) + gc * fp.x + f * fy.x, __uint_as_float(rr[2 * i + 1]) + gc * fp.y + f * fy.y);
          }
          st_sw128_16B(blk, v, e, make_uint4(o[e8 * 4], o[e8 * 4 + 1], o[e8 * 4 + 2], o[e8 * 4 + 3]));
        }
        tmem_st_x8(tAq + 8 * hh, o);
      }
      tmem_st_wait();
      gp *= gc;
      K15_MARK(7);
      fence_proxy_async_smem();
      tcgen05_fence_before();
      named_sync(1, K15_CTHREADS);
      K15_MARK(3);
      if (tid == 0) {
        sGp[sl] = gp;
        mbar_arrive(&bar_ready[sl]);   // prefix r complete in its tile -> storer
        K15_MARK(5);
      }
    }
#undef K15_MARK
  }
  tcgen05_fence_before();
  named_sync(1, K15_CTHREADS);
  if (p.ts && tid == 0 && blockIdx.x < 128) p.ts[2048 + 2 * (PART * 128 + blockIdx.x) + 1] = globaltimer_ns();
  if (warp == 0) tmem_dealloc(tmem, 256);
}

// ===========================================================================
// K2: per (head, sequence) chain over groups.  S^T (fp32, [v][k]) lives in TMEM (2 buffers).
// A producer warp waits for the K15/K1 flags of each group and feeds the stages; the bf16 copy of
// the state entering each group is stored for K3 and flagged (fS) once the store has completed.
// ===========================================================================
constexpr int K2_STAGES = 2;
constexpr int K2_STAGE_BYTES = 32768 /*NP^T bf16*/ + 32768 /*BP^T bf16*/;
constexpr int K2_NSB = 3;   // bf16 state staging buffers (the TMA store of one takes ~1.5-3 us to complete in situ)
constexpr int K2_SMEM = K2_STAGES * K2_STAGE_BYTES + K2_NSB * 32768 /*S bf16 (initial fp32 state staged here)*/ + 1024;

constexpr int K2_CTHREADS = 512;   // 16 compute warps: 4 column groups x 4 lane quadrants
constexpr int K2_THREADS = K2_CTHREADS + 64;   // + producer warp + storer warp
__global__ void __launch_bounds__(K2_THREADS, 1) k2_chain(const Params p, const __grid_constant__ CUtensorMap tm_np,
                                                         const __grid_constant__ CUtensorMap tm_bp,
                                                         const __grid_constant__ CUtensorMap tm_sc,
                                                         const __grid_constant__ CUtensorMap tm_nt,
                                                         const __grid_constant__ CUtensorMap tm_bt,
                                                         const __grid_constant__ CUtensorMap tm_so,
                                                         const __grid_constant__ CUtensorMap tm_si) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sSbuf = smem + K2_STAGES * K2_STAGE_BYTES;   // K2_NSB x [128 v][2 blocks of 64 k] bf16
  __shared__ __align__(8) uint64_t bar_full[K2_STAGES];
  __shared__ __align__(8) uint64_t bar_empty[K2_STAGES];
  __shared__ __align__(8) uint64_t bar_mma;
  __shared__ __align__(8) uint64_t bar_state;
  __shared__ __align__(8) uint64_t bar_sready[K2_NSB];   // compute -> storer: bf16 S of this step is in sS[b]
  __shared__ __align__(8) uint64_t bar_sfree[K2_NSB];    // storer -> compute: sS[b] stored (and flagged)
  __shared__ uint32_t tmem_slot;
  __shared__ float sGam[K2_STAGES];
  __shared__ int s_cu[MAX_CU];
  unsigned long long t_kernel_start = clock64();
  const int h = blockIdx.x, s = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#define K2_TR(i) do { if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[32 + (i)] = clock64(); } while (0)
  K2_TR(0);
  if (p.ts && tid == 0) p.ts[2560 + 2 * (s * HV + h)] = globaltimer_ns();
  griddep_launch_dependents();   // dependencies are tracked with flags
  if (tid == 0) {
    tma_prefetch_desc(&tm_np); tma_prefetch_desc(&tm_bp); tma_prefetch_desc(&tm_sc); tma_prefetch_desc(&tm_nt); tma_prefetch_desc(&tm_bt);
    tma_prefetch_desc(&tm_so); tma_prefetch_desc(&tm_si);
  }
  const CuView cu = load_cu(s_cu, p.cu_seqlens, p.num_seqs);
  K2_TR(1);
  const int L = cu[s + 1] - cu[s];
  const int nc = (L + C - 1) / C;
  const int ng = (nc + G - 1) / G;
  const int cbase = seq_chunk_base(cu, s);
  int g_base = 0;
  for (int i = 0; i < s; ++i) { int Li = cu[i + 1] - cu[i]; g_base += ((Li + C - 1) / C + G - 1) / G; }
  const int q4 = warp & 3;        // TMEM lane quadrant
  const int cg = warp >> 2;       // column group: columns [32cg, 32cg+32)
  const int v = 32 * q4 + lane;   // row of S^T handled by this thread (M=128 layout)
  const uint32_t lane_off = ((uint32_t)(32 * q4) << 16) + 32 * cg;

  if (nc == 0) {
    // empty sequence: new_state = initial state (or zero)
    if (tid < 128) {
      float* gstate_out = p.new_state + ((size_t)(s * HV + h) * D + tid) * D;
      const float* gstate_in = p.state ? p.state + ((size_t)(s * HV + h) * D + tid) * D : nullptr;
#pragma unroll 4
      for (int e4 = 0; e4 < 32; ++e4)
        *reinterpret_cast<float4*>(gstate_out + 4 * e4) = gstate_in ? *reinterpret_cast<const float4*>(gstate_in + 4 * e4) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    return;
  }

  if (tid == 0) {
    for (int i = 0; i < K2_STAGES; ++i) { mbar_init(&bar_full[i], 1); mbar_init(&bar_empty[i], 1); }
    mbar_init(&bar_mma, 1);
    mbar_init(&bar_state, 1);
    for (int i = 0; i < K2_NSB; ++i) { mbar_init(&bar_sready[i], 1); mbar_init(&bar_sfree[i], 1); }
    fence_mbar_init();
  }
  if (warp == 0) { tmem_alloc(&tmem_slot, 256); tmem_relinquish(); }
  __syncthreads();
  K2_TR(2);
  const uint32_t tmem = tmem_slot;

  if (warp == K2_CTHREADS / 32) {
    // ---- producer warp
    if (lane == 0) {
      int uses[K2_STAGES] = {};
      if (p.state) {
        // initial state -> fp32 [128][128] as 4 column blocks (SW128) in the sS region (inputs only: no flag needed)
        uint8_t* dst = sSbuf;
        mbar_arrive_expect_tx(&bar_state, 65536);
        for (int q = 0; q < 4; ++q) tma_load_2d(dst + q * 16384, &tm_si, &bar_state, q * 32, (s * HV + h) * D);
      }
      const bool ptrace = p.ts && h == 0 && s == 0;
      for (int step = 0; step < ng; ++step) {
        const int st = step % K2_STAGES;
        if (ptrace) p.ts[3072 + step * 4 + 0] = globaltimer_ns();
        if (uses[st] > 0) mbar_wait(&bar_empty[st], (uses[st] - 1) & 1);
        uses[st]++;
        if (ptrace) p.ts[3072 + step * 4 + 1] = globaltimer_ns();
        const int m = min(G, nc - step * G);
        const int cl = cbase + step * G + m - 1;   // last chunk of the group: its prefix map is the group map
        // payload of fNP / fK1 = prefix gamma / chunk gamma
        sGam[st] = (m > 1) ? flag_wait2(&p.fNP[cl * HV + h], &p.fBP[cl * HV + h], p.epoch) : flag_wait(&p.fK1[cl * HV + h], p.epoch);
        if (ptrace) p.ts[3072 + step * 4 + 2] = globaltimer_ns();
        const CUtensorMap* mn = (m > 1) ? &tm_np : &tm_nt;
        const CUtensorMap* mb = (m > 1) ? &tm_bp : &tm_bt;
        uint8_t* sN = smem + st * K2_STAGE_BYTES;
        uint8_t* sB = sN + 32768;
        mbar_arrive_expect_tx(&bar_full[st], K2_STAGE_BYTES);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(sN + kb * 16384, mn, &bar_full[st], kb * 64, (cl * HV + h) * D);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(sB + kb * 16384, mb, &bar_full[st], kb * 64, (cl * HV + h) * D);
        if (ptrace) p.ts[3072 + step * 4 + 3] = globaltimer_ns();
      }
    } else if (lane == 1 && p.ts && h == 0 && s == 0) {
      // diagnostic: when does each stage's data land?
      for (int step = 0; step < ng; ++step) { mbar_wait(&bar_full[step % K2_STAGES], (step / K2_STAGES) & 1); p.ts[3400 + step] = globaltimer_ns(); }
    }
    return;
  }
  if (warp == K2_CTHREADS / 32 + 1) {
    // ---- storer warp: bf16 S_c0 of each step -> workspace (TMA store), flagged for K3 once complete.
    if (lane == 0) {
      for (int step = 0; step < ng; ++step) {
        const int b = step % K2_NSB;
        const int c0 = cbase + step * G;
        mbar_wait(&bar_sready[b], (step / K2_NSB) & 1);
        if (!(p.dbg & 1)) {
          for (int kb = 0; kb < 2; ++kb) tma_store_2d(&tm_sc, sSbuf + b * 32768 + kb * 16384, kb * 64, (c0 * HV + h) * D);
          tma_store_commit();
          tma_store_wait_all<0>();
        }
        flag_set(&p.fS[c0 * HV + h], p.epoch);
        mbar_arrive(&bar_sfree[b]);
        if (p.ts && tid == K2_CTHREADS + 32 && h == 0 && s == 0) p.ts[3700 + step] = globaltimer_ns();
      }
    }
    return;
  }

  // ---- compute warps (named barrier 1, 512 threads)
  K2_TR(3);
  int cur = 0;
  // init S^T in buffer cur
  {
    const uint32_t tb = tmem + cur * 128 + lane_off;
    uint32_t r[32];
    if (p.state) {
      mbar_wait_cta_n(&bar_state, 0, K2_CTHREADS);
      const uint8_t* blk = sSbuf + cg * 16384;
#pragma unroll
      for (int i4 = 0; i4 < 8; ++i4) {
        uint32_t chunk = (uint32_t)i4 ^ (uint32_t)(v & 7);
        uint4 f = *reinterpret_cast<const uint4*>(blk + v * 128 + (chunk << 4));
        r[4 * i4] = f.x; r[4 * i4 + 1] = f.y; r[4 * i4 + 2] = f.z; r[4 * i4 + 3] = f.w;
      }
      named_sync(1, K2_CTHREADS);   // sS region read by everyone before step 0 overwrites it
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) r[j] = 0u;
    }
    tmem_st_x32(tb, r);
    tmem_st_wait();
  }
  K2_TR(6);
  unsigned long long acc_ts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int step = 0; step < ng; ++step) {
    const int st = step % K2_STAGES;
    const int nxt = cur ^ 1;
    const int c0 = cbase + step * G;   // state entering this group is S_{c0}
    uint8_t* sN = smem + st * K2_STAGE_BYTES;
    const uint8_t* sB = sN + 32768;
    uint8_t* sS = sSbuf + (step % K2_NSB) * 32768;
    unsigned long long t_a = clock64();
#define K2_TS(i) do { if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[64 + step * 16 + (i)] = clock64(); } while (0)
    K2_TS(0);
    // the store of the state written to this sS buffer two steps ago has completed (storer warp)
    if (step >= K2_NSB) mbar_wait(&bar_sfree[step % K2_NSB], (step / K2_NSB - 1) & 1);
    unsigned long long t_b = clock64();
    K2_TS(1);
    if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[3300 + step] = globaltimer_ns();
    mbar_wait_cta_n(&bar_full[st], (step / K2_STAGES) & 1, K2_CTHREADS);
    if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[3200 + step] = globaltimer_ns();
    unsigned long long t_c = clock64();
    K2_TS(2);
    const float gam = sGam[st];
    // ---- threads: S_c0 -> bf16 smem ; S' = gamma S + B -> TMEM nxt
    if (!(p.dbg & 4)) {
      const uint32_t tcur = tmem + cur * 128 + lane_off;
      const uint32_t tnxt = tmem + nxt * 128 + lane_off;
      uint32_t r[32];
      tmem_ld_x32(tcur, r);
      tmem_ld_wait();
      {
        const int q = cg;
        uint8_t* blk = sS + (q >> 1) * 16384;
        const uint8_t* bblk = sB + (q >> 1) * 16384;
#pragma unroll
        for (int e8 = 0; e8 < 4; ++e8) {
          const int e = (32 * q + e8 * 8) & 63;
          uint32_t* rr = r + e8 * 8;
          uint4 w;
          w.x = pack_bf16x2(__uint_as_float(rr[0]), __uint_as_float(rr[1]));
          w.y = pack_bf16x2(__uint_as_float(rr[2]), __uint_as_float(rr[3]));
          w.z = pack_bf16x2(__uint_as_float(rr[4]), __uint_as_float(rr[5]));
          w.w = pack_bf16x2(__uint_as_float(rr[6]), __uint_as_float(rr[7]));
          st_sw128_16B(blk, v, e, w);
          uint4 bb = ld_sw128_16B(bblk, v, e);
          uint32_t bu[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
          for (int mm = 0; mm < 4; ++mm) {
            float2 f = unpack_bf16x2(bu[mm]);
            rr[2 * mm] = __float_as_uint(gam * __uint_as_float(rr[2 * mm]) + f.x);
            rr[2 * mm + 1] = __float_as_uint(gam * __uint_as_float(rr[2 * mm + 1]) + f.y);
          }
        }
        tmem_st_x32(tnxt, r);
      }
      tmem_st_wait();
    }
    unsigned long long t_d = clock64();
    K2_TS(3);
    tcgen05_fence_before();
    fence_proxy_async_smem();
    named_sync(1, K2_CTHREADS);
    unsigned long long t_e = clock64(), t_f = t_e, t_g = t_e;
    K2_TS(4);
    if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[3500 + step] = globaltimer_ns();
    if (tid == 0) mbar_arrive(&bar_sready[step % K2_NSB]);   // bf16 S_c0 complete in sS -> storer warp
    if (tid == 0) {
      tcgen05_fence_after();
      const uint32_t idesc = make_idesc_bf16(128, 128, 0, 1);
      const uint32_t tD = tmem + nxt * 128;
      if (!(p.dbg & 2)) {
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          umma_bf16_ss(tD, desc_kmajor(smem_u32(sS), 128, ks), desc_mnmajor(smem_u32(sN), 128, ks), idesc, 1);
      }
      umma_commit(&bar_mma);
      t_f = clock64();
      K2_TS(5);
    }
    if (tid == 0) t_g = clock64();
    K2_TS(6);
    mbar_wait_cta_n(&bar_mma, step & 1, K2_CTHREADS);
    tcgen05_fence_after();
    if (tid == 0) mbar_arrive(&bar_empty[st]);   // stage consumed (B by the threads, N by the MMA)
    K2_TS(7);
    if (p.ts && tid == 0 && h == 0 && s == 0) p.ts[3600 + step] = globaltimer_ns();
    if (p.ts && tid == 0 && h == 0 && s == 0) {
      unsigned long long t_h = clock64();
      acc_ts[0] += t_b - t_a; acc_ts[1] += t_c - t_b; acc_ts[2] += t_d - t_c; acc_ts[3] += t_e - t_d;
      acc_ts[4] += t_f - t_e; acc_ts[5] += t_g - t_f; acc_ts[6] += t_h - t_g; acc_ts[7] += 1;
    }
    if (step == ng - 1) {
      // final state of this sequence -> new_state (fp32, k-last [v][k]) staged in the consumed stage buffer, TMA store
      const uint32_t tn = tmem + nxt * 128 + lane_off;
      uint32_t r[32];
      tmem_ld_x32(tn, r);
      tmem_ld_wait();
      uint8_t* stg = sN + cg * 16384;   // block cg: [128 rows][32 fp32], SW128
#pragma unroll
      for (int i4 = 0; i4 < 8; ++i4) {
        uint32_t chunk = (uint32_t)i4 ^ (uint32_t)(v & 7);
        *reinterpret_cast<uint4*>(stg + v * 128 + (chunk << 4)) = make_uint4(r[4 * i4], r[4 * i4 + 1], r[4 * i4 + 2], r[4 * i4 + 3]);
      }
      fence_proxy_async_smem();
      named_sync(1, K2_CTHREADS);
      if (tid == 32) {
        for (int q = 0; q < 4; ++q) tma_store_2d(&tm_so, sN + q * 16384, q * 32, (s * HV + h) * D);
        tma_store_commit();
      }
    }
    K2_TS(8);
    cur = nxt;
  }
  K2_TR(10);
  if (tid == 32) tma_store_wait_all<0>();   // final state store
  tcgen05_fence_before();
  named_sync(1, K2_CTHREADS);
  K2_TR(11);
  if (p.ts && tid == 0) p.ts[2560 + 2 * (s * HV + h) + 1] = globaltimer_ns();
  if (p.ts && tid == 0 && h == 0 && s == 0) {
    for (int i = 0; i < 8; ++i) p.ts[i] = acc_ts[i];
    p.ts[29] = 0;
  }
#undef K2_TR
#undef K2_TS
  if (warp == 0) tmem_dealloc(tmem, 256);
}

// ===========================================================================
// K3: persistent output kernel with two item types in one queue (no inter-grid dispatch dependency):
//   items [0, n3):       K3a1 per (chunk, head): M = Qg - P W, PU = P U            (needs only K1 of the chunk)
//   items [n3, 2 n3):    K3b  per (chunk, head): O = (gp M + M NP) S_c0 + M BP + PU  (needs K15 prefix, chain state, M/PU)
// CTAs walk the queue in order, so every K3a1 item is taken before any K3b item; K3b items wait on flags.
// smem: 6 regions of 16 KB + gates 1 KB (98 KB -> 2 CTAs/SM); TMEM 256; 128 threads.
//   K3a1: RA: Q -> M | RB: V -> U -> PU | RC: K -> P (8K) | RD: Tk,Tv -> W
//   K3b:  R0+R1: S0 | R2: M | R3: PU (then A') | R4+R5: NP^T, then BP^T, then O staging
// ===========================================================================
constexpr int K3_SMEM = 16384 * 6 + 1024 + 1024;      // mode 1 (K3b items): 98 KB, TMEM 256 -> 2 CTAs/SM
constexpr int K3A_SMEM = 16384 * 4 + 1024 + 1024;     // mode 0 (K3a1 items): 66 KB (+ G row), TMEM 128 -> 3 CTAs/SM
constexpr int K3_THREADS = 128;
// mode 0: K3a1 items only (persistent); mode 1: K3b items only; mode 2: both in one queue (persistent)
__global__ void __launch_bounds__(K3_THREADS, 4) k3_main(const Params p, const __grid_constant__ CUtensorMap tm_q,
                                                        const __grid_constant__ CUtensorMap tm_k,
                                                        const __grid_constant__ CUtensorMap tm_tk,
                                                        const __grid_constant__ CUtensorMap tm_up,
                                                        const __grid_constant__ CUtensorMap tm_gs,
                                                        const __grid_constant__ CUtensorMap tm_mp,
                                                        const __grid_constant__ CUtensorMap tm_pup,
                                                        const __grid_constant__ CUtensorMap tm_sc,
                                                        const __grid_constant__ CUtensorMap tm_np,
                                                        const __grid_constant__ CUtensorMap tm_bp,
                                                        const __grid_constant__ CUtensorMap tm_nt,
                                                        const __grid_constant__ CUtensorMap tm_bt,
                                                        const __grid_constant__ CUtensorMap tm_out, const int n3, const int mode) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* R0 = smem;
  uint8_t* R1 = R0 + 16384;
  uint8_t* R2 = R1 + 16384;
  uint8_t* R3 = R2 + 16384;
  uint8_t* R4 = R3 + 16384;
  float* sG = reinterpret_cast<float*>(smem + 16384 * (mode == 0 ? 4 : 6));   // G cumsum of the chunk (from K1, TMA-loaded)
  __shared__ __align__(8) uint64_t bar_tma;
  __shared__ __align__(8) uint64_t bar_tma2;
  __shared__ __align__(8) uint64_t bar_mma;
  __shared__ uint32_t tmem_slot;
  __shared__ float s_gp;
  __shared__ int s_pref;         // K3b: the next item's loads were issued by the previous item (its flags were ready)
  __shared__ int s_cu[MAX_CU];
  griddep_launch_dependents();   // (the next call's K2 waits for this grid in plain stream order)
  if (threadIdx.x == 0) {
    tma_prefetch_desc(&tm_q); tma_prefetch_desc(&tm_k); tma_prefetch_desc(&tm_tk); tma_prefetch_desc(&tm_up); tma_prefetch_desc(&tm_gs);
    tma_prefetch_desc(&tm_mp); tma_prefetch_desc(&tm_pup); tma_prefetch_desc(&tm_sc); tma_prefetch_desc(&tm_np);
    tma_prefetch_desc(&tm_bp); tma_prefetch_desc(&tm_nt); tma_prefetch_desc(&tm_bt); tma_prefetch_desc(&tm_out);
  }
  const CuView cu = load_cu(s_cu, p.cu_seqlens, p.num_seqs);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int t = 16 * warp + lane;            // M=64 accumulator layout: row t lives in lane 16*warp+lane (lane < 16)
  const bool rowv = lane < 16;
  if (tid == 0) { mbar_init(&bar_tma, 1); mbar_init(&bar_tma2, 1); mbar_init(&bar_mma, 1); fence_mbar_init(); }
  const uint32_t tmem_cols = (mode == 0) ? 128 : 256;
  if (warp == 0) { tmem_alloc(&tmem_slot, tmem_cols); tmem_relinquish(); }
  __syncthreads();
  const uint32_t tmem = tmem_slot;
  const uint32_t tq1 = tmem + ((uint32_t)(32 * warp) << 16), tq2 = tmem + 128 + ((uint32_t)(32 * warp) << 16);
  int mma_phase = 0, tma_phase = 0, tma2_phase = 0;

  // readback helper: D (64 x 128 fp32) at tq -> bf16 tile dst [64][128] (2 blocks) with per-element op
  auto readback = [&](uint32_t tq, uint8_t* dst, auto op) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t rw[32];
      tmem_ld_x32(tq + 32 * q, rw);
      tmem_ld_wait();
      if (rowv) {
#pragma unroll
        for (int e8 = 0; e8 < 4; ++e8) {
          const int e = (32 * q + e8 * 8) & 63;
          uint32_t o[4];
          op(q, e, rw + e8 * 8, o);
          st_sw128_16B(dst + (q >> 1) * 8192, t, e, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
    }
  };
  auto cvt = [](int, int, const uint32_t* rw, uint32_t* o) {
#pragma unroll
    for (int m = 0; m < 4; ++m) o[m] = pack_bf16x2(__uint_as_float(rw[2 * m]), __uint_as_float(rw[2 * m + 1]));
  };
  auto mma_wait = [&]() { mbar_wait_cta(&bar_mma, mma_phase & 1); mma_phase++; tcgen05_fence_after(); };
  auto tma_wait = [&]() { mbar_wait_cta(&bar_tma, tma_phase & 1); tma_phase++; };
  auto sync_smem = [&]() { fence_proxy_async_smem(); tcgen05_fence_before(); __syncthreads(); };
  auto mma_k64 = [&](const uint8_t* a, const uint8_t* b) {   // D(cols 0-127) = A[64][64] (K-major) x B[64][128] (MN-major)
    tcgen05_fence_after();
    const uint32_t idesc = make_idesc_bf16(64, 128, 0, 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) umma_bf16_ss(tmem, desc_kmajor(smem_u32(a), 64, ks), desc_mnmajor(smem_u32(b), 64, ks), idesc, ks > 0);
    umma_commit(&bar_mma);
  };
  auto mma_k128 = [&](uint32_t d, const uint8_t* a, const uint8_t* b, uint32_t acc) {   // D (+)= A[64][128] (K-major) x B[128 n][128 k] (K-major)
    const uint32_t idesc = make_idesc_bf16(64, 128, 0, 0);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) umma_bf16_ss(d, desc_kmajor(smem_u32(a), 64, ks), desc_kmajor(smem_u32(b), 128, ks), idesc, acc | (ks > 0));
  };

  // ------------------------------------------------------------------ K3a1 items (software pipelined)
  // Loads of the next item are issued as soon as their regions are dead (K, Tk/Tv after the PW MMA; Q, V after the
  // M/PU stores have read R0/R1); the completion wait + flag of an item happen one item later (flag by a thread that
  // issued no bulk stores, so that its release fence does not drain the in-flight stores).
  auto k3a1_decode = [&](int item, int& c, int& h, int& hq, int& t0, int& len) {
    c = item / HV; h = item % HV; hq = h / (HV / HQ);
    int seq; bool first, last;
    return decode_chunk(cu, p.num_seqs, c, seq, t0, len, first, last);
  };
  auto k3a1_next = [&](int item) {   // next valid item of this CTA (or -1)
    int c, h, hq, t0, len;
    for (int nx = item + gridDim.x; nx < n3; nx += gridDim.x) if (k3a1_decode(nx, c, h, hq, t0, len)) return nx;
    return -1;
  };
  auto k3a1_issue_k = [&](int item) {   // K -> R2, Tk -> R3, G -> sG (bar_tma2); waits for the chunk's K1 flag
    int c, h, hq, t0, len; k3a1_decode(item, c, h, hq, t0, len);
    flag_wait(&p.fK1[c * HV + h], p.epoch);
    const size_t ws_row = (size_t)(c * HV + h) * C;
    mbar_arrive_expect_tx(&bar_tma2, 16384 + 8192 + 256);
    for (int kb = 0; kb < 2; ++kb) tma_load_2d(R2 + kb * 8192, &tm_k, &bar_tma2, hq * D + kb * 64, t0);
    tma_load_2d(R3, &tm_tk, &bar_tma2, 0, ws_row);
    tma_load_2d(sG, &tm_gs, &bar_tma2, 0, c * HV + h);
  };
  auto k3a1_issue_qv = [&](int item) {   // Q -> R0, U -> R1 (bar_tma)
    int c, h, hq, t0, len; k3a1_decode(item, c, h, hq, t0, len);
    const size_t ws_row = (size_t)(c * HV + h) * C;
    mbar_arrive_expect_tx(&bar_tma, 16384 * 2);
    for (int kb = 0; kb < 2; ++kb) {
      tma_load_2d(R0 + kb * 8192, &tm_q, &bar_tma, hq * D + kb * 64, t0);
      tma_load_2d(R1 + kb * 8192, &tm_up, &bar_tma, kb * 64, ws_row);
    }
  };
  auto k3a1_item = [&](int item, int next, int& prev_flag) {
    uint8_t* sQ = R0; uint8_t* sM = R0;
    uint8_t* sU = R1; uint8_t* sPU = R1;
    uint8_t* sK = R2; uint8_t* sP = R2;
    uint8_t* sTk = R3; uint8_t* sW = R3;
    int c, h, hq, t0, len; k3a1_decode(item, c, h, hq, t0, len);
    unsigned long long k3t[12]; int k3n = 0;
#define K3_MARK() do { if (tid == 0) k3t[k3n] = clock64(); k3n++; } while (0)
    K3_MARK();
    if (p.ts && tid == 0) p.ts[4096 + 2 * item] = globaltimer_ns();
    const size_t ws_row = (size_t)(c * HV + h) * C;
    mbar_wait_cta(&bar_tma2, tma2_phase & 1); tma2_phase++;   // K, Tk, G
    K3_MARK();   // 1: K/Tk/G landed
    // ---- W = Tk K -> sW (over Tk)
    if (tid == 0) mma_k64(sTk, sK);
    mma_wait();
    readback(tq1, sW, cvt);
    sync_smem();
    tma_wait();                                                // Q, U
    K3_MARK();   // 2: W, loads landed
    K3_MARK();   // 3: (U from K1)
    // ---- P = Q K^T (cols 0-63) -> sP (over K), masked / decayed / scaled
    if (tid == 0) {
      tcgen05_fence_after();
      const uint32_t idesc_p = make_idesc_bf16(64, 64, 0, 0);
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) umma_bf16_ss(tmem, desc_kmajor(smem_u32(sQ), 64, ks), desc_kmajor(smem_u32(sK), 64, ks), idesc_p, ks > 0);
      umma_commit(&bar_mma);
    }
    mma_wait();
    {
      uint32_t rr[64];
      tmem_ld_x32(tq1, rr);
      tmem_ld_x32(tq1 + 32, rr + 32);
      tmem_ld_wait();
      if (rowv) {
        const float Gt = sG[t];
#pragma unroll
        for (int e8 = 0; e8 < 8; ++e8) {
          uint32_t o[4];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            int j = e8 * 8 + 2 * m;
            float p0 = (j <= t) ? __expf(Gt - sG[j]) * p.scale * __uint_as_float(rr[j]) : 0.f;
            float p1 = (j + 1 <= t) ? __expf(Gt - sG[j + 1]) * p.scale * __uint_as_float(rr[j + 1]) : 0.f;
            o[m] = pack_bf16x2(p0, p1);
          }
          st_sw128_16B(sP, t, e8 * 8, make_uint4(o[0], o[1], o[2], o[3]));
        }
      }
    }
    sync_smem();
    K3_MARK();   // 4: P
    // ---- PU = P U -> sPU (over U) -> workspace
    if (tid == 0) mma_k64(sP, sU);
    mma_wait();
    readback(tq1, sPU, cvt);
    sync_smem();
    if (tid == 0) {
      for (int kb = 0; kb < 2; ++kb) tma_store_2d(&tm_pup, sPU + kb * 8192, kb * 64, ws_row);
      tma_store_commit();
    }
    // ---- PW -> M = Qg - PW -> sM (over Q, in place per thread) -> workspace
    if (tid == 0) mma_k64(sP, sW);
    mma_wait();
    if (tid == 0 && next >= 0) k3a1_issue_k(next);   // P (R2) and W (R3) are dead: prefetch the next item's K, Tk, Tv
    {
      const float qs = rowv ? __expf(sG[t]) * p.scale : 0.f;
      readback(tq1, sM, [&](int q, int e, const uint32_t* rw, uint32_t* o) {
        uint4 u = ld_sw128_16B(sQ + (q >> 1) * 8192, t, e);
        const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          float2 f = unpack_bf16x2(uu[m]);
          o[m] = pack_bf16x2(f.x * qs - __uint_as_float(rw[2 * m]), f.y * qs - __uint_as_float(rw[2 * m + 1]));
        }
      });
    }
    sync_smem();
    K3_MARK();   // 5: M
    if (tid == 0) {
      for (int kb = 0; kb < 2; ++kb) tma_store_2d(&tm_mp, sM + kb * 8192, kb * 64, ws_row);
      tma_store_commit();
      if (prev_flag >= 0) tma_store_wait_all<2>();   // the previous item's PU/M stores are complete
      if (next >= 0) { tma_store_wait<0>(); k3a1_issue_qv(next); }   // this item's stores have read R0/R1: prefetch Q, V
    }
    __syncthreads();
    if (tid == 32 && prev_flag >= 0) flag_set(&p.fM[prev_flag], p.epoch);
    prev_flag = c * HV + h;
    K3_MARK();   // 6: stores issued, previous item flagged
    if (p.ts && tid == 0) {
      for (int i = 1; i < k3n; ++i) atomicAdd(&p.ts[48 + i - 1], k3t[i] - k3t[i - 1]);
      atomicAdd(&p.ts[63], 1ull);
      p.ts[4096 + 2 * item + 1] = globaltimer_ns();
    }
#undef K3_MARK
  };

  // ------------------------------------------------------------------ K3b items
  // Regions: S0 = R0+R1 | M = R2 | PU -> A' -> O staging = R3 | NP -> BP = R4+R5.
  // When the next item's flags are already set at the end of an item, its S0/M/NP loads are issued right after the
  // last MMAs (those regions are dead) and PU after the O store has read R3: the next item then starts without a
  // load wait.
  struct K3bItem { int c, h, r, c0, t0, len; size_t ws_row; int prow; bool valid; };
  auto k3b_decode = [&](int item) {
    K3bItem it; it.valid = false;
    if (item < 0 || item >= n3) return it;
    it.c = item / HV; it.h = item % HV;
    int seq; bool first, last;
    if (!decode_chunk(cu, p.num_seqs, it.c, seq, it.t0, it.len, first, last)) return it;
    const int cbase = seq_chunk_base(cu, seq);
    it.r = (it.c - cbase) % G; it.c0 = it.c - it.r;
    it.ws_row = (size_t)(it.c * HV + it.h) * C; it.prow = ((it.c - 1) * HV + it.h) * D; it.valid = true;
    return it;
  };
  // thread 0: are all flags of the item set?  (single non-blocking poll; stores the prefix gamma payload)
  auto k3b_flags_ready = [&](const K3bItem& it) {
    const uint32_t e = p.epoch;
    uint64_t a = flag_load(&p.fM[it.c * HV + it.h]), b = flag_load(&p.fS[it.c0 * HV + it.h]);
    if ((uint32_t)(a >> 32) != e || (uint32_t)(b >> 32) != e) return false;
    if (it.r > 1) {
      uint64_t x = flag_load(&p.fNP[(it.c - 1) * HV + it.h]), y = flag_load(&p.fBP[(it.c - 1) * HV + it.h]);
      if ((uint32_t)(x >> 32) != e || (uint32_t)(y >> 32) != e) return false;
      s_gp = __uint_as_float((uint32_t)x);
    } else if (it.r == 1) {
      uint64_t x = flag_load(&p.fK1[(it.c - 1) * HV + it.h]);
      if ((uint32_t)(x >> 32) != e) return false;
      s_gp = __uint_as_float((uint32_t)x);
    }
    flag_fence();
    return true;
  };
  auto k3b_wait_flags = [&](const K3bItem& it) {   // thread 0: blocking version
    if (it.r > 1) s_gp = flag_wait2(&p.fNP[(it.c - 1) * HV + it.h], &p.fBP[(it.c - 1) * HV + it.h], p.epoch);
    else if (it.r == 1) s_gp = flag_wait(&p.fK1[(it.c - 1) * HV + it.h], p.epoch);
    flag_wait2(&p.fM[it.c * HV + it.h], &p.fS[it.c0 * HV + it.h], p.epoch);
  };
  auto k3b_issue_smn = [&](const K3bItem& it) {   // thread 0: arm the barrier for the item's first phase; S0, M, NP
    const CUtensorMap* mn = (it.r > 1) ? &tm_np : &tm_nt;
    mbar_arrive_expect_tx(&bar_tma, 65536 + (it.r > 0 ? 32768 : 0));
    for (int kb = 0; kb < 2; ++kb) {
      tma_load_2d(R0 + kb * 16384, &tm_sc, &bar_tma, kb * 64, (it.c0 * HV + it.h) * D);
      tma_load_2d(R2 + kb * 8192, &tm_mp, &bar_tma, kb * 64, it.ws_row);
      if (it.r > 0) tma_load_2d(R4 + kb * 16384, mn, &bar_tma, kb * 64, it.prow);
    }
  };
  auto k3b_issue_pu = [&](const K3bItem& it) {   // thread 0: PU -> R3 (same barrier phase)
    for (int kb = 0; kb < 2; ++kb) tma_load_2d(R3 + kb * 8192, &tm_pup, &bar_tma, kb * 64, it.ws_row);
  };
  auto k3b_item = [&](int item, int next) {
    uint8_t* sS = R0;            // 32 KB (R0+R1)
    uint8_t* sM = R2;
    uint8_t* sPU = R3; uint8_t* sA = R3; uint8_t* sO = R3;
    uint8_t* sNP = R4;           // 32 KB (R4+R5)
    const K3bItem it = k3b_decode(item);
    if (!it.valid) return;
    const int c = it.c, h = it.h, r = it.r, c0 = it.c0, t0 = it.t0, len = it.len;
    const size_t ws_row = it.ws_row;
    const int prow = it.prow;
    const CUtensorMap* mb = (r > 1) ? &tm_bp : &tm_bt;
    unsigned long long k3t[6]; int k3n = 0;
#define K3_MARK() do { if (tid == 0) k3t[k3n] = clock64(); k3n++; } while (0)
    K3_MARK();
    if (p.ts && tid == 0) p.ts[6144 + 2 * item] = globaltimer_ns();
    if (tid == 0 && !s_pref) {
      // dependencies: M/PU of the chunk (K3a1 items), prefix map of c-1 (K15 / K1), chain state S_c0 (K2)
      k3b_wait_flags(it);
      K3_MARK();   // 1: flags
      k3b_issue_smn(it);
      k3b_issue_pu(it);
    } else { k3n++; }
    tma_wait();
    K3_MARK();   // 2: loads
    // ---- D2 := PU (fp32) from the bf16 tile (tcgen05.st is warp-aligned: lanes 16-31, which hold no row, store zeros)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t rw[32];
#pragma unroll
      for (int e8 = 0; e8 < 4; ++e8) {
        uint4 u = ld_sw128_16B(sPU + (q >> 1) * 8192, rowv ? t : 0, (32 * q + e8 * 8) & 63);
        const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          float2 f = unpack_bf16x2(uu[m]);
          rw[e8 * 8 + 2 * m] = rowv ? __float_as_uint(f.x) : 0u;
          rw[e8 * 8 + 2 * m + 1] = rowv ? __float_as_uint(f.y) : 0u;
        }
      }
      tmem_st_x32(tq2 + 32 * q, rw);
    }
    tmem_st_wait();
    tcgen05_fence_before();
    __syncthreads();
    if (r > 0) {
      // ---- D1 = M NP;  A' = D1 + gp M -> over PU;  BP^T -> over NP
      if (tid == 0) { tcgen05_fence_after(); mma_k128(tmem, sM, sNP, 0); umma_commit(&bar_mma); }
      mma_wait();
      if (tid == 0) {
        mbar_arrive_expect_tx(&bar_tma, 32768);
        for (int kb = 0; kb < 2; ++kb) tma_load_2d(sNP + kb * 16384, mb, &bar_tma, kb * 64, prow);
      }
      {
        const float gp = s_gp;
        readback(tq1, sA, [&](int q, int e, const uint32_t* rw, uint32_t* o) {
          uint4 u = ld_sw128_16B(sM + (q >> 1) * 8192, t, e);
          const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            float2 f = unpack_bf16x2(uu[m]);
            o[m] = pack_bf16x2(__uint_as_float(rw[2 * m]) + gp * f.x, __uint_as_float(rw[2 * m + 1]) + gp * f.y);
          }
        });
      }
      sync_smem();
      K3_MARK();   // 3: A'
      tma_wait();
      // ---- D2 += A' S0 + M BP
      if (tid == 0) { tcgen05_fence_after(); mma_k128(tmem + 128, sA, sS, 1); mma_k128(tmem + 128, sM, sNP, 1); umma_commit(&bar_mma); }
    } else {
      K3_MARK();
      if (tid == 0) { tcgen05_fence_after(); mma_k128(tmem + 128, sM, sS, 1); umma_commit(&bar_mma); }
    }
    mma_wait();
    // ---- next item: if its flags are already set, issue its S0 / M / NP loads now (R0-R2, R4-R5 are dead)
    const K3bItem nx = k3b_decode(next);
    if (tid == 0) {
      s_pref = 0;
      if (nx.valid && k3b_flags_ready(nx)) { k3b_issue_smn(nx); s_pref = 1; }
    }
    // ---- O -> bf16: full chunks via smem staging (R3: A' / PU are dead) + TMA store, partial chunks via masked direct stores
    {
      bf16* go = p.out + ((size_t)(t0 + t) * HV + h) * D;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t rw[32];
        tmem_ld_x32(tq2 + 32 * q, rw);
        tmem_ld_wait();
        if (rowv) {
#pragma unroll
          for (int e8 = 0; e8 < 4; ++e8) {
            const int e = (32 * q + e8 * 8) & 63;
            uint32_t o[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) o[m] = pack_bf16x2(__uint_as_float(rw[e8 * 8 + 2 * m]), __uint_as_float(rw[e8 * 8 + 2 * m + 1]));
            if (len == C) st_sw128_16B(sO + (q >> 1) * 8192, t, e, make_uint4(o[0], o[1], o[2], o[3]));
            else if (t < len) *reinterpret_cast<uint4*>(go + 32 * q + e8 * 8) = make_uint4(o[0], o[1], o[2], o[3]);
          }
        }
      }
    }
    if (len == C) {
      fence_proxy_async_smem();
      __syncthreads();
      if (tid == 0) {
        for (int kb = 0; kb < 2; ++kb) tma_store_2d(&tm_out, sO + kb * 8192, h * D + kb * 64, t0);
        tma_store_commit();
        tma_store_wait<0>();   // (read: R3 is free again)
      }
    }
    if (tid == 0 && s_pref) k3b_issue_pu(nx);   // completes the next item's barrier phase
    tcgen05_fence_before();
    __syncthreads();   // (s_pref visible to all threads for the next item)
    K3_MARK();   // 4: stored
    if (p.ts && tid == 0) {
      for (int i = 1; i < k3n; ++i) atomicAdd(&p.ts[3900 + i - 1], k3t[i] - k3t[i - 1]);
      atomicAdd(&p.ts[3907], 1ull);
      p.ts[6144 + 2 * item + 1] = globaltimer_ns();
    }
#undef K3_MARK
  };

  if (mode == 0) {
    int prev_flag = -1;
    int item = k3a1_next((int)blockIdx.x - (int)gridDim.x);
    if (item >= 0 && tid == 0) { k3a1_issue_k(item); k3a1_issue_qv(item); }
    for (; item >= 0;) {
      const int next = k3a1_next(item);
      k3a1_item(item, next, prev_flag);
      item = next;
    }
    if (tid == 0) tma_store_wait_all<0>();
    __syncthreads();
    if (tid == 32 && prev_flag >= 0) flag_set(&p.fM[prev_flag], p.epoch);
  } else {
    if (tid == 0) s_pref = 0;
    __syncthreads();
    for (int item = blockIdx.x; item < n3; item += gridDim.x) k3b_item(item, item + (int)gridDim.x);
  }
  tcgen05_fence_before();
  __syncthreads();
  if (tid == 0) tma_store_wait_all<0>();
  if (warp == 0) tmem_dealloc(tmem, tmem_cols);
}

// ===========================================================================
// host
// ===========================================================================
typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                             const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                             CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeFn get_encode() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult qres;
    void* f = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &qres);
    TORCH_CHECK(f != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<EncodeFn>(f);
  }
  return fn;
}

static CUtensorMap make_tmap_2d(const void* ptr, CUtensorMapDataType dt, int elem_bytes, uint64_t inner, uint64_t outer,
                                uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer};
  cuuint64_t strides[1] = {inner * (uint64_t)elem_bytes};
  cuuint32_t box[2] = {box_inner, box_outer};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, dt, 2, const_cast<void*>(ptr), dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
  return m;
}

struct Workspace {
  torch::Tensor Tk, Tv, NT, BT, Sc, gamma, ts, NP, BP, gamma_p, flags, Mp, PUp, Up, Gs;
  int max_chunks = 0;
  uint32_t epoch = 0;
};
static Workspace g_ws;
static int g_num_sms = 148;

static void ensure_workspace(int max_chunks, const torch::TensorOptions& opt_bf16) {
  if (g_ws.max_chunks >= max_chunks) return;
  auto opt_f32 = opt_bf16.dtype(torch::kFloat32);
  g_ws.Tk = torch::empty({(int64_t)max_chunks * HV * C, C}, opt_bf16);
  g_ws.Tv = torch::empty({(int64_t)max_chunks * HV * C, C}, opt_bf16);
  g_ws.NT = torch::empty({(int64_t)max_chunks * HV * D, D}, opt_bf16);
  g_ws.BT = torch::empty({(int64_t)max_chunks * HV * D, D}, opt_bf16);
  g_ws.Sc = torch::empty({(int64_t)max_chunks * HV * D, D}, opt_bf16);
  g_ws.gamma = torch::empty({(int64_t)max_chunks * HV}, opt_f32);
  g_ws.NP = torch::empty({(int64_t)max_chunks * HV * D, D}, opt_bf16);
  g_ws.BP = torch::empty({(int64_t)max_chunks * HV * D, D}, opt_bf16);
  g_ws.gamma_p = torch::empty({(int64_t)max_chunks * HV}, opt_f32);
  g_ws.Mp = torch::empty({(int64_t)max_chunks * HV * C, D}, opt_bf16);
  g_ws.PUp = torch::empty({(int64_t)max_chunks * HV * C, D}, opt_bf16);
  g_ws.Up = torch::empty({(int64_t)max_chunks * HV * C, D}, opt_bf16);
  g_ws.Gs = torch::empty({(int64_t)max_chunks * HV, C}, opt_f32);
  g_ws.ts = torch::zeros({16384}, opt_f32.dtype(torch::kInt64));
  g_ws.flags = torch::zeros({5, (int64_t)max_chunks * HV}, opt_f32.dtype(torch::kInt64));
  g_ws.max_chunks = max_chunks;
}

void gdn_prefill(torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> state, torch::Tensor A_log,
                 torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b, torch::Tensor cu_seqlens, double scale,
                 torch::Tensor out, torch::Tensor new_state, int64_t dbg) {
  const int total = q.size(0);
  const int num_seqs = cu_seqlens.numel() - 1;
  const int max_chunks = total / C + num_seqs;
  ensure_workspace(max_chunks, q.options());

  Params p;
  p.q = reinterpret_cast<const bf16*>(q.data_ptr());
  p.k = reinterpret_cast<const bf16*>(k.data_ptr());
  p.v = reinterpret_cast<const bf16*>(v.data_ptr());
  p.state = (state.has_value() && state->defined()) ? state->data_ptr<float>() : nullptr;
  p.A_log = A_log.data_ptr<float>();
  p.a = reinterpret_cast<const bf16*>(a.data_ptr());
  p.dt_bias = dt_bias.data_ptr<float>();
  p.b = reinterpret_cast<const bf16*>(b.data_ptr());
  p.cu_seqlens = cu_seqlens.data_ptr<int64_t>();
  p.num_seqs = num_seqs; p.total = total; p.scale = (float)scale; p.dbg = (int)dbg;
  p.out = reinterpret_cast<bf16*>(out.data_ptr());
  p.new_state = new_state.data_ptr<float>();
  p.Tk = reinterpret_cast<bf16*>(g_ws.Tk.data_ptr());
  p.Tv = reinterpret_cast<bf16*>(g_ws.Tv.data_ptr());
  p.NT = reinterpret_cast<bf16*>(g_ws.NT.data_ptr());
  p.BT = reinterpret_cast<bf16*>(g_ws.BT.data_ptr());
  p.Sc = reinterpret_cast<bf16*>(g_ws.Sc.data_ptr());
  p.gamma = g_ws.gamma.data_ptr<float>();
  p.NP = reinterpret_cast<bf16*>(g_ws.NP.data_ptr());
  p.BP = reinterpret_cast<bf16*>(g_ws.BP.data_ptr());
  p.gamma_p = g_ws.gamma_p.data_ptr<float>();
  p.Mp = reinterpret_cast<bf16*>(g_ws.Mp.data_ptr());
  p.PUp = reinterpret_cast<bf16*>(g_ws.PUp.data_ptr());
  p.Up = reinterpret_cast<bf16*>(g_ws.Up.data_ptr());
  p.Gs = g_ws.Gs.data_ptr<float>();
  p.ts = (dbg & 8) ? reinterpret_cast<unsigned long long*>(g_ws.ts.data_ptr<int64_t>()) : nullptr;
  // dbg 64: profiling mode (Nsight Compute replays kernels one at a time, which would deadlock the flag waits): reuse the
  // previous call's epoch so that every flag wait passes immediately; the kernels then consume the previous call's
  // intermediate data (same inputs -> same values).  Not for timing or correctness runs.
  if (!(dbg & 64) || g_ws.epoch == 0) g_ws.epoch += 1;
  if (g_ws.epoch == 0) { g_ws.flags.zero_(); g_ws.epoch = 1; }
  p.epoch = g_ws.epoch;
  uint64_t* fl = reinterpret_cast<uint64_t*>(g_ws.flags.data_ptr<int64_t>());
  const int64_t fstride = (int64_t)g_ws.max_chunks * HV;
  p.fK1 = fl; p.fNP = fl + fstride; p.fBP = fl + 2 * fstride; p.fS = fl + 3 * fstride; p.fM = fl + 4 * fstride;

  const auto BF = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  const auto F32 = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  const auto SW = CU_TENSOR_MAP_SWIZZLE_128B;
  const int64_t ws_rows_c = (int64_t)g_ws.max_chunks * HV * C;
  const int64_t ws_rows_d = (int64_t)g_ws.max_chunks * HV * D;
  CUtensorMap tm_q = make_tmap_2d(p.q, BF, 2, HQ * D, total, 64, C, SW);
  CUtensorMap tm_k = make_tmap_2d(p.k, BF, 2, HQ * D, total, 64, C, SW);
  CUtensorMap tm_v = make_tmap_2d(p.v, BF, 2, HV * D, total, 64, C, SW);
  CUtensorMap tm_tk = make_tmap_2d(p.Tk, BF, 2, C, ws_rows_c, 64, C, SW);
  CUtensorMap tm_tv = make_tmap_2d(p.Tv, BF, 2, C, ws_rows_c, 64, C, SW);
  CUtensorMap tm_nt = make_tmap_2d(p.NT, BF, 2, D, ws_rows_d, 64, D, SW);
  CUtensorMap tm_bt = make_tmap_2d(p.BT, BF, 2, D, ws_rows_d, 64, D, SW);
  CUtensorMap tm_sc = make_tmap_2d(p.Sc, BF, 2, D, ws_rows_d, 64, D, SW);
  CUtensorMap tm_np = make_tmap_2d(p.NP, BF, 2, D, ws_rows_d, 64, D, SW);
  CUtensorMap tm_bp = make_tmap_2d(p.BP, BF, 2, D, ws_rows_d, 64, D, SW);
  CUtensorMap tm_mp = make_tmap_2d(p.Mp, BF, 2, D, ws_rows_c, 64, C, SW);
  CUtensorMap tm_pup = make_tmap_2d(p.PUp, BF, 2, D, ws_rows_c, 64, C, SW);
  CUtensorMap tm_up = make_tmap_2d(p.Up, BF, 2, D, ws_rows_c, 64, C, SW);
  CUtensorMap tm_gs = make_tmap_2d(p.Gs, F32, 4, C, (int64_t)g_ws.max_chunks * HV, C, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  CUtensorMap tm_out = make_tmap_2d(p.out, BF, 2, HV * D, total, 64, C, SW);
  CUtensorMap tm_a = make_tmap_2d(p.a, BF, 2, HV, total, HV, C, CU_TENSOR_MAP_SWIZZLE_NONE);
  CUtensorMap tm_b = make_tmap_2d(p.b, BF, 2, HV, total, HV, C, CU_TENSOR_MAP_SWIZZLE_NONE);
  const int max_groups = max_chunks / G + num_seqs;
  CUtensorMap tm_so = make_tmap_2d(p.new_state, F32, 4, D, (int64_t)num_seqs * HV * D, 32, D, SW);
  CUtensorMap tm_si = p.state ? make_tmap_2d(p.state, F32, 4, D, (int64_t)num_seqs * HV * D, 32, D, SW) : tm_so;

  static bool attr_set = false;
  if (!attr_set) {
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, q.get_device());
    cudaFuncSetAttribute(k1_prep, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM);
    cudaFuncSetAttribute(k15_compose<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, K15_SMEM_N);
    cudaFuncSetAttribute(k15_compose<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, K15_SMEM_B);
    cudaFuncSetAttribute(k2_chain, cudaFuncAttributeMaxDynamicSharedMemorySize, K2_SMEM);
    cudaFuncSetAttribute(k3_main, cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM);
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  bool pdl = true;
  auto launch = [&](auto kernel, dim3 grid, int block, int smem, auto... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = dim3(block); cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr; cfg.numAttrs = (pdl && !(p.dbg & 16)) ? 1 : 0;   // dbg 16: no PDL (true per-kernel timing)
    cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args...);
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
  };
  // Launch order (all dependencies are flag-based; PDL only gates CTA dispatch):
  //   K2 first, in plain stream order (after the previous call has fully completed): its 16 chain CTAs need whole SMs
  //   and sit on them from the start.  K1 (parallel prep) then K15 N/B (dispatched once all K1 CTAs started: they
  //   compose early groups next to K1's last wave), K3a1 fills the remaining holes, K3a2 and K3b follow.
  pdl = false;
  launch(k2_chain, dim3(HV, num_seqs), K2_THREADS, K2_SMEM, p, tm_np, tm_bp, tm_sc, tm_nt, tm_bt, tm_so, tm_si);
  pdl = true;
  const int n3 = max_chunks * HV;
  // K1 persistent (3 CTAs per SM; GDN_K1_GRID overrides)
  static int k1_cap = -1;
  if (k1_cap < 0) { const char* e = getenv("GDN_K1_GRID"); k1_cap = e ? atoi(e) : 0; }
  // 3 CTAs per SM on the SMs not held by the chain CTAs (K2 is resident first): every K1 CTA starts at once
  const int k1_grid = std::min(n3, k1_cap > 0 ? k1_cap : 3 * std::max(1, g_num_sms - HV * num_seqs));
  launch(k1_prep, dim3(k1_grid), NTHREADS, K1_SMEM, p, tm_k, tm_v, tm_tk, tm_up, tm_nt, tm_bt, tm_a, tm_b, n3);
  const int n_items = max_groups * HV;
  static int k15_cap = -1;
  if (k15_cap < 0) { const char* e = getenv("GDN_K15_GRID"); k15_cap = e ? atoi(e) : 0; }
  const int k15_grid = k15_cap > 0 ? std::min(n_items, k15_cap) : n_items;
  // Both K15 parts request 130 KB of smem (they use 65 / 129 KB): exactly one 66 KB K3a1 CTA then fits next to a K15
  // CTA (two per N-part SM stretched the compositions from 1.7 to 2.4-3 us; none at all starves the K3 work).
  // GDN_K15_SMEM_KB overrides (experiments).
  static int k15_kb = -1;
  if (k15_kb < 0) { const char* e = getenv("GDN_K15_SMEM_KB"); k15_kb = e ? atoi(e) : 130; }
  const int smem_n = std::max(k15_kb * 1024, K15_SMEM_N), smem_b = std::max(k15_kb * 1024, K15_SMEM_B);
  if (k15_kb > 0) {
    cudaFuncSetAttribute(k15_compose<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_n);
    cudaFuncSetAttribute(k15_compose<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_b);
  }
  launch(k15_compose<0>, dim3(k15_grid), K15_THREADS, smem_n, p, tm_nt, tm_bt, tm_np, tm_bp, n_items);
  launch(k15_compose<1>, dim3(k15_grid), K15_THREADS, smem_b, p, tm_nt, tm_bt, tm_np, tm_bp, n_items);
  if (!(dbg & 32)) {
    // K3a1 items: persistent, 3 CTAs per SM (66 KB, TMEM 128); then the K3b output items, one CTA each (98 KB, TMEM 256).
    // GDN_K3_GRID = <k3a1 grid>[,<k3b grid>] overrides (0 = default).
    static int k3_cap0 = -1, k3_cap1 = 0;
    if (k3_cap0 < 0) { const char* e = getenv("GDN_K3_GRID"); k3_cap0 = e ? atoi(e) : 0; const char* f = e ? strchr(e, ',') : nullptr; k3_cap1 = f ? atoi(f + 1) : 0; }
    const int g0 = std::min(n3, k3_cap0 > 0 ? k3_cap0 : 3 * g_num_sms);
    const int g1 = std::min(n3, k3_cap1 > 0 ? k3_cap1 : 2 * g_num_sms);   // persistent: 2 per SM (measured best)
    launch(k3_main, dim3(g0), K3_THREADS, K3A_SMEM, p, tm_q, tm_k, tm_tk, tm_up, tm_gs, tm_mp, tm_pup, tm_sc, tm_np, tm_bp, tm_nt, tm_bt, tm_out, n3, 0);
    launch(k3_main, dim3(g1), K3_THREADS, K3_SMEM, p, tm_q, tm_k, tm_tk, tm_up, tm_gs, tm_mp, tm_pup, tm_sc, tm_np, tm_bp, tm_nt, tm_bt, tm_out, n3, 1);
  }
}

// debug access to the workspace
std::vector<torch::Tensor> debug_workspace() {
  return {g_ws.Tk, g_ws.Tv, g_ws.NT, g_ws.BT, g_ws.Sc, g_ws.gamma, g_ws.ts, g_ws.NP, g_ws.BP, g_ws.gamma_p};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_prefill", &gdn_prefill);
  m.def("debug_workspace", &debug_workspace);
}
