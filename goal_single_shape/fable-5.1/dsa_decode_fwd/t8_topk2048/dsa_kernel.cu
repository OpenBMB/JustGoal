// DSA sparse attention forward (decode), B200 / sm_100a.
//
// Grid = num_tokens x 32 CTAs of 8 warps; CTA (t, c) owns rows [64c, 64c+64) of
// token t's top-k list (empty CTAs only publish "no data"). Two CTAs are
// resident per SM so the whole grid is co-resident and CTAs may spin on each
// other. Q (16 x 576) and the chunk's KV rows are fetched with 1D TMA bulk
// copies (one mbarrier); S = Q K^T runs on mma.sync bf16 tensor cores with the
// 8 warps split 4 (contraction slices) x 2 (row halves), O = P V with one
// 64-dim slice per warp, fragment loads software-pipelined one step ahead.
// The chunk's softmax statistics (m, l) and normalised P V partial (bf16) are
// published to global scratch; every thread fences, then one thread adds to
// the token's arrival counter. CTA b then combines slice b (token b/32, head
// (b%32)/2, half of the 512 dims) from the live partials once all 32 chunk
// CTAs of its token have arrived. Counters are monotonic (epoch = counter /
// 32, read before this CTA's own arrival), so no reset is needed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int H = 16;         // query heads
constexpr int DC = 512;       // ckv head dim (also the value dim)
constexpr int DP = 64;        // kpe head dim
constexpr int DQ = DC + DP;   // 576
constexpr int TOPK = 2048;
constexpr int R = 64;                   // KV rows per CTA
constexpr int NCHUNK = TOPK / R;        // CTAs per token (32)
constexpr int SLICES = 32;              // reduction slices per token (== NCHUNK: CTA b reduces slice b)
constexpr int SLICE_DIMS = H * DC / SLICES;  // 256 output elements per slice
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS / 32;    // 8
constexpr int CTAS_PER_SM = 2;
constexpr int KSLICES = 4;                  // warps splitting the contraction
constexpr int KSPLIT = DQ / KSLICES;        // 144 dims per k-slice
constexpr int KSTEPS = KSPLIT / 16;         // 9 mma k-steps
constexpr int ROWHALF = R / 2;              // 32 rows per warp in S
constexpr int OSPLIT = DC / NWARPS;         // 64 output dims per warp in P V
constexpr int ONT = OSPLIT / 8;             // 8 n-tiles per warp
constexpr int SK_STRIDE = DC + 8;   // bf16 elements per smem K row (1040 B, conflict-free ldmatrix)
constexpr int SP_STRIDE = DP + 8;   // bf16 elements per smem kpe row (144 B)
constexpr int SQ_STRIDE = DQ + 8;   // bf16 elements per smem Q row (1168 B)
constexpr int SS_STRIDE = R + 4;    // fp32 elements per partial-S row (68)
constexpr int SR_STRIDE = R + 8;    // fp32 elements per reduced-S row (72: rows 8 banks apart)
constexpr int ROW_BYTES_K = DC * 2;  // 1024 B per ckv row
constexpr int ROW_BYTES_P = DP * 2;  // 128 B per kpe row
constexpr int Q_BYTES = H * (ROW_BYTES_K + ROW_BYTES_P);  // 18432 B per token
constexpr int MAX_T = 32;
constexpr int CTR_PAD = 16;  // u64 per counter slot (128 B apart -> different L2 lines)
constexpr int SP_STRIDE_BF = R + 8;  // bf16 elements per smem P row (144 B, conflict-free ldmatrix)
constexpr int SMEM_SR = H * SR_STRIDE * 4;      // reduced S (fp32) in the dead kpe tile
constexpr int SMEM_PB = H * SP_STRIDE_BF * 2;   // P (bf16) after it
constexpr int SMEM_ML = 2 * H * 4;              // per-head m and l

constexpr int SMEM_K = R * SK_STRIDE * 2;        // 66560
constexpr int SMEM_P = R * SP_STRIDE * 2;        // 9216
constexpr int SMEM_Q = H * SQ_STRIDE * 2;        // 18688 (the partial-S exchange aliases it)
constexpr int SMEM_S = KSLICES * H * SS_STRIDE * 4;  // 17408
constexpr int SMEM_IDX = 2 * R * 4;              // indices of both candidate chunks
constexpr int SMEM_MISC = 16;                    // slot broadcast
constexpr int SMEM_BAR = 16;
constexpr int SMEM_TOTAL = SMEM_K + SMEM_P + SMEM_Q + SMEM_IDX + SMEM_MISC + SMEM_BAR;

static_assert(KSPLIT % 16 == 0, "k-slice must be a multiple of the mma k");
static_assert(NWARPS == KSLICES * 2, "S phase uses 4 k-slices x 2 row halves");
static_assert(SLICE_DIMS == 2 * 128, "two output dims per thread for 128 reducing threads");
static_assert(SLICES == 2 * H && SLICES == NCHUNK, "a slice is half a head; one slice per CTA");
static_assert(SMEM_S <= SMEM_Q, "partial-S exchange must fit in the (dead) Q tile");
static_assert(SMEM_SR + SMEM_PB + SMEM_ML <= SMEM_P, "reduced S, P and stats must fit in the (dead) kpe tile");
static_assert(NWARPS * 2 == H, "softmax: two heads (rows) per warp");
static_assert(THREADS >= 2 * R + 2 * H, "one thread per bulk copy");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- mbarrier + 1D TMA bulk copy ----
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

// ---- tensor core fragments ----
// ldmatrix reads shared memory written by TMA/other threads: volatile + memory clobber pins it
// after the mbarrier wait / __syncthreads (a plain asm may legally be hoisted above them).
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                                  uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}
// D(16x8,f32) += A(16x16,bf16,row) * B(16x8,bf16,col)
__device__ __forceinline__ void mma_bf16_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
  return __bfloat1622float2(v);
}

// ---- inter-CTA signalling (monotonic 64-bit counters) ----
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void red_relaxed_add_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned int atom_add_u32(unsigned int* p, unsigned int v) {
  unsigned int old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
// Each warp orders its own prior global stores before the (relaxed) arrival add that follows.
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(p) : "memory");
  return v;
}
// Predicated L2 load of 4 bytes (no branch, so consecutive loads stay in flight together).
__device__ __forceinline__ uint32_t ldcg_u32_if(const void* p, bool pred) {
  uint32_t v = 0u;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "setp.ne.b32 P1, %2, 0;\n"
      "@P1 ld.global.cg.b32 %0, [%1];\n"
      "}\n"
      : "+r"(v)
      : "l"(p), "r"(static_cast<int>(pred))
      : "memory");
  return v;
}


constexpr int NPROF = 12;  // per-CTA phase timestamps when profiling (see dsa_kernel.py)
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long v;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(v));
  return v;
}
__device__ __forceinline__ unsigned int smid() {
  unsigned int v;
  asm volatile("mov.u32 %0, %%smid;\n" : "=r"(v));
  return v;
}
#define PROF(k)                                                               \
  do {                                                                        \
    if (prof != nullptr && threadIdx.x == 0) prof[b * NPROF + (k)] = gtimer(); \
  } while (0)

__global__ void __launch_bounds__(THREADS, CTAS_PER_SM)
dsa_fwd_kernel(const __nv_bfloat16* __restrict__ q_nope,   // [T][H][DC]
               const __nv_bfloat16* __restrict__ q_pe,     // [T][H][DP]
               const __nv_bfloat16* __restrict__ ckv,      // [num_pages*64][DC]
               const __nv_bfloat16* __restrict__ kpe,      // [num_pages*64][DP]
               const int* __restrict__ indices,            // [T][TOPK]
               float scale_log2,                           // sm_scale * log2(e)
               int T,
               __nv_bfloat16* __restrict__ part_o,         // [T][NCHUNK][H][DC] normalised partials
               float* __restrict__ part_m,                 // [T][NCHUNK][H]
               float* __restrict__ part_l,                 // [T][NCHUNK][H]
               unsigned long long* __restrict__ arrive,    // [T*CTR_PAD] monotonic CTA arrivals
               unsigned int* __restrict__ smctr,           // [num_sms*32] monotonic per-SM CTA counter (2 per launch)
               __nv_bfloat16* __restrict__ out,            // [T][H][DC]
               float* __restrict__ lse,                    // [T][H]
               unsigned long long* __restrict__ prof) {    // [grid][NPROF] or nullptr
  extern __shared__ __align__(128) unsigned char smem[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_K);
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_K + SMEM_P);
  float* sS = reinterpret_cast<float*>(smem + SMEM_K + SMEM_P);  // aliases sQ once S is done
  float* sR = reinterpret_cast<float*>(smem + SMEM_K);            // aliases sP once S is done
  __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_K + SMEM_SR);  // P (bf16), dead kpe tile
  float* sM = reinterpret_cast<float*>(smem + SMEM_K + SMEM_SR + SMEM_PB);
  float* sL = sM + H;
  int* sIdx2 = reinterpret_cast<int*>(smem + SMEM_K + SMEM_P + SMEM_Q);  // [2][R]
  int* sMisc = reinterpret_cast<int*>(smem + SMEM_K + SMEM_P + SMEM_Q + SMEM_IDX);
  const uint32_t bar = smem_u32(smem + SMEM_K + SMEM_P + SMEM_Q + SMEM_IDX + SMEM_MISC);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int frow = lane >> 2;
  const int fcol = (lane & 3) * 2;
  const int num_sms = gridDim.x / 2;
  const int sm = static_cast<int>(smid());
  const int NIT = T * NCHUNK;
  const unsigned long long t_start = (prof != nullptr) ? gtimer() : 0ull;

  // Two CTAs land on every SM (occupancy is exactly 2 and grid = 2 * #SMs); the per-SM counter
  // tells them apart. Chunk index i = slot * #SMs + sm in chunk-major order, so the low (busy)
  // chunks of all tokens go to slot 0, one per SM. Both candidates' indices are fetched up front.
  if (tid < 2 * R) {
    const int cand = tid / R;
    const int i = cand * num_sms + sm;
    sIdx2[tid] = (i < NIT) ? indices[static_cast<size_t>(i % T) * TOPK + (i / T) * R + (tid % R)] : -1;
  }
  if (tid == 0) {
    mbar_init(bar, 1);
    mbar_fence_init();
    sMisc[0] = static_cast<int>(atom_add_u32(smctr + static_cast<size_t>(sm) * 32, 1u) & 1u);
  }
  __syncthreads();
  const int slot = sMisc[0];
  const int i = slot * num_sms + sm;
  const bool has_chunk = i < NIT;
  const int t = has_chunk ? (i % T) : 0;
  const int c = has_chunk ? (i / T) : 0;
  const int* sIdx = sIdx2 + slot * R;
  unsigned long long* my_arrive = arrive + static_cast<size_t>(t) * CTR_PAD;
  const int b = i;  // used for profiling rows
  if (!has_chunk) {
    if (prof != nullptr && tid == 0) {
      prof[(NIT + (sm * 2 + slot) % (gridDim.x - NIT)) * NPROF + 10] = sm;  // idle CTA
    }
    return;
  }
  unsigned long long arrive_at_start = 0ull;  // < NCHUNK*(E+1): this CTA has not arrived yet
  if (tid == 0) arrive_at_start = ld_relaxed_u64(my_arrive);
  const int nvalid = __syncthreads_count(tid < R && sIdx[tid] >= 0);
  if (prof != nullptr && tid == 0) prof[b * NPROF + 0] = t_start;
  PROF(1);

  if (nvalid == 0) {
    if (tid < H) {
      part_m[(static_cast<size_t>(t) * NCHUNK + c) * H + tid] = -INFINITY;
      part_l[(static_cast<size_t>(t) * NCHUNK + c) * H + tid] = 0.f;
    }
  } else {
    // ---- gather: Q rows + the chunk's KV rows, one bulk copy each; zero-fill masked rows ----
    if (tid == 0) mbar_arrive_expect_tx(bar, static_cast<uint32_t>(nvalid) * (ROW_BYTES_K + ROW_BYTES_P) + Q_BYTES);
    __syncthreads();
    if (tid >= 2 * R && tid < 2 * R + H) {
      const int h = tid - 2 * R;
      tma_load_1d(smem_u32(sQ + h * SQ_STRIDE), q_nope + (static_cast<size_t>(t) * H + h) * DC, ROW_BYTES_K, bar);
    } else if (tid >= 2 * R + H && tid < 2 * R + 2 * H) {
      const int h = tid - 2 * R - H;
      tma_load_1d(smem_u32(sQ + h * SQ_STRIDE + DC), q_pe + (static_cast<size_t>(t) * H + h) * DP, ROW_BYTES_P, bar);
    }
    if (tid < R) {
      const int i = sIdx[tid];
      __nv_bfloat16* dst = sK + tid * SK_STRIDE;
      if (i >= 0) {
        tma_load_1d(smem_u32(dst), ckv + static_cast<size_t>(i) * DC, ROW_BYTES_K, bar);
      } else {
        uint4* d4 = reinterpret_cast<uint4*>(dst);
#pragma unroll 8
        for (int k = 0; k < ROW_BYTES_K / 16; ++k) d4[k] = make_uint4(0u, 0u, 0u, 0u);
      }
    } else if (tid < 2 * R) {
      const int r = tid - R;
      const int i = sIdx[r];
      __nv_bfloat16* dst = sP + r * SP_STRIDE;
      if (i >= 0) {
        tma_load_1d(smem_u32(dst), kpe + static_cast<size_t>(i) * DP, ROW_BYTES_P, bar);
      } else {
        uint4* d4 = reinterpret_cast<uint4*>(dst);
#pragma unroll
        for (int k = 0; k < ROW_BYTES_P / 16; ++k) d4[k] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    mbar_wait(bar, 0u);
    __syncthreads();
    PROF(2);

    // ---- partial S: warp = (k-slice ks, row half rh); 16 heads x 32 rows over 144 dims ----
    const int ks = warp & (KSLICES - 1);
    const int rh = warp >> 2;
    const int kbase = ks * KSPLIT;
    float sacc[ROWHALF / 8][4];
#pragma unroll
    for (int n = 0; n < ROWHALF / 8; ++n) sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;   // A fragment row for ldmatrix
    const int acol = (lane >> 4) * 8;                       // A fragment k offset
    const int brow0 = rh * ROWHALF + (lane & 7) + ((lane >> 4) << 3);  // B fragment row (n)
    const int bcol = ((lane >> 3) & 1) * 8;                 // B fragment k offset
    uint32_t qa[2][4], kb[2][ROWHALF / 16][4];
    auto load_step = [&](int kstep, int buf) {
      const int kd = kbase + kstep * 16;
      ldmatrix_x4(qa[buf][0], qa[buf][1], qa[buf][2], qa[buf][3], smem_u32(sQ + arow * SQ_STRIDE + kd + acol));
#pragma unroll
      for (int np = 0; np < ROWHALF / 16; ++np) {
        const int row = brow0 + np * 16;
        const uint32_t addr = (kd < DC) ? smem_u32(sK + row * SK_STRIDE + kd + bcol)
                                        : smem_u32(sP + row * SP_STRIDE + (kd - DC) + bcol);
        ldmatrix_x4(kb[buf][np][0], kb[buf][np][1], kb[buf][np][2], kb[buf][np][3], addr);
      }
    };
    load_step(0, 0);
#pragma unroll
    for (int kstep = 0; kstep < KSTEPS; ++kstep) {
      const int cur = kstep & 1;
      if (kstep + 1 < KSTEPS) load_step(kstep + 1, cur ^ 1);
#pragma unroll
      for (int np = 0; np < ROWHALF / 16; ++np) {
        mma_bf16_16816(sacc[2 * np], qa[cur], kb[cur][np][0], kb[cur][np][1]);
        mma_bf16_16816(sacc[2 * np + 1], qa[cur], kb[cur][np][2], kb[cur][np][3]);
      }
    }
    PROF(3);

    // ---- sum the 4 k-slice partials through smem (reusing the Q tile) ----
    __syncthreads();  // all warps are done reading sQ
    float* sSw = sS + ks * H * SS_STRIDE;
#pragma unroll
    for (int n = 0; n < ROWHALF / 8; ++n) {
      const int col = rh * ROWHALF + n * 8 + fcol;
      *reinterpret_cast<float2*>(sSw + frow * SS_STRIDE + col) = make_float2(sacc[n][0], sacc[n][1]);
      *reinterpret_cast<float2*>(sSw + (frow + 8) * SS_STRIDE + col) = make_float2(sacc[n][2], sacc[n][3]);
    }
    __syncthreads();

    // all 256 threads sum the 4 k-slices once (1024 elements, 4 per thread) into sR
#pragma unroll
    for (int k = 0; k < (H * R) / THREADS; ++k) {
      const int e = tid + k * THREADS;
      const int row = e / R, col = e % R;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < KSLICES; ++w) v += sS[(w * H + row) * SS_STRIDE + col];
      sR[row * SR_STRIDE + col] = v;
    }
    __syncthreads();

    // softmax once: warp w owns heads 2w, 2w+1; lane owns columns lane and lane+32
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int row = warp * 2 + hh;
      float s0 = sR[row * SR_STRIDE + lane];
      float s1 = sR[row * SR_STRIDE + lane + 32];
      s0 = (sIdx[lane] >= 0) ? s0 * scale_log2 : -INFINITY;
      s1 = (sIdx[lane + 32] >= 0) ? s1 * scale_log2 : -INFINITY;
      float m = fmaxf(s0, s1);
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      const float p0 = exp2f(s0 - m);
      const float p1 = exp2f(s1 - m);
      float l = p0 + p1;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      sPb[row * SP_STRIDE_BF + lane] = __float2bfloat16_rn(p0);
      sPb[row * SP_STRIDE_BF + lane + 32] = __float2bfloat16_rn(p1);
      if (lane == 0) {
        sM[row] = m;
        sL[row] = l;
      }
    }
    __syncthreads();

    // P as bf16 A fragments (16 heads x 16 rows per k-step) via ldmatrix
    uint32_t pa[R / 16][4];
#pragma unroll
    for (int kk = 0; kk < R / 16; ++kk)
      ldmatrix_x4(pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], smem_u32(sPb + arow * SP_STRIDE_BF + kk * 16 + acol));
    const float mrow0 = sM[frow], mrow1 = sM[frow + 8];
    const float lrow0 = sL[frow], lrow1 = sL[frow + 8];

    // ---- O partial = P V[:, warp slice] (16 heads x 64 dims), V fragments loaded one pair ahead ----
    float oacc[ONT][4];
#pragma unroll
    for (int n = 0; n < ONT; ++n) oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f;
    const int vrow0 = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int vcol0 = warp * OSPLIT + (lane >> 4) * 8;
    uint32_t vb[2][4];
    constexpr int NPAIR = ONT / 2;
    auto load_v = [&](int step, int buf) {  // step = kk * NPAIR + np
      const int kk = step / NPAIR, np = step % NPAIR;
      ldmatrix_x4_trans(vb[buf][0], vb[buf][1], vb[buf][2], vb[buf][3],
                        smem_u32(sK + (kk * 16 + vrow0) * SK_STRIDE + vcol0 + np * 16));
    };
    constexpr int VSTEPS = (R / 16) * NPAIR;
    load_v(0, 0);
#pragma unroll
    for (int st = 0; st < VSTEPS; ++st) {
      const int cur = st & 1;
      if (st + 1 < VSTEPS) load_v(st + 1, cur ^ 1);
      const int kk = st / NPAIR, np = st % NPAIR;
      mma_bf16_16816(oacc[2 * np], pa[kk], vb[cur][0], vb[cur][1]);
      mma_bf16_16816(oacc[2 * np + 1], pa[kk], vb[cur][2], vb[cur][3]);
    }
    PROF(4);

    // ---- publish: normalised partial (bf16) + stats; l >= 1 since the row max contributes exp2(0) ----
    const float inv0 = 1.f / lrow0, inv1 = 1.f / lrow1;
    __nv_bfloat16* my_o = part_o + (static_cast<size_t>(t) * NCHUNK + c) * H * DC;
#pragma unroll
    for (int n = 0; n < ONT; ++n) {
      const int d = warp * OSPLIT + n * 8 + fcol;
      *reinterpret_cast<uint32_t*>(my_o + frow * DC + d) = pack_bf16x2(oacc[n][0] * inv0, oacc[n][1] * inv0);
      *reinterpret_cast<uint32_t*>(my_o + (frow + 8) * DC + d) = pack_bf16x2(oacc[n][2] * inv1, oacc[n][3] * inv1);
    }
    if (warp == 0 && (lane & 3) == 0) {
      float* my_m = part_m + (static_cast<size_t>(t) * NCHUNK + c) * H;
      float* my_l = part_l + (static_cast<size_t>(t) * NCHUNK + c) * H;
      my_m[frow] = mrow0;
      my_m[frow + 8] = mrow1;
      my_l[frow] = lrow0;
      my_l[frow + 8] = lrow1;
    }
  }
  // ---- arrive: CTA barrier, then one thread fences (cumulative over the barrier) and adds ----
  __syncthreads();
  if (tid == 0) {
    fence_acq_rel_gpu();
    red_relaxed_add_u64(my_arrive, 1ull);
  }
  PROF(5);

  // ---- combine slice c of token t: head h, dims [(c&1)*256, +256), 2 dims per thread (128 threads) ----
  const int h = c >> 1;
  if (tid == 0) {
    const unsigned long long target = (arrive_at_start / NCHUNK + 1ull) * NCHUNK;
    while (ld_acquire_u64(my_arrive) < target) {
    }
  }
  __syncthreads();
  PROF(6);
  if (tid < 128) {
    const int d0 = (c & 1) * (DC / 2) + tid * 2;
    const float* pm = part_m + static_cast<size_t>(t) * NCHUNK * H + h;
    const float* pl = part_l + static_cast<size_t>(t) * NCHUNK * H + h;
    const float m_i = ld_cg_f32(pm + lane * H);
    const float l_i = ld_cg_f32(pl + lane * H);
    const bool live_i = m_i > -INFINITY;
    const uint32_t mask = __ballot_sync(0xffffffffu, live_i);
    const __nv_bfloat16* po = part_o + (static_cast<size_t>(t) * NCHUNK * H + h) * DC + d0;
    uint32_t oc[NCHUNK];
#pragma unroll
    for (int i = 0; i < NCHUNK; ++i) oc[i] = ldcg_u32_if(po + static_cast<size_t>(i) * H * DC, (mask >> i) & 1u);
    float M = m_i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    const float w_i = live_i ? exp2f(m_i - M) * l_i : 0.f;
    float L = w_i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
    for (int i = 0; i < NCHUNK; ++i) {
      const float w = __shfl_sync(0xffffffffu, w_i, i);
      const float2 f = unpack_bf16x2(oc[i]);
      acc0 += w * f.x;
      acc1 += w * f.y;
    }
    if (mask != 0u) {
      const float inv = 1.f / L;
      acc0 *= inv;
      acc1 *= inv;
    } else {
      acc0 = acc1 = 0.f;
    }
    *reinterpret_cast<uint32_t*>(out + (static_cast<size_t>(t) * H + h) * DC + d0) = pack_bf16x2(acc0, acc1);
    if ((c & 1) == 0 && tid == 0) lse[t * H + h] = (mask != 0u) ? (M + log2f(L)) : -INFINITY;
  }
  PROF(7);
  if (prof != nullptr && tid == 0) {
    prof[b * NPROF + 10] = sm;
    prof[b * NPROF + 11] = static_cast<unsigned long long>(nvalid > 0);
  }
}

__global__ void noop_kernel() {}

int g_num_sms = 0;

void ensure_launchable(int grid_blocks) {
  static bool configured = false;
  if (configured) return;
  cudaError_t err = cudaFuncSetAttribute(dsa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
  int dev = 0;
  cudaGetDevice(&dev);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
  int per_sm = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dsa_fwd_kernel, THREADS, SMEM_TOTAL);
  TORCH_CHECK(err == cudaSuccess, "occupancy query failed: ", cudaGetErrorString(err));
  // Exactly two CTAs per SM: the spin barrier needs the whole grid resident, and the per-SM
  // slot counter needs every SM to receive exactly two CTAs per launch.
  TORCH_CHECK(per_sm == CTAS_PER_SM, "kernel must fit exactly ", CTAS_PER_SM, " CTAs per SM, got ", per_sm);
  g_num_sms = num_sms;
  configured = true;
}

}  // namespace

std::vector<torch::Tensor> run(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
                               torch::Tensor kpe_cache, torch::Tensor sparse_indices, double sm_scale,
                               torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
                               torch::Tensor arrive, torch::Tensor smctr, torch::Tensor prof, bool do_prof) {
  TORCH_CHECK(q_nope.is_cuda() && q_nope.is_contiguous() && q_nope.dtype() == torch::kBFloat16);
  TORCH_CHECK(q_pe.is_contiguous() && ckv_cache.is_contiguous() && kpe_cache.is_contiguous());
  TORCH_CHECK(sparse_indices.is_contiguous() && sparse_indices.dtype() == torch::kInt32);
  const int T = q_nope.size(0);
  TORCH_CHECK(T >= 1 && T <= MAX_T, "num_tokens must be in [1, ", MAX_T, "]");
  TORCH_CHECK(q_nope.size(1) == H && q_nope.size(2) == DC && q_pe.size(2) == DP);
  TORCH_CHECK(ckv_cache.size(1) == 64 && ckv_cache.size(2) == DC && kpe_cache.size(2) == DP);
  TORCH_CHECK(sparse_indices.size(0) == T && sparse_indices.size(1) == TOPK);
  TORCH_CHECK(part_o.dtype() == torch::kBFloat16 && part_o.numel() >= static_cast<int64_t>(T) * NCHUNK * H * DC);
  TORCH_CHECK(part_m.numel() >= T * NCHUNK * H && part_l.numel() >= T * NCHUNK * H);
  TORCH_CHECK(arrive.dtype() == torch::kInt64 && arrive.numel() >= T * CTR_PAD);

  auto out = torch::empty({T, H, DC}, q_nope.options());
  auto lse = torch::empty({T, H}, q_nope.options().dtype(torch::kFloat32));
  ensure_launchable(0);
  const int G = 2 * g_num_sms;
  TORCH_CHECK(smctr.dtype() == torch::kInt32 && smctr.numel() >= g_num_sms * 32);
  TORCH_CHECK(!do_prof || prof.numel() >= static_cast<int64_t>(std::max(G, T * NCHUNK)) * NPROF);
  const float scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
  auto stream = at::cuda::getCurrentCUDAStream();
  dsa_fwd_kernel<<<G, THREADS, SMEM_TOTAL, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()), sparse_indices.data_ptr<int>(), scale_log2, T,
      reinterpret_cast<__nv_bfloat16*>(part_o.data_ptr()), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(arrive.data_ptr()),
      reinterpret_cast<unsigned int*>(smctr.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      lse.data_ptr<float>(), do_prof ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "dsa_fwd_kernel launch failed: ", cudaGetErrorString(err));
  return {out, lse};
}

// Empty kernel launch: calibrates the fixed overhead of the timing method.
void run_noop(int grid, int smem_bytes) {
  static int configured_for = -1;
  if (smem_bytes > 48 * 1024 && configured_for < smem_bytes) {
    cudaFuncSetAttribute(noop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured_for = smem_bytes;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  noop_kernel<<<grid, 128, smem_bytes, stream>>>();
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "noop launch failed: ", cudaGetErrorString(err));
}

int num_chunks() { return NCHUNK; }
int num_prof() { return NPROF; }
int counter_pad() { return CTR_PAD; }
int grid_for(int num_tokens) {
  ensure_launchable(0);
  return std::max(2 * g_num_sms, num_tokens * NCHUNK);
}
int num_sms() {
  ensure_launchable(0);
  return g_num_sms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA sparse attention forward");
  m.def("run_noop", &run_noop, "empty kernel launch (grid, dynamic smem bytes)");
  m.def("num_chunks", &num_chunks, "chunks per token");
  m.def("num_prof", &num_prof, "profile stamps per CTA");
  m.def("counter_pad", &counter_pad, "u64 slots per arrival counter");
  m.def("grid_for", &grid_for, "profile rows for a token count");
  m.def("num_sms", &num_sms, "SM count (per-SM counter slots)");
}
