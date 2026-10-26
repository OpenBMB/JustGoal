// DSA sparse paged attention (decode) — v16: split-KV, 512 chunk blocks (4 per SM), fence-free progressive combine.
//  * grid = T x 64 blocks (one 32-row chunk of index slots per block), 128 threads, 4 blocks / SM, single launch, no clusters.
//  * block b -> (token b % T, chunk b / T) so that the (typically prefix-valid) active chunks spread over all SMs.
//  * work donation (SM balance): an active block counts itself on its SM (atomic per-SM counter); the 2nd+ active block
//    on an SM donates its chunk to a global queue (it still issues its loads); empty blocks on idle SMs claim donated
//    chunks (one per idle SM) and compute them; an unclaimed donation is reclaimed by the donor after a deadline (CAS), so
//    every chunk is computed exactly once.
//  * indices first (critical path); non-empty chunks then issue Q (two dense TMA bulk copies: q_nope[t] 16 KB, q_pe[t]
//    2 KB), the 32 ckv rows (one 1 KB TMA bulk copy each, zero row for invalid slots) and the kpe rows (16 B cp.async,
//    zero-filled). Empty chunks publish immediately.
//  * S = Q K^T: k-split over the 4 warps (Q fragments in registers), per-n-tile reduction (warp w owns rows 8w..8w+7),
//    softmax statistics exchanged through smem, P fragments assembled from smem, O = P V with warp w owning dims 128w..+127.
//  * bf16 partials (fragment layout, 128 B lines) + (m, l) records go to global; publish = barrier + red.release of a
//    per-token completion bit (empty bit first).
//  * combine (block (t, c) owns 16 heads x 8 dims of token t): warp 0 polls the token mask with relaxed loads (no acquire /
//    fence anywhere in the consumer: the producers' release puts the lines in L2 before the bit and the consumer reads them
//    through L2-only loads) and publishes the snapshot in smem; every warp loads the lines of its fresh chunks (q = 4i + w)
//    straight into registers and merges them in batches (one rescale + one ex2 per chunk) into a running (M, L, acc) state;
//    cross-warp merge, output / lse store.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace {

constexpr int H = 16;
constexpr int D = 512;
constexpr int DP = 64;
constexpr int DK = D + DP;          // 576
constexpr int TOPK = 2048;
constexpr int CS = 32;              // rows per chunk
constexpr int NCHUNK = TOPK / CS;   // 64 chunks per token
constexpr int NTHREADS = 128;
constexpr int ROW_BYTES = DK * 2;         // 1152
constexpr int ROW_STRIDE = ROW_BYTES + 16;   // 1168 B padded rows (conflict-free ldmatrix)
constexpr int Q_OFF = 0;                  // q_nope[t] dense (16 x 1024 B), then q_pe[t] dense (16 x 128 B)
constexpr int QPE_OFF = H * D * 2;        // 16384
constexpr int Q_SMEM = H * ROW_STRIDE;    // 18688 (>= 18432 used); reused as scratch after the Q fragments are extracted
constexpr int K_OFF = Q_OFF + Q_SMEM;
constexpr int K_SMEM = CS * ROW_STRIDE;   // 37376
constexpr int IDX_OFF = K_OFF + K_SMEM;
constexpr int IDX_SMEM = CS * 4;
constexpr int MISC_OFF = IDX_OFF + IDX_SMEM;   // mask copy (2 x u64), qbar, kbar
constexpr int MISC_SMEM = 64;
constexpr int STX_OFF = K_OFF;                       // combine stash (K region is free by then): [64 q][2 halves][32] u32 = 16 KB
constexpr int STR_OFF = K_OFF + 64 * 256;            // [64 q][16 heads] float2 = 8 KB
static_assert(64 * 256 + 64 * 128 <= K_SMEM, "stash must fit in the K region");
constexpr int SMEM_TOTAL = MISC_OFF + MISC_SMEM;    // 56256 B -> 4 blocks per SM
// scratch layout inside the Q region (after fragment extraction)
constexpr int SRED_OFF = Q_OFF;                       // [4 warps][4 n-tiles][32 lanes] float4 = 8192 B
constexpr int MX_OFF = SRED_OFF + 8192;               // [4 warps][16] float (row max per head)
constexpr int LS_OFF = MX_OFF + 256;                  // [4 warps][16] float (row sum per head)
constexpr int PB_OFF = LS_OFF + 256;                  // [4 warps][2][32] u32 (bf16x2 P fragments) = 1024 B
constexpr int CRED_OFF = PB_OFF + 1024;               // combine reduction [10 fields][4 warps][32] float = 5120 B
static_assert(CRED_OFF + 5120 <= Q_SMEM, "scratch must fit in the Q region");
constexpr int MASK_STRIDE = 16;   // u64 words between per-token mask pairs (128 B each)
constexpr int DW_STRIDE = 256 + 2 + 512;   // donation workspace ints per launch parity
#ifndef DSA_DONATE_MIN
#define DSA_DONATE_MIN 99           // the k-th (0-based) active block on an SM donates when k >= this (99 = off)
#endif
#ifndef DSA_DONATE_WAIT_NS
#define DSA_DONATE_WAIT_NS 3000
#endif
#ifndef DSA_CLAIM_DEADLINE_NS
#define DSA_CLAIM_DEADLINE_NS 3500
#endif

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#ifdef DSA_TRACE
#define TRACE(i) do { if (tid == 0) trace[(size_t)b * 16 + (i)] = gtimer(); } while (0)
#else
#define TRACE(i) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(void* bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(void* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(void* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s(void* dst, const void* src, uint32_t bytes, void* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void ld_acquire_u64x2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.acquire.gpu.global.v2.b64 {%0,%1}, [%2];\n" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ void tma_bulk_g2s_hint(void* dst, const void* src, uint32_t bytes, void* bar, unsigned long long pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
}
__device__ __forceinline__ unsigned long long policy_evict_first() {
  unsigned long long p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async16_pred(void* dst, const void* src, bool pred) {
  asm volatile(
      "{\n"
      ".reg .pred q;\n"
      "setp.ne.b32 q, %2, 0;\n"
      "@q cp.async.cg.shared.global [%0], [%1], 16;\n"
      "}\n" ::"r"(smem_u32(dst)), "l"(src), "r"((int)pred) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_group1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}
__device__ __forceinline__ void ld_relaxed_u64x2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0,%1}, [%2];\n" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ int ld_relaxed_s32(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_or_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.relaxed.gpu.global.or.b64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void red_release_or_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.or.b64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float lg2_approx(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t ld_cg_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_cg_u4(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ld_cg_u2(const uint2* p) {
  uint2 v;
  asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float2 ld_cg_f2(const float2* p) {
  float2 v;
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p) : "memory");
  return v;
}

// online-softmax merge of one (m, l, x) record into (M, L, a)
__device__ __forceinline__ void merge1(float& M, float& L, float2& a, float m, float l, float2 x) {
  const float Mn = fmaxf(M, m);
  const float al = ex2_approx(M - Mn);   // 0 when M == -inf (Mn finite)
  const float be = ex2_approx(m - Mn);
  L = L * al + l * be;
  a.x = a.x * al + x.x * be;
  a.y = a.y * al + x.y * be;
  M = Mn;
}

struct PermArgs { int p[512]; };   // placement-aware chunk permutation, passed by value (kernel parameter space)

__global__ void __launch_bounds__(NTHREADS, 4)
dsa_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
           const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
           const int* __restrict__ sidx, __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
           const __nv_bfloat16* __restrict__ zero_row,
           uint32_t* __restrict__ part, float2* __restrict__ rec, unsigned long long* __restrict__ masks,
           int* __restrict__ dws, const __grid_constant__ PermArgs perm, int* __restrict__ probe_out,
           int T, float scale_log2, unsigned gen, long long* __restrict__ trace) {
  extern __shared__ __align__(128) unsigned char smem[];
  if (probe_out != nullptr) {   // placement probe: record this block's SM and exit (same launch footprint as the real run)
    if (threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); probe_out[blockIdx.x] = (int)sm; }
    return;
  }
  unsigned char* qs = smem + Q_OFF;
  unsigned char* ks = smem + K_OFF;
  int* idx_s = reinterpret_cast<int*>(smem + IDX_OFF);
  volatile unsigned long long* mask_s = reinterpret_cast<unsigned long long*>(smem + MISC_OFF);   // [0] done, [1] empty
  unsigned long long* qbar = reinterpret_cast<unsigned long long*>(smem + MISC_OFF) + 2;
  unsigned long long* kbar = qbar + 1;
  volatile int* ctl = reinterpret_cast<volatile int*>(smem + MISC_OFF + 32);   // donation control words
  uint32_t* stx = reinterpret_cast<uint32_t*>(smem + STX_OFF);
  volatile unsigned long long* snap_s = reinterpret_cast<volatile unsigned long long*>(smem + STX_OFF);   // [4 warps][2] poll snapshots (combine phase; K region is free then)
  float2* str_ = reinterpret_cast<float2*>(smem + STR_OFF);
  float* sred = reinterpret_cast<float*>(smem + SRED_OFF);
  float* mx_s = reinterpret_cast<float*>(smem + MX_OFF);
  float* ls_s = reinterpret_cast<float*>(smem + LS_OFF);
  uint32_t* pb_s = reinterpret_cast<uint32_t*>(smem + PB_OFF);
  float* cred = reinterpret_cast<float*>(smem + CRED_OFF);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int cc = lane & 3;
  const int b = blockIdx.x;
  const int kid = perm.p[b];         // chunk / unit id (placement-aware permutation of c * T + t, built once at init)
  const int t = kid % T;             // token
  const int c = kid / T;             // chunk / combine unit
  unsigned long long* mask = masks + ((size_t)(gen & 1) * T + t) * MASK_STRIDE;
  TRACE(0);
#ifdef DSA_TRACE
  if (tid == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); trace[(size_t)b * 16 + 15] = sm; }
#endif

  // ---- 1. indices (critical path) ----
  int my_idx = -1;
  if (tid < CS) my_idx = __ldg(sidx + (size_t)t * TOPK + c * CS + tid);
  if (tid == 0) {
    mask_s[0] = 0ull; mask_s[1] = 0ull;
    mbar_init(qbar, 1);
    mbar_init(kbar, 1);
    fence_mbar_init();
    if (b == 0) {   // reset the other parity's masks for the next launch
      for (int i = 0; i < T; ++i) {
        masks[((size_t)((gen + 1) & 1) * T + i) * MASK_STRIDE] = 0ull;
        masks[((size_t)((gen + 1) & 1) * T + i) * MASK_STRIDE + 1] = 0ull;
      }
    }
  }
  if (b == 0) for (int i = tid; i < DW_STRIDE; i += NTHREADS) dws[(size_t)((gen + 1) & 1) * DW_STRIDE + i] = 0;
  if (tid < CS) idx_s[tid] = my_idx;
  const int nvalid = __syncthreads_count(tid < CS && my_idx >= 0);
  TRACE(1);
  const size_t chunk_id = (size_t)t * NCHUNK + c;

  // ---- donation bookkeeping (parity-selected workspace: sm_cnt[256], dq_cnt, dq_claim, dq[512]) ----
  int* dw = dws + (size_t)(gen & 1) * DW_STRIDE;
  int* sm_cnt = dw;
  int* dq_cnt = dw + 256;
  int* dq_claim = dw + 257;
  int* dq = dw + 258;
  unsigned smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  const long long t_start = gtimer();
  int pt = t, pc = c;          // chunk this block computes (may change by donation / claim)
  bool do_compute = false;

  auto issue_loads = [&](int lt) {
    // K rows first (DRAM latency is the long pole), then Q rows (mostly L2 hits); everything into padded 1168 B rows
    if (tid == 0) {
#if DSA_QCP == 0
      mbar_expect_tx(qbar, (uint32_t)H * (D * 2));
#endif
      mbar_expect_tx(kbar, (uint32_t)CS * (D * 2));
    }
#ifndef DSA_KEF
#define DSA_KEF 0
#endif
    if (lane < 8) {
      const int r = warp * 8 + lane;
      const int ix = idx_s[r];
#if DSA_KEF
      tma_bulk_g2s_hint(ks + r * ROW_STRIDE, (ix >= 0) ? ckv + (size_t)ix * D : zero_row, D * 2, kbar, policy_evict_first());
#else
      tma_bulk_g2s(ks + r * ROW_STRIDE, (ix >= 0) ? ckv + (size_t)ix * D : zero_row, D * 2, kbar);
#endif
    }
#ifndef DSA_QCP
#define DSA_QCP 0
#endif
#if DSA_QCP == 0
    if (tid < H) tma_bulk_g2s(qs + tid * ROW_STRIDE, q_nope + ((size_t)lt * H + tid) * D, D * 2, qbar);
#else
    // q_nope rows via cp.async (16 rows x 64 pieces = 8 per thread) -> the TMA queue only carries the K rows
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = tid + j * NTHREADS, r = e >> 6, pc = e & 63;
      cp_async16(qs + r * ROW_STRIDE + pc * 16, q_nope + ((size_t)lt * H + r) * D + pc * 8, 16);
    }
#endif
    {   // q_pe rows: 16 x 8 pieces of 16 B -> one per thread (group 0)
      const int r = tid >> 3, pc = tid & 7;
      cp_async16(qs + r * ROW_STRIDE + D * 2 + pc * 16, q_pe + ((size_t)lt * H + r) * DP + pc * 8, 16);
    }
    cp_async_commit();
#pragma unroll
    for (int j = 0; j < 2; ++j) {   // kpe rows: 32 x 8 pieces (group 1)
      const int e = tid + j * NTHREADS, r = e >> 3, pc = e & 7;
      const int ix = idx_s[r];
      cp_async16(ks + r * ROW_STRIDE + D * 2 + pc * 16, (ix >= 0) ? kpe + (size_t)ix * DP + pc * 8 : zero_row, (ix >= 0) ? 16 : 0);
    }
    cp_async_commit();
  };

  if (nvalid > 0) {
    // owner of an active chunk: register on this SM (returns the number of active blocks already here), issue the loads
    if (DSA_DONATE_MIN < 99) { if (tid == 0) ctl[0] = atomicAdd(sm_cnt + (int)smid, 1); }
    issue_loads(t);
    TRACE(2);
    int k = 0;
    if (DSA_DONATE_MIN < 99) { __syncthreads(); k = ctl[0]; }
    if (k < DSA_DONATE_MIN) {
      do_compute = true;
    } else {
      // donate the chunk; wait for a claim (or reclaim it after the deadline)
      if (tid == 0) {
        const int slot = atomicAdd(dq_cnt, 1);
        const int entry = (int)chunk_id + 1;
        atomicExch(dq + slot, entry);
        int decision = 0;
        for (;;) {
          const int v = ld_relaxed_s32(dq + slot);
          if (v == -1) { decision = 0; break; }                              // claimed by an idle block
          if (gtimer() - t_start > DSA_DONATE_WAIT_NS) {
            decision = (atomicCAS(dq + slot, entry, -2) == entry) ? 1 : 0;   // reclaim (unless a claim races in)
            break;
          }
          __nanosleep(100);
        }
        ctl[1] = decision;
#ifdef DSA_TRACE
        trace[(size_t)b * 16 + 11] = decision ? 2 : 1; trace[(size_t)b * 16 + 13] = gtimer();
#endif
      }
      __syncthreads();
      do_compute = ctl[1] != 0;
    }
  } else {
    // empty chunk: empty bit first, then (release) the completion bit; then try to claim a donated chunk if the SM is idle
    if (tid == 0) {
      red_or_u64(mask + 1, 1ull << c);
      red_release_or_u64(mask, 1ull << c);
      int got = -1;
      if (ld_relaxed_s32(sm_cnt + (int)smid) == 0 && atomicAdd(sm_cnt + (int)smid, 1) == 0) {
        for (;;) {
          const int dcnt = ld_relaxed_s32(dq_cnt);
          const int dcl = ld_relaxed_s32(dq_claim);
          if (dcnt > dcl) {
            const int ci = atomicAdd(dq_claim, 1);
            int v = 0;
            for (;;) {
              v = ld_relaxed_s32(dq + ci);
              if (v != 0 || gtimer() - t_start > DSA_CLAIM_DEADLINE_NS) break;
              __nanosleep(100);
            }
            if (v > 0 && atomicCAS(dq + ci, v, -1) == v) got = v - 1;
            break;   // a single attempt per idle block
          }
          if (gtimer() - t_start > DSA_CLAIM_DEADLINE_NS) break;
          __nanosleep(200);
        }
      }
      ctl[1] = got;
#ifdef DSA_TRACE
      if (got >= 0) { trace[(size_t)b * 16 + 11] = 3; trace[(size_t)b * 16 + 13] = gtimer(); }
#endif
    }
    __syncthreads();
    const int got = ctl[1];
    if (got >= 0) {
      pt = got / NCHUNK; pc = got - pt * NCHUNK;
      if (tid < CS) idx_s[tid] = __ldg(sidx + (size_t)pt * TOPK + pc * CS + tid);
      __syncthreads();
      issue_loads(pt);
      do_compute = true;
    }
  }

  if (do_compute) {
    const size_t pchunk = (size_t)pt * NCHUNK + pc;
    unsigned long long* pmask = masks + ((size_t)(gen & 1) * T + pt) * MASK_STRIDE;
    // ---- 3. Q fragments (warp w: k in [144w, 144w+144) = 9 k-steps; q_pe columns follow q_nope in the padded row) ----
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");   // Q pieces of this thread
#if DSA_QCP == 0
    mbar_wait(qbar, 0u);
#endif
    __syncthreads();                                           // Q pieces of every thread
    uint32_t qf[9][4];
    {
      const unsigned char* qa = qs + (lane & 15) * ROW_STRIDE + (lane >> 4) * 16 + warp * 9 * 32;
#pragma unroll
      for (int i = 0; i < 9; ++i) ldmatrix_x4(qf[i], qa + i * 32);
    }
    cp_async_wait_all();
    mbar_wait(kbar, 0u);
    __syncthreads();
    TRACE(3);

    // ---- 4. S = Q K^T, k-split ----
    float s[4][4], s2[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; s2[n][0] = s2[n][1] = s2[n][2] = s2[n][3] = 0.f; }
    {
      const int mi = lane >> 3;
      const unsigned char* kb = ks + ((mi >> 1) * 8 + (lane & 7)) * ROW_STRIDE + (mi & 1) * 16 + warp * 9 * 32;
      uint32_t bb0[2][4], bb1[2][4];
      ldmatrix_x4(bb0[0], kb);
      ldmatrix_x4(bb1[0], kb + 16 * ROW_STRIDE);
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        if (i + 1 < 9) {
          ldmatrix_x4(bb0[(i + 1) & 1], kb + (i + 1) * 32);
          ldmatrix_x4(bb1[(i + 1) & 1], kb + (i + 1) * 32 + 16 * ROW_STRIDE);
        }
        float (*acc)[4] = (i & 1) ? s2 : s;
        mma_bf16(acc[0], qf[i], bb0[i & 1][0], bb0[i & 1][1]);
        mma_bf16(acc[1], qf[i], bb0[i & 1][2], bb0[i & 1][3]);
        mma_bf16(acc[2], qf[i], bb1[i & 1][0], bb1[i & 1][1]);
        mma_bf16(acc[3], qf[i], bb1[i & 1][2], bb1[i & 1][3]);
      }
    }
    // Q region is free: every warp extracted its fragments before the barrier that preceded the S loop.
    TRACE(9);
#pragma unroll
    for (int n = 0; n < 4; ++n)
      *reinterpret_cast<float4*>(sred + ((warp * 4 + n) * 32 + lane) * 4) =
          make_float4(s[n][0] + s2[n][0], s[n][1] + s2[n][1], s[n][2] + s2[n][2], s[n][3] + s2[n][3]);
    __syncthreads();
    // warp w owns n-tile w (rows 8w..8w+7): sum the 4 k-slices
    float4 sv = *reinterpret_cast<const float4*>(sred + ((0 * 4 + warp) * 32 + lane) * 4);
#pragma unroll
    for (int w = 1; w < 4; ++w) {
      const float4 v = *reinterpret_cast<const float4*>(sred + ((w * 4 + warp) * 32 + lane) * 4);
      sv.x += v.x; sv.y += v.y; sv.z += v.z; sv.w += v.w;
    }
    TRACE(11);
    const int r0 = warp * 8 + 2 * cc;
    const bool v0 = idx_s[r0] >= 0, v1 = idx_s[r0 + 1] >= 0;
    const float s00 = v0 ? sv.x * scale_log2 : -INFINITY;   // head g, row r0
    const float s01 = v1 ? sv.y * scale_log2 : -INFINITY;   // head g, row r0+1
    const float s10 = v0 ? sv.z * scale_log2 : -INFINITY;   // head g+8, row r0
    const float s11 = v1 ? sv.w * scale_log2 : -INFINITY;
    float mx0 = fmaxf(s00, s01), mx1 = fmaxf(s10, s11);
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    if (cc == 0) { mx_s[warp * 16 + g] = mx0; mx_s[warp * 16 + g + 8] = mx1; }
    __syncthreads();
    TRACE(13);
    float M0 = mx_s[g], M1 = mx_s[g + 8];
#pragma unroll
    for (int w = 1; w < 4; ++w) { M0 = fmaxf(M0, mx_s[w * 16 + g]); M1 = fmaxf(M1, mx_s[w * 16 + g + 8]); }
    // chunk is non-empty -> at least one valid row -> M finite for every head
    const float p00 = ex2_approx(s00 - M0), p01 = ex2_approx(s01 - M0);
    const float p10 = ex2_approx(s10 - M1), p11 = ex2_approx(s11 - M1);
    float l0 = p00 + p01, l1 = p10 + p11;
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    if (cc == 0) { ls_s[warp * 16 + g] = l0; ls_s[warp * 16 + g + 8] = l1; }
    // P fragment pieces: warp w holds (head g | g+8, rows 8w+2cc..+1) -> A fragment regs of k-step w>>1, slot (w&1)*2 + {0,1}
    pb_s[(warp * 2 + 0) * 32 + lane] = pack_bf16x2(p00, p01);
    pb_s[(warp * 2 + 1) * 32 + lane] = pack_bf16x2(p10, p11);
    __syncthreads();
    if (warp == 0 && cc == 0) {
      const float L0 = ls_s[g] + ls_s[16 + g] + ls_s[32 + g] + ls_s[48 + g];
      const float L1 = ls_s[g + 8] + ls_s[16 + g + 8] + ls_s[32 + g + 8] + ls_s[48 + g + 8];
      __stcg(reinterpret_cast<float4*>(rec) + pchunk * 8 + g, make_float4(M0, L0, M1, L1));   // heads g and g+8
    }
    TRACE(14);
    uint32_t pa[2][4];
#pragma unroll
    for (int kstep = 0; kstep < 2; ++kstep) {
      pa[kstep][0] = pb_s[((kstep * 2 + 0) * 2 + 0) * 32 + lane];   // head g,   rows 16ks + 2cc
      pa[kstep][1] = pb_s[((kstep * 2 + 0) * 2 + 1) * 32 + lane];   // head g+8, rows 16ks + 2cc
      pa[kstep][2] = pb_s[((kstep * 2 + 1) * 2 + 0) * 32 + lane];   // head g,   rows 16ks + 8 + 2cc
      pa[kstep][3] = pb_s[((kstep * 2 + 1) * 2 + 1) * 32 + lane];   // head g+8, rows 16ks + 8 + 2cc
    }
    TRACE(4);

    // ---- 5. O = P V (warp w: dims 128w..128w+127), bf16 partials in fragment layout: line (warp, nt, half) = 32 lanes x 4 B ----
    uint32_t* pp = part + ((pchunk * 4 + warp) * 16) * 64 + lane;   // + nt*64 + half*32
    {
      const int mi = lane >> 3;
      const unsigned char* vb = ks + ((mi & 1) * 8 + (lane & 7)) * ROW_STRIDE + (warp * 128 + (mi >> 1) * 8) * 2;
      uint32_t bb[2][2][4];
      ldmatrix_x4_trans(bb[0][0], vb);
      ldmatrix_x4_trans(bb[0][1], vb + 16 * ROW_STRIDE);
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const int nt = 2 * p;
        if (p + 1 < 8) {
          ldmatrix_x4_trans(bb[(p + 1) & 1][0], vb + (nt + 2) * 16);
          ldmatrix_x4_trans(bb[(p + 1) & 1][1], vb + 16 * ROW_STRIDE + (nt + 2) * 16);
        }
        float o0[4] = {0.f, 0.f, 0.f, 0.f}, o1[4] = {0.f, 0.f, 0.f, 0.f};
        mma_bf16(o0, pa[0], bb[p & 1][0][0], bb[p & 1][0][1]);
        mma_bf16(o1, pa[0], bb[p & 1][0][2], bb[p & 1][0][3]);
        mma_bf16(o0, pa[1], bb[p & 1][1][0], bb[p & 1][1][1]);
        mma_bf16(o1, pa[1], bb[p & 1][1][2], bb[p & 1][1][3]);
        __stcg(pp + nt * 64, pack_bf16x2(o0[0], o0[1]));
        __stcg(pp + nt * 64 + 32, pack_bf16x2(o0[2], o0[3]));
        __stcg(pp + (nt + 1) * 64, pack_bf16x2(o1[0], o1[1]));
        __stcg(pp + (nt + 1) * 64 + 32, pack_bf16x2(o1[2], o1[3]));
      }
    }
    TRACE(5);
    __syncthreads();
    if (tid == 0) red_release_or_u64(pmask, 1ull << pc);
  }
  TRACE(6);
  // ---- 6. combine: every block consumes unit (tu, cu) = (token t + T/2, n-tile c) of the mirrored token: 16 heads x 8
  //      dims; lane l: head hh = (l%16)/2 + 8 (l/16), dims nt*8 + 4 (l%2) .. +3 -> 8 B of data + 8 B record per chunk,
  //      so a warp loads all of its 16 chunk slots in a single round trip ----
  {
#ifndef DSA_USHIFT
#define DSA_USHIFT 2   // token shift of the consumer mapping (T/4 for T = 8): with the plain permutation an SM holds tokens
                       // i and i + T/2, so a shift of T/4 keeps a token's consumers off the SMs that produce that token
#endif
    const int tu = (t + (T >= 4 ? DSA_USHIFT : 1)) % T, cu = c;   // cu in [0, 64): producer warp wp = cu / 16, n-tile nt = cu % 16
    unsigned long long* umask = masks + ((size_t)(gen & 1) * T + tu) * MASK_STRIDE;
    const int wp = cu >> 4, nt = cu & 15;
    const int hh = ((lane & 15) >> 1) + 8 * (lane >> 4);
    const uint2* px = reinterpret_cast<const uint2*>(part + (((size_t)tu * NCHUNK) * 4 + wp) * 16 * 64 + nt * 64) + lane;   // + q*2048 (uint2)
    const float2* rc = reinterpret_cast<const float2*>(rec) + (size_t)tu * NCHUNK * 16 + (hh & 7) * 2 + (hh >> 3);           // + q*16
    float RM = -INFINITY, RL = 0.f;
    float RA[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned long long seen = 0ull, mk, me;
    if (tid < 8) snap_s[tid] = 0ull;
    __syncthreads();
#ifdef DSA_TRACE
    int n_iter = 0;
#endif
    for (;;) {
#ifdef DSA_TRACE
      ++n_iter;
#endif
      // every warp polls on its own (relaxed; data is read via L2 .cg loads) and shares its snapshot through smem;
      // the union of the 4 snapshots is consistent because the bits are monotone and each 16 B snapshot has empty(q)
      // whenever it has done(q) for an empty chunk (producer order: empty bit, then release of the done bit)
      if (lane == 0) {
        ld_relaxed_u64x2(umask, mk, me);
        snap_s[warp * 2 + 1] = me;
        __threadfence_block();
        snap_s[warp * 2] = mk;
      }
#ifndef DSA_SHAREPOLL
#define DSA_SHAREPOLL 0
#endif
#if DSA_SHAREPOLL
      if (lane == 0) {
        unsigned long long um = 0ull, ue = 0ull;
#pragma unroll
        for (int w = 0; w < 4; ++w) { const unsigned long long d_ = snap_s[w * 2]; const unsigned long long e_ = snap_s[w * 2 + 1]; um |= d_; ue |= e_ & d_; }
        mk = um; me = ue;   // (empty bits are only used together with done bits)
      }
#endif
      mk = __shfl_sync(0xffffffff, mk, 0);
      me = __shfl_sync(0xffffffff, me, 0);
      const unsigned long long fresh = mk & ~seen & ~me;
      seen = mk;
      if (fresh) {
#ifdef DSA_TRACE
        if (lane == 0 && warp == 0 && trace[(size_t)b * 16 + 10] == 0) trace[(size_t)b * 16 + 10] = gtimer();
#endif
        float2 r[16];
        uint2 x[16];
        bool has = false;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int q = i * 4 + warp;
          r[i] = make_float2(-INFINITY, 0.f); x[i] = make_uint2(0u, 0u);
          if ((fresh >> q) & 1ull) {   // warp-uniform
            has = true;
            r[i] = ld_cg_f2(rc + q * 16);
            x[i] = ld_cg_u2(px + (size_t)q * 2048);
          }
        }
        if (has) {
          float mb = -INFINITY;
#pragma unroll
          for (int i = 0; i < 16; ++i) mb = fmaxf(mb, r[i].x);
          const float Mn = fmaxf(RM, mb);              // finite: at least one fresh non-empty chunk
          const float al = ex2_approx(RM - Mn);
          float L = RL * al;
          float A0 = RA[0] * al, A1 = RA[1] * al, A2 = RA[2] * al, A3 = RA[3] * al;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const float w = ex2_approx(r[i].x - Mn);   // 0 for unused slots
            const float2 v0 = unpack_bf16x2(x[i].x), v1 = unpack_bf16x2(x[i].y);
            L += w * r[i].y;
            A0 += w * v0.x; A1 += w * v0.y; A2 += w * v1.x; A3 += w * v1.y;
          }
          RM = Mn; RL = L; RA[0] = A0; RA[1] = A1; RA[2] = A2; RA[3] = A3;
        }
      }
      if (mk == ~0ull) break;
#ifndef DSA_POLL_NS
#define DSA_POLL_NS 0
#endif
      if (!fresh && DSA_POLL_NS > 0) __nanosleep(DSA_POLL_NS);
    }
#ifdef DSA_TRACE
    if (tid == 0) trace[(size_t)b * 16 + 12] = n_iter;
#endif
    TRACE(7);
    // cross-warp merge: fields [M, L, A0..A3][warp][lane] in the Q region
    {
      float* r = cred + warp * 32 + lane;
      r[0 * 128] = RM; r[1 * 128] = RL;
#pragma unroll
      for (int d = 0; d < 4; ++d) r[(2 + d) * 128] = RA[d];
    }
    __syncthreads();
    if (warp == 0) {
      float Mt = -INFINITY;
#pragma unroll
      for (int w = 0; w < 4; ++w) Mt = fmaxf(Mt, cred[w * 32 + lane]);
      const float Mc = (Mt > -INFINITY) ? Mt : 0.f;
      float Lt = 0.f, O[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float* rr = cred + w * 32 + lane;
        const float wt = ex2_approx(rr[0] - Mc);   // 0 for warps that saw nothing (M = -inf)
        Lt += wt * rr[1 * 128];
#pragma unroll
        for (int d = 0; d < 4; ++d) O[d] += wt * rr[(2 + d) * 128];
      }
      const float inv = (Lt > 0.f) ? __frcp_rn(Lt) : 0.f;
      uint2 ov;
      ov.x = pack_bf16x2(O[0] * inv, O[1] * inv); ov.y = pack_bf16x2(O[2] * inv, O[3] * inv);
      *reinterpret_cast<uint2*>(out + ((size_t)tu * H + hh) * D + wp * 128 + nt * 8 + 4 * (lane & 1)) = ov;
      if (cu == 0 && (lane & 1) == 0) lse[(size_t)tu * H + hh] = (Lt > 0.f) ? (Mt + lg2_approx(Lt)) : -INFINITY;
    }
  }
  TRACE(8);
}

}  // namespace

std::vector<torch::Tensor> dsa_forward(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
                                       torch::Tensor kpe_cache, torch::Tensor sparse_indices, double sm_scale,
                                       torch::Tensor zero_row, torch::Tensor part, torch::Tensor rec, torch::Tensor masks,
                                       torch::Tensor dws, torch::Tensor perm, int64_t gen, torch::Tensor trace) {
  const int T = q_nope.size(0);
  TORCH_CHECK(q_nope.size(1) == H && q_nope.size(2) == D);
  TORCH_CHECK(q_pe.size(1) == H && q_pe.size(2) == DP);
  TORCH_CHECK(ckv_cache.size(1) == 64 && ckv_cache.size(2) == D);
  TORCH_CHECK(kpe_cache.size(1) == 64 && kpe_cache.size(2) == DP);
  TORCH_CHECK(sparse_indices.size(0) == T && sparse_indices.size(1) == TOPK);
  TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous() && ckv_cache.is_contiguous() &&
              kpe_cache.is_contiguous() && sparse_indices.is_contiguous());
  TORCH_CHECK(sparse_indices.scalar_type() == torch::kInt32);
  TORCH_CHECK(zero_row.numel() >= DK && zero_row.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(part.numel() >= (int64_t)T * NCHUNK * 4 * 16 * 64 && part.scalar_type() == torch::kInt32);
  TORCH_CHECK(rec.numel() >= (int64_t)T * NCHUNK * H * 2 && rec.scalar_type() == torch::kFloat32);
  TORCH_CHECK(masks.numel() >= 2 * T * MASK_STRIDE && masks.scalar_type() == torch::kInt64);
  TORCH_CHECK(dws.numel() >= 2 * DW_STRIDE && dws.scalar_type() == torch::kInt32);
  TORCH_CHECK(perm.numel() >= T * NCHUNK && perm.scalar_type() == torch::kInt32);
  TORCH_CHECK(T * NCHUNK <= 512, "donation queue sized for 512 chunks");
  auto out = torch::empty({T, H, D}, q_nope.options());
  auto lse = torch::empty({T, H}, q_nope.options().dtype(torch::kFloat32));
  static int max_blocks = -1;
  if (max_blocks < 0) {
    cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    int per_sm = 0;
    cudaError_t oe = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dsa_kernel, NTHREADS, SMEM_TOTAL);
    TORCH_CHECK(oe == cudaSuccess, "occupancy query failed: ", cudaGetErrorString(oe));
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    max_blocks = per_sm * nsm;
    if (getenv("DSA_VERBOSE")) printf("dsa v14: blocks/SM = %d, SMs = %d, max co-resident = %d, smem = %d\n", per_sm, nsm, max_blocks, SMEM_TOTAL);
  }
  TORCH_CHECK(T * NCHUNK <= max_blocks, "grid does not fit co-resident: ", T * NCHUNK, " > ", max_blocks);
  const float scale_log2 = (float)(sm_scale * 1.4426950408889634);
  auto stream = at::cuda::getCurrentCUDAStream();
  static PermArgs pargs;
  static const void* pargs_src = nullptr;
  if (pargs_src != perm.data_ptr()) {   // one-time (per permutation table): copy to the host-side parameter struct
    auto pc = perm.to(torch::kCPU).contiguous();
    const int n = std::min<int64_t>(512, pc.numel());
    for (int i = 0; i < 512; ++i) pargs.p[i] = (i < n) ? pc.data_ptr<int>()[i] : i;
    pargs_src = perm.data_ptr();
  }
  dsa_kernel<<<T * NCHUNK, NTHREADS, SMEM_TOTAL, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
      sparse_indices.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(zero_row.data_ptr()),
      reinterpret_cast<uint32_t*>(part.data_ptr<int>()), reinterpret_cast<float2*>(rec.data_ptr<float>()),
      reinterpret_cast<unsigned long long*>(masks.data_ptr<int64_t>()), dws.data_ptr<int>(), pargs, nullptr,
      T, scale_log2, (unsigned)gen, trace.numel() ? reinterpret_cast<long long*>(trace.data_ptr<int64_t>()) : nullptr);
  return {out, lse};
}

// placement probe: launch the kernel with the real footprint for `nblocks` blocks and return the SM id of every block
torch::Tensor dsa_probe(int64_t nblocks) {
  cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  auto out = torch::zeros({nblocks}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto stream = at::cuda::getCurrentCUDAStream();
  PermArgs dummy;
  for (int i = 0; i < 512; ++i) dummy.p[i] = i;
  dsa_kernel<<<(int)nblocks, NTHREADS, SMEM_TOTAL, stream>>>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, dummy, out.data_ptr<int>(), 1, 0.f, 0u, nullptr);
  cudaStreamSynchronize(stream);
  return out;
}

int dsa_max_blocks() {
  int per_sm = 0;
  cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, dsa_kernel, NTHREADS, SMEM_TOTAL);
  int dev = 0, nsm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  return std::min(per_sm * nsm, 512);   // co-resident blocks and the 512-entry permutation parameter (T <= 8 chunks grid)
}
