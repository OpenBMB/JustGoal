// DSA sparse attention (DeepSeek V3.2 decode), B200 / sm_100a, CUDA 13.
// Definition: dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
//
// v2 design (Phase 2): one launch, one 16-CTA thread-block cluster per token.
//   grid = (16 chunks, num_tokens), cluster = (16,1,1), block = 256 threads (8 warps).
//   CTA (rank r) owns index slots [128r, 128r+128) of its token:
//     - cp.async gathers its KV rows into smem (ckv rows 1040 B stride, kpe rows 144 B),
//     - S = Q K^T with mma.sync m16n8k16 bf16 (M = 16 heads, N = 128 keys, K = 576),
//       2 n-tiles per warp, 4 interleaved accumulators per n-tile,
//     - base-2 softmax -> P (bf16) and per-head (m, l),
//     - O = P V with mma.sync (K = 128 keys, N = 512 dims),
//     - push model: every CTA st.async-es its fp32 partial slice for dims [32r, 32r+32) and
//       its (m, l) straight into rank r's receive buffer (DSMEM) with mbarrier transaction
//       counting; rank r waits on its local mbarrier, merges locally, writes bf16 output
//       (rank 0 also writes lse). No mid-kernel cluster barrier, no global workspace.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int H       = 16;    // query heads
constexpr int DC      = 512;   // head_dim_ckv
constexpr int DP      = 64;    // head_dim_kpe
constexpr int D       = DC + DP;   // 576
constexpr int TOPK    = 2048;
constexpr int CHUNK   = 128;   // index slots per CTA
constexpr int NCHUNK  = TOPK / CHUNK;  // 16 == cluster size
constexpr int THREADS = 256;
constexpr int NWARPS  = THREADS / 32;

// smem strides (bytes). Row strides are ≡ 16 (mod 128) so 8 consecutive rows hit distinct
// bank groups for ldmatrix.
constexpr int Q_STRIDE   = D * 2 + 16;        // 1168
constexpr int K_STRIDE   = DC * 2 + 16;       // 1040
constexpr int KPE_STRIDE = DP * 2 + 16;       // 144
constexpr int S_STRIDE   = CHUNK + 4;         // floats (132)
constexpr int P_STRIDE   = CHUNK * 2 + 16;    // bytes (272)

constexpr int SMEM_Q    = H * Q_STRIDE;            // 18688
constexpr int SMEM_K    = CHUNK * K_STRIDE;        // 133120
constexpr int SMEM_KPE  = CHUNK * KPE_STRIDE;      // 18432
constexpr int SMEM_S    = H * S_STRIDE * 4;        // 8448
constexpr int SMEM_P    = H * P_STRIDE;            // 4352
constexpr int SMEM_MISC = 2048;
constexpr int SLOT_O     = H * 32 * 4;              // 2048 B: [16 heads][32 dims] fp32
constexpr int SLOT_BYTES = SLOT_O + H * 8;          // 2176 B: + [16 heads](m, l)
constexpr int SMEM_RECV  = NCHUNK * SLOT_BYTES;     // [16 src ranks] = 34816 B
constexpr int RECV_BYTES = SMEM_RECV;               // expected mbarrier tx bytes
constexpr int SMEM_TOTAL = SMEM_Q + SMEM_K + SMEM_KPE + SMEM_S + SMEM_P + SMEM_MISC + SMEM_RECV;
static_assert(NCHUNK * SLOT_BYTES <= SMEM_K, "staging buffer must fit in the K region");
static_assert(SLOT_BYTES % 16 == 0, "bulk copy size must be a multiple of 16");

#ifdef DSA_TIMING
__device__ unsigned long long g_dbg[64 * 16];
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define STAMP(ph) do { if (threadIdx.x == 0) g_dbg[(blockIdx.y * NCHUNK + blockIdx.x) * 16 + (ph)] = gtimer(); } while (0)
#else
#define STAMP(ph) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async_f32x2(uint32_t raddr, float a, float b, uint32_t rbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               ::"r"(raddr), "f"(a), "f"(b), "r"(rbar) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
// bulk copy local smem -> remote CTA smem, completing on the remote mbarrier
__device__ __forceinline__ void bulk_s2s_cluster(uint32_t rdst, uint32_t lsrc, uint32_t bytes, uint32_t rbar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(rdst), "r"(lsrc), "r"(bytes), "r"(rbar) : "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16_16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__global__ void __launch_bounds__(THREADS, 1)
dsa_sparse_attn_kernel(const __nv_bfloat16* __restrict__ q_nope,   // [T][H][DC]
                       const __nv_bfloat16* __restrict__ q_pe,     // [T][H][DP]
                       const __nv_bfloat16* __restrict__ ckv,      // [P*64][DC]
                       const __nv_bfloat16* __restrict__ kpe,      // [P*64][DP]
                       const int32_t* __restrict__ indices,        // [T][TOPK]
                       float scale_log2,                           // sm_scale * log2(e)
                       __nv_bfloat16* __restrict__ out,            // [T][H][DC]
                       float* __restrict__ lse) {                  // [T][H]
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sQ   = smem;
  uint8_t* sK   = smem + SMEM_Q;
  uint8_t* sKpe = sK + SMEM_K;
  float*   sS   = reinterpret_cast<float*>(sKpe + SMEM_KPE);
  uint8_t* sP   = sKpe + SMEM_KPE + SMEM_S;
  uint8_t* sMisc = sP + SMEM_P;
  int32_t* sIdx = reinterpret_cast<int32_t*>(sMisc);               // [128]  512 B
  float*   sM   = reinterpret_cast<float*>(sMisc + 512);            // [16]
  float*   sL   = sM + 16;                                          // [16]
  uint64_t* sBar = reinterpret_cast<uint64_t*>(sMisc + 1024);      // mbarrier
  uint8_t* sRecv  = sMisc + SMEM_MISC;                               // [16 src][SLOT_BYTES]
  uint8_t* sStage = sK;                                              // [16 dst][SLOT_BYTES] after PV

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int rank = blockIdx.x;          // == cluster.block_rank() for cluster dims (16,1,1)
  const int tok  = blockIdx.y;

  STAMP(0);
  // receive-side mbarrier: 1 local arrival (expect_tx) + RECV_BYTES of remote st.async data
  const uint32_t bar = smem_u32(sBar);
  if (tid == 0) {
    mbar_init(bar, 1);
    fence_mbar_init();
    mbar_expect_tx(bar, RECV_BYTES);
  }
  __syncthreads();
  cluster_arrive_relaxed();   // "my mbarrier is initialised"; waited on right before pushing
  // ---- 1. indices (first thing on the critical path) ------------------------
  int my_idx = -1;
  if (tid < CHUNK) {
    my_idx = __ldg(indices + (size_t)tok * TOPK + rank * CHUNK + tid);
    sIdx[tid] = my_idx;
  }
  // Q does not depend on the indices: issue it right away (16 rows x 72 chunks of 16 B).
  {
    const __nv_bfloat16* qn = q_nope + (size_t)tok * H * DC;
    const __nv_bfloat16* qp = q_pe + (size_t)tok * H * DP;
#pragma unroll
    for (int i = 0; i < (H * 72 + THREADS - 1) / THREADS; ++i) {
      const int g = tid + i * THREADS;
      if (g < H * 72) {
        const int row = g / 72, j = g - row * 72;
        const uint32_t dst = smem_u32(sQ + row * Q_STRIDE + j * 16);
        if (j < 64) cp_async16(dst, reinterpret_cast<const uint8_t*>(qn + row * DC) + j * 16);
        else        cp_async16(dst, reinterpret_cast<const uint8_t*>(qp + row * DP) + (j - 64) * 16);
      }
    }
  }
  const int nvalid = __syncthreads_count(tid < CHUNK && my_idx >= 0);
  STAMP(1);

  float o[8][4];
  if (nvalid > 0) {
    // ---- 2. gather KV rows ----------------------------------------------------
    // ckv: thread owns 16-B column j = tid & 63 of rows (tid >> 6) + 4i, i = 0..31.
    {
      const int j = tid & 63;
      const int r0 = tid >> 6;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int i = 0; i < CHUNK / 4; ++i) {
        const int row = r0 + 4 * i;
        const int idx = sIdx[row];
        uint8_t* dstp = sK + row * K_STRIDE + j * 16;
        if (idx >= 0) cp_async16(smem_u32(dstp), reinterpret_cast<const uint8_t*>(ckv + (size_t)idx * DC) + j * 16);
        else          *reinterpret_cast<uint4*>(dstp) = z;
      }
      // kpe: column j = tid & 7 of rows (tid >> 3) + 32i, i = 0..3.
      const int jp = tid & 7;
      const int rp = tid >> 3;
#pragma unroll
      for (int i = 0; i < CHUNK / 32; ++i) {
        const int row = rp + 32 * i;
        const int idx = sIdx[row];
        uint8_t* dstp = sKpe + row * KPE_STRIDE + jp * 16;
        if (idx >= 0) cp_async16(smem_u32(dstp), reinterpret_cast<const uint8_t*>(kpe + (size_t)idx * DP) + jp * 16);
        else          *reinterpret_cast<uint4*>(dstp) = z;
      }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
    STAMP(2);

    // ---- 3. S = Q K^T : warp w owns keys [16w, 16w+16) = n-tiles 2w, 2w+1 -------
    float acc[2][4][4];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int g = 0; g < 4; ++g) { acc[j][g][0] = acc[j][g][1] = acc[j][g][2] = acc[j][g][3] = 0.f; }
    {
      const int mi = lane >> 3;
      // A (Q): matrices (rows 0-7,k0-7),(rows 8-15,k0-7),(rows 0-7,k8-15),(rows 8-15,k8-15)
      const int a_row = (mi & 1) * 8 + (lane & 7);
      const int a_kcol = (mi >> 1) * 8;
      const uint32_t a_base = smem_u32(sQ + a_row * Q_STRIDE + a_kcol * 2);
      // B (K rows): matrices (keys 0-7,k0-7),(keys 0-7,k8-15),(keys 8-15,k0-7),(keys 8-15,k8-15)
      const int b_key = warp * 16 + (mi >> 1) * 8 + (lane & 7);
      const int b_kcol = (mi & 1) * 8;
      const uint32_t bk_base = smem_u32(sK + b_key * K_STRIDE + b_kcol * 2);
      const uint32_t bp_base = smem_u32(sKpe + b_key * KPE_STRIDE + b_kcol * 2);
#pragma unroll
      for (int ks = 0; ks < DC / 16; ++ks) {
        uint32_t a[4], b[4];
        ldmatrix_x4(a[0], a[1], a[2], a[3], a_base + ks * 32);
        ldmatrix_x4(b[0], b[1], b[2], b[3], bk_base + ks * 32);
        mma_bf16_16816(acc[0][ks & 3], a, b);
        mma_bf16_16816(acc[1][ks & 3], a, b + 2);
      }
#pragma unroll
      for (int ks = 0; ks < DP / 16; ++ks) {
        uint32_t a[4], b[4];
        ldmatrix_x4(a[0], a[1], a[2], a[3], a_base + (DC + ks * 16) * 2);
        ldmatrix_x4(b[0], b[1], b[2], b[3], bp_base + ks * 32);
        mma_bf16_16816(acc[0][ks & 3], a, b);
        mma_bf16_16816(acc[1][ks & 3], a, b + 2);
      }
    }
    // reduce k-groups, scale, mask, store S[h][key]
    {
      const int h0 = lane >> 2;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int k0 = warp * 16 + j * 8 + (lane & 3) * 2;
        const float s0 = (acc[j][0][0] + acc[j][1][0]) + (acc[j][2][0] + acc[j][3][0]);
        const float s1 = (acc[j][0][1] + acc[j][1][1]) + (acc[j][2][1] + acc[j][3][1]);
        const float s2 = (acc[j][0][2] + acc[j][1][2]) + (acc[j][2][2] + acc[j][3][2]);
        const float s3 = (acc[j][0][3] + acc[j][1][3]) + (acc[j][2][3] + acc[j][3][3]);
        const bool v0 = sIdx[k0] >= 0, v1 = sIdx[k0 + 1] >= 0;
        sS[h0 * S_STRIDE + k0]           = v0 ? s0 * scale_log2 : -INFINITY;
        sS[h0 * S_STRIDE + k0 + 1]       = v1 ? s1 * scale_log2 : -INFINITY;
        sS[(h0 + 8) * S_STRIDE + k0]     = v0 ? s2 * scale_log2 : -INFINITY;
        sS[(h0 + 8) * S_STRIDE + k0 + 1] = v1 ? s3 * scale_log2 : -INFINITY;
      }
    }
    __syncthreads();
    STAMP(3);

    // ---- 4. softmax (base 2): warp w handles heads 2w, 2w+1 ---------------------
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int h = warp * 2 + i;
      const float* srow = sS + h * S_STRIDE;
      const float s0 = srow[lane], s1 = srow[lane + 32], s2 = srow[lane + 64], s3 = srow[lane + 96];
      const float m = warp_max(fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
      const float p0 = exp2f(s0 - m), p1 = exp2f(s1 - m), p2 = exp2f(s2 - m), p3 = exp2f(s3 - m);
      const float l = warp_sum((p0 + p1) + (p2 + p3));
      __nv_bfloat16* prow = reinterpret_cast<__nv_bfloat16*>(sP + h * P_STRIDE);
      prow[lane]      = __float2bfloat16_rn(p0);
      prow[lane + 32] = __float2bfloat16_rn(p1);
      prow[lane + 64] = __float2bfloat16_rn(p2);
      prow[lane + 96] = __float2bfloat16_rn(p3);
      if (lane == 0) { sM[h] = m; sL[h] = l; }
    }
    __syncthreads();
    STAMP(4);

    // ---- 5. O = P V : warp w owns dims [64w, 64w+64) (8 n-tiles) ----------------
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f; }
    {
      const int mi = lane >> 3;
      const int a_row = (mi & 1) * 8 + (lane & 7);
      const int a_kcol = (mi >> 1) * 8;
      const uint32_t a_base = smem_u32(sP + a_row * P_STRIDE + a_kcol * 2);
      // B (V rows, .trans): lanes 0-7 keys k0+l dims n0; 8-15 keys k0+8+l dims n0;
      //                     16-23 keys k0+l dims n0+8; 24-31 keys k0+8+l dims n0+8
      const int b_key = (mi & 1) * 8 + (lane & 7);
      const int b_dcol = (mi >> 1) * 8;
      const uint32_t b_base = smem_u32(sK + b_key * K_STRIDE + (warp * 64 + b_dcol) * 2);
#pragma unroll
      for (int ks = 0; ks < CHUNK / 16; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a[0], a[1], a[2], a[3], a_base + ks * 32);
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t b[4];
          ldmatrix_x4_trans(b[0], b[1], b[2], b[3], b_base + ks * 16 * K_STRIDE + jp * 32);
          mma_bf16_16816(o[jp * 2], a, b);
          mma_bf16_16816(o[jp * 2 + 1], a, b + 2);
        }
      }
    }
    STAMP(5);
  } else {
    cp_async_commit();
    cp_async_wait_all();
    if (tid < H) { sM[tid] = -INFINITY; sL[tid] = 0.f; }
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f; }
  }

  // ---- 6. stage (o slice, m, l) per target rank in local smem, bulk-copy to targets ----
  __syncthreads();   // all warps are done reading sK (PV) before it is reused as staging
  {
    // O: thread holds (h0, d0+8j) and (h0+8, d0+8j); dims [64w, 64w+64) -> ranks 2w, 2w+1
    const int h0 = lane >> 2;
    const int d0 = warp * 64 + (lane & 3) * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int dim = d0 + j * 8;
      const int tr = dim >> 5, dl = dim & 31;
      float* slot = reinterpret_cast<float*>(sStage + tr * SLOT_BYTES);
      *reinterpret_cast<float2*>(slot + h0 * 32 + dl)       = make_float2(o[j][0], o[j][1]);
      *reinterpret_cast<float2*>(slot + (h0 + 8) * 32 + dl) = make_float2(o[j][2], o[j][3]);
    }
    // (m, l) for every target: thread t -> target t>>4, head t&15
    {
      const int tr = tid >> 4, h = tid & 15;
      *reinterpret_cast<float2*>(sStage + tr * SLOT_BYTES + SLOT_O + h * 8) = make_float2(sM[h], sL[h]);
    }
  }
  STAMP(9);
  fence_proxy_async_smem();   // make generic-proxy smem writes visible to the async (bulk) proxy
  __syncthreads();
  STAMP(10);
#ifndef DSA_NO_CLUSTER_WAIT
  cluster_wait_acquire();     // all 16 mbarriers are initialised (long done by now)
#endif
  STAMP(11);
  if (tid < NCHUNK) {
    const uint32_t tr = tid;
    const uint32_t rdst = mapa_shared(smem_u32(sRecv) + rank * SLOT_BYTES, tr);
    bulk_s2s_cluster(rdst, smem_u32(sStage + tr * SLOT_BYTES), SLOT_BYTES, mapa_shared(bar, tr));
  }
  STAMP(6);

  // ---- 7. wait for all 16 slots, merge locally, write output slice ------------
  mbar_wait_parity(bar, 0);
  STAMP(7);
  {
    const int h = tid >> 4;
    const int dl = (tid & 15) * 2;
    float M = -INFINITY;
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c)
      M = fmaxf(M, *reinterpret_cast<const float*>(sRecv + c * SLOT_BYTES + SLOT_O + h * 8));
    float L = 0.f, ax = 0.f, ay = 0.f;
#pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      const float2 ml = *reinterpret_cast<const float2*>(sRecv + c * SLOT_BYTES + SLOT_O + h * 8);
      const float wc = (ml.x == -INFINITY) ? 0.f : exp2f(ml.x - M);
      L += wc * ml.y;
      const float2 v = *reinterpret_cast<const float2*>(sRecv + c * SLOT_BYTES + (h * 32 + dl) * 4);
      ax += wc * v.x;
      ay += wc * v.y;
    }
    const float inv = (L > 0.f) ? 1.f / L : 0.f;
    *reinterpret_cast<uint32_t*>(out + ((size_t)tok * H + h) * DC + rank * 32 + dl) =
        pack_bf16x2(ax * inv, ay * inv);
    if (rank == 0 && dl == 0) lse[(size_t)tok * H + h] = (L > 0.f) ? (M + log2f(L)) : -INFINITY;
  }
  STAMP(8);
}

}  // namespace

extern "C" {

int dsa_launch(const void* q_nope, const void* q_pe, const void* ckv, const void* kpe,
               const int32_t* indices, float sm_scale, void* out, float* lse,
               int num_tokens, cudaStream_t stream) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(dsa_sparse_attn_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(dsa_sparse_attn_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
    attr_set = true;
  }
  const float scale_log2 = sm_scale * 1.4426950408889634f;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCHUNK, num_tokens, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_TOTAL;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = NCHUNK;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&cfg, dsa_sparse_attn_kernel,
      reinterpret_cast<const __nv_bfloat16*>(q_nope), reinterpret_cast<const __nv_bfloat16*>(q_pe),
      reinterpret_cast<const __nv_bfloat16*>(ckv), reinterpret_cast<const __nv_bfloat16*>(kpe),
      indices, scale_log2, reinterpret_cast<__nv_bfloat16*>(out), lse);
  return (int)e;
}

int dsa_smem_bytes() { return SMEM_TOTAL; }

#ifdef DSA_TIMING
int dsa_read_dbg(void* host_dst) { return (int)cudaMemcpyFromSymbol(host_dst, g_dbg, sizeof(g_dbg)); }
int dsa_clear_dbg() {
  static unsigned long long z[64 * 16] = {};
  return (int)cudaMemcpyToSymbol(g_dbg, z, sizeof(z));
}
#endif

}  // extern "C"
