// Sparse MLA decode forward (DSA), bf16 in / bf16 out + fp32 lse.
// out[t,h,:] = softmax(( q_nope[t,h]@Kc^T + q_pe[t,h]@Kp^T ) * sm_scale) @ Kc
// lse[t,h]   = m * log2e + log2(sum_j exp2((s_j - m) * log2e))
// Kc / Kp are gathered from paged caches flattened to [num_pages*page_size, dim].
// sparse_indices entries < 0 are padding.
//
// The grid is tiny (T*16 blocks) and each block sees only a handful of KV rows,
// so this kernel is pure latency: the design is about shortening the dependency
// chain and cutting instruction count.
//
//  - Each warp loads the indices for its own rows, so the index -> gather chain
//    needs no barrier and no shared round trip.
//  - Kc and Kp are gathered into one contiguous K=576 shared tile, which lets
//    the score be a single 576-deep mma chain instead of two dot products.
//  - The scores come out of tensor cores. The GEMV has only one column, so the
//    mma's 8 n-lanes all carry the same q vector; the arithmetic is wasted but
//    it replaces ~110 fp32 FFMA/cvt per thread with 9 mma.
//  - Valid indices form a strict prefix, so the row count is a population
//    count, which __syncthreads_count folds into an already-needed barrier.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#define FULLMASK 0xffffffffu
#define NWARP 8
#define NTHREAD (32 * NWARP)
#define CHUNK 32
#define DCKV 512
#define DPE 64
#define KDIM (DCKV + DPE)  // 576: ckv and kpe concatenated along the k axis
// Pad the row stride so that the 16 rows an ldmatrix touches land on distinct
// bank groups: 584*2/4 % 32 == 4, so 8 consecutive rows cover all 32 banks.
#define KPAD (KDIM + 8)
#define NKSTEP (KDIM / 16)       // 36 mma steps of k=16
#define KPERWARP (NKSTEP / 4)    // 4 warps split the k axis, 9 steps each
#define NHEAD 16
#define TOPK 2048
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ unsigned smem_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)),
               "l"(src));
}

// One k=16 slice of the score GEMV. The caller picks which accumulator the step
// lands in, so the 9 steps can be spread over several independent chains.
#define MMA_STEP(IDX, D0, D1, D2, D3)                                             \
  {                                                                               \
    const uint2 bq = *reinterpret_cast<const uint2 *>(&s_qb[kst + (IDX)][bk]);    \
    unsigned a0, a1, a2, a3;                                                      \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" \
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)                         \
                 : "r"(smem_u32(&s_k[mt * 16 + arow][(kst + (IDX)) * 16 + acol]))); \
    asm volatile(                                                                 \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                    \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                 \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                                  \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bq.x), "r"(bq.y));              \
  }

// Warp-wide max in one instruction. redux only reduces integers, so map fp32 to
// an unsigned key that preserves order (flip the sign bit for non-negatives,
// invert everything for negatives) and map back afterwards.
__device__ __forceinline__ float warp_max(float x) {
  unsigned b = __float_as_uint(x);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  unsigned m;
  asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(m) : "r"(b));
  return __uint_as_float((m & 0x80000000u) ? (m & 0x7fffffffu) : ~m);
}

__global__ __launch_bounds__(NTHREAD) void dsa_kernel(
    const __nv_bfloat16 *__restrict__ q_nope,
    const __nv_bfloat16 *__restrict__ q_pe, const __nv_bfloat16 *__restrict__ ckv,
    const __nv_bfloat16 *__restrict__ kpe, const int *__restrict__ sidx,
    __nv_bfloat16 *__restrict__ out, float *__restrict__ lse, float sm_scale) {
  __shared__ __align__(16) __nv_bfloat16 s_k[CHUNK][KPAD];
  __shared__ unsigned s_qb[NKSTEP][8];
  // [m_tile][row][k_slice]: the 4 slices a row needs are adjacent, so folding
  // them is one conflict-free 16B load.
  __shared__ float s_part[2][16][4];

  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int sub = lane & 7;  // 8 lanes cooperate on one key row
  const int r = tid >> 3;    // this thread's row within the chunk

  // Issue the index load first: the gather depends on it. The 8 lanes of a row
  // read the same int, so a warp's 4 indices cost one transaction.
  const int *irow = sidx + (size_t)t * TOPK;
  int row = irow[r];

  // Stage q once, one thread per k=16 slice, already permuted into the order
  // the mma B operand wants: k pairs {2j,2j+1} and {2j+8,2j+9} land adjacent,
  // so each step reads its two B registers with a single 8B load.
  if (tid < NKSTEP) {
    const size_t qh = (size_t)t * NHEAD + h;
    const int o = tid * 16;
    const __nv_bfloat16 *src =
        (o < DCKV) ? (q_nope + qh * DCKV + o) : (q_pe + qh * DPE + (o - DCKV));
    const uint4 lo = *reinterpret_cast<const uint4 *>(src);
    const uint4 hi = *reinterpret_cast<const uint4 *>(src + 8);
    *reinterpret_cast<uint4 *>(&s_qb[tid][0]) = make_uint4(lo.x, hi.x, lo.y, hi.y);
    *reinterpret_cast<uint4 *>(&s_qb[tid][4]) = make_uint4(lo.z, hi.z, lo.w, hi.w);
  }

  // Fixed per-thread coordinates for the mma pass.
  const int mt = warp & 1;               // which 16-row tile this warp scores
  const int kst = (warp >> 1) * KPERWARP;  // which slice of the k axis
  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;  // ldmatrix x4 quadrants
  const int acol = (lane >> 4) * 8;
  const int bk = (lane & 3) * 2;  // this lane's pair of B registers in s_qb

  const int d2 = 2 * tid;
  float acc0 = 0.f, acc1 = 0.f;
  float m_run = -1e30f, l_run = 0.f;

  for (int c0 = 0; c0 < TOPK; c0 += CHUNK) {
    const bool act = (row >= 0);

    if (act) {
#pragma unroll
      for (int i = 0; i < DCKV / 64; ++i)
        cp_async16(&s_k[r][8 * sub + 64 * i],
                   ckv + (size_t)row * DCKV + 8 * sub + 64 * i);
      cp_async16(&s_k[r][DCKV + 8 * sub], kpe + (size_t)row * DPE + 8 * sub);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);

    // One barrier publishes s_k and s_q and counts the live rows at the same
    // time; 8 lanes per row, hence the shift.
    const int cn = __syncthreads_count(act) >> 3;
    if (cn == 0) break;

    // The output tile of s_k is live as soon as the gather lands, and only p
    // depends on the mma, so pull the first 8 rows into registers now: these
    // LDS hide under the mma and the s_part barrier instead of stalling after
    // them. Rows past cn are never written, hence the guard.
    const __nv_bfloat16 *kbase = &s_k[0][d2];
    unsigned wpre[18];
#pragma unroll
    for (int j = 0; j < 18; ++j)
      wpre[j] = (j < cn) ? *reinterpret_cast<const unsigned *>(kbase + j * KPAD) : 0u;

    // Three independent accumulator chains: the 9 mma steps become 3 chains of
    // 3 instead of one chain of 9, and the folding adds sit off that chain.
    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
    float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;
    float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
#pragma unroll
    for (int i = 0; i < KPERWARP / 3; ++i) {
      MMA_STEP(3 * i + 0, e0, e1, e2, e3)
      MMA_STEP(3 * i + 1, f0, f1, f2, f3)
      MMA_STEP(3 * i + 2, g0, g1, g2, g3)
    }
    // Only mma column 0 is real; it lives in D0/D2 of every fourth lane.
    if ((lane & 3) == 0) {
      s_part[mt][lane >> 2][warp >> 1] = (e0 + f0) + g0;
      s_part[mt][(lane >> 2) + 8][warp >> 1] = (e2 + f2) + g2;
    }
    __syncthreads();

    // Every warp folds the 4 k-slices itself, so the scores stay in registers
    // and the softmax needs no further shared traffic.
    const float4 v4 =
        *reinterpret_cast<const float4 *>(&s_part[lane >> 4][lane & 15][0]);
    const float v = (v4.x + v4.y) + (v4.z + v4.w);
    const float s = (lane < cn) ? v * sm_scale : -INFINITY;

    // Online softmax over the chunk (every warp recomputes it; the rescale
    // factor is needed by all of them anyway).
    const float mnew = fmaxf(m_run, warp_max(s));
    const float corr = exp2f((m_run - mnew) * LOG2E);
    const float p = exp2f((s - mnew) * LOG2E);
    float ls = p;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ls += __shfl_xor_sync(FULLMASK, ls, o);
    l_run = l_run * corr + ls;
    m_run = mnew;
    acc0 *= corr;
    acc1 *= corr;

    // Output: each thread owns two ckv dims. Every warp holds the full p vector
    // across its lanes, so a shuffle broadcast replaces a shared round trip.
    float b0 = 0.f, b1 = 0.f;
#pragma unroll
    for (int j = 0; j < 18; ++j) {  // lanes past cn hold p == 0, so wpre == 0 is safe
      const float pj = __shfl_sync(FULLMASK, p, j);
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&wpre[j]));
      b0 = fmaf(pj, f.x, b0);
      b1 = fmaf(pj, f.y, b1);
    }
#pragma unroll 8
    for (int j = 18; j < cn; ++j) {
      const float pj = __shfl_sync(FULLMASK, p, j);
      const unsigned w = *reinterpret_cast<const unsigned *>(kbase + j * KPAD);
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&w));
      b0 = fmaf(pj, f.x, b0);
      b1 = fmaf(pj, f.y, b1);
    }
    acc0 += b0;
    acc1 += b1;

    if (cn < CHUNK || c0 + CHUNK >= TOPK) break;
    row = irow[c0 + CHUNK + r];
    __syncthreads();  // all reads of s_k / s_part done before they are rewritten
  }

  const float inv = (l_run > 0.f) ? 1.f / l_run : 0.f;
  *reinterpret_cast<__nv_bfloat162 *>(out + ((size_t)t * NHEAD + h) * DCKV + d2) =
      __floats2bfloat162_rn(acc0 * inv, acc1 * inv);
  if (tid == 0)
    lse[(size_t)t * NHEAD + h] =
        (l_run > 0.f) ? (m_run * LOG2E + log2f(l_run)) : -INFINITY;
}

std::vector<at::Tensor> run(at::Tensor q_nope, at::Tensor q_pe, at::Tensor ckv_cache,
                            at::Tensor kpe_cache, at::Tensor sparse_indices,
                            double sm_scale) {
  const int T = q_nope.size(0);
  TORCH_CHECK(sparse_indices.size(1) == TOPK, "topk must be ", TOPK);
  auto out = at::empty({T, NHEAD, DCKV}, q_nope.options());
  auto lse = at::empty({T, NHEAD}, q_nope.options().dtype(at::kFloat));
  dsa_kernel<<<dim3(T, NHEAD), NTHREAD, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(ckv_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(kpe_cache.data_ptr()),
      sparse_indices.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr()), lse.data_ptr<float>(),
      (float)sm_scale);
  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "DSA sparse MLA decode"); }
