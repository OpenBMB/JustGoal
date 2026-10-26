// DSA FWD decode (sparse MLA decode) for SM100 / B200, written from scratch.
//
// Shape: T decode tokens, H=16 heads, ckv (== V) dim 512, kpe dim 64,
// top-k sparse index list into a paged pool, bf16 in / bf16 out.
//
// Structure: split-KV flash decoding.
//
//   CTA (s, t) owns index slots [s*BN, s*BN+BN) of request t. The slice is
//   deliberately small: one SM only sustains ~25 GB/s on this row gather, so the
//   KV bytes have to be spread over many SMs for the memory time to hide behind
//   the kernel-launch floor. A slice with no valid index writes lse = -inf and
//   skips the compute, which also balances the work for a short valid prefix.
//   Exactly one KV tile per CTA, so the softmax is a plain one-pass softmax with
//   no accumulator rescaling.
//
//     QK : S[16 heads][8 rows] per warp, k = 576.
//          A = Q from smem via ldmatrix.x4; B = K read straight out of smem with
//          32-bit loads (the m16n8k16 B fragment wants 2 consecutive k for one n,
//          which is exactly 2 consecutive halves of a row-major K row).
//     PV : O[16 heads][D/NWARP dims] per warp, k = BN rows.
//          A = P from smem via ldmatrix.x4; B = V via ldmatrix.x4.trans.
//
//   With FUSED=true the log-sum-exp combine runs in the same kernel: each CTA
//   signals a per-request counter, waits for its siblings, then reduces an equal
//   slice of that request's output. That removes a whole second launch and keeps
//   the partials L2-hot.
//
// Shared-memory row strides are padded so every ldmatrix / LDS access is
// bank-conflict free: the K/Q stride of 584 halves is 292 words (4 banks per
// row), and the P stride of BN+8 halves gives 8 distinct 4-bank groups.

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#define DSA_H 16    // heads
#define DSA_D 512   // ckv / value / output dim
#define DSA_DPE 64  // kpe dim
#define DSA_DK 576  // total qk dim
#define DKP 584     // padded row stride of the Q and K smem matrices, in halves
#define OSTRIDE (DSA_D + 4)  // padded row stride of the fp32 output transpose, in floats
// The K tile doubles as the output transpose buffer in the fused path, so the
// region has to be big enough for both.
#define KREGION(BN_) \
  ((BN_) * DKP > DSA_H * OSTRIDE * 2 ? (BN_) * DKP : DSA_H * OSTRIDE * 2)

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void cp_commit_wait() {
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// Vector reduction-add: one instruction covers 4 consecutive fp32 outputs.
// The L2 atomic units are per-slice, so keeping the addresses spread across
// cache lines is good, but the *instruction count* is what costs - dropping it
// 4x is the single biggest win in the accumulate phase. sm_90+.
// bf16x2 reduction-add: covers the same 2 dims per lane as half a v4.f32 does
// per lane-pair, i.e. the same dims per warp instruction for half the bytes.
// The numerators are unnormalised softmax sums, and the accuracy budget here is
// ~5x wider than the rounding this costs.
__device__ __forceinline__ void red_add_bf16x2(__nv_bfloat16 *p, __nv_bfloat162 v) {
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" ::"l"(p), "r"(*reinterpret_cast<const unsigned *>(&v))
               : "memory");
}

__device__ __forceinline__ void red_add_v4(float *p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "f"(a), "f"(b), "f"(c),
               "f"(d)
               : "memory");
}

// Arrive/wait with release-acquire ordering: the release on the counter carries
// this block's prior reductions, so no separate (and separately timed)
// __threadfence() is needed before signalling.
__device__ __forceinline__ int atom_add_release(int *p, int v) {
  int old;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(p), "r"(v)
               : "memory");
  return old;
}

__device__ __forceinline__ int ld_acquire(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void mma_16x8x16(float (&c)[4], const uint32_t (&a)[4],
                                            uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}


// Softmax weights across the splits of one head. Every thread returns the same
// (1/sum, bound) pair; s_w[0..nsplit) gets the weight of each split (0 for the
// splits that had no valid index). `bound` is one past the last non-empty split,
// so the reduction loop can stay branch-free and still skip the empty tail.
// Requires blockDim.x >= nsplit.
struct SplitWeights {
  float inv_sum;
  int bound;
};

__device__ __forceinline__ SplitWeights split_weights(const float *lcol, int nsplit, float *s_w,
                                                      float *s_f, int *s_i) {
  const int tid = threadIdx.x;
  const int nw = blockDim.x >> 5;
  const float v = (tid < nsplit) ? lcol[tid] : -INFINITY;

  float m = v;
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) s_f[tid >> 5] = m;
  __syncthreads();
  m = -INFINITY;
  for (int i = 0; i < nw; ++i) m = fmaxf(m, s_f[i]);

  const float wt = (v == -INFINITY) ? 0.f : __expf(v - m);
  if (tid < nsplit) s_w[tid] = wt;
  float sw = wt;
  int bd = (wt > 0.f) ? tid + 1 : 0;
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    sw += __shfl_xor_sync(0xffffffffu, sw, o);
    bd = max(bd, __shfl_xor_sync(0xffffffffu, bd, o));
  }
  __syncthreads();
  if ((tid & 31) == 0) {
    s_f[tid >> 5] = sw;
    s_i[tid >> 5] = bd;
  }
  __syncthreads();
  SplitWeights r;
  float tot = 0.f;
  int bmax = 0;
  for (int i = 0; i < nw; ++i) {
    tot += s_f[i];
    bmax = max(bmax, s_i[i]);
  }
  r.inv_sum = 1.f / tot;
  r.bound = bmax;
  return r;
}

// Q (nope || pe) -> smem. Issued before the index load so its latency overlaps
// with it; the copies stay uncommitted until the first KV tile commits them.
template <int NWARP, int HG>
__device__ __forceinline__ void stage_q(const __nv_bfloat16 *__restrict__ q_nope,
                                        const __nv_bfloat16 *__restrict__ q_pe,
                                        __nv_bfloat16 *s_q, int req, int hg) {
  constexpr int HPC = DSA_H / HG;
  constexpr int NTHREADS = NWARP * 32;
  const int tid = threadIdx.x;
  const __nv_bfloat16 *qn = q_nope + ((size_t)req * DSA_H + hg * HPC) * DSA_D;
  for (int i = tid; i < HPC * (DSA_D / 8); i += NTHREADS) {
    int h = i / (DSA_D / 8), c = i % (DSA_D / 8);
    cp_async16(smem_addr(&s_q[h * DKP + c * 8]), &qn[h * DSA_D + c * 8]);
  }
  const __nv_bfloat16 *qp = q_pe + ((size_t)req * DSA_H + hg * HPC) * DSA_DPE;
  for (int i = tid; i < HPC * (DSA_DPE / 8); i += NTHREADS) {
    int h = i / (DSA_DPE / 8), c = i % (DSA_DPE / 8);
    cp_async16(smem_addr(&s_q[h * DKP + DSA_D + c * 8]), &qp[h * DSA_DPE + c * 8]);
  }
  if (HG > 1) {  // zeroed rows keep the discarded fragment half finite
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = tid; i < (DSA_H - HPC) * (DKP / 8); i += NTHREADS) {
      int h = HPC + i / (DKP / 8), c = i % (DKP / 8);
      *reinterpret_cast<float4 *>(&s_q[h * DKP + c * 8]) = z;
    }
  }
}

// Attention for this CTA's head group over its tiles of <= BN KV rows.
// HG = 1 -> all 16 heads (both halves of every mma fragment are used).
// HG = 2 -> heads [hg*8, hg*8+8); the upper half of each fragment is fed zeroed
//           Q rows and discarded. Halving Q and the output bytes per CTA is
//           worth more than the wasted mma slots: this kernel is bound by
//           bytes-and-latency per CTA, not by tensor-core throughput.
// TPP   -> tiles per CTA, strided by the grid's split count. >1 shrinks the grid
//           (shorter launch ramp, cheaper barrier) while keeping the per-CTA KV
//           slice small. Only valid for MODE 1, whose fixed exponent reference
//           lets tiles be summed with no rescaling.
template <int BN, int NWARP, int HG, int MODE, int TPP>
__device__ __forceinline__ void dsa_tiles_body(
    const __nv_bfloat16 *__restrict__ q_nope, const __nv_bfloat16 *__restrict__ q_pe,
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    __nv_bfloat16 *__restrict__ o_part, float *__restrict__ lse, float *__restrict__ num,
    float *__restrict__ den, __nv_bfloat16 *s_q, __nv_bfloat16 *s_k, __nv_bfloat16 *s_p,
    float *s_max, float *s_sum, const int *s_slot, const int (&anyt)[TPP],
    const int (&mine)[TPP], float *s_out, float *s_l,
    int req, int split, int hg, int nsplit, float sm_scale, int stop,
    __nv_bfloat16 *__restrict__ sink) {
  constexpr int HPC = DSA_H / HG;  // heads per CTA
  constexpr int NTHREADS = NWARP * 32;
  constexpr int BNP = BN + 8;
  constexpr int NQK = BN / 8;            // warps that participate in QK
  constexpr int NT = DSA_D / NWARP / 8;  // n-tiles of 8 dims owned by one warp
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2;  // head index within 0..7
  const int tig = lane & 3;   // column pair

  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
  const int a_col = 8 * (lane >> 4);
  // Rotate which dim slice each warp owns by the CTA's split index: all CTAs
  // reduce into the same addresses, and issuing them in the same order turns
  // every address into a serialisation point. This is a scalar offset, so it
  // does not make the accumulator array dynamically indexed.
  const int d0w = ((warp + split) & (NWARP - 1)) * (DSA_D / NWARP);

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[j][k] = 0.f;
  float q0 = 0.f, q1 = 0.f;  // running softmax denominators (MODE 1)
  float m0 = 0.f, m1 = 0.f;

#pragma unroll
  for (int it = 0; it < TPP; ++it) {
    if (!anyt[it]) continue;
    const int *slot = s_slot + it * BN;
    // BN == 32: every warp holds the whole index slice, so a shuffle replaces
    // the smem round trip (and the barrier that would have to precede it).
    const int myslot = (BN == 32) ? mine[it] : 0;

    for (int j = tid; j < BN * (DSA_D / 8); j += NTHREADS) {
      int r = j / (DSA_D / 8), c = j % (DSA_D / 8);
      int sl = (BN == 32) ? __shfl_sync(0xffffffffu, myslot, r) : slot[r];
      sl = sl < 0 ? 0 : sl;
      cp_async16(smem_addr(&s_k[r * DKP + c * 8]), &ckv[(size_t)sl * DSA_D + c * 8]);
    }
    for (int j = tid; j < BN * (DSA_DPE / 8); j += NTHREADS) {
      int r = j / (DSA_DPE / 8), c = j % (DSA_DPE / 8);
      int sl = (BN == 32) ? __shfl_sync(0xffffffffu, myslot, r) : slot[r];
      sl = sl < 0 ? 0 : sl;
      cp_async16(smem_addr(&s_k[r * DKP + DSA_D + c * 8]), &kpe[(size_t)sl * DSA_DPE + c * 8]);
    }
    cp_commit_wait();
    __syncthreads();
    if (stop == 2) {
      if (tid == 0) sink[0] = s_k[0];
      return;
    }

    // ---- S = Q . K^T ; barriers stay outside the branch ----
    float s0 = -INFINITY, s1 = -INFINITY, s2 = -INFINITY, s3 = -INFINITY;
    const int rloc = warp * 8 + tig * 2;
    if (warp < NQK) {
      // 4 independent accumulators: a single chain of 36 dependent mma
      // instructions would cost ~36x the mma latency with nothing to hide it.
      float sca[4][4] = {};
      // ldmatrix.x2 lands exactly in the m16n8k16 B layout when the KV rows are
      // the matrix rows, so one instruction replaces two 32-bit smem loads.
      const int b_arow = warp * 8 + (lane & 7);
      const int b_acol = 8 * ((lane >> 3) & 1);
#pragma unroll
      for (int kk = 0; kk < DSA_DK / 16; ++kk) {
        uint32_t a[4], b[2];
        ldm_x4(a, smem_addr(&s_q[a_row * DKP + kk * 16 + a_col]));
        ldm_x2(b, smem_addr(&s_k[b_arow * DKP + kk * 16 + b_acol]));
        mma_16x8x16(sca[kk & 3], a, b[0], b[1]);
      }
      float sc[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sc[i] = (sca[0][i] + sca[1][i]) + (sca[2][i] + sca[3][i]);
      s0 = sc[0] * sm_scale;  // head gid,     row rloc
      s1 = sc[1] * sm_scale;  // head gid,     row rloc + 1
      s2 = sc[2] * sm_scale;  // head gid + 8, row rloc
      s3 = sc[3] * sm_scale;  // head gid + 8, row rloc + 1
      const int sl0 = (BN == 32) ? __shfl_sync(0xffffffffu, myslot, rloc) : slot[rloc];
      const int sl1 = (BN == 32) ? __shfl_sync(0xffffffffu, myslot, rloc + 1) : slot[rloc + 1];
      if (sl0 < 0) { s0 = -INFINITY; s2 = -INFINITY; }
      if (sl1 < 0) { s1 = -INFINITY; s3 = -INFINITY; }

      if (MODE == 0) {
        float mx0 = fmaxf(s0, s1), mx1 = fmaxf(s2, s3);
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        if (tig == 0) {
          s_max[warp * DSA_H + gid] = mx0;
          s_max[warp * DSA_H + gid + 8] = mx1;
        }
      }
    }
    if (stop == 3) {
      if (tid == 0) sink[0] = __float2bfloat16(s0);
      return;
    }
    if (MODE == 0) {
      __syncthreads();
      m0 = -INFINITY;
      m1 = -INFINITY;
#pragma unroll
      for (int ww = 0; ww < NQK; ++ww) {
        m0 = fmaxf(m0, s_max[ww * DSA_H + gid]);
        m1 = fmaxf(m1, s_max[ww * DSA_H + gid + 8]);
      }
    }
    if (warp < NQK) {
      // MODE 1 exponentiates against a fixed reference of 0 so partial
      // numerators/denominators from different CTAs and tiles are directly
      // addable with plain fp32 atomics. Ceiling: exact while every logit is
      // <= 80 (exp(80) ~ 5e34, and 343 rows of it still fit fp32); above that
      // the clamp degrades instead of producing inf. Attention logits are
      // O(1)-O(10) in practice; MODE 0 is the exact log-sum-exp fallback.
      const float c0 = (MODE == 0) ? s0 : fminf(s0, 80.f);
      const float c1 = (MODE == 0) ? s1 : fminf(s1, 80.f);
      const float c2 = (MODE == 0) ? s2 : fminf(s2, 80.f);
      const float c3 = (MODE == 0) ? s3 : fminf(s3, 80.f);
      const float p0 = __expf(c0 - m0), p1 = __expf(c1 - m0);
      const float p2 = __expf(c2 - m1), p3 = __expf(c3 - m1);
      q0 += p0 + p1;
      q1 += p2 + p3;
      *reinterpret_cast<__nv_bfloat162 *>(&s_p[gid * BNP + rloc]) =
          __floats2bfloat162_rn(p0, p1);
      *reinterpret_cast<__nv_bfloat162 *>(&s_p[(gid + 8) * BNP + rloc]) =
          __floats2bfloat162_rn(p2, p3);
    }
    __syncthreads();
    if (stop == 4) {
      if (tid == 0) sink[1] = s_p[0];
      return;
    }

    // ---- O += P . V ----
#pragma unroll
    for (int kt = 0; kt < BN / 16; ++kt) {
      uint32_t a[4];
      ldm_x4(a, smem_addr(&s_p[a_row * BNP + kt * 16 + a_col]));
#pragma unroll
      for (int jj = 0; jj < NT / 2; ++jj) {
        uint32_t b[4];
        const int brow = kt * 16 + 8 * ((lane >> 3) & 1) + (lane & 7);
        const int bdim = d0w + jj * 16 + 8 * (lane >> 4);
        ldm_x4_trans(b, smem_addr(&s_k[brow * DKP + bdim]));
        mma_16x8x16(acc[jj * 2], a, b[0], b[1]);
        mma_16x8x16(acc[jj * 2 + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();  // s_k / s_p are rewritten by the next tile
  }

  // cross-warp softmax denominators (once, after all tiles)
  if (warp < NQK) {
    float r0 = q0, r1 = q1;
    r0 += __shfl_xor_sync(0xffffffffu, r0, 1);
    r0 += __shfl_xor_sync(0xffffffffu, r0, 2);
    r1 += __shfl_xor_sync(0xffffffffu, r1, 1);
    r1 += __shfl_xor_sync(0xffffffffu, r1, 2);
    if (tig == 0) {
      s_sum[warp * DSA_H + gid] = r0;
      s_sum[warp * DSA_H + gid + 8] = r1;
    }
  }
  __syncthreads();
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int ww = 0; ww < NQK; ++ww) {
    l0 += s_sum[ww * DSA_H + gid];
    l1 += s_sum[ww * DSA_H + gid + 8];
  }
  if (stop == 5) {
    if (tid == 0) sink[2] = __float2bfloat16(acc[0][0] + l0);
    return;
  }

  if (MODE == 2) {
    // Leave the unnormalised result in shared memory; the cluster reduces it
    // through distributed shared memory instead of through L2 atomics.
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int d = d0w + j * 8 + tig * 2;
      *reinterpret_cast<float2 *>(&s_out[gid * OSTRIDE + d]) = make_float2(acc[j][0], acc[j][1]);
      *reinterpret_cast<float2 *>(&s_out[(gid + 8) * OSTRIDE + d]) =
          make_float2(acc[j][2], acc[j][3]);
    }
    if (warp == 0 && tig == 0) {
      s_l[gid] = l0;
      s_l[gid + 8] = l1;
    }
    return;
  }

  if (MODE == 1) {
    // Keep cross-CTA sums in FP32 to avoid cumulative BF16 atomic rounding.
    float *nm = num + ((size_t)req * DSA_H + hg * HPC) * DSA_D;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int d = d0w + j * 8 + tig * 2;
      if (gid < HPC) {
        atomicAdd(&nm[gid * DSA_D + d], acc[j][0]);
        atomicAdd(&nm[gid * DSA_D + d + 1], acc[j][1]);
      }
      if (HG == 1) {
        atomicAdd(&nm[(gid + 8) * DSA_D + d], acc[j][2]);
        atomicAdd(&nm[(gid + 8) * DSA_D + d + 1], acc[j][3]);
      }
    }
    if (warp == 0 && tig == 0 && gid < HPC) {
      atomicAdd(&den[req * DSA_H + hg * HPC + gid], l0);
      if (HG == 1) atomicAdd(&den[req * DSA_H + hg * HPC + gid + 8], l1);
    }
    return;
  }

  // MODE 0: bf16 partials, already softmax-normalised.
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
  __nv_bfloat16 *op = o_part + ((size_t)req * nsplit + split) * DSA_H * DSA_D + hg * HPC * DSA_D;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int d = d0w + j * 8 + tig * 2;
    if (gid < HPC)
      *reinterpret_cast<__nv_bfloat162 *>(&op[gid * DSA_D + d]) =
          __floats2bfloat162_rn(acc[j][0] * inv0, acc[j][1] * inv0);
    if (HG == 1)
      *reinterpret_cast<__nv_bfloat162 *>(&op[(gid + 8) * DSA_D + d]) =
          __floats2bfloat162_rn(acc[j][2] * inv1, acc[j][3] * inv1);
  }
  // lse is stored [req][head][split] so the combine reads one head's splits
  // contiguously instead of striding by H floats per split.
  if (warp == 0 && tig == 0 && gid < HPC) {
    float *lp = lse + (size_t)(req * DSA_H + hg * HPC + gid) * nsplit + split;
    *lp = m0 + __logf(l0);
    if (HG == 1) lp[(size_t)8 * nsplit] = m1 + __logf(l1);
  }
}

// MODE 0: write per-split partials; a second kernel does the log-sum-exp merge.
// MODE 1: atomically accumulate numerator/denominator into the output buffers,
//         barrier once, then divide. One launch, no partial buffer.
template <int BN, int NWARP, int HG, int MODE, int TPP>
__global__ __launch_bounds__(NWARP * 32) void dsa_partial_kernel(
    const __nv_bfloat16 *__restrict__ q_nope, const __nv_bfloat16 *__restrict__ q_pe,
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    const int *__restrict__ indices, __nv_bfloat16 *__restrict__ o_part,
    float *__restrict__ lse, float *__restrict__ num, float *__restrict__ den,
    __nv_bfloat16 *__restrict__ out, int *__restrict__ tick, int *__restrict__ done, int topk,
    int nsplit, int T, float sm_scale, int stop) {
  constexpr int NTHREADS = NWARP * 32;
  constexpr int BNP = BN + 8;
  constexpr int NQK = BN / 8;
  constexpr int HPC = DSA_H / HG;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *s_q = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *s_k = s_q + DSA_H * DKP;
  __nv_bfloat16 *s_p = s_k + KREGION(BN);
  float *s_max = reinterpret_cast<float *>(s_p + DSA_H * BNP);
  float *s_sum = s_max + NQK * DSA_H;
  int *s_slot = reinterpret_cast<int *>(s_sum + NQK * DSA_H);

  const int split = blockIdx.x;
  const int hg = blockIdx.y;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int nblk = nsplit * HG;  // CTAs per request

  // Generation ticket, drawn before any accumulation: it selects which of the
  // two numerator/denominator buffers this launch owns, so nothing has to be
  // zeroed between launches (CUDA-graph safe).
  __shared__ int s_gen;
  constexpr int CTR_PAD = 32;  // ints: one counter per 128-byte line
  if (MODE == 1 && tid == 0) s_gen = atomicAdd(&tick[req * CTR_PAD], 1) / nblk;
  if (MODE == 1) __syncthreads();
  const int gen = (MODE == 1) ? s_gen : 0;
  const size_t bufn = 0;  // numerator is single-buffered: zeroed after reading
  const size_t bufd = (size_t)(gen & 1) * T * DSA_H;
  const size_t nxtd = (size_t)((gen + 1) & 1) * T * DSA_H;

  stage_q<NWARP, HG>(q_nope, q_pe, s_q, req, hg);

  // All TPP index slices are fetched with independent loads, so they cost one
  // memory round trip rather than TPP of them. At BN == 32 every warp loads the
  // whole slice, which makes the validity test a warp vote (no barrier) and lets
  // the staging addresses come from a shuffle instead of shared memory.
  int mine[TPP];
  const int ilane = tid & 31;
  (void)ilane;
#pragma unroll
  for (int it = 0; it < TPP; ++it) {
    const int row0 = (split + it * nsplit) * BN;
    mine[it] = -1;
    if (BN == 32) {
      mine[it] = (row0 + ilane < topk) ? __ldg(&indices[(size_t)req * topk + row0 + ilane]) : -1;
    } else {
      for (int i = tid; i < BN; i += NTHREADS) {
        const int v = (row0 + i < topk) ? indices[(size_t)req * topk + row0 + i] : -1;
        s_slot[it * BN + i] = v;
        mine[it] = v;
      }
    }
  }
  int anyt[TPP];
  int anyhit = 0;
#pragma unroll
  for (int it = 0; it < TPP; ++it) {
    anyt[it] = (BN == 32) ? __any_sync(0xffffffffu, mine[it] >= 0)
                          : __syncthreads_or(mine[it] >= 0);
    anyhit |= anyt[it];
  }
  if (BN != 32) __syncthreads();
  if (stop == 1) {
    if (tid == 0) out[0] = __float2bfloat16((float)anyhit);
    return;
  }

  if (!anyhit) {
    if (MODE == 0)
      for (int h = tid; h < HPC; h += NTHREADS)
        lse[(size_t)(req * DSA_H + hg * HPC + h) * nsplit + split] = -INFINITY;
  } else {
    dsa_tiles_body<BN, NWARP, HG, MODE, TPP>(q_nope, q_pe, ckv, kpe, o_part, lse, num + bufn,
                                             den + bufd, s_q, s_k, s_p, s_max, s_sum, s_slot,
                                             anyt, mine, nullptr, nullptr, req, split, hg,
                                             nsplit, sm_scale, stop, out);
  }
  if (MODE == 0 || (stop >= 2 && stop <= 6)) return;

  // ---- barrier, then divide this CTA's slice of the output ----------------
  const int cid = hg * nsplit + split;  // rank within the request
  const int elems = DSA_H * DSA_D / nblk;
  const int h_out = (cid * elems) / DSA_D;
  const int d_out = (cid * elems) % DSA_D;

  __syncthreads();
  if (stop == 7) return;
  if (tid == 0) {
    // The CTA that arrives last sets the kernel's duration, and the atomic
    // already tells it that everyone else is in - so it never polls.
    const int seen = atom_add_release(done + req * CTR_PAD, 1) + 1;
    const int target = (gen + 1) * nblk;
    if (seen < target) {
      unsigned ns = 16;
      while (ld_acquire(done + req * CTR_PAD) < target) {
        __nanosleep(ns);
        if (ns < 64) ns <<= 1;
      }
    }
  }
  __syncthreads();

  const float dv = den[bufd + req * DSA_H + h_out];
  const float inv = (dv > 0.f) ? 1.f / dv : 0.f;
  float *nm = num + bufn + ((size_t)req * DSA_H + h_out) * DSA_D + d_out;
  __nv_bfloat16 *ob = out + ((size_t)req * DSA_H + h_out) * DSA_D + d_out;
  for (int i = tid; i < elems; i += NTHREADS) {
    ob[i] = __float2bfloat16(nm[i] * inv);
    nm[i] = 0.f;  // ready for the next launch
  }
  if (tid == 0 && d_out == 0) {
    // MODE 1 has no partial LSE consumers; reuse its scratch for final base-2 LSE.
    lse[req * DSA_H + h_out] = log2f(dv);
    den[nxtd + req * DSA_H + h_out] = 0.f;
  }
}

template <int DBLK>
__global__ __launch_bounds__(DBLK) void dsa_combine_kernel(
    const __nv_bfloat16 *__restrict__ o_part, const float *__restrict__ lse,
    __nv_bfloat16 *__restrict__ out, int nsplit) {
  extern __shared__ float s_dyn[];
  float *s_w = s_dyn;
  __shared__ float s_f[32];
  __shared__ int s_i[32];
  const int d = blockIdx.x * DBLK + threadIdx.x;
  const int h = blockIdx.y;
  const int req = blockIdx.z;

  const SplitWeights sw =
      split_weights(lse + (size_t)(req * DSA_H + h) * nsplit, nsplit, s_w, s_f, s_i);

  const __nv_bfloat16 *base = o_part + ((size_t)req * nsplit) * DSA_H * DSA_D + h * DSA_D + d;
  float acc = 0.f;
#pragma unroll 8
  for (int s = 0; s < sw.bound; ++s)
    acc = fmaf(s_w[s], __bfloat162float(base[(size_t)s * DSA_H * DSA_D]), acc);
  out[((size_t)req * DSA_H + h) * DSA_D + d] = __float2bfloat16(acc * sw.inv_sum);
}

template <int BN>
static size_t partial_smem() {
  constexpr int BNP = BN + 8;
  constexpr int NQK = BN / 8;
  return (size_t)(DSA_H * DKP + KREGION(BN) + DSA_H * BNP) * sizeof(__nv_bfloat16) +
         (size_t)2 * NQK * DSA_H * sizeof(float) + (size_t)8 * BN * sizeof(int) + 16;
}

template <int BN, int NWARP, int HG, int MODE, int TPP>
static void launch_partial(const torch::Tensor &q_nope, const torch::Tensor &q_pe,
                           const torch::Tensor &ckv, const torch::Tensor &kpe,
                           const torch::Tensor &indices, torch::Tensor &o_part,
                           torch::Tensor &lse, torch::Tensor &num, torch::Tensor &den,
                           torch::Tensor &out, torch::Tensor &tick, torch::Tensor &done,
                           int nsplit, float sm_scale, int stop) {
  const int T = q_nope.size(0);
  const int topk = indices.size(1);
  const size_t smem = partial_smem<BN>();
  nsplit /= TPP;
  static int configured = 0;
  if ((int)smem > configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dsa_partial_kernel<BN, NWARP, HG, MODE, TPP>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    configured = (int)smem;
  }
  dsa_partial_kernel<BN, NWARP, HG, MODE, TPP>
      <<<dim3(nsplit, HG, T), NWARP * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(q_nope.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(q_pe.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(ckv.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(kpe.data_ptr<at::BFloat16>()),
          indices.data_ptr<int>(),
          reinterpret_cast<__nv_bfloat16 *>(o_part.data_ptr<at::BFloat16>()),
          lse.data_ptr<float>(),
          num.data_ptr<float>(),
          den.data_ptr<float>(),
          reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
          tick.data_ptr<int>(), done.data_ptr<int>(), topk, nsplit, T, sm_scale, stop);
  C10_CUDA_CHECK(cudaGetLastError());
}

#define DSA_CASE(BN_, NW_, HG_, TPP_, MODE_)                                               \
  if (bn == BN_ && nwarp == NW_ && hg == HG_ && tpp == TPP_) {                               \
    launch_partial<BN_, NW_, HG_, MODE_, TPP_>(q_nope, q_pe, ckv, kpe, indices, o_part, lse, \
                                               num, den, out, tick, done, nsplit, scale,     \
                                               stop);                                        \
    launched = true;                                                                         \
  }

#define DSA_DISPATCH(MODE_)                  \
  DSA_CASE(32, 8, 2, 1, MODE_)               \
  DSA_CASE(32, 8, 2, 2, MODE_)               \
  DSA_CASE(32, 8, 2, 4, MODE_)               \
  DSA_CASE(32, 8, 1, 1, MODE_)               \
  DSA_CASE(32, 8, 1, 2, MODE_)               \
  DSA_CASE(32, 16, 2, 1, MODE_)              \
  DSA_CASE(32, 16, 2, 2, MODE_)              \
  DSA_CASE(32, 16, 1, 2, MODE_)              \
  DSA_CASE(16, 8, 2, 1, MODE_)               \
  DSA_CASE(64, 8, 2, 2, MODE_)

void dsa_fwd(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
             torch::Tensor indices, torch::Tensor o_part, torch::Tensor lse, torch::Tensor num,
             torch::Tensor den, torch::Tensor out, torch::Tensor tick, torch::Tensor done,
             double sm_scale, int64_t bn, int64_t nwarp, int64_t hg, int64_t tpp, bool fused,
             int64_t stop_) {
  TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "q_nope must be bf16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  const int topk = indices.size(1);
  const int nsplit = (int)((topk + bn - 1) / bn);
  const float scale = (float)sm_scale;
  const int stop = (int)stop_;
  TORCH_CHECK(fused || tpp == 1, "MODE 0 has no cross-tile accumulation");
  const int nblk = (nsplit / (int)tpp) * (int)hg;
  TORCH_CHECK(!fused || nblk >= DSA_H, "fused combine needs nsplit/tpp*hg >= H");
  TORCH_CHECK(!fused || (DSA_H * DSA_D) % nblk == 0, "fused combine needs nsplit/tpp*hg | H*D");
  bool launched = false;

  if (fused) {
    DSA_DISPATCH(1)
    TORCH_CHECK(launched, "unsupported (bn, nwarp, hg, tpp) combination");
    return;
  }
  DSA_DISPATCH(0)
  TORCH_CHECK(launched, "unsupported (bn, nwarp, hg) combination");

  constexpr int DBLK = 128;
  const int T = q_nope.size(0);
  dsa_combine_kernel<DBLK><<<dim3(DSA_D / DBLK, DSA_H, T), DBLK, 2 * nsplit * sizeof(float),
                             at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(o_part.data_ptr<at::BFloat16>()),
      lse.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), nsplit);
  C10_CUDA_CHECK(cudaGetLastError());
}


// Cost decomposition probes, same grid / block / smem shape as the real kernel.
// MODE 0: launch only.  1: + read this CTA's index slice.  2: + stage its KV rows.
template <int BN, int NWARP, int MODE>
__global__ __launch_bounds__(NWARP * 32) void dsa_probe_kernel(
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    const int *__restrict__ indices, float *__restrict__ sink, int topk) {
  constexpr int NTHREADS = NWARP * 32;
  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *s_k = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  int *s_slot = reinterpret_cast<int *>(s_k + BN * DKP);  // probe: no transpose buffer
  const int split = blockIdx.x;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;

  if (MODE == 0) {
    if (tid == 0) sink[split] = 1.f;
    return;
  }
  const int row0 = split * BN;
  for (int i = tid; i < BN; i += NTHREADS)
    s_slot[i] = (row0 + i < topk) ? indices[(size_t)req * topk + row0 + i] : -1;
  __syncthreads();
  int any = 0;
#pragma unroll
  for (int i = 0; i < BN; ++i) any |= (s_slot[i] >= 0);
  if (MODE == 1 || !any) {
    if (tid == 0) sink[split] = (float)any;
    return;
  }
  for (int j = tid; j < BN * (DSA_D / 8); j += NTHREADS) {
    int r = j / (DSA_D / 8), c = j % (DSA_D / 8);
    int slot = s_slot[r] < 0 ? 0 : s_slot[r];
    cp_async16(smem_addr(&s_k[r * DKP + c * 8]), &ckv[(size_t)slot * DSA_D + c * 8]);
  }
  for (int j = tid; j < BN * (DSA_DPE / 8); j += NTHREADS) {
    int r = j / (DSA_DPE / 8), c = j % (DSA_DPE / 8);
    int slot = s_slot[r] < 0 ? 0 : s_slot[r];
    cp_async16(smem_addr(&s_k[r * DKP + DSA_D + c * 8]), &kpe[(size_t)slot * DSA_DPE + c * 8]);
  }
  cp_commit_wait();
  __syncthreads();
  if (tid == 0) sink[split] = __bfloat162float(s_k[0]) + __bfloat162float(s_k[DSA_D]);
}

void dsa_probe(torch::Tensor ckv, torch::Tensor kpe, torch::Tensor indices, torch::Tensor sink,
               int64_t bn, int64_t mode) {
  const int T = 2;
  const int topk = indices.size(1);
  const int nsplit = (int)((topk + bn - 1) / bn);
  const size_t smem = (size_t)32 * DKP * sizeof(__nv_bfloat16) + 32 * sizeof(int) + 16;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto *ck = reinterpret_cast<const __nv_bfloat16 *>(ckv.data_ptr<at::BFloat16>());
  const auto *kp = reinterpret_cast<const __nv_bfloat16 *>(kpe.data_ptr<at::BFloat16>());
  dim3 grid(nsplit, 1, T);
  if (bn == 32 && mode == 0)
    dsa_probe_kernel<32, 8, 0><<<grid, 256, smem, stream>>>(ck, kp, indices.data_ptr<int>(),
                                                            sink.data_ptr<float>(), topk);
  else if (bn == 32 && mode == 1)
    dsa_probe_kernel<32, 8, 1><<<grid, 256, smem, stream>>>(ck, kp, indices.data_ptr<int>(),
                                                            sink.data_ptr<float>(), topk);
  else if (bn == 32 && mode == 2)
    dsa_probe_kernel<32, 8, 2><<<grid, 256, smem, stream>>>(ck, kp, indices.data_ptr<int>(),
                                                            sink.data_ptr<float>(), topk);
  else
    TORCH_CHECK(false, "probe supports bn=32, mode in {0,1,2}");
  C10_CUDA_CHECK(cudaGetLastError());
}


// Cluster variant: CSIZE CTAs of one request form a thread-block cluster, so the
// cross-CTA reduction runs over distributed shared memory behind a hardware
// barrier instead of over L2 atomics behind a software one. CSIZE == DSA_H, so
// after the barrier each CTA owns exactly one head's 512 outputs.
template <int BN, int NWARP, int CSIZE, int TPP>
__global__ void __cluster_dims__(CSIZE, 1, 1) __launch_bounds__(NWARP * 32) dsa_cluster_kernel(
    const __nv_bfloat16 *__restrict__ q_nope, const __nv_bfloat16 *__restrict__ q_pe,
    const __nv_bfloat16 *__restrict__ ckv, const __nv_bfloat16 *__restrict__ kpe,
    const int *__restrict__ indices, __nv_bfloat16 *__restrict__ out, int topk, int nsplit,
    float sm_scale) {
  namespace cg = cooperative_groups;
  constexpr int NTHREADS = NWARP * 32;
  constexpr int BNP = BN + 8;
  constexpr int NQK = BN / 8;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16 *s_q = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  __nv_bfloat16 *s_k = s_q + DSA_H * DKP;
  // The K tile is dead once PV is done and nothing writes it after the cluster
  // barrier, so the fp32 result lives in the same bytes.
  float *s_out = reinterpret_cast<float *>(s_k);
  __nv_bfloat16 *s_p = s_k + KREGION(BN);
  float *s_max = reinterpret_cast<float *>(s_p + DSA_H * BNP);
  float *s_sum = s_max + NQK * DSA_H;
  float *s_l = s_sum + NQK * DSA_H;
  int *s_slot = reinterpret_cast<int *>(s_l + DSA_H);

  const int split = blockIdx.x;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;

  stage_q<NWARP, 1>(q_nope, q_pe, s_q, req, 0);
  int mine[TPP];
  const int ilane = tid & 31;
#pragma unroll
  for (int it = 0; it < TPP; ++it) {
    const int row0 = (split + it * nsplit) * BN;
    mine[it] = -1;
    if (BN == 32) {
      mine[it] = (row0 + ilane < topk) ? __ldg(&indices[(size_t)req * topk + row0 + ilane]) : -1;
    } else {
      for (int i = tid; i < BN; i += NTHREADS) {
        const int v = (row0 + i < topk) ? indices[(size_t)req * topk + row0 + i] : -1;
        s_slot[it * BN + i] = v;
        mine[it] = v;
      }
    }
  }
  int anyt[TPP];
  int anyhit = 0;
#pragma unroll
  for (int it = 0; it < TPP; ++it) {
    anyt[it] = (BN == 32) ? __any_sync(0xffffffffu, mine[it] >= 0)
                          : __syncthreads_or(mine[it] >= 0);
    anyhit |= anyt[it];
  }
  if (BN != 32) __syncthreads();

  if (anyhit) {
    dsa_tiles_body<BN, NWARP, 1, 2, TPP>(q_nope, q_pe, ckv, kpe, nullptr, nullptr, nullptr,
                                         nullptr, s_q, s_k, s_p, s_max, s_sum, s_slot, anyt,
                                         mine, s_out, s_l, req, split, 0, nsplit, sm_scale, 0,
                                         nullptr);
  } else {
    for (int i = tid; i < DSA_H * OSTRIDE; i += NTHREADS) s_out[i] = 0.f;
    for (int i = tid; i < DSA_H; i += NTHREADS) s_l[i] = 0.f;
  }

  cg::cluster_group cluster = cg::this_cluster();
  cluster.sync();

  const int h = cluster.block_rank();  // CSIZE == DSA_H
  float sum_l = 0.f;
#pragma unroll
  for (int k = 0; k < CSIZE; ++k) sum_l += cluster.map_shared_rank(s_l, k)[h];
  const float inv = (sum_l > 0.f) ? 1.f / sum_l : 0.f;
  __nv_bfloat16 *ob = out + ((size_t)req * DSA_H + h) * DSA_D;
  for (int d = tid; d < DSA_D; d += NTHREADS) {
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < CSIZE; ++k) a += cluster.map_shared_rank(s_out, k)[h * OSTRIDE + d];
    ob[d] = __float2bfloat16(a * inv);
  }
  cluster.sync();  // nobody may exit while a sibling still reads its smem
}

template <int BN>
static size_t cluster_smem() {
  constexpr int BNP = BN + 8;
  constexpr int NQK = BN / 8;
  return (size_t)(DSA_H * DKP + KREGION(BN) + DSA_H * BNP) * sizeof(__nv_bfloat16) +
         (size_t)(2 * NQK * DSA_H + DSA_H) * sizeof(float) + (size_t)8 * BN * sizeof(int) + 16;
}

template <int BN, int NWARP, int CSIZE, int TPP>
static void launch_cluster(const torch::Tensor &q_nope, const torch::Tensor &q_pe,
                           const torch::Tensor &ckv, const torch::Tensor &kpe,
                           const torch::Tensor &indices, torch::Tensor &out, float sm_scale) {
  const int T = q_nope.size(0);
  const int topk = indices.size(1);
  const size_t smem = cluster_smem<BN>();
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dsa_cluster_kernel<BN, NWARP, CSIZE, TPP>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    if (CSIZE > 8)
      C10_CUDA_CHECK(cudaFuncSetAttribute(dsa_cluster_kernel<BN, NWARP, CSIZE, TPP>,
                                          cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    configured = true;
  }
  dsa_cluster_kernel<BN, NWARP, CSIZE, TPP>
      <<<dim3(CSIZE, 1, T), NWARP * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16 *>(q_nope.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(q_pe.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(ckv.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16 *>(kpe.data_ptr<at::BFloat16>()),
          indices.data_ptr<int>(),
          reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()), topk, CSIZE,
          sm_scale);
  C10_CUDA_CHECK(cudaGetLastError());
}

void dsa_fwd_cluster(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv,
                     torch::Tensor kpe, torch::Tensor indices, torch::Tensor out,
                     double sm_scale, int64_t bn) {
  const float scale = (float)sm_scale;
  const int topk = indices.size(1);
  if (bn == 32) {
    TORCH_CHECK(topk / 32 % DSA_H == 0, "topk/bn must be a multiple of 16");
    launch_cluster<32, 8, DSA_H, 4>(q_nope, q_pe, ckv, kpe, indices, out, scale);
  } else if (bn == 64) {
    launch_cluster<64, 8, DSA_H, 2>(q_nope, q_pe, ckv, kpe, indices, out, scale);
  } else if (bn == 16) {
    launch_cluster<16, 8, DSA_H, 8>(q_nope, q_pe, ckv, kpe, indices, out, scale);
  } else {
    TORCH_CHECK(false, "cluster path supports bn in {16,32,64}");
  }
}

int64_t dsa_partial_smem(int64_t bn) {
  if (bn == 16) return (int64_t)partial_smem<16>();
  if (bn == 32) return (int64_t)partial_smem<32>();
  if (bn == 64) return (int64_t)partial_smem<64>();
  return -1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsa_fwd", &dsa_fwd, "DSA FWD decode (split-KV sparse MLA)");
  m.def("dsa_partial_smem", &dsa_partial_smem, "partial kernel shared memory bytes");
  m.def("dsa_probe", &dsa_probe, "cost decomposition probe");
  m.def("dsa_fwd_cluster", &dsa_fwd_cluster, "DSA FWD decode (cluster / DSMEM reduction)");
}
