// DSA sparse attention (DeepSeek-V3.2), h16 / ckv512 / kpe64 / topk2048 / page_size64, B200.
//
// ONE kernel launch does everything.  The CUPTI harness measures
// max(kernel_end) - min(kernel_start) over every kernel launched in a timed iteration,
// so a second launch costs ~10 us of host launch latency, and memsets / H2D-D2H copies
// are counted too.  Host-side work before the launch is free.
//
// ---------------------------------------------------------------------------------
// Decomposition.  One CLUSTER of CSIZE = 16 CTAs per query token, grid = T * CSIZE.
// topk = 2048 is split into 128 groups of 16 KV slots; CTA s of the cluster owns the
// *strided* groups { s, s+16, s+32, ..., s+112 }.  Two consequences:
//
//   * the mapping is static, so a worker reads its own 128 indices in the *same* HBM
//     round as the coarse probes -- there is no schedule-dependent second round; and
//   * because valid indices are a prefix of sparse_indices, the valid groups of a
//     worker are a prefix in k, so the work self-balances: a token with n_t = 300 gives
//     every CTA one or two 16-row groups, while n_t = 2048 gives every CTA eight.
//
// Reduction.  All partials of a token live in one cluster, so the flash-decoding
// combine is a distributed-shared-memory reduce-scatter: CTA c's warp w ships its
// 16x32 output slice straight into peer w's shared memory, one cluster barrier later
// every CTA owns 1/16 of the output and divides by the combined denominator.  This
// replaces the global workspace round trip (store + __threadfence + peer wait + reload,
// measured at 5.9 us of the previous 11.6 us split path) with two cluster barriers and
// 16 KB of DSMEM traffic, which measured as free.  No workspace, no fence, no atomics.
//
// MMA blocking minimises LDSM traffic, the binding constraint at these sizes: the Q
// A-fragments are hoisted into registers once per CTA, one ldmatrix.x4 supplies the B
// fragments of two MMAs, the QK partials are staged in fp16, and entirely-invalid KV
// tiles are skipped so light tokens pay almost no MMA cost.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef DSA_NO_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
namespace ffi = tvm::ffi;
#endif

namespace cg = cooperative_groups;

#define H 16
#define DCKV 512
#define DKPE 64
#define DQK 576
#define TOPK 2048
#define GRP 64                    // KV slots per group
#define NGRP (TOPK / GRP)         // 128 groups
#define CSIZE 16                  // CTAs per cluster == partials per token
#define KPC (NGRP / CSIZE)        // 2 groups per CTA
#define CHUNK (KPC * GRP)         // 128 KV rows staged per CTA
#define SUBR (CHUNK / 2)          // the two halves the strided n-tile pairs consume
#define NTHR 512
#define NWARP (NTHR / 32)              // 16
#define SKSTRIDE 584                   // 73 * 16B
#define SPSTRIDE 136                   // 17 * 16B
#define SSTRIDE 136                    // fp16 QK partials
#define SOSTRIDE 520                   // 65 * 16B
#define NQKT (CHUNK / 8)   // 16 QK n-tiles
#define KSLICE 4           // QK k-slices
#define NPG (NWARP / KSLICE)      // 4 QK n-groups, one 16-row pair each per stride
#define NPAIR (NQKT / 2 / NPG)    // 2 strided pairs per warp
#define QKSTEP (DQK / 16 / KSLICE)     // 9 k-steps per slice
#define DPW (DCKV / NWARP)             // 32 dims per warp == 1/CSIZE of the output row
#define NTPV (DPW / 8)                 // 4 PV n-tiles
#define LOG2E 1.4426950408889634f

// ---------------------------------------------------------------------------- PTX
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(unsigned (&r)[4], const void* p) {
  unsigned a = smem_u32(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldm_x4_trans(unsigned (&r)[4], const void* p) {
  unsigned a = smem_u32(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// One ldmatrix.x4 supplies the B fragments of two m16n8k16 MMAs, halving the LDSM
// instruction count relative to the x2 form.
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0,
                                         unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float lg2(float x) {
  float r;
  asm("lg2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void cpasync16(void* dst, const void* src) {
  unsigned d = smem_u32(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cpasync_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cpasync_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ int ldm4_row(int lane) { return (lane & 7) + (((lane >> 3) & 1) << 3); }
__device__ __forceinline__ int ldm4_col(int lane) { return (lane >> 4) << 3; }

// ------------------------------------------------------------------------- kernel
extern "C" __global__ void __launch_bounds__(NTHR, 1) __cluster_dims__(CSIZE, 1, 1)
    dsa_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
               const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
               const int* __restrict__ sidx, float scale, __nv_bfloat16* __restrict__ out,
               float* __restrict__ lse, unsigned long long* prof) {
#ifdef DSA_PROF
#define PROF(k)                                                \
  do {                                                         \
    if (prof && threadIdx.x == 0) {                            \
      unsigned long long gt_;                                  \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_));  \
      prof[blockIdx.x * 32 + (k)] = gt_;                       \
    }                                                          \
  } while (0)
#else
#define PROF(k) ((void)0)
#endif
  PROF(0);
  extern __shared__ char smem[];
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);      // [H][SKSTRIDE]
  __nv_bfloat16* sK = sQ + H * SKSTRIDE;                           // [CHUNK][SKSTRIDE]
  __nv_bfloat16* sO = sK;                                          // [H][SOSTRIDE], aliases sK
  float* sSp = reinterpret_cast<float*>(sK + CHUNK * SKSTRIDE);     // [KSLICE][H][SSTRIDE]
  __nv_bfloat16* sP =
      reinterpret_cast<__nv_bfloat16*>(sSp + KSLICE * H * SSTRIDE);  // [H][SPSTRIDE]
  float* sM = reinterpret_cast<float*>(sP + H * SPSTRIDE);           // [H]
  float* sL = sM + H;                                                // [H]
  int* sI = reinterpret_cast<int*>(sL + H);                           // [CHUNK]
  __nv_bfloat16* sStg = reinterpret_cast<__nv_bfloat16*>(sI + CHUNK + 8);  // [CSIZE][DCKV]
  float2* sStat = reinterpret_cast<float2*>(sStg + CSIZE * DCKV);          // [CSIZE]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int t = blockIdx.x / CSIZE;
  const int s = blockIdx.x - t * CSIZE;  // == cluster rank
  const int* sbase = sidx + (long long)t * TOPK;

  // ---- round 1: this worker's CHUNK indices, plus two broadcast probes -----------
  // Statically addressed and independent, so one HBM round.  The per-worker facts
  // (am I active, how many rows) come out of sI itself; only "is group 0/1 non-empty",
  // which decides the cluster-wide single-worker fast path, has to be read globally.
#pragma unroll
  for (int u = 0; u < CHUNK / NTHR + 1; ++u) {
    const int g = tid + u * NTHR;
    if (g < CHUNK) sI[g] = sbase[(s + (g >> 6) * CSIZE) * GRP + (g & 63)];
  }
  const int p0 = sbase[0];
  const int p1 = sbase[GRP];
  const bool single = (p0 >= 0) && (p1 < 0);  // exactly one worker has rows
  {
    const __nv_bfloat16* qn = q_nope + (long long)t * H * DCKV;
    const __nv_bfloat16* qp = q_pe + (long long)t * H * DKPE;
    for (int g = tid; g < H * DCKV / 8; g += NTHR)
      cpasync16(sQ + (g >> 6) * SKSTRIDE + ((g & 63) << 3), qn + ((long long)g << 3));
    if (tid < H * DKPE / 8)
      cpasync16(sQ + (tid >> 3) * SKSTRIDE + DCKV + ((tid & 7) << 3), qp + ((long long)tid << 3));
    cpasync_commit();  // group 0: Q only
  }
  PROF(1);

  if (p0 < 0) {  // degenerate: no valid index at all; does not occur in the dataset
    if (s == 0) {
      for (int i = tid; i < H * DCKV; i += NTHR)
        out[(long long)t * H * DCKV + i] = (__nv_bfloat16)0.f;
      if (tid < H) lse[(long long)t * H + tid] = -3.0e38f;
    }
    return;
  }

  PROF(13);  // the p0 branch has resolved
  cg::cluster_group cl = cg::this_cluster();
  __syncthreads();                   // sI is written by round 1 and read below
  PROF(14);  // the index round has landed everywhere
  const bool active = sI[0] >= 0;
  int nrows = 0;  // 32 * number of valid 32-row sub-blocks (they are a prefix)
#pragma unroll
  for (int k = 0; k < CHUNK / 32; ++k) nrows += (sI[k * 32] >= 0) ? 32 : 0;

  const int arow = ldm4_row(lane);
  const int acol = ldm4_col(lane);
  const int m0 = lane >> 2;
  const int n0 = (lane & 3) << 1;
  const int dim0 = warp * DPW;
  float o[NTPV][4];
#pragma unroll
  for (int i = 0; i < NTPV; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) o[i][j] = 0.f;

  const int ng = warp & (NPG - 1);  // QK n-group
  const int ks = warp / NPG;        // QK k-slice
  unsigned aQ[QKSTEP][4];           // Q A-fragments, hoisted out of the MMA loops

  if (active) {
    // -------- round 2: Q and this worker's KV rows, all independent ------------

    // cp.async (16 B LDGSTS) beats cp.async.bulk here: the bulk engine charges ~20
    // cycles per request and we would issue one request per 1 KB row.
    {
      auto issue_rows = [&](int r0, int r1) {
        for (int g = tid + r0 * (DCKV / 8); g < r1 * (DCKV / 8); g += NTHR) {
          const int j = sI[g >> 6];
          cpasync16(sK + (g >> 6) * SKSTRIDE + ((g & 63) << 3),
                    ckv + (long long)(j < 0 ? 0 : j) * DCKV + ((g & 63) << 3));
        }
        for (int g = tid + r0 * (DKPE / 8); g < r1 * (DKPE / 8); g += NTHR) {
          const int j = sI[g >> 3];
          cpasync16(sK + (g >> 3) * SKSTRIDE + DCKV + ((g & 7) << 3),
                    kpe + (long long)(j < 0 ? 0 : j) * DKPE + ((g & 7) << 3));
        }
        cpasync_commit();
      };
      issue_rows(0, min(nrows, SUBR));     // group 1: the rows pair 0 consumes
      issue_rows(SUBR, max(nrows, SUBR));  // group 2: the rows pair 1 consumes
    }
    PROF(15);  // every KV cp.async is issued
    cpasync_wait<2>();  // Q has landed; both KV halves are still in flight
    __syncthreads();
    PROF(2);
#pragma unroll
    for (int st = 0; st < QKSTEP; ++st)
      ldm_x4(aQ[st], sQ + arow * SKSTRIDE + (ks * QKSTEP + st) * 16 + acol);
    PROF(3);
  }

  if (active) {
    cpasync_wait<1>();  // the first KV half has landed
    __syncthreads();

    // ---- QK: S[H][nrows] = Q[H][DQK] * K^T -----------------------------------
    // Warp ng owns the KV-row pairs { 16*ng, 16*ng + 64 } (strided, not contiguous), so
    // on a short row count the valid pairs still spread over all NPG n-groups and every
    // warp stays busy; with 16 contiguous rows per warp instead, a 32-row token would
    // only light up 4 of the 16 warps.  A pair is 16 rows = two MMA n-tiles, which is
    // exactly what one ldmatrix.x4 feeds.
    {
      float d[NPAIR][2][4];
#pragma unroll
      for (int pp = 0; pp < NPAIR; ++pp)
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
          for (int j = 0; j < 4; ++j) d[pp][q][j] = 0.f;
      // The pair-validity test is hoisted out of the k-step loop so each variant unrolls
      // fully; with the test inside, the branch serialises the ldmatrix -> MMA chain.
      const int kv0a = ng * 16;            // in the first  64-row half
      const int kv0b = kv0a + NPG * 16;    // in the second 64-row half
      if (kv0a < nrows) {
#pragma unroll
        for (int st = 0; st < QKSTEP; ++st) {
          const int k0 = (ks * QKSTEP + st) * 16;
          unsigned ba[4];
          ldm_x4(ba, sK + (kv0a + arow) * SKSTRIDE + k0 + acol);
          mma16816(d[0][0], aQ[st], ba[0], ba[2]);
          mma16816(d[0][1], aQ[st], ba[1], ba[3]);
        }
      }
      cpasync_wait<0>();  // the second KV half has landed
      __syncthreads();
      if (kv0b < nrows) {
#pragma unroll
        for (int st = 0; st < QKSTEP; ++st) {
          const int k0 = (ks * QKSTEP + st) * 16;
          unsigned bb[4];
          ldm_x4(bb, sK + (kv0b + arow) * SKSTRIDE + k0 + acol);
          mma16816(d[1][0], aQ[st], bb[0], bb[2]);
          mma16816(d[1][1], aQ[st], bb[1], bb[3]);
        }
      }
      const int np = (kv0b < nrows) ? 2 : ((kv0a < nrows) ? 1 : 0);
      float* dst = sSp + ks * H * SSTRIDE;
#pragma unroll
      for (int pp = 0; pp < NPAIR; ++pp) {
        if (pp >= np) break;
        const int kv0 = ng * 16 + pp * (NPG * 16);
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int c0 = kv0 + q * 8 + n0;
          *reinterpret_cast<float2*>(dst + m0 * SSTRIDE + c0) =
              make_float2(d[pp][q][0], d[pp][q][1]);
          *reinterpret_cast<float2*>(dst + (m0 + 8) * SSTRIDE + c0) =
              make_float2(d[pp][q][2], d[pp][q][3]);
        }
      }
    }
    __syncthreads();
    PROF(4);

    // ---- softmax: one warp per head, lane owns 4 KV slots --------------------
    {
      const int h = warp;  // NWARP == H
      float y[CHUNK / 32];
      float mx = -INFINITY;
      const int nu = (nrows + 63) >> 6;  // 64-slot passes actually needed
#pragma unroll
      for (int u = 0; u < CHUNK / 64; ++u) {
        if (u >= nu) break;
        const int kv = (lane + u * 32) << 1;
        const float2 a = *reinterpret_cast<const float2*>(sSp + h * SSTRIDE + kv);
        const float2 b =
            *reinterpret_cast<const float2*>(sSp + H * SSTRIDE + h * SSTRIDE + kv);
        const float2 c =
            *reinterpret_cast<const float2*>(sSp + 2 * H * SSTRIDE + h * SSTRIDE + kv);
        const float2 e =
            *reinterpret_cast<const float2*>(sSp + 3 * H * SSTRIDE + h * SSTRIDE + kv);
        const float v0 = a.x + b.x + c.x + e.x;
        const float v1 = a.y + b.y + c.y + e.y;
        y[u * 2] = (kv < nrows && sI[kv] >= 0) ? v0 * scale : -INFINITY;
        y[u * 2 + 1] = (kv + 1 < nrows && sI[kv + 1] >= 0) ? v1 * scale : -INFINITY;
        mx = fmaxf(mx, fmaxf(y[u * 2], y[u * 2 + 1]));
      }
#pragma unroll
      for (int o2 = 16; o2; o2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2));
      float l = 0.f;
#pragma unroll
      for (int u = 0; u < CHUNK / 64; ++u) {
        if (u >= nu) break;
        const int kv = (lane + u * 32) << 1;
        const float p0 = ex2(y[u * 2] - mx);
        const float p1 = ex2(y[u * 2 + 1] - mx);
        l += p0 + p1;
        *reinterpret_cast<__nv_bfloat162*>(sP + h * SPSTRIDE + kv) =
            __floats2bfloat162_rn(p0, p1);
      }
#pragma unroll
      for (int o2 = 16; o2; o2 >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o2);
      if (lane == 0) {
        sM[h] = mx;
        sL[h] = l;
      }
    }
    __syncthreads();
    PROF(5);
  }

  if (active) {
    // ---- PV: O[H][DCKV] = P[H][nrows] * V[nrows][DCKV] -----------------------
    {
      const int ksteps = nrows >> 4;
#pragma unroll 2
      for (int st = 0; st < ksteps; ++st) {
        unsigned a[4];
        ldm_x4(a, sP + arow * SPSTRIDE + st * 16 + acol);
        unsigned b[NTPV / 2][4];
#pragma unroll
        for (int qp = 0; qp < NTPV / 2; ++qp)
          ldm_x4_trans(b[qp], sK + (st * 16 + arow) * SKSTRIDE + dim0 + qp * 16 + acol);
#pragma unroll
        for (int qp = 0; qp < NTPV / 2; ++qp) {
          mma16816(o[qp * 2], a, b[qp][0], b[qp][1]);
          mma16816(o[qp * 2 + 1], a, b[qp][2], b[qp][3]);
        }
      }
    }
    PROF(6);
  }

  // ---- single-worker fast path: write the result straight out -------------------
  if (single) {
    if (s != 0) {
      cpasync_wait<0>();  // do not leave the CTA with its Q copy still in flight
      return;
    }
    const float inv0 = 1.f / sL[m0];
    const float inv8 = 1.f / sL[m0 + 8];
    const int dcol = dim0 + n0;
    __syncthreads();  // sO aliases sK, which PV has just finished reading
#pragma unroll
    for (int q = 0; q < NTPV; ++q) {
      *reinterpret_cast<__nv_bfloat162*>(sO + m0 * SOSTRIDE + dcol + q * 8) =
          __floats2bfloat162_rn(o[q][0] * inv0, o[q][1] * inv0);
      *reinterpret_cast<__nv_bfloat162*>(sO + (m0 + 8) * SOSTRIDE + dcol + q * 8) =
          __floats2bfloat162_rn(o[q][2] * inv8, o[q][3] * inv8);
    }
    if (tid < H) lse[(long long)t * H + tid] = sM[tid] + lg2(sL[tid]);
    __syncthreads();
    __nv_bfloat16* dst = out + (long long)t * H * DCKV;
#pragma unroll
    for (int u = 0; u < H * DCKV / 8 / NTHR; ++u) {
      const int g = tid + u * NTHR;
      *reinterpret_cast<int4*>(dst + ((long long)g << 3)) =
          *reinterpret_cast<const int4*>(sO + (g >> 6) * SOSTRIDE + ((g & 63) << 3));
    }
    PROF(7);
    return;
  }

  // ============================ cluster combine =================================
  // No barrier is needed before writing a peer: the cluster's CTAs are co-resident for the
  // whole kernel and sStg/sStat are dead until here.  Every CTA publishes (m, l) for every
  // head to every peer; an inactive worker publishes (-inf, 0) so it contributes a zero
  // coefficient.
  if (tid < CSIZE)
    cl.map_shared_rank(sStat, tid)[s] =
        active ? make_float2(sM[tid], sL[tid]) : make_float2(-INFINITY, 0.f);
  if (!active) cpasync_wait<0>();  // must not exit with the Q copy still in flight
  if (active) {
    // Stage the normalised 16 x DCKV tile locally, then hand peer p the row of head p.
    // bf16 is what the peer receives, so stage in bf16: 16 KB instead of 32 KB written,
    // and the scatter becomes a straight copy of 8 B units.
    const float inv0 = 1.f / sL[m0];
    const float inv8 = 1.f / sL[m0 + 8];
    const int dcol = dim0 + n0;
    __syncthreads();  // sO aliases sK, which PV has just finished reading
#pragma unroll
    for (int q = 0; q < NTPV; ++q) {
      *reinterpret_cast<__nv_bfloat162*>(sO + m0 * SOSTRIDE + dcol + q * 8) =
          __floats2bfloat162_rn(o[q][0] * inv0, o[q][1] * inv0);
      *reinterpret_cast<__nv_bfloat162*>(sO + (m0 + 8) * SOSTRIDE + dcol + q * 8) =
          __floats2bfloat162_rn(o[q][2] * inv8, o[q][3] * inv8);
    }
    __syncthreads();
    // CSIZE * DCKV bf16 = 1024 16 B units; consecutive lanes take consecutive units of one
    // peer, so each warp reads 512 B of sO and stores 512 B into one peer.
#pragma unroll
    for (int u = 0; u < CSIZE * DCKV / 8 / NTHR; ++u) {
      const int j = tid + u * NTHR;
      const int p = j >> 6;
      const int d = (j & 63) << 3;
      *reinterpret_cast<int4*>(cl.map_shared_rank(sStg, p) + s * DCKV + d) =
          *reinterpret_cast<const int4*>(sO + p * SOSTRIDE + d);
    }
  }
  PROF(10);
  cl.sync();
  PROF(11);

  // CTA s now owns head s of this token.  Every thread that reduces derives the CSIZE
  // combine weights itself: sStat is 128 B read by broadcast and the arithmetic is a chain
  // of independent FMAs, which is cheaper than a ballot, ten shuffles and a __syncthreads(),
  // and -- the point -- leaves the sStg loads free to issue while it is still running.
  if (tid < DCKV / 4) {
    float mx = -INFINITY;
#pragma unroll
    for (int c = 0; c < CSIZE; ++c) mx = fmaxf(mx, sStat[c].x);
    float w[CSIZE], sum = 0.f;
#pragma unroll
    for (int c = 0; c < CSIZE; ++c) {
      const float2 st = sStat[c];
      w[c] = st.y * ex2(st.x - mx);  // an inactive rank published (-inf, 0), so this is 0
      sum += w[c];
    }
    // Four dims per thread, fully unrolled over CSIZE so all 16 shared loads are
    // independent and complete in one round.  A slot no rank wrote must stay unread, not
    // merely zero-weighted: it could hold a NaN, and 0 * NaN is NaN.
    const int d = tid << 2;
    const float inv = 1.f / sum;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < CSIZE; ++c) {
      const bool ok = w[c] > 0.f;  // slot 0 always exists, so the clamp keeps the load legal
      const __nv_bfloat162* v =
          reinterpret_cast<const __nv_bfloat162*>(sStg + (ok ? c : 0) * DCKV + d);
      const float2 lo = __bfloat1622float2(v[0]);
      const float2 hi = __bfloat1622float2(v[1]);
      acc[0] += w[c] * lo.x;
      acc[1] += w[c] * lo.y;
      acc[2] += w[c] * hi.x;
      acc[3] += w[c] * hi.y;
    }
    __nv_bfloat162 r[2] = {__floats2bfloat162_rn(acc[0] * inv, acc[1] * inv),
                           __floats2bfloat162_rn(acc[2] * inv, acc[3] * inv)};
    *reinterpret_cast<int2*>(out + ((long long)t * H + s) * DCKV + d) =
        *reinterpret_cast<const int2*>(r);
    if (tid == 0) lse[(long long)t * H + s] = mx + lg2(sum);
  }
  PROF(12);
#undef PROF
}

// --------------------------------------------------------------------------- host
static bool g_init = false;

static constexpr size_t kSmemBytes =
    sizeof(__nv_bfloat16) * (H * SKSTRIDE + CHUNK * SKSTRIDE) +
    sizeof(float) * (KSLICE * H * SSTRIDE) + sizeof(__nv_bfloat16) * (H * SPSTRIDE) +
    sizeof(float) * (2 * H) + sizeof(int) * (CHUNK + 8) +
    sizeof(__nv_bfloat16) * (CSIZE * DCKV) + sizeof(float2) * CSIZE;
// sO aliases sK, which PV has finished reading by the time sO is written
static_assert(sizeof(__nv_bfloat16) * (H * SOSTRIDE) <= sizeof(__nv_bfloat16) * CHUNK * SKSTRIDE,
              "output tile does not fit in the sK alias");
static_assert(kSmemBytes <= 232448, "over the sm_100 opt-in shared memory limit");

const char* dsa_launch(const __nv_bfloat16* q_nope, const __nv_bfloat16* q_pe,
                       const __nv_bfloat16* ckv, const __nv_bfloat16* kpe, const int* sidx,
                       float sm_scale, __nv_bfloat16* out, float* lse, int T, int device,
                       cudaStream_t stream, unsigned long long* prof) {
  if (T <= 0) return nullptr;
  if (!g_init) {
    // CSIZE > 8 is a non-portable cluster size and has to be opted into.
    cudaError_t e =
        cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(dsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)kSmemBytes);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    g_init = true;
  }
  dsa_kernel<<<T * CSIZE, NTHR, kSmemBytes, stream>>>(q_nope, q_pe, ckv, kpe, sidx,
                                                      sm_scale * LOG2E, out, lse, prof);
  return nullptr;
}

#ifndef DSA_NO_FFI
void dsa_sparse_attention(ffi::TensorView q_nope, ffi::TensorView q_pe, ffi::TensorView ckv_cache,
                          ffi::TensorView kpe_cache, ffi::TensorView sparse_indices, double sm_scale,
                          ffi::TensorView output, ffi::TensorView lse) {
  DLDevice dev = q_nope.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const char* err = dsa_launch(static_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
                               static_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
                               static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
                               static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
                               static_cast<const int*>(sparse_indices.data_ptr()),
                               static_cast<float>(sm_scale),
                               static_cast<__nv_bfloat16*>(output.data_ptr()),
                               static_cast<float*>(lse.data_ptr()),
                               static_cast<int>(q_nope.size(0)), dev.device_id, stream, nullptr);
  if (err) TVM_FFI_THROW(RuntimeError) << "dsa_sparse_attention: " << err;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsa_sparse_attention, dsa_sparse_attention);
#endif
