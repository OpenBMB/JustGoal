// ============================================================================
// GQA paged decode, page_size = 1, bf16 -- written from scratch for B200 (sm100).
//
// Shape it is tuned for:  B=16, kv_len 118..455 (2841 tokens total),
// 32 q heads / 8 kv heads (group = 4), head_dim = 128.
//   q (B,32,128) bf16, k_cache/v_cache (P,1,8,128) bf16,
//   kv_indptr (B+1) i32, kv_indices (total) i32, sm_scale
//   -> out (B,32,128) bf16, lse (B,32) f32   (lse in natural log)
//
// Why it is shaped this way
// ------------------------------------------------------------------
// The whole problem only touches 11.6 MB of KV, so a launch large enough to
// fill the GPU is entirely resident: the kernel is limited by issued
// instructions and by memory latency, not by DRAM bandwidth.  The design
// therefore (a) keeps the instruction count per KV token low, (b) keeps the
// register count low enough for several CTAs per SM, (c) keeps the dependency
// chain short (one indirection, then one batch of K/V loads), and (d) does the
// whole operator, split-KV reduction included, in a single kernel launch.
//
//   * one CTA per (batch, kv_head, kv_chunk), chunk bounds balanced by the host
//     side plan; a CTA has WARPS warps and each warp owns TPW contiguous tokens
//   * Q for the 4 q-heads that share this kv-head is loaded once into registers
//     and pre-scaled by sm_scale*log2(e), so all softmax math is base 2
//   * QK: 16 lanes per token, 8 head dims per lane (16B loads; 16 lanes cover
//     the 256B contiguous K row of one token, so a warp does 2 tokens per step).
//     A 16-lane butterfly finishes the 4 dot products at once, and one K
//     element feeds all four q-heads.
//   * softmax: chunk-wide, in shared memory, one warp per q-head
//   * PV: 32 lanes per token, 4 head dims per lane, so the output accumulators
//     are 16 registers per thread and need no cross-lane reduction at all
//   * output: cross-warp sum in shared memory; single-chunk groups write the
//     final result directly, otherwise the CTA stores its partial and the last
//     CTA of the group (device-scope release/acquire arrival counter) combines.
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda/atomic>

#define HD 128            // head_dim
#define NQPK 4            // q heads per kv head
#define LN2 0.6931471805599453f
#define UB 8              // partial loads issued per batch in the combine step

__device__ __forceinline__ void cvt4(const uint2& raw, float* o) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;
}
__device__ __forceinline__ void cvt8(const uint4& raw, float* o) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) { float2 f = __bfloat1622float2(h[i]); o[2*i] = f.x; o[2*i+1] = f.y; }
}

// tasks [n][4]  = (batch, kv_head, indptr[b]+chunk_start, chunk_len)
// tasks2[n][4]  = (chunks_in_group, first_task_of_group, group_id, unused)
template <int TPW, int WARPS>
__global__ __launch_bounds__(WARPS*32) void gqa_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const int* __restrict__ indices,
    const int4* __restrict__ tasks,
    const int4* __restrict__ tasks2,
    float* __restrict__ part_acc,      // [n][NQPK*HD]
    float2* __restrict__ part_ml,      // [n][NQPK]
    int* __restrict__ counters,        // [groups*32]  (one 128B line each)
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int num_kv_heads,
    int gstride,                       // task stride between chunks of one group
    float qscale)                      // sm_scale * log2(e)
{
  constexpr int CHUNK = TPW * WARPS;
  constexpr int NS1 = TPW / 2;         // QK steps per warp (2 tokens per step)

  const int task = blockIdx.x;
  const int4 td = tasks[task];
  const int4 td2 = tasks2[task];
  const int b = td.x, kvh = td.y, ibase = td.z, nt = td.w;
  const int gn = td2.x, first = td2.y, group = td2.z;
  const int HQ = num_kv_heads * NQPK;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int half = lane >> 4, l16 = lane & 15;
  const int dimA = l16 * 8;            // QK / K  : 8 dims per lane
  const int dimB = lane * 4;           // PV / V  : 4 dims per lane
  const unsigned hmask = 0xffffu << (half * 16);
  const int t0w = warp * TPW;

  __shared__ float sS[CHUNK][NQPK];
  __shared__ float sm[NQPK], sl[NQPK];
  __shared__ float sacc[WARPS][NQPK * HD];
  __shared__ int slast;

  // ---- Q into registers, pre-scaled ----
  float qr[NQPK][8];
  {
    const __nv_bfloat16* qp = q + ((size_t)b * HQ + kvh * NQPK) * HD + dimA;
    uint4 raw[NQPK];
#pragma unroll
    for (int j = 0; j < NQPK; ++j) raw[j] = *reinterpret_cast<const uint4*>(qp + j * HD);
#pragma unroll
    for (int j = 0; j < NQPK; ++j) {
      cvt8(raw[j], qr[j]);
#pragma unroll
      for (int d = 0; d < 8; ++d) qr[j][d] *= qscale;
    }
  }

  // ---- page ids for this warp's tokens, then all of its K loads ----
  const int* idxp = indices + ibase;
  int pg1[NS1], pg3[TPW];
  bool ok1[NS1], ok3[TPW];
#pragma unroll
  for (int s = 0; s < NS1; ++s) {
    const int it = t0w + s * 2 + half;
    ok1[s] = (it < nt);
    pg1[s] = idxp[ok1[s] ? it : 0];               // clamped: always a legal page
  }
#pragma unroll
  for (int i = 0; i < TPW; ++i) {
    const int it = t0w + i;
    ok3[i] = (it < nt);
    pg3[i] = idxp[ok3[i] ? it : 0];
  }
  uint4 kraw[NS1];
#pragma unroll
  for (int s = 0; s < NS1; ++s)
    kraw[s] = *reinterpret_cast<const uint4*>(kc + ((size_t)pg1[s] * num_kv_heads + kvh) * HD + dimA);

  // ---- S = Q@K^T for this chunk ----
#pragma unroll
  for (int s = 0; s < NS1; ++s) {
    float kk[8];
    cvt8(kraw[s], kk);
    float a[NQPK];
#pragma unroll
    for (int j = 0; j < NQPK; ++j) {
      float t = 0.f;
#pragma unroll
      for (int d = 0; d < 8; ++d) t = fmaf(qr[j][d], kk[d], t);
      a[j] = t;
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
#pragma unroll
      for (int j = 0; j < NQPK; ++j) a[j] += __shfl_xor_sync(hmask, a[j], off);
    if (l16 == 0) {                               // 0 for padding tokens -> weight 0 in PV
      float4 v4;
      v4.x = ok1[s] ? a[0] : 0.f; v4.y = ok1[s] ? a[1] : 0.f;
      v4.z = ok1[s] ? a[2] : 0.f; v4.w = ok1[s] ? a[3] : 0.f;
      *reinterpret_cast<float4*>(&sS[t0w + s * 2 + half][0]) = v4;
    }
  }
  __syncthreads();

  // ---- chunk-local base-2 softmax, one warp per q-head ----
  if (warp < NQPK) {
    const int j = warp;
    float mx = -INFINITY;
    for (int t = lane; t < nt; t += 32) mx = fmaxf(mx, sS[t][j]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.f;
    for (int t = lane; t < nt; t += 32) { const float p = exp2f(sS[t][j] - mx); sS[t][j] = p; sum += p; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) { sm[j] = mx; sl[j] = sum; }
  }
  __syncthreads();

  // ---- acc = P@V ----
  float acc[NQPK][4];
#pragma unroll
  for (int j = 0; j < NQPK; ++j)
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[j][d] = 0.f;
#pragma unroll
  for (int i = 0; i < TPW; ++i) {
    const uint2 vr = *reinterpret_cast<const uint2*>(vc + ((size_t)pg3[i] * num_kv_heads + kvh) * HD + dimB);
    float vv[4];
    cvt4(vr, vv);
    const float4 p4 = *reinterpret_cast<const float4*>(&sS[t0w + i][0]);
    const float p[NQPK] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
    for (int j = 0; j < NQPK; ++j)
#pragma unroll
      for (int d = 0; d < 4; ++d) acc[j][d] = fmaf(p[j], vv[d], acc[j][d]);
  }

  // ---- cross-warp sum (all warps share one chunk max, so no rescaling) ----
#pragma unroll
  for (int j = 0; j < NQPK; ++j)
    *reinterpret_cast<float4*>(&sacc[warp][j * HD + dimB]) = *reinterpret_cast<float4*>(acc[j]);
  __syncthreads();

  if (tid >= 128) return;                       // 128 threads x 4 floats = NQPK*HD
  const int e0 = tid * 4, jj = e0 >> 7, dd = e0 & (HD - 1);
  float r[4];
  {
    const float4 x = *reinterpret_cast<float4*>(&sacc[0][e0]);
    r[0] = x.x; r[1] = x.y; r[2] = x.z; r[3] = x.w;
  }
#pragma unroll
  for (int w = 1; w < WARPS; ++w) {
    const float4 x = *reinterpret_cast<float4*>(&sacc[w][e0]);
    r[0] += x.x; r[1] += x.y; r[2] += x.z; r[3] += x.w;
  }

  __nv_bfloat16* op = out + ((size_t)b * HQ + kvh * NQPK + jj) * HD + dd;
  const float cm = sm[jj], cl = sl[jj];
  if (gn == 1) {
    const float inv = 1.f / cl;
    __nv_bfloat16 t[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = __float2bfloat16(r[i] * inv);
    *reinterpret_cast<uint2*>(op) = *reinterpret_cast<uint2*>(t);
    if (dd == 0) lse[b * HQ + kvh * NQPK + jj] = (cm + log2f(cl)) * LN2;
    return;
  }

  // ---- split-KV: publish the partial, last CTA of the group combines ----
  __stcg(reinterpret_cast<float4*>(part_acc + (size_t)task * (NQPK * HD) + e0),
         *reinterpret_cast<float4*>(r));
  if (dd == 0) __stcg(part_ml + (size_t)task * NQPK + jj, make_float2(cm, cl));
  if (tid == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> ctr(counters[group * 32]);
    const int old = ctr.fetch_add(1, cuda::memory_order_acq_rel);   // release: partials visible
    slast = (old == gn - 1);
    if (slast) ctr.store(0, cuda::memory_order_relaxed);            // armed for the next launch
  }
  __syncthreads();
  if (!slast) return;

  float gm = -INFINITY, gd = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c0 = 0; c0 < gn; c0 += UB) {
    float2 mlb[UB];
    float4 xb[UB];
#pragma unroll
    for (int u = 0; u < UB; ++u) {               // one batch of loads in flight
      const int c = c0 + u;
      if (c < gn) {
        const int tk = first + c * gstride;
        mlb[u] = __ldcg(part_ml + (size_t)tk * NQPK + jj);
        xb[u] = __ldcg(reinterpret_cast<const float4*>(part_acc + (size_t)tk * (NQPK * HD) + e0));
      }
    }
#pragma unroll
    for (int u = 0; u < UB; ++u) {
      if (c0 + u < gn) {
        const float nm = fmaxf(gm, mlb[u].x);
        const float sa = exp2f(gm - nm), sb = exp2f(mlb[u].x - nm);
        gd = fmaf(gd, sa, mlb[u].y * sb);
        o[0] = fmaf(o[0], sa, xb[u].x * sb); o[1] = fmaf(o[1], sa, xb[u].y * sb);
        o[2] = fmaf(o[2], sa, xb[u].z * sb); o[3] = fmaf(o[3], sa, xb[u].w * sb);
        gm = nm;
      }
    }
  }
  const float inv = 1.f / gd;
  __nv_bfloat16 t[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) t[i] = __float2bfloat16(o[i] * inv);
  *reinterpret_cast<uint2*>(op) = *reinterpret_cast<uint2*>(t);
  if (dd == 0) lse[b * HQ + kvh * NQPK + jj] = (gm + log2f(gd)) * LN2;
}

void gqa_decode(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor indices,
                torch::Tensor tasks, torch::Tensor tasks2, torch::Tensor part_acc,
                torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out,
                torch::Tensor lse, double sm_scale, int64_t tpw, int64_t warps, int64_t gstride) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kc.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.size(2) == HD && kc.size(1) == 1, "page_size must be 1 and head_dim 128");
  TORCH_CHECK(q.size(1) == kc.size(2) * NQPK, "this kernel expects 4 q heads per kv head");
  const int num_kv_heads = kc.size(2);
  const dim3 grid(tasks.size(0));
  const float qscale = (float)(sm_scale * 1.4426950408889634);
#define LAUNCH(T, W)                                                                        \
  gqa_decode_kernel<T, W><<<grid, W * 32, 0, at::cuda::getCurrentCUDAStream()>>>(            \
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)kc.data_ptr(),               \
      (const __nv_bfloat16*)vc.data_ptr(), indices.data_ptr<int>(),                          \
      (const int4*)tasks.data_ptr<int>(), (const int4*)tasks2.data_ptr<int>(),                \
      part_acc.data_ptr<float>(), (float2*)part_ml.data_ptr<float>(),                         \
      counters.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(), lse.data_ptr<float>(),        \
      num_kv_heads, (int)gstride, qscale)
  const int key = (int)tpw * 100 + (int)warps;
  switch (key) {
    case 1604: LAUNCH(16, 4); break;      // the tuned default (chunk 64, 424 CTAs)
    case 1204: LAUNCH(12, 4); break;
    case 804:  LAUNCH(8, 4); break;
    case 2004: LAUNCH(20, 4); break;
    case 808:  LAUNCH(8, 8); break;
    case 1608: LAUNCH(16, 8); break;
    default: TORCH_CHECK(false, "unsupported tpw/warps combination");
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gqa_decode", &gqa_decode); }
