// GDN (Gated Delta Net) decode kernel — Qwen3-Next, qk4 / v8 / head_dim 128, fp32 k-last state.
// Written from the op definition only (no reference-kernel code consulted).
//
// Math (per v-head h, k/q head hk = h/2):
//   g = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h])),  beta = sigmoid(b[h])
//   S      = g * state[h]                    (S is [V=128, K=128], k contiguous)
//   kv[v]  = sum_k S[v,k] * k[k]
//   delta  = beta * (v[v] - kv[v])
//   S[v,k] += delta[v] * k[k]                -> new_state
//   out[v] = scale * sum_k S[v,k] * q[k]     = scale * (g*(state[v,:].q) + delta[v]*(k.q))
//
// Mapping: one row (h, v) = 512 B of state is owned by 16 lanes (2 x float4 each); a warp handles 2 rows;
// a 128-thread block handles 8 rows; grid = 1024 rows / 8 = 128 blocks (one wave on B200).
// The whole state is requested in one shot, the three dot products S.k, S.q, k.q are reduced in one
// interleaved 4-step shuffle pass, the state write-back is issued right after the update, and the
// output is derived algebraically so no second dependent reduction sits on the tail.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn {

constexpr int HQ = 4, HV = 8, D = 128;
constexpr int LPR = 16;              // lanes per row
constexpr int F = D / (4 * LPR);     // float4 per lane (=2)
constexpr int WARPS = 4;             // warps per block
constexpr int RPW = 32 / LPR;        // rows per warp (=2)
constexpr int ROWS = HV * D;         // 1024 rows
constexpr int THREADS = WARPS * 32;  // 128
constexpr int BLOCKS = ROWS / (WARPS * RPW);  // 128

struct Params {
  const __nv_bfloat16* q; const __nv_bfloat16* k; const __nv_bfloat16* v;
  const float* state; const float* A_log; const __nv_bfloat16* a; const float* dt_bias; const __nv_bfloat16* b;
  float scale; __nv_bfloat16* out; float* new_state;
};

__device__ __forceinline__ void bf16x4_to_float(uint2 u, float* f) {
  __nv_bfloat162 p0 = *reinterpret_cast<__nv_bfloat162*>(&u.x);
  __nv_bfloat162 p1 = *reinterpret_cast<__nv_bfloat162*>(&u.y);
  float2 a = __bfloat1622float2(p0), b = __bfloat1622float2(p1);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__global__ void __launch_bounds__(THREADS) gdn_decode_kernel(Params p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int sub = lane / LPR, l = lane % LPR;
  const int row = (blockIdx.x * WARPS + warp) * RPW + sub;   // (h, v) flattened
  const int h = row / D, hk = h / (HV / HQ);

  // 1) issue all loads up front: state row slice, k/q slices, scalars
  const float4* srow = reinterpret_cast<const float4*>(p.state + (size_t)row * D);
  float4 s[F];
#pragma unroll
  for (int f = 0; f < F; f++) s[f] = __ldg(srow + f * LPR + l);
  uint2 ku[F], qu[F];
#pragma unroll
  for (int f = 0; f < F; f++) {
    ku[f] = __ldg(reinterpret_cast<const uint2*>(p.k + hk * D) + f * LPR + l);
    qu[f] = __ldg(reinterpret_cast<const uint2*>(p.q + hk * D) + f * LPR + l);
  }
  const float vv = __bfloat162float(p.v[row]);
  const float A = __ldg(p.A_log + h), dtb = __ldg(p.dt_bias + h);
  const float av = __bfloat162float(p.a[h]), bv = __bfloat162float(p.b[h]);

  // 2) gates (overlap with the state load latency)
  const float g = expf(-expf(A) * softplus_f(av + dtb));
  const float beta = 1.f / (1.f + expf(-bv));
  float kf[F][4], qf[F][4];
#pragma unroll
  for (int f = 0; f < F; f++) { bf16x4_to_float(ku[f], kf[f]); bf16x4_to_float(qu[f], qf[f]); }
  float kq = 0.f;
#pragma unroll
  for (int f = 0; f < F; f++) kq += kf[f][0] * qf[f][0] + kf[f][1] * qf[f][1] + kf[f][2] * qf[f][2] + kf[f][3] * qf[f][3];

  // 3) decay + partial dot products, one interleaved reduction across the 16 lanes of the row
  float kv = 0.f, sq = 0.f;
#pragma unroll
  for (int f = 0; f < F; f++) {
    s[f].x *= g; s[f].y *= g; s[f].z *= g; s[f].w *= g;
    kv += s[f].x * kf[f][0] + s[f].y * kf[f][1] + s[f].z * kf[f][2] + s[f].w * kf[f][3];
    sq += s[f].x * qf[f][0] + s[f].y * qf[f][1] + s[f].z * qf[f][2] + s[f].w * qf[f][3];
  }
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) {
    kv += __shfl_xor_sync(0xffffffffu, kv, o);
    sq += __shfl_xor_sync(0xffffffffu, sq, o);
    kq += __shfl_xor_sync(0xffffffffu, kq, o);
  }

  // 4) delta-rule update, write state back immediately, then the output
  const float delta = (vv - kv) * beta;
  float4* drow = reinterpret_cast<float4*>(p.new_state + (size_t)row * D);
#pragma unroll
  for (int f = 0; f < F; f++) {
    s[f].x = fmaf(delta, kf[f][0], s[f].x); s[f].y = fmaf(delta, kf[f][1], s[f].y);
    s[f].z = fmaf(delta, kf[f][2], s[f].z); s[f].w = fmaf(delta, kf[f][3], s[f].w);
    drow[f * LPR + l] = s[f];
  }
  if (l == 0) p.out[row] = __float2bfloat16(fmaf(delta, kq, sq) * p.scale);
}

}  // namespace gdn

// C ABI launcher. `variant` is ignored (kept for interface compatibility with the exploration lib).
// new_state may alias state (in-place update).
extern "C" int gdn_launch(int /*variant*/, const void* q, const void* k, const void* v, const void* state,
                          const void* A_log, const void* a, const void* dt_bias, const void* b, float scale,
                          void* out, void* new_state, cudaStream_t stream) {
  gdn::Params p;
  p.q = (const __nv_bfloat16*)q; p.k = (const __nv_bfloat16*)k; p.v = (const __nv_bfloat16*)v;
  p.state = (const float*)state; p.A_log = (const float*)A_log; p.a = (const __nv_bfloat16*)a;
  p.dt_bias = (const float*)dt_bias; p.b = (const __nv_bfloat16*)b; p.scale = scale;
  p.out = (__nv_bfloat16*)out; p.new_state = (float*)new_state;
  gdn::gdn_decode_kernel<<<gdn::BLOCKS, gdn::THREADS, 0, stream>>>(p);
  return (int)cudaGetLastError();
}
