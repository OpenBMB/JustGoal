// MoE FP8 block-scale kernels for B200 (sm_100).
// Phase 3: CUDA-graph path for tiny/mid seq. Graph-safe ops write into
// caller-provided persistent buffers (no alloc, no host sync), and guard rows >= R
// (R read from device offsets[E_LOCAL]) so a fixed M_max grid is correct for any R.
// The eager D4 ops (route_gather/swiglu_quant/finalize) remain for the large-seq
// fallback. Two grouped GEMMs run in Python via flashinfer trtllm groupwise.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <algorithm>

namespace {

constexpr int H = 7168;
constexpr int I = 2048;
constexpr int G1OUT = 4096;      // 2*I
constexpr int E_GLOBAL = 256;
constexpr int E_LOCAL = 32;
constexpr int N_GROUP = 8;
constexpr int GROUP_SZ = 32;     // E_GLOBAL / N_GROUP
constexpr int TOPK_GROUP = 4;
constexpr int TOP_K = 8;
constexpr int BLK = 128;
constexpr int HB = H / BLK;      // 56
constexpr int IB = I / BLK;      // 16
constexpr int MAXK = 8;          // <=8 local-expert contributions per token (top_k=8)

// ---------------- Routing (DeepSeek-V3 no-aux) ----------------
// One block (256 threads) per token. Group-score + top-4-group selection on thread 0
// (cheap: 8 groups). Top-8 expert selection parallelized across all 256 threads via
// 8 block-argmax reductions (leftmost-max on ties == reference strict-> order).
__global__ void routing_kernel(const float* __restrict__ logits,       // [T,256]
                               const __nv_bfloat16* __restrict__ bias,  // [256]
                               float* __restrict__ sel_w,               // [T,32]
                               int* __restrict__ expert_count,          // [32]
                               int T, int offset, float rsf) {
    int t = blockIdx.x;
    int e = threadIdx.x;              // 0..255
    __shared__ float s_s[E_GLOBAL];
    __shared__ float s_sb[E_GLOBAL];
    __shared__ int   kept[N_GROUP];
    __shared__ float wv[N_GROUP];     // per-warp (group) argmax value
    __shared__ int   wi[N_GROUP];     // per-warp (group) argmax global index
    __shared__ int   sel_idx[TOP_K];

    __shared__ float gsc[N_GROUP];
    float lg = logits[(long)t * E_GLOBAL + e];
    float s = 1.0f / (1.0f + __expf(-lg));
    float sb = s + __bfloat162float(bias[e]);
    s_s[e] = s;
    s_sb[e] = sb;
    if (e < E_LOCAL) sel_w[(long)t * E_LOCAL + e] = 0.0f;

    // group top-2 sum: one warp (32 lanes) per group via shuffle reduction
    {
        float m1 = sb, m2 = -1e30f;
        for (int o = 16; o >= 1; o >>= 1) {
            float om1 = __shfl_down_sync(0xffffffffu, m1, o);
            float om2 = __shfl_down_sync(0xffffffffu, m2, o);
            float nm1 = fmaxf(m1, om1);
            float nm2 = fmaxf(fminf(m1, om1), fmaxf(m2, om2));
            m1 = nm1; m2 = nm2;
        }
        if ((e & 31) == 0) gsc[e >> 5] = m1 + m2;
    }
    __syncthreads();

    // thread 0: keep top-4 groups (leftmost on ties), 8 elements
    if (e == 0) {
        for (int g = 0; g < N_GROUP; ++g) kept[g] = 0;
        for (int k = 0; k < TOPK_GROUP; ++k) {
            int best = -1; float bv = -1e30f;
            for (int g = 0; g < N_GROUP; ++g)
                if (kept[g] == 0 && gsc[g] > bv) { bv = gsc[g]; best = g; }
            kept[best] = 1;
        }
    }
    __syncthreads();

    // parallel top-8 over eligible experts (experts in kept groups). Two-level
    // leftmost-argmax: each warp (= one 32-expert group) reduces its 32 lanes with
    // shuffles, then thread 0 scans the 8 warp leaders. On ties the LOWER global
    // index wins (== reference leftmost-max): a warp's lanes are a contiguous
    // ascending index range and warp g covers strictly-lower indices than warp g+1,
    // so updating ONLY on strictly-greater value keeps the leftmost. This replaces
    // the old per-k 256-wide shared-mem tree reduction (~11 syncs/k -> 3 syncs/k).
    const int lane = e & 31;
    const int wid  = e >> 5;          // 0..7, one warp per group of 32 experts
    float ev = kept[wid] ? s_sb[e] : -1e30f;
    for (int k = 0; k < TOP_K; ++k) {
        float v = ev; int idx = e;
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, o);
            int   oi = __shfl_down_sync(0xffffffffu, idx, o);
            // keep max; on value ties keep the LOWER global index (leftmost-max).
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { wv[wid] = v; wi[wid] = idx; }
        __syncthreads();
        if (e == 0) {
            float bv = wv[0]; int bi = wi[0];
            for (int g = 1; g < N_GROUP; ++g)
                if (wv[g] > bv) { bv = wv[g]; bi = wi[g]; }  // strict-> leftmost
            sel_idx[k] = bi;
        }
        __syncthreads();
        if (e == sel_idx[k]) ev = -1e30f;   // mask winner out for next round
        __syncthreads();
    }

    // thread 0: normalize from unbiased sigmoid scores and scatter weights
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) sum += s_s[sel_idx[k]];
        float inv = rsf / (sum + 1e-20f);
        for (int k = 0; k < TOP_K; ++k) {
            int ge = sel_idx[k];
            int le = ge - offset;
            if (le >= 0 && le < E_LOCAL) {
                sel_w[(long)t * E_LOCAL + le] = s_s[ge] * inv;
                atomicAdd(&expert_count[le], 1);
            }
        }
    }
}

// prefix-sum of counts -> offsets[33]; also zeros cursor[32] (fused, saves a launch).
__global__ void prefix_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                              int* __restrict__ cursor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc = 0;
        for (int i = 0; i < E_LOCAL; ++i) { offsets[i] = acc; acc += counts[i]; if (cursor) cursor[i] = 0; }
        offsets[E_LOCAL] = acc;
    }
}

__global__ void scatter_kernel(const float* __restrict__ sel_w,   // [T,32]
                               const int* __restrict__ offsets,    // [33]
                               int* __restrict__ cursor,           // [32]
                               int* __restrict__ perm_token,       // [R]
                               float* __restrict__ perm_w,         // [R]
                               int T,
                               int* __restrict__ tok_rows,         // [T,MAXK] or null
                               float* __restrict__ tok_w,          // [T,MAXK] or null
                               int* __restrict__ tok_n) {          // [T] or null
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * E_LOCAL;
    if (idx >= total) return;
    int t = idx / E_LOCAL;
    int le = idx % E_LOCAL;
    float w = sel_w[idx];
    if (w > 0.0f) {
        int pos = offsets[le] + atomicAdd(&cursor[le], 1);
        perm_token[pos] = t;
        perm_w[pos] = w;
        // build per-token inverse list for the atomic-free finalize
        if (tok_n) {
            int k = atomicAdd(&tok_n[t], 1);
            if (k < MAXK) { tok_rows[t * MAXK + k] = pos; tok_w[t * MAXK + k] = w; }
        }
    }
}

// Fixed grids for the graph path (grid-stride; decouple launch count from M_max).
constexpr int GRID_EW = 148 * 16;   // element-wise saturation
constexpr int GRID_ROWS = 148 * 4;  // row/row-block based

// Fused gather: copy fp8 row (448 uint4) AND the 56 per-token block scales in one
// launch. block=H/16=448 threads; threads 0..55 additionally write a_scale.
__global__ void gather_rows_scale(const uint4* __restrict__ hid,       // [T,H/16]
                                  const float* __restrict__ hscale,     // [56,T]
                                  const int* __restrict__ perm_token,   // [R]
                                  uint4* __restrict__ a_fp8,            // [R,H/16]
                                  float* __restrict__ a_scale,          // [R,56]
                                  int T, long R, const int* __restrict__ Rptr) {
    long Rn = Rptr ? (long)*Rptr : R;
    int c = threadIdx.x;                 // 0..447
    for (long r = blockIdx.x; r < Rn; r += gridDim.x) {
        int t = perm_token[r];
        a_fp8[r * (H / 16) + c] = hid[(long)t * (H / 16) + c];
        if (c < HB) a_scale[r * HB + c] = hscale[(long)c * T + t];
    }
}

// SwiGLU + per-token 1x128 fp8 quant. grid=(G, IB), block=128; grid-stride over rows.
__global__ void swiglu_quant_kernel(const __nv_bfloat16* __restrict__ g1, // [M,4096]
                                    __nv_fp8_e4m3* __restrict__ c_fp8,     // [M,2048]
                                    float* __restrict__ c_scale,           // [M,16]
                                    long R, const int* __restrict__ Rptr) {
    long Rn = Rptr ? (long)*Rptr : R;
    int kb = blockIdx.y;               // 0..15
    int tid = threadIdx.x;             // 0..127
    int j = kb * BLK + tid;            // 0..2047
    __shared__ float sred[128];
    for (long r = blockIdx.x; r < Rn; r += gridDim.x) {
        float gate = __bfloat162float(g1[r * G1OUT + j]);
        float up   = __bfloat162float(g1[r * G1OUT + I + j]);
        float val  = (up / (1.0f + __expf(-up))) * gate;
        sred[tid] = fabsf(val);
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) sred[tid] = fmaxf(sred[tid], sred[tid + s]);
            __syncthreads();
        }
        float scale = fmaxf(sred[0], 1e-4f) / 448.0f;
        if (tid == 0) c_scale[r * IB + kb] = scale;
        c_fp8[r * I + j] = (__nv_fp8_e4m3)(val / scale);
        __syncthreads();
    }
}

// Faster SwiGLU + per-token 1x128 fp8 quant: ONE WARP per (row, kb) 128-col group.
// Each lane handles 4 strided columns (lane, lane+32, lane+64, lane+96 -> coalesced
// 32-element reads per step), computes SwiGLU, tracks its max-abs, then the 128-element
// block max is a pure warp shuffle reduction (no shared memory, no __syncthreads). The
// old kernel paid a 7-sync shared-mem tree reduction PER ROW and ran at <10% of HBM BW;
// this removes all syncs/shared and is memory-bound. Math identical to swiglu_quant_kernel.
__global__ void swiglu_quant_warp(const __nv_bfloat16* __restrict__ g1, // [M,4096]
                                  __nv_fp8_e4m3* __restrict__ c_fp8,     // [M,2048]
                                  float* __restrict__ c_scale,           // [M,16]
                                  long R, const int* __restrict__ Rptr) {
    long Rn = Rptr ? (long)*Rptr : R;
    long total = Rn * IB;                                   // (row, kb) groups
    long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    long nwarps = ((long)gridDim.x * blockDim.x) >> 5;
    for (long gid = warp; gid < total; gid += nwarps) {
        long r = gid / IB;
        int kb = (int)(gid % IB);
        long bg = r * G1OUT + (long)kb * BLK;               // gate base
        long bu = r * G1OUT + I + (long)kb * BLK;           // up base
        float v[4];
        float m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = lane + i * 32;                        // 0..127
            float gate = __bfloat162float(g1[bg + col]);
            float up   = __bfloat162float(g1[bu + col]);
            float val  = (up / (1.0f + __expf(-up))) * gate;
            v[i] = val;
            m = fmaxf(m, fabsf(val));
        }
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float scale = fmaxf(m, 1e-4f) / 448.0f;
        if (lane == 0) c_scale[r * IB + kb] = scale;
        long bc = r * I + (long)kb * BLK;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = lane + i * 32;
            c_fp8[bc + col] = (__nv_fp8_e4m3)(v[i] / scale);
        }
    }
}

// Weighted scatter-accumulate directly into the bf16 output via bf16 atomics.
// Fuses the f32 accumulate + cast (removes a separate f32 buffer, its zeroing, and a
// cast kernel). Under the loose MoE tolerance (atol 1, rtol 0.3) bf16 accumulation of
// the <=8 per-token expert contributions is well within bounds.
__global__ void scatter_acc_bf16_kernel(const __nv_bfloat16* __restrict__ O, // [R,H]
                                        const int* __restrict__ perm_token,
                                        const float* __restrict__ perm_w,
                                        __nv_bfloat16* __restrict__ out,       // [T,H]
                                        long R, const int* __restrict__ Rptr) {
    long Rn = Rptr ? (long)*Rptr : R;
    long total = Rn * H;
    long stride = (long)gridDim.x * blockDim.x;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        long r = idx / H;
        int h = idx % H;
        float v = __bfloat162float(O[idx]) * perm_w[r];
        atomicAdd(&out[(long)perm_token[r] * H + h], __float2bfloat16(v));
    }
}

// Atomic-free weighted finalize: one block per output token. Each token has <=MAXK
// local-expert contributions (routed rows in O). The block loads its (row,weight)
// list once, then each thread accumulates its H-columns across the <=MAXK rows in
// registers and writes the result EXACTLY ONCE (vectorized bf16x2). No atomics, each
// O element read once and each out row written once -> ~HBM-floor traffic. Tokens with
// no local contribution (n==0) write zeros, so `out` needs no pre-zeroing.
__global__ void finalize_pertoken(const __nv_bfloat162* __restrict__ O,   // [R,H/2]
                                  const int* __restrict__ tok_rows,        // [T,MAXK]
                                  const float* __restrict__ tok_w,         // [T,MAXK]
                                  const int* __restrict__ tok_n,           // [T]
                                  __nv_bfloat162* __restrict__ out,        // [T,H/2]
                                  int T) {
    constexpr int H2 = H / 2;
    __shared__ int   s_r[MAXK];
    __shared__ float s_w[MAXK];
    for (int t = blockIdx.x; t < T; t += gridDim.x) {
        int n = tok_n[t];
        if (n > MAXK) n = MAXK;
        if (threadIdx.x < n) {
            s_r[threadIdx.x] = tok_rows[t * MAXK + threadIdx.x];
            s_w[threadIdx.x] = tok_w[t * MAXK + threadIdx.x];
        }
        __syncthreads();
        for (int h = threadIdx.x; h < H2; h += blockDim.x) {
            float2 acc = make_float2(0.f, 0.f);
            for (int k = 0; k < n; ++k) {
                float2 vf = __bfloat1622float2(O[(long)s_r[k] * H2 + h]);
                acc.x += vf.x * s_w[k];
                acc.y += vf.y * s_w[k];
            }
            out[(long)t * H2 + h] = __float22bfloat162_rn(acc);
        }
        __syncthreads();  // ensure all reads of s_r/s_w done before next t overwrites
    }
}

inline long ceil_div(long a, long b) { return (a + b - 1) / b; }

} // namespace

// ===================== EAGER path (large-seq fallback) =====================
std::vector<torch::Tensor> route_gather(torch::Tensor logits, torch::Tensor bias,
                                        torch::Tensor hid, torch::Tensor hscale,
                                        int64_t offset, double rsf) {
    int T = logits.size(0);
    auto dev = hid.device();
    auto i32 = torch::dtype(torch::kInt32).device(dev);
    auto f32 = torch::dtype(torch::kFloat32).device(dev);
    auto fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(dev);
    auto cs = at::cuda::getCurrentCUDAStream().stream();

    auto sel_w = torch::empty({T, E_LOCAL}, f32);
    auto counts = torch::zeros({E_LOCAL}, i32);
    routing_kernel<<<T, E_GLOBAL, 0, cs>>>(
        logits.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        sel_w.data_ptr<float>(), counts.data_ptr<int>(), T, (int)offset, (float)rsf);
    auto offsets = torch::empty({E_LOCAL + 1}, i32);
    prefix_kernel<<<1, 1, 0, cs>>>(counts.data_ptr<int>(), offsets.data_ptr<int>(), nullptr);
    auto offsets_cpu = offsets.to(torch::kCPU);
    long R = offsets_cpu.data_ptr<int>()[E_LOCAL];
    auto a_fp8 = torch::empty({R, H}, fp8);
    auto a_scale = torch::empty({R, HB}, f32);
    auto perm_token = torch::empty({R}, i32);
    auto perm_w = torch::empty({R}, f32);
    // per-token inverse list (token -> its routed rows + weights) for atomic-free finalize
    auto tok_rows = torch::empty({T, MAXK}, i32);
    auto tok_w = torch::empty({T, MAXK}, f32);
    auto tok_n = torch::zeros({T}, i32);
    if (R == 0) return {a_fp8, a_scale, perm_token, perm_w, offsets, tok_rows, tok_w, tok_n};
    auto cursor = torch::zeros({E_LOCAL}, i32);
    scatter_kernel<<<ceil_div((long)T * E_LOCAL, 256), 256, 0, cs>>>(
        sel_w.data_ptr<float>(), offsets.data_ptr<int>(), cursor.data_ptr<int>(),
        perm_token.data_ptr<int>(), perm_w.data_ptr<float>(), T,
        tok_rows.data_ptr<int>(), tok_w.data_ptr<float>(), tok_n.data_ptr<int>());
    gather_rows_scale<<<R, H / 16, 0, cs>>>(
        reinterpret_cast<const uint4*>(hid.data_ptr()), hscale.data_ptr<float>(),
        perm_token.data_ptr<int>(), reinterpret_cast<uint4*>(a_fp8.data_ptr()),
        a_scale.data_ptr<float>(), T, R, nullptr);
    return {a_fp8, a_scale, perm_token, perm_w, offsets, tok_rows, tok_w, tok_n};
}

std::vector<torch::Tensor> swiglu_quant(torch::Tensor g1) {
    long R = g1.size(0);
    auto dev = g1.device();
    auto fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(dev);
    auto f32 = torch::dtype(torch::kFloat32).device(dev);
    auto c_fp8 = torch::empty({R, I}, fp8);
    auto c_scale = torch::empty({R, IB}, f32);
    if (R == 0) return {c_fp8, c_scale};
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    // one warp per (row,kb) group; block=256 (8 warps), grid-stride, capped grid.
    long ngroups = R * IB;
    int blocks = (int)std::min((ngroups * 32 + 255) / 256, (long)(148 * 32));
    swiglu_quant_warp<<<blocks, 256, 0, cs>>>(
        reinterpret_cast<const __nv_bfloat16*>(g1.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(c_fp8.data_ptr()), c_scale.data_ptr<float>(), R, nullptr);
    return {c_fp8, c_scale};
}

torch::Tensor finalize(torch::Tensor O, torch::Tensor tok_rows, torch::Tensor tok_w,
                       torch::Tensor tok_n, int64_t T) {
    auto dev = O.device();
    auto bf16 = torch::dtype(torch::kBFloat16).device(dev);
    // atomic-free: every token row is written exactly once, so no pre-zeroing needed.
    auto out = torch::empty({T, H}, bf16);
    long R = O.size(0);
    if (R == 0) { out.zero_(); return out; }
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    int gr = (int)std::min((long)T, (long)(148 * 16));
    finalize_pertoken<<<gr, 512, 0, cs>>>(
        reinterpret_cast<const __nv_bfloat162*>(O.data_ptr()),
        tok_rows.data_ptr<int>(), tok_w.data_ptr<float>(), tok_n.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat162*>(out.data_ptr()), (int)T);
    return out;
}

// ===================== GRAPH-SAFE ops (write into provided buffers) ==========
void route_g(torch::Tensor logits, torch::Tensor bias, torch::Tensor sel_w,
             torch::Tensor counts, int64_t offset, double rsf) {
    int T = logits.size(0);
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    routing_kernel<<<T, E_GLOBAL, 0, cs>>>(
        logits.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        sel_w.data_ptr<float>(), counts.data_ptr<int>(), T, (int)offset, (float)rsf);
}

void prefix_g(torch::Tensor counts, torch::Tensor offsets, torch::Tensor cursor) {
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    prefix_kernel<<<1, 1, 0, cs>>>(counts.data_ptr<int>(), offsets.data_ptr<int>(),
                                   cursor.data_ptr<int>());
}

void scatter_g(torch::Tensor sel_w, torch::Tensor offsets, torch::Tensor cursor,
               torch::Tensor perm_token, torch::Tensor perm_w) {
    int T = sel_w.size(0);
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    scatter_kernel<<<ceil_div((long)T * E_LOCAL, 256), 256, 0, cs>>>(
        sel_w.data_ptr<float>(), offsets.data_ptr<int>(), cursor.data_ptr<int>(),
        perm_token.data_ptr<int>(), perm_w.data_ptr<float>(), T, nullptr, nullptr, nullptr);
}

// gather into fixed [M_max,H] buffer, guard rows>=R via offsets[E_LOCAL].
void gather_g(torch::Tensor hid, torch::Tensor hscale, torch::Tensor perm_token,
              torch::Tensor offsets, torch::Tensor a_fp8, torch::Tensor a_scale) {
    int T = hid.size(0);
    long M = a_fp8.size(0);
    const int* Rptr = offsets.data_ptr<int>() + E_LOCAL;
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    int gr = (int)std::min((long)GRID_ROWS, M);
    gather_rows_scale<<<gr, H / 16, 0, cs>>>(
        reinterpret_cast<const uint4*>(hid.data_ptr()), hscale.data_ptr<float>(),
        perm_token.data_ptr<int>(), reinterpret_cast<uint4*>(a_fp8.data_ptr()),
        a_scale.data_ptr<float>(), T, M, Rptr);
}

void swiglu_quant_g(torch::Tensor g1, torch::Tensor offsets, torch::Tensor c_fp8,
                    torch::Tensor c_scale) {
    long M = g1.size(0);
    const int* Rptr = offsets.data_ptr<int>() + E_LOCAL;
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned)std::min((long)GRID_ROWS, M), IB);
    swiglu_quant_kernel<<<grid, BLK, 0, cs>>>(
        reinterpret_cast<const __nv_bfloat16*>(g1.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(c_fp8.data_ptr()), c_scale.data_ptr<float>(), M, Rptr);
}

// Graph path: weighted scatter-accumulate directly into the bf16 output (bf16 atomics).
void scatter_acc_g(torch::Tensor O, torch::Tensor perm_token, torch::Tensor perm_w,
                   torch::Tensor offsets, torch::Tensor out) {
    long M = O.size(0);
    const int* Rptr = offsets.data_ptr<int>() + E_LOCAL;
    auto cs = at::cuda::getCurrentCUDAStream().stream();
    int ge = (int)std::min((long)GRID_EW, ceil_div(M * H, 256));
    scatter_acc_bf16_kernel<<<ge, 256, 0, cs>>>(
        reinterpret_cast<const __nv_bfloat16*>(O.data_ptr()), perm_token.data_ptr<int>(),
        perm_w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, Rptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("route_gather", &route_gather);
    mod.def("swiglu_quant", &swiglu_quant);
    mod.def("finalize", &finalize);
    mod.def("route_g", &route_g);
    mod.def("prefix_g", &prefix_g);
    mod.def("scatter_g", &scatter_g);
    mod.def("gather_g", &gather_g);
    mod.def("swiglu_quant_g", &swiglu_quant_g);
    mod.def("scatter_acc_g", &scatter_acc_g);
}
