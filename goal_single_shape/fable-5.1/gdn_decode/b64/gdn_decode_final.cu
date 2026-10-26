// gdn_decode_final.cu — Gated Delta Net decode (Qwen3-Next), B200-tuned, written from scratch.
// Definition: gdn_decode_qk4_v8_d128_k_last (flashinfer-trace). State layout [B, HV, V, K] (k-last), K = V = 128,
// HQ = HK = 4, HV = 8 (GVA: value head h uses q/k head h/2).
//
// Math per value head (b, h), per state row v (rows are independent):
//   g      = exp(-exp(A_log[h]) * softplus(a[b,h] + dt_bias[h]))
//   beta   = sigmoid(b[b,h])
//   old_v  = g * <S[v,:], k>
//   new_v  = beta * v[v] + (1 - beta) * old_v
//   S'[v,:]= g * S[v,:] + k * (new_v - old_v)
//   out[v] = scale * <q, S'[v,:]>
//
// Design (memory-bound kernel: 33.5 MB read + 33.5 MB write of fp32 state):
//   * persistent CTAs (3 per SM), each streaming 16 KB state chunks (32 rows) through a 4-slot TMA (cp.async.bulk) ring;
//     chunk c = blockIdx.x + it * gridDim.x keeps the global access front contiguous in memory;
//   * every operand a chunk needs (state rows, k/q/v rows, a/b) is fetched by TMA into the slot — no LSU global loads
//     in the loop (they proved pathologically slow next to the TMA stream);
//   * 8 compute warps, 8 lanes per row (4 rows per warp-instruction), 3-step shuffle reductions; rows are updated in place
//     in shared memory and written back with one 16 KB TMA bulk store per chunk, bf16 outputs staged and bulk-stored too;
//   * a 9th (producer) warp issues all TMA stores and slot refills, so compute warps never wait on TMA issue.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gdn {

constexpr int HQ = 4, HV = 8, D = 128;
constexpr int CHUNK_ROWS = 32;                        // rows per chunk
constexpr int CHUNK = CHUNK_ROWS * D * 4;             // 16 KB
constexpr int SLOT = CHUNK + 1024;                    // + k 256 | q 256 | v 256 | a 16 | b 16 | out 64
constexpr int CPH = D / CHUNK_ROWS;                   // chunks per head
constexpr int NBUF = 4;                               // ring depth
constexpr int NW = CHUNK_ROWS / 4;                    // compute warps (8)
constexpr int THREADS = NW * 32 + 32;                 // + producer warp
constexpr int SMEM = SLOT * NBUF;
constexpr int CTAS_PER_SM = 3;
constexpr uint32_t TX = CHUNK + 256 + 256 + 256 + 16 + 16;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count)); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra LAB_WAIT;\n}\n" ::"r"(smem_u32(bar)), "r"(phase) : "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ void bulk_s2g(void* dst, const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory"); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(__expf(x)); }
__device__ __forceinline__ float sum8(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 4); x += __shfl_xor_sync(0xffffffffu, x, 2); x += __shfl_xor_sync(0xffffffffu, x, 1); return x; }
__device__ __forceinline__ float bf16_to_f(uint16_t u) { return __uint_as_float(((uint32_t)u) << 16); }

__global__ void __launch_bounds__(THREADS)
gdn_decode_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                  const __nv_bfloat16* __restrict__ v, const float* __restrict__ state,
                  const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ a,
                  const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ b, float scale,
                  __nv_bfloat16* __restrict__ out, float* __restrict__ new_state, int nchunks) {
    extern __shared__ __align__(1024) char smem[];
    __shared__ __align__(8) uint64_t full[NBUF], computed[NBUF];
    __shared__ float s_alog[HV], s_dtb[HV];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sub = lane & 7, rsel = lane >> 3;
    const int rloc = warp * 4 + rsel;                                   // row within chunk for compute warps

    auto issue = [&](int c, int j) {                                     // fetch chunk c into slot j
        const int bh = c / CPH, bidx = bh / HV, h = bh % HV, qkh = bidx * HQ + (h >> 1);
        char* slot = smem + j * SLOT;
        mbar_expect_tx(&full[j], TX);
        bulk_g2s(slot, state + (size_t)c * (CHUNK / 4), CHUNK, &full[j]);
        bulk_g2s(slot + CHUNK, k + (size_t)qkh * D, 256, &full[j]);
        bulk_g2s(slot + CHUNK + 256, q + (size_t)qkh * D, 256, &full[j]);
        bulk_g2s(slot + CHUNK + 512, v + (size_t)bh * D, 256, &full[j]);
        bulk_g2s(slot + CHUNK + 768, a + (size_t)bidx * HV, 16, &full[j]);
        bulk_g2s(slot + CHUNK + 784, b + (size_t)bidx * HV, 16, &full[j]);
    };
    if (threadIdx.x == 0) {
        for (int i = 0; i < NBUF; ++i) { mbar_init(&full[i], 1); mbar_init(&computed[i], NW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int j = 0; j < NBUF; ++j) { const int c = blockIdx.x + j * gridDim.x; if (c < nchunks) issue(c, j); }
    }
    if (threadIdx.x >= 32 && threadIdx.x < 32 + HV) { s_alog[threadIdx.x - 32] = A_log[threadIdx.x - 32]; s_dtb[threadIdx.x - 32] = dt_bias[threadIdx.x - 32]; }
    __syncthreads();

    if (warp == NW) {                                                    // ---- producer warp: stores + refills
        if (lane == 0) {
            for (int it = 0, c = blockIdx.x; c < nchunks; ++it, c += gridDim.x) {
                const int j = it % NBUF;
                const int bh = c / CPH, rowbase = (c % CPH) * CHUNK_ROWS;
                char* slot = smem + j * SLOT;
                mbar_wait(&computed[j], (it / NBUF) & 1);
                bulk_s2g(new_state + ((size_t)bh * D + rowbase) * D, slot, CHUNK);
                bulk_s2g(out + (size_t)bh * D + rowbase, slot + CHUNK + 800, CHUNK_ROWS * 2);
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                const int cn = c + NBUF * gridDim.x;
                if (cn < nchunks) { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); issue(cn, j); }
            }
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        return;
    }
    // ---- compute warps
    for (int it = 0, c = blockIdx.x; c < nchunks; ++it, c += gridDim.x) {
        const int j = it % NBUF;
        const int bh = c / CPH, h = bh % HV;
        const int row = (c % CPH) * CHUNK_ROWS + rloc;
        char* slot = smem + j * SLOT;
        mbar_wait(&full[j], (it / NBUF) & 1);
        const uint2* ks = reinterpret_cast<const uint2*>(slot + CHUNK) + sub;
        const uint2* qs = reinterpret_cast<const uint2*>(slot + CHUNK + 256) + sub;
        float kk[16], qq[16];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {                                 // this lane's 16 columns: sub*4 + 32*jj + {0..3}
            uint2 kraw = ks[jj * 8], qraw = qs[jj * 8];
            float2 t;
            t = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&kraw.x)); kk[4 * jj] = t.x; kk[4 * jj + 1] = t.y;
            t = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&kraw.y)); kk[4 * jj + 2] = t.x; kk[4 * jj + 3] = t.y;
            t = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&qraw.x)); qq[4 * jj] = t.x; qq[4 * jj + 1] = t.y;
            t = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&qraw.y)); qq[4 * jj + 2] = t.x; qq[4 * jj + 3] = t.y;
        }
        const float vv = bf16_to_f(reinterpret_cast<const uint16_t*>(slot + CHUNK + 512)[row]);
        const float av = bf16_to_f(reinterpret_cast<const uint16_t*>(slot + CHUNK + 768)[h]);
        const float bv = bf16_to_f(reinterpret_cast<const uint16_t*>(slot + CHUNK + 784)[h]);
        const float gdec = __expf(-__expf(s_alog[h]) * softplus_f(av + s_dtb[h]));
        const float beta = 1.f / (1.f + __expf(-bv));
        float4* srow = reinterpret_cast<float4*>(slot) + rloc * (D / 4) + sub;
        float4 s[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) s[jj] = srow[jj * 8];
        const float d0 = s[0].x * kk[0] + s[0].y * kk[1] + s[0].z * kk[2] + s[0].w * kk[3];
        const float d1 = s[1].x * kk[4] + s[1].y * kk[5] + s[1].z * kk[6] + s[1].w * kk[7];
        const float d2 = s[2].x * kk[8] + s[2].y * kk[9] + s[2].z * kk[10] + s[2].w * kk[11];
        const float d3 = s[3].x * kk[12] + s[3].y * kk[13] + s[3].z * kk[14] + s[3].w * kk[15];
        const float old_v = gdec * sum8((d0 + d1) + (d2 + d3));
        const float new_v = beta * vv + (1.f - beta) * old_v;
        const float delta = new_v - old_v;
        float oo[4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            float4 hrow;
            hrow.x = fmaf(gdec, s[jj].x, kk[4 * jj] * delta);
            hrow.y = fmaf(gdec, s[jj].y, kk[4 * jj + 1] * delta);
            hrow.z = fmaf(gdec, s[jj].z, kk[4 * jj + 2] * delta);
            hrow.w = fmaf(gdec, s[jj].w, kk[4 * jj + 3] * delta);
            srow[jj * 8] = hrow;                                          // update in place (store source)
            oo[jj] = hrow.x * qq[4 * jj] + hrow.y * qq[4 * jj + 1] + hrow.z * qq[4 * jj + 2] + hrow.w * qq[4 * jj + 3];
        }
        const float o = sum8((oo[0] + oo[1]) + (oo[2] + oo[3]));
        if (sub == 0) reinterpret_cast<uint16_t*>(slot + CHUNK + 800)[rloc] = __bfloat16_as_ushort(__float2bfloat16(scale * o));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");    // make smem writes visible to the TMA store
        __syncwarp();
        if (lane == 0) mbar_arrive(&computed[j]);
    }
}

}  // namespace gdn

// Host launcher. All pointers are device pointers; B is the batch size; out is bf16 [B,1,8,128]; new_state fp32 [B,8,128,128].
extern "C" int gdn_decode_launch(const void* q, const void* k, const void* v, const void* state, const void* A_log,
                                 const void* a, const void* dt_bias, const void* b, float scale, void* out,
                                 void* new_state, int B, cudaStream_t stream) {
    using namespace gdn;
    static int grid = 0;
    if (grid == 0) {
        cudaError_t e = cudaFuncSetAttribute(gdn_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        if (e != cudaSuccess) return (int)e;
        int dev, sms; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        grid = sms * CTAS_PER_SM;
    }
    const int nchunks = B * HV * D / CHUNK_ROWS;
    gdn_decode_kernel<<<grid, THREADS, SMEM, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v, (const float*)state, (const float*)A_log,
        (const __nv_bfloat16*)a, (const float*)dt_bias, (const __nv_bfloat16*)b, scale, (__nv_bfloat16*)out, (float*)new_state, nchunks);
    return (int)cudaGetLastError();
}
// Variant-style entry point (kept for the benchmark scripts): variant is ignored.
extern "C" int gdn_launch(int, const void* q, const void* k, const void* v, const void* state, const void* A_log,
                          const void* a, const void* dt_bias, const void* b, float scale, void* out, void* new_state,
                          int B, cudaStream_t stream) {
    return gdn_decode_launch(q, k, v, state, A_log, a, dt_bias, b, scale, out, new_state, B, stream);
}
