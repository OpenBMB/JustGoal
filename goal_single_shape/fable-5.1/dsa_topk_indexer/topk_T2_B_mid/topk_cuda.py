"""CUDA kernel for DSA top-k indexer — single .cu file compilation."""

import torch
import torch.utils.cpp_extension as ext
import os, tempfile

# Single .cu file: kernel + torch binding in one NVCC translation unit.
CUDA_FULL_SRC = r"""
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define H_DIM    64
#define D_DIM    128
#define PAGE_SZ  64
#define K_STRIDE 132     /* 128 fp8 bytes + 4 float32 bytes per token */
#define TOPK_K   2048
#define BSORT_N  4096    /* next power-of-2 >= max_seq_len (2817) */

/* Shared memory offsets (bytes):
 *  q_sh      [0 ..  8191]  : H_DIM * D_DIM  = 8192
 *  wt_sh     [8192 .. 8447]: H_DIM * 4      =  256
 *  scores_sh [8448..24831] : BSORT_N * 4    = 16384
 *  ids_sh    [24832..41215]: BSORT_N * 4    = 16384
 *  Total = 41216 bytes
 */
#define SMEM_Q_OFF  0
#define SMEM_W_OFF  (H_DIM * D_DIM)
#define SMEM_S_OFF  (SMEM_W_OFF + H_DIM * 4)
#define SMEM_I_OFF  (SMEM_S_OFF + BSORT_N * 4)
#define SMEM_BYTES  (SMEM_I_OFF + BSORT_N * 4)

/* fp8_e4m3fn -> float via half intrinsic */
__device__ __forceinline__ float fp8_f32(uint8_t x) {
    return __half2float(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3));
}

/* Convert 4 packed fp8 bytes (uint32) to 4 floats and dot with another 4 fp8. */
__device__ __forceinline__ float dot4(uint32_t q4, uint32_t k4, float scale) {
    __half2 qh0 = __nv_cvt_fp8x2_to_halfraw2(q4 & 0xFFFFu, __NV_E4M3);
    __half2 qh1 = __nv_cvt_fp8x2_to_halfraw2(q4 >> 16,     __NV_E4M3);
    __half2 kh0 = __nv_cvt_fp8x2_to_halfraw2(k4 & 0xFFFFu, __NV_E4M3);
    __half2 kh1 = __nv_cvt_fp8x2_to_halfraw2(k4 >> 16,     __NV_E4M3);
    float2 qf0 = __half22float2(qh0), qf1 = __half22float2(qh1);
    float2 kf0 = __half22float2(kh0), kf1 = __half22float2(kh1);
    kf0.x *= scale; kf0.y *= scale; kf1.x *= scale; kf1.y *= scale;
    return qf0.x*kf0.x + qf0.y*kf0.y + qf1.x*kf1.x + qf1.y*kf1.y;
}

__global__ void dsa_topk_kernel(
    const uint8_t* __restrict__ q_fp8,   /* (B, H_DIM, D_DIM) uint8  */
    const int8_t*  __restrict__ k_cache, /* (num_pages, PAGE_SZ, 1, K_STRIDE) int8 */
    const float*   __restrict__ weights, /* (B, H_DIM) float32 */
    const int*     __restrict__ seq_lens,/* (B,) int32 */
    const int*     __restrict__ blk_tbl, /* (B, max_pages) int32 */
    int*           __restrict__ out,     /* (B, TOPK_K) int32 */
    int max_pages
) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bsz = blockDim.x;   /* 256 */

    const int seq_len = seq_lens[b];

    extern __shared__ char smem[];
    uint8_t* q_sh  = (uint8_t*)(smem + SMEM_Q_OFF);
    float*   wt_sh = (float*)  (smem + SMEM_W_OFF);
    float*   sc_sh = (float*)  (smem + SMEM_S_OFF);
    int*     id_sh = (int*)    (smem + SMEM_I_OFF);

    /* Load q[b] into shared (8192 bytes; 256 threads -> 32 bytes each) */
    for (int i = tid; i < H_DIM * D_DIM / 4; i += bsz)
        ((uint32_t*)q_sh)[i] = ((const uint32_t*)(q_fp8 + (long)b * H_DIM * D_DIM))[i];
    /* Load weights[b] */
    for (int i = tid; i < H_DIM; i += bsz)
        wt_sh[i] = weights[b * H_DIM + i];
    __syncthreads();

    /* Score each token: each thread handles stride-bsz tokens. */
    for (int t = tid; t < seq_len; t += bsz) {
        const int page   = blk_tbl[b * max_pages + t / PAGE_SZ];
        const int pg_off = t % PAGE_SZ;
        /* Each page stores all FP8 rows, followed by all FP32 scales. */
        const uint8_t* kp =
            (const uint8_t*)k_cache + (long)page * PAGE_SZ * K_STRIDE + pg_off * D_DIM;
        const float kscale = ((const float*)((const uint8_t*)k_cache + (long)page * PAGE_SZ * K_STRIDE + PAGE_SZ * D_DIM))[pg_off];

        float score = 0.0f;
        for (int h = 0; h < H_DIM; h++) {
            const uint32_t* qh32 = (const uint32_t*)(q_sh + h * D_DIM);
            const uint32_t* kh32 = (const uint32_t*)kp;
            float dot = 0.0f;
            /* 32 uint32 loads per head (4 fp8 per load -> 128 total) */
            for (int d = 0; d < D_DIM / 4; d++)
                dot += dot4(qh32[d], kh32[d], kscale);
            score += (isnan(dot) ? dot : fmaxf(dot, 0.0f)) * wt_sh[h];
        }
        sc_sh[t] = score;
        id_sh[t] = page * PAGE_SZ + pg_off;
    }

    /* Pad to BSORT_N */
    for (int t = seq_len + tid; t < BSORT_N; t += bsz) {
        sc_sh[t] = -FLT_MAX;
        id_sh[t] = -1;
    }
    __syncthreads();

    int* out_b = out + (long)b * TOPK_K;

    if (seq_len <= TOPK_K) {
        for (int t = tid; t < seq_len; t += bsz)  out_b[t] = id_sh[t];
        for (int t = seq_len + tid; t < TOPK_K; t += bsz) out_b[t] = -1;
    } else {
        /* Bitonic sort descending over BSORT_N elements */
        for (int k = 2; k <= BSORT_N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < BSORT_N; i += bsz) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        bool swap;
                        if ((i & k) == 0) swap = (sc_sh[i] < sc_sh[ixj] || (!isnan(sc_sh[i]) && isnan(sc_sh[ixj]))); /* even: desc */
                        else              swap = (sc_sh[i] > sc_sh[ixj] || (isnan(sc_sh[i]) && !isnan(sc_sh[ixj]))); /* odd:  asc  */
                        if (swap) {
                            float ts = sc_sh[i]; sc_sh[i] = sc_sh[ixj]; sc_sh[ixj] = ts;
                            int   ti = id_sh[i]; id_sh[i] = id_sh[ixj]; id_sh[ixj] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int t = tid; t < TOPK_K; t += bsz) out_b[t] = id_sh[t];
    }
}

/* ── Torch binding ──────────────────────────────────────────────────────── */

torch::Tensor dsa_topk(
    torch::Tensor q_u8,     /* (B, H_DIM, D_DIM) uint8 */
    torch::Tensor k_cache,  /* (num_pages, PAGE_SZ, 1, K_STRIDE) int8 */
    torch::Tensor weights,  /* (B, H_DIM) float32 */
    torch::Tensor seq_lens, /* (B,) int32 */
    torch::Tensor blk_tbl   /* (B, max_pages) int32 */
) {
    int B         = q_u8.size(0);
    int max_pages = blk_tbl.size(1);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q_u8.device());
    auto out  = torch::full({B, TOPK_K}, -1, opts);

    dsa_topk_kernel<<<B, 256, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
        (const uint8_t*)q_u8.data_ptr(),
        (const int8_t*) k_cache.data_ptr(),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        blk_tbl.data_ptr<int>(),
        out.data_ptr<int>(),
        max_pages
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dsa_topk", &dsa_topk, "DSA top-k CUDA kernel");
}
"""

_mod = None


def _load():
    global _mod
    if _mod is None:
        tmpdir = tempfile.mkdtemp()
        src_path = os.path.join(tmpdir, "dsa_topk.cu")
        with open(src_path, "w") as f:
            f.write(CUDA_FULL_SRC)
        _mod = ext.load(
            name="dsa_topk_ext",
            sources=[src_path],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=True,
        )
    return _mod


def run(q_fp8, k_cache, weights, seq_lens, blk_tbl):
    """Run the new DSA top-k kernel.
    q_fp8: (B,64,128) float8_e4m3fn or uint8
    k_cache: (num_pages,64,1,132) int8
    weights: (B,64) float32
    seq_lens: (B,) int32
    blk_tbl: (B,max_pages) int32
    Returns: (B,2048) int32
    """
    mod = _load()
    q_u8 = q_fp8.view(torch.uint8) if q_fp8.dtype != torch.uint8 else q_fp8
    return mod.dsa_topk(
        q_u8.contiguous(), k_cache.contiguous(),
        weights.contiguous(), seq_lens.contiguous(),
        blk_tbl.contiguous(),
    )
