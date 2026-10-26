// Routing + grouping kernels for large T (lists live in global memory).
#pragma once
#include "moe_common.cuh"

namespace moe {

// Per call, in global memory (self-cleaning: zeroed by the main kernel's last CTA):
//   rcount[32]  : local pairs per expert
//   rfill[32]   : scatter fill counters
//   rstart[32]  : exclusive prefix of rcount (written by the scatter kernel, CTA 0)
//   npairs[T]   : number of local pairs per token

// one warp per token: top-8 ids/weights, local counts
__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ logits, const bf16_t* __restrict__ bias,
                                                    int T, float scaling, int local_offset,
                                                    uint8_t* __restrict__ ids, float* __restrict__ w,
                                                    int* __restrict__ rcount, int* __restrict__ npairs) {
    __shared__ float sbias[E_GLOBAL];
    __shared__ int scount[E_LOCAL];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < E_GLOBAL; i += 256) sbias[i] = __bfloat162float(bias[i]);
    if (tid < E_LOCAL) scount[tid] = 0;
    __syncthreads();
    const int t = blockIdx.x * 8 + warp;
    if (t < T) {
        int id; float wt;
        route_token_warp(logits + (size_t)t * E_GLOBAL, sbias, scaling, id, wt);
        int loc = 0;
        if (lane < TOP_K) {
            ids[t * TOP_K + lane] = (uint8_t)id;
            w[t * TOP_K + lane] = wt;
            const int le = id - local_offset;
            loc = (le >= 0 && le < E_LOCAL) ? 1 : 0;
            if (loc) atomicAdd(&scount[le], 1);
        }
        loc = __reduce_add_sync(0xffffffffu, loc);
        if (lane == 0) npairs[t] = loc;
    }
    __syncthreads();
    if (tid < E_LOCAL && scount[tid]) atomicAdd(&rcount[tid], scount[tid]);
}

// scatter pairs grouped by expert: tok_list[pos], w_list[pos]; pos = start[le] + atomic fill
__global__ void __launch_bounds__(256) scatter_kernel(const uint8_t* __restrict__ ids, const float* __restrict__ w, int T, int local_offset,
                                                      const int* __restrict__ rcount, int* __restrict__ rfill, int* __restrict__ rstart,
                                                      int* __restrict__ tok_list, float* __restrict__ w_list, int* __restrict__ tok_pairs) {
    __shared__ int sstart[E_LOCAL];
    const int tid = threadIdx.x;
    if (tid < 32) {
        int c = rcount[tid];
        int incl = c;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int o = __shfl_up_sync(0xffffffffu, incl, off);
            if (tid >= off) incl += o;
        }
        sstart[tid] = incl - c;
        if (blockIdx.x == 0) rstart[tid] = incl - c;
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + tid;
    if (i < T * TOP_K) {
        const int le = (int)ids[i] - local_offset;
        int pos = -1;
        if (le >= 0 && le < E_LOCAL) {
            pos = sstart[le] + atomicAdd(&rfill[le], 1);
            tok_list[pos] = i / TOP_K;
            w_list[pos] = w[i];
        }
        tok_pairs[i] = pos;
    }
}

}  // namespace moe
