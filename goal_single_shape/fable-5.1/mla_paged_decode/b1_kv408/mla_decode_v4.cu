// MLA paged decode (page_size = 1), written from scratch.
//   q_nope [B,16,512] bf16, q_pe [B,16,64] bf16, ckv_cache [P,1,512] bf16, kpe_cache [P,1,64] bf16,
//   kv_indptr [B+1] i32, kv_indices [nnz] i32, sm_scale -> out [B,16,512] bf16, lse2 [B,16] f32.
//
// Design: one thread-block cluster of CS CTAs per batch row. The kv tokens of the row are split
// evenly across the CS CTAs (split-KV). Each CTA gathers its rows of the latent cache into shared
// memory with cp.async, computes S = Q K^T and O = P V with mma.sync (bf16 -> fp32), keeping an
// online-softmax state per head. Every CTA then *pushes* its partial (m, l, O) for head h into the
// shared memory of the CTA that owns head h (DSMEM stores); after one cluster barrier each owner
// merges its head from local shared memory and writes the bf16 output and the base-2 LSE.
// No global workspace, no atomics, one launch, one cluster barrier.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 16
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

constexpr int H = 16;         // query heads (== mma M)
constexpr int D_CKV = 512;
constexpr int D_KPE = 64;
constexpr int D_QK = D_CKV + D_KPE;   // 576
constexpr int KSTEPS = D_QK / 16;     // 36
constexpr int TILE_T = 32;    // kv tokens per chunk (mma N for QK, K for PV)
constexpr int NWARPS = NTHREADS / 32;
constexpr int HPC = H / CLUSTER_SIZE;         // heads finalised per CTA
constexpr int ROW_BYTES = D_QK * 2;          // 1152
constexpr int ROW_STRIDE = ROW_BYTES + 16;   // 1168: 73 x 16B  -> conflict-free ldmatrix
constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;   // 72
constexpr int P_STRIDE = TILE_T * 2 + 16;    // 80 B
static_assert(H % CLUSTER_SIZE == 0, "cluster size must divide the head count");
static_assert(NWARPS == 8, "kernel assumes 8 warps (2 heads per warp in softmax, 64 dims per warp in PV)");

// shared memory carve-up (bytes)
constexpr int SQ_OFF = 0;
constexpr int SQ_BYTES = H * ROW_STRIDE;                 // 18688
constexpr int SKV_OFF = SQ_OFF + SQ_BYTES;
constexpr int SKV_BYTES = TILE_T * ROW_STRIDE;           // 37376
constexpr int SS_OFF = SKV_OFF + SKV_BYTES;
constexpr int SS_BYTES = NWARPS * H * TILE_T * 4;        // 16384 (split-K partials of S, per warp)
constexpr int SP_OFF = SS_OFF + SS_BYTES;
constexpr int SP_BYTES = H * P_STRIDE;                   // 1280
#ifndef PARTIAL_FP32
#define PARTIAL_FP32 0
#endif
#if PARTIAL_FP32
typedef float2 opair_t;   // partial O exchanged as fp32
constexpr int SLOT_BYTES = D_CKV * 4 + 16;              // O[512] f32, m, l, pad, pad (2064 B)
#else
typedef __half2 opair_t;  // partial O exchanged as fp16 (2^-11 relative rounding, far below bf16 output rounding)
constexpr int SLOT_BYTES = D_CKV * 2 + 16;              // O[512] f16, m, l, pad, pad (1040 B)
#endif
constexpr int SLOT_ML_OFF = SLOT_BYTES - 16;             // byte offset of (m, l) inside a slot
constexpr int SIN_OFF = SP_OFF + SP_BYTES;               // receive buffer: [HPC][CLUSTER_SIZE] slots pushed by peers
constexpr int SIN_BYTES = HPC * CLUSTER_SIZE * SLOT_BYTES;   // 33024
constexpr int SBAR_OFF = SIN_OFF + SIN_BYTES;            // mbarrier (8 B) + alpha[16]
constexpr int SBAR_BYTES = 128;
constexpr int SMEM_TOTAL = SBAR_OFF + SBAR_BYTES;
// staging buffer for my own partials ([H][516] floats = 33024 B) aliases sKV+sS once the chunk loop is done
constexpr int SSTAGE_OFF = SKV_OFF;
static_assert(SKV_BYTES + SS_BYTES >= H * SLOT_BYTES, "staging alias does not fit");
static_assert(SIN_OFF % 16 == 0 && SSTAGE_OFF % 16 == 0 && SBAR_OFF % 8 == 0, "alignment");

#ifdef PROFILE
__device__ unsigned long long g_prof[16 * 16];
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define PROF(i) do { if (blockIdx.y == 0 && threadIdx.x == 0) g_prof[blockIdx.x * 16 + (i)] = gtimer(); } while (0)
#else
#define PROF(i) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16_16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "WAIT_%=:\n"
        " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra WAIT_%=;\n"
        "}\n" ::"r"(bar), "r"(parity));
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}
// smem (this CTA) -> smem (peer CTA), completion signalled on the peer's mbarrier
__device__ __forceinline__ void bulk_push(uint32_t dst_cluster, uint32_t src_cta, uint32_t bytes, uint32_t bar_cluster) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dst_cluster), "r"(src_cta), "r"(bytes), "r"(bar_cluster));
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::); }

__device__ __forceinline__ opair_t pack_o(float x, float y) {
#if PARTIAL_FP32
    return make_float2(x, y);
#else
    return __floats2half2_rn(x, y);
#endif
}
__device__ __forceinline__ float2 unpack_o(opair_t v) {
#if PARTIAL_FP32
    return v;
#else
    return __half22float2(v);
#endif
}
// reductions over the 16 lanes of a half-warp (lanes 0-15 and 16-31 reduce independently)
__device__ __forceinline__ float half_max(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float half_sum(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(NTHREADS, 1)
mla_decode_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                  const __nv_bfloat16* __restrict__ ckv_cache, const __nv_bfloat16* __restrict__ kpe_cache,
                  const int* __restrict__ kv_indptr, const int* __restrict__ kv_indices, float scale_log2,
                  __nv_bfloat16* __restrict__ out, float* __restrict__ lse) {
    extern __shared__ __align__(128) unsigned char smem[];
    PROF(0);
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();       // 0..CLUSTER_SIZE-1
    const int b = blockIdx.y;                          // batch row
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c4 = lane & 3;

    float* sS = reinterpret_cast<float*>(smem + SS_OFF);
    unsigned char* sIn = smem + SIN_OFF;                              // [HPC][CS] slots
    unsigned char* sStage = smem + SSTAGE_OFF;                        // [H] slots (alias)
    float* sAlpha = reinterpret_cast<float*>(smem + SBAR_OFF + 16);   // [16]
    const uint32_t bar = smem_u32(smem + SBAR_OFF);

    // mbarrier: 1 local arrival (expect_tx) + HPC*CS incoming bulk pushes worth of bytes
    if (tid == 0) { mbar_init(bar, 1); fence_mbar_init(); }
    cluster_arrive();   // peers may only push into my buffer after this barrier completes

    // ---- issue the Q gather first: it does not depend on the index chain ----------------------
    {
        const __nv_bfloat16* qn = q_nope + (size_t)b * H * D_CKV;
        const __nv_bfloat16* qp = q_pe + (size_t)b * H * D_KPE;
#pragma unroll
        for (int i = 0; i < (H * CHUNKS_PER_ROW + NTHREADS - 1) / NTHREADS; ++i) {
            const int c = tid + i * NTHREADS;
            if (c < H * CHUNKS_PER_ROW) {
                const int r = c / CHUNKS_PER_ROW, part = c - r * CHUNKS_PER_ROW;
                const void* src = part < 64 ? (const void*)(qn + r * D_CKV + part * 8)
                                            : (const void*)(qp + r * D_KPE + (part - 64) * 8);
                cp_async16(smem_u32(smem + SQ_OFF + r * ROW_STRIDE + part * 16), src);
            }
        }
    }
    const int kv_beg = __ldg(kv_indptr + b), kv_end = __ldg(kv_indptr + b + 1);
    PROF(1);
    const int kv_len = kv_end - kv_beg;
    const int per = (kv_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE;      // tokens per CTA
    const int my_beg = min(kv_len, rank * per);
    const int my_len = max(0, min(kv_len, my_beg + per) - my_beg);
    const int nchunks = max(1, (my_len + TILE_T - 1) / TILE_T);
    const int* my_idx = kv_indices + kv_beg + my_beg;

    // online-softmax state: lanes 0-15 of warp w own head 2w, lanes 16-31 own head 2w+1
    float m_run = -INFINITY, l_run = 0.f;
    float acc_o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc_o[j][i] = 0.f;

    for (int chunk = 0; chunk < nchunks; ++chunk) {
        const int t0 = chunk * TILE_T;
        const int ntok = min(TILE_T, my_len - t0);   // may be <= 0 only when my_len == 0
        if (chunk > 0) __syncthreads();              // previous chunk done with sKV / sP
        // ---- gather KV rows [ckv | kpe] into sKV: every thread fetches the indices it needs -----
        {
            constexpr int NCH = TILE_T * CHUNKS_PER_ROW / NTHREADS;   // 9
            int idx[NCH];
#pragma unroll
            for (int i = 0; i < NCH; ++i) {
                const int r = (tid + i * NTHREADS) / CHUNKS_PER_ROW;
                idx[i] = (r < ntok) ? __ldg(my_idx + t0 + r) : -1;
            }
            PROF(2);
#pragma unroll
            for (int i = 0; i < NCH; ++i) {
                const int c = tid + i * NTHREADS;
                const int r = c / CHUNKS_PER_ROW, part = c - r * CHUNKS_PER_ROW;
                const uint32_t dst = smem_u32(smem + SKV_OFF + r * ROW_STRIDE + part * 16);
                if (idx[i] >= 0) {
                    const void* src = part < 64 ? (const void*)(ckv_cache + (size_t)idx[i] * D_CKV + part * 8)
                                                : (const void*)(kpe_cache + (size_t)idx[i] * D_KPE + (part - 64) * 8);
                    cp_async16(dst, src);
                } else {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%1,%1,%1};\n" ::"r"(dst), "r"(0));
                }
            }
        }
        cp_async_commit();
        PROF(3);
        cp_async_wait_all();
        __syncthreads();
        PROF(4);

        // ---- S = Q K^T : warp w takes k-steps w, w+8, ... (split-K), all 4 n-tiles -------------
        {
            float acc[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;
            const uint32_t sQ = smem_u32(smem + SQ_OFF), sK = smem_u32(smem + SKV_OFF);
#pragma unroll
            for (int i = 0; i < (KSTEPS + NWARPS - 1) / NWARPS; ++i) {
                const int ks = warp + i * NWARPS;
                if (ks < KSTEPS) {
                    uint32_t a[4], b0[4], b1[4];
                    ldmatrix_x4(sQ + (lane & 15) * ROW_STRIDE + (ks * 16 + (lane >> 4) * 8) * 2, a[0], a[1], a[2], a[3]);
                    const int n = (lane & 7) + (lane >> 4) * 8;
                    const int k = ks * 16 + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(sK + n * ROW_STRIDE + k * 2, b0[0], b0[1], b0[2], b0[3]);
                    ldmatrix_x4(sK + (n + 16) * ROW_STRIDE + k * 2, b1[0], b1[1], b1[2], b1[3]);
                    mma_bf16_16816(acc[0], a, b0);
                    mma_bf16_16816(acc[1], a, b0 + 2);
                    mma_bf16_16816(acc[2], a, b1);
                    mma_bf16_16816(acc[3], a, b1 + 2);
                }
            }
            float* dst = sS + warp * (H * TILE_T);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                *reinterpret_cast<float2*>(dst + g * TILE_T + j * 8 + c4 * 2) = make_float2(acc[j][0], acc[j][1]);
                *reinterpret_cast<float2*>(dst + (g + 8) * TILE_T + j * 8 + c4 * 2) = make_float2(acc[j][2], acc[j][3]);
            }
        }
        __syncthreads();
        PROF(5);

        // ---- reduce split-K partials + online softmax ----------------------------------------
        // warp w: lanes 0-15 -> head 2w, lanes 16-31 -> head 2w+1; each lane holds 2 tokens.
        {
            const int h = 2 * warp + (lane >> 4);
            const int t = (lane & 15) * 2;
            float2 s = make_float2(0.f, 0.f);
#pragma unroll
            for (int p = 0; p < NWARPS; ++p) {
                const float2 v = *reinterpret_cast<const float2*>(sS + p * (H * TILE_T) + h * TILE_T + t);
                s.x += v.x; s.y += v.y;
            }
            s.x = (t < ntok) ? s.x * scale_log2 : -INFINITY;
            s.y = (t + 1 < ntok) ? s.y * scale_log2 : -INFINITY;
            const float m_new = fmaxf(m_run, half_max(fmaxf(s.x, s.y)));
            const float mu = (m_new == -INFINITY) ? 0.f : m_new;
            const float p0 = exp2f(s.x - mu), p1 = exp2f(s.y - mu);
            const float alpha = exp2f(m_run - mu);
            l_run = l_run * alpha + half_sum(p0 + p1);
            m_run = m_new;
            *reinterpret_cast<__nv_bfloat162*>(smem + SP_OFF + h * P_STRIDE + t * 2) = __floats2bfloat162_rn(p0, p1);
            if ((lane & 15) == 0) sAlpha[h] = alpha;
        }
        __syncthreads();
        PROF(6);

        // ---- O += P V : warp w owns dims [64w, 64w+64) = 8 n-tiles, 2 k-steps of 16 tokens -------
        {
            const float ar0 = sAlpha[g], ar1 = sAlpha[g + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc_o[j][0] *= ar0; acc_o[j][1] *= ar0; acc_o[j][2] *= ar1; acc_o[j][3] *= ar1;
            }
            const uint32_t sP = smem_u32(smem + SP_OFF), sV = smem_u32(smem + SKV_OFF);
#pragma unroll
            for (int ks = 0; ks < TILE_T / 16; ++ks) {
                uint32_t a[4];
                ldmatrix_x4(sP + (lane & 15) * P_STRIDE + (ks * 16 + (lane >> 4) * 8) * 2, a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {   // two n-tiles per ldmatrix.x4.trans
                    uint32_t bb[4];
                    const int t = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int d = warp * 64 + jj * 16 + (lane >> 4) * 8;
                    ldmatrix_x4_trans(sV + t * ROW_STRIDE + d * 2, bb[0], bb[1], bb[2], bb[3]);
                    mma_bf16_16816(acc_o[jj * 2], a, bb);
                    mma_bf16_16816(acc_o[jj * 2 + 1], a, bb + 2);
                }
            }
        }
    }
    PROF(7);

    // ---- stage my partials in local smem as [head][O 512 | m | l] rows ----------------------
    __syncthreads();   // everyone is done reading sKV / sS (staging aliases them)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = warp * 64 + j * 8 + c4 * 2;
            *reinterpret_cast<opair_t*>(sStage + g * SLOT_BYTES + d * sizeof(opair_t) / 2) = pack_o(acc_o[j][0], acc_o[j][1]);
            *reinterpret_cast<opair_t*>(sStage + (g + 8) * SLOT_BYTES + d * sizeof(opair_t) / 2) = pack_o(acc_o[j][2], acc_o[j][3]);
        }
        if ((lane & 15) == 0) {
            const int h = 2 * warp + (lane >> 4);
            *reinterpret_cast<float2*>(sStage + h * SLOT_BYTES + SLOT_ML_OFF) = make_float2(m_run, l_run);
        }
    }
    fence_proxy_async();   // make generic-proxy smem writes visible to the bulk-copy engine
    __syncthreads();
    cluster_wait();        // every peer's mbarrier is initialised
    PROF(8);
    if (tid == 0) mbar_expect_tx(bar, HPC * CLUSTER_SIZE * SLOT_BYTES);
    if (tid < H) {
        const int h = tid;
        const uint32_t owner = h % CLUSTER_SIZE;
        const uint32_t dst = mapa(smem_u32(smem + SIN_OFF + ((h / CLUSTER_SIZE) * CLUSTER_SIZE + rank) * SLOT_BYTES), owner);
        bulk_push(dst, smem_u32(smem + SSTAGE_OFF + h * SLOT_BYTES), SLOT_BYTES, mapa(bar, owner));
    }
    mbar_wait(bar, 0);     // all HPC*CS partial rows for my head(s) have landed
    cluster_arrive();      // (a) all my incoming copies done => after everyone arrives, all copies are done
    PROF(9);

    // ---- merge my head(s) from local shared memory ------------------------------------------
#pragma unroll
    for (int hl = 0; hl < HPC; ++hl) {
        const int h = hl * CLUSTER_SIZE + rank;
        const unsigned char* slots = sIn + hl * CLUSTER_SIZE * SLOT_BYTES;
        float wj[CLUSTER_SIZE], lj[CLUSTER_SIZE];
        float M = -INFINITY;
#pragma unroll
        for (int j = 0; j < CLUSTER_SIZE; ++j) {
            const float2 ml = *reinterpret_cast<const float2*>(slots + j * SLOT_BYTES + SLOT_ML_OFF);
            wj[j] = ml.x; lj[j] = ml.y; M = fmaxf(M, ml.x);
        }
        const float Mu = (M == -INFINITY) ? 0.f : M;
        float L = 0.f;
#pragma unroll
        for (int j = 0; j < CLUSTER_SIZE; ++j) { wj[j] = exp2f(wj[j] - Mu); L += wj[j] * lj[j]; }
        const float invL = (L > 0.f) ? 1.f / L : 0.f;
#pragma unroll
        for (int d = tid * 2; d < D_CKV; d += NTHREADS * 2) {
            float2 acc = make_float2(0.f, 0.f);
#pragma unroll
            for (int j = 0; j < CLUSTER_SIZE; ++j) {
                const float2 v = unpack_o(*reinterpret_cast<const opair_t*>(slots + j * SLOT_BYTES + d * sizeof(opair_t) / 2));
                acc.x += wj[j] * v.x;
                acc.y += wj[j] * v.y;
            }
            *reinterpret_cast<__nv_bfloat162*>(out + ((size_t)b * H + h) * D_CKV + d) =
                __floats2bfloat162_rn(acc.x * invL, acc.y * invL);
        }
        if (tid == 0) lse[(size_t)b * H + h] = (L > 0.f) ? (M + log2f(L)) : -INFINITY;
    }
    cluster_wait();        // (b) nobody exits while a peer could still be copying out of its smem
    PROF(10);
}

extern "C" int mla_decode_launch(const void* q_nope, const void* q_pe, const void* ckv_cache, const void* kpe_cache,
                                 const int* kv_indptr, const int* kv_indices, float sm_scale, void* out, float* lse,
                                 int batch, cudaStream_t stream) {
    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
        if (e != cudaSuccess) { fprintf(stderr, "smem attr: %s\n", cudaGetErrorString(e)); return 1; }
        e = cudaFuncSetAttribute(mla_decode_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        if (e != cudaSuccess) { fprintf(stderr, "cluster attr: %s\n", cudaGetErrorString(e)); return 1; }
        configured = true;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CLUSTER_SIZE, batch, 1);
    cfg.blockDim = dim3(NTHREADS, 1, 1);
    cfg.dynamicSmemBytes = SMEM_TOTAL;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER_SIZE;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const float scale_log2 = sm_scale * 1.4426950408889634f;
    cudaError_t e = cudaLaunchKernelEx(&cfg, mla_decode_kernel, (const __nv_bfloat16*)q_nope, (const __nv_bfloat16*)q_pe,
                                       (const __nv_bfloat16*)ckv_cache, (const __nv_bfloat16*)kpe_cache, kv_indptr,
                                       kv_indices, scale_log2, (__nv_bfloat16*)out, lse);
    if (e != cudaSuccess) { fprintf(stderr, "launch: %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}

extern "C" int mla_decode_smem_bytes() { return SMEM_TOTAL; }
extern "C" int mla_decode_prof(unsigned long long* host) {
#ifdef PROFILE
    return cudaMemcpyFromSymbol(host, g_prof, sizeof(unsigned long long) * 256) == cudaSuccess ? 0 : 1;
#else
    return 2;
#endif
}
