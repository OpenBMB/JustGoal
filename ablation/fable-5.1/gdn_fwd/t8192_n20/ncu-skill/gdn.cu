// GDN prefill (gated delta rule, chunked) for B200. Written from scratch.
//   q,k: [T,4,128] bf16   v: [T,8,128] bf16   a,b: [T,8] bf16   A_log,dt_bias: [8] f32
//   state/new_state: [N,8,128(v),128(k)] f32 (k-last)   output: [T,8,128] bf16
// Algorithm (chunk C=64, per v-head h, k-head h/2):
//   lg_i = cumsum(log g), G_i = exp(lg_i), beta_i = sigmoid(b_i)
//   A[i][j] = beta_i (k_i.k_j) G_i/G_j (j<i);  Tm = (I+A)^-1
//   w = Tm (beta G k);  u = Tm (beta v);  Aqk[i][j] = (q_i.k_j) G_i/G_j (j<=i)
//   qt = scale (G q - Aqk w);  ob = scale Aqk u;  kp = k G_C/G
//   per chunk (sequential):  ut = u - w S;  o = qt S + ob;  S = G_C S + kp^T ut
// Kernel 1 (prepass): per (chunk, v-head) CTA -> tiles Wn=-w, Qg=scale*G*q, Kp (bf16 [64][128] swizzled), U (bf16 [4][64][32] swizzled), Aq=scale*Aqk (bf16 [64][64] swizzled), gC.
// Kernel 2 (chain):   per (seq, v-head, 32-wide v-slice) CTA, S^T slice kept in registers (fp32), mma.sync bf16.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

namespace {

constexpr int C = 64, D = 128, HV = 8, HQ = 4, VS = 32, NVS = D / VS;
constexpr int TILE_BYTES = C * D * 2;    // 16 KB
constexpr int SLICE_BYTES = C * VS * 2;  // 4 KB
constexpr int AQ_BYTES = C * C * 2;      // 8 KB
// per chunk-head workspace block: [Wn 16K][Qg 16K][Kp 16K][Aq 8K][gC + pad 16 B][U: 4 slices x 4 KB]
constexpr int BLK_WN = 0, BLK_QG = TILE_BYTES, BLK_KP = 2 * TILE_BYTES, BLK_AQ = 3 * TILE_BYTES, BLK_GC = 3 * TILE_BYTES + AQ_BYTES,
              BLK_U = BLK_GC + 64, BLK_HEAD = BLK_U /* bytes shared by the 4 v-slice CTAs (multicast) */,
              BLK_BYTES = BLK_U + NVS * SLICE_BYTES;
static_assert(BLK_HEAD % 64 == 0, "BLK_HEAD must split into 4 x 16B-aligned quarters");
constexpr int NTHREADS = 128;

struct Params {
    const __nv_bfloat16 *q, *k, *v, *a, *b;
    const float *state, *A_log, *dt_bias;
    const int64_t* cu;
    int N, T;
    float scale;
    __nv_bfloat16* out;
    float* new_state;
    uint8_t* blk;  // workspace blocks [maxChunks*HV][BLK_BYTES]
    int* plan;  // [N][4]: seq, cbase, L, pad (ordered by length desc)
    int nFine;  // number of longest sequences run with 16-wide v slices
    long long* dbg;  // timing debug (CHAIN_TIMING)
    long long* dbg2; // per-CTA timeline (CHAIN_TIMING)
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// byte offset of element (row, col) in a swizzled bf16 tile with COLS columns
template <int COLS>
__device__ __forceinline__ int swz(int row, int col) {
    if constexpr (COLS == 32) return row * 64 + ((((col >> 3) ^ ((row >> 1) & 3))) << 4) + ((col & 7) << 1);
    else return row * (COLS * 2) + ((((col >> 3) ^ (row & 7))) << 4) + ((col & 7) << 1);
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
    __nv_bfloat162 t = *reinterpret_cast<__nv_bfloat162*>(&u);
    return __bfloat1622float2(t);
}
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }


#ifdef CHAIN_TIMING
#define TS(i) do { if (blockIdx.x == 0 && tid == 0) p.dbg[(size_t)c * 16 + (i)] = clock64(); } while (0)
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int smid() { int r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
#define CTA_TS(i) do { if (threadIdx.x == 0) { p.dbg2[(size_t)blockIdx.x * 8 + (i)] = (i) == 0 ? smid() : gtimer(); p.dbg2[(size_t)blockIdx.x * 8 + 4 + (i)] = clock64(); } } while (0)
#else
#define TS(i) do {} while (0)
#define CTA_TS(i) do {} while (0)
#endif
// ---- mbarrier + TMA 1D bulk copy helpers ----
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes));
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0, 1, 0, P1;\n"
        "}\n" : "=r"(ok) : "r"(bar), "r"(parity));
    return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    if (mbar_test(bar, parity)) return;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(bar), "r"(parity));
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa_cluster(uint32_t addr, uint32_t cta) {
    uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta)); return r;
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t cluster_addr) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t mask) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// ---- ldmatrix address helpers (lane-dependent) ----
// A operand (m16 x k16) from row-major [M][K] tile
template <int COLS> __device__ __forceinline__ uint32_t addrA(uint32_t base, int m0, int k0, int lane) {
    return base + swz<COLS>(m0 + (lane & 15), k0 + ((lane >> 4) << 3));
}
// B operand from [N][K] row-major tile (non-trans): regs (0,1)->n-tile n0, (2,3)->n-tile n0+8
template <int COLS> __device__ __forceinline__ uint32_t addrBn(uint32_t base, int n0, int k0, int lane) {
    return base + swz<COLS>(n0 + (lane & 7) + ((lane >> 4) << 3), k0 + (((lane >> 3) & 1) << 3));
}
// B operand from [K][N] row-major tile (trans): regs (0,1)->n-tile n0, (2,3)->n-tile n0+8
template <int COLS> __device__ __forceinline__ uint32_t addrBt(uint32_t base, int k0, int n0, int lane) {
    return base + swz<COLS>(k0 + (lane & 7) + (((lane >> 3) & 1) << 3), n0 + ((lane >> 4) << 3));
}

__device__ __forceinline__ int nchunks_of(const int64_t* cu, int s) {
    long long L = cu[s + 1] - cu[s];
    return L > 0 ? (int)((L + C - 1) / C) : 0;
}

// Block-parallel: find (seq, chunk_in_seq) for global chunk id cid. Returns false if cid >= total.
__device__ bool locate_chunk(const int64_t* cu, int N, int cid, int& seq, int& cis, int* scratch /*>=136 ints*/) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int per = (N + NTHREADS - 1) / NTHREADS;
    const int s0 = tid * per, s1 = min(N, s0 + per);
    int cnt = 0;
    for (int s = s0; s < s1; ++s) cnt += nchunks_of(cu, s);
    // exclusive block scan: warp shuffle scan + warp totals
    int incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += t; }
    if (lane == 31) scratch[warp] = incl;
    if (tid == 0) scratch[NTHREADS] = -1;
    __syncthreads();
    int base = incl - cnt;
#pragma unroll
    for (int w = 0; w < 4; ++w) base += (w < warp) ? scratch[w] : 0;
    for (int s = s0; s < s1; ++s) {
        int nc = nchunks_of(cu, s);
        if (cid >= base && cid < base + nc) { scratch[NTHREADS] = s; scratch[NTHREADS + 1] = cid - base; }
        base += nc;
    }
    __syncthreads();
    seq = scratch[NTHREADS]; cis = scratch[NTHREADS + 1];
    return seq >= 0;
}

// ============================================================================================
// Kernel 1: prepass. grid = maxChunks * HV, block = 128.
// ============================================================================================
// prepass smem: Qs is dead after the QK product and Qg conversion, so Ah/Th alias it afterwards.
constexpr int PP_QS = 0, PP_KS = 16384, PP_VS = 32768, PP_AS = 49152 /*fp32 A, later Ws*/,
              PP_AH = 0, PP_TH = 8192, PP_R = 65536 /*2 KB R staging*/, PP_G = 65536 + 2048 /* lg[64], beta[64], Gi[64] */,
              PP_SCR = PP_G + 768, PP_SMEM = PP_SCR + 136 * 4;

__device__ void build_plan(const Params& p, uint8_t* smem) {
    const int tid = threadIdx.x, N = p.N;
    int64_t* cus = reinterpret_cast<int64_t*>(smem);
    const bool fits = (size_t)(N + 1) * 8 <= (size_t)PP_G;
    const int64_t* cu = fits ? cus : p.cu;
    if (fits) {
        for (int i = tid; i <= N; i += NTHREADS) cus[i] = p.cu[i];
        __syncthreads();
    }
    const bool do_rank = N <= 4096;
    for (int s = tid; s < N; s += NTHREADS) {
        long long Ls = cu[s + 1] - cu[s];
        int rank = 0, cbase = 0;
        if (do_rank) {
            for (int r = 0; r < N; ++r) {
                long long Lr = cu[r + 1] - cu[r];
                rank += (Lr > Ls) || (Lr == Ls && r < s);
                cbase += (r < s && Lr > 0) ? (int)((Lr + C - 1) / C) : 0;
            }
        } else {
            rank = s;
            for (int r = 0; r < s; ++r) { long long Lr = cu[r + 1] - cu[r]; cbase += Lr > 0 ? (int)((Lr + C - 1) / C) : 0; }
        }
        p.plan[rank * 4 + 0] = s; p.plan[rank * 4 + 1] = cbase; p.plan[rank * 4 + 2] = (int)Ls; p.plan[rank * 4 + 3] = 0;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(NTHREADS, 3) prepass_kernel(Params p) {
    extern __shared__ __align__(1024) uint8_t smem[];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // let the chain grid start scheduling early (PDL)
    if (blockIdx.x == 0) build_plan(p, smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int cid = blockIdx.x >> 3, h = blockIdx.x & 7, kh = h >> 1;
    int* scratch = reinterpret_cast<int*>(smem + PP_SCR);
    int seq, cis;
    if (!locate_chunk(p.cu, p.N, cid, seq, cis, scratch)) return;
    const long long t0 = p.cu[seq] + (long long)cis * C;
    const int L = (int)min((long long)C, p.cu[seq + 1] - t0);

    float* lg = reinterpret_cast<float*>(smem + PP_G);
    float* beta = lg + 64;
    float* Gi = lg + 128;
    uint8_t* blk = p.blk + (size_t)(cid * HV + h) * BLK_BYTES;
    const uint32_t sQ = smem_u32(smem + PP_QS), sK = smem_u32(smem + PP_KS), sV = smem_u32(smem + PP_VS);
    const uint32_t sAh = smem_u32(smem + PP_AH), sTh = smem_u32(smem + PP_TH);
    float* As = reinterpret_cast<float*>(smem + PP_AS);

    // gate inputs for this thread's token (prefetched before the tile loads)
    float av = 0.f, bv = 0.f;
    if (tid < L) {
        av = __bfloat162float(p.a[(t0 + tid) * HV + h]);
        bv = __bfloat162float(p.b[(t0 + tid) * HV + h]);
    }
    // ---- async loads of q, k (k-head), v (v-head) tiles; zero-fill rows >= L ----
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int idx = tid + j * NTHREADS, row = idx >> 4, ch = idx & 15;
        bool valid = row < L;
        long long t = t0 + row;
        int soff = swz<D>(row, ch * 8);
        cp_async16(sQ + soff, p.q + (t * HQ + kh) * D + ch * 8, valid);
        cp_async16(sK + soff, p.k + (t * HQ + kh) * D + ch * 8, valid);
        cp_async16(sV + soff, p.v + (t * HV + h) * D + ch * 8, valid);
    }
    cp_async_commit();

    // ---- gates (threads 0..63) ----
    if (tid < 64) {
        bool valid = tid < L;
        float lgi = 0.f, be = 0.f;
        if (valid) {
            float x = av + p.dt_bias[h];
            float sp = x > 20.f ? x : log1pf(__expf(x));
            lgi = -__expf(p.A_log[h]) * sp;
            be = 1.f / (1.f + __expf(-bv));
        }
        // inclusive scan over 64 threads (2 warps)
        float v = lgi;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float t = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += t;
        }
        lg[tid] = v;  // partial (per-warp inclusive)
        beta[tid] = be;
    }
    __syncthreads();
    if (tid >= 32 && tid < 64) lg[tid] += lg[31];
    __syncthreads();
    if (tid < 64) Gi[tid] = __expf(lg[tid]);
    if (tid == 0) *reinterpret_cast<float*>(blk + BLK_GC) = __expf(lg[63]);
    cp_async_wait<0>();
    __syncthreads();
    // ---- Qg = scale * G * q -> global (all threads; Qs stays intact for the QK product) ----
    {
        __nv_bfloat16* Qg = reinterpret_cast<__nv_bfloat16*>(blk + BLK_QG);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = tid + j * NTHREADS, row = idx >> 4;
            const float fqg = p.scale * Gi[row];
            uint4 qraw = *reinterpret_cast<uint4*>(smem + PP_QS + idx * 16);
            uint32_t* qw = reinterpret_cast<uint32_t*>(&qraw);
            uint4 qgo; uint32_t* qgw = reinterpret_cast<uint32_t*>(&qgo);
#pragma unroll
            for (int e = 0; e < 4; ++e) { float2 qf = unpack2(qw[e]); qgw[e] = pack2(qf.x * fqg, qf.y * fqg); }
            *reinterpret_cast<uint4*>(Qg + idx * 8) = qgo;
        }
    }

    // ---- KK = k k^T, QK = q k^T (rows 16*warp.., n-tiles 0..2*warp+1) ----
    float KK[8][4], QK[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) { KK[t][i] = 0.f; QK[t][i] = 0.f; }
    const int m0 = warp * 16;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        uint32_t ak[4], aq[4], b[4];
        ldsm4(ak, addrA<D>(sK, m0, kk * 16, lane));
        ldsm4(aq, addrA<D>(sQ, m0, kk * 16, lane));
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            if (t <= warp) {
                ldsm4(b, addrBn<D>(sK, t * 16, kk * 16, lane));
                mma16816(KK[2 * t], ak, b[0], b[1]);
                mma16816(KK[2 * t + 1], ak, b[2], b[3]);
                mma16816(QK[2 * t], aq, b[0], b[1]);
                mma16816(QK[2 * t + 1], aq, b[2], b[3]);
            }
        }
    }
    __syncthreads();  // all warps done reading Qs/Ks (QK/KK) -> Ah/Th may overwrite Qs, X1 may overwrite Ks
    // ---- rhs prep (all threads): Kp = k * G_C/G_i -> global; X1 = beta*G*k (in place in Ks); X2 = beta*v (in Vs) ----
    {
        const float lgC = lg[63];
        __nv_bfloat16* Kp = reinterpret_cast<__nv_bfloat16*>(blk + BLK_KP);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = tid + j * NTHREADS, row = idx >> 4;
            const float fkp = __expf(lgC - lg[row]), fx1 = beta[row] * Gi[row], fx2 = beta[row];
            uint4 kraw = *reinterpret_cast<uint4*>(smem + PP_KS + idx * 16);
            uint4 vraw = *reinterpret_cast<uint4*>(smem + PP_VS + idx * 16);
            uint32_t* kw = reinterpret_cast<uint32_t*>(&kraw);
            uint32_t* vw = reinterpret_cast<uint32_t*>(&vraw);
            uint4 kpo, x1o, x2o;
            uint32_t* kpw = reinterpret_cast<uint32_t*>(&kpo);
            uint32_t* x1w = reinterpret_cast<uint32_t*>(&x1o);
            uint32_t* x2w = reinterpret_cast<uint32_t*>(&x2o);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float2 kf = unpack2(kw[e]), vf = unpack2(vw[e]);
                kpw[e] = pack2(kf.x * fkp, kf.y * fkp);
                x1w[e] = pack2(kf.x * fx1, kf.y * fx1);
                x2w[e] = pack2(vf.x * fx2, vf.y * fx2);
            }
            *reinterpret_cast<uint4*>(Kp + idx * 8) = kpo;
            *reinterpret_cast<uint4*>(smem + PP_KS + idx * 16) = x1o;
            *reinterpret_cast<uint4*>(smem + PP_VS + idx * 16) = x2o;
        }
    }
    // zero Th (aliases Qs)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int idx = tid + j * NTHREADS;  // 512 x 16B
        *reinterpret_cast<uint4*>(smem + PP_TH + idx * 16) = make_uint4(0, 0, 0, 0);
    }
    // ---- A (fp32 + bf16), Aqk (bf16, scaled) -> global ----
    __nv_bfloat16* Aq = reinterpret_cast<__nv_bfloat16*>(blk + BLK_AQ);
    {
        const int i0 = m0 + g, i1 = m0 + g + 8;
        const float lg0 = lg[i0], lg1 = lg[i1], be0 = beta[i0], be1 = beta[i1];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int j = t * 8 + tig * 2;
            float a00 = 0, a01 = 0, a10 = 0, a11 = 0, q00 = 0, q01 = 0, q10 = 0, q11 = 0;
            if (t <= 2 * warp + 1) {
                const float lj0 = lg[j], lj1 = lg[j + 1];
                float e00 = __expf(lg0 - lj0), e01 = __expf(lg0 - lj1), e10 = __expf(lg1 - lj0), e11 = __expf(lg1 - lj1);
                a00 = (j < i0) ? be0 * e00 * KK[t][0] : 0.f;
                a01 = (j + 1 < i0) ? be0 * e01 * KK[t][1] : 0.f;
                a10 = (j < i1) ? be1 * e10 * KK[t][2] : 0.f;
                a11 = (j + 1 < i1) ? be1 * e11 * KK[t][3] : 0.f;
                q00 = (j <= i0) ? e00 * QK[t][0] : 0.f;
                q01 = (j + 1 <= i0) ? e01 * QK[t][1] : 0.f;
                q10 = (j <= i1) ? e10 * QK[t][2] : 0.f;
                q11 = (j + 1 <= i1) ? e11 * QK[t][3] : 0.f;
                As[i0 * 64 + j] = a00; As[i0 * 64 + j + 1] = a01; As[i1 * 64 + j] = a10; As[i1 * 64 + j + 1] = a11;
                *reinterpret_cast<uint32_t*>(smem + PP_AH + swz<64>(i0, j)) = pack2(a00, a01);
                *reinterpret_cast<uint32_t*>(smem + PP_AH + swz<64>(i1, j)) = pack2(a10, a11);
            }
            *reinterpret_cast<uint32_t*>(Aq + swz<64>(i0, j) / 2) = pack2(p.scale * q00, p.scale * q01);
            *reinterpret_cast<uint32_t*>(Aq + swz<64>(i1, j) / 2) = pack2(p.scale * q10, p.scale * q11);
        }
    }
    __syncthreads();

    // ---- diagonal 16x16 block inverses (fp32, serial), warp w -> block w ----
    {
        const int c = lane & 15, b0 = warp * 16;
        float Tc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float val = (i == c) ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < i; ++j)
                if (j >= c) val -= As[(b0 + i) * 64 + b0 + j] * Tc[j];
            Tc[i] = val;
        }
        if (lane < 16) {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                *reinterpret_cast<__nv_bfloat16*>(smem + PP_TH + swz<64>(b0 + i, b0 + c)) = __float2bfloat16(Tc[i]);
        }
    }
    __syncthreads();

    // ---- level 1 merges (warps 0,1): T10 = -T11 A10 T00 for pair p (bf16 operands, fp32 accumulate) ----
    if (warp < 2) {
        const int pr = warp, r0 = 16 * (2 * pr + 1), c0 = 16 * (2 * pr);
        float acc[2][4] = {};
        {
            uint32_t xh[4], yh[4];
            ldsm4(xh, addrA<64>(sTh, r0, r0, lane));
            ldsm4t(yh, addrBt<64>(sAh, r0, c0, lane));
#pragma unroll
            for (int t = 0; t < 2; ++t) mma16816(acc[t], xh, yh[2 * t], yh[2 * t + 1]);
        }
        float acc2[2][4] = {};
        {
            uint32_t ah[4], yh[4];
            ah[0] = pack2(acc[0][0], acc[0][1]); ah[1] = pack2(acc[0][2], acc[0][3]); ah[2] = pack2(acc[1][0], acc[1][1]); ah[3] = pack2(acc[1][2], acc[1][3]);
            ldsm4t(yh, addrBt<64>(sTh, c0, c0, lane));
#pragma unroll
            for (int t = 0; t < 2; ++t) mma16816(acc2[t], ah, yh[2 * t], yh[2 * t + 1]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int j = c0 + t * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(smem + PP_TH + swz<64>(r0 + g, j)) = pack2(-acc2[t][0], -acc2[t][1]);
            *reinterpret_cast<uint32_t*>(smem + PP_TH + swz<64>(r0 + g + 8, j)) = pack2(-acc2[t][2], -acc2[t][3]);
        }
    }
    __syncthreads();

    // ---- level 2 merge (all 4 warps): R = A_BL T_A (bf16, staged in PP_R), then T[32..63][0..31] = -T_B R ----
    {
        const int mt = warp & 1, nh = warp >> 1;   // m-tile (rows 32+16mt..), n-half (cols 16nh..)
        const int r0 = 32 + 16 * mt;
        float acc[2][4] = {};
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t xh[4], yh[4];
            ldsm4(xh, addrA<64>(sAh, r0, kk * 16, lane));           // A_BL rows r0.., cols kk*16..
            ldsm4t(yh, addrBt<64>(sTh, kk * 16, nh * 16, lane));    // T_A rows kk*16.., cols nh*16..
#pragma unroll
            for (int t = 0; t < 2; ++t) mma16816(acc[t], xh, yh[2 * t], yh[2 * t + 1]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int j = nh * 16 + t * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(smem + PP_R + swz<32>(16 * mt + g, j)) = pack2(acc[t][0], acc[t][1]);
            *reinterpret_cast<uint32_t*>(smem + PP_R + swz<32>(16 * mt + g + 8, j)) = pack2(acc[t][2], acc[t][3]);
        }
        __syncthreads();
        float acc2[2][4] = {};
        const uint32_t sR = smem_u32(smem + PP_R);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t xh[4], yh[4];
            ldsm4(xh, addrA<64>(sTh, r0, 32 + kk * 16, lane));       // T_B rows r0.., cols 32+kk*16..
            ldsm4t(yh, addrBt<32>(sR, kk * 16, nh * 16, lane));      // R rows kk*16.., cols nh*16..
#pragma unroll
            for (int t = 0; t < 2; ++t) mma16816(acc2[t], xh, yh[2 * t], yh[2 * t + 1]);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int j = nh * 16 + t * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(smem + PP_TH + swz<64>(r0 + g, j)) = pack2(-acc2[t][0], -acc2[t][1]);
            *reinterpret_cast<uint32_t*>(smem + PP_TH + swz<64>(r0 + g + 8, j)) = pack2(-acc2[t][2], -acc2[t][3]);
        }
    }
    __syncthreads();

    // ---- w = T X1 (X1 in Ks) -> Ws = -w (smem, aliases As) ; u = T X2 (X2 in Vs) -> Vs in place ----
    __nv_bfloat16* Wn = reinterpret_cast<__nv_bfloat16*>(blk + BLK_WN);
    __nv_bfloat16* U = reinterpret_cast<__nv_bfloat16*>(blk + BLK_U);
    {
        float acc[16][4] = {};
        {
            uint32_t xh[4], y[4], yn[4];
            ldsm4(xh, addrA<64>(sTh, m0, 0, lane));
            ldsm4t(y, addrBt<D>(sK, 0, 0, lane));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int t2 = 0; t2 < 8; ++t2) {
                    const bool more = (kk < 3) || (t2 < 7);
                    if (more) ldsm4t(yn, addrBt<D>(sK, (t2 < 7 ? kk : kk + 1) * 16, (t2 < 7 ? t2 + 1 : 0) * 16, lane));
                    mma16816(acc[2 * t2], xh, y[0], y[1]);
                    mma16816(acc[2 * t2 + 1], xh, y[2], y[3]);
                    if (more) { y[0] = yn[0]; y[1] = yn[1]; y[2] = yn[2]; y[3] = yn[3]; }
                }
                if (kk < 3) ldsm4(xh, addrA<64>(sTh, m0, (kk + 1) * 16, lane));
            }
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int j = t * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(smem + PP_AS + swz<D>(m0 + g, j)) = pack2(-acc[t][0], -acc[t][1]);
            *reinterpret_cast<uint32_t*>(smem + PP_AS + swz<D>(m0 + g + 8, j)) = pack2(-acc[t][2], -acc[t][3]);
        }
    }
    {
        float acc[16][4] = {};
        {
            uint32_t xh[4], y[4], yn[4];
            ldsm4(xh, addrA<64>(sTh, m0, 0, lane));
            ldsm4t(y, addrBt<D>(sV, 0, 0, lane));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int t2 = 0; t2 < 8; ++t2) {
                    const bool more = (kk < 3) || (t2 < 7);
                    if (more) ldsm4t(yn, addrBt<D>(sV, (t2 < 7 ? kk : kk + 1) * 16, (t2 < 7 ? t2 + 1 : 0) * 16, lane));
                    mma16816(acc[2 * t2], xh, y[0], y[1]);
                    mma16816(acc[2 * t2 + 1], xh, y[2], y[3]);
                    if (more) { y[0] = yn[0]; y[1] = yn[1]; y[2] = yn[2]; y[3] = yn[3]; }
                }
                if (kk < 3) ldsm4(xh, addrA<64>(sTh, m0, (kk + 1) * 16, lane));
            }
        }
        __syncthreads();  // all warps done reading Vs (X2) and Ws written
        // copy Ws -> global Wn (linear, already swizzled)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int idx = tid + j * NTHREADS;
            *reinterpret_cast<uint4*>(Wn + idx * 8) = *reinterpret_cast<uint4*>(smem + PP_AS + idx * 16);
        }
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int j = t * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(smem + PP_VS + swz<D>(m0 + g, j)) = pack2(acc[t][0], acc[t][1]);
            *reinterpret_cast<uint32_t*>(smem + PP_VS + swz<D>(m0 + g + 8, j)) = pack2(acc[t][2], acc[t][3]);
        }
    }
    __syncthreads();
    // copy Us (in Vs) -> global U as [vslice][64][32] (COLS=32 swizzle)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int idx = tid + j * NTHREADS;
        int row = idx >> 4, pch = idx & 15, lch = pch ^ (row & 7);  // logical 16B chunk (of 16 per row)
        int vs = lch >> 2, cj = lch & 3;
        uint4 val = *reinterpret_cast<uint4*>(smem + PP_VS + idx * 16);
        *reinterpret_cast<uint4*>(U + vs * (C * VS) + row * VS + ((cj ^ ((row >> 1) & 3)) << 3)) = val;
    }
}

// ============================================================================================
// Kernel 2: chain. grid = N * HV * NVS, block = 160 (4 compute warps + TMA producer warp).
// CTA = (seq by length rank, head, v-slice of 32). Compute warp w = cq: all 32 v rows; it owns k-quarter cq of the
// state and computes c-quarter cq of ut / o. Register "slots" are rotated so that slot j holds quarter (cq + j) & 3:
// every register index is static and all warps run the same code (one instantiation -> small i-cache footprint).
// ============================================================================================
constexpr int CH_THREADS = 160;
#ifndef NFINE
#define NFINE 1
#endif
constexpr int CH_STAGES = 3;
constexpr int CH_STAGE_BYTES = BLK_HEAD + SLICE_BYTES;
constexpr int CH_XU = CH_STAGES * CH_STAGE_BYTES;  // 4 warps x 8 regs x 32 lanes x 4 B = 4 KB
constexpr int CH_XS = CH_XU + 4 * 1024;            // 4 warps x 16 regs x 32 lanes x 4 B = 8 KB
constexpr int CH_BAR = CH_XS + 8 * 1024;           // 2*CH_STAGES mbarriers (full, empty; 8 B each)
constexpr int CH_SMEM = CH_BAR + 64;

__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t count) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}

// Producer (one thread): TMA bulk copies of a chunk's block into a stage; completion tracked by the stage's mbarrier.
__device__ __forceinline__ void chain_issue_loads(const Params& p, uint32_t stage, uint32_t bar, int chid, int vs) {
    const uint8_t* blk = p.blk + (size_t)chid * BLK_BYTES;
    mbar_expect_tx(bar, CH_STAGE_BYTES);
    bulk_g2s(stage, blk, BLK_HEAD, bar);
    bulk_g2s(stage + BLK_HEAD, blk + BLK_U + vs * SLICE_BYTES, SLICE_BYTES, bar);
}

__device__ __forceinline__ void chain_producer(const Params& p, uint8_t* smem, int h, int vs, int cbase, int nch) {
    const uint32_t sbase = smem_u32(smem);
    const uint32_t full0 = sbase + CH_BAR, empty0 = full0 + 8 * CH_STAGES;
    if ((threadIdx.x & 31) != 0) return;
    for (int c = 0; c < nch; ++c) {
        const int stg = c % CH_STAGES;
        if (c >= CH_STAGES) mbar_wait(empty0 + 8 * stg, ((c / CH_STAGES) - 1) & 1);
        chain_issue_loads(p, sbase + stg * CH_STAGE_BYTES, full0 + 8 * stg, (cbase + c) * HV + h, vs);
    }
}

template <int VG>
__device__ __forceinline__ void chain_loop(const Params& p, uint8_t* smem, int h, int vs, int cq, int cbase, int nch,
                                           long long t0, int L, float (&Sacc)[2][4][4], uint32_t (&Sb)[2][16][2]) {
    const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int v0 = vs * 16 * VG;                    // first v row of this CTA within the head
    const int voff = (VG == 1) ? (vs & 1) * 16 : 0;  // offset inside the 32-wide U slice
    const uint32_t sbase = smem_u32(smem);
    uint32_t* xU = reinterpret_cast<uint32_t*>(smem + CH_XU);
    uint32_t* xS = reinterpret_cast<uint32_t*>(smem + CH_XS);
    const uint32_t full0 = sbase + CH_BAR, empty0 = full0 + 8 * CH_STAGES;
    const int qn[4] = {cq, (cq + 1) & 3, (cq + 2) & 3, (cq + 3) & 3};  // slot -> quarter
    // lane-dependent address parts (hoisted)
    const int ldA_row = cq * 16 + (lane & 15), ldA_chunk = (lane >> 4);                 // A frag: rows cq*16.., k chunk +0/1
    const int ldBn_row = cq * 16 + (lane & 7) + ((lane >> 4) << 3), ldBn_chunk = ((lane >> 3) & 1);
    const int ldBt_rowoff = (lane & 7) + (((lane >> 3) & 1) << 3), ldBt_col0 = cq * 32 + ((lane >> 4) << 3);
    const int mi = lane >> 3;

    mbar_wait(full0, 0);
    bool ready_next = false;
    for (int c = 0; c < nch; ++c) {
        const int stg = c % CH_STAGES;
        const uint32_t sWn = sbase + stg * CH_STAGE_BYTES, sQg = sWn + BLK_QG, sKp = sWn + BLK_KP, sAq = sWn + BLK_AQ,
                       sU = sWn + BLK_HEAD;
        const float gc = *reinterpret_cast<const float*>(smem + stg * CH_STAGE_BYTES + BLK_GC);
        TS(0);
        // (1) ut^T[32v x 16c] = u^T + S^T (-w)^T   and  (2a) o[16c x 32v] = Qg S, interleaved over k-quarters
        float Ua[2][2][4], Oacc[2][2][4] = {};
#pragma unroll
        for (int vg = 0; vg < VG; ++vg) {
            uint32_t r[4];
            ldsm4t(r, sU + swz<VS>(cq * 16 + (mi >> 1) * 8 + (lane & 7), voff + vg * 16 + (mi & 1) * 8));
            float2 f0 = unpack2(r[0]), f1 = unpack2(r[1]), f2 = unpack2(r[2]), f3 = unpack2(r[3]);
            Ua[vg][0][0] = f0.x; Ua[vg][0][1] = f0.y; Ua[vg][0][2] = f1.x; Ua[vg][0][3] = f1.y;
            Ua[vg][1][0] = f2.x; Ua[vg][1][1] = f2.y; Ua[vg][1][2] = f3.x; Ua[vg][1][3] = f3.y;
        }
        {
            uint32_t b[4], bn[4], a[4], an[4];
            ldsm4(b, sWn + swz<D>(ldBn_row, (4 * cq + ldBn_chunk) * 8));  // k16-step 2*cq (first step of own quarter)
            ldsm4(a, sQg + swz<D>(ldA_row, (4 * cq + ldA_chunk) * 8));
            if (c > 0) {  // gather slot 1 (at c == 0 all slots come from the initial-state registers)
                const int q1 = qn[1];
#pragma unroll
                for (int vg = 0; vg < VG; ++vg)
#pragma unroll
                    for (int t4 = 0; t4 < 4; ++t4) {
                        Sb[vg][4 + t4][0] = xS[q1 * 512 + (vg * 8 + t4 * 2) * 32 + lane];
                        Sb[vg][4 + t4][1] = xS[q1 * 512 + (vg * 8 + t4 * 2 + 1) * 32 + lane];
                    }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j + 2 < 4 && c > 0) {
                    const int q2 = qn[j + 2];
#pragma unroll
                    for (int vg = 0; vg < VG; ++vg)
#pragma unroll
                        for (int t4 = 0; t4 < 4; ++t4) {
                            Sb[vg][(j + 2) * 4 + t4][0] = xS[q2 * 512 + (vg * 8 + t4 * 2) * 32 + lane];
                            Sb[vg][(j + 2) * 4 + t4][1] = xS[q2 * 512 + (vg * 8 + t4 * 2 + 1) * 32 + lane];
                        }
                }
#pragma unroll
                for (int kq = 0; kq < 2; ++kq) {
                    const int ks = 2 * j + kq;  // static k16-step slot
                    const bool more = (j + 1 < 4) || (kq == 0);
                    if (more) {
                        const int kkn = (kq == 0) ? 2 * qn[j] + 1 : 2 * qn[j + 1 < 4 ? j + 1 : 0];
                        ldsm4(bn, sWn + swz<D>(ldBn_row, (2 * kkn + ldBn_chunk) * 8));
                        ldsm4(an, sQg + swz<D>(ldA_row, (2 * kkn + ldA_chunk) * 8));
                    }
#pragma unroll
                    for (int vg = 0; vg < VG; ++vg) {
                        uint32_t sa[4] = {Sb[vg][2 * ks][0], Sb[vg][2 * ks][1], Sb[vg][2 * ks + 1][0], Sb[vg][2 * ks + 1][1]};
                        mma16816(Ua[vg][0], sa, b[0], b[1]);
                        mma16816(Ua[vg][1], sa, b[2], b[3]);
                        mma16816(Oacc[vg][0], a, Sb[vg][2 * ks][0], Sb[vg][2 * ks + 1][0]);
                        mma16816(Oacc[vg][1], a, Sb[vg][2 * ks][1], Sb[vg][2 * ks + 1][1]);
                    }
                    if (more) {
#pragma unroll
                        for (int i = 0; i < 4; ++i) { b[i] = bn[i]; a[i] = an[i]; }
                    }
                }
            }
        }
        TS(1);
        uint32_t Ubf[2][8][2];
#pragma unroll
        for (int vg = 0; vg < VG; ++vg)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                Ubf[vg][t][0] = pack2(Ua[vg][t][0], Ua[vg][t][1]);
                Ubf[vg][t][1] = pack2(Ua[vg][t][2], Ua[vg][t][3]);
                xU[cq * 256 + (vg * 4 + t * 2) * 32 + lane] = Ubf[vg][t][0];
                xU[cq * 256 + (vg * 4 + t * 2 + 1) * 32 + lane] = Ubf[vg][t][1];
            }
        TS(2);
        named_bar_sync(1, 128);  // sync A: xU visible
        TS(3);
        {
            const int q1 = qn[1];
#pragma unroll
            for (int vg = 0; vg < VG; ++vg) {
                Ubf[vg][2][0] = xU[q1 * 256 + (vg * 4 + 0) * 32 + lane];
                Ubf[vg][2][1] = xU[q1 * 256 + (vg * 4 + 1) * 32 + lane];
                Ubf[vg][3][0] = xU[q1 * 256 + (vg * 4 + 2) * 32 + lane];
                Ubf[vg][3][1] = xU[q1 * 256 + (vg * 4 + 3) * 32 + lane];
            }
        }
        TS(4);
        // (3) S^T[32v x 32k] = gc S^T + ut^T kp   and  (2b) o += Aq ut, interleaved over c-quarters
#pragma unroll
        for (int vg = 0; vg < VG; ++vg)
#pragma unroll
            for (int t4 = 0; t4 < 4; ++t4)
#pragma unroll
                for (int i = 0; i < 4; ++i) Sacc[vg][t4][i] *= gc;
        {
            uint32_t b[2][4], bn[2][4], a[4], an[4];
            ldsm4t(b[0], sKp + swz<D>(cq * 16 + ldBt_rowoff, ldBt_col0));
            ldsm4t(b[1], sKp + swz<D>(cq * 16 + ldBt_rowoff, ldBt_col0 + 16));
            ldsm4(a, sAq + swz<64>(ldA_row, (2 * cq + ldA_chunk) * 8));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j + 2 < 4) {
                    const int q2 = qn[j + 2];
#pragma unroll
                    for (int vg = 0; vg < VG; ++vg) {
                        Ubf[vg][(j + 2) * 2][0] = xU[q2 * 256 + (vg * 4 + 0) * 32 + lane];
                        Ubf[vg][(j + 2) * 2][1] = xU[q2 * 256 + (vg * 4 + 1) * 32 + lane];
                        Ubf[vg][(j + 2) * 2 + 1][0] = xU[q2 * 256 + (vg * 4 + 2) * 32 + lane];
                        Ubf[vg][(j + 2) * 2 + 1][1] = xU[q2 * 256 + (vg * 4 + 3) * 32 + lane];
                    }
                }
                if (j + 1 < 4) {
                    const int ccn = qn[j + 1 < 4 ? j + 1 : 0];
                    ldsm4t(bn[0], sKp + swz<D>(ccn * 16 + ldBt_rowoff, ldBt_col0));
                    ldsm4t(bn[1], sKp + swz<D>(ccn * 16 + ldBt_rowoff, ldBt_col0 + 16));
                    ldsm4(an, sAq + swz<64>(ldA_row, (2 * ccn + ldA_chunk) * 8));
                }
#pragma unroll
                for (int vg = 0; vg < VG; ++vg) {
                    uint32_t ua[4] = {Ubf[vg][2 * j][0], Ubf[vg][2 * j][1], Ubf[vg][2 * j + 1][0], Ubf[vg][2 * j + 1][1]};
                    mma16816(Sacc[vg][0], ua, b[0][0], b[0][1]);
                    mma16816(Sacc[vg][1], ua, b[0][2], b[0][3]);
                    mma16816(Sacc[vg][2], ua, b[1][0], b[1][1]);
                    mma16816(Sacc[vg][3], ua, b[1][2], b[1][3]);
                    mma16816(Oacc[vg][0], a, Ubf[vg][2 * j][0], Ubf[vg][2 * j + 1][0]);
                    mma16816(Oacc[vg][1], a, Ubf[vg][2 * j][1], Ubf[vg][2 * j + 1][1]);
                }
                if (j + 1 < 4) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) { b[0][i] = bn[0][i]; b[1][i] = bn[1][i]; a[i] = an[i]; }
                }
            }
        }
        TS(5);
        if (c + 1 < nch) ready_next = mbar_test(full0 + 8 * ((c + 1) % CH_STAGES), ((c + 1) / CH_STAGES) & 1);
#pragma unroll
        for (int vg = 0; vg < VG; ++vg)
#pragma unroll
            for (int t4 = 0; t4 < 4; ++t4) {
                Sb[vg][t4][0] = pack2(Sacc[vg][t4][0], Sacc[vg][t4][1]);
                Sb[vg][t4][1] = pack2(Sacc[vg][t4][2], Sacc[vg][t4][3]);
                xS[cq * 512 + (vg * 8 + t4 * 2) * 32 + lane] = Sb[vg][t4][0];
                xS[cq * 512 + (vg * 8 + t4 * 2 + 1) * 32 + lane] = Sb[vg][t4][1];
            }
        TS(6);
        {
            const int r0 = c * C + cq * 16 + g, r1 = r0 + 8;
            __nv_bfloat16* o0 = p.out + ((t0 + r0) * HV + h) * D + v0 + tig * 2;
#pragma unroll
            for (int vg = 0; vg < VG; ++vg)
#pragma unroll
                for (int nv = 0; nv < 2; ++nv) {
                    if (r0 < L) *reinterpret_cast<uint32_t*>(o0 + vg * 16 + nv * 8) = pack2(Oacc[vg][nv][0], Oacc[vg][nv][1]);
                    if (r1 < L) *reinterpret_cast<uint32_t*>(o0 + 8 * HV * D + vg * 16 + nv * 8) = pack2(Oacc[vg][nv][2], Oacc[vg][nv][3]);
                }
        }
        TS(7);
        named_bar_sync(1, 128);  // sync B: xS visible; stage of chunk c free for reuse
        if (lane == 0) mbar_arrive(empty0 + 8 * stg);
        TS(8);
        if (c + 1 < nch && !ready_next) mbar_wait(full0 + 8 * ((c + 1) % CH_STAGES), ((c + 1) / CH_STAGES) & 1);
        TS(9);
    }
}

template <int VG>
__device__ __forceinline__ void chain_cta(const Params& p, uint8_t* smem, int ord, int h, int vs) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int slice = (VG == 2) ? vs : (vs >> 1);  // 32-wide U slice holding this CTA's v rows
    if (tid == 0) {
        const uint32_t bar0 = smem_u32(smem) + CH_BAR;
#pragma unroll
        for (int i = 0; i < CH_STAGES; ++i) { mbar_init(bar0 + 8 * i, 1); mbar_init(bar0 + 8 * (CH_STAGES + i), 4); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");  // PDL: prepass grid complete and its writes visible
    const int4 pl = *reinterpret_cast<const int4*>(p.plan + ord * 4);
    const int seq = pl.x, cbase = pl.y, L = pl.z;
    const long long t0 = p.cu[seq];
    const int cq = warp & 3;
    const int v0 = vs * 16 * VG;
    float* ns = p.new_state + ((size_t)seq * HV + h) * D * D;
    if (L <= 0) {  // definition: new_state stays zero for empty sequences
        if (warp < 4) {
#pragma unroll
            for (int vg = 0; vg < VG; ++vg)
#pragma unroll
                for (int t4 = 0; t4 < 4; ++t4) {
                    const int kk = cq * 32 + t4 * 8 + tig * 2;
                    *reinterpret_cast<float2*>(ns + (v0 + vg * 16 + g) * D + kk) = make_float2(0.f, 0.f);
                    *reinterpret_cast<float2*>(ns + (v0 + vg * 16 + g + 8) * D + kk) = make_float2(0.f, 0.f);
                }
        }
        return;
    }
    const int nch = (L + C - 1) / C;
    __syncthreads();
    if (warp == 4) { chain_producer(p, smem, h, slice, cbase, nch); return; }
    float Sacc[2][4][4];
    uint32_t Sb[2][16][2];
    {
        const float* st = p.state ? p.state + ((size_t)seq * HV + h) * D * D : nullptr;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = (cq + j) & 3;
#pragma unroll
            for (int vg = 0; vg < VG; ++vg)
#pragma unroll
                for (int t4 = 0; t4 < 4; ++t4) {
                    const int kk = q * 32 + t4 * 8 + tig * 2;
                    float2 r0 = make_float2(0.f, 0.f), r1 = make_float2(0.f, 0.f);
                    if (st) {
                        r0 = *reinterpret_cast<const float2*>(st + (v0 + vg * 16 + g) * D + kk);
                        r1 = *reinterpret_cast<const float2*>(st + (v0 + vg * 16 + g + 8) * D + kk);
                    }
                    Sb[vg][j * 4 + t4][0] = pack2(r0.x, r0.y);
                    Sb[vg][j * 4 + t4][1] = pack2(r1.x, r1.y);
                    if (j == 0) { Sacc[vg][t4][0] = r0.x; Sacc[vg][t4][1] = r0.y; Sacc[vg][t4][2] = r1.x; Sacc[vg][t4][3] = r1.y; }
                }
        }
    }
    CTA_TS(2);
    chain_loop<VG>(p, smem, h, vs, cq, cbase, nch, t0, L, Sacc, Sb);
#pragma unroll
    for (int vg = 0; vg < VG; ++vg)
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
            const int kk = cq * 32 + t4 * 8 + tig * 2;
            *reinterpret_cast<float2*>(ns + (v0 + vg * 16 + g) * D + kk) = make_float2(Sacc[vg][t4][0], Sacc[vg][t4][1]);
            *reinterpret_cast<float2*>(ns + (v0 + vg * 16 + g + 8) * D + kk) = make_float2(Sacc[vg][t4][2], Sacc[vg][t4][3]);
        }
    CTA_TS(3);
}

// grid: the first p.nFine sequences (by length rank) use 16-wide v slices (64 CTAs per sequence, half the per-step
// work on the critical path); the rest use 32-wide slices (32 CTAs per sequence).
__global__ void __launch_bounds__(CH_THREADS, 1) chain_kernel(Params p) {
    extern __shared__ __align__(1024) uint8_t smem[];
    const int fineCTAs = p.nFine * HV * 8;
    CTA_TS(0); CTA_TS(1);
    if ((int)blockIdx.x < fineCTAs) {
        const int ord = blockIdx.x >> 6, h = (blockIdx.x >> 3) & 7, vs = blockIdx.x & 7;
        chain_cta<1>(p, smem, ord, h, vs);
    } else {
        const int b = blockIdx.x - fineCTAs;
        const int ord = p.nFine + (b >> 5), h = (b >> 2) & 7, vs = b & 3;
        chain_cta<2>(p, smem, ord, h, vs);
    }
}

}  // namespace

extern "C" {

size_t gdn_workspace_bytes(int T, int N) {
    size_t maxChunks = (size_t)(T / C + N);
    return maxChunks * HV * (size_t)BLK_BYTES + (size_t)N * 16 + 256 + 128 * 16 * 8 * 2 + (size_t)N * HV * 8 * 64;
}

int gdn_forward(const void* q, const void* k, const void* v, const float* state, const float* A_log, const void* a,
                const float* dt_bias, const void* b, const int64_t* cu_seqlens, float scale, int T, int N, void* out,
                float* new_state, void* workspace, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(prepass_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, PP_SMEM);
        cudaFuncSetAttribute(chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, CH_SMEM);
        attr_set = true;
    }
    Params p;
    p.q = (const __nv_bfloat16*)q; p.k = (const __nv_bfloat16*)k; p.v = (const __nv_bfloat16*)v;
    p.a = (const __nv_bfloat16*)a; p.b = (const __nv_bfloat16*)b;
    p.state = state; p.A_log = A_log; p.dt_bias = dt_bias; p.cu = cu_seqlens; p.N = N; p.T = T; p.scale = scale;
    p.out = (__nv_bfloat16*)out; p.new_state = new_state;
    const size_t maxChunks = (size_t)(T / C + N);
    uint8_t* ws = (uint8_t*)workspace;
    p.blk = ws;
    p.plan = (int*)(ws + maxChunks * HV * (size_t)BLK_BYTES);
    p.nFine = N < NFINE ? N : NFINE;
    p.dbg = (long long*)(ws + maxChunks * HV * (size_t)BLK_BYTES + (size_t)N * 16);
    p.dbg2 = p.dbg + 128 * 16;
    prepass_kernel<<<(unsigned)(maxChunks * HV), NTHREADS, PP_SMEM, stream>>>(p);
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3((unsigned)(N * HV * NVS)); cfg.blockDim = dim3(CH_THREADS); cfg.dynamicSmemBytes = CH_SMEM; cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr; cfg.numAttrs = 1;
        cfg.gridDim = dim3((unsigned)(p.nFine * HV * 8 + (N - p.nFine) * HV * NVS));
        cudaLaunchKernelEx(&cfg, chain_kernel, p);
    }
    return (int)cudaGetLastError();
}

int gdn_smem_sizes(int* pp, int* ch) { *pp = PP_SMEM; *ch = CH_SMEM; return 0; }
}
