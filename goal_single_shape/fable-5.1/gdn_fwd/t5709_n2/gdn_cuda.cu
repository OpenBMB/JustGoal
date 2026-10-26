// Gated Delta Net prefill helpers for sm_100a: meta, prep (K K^T + gates + T elimination on 64-token
// blocks), sequential state kernel on 128-token chunks.  Written from scratch (mma.sync / ldmatrix / TMA bulk).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define CB 64          // elimination block (tokens)
#define CH 128         // state chunk (tokens)
#define DK 128
#define DV 128
#define VW 16          // v-slice width per CTA in the sequential kernel
#define NVS (DV / VW)

typedef __nv_bfloat16 bf16;
__device__ long long* g_dbg = nullptr;
#ifdef GDN_DEBUG
#define DBG_REC(idx) do { if (g_dbg && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) g_dbg[idx] = clock64(); } while (0)
#define DBG_ON 1
#else
#define DBG_REC(idx) do {} while (0)
#define DBG_ON 0
#endif
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define DBG_CTA(slot) do { if (DBG_ON && g_dbg && threadIdx.x == 0) { int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); g_dbg[1024 + 2 * bid + (slot)] = gtimer(); } } while (0)

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&t);
}
__device__ __forceinline__ float softplus_f(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\n.reg .pred p;\nWAIT_LOOP_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_LOOP_%=;\n}\n" :: "r"(bar), "r"(parity));
}
__device__ __forceinline__ void tma_bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// ------------------------------------------------------------------ meta kernel
// cstart64[n], cstart128[n]: chunk prefix counts; cinfo{64,128}[c] = {seq, tok0, len, 0}
#define SHORT_CHUNKS 2   // sequences with at most this many 128-chunks are processed first (their seq items then never wait long)
// cstart64[n]: 64-block prefix counts (natural order, Triton path only; skipped when C64max == 0).
// cstart128: [0..N-1] = first chunk (cbeg) of sequence n, [N] = total chunks, [N+1 .. 2N] = end chunk (cend) of sequence n.
// Chunk order (fused path): short sequences first (natural order among them), then the others; cinfo128[c] = {seq, tok0, len, 0}.
__global__ void meta_kernel(const int64_t* __restrict__ cu_seqlens, int N, int C64max, int C128max,
                            int* __restrict__ cstart64, int* __restrict__ cstart128,
                            int* __restrict__ cinfo64, int* __restrict__ cinfo128, int* __restrict__ counters, int* __restrict__ ready, int* __restrict__ ready2, int HVm,
                            int* __restrict__ seq_order, int Ksplit, int max_early) {
    // smem: s64[N+1], s_pos[N+1] (chunk prefix over positions), s_order[N] (position -> sequence), s_beg[N], s_base[N]
    extern __shared__ int s_cs[];
    if (threadIdx.x < 16) counters[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < C128max; i += blockDim.x) ready[i] = 0;
    for (int i = threadIdx.x; i < C128max * HVm; i += blockDim.x) ready2[i] = 0;
    int* s64 = s_cs; int* s_pos = s_cs + (N + 1); int* s_order = s_cs + 2 * (N + 1); int* s_beg = s_order + N; int* s_base = s_beg + N;
    const int per = HVm * NVS;
    const bool permute = (C64max == 0);
    if (threadIdx.x == 0) {
        int a64 = 0;
        for (int n = 0; n < N; ++n) { s64[n] = a64; a64 += (int)((cu_seqlens[n + 1] - cu_seqlens[n] + CB - 1) / CB); }
        s64[N] = a64;
        int pos = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (int n = 0; n < N; ++n) {
                const int nch = (int)((cu_seqlens[n + 1] - cu_seqlens[n] + CH - 1) / CH);
                const bool first_group = permute ? (nch <= SHORT_CHUNKS) : true;
                if ((pass == 0) != first_group) continue;
                s_order[pos++] = n;
            }
        }
        int a128 = 0;
        for (int k = 0; k < N; ++k) {
            const int n = s_order[k];
            s_pos[k] = a128; s_beg[n] = a128;
            a128 += (int)((cu_seqlens[n + 1] - cu_seqlens[n] + CH - 1) / CH);
        }
        s_pos[N] = a128;
        // seq work items: (n, hv, vs) encoded as (n*HVm + hv)*NVS + vs. Early group: sequences whose first chunk's intra
        // items lie before Ksplit (at most max_early items in total); the rest go to the late group. counters[1] = n_early.
        int cnt = 0;
        for (int n = 0; n < N; ++n) {
            const int nch = (int)((cu_seqlens[n + 1] - cu_seqlens[n] + CH - 1) / CH);
            const bool early = (s_beg[n] * HVm < Ksplit) && (nch > 0) && (cnt + per <= max_early);
            s_base[n] = early ? cnt : -1;
            if (early) cnt += per;
        }
        int p2 = cnt;
        for (int n = 0; n < N; ++n) { if (s_base[n] < 0) { s_base[n] = p2; p2 += per; } }
        counters[1] = cnt;
    }
    __syncthreads();
    if (C64max > 0) for (int n = threadIdx.x; n <= N; n += blockDim.x) cstart64[n] = s64[n];
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        const int nch = (int)((cu_seqlens[n + 1] - cu_seqlens[n] + CH - 1) / CH);
        cstart128[n] = s_beg[n]; cstart128[N + 1 + n] = s_beg[n] + nch;
    }
    if (threadIdx.x == 0) cstart128[N] = s_pos[N];
    for (int idx = threadIdx.x; idx < N * per; idx += blockDim.x) seq_order[s_base[idx / per] + (idx % per)] = idx;
    // cinfo64 (natural order) and cinfo128 (position order)
    for (int pass = 0; pass < 2; ++pass) {
        const int* sc = pass == 0 ? s64 : s_pos;
        int* ci = pass == 0 ? cinfo64 : cinfo128;
        const int csz = pass == 0 ? CB : CH;
        const int cmax = pass == 0 ? C64max : C128max;
        const int total = sc[N];
        for (int c = threadIdx.x; c < cmax; c += blockDim.x) {
            if (c >= total) { ci[c * 4 + 0] = -1; ci[c * 4 + 1] = 0; ci[c * 4 + 2] = 0; ci[c * 4 + 3] = 0; continue; }
            int lo = 0, hi = N - 1;
            while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (sc[mid] <= c) lo = mid; else hi = mid - 1; }
            const int n = pass == 0 ? lo : s_order[lo];
            int64_t t0 = cu_seqlens[n] + (int64_t)(c - sc[lo]) * csz;
            int len = (int)min((int64_t)csz, cu_seqlens[n + 1] - t0);
            ci[c * 4 + 0] = n; ci[c * 4 + 1] = (int)t0; ci[c * 4 + 2] = len; ci[c * 4 + 3] = 0;
        }
    }
}

// ------------------------------------------------------------------ prep kernel (64-token blocks)
// grid (C64max, HQK), block 256.  For block c and qk-head hq (v-heads hq*2, hq*2+1):
//   gam[c][hv][64] (cumulative log decay within block, constant over padding), bet[c][hv][64] (0 on padding)
//   Tm[c][hv][64][64] = T - I  (bf16, strictly lower), T = (I + A)^-1, A[t][s] = beta_t exp(gam_t-gam_s) k_t.k_s (s<t)
// T is computed blockwise (4 blocks of 16): diagonal blocks by exact fp32 elimination (warp-synchronous),
// off-diagonal blocks T_ij = -T_ii sum_m A_im T_mj with bf16 mma.
#define KS_STRIDE (DK + 8)
#define TS_STRIDE (CB + 8)     // bf16 elements per row of Ts / Ab (144 B)
#define XS_STRIDE 24

template <int REP>
__global__ void __launch_bounds__(256, 4) prep_kernel(
    const bf16* __restrict__ k, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cinfo, int HQK, int HV,
    float* __restrict__ gam, float* __restrict__ bet, bf16* __restrict__ Tm)
{
    static_assert(REP == 2, "block layout assumes REP == 2");
    __shared__ __align__(16) union { bf16 Ks[CB * KS_STRIDE]; bf16 Ab[REP][CB][TS_STRIDE]; } ka;  // 18432 B (K dead after S)
    __shared__ __align__(16) union { float S[CB][CB]; bf16 Ts[REP][CB][TS_STRIDE]; } u;   // 18432 B
    bf16* Ks = ka.Ks;
    auto Ab = ka.Ab;
    __shared__ float Ad[REP][4][16][16];                                  // 8192 B   fp32 diagonal blocks of A
    __shared__ __align__(16) bf16 Xs[REP][3][16][XS_STRIDE];              // 4608 B
    __shared__ float sG[REP][CB], sB[REP][CB];
    __shared__ float rowbuf[8][2][16];

    const int c = blockIdx.x, hq = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int len = cinfo[c * 4 + 2];
    if (len <= 0) return;
    const int tok0 = cinfo[c * 4 + 1];
    DBG_REC(0); DBG_CTA(0);

    // ---- load K block (zero padded rows): 1024 x 16B, 4 per thread
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int i = tid + q * 256;
        int row = i >> 4, col8 = i & 15;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (row < len) val = *reinterpret_cast<const uint4*>(k + ((size_t)(tok0 + row) * HQK + hq) * DK + col8 * 8);
        *reinterpret_cast<uint4*>(Ks + row * KS_STRIDE + col8 * 8) = val;
    }
    // ---- gates: warp r < REP handles v-head hq*REP + r; lane covers tokens 2*lane, 2*lane+1
    if (warp < REP) {
        const int hv = hq * REP + warp;
        const float ea = __expf(A_log[hv]);
        const float dtb = dt_bias[hv];
        float lg[2], be[2];
        #pragma unroll
        for (int u2 = 0; u2 < 2; ++u2) {
            int t = lane * 2 + u2;
            if (t < len) {
                float av = __bfloat162float(a[(size_t)(tok0 + t) * HV + hv]) + dtb;
                float bv = __bfloat162float(b[(size_t)(tok0 + t) * HV + hv]);
                lg[u2] = -ea * softplus_f(av);
                be[u2] = 1.f / (1.f + __expf(-bv));
            } else { lg[u2] = 0.f; be[u2] = 0.f; }
        }
        float loc = lg[0] + lg[1];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { float v = __shfl_up_sync(0xffffffffu, loc, o); if (lane >= o) loc += v; }
        float g1 = loc, g0 = loc - lg[1];
        sG[warp][lane * 2] = g0; sG[warp][lane * 2 + 1] = g1;
        sB[warp][lane * 2] = be[0]; sB[warp][lane * 2 + 1] = be[1];
        size_t base = ((size_t)c * HV + hv) * CB;
        *reinterpret_cast<float2*>(gam + base + lane * 2) = make_float2(g0, g1);
        *reinterpret_cast<float2*>(bet + base + lane * 2) = make_float2(be[0], be[1]);
    }
    __syncthreads();
    DBG_REC(1);

    // ---- S = K K^T via mma: 8 warps: rows 16*(warp&3), cols 32*(warp>>2)
    {
        const int r0 = (warp & 3) * 16, n0 = (warp >> 2) * 32;
        float acc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
        const uint32_t ks_base = smem_u32(Ks);
        #pragma unroll
        for (int kk = 0; kk < DK / 16; ++kk) {
            uint32_t af[4];
            int row = r0 + (lane & 7) + ((lane >> 3) & 1) * 8;
            int kc = kk * 2 + (lane >> 4);
            ldsm_x4(af[0], af[1], af[2], af[3], ks_base + (row * KS_STRIDE + kc * 8) * 2);
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t bfr[4];
                int nrow = n0 + np * 16 + (lane & 7) + (lane >> 4) * 8;
                int kc2 = kk * 2 + ((lane >> 3) & 1);
                ldsm_x4(bfr[0], bfr[1], bfr[2], bfr[3], ks_base + (nrow * KS_STRIDE + kc2 * 8) * 2);
                mma_bf16(acc[np * 2], af, bfr);
                mma_bf16(acc[np * 2 + 1], af, bfr + 2);
            }
        }
        const int g = lane >> 2, cc = (lane & 3) * 2;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            int col = n0 + nt * 8 + cc;
            *reinterpret_cast<float2*>(&u.S[r0 + g][col]) = make_float2(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<float2*>(&u.S[r0 + g + 8][col]) = make_float2(acc[nt][2], acc[nt][3]);
        }
    }
    __syncthreads();
    DBG_REC(2);

    // ---- A[r][t][s] = beta_t exp(G_t - G_s) S[t][s] (s < t): bf16 full into Ab, fp32 diagonal blocks into Ad
    {   // thread -> (r, s2 fixed, t half); consecutive lanes -> consecutive s2 (conflict-free)
        const int r = tid >> 7, s2 = tid & 63, th = (tid >> 6) & 1;
        const float gs = sG[r][s2];
        const int sb = s2 >> 4;
        #pragma unroll
        for (int q = 0; q < 32; ++q) {
            int t = th * 32 + q;
            float v = (s2 < t) ? sB[r][t] * __expf(sG[r][t] - gs) * u.S[t][s2] : 0.f;
            Ab[r][t][s2] = __float2bfloat16(v);
            if ((t >> 4) == sb) Ad[r][sb][t & 15][s2 & 15] = v;
        }
    }
    __syncthreads();
    // zero Ts (aliases S, which is dead now): REP*64*72 bf16 = 18432 B = 1152 x 16B
    for (int i = tid; i < REP * CB * TS_STRIDE / 8; i += 256)
        reinterpret_cast<uint4*>(&u.Ts[0][0][0])[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();
    DBG_REC(3);

    // ---- Phase D: diagonal blocks T_bb = (I + A_bb)^-1, warp-synchronous. warp -> (r = warp>>2, b = warp&3)
    {
        const int r = warp >> 2, bb = warp & 3;
        const int j = lane & 15, half = lane >> 4;     // rows 8*half .. 8*half+7
        float Tr[8];
        #pragma unroll
        for (int q = 0; q < 8; ++q) Tr[q] = 0.f;
        #pragma unroll
        for (int m = 0; m < 15; ++m) {
            if ((m >> 3) == half) rowbuf[warp][m & 1][j] = (j < m) ? Tr[m & 7] : (j == m ? 1.f : 0.f);
            __syncwarp();
            float tmj = rowbuf[warp][m & 1][j];
            if (j <= m) {
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    int i = half * 8 + q;
                    if (i > m) Tr[q] -= Ad[r][bb][i][m] * tmj;
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            int i = half * 8 + q;
            float tv = (i > j) ? Tr[q] : (i == j ? 1.f : 0.f);
            u.Ts[r][bb * 16 + i][bb * 16 + j] = __float2bfloat16(tv);
        }
    }
    __syncthreads();
    DBG_REC(4);

    // ---- Phase O: block rows i = 1..3 ; warp (r, j) with j < i computes X_ij then T_ij = -T_ii X_ij
    #pragma unroll 1
    for (int bi = 1; bi < 4; ++bi) {
        const int r = warp / 3, bj = warp % 3;
        const bool active = (warp < 6) && (bj < bi);
        const int g = lane >> 2, cc = (lane & 3) * 2;
        float acc[2][4];
        if (active) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
            for (int bm = bj; bm < bi; ++bm) {
                uint32_t af[4], bfr[4];
                int row = bi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int kc = bm * 2 + (lane >> 4);
                ldsm_x4(af[0], af[1], af[2], af[3], smem_u32(&Ab[r][row][kc * 8]));
                int krow = bm * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int nc = bj * 2 + (lane >> 4);
                ldsm_x4_trans(bfr[0], bfr[1], bfr[2], bfr[3], smem_u32(&u.Ts[r][krow][nc * 8]));
                mma_bf16(acc[0], af, bfr);
                mma_bf16(acc[1], af, bfr + 2);
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                *reinterpret_cast<uint32_t*>(&Xs[r][bj][g][nt * 8 + cc]) = pack_bf16x2(acc[nt][0], acc[nt][1]);
                *reinterpret_cast<uint32_t*>(&Xs[r][bj][g + 8][nt * 8 + cc]) = pack_bf16x2(acc[nt][2], acc[nt][3]);
            }
        }
        __syncthreads();
        if (active) {
            uint32_t af[4], bfr[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
            int row = bi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            int kc = bi * 2 + (lane >> 4);
            ldsm_x4(af[0], af[1], af[2], af[3], smem_u32(&u.Ts[r][row][kc * 8]));
            int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
            int nc = lane >> 4;
            ldsm_x4_trans(bfr[0], bfr[1], bfr[2], bfr[3], smem_u32(&Xs[r][bj][krow][nc * 8]));
            mma_bf16(acc[0], af, bfr);
            mma_bf16(acc[1], af, bfr + 2);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                int col = bj * 16 + nt * 8 + cc;
                *reinterpret_cast<uint32_t*>(&u.Ts[r][bi * 16 + g][col]) = pack_bf16x2(-acc[nt][0], -acc[nt][1]);
                *reinterpret_cast<uint32_t*>(&u.Ts[r][bi * 16 + g + 8][col]) = pack_bf16x2(-acc[nt][2], -acc[nt][3]);
            }
        }
        __syncthreads();
    }
    DBG_REC(5);

    // ---- write Tm = T - I (bf16 [64][64] contiguous): 2 heads x 512 vectors of 8, 4 per thread
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int i = tid + q * 256;
        int r = i >> 9, rem = i & 511, row = rem >> 3, grp = rem & 7;
        uint4 val = *reinterpret_cast<const uint4*>(&u.Ts[r][row][grp * 8]);
        if ((row >> 3) == grp) {   // diagonal element inside this group -> zero it
            bf16* e = reinterpret_cast<bf16*>(&val);
            e[row & 7] = __float2bfloat16(0.f);
        }
        if ((row >> 3) < grp) val = make_uint4(0, 0, 0, 0);   // strictly upper (already zero, keep explicit)
        const int hv = hq * REP + r;
        *reinterpret_cast<uint4*>(Tm + ((size_t)c * HV + hv) * CB * CB + row * CB + grp * 8) = val;
    }
    DBG_REC(6); DBG_CTA(1);
}

#ifndef IT_GRID_PER_SM
#define IT_GRID_PER_SM 1
#endif
// ===================== (inlined) gdn_intra =====================
// Fused intra-chunk kernel (128-token chunks), tcgen05 version.
// One persistent CTA per SM (256 threads = 8 warps).  For each item (128-chunk c, v-head hv):
//   gates, S = K K^T, A-build, T = (I+A)^-1 (blockwise: fp32 16x16 diagonal blocks + mma.sync off-diagonal),
//   W = (T-I) diag(bg) K + bg K,  Qt = scale e^G Q - A_att W,  Mp = -K^T (dL W),
//   U = (T-I) diag(beta) V + beta V,  Ot = A_att U,  B = K^T (dL U).
// bf16 128x128 smem tiles use two 64-col halves, each [128 rows][128 B] with 16B-chunk XOR swizzle (row & 7)
// == canonical SWIZZLE_128B layout usable as K-major (rows = M/N) or MN-major (rows = K) tcgen05 operands.

#define IT_TILE 32768
#define IT_OFF_K 0
#define IT_OFF_T 32768
#define IT_OFF_WV 65536
#define IT_OFF_QA 98304
#define IT_OFF_V 131072
#define IT_OFF_K2 163840          // second K tile (next item's K prefetch)
#define IT_OFF_X 196608           // 16 KB scratch: X21 (8 KB) / per-warp X blocks / tf32 doubling scratch (8 x 2 KB)
#define IT_OFF_G 212992           // gates: G, beta, eG, bg, dL (5 x 128 fp32)
#define IT_OFF_MISC (IT_OFF_G + 5 * 128 * 4)   // mbarrier (8B), tmem holder (4B), rowbuf
#define IT_MISC_BYTES 5120
#define IT_GVALS_OFF 1088                     // per-chunk gate values of a seq item (fp32), IT_GVALS_CAP entries
#define IT_GVALS_CAP ((IT_MISC_BYTES - IT_GVALS_OFF) / 4)
#define IT_SMEM (IT_OFF_MISC + IT_MISC_BYTES + 1024)   // + 1024 for base alignment ; misc: mbar x4 @0, tmem holder @32, out-load barrier @40, seq barriers @64..128, flags @128..192, gvals @1088
#define IT_AD_OFF (IT_OFF_T + 16384)          // Ad (fp32 8x16x16 = 8 KB) aliases T block (0,1) region during elimination

__device__ __forceinline__ uint32_t tile_off(int r, int c) {
    return ((c >> 6) << 14) + (r << 7) + (((((c >> 3) & 7) ^ (r & 7))) << 4) + ((c & 7) << 1);
}
__device__ __forceinline__ void ldA(uint32_t* a, uint32_t base, int m0, int k0, int lane) {
    int r = m0 + (lane & 7) + ((lane >> 3) & 1) * 8, c = k0 + (lane >> 4) * 8;
    ldsm_x4(a[0], a[1], a[2], a[3], base + tile_off(r, c));
}
__device__ __forceinline__ void ldB_trans(uint32_t* b, uint32_t base, int n0, int k0, int lane) {
    int r = k0 + (lane & 7) + ((lane >> 3) & 1) * 8, c = n0 + (lane >> 4) * 8;
    ldsm_x4_trans(b[0], b[1], b[2], b[3], base + tile_off(r, c));
}
__device__ __forceinline__ uint32_t scale_pair(uint32_t p, float s0, float s1) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&p);
    float2 f = __bfloat1622float2(v);
    return pack_bf16x2(f.x * s0, f.y * s1);
}
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t p, uint32_t s2) {   // bf16x2 * bf16x2
    __nv_bfloat162 a = *reinterpret_cast<__nv_bfloat162*>(&p), b = *reinterpret_cast<__nv_bfloat162*>(&s2);
    __nv_bfloat162 r = __hmul2(a, b);
    return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ float2 ld_bf16x2(const unsigned char* sm, uint32_t off) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(sm + off);
    return __bfloat1622float2(v);
}
__device__ __forceinline__ void st_bf16x2(unsigned char* sm, uint32_t off, float a, float b) {
    *reinterpret_cast<uint32_t*>(sm + off) = pack_bf16x2(a, b);
}
// ---- tf32 mma.sync m16n8k8 (fp32 registers interpreted as tf32)
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t f2tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x)); return r; }
// 16x16 fp32 product C = X * Y from smem (row-major, 16 floats per row), one warp; result in C fragments
__device__ __forceinline__ void mm16_tf32(float (&acc)[2][4], const float* X, const float* Y, int lane) {
    const int g = lane >> 2, c = lane & 3;
    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4], b0[2], b1[2];
        a[0] = f2tf32(X[g * 16 + ks * 8 + c]);       a[1] = f2tf32(X[(g + 8) * 16 + ks * 8 + c]);
        a[2] = f2tf32(X[g * 16 + ks * 8 + c + 4]);   a[3] = f2tf32(X[(g + 8) * 16 + ks * 8 + c + 4]);
        b0[0] = f2tf32(Y[(ks * 8 + c) * 16 + g]);     b0[1] = f2tf32(Y[(ks * 8 + c + 4) * 16 + g]);
        b1[0] = f2tf32(Y[(ks * 8 + c) * 16 + 8 + g]); b1[1] = f2tf32(Y[(ks * 8 + c + 4) * 16 + 8 + g]);
        mma_tf32(acc[0], a, b0);
        mma_tf32(acc[1], a, b1);
    }
}
__device__ __forceinline__ void zero_acc2(float (&acc)[2][4]) { acc[0][0] = acc[0][1] = acc[0][2] = acc[0][3] = acc[1][0] = acc[1][1] = acc[1][2] = acc[1][3] = 0.f; }
__device__ __forceinline__ void st16(float* Z, const float (&acc)[2][4], int lane) {
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        *reinterpret_cast<float2*>(Z + g * 16 + nt * 8 + c2) = make_float2(acc[nt][0], acc[nt][1]);
        *reinterpret_cast<float2*>(Z + (g + 8) * 16 + nt * 8 + c2) = make_float2(acc[nt][2], acc[nt][3]);
    }
}
// ---- tcgen05 helpers
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // 128B swizzle
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(int ta, int tb, int negA) {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)negA << 13) | ((uint32_t)ta << 15) | ((uint32_t)tb << 16)
         | ((uint32_t)(128 >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
}
__device__ __forceinline__ void tc_mma(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t idesc, int accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
#define TC_REGS32(r) "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), \
    "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), \
    "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), \
    "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
#define TC_REGS32_IN(r) "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), \
    "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), \
    "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), \
    "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31])
__device__ __forceinline__ void tc_ld32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
        : TC_REGS32(r) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
#define TC_REGS64(r) TC_REGS32(r), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), \
    "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), \
    "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), \
    "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
__device__ __forceinline__ void tc_ld64(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
                 "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];\n"
        : TC_REGS64(r) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tc_st32(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"
        :: TC_REGS32_IN(r), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async16_(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit_() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait_() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// descriptor helpers for our tiles: K-major operand (rows = M or N, k along columns): k-step kk -> half kk/4, +32B*(kk%4)
__device__ __forceinline__ uint64_t desc_kmajor(uint32_t tile, int kk) { return make_sdesc(tile + (kk >> 2) * 16384 + (kk & 3) * 32, 16, 1024); }
// MN-major operand (rows = K): k-step kk -> 16 rows * 128 B ; LBO = 16384 (second 64-wide MN block), SBO = 1024
__device__ __forceinline__ uint64_t desc_mnmajor(uint32_t tile, int kk) { return make_sdesc(tile + kk * 2048, 16384, 1024); }

// issue a full 128x128x128 MMA (8 k-steps) into TMEM accumulator taddr; accumulate into existing D if acc_init
// adesc0/bdesc0: descriptors for k-step 0 (from desc_kmajor/desc_mnmajor with kk = 0); per-k-step start-address deltas
// are added in the low 14 bits (encoded >> 4): K-major: +2 per k-step within a half, +1024 between halves; MN-major: +128.
__device__ __forceinline__ void issue_mma_d(uint32_t taddr, uint64_t adesc0, int ta, uint64_t bdesc0, int tb, int negA, int acc_init, uint32_t bar) {
    const uint32_t idesc = make_idesc(ta, tb, negA);
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        const uint32_t koffA = ta ? (kk * 2048) : ((kk >> 2) * 16384 + (kk & 3) * 32);
        const uint32_t koffB = tb ? (kk * 2048) : ((kk >> 2) * 16384 + (kk & 3) * 32);
        tc_mma(taddr, adesc0 + (koffA >> 4), bdesc0 + (koffB >> 4), idesc, (kk > 0) || acc_init);
    }
    tc_commit(bar);
}
__device__ __forceinline__ void issue_mma(uint32_t taddr, uint32_t atile, int ta, uint32_t btile, int tb, int negA, int acc_init, uint32_t bar) {
    issue_mma_d(taddr, ta ? desc_mnmajor(atile, 0) : desc_kmajor(atile, 0), ta, tb ? desc_mnmajor(btile, 0) : desc_kmajor(btile, 0), tb, negA, acc_init, bar);
}

// ===================== (inlined) gdn_seq_out =====================
// ---------------- sequential state kernel v3: TMA producer warp + 4 consumer warps, 128-token chunks
// Mp global: fragment-major bf16 (32 KB per chunk-head), B: fragment-major bf16 (4 KB per chunk-head-vslice)
#ifndef SQ_NST
#define SQ_NST 4
#endif
#ifndef SEQ_X_NOMMA
#define SEQ_X_NOMMA 0
#endif
#ifndef SEQ_X_NOTMA
#define SEQ_X_NOTMA 0
#endif
#ifndef POLL_NS
#define POLL_NS 64
#endif
#define SQ_MP_BYTES (DK * DK * 2)
#define SQ_BS_BYTES (DK * VW * 2)
#define SQ_STAGE (SQ_MP_BYTES + SQ_BS_BYTES)
#define SQ_HB_STRIDE 24
#define SQ_SMEM (SQ_NST * SQ_STAGE + 2 * DK * SQ_HB_STRIDE * 2 + 256 + 4096)

__device__ __forceinline__ int ld_acquire_cta_shared(const int* p) { int v; asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v; }
__device__ __forceinline__ void st_release_cta_shared(int* p, int v) { asm volatile("st.release.cta.shared.b32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v) : "memory"); }
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
    int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void bar_consumers() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }

// seq work item executed by warps 0-4 of a 256-thread CTA (warps 5-7 must skip); caller handles barriers init/sync
#ifdef SEQ_INLINE
#define SEQ_ATTR __forceinline__
#else
#define SEQ_ATTR __noinline__
#endif
__device__ SEQ_ATTR void seq_item(unsigned char* sq_smem, uint64_t* full, uint64_t* empty, float* gstage, volatile int* hs_done,
    volatile int* s_ready_upto, float* gvals, int itb, int vs, int hv, int n,
    const bf16* __restrict__ Mp, const bf16* __restrict__ Bp, const float* __restrict__ gchunk,
    const float* __restrict__ state, float* __restrict__ new_state, bf16* __restrict__ Hs,
    const int* __restrict__ cbeg, const int* __restrict__ cend, const int* __restrict__ ready, int HV, int* __restrict__ ready2, int* hs_copied)
{
    unsigned char* stages = sq_smem;
    bf16* Hb = reinterpret_cast<bf16*>(sq_smem + SQ_NST * SQ_STAGE);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c0 = cbeg[n], c1 = cend[n];
    const int nc = c1 - c0;
    const int v0 = vs * VW;

    if (warp == 4) {
        // ---------------- producer
        if (lane == 0) {
            for (int it = 0; it < nc; ++it) {
                const int gi = itb + it;                       // stage/phase index continues across the CTA's seq items
                const int c = c0 + it, st = gi % SQ_NST;
                if (gi >= SQ_NST) mbar_wait(smem_u32(&empty[st]), ((gi / SQ_NST) - 1) & 1);   // arrival of global iteration gi-NST
                while (*s_ready_upto <= it) { __nanosleep(32); }          // poller warp certifies chunks < ready_upto
                
                uint32_t bar = smem_u32(&full[st]);
                uint32_t dst = smem_u32(stages + st * SQ_STAGE);
                if (SEQ_X_NOTMA) { mbar_arrive(bar); }
                else {
                mbar_expect_tx(bar, SQ_STAGE);
                const bf16* src = Mp + ((size_t)c * HV + hv) * DK * DK;
                #pragma unroll
                for (int qq = 0; qq < 4; ++qq)
                    tma_bulk_g2s(dst + qq * (SQ_MP_BYTES / 4), src + qq * (DK * DK / 4), SQ_MP_BYTES / 4, bar);
                const bf16* bsrc = Bp + (((size_t)c * HV + hv) * NVS + vs) * DK * VW;
                tma_bulk_g2s(dst + SQ_MP_BYTES, bsrc, SQ_BS_BYTES, bar);
                }
            }
        }
        return;
    }
    if (warp == 5) {
        // ---------------- poller: certify chunks in batches of 16 lanes; stash gchunk values; publish s_ready_upto
        int base = 0;
        while (base < nc) {
            int cnt = min(16, nc - base);
            int ok = 0; float gv = 0.f;
            if (lane < cnt) {
                ok = (ld_acquire_gpu(ready + c0 + base + lane) >= HV) ? 1 : 0;
                if (ok) gv = gchunk[(size_t)(c0 + base + lane) * HV + hv];
            }
            unsigned m = __ballot_sync(0xffffffffu, ok);
            int nready = (~m == 0u) ? 32 : __ffs(~m) - 1;    // consecutive ready lanes from lane 0
            nready = min(nready, cnt);
            if (lane < nready) gvals[base + lane] = gv;
            __syncwarp();
            if (nready > 0) {
                __threadfence_block();
                if (lane == 0) *s_ready_upto = base + nready;
                base += nready;
            } else {
                __nanosleep(POLL_NS);
            }
        }
        return;
    }
    if (warp == 6) {
        // ---------------- publisher (fused mode only): Hs[c0+i] readiness for the out items. Consumers publish the count of
        // completed Hs copies in hs_copied (st.release.cta after their barrier); this warp acquires it, issues the gpu-scope
        // fence (cumulative over the observed consumer stores) and bumps ready2 for every newly completed chunk.
        if (ready2 != nullptr && lane == 0) {
            int published = 0;
            while (published < nc) {
                int cur;
                while ((cur = ld_acquire_cta_shared(hs_copied)) <= published) { __nanosleep(32); }
                __threadfence();
                for (int i = published; i < cur; ++i) atomicAdd(ready2 + (size_t)(c0 + i) * HV + hv, 1);
                published = cur;
            }
        }
        return;
    }
    if (warp > 5) return;

    // ---------------- consumers (warps 0-3)
    const int g = lane >> 2, cc = (lane & 3) * 2;
    constexpr int NT = VW / 8;
    float acc[2][NT][4];
    {
        const float* stt = state + ((size_t)n * HV + hv) * DV * DK;
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int i = warp * 32 + m * 16 + g, v = v0 + nt * 8 + cc;
                acc[m][nt][0] = stt[(size_t)v * DK + i];
                acc[m][nt][1] = stt[(size_t)(v + 1) * DK + i];
                acc[m][nt][2] = stt[(size_t)v * DK + i + 8];
                acc[m][nt][3] = stt[(size_t)(v + 1) * DK + i + 8];
            }
    }
    auto store_h = [&](int hb) {
        bf16* hbp = Hb + hb * DK * SQ_HB_STRIDE;
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int i = warp * 32 + m * 16 + g, v = nt * 8 + cc;
                *reinterpret_cast<uint32_t*>(hbp + i * SQ_HB_STRIDE + v) = pack_bf16x2(acc[m][nt][0], acc[m][nt][1]);
                *reinterpret_cast<uint32_t*>(hbp + (i + 8) * SQ_HB_STRIDE + v) = pack_bf16x2(acc[m][nt][2], acc[m][nt][3]);
            }
    };
    // copy Hb[hb] rows (VW bf16 = 32 B each) to Hs[c][hv][i][v0..]: 128 rows x 2 x 16 B, 2 chunks per thread
    auto copy_h_out = [&](int hb, int c) {
        const bf16* hbp = Hb + hb * DK * SQ_HB_STRIDE;
        unsigned char* hsb = reinterpret_cast<unsigned char*>(Hs + ((size_t)c * HV + hv) * DK * DV);   // Hs[c][hv] as the swizzled tile image
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            int idx = tid + q * 128, row = idx >> 1, half = idx & 1;
            *reinterpret_cast<uint4*>(hsb + tile_off(row, v0 + half * 8)) = *reinterpret_cast<const uint4*>(hbp + row * SQ_HB_STRIDE + half * 8);
        }
    };
    if (nc > 0) store_h(0);
    const int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int nchunk = lane >> 4;

    for (int it = 0; it < nc; ++it) {
        const int gi = itb + it;
        const int c = c0 + it, st = gi % SQ_NST, hcur = it & 1;
        bar_consumers();                         // Hb[hcur] complete
        copy_h_out(hcur, c);                     // Hs[c] = h at the start of chunk c
        if (ready2 != nullptr && tid == 0 && it > 0) st_release_cta_shared(hs_copied, it);   // Hs[c0 .. c0+it-1] complete (ordered by the barrier above)
        if (DBG_ON && g_dbg && tid == 0 && vs == 0 && hv == 0 && n == 0 && it < 40) g_dbg[3000 + it * 4 + 0] = clock64();
        mbar_wait(smem_u32(&full[st]), (gi / SQ_NST) & 1);
        if (DBG_ON && g_dbg && tid == 0 && vs == 0 && hv == 0 && n == 0 && it < 40) g_dbg[3000 + it * 4 + 1] = clock64();
        const float gc = gvals[it];
        const unsigned char* stg = stages + st * SQ_STAGE;
        const uint32_t stg_base = smem_u32(stg);
        const unsigned char* bsl = stg + SQ_MP_BYTES;    // [128 i][VW v] bf16 rows (VW*2 B)
        uint32_t bfc[2][NT][2];                         // bf16x2 pairs: (row g, cols cc..), (row g+8, cols cc..)
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int i = warp * 32 + m * 16 + g, v = nt * 8 + cc;
                bfc[m][nt][0] = *reinterpret_cast<const uint32_t*>(bsl + i * (VW * 2) + v * 2);
                bfc[m][nt][1] = *reinterpret_cast<const uint32_t*>(bsl + (i + 8) * (VW * 2) + v * 2);
            }
        const uint32_t hb_base = smem_u32(Hb + hcur * DK * SQ_HB_STRIDE);
        float acc2[2][NT][4];
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) { acc2[m][nt][0] = acc2[m][nt][1] = acc2[m][nt][2] = acc2[m][nt][3] = 0.f; }
        uint32_t bfr[2][NT / 2][4];
        uint4 af[2][8];
        {
            const int arow = (lane & 7) + ((lane >> 3) & 1) * 8, achunk = lane >> 4;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    uint32_t* a = reinterpret_cast<uint32_t*>(&af[m][kk]);
                    ldsm_x4(a[0], a[1], a[2], a[3], stg_base + tile_off(warp * 32 + m * 16 + arow, kk * 16 + achunk * 8));
                }
            }
        }
        #pragma unroll
        for (int np = 0; np < NT / 2; ++np)
            ldsm_x4_trans(bfr[0][np][0], bfr[0][np][1], bfr[0][np][2], bfr[0][np][3], hb_base + ((0 * 16 + krow) * SQ_HB_STRIDE + (np * 2 + nchunk) * 8) * 2);
        #pragma unroll
        for (int kk = 0; kk < (SEQ_X_NOMMA ? 0 : 8); ++kk) {
            const int slot = kk & 1;
            if (kk + 1 < 8) {
                #pragma unroll
                for (int np = 0; np < NT / 2; ++np)
                    ldsm_x4_trans(bfr[slot ^ 1][np][0], bfr[slot ^ 1][np][1], bfr[slot ^ 1][np][2], bfr[slot ^ 1][np][3],
                                  hb_base + (((kk + 1) * 16 + krow) * SQ_HB_STRIDE + (np * 2 + nchunk) * 8) * 2);
            }
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const uint32_t* a = reinterpret_cast<const uint32_t*>(&af[m][kk]);
                #pragma unroll
                for (int np = 0; np < NT / 2; ++np) {
                    mma_bf16(acc2[m][np * 2], a, bfr[slot][np]);
                    mma_bf16(acc2[m][np * 2 + 1], a, bfr[slot][np] + 2);
                }
            }
        }
        if (lane == 0) mbar_arrive(smem_u32(&empty[st]));   // this warp is done reading the stage
        if (DBG_ON && g_dbg && tid == 0 && vs == 0 && hv == 0 && n == 0 && it < 40) g_dbg[3000 + it * 4 + 2] = clock64();
        {
            #pragma unroll
            for (int m = 0; m < 2; ++m)
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    float2 b0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bfc[m][nt][0]));
                    float2 b1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bfc[m][nt][1]));
                    acc[m][nt][0] = fmaf(gc, acc[m][nt][0], acc2[m][nt][0] + b0.x);
                    acc[m][nt][1] = fmaf(gc, acc[m][nt][1], acc2[m][nt][1] + b0.y);
                    acc[m][nt][2] = fmaf(gc, acc[m][nt][2], acc2[m][nt][2] + b1.x);
                    acc[m][nt][3] = fmaf(gc, acc[m][nt][3], acc2[m][nt][3] + b1.y);
                }
        }
        store_h(hcur ^ 1);
        if (DBG_ON && g_dbg && tid == 0 && vs == 0 && hv == 0 && n == 0 && it < 40) g_dbg[3000 + it * 4 + 3] = clock64();
    }
    if (ready2 != nullptr && nc > 0) {   // all Hs[c0 .. c1-1] copies complete after this barrier -> publisher warp finishes
        bar_consumers();
        if (tid == 0) st_release_cta_shared(hs_copied, nc);
    }
    {
        float* stt = new_state + ((size_t)n * HV + hv) * DV * DK;
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int i = warp * 32 + m * 16 + g, v = v0 + nt * 8 + cc;
                stt[(size_t)v * DK + i] = acc[m][nt][0];
                stt[(size_t)(v + 1) * DK + i] = acc[m][nt][1];
                stt[(size_t)v * DK + i + 8] = acc[m][nt][2];
                stt[(size_t)(v + 1) * DK + i + 8] = acc[m][nt][3];
            }
    }
}


__global__ void __launch_bounds__(224) seq_kernel3(
    const bf16* __restrict__ Mp, const bf16* __restrict__ Bp, const float* __restrict__ gchunk,
    const float* __restrict__ state, float* __restrict__ new_state, bf16* __restrict__ Hs,
    const int* __restrict__ cstart, const int* __restrict__ ready, int HV)
{
    extern __shared__ __align__(128) unsigned char sq_smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(sq_smem + SQ_NST * SQ_STAGE + 2 * DK * SQ_HB_STRIDE * 2);
    uint64_t* empty = full + SQ_NST;
    if (threadIdx.x == 0) {
        for (int i = 0; i < SQ_NST; ++i) { mbar_init(smem_u32(&full[i]), 1); mbar_init(smem_u32(&empty[i]), 4); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    float* gstage = reinterpret_cast<float*>(empty + SQ_NST);
    volatile int* hs_done = reinterpret_cast<volatile int*>(gstage + SQ_NST);
    volatile int* s_ready_upto = hs_done + 4;
    int* hs_copied = const_cast<int*>(hs_done) + 8;
    float* gvals = reinterpret_cast<float*>(sq_smem + SQ_NST * SQ_STAGE + 2 * DK * SQ_HB_STRIDE * 2 + 256);
    if (threadIdx.x == 0) { *s_ready_upto = 0; *hs_copied = 0; }
    __syncthreads();
    seq_item(sq_smem, full, empty, gstage, hs_done, s_ready_upto, gvals, 0, blockIdx.x, blockIdx.y, blockIdx.z, Mp, Bp, gchunk, state, new_state, Hs, cstart, cstart + gridDim.z + 1, ready, HV, nullptr, hs_copied);
}

// out work item for the persistent kernel: waits for Qt/Ot (ready[c] >= HV) and Hs[c][hv] (ready2 >= NVS)
__device__ __noinline__ void out_item(unsigned char* sm, uint32_t tm0, uint32_t barb, uint32_t& phase, uint32_t barld, uint32_t& phase_ld, int c, int hv,
    const bf16* __restrict__ Qt, const bf16* __restrict__ Ot, const bf16* __restrict__ Hs,
    const int* __restrict__ cinfo, bf16* __restrict__ out, int HV, const int* __restrict__ ready, const int* __restrict__ ready2)
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int len = cinfo[c * 4 + 2];
    if (len <= 0) return;
    const int tok0 = cinfo[c * 4 + 1];
    const size_t ch = (size_t)c * HV + hv;
    unsigned char* Qs = sm; unsigned char* Hsm = sm + 32768; unsigned char* Os = sm + 65536;
    // Qt / Ot only need the intra items of chunk c: fetch them while (possibly) waiting for the Hs slices of the seq items
    // Qt / Ot / Hs are stored as 32 KB swizzled tile images: 4 x 8 KB bulk copies each, one mbarrier phase for all 96 KB
    const uint32_t Qb = smem_u32(Qs), Hbb = smem_u32(Hsm), Ob = smem_u32(Os);
    if (tid == 0) {
        while (ld_acquire_gpu(ready + c) < HV) { __nanosleep(POLL_NS); }
        mbar_expect_tx(barld, 3 * 32768);
        const unsigned char* qsrc = reinterpret_cast<const unsigned char*>(Qt + ch * CH * DK);
        const unsigned char* osrc = reinterpret_cast<const unsigned char*>(Ot + ch * CH * DV);
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
            tma_bulk_g2s(Qb + qq * 8192, qsrc + qq * 8192, 8192, barld);
            tma_bulk_g2s(Ob + qq * 8192, osrc + qq * 8192, 8192, barld);
        }
        while (ld_acquire_gpu(ready2 + ch) < NVS) { __nanosleep(POLL_NS); }
        const unsigned char* hsrc = reinterpret_cast<const unsigned char*>(Hs + ch * DK * DV);
        #pragma unroll
        for (int qq = 0; qq < 4; ++qq) tma_bulk_g2s(Hbb + qq * 8192, hsrc + qq * 8192, 8192, barld);
    }
    mbar_wait(barld, phase_ld & 1); ++phase_ld;
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (tid == 0) issue_mma(tm0, Qb, 0, Hbb, 1, 0, 0, barb);
    mbar_wait(barb, phase & 1); ++phase;
    tc_fence_after();
    const int t = 32 * (warp & 3) + lane;
    const int c0 = (warp >> 2) * 64;
    const uint32_t tlane = (uint32_t)(32 * (warp & 3)) << 16;
    uint32_t rr[64];
    tc_ld64(tm0 + tlane + c0, rr);
    // epilogue: o = acc + Ot written back in place into the Ot tile (smem), then a coalesced copy-out (256 B per row)
    if (t < len) {
        #pragma unroll
        for (int q8 = 0; q8 < 8; ++q8) {
            uint4* oaddr = reinterpret_cast<uint4*>(Os + tile_off(t, c0 + q8 * 8));
            uint4 ov = *oaddr;
            const uint32_t* op = reinterpret_cast<const uint32_t*>(&ov);
            uint4 pk; uint32_t* pp = reinterpret_cast<uint32_t*>(&pk);
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float2 of = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&op[e]));
                pp[e] = pack_bf16x2(__uint_as_float(rr[q8 * 8 + e * 2]) + of.x, __uint_as_float(rr[q8 * 8 + e * 2 + 1]) + of.y);
            }
            *oaddr = pk;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int idx = tid + p * 256, row = idx >> 4, seg = idx & 15;
        if (row < len)
            *reinterpret_cast<uint4*>(out + ((size_t)(tok0 + row) * HV + hv) * DV + seg * 8) = *reinterpret_cast<const uint4*>(Os + tile_off(row, seg * 8));
    }
    tc_fence_before();
    __syncthreads();
}

// ---------------- out kernel (tcgen05): o[t][v] = Qt[t][:] . Hs[:][v] + Ot[t][v]   per (128-chunk, v-head)
#define OUT_SMEM (3 * 32768 + 1024 + 64)
__global__ void __launch_bounds__(128, 2) out_kernel_tc(
    const bf16* __restrict__ Qt, const bf16* __restrict__ Ot, const bf16* __restrict__ Hs,
    const int* __restrict__ cinfo, bf16* __restrict__ out, int HV)
{
    extern __shared__ __align__(1024) unsigned char osm_raw[];
    unsigned char* sm = (unsigned char*)(((uintptr_t)osm_raw + 1023) & ~(uintptr_t)1023);
    unsigned char* Qs = sm; unsigned char* Hsm = sm + 32768; unsigned char* Os = sm + 65536;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(sm + 98304);
    uint32_t* tmem_holder = reinterpret_cast<uint32_t*>(sm + 98304 + 8);
    const int c = blockIdx.x, hv = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int len = cinfo[c * 4 + 2];
    if (len <= 0) return;
    const int tok0 = cinfo[c * 4 + 1];
    const size_t ch = (size_t)c * HV + hv;
    const uint32_t Qb = smem_u32(Qs), Hbb = smem_u32(Hsm);
    #pragma unroll
    for (int qq = 0; qq < 16; ++qq) {
        int i = tid + qq * 128, row = i >> 4, cidx = i & 15;
        cp_async16_(Qb + i * 16, reinterpret_cast<const unsigned char*>(Qt + ch * CH * DK) + i * 16);       // tile images: linear copy
        cp_async16_(Hbb + i * 16, reinterpret_cast<const unsigned char*>(Hs + ch * DK * DV) + i * 16);
        cp_async16_(smem_u32(Os) + i * 16, reinterpret_cast<const unsigned char*>(Ot + ch * CH * DV) + i * 16);
        (void)row; (void)cidx;
    }
    cp_async_commit_();
    if (tid == 0) { mbar_init(smem_u32(mbar), 1); asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(tmem_holder)), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    cp_async_wait_<0>();
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tm0 = *tmem_holder;
    if (tid == 0) issue_mma(tm0, Qb, 0, Hbb, 1, 0, 0, smem_u32(mbar));
    mbar_wait(smem_u32(mbar), 0);
    tc_fence_after();
    const int t = warp * 32 + lane;
    const uint32_t tlane = (uint32_t)(32 * warp) << 16;
    bf16* orow = out + ((size_t)(tok0 + t) * HV + hv) * DV;
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const int c0 = hh * 32;
        uint32_t r[32];
        tc_ld32(tm0 + tlane + c0, r);
        if (t < len) {
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 ov = *reinterpret_cast<const uint4*>(Os + tile_off(t, c0 + q8 * 8));
                const uint32_t* op = reinterpret_cast<const uint32_t*>(&ov);
                uint4 pk; uint32_t* pp = reinterpret_cast<uint32_t*>(&pk);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float2 of = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&op[e]));
                    pp[e] = pack_bf16x2(__uint_as_float(r[q8 * 8 + e * 2]) + of.x, __uint_as_float(r[q8 * 8 + e * 2 + 1]) + of.y);
                }
                *reinterpret_cast<uint4*>(orow + c0 + q8 * 8) = pk;
            }
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tm0), "r"(128));
}



#ifndef IT_MINB
#define IT_MINB 1
#endif
#ifndef IT_STOP_AFTER
#define IT_STOP_AFTER 99
#endif
#define IT_STOP(n)
__global__ void __launch_bounds__(256, IT_MINB) intra_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const bf16* __restrict__ a, const bf16* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cinfo, int HQK, int HV, float scale, int nitems, int* __restrict__ counter,
    bf16* __restrict__ Mp, bf16* __restrict__ Bp, bf16* __restrict__ Qt, bf16* __restrict__ Ot, float* __restrict__ gchunk,
    int* __restrict__ ready,
    const float* __restrict__ state, float* __restrict__ new_state, bf16* __restrict__ Hs, const int* __restrict__ cstart128, int nseq,
    int* __restrict__ ready2, bf16* __restrict__ out, int nout, const int* __restrict__ seq_order, int Ksplit)
{
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* sm = (unsigned char*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
    __shared__ int s_item;
    __shared__ float rowbuf[8][2][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    unsigned char* Ktiles[2] = { sm + IT_OFF_K, sm + IT_OFF_K2 };
    unsigned char* Tsm = sm + IT_OFF_T;
    unsigned char* WV = sm + IT_OFF_WV;
    unsigned char* QA = sm + IT_OFF_QA;
    unsigned char* Vsm = sm + IT_OFF_V;
    unsigned char* Xs = sm + IT_OFF_X;
    float* sG = reinterpret_cast<float*>(sm + IT_OFF_G);
    float* sBeta = sG + 128; float* sEG = sG + 256; float* sBG = sG + 384; float* sDL = sG + 512;
    float* Ad = reinterpret_cast<float*>(sm + IT_AD_OFF);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(sm + IT_OFF_MISC);      // [0..3]: tm0..tm3 MMAs
    uint32_t* tmem_holder = reinterpret_cast<uint32_t*>(sm + IT_OFF_MISC + 32);
    const uint32_t Tb = smem_u32(Tsm), WVb = smem_u32(WV), QAb = smem_u32(QA), Vb = smem_u32(Vsm);
    int kcur = 0;            // which K tile holds the current item's K
    int prefetched_item = -1;
    // precomputed k-step-0 descriptors (K-major: rows = M/N ; MN-major: rows = K)
    const uint64_t dK_k[2] = { desc_kmajor(smem_u32(Ktiles[0]), 0), desc_kmajor(smem_u32(Ktiles[1]), 0) };
    const uint64_t dK_mn[2] = { desc_mnmajor(smem_u32(Ktiles[0]), 0), desc_mnmajor(smem_u32(Ktiles[1]), 0) };
    const uint64_t dT_k = desc_kmajor(Tb, 0), dT_mn = desc_mnmajor(Tb, 0);
    const uint64_t dWV_mn = desc_mnmajor(WVb, 0);
    const uint64_t dQA_k = desc_kmajor(QAb, 0);
    const uint64_t dV_mn = desc_mnmajor(Vb, 0);
    const uint32_t barb0 = smem_u32(mbar), barb1 = smem_u32(mbar + 1), barb2 = smem_u32(mbar + 2), barb3 = smem_u32(mbar + 3);

    const uint32_t barld = smem_u32(mbar + 5);     // out-item bulk-load barrier (misc @40)
    // seq-item stage barriers (misc @64: full[NST], empty[NST]) are initialised ONCE; their phases continue across the CTA's
    // seq items (mbarrier arrivals complete asynchronously: re-initialising per item raced with late arrivals).
    uint64_t* sq_full_init = reinterpret_cast<uint64_t*>(sm + IT_OFF_MISC + 64);
    if (tid == 0) {
        mbar_init(barb0, 1); mbar_init(barb1, 1); mbar_init(barb2, 1); mbar_init(barb3, 1); mbar_init(barld, 1);
        for (int i = 0; i < SQ_NST; ++i) { mbar_init(smem_u32(&sq_full_init[i]), 1); mbar_init(smem_u32(&sq_full_init[SQ_NST + i]), 4); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    int seq_itb = 0;   // total seq iterations processed by this CTA (stage/phase continuity)
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(smem_u32(tmem_holder)), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tm0 = *tmem_holder, tm1 = tm0 + 128, tm2 = tm0 + 256, tm3 = tm0 + 384;
    uint32_t phase0 = 0, phase1 = 0, phase2 = 0, phase3 = 0, phase_ld = 0;
    // epilogue mapping: warp w -> TMEM lanes 32*(w&3).. (row = 32*(w&3) + lane), column half (w>>2)*64
    const int erow = 32 * (warp & 3) + lane;
    const int ecol0 = (warp >> 2) * 64;
    const uint32_t tlane = (uint32_t)(32 * (warp & 3)) << 16;

    auto wait0 = [&]() { mbar_wait(barb0, phase0 & 1); ++phase0; tc_fence_after(); };
    auto wait1 = [&]() { mbar_wait(barb1, phase1 & 1); ++phase1; tc_fence_after(); };
    auto wait2 = [&]() { mbar_wait(barb2, phase2 & 1); ++phase2; tc_fence_after(); };
    auto wait3 = [&]() { mbar_wait(barb3, phase3 & 1); ++phase3; tc_fence_after(); };

    __shared__ int s_next, s_nlen, s_ntok0;
    // work queue layout (global index g): [intra 0..Ksplit) [early seq (n_early)] [intra Ksplit..nitems) [late seq] [out]
    const int n_early = min(counter[1], nseq);
    const int total_items = nitems + nseq + nout;
    // decode: returns kind (0 intra, 1 seq, 2 out, 3 end) and index
    auto decode = [&](int g, int& idx) -> int {
        if (g < Ksplit) { idx = g; return 0; }
        g -= Ksplit;
        if (g < n_early) { idx = g; return 1; }
        g -= n_early;
        if (g < nitems - Ksplit) { idx = Ksplit + g; return 0; }
        g -= (nitems - Ksplit);
        if (g < nseq - n_early) { idx = n_early + g; return 1; }
        g -= (nseq - n_early);
        if (g < nout) { idx = g; return 2; }
        return 3;
    };
    if (tid == 0) s_item = atomicAdd(counter, 1);
    __syncthreads();
    int gcur = s_item;
    int cur_len = -2, cur_tok0 = 0;    // -2: unknown, load now
    int gp_item = -1; float gp_ea = 0.f, gp_dtb = 0.f; bf16 gp_a[4], gp_b[4];   // gate inputs prefetched for the next item (warp 0)
    uint64_t* sq_full = reinterpret_cast<uint64_t*>(sm + IT_OFF_MISC + 64);
    uint64_t* sq_empty = sq_full + SQ_NST;
  for (;;) {
    int item;
    const int kind = decode(gcur, item);
    if (kind == 3) break;
    if (kind != 0) {
        // ---------------- seq / out work item
        if (DBG_ON && g_dbg && tid == 0) { g_dbg[1024 + 2 * gcur] = gtimer(); g_dbg[8000 + gcur] = kind; }
        if (kind == 2 && tid == 0) s_item = atomicAdd(counter, 1);   // one-ahead dequeue (short items only)
        if (kind == 1) {
            const int code = seq_order[item];
            const int svs = code % NVS, shv = (code / NVS) % HV, sn = code / (NVS * HV);
            volatile int* hs_done = reinterpret_cast<volatile int*>(reinterpret_cast<float*>(sq_empty + SQ_NST) + SQ_NST);
            volatile int* s_ready_upto = hs_done + 4;
            int* hs_copied = const_cast<int*>(hs_done) + 8;
            float* gvals = reinterpret_cast<float*>(sm + IT_OFF_MISC + IT_GVALS_OFF);
            if (tid == 0) { *s_ready_upto = 0; *hs_copied = 0; }
            __syncthreads();
            seq_item(sm, sq_full, sq_empty, reinterpret_cast<float*>(sq_empty + SQ_NST), hs_done, s_ready_upto, gvals, seq_itb, svs, shv, sn, Mp, Bp, gchunk, state, new_state, Hs, cstart128, cstart128 + nseq / (NVS * HV) + 1, ready, HV, ready2, hs_copied);
        } else {
#ifdef OUT_AFTER_SEQ
            if (tid == 0) { while (ld_acquire_gpu(counter + 2) < nseq) __nanosleep(256); }
            __syncthreads();
#endif
            out_item(sm, tm0, barb0, phase0, barld, phase_ld, item / HV, item % HV, Qt, Ot, Hs, cinfo, out, HV, ready, ready2);
        }
        __syncthreads();
        if (kind == 1) { const int sn_ = seq_order[item] / (NVS * HV); seq_itb += cstart128[nseq / (NVS * HV) + 1 + sn_] - cstart128[sn_]; }
        if (kind == 1 && tid == 0) atomicAdd(counter + 2, 1);
        if (DBG_ON && g_dbg && tid == 0) g_dbg[1025 + 2 * gcur] = gtimer();
        prefetched_item = -1; cur_len = -2; gp_item = -1;      // smem reused: any K prefetch is void
        if (kind == 1 && tid == 0) s_item = atomicAdd(counter, 1);   // out items dequeued their successor at item start
        __syncthreads();
        gcur = s_item;
        continue;
    }
    if (tid == 0) s_next = atomicAdd(counter, 1);   // one-ahead
    const int c = item / HV, hv = item % HV;
    const int len = (cur_len == -2) ? cinfo[c * 4 + 2] : cur_len;
    if (len <= 0) { __syncthreads(); gcur = s_next; cur_len = -2; __syncthreads(); continue; }
    const int tok0 = (cur_len == -2) ? cinfo[c * 4 + 1] : cur_tok0;
    const int hq = hv / (HV / HQK);
    if (DBG_ON && g_dbg && tid == 0) { g_dbg[1024 + 2 * gcur] = gtimer(); g_dbg[8000 + gcur] = 0; }
    DBG_REC(13);
    const size_t ch = (size_t)c * HV + hv;

    // ---------------- P0: K (prefetched into Ktiles[kcur] by the previous item, else load now); Q -> QA, V -> Vs; zero T; gates
    unsigned char* Ksm = Ktiles[kcur];
    const uint32_t Kb = smem_u32(Ksm);
    {
        if (prefetched_item != item) {
            #pragma unroll
            for (int qq = 0; qq < 8; ++qq) {
                int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
                if (row < len) cp_async16_(Kb + tile_off(row, cidx * 8), k + ((size_t)(tok0 + row) * HQK + hq) * DK + cidx * 8);
                else *reinterpret_cast<uint4*>(Ksm + tile_off(row, cidx * 8)) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit_();
        #pragma unroll
        for (int qq = 0; qq < 8; ++qq) {
            int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
            if (row < len) cp_async16_(QAb + tile_off(row, cidx * 8), q + ((size_t)(tok0 + row) * HQK + hq) * DK + cidx * 8);
            else *reinterpret_cast<uint4*>(QA + tile_off(row, cidx * 8)) = make_uint4(0, 0, 0, 0);
            if (row < len) cp_async16_(Vb + tile_off(row, cidx * 8), v + ((size_t)(tok0 + row) * HV + hv) * DV + cidx * 8);
            else *reinterpret_cast<uint4*>(Vsm + tile_off(row, cidx * 8)) = make_uint4(0, 0, 0, 0);
        }
        cp_async_commit_();
        #pragma unroll
        for (int qq = 0; qq < 8; ++qq) reinterpret_cast<uint4*>(Tsm)[tid + qq * 256] = make_uint4(0, 0, 0, 0);
        if (warp == 0) {
            if (gp_item != item) {     // not prefetched: load now
                gp_ea = __expf(A_log[hv]); gp_dtb = dt_bias[hv];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    int t = lane * 4 + u;
                    gp_a[u] = (t < len) ? a[(size_t)(tok0 + t) * HV + hv] : __float2bfloat16(0.f);
                    gp_b[u] = (t < len) ? b[(size_t)(tok0 + t) * HV + hv] : __float2bfloat16(0.f);
                }
            }
            const float ea = gp_ea, dtb = gp_dtb;
            float lg[4], be[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int t = lane * 4 + u;
                if (t < len) {
                    float av = __bfloat162float(gp_a[u]) + dtb;
                    float bv = __bfloat162float(gp_b[u]);
                    lg[u] = -ea * softplus_f(av);
                    be[u] = 1.f / (1.f + __expf(-bv));
                } else { lg[u] = 0.f; be[u] = 0.f; }
            }
            float loc = lg[0] + lg[1] + lg[2] + lg[3];
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { float vv = __shfl_up_sync(0xffffffffu, loc, o); if (lane >= o) loc += vv; }
            float gl = __shfl_sync(0xffffffffu, loc, 31);
            float run = loc - lg[3] - lg[2] - lg[1];
            float Gt[4];
            Gt[0] = run; Gt[1] = run + lg[1]; Gt[2] = Gt[1] + lg[2]; Gt[3] = Gt[2] + lg[3];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int t = lane * 4 + u;
                float eg = __expf(Gt[u]);
                sG[t] = Gt[u]; sBeta[t] = be[u]; sEG[t] = eg; sBG[t] = be[u] * eg; sDL[t] = __expf(gl - Gt[u]);
            }
            if (lane == 0) gchunk[ch] = __expf(gl);
        }
        cp_async_wait_<1>();      // K landed (this thread's copies); Q/V group may still be in flight
        fence_async_smem();
        __syncthreads();
        DBG_REC(0);
    }
    // ---------------- P1: S = K K^T -> tm0 ; epilogue builds A into WV, Ad   (Sqk issued after Phase D)
    if (tid == 0) issue_mma_d(tm0, dK_k[kcur], 0, dK_k[kcur], 0, 0, 0, barb0);
    DBG_REC(11);
    wait0();
    DBG_REC(12);
    if (IT_STOP_AFTER > 1) {
        const int t = erow;
        const float bt = sBeta[t], gt = sG[t];
        uint32_t rr[64];
        tc_ld64(tm0 + tlane + ecol0, rr);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            if (32 * (warp & 3) + 31 < c0) continue;    // whole warp-slice is above the diagonal: never read
            const uint32_t* r = rr + hh * 32;
            float av[32];
            if (32 * (warp & 3) >= c0 + 32) {           // all rows of this warp are below the slice: all 32 entries valid
                // e^{G_t - G_s} = e^{G_t - G_ref} * e^{G_ref - G_s}, G_ref = G[c0+31]; both exponents <= 0 (no overflow)
                const float gref = sG[c0 + 31];
                const float ft = bt * __expf(gt - gref);
                const float Ej = __expf(gref - sG[c0 + lane]);
                #pragma unroll
                for (int e = 0; e < 32; ++e) av[e] = ft * __shfl_sync(0xffffffffu, Ej, e) * __uint_as_float(r[e]);
            } else {
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    int s = c0 + e;
                    av[e] = (s < t) ? bt * __expf(gt - sG[s]) * __uint_as_float(r[e]) : 0.f;
                }
            }
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 pk;
                pk.x = pack_bf16x2(av[q8 * 8 + 0], av[q8 * 8 + 1]); pk.y = pack_bf16x2(av[q8 * 8 + 2], av[q8 * 8 + 3]);
                pk.z = pack_bf16x2(av[q8 * 8 + 4], av[q8 * 8 + 5]); pk.w = pack_bf16x2(av[q8 * 8 + 6], av[q8 * 8 + 7]);
                *reinterpret_cast<uint4*>(WV + tile_off(t, c0 + q8 * 8)) = pk;
            }
            if ((c0 >> 4) == (t >> 4)) {          // diagonal 16-block = first half of this 32-col slice
                float* ad = Ad + (t >> 4) * 256 + (t & 15) * 16;
                #pragma unroll
                for (int e = 0; e < 16; e += 4)
                    *reinterpret_cast<float4*>(ad + e) = make_float4(av[e], av[e + 1], av[e + 2], av[e + 3]);
            } else if (((c0 + 16) >> 4) == (t >> 4)) {   // second half
                float* ad = Ad + (t >> 4) * 256 + (t & 15) * 16;
                #pragma unroll
                for (int e = 0; e < 16; e += 4)
                    *reinterpret_cast<float4*>(ad + e) = make_float4(av[16 + e], av[16 + e + 1], av[16 + e + 2], av[16 + e + 3]);
            }
        }
    }
    __syncthreads();      // s_next visible; A-build done
    DBG_REC(1);
    // prefetch the next item's K into the other K tile (previous item's K, dead)
    const int next_g = s_next;
    int next_item = -1;
    const bool next_is_intra = (decode(next_g, next_item) == 0);
    int nlen = 0, ntok0 = 0;
    if (next_is_intra) {
        const int nc_ = next_item / HV, nhv = next_item % HV;
        nlen = cinfo[nc_ * 4 + 2];
        ntok0 = cinfo[nc_ * 4 + 1];
        if (nlen > 0) {
            const int nhq = nhv / (HV / HQK);
            unsigned char* Kn = Ktiles[kcur ^ 1];
            const uint32_t Knb = smem_u32(Kn);
            #pragma unroll
            for (int qq = 0; qq < 8; ++qq) {
                int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
                if (row < nlen) cp_async16_(Knb + tile_off(row, cidx * 8), k + ((size_t)(ntok0 + row) * HQK + nhq) * DK + cidx * 8);
                else *reinterpret_cast<uint4*>(Kn + tile_off(row, cidx * 8)) = make_uint4(0, 0, 0, 0);
            }
            cp_async_commit_();
            prefetched_item = next_item;
        }
        if (warp == 0 && nlen > 0) {
            gp_ea = __expf(A_log[nhv]); gp_dtb = dt_bias[nhv];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int t = lane * 4 + u;
                gp_a[u] = (t < nlen) ? a[(size_t)(ntok0 + t) * HV + nhv] : __float2bfloat16(0.f);
                gp_b[u] = (t < nlen) ? b[(size_t)(ntok0 + t) * HV + nhv] : __float2bfloat16(0.f);
            }
            gp_item = next_item;
        }
    }

    // ---------------- P2: Phase D: T_bb = (I + A_bb)^-1 = (I - A)(I + A^2)(I + A^4)(I + A^8) (exact: A nilpotent), tf32 mma
    if (IT_STOP_AFTER > 2) {
        const float* ad = Ad + warp * 256;                          // A block (fp32 16x16)
        float* scr = reinterpret_cast<float*>(Xs) + warp * 512;     // per-warp scratch: P (256 floats), Apow (256 floats)
        float* P = scr; float* Ap = scr + 256;
        float acc[2][4];
        // Ap = A*A ; P = I - A
        zero_acc2(acc); mm16_tf32(acc, ad, ad, lane);
        {
            const int g = lane >> 2, c2 = (lane & 3) * 2;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                int cc = nt * 8 + c2;
                float p00 = (g == cc ? 1.f : 0.f) - ad[g * 16 + cc], p01 = (g == cc + 1 ? 1.f : 0.f) - ad[g * 16 + cc + 1];
                float p10 = (g + 8 == cc ? 1.f : 0.f) - ad[(g + 8) * 16 + cc], p11 = (g + 8 == cc + 1 ? 1.f : 0.f) - ad[(g + 8) * 16 + cc + 1];
                *reinterpret_cast<float2*>(P + g * 16 + cc) = make_float2(p00, p01);
                *reinterpret_cast<float2*>(P + (g + 8) * 16 + cc) = make_float2(p10, p11);
            }
        }
        __syncwarp();
        st16(Ap, acc, lane);            // A^2
        __syncwarp();
        #pragma unroll
        for (int it2 = 0; it2 < 3; ++it2) {   // P <- P + P*Ap ; Ap <- Ap*Ap (last squaring unnecessary but harmless)
            // acc = P * Ap (init acc with P)
            {
                const int g = lane >> 2, c2 = (lane & 3) * 2;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    float2 p0 = *reinterpret_cast<const float2*>(P + g * 16 + nt * 8 + c2);
                    float2 p1 = *reinterpret_cast<const float2*>(P + (g + 8) * 16 + nt * 8 + c2);
                    acc[nt][0] = p0.x; acc[nt][1] = p0.y; acc[nt][2] = p1.x; acc[nt][3] = p1.y;
                }
            }
            mm16_tf32(acc, P, Ap, lane);
            float acc2[2][4];
            zero_acc2(acc2);
            if (it2 < 2) mm16_tf32(acc2, Ap, Ap, lane);
            __syncwarp();
            st16(P, acc, lane);
            if (it2 < 2) st16(Ap, acc2, lane);
            __syncwarp();
        }
        // write T - I (strictly lower part of P) into Ts block (warp, warp)
        {
            const int g = lane >> 2, c2 = (lane & 3) * 2;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                int cc = nt * 8 + c2;
                float2 p0 = *reinterpret_cast<const float2*>(P + g * 16 + cc);
                float2 p1 = *reinterpret_cast<const float2*>(P + (g + 8) * 16 + cc);
                st_bf16x2(Tsm, tile_off(warp * 16 + g, warp * 16 + cc), (g > cc) ? p0.x : 0.f, (g > cc + 1) ? p0.y : 0.f);
                st_bf16x2(Tsm, tile_off(warp * 16 + g + 8, warp * 16 + cc), (g + 8 > cc) ? p1.x : 0.f, (g + 8 > cc + 1) ? p1.y : 0.f);
            }
        }
    }
    cp_async_wait_<0>();          // Q, V landed
    fence_async_smem();
    __syncthreads();
    if (tid == 224) issue_mma_d(tm1, dQA_k, 0, dK_k[kcur], 0, 0, 0, barb1);   // Sqk = Q K^T -> tm1 (overlaps Phase O; warp 7 idle there)
    DBG_REC(2);
    IT_STOP(2);
    if (IT_STOP_AFTER > 3) {   // Phase O within each 64-block; then T21
        #pragma unroll 1
        for (int bi = 1; bi < 4; ++bi) {
            const int bb = warp / 3, bj = warp % 3;
            const bool active = (warp < 6) && (bj < bi);
            float acc[2][4];
            if (active) {
                const int rowb = bb * 64 + bi * 16, colb = bb * 64 + bj * 16;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    float2 v0 = ld_bf16x2(WV, tile_off(rowb + g, colb + nt * 8 + c2));
                    float2 v1 = ld_bf16x2(WV, tile_off(rowb + g + 8, colb + nt * 8 + c2));
                    acc[nt][0] = v0.x; acc[nt][1] = v0.y; acc[nt][2] = v1.x; acc[nt][3] = v1.y;
                }
                for (int bm = bj; bm < bi; ++bm) {
                    uint32_t af[4], bf[4];
                    ldA(af, WVb, rowb, bb * 64 + bm * 16, lane);
                    ldB_trans(bf, Tb, colb, bb * 64 + bm * 16, lane);
                    mma_bf16(acc[0], af, bf);
                    mma_bf16(acc[1], af, bf + 2);
                }
                unsigned char* xw = Xs + warp * 512;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    *reinterpret_cast<uint32_t*>(xw + g * 32 + (nt * 8 + c2) * 2) = pack_bf16x2(acc[nt][0], acc[nt][1]);
                    *reinterpret_cast<uint32_t*>(xw + (g + 8) * 32 + (nt * 8 + c2) * 2) = pack_bf16x2(acc[nt][2], acc[nt][3]);
                }
                __syncwarp();
                uint32_t af[4], bf[4];
                ldA(af, Tb, rowb, rowb, lane);
                {
                    int r = (lane & 7) + ((lane >> 3) & 1) * 8, cidx = (lane >> 4);
                    ldsm_x4_trans(bf[0], bf[1], bf[2], bf[3], smem_u32(xw) + r * 32 + cidx * 16);
                }
                mma_bf16(acc[0], af, bf);
                mma_bf16(acc[1], af, bf + 2);
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    st_bf16x2(Tsm, tile_off(rowb + g, colb + nt * 8 + c2), -acc[nt][0], -acc[nt][1]);
                    st_bf16x2(Tsm, tile_off(rowb + g + 8, colb + nt * 8 + c2), -acc[nt][2], -acc[nt][3]);
                }
            }
            __syncthreads();
        }
        {   // T21 (rows 64..127, cols 0..63): X = A21 + A21 T1off ; T21 = -(X + T2off X)
            const int r0 = 64 + (warp & 3) * 16, n0 = (warp >> 2) * 32;
            float acc[4][4];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                float2 v0 = ld_bf16x2(WV, tile_off(r0 + g, n0 + nt * 8 + c2));
                float2 v1 = ld_bf16x2(WV, tile_off(r0 + g + 8, n0 + nt * 8 + c2));
                acc[nt][0] = v0.x; acc[nt][1] = v0.y; acc[nt][2] = v1.x; acc[nt][3] = v1.y;
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t af[4];
                ldA(af, WVb, r0, kk * 16, lane);
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t bf[4];
                    ldB_trans(bf, Tb, n0 + np * 16, kk * 16, lane);
                    mma_bf16(acc[np * 2], af, bf);
                    mma_bf16(acc[np * 2 + 1], af, bf + 2);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                st_bf16x2(Xs, tile_off(r0 - 64 + g, n0 + nt * 8 + c2), acc[nt][0], acc[nt][1]);
                st_bf16x2(Xs, tile_off(r0 - 64 + g + 8, n0 + nt * 8 + c2), acc[nt][2], acc[nt][3]);
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t af[4];
                ldA(af, Tb, r0, 64 + kk * 16, lane);
                #pragma unroll
                for (int np = 0; np < 2; ++np) {
                    uint32_t bf[4];
                    ldB_trans(bf, smem_u32(Xs), n0 + np * 16, kk * 16, lane);
                    mma_bf16(acc[np * 2], af, bf);
                    mma_bf16(acc[np * 2 + 1], af, bf + 2);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                st_bf16x2(Tsm, tile_off(r0 + g, n0 + nt * 8 + c2), -acc[nt][0], -acc[nt][1]);
                st_bf16x2(Tsm, tile_off(r0 + g + 8, n0 + nt * 8 + c2), -acc[nt][2], -acc[nt][3]);
            }
            #pragma unroll
            for (int qq = 0; qq < 2; ++qq) reinterpret_cast<uint4*>(sm + IT_AD_OFF)[tid + qq * 256] = make_uint4(0, 0, 0, 0);
        }
    }
    // ---------------- P3: Tb = (T-I) diag(beta) in place ; Kg = bf16(eG[t] K[t][:]) -> WV ;
    //                     W = Tb Kg -> tm0 and U = Tb V -> tm2 issued together ; W epilogue (+ bg K) -> WV
    if (IT_STOP_AFTER > 4) {
        __syncthreads();      // T block (0,1) zeroing (end of T21) must be complete before scaling T in place
        #pragma unroll
        for (int qq = 0; qq < 8; ++qq) {
            int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
            uint4* p = reinterpret_cast<uint4*>(Tsm + tile_off(row, cidx * 8));
            uint4 val = *p;
            const float4 b0 = *reinterpret_cast<const float4*>(sBeta + cidx * 8), b1 = *reinterpret_cast<const float4*>(sBeta + cidx * 8 + 4);
            uint32_t* w = reinterpret_cast<uint32_t*>(&val);
            w[0] = scale_pair(w[0], b0.x, b0.y); w[1] = scale_pair(w[1], b0.z, b0.w);
            w[2] = scale_pair(w[2], b1.x, b1.y); w[3] = scale_pair(w[3], b1.z, b1.w);
            *p = val;
            uint32_t offk = tile_off(row, cidx * 8);
            uint4 kv = *reinterpret_cast<const uint4*>(Ksm + offk);
            const uint32_t s2 = pack_bf16x2(sEG[row], sEG[row]);
            uint32_t* kw = reinterpret_cast<uint32_t*>(&kv);
            #pragma unroll
            for (int e = 0; e < 4; ++e) kw[e] = hmul2_bf16(kw[e], s2);
            *reinterpret_cast<uint4*>(WV + offk) = kv;
        }
        fence_async_smem();
        __syncthreads();
        DBG_REC(3);
        if (tid == 0) issue_mma_d(tm0, dT_k, 0, dWV_mn, 1, 0, 0, barb0);     // W = Tb Kg
        if (tid == 32) issue_mma_d(tm2, dT_k, 0, dV_mn, 1, 0, 0, barb2);     // U = Tb V
        wait0();
        const int t = erow;
        const float bgt = sBG[t];
        uint32_t wpk[2][16];
        uint32_t rr[64];
        tc_ld64(tm0 + tlane + ecol0, rr);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            const uint32_t* r = rr + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 kv = *reinterpret_cast<const uint4*>(Ksm + tile_off(t, c0 + q8 * 8));
                const uint32_t* kp = reinterpret_cast<const uint32_t*>(&kv);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float2 kf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kp[e]));
                    wpk[hh][q8 * 4 + e] = pack_bf16x2(__uint_as_float(r[q8 * 8 + e * 2]) + bgt * kf.x, __uint_as_float(r[q8 * 8 + e * 2 + 1]) + bgt * kf.y);
                }
            }
        }
        __syncthreads();   // W MMA done (Kg consumed) -> WV may be overwritten
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8)
                *reinterpret_cast<uint4*>(WV + tile_off(t, c0 + q8 * 8)) = make_uint4(wpk[hh][q8 * 4], wpk[hh][q8 * 4 + 1], wpk[hh][q8 * 4 + 2], wpk[hh][q8 * 4 + 3]);
        }
        fence_async_smem();
        __syncthreads();
        DBG_REC(4);
    }
    if (IT_STOP_AFTER > 2 && IT_STOP_AFTER <= 5) { wait1(); wait2(); }
    // ---------------- P4: (Sqk in tm1) ; Qbar -> tm0 (tcgen05.st) ; A_att (bf16) -> QA
    if (IT_STOP_AFTER > 5) {
        const int t = erow;
        const float e_t = scale * sEG[t];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            uint32_t r[32];
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 qv = *reinterpret_cast<const uint4*>(QA + tile_off(t, c0 + q8 * 8));
                const uint32_t* qp = reinterpret_cast<const uint32_t*>(&qv);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float2 qf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qp[e]));
                    r[q8 * 8 + e * 2] = __float_as_uint(e_t * qf.x);
                    r[q8 * 8 + e * 2 + 1] = __float_as_uint(e_t * qf.y);
                }
            }
            tc_st32(tm0 + tlane + c0, r);
        }
        tc_wait_st();
        wait1();
        const float gt = sG[t];
        uint32_t apk[2][16];
        uint32_t rr[64];
        tc_ld64(tm1 + tlane + ecol0, rr);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            const uint32_t* r = rr + hh * 32;
            if (32 * (warp & 3) + 31 < c0) {
                #pragma unroll
                for (int e = 0; e < 16; ++e) apk[hh][e] = 0u;
                continue;
            }
            if (32 * (warp & 3) >= c0 + 32) {           // all valid: split exponent (see A-build)
                const float gref = sG[c0 + 31];
                const float ft = scale * __expf(gt - gref);
                const float Ej = __expf(gref - sG[c0 + lane]);
                #pragma unroll
                for (int e = 0; e < 32; e += 2) {
                    float a0 = ft * __shfl_sync(0xffffffffu, Ej, e) * __uint_as_float(r[e]);
                    float a1 = ft * __shfl_sync(0xffffffffu, Ej, e + 1) * __uint_as_float(r[e + 1]);
                    apk[hh][e >> 1] = pack_bf16x2(a0, a1);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 32; e += 2) {
                    int s = c0 + e;
                    float a0 = (s <= t) ? scale * __expf(gt - sG[s]) * __uint_as_float(r[e]) : 0.f;
                    float a1 = (s + 1 <= t) ? scale * __expf(gt - sG[s + 1]) * __uint_as_float(r[e + 1]) : 0.f;
                    apk[hh][e >> 1] = pack_bf16x2(a0, a1);
                }
            }
        }
        tc_fence_before();
        __syncthreads();   // everyone finished reading Q (for Qbar) and tm1
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8)
                *reinterpret_cast<uint4*>(QA + tile_off(t, c0 + q8 * 8)) = make_uint4(apk[hh][q8 * 4], apk[hh][q8 * 4 + 1], apk[hh][q8 * 4 + 2], apk[hh][q8 * 4 + 3]);
        }
        fence_async_smem();
        __syncthreads();
        DBG_REC(5);
    }
    // ---------------- P5: issue Qt = Qbar - A W (tm0) ; U epilogue (U -> Vs, Ud -> Ts) ; issue Ot (tm3) and B (tm2)
    if (IT_STOP_AFTER > 6) {
        if (tid == 0) issue_mma_d(tm0, dQA_k, 0, dWV_mn, 1, 1, 1, barb0);
        wait2();                      // U done (long ago)
        const int t = erow;
        const float bt = sBeta[t], dl = sDL[t];
        uint32_t upk[2][16], udpk[2][16];
        uint32_t rr[64];
        tc_ld64(tm2 + tlane + ecol0, rr);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            const uint32_t* r = rr + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 vv = *reinterpret_cast<const uint4*>(Vsm + tile_off(t, c0 + q8 * 8));
                const uint32_t* vp = reinterpret_cast<const uint32_t*>(&vv);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float2 vf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vp[e]));
                    float u0 = __uint_as_float(r[q8 * 8 + e * 2]) + bt * vf.x, u1 = __uint_as_float(r[q8 * 8 + e * 2 + 1]) + bt * vf.y;
                    upk[hh][q8 * 4 + e] = pack_bf16x2(u0, u1);
                    udpk[hh][q8 * 4 + e] = pack_bf16x2(dl * u0, dl * u1);
                }
            }
        }
        __syncthreads();   // all V reads done; T (A operand of W/U) fully consumed
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                *reinterpret_cast<uint4*>(Vsm + tile_off(t, c0 + q8 * 8)) = make_uint4(upk[hh][q8 * 4], upk[hh][q8 * 4 + 1], upk[hh][q8 * 4 + 2], upk[hh][q8 * 4 + 3]);
                *reinterpret_cast<uint4*>(Tsm + tile_off(t, c0 + q8 * 8)) = make_uint4(udpk[hh][q8 * 4], udpk[hh][q8 * 4 + 1], udpk[hh][q8 * 4 + 2], udpk[hh][q8 * 4 + 3]);
            }
        }
        fence_async_smem();
        __syncthreads();
        if (tid == 224) issue_mma_d(tm3, dQA_k, 0, dV_mn, 1, 0, 0, barb3);          // Ot = A U
        if (tid == 192) issue_mma_d(tm2, dK_mn[kcur], 1, dT_mn, 1, 0, 0, barb2);    // B = K^T Ud
        DBG_REC(6);
        // Qt epilogue -> global
        wait0();
        unsigned char* qtb = reinterpret_cast<unsigned char*>(Qt + ch * CH * DK);   // Qt[c][hv] stored as the 32 KB swizzled tile image
        uint32_t rq[64];
        tc_ld64(tm0 + tlane + ecol0, rq);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c0 = ecol0 + hh * 32;
            const uint32_t* r = rq + hh * 32;
            #pragma unroll
            for (int q8 = 0; q8 < 4; ++q8) {
                uint4 pk;
                pk.x = pack_bf16x2(__uint_as_float(r[q8 * 8]), __uint_as_float(r[q8 * 8 + 1]));
                pk.y = pack_bf16x2(__uint_as_float(r[q8 * 8 + 2]), __uint_as_float(r[q8 * 8 + 3]));
                pk.z = pack_bf16x2(__uint_as_float(r[q8 * 8 + 4]), __uint_as_float(r[q8 * 8 + 5]));
                pk.w = pack_bf16x2(__uint_as_float(r[q8 * 8 + 6]), __uint_as_float(r[q8 * 8 + 7]));
                *reinterpret_cast<uint4*>(qtb + tile_off(t, c0 + q8 * 8)) = pk;
            }
        }
        // Wd = dL W in place (Qt MMA done -> W free), issue Mp = -K^T Wd (tm1)
        #pragma unroll
        for (int qq = 0; qq < 8; ++qq) {
            int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
            uint4* p = reinterpret_cast<uint4*>(WV + tile_off(row, cidx * 8));
            uint4 val = *p;
            const uint32_t s2 = pack_bf16x2(sDL[row], sDL[row]);
            uint32_t* w = reinterpret_cast<uint32_t*>(&val);
            #pragma unroll
            for (int e = 0; e < 4; ++e) w[e] = hmul2_bf16(w[e], s2);
            *p = val;
        }
        fence_async_smem();
        tc_fence_before();
        __syncthreads();
        if (tid == 160) issue_mma_d(tm1, dK_mn[kcur], 1, dWV_mn, 1, 1, 0, barb1);   // Mp = -K^T Wd
        DBG_REC(7);
    }
    // ---------------- P6: Ot epilogue -> global ; B epilogue (stage in Vs) ; Mp epilogue (stage in WV)
    if (IT_STOP_AFTER > 7) {
        const int t = erow;
        wait3();
        {   // Ot: stage row-major [t][v] (256 B rows, chunk swizzle by (t&7) to spread banks) in Vs, then contiguous copy
            uint32_t rr[64];
            tc_ld64(tm3 + tlane + ecol0, rr);
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int c0 = ecol0 + hh * 32;
                const uint32_t* r = rr + hh * 32;
                #pragma unroll
                for (int q8 = 0; q8 < 4; ++q8) {
                    uint4 pk;
                    pk.x = pack_bf16x2(__uint_as_float(r[q8 * 8]), __uint_as_float(r[q8 * 8 + 1]));
                    pk.y = pack_bf16x2(__uint_as_float(r[q8 * 8 + 2]), __uint_as_float(r[q8 * 8 + 3]));
                    pk.z = pack_bf16x2(__uint_as_float(r[q8 * 8 + 4]), __uint_as_float(r[q8 * 8 + 5]));
                    pk.w = pack_bf16x2(__uint_as_float(r[q8 * 8 + 6]), __uint_as_float(r[q8 * 8 + 7]));
                    const int chunk = ((c0 + q8 * 8) >> 3) ^ (t & 15);
                    *reinterpret_cast<uint4*>(Vsm + t * 256 + chunk * 16) = pk;
                }
            }
            __syncthreads();
            unsigned char* dstb = reinterpret_cast<unsigned char*>(Ot + ch * CH * DV);   // Ot[c][hv] stored as the swizzled tile image
            #pragma unroll
            for (int qq = 0; qq < 8; ++qq) {
                int i = tid + qq * 256, row = i >> 4, cidx = i & 15;
                *reinterpret_cast<uint4*>(dstb + tile_off(row, cidx * 8)) = *reinterpret_cast<const uint4*>(Vsm + row * 256 + ((cidx ^ (row & 15)) * 16));
            }
        }
        DBG_REC(8);
        wait2();      // B done
        {   // B global layout: [vs][128 i][VW v] bf16; stage in Vs then contiguous copy
            __syncthreads();   // Ot copy-out reads of Vs done
            uint32_t rrb[64];
            tc_ld64(tm2 + tlane + ecol0, rrb);
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int c0 = ecol0 + hh * 32;
                const uint32_t* r = rrb + hh * 32;
                #pragma unroll
                for (int q8 = 0; q8 < 4; ++q8) {
                    uint4 pk;
                    pk.x = pack_bf16x2(__uint_as_float(r[q8 * 8]), __uint_as_float(r[q8 * 8 + 1]));
                    pk.y = pack_bf16x2(__uint_as_float(r[q8 * 8 + 2]), __uint_as_float(r[q8 * 8 + 3]));
                    pk.z = pack_bf16x2(__uint_as_float(r[q8 * 8 + 4]), __uint_as_float(r[q8 * 8 + 5]));
                    pk.w = pack_bf16x2(__uint_as_float(r[q8 * 8 + 6]), __uint_as_float(r[q8 * 8 + 7]));
                    const int v = c0 + q8 * 8;
                    *reinterpret_cast<uint4*>(Vsm + (size_t)(v / VW) * (DK * VW * 2) + t * (VW * 2) + (v % VW) * 2) = pk;
                }
            }
            __syncthreads();
            const uint4* src = reinterpret_cast<const uint4*>(Vsm);
            uint4* dst = reinterpret_cast<uint4*>(Bp + ch * DK * DV);
            #pragma unroll
            for (int qq = 0; qq < 8; ++qq) dst[tid + qq * 256] = src[tid + qq * 256];
        }
        DBG_REC(9);
        wait1();      // Mp done
        {   // Mp global layout == swizzled smem tile layout (32 KB per chunk-head): stage in WV, then contiguous copy
            uint32_t rr[64];
            tc_ld64(tm1 + tlane + ecol0, rr);
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int c0 = ecol0 + hh * 32;
                const uint32_t* r = rr + hh * 32;
                #pragma unroll
                for (int q8 = 0; q8 < 4; ++q8) {
                    uint4 pk;
                    pk.x = pack_bf16x2(__uint_as_float(r[q8 * 8]), __uint_as_float(r[q8 * 8 + 1]));
                    pk.y = pack_bf16x2(__uint_as_float(r[q8 * 8 + 2]), __uint_as_float(r[q8 * 8 + 3]));
                    pk.z = pack_bf16x2(__uint_as_float(r[q8 * 8 + 4]), __uint_as_float(r[q8 * 8 + 5]));
                    pk.w = pack_bf16x2(__uint_as_float(r[q8 * 8 + 6]), __uint_as_float(r[q8 * 8 + 7]));
                    *reinterpret_cast<uint4*>(WV + tile_off(t, c0 + q8 * 8)) = pk;
                }
            }
            __syncthreads();
            const uint4* src = reinterpret_cast<const uint4*>(WV);
            uint4* dst = reinterpret_cast<uint4*>(Mp + ch * DK * DK);
            #pragma unroll
            for (int qq = 0; qq < 8; ++qq) dst[tid + qq * 256] = src[tid + qq * 256];
        }
        tc_fence_before();
        __syncthreads();
        if (tid == 0) { __threadfence(); atomicAdd(ready + c, 1); }
        DBG_REC(10);
    }
    if (DBG_ON && g_dbg && tid == 0) g_dbg[1025 + 2 * gcur] = gtimer();
    gcur = next_g;
    if (next_is_intra && prefetched_item == next_item) kcur ^= 1;
    cur_len = next_is_intra ? nlen : -2; cur_tok0 = ntok0;
  }
    tc_fence_before();
    __syncthreads();
    tc_fence_before();
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tm0), "r"(512));
}

// ------------------------------------------------------------------ host launchers
extern "C" {
int gdn_meta(const void* cu_seqlens, int N, int C64max, int C128max, void* cstart64, void* cstart128,
             void* cinfo64, void* cinfo128, void* counters, void* ready, void* ready2, int HV, void* seq_order, int Ksplit, int max_early, void* stream) {
    size_t sm = (size_t)(2 * (N + 1) + 3 * N) * sizeof(int);
    meta_kernel<<<1, 1024, sm, (cudaStream_t)stream>>>((const int64_t*)cu_seqlens, N, C64max, C128max,
        (int*)cstart64, (int*)cstart128, (int*)cinfo64, (int*)cinfo128, (int*)counters, (int*)ready, (int*)ready2, HV,
        (int*)seq_order, Ksplit, max_early);
    return (int)cudaGetLastError();
}
int gdn_prep(const void* k, const void* a, const void* b, const void* A_log, const void* dt_bias,
             const void* cinfo64, int C64max, int HQK, int HV, void* gam, void* bet, void* Tm, void* stream) {
    if (HV != 2 * HQK) return -100;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(prep_kernel<2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        attr_set = true;
    }
    dim3 grid(C64max, HQK);
    prep_kernel<2><<<grid, 256, 0, (cudaStream_t)stream>>>((const bf16*)k, (const bf16*)a, (const bf16*)b,
        (const float*)A_log, (const float*)dt_bias, (const int*)cinfo64, HQK, HV, (float*)gam, (float*)bet, (bf16*)Tm);
    return (int)cudaGetLastError();
}
int gdn_intra(const void* q, const void* k, const void* v, const void* a, const void* b, const void* A_log, const void* dt_bias,
              const void* cinfo128, int C128max, int HQK, int HV, float scale, void* counter, void* ready,
              void* Mp, void* Bp, void* Qt, void* Ot, void* gchunk,
              const void* state, void* new_state, void* Hs, const void* cstart128, int N, int fuse_seq,
              void* ready2, void* out, int fuse_out, const void* seq_order, int Ksplit, void* stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(intra_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, IT_SMEM);
        cudaFuncSetAttribute(intra_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        attr_set = true;
    }
    int nitems = C128max * HV;
    static int nsm = 0;
    if (!nsm) { cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0); }
    int grid = min(nitems, nsm * IT_GRID_PER_SM);
    intra_kernel<<<grid, 256, IT_SMEM, (cudaStream_t)stream>>>((const bf16*)q, (const bf16*)k, (const bf16*)v, (const bf16*)a, (const bf16*)b,
        (const float*)A_log, (const float*)dt_bias, (const int*)cinfo128, HQK, HV, scale, nitems, (int*)counter,
        (bf16*)Mp, (bf16*)Bp, (bf16*)Qt, (bf16*)Ot, (float*)gchunk, (int*)ready,
        (const float*)state, (float*)new_state, (bf16*)Hs, (const int*)cstart128, fuse_seq ? NVS * HV * N : 0,
        (int*)ready2, (bf16*)out, fuse_out ? C128max * HV : 0, (const int*)seq_order, Ksplit);
    return (int)cudaGetLastError();
}
int gdn_seq3(const void* Mp, const void* Bp, const void* gchunk, const void* state, void* new_state, void* Hs,
             const void* cstart128, const void* ready, int HV, int N, void* stream) {
    static bool attr_set = false;
    if (!attr_set) { cudaFuncSetAttribute(seq_kernel3, cudaFuncAttributeMaxDynamicSharedMemorySize, SQ_SMEM); attr_set = true; }
    dim3 grid(NVS, HV, N);
    seq_kernel3<<<grid, 224, SQ_SMEM, (cudaStream_t)stream>>>((const bf16*)Mp, (const bf16*)Bp, (const float*)gchunk,
        (const float*)state, (float*)new_state, (bf16*)Hs, (const int*)cstart128, (const int*)ready, HV);
    return (int)cudaGetLastError();
}
int gdn_out(const void* Qt, const void* Ot, const void* Hs, const void* cinfo128, void* out, int C128max, int HV, void* stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(out_kernel_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, OUT_SMEM);
        cudaFuncSetAttribute(out_kernel_tc, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        attr_set = true;
    }
    dim3 grid(C128max, HV);
    out_kernel_tc<<<grid, 128, OUT_SMEM, (cudaStream_t)stream>>>((const bf16*)Qt, (const bf16*)Ot, (const bf16*)Hs, (const int*)cinfo128, (bf16*)out, HV);
    return (int)cudaGetLastError();
}
int gdn_set_dbg(void* p) { return (int)cudaMemcpyToSymbol(g_dbg, &p, sizeof(void*)); }
int gdn_occupancy(int which) {
    int nb = -1;
    if (which == 0) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, prep_kernel<2>, 256, 0);
    else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, seq_kernel3, 224, 0);
    return nb;
}
}
