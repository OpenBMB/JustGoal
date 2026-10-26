// DSA sparse MLA decode forward (T tokens, 16 heads, ckv 512 + kpe 64, bf16) for Blackwell (sm_100a).
// Written from the operation spec only.
//
// One cooperative kernel, N = 32 CTAs per token, no clusters:
//   1. every CTA loads the token's 2048 indices, compacts the valid ones (n rows), and owns
//      (a) the contiguous row block [s*RB, s*RB+RB) for S = Q K^T (RB multiple of 8, <= 64), and
//      (b) the output dims [16 s, 16 s + 16) for O = P V.
//   2. K rows (full 1152 B) of (a) and V columns (32 B of every valid row) of (b) are gathered
//      with cp.async into swizzled smem; Q goes to smem as well.
//   3. S with mma.sync (M = 16 heads, two warps per 8-row tile split over k), local softmax over
//      the block with the same arithmetic as the fp32 spec (ls = S * sm_scale, p = exp(ls - max)).
//      P is split exactly into three bf16 parts (hi + mid + lo == fp32 p) and published to a global
//      workspace in mma A-fragment order as self-validating 8-byte words (payload + 16-bit epoch,
//      NCCL "LL" style): no fence, no flag, a single round trip for the consumer.
//   4. every CTA consumes all source blocks of its token (coalesced 16 B volatile loads re-issued
//      until every word carries the current epoch), computes the global row max M from the
//      published block maxima, accumulates P V for its 16 dims scaled by exp(m_src - M), reduces
//      across warps and writes out / lse.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

#ifdef DSA_TIMING
constexpr int NSTAMP = 32;
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long x;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x));
  return x;
}
#define TSTAMP(i)                                        \
  if (tid == 0) {                                        \
    tbuf[(t * N + s) * NSTAMP + (i)] = clock64();        \
    gbuf[(t * N + s) * NSTAMP + (i)] = gtimer();         \
  }
#else
#define TSTAMP(i)
#endif

constexpr int H = 16, DC = 512, DP = 64, TOPK = 2048;
constexpr int NTHREADS = 256, NWARPS = 8;
constexpr int N = 32;                  // CTAs per token
constexpr int RBMAX = TOPK / N;        // 64: max rows per source block
constexpr int KSMAX = RBMAX / 16;      // 4:  max k-steps per source block
constexpr int CWD = DC / N;            // 16: output dims per CTA
constexpr int NT = CWD / 8;            // 2:  n-tiles per CTA
constexpr int KV_ROW_B = DC * 2;       // 1024
constexpr int PE_ROW_B = DP * 2;       // 128
constexpr int V_ROW_B = CWD * 2;       // 32
constexpr int KSTEPS = (DC + DP) / 16;  // 36 k-steps of S
constexpr int KPAIRS = KSTEPS / 2;      // 18
constexpr int VROWS = TOPK + 16;       // V rows in smem incl. zero tail
constexpr int UBATCH = 4;              // consume units per batch

// workspace record per (token, source): 16-byte chunks, chunk c of lane l at (c*32 + l)*16.
//   P chunks [KSMAX][4]: chunk k of k-step ks holds elements (2k, 2k+1) of the 8-element
//     C/A-fragment vector (p(g,2q), p(g,2q+1), p(g+8,2q), p(g+8,2q+1)) x (tile 2ks, tile 2ks+1),
//     each element as an 8-byte word (hi16 | mid16<<16, lo16 | epoch16<<16).
//   then 1 chunk: (m[g], epoch, m[g+8], epoch). The row sums l are recomputed by the consumers
//   from the exact P payload, so the source needs a single block barrier after S.
constexpr int REC_CHUNKS = KSMAX * 4 + 1;
constexpr int REC_B = REC_CHUNKS * 32 * 16;  // 8704

// shared memory layout (bytes)
constexpr int SM_QN = 0;
constexpr int SM_QP = SM_QN + H * KV_ROW_B;            // 16384
constexpr int SM_KV = SM_QP + H * PE_ROW_B;            // 18432
constexpr int SM_KP = SM_KV + RBMAX * KV_ROW_B;        // 83968
constexpr int SM_V = SM_KP + RBMAX * PE_ROW_B;         // 92160
constexpr int SM_SEL = SM_V + VROWS * V_ROW_B;         // 158208
constexpr int SM_RED = SM_SEL + TOPK * 4;              // 166400: red_max[8][16], red_sum[8][16]
constexpr int SM_SCAN = SM_RED + 2 * NWARPS * H * 4;   // 167424
constexpr int SM_CMB = SM_SCAN + 128;                  // 167552: [8][16][16] fp32 (also split-k S handoff)
constexpr int SM_CML = SM_CMB + NWARPS * H * CWD * 4;  // 175744: l[8][16], M[16]
constexpr int SM_MS = SM_CML + (NWARPS + 1) * H * 4;   // 176320: per-warp max of block maxima [8][16]
constexpr int SM_TOTAL = SM_MS + NWARPS * H * 4;       // 176832

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(__cvta_generic_to_global(src)));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int M>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(M));
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(__cvta_generic_to_global(p))
               : "memory");
  return v;
}
__device__ __forceinline__ void st_16(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(__cvta_generic_to_global(p)), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w)
               : "memory");
}
// 16-byte chunk swizzles keeping ldmatrix conflict-free.
__device__ __forceinline__ int kv_off(int row, int chunk) { return row * KV_ROW_B + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ int pe_off(int row, int chunk) { return row * PE_ROW_B + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ int v_off(int row, int chunk) { return row * V_ROW_B + ((chunk ^ ((row >> 2) & 1)) << 4); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
// exact split p == hi + mid + lo (three bf16 obtained by truncation); returns the two LL payload words
__device__ __forceinline__ uint2 split_ll(float p, uint32_t ep16) {
  const uint32_t hb = __float_as_uint(p) & 0xffff0000u;
  const float r1 = p - __uint_as_float(hb);
  const uint32_t mb = __float_as_uint(r1) & 0xffff0000u;
  const float r2 = r1 - __uint_as_float(mb);
  const uint32_t lb = __float_as_uint(r2) & 0xffff0000u;
  return make_uint2((hb >> 16) | mb, (lb >> 16) | (ep16 << 16));
}

__global__ void __launch_bounds__(NTHREADS, 1)
dsa_decode_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
                  const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
                  const int32_t* __restrict__ indices, __nv_bfloat16* __restrict__ out,
                  float* __restrict__ lse, uint8_t* __restrict__ ws, int epoch, float sm_scale,
                  long long* __restrict__ tbuf, unsigned long long* __restrict__ gbuf) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int s = blockIdx.x;  // source block id == output dim slice id
  const int t = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const uint32_t sbase = smem_u32(smem);
  const uint32_t ep = static_cast<uint32_t>(epoch);  // 1..65535
  TSTAMP(0);

  // 1. Q -> smem (cp.async group 0), issued first so it overlaps the index round trip.
  {
    const __nv_bfloat16* qn = q_nope + static_cast<size_t>(t) * H * DC;
    const __nv_bfloat16* qp = q_pe + static_cast<size_t>(t) * H * DP;
#pragma unroll
    for (int i = 0; i < (H * 64) / NTHREADS; ++i) {
      const int item = tid + i * NTHREADS, row = item >> 6, c = item & 63;
      cp_async16(sbase + SM_QN + kv_off(row, c), qn + row * DC + c * 8);
    }
    if (tid < H * 8) {
      const int row = tid >> 3, c = tid & 7;
      cp_async16(sbase + SM_QP + pe_off(row, c), qp + row * DP + c * 8);
    }
    cp_async_commit();
  }

  // 2. compact the token's indices: sel[0..n) = valid (>= 0) entries in slot order.
  int* sel = reinterpret_cast<int*>(smem + SM_SEL);
  int* scan = reinterpret_cast<int*>(smem + SM_SCAN);
  int n;
  {
    const int4* ip = reinterpret_cast<const int4*>(indices + static_cast<size_t>(t) * TOPK) + 2 * tid;
    const int4 lo = ip[0], hi = ip[1];
    const int v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    int cnt = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) cnt += (v[i] >= 0) ? 1 : 0;
    int incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      incl += (lane >= o) ? y : 0;
    }
    if (lane == 31) scan[warp] = incl;
    __syncthreads();
    int wbase = 0;
    n = 0;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      const int c = scan[w];
      wbase += (w < warp) ? c : 0;
      n += c;
    }
    int pos = wbase + incl - cnt;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (v[i] >= 0) sel[pos++] = v[i];
    }
    __syncthreads();
  }
  const int RB = min(RBMAX, (((n + N - 1) / N) + 7) & ~7);  // rows per source block (multiple of 8)
  const int nsrc = (RB > 0) ? (n + RB - 1) / RB : 0;        // source blocks with rows: 0..nsrc-1
  const int my0 = s * RB;
  const int my_cnt = max(0, min(RB, n - my0));               // rows of this block
  const int KS = (my_cnt + 15) >> 4;                         // k-steps this block publishes
  const int ntiles = (my_cnt + 7) >> 3;                      // 8-row S tiles
  TSTAMP(1);

  // 3a. gather own K rows (group 1): one warp per row, 64 + 8 chunks.
  for (int r = warp; r < my_cnt; r += NWARPS) {
    const int tok = sel[my0 + r];
    const __nv_bfloat16* src = ckv + static_cast<size_t>(tok) * DC;
    cp_async16(sbase + SM_KV + kv_off(r, lane), src + lane * 8);
    cp_async16(sbase + SM_KV + kv_off(r, lane + 32), src + (lane + 32) * 8);
    if (lane < 8) cp_async16(sbase + SM_KP + pe_off(r, lane), kpe + static_cast<size_t>(tok) * DP + lane * 8);
  }
  cp_async_commit();
  // 3b. gather V columns [16 s, 16 s + 16) of every valid row (group 2): 16 rows per warp instruction.
  for (int base = warp * 16; base < n; base += NWARPS * 16) {
    const int row = base + (lane >> 1);
    if (row < n) {
      const int tok = sel[row];
      cp_async16(sbase + SM_V + v_off(row, lane & 1), ckv + static_cast<size_t>(tok) * DC + s * CWD + (lane & 1) * 8);
    }
  }
  if (tid < 32) {  // zero tail rows [n, n + 16) read by padded k-steps
    const int row = n + (tid >> 1);
    *reinterpret_cast<uint4*>(smem + SM_V + v_off(row, tid & 1)) = make_uint4(0u, 0u, 0u, 0u);
  }
  cp_async_commit();

  // 4. S = Q K^T: tile warp w < ntiles does k-pairs [0, kp_split), helper warp ntiles + w does
  //    [kp_split, 18) when ntiles <= 4; partial S handed over through smem.
  const int nhelp = (ntiles <= NWARPS / 2) ? ntiles : 0;
  const bool tile_w = warp < ntiles;
  const bool help_w = (warp >= ntiles) && (warp < ntiles + nhelp);
  const int my_tile = tile_w ? warp : (warp - ntiles);
  const int kp0 = help_w ? KPAIRS / 2 : 0;
  const int kp1 = tile_w ? (nhelp ? KPAIRS / 2 : KPAIRS) : (help_w ? KPAIRS : 0);
  float m0 = neg_inf(), m1 = neg_inf();  // block maxima for rows g, g+8
  float* red_max = reinterpret_cast<float*>(smem + SM_RED);
  float* s_hand = reinterpret_cast<float*>(smem + SM_CMB);  // [4 helpers][32 lanes][4]
  uint8_t* rec = ws + (static_cast<size_t>(t) * N + s) * REC_B;
  {
    // Q A-fragments for one half (18 k-steps) of the reduction; K B-fragments streamed from smem.
    uint32_t qa[KSTEPS / 2][4];
    const bool qk_w = tile_w || help_w;
    const int ks_base = help_w ? KSTEPS / 2 : 0;
    auto preload_q = [&](int ksb) {
#pragma unroll
      for (int i = 0; i < KSTEPS / 2; ++i) {
        const int ks = ksb + i;
        const int row = lane & 15;
        const uint32_t addr = (ks < DC / 16)
                                  ? sbase + SM_QN + kv_off(row, ks * 2 + (lane >> 4))
                                  : sbase + SM_QP + pe_off(row, (ks - DC / 16) * 2 + (lane >> 4));
        ldsm_x4(addr, qa[i]);
      }
    };
    float sp[4][4];  // 4 independent accumulation chains
#pragma unroll
    for (int i = 0; i < 4; ++i) sp[i][0] = sp[i][1] = sp[i][2] = sp[i][3] = 0.f;
    auto mma_half = [&](int kpb) {
      const int row = my_tile * 8 + (lane & 7);
#pragma unroll
      for (int i = 0; i < KPAIRS / 2; ++i) {
        const int kp = kpb + i;
        uint32_t b[4];
        const uint32_t addr = (kp < DC / 32)
                                  ? sbase + SM_KV + kv_off(row, kp * 4 + (lane >> 3))
                                  : sbase + SM_KP + pe_off(row, (kp - DC / 32) * 4 + (lane >> 3));
        ldsm_x4(addr, b);
        mma_bf16(sp[i & 3], qa[2 * i], b[0], b[1]);
        mma_bf16(sp[(i + 2) & 3], qa[2 * i + 1], b[2], b[3]);
      }
    };
    cp_async_wait<2>();
    __syncthreads();
    TSTAMP(2);
    if (qk_w) preload_q(ks_base);
    cp_async_wait<1>();
    __syncthreads();
    TSTAMP(3);
    if (qk_w) {
      mma_half(ks_base / 2);
      if (tile_w && !nhelp) {  // > 4 tiles: this warp also does the second half
        preload_q(KSTEPS / 2);
        mma_half(KPAIRS / 2);
      }
    }
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sc[0] += sp[i][0]; sc[1] += sp[i][1]; sc[2] += sp[i][2]; sc[3] += sp[i][3];
    }
    if (help_w) *reinterpret_cast<float4*>(s_hand + (my_tile * 32 + lane) * 4) = make_float4(sc[0], sc[1], sc[2], sc[3]);
    __syncthreads();
    if (tile_w && nhelp) {
      const float4 h = *reinterpret_cast<const float4*>(s_hand + (my_tile * 32 + lane) * 4);
      sc[0] += h.x; sc[1] += h.y; sc[2] += h.z; sc[3] += h.w;
    }
    TSTAMP(4);
    // local softmax; helper / idle warps hold masked columns only (col >= my_cnt).
    const int col = warp * 8 + 2 * q;
    sc[0] = (col < my_cnt) ? sc[0] * sm_scale : neg_inf();
    sc[1] = (col + 1 < my_cnt) ? sc[1] * sm_scale : neg_inf();
    sc[2] = (col < my_cnt) ? sc[2] * sm_scale : neg_inf();
    sc[3] = (col + 1 < my_cnt) ? sc[3] * sm_scale : neg_inf();
    float mx0 = fmaxf(sc[0], sc[1]), mx1 = fmaxf(sc[2], sc[3]);
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    if (q == 0) {
      red_max[warp * H + g] = mx0;
      red_max[warp * H + g + 8] = mx1;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      m0 = fmaxf(m0, red_max[w * H + g]);
      m1 = fmaxf(m1, red_max[w * H + g + 8]);
    }
    if (KS > 0 && warp == 0)
      st_16(rec + ((KSMAX * 4) * 32 + lane) * 16, make_uint4(__float_as_uint(m0), ep, __float_as_uint(m1), ep));
    const float p0 = expf(sc[0] - m0), p1 = expf(sc[1] - m0);  // exp(-inf) = 0 for masked / empty
    const float p2 = expf(sc[2] - m1), p3 = expf(sc[3] - m1);
    // publish P: tile-warp w (k-step w>>1, half w&1) writes chunks 2(w&1) and 2(w&1)+1
    if (warp < 2 * KS) {
      const uint2 w0 = split_ll(p0, ep), w1 = split_ll(p1, ep), w2 = split_ll(p2, ep), w3 = split_ll(p3, ep);
      const int c0 = (warp >> 1) * 4 + (warp & 1) * 2;
      st_16(rec + (c0 * 32 + lane) * 16, make_uint4(w0.x, w0.y, w1.x, w1.y));
      st_16(rec + ((c0 + 1) * 32 + lane) * 16, make_uint4(w2.x, w2.y, w3.x, w3.y));
    }
    TSTAMP(5);
  }

  // 5. consume: units (src, ks) of this warp with src in {warp, warp+8, ...}, batches of UBATCH;
  //    all loads of a batch in flight together, re-issued until every word carries this epoch.
  cp_async_wait<0>();
  __syncthreads();
  TSTAMP(6);
  float* mS = reinterpret_cast<float*>(smem + SM_MS);      // [warp][16] partial maxima
  float* cmb = reinterpret_cast<float*>(smem + SM_CMB);    // [warp][row][col]
  float* cl = reinterpret_cast<float*>(smem + SM_CML);     // [warp][row]
  float* cM = cl + NWARPS * H;                             // [row]
  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float lr0 = 0.f, lr1 = 0.f;
  const int KSU = (RB + 15) >> 4;                 // k-steps of a full source block
  const int nunits = (N / NWARPS) * KSU;          // unit slots per warp (some empty)
  // batch 0 is loaded before the global max is known; its units are processed after the M exchange.
  uint4 buf[UBATCH][5];
  bool live[UBATCH];
  int usrc[UBATCH], uks[UBATCH];
  auto load_batch = [&](int u0) {
#pragma unroll
    for (int i = 0; i < UBATCH; ++i) {
      const int u = u0 + i;
      usrc[i] = warp + NWARPS * (u / KSU);
      uks[i] = u % KSU;
      const int cnt = (usrc[i] < N) ? min(RB, n - usrc[i] * RB) : 0;
      live[i] = (u < nunits) && (cnt > 0) && (uks[i] < ((cnt + 15) >> 4));
    }
    bool pend[UBATCH];
#pragma unroll
    for (int i = 0; i < UBATCH; ++i) pend[i] = live[i];
    bool any;
    do {
#pragma unroll
      for (int i = 0; i < UBATCH; ++i) {
        if (pend[i]) {
          const uint8_t* r = ws + (static_cast<size_t>(t) * N + usrc[i]) * REC_B;
#pragma unroll
          for (int k = 0; k < 4; ++k) buf[i][k] = ld_volatile16(r + ((uks[i] * 4 + k) * 32 + lane) * 16);
          buf[i][4] = ld_volatile16(r + ((KSMAX * 4) * 32 + lane) * 16);
        }
      }
      any = false;
#pragma unroll
      for (int i = 0; i < UBATCH; ++i) {
        if (pend[i]) {
          bool ok = (buf[i][4].y == ep) && (buf[i][4].w == ep);
#pragma unroll
          for (int k = 0; k < 4; ++k) ok = ok && ((buf[i][k].y >> 16) == ep) && ((buf[i][k].w >> 16) == ep);
          pend[i] = !ok;
          any = any || !ok;
        }
      }
    } while (any);
  };
  auto consume_batch = [&](float M0, float M1) {
#pragma unroll
    for (int i = 0; i < UBATCH; ++i) {
      if (!live[i]) continue;
      uint32_t a[3][4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {  // elements (2k, 2k+1) -> A-fragment register k
        a[0][k] = __byte_perm(buf[i][k].x, buf[i][k].z, 0x5410);  // hi
        a[1][k] = __byte_perm(buf[i][k].x, buf[i][k].z, 0x7632);  // mid
        a[2][k] = __byte_perm(buf[i][k].y, buf[i][k].w, 0x5410);  // lo
      }
      float c[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
      const int vrow = usrc[i] * RB + uks[i] * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
      uint32_t b[4];
      ldsm_x4_trans(sbase + SM_V + v_off(vrow, lane >> 4), b);
#pragma unroll
      for (int part = 0; part < 3; ++part) {
        mma_bf16(c[0], a[part], b[0], b[1]);
        mma_bf16(c[1], a[part], b[2], b[3]);
      }
      const float ms0 = __uint_as_float(buf[i][4].x), ms1 = __uint_as_float(buf[i][4].z);
      const float b0 = expf(ms0 - M0), b1 = expf(ms1 - M1);
      // row sums of this unit from the exact payload (p == hi + mid + lo), reduced over the quad
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint32_t e0 = buf[i][k].x, e1 = buf[i][k].z;  // elements 2k, 2k+1: rows g (k even) / g+8 (k odd)
        const float v0 = __uint_as_float(e0 << 16) + __uint_as_float(e0 & 0xffff0000u) + __uint_as_float(buf[i][k].y << 16);
        const float v1 = __uint_as_float(e1 << 16) + __uint_as_float(e1 & 0xffff0000u) + __uint_as_float(buf[i][k].w << 16);
        if (k & 1) rs1 += v0 + v1; else rs0 += v0 + v1;
      }
      rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
      rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
      rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
      rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
      lr0 += rs0 * b0;
      lr1 += rs1 * b1;
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        acc[j][0] += c[j][0] * b0;
        acc[j][1] += c[j][1] * b0;
        acc[j][2] += c[j][2] * b1;
        acc[j][3] += c[j][3] * b1;
      }
    }
  };
  load_batch(0);
  TSTAMP(7);
  // global row maxima: warp max over its batch-0 units, then across warps through smem.
  float M0 = neg_inf(), M1 = neg_inf();
#pragma unroll
  for (int i = 0; i < UBATCH; ++i) {
    if (live[i] && uks[i] == 0) {
      M0 = fmaxf(M0, __uint_as_float(buf[i][4].x));
      M1 = fmaxf(M1, __uint_as_float(buf[i][4].z));
    }
  }
  if (KSU > UBATCH) {  // generic path (block > 64 rows never happens: RBMAX == 64); sources beyond batch 0
    for (int src = warp + NWARPS * (UBATCH / KSU + 1); src < nsrc; src += NWARPS) {
      const uint8_t* r = ws + (static_cast<size_t>(t) * N + src) * REC_B;
      uint4 v;
      do { v = ld_volatile16(r + ((KSMAX * 4) * 32 + lane) * 16); } while (v.y != ep || v.w != ep);
      M0 = fmaxf(M0, __uint_as_float(v.x));
      M1 = fmaxf(M1, __uint_as_float(v.z));
    }
  }
  if (q == 0) { mS[warp * H + g] = M0; mS[warp * H + g + 8] = M1; }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < NWARPS; ++w) {
    M0 = fmaxf(M0, mS[w * H + g]);
    M1 = fmaxf(M1, mS[w * H + g + 8]);
  }
  consume_batch(M0, M1);
  for (int u0 = UBATCH; u0 < nunits; u0 += UBATCH) {
    load_batch(u0);
    consume_batch(M0, M1);
  }
  TSTAMP(8);

  // 6. cross-warp sum and store.
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int cidx = j * 8 + 2 * q;
    *reinterpret_cast<float2*>(cmb + (warp * H + g) * CWD + cidx) = make_float2(acc[j][0], acc[j][1]);
    *reinterpret_cast<float2*>(cmb + (warp * H + g + 8) * CWD + cidx) = make_float2(acc[j][2], acc[j][3]);
  }
  if (q == 0) {
    cl[warp * H + g] = lr0;
    cl[warp * H + g + 8] = lr1;
    if (warp == 0) { cM[g] = M0; cM[g + 8] = M1; }
  }
  __syncthreads();
  {
    const int row = tid >> 4, col = tid & 15;
    const float M = cM[row];
    float o = 0.f, lse_v = neg_inf();
    if (M > neg_inf()) {
      float L = 0.f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) {
        L += cl[w * H + row];
        o += cmb[(w * H + row) * CWD + col];
      }
      o /= L;
      lse_v = (M + logf(L)) / 0.6931471805599453f;  // base-2 lse exactly as the spec
    }
    out[(static_cast<size_t>(t) * H + row) * DC + s * CWD + col] = __float2bfloat16_rn(o);
    if (s == 0 && col == 0) lse[t * H + row] = lse_v;
  }
  TSTAMP(9);
}

std::vector<at::Tensor> run(const at::Tensor& q_nope, const at::Tensor& q_pe, const at::Tensor& ckv,
                            const at::Tensor& kpe, const at::Tensor& idx, double sm_scale, at::Tensor ws,
                            int64_t epoch, bool cooperative, c10::optional<at::Tensor> tstamps,
                            c10::optional<at::Tensor> gstamps) {
  TORCH_CHECK(q_nope.is_cuda() && q_nope.dtype() == at::kBFloat16 && q_nope.is_contiguous());
  TORCH_CHECK(q_pe.dtype() == at::kBFloat16 && q_pe.is_contiguous());
  TORCH_CHECK(ckv.dtype() == at::kBFloat16 && ckv.is_contiguous());
  TORCH_CHECK(kpe.dtype() == at::kBFloat16 && kpe.is_contiguous());
  TORCH_CHECK(idx.dtype() == at::kInt && idx.is_contiguous());
  TORCH_CHECK(epoch >= 1 && epoch <= 65535);
  const int T = q_nope.size(0);
  TORCH_CHECK(q_nope.size(1) == H && q_nope.size(2) == DC && q_pe.size(2) == DP);
  TORCH_CHECK(ckv.size(2) == DC && kpe.size(2) == DP && idx.size(0) == T && idx.size(1) == TOPK);
  TORCH_CHECK(ws.numel() >= static_cast<int64_t>(T) * N * REC_B);
  at::Tensor out = at::empty({T, H, DC}, q_nope.options());
  at::Tensor lse = at::empty({T, H}, q_nope.options().dtype(at::kFloat));
  long long* tbuf = tstamps.has_value() ? reinterpret_cast<long long*>(tstamps->data_ptr<int64_t>()) : nullptr;
  unsigned long long* gbuf =
      gstamps.has_value() ? reinterpret_cast<unsigned long long*>(gstamps->data_ptr<int64_t>()) : nullptr;

  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(dsa_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
    configured = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N, T, 1);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = SM_TOTAL;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = cooperative ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, dsa_decode_kernel, reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(ckv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe.data_ptr()), idx.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), lse.data_ptr<float>(), ws.data_ptr<uint8_t>(),
      static_cast<int>(epoch), static_cast<float>(sm_scale), tbuf, gbuf));
  return {out, lse};
}

int64_t rec_bytes() { return REC_B; }
int64_t num_ctas() { return N; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA sparse MLA decode forward");
  m.def("rec_bytes", &rec_bytes);
  m.def("num_ctas", &num_ctas);
}
