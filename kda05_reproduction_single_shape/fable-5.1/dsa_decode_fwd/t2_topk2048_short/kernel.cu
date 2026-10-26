// DSA sparse attention (h16, ckv512, kpe64, topk2048, page_size 64) — v13: 4-CTA cluster split.
//
// One cluster of 4 CTAs (8 warps each) per token.  CTA rank g owns ckv dims [128g, 128g+128) and
// kpe dims [16g, 16g+16): it gathers only that 288-byte slice of every selected KV row, computes
// the partial logits S_g = Q_g K_g^T over its 144 dims with mma.sync, pushes the 16x32 fp32
// partial tile into the shared memory of its 3 peers (st.async DSMEM stores that signal the
// receiver's mbarrier by transaction bytes), sums the four partials, runs the (identical,
// deterministic) online softmax, and computes O_g = P V_g for its own 128 output dims from the V
// slice it already holds.
//
// Why: on B200 a single SM moves at most ~one 128 B line per 2 cycles through its load path and
// the dependent chain idx -> KV rows costs two DRAM trips; splitting the bytes (Q 18 KB + KV
// 21 KB + idx 8 KB for one chunk) over four SMs shortens every phase, while the exchange is only
// 2 KB per CTA and uses no global memory.  Measured cluster launch cost equals a plain launch.
//
// Phase 3 (v11-v13 vs v6j): the exchange is bandwidth-bound (~25 B/cycle received), so only the key
// pairs with a valid key are pushed (every CTA derives the same mask from the same index line);
// the row sums of P come from an all-ones B tile (every lane holds them: no quad shuffle); the Q
// A-fragments are loaded once, under the gather; the per-head softmax max uses two full-warp
// redux.sync ops (a partial-mask redux costs ~150 cycles more); the gather base pointers are
// materialised before the index lands.
//
// All global->shared movement uses plain 128-bit loads (cp.async / TMA bulk copies were measured
// to be slower to issue for these sizes).  Chunk 0 is gathered speculatively: every warp loads
// the first 32 indices itself and issues its KV loads as soon as they land.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifdef DSA_TIMING
__device__ long long g_dsa_ts[64][32];
#define TS(i) do { if (threadIdx.x == 0) { long long c_; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c_) :: "memory"); g_dsa_ts[blockIdx.x][i] = c_; } } while (0)
#define TSG(i) do { if (threadIdx.x == 0) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g) :: "memory"); g_dsa_ts[blockIdx.x][i] = (long long)g; } } while (0)
// stamp after a real consumer of a float value (a MOV that reads the register, so the hardware waits for it)
#define TSF(i, v) do { int d_; asm volatile("mov.b32 %0, %1;" : "=r"(d_) : "f"(v) : "memory"); asm volatile("" :: "r"(d_)); TS(i); } while (0)
#else
#define TS(i) do {} while (0)
#define TSG(i) do {} while (0)
#define TSF(i, v) do {} while (0)
#endif

namespace {

constexpr int H = 16;         // query heads  (= MMA M)
constexpr int DC = 512;       // head_dim_ckv
constexpr int DP = 64;        // head_dim_kpe
constexpr int TOPK = 2048;
constexpr int PS = 64;        // page size
constexpr int KC = 32;        // keys per chunk (4 n-tiles)
constexpr int NCHUNK = TOPK / KC;   // 64
constexpr int NT = 256;       // 8 warps
constexpr int NWARP = NT / 32;
constexpr int CL = 4;         // cluster size
constexpr int DCS = DC / CL;  // 128 ckv dims per CTA
constexpr int DPS = DP / CL;  // 16 kpe dims per CTA
constexpr int DS = DCS + DPS; // 144 dims per CTA = 9 k-steps
constexpr int KSTEPS = DS / 16;     // 9
constexpr int SROW = DS + 8;  // padded bf16 row (304 B): conflict-free ldmatrix
constexpr int PROW = KC + 8;  // padded P row (80 B)
constexpr int SPROW = KC + 8; // padded fp32 partial row (160 B): conflict-free STS.64 from the C layout
constexpr int OROW = DCS + 8; // padded output staging row (272 B)
constexpr int SEGS = DS * 2 / 16;   // 18 x 16 B per row slice (16 ckv + 2 kpe)
constexpr int NTILE_O = DCS / 8;    // 16 output n-tiles per CTA -> 2 per warp

struct __align__(16) Smem {
  __nv_bfloat16 q[H][SROW];          // 4864 B
  __nv_bfloat16 k[KC][SROW];         // 9728 B (reused as output staging [H][OROW])
  __nv_bfloat16 p[H][PROW];          // 1280 B
  float sp[NWARP][H][SPROW];         // 20480 B  per-warp partial logits (k-steps)
  float sx[2][CL][H][KC];            // 16384 B  partial logits from each cluster rank (double-buffered)
  float alpha[H];
  float lsum[H];
  int idx[TOPK];                     // 8192 B
  int chunk_valid[NCHUNK];
  int chunk_list[NCHUNK];
  int num_chunks;
  unsigned long long xbar;           // mbarrier: 3 peers x 2 KB of st.async bytes per chunk
};
static_assert(H * OROW * 2 <= KC * SROW * 2, "staging tile must fit in k");

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned cluster_rank() {
  unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r;
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned rank) {
  unsigned r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void st_async_f2(unsigned cluster_addr, float a, float b, unsigned cluster_bar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               ::"r"(cluster_addr), "f"(a), "f"(b), "r"(cluster_bar) : "memory");
}
__device__ __forceinline__ float ex2_approx(float x) {
  float r; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float r; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float redux_max_f32(float v, unsigned mask) {
  float r; asm volatile("redux.sync.max.f32 %0, %1, %2;\n" : "=f"(r) : "f"(v), "r"(mask)); return r;
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(unsigned cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void st_cluster_f2(unsigned cluster_addr, float a, float b) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" ::"r"(cluster_addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void ldsm_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void mma_16816(float c[4], const unsigned a[4], unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float group16_max(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float group16_sum(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void chunk_flags(Smem& sm, bool any0, bool any1, int warp, int lane) {
  // int4 i covers slots 4i..4i+3 -> chunk i/8.  warp w, lane l: i = 32w + l -> chunk 4w + l/8.
  const unsigned b0 = __ballot_sync(0xffffffffu, any0);
  const unsigned b1 = __ballot_sync(0xffffffffu, any1);
  if (lane < 4) sm.chunk_valid[4 * warp + lane] = ((b0 >> (8 * lane)) & 0xffu) ? 1 : 0;
  else if (lane < 8) sm.chunk_valid[NCHUNK / 2 + 4 * warp + (lane - 4)] = ((b1 >> (8 * (lane - 4))) & 0xffu) ? 1 : 0;
}
__device__ __forceinline__ void build_chunk_list(Smem& sm, int lane) {   // warp 0 only
  int base = 0;
#pragma unroll
  for (int half = 0; half < NCHUNK / 32; ++half) {
    const int c = half * 32 + lane;
    const bool v = sm.chunk_valid[c] != 0;
    const unsigned b = __ballot_sync(0xffffffffu, v);
    if (v) sm.chunk_list[base + __popc(b & ((1u << lane) - 1u))] = c;
    base += __popc(b);
  }
  if (lane == 0) sm.num_chunks = base;
}

__global__ void __launch_bounds__(NT) __cluster_dims__(CL, 1, 1)
dsa_sparse_attn_v13_kernel(const __nv_bfloat16* __restrict__ q_nope,   // [T, H, DC]
                          const __nv_bfloat16* __restrict__ q_pe,     // [T, H, DP]
                          const __nv_bfloat16* __restrict__ ckv,      // [P, PS, DC]
                          const __nv_bfloat16* __restrict__ kpe,      // [P, PS, DP]
                          const int* __restrict__ indices,            // [T, TOPK]
                          float scale_log2,                           // sm_scale * log2(e)
                          __nv_bfloat16* __restrict__ out,            // [T, H, DC]
                          float* __restrict__ lse,                    // [T, H]
                          long long num_kv_tokens) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = static_cast<int>(cluster_rank());   // dim slice owned by this CTA
  const int t = blockIdx.x / CL;                     // token
  TS(0); TSG(30);
  // exchange barrier init + relaxed cluster arrive before any memory traffic (the init fence must
  // not have to wait for outstanding loads); the matching wait is taken before the first gather.
  if (tid == 0) {
    mbar_init(&sm.xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  cluster_arrive_relaxed();
  TS(23);

  // ---- 1. index loads first; Q slice loads; speculative chunk-0 gather -----------------------
  const int4* gi = reinterpret_cast<const int4*>(indices + (size_t)t * TOPK);
  const int idx0 = indices[(size_t)t * TOPK + lane];   // chunk-0 index (same 128 B line in every warp)
  const int4 v0 = gi[tid];
  const int4 v1 = gi[tid + NT];
  int4 qv[2];
  {
    // Q slice: 16 rows x 18 segments = 288 x 16 B; thread i -> (row i/18, seg i%18), i = tid, tid+256
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int i = tid + k * NT;
      if (i < H * SEGS) {
        const int h = i / SEGS, seg = i - h * SEGS;
        const __nv_bfloat16* src = (seg < DCS / 8) ? q_nope + ((size_t)t * H + h) * DC + g * DCS + seg * 8
                                                   : q_pe + ((size_t)t * H + h) * DP + g * DPS + (seg - DCS / 8) * 8;
        qv[k] = *reinterpret_cast<const int4*>(src);
      }
    }
  }
  // gather geometry: thread -> row j = tid/8, segments s0 + 8k (k = 0..2), 18 segments per row
  const int gj = tid >> 3, gs0 = tid & 7;
  int4 kv[3];
  // per-thread gather base pointers, materialised before the index lands (the ckv/kpe parameter
  // loads and the slice offsets would otherwise sit between the shuffle and the gather loads)
  const __nv_bfloat16* rc_base = ckv + g * DCS + gs0 * 8;
  const __nv_bfloat16* rp_base = kpe + g * DPS + gs0 * 8;
  cluster_wait();   // peers' mbarriers initialised (overlaps the index DRAM trip)
  TS(26);
  const int ix0 = __shfl_sync(0xffffffffu, idx0, gj);
  TS(24);
  const bool g0_valid = (ix0 >= 0) && (ix0 < num_kv_tokens);
  if (g0_valid) {
    const __nv_bfloat16* rc = rc_base + (size_t)ix0 * DC;
    kv[0] = *reinterpret_cast<const int4*>(rc);
    kv[1] = *reinterpret_cast<const int4*>(rc + 64);
    if (gs0 < 2) kv[2] = *reinterpret_cast<const int4*>(rp_base + (size_t)ix0 * DP);
  }
  TS(25);
  const unsigned vb0 = __ballot_sync(0xffffffffu, (idx0 >= 0) && (idx0 < num_kv_tokens));
  // key pairs (2kp, 2kp+1) with at least one valid key: only their partial logits are exchanged
  // (bit 2kp); every CTA derives the same mask from the same index line, so the expected byte
  // count of the receiver matches what the senders push.
  const unsigned pair_mask0 = (vb0 | (vb0 >> 1)) & 0x55555555u;
  TS(10);

  // ---- 2. bookkeeping while loads are in flight -------------------------------------------
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int i = tid + k * NT;
    if (i < H * SEGS) {
      const int h = i / SEGS, seg = i - h * SEGS;
      *reinterpret_cast<int4*>(&sm.q[h][seg * 8]) = qv[k];
    }
  }
  reinterpret_cast<int4*>(sm.idx)[tid] = v0;
  reinterpret_cast<int4*>(sm.idx)[tid + NT] = v1;
  TS(11);
  // chunk validity flags (ballots over the register-resident index row); the compacted chunk
  // list is only needed after chunk 0, so on the fast path (chunk 0 has a valid key) both are
  // computed while the CTA waits for its peers in the first exchange.
  const bool any0 = (v0.x >= 0) | (v0.y >= 0) | (v0.z >= 0) | (v0.w >= 0);
  const bool any1 = (v1.x >= 0) | (v1.y >= 0) | (v1.z >= 0) | (v1.w >= 0);
  TS(1);
  const bool fast0 = (vb0 != 0u);
  if (!fast0) {
    chunk_flags(sm, any0, any1, warp, lane);
    __syncthreads();
    if (warp == 0) build_chunk_list(sm, lane);
    __syncthreads();
  }
  TS(2);
  TS(3);
  int num_chunks = fast0 ? 1 : sm.num_chunks;   // refreshed after chunk 0 on the fast path

  // per-thread softmax state: thread -> (head = tid>>4, key pair kp = tid&15)
  const int sh = tid >> 4, kp = tid & 15;
  float m_run = -INFINITY, l_part = 0.f;

  // PV accumulators: warp owns dims [warp*16, +16) of this CTA's 128 = 2 n-tiles
  // two accumulator sets (keys 0-15 / 16-31) keep the dependent mma chain one deep; summed at the end
  float acc[2][2][4];
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int i = 0; i < 2; ++i) { acc[k][i][0] = acc[k][i][1] = acc[k][i][2] = acc[k][i][3] = 0.f; }
  float acc_l[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};   // row sums of (bf16) P (all-ones tile MMA)

  // QK: warp w takes k-step w; warp 0 also takes k-step 8 (the kpe slice)
  const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
  const int lcol8 = 8 * (lane >> 4);
  const int mi = lane >> 3;

  // remote addresses of the peers' sx slots for our rank (double-buffered) and their barriers
  // Remote addresses of the peers' sx slots for our rank (buffer 0) and barriers.  mapa is linear
  // inside a CTA's window, so the buffer offset can be added at runtime.
  const unsigned sx_local = smem_u32(&sm.sx[0][g][sh][2 * kp]);
  const unsigned bar_local = smem_u32(&sm.xbar);
  constexpr unsigned SX_BUF_BYTES = sizeof(float) * CL * H * KC;
  const unsigned sx_p1 = mapa(sx_local, (g + 1) & 3), bar_p1 = mapa(bar_local, (g + 1) & 3);
  const unsigned sx_p2 = mapa(sx_local, (g + 2) & 3), bar_p2 = mapa(bar_local, (g + 2) & 3);
  const unsigned sx_p3 = mapa(sx_local, (g + 3) & 3), bar_p3 = mapa(bar_local, (g + 3) & 3);
  const __nv_bfloat16* qbase = &sm.q[lrow][lcol8];
  unsigned qa[4], qa2[4];

  for (int ci = 0; ci < num_chunks; ++ci) {
    const int c = (fast0 && ci == 0) ? 0 : sm.chunk_list[ci];
    const int* cidx = sm.idx + c * KC;
    const int buf = ci & 1;
    // ---- gather (chunk 0 was issued in the prologue) -----------------------------------------
    unsigned pair_mask = pair_mask0;
    {
      const int ix = (c == 0) ? ix0 : cidx[gj];
      const bool valid = (ix >= 0) && (ix < num_kv_tokens);
      if (c != 0) {
        const int ixl = cidx[lane];
        const unsigned vb = __ballot_sync(0xffffffffu, (ixl >= 0) && (ixl < num_kv_tokens));
        pair_mask = (vb | (vb >> 1)) & 0x55555555u;
      }
      if (valid && c != 0) {
        const __nv_bfloat16* rc = rc_base + (size_t)ix * DC;
        kv[0] = *reinterpret_cast<const int4*>(rc);
        kv[1] = *reinterpret_cast<const int4*>(rc + 64);
        if (gs0 < 2) kv[2] = *reinterpret_cast<const int4*>(rp_base + (size_t)ix * DP);
      }
      TS(12);
      if (fast0 && ci == 0) {   // chunk flags + list while the gather loads are in flight
        chunk_flags(sm, any0, any1, warp, lane);
        __syncthreads();
        if (warp == 0) build_chunk_list(sm, lane);
      }
      if (ci == 0) {   // Q A-fragments (loop-invariant): sm.q is visible after the sync above / before the loop
        ldsm_x4(qa[0], qa[1], qa[2], qa[3], qbase + warp * 16);
        if (warp < 2) ldsm_x4(qa2[0], qa2[1], qa2[2], qa2[3], qbase + (KSTEPS - 1) * 16);
      }
      if (valid) {
        *reinterpret_cast<int4*>(&sm.k[gj][gs0 * 8]) = kv[0];
        *reinterpret_cast<int4*>(&sm.k[gj][64 + gs0 * 8]) = kv[1];
        if (gs0 < 2) *reinterpret_cast<int4*>(&sm.k[gj][DCS + gs0 * 8]) = kv[2];
      } else {  // invalid row must be finite (P is exactly 0 there)
        *reinterpret_cast<int4*>(&sm.k[gj][gs0 * 8]) = make_int4(0, 0, 0, 0);
        *reinterpret_cast<int4*>(&sm.k[gj][64 + gs0 * 8]) = make_int4(0, 0, 0, 0);
        if (gs0 < 2) *reinterpret_cast<int4*>(&sm.k[gj][DCS + gs0 * 8]) = make_int4(0, 0, 0, 0);
      }
    }
    TS(4);
    __syncthreads();   // sm.q / sm.idx / sm.k visible
    TS(5);

    // ---- S_g partial = Q_g K_g^T: warp w takes k-step w for all 4 key tiles; the kpe k-step
    //      (8) is shared by warps 0 (tiles 0,1) and 1 (tiles 2,3).  Straight-line code so all
    //      ldmatrix loads issue before the MMAs (invalid rows are zero and masked later anyway).
    {
      float c4[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) { c4[n][0] = c4[n][1] = c4[n][2] = c4[n][3] = 0.f; }
      const __nv_bfloat16* kbase = &sm.k[(lane & 7) + 8 * (mi >> 1)][8 * (mi & 1)];
      const unsigned* a = qa;
      unsigned b0, b1, b2, b3, b4, b5, b6, b7;
      ldsm_x4(b0, b1, b2, b3, kbase + warp * 16);
      ldsm_x4(b4, b5, b6, b7, kbase + 16 * SROW + warp * 16);
      TS(27);
      // NB: accumulator indices must be compile-time constants (a warp-dependent index would put
      // the accumulators in local memory).
      if (warp < 2) {
        const unsigned* a2 = qa2;
        unsigned e0, e1, e2, e3;
        ldsm_x4(e0, e1, e2, e3, kbase + warp * 16 * SROW + (KSTEPS - 1) * 16);
        mma_16816(c4[0], a, b0, b1);
        mma_16816(c4[1], a, b2, b3);
        mma_16816(c4[2], a, b4, b5);
        mma_16816(c4[3], a, b6, b7);
        float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
        mma_16816(d0, a2, e0, e1);
        mma_16816(d1, a2, e2, e3);
        if (warp == 0) {
#pragma unroll
          for (int e = 0; e < 4; ++e) { c4[0][e] += d0[e]; c4[1][e] += d1[e]; }
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e) { c4[2][e] += d0[e]; c4[3][e] += d1[e]; }
        }
      } else {
        mma_16816(c4[0], a, b0, b1);
        mma_16816(c4[1], a, b2, b3);
        mma_16816(c4[2], a, b4, b5);
        mma_16816(c4[3], a, b6, b7);
      }
      TS(16);
      const int r = lane >> 2, cc = (lane & 3) * 2;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        *reinterpret_cast<float2*>(&sm.sp[warp][r][n * 8 + cc]) = make_float2(c4[n][0], c4[n][1]);
        *reinterpret_cast<float2*>(&sm.sp[warp][r + 8][n * 8 + cc]) = make_float2(c4[n][2], c4[n][3]);
      }
      TS(29);
    }
    __syncthreads();
    TS(6);

    // ---- reduce the 8 warp partials and push the CTA partial to the 3 peers ------------------
    float2 s_own;
    {
      const int k0 = 2 * kp;
      float2 v[NWARP];
#pragma unroll
      for (int w = 0; w < NWARP; ++w) v[w] = *reinterpret_cast<const float2*>(&sm.sp[w][sh][k0]);
      const float2 s = make_float2(((v[0].x + v[1].x) + (v[2].x + v[3].x)) + ((v[4].x + v[5].x) + (v[6].x + v[7].x)),
                                   ((v[0].y + v[1].y) + (v[2].y + v[3].y)) + ((v[4].y + v[5].y) + (v[6].y + v[7].y)));
      s_own = s;
      if (tid == 0) mbar_expect_tx(&sm.xbar, (CL - 1) * H * __popc(pair_mask) * 8);
      if ((pair_mask >> (2 * kp)) & 1u) {   // pairs without a valid key are masked by the receiver anyway
        const unsigned off = buf ? SX_BUF_BYTES : 0u;
        st_async_f2(sx_p1 + off, s.x, s.y, bar_p1);
        st_async_f2(sx_p2 + off, s.x, s.y, bar_p2);
        st_async_f2(sx_p3 + off, s.x, s.y, bar_p3);
      }
    }
    TS(17);
    // V fragments for PV do not depend on the softmax: load them while the peers' data is in flight
    unsigned vb[8];
    {
      const __nv_bfloat16* vbase = &sm.k[(lane & 7) + 8 * (mi & 1)][warp * 16 + 8 * (mi >> 1)];
      ldsm_x4_trans(vb[0], vb[1], vb[2], vb[3], vbase);
      ldsm_x4_trans(vb[4], vb[5], vb[6], vb[7], vbase + 16 * SROW);
    }
    mbar_wait_cluster(&sm.xbar, (unsigned)(ci & 1));
    if (fast0 && ci == 0) num_chunks = sm.num_chunks;   // written by warp 0 before the gather sync
    TS(18);

    // ---- online softmax (log2 domain), identical in all 4 CTAs ------------------------------
    {
      const int k0 = 2 * kp;
      const int ix0c = cidx[k0], ix1c = cidx[k0 + 1];
      const bool vv0 = (ix0c >= 0) && (ix0c < num_kv_tokens);
      const bool vv1 = (ix1c >= 0) && (ix1c < num_kv_tokens);
      float2 s = make_float2(0.f, 0.f);
#pragma unroll
      for (int r = 0; r < CL; ++r) {   // fixed summation order in every CTA -> identical softmax
        const float2 v = (r == g) ? s_own : *reinterpret_cast<const float2*>(&sm.sx[buf][r][sh][k0]);
        s.x += v.x; s.y += v.y;
      }
      TSF(2, s.x + s.y);
      const float s0 = vv0 ? s.x * scale_log2 : -INFINITY;
      const float s1 = vv1 ? s.y * scale_log2 : -INFINITY;
      // per-head max over the 16 lanes of the head: two full-warp redux ops (a partial-mask redux
      // was measured ~150 cycles slower)
      const float mxx = fmaxf(s0, s1);
      const float mlo = redux_max_f32((lane < 16) ? mxx : -INFINITY, 0xffffffffu);
      const float mhi = redux_max_f32((lane < 16) ? -INFINITY : mxx, 0xffffffffu);
      const float m_new = fmaxf(m_run, (lane < 16) ? mlo : mhi);
      // m_new == -inf only if no key so far was valid (out-of-range indices): all p = 0, alpha = 1
      TSF(6, m_new);
      const bool none = (m_new == -INFINITY);
      const float alpha = none ? 1.f : ex2_approx(m_run - m_new);
      const float p0 = (vv0 && !none) ? ex2_approx(s0 - m_new) : 0.f;
      const float p1 = (vv1 && !none) ? ex2_approx(s1 - m_new) : 0.f;
      m_run = m_new;
      const __nv_bfloat162 pb = __floats2bfloat162_rn(p0, p1);
      *reinterpret_cast<__nv_bfloat162*>(&sm.p[sh][k0]) = pb;
      l_part = l_part * alpha + (p0 + p1);
      if (kp == 0) sm.alpha[sh] = alpha;
      TS(19);
    }
    __syncthreads();
    TS(7);

    // ---- O_g = alpha * O_g + P V_g ; warp owns dims [warp*16, +16) --------------------------
    {
      if (ci > 0) {
        const float a_lo = sm.alpha[lane >> 2], a_hi = sm.alpha[(lane >> 2) + 8];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
#pragma unroll
          for (int i = 0; i < 2; ++i) { acc[k][i][0] *= a_lo; acc[k][i][1] *= a_lo; acc[k][i][2] *= a_hi; acc[k][i][3] *= a_hi; }
          acc_l[k][0] *= a_lo; acc_l[k][1] *= a_lo; acc_l[k][2] *= a_hi; acc_l[k][3] *= a_hi;
        }
      }
      unsigned a0[4], a1[4];
      ldsm_x4(a0[0], a0[1], a0[2], a0[3], &sm.p[lrow][lcol8]);
      ldsm_x4(a1[0], a1[1], a1[2], a1[3], &sm.p[lrow][16 + lcol8]);
      mma_16816(acc[0][0], a0, vb[0], vb[1]);
      mma_16816(acc[0][1], a0, vb[2], vb[3]);
      mma_16816(acc[1][0], a1, vb[4], vb[5]);
      mma_16816(acc[1][1], a1, vb[6], vb[7]);
      // all-ones B tile: every column of acc_l is the row sum of P, so every lane holds its rows' sums
      const unsigned b_ones = 0x3f803f80u;
      mma_16816(acc_l[0], a0, b_ones, b_ones);
      mma_16816(acc_l[1], a1, b_ones, b_ones);
    }
    TS(20);
    if (kp == 0) sm.alpha[sh] = m_run;                 // m per head for the lse (alpha[] is free now)
    if (ci + 1 < num_chunks) __syncthreads();          // sm.k / sm.p / sm.sp reuse in the next chunk
    TS(8);
  }

  // ---- epilogue: O = (acc0 + acc1) / l, staged through the (free) sp region, coalesced stores ----
  {
    // row sums of P (all-ones MMA): rows r (acc_l[.][0]) and r+8 (acc_l[.][2]) in every lane
    const float l_lo = acc_l[0][0] + acc_l[1][0];
    const float l_hi = acc_l[0][2] + acc_l[1][2];
    const float inv_lo = (l_lo > 0.f) ? rcp_approx(l_lo) : 0.f;
    const float inv_hi = (l_hi > 0.f) ? rcp_approx(l_hi) : 0.f;
    TS(14);
    const int r = lane >> 2;
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(&sm.sp[0][0][0]);  // [H][OROW]
    static_assert(H * OROW * 2 <= sizeof(float) * NWARP * H * SPROW, "staging tile must fit in sp");
    __nv_bfloat16* s_lo = stage + r * OROW + warp * 16 + (lane & 3) * 2;
    __nv_bfloat16* s_hi = s_lo + 8 * OROW;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      *reinterpret_cast<__nv_bfloat162*>(s_lo + i * 8) =
          __floats2bfloat162_rn((acc[0][i][0] + acc[1][i][0]) * inv_lo, (acc[0][i][1] + acc[1][i][1]) * inv_lo);
      *reinterpret_cast<__nv_bfloat162*>(s_hi + i * 8) =
          __floats2bfloat162_rn((acc[0][i][2] + acc[1][i][2]) * inv_hi, (acc[0][i][3] + acc[1][i][3]) * inv_hi);
    }
    TS(21);
    __syncthreads();
    TS(22);
    // 16 rows x 256 B = 256 x 16 B -> one coalesced store per thread
    {
      const int h = tid >> 4, seg = tid & 15;
      *reinterpret_cast<int4*>(out + ((size_t)t * H + h) * DC + g * DCS + seg * 8) =
          *reinterpret_cast<const int4*>(stage + h * OROW + seg * 8);
    }
    // lse = m + log2(l) by rank 0, warp 7, after its output store (off the critical path)
    if (g == 0 && warp == NWARP - 1 && (lane & 3) == 0) {
      lse[(size_t)t * H + r] = (l_lo > 0.f) ? (sm.alpha[r] + __log2f(l_lo)) : -INFINITY;
      lse[(size_t)t * H + r + 8] = (l_hi > 0.f) ? (sm.alpha[r + 8] + __log2f(l_hi)) : -INFINITY;
    }
  }
  TS(9); TSG(31);
}

}  // namespace

void dsa_sparse_attention(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
                          torch::Tensor kpe_cache, torch::Tensor sparse_indices, double sm_scale,
                          torch::Tensor output, torch::Tensor lse) {
  TORCH_CHECK(q_nope.is_cuda() && q_nope.dtype() == torch::kBFloat16 && q_nope.is_contiguous());
  TORCH_CHECK(q_pe.is_contiguous() && ckv_cache.is_contiguous() && kpe_cache.is_contiguous());
  TORCH_CHECK(sparse_indices.dtype() == torch::kInt32 && sparse_indices.is_contiguous());
  TORCH_CHECK(output.is_contiguous() && lse.is_contiguous());
  TORCH_CHECK(q_nope.size(1) == H && q_nope.size(2) == DC && q_pe.size(2) == DP);
  TORCH_CHECK(ckv_cache.size(1) == PS && sparse_indices.size(1) == TOPK);
  static bool attr_set = false;
  constexpr int smem = sizeof(Smem);
  if (!attr_set) {
    cudaFuncSetAttribute(dsa_sparse_attn_v13_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int T = static_cast<int>(q_nope.size(0));
  const long long num_kv_tokens = (long long)ckv_cache.size(0) * PS;
  const float scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
  dsa_sparse_attn_v13_kernel<<<T * CL, NT, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()), sparse_indices.data_ptr<int>(),
      scale_log2, reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), lse.data_ptr<float>(),
      num_kv_tokens);
}

torch::Tensor read_timing(int num_ctas) {
  auto t = torch::zeros({num_ctas, 32}, torch::dtype(torch::kInt64));
#ifdef DSA_TIMING
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(t.data_ptr<int64_t>(), g_dsa_ts, sizeof(long long) * 32 * num_ctas);
#endif
  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsa_sparse_attention", &dsa_sparse_attention, "DSA sparse attention (v6, 4-CTA cluster)");
  m.def("read_timing", &read_timing, "read per-CTA clock64 phase stamps (dev builds only)");
}
