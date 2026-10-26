// Single-token (T=1) DeepSeek-V3 MoE FP8 block-scale decode kernel for Blackwell (sm_100a).
// Written from scratch. One persistent, warp-specialised kernel:
//   * every block redundantly computes the no-aux routing (256 logits) -> list of local hit experts
//   * work queue: a GEMM1 queue (128 items per hit: 128 rows x K/4) and per-hit GEMM2 queues
//     (56 items per hit: 128 rows x 2048) that are handed out only once the hit's SwiGLU output is
//     ready; weights stream through a TMA (cp.async.bulk.tensor) smem ring of STAGES x (128 rows x
//     256 B). Producer warp: lane L owns ring slot L, so a round of stages costs one latency.
//   * 8 consumer warps (16 rows each) run f16 x f16 -> fp32 mma.sync (m16n8k16); fp8 weights are
//     converted with cvt.rn.f16x2.e4m3x2. The k-block stream is software pipelined: while k-block
//     j's MMAs run, k-block j+1 is loaded and converted (keeps the XU and tensor pipes both busy).
//   * GEMM1/GEMM2 items write K-split partial sums as tagged 16-byte records (3 floats + the launch
//     epoch); readers validate the tags themselves, so no fence, counter or completion wait sits on the
//     hot path. Every block that needs an expert's intermediate computes SwiGLU + block-scaled f16
//     itself into shared memory; output tiles are summed/converted to bf16 by one block per tile.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

namespace {

constexpr int H = 7168, I = 2048, I2 = 4096, E_GLOBAL = 256, E_LOCAL = 32;
constexpr int BLK = 128;
constexpr int KB1 = H / BLK;        // 56 k-blocks of W13
constexpr int KB2 = I / BLK;        // 16 k-blocks of W2
constexpr int RB1 = I2 / BLK;       // 32 row-blocks of W13
constexpr int RB2 = H / BLK;        // 56 row-blocks of W2
constexpr int KPS = 2;              // k-blocks per stage
constexpr int G1_KSPLIT = 4;
constexpr int G1_NKB = KB1 / G1_KSPLIT;            // 14 k-blocks per item
constexpr int G1_NK = G1_NKB / KPS;                // 7 stages per item
constexpr int G1_IPH = RB1 * G1_KSPLIT;            // 128 items per hit
constexpr int G2_KSPLIT = 2;
constexpr int G2_NKB = KB2 / G2_KSPLIT;            // 8 k-blocks per item
constexpr int G2_NK = G2_NKB / KPS;                // 4 stages per item
constexpr int G2_IPH = RB2 * G2_KSPLIT;            // 112 items per hit
constexpr int MAXHITS = 8;
constexpr int NCONS = 8;                           // consumer warps (16 rows each)
constexpr int NCONS_THREADS = NCONS * 32;          // 256
constexpr int NTHREADS = NCONS_THREADS + 32;       // + 1 producer warp
constexpr int STAGES = 6;
constexpr int STAGE_ROW = BLK * KPS;               // 256 B per row per stage
constexpr int STAGE_BYTES = BLK * STAGE_ROW;       // 32 KB
constexpr float C_TARGET = 1024.f;                 // f16 magnitude target for the intermediate blocks
// a producer round publishes min(STAGES, stages available in the first 3 queued items) stages

// Tagged record: 8 consecutive rows are stored as 3 float4 chunks {3 values, epoch tag} (last chunk pads).
struct Workspace {
  float4 p_rec[G1_KSPLIT][MAXHITS][I2 / 8][3];     // GEMM1 K-split partial sums
  float4 o_rec[MAXHITS * G2_KSPLIT][H / 8][3];     // GEMM2 partial outputs (routing weight applied)
  float4 c_rec[MAXHITS][I / 4];                    // published intermediate: {f16x2, f16x2, block scale, tag} per 4 values
  int q1;                 // GEMM1 work queue
  int q2[MAXHITS];        // per-hit GEMM2 work queues (block b prefers a ready hit, steals when exhausted)
  int blocks_done;
  int g1_done[MAXHITS];   // readiness hint only: warps that stored GEMM1 partials (relaxed increments)
  int cb_cnt[MAXHITS][KB2];   // per (expert, 128-block of the intermediate): warps that stored partials (target 64)
};

struct StageMeta { int item; int kidx; float sc[KPS]; };

#ifndef MOE_DEBUG
#define MOE_DEBUG 0
#endif
#ifndef MOE_MODE
#define MOE_MODE 0   // 0 normal, 1 producer skips TMA (compute-only), 2 consumers skip math (stream-only)
#endif
constexpr int DBG_PER_BLOCK = 4 + 3 * 64 + 2 * 256 + 256 + 32;
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

struct __align__(1024) Smem {
  uint8_t stage[STAGES][STAGE_BYTES];
  uint16_t act[H];                // hidden_states as exact f16
  uint16_t c[2][I];               // GEMM2 B operand (block-scaled f16), double buffered by G2 item
  float s_c[2][KB2];
  float hscale[KB1];
  StageMeta meta[STAGES];
  uint64_t full[STAGES];
  uint64_t empty[STAGES];
  // routing scratch
  float s[E_GLOBAL];
  float swb[E_GLOBAL];
  float gscore[8];
  int gmask[8];
  float cval[8][8];
  int cidx[8][8];
  int topk[8];
  int hits[MAXHITS];
  float hitw[MAXHITS];
  int nhits;
  int flag;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n .reg .pred p;\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      " selp.u32 %0, 1, 0, p;\n}"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) { }
}
#ifndef MOE_L2_HINT
#define MOE_L2_HINT 1   // 1: TMA loads carry an evict_first L2 policy (streamed weights should not displace other lines)
#endif
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
  uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol)); return pol;
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* tm, uint64_t* bar, int x, int y, uint64_t pol) {
#if MOE_L2_HINT
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(x), "r"(y), "l"(pol)
      : "memory");
#else
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(x), "r"(y)
      : "memory");
#endif
}
__device__ __forceinline__ void mma_f16(float& d0, float& d1, float& d2, float& d3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_relaxed_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void consumer_bar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void red_relaxed_add(int* p, int v) { asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release_gpu(int* p, int v) { asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// two fp8 e4m3 (low / high half of a word) -> exact f16x2
__device__ __forceinline__ uint32_t f8x2_lo(uint32_t w) { uint32_t r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(w & 0xffffu))); return r; }
__device__ __forceinline__ uint32_t f8x2_hi(uint32_t w) { uint32_t r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(w >> 16))); return r; }

// A fragments of one 128-wide k-block for this warp's 16 rows: 2 x 64 B chunks, this thread owns
// bytes [16*tig, +16) of rows g and g+8 in each -> 8 mma.m16n8k16 worth of (a0,a1,a2,a3).
struct KBlockA { uint32_t a[8][4]; };

__device__ __forceinline__ void load_convert_kblock(KBlockA& A, const uint8_t* wrow, int row_stride) {
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const uint4 q = lds128(wrow + 64 * j);
    const uint4 r = lds128(wrow + 8 * row_stride + 64 * j);
    const uint32_t qw[4] = {q.x, q.y, q.z, q.w}, rw[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int m = 0; m < 4; m++) {
      A.a[4 * j + m][0] = f8x2_lo(qw[m]); A.a[4 * j + m][1] = f8x2_lo(rw[m]);
      A.a[4 * j + m][2] = f8x2_hi(qw[m]); A.a[4 * j + m][3] = f8x2_hi(rw[m]);
    }
  }
}
// 8 independent MMAs against the 32 f16 activations of the k-block, summed and scaled into the row accumulators.
__device__ __forceinline__ void mma_kblock(float& acc0, float& acc2, const KBlockA& A, const uint16_t* brow, float sc) {
  float d[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++) { d[i][0] = 0.f; d[i][1] = 0.f; d[i][2] = 0.f; d[i][3] = 0.f; }
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const uint4 bA = lds128(brow + 64 * j);
    const uint4 bB = lds128(brow + 64 * j + 8);
    const uint32_t bw[8] = {bA.x, bA.y, bA.z, bA.w, bB.x, bB.y, bB.z, bB.w};
#pragma unroll
    for (int m = 0; m < 4; m++)
      mma_f16(d[4 * j + m][0], d[4 * j + m][1], d[4 * j + m][2], d[4 * j + m][3],
              A.a[4 * j + m][0], A.a[4 * j + m][1], A.a[4 * j + m][2], A.a[4 * j + m][3], bw[2 * m], bw[2 * m + 1]);
  }
  float s0 = ((d[0][0] + d[1][0]) + (d[2][0] + d[3][0])) + ((d[4][0] + d[5][0]) + (d[6][0] + d[7][0]));
  float s2 = ((d[0][2] + d[1][2]) + (d[2][2] + d[3][2])) + ((d[4][2] + d[5][2]) + (d[6][2] + d[7][2]));
  acc0 += s0 * sc; acc2 += s2 * sc;
}

// ---------------------------------------------------------------------------------------------
// DeepSeek-V3 no-aux routing, computed by threads 0..255 (expert = thread, group = warp).
__device__ void routing(Smem& sm, const float* __restrict__ logits, const __nv_bfloat16* __restrict__ bias,
                        int offset, float rsf, unsigned long long* rstamp) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
#if MOE_DEBUG
#define RSTAMP(i) if (t == 0) rstamp[i] = gtimer();
#else
#define RSTAMP(i)
#endif
  RSTAMP(0)
  float s = 0.f, swb = -INFINITY;
  if (t < E_GLOBAL) {
    s = 1.0f / (1.0f + expf(-logits[t]));
    swb = s + __bfloat162float(bias[t]);
    sm.s[t] = s;
    float m1 = warp_max(swb);
    unsigned b = __ballot_sync(0xffffffffu, swb == m1);
    int first = __ffs(b) - 1;
    float v2 = (lane == first) ? -INFINITY : swb;
    float m2 = warp_max(v2);
    if (lane == 0) sm.gscore[warp] = m1 + m2;
  }
  __syncthreads();
  RSTAMP(1)
  if (t < 8) {
    float my = sm.gscore[t];
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) { float o = sm.gscore[j]; rank += (o > my) || (o == my && j < t); }
    sm.gmask[t] = rank < 4;
  }
  __syncthreads();
  RSTAMP(2)
  if (t < E_GLOBAL) {
    if (sm.gmask[warp]) {
      // per selected group: its top-8 experts (value, index) via 8 argmax rounds
      float v = swb;
#pragma unroll
      for (int r = 0; r < 8; r++) {
        float m = warp_max(v);
        unsigned b = __ballot_sync(0xffffffffu, v == m);
        int first = __ffs(b) - 1;
        if (lane == first) { sm.cval[warp][r] = m; sm.cidx[warp][r] = t; v = -INFINITY; }
      }
    } else if (lane < 8) { sm.cval[warp][lane] = -INFINITY; sm.cidx[warp][lane] = 0x7fffffff; }
  }
  __syncthreads();
  RSTAMP(3)
  if (t < E_GLOBAL) {
    // warp w ranks the 8 candidates of group w against all 64 (ties -> lower expert index wins):
    // lane holds candidates lane and lane+32, each candidate's rank is a popcount of two ballots
    const float* cv = &sm.cval[0][0]; const int* ci = &sm.cidx[0][0];
    const float v0 = cv[lane], v1 = cv[lane + 32];
    const int i0 = ci[lane], i1 = ci[lane + 32];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const float c = sm.cval[warp][r]; const int cidx = sm.cidx[warp][r];
      const unsigned b0 = __ballot_sync(0xffffffffu, (v0 > c) || (v0 == c && i0 < cidx));
      const unsigned b1 = __ballot_sync(0xffffffffu, (v1 > c) || (v1 == c && i1 < cidx));
      const int rank = __popc(b0) + __popc(b1);
      if (lane == 0 && rank < 8 && cidx != 0x7fffffff) sm.topk[rank] = cidx;
    }
  }
  RSTAMP(7)
  __syncthreads();
  RSTAMP(5)
  if (t == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) sum += sm.s[sm.topk[i]];
    float inv = rsf / (sum + 1e-20f);
    int n = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      int e = sm.topk[i];
      if (e >= offset && e < offset + E_LOCAL) { sm.hits[n] = e - offset; sm.hitw[n] = sm.s[e] * inv; n++; }
    }
    sm.nhits = n;
    RSTAMP(6)
  }
  __syncthreads();
  RSTAMP(4)
}

struct ItemInfo { int phase; int slot; int tile; int kpart; int nk; };
// item ids: [0, nhits*G1_IPH) GEMM1 (hit = id % nhits), then [nhits*G1_IPH, total) GEMM2 (slot-major)
__device__ __forceinline__ ItemInfo decode_item(int item, int nhits) {
  ItemInfo it;
  const int n1 = nhits * G1_IPH;
  item = max(item, 0);
  if (item < n1) {
    it.phase = 1; it.slot = item / G1_IPH; int r = item - it.slot * G1_IPH;
    it.tile = r / G1_KSPLIT; it.kpart = r - it.tile * G1_KSPLIT; it.nk = G1_NK;
  } else {
    item -= n1;
    it.phase = 2; it.slot = item / G2_IPH; int r = item - it.slot * G2_IPH;
    it.tile = r / G2_KSPLIT; it.kpart = r - it.tile * G2_KSPLIT; it.nk = G2_NK;
  }
  return it;
}

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS, 1)
moe_kernel(const __grid_constant__ CUtensorMap tm1, const __grid_constant__ CUtensorMap tm2,
           const float* __restrict__ logits, const __nv_bfloat16* __restrict__ bias,
           const uint8_t* __restrict__ hidden, const float* __restrict__ hscale,
           const float* __restrict__ w1s, const float* __restrict__ w2s,
           int offset, float rsf, Workspace* ws, __nv_bfloat16* __restrict__ out, unsigned epoch, unsigned long long* dbg) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (t == 0 && (smem_u32(smem_raw) & 1023) != 0) { printf("moe_kernel: dynamic smem not 1024-aligned\n"); __trap(); }
#if MOE_DEBUG
  unsigned long long* dbgb = dbg + blockIdx.x * DBG_PER_BLOCK;
  if (t == 0) dbgb[0] = gtimer();
  int dbg_n = 0;
#endif

  // token vector (exact f16) + its scales -> smem, by the producer warp while the others route.
  // All loads are issued first (one DRAM latency), then converted.
  if (warp == NCONS) {
    constexpr int NCH = H / 16 / 32;   // 14 x 16 B chunks per lane
    uint4 v[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) v[i] = __ldg(reinterpret_cast<const uint4*>(hidden) + lane + 32 * i);
    float hs0 = (lane < KB1) ? __ldg(hscale + lane) : 0.f;
    float hs1 = (lane + 32 < KB1) ? __ldg(hscale + lane + 32) : 0.f;
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      const int c = lane + 32 * i;
      uint4 o0, o1;
      o0.x = f8x2_lo(v[i].x); o0.y = f8x2_hi(v[i].x); o0.z = f8x2_lo(v[i].y); o0.w = f8x2_hi(v[i].y);
      o1.x = f8x2_lo(v[i].z); o1.y = f8x2_hi(v[i].z); o1.z = f8x2_lo(v[i].w); o1.w = f8x2_hi(v[i].w);
      reinterpret_cast<uint4*>(sm.act)[2 * c] = o0;
      reinterpret_cast<uint4*>(sm.act)[2 * c + 1] = o1;
    }
    if (lane < KB1) sm.hscale[lane] = hs0;
    if (lane + 32 < KB1) sm.hscale[lane + 32] = hs1;
    if (lane == 0) {
      for (int s = 0; s < STAGES; s++) { mbar_init(&sm.full[s], 1); mbar_init(&sm.empty[s], NCONS); }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#ifndef MOE_NO_PREFETCH
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm1)) : "memory");
      asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tm2)) : "memory");
#endif
    }
  }
#if MOE_DEBUG
  routing(sm, logits, bias, offset, rsf, dbgb + 4 + 3 * 64 + 2 * 256 + 256);
#else
  routing(sm, logits, bias, offset, rsf, nullptr);
#endif
  const int nhits = sm.nhits;
  const int n1 = nhits * G1_IPH;
  const int total = nhits * (G1_IPH + G2_IPH);
#if MOE_DEBUG
  if (t == 0) dbgb[1] = gtimer();
#endif

  if (warp == NCONS) {
    // ===================== producer warp: lane L (< STAGES) owns ring slot L =====================
    // Item sequence: GEMM1 items from one global queue, then GEMM2 items from per-hit queues (this block
    // prefers hit blockIdx.x % nhits so that it rarely has to switch experts, and steals from the other
    // hits' queues once its own is exhausted). Claims are issued one round ahead of their use.
    // Consumers wait for a hit's intermediate themselves, so the producer never blocks on others.
    const uint64_t l2pol = l2_evict_first_policy();
    constexpr int QD = 3;
    int qid[QD]; ItemInfo qit[QD]; float qsc[QD]; int qn = 0;
    int cur_rem = 0;                     // stages of qid[0] not yet published
    bool terminal = false, inflight = false, g1_exhausted = false;
    int v_inflight = 0;
    int pref = (nhits > 0) ? (int)(blockIdx.x % nhits) : 0;   // preferred GEMM2 hit
    unsigned exhausted = 0;              // bit h: q2[h] exhausted
    int seen = 0;                        // last observed value of the active queue counter (one round stale)
    auto peek_queue = [&]() {            // refresh `seen` (relaxed load, consumed next round)
      int v = 0;
      if (lane == 0) v = ld_relaxed_gpu(g1_exhausted ? &ws->q2[pref] : &ws->q1);
      seen = __shfl_sync(0xffffffffu, v, 0);
    };
    // lane kb holds the combined scale of k-block kb of the item
    auto load_sc = [&](int item, const ItemInfo& it) -> float {
      int e = sm.hits[it.slot];
      if (it.phase == 1) {
        if (lane < G1_NKB) { int kb = it.kpart * G1_NKB + lane; return __ldg(w1s + (e * RB1 + it.tile) * KB1 + kb) * sm.hscale[kb]; }
      } else {
        if (lane < G2_NKB) { int kb = it.kpart * G2_NKB + lane; return __ldg(w2s + (e * RB2 + it.tile) * KB2 + kb); }
      }
      return 0.f;
    };
    auto push = [&](int id) {
      qid[qn] = id; qit[qn] = decode_item(id, nhits); qsc[qn] = load_sc(id, qit[qn]);
      if (qn == 0) cur_rem = qit[0].nk;
      qn++;
    };
    auto resolve = [&]() {
      if (!inflight) return;
      inflight = false;
      const int vv = __shfl_sync(0xffffffffu, v_inflight, 0);
      if (!g1_exhausted) {
        if (vv < n1) push(vv);
        else {
          g1_exhausted = true;
          if (nhits == 0) { terminal = true; return; }
          // prefer a hit whose GEMM1 partials are already complete (spread blocks over the ready hits);
          // if none is ready yet, take the one that will complete first (hits complete in slot order)
          int ready_mask = 0;
          if (lane < nhits) ready_mask = (ld_acquire_gpu(&ws->g1_done[lane]) >= NCONS * G1_IPH) ? 1 : 0;
          ready_mask = __ballot_sync(0xffffffffu, ready_mask != 0) & ((1u << nhits) - 1u);
          if (ready_mask != 0) {
            const int nready = __popc(ready_mask);
            int k = (int)(blockIdx.x % nready);
            int m = ready_mask;
            for (int q = 0; q < k; q++) m &= m - 1;      // drop the k lowest set bits
            pref = __ffs(m) - 1;
          } else pref = 0;
        }
      } else {
        if (vv < G2_IPH) push(n1 + pref * G2_IPH + vv);
        else {
          exhausted |= 1u << pref;
          if (exhausted == ((1u << nhits) - 1u)) terminal = true;
          else {
            // next queue: prefer hits whose intermediate is ready (so that unready hits' items are left
            // for blocks that have nothing else to do), rotating from the current one
            int rdy = 0;
            if (lane < nhits) rdy = (ld_acquire_gpu(&ws->g1_done[lane]) >= NCONS * G1_IPH) ? 1 : 0;
            const unsigned ready_mask = __ballot_sync(0xffffffffu, rdy != 0) & ((1u << nhits) - 1u) & ~exhausted;
            const unsigned any_mask = ((1u << nhits) - 1u) & ~exhausted;
            const unsigned pick_from = ready_mask ? ready_mask : any_mask;
            int np = pref;
            do { np = (np + 1) % nhits; } while (!(pick_from & (1u << np)));
            pref = np;
          }
        }
      }
    };
    auto issue = [&]() {
      if (terminal || inflight || qn > QD - 1 - 1) return;   // keep up to two items queued behind the current one
      // near the end of a queue, do not hoard: claim ahead only while more items remain than there are blocks
      const int remaining = (g1_exhausted ? G2_IPH : n1) - seen;
      if (qn >= 1 && remaining <= (int)gridDim.x) return;
      if (lane == 0) v_inflight = atomicAdd(g1_exhausted ? &ws->q2[pref] : &ws->q1, 1);
      inflight = true;
    };
    // first item synchronously (one atomic latency), the next claim in flight
    issue(); resolve(); peek_queue(); issue();
    uint32_t seq = 0;                    // stages published so far
    while (true) {
      if (qn == 0) {
        if (terminal) {
          const int s = seq % STAGES; const uint32_t par = ((seq / STAGES) & 1) ^ 1;
          if (lane == 0) {
            mbar_wait(&sm.empty[s], par);
            sm.meta[s].item = -1;
            mbar_arrive(&sm.full[s]);
          }
          break;
        }
        resolve(); peek_queue(); issue();   // waits for the in-flight claim
        continue;
      }
      int lim[3];
      lim[0] = cur_rem;
#pragma unroll
      for (int k = 1; k < 3; k++) lim[k] = lim[k - 1] + ((k < qn) ? qit[k].nk : 0);
      const int take = min(STAGES, lim[2]);     // a round spans at most 3 items (nk >= 4)
      const bool active = lane < take;
      const int which = (lane < lim[0]) ? 0 : ((lane < lim[1]) ? 1 : 2);
      const int kidx = (which == 0) ? (qit[0].nk - cur_rem + lane) : (lane - lim[which - 1]);
      const int item = qid[which];
      const ItemInfo it = qit[which];
      const int ka = (KPS * kidx) & 31, kb_ = (KPS * kidx + 1) & 31;
      const float a0 = __shfl_sync(0xffffffffu, qsc[0], ka), b0 = __shfl_sync(0xffffffffu, qsc[0], kb_);
      const float a1 = __shfl_sync(0xffffffffu, qsc[1], ka), b1 = __shfl_sync(0xffffffffu, qsc[1], kb_);
      const float a2 = __shfl_sync(0xffffffffu, qsc[2], ka), b2 = __shfl_sync(0xffffffffu, qsc[2], kb_);
      if (active) {
        const uint32_t si = seq + lane;                 // this lane's stage index
        const int s = si % STAGES; const uint32_t par = ((si / STAGES) & 1) ^ 1;
        mbar_wait(&sm.empty[s], par);
#if MOE_DEBUG
        if (si < 256) dbgb[4 + 3 * 64 + 2 * 256 + si] = gtimer();
#endif
        sm.meta[s].item = item; sm.meta[s].kidx = kidx;
        sm.meta[s].sc[0] = (which == 0) ? a0 : ((which == 1) ? a1 : a2);
        sm.meta[s].sc[1] = (which == 0) ? b0 : ((which == 1) ? b1 : b2);
#if MOE_MODE == 1
        mbar_arrive(&sm.full[s]);
#else
        mbar_arrive_expect_tx(&sm.full[s], STAGE_BYTES);
        const int e = sm.hits[it.slot];
        if (it.phase == 1) tma_load_2d(sm.stage[s], &tm1, &sm.full[s], (it.kpart * G1_NKB + KPS * kidx) * BLK, e * I2 + it.tile * BLK, l2pol);
        else               tma_load_2d(sm.stage[s], &tm2, &sm.full[s], (it.kpart * G2_NKB + KPS * kidx) * BLK, e * H + it.tile * BLK, l2pol);
#endif
      }
      __syncwarp();
      seq += take;
      // consume `take` stages from the front of the queue, popping finished items
      int used = take;
      while (true) {
        const int c = min(used, cur_rem);
        cur_rem -= c; used -= c;
        if (cur_rem > 0) break;
#pragma unroll
        for (int k = 1; k < QD; k++) { qid[k - 1] = qid[k]; qit[k - 1] = qit[k]; qsc[k - 1] = qsc[k]; }
        qn--;
        cur_rem = (qn > 0) ? qit[0].nk : 0;
        if (qn == 0) break;
      }
      resolve(); peek_queue(); issue();
    }
    __syncwarp();
  } else {
    // ===================== consumer warps: software-pipelined k-block stream =====================
    const int g = lane >> 2, tig = lane & 3;
    const uint8_t* const wbase = sm.stage[0] + (warp * 16 + g) * STAGE_ROW + tig * 16;
    float acc0 = 0.f, acc2 = 0.f;
    // state of the k-block being loaded/converted ("next") and the one being multiplied ("cur")
    ItemInfo it_n{}, it_c{}; int item_n = -1, item_c = -1; int cbuf_n = 0, cbuf_c = 0; int g2_count = 0;
    KBlockA A0, A1;
    uint32_t j = 0;          // k-block stream index; stage = j >> 1, kk = j & 1
    int cslot[2] = {-1, -1};   // expert slot currently held in sm.c[0] / sm.c[1]
    const float ftag = __int_as_float((int)epoch);
    bool pend = false; int pend_slot = 0, pend_cb = 0, pend_old = 0;   // deferred "did this warp complete a 128-block?" check

    // Resolve the counter increment issued by this warp's previous GEMM1 item. The warp that completed all
    // 64 contributions (4 K-parts x 2 halves x 8 warps) of a 128-block computes its SwiGLU + block scale and
    // publishes it as 32 tagged records. Readers spin on the records, so this must run before any wait.
    auto check_pend = [&]() {
      if (!pend) return;
      pend = false;
      const int old = __shfl_sync(0xffffffffu, pend_old, 0);
      if (old != 8 * NCONS - 1) return;
      const int slot = pend_slot, cb = pend_cb;
      const int r0 = 4 * lane;                       // values r0..r0+3 of the 128-block
      const int grp1 = cb * 16 + (r0 >> 3), grp2 = I / 8 + grp1;
      const int c0 = (r0 & 7) ? 1 : 0;               // chunks {0,1} for values 0..3, {1,2} for 4..7
      float x1[4] = {0.f, 0.f, 0.f, 0.f}, x2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int p = 0; p < G1_KSPLIT; p++) {
        float4 a0, a1, b0, b1;
        do {
          a0 = __ldcg(&ws->p_rec[p][slot][grp1][c0]); a1 = __ldcg(&ws->p_rec[p][slot][grp1][c0 + 1]);
          b0 = __ldcg(&ws->p_rec[p][slot][grp2][c0]); b1 = __ldcg(&ws->p_rec[p][slot][grp2][c0 + 1]);
          if (__float_as_int(a0.w) == (int)epoch && __float_as_int(a1.w) == (int)epoch &&
              __float_as_int(b0.w) == (int)epoch && __float_as_int(b1.w) == (int)epoch) break;
          __nanosleep(64);
        } while (true);
        // values 0..3 -> a0.xyz, a1.x ; values 4..7 -> a0.y a0.z a1.x a1.y (a0 = chunk1 = values 3,4,5; a1 = chunk2 = 6,7,8)
        if (r0 & 7) { x1[0] += a0.y; x1[1] += a0.z; x1[2] += a1.x; x1[3] += a1.y; x2[0] += b0.y; x2[1] += b0.z; x2[2] += b1.x; x2[3] += b1.y; }
        else        { x1[0] += a0.x; x1[1] += a0.y; x1[2] += a0.z; x1[3] += a1.x; x2[0] += b0.x; x2[1] += b0.y; x2[2] += b0.z; x2[3] += b1.x; }
      }
      float c[4], amax = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) { c[i] = x2[i] / (1.0f + expf(-x2[i])) * x1[i]; amax = fmaxf(amax, fabsf(c[i])); }
      amax = warp_max(amax);
      const float scl = (amax > 0.f) ? amax * (1.0f / C_TARGET) : 1.0f;
      const float inv = 1.0f / scl;
      __half2 h0 = __floats2half2_rn(c[0] * inv, c[1] * inv), h1 = __floats2half2_rn(c[2] * inv, c[3] * inv);
      ws->c_rec[slot][cb * 32 + lane] = make_float4(*reinterpret_cast<float*>(&h0), *reinterpret_cast<float*>(&h1), scl, ftag);
#if MOE_DEBUG
      if (lane == 0 && slot == nhits - 1) { dbgb[4 + 3 * 64 + 2 * 256 + 256 + 9] = gtimer(); }
#endif
    };

    // SwiGLU + block-scaled f16 quantisation of expert `slot`'s 2048 intermediates into sm.c[buf] (all consumer threads)
    // the expert's intermediate (published by the warps that completed each 128-block): 2 tagged records
    // per thread; spin until valid (the producers never wait, so completion is guaranteed)
    auto compute_c = [&](int slot, int buf) {
      const int base = t * 8;
      float4 r0, r1;
      do {
        r0 = __ldcg(&ws->c_rec[slot][2 * t]); r1 = __ldcg(&ws->c_rec[slot][2 * t + 1]);
        if (__float_as_int(r0.w) == (int)epoch && __float_as_int(r1.w) == (int)epoch) break;
        __nanosleep(128);
      } while (true);
      *reinterpret_cast<uint4*>(&sm.c[buf][base]) = make_uint4(__float_as_uint(r0.x), __float_as_uint(r0.y), __float_as_uint(r1.x), __float_as_uint(r1.y));
      if ((t & 15) == 0) sm.s_c[buf][t >> 4] = r0.z;
#if MOE_DEBUG
      if (t == 0 && slot == nhits - 1) { dbgb[4 + 3 * 64 + 2 * 256 + 256 + 8] = gtimer(); }
#endif
    };
    auto wait_stage = [&](uint32_t sidx) {
      mbar_wait(&sm.full[sidx % STAGES], (sidx / STAGES) & 1);   // uniform across the warp: no divergence
    };
    // called when the load side meets a new item; returns false on the sentinel
    auto begin_item = [&](int item) -> bool {
      if (item < 0) return false;
      item_n = item; it_n = decode_item(item, nhits);
      if (it_n.phase == 2) {
        const int slot = it_n.slot;
        if (cslot[cbuf_n] == slot) return true;              // same expert as the previous GEMM2 item: B operand already in smem
        cbuf_n = (cslot[cbuf_n] < 0) ? cbuf_n : (cbuf_n ^ 1);
#if MOE_DEBUG
        const bool firstg2 = (cslot[0] < 0 && cslot[1] < 0);   // no expert loaded yet: first GEMM2 item of this block
        if (t == 0 && firstg2) dbgb[4 + 3 * 64 + 2 * 256 + 256 + 10] = gtimer();
#endif
        check_pend();     // this warp may have to publish part of the intermediate everybody waits for
        consumer_bar();   // everyone has finished loading the k-blocks that used this buffer two items ago
        compute_c(slot, cbuf_n);
        cslot[cbuf_n] = slot;
        consumer_bar();
#if MOE_DEBUG
        if (t == 0 && firstg2) dbgb[4 + 3 * 64 + 2 * 256 + 256 + 13] = gtimer();
#endif
      }
      return true;
    };
    auto load_kb = [&](uint32_t jj, KBlockA& A) {
      const int s = (jj >> 1) % STAGES, kk = jj & 1;
      load_convert_kblock(A, wbase + s * STAGE_BYTES + kk * BLK, STAGE_ROW);
    };

    // one pipeline step: MMAs of k-block j (fragments in Acur) while k-block j+1 is loaded into Anext
    // prefetch (wait for stage, start item if needed, load + convert) k-block jn into A; false on sentinel
    auto prefetch = [&](uint32_t jn, KBlockA& A) -> bool {
      if ((jn & 1) == 0) {
        wait_stage(jn >> 1);
        const int itn = sm.meta[(jn >> 1) % STAGES].item;
        if (itn != item_n && !begin_item(itn)) return false;
      }
      load_kb(jn, A);
      return true;
    };
    auto step = [&](KBlockA& Acur, KBlockA& Anext) -> bool {
      const uint32_t jn = j + 1;
      const int s = (j >> 1) % STAGES, kk = j & 1;
      const int kidx = sm.meta[s].kidx;
      const bool last_kb = (kk == 1) && (kidx == it_c.nk - 1);
      bool have_next = true;
      // the epilogue of an item must not depend on the next stage being published (the producer may
      // be waiting for this very item's result), so at the item's last k-block prefetch afterwards
      if (!last_kb) have_next = prefetch(jn, Anext);

      if (it_c.phase == 1 && kidx == it_c.nk - 1 && kk == 0 && lane == 0)
        pend_old = atomicAdd(&ws->cb_cnt[it_c.slot][it_c.tile & (KB2 - 1)], 1);   // election for publishing this 128-block; resolved after the stores
      const int kb = it_c.kpart * (it_c.phase == 1 ? G1_NKB : G2_NKB) + KPS * kidx + kk;
      float sc = sm.meta[s].sc[kk];
      const uint16_t* brow;
      if (it_c.phase == 1) brow = sm.act + kb * BLK + tig * 16;
      else { brow = sm.c[cbuf_c] + kb * BLK + tig * 16; sc *= sm.s_c[cbuf_c][kb]; }
#if MOE_MODE != 2
      mma_kblock(acc0, acc2, Acur, brow, sc);
#else
      acc0 += sc * (float)brow[0] + (float)Acur.a[0][0];
#endif
#if MOE_DEBUG
      if (t == 0 && j < 256) { dbgb[4 + 3 * 64 + 2 * j] = ((unsigned long long)item_c << 16) | (kidx << 8) | (kk << 4) | 1; dbgb[5 + 3 * 64 + 2 * j] = gtimer(); }
#endif
      if (kk == 1) {
        __syncwarp();
        if (lane == 0) mbar_arrive(&sm.empty[s]);   // both k-blocks of this stage are in registers now
      }
      if (last_kb) {
        // ---- item epilogue
#if MOE_DEBUG
        if (t == 0 && dbg_n < 64) { dbgb[6 + 3 * dbg_n] = gtimer(); }
        dbg_n++;
#endif
        // ---- item epilogue: this warp's 16 row sums (rows g: acc0, rows g+8: acc2 of the tig==0 lanes) are
        // written as 2 groups x 3 tagged float4 records by lanes 0..5 (no fence: readers validate the tag)
        {
          const float w = (it_c.phase == 1) ? 1.f : sm.hitw[it_c.slot];
          const int grp = lane / 3, ch = lane - grp * 3;          // meaningful for lanes 0..5
          const int r0 = 3 * ch;                                 // rows r0..r0+2 of the group (r0+2 == 8 -> pad)
          const int s0 = ((r0 + 0) & 7) * 4, s1 = ((r0 + 1) & 7) * 4, s2 = ((r0 + 2) & 7) * 4;
          const float a0 = __shfl_sync(0xffffffffu, acc0, s0), b0 = __shfl_sync(0xffffffffu, acc2, s0);
          const float a1 = __shfl_sync(0xffffffffu, acc0, s1), b1 = __shfl_sync(0xffffffffu, acc2, s1);
          const float a2 = __shfl_sync(0xffffffffu, acc0, s2), b2 = __shfl_sync(0xffffffffu, acc2, s2);
          if (lane < 6) {
            const float v0 = (grp == 0 ? a0 : b0) * w, v1 = (grp == 0 ? a1 : b1) * w;
            const float v2 = (r0 + 2 < 8) ? (grp == 0 ? a2 : b2) * w : 0.f;
            const int group8 = (it_c.tile * BLK + warp * 16) / 8 + grp;
            float4* dst = (it_c.phase == 1) ? &ws->p_rec[it_c.kpart][it_c.slot][group8][ch]
                                            : &ws->o_rec[it_c.slot * G2_KSPLIT + it_c.kpart][group8][ch];
            *dst = make_float4(v0, v1, v2, ftag);
          }
          if (it_c.phase == 1) {
#if MOE_DEBUG
            if (lane == 0) { dbgb[4 + 3 * 64 + 2 * 256 + 256 + 16 + warp] = gtimer(); dbgb[4 + 3 * 64 + 2 * 256 + 256 + 24 + warp] = it_c.slot; }
#endif
            if (lane == 0) red_relaxed_add(&ws->g1_done[it_c.slot], 1);                 // readiness hint
            pend = true; pend_slot = it_c.slot; pend_cb = it_c.tile & (KB2 - 1);
            check_pend();   // the election atomic was issued a stage ago: publish now if this warp completed the block
          }
        }
        wait_stage(jn >> 1);
        const int itn = sm.meta[(jn >> 1) % STAGES].item;
#if MOE_DEBUG
        if (t == 0 && dbg_n == 1) dbgb[4 + 3 * 64 + 2 * 256 + 256 + 14] = gtimer();
#endif
        acc0 = 0.f; acc2 = 0.f;
        have_next = (itn != item_n) ? begin_item(itn) : true;
        if (have_next) load_kb(jn, Anext);
        it_c = it_n; item_c = item_n; cbuf_c = cbuf_n;    // the load side has moved on to the next item
#if MOE_DEBUG
        if (t == 0 && have_next && dbg_n < 64) { dbgb[4 + 3 * dbg_n] = item_c; dbgb[5 + 3 * dbg_n] = gtimer(); }
#endif
      }
      j = jn;
      return have_next;
    };

    // prologue: first stage
    wait_stage(0);
    bool have = begin_item(sm.meta[0].item);
    if (have) {
      load_kb(0, A0);
      it_c = it_n; item_c = item_n; cbuf_c = cbuf_n;
#if MOE_DEBUG
      if (t == 0) { dbgb[4] = item_c; dbgb[5] = gtimer(); }
#endif
    }
    while (have) {
      have = step(A0, A1);          // j even: multiply A0, load A1
      if (!have) break;
      have = step(A1, A0);          // j odd:  multiply A1, load A0
    }
    check_pend();
  }
  __syncthreads();
#if MOE_DEBUG
  if (t == 0) { dbgb[2] = gtimer(); dbgb[3] = dbg_n; }
#endif
  // ---- count finished blocks; the last one resets the queue / per-hit state (nobody reads them any more)
  if (t == 0) {
    int old = atomicAdd(&ws->blocks_done, 1);
    if (old == (int)gridDim.x - 1) {
      ws->q1 = 0; ws->blocks_done = 0;
      for (int h = 0; h < MAXHITS; h++) { ws->g1_done[h] = 0; ws->q2[h] = 0; }
      for (int k = 0; k < MAXHITS * KB2; k++) (&ws->cb_cnt[0][0])[k] = 0;
    }
  }
  // ---- output: block b < RB2 sums and converts tile b; thread (group, chunk) spins until every GEMM2
  // partial record of its 3 rows carries this launch's tag
  if (blockIdx.x < RB2 && t < 48) {
    const int tile = blockIdx.x;
    const int nparts = nhits * G2_KSPLIT;
    const int grp = t / 3, ch = t - grp * 3;
    const int group8 = tile * (BLK / 8) + grp;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f;
    for (int p = 0; p < nparts; p++) {
      float4 r;
      do { r = __ldcg(&ws->o_rec[p][group8][ch]); if (__float_as_int(r.w) == (int)epoch) break; __nanosleep(128); } while (true);
      v0 += r.x; v1 += r.y; v2 += r.z;
    }
    const int row = tile * BLK + grp * 8 + 3 * ch;
    out[row] = __float2bfloat16(v0);
    out[row + 1] = __float2bfloat16(v1);
    if (3 * ch + 2 < 8) out[row + 2] = __float2bfloat16(v2);
#if MOE_DEBUG
    if (t == 0) dbgb[4 + 3 * 64 + 2 * 256 + 256 + 15] = gtimer();
#endif
  }
}

constexpr size_t SMEM_BYTES = sizeof(Smem);

int encode_map(CUtensorMap* tm, const void* ptr, uint64_t rows, uint64_t cols) {
  cuuint64_t gdim[2] = {cols, rows};
  cuuint64_t gstride[1] = {cols};
  cuuint32_t box[2] = {STAGE_ROW, BLK};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), gdim, gstride, box, estride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (int)r;
}

}  // namespace

extern "C" {

size_t moe_ws_bytes() { return sizeof(Workspace); }
int moe_dbg_per_block() { return DBG_PER_BLOCK; }

int moe_num_sms() {
  int dev = 0; cudaGetDevice(&dev);
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  return n;
}

int moe_init() {
  return (int)cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
}

int moe_run(void* stream, const void* w1, const void* w2, const void* logits, const void* bias,
            const void* hidden, const void* hscale, const void* w1s, const void* w2s,
            int offset, float rsf, void* ws, void* out, int grid, unsigned epoch, void* dbg) {
  CUtensorMap tm1, tm2;
  int r = encode_map(&tm1, w1, (uint64_t)E_LOCAL * I2, H);
  if (r) return 1000 + r;
  r = encode_map(&tm2, w2, (uint64_t)E_LOCAL * H, I);
  if (r) return 2000 + r;
  moe_kernel<<<grid, NTHREADS, SMEM_BYTES, (cudaStream_t)stream>>>(
      tm1, tm2, (const float*)logits, (const __nv_bfloat16*)bias, (const uint8_t*)hidden, (const float*)hscale,
      (const float*)w1s, (const float*)w2s, offset, rsf, (Workspace*)ws, (__nv_bfloat16*)out, epoch, (unsigned long long*)dbg);
  return (int)cudaGetLastError();
}

}  // extern "C"
