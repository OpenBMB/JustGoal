// MoE FP8 block-scale (DeepSeek-V3 no-aux routing) — B200 (sm_100a), CUDA 13.
// Phase 2 (D1+D3): one persistent, cooperative kernel per call.
//   P0  every CTA: routing (sigmoid+bias, 8 groups x top2, top4 groups, top8), token + scales -> smem
//   P1  GEMM1: TMA ring -> ldmatrix -> mma.sync e4m3 (token as B), fp32 promotion per 128-k block,
//       balanced (expert, 32-row tile, k-block) units, atomicAdd of gate/up partial sums -> gu_acc
//   --  grid barrier (W2 streaming continues through it), C = silu(up)*gate -> block-scaled f16 in smem
//   P2  GEMM2: TMA ring -> ldmatrix -> cvt f16 -> mma.sync f16 (C as B), atomicAdd -> out_acc;
//       the warp completing a 64-row output tile converts it to bf16 and re-zeroes it; the last CTA
//       re-zeroes gu_acc and the counters so the workspace is clean for the next call.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

namespace {

constexpr int HIDDEN = 7168;
constexpr int INTER = 2048;
constexpr int G1_OUT = 4096;
constexpr int E_GLOBAL = 256;
constexpr int E_LOCAL = 32;
constexpr int TOPK = 8;
constexpr int N_GROUP = 8;
constexpr int TOPK_GROUP = 4;
constexpr int HB = 56;   // hidden blocks (7168/128)
constexpr int IB = 16;   // intermediate blocks (2048/128)
constexpr int G1B = 32;  // gemm1-out blocks (4096/128)

// ---- pipeline geometry
constexpr int TM = 32;                       // rows per TMA box
constexpr int BOX_BYTES = TM * 128;          // 4 KB
constexpr int UNIT_BYTES = 2 * BOX_BYTES;    // 8 KB: one (row tile, k-block) unit = 2 boxes
#ifndef KPS
#define KPS 2                                // k-blocks (units) per stage
#endif
constexpr int STAGE_BYTES = KPS * UNIT_BYTES; // 16 KB stages: halves the per-stage handshake + bookkeeping per byte
#ifndef STAGES
#define STAGES 10                            // 16 KB stages in flight (ring depth)
#endif
constexpr int RG = 4;                        // row groups (16 rows each) per stage
constexpr int KW = 2;                        // k-interleaved consumer groups
constexpr int NCONS = RG * KW;               // 8 consumer warps
constexpr int THREADS = 32 * (2 + NCONS);    // + producer warp + signaler warp
constexpr int G1_TILES = INTER / TM;         // 64 gate-row tiles per expert
constexpr int G2_TM = 2 * TM;                // 64 output rows per GEMM2 tile
constexpr int G2_TILES = HIDDEN / G2_TM;     // 112
constexpr int U1S = G1_TILES * HB;            // GEMM1 units per expert slot (3584)
constexpr int U2S = G2_TILES * IB;            // GEMM2 units per expert slot (1792)
constexpr int SCP_MAX = 32;                   // staged scales per phase per CTA (>= units per CTA per phase for G >= 112)
static_assert(STAGES % KW == 0, "stage -> consumer warp mapping must be static");
static_assert(HB % KPS == 0 && IB % KPS == 0, "stages must not straddle row tiles");


// ---- workspace layout (floats / ints), per token
constexpr int WS_GU = TOPK * 2 * INTER;                 // gate/up accumulators
constexpr int WS_OUT = WS_GU + HIDDEN;                  // out_acc
constexpr int WS_TILE_DONE = WS_OUT + HIDDEN;           // [G2_TILES] ints (padded to 128)
constexpr int WS_G1_DONE = WS_TILE_DONE + 128;
constexpr int WS_DONE_CTR = WS_G1_DONE + 16;           // [0..7] per-slot GEMM1-done CTA counters, [8] GEMM2-done
constexpr int WS_DBG = WS_DONE_CTR + 4;                 // debug timestamps [grid][8] (u64)
constexpr int WS_FLOATS = WS_DBG + 32768;

constexpr uint64_t POLICY_EVICT_FIRST = 0x12F0000000000000ull;

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\t"
      "bra LAB_WAIT;\n\t"
      "DONE:\n\t}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* map, uint64_t* bar, int c0,
                                            int c1, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(smem_dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_fp8(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t v16) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"((uint16_t)v16));
  return h2;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// monotonic float -> uint32 key (handles negative values)
__device__ __forceinline__ uint32_t f2key(float f) {
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// warp arg-max by redux.sync: returns the max key and the lowest lane holding it
__device__ __forceinline__ void warp_argmax_key(uint32_t key, uint32_t& best, int& lane_of_best) {
  best = __reduce_max_sync(0xffffffffu, key);
  lane_of_best = __ffs(__ballot_sync(0xffffffffu, key == best)) - 1;
}
__device__ __forceinline__ void warp_argmax(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o);
    int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
}
// loads that the compiler may not sink below later volatile asm (true prefetches)
__device__ __forceinline__ float ldg_f32_nc(const float* p) {
  float v; asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v;
}
__device__ __forceinline__ float4 ldg_f4_nc(const float* p) {
  float4 v; asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ float4 ldcg_f4(const float* p) {
  float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ int atom_add_acq_rel_gpu(int* p, int v) {
  int old; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#ifdef DEBUG_TIMING
#define STAMP(k) do { if (threadIdx.x == 0) { dbg[(k)] = globaltimer(); dbg[(k) + 16] = clock64(); } } while (0)
#define STAMP_T(k, cond) do { if (cond) { dbg[(k)] = globaltimer(); dbg[(k) + 16] = clock64(); } } while (0)
// per-warp wait / compute cycle accounting (lane 0): dbg[base + warp] = wait_cycles << 32 | busy_cycles
#ifdef NO_ACCT
#define DBG_T0() do { } while (0)
#define DBG_T1() do { } while (0)
#define DBG_T2() do { } while (0)
#else
#define DBG_T0() const long long _t0 = clock64()
#define DBG_T1() const long long _t1 = clock64(); _dbg_wait += _t1 - _t0
#define DBG_T2() _dbg_busy += clock64() - _t1
#endif
#define DBG_DUMP(base) do { if (lane == 0) dbg[(base) + warp] = ((uint64_t)(uint32_t)_dbg_wait << 32) | (uint32_t)_dbg_busy; _dbg_wait = _dbg_busy = 0; } while (0)
#else
#define STAMP(k) do { } while (0)
#define STAMP_T(k, cond) do { } while (0)
#define DBG_T0() do { } while (0)
#define DBG_T1() do { } while (0)
#define DBG_T2() do { } while (0)
#define DBG_DUMP(base) do { } while (0)
#endif
__device__ __forceinline__ void consumers_sync() { asm volatile("bar.sync 1, %0;" ::"r"(NCONS * 32) : "memory"); }

// ---------------------------------------------------------------- flush helpers (one code copy each)
__device__ __noinline__ void g1_flush(float* gp, float acc0, float acc8, bool leader) {
  if (leader) {
    atomicAdd(gp, acc0);
    atomicAdd(gp + 8, acc8);
  }
}
// ---------------------------------------------------------------- shared memory
struct __align__(1024) Smem {
  uint8_t stages[STAGES][STAGE_BYTES];
  __half c16[TOPK][INTER];   // block-scaled f16 SwiGLU output (32 KB)
  uint8_t token[HIDDEN];     // fp8 token (7 KB)
  float cs[TOPK][IB];        // 2^e * w_e per (slot, i-block)
  float sa[HB];              // token block scales
  float sc[2 * TOPK][SCP_MAX][2];   // raw weight block scales of this CTA's units, per phase (G1: gate, up)
  int unit_tag[STAGES];      // unit id within the phase carried by the stage (-1 = end of phase, one per consumer group)
  int stage_pay[STAGES];     // sentinel payload: sequence index of the next phase's first stage
  uint64_t full[STAGES];
  uint64_t empty[STAGES];
  // routing
  float gscore[N_GROUP];
  int kept[N_GROUP];           // per-group keep flags
  __align__(16) float swb_sm[E_GLOBAL];   // biased scores (float4 broadcast reads in the rank loop)
  float s_sm[E_GLOBAL];
  int count;
  int local_id[TOPK];
  float weight[TOPK];
  int sel[TOPK];
  int flag;
};
constexpr int SMEM_BYTES = sizeof(Smem) + 1024;

// ---------------------------------------------------------------- routing (threads 0..255 of the CTA)
// Rank-based selection: every candidate computes its own rank (ties -> lower index first, matching torch.topk
// on distinct values) with broadcast smem reads, so there are no serial arg-max rounds.
__device__ __forceinline__ float key2f(uint32_t k) { return __uint_as_float((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k); }
__device__ __forceinline__ void routing_cta(const float* __restrict__ logits, const __nv_bfloat16* __restrict__ bias,
                                            int local_offset, float scaling, Smem& S, uint64_t* dbg) {
  const int e = threadIdx.x;
  const int lane = e & 31, warp = e >> 5;
  float s = 0.f, swb = -INFINITY;
  if (e < E_GLOBAL) {
    const float lg = logits[e];
    s = 1.0f / (1.0f + __expf(-lg));
    swb = s + __bfloat162float(bias[e]);
    S.s_sm[e] = s;
    S.swb_sm[e] = swb;
    // group score = top-2 sum of the biased scores (group = warp)
    const uint32_t k = f2key(swb);
    const uint32_t k1 = __reduce_max_sync(0xffffffffu, k);
    const int first = __ffs(__ballot_sync(0xffffffffu, k == k1)) - 1;
    const uint32_t k2 = __reduce_max_sync(0xffffffffu, lane == first ? 0u : k);
    if (lane == 0) S.gscore[warp] = key2f(k1) + key2f(k2);
  }
  STAMP(8);
  __syncthreads();
  STAMP(9);
  if (e < N_GROUP) {   // top-4 groups by rank
    const float gs = S.gscore[e];
    int rank = 0;
#pragma unroll
    for (int j = 0; j < N_GROUP; ++j) { const float gj = S.gscore[j]; rank += (gj > gs) || (gj == gs && j < e); }
    S.kept[e] = rank < TOPK_GROUP;
  }
  STAMP(14);
  __syncthreads();
  STAMP(10);
  if (e < E_GLOBAL && S.kept[warp]) {   // top-8 experts among the kept groups' 128 candidates, by rank
    int rank = 0;
#pragma unroll 1
    for (int gq = 0; gq < N_GROUP; ++gq) {
      if (!S.kept[gq]) continue;
      const float4* v4 = reinterpret_cast<const float4*>(&S.swb_sm[gq * 32]);
#pragma unroll
      for (int j4 = 0; j4 < 8; ++j4) {
        const float4 v = v4[j4];   // broadcast read
        const int j = gq * 32 + j4 * 4;
        rank += (v.x > swb) || (v.x == swb && j < e);
        rank += (v.y > swb) || (v.y == swb && j + 1 < e);
        rank += (v.z > swb) || (v.z == swb && j + 2 < e);
        rank += (v.w > swb) || (v.w == swb && j + 3 < e);
      }
    }
    if (rank < TOPK) S.sel[rank] = e;
  }
  __syncthreads();
  if (e < TOPK) {   // weights (unbiased scores, normalised, scaled) and local-expert compaction in rank order
    const int ge = S.sel[e];
    const float sv = S.s_sm[ge];
    float sum = sv;
#pragma unroll
    for (int o = 1; o < TOPK; o <<= 1) sum += __shfl_xor_sync(0x000000ffu, sum, o, TOPK);
    sum += 1e-20f;
    const int le = ge - local_offset;
    const bool local = (le >= 0 && le < E_LOCAL);
    const uint32_t m = __ballot_sync(0x000000ffu, local) & 0xffu;
    const int pos = __popc(m & ((1u << e) - 1u));
    if (local) { S.local_id[pos] = le; S.weight[pos] = sv / sum * scaling; }
    if (e == 0) S.count = __popc(m);
  }
  STAMP(11);
  __syncthreads();
}

// ---------------------------------------------------------------- the fused kernel
__global__ void __launch_bounds__(THREADS, 1) moe_fused_kernel(const __grid_constant__ CUtensorMap map_w13,
                                                               const __grid_constant__ CUtensorMap map_w2,
                                                               const float* __restrict__ logits,
                                                               const __nv_bfloat16* __restrict__ bias,
                                                               int local_offset, float scaling,
                                                               const uint8_t* __restrict__ hs,
                                                               const float* __restrict__ hs_scale,
                                                               const float* __restrict__ s13,
                                                               const float* __restrict__ s2,
                                                               float* __restrict__ ws_all,
                                                               __nv_bfloat16* __restrict__ out, int T) {
  const int t = blockIdx.y;
  const int G = gridDim.x;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // align to 1 KB with pointer arithmetic on the shared array so the compiler keeps the shared address
  // space (an integer round-trip would turn every S.* access into a slow generic LD.E/ST.E)
  Smem& S = *reinterpret_cast<Smem*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* ws = ws_all + (size_t)t * WS_FLOATS;
  float* gu_acc = ws;
  float* out_acc = ws + WS_OUT;
  int* tile_done = reinterpret_cast<int*>(ws + WS_TILE_DONE);
  int* g1_done = reinterpret_cast<int*>(ws + WS_G1_DONE);
  int* g2_done = g1_done + 8;
  int* done_ctr = reinterpret_cast<int*>(ws + WS_DONE_CTR);
  __nv_bfloat16* out_t = out + (size_t)t * HIDDEN;
  uint64_t* dbg = reinterpret_cast<uint64_t*>(ws + WS_DBG) + (size_t)blockIdx.x * 64;
  STAMP(0);

  // ---- P0: barriers, token, scales (warp 8) and routing (warps 0..7)
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(&S.full[s], 1); mbar_init(&S.empty[s], RG); }
    fence_barrier_init();
  }
  if (warp == NCONS) {   // token + scales: all loads issued before any store (cold DRAM latency paid once)
    constexpr int TOK_PER_LANE = HIDDEN / 16 / 32;  // 14
    uint4 tv[TOK_PER_LANE];
    const uint4* src = reinterpret_cast<const uint4*>(hs + (size_t)t * HIDDEN);
#pragma unroll
    for (int q = 0; q < TOK_PER_LANE; ++q) tv[q] = __ldg(src + lane + q * 32);
    float sv0 = (lane < HB) ? __ldg(hs_scale + (size_t)lane * T + t) : 0.f;
    float sv1 = (lane + 32 < HB) ? __ldg(hs_scale + (size_t)(lane + 32) * T + t) : 0.f;
#pragma unroll
    for (int q = 0; q < TOK_PER_LANE; ++q) reinterpret_cast<uint4*>(S.token)[lane + q * 32] = tv[q];
    if (lane < HB) S.sa[lane] = sv0;
    if (lane + 32 < HB) S.sa[lane + 32] = sv1;
  }
  routing_cta(logits + (size_t)t * E_GLOBAL, bias, local_offset, scaling, S, dbg);   // ends with __syncthreads
  const int count = S.count;
  STAMP(1);

  if (count == 0) {  // nothing routed here: zero output, workspace untouched
    for (int h = threadIdx.x; h < HIDDEN; h += THREADS) out_t[h] = __float2bfloat16_rn(0.f);
    return;
  }

  // ---- phase plan: G1(0), G1(1), G2(0), G1(2), G2(1), ..., G1(n-1), G2(n-2), G2(n-1). Every G2(s) comes one
  // full phase after G1(s), so its grid-wide dependency is already satisfied when the consumers reach it:
  // the only exposed grid barrier is the final one. Within a phase each CTA streams a balanced static range.
  const int nph = 2 * count;
  auto phase_of = [&](int p, int& ph, int& slot) {
    if (p < 2 && p < count) { ph = 0; slot = p; }
    else if (p == nph - 1) { ph = 1; slot = count - 1; }
    else { const int q = p - 2; ph = (q & 1) ? 0 : 1; slot = ph ? (q >> 1) : (q >> 1) + 2; }
  };
  auto range_of = [&](int ph, int& a, int& b) {   // this CTA's static unit range within a phase (whole stages)
    const int U = ph ? U2S : U1S;
    a = (int)(((long long)U * blockIdx.x) / G) / KPS * KPS;
    b = (int)(((long long)U * (blockIdx.x + 1)) / G) / KPS * KPS;
  };

  if (warp == NCONS) {
    // ================= producer: streams the phases back to back through one ring; per unit the issuing
    // thread does wait / tag / expect / 2 TMA issues (its instruction count per 8 KB bounds the stream rate,
    // dev/bw_probe.cu). KW sentinel stages (tag -1, payload = next phase's first sequence index) end a phase.
    if (lane == 0) {
      int stage = 0; uint32_t phase = 0; int seq = 0;
      auto issue = [&](const CUtensorMap* map, int c0, int r0, int r1, int tag) {   // units tag, tag+1
        mbar_wait(&S.empty[stage], phase ^ 1);
        S.unit_tag[stage] = tag;
#ifdef NO_TMA   // diagnostic: stages complete immediately (garbage data) -> measures consumer-side throughput
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&S.full[stage])) : "memory");
        (void)map; (void)c0; (void)r0; (void)r1;
#else
        mbar_expect_tx(&S.full[stage], STAGE_BYTES);
        uint8_t* dst = S.stages[stage];
#pragma unroll
        for (int q = 0; q < KPS; ++q, dst += UNIT_BYTES, c0 += 128) {
          tma_load_2d(dst, map, &S.full[stage], c0, r0, POLICY_EVICT_FIRST);
          tma_load_2d(dst + BOX_BYTES, map, &S.full[stage], c0, r1, POLICY_EVICT_FIRST);
        }
#endif
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
        ++seq;
      };
      auto sentinel = [&](int payload) {   // a stage without data: completes the full barrier by plain arrival
        mbar_wait(&S.empty[stage], phase ^ 1);
        S.unit_tag[stage] = -1;
        S.stage_pay[stage] = payload;
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&S.full[stage])) : "memory");
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
        ++seq;
      };
      for (int p = 0; p < nph; ++p) {
        int ph, slot, a, b;
        phase_of(p, ph, slot);
        range_of(ph, a, b);
        const int le = S.local_id[slot];
        if (ph == 0) {
          for (int u = a; u < b;) {
            const int kb0 = u % HB, tile = u / HB;
            const int nseg = min(HB - kb0, b - u);
            const int r0 = le * G1_OUT + tile * TM;
            int c0 = kb0 * 128;
            for (int k = 0; k < nseg; k += KPS, c0 += KPS * 128) issue(&map_w13, c0, r0, r0 + INTER, u + k);
            u += nseg;
          }
        } else {
          for (int u = a; u < b;) {
            const int ib0 = u % IB, tile = u / IB;
            const int nseg = min(IB - ib0, b - u);
            const int r0 = le * HIDDEN + tile * G2_TM;
            int c0 = ib0 * 128;
            for (int k = 0; k < nseg; k += KPS, c0 += KPS * 128) issue(&map_w2, c0, r0, r0 + TM, u + k);
            u += nseg;
          }
        }
        for (int k = 0; k < KW; ++k) sentinel(seq + KW - k);
      }
    }
  } else if (warp == NCONS + 1) {
    // ================= signaler: after every GEMM1 phase the consumer warps arrive (non-blocking) at named
    // barrier 2; this warp syncs on it, makes their partial-sum atomics visible GPU-wide and bumps the slot's
    // completion counter, so the ~2 us release fence never stalls a consumer warp.
    for (int p = 0; p < nph; ++p) {
      int ph, slot;
      phase_of(p, ph, slot);
      if (ph != 0) continue;
      asm volatile("bar.sync 2, %0;" ::"r"((NCONS + 1) * 32) : "memory");
      __threadfence();
      if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(g1_done + slot) : "memory");
      STAMP_T(2, lane == 0 && slot == 0);
    }
  } else {
    // ================= consumers
    const int cw = warp;
    const int rg = cw % RG, kw = cw / RG;
    const int j = lane >> 3, rr = lane & 7;
    const int t4 = lane & 3, g = lane >> 2;
    // this lane's ldmatrix row: box rg>>1, row 16*(rg&1) + 8*(j&1) + rr ; swizzle key = rr
    const uint32_t lane_off = (rg >> 1) * BOX_BYTES + (16 * (rg & 1) + 8 * (j & 1) + rr) * 128;
    const int chunk_hi = j >> 1;
    const bool is_up = rg >= 2;
#ifdef DEBUG_TIMING
    long long _dbg_wait = 0, _dbg_busy = 0;
#endif

    // ---------------- stage the weight block scales of this CTA's units of every phase into smem (one
    // coalesced pass by the 256 consumer threads; its DRAM latency overlaps the first TMA loads' latency)
    for (int e = threadIdx.x; e < nph * SCP_MAX; e += NCONS * 32) {
      const int p = e / SCP_MAX, k = e % SCP_MAX;
      int ph, slot, a, b;
      phase_of(p, ph, slot);
      range_of(ph, a, b);
      if (k < b - a) {
        const int u = a + k, le = S.local_id[slot];
        if (ph == 0) {
          const float* q = s13 + ((size_t)le * G1B + ((u / HB * TM) >> 7)) * HB + (u % HB);
          const float g0 = __ldg(q), g1 = __ldg(q + 16 * HB);
          S.sc[p][k][0] = g0;
          S.sc[p][k][1] = g1;
        } else {
          S.sc[p][k][0] = __ldg(s2 + ((size_t)le * HB + ((u / IB * G2_TM) >> 7)) * IB + (u % IB));
        }
      }
    }
    consumers_sync();
    STAMP(56);

    int i = kw;        // sequence index of this group's next stage (group kw owns stages i = kw mod KW)
    int pstart = 0;    // sequence index of the current phase's first stage
    for (int p = 0; p < nph; ++p) {
      int ph, slot, a, b;
      phase_of(p, ph, slot);
      range_of(ph, a, b);
      const int np = b - a;
      const int le = S.local_id[slot];
      if (ph == 1) {
        // ---- GEMM2(slot) needs GEMM1(slot) complete on every CTA (normally long done), then C(slot) staged
        if (threadIdx.x == 0) { while (ld_acquire_gpu(g1_done + slot) < G) { } }
        consumers_sync();
        STAMP_T(3, threadIdx.x == 0 && slot == 0);
        {   // C(slot) = silu(up) * gate -> block-scaled f16: 16 blocks, 2 per warp, loads first
          const float* gp0 = gu_acc + (size_t)slot * 2 * INTER + cw * 128 + lane * 4;
          const float* gp1 = gp0 + NCONS * 128;
          const float4 x1a = ldcg_f4(gp0), x2a = ldcg_f4(gp0 + INTER);
          const float4 x1b = ldcg_f4(gp1), x2b = ldcg_f4(gp1 + INTER);
          float cv[2][4];
          // silu(x2) * x1 with the fast divide: no FCHK slow-path subroutine when 1+exp(-x) overflows to inf
          cv[0][0] = __fdividef(x2a.x, 1.0f + __expf(-x2a.x)) * x1a.x; cv[0][1] = __fdividef(x2a.y, 1.0f + __expf(-x2a.y)) * x1a.y;
          cv[0][2] = __fdividef(x2a.z, 1.0f + __expf(-x2a.z)) * x1a.z; cv[0][3] = __fdividef(x2a.w, 1.0f + __expf(-x2a.w)) * x1a.w;
          cv[1][0] = __fdividef(x2b.x, 1.0f + __expf(-x2b.x)) * x1b.x; cv[1][1] = __fdividef(x2b.y, 1.0f + __expf(-x2b.y)) * x1b.y;
          cv[1][2] = __fdividef(x2b.z, 1.0f + __expf(-x2b.z)) * x1b.z; cv[1][3] = __fdividef(x2b.w, 1.0f + __expf(-x2b.w)) * x1b.w;
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            float m = fmaxf(fmaxf(fabsf(cv[q][0]), fabsf(cv[q][1])), fmaxf(fabsf(cv[q][2]), fabsf(cv[q][3])));
            m = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(m)));   // m >= 0: bits monotonic
            // power-of-two block scale: scaled max in [2^13, 2^14)
            const int ex = (m > 0.f) ? ((int)((__float_as_uint(m) >> 23) & 0xffu) - 127 + 1 - 14) : 0;
            const float inv = __uint_as_float((uint32_t)(127 - ex) << 23);
            __half2 h0 = __floats2half2_rn(cv[q][0] * inv, cv[q][1] * inv);
            __half2 h1 = __floats2half2_rn(cv[q][2] * inv, cv[q][3] * inv);
            uint2 packed;
            packed.x = *reinterpret_cast<uint32_t*>(&h0);
            packed.y = *reinterpret_cast<uint32_t*>(&h1);
            const int ib = cw + q * NCONS;
            *reinterpret_cast<uint2*>(&S.c16[slot][ib * 128 + lane * 4]) = packed;
            if (lane == 0) S.cs[slot][ib] = __uint_as_float((uint32_t)(127 + ex) << 23) * S.weight[slot];
          }
        }
        consumers_sync();
        STAMP_T(4, threadIdx.x == 0 && slot == 0);
      }

      int cur_tile = -1;
      float acc0 = 0.f, acc8 = 0.f;
      auto flush = [&]() {   // fire-and-forget partial sums
        if (ph == 0) {
          float* gp = gu_acc + ((size_t)slot * 2 + (is_up ? 1 : 0)) * INTER + cur_tile * TM + 16 * (rg & 1) + g;
          g1_flush(gp, acc0, acc8, t4 == 0);
        } else if (t4 == 0) {
          const int h0 = cur_tile * G2_TM + 16 * rg + g;
          atomicAdd(out_acc + h0, acc0);
          atomicAdd(out_acc + h0 + 8, acc8);
        }
      };
      for (;; i += KW) {
        const int stage = i % STAGES;
        const uint32_t sphase = (i / STAGES) & 1;
        DBG_T0();
        mbar_wait(&S.full[stage], sphase);
        DBG_T1();
        const int u = S.unit_tag[stage];
        if (u < 0) {   // sentinel: read its payload, release the stage, leave the phase
          pstart = S.stage_pay[stage];
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.empty[stage]);
          i += KW;
          break;
        }
        const int k = (i - pstart) * KPS;   // position of the stage's first unit within the phase's static range
        const uint32_t base = smem_u32(S.stages[stage]) + lane_off;
        if (ph == 0) {
          const int kb = u % HB, tile = u / HB;
          if (tile != cur_tile) { if (cur_tile >= 0) flush(); cur_tile = tile; acc0 = acc8 = 0.f; }
          float sc[KPS];
#pragma unroll
          for (int q = 0; q < KPS; ++q) {
            const float sraw = (k + q < SCP_MAX) ? S.sc[p][k + q][is_up ? 1 : 0]
                                                 : __ldg(s13 + ((size_t)le * G1B + ((tile * TM) >> 7) + (is_up ? 16 : 0)) * HB + kb + q);
            sc[q] = sraw * S.sa[kb + q];
          }
          float d[KPS][4][4];
#pragma unroll
          for (int q = 0; q < KPS; ++q)
#pragma unroll
            for (int s = 0; s < 4; ++s) { d[q][s][0] = d[q][s][1] = d[q][s][2] = d[q][s][3] = 0.f; }
#ifndef NO_COMPUTE
          uint32_t a4[KPS][4][4];
#pragma unroll
          for (int q = 0; q < KPS; ++q)
#pragma unroll
            for (int s = 0; s < 4; ++s) ldmatrix_x4(a4[q][s], base + q * UNIT_BYTES + (((2 * s + chunk_hi) ^ rr) * 16));   // all loads first
#pragma unroll
          for (int q = 0; q < KPS; ++q)
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              const uint32_t* tok = reinterpret_cast<const uint32_t*>(S.token + (kb + q) * 128 + s * 32 + 4 * t4);
              mma_fp8(d[q][s], a4[q][s], tok[0], tok[4]);
            }
#else
          (void)base; d[0][0][0] = __uint_as_float(u);
#endif
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.empty[stage]);
          __syncwarp();
#pragma unroll
          for (int q = 0; q < KPS; ++q) {
            acc0 = fmaf((d[q][0][0] + d[q][1][0]) + (d[q][2][0] + d[q][3][0]), sc[q], acc0);
            acc8 = fmaf((d[q][0][2] + d[q][1][2]) + (d[q][2][2] + d[q][3][2]), sc[q], acc8);
          }
        } else {
          const int ib = u % IB, tile = u / IB;
          if (tile != cur_tile) { if (cur_tile >= 0) flush(); cur_tile = tile; acc0 = acc8 = 0.f; }
          float sc[KPS];
#pragma unroll
          for (int q = 0; q < KPS; ++q) {
            const float sraw = (k + q < SCP_MAX) ? S.sc[p][k + q][0]
                                                 : __ldg(s2 + ((size_t)le * HB + ((tile * G2_TM) >> 7)) * IB + ib + q);
            sc[q] = sraw * S.cs[slot][ib + q];
          }
          float d[KPS][4][4];
#pragma unroll
          for (int q = 0; q < KPS; ++q)
#pragma unroll
            for (int s = 0; s < 4; ++s) { d[q][s][0] = d[q][s][1] = d[q][s][2] = d[q][s][3] = 0.f; }
#ifndef NO_COMPUTE
          uint32_t a8[KPS][4][4];
#pragma unroll
          for (int q = 0; q < KPS; ++q)
#pragma unroll
            for (int s = 0; s < 4; ++s) ldmatrix_x4(a8[q][s], base + q * UNIT_BYTES + (((2 * s + chunk_hi) ^ rr) * 16));   // all loads first
#pragma unroll
          for (int q = 0; q < KPS; ++q) {
            const __half* cp = &S.c16[slot][(ib + q) * 128];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              uint32_t af[4], bf[4];
              af[0] = cvt_e4m3x2_f16x2(a8[q][s][0] & 0xffffu);
              af[2] = cvt_e4m3x2_f16x2(a8[q][s][0] >> 16);
              af[1] = cvt_e4m3x2_f16x2(a8[q][s][1] & 0xffffu);
              af[3] = cvt_e4m3x2_f16x2(a8[q][s][1] >> 16);
              bf[0] = cvt_e4m3x2_f16x2(a8[q][s][2] & 0xffffu);
              bf[2] = cvt_e4m3x2_f16x2(a8[q][s][2] >> 16);
              bf[1] = cvt_e4m3x2_f16x2(a8[q][s][3] & 0xffffu);
              bf[3] = cvt_e4m3x2_f16x2(a8[q][s][3] >> 16);
              const uint4 bb = *reinterpret_cast<const uint4*>(cp + s * 32 + 4 * t4);        // k 4t..4t+3
              const uint4 bc = *reinterpret_cast<const uint4*>(cp + s * 32 + 16 + 4 * t4);   // k 16+4t..
              mma_f16(d[q][s & 1], af, bb.x, bb.y);
              mma_f16(d[q][2 + (s & 1)], bf, bc.x, bc.y);
            }
          }
#else
          (void)base; d[0][0][0] = __uint_as_float(u);
#endif
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.empty[stage]);
          __syncwarp();
#pragma unroll
          for (int q = 0; q < KPS; ++q) {
            acc0 = fmaf((d[q][0][0] + d[q][1][0]) + (d[q][2][0] + d[q][3][0]), sc[q], acc0);
            acc8 = fmaf((d[q][0][2] + d[q][1][2]) + (d[q][2][2] + d[q][3][2]), sc[q], acc8);
          }
        }
        DBG_T2();
      }
      if (cur_tile >= 0) flush();
      if (ph == 0) {
        // publish this phase's partial sums: CTA-scope fence + non-blocking arrival; the signaler warp does the
        // GPU-scope release (cumulative over everything ordered before its bar.sync)
        asm volatile("fence.acq_rel.cta;" ::: "memory");
        asm volatile("bar.arrive 2, %0;" ::"r"((NCONS + 1) * 32) : "memory");
        STAMP_T(57, threadIdx.x == 0 && slot == 0);
      }
      if (p == nph - 1) { STAMP(60); DBG_DUMP(48); }
      (void)np;
    }
  }

  // ---- grid barrier on GEMM2 completion, then every CTA converts its slice of out_acc to bf16 (and re-zeroes it)
  STAMP_T(5, threadIdx.x == 0);
  STAMP_T(6, threadIdx.x == NCONS * 32);
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(g2_done) : "memory");
    while (ld_acquire_gpu(g2_done) < G) { }
  }
  __syncthreads();
  STAMP(62);
  {
    const int h0 = (int)(((long long)HIDDEN * blockIdx.x) / G), h1 = (int)(((long long)HIDDEN * (blockIdx.x + 1)) / G);
    for (int h = h0 + threadIdx.x; h < h1; h += THREADS) {
      out_t[h] = __float2bfloat16_rn(__ldcg(out_acc + h));
#ifndef DEBUG_KEEP_WS
      out_acc[h] = 0.f;
#endif
    }
  }
  STAMP(7);
  // ---- the last CTA to finish re-zeroes gu_acc and the counters for the next call
  if (threadIdx.x == 0) {
    __threadfence();
    const int prev = atomicAdd(done_ctr, 1);
    S.flag = (prev == G - 1);
  }
  __syncthreads();
#ifndef DEBUG_KEEP_WS
  if (S.flag) {
    __threadfence();
    float4* z = reinterpret_cast<float4*>(gu_acc);
    for (int i = threadIdx.x; i < count * 2 * INTER / 4; i += THREADS) z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (threadIdx.x < 16) g1_done[threadIdx.x] = 0;
    if (threadIdx.x == 0) *done_ctr = 0;
  }
#endif
}

}  // namespace

// ---------------------------------------------------------------- host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}

// 2D byte matrix [rows][cols_bytes] viewed as boxes of (TM rows x 128 B) with the 128B swizzle
static bool make_map_2d(CUtensorMap* map, const void* base, uint64_t rows, uint64_t cols_bytes) {
  cuuint64_t gdim[2] = {cols_bytes, rows};
  cuuint64_t gstride[1] = {cols_bytes};
  cuuint32_t box[2] = {128, TM};
  cuuint32_t estride[2] = {1, 1};
  EncodeTiledFn fn = get_encode_fn();
  if (!fn) return false;
  return fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

struct MapCache {
  const void* base = nullptr;
  CUtensorMap map;
};

extern "C" int moe_workspace_bytes(int T) { return T * WS_FLOATS * 4; }

extern "C" int moe_run(const void* logits, const void* bias, const void* hs, const void* hs_scale,
                       const void* w13, const void* s13, const void* w2, const void* s2,
                       int local_offset, float scaling, void* out, void* workspace, int T,
                       void* stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  static bool init_done = false;
  static int num_sms = 148;
  static int coop = 1;
  static MapCache map13, map2;
  if (!init_done) {
    cudaFuncSetAttribute(moe_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moe_fused_kernel, THREADS, SMEM_BYTES);
    if (per_sm < 1) return -2;
    init_done = true;
  }
  if (map13.base != w13) {
    if (!make_map_2d(&map13.map, w13, (uint64_t)E_LOCAL * G1_OUT, HIDDEN)) return -1;
    map13.base = w13;
  }
  if (map2.base != w2) {
    if (!make_map_2d(&map2.map, w2, (uint64_t)E_LOCAL * HIDDEN, INTER)) return -1;
    map2.base = w2;
  }
  const float* logits_f = (const float*)logits;
  const __nv_bfloat16* bias_b = (const __nv_bfloat16*)bias;
  const uint8_t* hs_u = (const uint8_t*)hs;
  const float* hss = (const float*)hs_scale;
  const float* s13f = (const float*)s13;
  const float* s2f = (const float*)s2;
  float* ws = (float*)workspace;
  __nv_bfloat16* outb = (__nv_bfloat16*)out;
  void* args[] = {(void*)&map13.map, (void*)&map2.map, (void*)&logits_f, (void*)&bias_b, (void*)&local_offset,
                  (void*)&scaling,   (void*)&hs_u,     (void*)&hss,      (void*)&s13f,   (void*)&s2f,
                  (void*)&ws,        (void*)&outb,     (void*)&T};
  cudaError_t err;
  static int nocoop = -1;
  if (nocoop < 0) nocoop = getenv("MOE_NOCOOP") ? 1 : 0;   // dev experiment switch
  if (coop && !nocoop) {
    err = cudaLaunchCooperativeKernel((const void*)moe_fused_kernel, dim3(num_sms, T), dim3(THREADS), args,
                                      SMEM_BYTES, stream);
  } else {
    err = cudaLaunchKernel((const void*)moe_fused_kernel, dim3(num_sms, T), dim3(THREADS), args, SMEM_BYTES,
                           stream);
  }
  if (err != cudaSuccess) return (int)err;
  return (int)cudaGetLastError();
}
