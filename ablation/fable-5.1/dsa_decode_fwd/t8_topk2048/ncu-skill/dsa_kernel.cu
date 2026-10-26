// DSA sparse decode attention (DeepSeek-V3.2 style, 16 heads, ckv 512 + kpe 64, top-k 2048), written from
// scratch for B200.  One kernel launch, two roles per CTA:
//   chunk role : CTA b handles request t = b / NCHUNK, keys [c*S, (c+1)*S) with c = b % NCHUNK.  Gathers the S selected
//                rows (bf16) into a swizzled smem tile with cp.async, computes QK^T and P.V with mma.sync
//                tensor cores, writes the un-normalised partial (m, l, o) to a global workspace, sets its bit in
//                the request's active-chunk mask and bumps the request's arrival counter.
//   combine role: CTA b handles request t2 = b / 64, head h = (b % 64) / 4, dims [q*128, q*128+128) (one dim per
//                thread).  It polls the request's counter line (arrivals + mask in one 16B load), then reads
//                the active partials in one round trip and writes output (bf16) and lse (log2).
// Chunk work never waits, so with every CTA resident (512 CTAs, 4 per SM) there is no deadlock; the counter line
// is reset by the last combine CTA of each request.
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef KEYS_PER_CTA
#define KEYS_PER_CTA 32
#endif
#ifndef PARTIAL_BF16
#define PARTIAL_BF16 1
#endif
#ifndef POLL_NS
#define POLL_NS 128
#endif
#ifndef MAP_MODE
#define MAP_MODE 3   // 0: t-major; 1: rotated chunk-major; 3: placement-aware (SM table recorded by the previous launch)
#endif
#define MAXBLK 512   // max blocks supported by the schedule table (8 KB of kernel parameters)
// Per-block static schedule (host-built from the probed block->SM placement, data-independent):
//   e[0] = own (primary) chunk slot (t*64 + c)
//   e[7] = role: 0 = plain (no takeover logic), 1 = holder (slot-0 block), 2.. = slot k block (k = role - 1)
//   holder:  e[1],e[3],e[5] = candidate chunks (slot 1..3 of the next SMs), e[2],e[4],e[6] = their SMs' slot-0 chunks
//   slot-k:  e[1] = own SM's slot-0 chunk, e[2] = holder's slot-0 chunk, e[3],e[4] / e[5],e[6] = holder's earlier
//            candidates and their slot-0 chunks (k-1 of them).   0xFFFF = none.
struct MapTab { unsigned short e[MAXBLK][8]; };
#ifndef POLL_WARPS
#define POLL_WARPS 4  // warps that poll the counter line with staggered phase
#endif

namespace {

constexpr int H = 16, DC = 512, DP = 64, TOPK = 2048, DQK = DC + DP;
constexpr int S = KEYS_PER_CTA;
constexpr int WARPS = 4;
constexpr int NT = WARPS * 32;             // 128
constexpr int NCHUNK = TOPK / S;           // 64
constexpr int KSTEPS = DQK / 16;           // 36
constexpr int KSW = KSTEPS / WARPS;        // 9 k-steps per warp in QK^T
constexpr int NTILE = S / 8;               // key tiles of 8 in QK^T (4)
constexpr int DPW = DC / WARPS;            // dims per warp in P.V (128)
#ifndef GATHER_BULK
#define GATHER_BULK 0
#endif
#ifndef EVICT_FIRST
#define EVICT_FIRST 0
#endif
#ifndef PREFETCH_WS
#define PREFETCH_WS 0   // 1: prefetch this chunk's workspace lines into L2 during the gather
#endif
#ifndef PIPE
#define PIPE 1        // 1: two cp.async groups (dims 0..255 first) so QK^T on them overlaps the rest of the gather
#endif
#ifndef PIPE_SPLIT
#define PIPE_SPLIT 32  // 16B columns of Q/K in gather group A (dims 0 .. 8*PIPE_SPLIT-1); QK k-steps < PIPE_SPLIT/2 run first
#endif
#ifndef SPEC_CNT
#define SPEC_CNT 3    // PROGW==3: speculate only once the request's arrival count is within this of NCHUNK
#endif
#ifndef SPEC_RETRY_NS
#define SPEC_RETRY_NS 32
#endif
#ifndef PROLOAD
#define PROLOAD 0
#endif
#ifndef TAGVAL
#define TAGVAL 0      // PROGW==1: validate partial/m,l tags after the poll loop (allows SIGNAL=9: no producer fence)
#endif
#ifndef TAGPROD
#define TAGPROD 0   // producer writes tagged partials / m,l words
#endif
#ifndef GUARD1
#define GUARD1 0      // PROGW==3: issue loads only on warp-uniform change
#endif
#ifndef TAGPACK
#define TAGPACK 1
#endif
#ifndef SPEC_PROG
#define SPEC_PROG 1
#endif
#ifndef SPEC_CP
#define SPEC_CP 1     // PROGW==3: 1 = cp.async staging in smem, 0 = predicated register loads
#endif
#ifndef NOTAB
#define NOTAB 1
#endif
#ifndef CTAB
#define CTAB 0        // 1: read the schedule table from __constant__ memory (uploaded once) instead of the kernel param
#endif
#ifndef P5_SPIN
#define P5_SPIN 6      // PROGW==5: smem checks for another warp's final line while a poll is in flight
#endif
#ifndef P5_SPIN_NS
#define P5_SPIN_NS 64
#endif
#ifndef EARLYQ
#define EARLYQ 1      // NOTAB: issue the Q / q_pe tile loads at kernel start (before the index load returns)
#endif
#ifndef COMBROT
#define COMBROT 1     // combine role of request t runs on the chunk blocks of request (t + T/2) % T, whose chunk work is
                      // independent of t's arrivals (a block busy with an active chunk would otherwise issue all its
                      // progressive loads in one late burst)
#endif
#ifndef WPH2
#define WPH2 0        // PROGW==1 tail: 2 barriers (per-group weights + group merge) instead of 3
#endif
#ifndef POLL6_HALF
#define POLL6_HALF 200   // PROGW==6: offset between the two in-flight polls (ns)
#endif
#ifndef QKSPLIT
#define QKSPLIT 0     // 1: two independent QK^T accumulator sets (even/odd k-steps)
#endif
#ifndef IDXLD
#define IDXLD 0       // 1: index block loads via ld.global.cg instead of ld.global.nc
#endif
#ifndef PUBEARLY
#define PUBEARLY 0    // SIGNAL 9 only: active chunks publish their mask bit / arrival right after P.V, before the stores
#endif
#ifndef BCAST
#define BCAST 0       // PROGW==1: broadcast the final line to the other warps through smem
#endif
#ifndef P5ACQ
#define P5ACQ 1
#endif
#ifndef SPEC_MAX
#define SPEC_MAX 1    // PROGW==3: speculative loads of the lane's missing active chunks once at most this many remain
#endif
#ifndef CTR_REP
#define CTR_REP 1     // replicated counter lines per (request, parity): pollers spread over the copies
#endif
#ifndef PIPE_POLL
#define PIPE_POLL 4   // PROGW==2: polls in flight per warp
#endif
#ifndef POLLACQ
#define POLLACQ 0     // PROGW==2: 1 = ld.acquire polls (no trailing fence), 0 = relaxed polls + fence
#endif
#ifndef TIM_REL
#define TIM_REL 0
#endif
#ifndef SIGNAL
#define SIGNAL 1      // 9: no producer fence (partials are tag-validated by the consumer) ; 0: red.release ; 1: __threadfence + red.relaxed ; 2: fence.acq_rel.gpu + red.relaxed (fastest)
#endif
#ifndef OML
#define OML 0         // 1: (m, l) embedded in the o-partial rows (272 B segments: 256 B data + 16 B {m, l})
#endif
#define OML_ROW 1088  // bytes per head row when OML (4 x 272)
#ifndef MLPF
#define MLPF 0        // 1: combine CTAs prefetch their (m, l) lines + counter line into L2 at kernel start
#endif
#ifndef OML2
#define OML2 0        // 1: each 16 KB o-partial slot is followed by one full 128 B line {m[16], l[16]}
#endif
#define OSLOT_B (H * DC * 2 + 128)  // bytes per chunk slot when OML2
#ifndef PROGML
#define PROGML 0      // 1: (m, l) as one full line per chunk slot, prefetched progressively by the polling warps
#endif
#ifndef IDLE_DELAY
#define IDLE_DELAY 0  // ns: CTAs whose chunk the host marked likely-idle (table bit 15) wait before their index load
#endif
#ifndef MLTOUCH
#define MLTOUCH 0     // 1: combine CTAs load one word of each of their 4 (m, l) lines at kernel start (make lines valid in L2)
#endif
#ifndef CTRPAR
#define CTRPAR 1      // 1: two counter lines per request selected by launch parity; no done-ticket / reset atomics
#endif
#ifndef LATEDONE
#define LATEDONE 0    // 1: take the done ticket at the very end instead of right after the poll
#endif
#ifndef NOLSE
#define NOLSE 0       // measurement only: skip the lse writes
#endif
#ifndef LSEPF
#define LSEPF 0       // 1: prefetch the lse lines into L2 at kernel start (partial-line writes then merge into resident lines)
#endif
#ifndef PROGW
#define PROGW 0       // 1: each warp polls the line itself and loads its own chunks' partials as their bits appear
#endif
#ifndef MLVEC
#define MLVEC 0       // 1: m/l rows via 16 B vector loads + smem
#endif
#ifndef WARPCOPY
#define WARPCOPY 1    // 1: per-warp copy-out of the staged partial (no block barrier)
#endif
#ifndef POLL_STAGGER
#define POLL_STAGGER (POLL_NS + 150)
#endif
#ifndef POLLREL
#define POLLREL 0     // 1: relaxed polling loads + one fence.acq_rel.gpu after success
#endif
#ifndef CPA_HINT
#define CPA_HINT 0   // 1: cp.async gather with an L2 evict_first cache hint (swizzled layout kept)
#endif
constexpr int Q_STRIDE = DC * 2;           // Q rows: 1024 B, 16B-chunk XOR swizzle (chunk ^ (row & 7))
constexpr int QPE_STRIDE = DP * 2;         // 128 B rows, swizzled the same way
#if GATHER_BULK
constexpr int KV_STRIDE = DC * 2 + 16;     // 1040 B rows: bulk copies land contiguous, padding avoids bank conflicts
constexpr int PE_STRIDE = DP * 2;          // 128 B rows, unswizzled (8-way conflicts on the small PE part)
constexpr int P_STRIDE = S * 2;            // padding dropped to fit 4 CTAs/SM
#define XK(c) ((c) << 4)                   // byte offset of 16B chunk c inside a KV/PE row
#else
constexpr int KV_STRIDE = DC * 2;          // 1024 B rows, swizzled
constexpr int PE_STRIDE = DP * 2;
constexpr int P_STRIDE = S * 2 + 16;       // bf16 P rows, padded (80 B)
#define XK(c) (((c) ^ x) << 4)             // (x == row & 7 at every use)
#endif
constexpr int OFF_KV = 0;
constexpr int OFF_PE = OFF_KV + S * KV_STRIDE;
constexpr int OFF_Q = OFF_PE + S * PE_STRIDE;
constexpr int OFF_QPE = OFF_Q + H * Q_STRIDE;
constexpr int OFF_P = OFF_QPE + H * QPE_STRIDE;
constexpr int OFF_ML = OFF_P + H * P_STRIDE;           // float[32]: m[16], l[16]
constexpr int OFF_IDX = OFF_ML + 32 * 4;               // int[S]
constexpr int OFF_MISC = OFF_IDX + S * 4;              // u64 mbarrier at +8
constexpr int SMEM_BYTES = OFF_MISC + 32;
constexpr int OFF_SLOG = OFF_Q;                        // after QK^T: float[WARPS][16][S] partial logits (8 KB)
constexpr int OFF_STAGE = OFF_KV;
constexpr int OFF_CSTG = 8192;                         // combine role: progressive partial staging (16 KB + 1 KB)                      // after P.V: staged partial [16][512] (bf16 or fp32)
#if PARTIAL_BF16
typedef __nv_bfloat16 part_t;
constexpr int STG_STRIDE = DC * 2 + 16;                // 1040 B rows: conflict-free 16B row segments
#else
typedef float part_t;
constexpr int STG_STRIDE = DC * 4 + 32;                // 2080 B rows: 2 wavefronts per float2 warp store
#endif
constexpr int SLOG_STRIDE = S + 8;                     // floats per row of the partial-logit tiles (conflict-free)
constexpr int COMB_PER_REQ = H * (DC / NT);            // 64 combine CTAs per request
constexpr int CTR_STRIDE = 32;                         // uint32 per request: [0]=arrivals [1]=done [2],[3]=mask
static_assert(WARPS * 16 * SLOG_STRIDE * 4 <= H * Q_STRIDE, "partial logits fit in the Q region");
static_assert(H * STG_STRIDE <= S * KV_STRIDE, "staged partial fits in the KV region");
static_assert(NCHUNK <= 64 && NT >= NCHUNK, "combine role: one chunk per thread for m, l");
static_assert(S % 16 == 0 && KSTEPS % WARPS == 0 && DC % NT == 0, "shape");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
#if CPA_HINT
__device__ __forceinline__ void cp_async16_hint(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(pol) : "memory");
}
#define CP_KV(dst, src) cp_async16_hint((dst), (src), ef_pol)
#else
#define CP_KV(dst, src) cp_async16((dst), (src))
#endif
__device__ __forceinline__ uint64_t make_evict_first_policy() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}\n" ::"r"(mbar), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst), "l"(src), "r"(bytes), "r"(mbar), "l"(pol) : "memory");
}
__device__ __forceinline__ void cp_async16_pred(uint32_t saddr, const void* g, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %2, 0;\n@q cp.async.cg.shared.global [%0], [%1], 16;\n}" ::"r"(saddr), "l"(g), "r"(pred) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// bf16x2 with 6 explicit mantissa bits (RNE) and the per-element tag bit in bit 0
__device__ __forceinline__ uint32_t pack_bf16x2_tag(float a, float b, unsigned ta, unsigned tb) {
#if TAGPACK == 1
  // RNE bf16x2, then force bit 0 of each half to the tag by adding one where it differs (nearest neighbour)
  const uint32_t x = pack_bf16x2(a, b);
  return x + ((x ^ (ta | (tb << 16))) & 0x00010001u);
#else
  unsigned ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua = (ua + 0xFFFFu + ((ua >> 17) & 1u)) & 0xFFFE0000u;
  ub = (ub + 0xFFFFu + ((ub >> 17) & 1u)) & 0xFFFE0000u;
  return ((ua >> 16) | ta) | (((ub >> 16) | tb) << 16);
#endif
}
__device__ __forceinline__ unsigned lsbtag8(const uint4& v) {
  return (v.x & 1u) | (((v.x >> 16) & 1u) << 1) | ((v.y & 1u) << 2) | (((v.y >> 16) & 1u) << 3) |
         ((v.z & 1u) << 4) | (((v.z >> 16) & 1u) << 5) | ((v.w & 1u) << 6) | (((v.w >> 16) & 1u) << 7);
}
__device__ __forceinline__ uint4 ld_relaxed_v4(const unsigned* p) {
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ld_relaxed_v4_pred(uint4& d, const unsigned* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %4, 0;\n@q ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%5];\n}"
               : "+r"(d.x), "+r"(d.y), "+r"(d.z), "+r"(d.w) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ void ld_acquire_v4_pred(uint4& d, const unsigned* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %4, 0;\n@q ld.acquire.gpu.global.v4.u32 {%0,%1,%2,%3}, [%5];\n}"
               : "+r"(d.x), "+r"(d.y), "+r"(d.z), "+r"(d.w) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ uint4 ld_acquire_v4(const unsigned* p) {
  uint4 v;
  asm volatile("ld.acquire.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void red_relaxed_gpu_add(unsigned* p, unsigned v) { asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_relaxed_gpu_or(unsigned* p, unsigned v) { asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_gpu_add(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void red_release_gpu_or(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.or.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_add_relaxed_gpu(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
// predicated loads that write the destination register only when pred != 0 (consumer waits via scoreboard)
__device__ __forceinline__ void ld_cg_u16_pred(unsigned short& d, const void* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %1, 0;\n@q ld.global.cg.u16 %0, [%2];\n}"
               : "+h"(d) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ float bf16lo(uint32_t w) { return __bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xffffu))); }
__device__ __forceinline__ float bf16hi(uint32_t w) { return __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16))); }
__device__ __forceinline__ void ld_cg_v4_pred(uint4& d, const void* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %4, 0;\n@q ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n}"
               : "+r"(d.x), "+r"(d.y), "+r"(d.z), "+r"(d.w) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ void ld_cg_v4_pred_f(float4& d, const void* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %4, 0;\n@q ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%5];\n}"
               : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ void ld_cg_f32_pred(float& d, const void* p, unsigned pred) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %1, 0;\n@q ld.global.cg.f32 %0, [%2];\n}"
               : "+f"(d) : "r"(pred), "l"(p) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// swizzled byte offset of 16B chunk j in row r (rows of 1024 or 128 bytes)
__device__ __forceinline__ uint32_t swz(int r, int j, int stride) { return r * stride + ((j ^ (r & 7)) << 4); }
}  // namespace

#ifdef DSA_TIMING
extern "C" __device__ unsigned long long* g_tbuf;
#define TSTAMP(i) do { if (threadIdx.x == 0) g_tbuf[blockIdx.x * 16 + (i)] = gtimer(); } while (0)
#define TSTAMP_LANE(i) do { g_tbuf[blockIdx.x * 16 + (i)] = gtimer(); } while (0)
#define TDEP(x) asm volatile("" ::"f"(x))
#else
#define TSTAMP(i) do {} while (0)
#define TSTAMP_LANE(i) do {} while (0)
#define TDEP(x) do {} while (0)
#endif

#if CTAB
extern "C" __constant__ MapTab c_tab;
#define TAB c_tab
#else
#define TAB tab
#endif
extern "C" __global__ void __launch_bounds__(NT, 4) dsa_fwd(const __nv_bfloat16* __restrict__ q_nope,
                                                            const __nv_bfloat16* __restrict__ q_pe,
                                                            const __nv_bfloat16* __restrict__ ckv,
                                                            const __nv_bfloat16* __restrict__ kpe,
                                                            const int32_t* __restrict__ sidx_g, float scale_log2,
                                                            part_t* __restrict__ o_part,  // [T*NCHUNK][16][512]
                                                            float* __restrict__ ml_part,  // [T][2][16][NCHUNK] (m, l)
                                                            unsigned* __restrict__ ctr,   // [T][CTR_STRIDE] lines, then SM table [MAXBLK] at ctr + CTR_STRIDE * 64
                                                            __nv_bfloat16* __restrict__ out,
                                                            float* __restrict__ lse, int num_tokens,
                                                            const MapTab tab, unsigned* probe, unsigned epoch) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (probe != nullptr && tid == 0) { unsigned mysm; asm volatile("mov.u32 %0, %%smid;" : "=r"(mysm)); probe[blockIdx.x] = mysm; }
  const uint32_t sbase = smem_u32(smem);
  int* sidx = reinterpret_cast<int*>(smem + OFF_IDX);
  float* sml = reinterpret_cast<float*>(smem + OFF_ML);
  // ================= chunk role: static primary + data-dependent takeover =================
  TSTAMP(0);
#if EARLYQ && NOTAB
  {
    // the chunk's request is static (chunk == blockIdx): fetch its Q tile now, in the shadow of the index load
    const int tq = (int)blockIdx.x / NCHUNK;
#pragma unroll
    for (int k = 0; k < (H * 64) / NT; ++k) {
      const int i = tid + k * NT, r = i >> 6, j = i & 63;
      cp_async16(sbase + OFF_Q + swz(r, j, Q_STRIDE), q_nope + ((size_t)tq * H + r) * DC + j * 8);
    }
    {
      const int r = tid >> 3, j = tid & 7;  // NT == H * 8
      cp_async16(sbase + OFF_QPE + swz(r, j, QPE_STRIDE), q_pe + ((size_t)tq * H + r) * DP + j * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#endif
  int* idxbuf = reinterpret_cast<int*>(smem + OFF_KV);        // [7][32] index blocks (KV region, free for now)
  unsigned* actb = reinterpret_cast<unsigned*>(smem + OFF_KV + 7 * 32 * 4);  // [7] activity flags, [8] chosen chunk
  int my_chunk, primary_chunk;
  bool took_over, stood_down;
#if PROGW == 3 || PROGW == 7 || PROLOAD
  int4 ca = make_int4(-1, -1, -1, -1), cb = ca, cc4 = ca, cd = ca;
#endif
  {
    const int bb = (blockIdx.x < MAXBLK) ? (int)blockIdx.x : 0;
#if NOTAB
    const int role = 0;
#else
    const int role = (blockIdx.x < MAXBLK) ? (int)TAB.e[bb][7] : 0;
#endif
    // load the index blocks of up to 7 chunks (warp w: chunks w and w+4)
    int vv[2]; int chs[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {   // issue every index load first (independent), then evaluate
      const int i = warp + 4 * r;
      vv[r] = -1; chs[r] = 0xFFFF;
      if (i < 7) {
#if NOTAB
        chs[r] = (i == 0) ? (int)blockIdx.x : 0xFFFF;
#else
        chs[r] = (blockIdx.x < MAXBLK) ? (int)TAB.e[bb][i] : ((i == 0) ? (int)blockIdx.x : 0xFFFF);
#endif
#if IDXLD
        if (chs[r] != 0xFFFF) { int v_; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v_) : "l"(sidx_g + (size_t)(chs[r] / NCHUNK) * TOPK + (chs[r] % NCHUNK) * S + lane)); vv[r] = v_; }
#else
        if (chs[r] != 0xFFFF) vv[r] = __ldg(sidx_g + (size_t)(chs[r] / NCHUNK) * TOPK + (chs[r] % NCHUNK) * S + lane);
#endif
      }
    }
#if PROGW == 3 || PROGW == 7 || PROLOAD
    // the combine role's request activity (16 ints per thread -> chunk tid/2), issued now so its latency overlaps
    if (blockIdx.x < (unsigned)(num_tokens * COMB_PER_REQ)) {
#if COMBROT
      const int tq2 = (int)((blockIdx.x / COMB_PER_REQ + (unsigned)((num_tokens + 1) / 2)) % (unsigned)num_tokens);
#else
      const int tq2 = (int)(blockIdx.x / COMB_PER_REQ);
#endif
      const int4* ip = reinterpret_cast<const int4*>(sidx_g + (size_t)tq2 * TOPK + tid * 16);
      ca = __ldg(ip); cb = __ldg(ip + 1); cc4 = __ldg(ip + 2); cd = __ldg(ip + 3);
    }
#endif
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int i = warp + 4 * r;
      if (i < 7) {
        idxbuf[i * 32 + lane] = vv[r];
        const unsigned any = __ballot_sync(0xffffffffu, vv[r] >= 0);
        if (lane == 0) actb[i] = (chs[r] != 0xFFFF && any) ? 1u : 0u;
      }
    }
    __syncthreads();
    // decision (identical in every block that looks at the same chunks)
    int chosen = -1;  // index into idxbuf of the chunk to process, -1 = idle
    if (role == 0) {
      chosen = actb[0] ? 0 : -1;
    } else if (role == 1) {  // holder: own if active, else the first surplus candidate
      if (actb[0]) chosen = 0;
      else {
#pragma unroll
        for (int m = 1; m <= 3; ++m)
          if (chosen < 0 && actb[2 * m - 1] && actb[2 * m]) chosen = 2 * m - 1;
      }
    } else {                 // slot k = role - 1
      const int k = role - 1;
      bool taken = false;
      if (actb[0] && actb[1] && !actb[2]) {   // surplus and holder free: taken unless an earlier candidate was
        taken = true;
#pragma unroll
        for (int m = 1; m < 3; ++m)
          if (m < k && actb[3 + 2 * (m - 1)] && actb[4 + 2 * (m - 1)]) taken = false;
      }
      chosen = (actb[0] && !taken) ? 0 : -1;
    }
#if NOTAB
    my_chunk = (int)blockIdx.x; primary_chunk = my_chunk;
    if (tid < S) sidx[tid] = (chosen >= 0) ? idxbuf[tid] : -1;
#else
    if (chosen >= 0) {
      const int ch = (blockIdx.x < MAXBLK) ? (int)TAB.e[bb][chosen] : (int)blockIdx.x;
      my_chunk = ch;
      if (tid < S) sidx[tid] = idxbuf[chosen * 32 + tid];
    } else {
      my_chunk = (blockIdx.x < MAXBLK) ? (int)TAB.e[bb][0] : (int)blockIdx.x;  // idle: keep a valid slot id
      if (tid < S) sidx[tid] = -1;
    }
    primary_chunk = (blockIdx.x < MAXBLK) ? (int)TAB.e[bb][0] : (int)blockIdx.x;
#endif
    took_over = (role == 1 && chosen >= 1);                       // holder processing a candidate (own primary idle)
    stood_down = (role >= 2 && actb[0] && chosen < 0);            // slot-k block whose active chunk was taken
  }
#if PROGW == 3 || PROGW == 7 || PROLOAD
  unsigned cm16;
  {
    const bool anyv = (ca.x >= 0) | (ca.y >= 0) | (ca.z >= 0) | (ca.w >= 0) | (cb.x >= 0) | (cb.y >= 0) | (cb.z >= 0) | (cb.w >= 0) |
                      (cc4.x >= 0) | (cc4.y >= 0) | (cc4.z >= 0) | (cc4.w >= 0) | (cd.x >= 0) | (cd.y >= 0) | (cd.z >= 0) | (cd.w >= 0);
    const unsigned bal = __ballot_sync(0xffffffffu, anyv);
    cm16 = 0u;
#pragma unroll
    for (int i = 0; i < 16; ++i) cm16 |= (((bal >> (2 * i)) & 3u) ? 1u : 0u) << i;
  }
#if PROLOAD && PROGW != 3
  if (cm16 == 0xFFFFFFFFu && tid == 0) ctr[0] = cm16;  // never true: keeps the loads alive
#endif
#endif
  const int t = my_chunk / NCHUNK, c = my_chunk % NCHUNK;
  const int slot = t * NCHUNK + c;
  __syncthreads();
  const int nvalid = __syncthreads_count(tid < S && sidx[tid] >= 0);
  const int myidx = (tid < S) ? sidx[tid] : -1;
  const bool valid = (tid < S) && (myidx >= 0);
#if CTRPAR
  // the block whose PRIMARY chunk is chunk 0 of its request clears that request's other parity line for the next
  // launch (exactly one such block per request; it runs this even if it stands down)
  if (tid < CTR_REP && (primary_chunk % NCHUNK) == 0)
    *reinterpret_cast<uint4*>(ctr + (((primary_chunk / NCHUNK) * 2 + ((epoch & 1) ^ 1)) * CTR_REP + tid) * CTR_STRIDE) = make_uint4(0u, 0u, 0u, 0u);
#endif
  if (took_over && tid == 0) {
    // the holder's own primary chunk is idle: account for it right away (no mask bit)
    const int tp = primary_chunk / NCHUNK;
#if CTRPAR
#pragma unroll
    for (int r = 0; r < CTR_REP; ++r) red_relaxed_gpu_add(ctr + ((tp * 2 + (epoch & 1)) * CTR_REP + r) * CTR_STRIDE, 1u);
#else
    red_relaxed_gpu_add(ctr + tp * CTR_STRIDE, 1u);
#endif
  }
  TSTAMP(1);

  if (nvalid > 0) {
#if CPA_HINT
    const uint64_t ef_pol = make_evict_first_policy();
#endif
#if PREFETCH_WS
    {
      // warm the L2 lines this CTA will store to: its 16 KB o-partial slot (one line per thread), the 32 scattered
      // (m, l) words and the request's counter line, so the later stores hit resident lines (no dirty eviction)
      const char* op = reinterpret_cast<const char*>(o_part + (size_t)slot * H * DC) + tid * 128;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(op));
      if (tid < 32) {
        const float* mp = ml_part + (((size_t)t * 2 + (tid >> 4)) * H + (tid & 15)) * NCHUNK + c;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(mp));
      }
      if (tid == 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(ctr + t * CTR_STRIDE));
    }
#endif
    // Q tile: 64 chunks per head of q_nope, 8 per head of q_pe (same round trip as the KV gather)
#if PIPE
    // group A: 16B-columns 0..31 (dims 0..255) of Q and K ; group B: the rest (+ q_pe, kpe)
    static_assert((H * PIPE_SPLIT) % NT == 0 && (S * PIPE_SPLIT) % NT == 0 && (H * (64 - PIPE_SPLIT)) % NT == 0 && (S * (64 - PIPE_SPLIT)) % NT == 0, "split");
#if !(EARLYQ && NOTAB)
#pragma unroll
    for (int k = 0; k < (H * PIPE_SPLIT) / NT; ++k) {
      const int i = tid + k * NT, r = i / PIPE_SPLIT, j = i % PIPE_SPLIT;
      cp_async16(sbase + OFF_Q + swz(r, j, Q_STRIDE), q_nope + ((size_t)t * H + r) * DC + j * 8);
    }
#endif
#pragma unroll
    for (int k = 0; k < (S * PIPE_SPLIT) / NT; ++k) {
      const int i = tid + k * NT, r = i / PIPE_SPLIT, j = i % PIPE_SPLIT;
      const int ix = sidx[r];
      const uint32_t off = OFF_KV + swz(r, j, KV_STRIDE);
      if (ix >= 0) CP_KV(sbase + off, ckv + (size_t)ix * DC + j * 8);
      else *reinterpret_cast<int4*>(smem + off) = make_int4(0, 0, 0, 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#if !(EARLYQ && NOTAB)
#pragma unroll
    for (int k = 0; k < (H * (64 - PIPE_SPLIT)) / NT; ++k) {
      const int i = tid + k * NT, r = i / (64 - PIPE_SPLIT), j = PIPE_SPLIT + i % (64 - PIPE_SPLIT);
      cp_async16(sbase + OFF_Q + swz(r, j, Q_STRIDE), q_nope + ((size_t)t * H + r) * DC + j * 8);
    }
#endif
#pragma unroll
    for (int k = 0; k < (S * (64 - PIPE_SPLIT)) / NT; ++k) {
      const int i = tid + k * NT, r = i / (64 - PIPE_SPLIT), j = PIPE_SPLIT + i % (64 - PIPE_SPLIT);
      const int ix = sidx[r];
      const uint32_t off = OFF_KV + swz(r, j, KV_STRIDE);
      if (ix >= 0) CP_KV(sbase + off, ckv + (size_t)ix * DC + j * 8);
      else *reinterpret_cast<int4*>(smem + off) = make_int4(0, 0, 0, 0);
    }
#else
#pragma unroll
    for (int k = 0; k < (H * 64) / NT; ++k) {
      const int i = tid + k * NT, r = i >> 6, j = i & 63;
      cp_async16(sbase + OFF_Q + swz(r, j, Q_STRIDE), q_nope + ((size_t)t * H + r) * DC + j * 8);
    }
#endif
#if !(EARLYQ && NOTAB)
    {
      const int r = tid >> 3, j = tid & 7;  // NT == H * 8
      cp_async16(sbase + OFF_QPE + swz(r, j, QPE_STRIDE), q_pe + ((size_t)t * H + r) * DP + j * 8);
    }
#endif
#if GATHER_BULK
    const uint32_t mbar = sbase + OFF_MISC + 8;
    if (tid == 0) { mbar_init(mbar, 1); }
    __syncthreads();
    if (tid == 0) mbar_arrive_expect_tx(mbar, (uint32_t)nvalid * (DC + DP) * 2);
    if (tid < S) {
      if (myidx >= 0) {
#if EVICT_FIRST
        const uint64_t pol = make_evict_first_policy();
        bulk_g2s_hint(sbase + OFF_KV + tid * KV_STRIDE, ckv + (size_t)myidx * DC, DC * 2, mbar, pol);
        bulk_g2s_hint(sbase + OFF_PE + tid * PE_STRIDE, kpe + (size_t)myidx * DP, DP * 2, mbar, pol);
#else
        bulk_g2s(sbase + OFF_KV + tid * KV_STRIDE, ckv + (size_t)myidx * DC, DC * 2, mbar);
        bulk_g2s(sbase + OFF_PE + tid * PE_STRIDE, kpe + (size_t)myidx * DP, DP * 2, mbar);
#endif
      } else {
        int4* d = reinterpret_cast<int4*>(smem + OFF_KV + tid * KV_STRIDE);
#pragma unroll 8
        for (int i = 0; i < DC * 2 / 16; ++i) d[i] = make_int4(0, 0, 0, 0);
        int4* e = reinterpret_cast<int4*>(smem + OFF_PE + tid * PE_STRIDE);
#pragma unroll
        for (int i = 0; i < DP * 2 / 16; ++i) e[i] = make_int4(0, 0, 0, 0);
      }
    }
    cp_async_wait_all();   // Q
    mbar_wait(mbar, 0);    // KV
    __syncthreads();
    TSTAMP(2);
#else
#if !PIPE
#pragma unroll
    for (int k = 0; k < (S * 64) / NT; ++k) {
      const int i = tid + k * NT, r = i >> 6, j = i & 63;
      const int ix = sidx[r];
      const uint32_t off = OFF_KV + swz(r, j, KV_STRIDE);
      if (ix >= 0) CP_KV(sbase + off, ckv + (size_t)ix * DC + j * 8);
      else *reinterpret_cast<int4*>(smem + off) = make_int4(0, 0, 0, 0);
    }
#endif
#pragma unroll
    for (int k = 0; k < (S * 8) / NT; ++k) {
      const int i = tid + k * NT, r = i >> 3, j = i & 7;
      const int ix = sidx[r];
      const uint32_t off = OFF_PE + swz(r, j, PE_STRIDE);
      if (ix >= 0) CP_KV(sbase + off, kpe + (size_t)ix * DP + j * 8);
      else *reinterpret_cast<int4*>(smem + off) = make_int4(0, 0, 0, 0);
    }
#if PIPE
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
#else
    cp_async_wait_all();
#endif
    __syncthreads();
    TSTAMP(2);
#endif


    // ---- QK^T : warp w owns k-steps [w*KSW, (w+1)*KSW) of the 36 (16-dim) k-steps, all 16 heads x S keys ----
    float sacc[NTILE][4];
#pragma unroll
    for (int i = 0; i < NTILE; ++i) sacc[i][0] = sacc[i][1] = sacc[i][2] = sacc[i][3] = 0.f;
#if QKSPLIT
    float saccB[NTILE][4];
#pragma unroll
    for (int i = 0; i < NTILE; ++i) saccB[i][0] = saccB[i][1] = saccB[i][2] = saccB[i][3] = 0.f;
#define QACC(n) ((i & 1) ? saccB[n] : sacc[n])
#else
#define QACC(n) sacc[n]
#endif
    {
      const int arow = (lane & 7) + 8 * ((lane >> 3) & 1);  // A (Q) row for ldmatrix x4; chunk 2ks + (lane>>4)
      const int bsub = (lane & 7) + 8 * (lane >> 4);         // B (K) row within a pair of key tiles; chunk 2ks + ((lane>>3)&1)
      const int x = lane & 7;                                // row & 7 for the swizzle (all rows used are == lane&7 mod 8)
#if PIPE
      // k-steps {warp + 4i}: i < 4 are dims 0..255 (gather group A), the rest need group B
#define QK_KS(i) (warp + 4 * (i))
#else
#define QK_KS(i) (warp * KSW + (i))
#endif
#pragma unroll
      for (int i = 0; i < KSW; ++i) {
#if PIPE
        if (i == (PIPE_SPLIT / 8)) { asm volatile("cp.async.wait_group 0;" ::: "memory"); __syncthreads(); TSTAMP(12); }
#endif
        const int ks = QK_KS(i);  // warp-uniform
        uint32_t a0, a1, a2, a3;
        if (ks < DC / 16) {
          const int ch = 2 * ks;
          ldsm_x4(a0, a1, a2, a3, sbase + OFF_Q + arow * Q_STRIDE + (((ch + (lane >> 4)) ^ x) << 4));
#pragma unroll
          for (int n = 0; n < NTILE; n += 2) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3,
                    sbase + OFF_KV + (n * 8 + bsub) * KV_STRIDE + XK(ch + ((lane >> 3) & 1)));
            mma_bf16(QACC(n), a0, a1, a2, a3, b0, b1);
            mma_bf16(QACC(n + 1), a0, a1, a2, a3, b2, b3);
          }
        } else {
          const int ch = 2 * (ks - DC / 16);
          ldsm_x4(a0, a1, a2, a3, sbase + OFF_QPE + arow * QPE_STRIDE + (((ch + (lane >> 4)) ^ x) << 4));
#pragma unroll
          for (int n = 0; n < NTILE; n += 2) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3,
                    sbase + OFF_PE + (n * 8 + bsub) * PE_STRIDE + XK(ch + ((lane >> 3) & 1)));
            mma_bf16(QACC(n), a0, a1, a2, a3, b0, b1);
            mma_bf16(QACC(n + 1), a0, a1, a2, a3, b2, b3);
          }
        }
      }
    }
#if QKSPLIT
#pragma unroll
    for (int n = 0; n < NTILE; ++n) { sacc[n][0] += saccB[n][0]; sacc[n][1] += saccB[n][1]; sacc[n][2] += saccB[n][2]; sacc[n][3] += saccB[n][3]; }
#endif
#undef QACC
    TDEP(sacc[0][0]); TDEP(sacc[NTILE - 1][3]);
    TSTAMP(3);
    __syncthreads();  // all warps done reading Q before the Q region is reused for the partial logits
    {
      float* slog = reinterpret_cast<float*>(smem + OFF_SLOG) + warp * (16 * SLOG_STRIDE);
      const int hrow = lane >> 2, kcol = 2 * (lane & 3);
#pragma unroll
      for (int n = 0; n < NTILE; ++n) {
        *reinterpret_cast<float2*>(slog + hrow * SLOG_STRIDE + n * 8 + kcol) = make_float2(sacc[n][0], sacc[n][1]);
        *reinterpret_cast<float2*>(slog + (hrow + 8) * SLOG_STRIDE + n * 8 + kcol) = make_float2(sacc[n][2], sacc[n][3]);
      }
    }
    __syncthreads();

    // ---- softmax (log2 domain): thread -> head tid / (S/4), keys 4*(tid % (S/4)) .. +3 ----
    {
      constexpr int TPH = S / 4;  // threads per head (8)
      const int h = tid / TPH, k0 = 4 * (tid % TPH);
      const float* slog = reinterpret_cast<float*>(smem + OFF_SLOG);
      float4 v = *reinterpret_cast<const float4*>(slog + h * SLOG_STRIDE + k0);
#pragma unroll
      for (int w = 1; w < WARPS; ++w) {
        const float4 u = *reinterpret_cast<const float4*>(slog + (w * 16 + h) * SLOG_STRIDE + k0);
        v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
      }
      float sv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) sv[i] = (sidx[k0 + i] >= 0) ? sv[i] * scale_log2 : -INFINITY;
      float mx = fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3]));
#pragma unroll
      for (int o = 1; o < TPH; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float p[4], l = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { p[i] = exp2f(sv[i] - mx); l += p[i]; }
#pragma unroll
      for (int o = 1; o < TPH; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
      uint2 pk = make_uint2(pack_bf16x2(p[0], p[1]), pack_bf16x2(p[2], p[3]));
      *reinterpret_cast<uint2*>(smem + OFF_P + h * P_STRIDE + k0 * 2) = pk;
      if ((tid % TPH) == 0) { sml[h] = mx; sml[16 + h] = l; }
    }
    __syncthreads();
    TSTAMP(4);

    // ---- P.V : this warp owns dims [warp*DPW, +DPW) ----
    float acc[DPW / 8][4];
#pragma unroll
    for (int j = 0; j < DPW / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    {
      const int arow = (lane & 7) + 8 * ((lane >> 3) & 1);
      const uint32_t pa = sbase + OFF_P + arow * P_STRIDE + 16 * (lane >> 4);
      const int x = lane & 7;  // (kk*16 + arow) & 7 == lane & 7
#pragma unroll
      for (int kk = 0; kk < S / 16; ++kk) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, pa + kk * 32);
        const uint32_t vb = sbase + OFF_KV + (kk * 16 + arow) * KV_STRIDE;
#pragma unroll
        for (int j = 0; j < DPW / 16; ++j) {
          uint32_t b0, b1, b2, b3;
          const int chunk = (warp * DPW) / 8 + 2 * j + (lane >> 4);
          ldsm_x4_t(b0, b1, b2, b3, vb + XK(chunk));
          mma_bf16(acc[2 * j], a0, a1, a2, a3, b0, b1);
          mma_bf16(acc[2 * j + 1], a0, a1, a2, a3, b2, b3);
        }
      }
    }
    TDEP(acc[0][0]); TDEP(acc[DPW / 8 - 1][3]);
    TSTAMP(5);
#if PUBEARLY && SIGNAL == 9
    // publish now (the consumer validates the data tags and re-reads anything that has not landed yet)
    if (tid == 0 && !stood_down) {
      unsigned* line = ctr + (t * 2 + (epoch & 1)) * CTR_REP * CTR_STRIDE;
#pragma unroll
      for (int r = 0; r < CTR_REP; ++r) {
        if (nvalid > 0) red_relaxed_gpu_or(line + r * CTR_STRIDE + 2 + (c >> 5), 1u << (c & 31));
        red_relaxed_gpu_add(line + r * CTR_STRIDE, 1u);
      }
    }
#endif
    // ---- stage the partial in smem (KV region, free once every warp finished P.V), then coalesced stores ----
    __syncthreads();
    {
      uint8_t* stg = smem + OFF_STAGE;
      const int hrow = lane >> 2;
#pragma unroll
      for (int j = 0; j < DPW / 8; ++j) {
        const int d = warp * DPW + j * 8 + 2 * (lane & 3);
#if PARTIAL_BF16 && TAGPROD
        {
          const unsigned tag8 = (epoch % 255u) + 1u, e0 = 2 * (lane & 3), ta = (tag8 >> e0) & 1u, tb = (tag8 >> (e0 + 1)) & 1u;
          *reinterpret_cast<uint32_t*>(stg + hrow * STG_STRIDE + d * 2) = pack_bf16x2_tag(acc[j][0], acc[j][1], ta, tb);
          *reinterpret_cast<uint32_t*>(stg + (hrow + 8) * STG_STRIDE + d * 2) = pack_bf16x2_tag(acc[j][2], acc[j][3], ta, tb);
        }
#elif PARTIAL_BF16
        *reinterpret_cast<uint32_t*>(stg + hrow * STG_STRIDE + d * 2) = pack_bf16x2(acc[j][0], acc[j][1]);
        *reinterpret_cast<uint32_t*>(stg + (hrow + 8) * STG_STRIDE + d * 2) = pack_bf16x2(acc[j][2], acc[j][3]);
#else
        *reinterpret_cast<float2*>(stg + hrow * STG_STRIDE + d * 4) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(stg + (hrow + 8) * STG_STRIDE + d * 4) = make_float2(acc[j][2], acc[j][3]);
#endif
      }
    }
#if OML
    __syncthreads();
    {
      // row r = 68 x 16 B chunks: segment q = u / 17 (272 B): 16 data chunks (dims 128q + 8i) then {m[r], l[r], 0, 0}
      const uint8_t* stg = smem + OFF_STAGE;
      int4* op = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(o_part) + (size_t)slot * (H * OML_ROW));
#pragma unroll
      for (int k = 0; k < 9; ++k) {
        const int i = tid + k * NT;
        if (i < H * 68) {
          const int r = i / 68, u = i % 68, qq = u / 17, w = u % 17;
          int4 v;
          if (w < 16) v = *reinterpret_cast<const int4*>(stg + r * STG_STRIDE + (qq * 16 + w) * 16);
          else v = make_int4(__float_as_int(sml[r]), __float_as_int(sml[16 + r]), 0, 0);
          op[i] = v;
        }
      }
#elif WARPCOPY
    __syncwarp();
    {
      // each warp copies out its own dims [128 w, 128 w + 128) of all 16 rows: 16 x 256 B = 8 x (32 lanes x 16 B)
      const uint8_t* stg = smem + OFF_STAGE;
      int4* op = reinterpret_cast<int4*>(o_part + (size_t)slot * H * DC);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int r = 2 * k + (lane >> 4), j = warp * 16 + (lane & 15);
        op[r * 64 + j] = *reinterpret_cast<const int4*>(stg + r * STG_STRIDE + j * 16);
      }
#else
    __syncthreads();
    {
      constexpr int CPR = DC * sizeof(part_t) / 16;  // 16B chunks per row
      const uint8_t* stg = smem + OFF_STAGE;
#if OML2
      int4* op = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(o_part) + (size_t)slot * OSLOT_B);
#else
      int4* op = reinterpret_cast<int4*>(o_part + (size_t)slot * H * DC);
#endif
#pragma unroll
      for (int k = 0; k < (H * CPR) / NT; ++k) {
        const int i = tid + k * NT, r = i / CPR, j = i % CPR;
        op[i] = *reinterpret_cast<const int4*>(stg + r * STG_STRIDE + j * 16);
      }
#endif
#if OML2
      if (tid < 32) reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(o_part) + (size_t)slot * OSLOT_B + H * DC * 2)[tid] = sml[tid];
#elif PROGML
      if (tid < 32) ml_part[(size_t)slot * 32 + tid] = sml[tid];  // one full 128 B line per chunk: m[16], l[16]
#elif TAGPROD
      if (tid < 16)  // tagged {m, l, tag, tag} word per (chunk, head): [t][H][NCHUNK] (contiguous per (t, h))
        *reinterpret_cast<uint4*>(ml_part + (((size_t)t * H + tid) * NCHUNK + c) * 4) = make_uint4(__float_as_uint(sml[tid]), __float_as_uint(sml[16 + tid]), epoch + 1u, epoch + 1u);  // tag twice: keeps .w live (no register reuse -> no scoreboard stall)
#elif !OML
      if (tid < 32) ml_part[(((size_t)t * 2 + (tid >> 4)) * H + (tid & 15)) * NCHUNK + c] = sml[tid];  // [t][2][16][NCHUNK]
#endif
    }
  }

  // ---------------- publish: mask bit (active chunks only) + arrival ----------------
#if SIGNAL == 3
  asm volatile("fence.acq_rel.gpu;" ::: "memory");  // each thread orders its own stores before the barrier
#endif
  __syncthreads();
  TSTAMP(6);
  if (tid == 0 && !stood_down && !(PUBEARLY && SIGNAL == 9 && CTRPAR && nvalid > 0)) {
#if CTRPAR
    // two counter lines per request, selected by launch parity (the other one is cleared in the prologue)
    unsigned* line = ctr + (t * 2 + (epoch & 1)) * CTR_REP * CTR_STRIDE;
#else
    unsigned* line = ctr + t * CTR_REP * CTR_STRIDE;
#endif
    // the fence orders the CTA's partial stores (via bar.sync) before the flags
#if SIGNAL == 0
    if (nvalid > 0) red_release_gpu_or(line + 2 + (c >> 5), 1u << (c & 31));
    red_release_gpu_add(line, 1u);
#elif SIGNAL == 1
    __threadfence();
    if (nvalid > 0) red_relaxed_gpu_or(line + 2 + (c >> 5), 1u << (c & 31));
    red_relaxed_gpu_add(line, 1u);
#elif SIGNAL == 2
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#pragma unroll
    for (int r = 0; r < CTR_REP; ++r) {
      if (nvalid > 0) red_relaxed_gpu_or(line + r * CTR_STRIDE + 2 + (c >> 5), 1u << (c & 31));
      red_relaxed_gpu_add(line + r * CTR_STRIDE, 1u);
    }
#else
#pragma unroll
    for (int r = 0; r < CTR_REP; ++r) {
      if (nvalid > 0) red_relaxed_gpu_or(line + r * CTR_STRIDE + 2 + (c >> 5), 1u << (c & 31));
      red_relaxed_gpu_add(line + r * CTR_STRIDE, 1u);
    }
#endif
    TSTAMP_LANE(12);  // after the release reds were issued (membar drained)
  }

  // ================= combine role =================
  if (blockIdx.x < (unsigned)(num_tokens * COMB_PER_REQ)) {
#if COMBROT
    const int t2 = (int)((blockIdx.x / COMB_PER_REQ + (unsigned)((num_tokens + 1) / 2)) % (unsigned)num_tokens), rem = blockIdx.x % COMB_PER_REQ;
#else
    const int t2 = blockIdx.x / COMB_PER_REQ, rem = blockIdx.x % COMB_PER_REQ;
#endif
    const int h = rem / (DC / NT), q = rem % (DC / NT);
    const int dim = q * NT + tid;
#if CTRPAR
    unsigned* my_ctr = ctr + ((t2 * 2 + (epoch & 1)) * CTR_REP + (rem % CTR_REP)) * CTR_STRIDE;
#else
    unsigned* my_ctr = ctr + (t2 * CTR_REP + (rem % CTR_REP)) * CTR_STRIDE;
#endif
    float* cw = reinterpret_cast<float*>(smem);                   // [NCHUNK] weights + [WARPS] warp maxima
    float* cwl = cw + NCHUNK + WARPS;                             // [WARPS] warp sums of w*l
    unsigned* cl = reinterpret_cast<unsigned*>(cwl + WARPS);      // [4] polled line: count, mask_lo, mask_hi
    const part_t* ob = o_part + ((size_t)t2 * NCHUNK * H + h) * DC + dim;
    // poll the counter line: POLL_WARPS warps poll independently with staggered phase; the first to see all
    // arrivals publishes the line in smem
    volatile unsigned* vcl = cl;
    if (tid < 4) vcl[tid] = 0u;
    __syncthreads();
#if PROGW == 4
    // ---- online progressive combine ----
    // 16-lane group cg (two per warp) owns chunks cc = cg + 8k; lane dgrp owns dims 8*dgrp..+7.  As a chunk's mask
    // bit shows up its tagged partial word (per lane) and its {m, l, tag, tag} word (lane dgrp == k) are cp.async'd
    // into smem; one poll later they are validated (tags: no producer fence needed) and merged with an online
    // softmax (running max / sum per lane).  After the last chunk only its own merge remains, then one block
    // barrier merges the 8 groups.
    const int dgrp = tid & 15, cg = tid >> 4;
    const __nv_bfloat16* ob8 = o_part + ((size_t)t2 * NCHUNK * H + h) * DC + q * NT + 8 * dgrp;
    const uint4* mlp = reinterpret_cast<const uint4*>(ml_part) + ((size_t)t2 * H + h) * NCHUNK;
    const uint32_t cstg = sbase + OFF_CSTG;
    const uint8_t* cstg_p = smem + OFF_CSTG;
    const unsigned tag8 = (epoch % 255u) + 1u, tag32 = epoch + 1u;
    float acc8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc8[i] = 0.f;
    float Mrun = -INFINITY, Lrun = 0.f;
    unsigned issued = 0u, done = 0u, redo = 0u, seen_lo = 0u, seen_hi = 0u, cnt = 0u;
#define P4_ISSUE(mask)                                                                                              \
    {                                                                                                               \
      _Pragma("unroll")                                                                                             \
      for (int k = 0; k < 8; ++k) {                                                                                 \
        const int cc = cg + 8 * k;                                                                                  \
        const unsigned pr = ((mask) >> k) & 1u;                                                                     \
        cp_async16_pred(cstg + ((k * 8 + cg) * 16 + dgrp) * 16, ob8 + (size_t)cc * H * DC, pr);                     \
        cp_async16_pred(cstg + 16384 + (k * 8 + cg) * 16, mlp + cc, pr & (dgrp == k ? 1u : 0u));                    \
      }                                                                                                             \
      asm volatile("cp.async.commit_group;" ::: "memory");                                                          \
    }
#define P4_CONSUME(mask)                                                                                            \
    {                                                                                                               \
      _Pragma("unroll")                                                                                             \
      for (int k = 0; k < 8; ++k) {                                                                                 \
        if (((mask) >> k) & 1u) {                                                                                   \
          const uint4 w = *reinterpret_cast<const uint4*>(cstg_p + ((k * 8 + cg) * 16 + dgrp) * 16);                \
          const uint4 ml = *reinterpret_cast<const uint4*>(cstg_p + 16384 + (k * 8 + cg) * 16);                     \
          if (lsbtag8(w) != tag8 || ml.z != tag32 || ml.w != tag32) { redo |= 1u << k; }                            \
          else {                                                                                                    \
            done |= 1u << k;                                                                                        \
            const float m = __uint_as_float(ml.x), l = __uint_as_float(ml.y);                                       \
            const float Mn = fmaxf(Mrun, m);                                                                        \
            const float sc = (Mrun == -INFINITY) ? 0.f : exp2f(Mrun - Mn);                                          \
            const float wt = exp2f(m - Mn);                                                                         \
            Lrun = Lrun * sc + wt * l;                                                                              \
            acc8[0] = acc8[0] * sc + wt * bf16lo(w.x & ~1u); acc8[1] = acc8[1] * sc + wt * bf16hi(w.x & ~0x10000u); \
            acc8[2] = acc8[2] * sc + wt * bf16lo(w.y & ~1u); acc8[3] = acc8[3] * sc + wt * bf16hi(w.y & ~0x10000u); \
            acc8[4] = acc8[4] * sc + wt * bf16lo(w.z & ~1u); acc8[5] = acc8[5] * sc + wt * bf16hi(w.z & ~0x10000u); \
            acc8[6] = acc8[6] * sc + wt * bf16lo(w.w & ~1u); acc8[7] = acc8[7] * sc + wt * bf16hi(w.w & ~0x10000u); \
            Mrun = Mn;                                                                                              \
          }                                                                                                         \
        }                                                                                                           \
      }                                                                                                             \
    }
    unsigned prev1 = 0u;   // issued in the previous iteration (its group may still be in flight)
#ifdef DSA_TIMING
    unsigned n_it = 0u, n_dr = 0u;
#endif
    if (warp > 0) __nanosleep(warp * POLL_STAGGER);
    for (;;) {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (lane == 0) { TSTAMP_LANE(7); x = ld_relaxed_v4(my_ctr); }
      // while the poll is in flight: merge the copies issued two iterations ago (all groups but the newest done)
      const unsigned ready = issued & ~done & ~prev1 & ~redo;
      asm volatile("cp.async.wait_group 1;" ::: "memory");
      __syncwarp();
      P4_CONSUME(ready)
      x.x = __shfl_sync(0xffffffffu, x.x, 0); x.z = __shfl_sync(0xffffffffu, x.z, 0); x.w = __shfl_sync(0xffffffffu, x.w, 0);
#ifdef DSA_TIMING
      ++n_it;
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll returned
#endif
      cnt = x.x;
      const unsigned new_lo = x.z & ~seen_lo, new_hi = x.w & ~seen_hi;
      seen_lo = x.z; seen_hi = x.w;
      unsigned iss = redo;
#pragma unroll
      for (int k = 0; k < 8; ++k) { const int cc = cg + 8 * k; iss |= (((cc < 32 ? new_lo : new_hi) >> (cc & 31)) & 1u) << k; }
      issued |= iss; redo = 0u; prev1 = iss;
      P4_ISSUE(iss)
      if (cnt >= (unsigned)NCHUNK) {
        // everything has arrived: drain what is still in flight, re-reading words that were copied too early
        for (;;) {
#ifdef DSA_TIMING
          ++n_dr;
#endif
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          P4_CONSUME(issued & ~done)
          if (__all_sync(0xffffffffu, redo == 0u)) break;
          P4_ISSUE(redo)
          redo = 0u;
          __nanosleep(SPEC_RETRY_NS);
        }
        break;
      }
      __nanosleep(POLL_NS);
    }
#undef P4_ISSUE
#undef P4_CONSUME
#ifdef DSA_TIMING
    if (tid == 0) { g_tbuf[8192 + blockIdx.x * 16 + 1] = gtimer(); g_tbuf[8192 + blockIdx.x * 16 + 2] = n_it; g_tbuf[8192 + blockIdx.x * 16 + 3] = n_dr; }   // E1: loop exit
#endif
    // merge the 8 groups: X[cg][dgrp] = {acc8[0..7], M, L, -, -}
    float* X = reinterpret_cast<float*>(smem);
    {
      float4* xp = reinterpret_cast<float4*>(X + (cg * 16 + dgrp) * 12);
      xp[0] = make_float4(acc8[0], acc8[1], acc8[2], acc8[3]);
      xp[1] = make_float4(acc8[4], acc8[5], acc8[6], acc8[7]);
      xp[2] = make_float4(Mrun, Lrun, 0.f, 0.f);
    }
    __syncthreads();
    TSTAMP(8);
    float M = -INFINITY, L = 0.f, acc = 0.f;
    {
      const int dg = tid >> 3, di = tid & 7;   // dim = q*128 + 8*dg + di
#pragma unroll
      for (int g = 0; g < 8; ++g) M = fmaxf(M, X[(g * 16 + dg) * 12 + 8]);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const float Mg = X[(g * 16 + dg) * 12 + 8];
        const float f = (Mg == -INFINITY) ? 0.f : exp2f(Mg - M);
        acc += X[(g * 16 + dg) * 12 + di] * f;
        L += X[(g * 16 + dg) * 12 + 9] * f;
      }
    }
    TSTAMP(9);
    TSTAMP(10);
    __nv_bfloat16* dst = out + ((size_t)t2 * H + h) * DC + dim;
    *dst = __float2bfloat16((L > 0.f) ? acc / L : 0.f);
#if !NOLSE
    if (dim == 0) lse[t2 * H + h] = (L > 0.f) ? M + log2f(L) : -INFINITY;
#endif
#elif PROGW == 5
    // ---- progressive loads (registers) + post-loop online merge ----
    // 16-lane group cg owns chunks cc = cg + 8k; lane dgrp owns dims 8*dgrp..+7 (vq[k]); lane dgrp == k < 8 also
    // loads chunk k's {m, l, tag, tag} word (mlq).  Loads are issued as mask bits show up; after the last arrival the
    // tags are validated (no producer fence), each lane merges its 8 chunks with an online softmax, and one block
    // barrier merges the 8 groups.
    const int dgrp = tid & 15, cg = tid >> 4;
    const __nv_bfloat16* ob8 = o_part + ((size_t)t2 * NCHUNK * H + h) * DC + q * NT + 8 * dgrp;
    const uint4* mlp = reinterpret_cast<const uint4*>(ml_part) + ((size_t)t2 * H + h) * NCHUNK;
    const unsigned tag8 = (epoch % 255u) + 1u, tag32 = epoch + 1u;
    uint4 vq[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) vq[k] = make_uint4(0u, 0u, 0u, 0u);
    uint4 mlq = make_uint4(0u, 0u, 0u, 0u);
    unsigned seen_lo = 0u, seen_hi = 0u, lo = 0u, hi = 0u;
    const int mycc = cg + 8 * (dgrp & 7);   // chunk whose m/l word this lane loads (dgrp < 8 only)
    if (warp > 0) __nanosleep(warp * POLL_STAGGER);
    for (;;) {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
#if P5ACQ
      if (lane == 0) { TSTAMP_LANE(7); x = ld_acquire_v4(my_ctr); }
#else
      if (lane == 0) { TSTAMP_LANE(7); x = ld_relaxed_v4(my_ctr); }
#endif
      // while the poll is in flight: did another warp of this block already see the final line?
      bool fin = false;
#pragma unroll 1
      for (int i = 0; i < P5_SPIN; ++i) {
        if (vcl[3] != 0u) { fin = true; break; }
        __nanosleep(P5_SPIN_NS);
      }
      if (!fin) {
        x.x = __shfl_sync(0xffffffffu, x.x, 0); x.z = __shfl_sync(0xffffffffu, x.z, 0); x.w = __shfl_sync(0xffffffffu, x.w, 0);
        if (vcl[3] != 0u) fin = true;
      }
      if (fin) { lo = vcl[1]; hi = vcl[2]; }
      else {
        lo = x.z; hi = x.w;
        if (x.x >= (unsigned)NCHUNK) { fin = true; if (lane == 0) { vcl[1] = lo; vcl[2] = hi; vcl[3] = 1u; } }
      }
#ifdef DSA_TIMING
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll returned
#endif
      const unsigned new_lo = lo & ~seen_lo, new_hi = hi & ~seen_hi;
      if (new_lo | new_hi) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int cc = cg + 8 * k;
          ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * H * DC, ((cc < 32 ? new_lo : new_hi) >> (cc & 31)) & 1u);
        }
        ld_cg_v4_pred(mlq, mlp + mycc, (dgrp < 8 ? 1u : 0u) & (((mycc < 32 ? new_lo : new_hi) >> (mycc & 31)) & 1u));
        seen_lo = lo; seen_hi = hi;
      }
      if (fin) break;
      __nanosleep(POLL_NS);
    }
#ifdef DSA_TIMING
    if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 1] = gtimer();   // E1: loop exit
    if (lane == 0) g_tbuf[8192 + blockIdx.x * 16 + 4 + warp] = gtimer();
#endif
    unsigned have = 0u;
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int cc = cg + 8 * k; have |= (((cc < 32 ? lo : hi) >> (cc & 31)) & 1u) << k; }
    const bool mlhave = (dgrp < 8) && ((((mycc < 32 ? lo : hi) >> (mycc & 31)) & 1u) != 0u);
    {
      // no producer fence: validate the per-launch tags, re-read words that were copied before they landed
#ifdef DSA_TIMING
      unsigned nval = 0u;
#endif
      for (;;) {
#ifdef DSA_TIMING
        ++nval;
#endif
        unsigned bad = 0u;
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (((have >> k) & 1u) && lsbtag8(vq[k]) != tag8) bad |= 1u << k;
        const bool mlbad = mlhave && (mlq.z != tag32 || mlq.w != tag32);
        if (__all_sync(0xffffffffu, bad == 0u && !mlbad)) break;
#pragma unroll
        for (int k = 0; k < 8; ++k) ld_cg_v4_pred(vq[k], ob8 + (size_t)(cg + 8 * k) * H * DC, (bad >> k) & 1u);
        if (mlbad) ld_cg_v4_pred(mlq, mlp + mycc, 1u);
        __nanosleep(SPEC_RETRY_NS);
      }
#ifdef DSA_TIMING
      if (tid == 0) { g_tbuf[8192 + blockIdx.x * 16 + 2] = gtimer(); g_tbuf[8192 + blockIdx.x * 16 + 3] = nval; }
      if (lane == 0) g_tbuf[8192 + blockIdx.x * 16 + 8 + warp] = gtimer();
#endif
    }
    // online merge of this lane's 8 chunks (m, l of chunk k come from lane k of the group)
    float acc8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc8[i] = 0.f;
    float mk[8], lk[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      mk[k] = __shfl_sync(0xffffffffu, __uint_as_float(mlq.x), (lane & 16) + k);
      lk[k] = __shfl_sync(0xffffffffu, __uint_as_float(mlq.y), (lane & 16) + k);
    }
    float Mrun = -INFINITY, Lrun = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) Mrun = ((have >> k) & 1u) ? fmaxf(Mrun, mk[k]) : Mrun;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if ((have >> k) & 1u) {
        const float wt = exp2f(mk[k] - Mrun);
        Lrun += wt * lk[k];
        const uint4 w = vq[k];
        acc8[0] += wt * bf16lo(w.x & ~1u); acc8[1] += wt * bf16hi(w.x & ~0x10000u);
        acc8[2] += wt * bf16lo(w.y & ~1u); acc8[3] += wt * bf16hi(w.y & ~0x10000u);
        acc8[4] += wt * bf16lo(w.z & ~1u); acc8[5] += wt * bf16hi(w.z & ~0x10000u);
        acc8[6] += wt * bf16lo(w.w & ~1u); acc8[7] += wt * bf16hi(w.w & ~0x10000u);
      }
    }
    // merge the 8 groups: X[cg][dgrp] = {acc8[0..7], M, L, -, -}
    float* X = reinterpret_cast<float*>(smem);
    {
      float4* xp = reinterpret_cast<float4*>(X + (cg * 16 + dgrp) * 12);
      xp[0] = make_float4(acc8[0], acc8[1], acc8[2], acc8[3]);
      xp[1] = make_float4(acc8[4], acc8[5], acc8[6], acc8[7]);
      xp[2] = make_float4(Mrun, Lrun, 0.f, 0.f);
    }
#ifdef DSA_TIMING
    if (lane == 0) g_tbuf[8192 + blockIdx.x * 16 + 12 + warp] = gtimer();
#endif
    __syncthreads();
    TSTAMP(8);
    float M = -INFINITY, L = 0.f, acc = 0.f;
    {
      const int dg = tid >> 3, di = tid & 7;   // dim = q*128 + 8*dg + di
#pragma unroll
      for (int g = 0; g < 8; ++g) M = fmaxf(M, X[(g * 16 + dg) * 12 + 8]);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const float Mg = X[(g * 16 + dg) * 12 + 8];
        const float f = (Mg == -INFINITY) ? 0.f : exp2f(Mg - M);
        acc += X[(g * 16 + dg) * 12 + di] * f;
        L += X[(g * 16 + dg) * 12 + 9] * f;
      }
    }
    TSTAMP(9);
    TSTAMP(10);
    __nv_bfloat16* dst = out + ((size_t)t2 * H + h) * DC + dim;
    *dst = __float2bfloat16((L > 0.f) ? acc / L : 0.f);
#if !NOLSE
    if (dim == 0) lse[t2 * H + h] = (L > 0.f) ? M + log2f(L) : -INFINITY;
#endif
#else  // PROGW != 4
#if PROGW
    // per-warp progressive combine: lane 0 of every warp polls the arrival line (acquire) and broadcasts it; the
    // warp's lanes own chunks cg + 8k (cg = tid/16 -> two groups per warp) and load each partial segment (16 B per
    // lane) as soon as its mask bit shows up; threads < 64 likewise fetch (m, l) of chunk tid.  No block barrier
    // until every chunk has arrived, so after the last arrival only that chunk's loads remain in flight.
    const int dgrp = tid & 15, cg = tid >> 4;
    const __nv_bfloat16* ob8 = o_part + ((size_t)t2 * NCHUNK * H + h) * DC + q * NT + 8 * dgrp;
    const float* mb = ml_part + (((size_t)t2 * 2) * H + h) * NCHUNK;
    uint4 vq[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) vq[k] = make_uint4(0u, 0u, 0u, 0u);
    float mc = -INFINITY, lc = 0.f;
    unsigned seen_lo = 0u, seen_hi = 0u, lo = 0u, hi = 0u;
#if TAGVAL
    uint4 mlw = make_uint4(0u, 0u, 0u, 0u);
    const uint4* mlp = reinterpret_cast<const uint4*>(ml_part) + ((size_t)t2 * H + h) * NCHUNK + (tid & 63);
#endif
#if PROGW == 3
    // activity of this request's chunks straight from the indices (16 ints per thread -> chunk tid/2), so the
    // warp knows which of its chunks it must wait for and can speculatively (re)load the last missing ones with
    // tag validation instead of waiting for their mask bits
    unsigned amask_lo, amask_hi;
    {
      if (lane == 0) cl[4 + warp] = cm16;
      __syncthreads();
      amask_lo = cl[4] | (cl[5] << 16); amask_hi = cl[6] | (cl[7] << 16);
    }
    unsigned need = 0u, pend = 0u;
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int cc = cg + 8 * k; need |= (((cc < 32 ? amask_lo : amask_hi) >> (cc & 31)) & 1u) << k; }
    bool ml_need = (tid < NCHUNK) && ((((tid < 32 ? amask_lo : amask_hi) >> (tid & 31)) & 1u) != 0u), ml_pend = false;
    uint4 mlw = make_uint4(0u, 0u, 0u, 0u);
    const uint4* mlp = reinterpret_cast<const uint4*>(ml_part) + ((size_t)t2 * H + h) * NCHUNK + (tid & 63);
    const unsigned tag8 = (epoch % 255u) + 1u, tag32 = epoch + 1u;
    const uint32_t cstg = sbase + OFF_CSTG;   // smem staging: [8 k][8 cg][16 dgrp] x 16 B, then [64] m/l words
    bool spec_seen = false;
#ifdef DSA_TIMING
    unsigned niter = 0u, nspec = 0u;
#endif
    if (warp > 0) __nanosleep(warp * POLL_STAGGER);
    // phase 1: poll until every needed load has been ISSUED (its mask bit is up, or the request has at most
    // SPEC_CNT chunks outstanding and the load is speculative)
    for (;;) {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (lane == 0) { TSTAMP_LANE(7); x = ld_acquire_v4(my_ctr); }
      x.x = __shfl_sync(0xffffffffu, x.x, 0); x.z = __shfl_sync(0xffffffffu, x.z, 0); x.w = __shfl_sync(0xffffffffu, x.w, 0);
      lo = x.z; hi = x.w;
#ifdef DSA_TIMING
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll returned (last writer wins)
#endif
#if SPEC_PROG
      // speculate once all but SPEC_CNT of the request's ACTIVE chunks have arrived (the rest are imminent)
      const unsigned spec = ((lo | hi) != 0u && __popc(amask_lo & ~lo) + __popc(amask_hi & ~hi) <= SPEC_CNT) ? 1u : 0u;
#else
      const unsigned spec = (x.x + SPEC_CNT >= (unsigned)NCHUNK) ? 1u : 0u;
#endif
#if GUARD1
      const bool changed = ((lo & ~seen_lo) | (hi & ~seen_hi)) != 0u || (spec && !spec_seen);
      seen_lo = lo; seen_hi = hi; spec_seen |= spec != 0u;
      if (changed) {
#endif
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int cc = cg + 8 * k;
        const unsigned pr = ((need >> k) & 1u) & ((((cc < 32 ? lo : hi) >> (cc & 31)) & 1u) | spec);
#if SPEC_CP
        cp_async16_pred(cstg + ((k * 8 + cg) * 16 + dgrp) * 16, ob8 + (size_t)cc * H * DC, pr);
#else
        ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * H * DC, pr);
#endif
        pend |= pr << k; need &= ~(pr << k);
      }
      if (ml_need) {
        const unsigned pr = (((tid < 32 ? lo : hi) >> (tid & 31)) & 1u) | spec;
#if SPEC_CP
        cp_async16_pred(cstg + 16384 + tid * 16, mlp, pr);
#else
        ld_cg_v4_pred(mlw, mlp, pr);
#endif
        if (pr) { ml_pend = true; ml_need = false; }
      }
#if SPEC_CP
      asm volatile("cp.async.commit_group;" ::: "memory");
#endif
#if GUARD1
      }
#endif
#ifdef DSA_TIMING
      ++niter; nspec += spec;
#endif
      if (__all_sync(0xffffffffu, need == 0u && !ml_need)) break;
      __nanosleep(POLL_NS);
    }
#ifdef DSA_TIMING
    if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 1] = gtimer();   // E1: phase-1 exit
#endif
#ifdef DSA_TIMING
    if (tid == 0) g_tbuf[blockIdx.x * 16 + 12] = (unsigned long long)niter | ((unsigned long long)nspec << 16) | ((unsigned long long)pend << 32) | ((unsigned long long)(ml_pend ? 1u : 0u) << 48);
#endif
    // phase 2: validate the tags (waits for the copies), re-issue whatever was read before it landed
    for (;;) {
#if SPEC_CP
      cp_async_wait_all();
#endif
      unsigned bad = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if ((pend >> k) & 1u) {
#if SPEC_CP
          vq[k] = *reinterpret_cast<const uint4*>(smem + OFF_CSTG + ((k * 8 + cg) * 16 + dgrp) * 16);
#endif
          if (lsbtag8(vq[k]) != tag8) bad |= 1u << k;
        }
#if SPEC_CP
      if (ml_pend) mlw = *reinterpret_cast<const uint4*>(smem + OFF_CSTG + 16384 + tid * 16);
#endif
      const bool mlbad = ml_pend && (mlw.z != tag32 || mlw.w != tag32);
      if (__all_sync(0xffffffffu, bad == 0u && !mlbad)) break;
#if SPEC_CP
#pragma unroll
      for (int k = 0; k < 8; ++k) cp_async16_pred(cstg + ((k * 8 + cg) * 16 + dgrp) * 16, ob8 + (size_t)(cg + 8 * k) * H * DC, (bad >> k) & 1u);
      if (mlbad) cp_async16_pred(cstg + 16384 + tid * 16, mlp, 1u);
      asm volatile("cp.async.commit_group;" ::: "memory");
#else
#pragma unroll
      for (int k = 0; k < 8; ++k) ld_cg_v4_pred(vq[k], ob8 + (size_t)(cg + 8 * k) * H * DC, (bad >> k) & 1u);
      if (mlbad) ld_cg_v4_pred(mlw, mlp, 1u);
#endif
      pend = bad; ml_pend = mlbad;
      __nanosleep(SPEC_RETRY_NS);
    }
#ifdef DSA_TIMING
    if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 2] = gtimer();   // E2: phase-2 exit
#endif
    if (tid < NCHUNK && ((((tid < 32 ? amask_lo : amask_hi) >> (tid & 31)) & 1u) != 0u)) { mc = __uint_as_float(mlw.x); lc = __uint_as_float(mlw.y); }
    __syncthreads();
    TSTAMP(8);
#endif
#if PROGW != 3
#if PROGW == 7
    // no counter line: the combine knows its request's active chunks from the indices (cm16, loaded in the
    // prologue); threads < 64 poll the tagged {m, l} word of their chunk, flag it in smem once its tag matches, and
    // the lanes load the partial words of flagged chunks.  Data tags are validated after the loop (TAGVAL).
    static_assert(TAGVAL, "PROGW 7 needs TAGVAL");
    unsigned* mlv = reinterpret_cast<unsigned*>(smem) + 96;   // [64] per-chunk ready flags
    unsigned amask_lo, amask_hi;
    {
      if (lane == 0) cl[4 + warp] = cm16;
      if (tid < NCHUNK) mlv[tid] = 0u;
      __syncthreads();
      amask_lo = cl[4] | (cl[5] << 16); amask_hi = cl[6] | (cl[7] << 16);
    }
    lo = amask_lo; hi = amask_hi;
    unsigned need = 0u, loaded = 0u;
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int cc = cg + 8 * k; need |= (((cc < 32 ? amask_lo : amask_hi) >> (cc & 31)) & 1u) << k; }
    bool mlwait = (tid < NCHUNK) && ((((tid < 32 ? amask_lo : amask_hi) >> (tid & 31)) & 1u) != 0u);
    const unsigned tag32p = epoch + 1u;
    for (;;) {
      if (mlwait) {
        if (lane == 0) { TSTAMP_LANE(7); }
        ld_cg_v4_pred(mlw, mlp, 1u);
        if (mlw.z == tag32p && mlw.w == tag32p) { mlwait = false; mc = __uint_as_float(mlw.x); lc = __uint_as_float(mlw.y); mlv[tid] = 1u; }
      }
      const int pending = __syncthreads_count(mlwait);
#ifdef DSA_TIMING
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll round done (last writer wins)
#endif
      unsigned ready = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) ready |= (mlv[cg + 8 * k] & 1u) << k;
      const unsigned newk = ready & need & ~loaded;
      if (__any_sync(0xffffffffu, newk != 0u)) {
#pragma unroll
        for (int k = 0; k < 8; ++k) ld_cg_v4_pred(vq[k], ob8 + (size_t)(cg + 8 * k) * H * DC, (newk >> k) & 1u);
        loaded |= newk;
      }
      if (pending == 0) break;
      __nanosleep(POLL_NS);
    }
#else
#if PROGW == 6
    // single poller: warp 0 keeps two acquire polls in flight (lanes 0/1, half a period apart); every changed line is
    // published to smem and the other warps are released through named barrier 1 (warp 0 arrives, warps 1-3 sync);
    // they acknowledge on barrier 2 before warp 0 may publish again.  Warps 1-3 never touch L2 for polling.
    static_assert(TAGVAL, "PROGW 6 needs TAGVAL");
    if (warp == 0) {
      uint4 xa = make_uint4(0u, 0u, 0u, 0u), xb = xa;
      bool pub = false;
      if (lane == 0) { TSTAMP_LANE(7); }
      ld_acquire_v4_pred(xa, my_ctr, lane == 0 ? 1u : 0u);
      __nanosleep(POLL6_HALF);
      ld_acquire_v4_pred(xb, my_ctr, lane == 1 ? 1u : 0u);
      for (int par = 0;; par ^= 1) {
        uint4 x;
        if (par == 0) { x.x = __shfl_sync(0xffffffffu, xa.x, 0); x.z = __shfl_sync(0xffffffffu, xa.z, 0); x.w = __shfl_sync(0xffffffffu, xa.w, 0); }
        else          { x.x = __shfl_sync(0xffffffffu, xb.x, 1); x.z = __shfl_sync(0xffffffffu, xb.z, 1); x.w = __shfl_sync(0xffffffffu, xb.w, 1); }
#ifdef DSA_TIMING
        if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll returned (last writer wins)
#endif
        lo = x.z; hi = x.w;
        const unsigned new_lo = lo & ~seen_lo, new_hi = hi & ~seen_hi;
        const bool fin = x.x >= (unsigned)NCHUNK;
        if ((new_lo | new_hi) != 0u || fin) {
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int cc = cg + 8 * k;
            ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * H * DC, ((cc < 32 ? new_lo : new_hi) >> (cc & 31)) & 1u);
          }
          ld_cg_v4_pred(mlw, mlp, (new_lo >> tid) & 1u);   // warp 0: tid < 32 -> chunk tid
          seen_lo = lo; seen_hi = hi;
          if (pub) asm volatile("barrier.cta.sync 2, 128;" ::: "memory");   // ack of the previous publish
          if (lane == 0) { vcl[0] = x.x; vcl[1] = lo; vcl[2] = hi; }
          __threadfence_block();
          asm volatile("barrier.cta.arrive 1, 128;" ::: "memory");
          pub = true;
        }
        if (fin) break;
        if (par == 0) { if (lane == 0) { TSTAMP_LANE(7); } ld_acquire_v4_pred(xa, my_ctr, lane == 0 ? 1u : 0u); }
        else ld_acquire_v4_pred(xb, my_ctr, lane == 1 ? 1u : 0u);
        __nanosleep(POLL6_HALF);
      }
    } else {
      for (;;) {
        asm volatile("barrier.cta.sync 1, 128;" ::: "memory");
        const unsigned cnt = vcl[0];
        lo = vcl[1]; hi = vcl[2];
        const unsigned new_lo = lo & ~seen_lo, new_hi = hi & ~seen_hi;
        if (new_lo | new_hi) {
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int cc = cg + 8 * k;
            ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * H * DC, ((cc < 32 ? new_lo : new_hi) >> (cc & 31)) & 1u);
          }
          if (tid < NCHUNK) ld_cg_v4_pred(mlw, mlp, (new_hi >> (tid & 31)) & 1u);   // warp 1: tid 32..63 -> chunk tid
          seen_lo = lo; seen_hi = hi;
        }
        asm volatile("barrier.cta.arrive 2, 128;" ::: "memory");
        if (cnt >= (unsigned)NCHUNK) break;
      }
    }
#else
#if PROGW == 2
    // pipelined polling: lanes 0..PIPE_POLL-1 each keep one relaxed poll in flight, issued POLL_NS apart, so the
    // line is sampled every POLL_NS while each sample still waits a full round trip
    uint4 xs[PIPE_POLL];
#pragma unroll
    for (int k = 0; k < PIPE_POLL; ++k) xs[k] = make_uint4(0u, 0u, 0u, 0u);
#ifdef DSA_TIMING
    unsigned long long tiss = 0ull;
#endif
#pragma unroll
    for (int k = 0; k < PIPE_POLL; ++k) {
#ifdef DSA_TIMING
      if (lane == k) tiss = gtimer();
#endif
#if POLLACQ
      ld_acquire_v4_pred(xs[k], my_ctr, lane == k);
#else
      ld_relaxed_v4_pred(xs[k], my_ctr, lane == k);
#endif
      __nanosleep(POLL_NS);
    }
    bool pdone = false;
    while (!pdone) {
#pragma unroll
      for (int k = 0; k < PIPE_POLL; ++k) {
      uint4 x;
      x.x = __shfl_sync(0xffffffffu, xs[k].x, k); x.z = __shfl_sync(0xffffffffu, xs[k].z, k); x.w = __shfl_sync(0xffffffffu, xs[k].w, k);
      lo = x.z; hi = x.w;
#else
    if (warp > 0) __nanosleep(warp * POLL_STAGGER);
    for (;;) {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (lane == 0) { TSTAMP_LANE(7); x = ld_acquire_v4(my_ctr); }
      x.x = __shfl_sync(0xffffffffu, x.x, 0); x.z = __shfl_sync(0xffffffffu, x.z, 0); x.w = __shfl_sync(0xffffffffu, x.w, 0);
      lo = x.z; hi = x.w;
#endif
#ifdef DSA_TIMING
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 0] = gtimer();   // E0: poll returned (last writer wins)
#endif
#if BCAST
      // another warp of this block may already have seen the final line: take it from smem
      bool fin = x.x >= (unsigned)NCHUNK;
      if (!fin && vcl[3] != 0u) { lo = vcl[1]; hi = vcl[2]; fin = true; }
      else if (fin && lane == 0) { vcl[1] = lo; vcl[2] = hi; __threadfence_block(); vcl[3] = 1u; }
      if (fin) x.x = (unsigned)NCHUNK;
#endif
      const unsigned new_lo = lo & ~seen_lo, new_hi = hi & ~seen_hi;
#ifdef DSA_TIMING
      if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 4] = (new_lo | new_hi) ? 1ull : 0ull;   // E4: this iteration issued loads
#endif
      if (new_lo | new_hi) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int cc = cg + 8 * k;
          ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * H * DC, ((cc < 32 ? new_lo : new_hi) >> (cc & 31)) & 1u);
        }
        if (tid < NCHUNK) {
          const unsigned pr = ((tid < 32 ? new_lo : new_hi) >> (tid & 31)) & 1u;
#if TAGVAL
          ld_cg_v4_pred(mlw, mlp, pr);
#else
          ld_cg_f32_pred(mc, mb + tid, pr);
          ld_cg_f32_pred(lc, mb + H * NCHUNK + tid, pr);
#endif
        }
        seen_lo = lo; seen_hi = hi;
      }
#if PROGW == 2
      if (x.x >= (unsigned)NCHUNK) {
#ifdef DSA_TIMING
        if (lane == k) g_tbuf[blockIdx.x * 16 + 7] = tiss;
#endif
        pdone = true; break;
      }
#ifdef DSA_TIMING
      if (lane == k) tiss = gtimer();
#endif
#if POLLACQ
      ld_acquire_v4_pred(xs[k], my_ctr, lane == k);
#else
      ld_relaxed_v4_pred(xs[k], my_ctr, lane == k);
#endif
      __nanosleep(POLL_NS);
      }
    }
#if !POLLACQ
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
#else
      if (x.x >= (unsigned)NCHUNK) break;
      __nanosleep(POLL_NS);
    }
#endif
#endif  // PROGW == 6
#endif  // PROGW == 7
#ifdef DSA_TIMING
    if (tid == 0) g_tbuf[8192 + blockIdx.x * 16 + 1] = gtimer();   // E1: loop exit
#endif
#if TAGVAL
    {
      // no producer fence: the flags may overtake the data, so validate the per-launch tags and re-read stale words
      const unsigned tag8 = (epoch % 255u) + 1u, tag32 = epoch + 1u;
      unsigned have = 0u;
#pragma unroll
      for (int k = 0; k < 8; ++k) { const int cc = cg + 8 * k; have |= (((cc < 32 ? lo : hi) >> (cc & 31)) & 1u) << k; }
      const bool mlhave = (tid < NCHUNK) && ((((tid < 32 ? lo : hi) >> (tid & 31)) & 1u) != 0u);
#ifdef DSA_TIMING
      unsigned nval = 0u;
#endif
      for (;;) {
#ifdef DSA_TIMING
        ++nval;
#endif
        unsigned bad = 0u;
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if (((have >> k) & 1u) && lsbtag8(vq[k]) != tag8) bad |= 1u << k;
        const bool mlbad = mlhave && (mlw.z != tag32 || mlw.w != tag32);
        if (__all_sync(0xffffffffu, bad == 0u && !mlbad)) break;
#pragma unroll
        for (int k = 0; k < 8; ++k) ld_cg_v4_pred(vq[k], ob8 + (size_t)(cg + 8 * k) * H * DC, (bad >> k) & 1u);
        if (mlbad) ld_cg_v4_pred(mlw, mlp, 1u);
        __nanosleep(SPEC_RETRY_NS);
      }
      if (mlhave) { mc = __uint_as_float(mlw.x); lc = __uint_as_float(mlw.y); }
#ifdef DSA_TIMING
      if (tid == 0) { g_tbuf[8192 + blockIdx.x * 16 + 2] = gtimer(); g_tbuf[8192 + blockIdx.x * 16 + 3] = nval; }
#endif
    }
#endif
#if WPH2
    if (tid < NCHUNK) { float* cmlw = reinterpret_cast<float*>(smem) + 1536; cmlw[tid] = mc; cmlw[64 + tid] = lc; }
#endif
    __syncthreads();
    TSTAMP(8);
#endif  // PROGW != 3
#else
#if PROGML
    float* cml = reinterpret_cast<float*>(smem) + 1536;  // [2][64]: m, l per chunk (-inf / 0 for inactive)
    if (warp < 2) {
      // every lane polls (one coalesced acquire load per warp); lane owns chunk cc = 32 warp + lane and fetches its
      // (m, l) as soon as its mask bit shows, writing them to smem one poll later (the load has landed by then)
      const int cc = warp * 32 + lane;
      const float* mlp = ml_part + ((size_t)t2 * NCHUNK + cc) * 32 + h;
      bool loaded = false, pending = false;
      float pm = -INFINITY, pl = 0.f;
      if (warp == 1) __nanosleep(POLL_STAGGER);
      for (;;) {
        if (lane == 0) TSTAMP_LANE(7);
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (lane == 0) x = ld_acquire_v4(my_ctr);
        x.x = __shfl_sync(0xffffffffu, x.x, 0); x.z = __shfl_sync(0xffffffffu, x.z, 0); x.w = __shfl_sync(0xffffffffu, x.w, 0);
        if (pending) { cml[cc] = pm; cml[64 + cc] = pl; pending = false; loaded = true; }
        const unsigned mybit = ((cc < 32 ? x.z : x.w) >> (cc & 31)) & 1u;
        if (x.x >= (unsigned)NCHUNK) {
          if (lane == 0) { vcl[0] = x.x; vcl[1] = x.z; vcl[2] = x.w; vcl[3] = 1u; }
          if (!loaded) {
            if (mybit) { cml[cc] = __ldcg(mlp); cml[64 + cc] = __ldcg(mlp + 16); }
            else { cml[cc] = -INFINITY; cml[64 + cc] = 0.f; }
          }
          break;
        }
        if (vcl[3]) {  // the other warp finished first: fetch anything still missing
          if (!loaded) {
            if (mybit || (((cc < 32 ? vcl[1] : vcl[2]) >> (cc & 31)) & 1u)) { cml[cc] = __ldcg(mlp); cml[64 + cc] = __ldcg(mlp + 16); }
            else { cml[cc] = -INFINITY; cml[64 + cc] = 0.f; }
          }
          break;
        }
        if (mybit && !loaded && !pending) { pm = __ldcg(mlp); pl = __ldcg(mlp + 16); pending = true; }
        __nanosleep(POLL_NS);
      }
    } else if (lane == 0) {
      __nanosleep(warp * POLL_STAGGER);
      for (;;) {
        const uint4 x = ld_acquire_v4(my_ctr);
        if (x.x >= (unsigned)NCHUNK) { vcl[0] = x.x; vcl[1] = x.z; vcl[2] = x.w; vcl[3] = 1u; break; }
        if (vcl[3]) break;
        __nanosleep(POLL_NS);
      }
    }
    __syncthreads();
    TSTAMP(8);
#else
#ifndef POLL8
#define POLL8 0
#endif
    if (tid < 32 * POLL_WARPS && (lane == 0 || (POLL8 && lane == 16))) {
      const int pidx = POLL8 ? (warp * 2 + (lane >> 4)) : warp;
      if (pidx > 0) __nanosleep(pidx * (POLL8 ? POLL_STAGGER / 2 : POLL_STAGGER));
      for (;;) {
        TSTAMP_LANE(7);   // issue time of this poll (last writer wins)
#if POLLREL
        const uint4 x = ld_relaxed_v4(my_ctr);
        if (x.x >= (unsigned)NCHUNK) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); vcl[0] = x.x; vcl[1] = x.z; vcl[2] = x.w; vcl[3] = 1u; break; }
#else
        const uint4 x = ld_acquire_v4(my_ctr);
        if (x.x >= (unsigned)NCHUNK) { vcl[0] = x.x; vcl[1] = x.z; vcl[2] = x.w; vcl[3] = 1u; break; }
#endif
        if (vcl[3]) break;
        __nanosleep(POLL_NS);
      }
    }
    __syncthreads();
    TSTAMP(8);
#endif
    const unsigned lo = cl[1], hi = cl[2];
    // one round trip: o of the active chunks as 16 B (8 dims) loads: thread -> dims q*128 + 8*dgrp, chunks cg + 8k
    const int dgrp = tid & 15, cg = tid >> 4;
#if OML
    const uint8_t* ob8 = reinterpret_cast<const uint8_t*>(o_part) + ((size_t)t2 * NCHUNK * H + h) * OML_ROW + q * 272 + 16 * dgrp;
    constexpr size_t CSTRIDE = (size_t)H * OML_ROW;  // bytes between chunk slots
#elif OML2
    const uint8_t* ob8 = reinterpret_cast<const uint8_t*>(o_part) + (size_t)t2 * NCHUNK * OSLOT_B + h * (DC * 2) + q * 256 + 16 * dgrp;
    constexpr size_t CSTRIDE = (size_t)OSLOT_B;
#else
    const __nv_bfloat16* ob8 = o_part + ((size_t)t2 * NCHUNK * H + h) * DC + q * NT + 8 * dgrp;
    constexpr size_t CSTRIDE = (size_t)H * DC;
#endif
    uint4 vq[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) vq[k] = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int cc = cg + 8 * k;
      ld_cg_v4_pred(vq[k], ob8 + (size_t)cc * CSTRIDE, ((cc < 32 ? lo : hi) >> (cc & 31)) & 1u);
    }
#if OML2
    // lanes 2k / 2k+1 fetch 16 B of the m / l line of chunk cg + 8k (its 4-head group containing h)
    uint4 mlq = make_uint4(0u, 0u, 0u, 0u);
    {
      const int k = dgrp >> 1, cc = cg + 8 * k;
      const uint8_t* mlline = reinterpret_cast<const uint8_t*>(o_part) + ((size_t)t2 * NCHUNK + cc) * OSLOT_B + H * DC * 2 + (dgrp & 1) * 64 + 16 * (h >> 2);
      ld_cg_v4_pred(mlq, mlline, ((cc < 32 ? lo : hi) >> (cc & 31)) & 1u);
    }
#endif
#if OML
    // lanes dgrp < 8 fetch the {m, l} chunk of chunk cg + 8*dgrp (same lines as the data) -> smem -> per-chunk threads
    uint4 mlq = make_uint4(0u, 0u, 0u, 0u);
    if (dgrp < 8) {
      const int cc = cg + 8 * dgrp;
      ld_cg_v4_pred(mlq, ob8 - 16 * dgrp + 256 + (size_t)cc * CSTRIDE, ((cc < 32 ? lo : hi) >> (cc & 31)) & 1u);
    }
#endif
    float mc = -INFINITY, lc = 0.f;
#if PROGML
    if (tid < NCHUNK) { mc = cml[tid]; lc = cml[64 + tid]; }
#elif OML2
    {
      float* cml = reinterpret_cast<float*>(smem) + 1536;  // [2][64]
      {
        const int k = dgrp >> 1, cc = cg + 8 * k;
        const bool act = ((cc < 32 ? lo : hi) >> (cc & 31)) & 1u;
        const float val = (h & 3) == 0 ? __uint_as_float(mlq.x) : (h & 3) == 1 ? __uint_as_float(mlq.y) : (h & 3) == 2 ? __uint_as_float(mlq.z) : __uint_as_float(mlq.w);
        if ((dgrp & 1) == 0) cml[cc] = act ? val : -INFINITY;
        else cml[64 + cc] = act ? val : 0.f;
      }
      __syncthreads();
      if (tid < NCHUNK) { mc = cml[tid]; lc = cml[64 + tid]; }
    }
#elif OML
    {
      float* cml = reinterpret_cast<float*>(smem) + 1536;  // [2][64]
      if (dgrp < 8) {
        const int cc = cg + 8 * dgrp;
        const bool act = ((cc < 32 ? lo : hi) >> (cc & 31)) & 1u;
        cml[cc] = act ? __uint_as_float(mlq.x) : -INFINITY;
        cml[64 + cc] = act ? __uint_as_float(mlq.y) : 0.f;
      }
      __syncthreads();
      if (tid < NCHUNK) { mc = cml[tid]; lc = cml[64 + tid]; }
    }
#elif MLVEC
    {
      // m[h][0..63] and l[h][0..63] are 256 B each: 32 threads x 16 B vector loads -> smem -> one chunk per thread
      float* mls = reinterpret_cast<float*>(smem) + 1536;  // [2][64]
      if (tid < 32) {
        const float* mb = ml_part + (((size_t)t2 * 2 + (tid >> 4)) * H + h) * NCHUNK + (tid & 15) * 4;
        float4 v4 = make_float4(0.f, 0.f, 0.f, 0.f);
        ld_cg_v4_pred_f(v4, mb, 1u);
        *reinterpret_cast<float4*>(mls + (tid >> 4) * 64 + (tid & 15) * 4) = v4;
      }
      __syncthreads();
      if (tid < NCHUNK) {
        const bool act = ((tid < 32 ? lo : hi) >> (tid & 31)) & 1u;
        mc = act ? mls[tid] : -INFINITY;
        lc = act ? mls[64 + tid] : 0.f;
      }
    }
#else
    if (tid < NCHUNK) {
      const unsigned pr = ((tid < 32 ? lo : hi) >> (tid & 31)) & 1u;
      const float* mb = ml_part + (((size_t)t2 * 2) * H + h) * NCHUNK;
      ld_cg_f32_pred(mc, mb + tid, pr);
      ld_cg_f32_pred(lc, mb + H * NCHUNK + tid, pr);
    }
#endif
#endif
#if WPH2
    // two-barrier tail: (m, l) of every chunk -> smem; each lane weights its own 8 chunks (group max) ; one more
    // barrier to merge the 8 groups
    const float* cml = reinterpret_cast<const float*>(smem) + 1536;   // [2][64], written before the barrier above
    {
      float mk[8], lk[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) { mk[k] = cml[cg + 8 * k]; lk[k] = cml[64 + cg + 8 * k]; }
      float Mg = -INFINITY;
#pragma unroll
      for (int k = 0; k < 8; ++k) Mg = fmaxf(Mg, mk[k]);   // inactive chunks hold -inf
      float Lg = 0.f, acc8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc8[i] = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float wt = (mk[k] == -INFINITY) ? 0.f : exp2f(mk[k] - Mg);
        Lg += wt * lk[k];
        const uint4 w = vq[k];
        acc8[0] += wt * bf16lo(w.x & ~1u); acc8[1] += wt * bf16hi(w.x & ~0x10000u);
        acc8[2] += wt * bf16lo(w.y & ~1u); acc8[3] += wt * bf16hi(w.y & ~0x10000u);
        acc8[4] += wt * bf16lo(w.z & ~1u); acc8[5] += wt * bf16hi(w.z & ~0x10000u);
        acc8[6] += wt * bf16lo(w.w & ~1u); acc8[7] += wt * bf16hi(w.w & ~0x10000u);
      }
      float* X = reinterpret_cast<float*>(smem) + 1792;   // [8 cg][16 dgrp][12] (after cml at 1536..1664)
      float4* xp = reinterpret_cast<float4*>(X + (cg * 16 + dgrp) * 12);
      xp[0] = make_float4(acc8[0], acc8[1], acc8[2], acc8[3]);
      xp[1] = make_float4(acc8[4], acc8[5], acc8[6], acc8[7]);
      xp[2] = make_float4(Mg, Lg, 0.f, 0.f);
    }
    __syncthreads();
    TSTAMP(9);
    float M = -INFINITY, L = 0.f, acc = 0.f;
    {
      const float* X = reinterpret_cast<const float*>(smem) + 1792;
      const int dg = tid >> 3, di = tid & 7;   // dim = q*128 + 8*dg + di
#pragma unroll
      for (int g = 0; g < 8; ++g) M = fmaxf(M, X[(g * 16 + dg) * 12 + 8]);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const float Mg = X[(g * 16 + dg) * 12 + 8];
        const float f = (Mg == -INFINITY) ? 0.f : exp2f(Mg - M);
        acc += X[(g * 16 + dg) * 12 + di] * f;
        L += X[(g * 16 + dg) * 12 + 9] * f;
      }
    }
    TSTAMP(10);
    __nv_bfloat16* dst = out + ((size_t)t2 * H + h) * DC + dim;
    *dst = __float2bfloat16((L > 0.f) ? acc / L : 0.f);
#if !NOLSE
    if (dim == 0) lse[t2 * H + h] = (L > 0.f) ? M + log2f(L) : -INFINITY;
#endif
#else  // !WPH2
    TDEP(mc); TDEP(lc);
#if !TIM_REL
    TSTAMP(12);  // this thread's (m, l) loads returned
#endif
    TDEP(__uint_as_float(vq[0].x)); TDEP(__uint_as_float(vq[7].w));
    TSTAMP(13);  // this thread's o loads returned
#ifdef DSA_TIMING
    { float tmp = 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k) tmp += __uint_as_float(vq[k].x) + __uint_as_float(vq[k].w);
      TDEP(tmp); __syncwarp();
      if (lane == 0) g_tbuf[16384 + blockIdx.x * 4 + warp] = gtimer(); }
#endif
    unsigned done = 0u;
#if !CTRPAR && !LATEDONE
    if (tid == 0) done = atom_add_relaxed_gpu(my_ctr + 1, 1u);  // consumed at the very end (reset decision)
#endif
    // block max of m over the chunks (warps 2,3 contribute -inf)
    float mw = mc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, o));
    if (lane == 0) cw[NCHUNK + warp] = mw;
    __syncthreads();
    TSTAMP(15);  // all warps' loads returned (first barrier of the weights phase)
    const float M = fmaxf(fmaxf(cw[NCHUNK], cw[NCHUNK + 1]), fmaxf(cw[NCHUNK + 2], cw[NCHUNK + 3]));
    const float w = (mc == -INFINITY) ? 0.f : exp2f(mc - M);
    float wl = w * lc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) wl += __shfl_xor_sync(0xffffffffu, wl, o);
    if (tid < NCHUNK) cw[tid] = w;
    if (lane == 0) cwl[warp] = wl;
    __syncthreads();
    TDEP(M);
    TSTAMP(9);
    const float L = (cwl[0] + cwl[1]) + (cwl[2] + cwl[3]);
    float a8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) a8[i] = 0.f;
#if PROGW == 3 || TAGVAL
#pragma unroll
    for (int k = 0; k < 8; ++k) { vq[k].x &= ~0x00010001u; vq[k].y &= ~0x00010001u; vq[k].z &= ~0x00010001u; vq[k].w &= ~0x00010001u; }
#endif
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float wk = cw[cg + 8 * k];
      a8[0] += wk * bf16lo(vq[k].x); a8[1] += wk * bf16hi(vq[k].x);
      a8[2] += wk * bf16lo(vq[k].y); a8[3] += wk * bf16hi(vq[k].y);
      a8[4] += wk * bf16lo(vq[k].z); a8[5] += wk * bf16hi(vq[k].z);
      a8[6] += wk * bf16lo(vq[k].w); a8[7] += wk * bf16hi(vq[k].w);
    }
    float* cx = reinterpret_cast<float*>(smem) + 256;  // [8 cg][16 dgrp][8] = 1024 floats
    {
      float4* d4 = reinterpret_cast<float4*>(cx + (cg * 16 + dgrp) * 8);
      d4[0] = make_float4(a8[0], a8[1], a8[2], a8[3]);
      d4[1] = make_float4(a8[4], a8[5], a8[6], a8[7]);
    }
    __syncthreads();
    float acc = 0.f;
    {
      const int dg = tid >> 3, di = tid & 7;   // dim = q*128 + 8*dg + di
#pragma unroll
      for (int g = 0; g < 8; ++g) acc += cx[(g * 16 + dg) * 8 + di];
    }
    TDEP(acc); TDEP(L);
    TSTAMP(10);
    __nv_bfloat16* dst = out + ((size_t)t2 * H + h) * DC + dim;
    *dst = __float2bfloat16((L > 0.f) ? acc / L : 0.f);
#if !NOLSE
    if (dim == 0) lse[t2 * H + h] = (L > 0.f) ? M + log2f(L) : -INFINITY;
#endif
#endif  // WPH2
#endif  // PROGW == 4
    // all COMB_PER_REQ combine CTAs of this request took a 'done' ticket after observing all arrivals;
    // the last one resets the line for the next launch
#if !CTRPAR
#if LATEDONE
    if (tid == 0) done = atom_add_relaxed_gpu(my_ctr + 1, 1u);
#endif
    if (tid == 0 && done == (unsigned)COMB_PER_REQ - 1) *reinterpret_cast<uint4*>(my_ctr) = make_uint4(0u, 0u, 0u, 0u);
#endif
  }
#if MLTOUCH
  if (touch == 0xffffffffu && tid == 0) ctr[0] = touch;  // never true in practice: keeps the early load alive
#endif
  TSTAMP(11);
#ifdef DSA_TIMING
  if (threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); g_tbuf[blockIdx.x * 16 + 14] = sm | ((unsigned long long)(unsigned)my_chunk << 16) | ((unsigned long long)(nvalid > 0) << 40); }
#endif
}

// exported constants for the host: S, NT, NCHUNK, SMEM_BYTES, COMB_PER_REQ, CTR_STRIDE
extern "C" __constant__ int dsa_cfg[8] = {S, NT, NCHUNK, SMEM_BYTES, COMB_PER_REQ, CTR_STRIDE, PARTIAL_BF16, 0};
