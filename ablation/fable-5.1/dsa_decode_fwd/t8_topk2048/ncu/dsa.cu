// DSA sparse decode attention (H=16, ckv=512, kpe=64, topk=2048, page=64), bf16 in / bf16 out + fp32 lse (base 2).
// v14: one kernel, two phases separated by a grid barrier (grid = all co-resident CTAs, from an occupancy query).
//  Phase 1: clusters of 2 CTAs (column halves). Chunk = (request t, 64 consecutive top-k slots); chunk c -> cluster c.
//   CTA (half ch): own 64 rows x ckv dims [256ch, +256) (+ kpe dims for ch == 0); Q half in registers (A fragments).
//   S_half = Q K^T (mma.sync, 2-way k-split x 2-way row-split over 4 warps) -> partial S (16 x 64, fp32) exchanged
//   with the partner through DSMEM -> softmax over the 64 rows (per head m, l) -> O_half = P V_half (256 cols)
//   -> bf16 partial [t][chunk][16][512] (+ m,l by half 0); empty chunks write m = -inf.
//  Phase 2: task = (t, head, 128-column quarter): 4 warps merge 32 chunk partials, write output / lse.
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;

#define CL 2
#define CS 80
#define H 16
#define DC 512
#define DP 64
#define TOPK 2048
#define NWARPS 4
#define NTHREADS (NWARPS * 32)
#define CPT ((TOPK + CS - 1) / CS)   // 26 chunks per request (last one covers rows 2000..2047)
#define HC (DC / 2)              // 256 ckv dims per column half
#define KW0 (HC + DP)            // 320 dims for half 0
#define KSTRIDE 328              // smem row stride (656 B): conflict-free ldmatrix
#define NT (CS / 8)              // 10 row tiles
#define KPV (CS / 16)            // 5 PV k-steps
#ifndef GATE
#define GATE 3
#endif
#ifndef POLL_A
#define POLL_A 64
#endif
#ifndef POLL_NS
#define POLL_NS 32
#endif
#define IDX_CG 1
// delay with exact cycle accounting (__nanosleep granularity is coarse and unpredictable)
#ifdef SPIN_CLK
__device__ __forceinline__ void poll_delay(unsigned ns) { const long long t0 = clock64(); const long long cyc = (long long)ns * 144 / 100; while (clock64() - t0 < cyc) {} }
#else
__device__ __forceinline__ void poll_delay(unsigned ns) { if (ns) __nanosleep(ns); }
#endif
#define CNT_STRIDE 32            // one 128 B line per counter (pollers + reductions of different requests do not share a line)
#define TASKS_PER_T (H * 2)      // combine tasks per request: 16 heads x 2 column halves (8 columns per lane)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
#if defined(CPA_CA)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_bytes));
#elif defined(CPA_L2_256)
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_bytes));
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_bytes));
#endif
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
// ---- mbarrier + st.async (DSMEM push with transaction counting) ----
__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile("{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra LAB_WAIT;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void st_async_f2(uint32_t daddr, float a, float b, uint32_t mbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n" :: "r"(daddr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
// generation-tagged words (no producer fence needed: the consumer validates the tag of every word it reads)
__device__ __forceinline__ uint32_t tag_bf16(float v, uint32_t tag16) { __nv_bfloat16 b = __float2bfloat16_rn(v); return ((uint32_t)*reinterpret_cast<unsigned short*>(&b) << 16) | tag16; }
__device__ __forceinline__ float untag_bf16(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t tag_f32(float v, uint32_t tag8) { return (__float_as_uint(v) & 0xffffff00u) | tag8; }
__device__ __forceinline__ float untag_f32(uint32_t w) { return __uint_as_float(w & 0xffffff00u); }
__device__ __forceinline__ float2 unpack_bf16(uint32_t u) { __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u); return __bfloat1622float2(v); }
__device__ __forceinline__ float fast_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint4 ld_cg_b128(const void* p) { uint4 v; asm volatile("ld.global.cg.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint2 ld_cg_b64(const void* p) { uint2 v; asm volatile("ld.global.cg.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }
__device__ __forceinline__ unsigned long long ld_cg_u64(const void* p) { unsigned long long v; asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_u64(void* p, unsigned long long v) { asm volatile("st.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ unsigned long long pack_ml(float m, float l, uint32_t tag8) { return ((unsigned long long)__float_as_uint(m) << 32) | tag_f32(l, tag8); }
__device__ __forceinline__ bool tags_ok(unsigned long long mw, uint4 a, uint4 b, uint32_t tag8, uint32_t tag16) {
  const uint32_t x = ((a.x ^ tag16) | (a.y ^ tag16) | (a.z ^ tag16) | (a.w ^ tag16) | (b.x ^ tag16) | (b.y ^ tag16) | (b.z ^ tag16) | (b.w ^ tag16)) & 0xffffu;
  return (x == 0u) & ((((uint32_t)mw ^ tag8) & 0xffu) == 0u);
}
__device__ __forceinline__ unsigned ld_cg_u32(const unsigned* p) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float2 ld_cg_f2(const float* p) { float2 v; asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p)); return v; }
__device__ __forceinline__ uint32_t ld_acquire_u32(const unsigned* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint32_t ldg_u32(const void* p) { uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint32_t mapa_u32(uint32_t saddr, uint32_t rank) { uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(saddr), "r"(rank)); return r; }
__device__ __forceinline__ uint2 ld_dsmem_b64(uint32_t a) { uint2 v; asm volatile("ld.shared::cluster.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ void cluster_arrive_smem() { __threadfence_block(); asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_rlx() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_rlx() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
#ifdef DEBUG_TS
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TS(i) do { if (threadIdx.x == 0) ts_s[(i)] = gtimer(); } while (0)
#define CK(i) do { if (threadIdx.x == 0) ts_s[(i)] = clock64(); } while (0)
#define TS_FLUSH() do { __syncthreads(); if (ts) for (int i = threadIdx.x; i < 32; i += blockDim.x) ts[blockIdx.x * 32 + i] = ts_s[i]; } while (0)
#else
#define TS(i) do {} while (0)
#define CK(i) do {} while (0)
#define TS_FLUSH() do {} while (0)
#endif

#define QSTRIDE 640              // Q half row stride in bytes (40 chunks of 16 B), XOR-swizzled chunks
struct Smem {
  __nv_bfloat16 k[CS * KSTRIDE];       // 41984 B: own 64 rows x (256 ckv [+64 kpe]) dims
  union {
    unsigned char q[H * QSTRIDE];      // 10240 B: Q half [16 rows][40 chunks] (live until the end of QK)
    float scratch[2][NT][32][4];       // 10240 B: [kq][row tile][lane][4] S partials (written after QK)
    struct { float tmax[NT][16], tl[NT][16]; };   // per-tile stats (between k-reduce and PV)
    uint32_t o[H * (HC / 2)];          // 8192 B: bf16 O staging [head][256] (after PV)
    struct { float cmerge[NWARPS][2][16]; float cacc[NWARPS][256]; unsigned cbusy; };   // combine phase
  };
  uint32_t p[NT][32][2];               // 2.5 KB: bf16 P tiles in A-fragment order (after B1)
  float rx[16 * CS];                   // 5 KB: the partner's S partial [head][80 rows] (fp32), pushed here via st.async
  unsigned long long xbar;             // mbarrier counting the partner's st.async bytes (one phase per busy chunk)
  float ml[32];                        // m[16], l[16]
  int idx[CS];                         // own 64 indices
};
#define SMEM_BYTES (sizeof(Smem))

extern "C" __global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(NTHREADS, 3)
dsa_kernel(const __nv_bfloat16* __restrict__ q_nope, const __nv_bfloat16* __restrict__ q_pe,
           const __nv_bfloat16* __restrict__ ckv, const __nv_bfloat16* __restrict__ kpe,
           const int* __restrict__ sidx, uint32_t* __restrict__ part, unsigned long long* __restrict__ pml,
           unsigned* __restrict__ cnt, __nv_bfloat16* __restrict__ out, float* __restrict__ lse,
           float scale_log2, int T, int nclusters, unsigned gen, unsigned long long* __restrict__ ts)
{
#ifdef DEBUG_TS
  __shared__ unsigned long long ts_s[32];
  if (threadIdx.x < 32) ts_s[threadIdx.x] = 0;
  __syncthreads();
#endif
  TS(0);
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const int ch = (int)cluster.block_rank();          // column half
  const int cid = blockIdx.x / CL;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lrow = lane & 7, lmat = lane >> 3;
  const int g = lane >> 2, qd = lane & 3;
  const int partner = ch ^ 1;
  const int kw = ch ? HC : KW0;                      // 256 or 320 dims
  const int ksteps_w = kw / 32;                      // k-steps per warp (2-way k-split): 8 or 10
  const int nchunks = T * CPT;
  const int nctas = nclusters * CL;
  int pending_wait = 0;
  unsigned xphase = 0;
  if (tid == 0) { mbar_init(&sm.xbar, 1); fence_mbar_init(); }
  __syncthreads();
  cluster_arrive_rlx(); pending_wait = 1;       // the partner must see my initialized mbarrier before it pushes
  const uint32_t tag16 = gen & 0xffffu, tag8 = gen & 0xffu;

  // =============================== phase 1: chunks ===============================
  // chunk <-> cluster mapping. Clusters run on SM pair (c mod 74) in slot (c / 74) on this GPU; for T == 8 the
  // mapping below spreads low-block (likely busy) chunks of different slots over different SM pairs.
  const int nids = (T == 8 && CPT == 26) ? 222 : nchunks;
  for (int chunk = cid; chunk < nids; chunk += nclusters) {
    int t, blk;
    if (T == 8 && CPT == 26) {
      // slot k = blk / 9 (0: blk 0-8, 1: 9-17, 2: 18-25), j = blk - 9k:  pair = (d_k*8*j + ((t + r_k) & 7) + s_k) mod 74.
      // Slots 1 and 2 run in descending block order so that their low (likely busy) blocks share SM pairs with the
      // high (likely idle) blocks of the lower slots; rotations r_k pair different requests.
      const int k = chunk / 74, pp = chunk - k * 74;
      if (k > 2) continue;
      const int dk = (k == 0) ? 1 : -1;
      const int rk = (k == 0) ? 0 : (k == 1 ? 2 : 4);
      const int sk = (k == 0) ? 0 : 66;
      const int q = (pp - sk + 74) % 74;
      int j = -1, tt = 0;
#pragma unroll
      for (int jj = 0; jj < 9; ++jj) {
        const int v = ((q - dk * 8 * jj) % 74 + 74) % 74;
        if (v < 8 && j < 0) { j = jj; tt = v; }
      }
      if (j < 0) continue;
      t = (tt - rk + 8) & 7;
      blk = 9 * k + j;
      if (blk >= CPT) continue;
    } else {
      t = chunk % T; blk = chunk / T;              // blk-major: likely-busy (low blk) chunks are launched first
    }
    // ---- own 64 indices (256 B) ----
    int myidx[3];
    {
      const int* ip = sidx + (size_t)t * TOPK + blk * CS;
#ifdef IDX_CG
      myidx[0] = (blk * CS + lane < TOPK) ? (int)ld_cg_u32((const unsigned*)(ip + lane)) : -1;
      myidx[1] = (blk * CS + 32 + lane < TOPK) ? (int)ld_cg_u32((const unsigned*)(ip + 32 + lane)) : -1;
      myidx[2] = (lane < 16 && blk * CS + 64 + lane < TOPK) ? (int)ld_cg_u32((const unsigned*)(ip + 64 + lane)) : -1;
#elif defined(IDX_PLAIN)
      myidx[0] = (blk * CS + lane < TOPK) ? ip[lane] : -1;
      myidx[1] = (blk * CS + 32 + lane < TOPK) ? ip[32 + lane] : -1;
      myidx[2] = (lane < 16 && blk * CS + 64 + lane < TOPK) ? ip[64 + lane] : -1;
#else
      myidx[0] = (blk * CS + lane < TOPK) ? __ldg(ip + lane) : -1;
      myidx[1] = (blk * CS + 32 + lane < TOPK) ? __ldg(ip + 32 + lane) : -1;
      myidx[2] = (lane < 16 && blk * CS + 64 + lane < TOPK) ? __ldg(ip + 64 + lane) : -1;
#endif
    }
    // ---- Q half -> swizzled smem (cp.async), independent of the indices ----
    {
      const __nv_bfloat16* qn = q_nope + (size_t)t * H * DC + ch * HC;
      const __nv_bfloat16* qp = q_pe + (size_t)t * H * DP;
      const int nch = ch ? 32 : 40;
      for (int c = tid; c < H * nch; c += NTHREADS) {
        const int r = c / nch, cc = c - r * nch;
        const __nv_bfloat16* src = (cc < 32) ? (qn + r * DC + cc * 8) : (qp + r * DP + (cc - 32) * 8);
        cp_async16(sm.q + r * QSTRIDE + ((cc ^ (r & 7)) << 4), src, 16);
      }
      cp_async_commit();
    }
    // touch the output pages now (fresh buffers in the harness: the address translation overlaps the gather instead of
    // stalling the final stores); the loaded values are discarded
    if (chunk == cid && tid == 0) {   // (asm volatile: the loads are kept; their results are never waited on)
      unsigned d0, d1;
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(d0) : "l"(out + ((size_t)(blockIdx.x % T) * H) * DC));
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(d1) : "l"(lse + (blockIdx.x % T) * H));
    }
    const unsigned vm0 = __ballot_sync(0xffffffffu, myidx[0] >= 0);
    const unsigned vm1 = __ballot_sync(0xffffffffu, myidx[1] >= 0);
    const unsigned vm2 = __ballot_sync(0xffffffffu, myidx[2] >= 0) & 0xffffu;
    const bool busy = (vm0 | vm1 | vm2) != 0u;    // identical in every warp (same loads)
    TS(1);
    if (busy) {
      if (pending_wait) { cluster_wait_rlx(); pending_wait = 0; }
      // ---- K half gather: warp w issues rows [20w, 20w+20); row index via shuffle (no smem table, no barrier) ----
      {
        const int r0 = warp * 20;
#pragma unroll
        for (int i = 0; i < 20; ++i) {
          const int rr = r0 + i;
          const int u = rr >> 5;                                   // warp-uniform
          const int sel = (u == 0) ? myidx[0] : ((u == 1) ? myidx[1] : myidx[2]);
          const int ix = __shfl_sync(0xffffffffu, sel, rr & 31);
          const bool valid = ix >= 0; const size_t row = valid ? (size_t)ix : 0;
          cp_async16(sm.k + rr * KSTRIDE + lane * 8, ckv + row * DC + ch * HC + lane * 8, valid ? 16 : 0);
        }
        if (ch == 0) {   // kpe: 8 chunks of 16 B per row; 4 rows per instruction
#pragma unroll
          for (int i = 0; i < 5; ++i) {
            const int rr = r0 + i * 4 + (lane >> 3);
            const int u = rr >> 5;
            const int sel = (u == 0) ? myidx[0] : ((u == 1) ? myidx[1] : myidx[2]);
            const int ix = __shfl_sync(0xffffffffu, sel, rr & 31);
            const bool valid = ix >= 0; const size_t row = valid ? (size_t)ix : 0;
            cp_async16(sm.k + rr * KSTRIDE + HC + (lane & 7) * 8, kpe + row * DP + (lane & 7) * 8, valid ? 16 : 0);
          }
        }
      }
      cp_async_commit();
      if (tid == 0) mbar_arrive_expect_tx(&sm.xbar, 16 * CS * 4);   // partner's S partial (5120 B) will land in rx
      cp_async_wait<1>();                          // Q (group 0) landed long ago
      __syncthreads();                             // Q visible to every warp
      cp_async_wait<0>();                          // own 20 K rows landed
      asm volatile("bar.sync %0, 64;" :: "r"(1 + (warp >> 1)) : "memory");   // warp pair: rows [40*(warp>>1), +40) visible
      TS(2); CK(20);
      // ---- S partial: warp (kq = warp&1, nq = warp>>1): k-steps of its k-half, row tiles [5nq, 5nq+5) ----
      {
        const int kq = warp & 1, nq = warp >> 1;
        float sacc[5][4];
#pragma unroll
        for (int n = 0; n < 5; ++n) { sacc[n][0] = sacc[n][1] = sacc[n][2] = sacc[n][3] = 0.f; }
        const int k0w = kq * (kw / 2);
        const int qrow = lrow + 8 * (lmat & 1);
#pragma unroll
        for (int ks = 0; ks < 10; ++ks) {
          if (ks >= ksteps_w) break;
          const int k0 = k0w + ks * 16;
          uint32_t aa[4];
          {
            const int cc = (k0 >> 3) + (lmat >> 1);
            ldsm_x4(aa[0], aa[1], aa[2], aa[3], sm.q + qrow * QSTRIDE + ((cc ^ (qrow & 7)) << 4));
          }
#pragma unroll
          for (int n = 0; n < 4; n += 2) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3, sm.k + ((nq * 5 + n) * 8 + lrow + 8 * (lmat >> 1)) * KSTRIDE + k0 + 8 * (lmat & 1));
            mma_bf16_16816(sacc[n], aa, b0, b1);
            mma_bf16_16816(sacc[n + 1], aa, b2, b3);
          }
          {
            uint32_t b0, b1;
            ldsm_x2(b0, b1, sm.k + ((nq * 5 + 4) * 8 + lrow) * KSTRIDE + k0 + 8 * ((lane >> 3) & 1));
            mma_bf16_16816(sacc[4], aa, b0, b1);
          }
        }
        CK(9);
        __syncthreads();   // every warp finished reading Q (scratch aliases it)
#pragma unroll
        for (int n = 0; n < 5; ++n)
          reinterpret_cast<float4*>(sm.scratch[kq][nq * 5 + n])[lane] = make_float4(sacc[n][0], sacc[n][1], sacc[n][2], sacc[n][3]);
      }
      __syncthreads();
      TS(3); CK(10);
      // ---- k-reduce: warp w owns tiles w, w+4, w+8 -> this half's S partial, pushed into the partner's rx ----
      if (pending_wait) { cluster_wait_rlx(); pending_wait = 0; }   // partner's mbarrier initialized / rx free
      float4 sp[3];
      const int ntile_w = (warp < 2) ? 3 : 2;      // tiles w, w+4, w+8 (< 10)
      {
        const uint32_t prx = mapa_u32(smem_u32(sm.rx), partner), pbar = mapa_u32(smem_u32(&sm.xbar), partner);
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          if (u < ntile_w) {
            const int n = warp + 4 * u;
            float4 v0 = reinterpret_cast<const float4*>(sm.scratch[0][n])[lane];
            float4 v1 = reinterpret_cast<const float4*>(sm.scratch[1][n])[lane];
            sp[u] = make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
            const int r0 = n * 8 + 2 * qd;
            st_async_f2(prx + (g * CS + r0) * 4, sp[u].x, sp[u].y, pbar);
            st_async_f2(prx + ((g + 8) * CS + r0) * 4, sp[u].z, sp[u].w, pbar);
          } else sp[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
      }
      CK(22);
      mbar_wait(&sm.xbar, xphase); xphase ^= 1u;   // B1: the partner's S partial has landed in rx
      TS(4); CK(23);
      float s[3][4];
#pragma unroll
      for (int u = 0; u < 3; ++u) {
        if (u < ntile_w) {
          const int n = warp + 4 * u;
          const int r0 = n * 8 + 2 * qd;
          const float2 w0 = *reinterpret_cast<const float2*>(sm.rx + g * CS + r0);
          const float2 w1 = *reinterpret_cast<const float2*>(sm.rx + (g + 8) * CS + r0);
          s[u][0] = sp[u].x + w0.x; s[u][1] = sp[u].y + w0.y;
          s[u][2] = sp[u].z + w1.x; s[u][3] = sp[u].w + w1.y;
        } else { s[u][0] = s[u][1] = s[u][2] = s[u][3] = -INFINITY; }
      }
      __syncwarp();
      CK(11);
      cluster_arrive_rlx(); pending_wait = 1;      // done reading rx (the partner may push the next chunk after my arrival)
      // ---- softmax over the 80 rows (identical in both halves) ----
      {
        float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          if (u < ntile_w) {
            const int n = warp + 4 * u;
            const int r0 = n * 8 + 2 * qd;
            const unsigned vm = (r0 < 32) ? vm0 : ((r0 < 64) ? vm1 : vm2);
            const int rb = r0 & 31;
            const bool v0 = (vm >> rb) & 1, v1 = (vm >> (rb + 1)) & 1;
            s[u][0] = v0 ? s[u][0] * scale_log2 : -INFINITY; s[u][1] = v1 ? s[u][1] * scale_log2 : -INFINITY;
            s[u][2] = v0 ? s[u][2] * scale_log2 : -INFINITY; s[u][3] = v1 ? s[u][3] * scale_log2 : -INFINITY;
            m0 = fmaxf(m0, fmaxf(s[u][0], s[u][1])); m1 = fmaxf(m1, fmaxf(s[u][2], s[u][3]));
          }
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        if (qd == 0) { sm.tmax[warp][g] = m0; sm.tmax[warp][g + 8] = m1; }   // per-warp max (over its tiles)
        __syncthreads();
        CK(12);
        float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) { M0 = fmaxf(M0, sm.tmax[tt][g]); M1 = fmaxf(M1, sm.tmax[tt][g + 8]); }
        const float mu0 = (M0 == -INFINITY) ? 0.f : M0, mu1 = (M1 == -INFINITY) ? 0.f : M1;
        float l0 = 0.f, l1 = 0.f;
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          if (u < ntile_w) {
            const int n = warp + 4 * u;
            const float p0 = fast_exp2(s[u][0] - mu0), p1 = fast_exp2(s[u][1] - mu0), p2 = fast_exp2(s[u][2] - mu1), p3 = fast_exp2(s[u][3] - mu1);
            l0 += p0 + p1; l1 += p2 + p3;
            sm.p[n][lane][0] = pack_bf16(p0, p1);
            sm.p[n][lane][1] = pack_bf16(p2, p3);
          }
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        if (qd == 0) { sm.tl[warp][g] = l0; sm.tl[warp][g + 8] = l1; }
        if (warp == 0 && qd == 0) { sm.ml[g] = M0; sm.ml[g + 8] = M1; }
        __syncthreads();
        CK(13);
        if (tid < 16) sm.ml[16 + tid] = sm.tl[0][tid] + sm.tl[1][tid] + sm.tl[2][tid] + sm.tl[3][tid];
      }
      // ---- O_half[:, warp cols] = P V_half : warp w -> cols [64w, 64w+64) ----
      uint32_t pa[KPV][4];
#pragma unroll
      for (int ks = 0; ks < KPV; ++ks) {
        uint2 w0 = *reinterpret_cast<const uint2*>(sm.p[2 * ks][lane]);
        uint2 w1 = *reinterpret_cast<const uint2*>(sm.p[2 * ks + 1][lane]);
        pa[ks][0] = w0.x; pa[ks][1] = w0.y; pa[ks][2] = w1.x; pa[ks][3] = w1.y;
      }
      float oacc[8][4];
#pragma unroll
      for (int n = 0; n < 8; ++n) { oacc[n][0] = oacc[n][1] = oacc[n][2] = oacc[n][3] = 0.f; }
      {
        const int c0w = warp * 64;
#pragma unroll
        for (int ks = 0; ks < KPV; ++ks) {
#pragma unroll
          for (int n = 0; n < 8; n += 2) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4_t(b0, b1, b2, b3, sm.k + (ks * 16 + lrow + 8 * (lmat & 1)) * KSTRIDE + c0w + n * 8 + 8 * (lmat >> 1));
            mma_bf16_16816(oacc[n], pa[ks], b0, b1);
            mma_bf16_16816(oacc[n + 1], pa[ks], b2, b3);
          }
        }
      }
      CK(14);
      TS(5); CK(16);
      // ---- tagged partial straight from registers: 8 B per lane per (row, n-tile); a quad covers one 32 B sector ----
      {
#ifdef STG128
        // quad exchange: lane qd=2h takes tile n cols 4h..4h+3, lane qd=2h+1 takes tile n+1 cols 4h..4h+3 -> 16 B stores
        uint32_t* pbase = part + (((size_t)t * CPT + blk) * H) * DC + ch * HC + warp * 64 + 4 * (qd >> 1);
#pragma unroll
        for (int n = 0; n < 8; n += 2) {
          const int nm = (qd & 1) ? n + 1 : n;            // tile this lane will store
          const int no = (qd & 1) ? n : n + 1;            // tile this lane gives away
          // my 2 cols of tile nm (rows g, g+8) + partner's 2 cols of tile nm
          const uint32_t a0 = tag_bf16(oacc[nm][0], tag16), a1 = tag_bf16(oacc[nm][1], tag16);
          const uint32_t a2 = tag_bf16(oacc[nm][2], tag16), a3 = tag_bf16(oacc[nm][3], tag16);
          const uint32_t b0 = tag_bf16(oacc[no][0], tag16), b1 = tag_bf16(oacc[no][1], tag16);
          const uint32_t b2 = tag_bf16(oacc[no][2], tag16), b3 = tag_bf16(oacc[no][3], tag16);
          const uint32_t p0 = __shfl_xor_sync(0xffffffffu, b0, 1), p1 = __shfl_xor_sync(0xffffffffu, b1, 1);
          const uint32_t p2 = __shfl_xor_sync(0xffffffffu, b2, 1), p3 = __shfl_xor_sync(0xffffffffu, b3, 1);
          // lane with qd even holds cols 2qd,2qd+1 = 4h,4h+1 of tile nm; partner (qd odd) holds 4h+2,4h+3
          const uint4 r0 = (qd & 1) ? make_uint4(p0, p1, a0, a1) : make_uint4(a0, a1, p0, p1);
          const uint4 r1 = (qd & 1) ? make_uint4(p2, p3, a2, a3) : make_uint4(a2, a3, p2, p3);
          *reinterpret_cast<uint4*>(pbase + g * DC + nm * 8) = r0;
          *reinterpret_cast<uint4*>(pbase + (g + 8) * DC + nm * 8) = r1;
        }
#else
        uint32_t* pbase = part + (((size_t)t * CPT + blk) * H) * DC + ch * HC + warp * 64 + 2 * qd;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          *reinterpret_cast<uint2*>(pbase + g * DC + n * 8) = make_uint2(tag_bf16(oacc[n][0], tag16), tag_bf16(oacc[n][1], tag16));
          *reinterpret_cast<uint2*>(pbase + (g + 8) * DC + n * 8) = make_uint2(tag_bf16(oacc[n][2], tag16), tag_bf16(oacc[n][3], tag16));
        }
#endif
        if (ch == 0 && tid < 16) st_u64(pml + ((size_t)t * CPT + blk) * H + tid, pack_ml(sm.ml[tid], sm.ml[16 + tid], tag8));
      }
      CK(17);
      __syncwarp();
      if (lane == 0) asm volatile("red.global.add.u32 [%0], 1;" :: "l"(cnt + t * CNT_STRIDE) : "memory");
      CK(19);
    } else {
      cp_async_wait<0>();
      if (ch == 0 && tid < 16) st_u64(pml + ((size_t)t * CPT + blk) * H + tid, pack_ml(-INFINITY, 0.f, tag8));
      if (tid == 0) asm volatile("red.global.add.u32 [%0], %1;" :: "l"(cnt + t * CNT_STRIDE), "r"((unsigned)NWARPS) : "memory");
    }
    TS(6); CK(21);
  }
  if (pending_wait) cluster_wait_rlx();
  // combine tasks are claimed dynamically (one claim per CTA per call, so the monotonic counter cnt[T] advances by
  // exactly nctas per call): CTAs whose chunks were empty are ready first and take all tasks. The exact busy mask of
  // the task's request is computed from its indices (already L2-resident) before polling.
  __shared__ int s_task;
  __syncthreads();
  if (tid == 0) { sm.cbusy = 0u; s_task = (int)(atomicAdd(cnt + T * CNT_STRIDE, 1u) - (gen - 1u) * (unsigned)nctas); }
  __syncthreads();
  const int task0 = s_task;
  if (task0 >= 0 && task0 < T * TASKS_PER_T) {
    const int t = task0 / TASKS_PER_T;
    const int4* ip = reinterpret_cast<const int4*>(sidx + (size_t)t * TOPK);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i4 = tid + k * NTHREADS;                        // int4 index -> chunk (4*i4)/CS (20 int4 per chunk)
      const int4 v = __ldg(ip + i4);
      if ((v.x >= 0) | (v.y >= 0) | (v.z >= 0) | (v.w >= 0)) atomicOr(&sm.cbusy, 1u << ((4 * i4) / CS));
    }
  }
  __syncthreads();
  TS(7);
  // =============================== phase 2: combine tasks ===============================
  for (int task = task0; task >= 0 && task < T * TASKS_PER_T; task += nctas) {
    const int t = task / TASKS_PER_T, head = (task % TASKS_PER_T) >> 1, quarter = task & 1;
    const int c = quarter * 256 + lane * 4;     // lane: columns [c, c+4) and [c+128, c+132) (contiguous 512 B per warp load)
    unsigned bmask = sm.cbusy;
    if (task != task0) {                        // (only for T large) recompute the mask for a later task
      __syncthreads();
      if (tid == 0) sm.cbusy = 0u;
      __syncthreads();
      const int4* ip = reinterpret_cast<const int4*>(sidx + (size_t)t * TOPK);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int i4 = tid + k * NTHREADS;
        const int4 v = __ldg(ip + i4);
        if ((v.x >= 0) | (v.y >= 0) | (v.z >= 0) | (v.w >= 0)) atomicOr(&sm.cbusy, 1u << ((4 * i4) / CS));
      }
      __syncthreads();
      bmask = sm.cbusy;
    }
    // wait until all 64 contributions (32 chunks x 2 halves) of request t for this call have been published
    if (tid == 0) {   // phase A: wait until at most one chunk pair (2 halves) of request t is still in flight
      const unsigned target = (gen * (unsigned)(CPT * CL) - (unsigned)GATE) * (unsigned)NWARPS;
      int spins = 0;
      while (ld_cg_u32(cnt + t * CNT_STRIDE) < target) { if (++spins > (1 << 22)) break; poll_delay(POLL_A); }
    }
    TS(9); CK(25);
#ifdef DEBUG_TS
    if (tid == 0) ts_s[29] = t + 1;
#endif
    __syncthreads();
    CK(26);
    // warp w merges chunks w, w+4, ..., w+28 (busy ones only)
    float M = -INFINITY, L = 0.f;
    float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f), acc1 = acc0;
#define CPW ((CPT + NWARPS - 1) / NWARPS)
    float2 mls[CPW];
    uint4 ov0[CPW], ov1[CPW];
    {
      unsigned long long mw[CPW];
#pragma unroll
      for (int i = 0; i < CPW; ++i) {   // first pass: all loads issued back to back
        const int s_ = warp + i * NWARPS;
        const bool bz = (s_ < CPT) && ((bmask >> s_) & 1u);
        const uint32_t* pp = part + (((size_t)t * CPT + s_) * H + head) * DC + c;
        mw[i] = bz ? ld_cg_u64(pml + ((size_t)t * CPT + s_) * H + head) : pack_ml(-INFINITY, 0.f, tag8);
        ov0[i] = bz ? ld_cg_b128(pp) : make_uint4(tag16, tag16, tag16, tag16);
#ifdef HALF_LOADS
        ov1[i] = ov0[i];
#else
        ov1[i] = bz ? ld_cg_b128(pp + 128) : make_uint4(tag16, tag16, tag16, tag16);
#endif
      }
      // phase B: validate every word's tag; the (at most one or two) slots whose stores have not landed yet are polled directly
      bool ok[CPW];
      int tries = 0;
      while (true) {
        bool allok = true;
#pragma unroll
        for (int i = 0; i < CPW; ++i) { ok[i] = __all_sync(0xffffffffu, tags_ok(mw[i], ov0[i], ov1[i], tag8, tag16)); allok &= ok[i]; }
        if (allok || ++tries > (1 << 20)) break;
#ifdef DEBUG_TS
        if (lane == 0) ts_s[28] += 1;
        if (lane == 0) ts_s[16 + warp] = tries;
#endif
        poll_delay(POLL_NS);
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
          if (!ok[i]) {
            const int s_ = warp + i * NWARPS;
            const uint32_t* pp = part + (((size_t)t * CPT + s_) * H + head) * DC + c;
            mw[i] = ld_cg_u64(pml + ((size_t)t * CPT + s_) * H + head);
            ov0[i] = ld_cg_b128(pp); ov1[i] = ld_cg_b128(pp + 128);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CPW; ++i) mls[i] = make_float2(__uint_as_float((uint32_t)(mw[i] >> 32)), untag_f32((uint32_t)mw[i]));
#ifdef DEBUG_TS
      if (lane == 0) ts_s[9 + warp] = clock64();
#endif
    }
#pragma unroll
    for (int i = 0; i < CPW; ++i) M = fmaxf(M, mls[i].x);
    const float Mu = (M == -INFINITY) ? 0.f : M;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {   // f == 0 exactly for absent/empty slots (m = -inf); tag bits (< 1% of a bf16 ulp) are left in
      const float f = fast_exp2(mls[i].x - Mu);
      acc0.x += f * __uint_as_float(ov0[i].x); acc0.y += f * __uint_as_float(ov0[i].y);
      acc0.z += f * __uint_as_float(ov0[i].z); acc0.w += f * __uint_as_float(ov0[i].w);
      acc1.x += f * __uint_as_float(ov1[i].x); acc1.y += f * __uint_as_float(ov1[i].y);
      acc1.z += f * __uint_as_float(ov1[i].z); acc1.w += f * __uint_as_float(ov1[i].w);
      L += f * mls[i].y;
    }
    // cross-warp merge through smem
    CK(27);
    if (lane == 0) { sm.cmerge[warp][0][0] = M; sm.cmerge[warp][1][0] = L; }
    reinterpret_cast<float4*>(sm.cacc[warp])[lane] = acc0;
    reinterpret_cast<float4*>(sm.cacc[warp])[32 + lane] = acc1;
    __syncthreads();
    if (warp == 0) {
      float Mg = -INFINITY;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) Mg = fmaxf(Mg, sm.cmerge[w][0][0]);
      const float Mgu = (Mg == -INFINITY) ? 0.f : Mg;
      float4 ag0 = make_float4(0.f, 0.f, 0.f, 0.f), ag1 = ag0; float Lg = 0.f;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) {
        const float mw = sm.cmerge[w][0][0];
        const bool ok = mw != -INFINITY;
        const float f = ok ? fast_exp2(mw - Mgu) : 0.f;
        const float4 v0 = reinterpret_cast<const float4*>(sm.cacc[w])[lane];
        const float4 v1 = reinterpret_cast<const float4*>(sm.cacc[w])[32 + lane];
        ag0.x += ok ? f * v0.x : 0.f; ag0.y += ok ? f * v0.y : 0.f; ag0.z += ok ? f * v0.z : 0.f; ag0.w += ok ? f * v0.w : 0.f;
        ag1.x += ok ? f * v1.x : 0.f; ag1.y += ok ? f * v1.y : 0.f; ag1.z += ok ? f * v1.z : 0.f; ag1.w += ok ? f * v1.w : 0.f;
        Lg += ok ? f * sm.cmerge[w][1][0] : 0.f;
      }
      const float inv = (Lg > 0.f) ? 1.f / Lg : 0.f;
      __nv_bfloat162 o0 = __floats2bfloat162_rn(ag0.x * inv, ag0.y * inv);
      __nv_bfloat162 o1 = __floats2bfloat162_rn(ag0.z * inv, ag0.w * inv);
      __nv_bfloat162 o2 = __floats2bfloat162_rn(ag1.x * inv, ag1.y * inv);
      __nv_bfloat162 o3 = __floats2bfloat162_rn(ag1.z * inv, ag1.w * inv);
      *reinterpret_cast<uint2*>(out + ((size_t)t * H + head) * DC + c) = make_uint2(*reinterpret_cast<uint32_t*>(&o0), *reinterpret_cast<uint32_t*>(&o1));
      *reinterpret_cast<uint2*>(out + ((size_t)t * H + head) * DC + c + 128) = make_uint2(*reinterpret_cast<uint32_t*>(&o2), *reinterpret_cast<uint32_t*>(&o3));
      if (lane == 0 && quarter == 0) lse[t * H + head] = (Lg > 0.f) ? (Mg + __log2f(Lg)) : -INFINITY;
    }
    CK(24);
    if (task + nctas < T * TASKS_PER_T) __syncthreads();
    CK(23);
  }
  TS(8);
  TS(15);
#ifdef DEBUG_TS
  if (threadIdx.x == 0) { unsigned sid; asm volatile("mov.u32 %0, %%smid;" : "=r"(sid)); ts_s[31] = sid; }
#endif
  TS_FLUSH();
}
