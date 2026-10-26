// k-outer with the whole GQA group collapsed into the CTA.
//
// hand63 (q-outer) is 1.448 ms against ref's 1.485.  Two probes bracket where
// the remaining time goes:
//
//   probe58  every reduction deleted        1.066 ms   <- pure compute
//   hand63   4.43 GB of dK/dV reduction     1.448 ms   <- +0.382
//   probe66  half the reductions dropped    1.255 ms   <- +0.190, exactly half
//   probe67  drain reads 8 TMEM cols not 64 1.414 ms   <- -0.023, TMEM is free
//
// So the gap is the reduction and nothing else, it scales linearly with the
// bytes, and the budget for hitting 1.238 is 1.238 - 1.066 = 0.172 ms.  4.43 GB
// costs 0.382 and 2.21 GB costs 0.190, so only the halved traffic fits.
//
// dQ's reduction axis is k; dK/dV's is q (and head).  They are orthogonal --
// whichever one a CTA walks, the other has to be reduced across CTAs -- so with
// TMEM holding exactly four 128x128 fp32 accumulators the choice is binary:
//
//   q-outer  dQ resident, dK/dV reduced   4.43 GB  (hand63)
//   k-outer  dK/dV resident, dQ reduced   2.21 GB  (this file)
//
// The asymmetry is GQA: H_KV = 2, so a CTA that owns (b, h_kv, k_tile) can walk
// all GROUP = 8 q heads and accumulate every one of their dK/dV contributions
// into the same pair of TMEM slots.  dK/dV then never touch memory until the
// epilogue, where each is written once, by its sole owner, as a plain bf16
// store -- no accumulator buffer, no memset, no cast pass.
//
// What k-outer gives up is delta.  hand63 computes rowsum(O * dO) once in its
// prologue because the CTA owns one q tile for its whole walk; here every
// iteration visits a new q tile, so delta has to be precomputed.  hand224_pre
// does that and zeroes dQ in the same pass (402 MB, ~72 us), and dQ's reduction
// is bf16 straight into the final (B,N,H,D) layout -- at an average of 16
// accumulations per element, against hand63's 128 per dK/dV element, bf16 has
// the headroom that ruled it out on the q-outer side.
//
// The five GEMMs are hand63's, with A and B swapped in residency only; the
// three instruction descriptors are bit-identical:
//
//   S  = Q . K^T    A = Q_s[slot] (1,64)  B = K_s  (1,64)     0x08200490
//   dP = dO . V^T   A = dO_s (1,64)       B = V_s  (1,64)     0x08200490
//   dV = P^T . dO   A = P_s  (1024,64)    B = dO_s (1024,64)  0x08218490
//   dK = dS^T . Q   A = dS_s (1024,64)    B = Q_s[slot] (1024,64) 0x08218490
//   dQ = dS . K     A = dS_s (1,64)       B = K_s  (1024,64)  0x08210490
//
// TMEM: S, dP, dK, dV -- and dQ has no slot of its own, so it timeshares dP's.
// That is the one serial point in the body (dQ(it) cannot issue until the
// element warps have read dP(it) out, and dP(it+1) cannot issue until the drain
// warps have read dQ(it) out), and it replaces hand63's two-way dV/dK
// timeshare, so the body has one handover instead of two.
//
// K and V are loaded once and stay; Q needs two slots (S(it+1) overlaps
// dK(it)); dO needs only one, because dV(it) is its sole reader and three GEMMs
// separate that from dP(it+1).
#include <cuda.h>
#include <cuda_fp16.h>
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/atomic.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/common.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

#define NTH 544
#define NQT 32     // N / 128
#define GROUP 8    // H / H_KV
#define SMEM_BYTES 229376

#define PSW(t, i)                                                              \
  (((t)*64) + (((((i) >> 2) + (((t)&7) >> 2)) & 1) * 32) +                     \
   (((((((i)&3) >> 1) + (((t)&3) >> 1)) & 1)) * 16) +                          \
   (((((i)&1) + ((t)&1)) & 1) * 8))

#define SCALE2 0x1.0527dbd87e24dp-3f  // softmax_scale * log2(e)
#define SCALE 0x1.6a09e667f3bcdp-4f   // softmax_scale
#define LOG2E 0x1.71547652b82fep+0f

// bf16, not fp16: this reduces straight into the final dQ, so there is no cast
// pass to undo an fp16 accumulator.  8 channels, 16 B, one instruction.
#define RED_BF16X2(ptr, src)                                                   \
  {                                                                            \
    __nv_bfloat162 hh[4];                                                      \
    _Pragma("unroll") for (int _v = 0; _v < 4; ++_v) {                         \
      hh[_v] = __float22bfloat162_rn(*(float2 *)((src) + (_v * 2)));           \
    }                                                                          \
    asm volatile(                                                              \
        "red.global.add.noftz.v4.bf16x2 [%0], {%1,%2,%3,%4};\n" ::"l"((ptr)),  \
        "r"(*(uint32_t *)(hh + 0)), "r"(*(uint32_t *)(hh + 1)),                \
        "r"(*(uint32_t *)(hh + 2)), "r"(*(uint32_t *)(hh + 3))                 \
        : "memory");                                                           \
  }

#define STS_TILE_H(dst, src, base, ibase)                                      \
  _Pragma("unroll") for (int i = 0; i < 4; ++i) {                              \
    _Pragma("unroll") for (int vec = 0; vec < 2; ++vec) {                      \
      uint2 u;                                                                 \
      float4 v = *(float4 *)((src) + (i * 8) + (vec * 4));                     \
      (reinterpret_cast<__nv_bfloat162 *>(&u))[0] =                            \
          __float22bfloat162_rn(((float2 *)(&v))[0]);                          \
      (reinterpret_cast<__nv_bfloat162 *>(&u))[1] =                            \
          __float22bfloat162_rn(((float2 *)(&v))[1]);                          \
      *(uint2 *)(cast8 + (vec * 4)) = u;                                       \
    }                                                                          \
    pk[(ibase) + i] = *(uint4 *)(cast8);                                       \
    *(uint4 *)(((bfloat16_t *)(dst)) + (base) + PSW(tid, (ibase) + i)) =       \
        pk[(ibase) + i];                                                       \
  }

// delta[b,h,n] = sum_d O * dO, and dQ zeroed for the reductions, in one pass.
// One 16-lane group per (b, n, h): 16 * 8 bf16 = the 128 channels.
extern "C" __global__ void __launch_bounds__(256)
    hand224_pre(const bfloat16_t *__restrict__ O,
               const bfloat16_t *__restrict__ dO, float *__restrict__ Delta,
               bfloat16_t *__restrict__ dQ) {
  const int g = (int)(blockIdx.x * 16 + (threadIdx.x >> 4));
  const int l = (int)(threadIdx.x & 15);
  const long base = ((long)g * 128) + (l * 8);
  uint4 ou = *(const uint4 *)(O + base);
  uint4 du = *(const uint4 *)(dO + base);
  *(uint4 *)(dQ + base) = make_uint4(0u, 0u, 0u, 0u);
  float s = 0.0f;
#pragma unroll
  for (int v = 0; v < 4; ++v) {
    float2 a = __bfloat1622float2((reinterpret_cast<__nv_bfloat162 *>(&ou))[v]);
    float2 c = __bfloat1622float2((reinterpret_cast<__nv_bfloat162 *>(&du))[v]);
    s = __fmaf_rn(a.x, c.x, s);
    s = __fmaf_rn(a.y, c.y, s);
  }
#pragma unroll
  for (int off = 8; off > 0; off >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, off, 16);
  if (l == 0) {
    const int h = g & 15, bn = g >> 4;
    Delta[((bn >> 12) * 65536) + (h * 4096) + (bn & 4095)] = s;
  }
}

extern "C" __global__ void __launch_bounds__(NTH, 1)
    __cluster_dims__(1, 2, 1)
    hand224_kernel(__grid_constant__ const CUtensorMap K_desc,
                  const float *__restrict__ LSE,
                  __grid_constant__ const CUtensorMap Q_desc,
                  __grid_constant__ const CUtensorMap V_desc,
                  bfloat16_t *__restrict__ dK,
                  __grid_constant__ const CUtensorMap dO_desc,
                  bfloat16_t *__restrict__ dQ, bfloat16_t *__restrict__ dV,
                  const float *__restrict__ Delta) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void *K_s = ((void *)((char *)buf_dyn_shmem + 0));        // resident
  void *V_s = ((void *)((char *)buf_dyn_shmem + 32768));    // resident
  void *Q_s = ((void *)((char *)buf_dyn_shmem + 65536));    // two slots
  void *dO_s = ((void *)((char *)buf_dyn_shmem + 131072));  // one slot
  void *P_s = ((void *)((char *)buf_dyn_shmem + 163840));
  void *dS_s = ((void *)((char *)buf_dyn_shmem + 196608));

  __shared__ __align__(16) uint64_t bar_mma_mem[6];
  auto bar_s = reinterpret_cast<Barrier *>(bar_mma_mem + 0);
  auto bar_p = reinterpret_cast<Barrier *>(bar_mma_mem + 1);  // dP done
  auto bar_v = reinterpret_cast<Barrier *>(bar_mma_mem + 2);  // dV done
  auto bar_k = reinterpret_cast<Barrier *>(bar_mma_mem + 3);  // dK done
  auto bar_q = reinterpret_cast<Barrier *>(bar_mma_mem + 4);  // dQ cols 0-63
  auto bar_q2 = reinterpret_cast<Barrier *>(bar_mma_mem + 5); // dQ cols 64-127
  __shared__ __align__(16) uint64_t bar_el_mem[9];
  auto bar_ld0 = reinterpret_cast<Barrier *>(bar_el_mem + 0);  // prologue in
  auto bar_sd = reinterpret_cast<Barrier *>(bar_el_mem + 1);   // S_m drained
  auto bar_pst = reinterpret_cast<Barrier *>(bar_el_mem + 2);  // P in P_s
  auto bar_pd = reinterpret_cast<Barrier *>(bar_el_mem + 3);   // dP read out
  auto bar_dst = reinterpret_cast<Barrier *>(bar_el_mem + 4);  // dS in dS_s
  auto bar_ldq = reinterpret_cast<Barrier *>(bar_el_mem + 5);  // Q(it+1) in
  auto bar_ldo = reinterpret_cast<Barrier *>(bar_el_mem + 6);  // dO(it+1) in
  auto bar_qd = reinterpret_cast<Barrier *>(bar_el_mem + 7);   // dQ lo read
  auto bar_qd2 = reinterpret_cast<Barrier *>(bar_el_mem + 8);  // dQ hi read

  __shared__ __align__(16) uint S_m[1];
  __shared__ __align__(16) uint P_m[1];  // timeshared by dP and dQ
  __shared__ __align__(16) uint dK_m[1];
  __shared__ __align__(16) uint dV_m[1];

  const int tid = (int)threadIdx.x;
  // Longest-processing-time first.  A CTA's work is GROUP * (NQT - kt)
  // iterations, from 256 at kt = 0 down to 8 at kt = 31, and blocks launch in
  // linear-id order with blockIdx.x fastest -- so kt on y puts all sixteen
  // 256-iteration blocks in the first wave and the 8-iteration ones last.
  const int kt = (int)blockIdx.y;
  const int bkv = (int)blockIdx.x;  // batch * H_KV + kv
  const int b = bkv >> 1;
  const int kv = bkv & 1;
  const int nq = NQT - kt;          // q tiles per head
  const int nit = GROUP * nq;

  if (tl::tl_shuffle_elect<0>()) {
    bar_s[0].init(1);
    bar_p[0].init(1);
    bar_v[0].init(1);
    bar_k[0].init(1);
    bar_q[0].init(1);
    bar_q2[0].init(1);
    bar_ld0[0].init(1);
    bar_sd[0].init(1);
    bar_pst[0].init(256);
    bar_pd[0].init(1);
    bar_dst[0].init(1);
    bar_ldq[0].init(1);
    bar_ldo[0].init(1);
    bar_qd[0].init(128);
    bar_qd2[0].init(128);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();

  if ((tid >> 5) == 0) {
    tl::tmem_allocate((&(S_m[0])), 128);
    tl::tmem_allocate((&(P_m[0])), 128);
    tl::tmem_allocate((&(dK_m[0])), 128);
    tl::tmem_allocate((&(dV_m[0])), 128);
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  // The four TMEM base addresses live in shared memory, so every use was a
  // separate LDS: the syncs between uses stop the compiler from reusing one.
  // They are fixed for the whole kernel, so hoist them into registers once.
  const uint32_t sm_a = S_m[0], pm_a = P_m[0], km_a = dK_m[0],
                 vm_a = dV_m[0];

  tl::tcgen05_after_thread_sync();

  if (tid < 256) {
    // ==================== element warps ====================
    float f[32];
    uint4 pk[8];
    bfloat16_t cast8[8];
    const int qrow = tid & 127;

    // K(kt) and V(kt) are this CTA's for the whole run; Q(0) and dO(0) open the
    // pipeline at (head 0, q tile kt).
    if (tid == 0) {
      bar_ld0[0].expect_transaction(131072);
      tl::fence_proxy_async();
      tl::tma_load(K_desc, bar_ld0[0], ((bfloat16_t *)K_s), 0, (kt * 128), kv,
                   b);
      tl::tma_load(K_desc, bar_ld0[0], ((bfloat16_t *)K_s) + 8192, 64,
                   (kt * 128), kv, b);
      tl::tma_load(V_desc, bar_ld0[0], ((bfloat16_t *)V_s), 0, (kt * 128), kv,
                   b);
      tl::tma_load(V_desc, bar_ld0[0], ((bfloat16_t *)V_s) + 8192, 64,
                   (kt * 128), kv, b);
      tl::tma_load(Q_desc, bar_ld0[0], ((bfloat16_t *)Q_s), 0, (kt * 128),
                   (kv * GROUP), b);
      tl::tma_load(Q_desc, bar_ld0[0], ((bfloat16_t *)Q_s) + 8192, 64,
                   (kt * 128), (kv * GROUP), b);
      tl::tma_load(dO_desc, bar_ld0[0], ((bfloat16_t *)dO_s), 0, (kt * 128),
                   (kv * GROUP), b);
      tl::tma_load(dO_desc, bar_ld0[0], ((bfloat16_t *)dO_s) + 8192, 64,
                   (kt * 128), (kv * GROUP), b);
      bar_ld0[0].arrive();
    }

    // lse and delta are per (head, q tile) now, so they move with the walk: one
    // LDG each per iteration, issued a full iteration ahead of their use.
    const int hbase = (b * 65536) + (kv * GROUP * 4096);
    float l_cur = LSE[hbase + (kt * 128) + qrow] * LOG2E;
    float dls_cur = Delta[hbase + (kt * 128) + qrow] * SCALE;

    tl::tcgen05_before_thread_sync();
    tl::__sync_thread_partial(3, 256);

    // Peel iteration 0's tail: drain S(0), exp2, publish P(0).  q tile kt with
    // head 0 is on the diagonal, so this one is masked.
    bar_s[0].wait(0);
    tl::tcgen05_after_thread_sync();
    {
      const float l = l_cur;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int c0 = ((tid >> 7) * 64) + (half * 32);
        tl::tcgen05_ld_32dp32bNx<32, false>(sm_a, c0, (&(f[0])));
        tl::tcgen05_before_thread_sync();
#pragma unroll
        for (int i = 0; i < 32; ++i) {
          float e = exp2f((f[i] * SCALE2) - l);
          f[i] = (c0 + i <= qrow) ? e : 0.0f;
        }
        STS_TILE_H(P_s, f, 0, half * 4)
      }
    }
    tl::tcgen05_before_thread_sync();
    tl::__sync_thread_partial(3, 256);
    if (tid == 0) bar_sd[0].arrive();
    bar_pst[0].arrive();

    int hh = 0, qi = kt;
    for (int it = 0; it < nit; ++it) {
      const int par = it & 1;
      // (head, q tile) of iteration it+1, clamped on the last iteration.
      int qi_n = qi + 1, hh_n = hh;
      if (qi_n == NQT) { qi_n = kt; hh_n = hh + 1; }
      if (hh_n == GROUP) { qi_n = kt; hh_n = GROUP - 1; }
      const int h_n = (kv * GROUP) + hh_n;

      // ---- Q(it+1) into the slot dK(it-1) released ----
      if (tid == 0) {
        if (it > 0) bar_k[0].wait(1 - par);
        bar_ldq[0].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(Q_desc, bar_ldq[0],
                     ((bfloat16_t *)Q_s) + ((1 - par) * 16384), 0, (qi_n * 128),
                     h_n, b);
        tl::tma_load(Q_desc, bar_ldq[0],
                     ((bfloat16_t *)Q_s) + ((1 - par) * 16384) + 8192, 64,
                     (qi_n * 128), h_n, b);
        bar_ldq[0].arrive();
      }

      // Issued here, consumed two phases down: covers the LDG latency.
      const float l_nx = LSE[(b * 65536) + (h_n * 4096) + (qi_n * 128) + qrow] *
                         LOG2E;
      const float dls_nx =
          Delta[(b * 65536) + (h_n * 4096) + (qi_n * 128) + qrow] * SCALE;

      // ---- dS = P * (dP - delta) * scale ----
      bar_p[0].wait(par);
      tl::tcgen05_after_thread_sync();
      {
        const float dls = dls_cur;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          tl::tcgen05_ld_32dp32bNx<32, false>(
              pm_a, ((tid >> 7) * 64) + (half * 32), (&(f[0])));
          tl::tcgen05_before_thread_sync();
          if (half == 1) {
            tl::__sync_thread_partial(3, 256);
            if (tid == 0) bar_pd[0].arrive();
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            uint4 pu = pk[(half * 4) + i];
            uint4 ou;
#pragma unroll
            for (int vec = 0; vec < 4; ++vec) {
              float2 d = *(float2 *)(f + (i * 8) + (vec * 2));
              float2 t;
              t.x = __fmaf_rn(d.x, SCALE, -dls);
              t.y = __fmaf_rn(d.y, SCALE, -dls);
              (reinterpret_cast<__nv_bfloat162 *>(&ou))[vec] =
                  __hmul2(__float22bfloat162_rn(t),
                          (reinterpret_cast<__nv_bfloat162 *>(&pu))[vec]);
            }
            *(uint4 *)(((bfloat16_t *)dS_s) + PSW(tid, (half * 4) + i)) = ou;
          }
        }
      }
      tl::tcgen05_before_thread_sync();
      tl::__sync_thread_partial(3, 256);
      if (tid == 0) bar_dst[0].arrive();

      // ---- dO(it+1): dV(it) was dO_s's only reader, and three GEMMs separate
      // that from dP(it+1), so one slot is enough.
      if (tid == 0) {
        bar_v[0].wait(par);
        bar_ldo[0].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(dO_desc, bar_ldo[0], ((bfloat16_t *)dO_s), 0, (qi_n * 128),
                     h_n, b);
        tl::tma_load(dO_desc, bar_ldo[0], ((bfloat16_t *)dO_s) + 8192, 64,
                     (qi_n * 128), h_n, b);
        bar_ldo[0].arrive();
      }

      // ---- P(it+1) = exp2(S(it+1)) ----
      bar_s[0].wait(1 - par);
      tl::tcgen05_after_thread_sync();
      {
        const float l = l_nx;
        // One tile per head straddles the diagonal (q tile == kt, the first of
        // every head's walk), so unlike hand63 it cannot all be peeled -- but
        // the predicate is CTA-uniform, so it costs a branch, not 64 SEL.
        const bool diag = (qi_n == kt);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int c0 = ((tid >> 7) * 64) + (half * 32);
          tl::tcgen05_ld_32dp32bNx<32, false>(sm_a, c0, (&(f[0])));
          tl::tcgen05_before_thread_sync();
          if (half == 1) {
            tl::__sync_thread_partial(3, 256);
            if (tid == 0) bar_sd[0].arrive();
          }
          if (diag) {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              float e = exp2f((f[i] * SCALE2) - l);
              f[i] = (c0 + i <= qrow) ? e : 0.0f;
            }
          } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) f[i] = exp2f((f[i] * SCALE2) - l);
          }
          STS_TILE_H(P_s, f, 0, half * 4)
        }
      }
      tl::tcgen05_before_thread_sync();
      bar_pst[0].arrive();

      l_cur = l_nx;
      dls_cur = dls_nx;
      qi = qi_n;
      hh = hh_n;
    }
  } else {
    // ====== warps 8..15 drain dQ and reduce; warp 16 issues every MMA ======
    const bool issuer = (tid >= 512);
    const int row = (tid - 256) & 127;
    const int col = ((tid - 256) >> 7) * 64;
    float o_f[64];
    tl::Tcgen05SMemDescriptor q_a, k_a, o_a, v_a, p_a, o_b, ds_b, q_b, ds_a,
        k_b;
    tl::initialize_tcgen05_descriptor(q_a, (&(((bfloat16_t *)Q_s)[0])), 1, 64, 0,
                                      0, 2);  // S:A  = Q[slot]
    tl::initialize_tcgen05_descriptor(k_a, (&(((bfloat16_t *)K_s)[0])), 1, 64, 0,
                                      0, 2);  // S:B  = K
    tl::initialize_tcgen05_descriptor(o_a, (&(((bfloat16_t *)dO_s)[0])), 1, 64,
                                      0, 0, 2);  // dP:A = dO
    tl::initialize_tcgen05_descriptor(v_a, (&(((bfloat16_t *)V_s)[0])), 1, 64, 0,
                                      0, 2);  // dP:B = V
    tl::initialize_tcgen05_descriptor(ds_a, (&(((bfloat16_t *)dS_s)[0])), 1, 64,
                                      0, 0, 2);  // dQ:A = dS
    tl::initialize_tcgen05_descriptor(p_a, (&(((bfloat16_t *)P_s)[0])), 1024, 64,
                                      0, 0, 2);  // dV:A = P^T
    tl::initialize_tcgen05_descriptor(o_b, (&(((bfloat16_t *)dO_s)[0])), 1024,
                                      64, 0, 0, 2);  // dV:B = dO
    tl::initialize_tcgen05_descriptor(ds_b, (&(((bfloat16_t *)dS_s)[0])), 1024,
                                      64, 0, 0, 2);  // dK:A = dS^T
    tl::initialize_tcgen05_descriptor(q_b, (&(((bfloat16_t *)Q_s)[0])), 1024, 64,
                                      0, 0, 2);  // dK:B = Q[slot]
    tl::initialize_tcgen05_descriptor(k_b, (&(((bfloat16_t *)K_s)[0])), 1024, 64,
                                      0, 0, 2);  // dQ:B = K

    if (issuer) {
      bar_ld0[0].wait(0);
      tl::tcgen05_after_thread_sync();
      tl::fence_proxy_async();
#pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
            uint64_t(q_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
            uint64_t(k_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
            sm_a + 0, (0 < ki) ? 1 : 0,
            static_cast<uint32_t>(136316048), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(bar_s[0])));
#pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
            uint64_t(o_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
            uint64_t(v_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
            pm_a + 0, (0 < ki) ? 1 : 0,
            static_cast<uint32_t>(136316048), 0, 0, 0, 0);
      }
      tl::tcgen05_mma_arrive((&(bar_p[0])));

      // Issuer and drain used to share one loop body behind `if (issuer)`,
      // which made ptxas rematerialise SR_TID.X and re-test it on every
      // iteration.  Two branch-free loops cost nothing in code size.
      for (int it = 0; it < nit; ++it) {
        const int par = it & 1;
        const int koff = par * 32768;  // byte offset of Q(it)'s slot
        const int noff = (1 - par) * 32768;
        // ---- dV += P^T . dO -- resident across every head and q tile ----
        bar_pst[0].wait(par);
        tl::tcgen05_after_thread_sync();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
              uint64_t(p_a + (ki * 2048)), uint64_t(o_b + (ki * 2048)),
              vm_a + 0,
              ((0 < ki) || (it > 0)) ? 1 : 0,
              static_cast<uint32_t>(136414352), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(bar_v[0])));

        // ---- S(it+1) into the freed S_m ----
        bar_ldq[0].wait(par);
        tl::tcgen05_after_thread_sync();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
              uint64_t(q_a + noff + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
              uint64_t(k_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
              sm_a + 0, (0 < ki) ? 1 : 0,
              static_cast<uint32_t>(136316048), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(bar_s[0])));

        // ---- dQ into P_m, once the element warps have read dP(it) out ----
        bar_dst[0].wait(par);
        tl::tcgen05_after_thread_sync();
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int ki = 0; ki < 8; ++ki) {
            tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
                uint64_t(ds_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
                uint64_t(k_b + (half * 16384) + (ki * 2048)),
                pm_a + (half * 64),
                (0 < ki) ? 1 : 0, static_cast<uint32_t>(135333008), 0, 0, 0, 0);
          }
          tl::tcgen05_mma_arrive(half ? (&(bar_q2[0])) : (&(bar_q[0])));
        }

        // ---- dK += dS^T . Q -- also resident; releases Q(it)'s slot ----
        bar_dst[0].wait(par);
        tl::tcgen05_after_thread_sync();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
              uint64_t(ds_b + (ki * 2048)), uint64_t(q_b + koff + (ki * 2048)),
              km_a + 0,
              ((0 < ki) || (it > 0)) ? 1 : 0,
              static_cast<uint32_t>(136414352), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(bar_k[0])));

        // ---- dP(it+1) once the drain warps have read dQ(it) out ----
        bar_qd[0].wait(par);
        bar_qd2[0].wait(par);
        bar_ldo[0].wait(par);
        tl::tcgen05_after_thread_sync();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
          tl::tcgen05mma_ss<tl::DataType::kBFloat16, false>(
              uint64_t(o_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
              uint64_t(v_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))),
              pm_a + 0, (0 < ki) ? 1 : 0,
              static_cast<uint32_t>(136316048), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(bar_p[0])));
      }
      return;
    }

    // dst walks the q tiles with the loop; recomputing it meant a 64-bit
    // multiply-add chain on every iteration in all eight drain warps.
    bfloat16_t *dst = dQ + (b * 8388608) + ((kv * GROUP) * 524288) +
                      ((col >> 3) * 32768) + ((kt * 128 + row) * 8);
    const int wrap = 524288 - ((nq - 1) * 1024);
    int qi = kt;
    for (int it = 0; it < nit; ++it) {
      const int par = it & 1;
      {
        // ---- the only reduction in the kernel: dQ, 2.21 GB, bf16, straight
        // into the final (B,N,H,D) layout.  Release P_m before reducing so the
        // L2 traffic runs underneath the next MMA group.
        // One 64-column read: the path from dQ(it) to bar_qd -- which is all
        // that gates dP(it+1) -- is a single ld plus its wait, with every red
        // behind the arrive instead of wedged between two halves of the load.
        // h is kept incrementally: it / nq is a runtime divide, and its
          // software expansion was the single largest stall in the kernel.
        Barrier *bq = col ? bar_q2 : bar_q;
        Barrier *bqd = col ? bar_qd2 : bar_qd;
        bq[0].wait(par);
        tl::tcgen05_after_thread_sync();
        tl::tcgen05_ld_32dp32bNx<64, false>(pm_a, col, (&(o_f[0])));
        tl::tcgen05_before_thread_sync();
        bqd[0].arrive();
        // dQ reduces into (B,H,D/8,N,8), not its final (B,N,H,D): a warp's
        // 32 lanes are 32 consecutive q rows, so in the final layout they
        // would land 4096 B apart and each red would split into 32 separate
        // L2 transactions.  Here they are adjacent 16 B slices that coalesce
        // into 512 B.  Measured: 1.483 ms scattered vs 1.262 ms coalesced.
        // hand224_post transposes the result back, for 50 us.
        // The reds drain under the next iteration's bar_q wait.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          RED_BF16X2((dst + (i * 32768)), (o_f + (i * 8)))
        }
      }
      if (++qi == NQT) { qi -= nq; dst += wrap; } else { dst += 1024; }  // == kt, without rematerialising blockIdx.y
    }

    // dK and dV never left TMEM and this CTA is their only writer: one drain,
    // one bf16 store each, no accumulator, no memset, no cast.
    const int last = (nit - 1) & 1;
    bar_v[0].wait(last);
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_ld_32dp32bNx<64, false>(vm_a, col, (&(o_f[0])));
    bfloat16_t *vdst =
        dV + (b * 1048576) + ((kt * 128 + row) * 256) + (kv * 128) + col;
    {
      bfloat16_t cast16[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
          uint2 u;
          float4 v = *(float4 *)(o_f + (i * 16) + (vec * 4));
          (reinterpret_cast<__nv_bfloat162 *>(&u))[0] =
              __float22bfloat162_rn(((float2 *)(&v))[0]);
          (reinterpret_cast<__nv_bfloat162 *>(&u))[1] =
              __float22bfloat162_rn(((float2 *)(&v))[1]);
          *(uint2 *)(cast16 + (vec * 4)) = u;
        }
        tl::store_global_256(&(*(ulonglong4 *)(vdst + (i * 16))),
                             *(ulonglong4 *)(cast16));
      }
    }
    bar_k[0].wait(last);
    tl::tcgen05_after_thread_sync();
    tl::tcgen05_ld_32dp32bNx<64, false>(km_a, col, (&(o_f[0])));
    bfloat16_t *kdst =
        dK + (b * 1048576) + ((kt * 128 + row) * 256) + (kv * 128) + col;
    {
      bfloat16_t cast16[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int vec = 0; vec < 4; ++vec) {
          uint2 u;
          float4 v = *(float4 *)(o_f + (i * 16) + (vec * 4));
          (reinterpret_cast<__nv_bfloat162 *>(&u))[0] =
              __float22bfloat162_rn(((float2 *)(&v))[0]);
          (reinterpret_cast<__nv_bfloat162 *>(&u))[1] =
              __float22bfloat162_rn(((float2 *)(&v))[1]);
          *(uint2 *)(cast16 + (vec * 4)) = u;
        }
        tl::store_global_256(&(*(ulonglong4 *)(kdst + (i * 16))),
                             *(ulonglong4 *)(cast16));
      }
    }
  }

  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((tid >> 5) == 0) {
    tl::tmem_deallocate((&(S_m[0])), 128);
    tl::tmem_deallocate((&(P_m[0])), 128);
    tl::tmem_deallocate((&(dK_m[0])), 128);
    tl::tmem_deallocate((&(dV_m[0])), 128);
  }
}


// acc (B,H,D/8,N,8) -> dQ (B,N,H,D).  One block per (b, h, q tile): reads 16
// contiguous 2 KB runs, writes 128 contiguous 256 B rows, with SMEM carrying
// the transpose so neither side scatters.  Row pitch 136 keeps the 16 B accesses aligned and spreads consecutive
// threads land on consecutive banks on the write side.
extern "C" __global__ void __launch_bounds__(256)
    hand224_post(const bfloat16_t *__restrict__ acc,
                bfloat16_t *__restrict__ dQ) {
  __shared__ bfloat16_t sm[128 * 136];
  const int blk = (int)blockIdx.x;  // (b * 16 + h) * 32 + q tile
  const int qi = blk & 31;
  const int h = (blk >> 5) & 15;
  const int b = blk >> 9;
  const bfloat16_t *src =
      acc + ((long)b * 8388608) + (h * 524288) + (qi * 1024);
  const int t = (int)threadIdx.x;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int idx = (j * 256) + t;
    const int d8 = idx >> 7, nl = idx & 127;
    *(uint4 *)(sm + (nl * 136) + (d8 * 8)) =
        *(const uint4 *)(src + (d8 * 32768) + (nl * 8));
  }
  __syncthreads();
  bfloat16_t *dst = dQ + ((long)b * 8388608) + ((long)qi * 262144) + (h * 128);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int idx = (j * 256) + t;
    const int nl = idx >> 4, c16 = idx & 15;
    *(uint4 *)(dst + (nl * 2048) + (c16 * 8)) =
        *(const uint4 *)(sm + (nl * 136) + (c16 * 8));
  }
}

// Q/dO are (B=8, N=4096, H=16, D=128); K/V are (B=8, N=4096, H_KV=2, D=128).
static CUresult encode_map(CUtensorMap *map, const void *ptr, int nh) {
  uint64_t gdim[4] = {128, 4096, (uint64_t)nh, 8};
  uint64_t gstr[3] = {(uint64_t)nh * 256, 256, (uint64_t)nh * 1048576};
  uint32_t bdim[4] = {64, 128, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void *>(ptr), gdim,
      gstr, bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern "C" int hand224_launch(const bfloat16_t *Out, const bfloat16_t *K,
                             const float *LSE, const bfloat16_t *Q,
                             const bfloat16_t *V, bfloat16_t *dK,
                             const bfloat16_t *dO, bfloat16_t *dQo,
                             bfloat16_t *dV, float *Delta,
                             bfloat16_t *dQacc, int gx, int gy, void *stream) {
  if (gy != NQT) return -2;
  static bool once = false;
  if (!once) {
    cudaError_t e = cudaFuncSetAttribute(
        hand224_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return (int)e;
    once = true;
  }
  static const void *q_c = nullptr, *o_c = nullptr, *k_c = nullptr,
                    *v_c = nullptr;
  static CUtensorMap q_map, o_map, k_map, v_map;
  if (Q != q_c) {
    CUresult r = encode_map(&q_map, Q, 16);
    if (r != CUDA_SUCCESS) return -(int)r;
    q_c = Q;
  }
  if (dO != o_c) {
    CUresult r = encode_map(&o_map, dO, 16);
    if (r != CUDA_SUCCESS) return -(int)r;
    o_c = dO;
  }
  if (K != k_c) {
    CUresult r = encode_map(&k_map, K, 2);
    if (r != CUDA_SUCCESS) return -(int)r;
    k_c = K;
  }
  if (V != v_c) {
    CUresult r = encode_map(&v_map, V, 2);
    if (r != CUDA_SUCCESS) return -(int)r;
    v_c = V;
  }
  // 8 * 4096 * 16 groups of 16 lanes, 16 groups per 256-thread block.
  hand224_pre<<<32768, 256, 0, (cudaStream_t)stream>>>(Out, dO, Delta, dQacc);
  hand224_kernel<<<dim3(gx, gy), NTH, SMEM_BYTES, (cudaStream_t)stream>>>(
      k_map, LSE, q_map, v_map, dK, o_map, dQacc, dV, Delta);
  hand224_post<<<4096, 256, 0, (cudaStream_t)stream>>>(dQacc, dQo);
  return (int)cudaGetLastError();
}
