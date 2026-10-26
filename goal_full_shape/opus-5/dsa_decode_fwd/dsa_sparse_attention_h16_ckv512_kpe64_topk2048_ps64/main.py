import os, hashlib, tempfile
_CU_SRC = r"""// DSA sparse MLA decode (h16, ckv512, kpe64, page_size 64, topk 2048).
// Written from scratch. Single fused kernel:
//   * per-CTA on-device compaction of the sparse index list (so only valid KV
//     rows are ever touched),
//   * a speculative slot-ordered row gather that starts before the compaction
//     scan finishes and is verified by it (fallback for rows whose -1 padding
//     is not a suffix),
//   * adaptive split of the compacted key list across CTAs,
//   * bf16 tensor-core (mma.m16n8k16) QK^T and PV out of padded shared memory
//     via ldmatrix,
//   * tokens that fit in one slice write their final output directly, otherwise
//     an in-kernel combine whose {generation, lse} release word is both the
//     completion flag and the value it needs finishes the softmax.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define DEV __device__ __forceinline__

namespace {

constexpr int HH = 16;    // heads
constexpr int DC = 512;   // ckv dim
constexpr int DR = 64;    // kpe dim
constexpr int DK = DC + DR;  // 576: ckv and kpe share one padded row
constexpr int NTH = 256;  // threads per CTA
constexpr int NW = 8;     // warps per CTA
constexpr int IPT = 8;    // index entries held per thread (topk <= NTH*IPT)
constexpr int MAXNC = 64;
constexpr int SPF = 640;  // floats reserved for the bf16 P staging
constexpr int CLU = 2;    // CTAs per cluster (scheduling locality; grid must divide)

// Shared-memory rows are padded by 8 bf16 (16B) so that the 8 (or 16) rows an
// ldmatrix reads land in different bank groups - no XOR swizzle maths needed.
template <int NCOL>
DEV constexpr int lds() { return NCOL + 8; }
template <int NCOL>
DEV int sof(int row, int col) { return (row * lds<NCOL>() + col) * 2; }
DEV uint32_t sptr(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

DEV void mma1688(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV void ldm_x4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEV void ldm_x4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEV void cpa16(uint32_t dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
DEV int4 ldg4(const void *p) {
  int4 r;
  asm volatile("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
DEV int ldg1(const void *p) {
  int r;
  asm volatile("ld.global.nc.s32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
DEV void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
DEV void fence_rel() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
DEV void st_release64(unsigned long long *p, unsigned long long v) {
  asm volatile("st.global.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
DEV unsigned long long ld_acquire64(const unsigned long long *p) {
  unsigned long long r;
  asm volatile("ld.global.acquire.gpu.u64 %0, [%1];\n" : "=l"(r) : "l"(p) : "memory");
  return r;
}
DEV void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

DEV float red4_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
DEV float red4_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}


// One depth-group of the QK gemm: [LO,HI) k-steps x 2 key n-groups.
// A single ldmatrix.x4 supplies the B fragments of BOTH n-groups, and NACC
// independent accumulators keep the tensor core pipeline full.
#define NACC 1
template <int LO, int HI, int NGW>
DEV void qk_dg(float (&pa)[NGW][NACC][4], uint32_t pQ, uint32_t pK, int arow, int acol,
               int kbase, int ngmax) {
  constexpr int NK4 = (NGW + 1) / 2;   // ldmatrix.x4 per k-step for the K operand
  uint32_t qa[3][4], kb[3][NK4][4];
#pragma unroll
  for (int st = LO; st <= HI + 1; ++st) {
    if (st < HI) {
      const int cur = st % 3;
      ldm_x4(qa[cur], pQ + sof<DK>(arow, st * 16 + acol));
#pragma unroll
      for (int q = 0; q < NK4; ++q)
        if (NK4 == 1 || 2 * q < ngmax)
          ldm_x4(kb[cur][q], pK + sof<DK>(kbase + q * 16 + arow, st * 16 + acol));
    }
    if (st >= LO + 2) {
      const int pv = (st - 2) % 3;
      const int acc = (st - 2) % NACC;
#pragma unroll
      for (int q = 0; q < NK4; ++q) {
        if (2 * q < ngmax) {
          uint32_t b0[2] = {kb[pv][q][0], kb[pv][q][2]};
          mma1688(pa[2 * q][acc], qa[pv], b0);
        }
        if (NGW > 2 * q + 1 && 2 * q + 1 < ngmax) {
          uint32_t b1[2] = {kb[pv][q][1], kb[pv][q][3]};
          mma1688(pa[NGW > 2 * q + 1 ? 2 * q + 1 : 0][acc], qa[pv], b1);
        }
      }
    }
  }
}

// ===========================================================================
template <int CS, int KGSZ, int NTHR, int SIMPLE, int FULLROW, int MINB>
__global__ __cluster_dims__(CLU, 1, 1) __launch_bounds__(NTHR, MINB) void dsa_kernel(
    const __nv_bfloat16 *__restrict__ Qn, const __nv_bfloat16 *__restrict__ Qp,
    const __nv_bfloat16 *__restrict__ CKV, const __nv_bfloat16 *__restrict__ KPE,
    const int *__restrict__ IDX, __nv_bfloat16 *__restrict__ OUT,
    float *__restrict__ LSE, __nv_bfloat16 *__restrict__ PO,
    unsigned long long *__restrict__ PMG, unsigned int gen, float sm_scale2, int T,
    int topk, int nchunk, int nclog2, int nkvtok) {
  constexpr int NTH = NTHR;            // shadow the file-scope defaults
  constexpr int NW = NTHR / 32;
  constexpr int IPT = 2048 / NTHR;
  constexpr int NKG = CS / KGSZ;      // key groups for QK
  constexpr int NGW = KGSZ / 8;       // n-groups per warp in phase 1
  constexpr int DW = DC / NW;         // dims per warp for PV (64)
  constexpr int NGO = DW / 8;         // PV n-groups per warp (8)

  extern __shared__ char smem[];
  __nv_bfloat16 *sK = (__nv_bfloat16 *)smem;      // [CS][576+8] ckv | kpe
  __nv_bfloat16 *sQ = sK + CS * lds<DK>();        // [16][576+8] q_nope | q_pe
  int *sIdx = (int *)(sQ + HH * lds<DK>());
  int *sScan = sIdx + CS;                   // [3*NW+4]
  float *sRed = (float *)(sScan + 3 * NW + 4);   // [NW*16+16]
  float *sRedL = sRed + NW * 16 + 16;           // [NW*16+16]
  float *sS = sRedL + NW * 16 + 16;             // [SPF] (only the P staging)
  __nv_bfloat16 *sP = (__nv_bfloat16 *)sS;  // overlaid (written after sS is consumed)
  float *sM = sS + SPF;                     // [MAXNC]
  float *sL = sM + MAXNC;                   // [MAXNC]

  const uint32_t pK = sptr(sK), pQ = sptr(sQ), pP = sptr(sP);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gsz = gridDim.x;

  const int arow = (lane & 7) | (((lane >> 3) & 1) << 3);
  const int acol = ((lane >> 4) & 1) << 3;
  const int r0 = lane >> 2;
  const int kcol = (lane & 3) * 2;

  float acc[NGO][4];
  int cur_tok = -1;
  const int nctot = T * nchunk;
  // combine unit of this CTA: with SIMPLE the mapping is chosen so that the
  // combine token equals this CTA's own chunk token (no second index scan).
  // combine unit of this CTA.  PARTS = 2 means a CTA owns half of one head's output
  // row, so twice as many CTAs share the combine work (half the bytes each).
  constexpr int PARTS =
      (SIMPLE && FULLROW && (2048 / CS) >= 2 * HH) ? 2 : 1;  // !SIMPLE combines whole rows
  constexpr int PL = PARTS == 2 ? 1 : 0;
  constexpr int DPART = DC / PARTS;
  const int ct = SIMPLE ? (int)(blockIdx.x >> nclog2) : (int)(blockIdx.x >> 4);
  const int cj = SIMPLE ? (int)(blockIdx.x & (nchunk - 1)) : (int)(blockIdx.x & 15);
  const int ch = SIMPLE ? (cj >> PL) : cj;
  const int cpart = SIMPLE ? (cj & (PARTS - 1)) : 0;
  const bool do_comb = SIMPLE ? (ct < T && (cj >> PL) < HH) : ((int)blockIdx.x < T * HH);
  int t_first = ((int)blockIdx.x < nctot) ? (int)(blockIdx.x >> nclog2) : -1;
  // this CTA's own chunk of slots, requested before the full row so that the
  // slot-ordered gather does not have to wait for the prefix scan
  const int j0 = (int)(blockIdx.x & (nchunk - 1));
  int sp0 = -1, sp1 = -1;
  if (t_first >= 0) {
    const int *ip = IDX + (size_t)t_first * topk + j0 * CS;
    const int s0 = j0 * CS + lane;
    sp0 = (s0 < topk) ? ldg1(ip + lane) : -1;
    if (CS > 32) sp1 = (s0 + 32 < topk) ? ldg1(ip + 32 + lane) : -1;
  }

  // --- read index rows first: the prefix scan is the kernel's critical path, so
  // the index requests must not queue behind the 18 KB of Q prefetch traffic ---
  int v[IPT], u[IPT];
  int cnt0 = 0, cnt1 = 0;
  if (t_first >= 0) {
#pragma unroll
    for (int i = 0; i < IPT; i += 4) {
      int p = tid * IPT + i;
      if (FULLROW) {
        int4 q = ldg4(IDX + (size_t)t_first * topk + p);
        v[i] = q.x; v[i + 1] = q.y; v[i + 2] = q.z; v[i + 3] = q.w;
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          v[i + k] = (p + k < topk) ? ldg1(IDX + (size_t)t_first * topk + p + k) : -1;
      }

    }
  }
  // --- Q prefetch, issued behind the index loads (Q is only needed at the QK) ---
  if (t_first >= 0) {
    const __nv_bfloat16 *qn = Qn + (size_t)t_first * HH * DC;
    for (int i = tid; i < HH * 64; i += NTH) {
      int row = i >> 6, g = i & 63;
      cpa16(pQ + (uint32_t)sof<DK>(row, g * 8), qn + row * DC + g * 8);
    }
    const __nv_bfloat16 *qp = Qp + (size_t)t_first * HH * DR;
    for (int i = tid; i < HH * 8; i += NTH) {
      int row = i >> 3, g = i & 7;
      cpa16(pQ + (uint32_t)sof<DK>(row, DC + g * 8), qp + row * DR + g * 8);
    }
    cur_tok = t_first;
  }
  // --- slot-ordered gather of this chunk's rows: right whenever the row's -1
  // padding is a suffix, which the scan below verifies ---
  if (t_first >= 0) {
    constexpr int RPW = CS / NW;                 // rows per warp
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
      const int row = warp * RPW + rr;
      const int ix = __shfl_sync(0xffffffffu, (row < 32) ? sp0 : sp1, row & 31);
      if (ix >= 0) {
        const int w = (unsigned)ix < (unsigned)nkvtok ? ix : nkvtok - 1;
#pragma unroll
        for (int h = 0; h < DC / 256; ++h)
          cpa16(pK + (uint32_t)sof<DK>(row, h * 256 + lane * 8),
                CKV + (size_t)w * DC + h * 256 + lane * 8);
      }
    }
    // the 128 B kpe halves: four rows per instruction, eight lanes each
    constexpr int KRW = 256 / DR;                // kpe rows per instruction
#pragma unroll
    for (int rr = 0; rr < RPW; rr += KRW) {
      const int row = warp * RPW + rr + (lane >> 3);
      const int ix = __shfl_sync(0xffffffffu, (row < 32) ? sp0 : sp1, row & 31);
      if (ix >= 0) {
        const int w = (unsigned)ix < (unsigned)nkvtok ? ix : nkvtok - 1;
        cpa16(pK + (uint32_t)sof<DK>(row, DC + (lane & 7) * 8),
              KPE + (size_t)w * DR + (lane & 7) * 8);
      }
    }
  }
  int mymax = -1;
  if (t_first >= 0) {
#pragma unroll
    for (int i = 0; i < IPT; ++i)
      if (v[i] >= 0) { ++cnt0; mymax = tid * IPT + i; }
  }
  const bool need_u = (!SIMPLE) && do_comb && (ct != t_first);
  if (need_u) {
#pragma unroll
    for (int i = 0; i < IPT; i += 4) {
      int p = tid * IPT + i;
      if (FULLROW) {
        int4 q = *(const int4 *)(IDX + (size_t)ct * topk + p);
        u[i] = q.x; u[i + 1] = q.y; u[i + 2] = q.z; u[i + 3] = q.w;
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          u[i + k] = (p + k < topk) ? IDX[(size_t)ct * topk + p + k] : -1;
      }

    }
#pragma unroll
    for (int i = 0; i < IPT; ++i) cnt1 += (u[i] >= 0);
  }
  // warp-local inclusive scan, then warp 0 turns the per-warp totals into
  // exclusive prefixes so every other warp only reads two words back.
  int texcl = cnt0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, texcl, o);
    if (lane >= o) texcl += y;
  }
  texcl -= cnt0;
  const unsigned wmax = __reduce_max_sync(0xffffffffu, (unsigned)(mymax + 1));
  if (lane == 31) sScan[warp] = texcl + cnt0;
  if (lane == 0) sScan[2 * NW + warp] = (int)wmax;
  if (!SIMPLE) {
    cnt1 = __reduce_add_sync(0xffffffffu, cnt1);
    if (lane == 0) sScan[NW + warp] = cnt1;
  }
  __syncthreads();
  int base0 = texcl, tot0 = 0, tot1 = 0, mxs = 0;
#pragma unroll
  for (int i = 0; i < NW; ++i) {
    int q = sScan[i];
    if (i < warp) base0 += q;
    tot0 += q;
    if (!SIMPLE) tot1 += sScan[NW + i];
    int m = sScan[2 * NW + i];
    if (m > mxs) mxs = m;
  }
  const int nval_ct = need_u ? tot1 : tot0;
  // the padding is a suffix exactly when the last valid slot is slot nval-1
  const bool dense = (mxs == tot0);

  for (int c = blockIdx.x; c < nctot; c += gsz) {
    const int t = c >> nclog2, j = c & (nchunk - 1);
    int nval, base;
    if (SIMPLE || t == t_first) {
      nval = tot0; base = base0;
    } else {
      // rare path (grid smaller than chunk count): re-read this token's row
#pragma unroll
      for (int i = 0; i < IPT; i += 4) {
        int p = tid * IPT + i;
        if (FULLROW) {
          int4 q = *(const int4 *)(IDX + (size_t)t * topk + p);
          v[i] = q.x; v[i + 1] = q.y; v[i + 2] = q.z; v[i + 3] = q.w;
        } else {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            v[i + k] = (p + k < topk) ? IDX[(size_t)t * topk + p + k] : -1;
        }

      }
      int cn = 0;
#pragma unroll
      for (int i = 0; i < IPT; ++i) cn += (v[i] >= 0);
      int tx = cn;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, tx, o);
        if (lane >= o) tx += y;
      }
      tx -= cn;
      __syncthreads();
      if (lane == 31) sScan[warp] = tx + cn;
      __syncthreads();
      base = 0; nval = 0;
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        int q = sScan[i];
        if (i < warp) base += q;
        nval += q;
      }
      base += tx;
      t_first = t; tot0 = nval; base0 = base;
    }
    constexpr int CSL = (CS == 32) ? 5 : ((CS == 64) ? 6 : 7);
    const int nactive = (nval + CS - 1) >> CSL;
    if (j >= nactive) continue;
    const int lo = j * CS;
    const int mykeys = (nval - lo < CS) ? (nval - lo) : CS;
    const int kst = (mykeys + 15) >> 4;
    const int kpad = kst << 4;

    const int hi = lo + mykeys;
    const bool spec = dense && (c == (int)blockIdx.x);
    if (!spec) {   // the slot order was wrong: compact and re-gather
      if (c == (int)blockIdx.x) cp_wait0();   // retire the speculative copies first
      if (base < hi && base + IPT > lo) {
        int pos = base;
#pragma unroll
        for (int i = 0; i < IPT; ++i) {
          if (v[i] >= 0) {
            if (pos >= lo && pos < hi) {
              int w = v[i] < nkvtok ? v[i] : nkvtok - 1;
              sIdx[pos - lo] = w;
            }
            ++pos;
          }
        }
      }
      if (!SIMPLE && t != cur_tok) {
        cur_tok = t;
        const __nv_bfloat16 *qn = Qn + (size_t)t * HH * DC;
        for (int i = tid; i < HH * 64; i += NTH) {
          int row = i >> 6, g = i & 63;
          cpa16(pQ + (uint32_t)sof<DK>(row, g * 8), qn + row * DC + g * 8);
        }
        const __nv_bfloat16 *qp = Qp + (size_t)t * HH * DR;
        for (int i = tid; i < HH * 8; i += NTH) {
          int row = i >> 3, g = i & 7;
          cpa16(pQ + (uint32_t)sof<DK>(row, DC + g * 8), qp + row * DR + g * 8);
        }
      }
      __syncthreads();

      // ---- gather only the valid rows (fallback: the row was not suffix-padded) ----
      // stride-NTH so each cp.async stays fully coalesced, 4 of them per iteration
      {
        const int g = tid & 63;
        for (int r = tid >> 6; r < mykeys; r += (NTH / 64) * 4) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int row = r + k * (NTH / 64);
            if (row < mykeys)
              cpa16(pK + (uint32_t)sof<DK>(row, g * 8),
                    CKV + (size_t)sIdx[row] * DC + g * 8);
          }
        }
        const int g2 = tid & 7;
        for (int r = tid >> 3; r < mykeys; r += (NTH / 8) * 4) {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int row = r + k * (NTH / 8);
            if (row < mykeys)
              cpa16(pK + (uint32_t)sof<DK>(row, DC + g2 * 8),
                    KPE + (size_t)sIdx[row] * DR + g2 * 8);
          }
        }
      }
    }
    // the padded rows the gather does not write: zeroed while it is in flight
    if (kpad > mykeys) {
      uint4 z = make_uint4(0, 0, 0, 0);
      constexpr int RU4 = lds<DK>() * 2 / 16;      // uint4 per padded key row
      for (int i = mykeys * RU4 + tid; i < kpad * RU4; i += NTH) ((uint4 *)sK)[i] = z;
    }
    cp_commit();
    cp_wait0();
    __syncthreads();

    // ---------------- QK ----------------
    // one warp per key group of KGSZ keys, each running all 36 k-steps: with no
    // depth split the partial logits never have to meet through shared memory
    const int kbase16 = warp * KGSZ;
    int ngmax = (mykeys - kbase16 + 7) >> 3;
    if (ngmax < 0) ngmax = 0;
    if (ngmax > NGW) ngmax = NGW;
    float sa[NGW][4];
    {
      float pa[NGW][NACC][4];
#pragma unroll
      for (int g = 0; g < NGW; ++g)
#pragma unroll
        for (int u = 0; u < NACC; ++u)
#pragma unroll
          for (int i = 0; i < 4; ++i) pa[g][u][i] = 0.f;
      if (ngmax > 0) qk_dg<0, 36, NGW>(pa, pQ, pK, arow, acol, kbase16, ngmax);
#pragma unroll
      for (int g = 0; g < NGW; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float v = 0.f;
#pragma unroll
          for (int u = 0; u < NACC; ++u) v += pa[g][u][i];
          sa[g][i] = v;
        }
    }
    const int nwa = (kpad + KGSZ - 1) / KGSZ;      // key groups that hold data
    const bool aw = (warp < nwa) && (warp < NKG);
    if (aw) {
      float u0 = -INFINITY, u1 = -INFINITY;
#pragma unroll
      for (int g = 0; g < NGW; ++g) {
        const int kk = kbase16 + g * 8 + kcol;
        bool vv0 = (kk < mykeys), vv1 = (kk + 1 < mykeys);
        float t0 = sa[g][0] * sm_scale2, t1 = sa[g][1] * sm_scale2;
        float t2 = sa[g][2] * sm_scale2, t3 = sa[g][3] * sm_scale2;
        sa[g][0] = vv0 ? t0 : -INFINITY;
        sa[g][1] = vv1 ? t1 : -INFINITY;
        sa[g][2] = vv0 ? t2 : -INFINITY;
        sa[g][3] = vv1 ? t3 : -INFINITY;
        u0 = fmaxf(u0, fmaxf(sa[g][0], sa[g][1]));
        u1 = fmaxf(u1, fmaxf(sa[g][2], sa[g][3]));
      }
      float mx0 = red4_max(u0), mx1 = red4_max(u1);
      if ((lane & 3) == 0) {
        sRed[warp * 16 + r0] = mx0;
        sRed[warp * 16 + r0 + 8] = mx1;
      }
    }
    __syncthreads();
    float m0 = -INFINITY, m1 = -INFINITY;
    for (int w = 0; w < nwa; ++w) {
      m0 = fmaxf(m0, sRed[w * 16 + r0]);
      m1 = fmaxf(m1, sRed[w * 16 + r0 + 8]);
    }
    if (aw) {
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int g = 0; g < NGW; ++g) {
        const int kk = kbase16 + g * 8 + kcol;
        float p0 = exp2f(sa[g][0] - m0), p1 = exp2f(sa[g][1] - m0);
        float p2 = exp2f(sa[g][2] - m1), p3 = exp2f(sa[g][3] - m1);
        __nv_bfloat162 q01 = __floats2bfloat162_rn(p0, p1);
        __nv_bfloat162 q23 = __floats2bfloat162_rn(p2, p3);
        *(uint32_t *)((char *)sP + sof<CS>(r0, kk)) = *(uint32_t *)&q01;
        *(uint32_t *)((char *)sP + sof<CS>(r0 + 8, kk)) = *(uint32_t *)&q23;
        s0 += p0 + p1;
        s1 += p2 + p3;
      }
      s0 = red4_sum(s0);
      s1 = red4_sum(s1);
      if ((lane & 3) == 0) {
        sRedL[warp * 16 + r0] = s0;
        sRedL[warp * 16 + r0 + 8] = s1;
      }
    }
    __syncthreads();

    // ---------------- PV ----------------
#pragma unroll
    for (int i = 0; i < NGO; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][k] = 0.f;
    const int dimw = warp * DW;
    for (int kk = 0; kk < kst; ++kk) {
      uint32_t a[4];
      ldm_x4(a, pP + sof<CS>(arow, kk * 16 + acol));
#pragma unroll
      for (int ng = 0; ng < NGO; ng += 2) {
        uint32_t r[4];
        ldm_x4t(r, pK + sof<DK>(kk * 16 + arow, dimw + ng * 8 + acol));
        uint32_t b0[2] = {r[0], r[1]};
        uint32_t b1[2] = {r[2], r[3]};
        mma1688(acc[ng], a, b0);
        mma1688(acc[ng + 1], a, b1);
      }
    }

    // the denominator fold only feeds the final scaling, so it rides behind the PV
    float l0 = 0.f, l1 = 0.f;
    for (int w = 0; w < nwa; ++w) {
      l0 += sRedL[w * 16 + r0];
      l1 += sRedL[w * 16 + r0 + 8];
    }

    // ---------------- emit ----------------
    // stage the normalised result through the warp's *own* column slice of the KV
    // buffer, then read it back warp-locally: 128 B per store, no barrier needed
    {
      float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
      for (int ng = 0; ng < NGO; ++ng) {
        int d = dimw + ng * 8 + kcol;
        __nv_bfloat162 uu = __floats2bfloat162_rn(acc[ng][0] * i0, acc[ng][1] * i0);
        __nv_bfloat162 ww = __floats2bfloat162_rn(acc[ng][2] * i1, acc[ng][3] * i1);
        *(uint32_t *)((char *)sK + sof<DK>(r0, d)) = *(uint32_t *)&uu;
        *(uint32_t *)((char *)sK + sof<DK>(r0 + 8, d)) = *(uint32_t *)&ww;
      }
    }
    __syncwarp();          // lanes read back what other lanes of this warp staged
    __nv_bfloat16 *dst = (nactive == 1) ? OUT + (size_t)t * HH * DC
                                        : PO + (size_t)c * (HH * DC);
#pragma unroll
    for (int row = 0; row < HH; row += 2) {
      const int rr = row + (lane >> 4);
      const int dd = dimw + (lane & 15) * 4;
      *(uint2 *)(dst + rr * DC + dd) = *(const uint2 *)((char *)sK + sof<DK>(rr, dd));
    }
    if (nactive == 1) {
      if (warp == 0 && (lane & 3) == 0) {
        LSE[t * HH + r0] = m0 + log2f(l0);
        LSE[t * HH + r0 + 8] = m1 + log2f(l1);
      }
    } else {
      // the {gen, lse} word is the completion flag *and* the chunk's lse, so the
      // combine gets both from the same request
      const unsigned long long g64 = (unsigned long long)gen << 32;
      const unsigned long long w0 = g64 | __float_as_uint(m0 + log2f(l0));
      const unsigned long long w1 = g64 | __float_as_uint(m1 + log2f(l1));
      __syncthreads();                 // every warp's partial row is in flight
      if (warp == 0 && (lane & 3) == 0) {
        fence_rel();                   // one release fence, then plain publications
        st_release64(PMG + ((size_t)t * HH + r0) * nchunk + j, w0);
        st_release64(PMG + ((size_t)t * HH + r0 + 8) * nchunk + j, w1);
      }
    }
  }

  // ---------------- combine ----------------
  // nval of this CTA's combine token is already known, so a CTA that has
  // nothing to combine skips the barrier and the drain as well as the loop
  constexpr int CSL1 = (CS == 32) ? 5 : ((CS == 64) ? 6 : 7);
  const bool skip_comb = SIMPLE && (!do_comb || ((nval_ct + CS - 1) >> CSL1) == 1);
  if (skip_comb) return;
  cp_wait0();       // no speculative copy may still be in flight
  __syncthreads();  // the combine reuses the KV staging as a reduction buffer
  for (int pp = SIMPLE ? (do_comb ? ct * HH + ch : T * HH) : (int)blockIdx.x; pp < T * HH;
       pp += (SIMPLE ? T * HH : gsz)) {
    const int t = SIMPLE ? ct : (pp >> 4), h = SIMPLE ? ch : (pp & 15);
    const int part = SIMPLE ? cpart : 0;
    int nval;
    if (SIMPLE || t == ct) nval = nval_ct;
    else {
      int cnt = 0;
#pragma unroll
      for (int i = 0; i < IPT; i += 4) {
        int p = tid * IPT + i;
        if (FULLROW) {
          int4 q = *(const int4 *)(IDX + (size_t)t * topk + p);
          cnt += (q.x >= 0) + (q.y >= 0) + (q.z >= 0) + (q.w >= 0);
        } else {
#pragma unroll
          for (int k = 0; k < 4; ++k)
            if (p + k < topk) cnt += (IDX[(size_t)t * topk + p + k] >= 0);
        }

      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
      __syncthreads();
      if (lane == 0) sScan[warp] = cnt;
      __syncthreads();
      nval = 0;
#pragma unroll
      for (int i = 0; i < NW; ++i) nval += sScan[i];
    }
    constexpr int CSL2 = (CS == 32) ? 5 : ((CS == 64) ? 6 : 7);
    const int nactive = (nval + CS - 1) >> CSL2;
    if (nactive == 1) continue;
    if (nactive == 0) {
      if (tid < 64) {
        uint4 z = make_uint4(0, 0, 0, 0);
        if (PARTS == 2) *(uint2 *)(OUT + (size_t)pp * DC + part * DPART + tid * 4) = *(uint2 *)&z;
        else *(uint4 *)(OUT + (size_t)pp * DC + tid * 8) = z;
        if (tid == 0 && part == 0) LSE[pp] = -INFINITY;
      }
      continue;
    }
    // one coalesced 256B poll for the whole token: lane jj watches chunk jj and
    // keeps the lse that arrives in the same word
    if (warp == 0) {
      const unsigned long long *pg = PMG + ((size_t)t * HH + h) * nchunk;
      unsigned long long w = 0;
      bool ok;
      int guard = 0;
      do {
        if (lane < nactive) w = ld_acquire64(pg + lane);
        unsigned f = (lane < nactive) ? (unsigned)(w >> 32) : gen;
        ok = __all_sync(0xffffffffu, f == gen);
        if (!ok) __nanosleep(20);
      } while (!ok && ++guard < (1 << 20));
      float ls = (lane < nactive) ? __uint_as_float((unsigned)w) : -INFINITY;
      if (lane < nactive) sM[lane] = ls;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ls = fmaxf(ls, __shfl_xor_sync(0xffffffffu, ls, o));
      if (lane == 0) sL[0] = ls;
    }
    __syncthreads();
    // thread -> (chunk group g of NGRP, dim block of 8)
    constexpr int NGRP = NTH / 64;
    const int g = tid >> 6, l64 = tid & 63;
    const __nv_bfloat16 *pb = PO + (size_t)(t * nchunk) * (HH * DC) + h * DC +
                              part * DPART + l64 * (DPART / 64);
    constexpr int DPT = DPART / 64;      // dims per thread
    // The partial rows and the per-chunk lse are two independent reads, so both
    // go in flight before either is consumed.  nchunk <= NGRP * 8 always, so one
    // batch of 8 covers every chunk of this token.
    uint2 raw2[8];
    uint4 raw[8];
    int jjs[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      int jj = g + k * NGRP;
      jjs[k] = jj;
      int jc = jj < nactive ? jj : nactive - 1;
      if (DPT == 4) raw2[k] = __ldcg((const uint2 *)(pb + (size_t)jc * (HH * DC)));
      else raw[k] = __ldcg((const uint4 *)(pb + (size_t)jc * (HH * DC)));
    }
    const float m = sL[0];
    float ow[DPT];
#pragma unroll
    for (int k = 0; k < DPT; ++k) ow[k] = 0.f;
    float lw = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (jjs[k] < nactive) {
        float w = exp2f(sM[jjs[k]] - m);
        lw += w;
        const __nv_bfloat162 *pv =
            (DPT == 4) ? (const __nv_bfloat162 *)&raw2[k] : (const __nv_bfloat162 *)&raw[k];
#pragma unroll
        for (int q = 0; q < DPT / 2; ++q) {
          float2 f = __bfloat1622float2(pv[q]);
          ow[2 * q] += w * f.x;
          ow[2 * q + 1] += w * f.y;
        }
      }
    // reduce the 4 chunk groups through smem
    {
      float *st = (float *)sK;             // NTH*DPT floats
      float *sl = st + NTH * DPT;          // NTH floats
#pragma unroll
      for (int k = 0; k < DPT; ++k) st[tid * DPT + k] = ow[k];
      sl[tid] = lw;
      __syncthreads();
      if (tid < 64) {
        float lt = 0.f;
        float ot[DPT];
#pragma unroll
        for (int k = 0; k < DPT; ++k) ot[k] = 0.f;
#pragma unroll
        for (int gg = 0; gg < NGRP; ++gg) {
          int src = gg * 64 + tid;
          lt += sl[src];
#pragma unroll
          for (int k = 0; k < DPT; ++k) ot[k] += st[src * DPT + k];
        }
        float inv = 1.f / lt;
        __nv_bfloat162 oo[DPT / 2];
#pragma unroll
        for (int q = 0; q < DPT / 2; ++q)
          oo[q] = __floats2bfloat162_rn(ot[2 * q] * inv, ot[2 * q + 1] * inv);
        __nv_bfloat16 *od = OUT + (size_t)pp * DC + part * DPART + tid * DPT;
        if (DPT == 4) *(uint2 *)od = *(const uint2 *)oo;
        else *(uint4 *)od = *(const uint4 *)oo;
        if (tid == 0 && part == 0) LSE[pp] = m + log2f(lt);
      }
      __syncthreads();
    }
  }
}

// ============================== host side ==============================
template <int CS, int KGSZ, int NTHR>
size_t smem_for() {
  constexpr int NGW = KGSZ / 8;
  constexpr int NWL = NTHR / 32;
  return (size_t)(CS * (DK + 8) + HH * (DK + 8)) * 2 +
         (size_t)(CS + 3 * NWL + 4 + 2 * NWL * 16 + 32 + SPF + 2 * MAXNC) * 4;
}

template <int CS, int KGSZ, int NTHR, int SIMPLE, int FULLROW, int NDG>
int occ_for() {
  int occ = 0;
  size_t sm = smem_for<CS, KGSZ, NTHR>();
  cudaFuncSetAttribute(dsa_kernel<CS, KGSZ, NTHR, SIMPLE, FULLROW, NDG>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, dsa_kernel<CS, KGSZ, NTHR, SIMPLE, FULLROW, NDG>, NTHR, sm);
  return occ < 1 ? 1 : occ;
}

}  // namespace

// (chunk size, keys per QK group, threads, SIMPLE, FULLROW, min blocks/SM).
// Only the two 256-thread shapes are instantiated: 512 threads measured 24 %
// slower and nothing selects it.
#define DISPATCH(FN)                                                      \
  if (simple && fullrow) FN(64, 16, 256, 1, 1, 2)                         \
  else if (fullrow) FN(64, 16, 256, 0, 1, 2)                              \
  else if (simple) FN(64, 16, 256, 1, 0, 2)                               \
  else FN(64, 16, 256, 0, 0, 2)

std::vector<int64_t> dsa_plan(int64_t cs, int64_t nth, int64_t T, int64_t topk) {
  int occ = 1, nthr = 256;
  size_t sm = 0;
  const bool simple = true, fullrow = (topk == 2048);
#define PLAN1(C, G, X, M, F, D) { occ = occ_for<C, G, X, M, F, D>(); sm = smem_for<C, G, X>(); nthr = X; }
  DISPATCH(PLAN1)
#undef PLAN1
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  int64_t cap = (int64_t)nsm * (occ > 4 ? 4 : occ);
  int64_t nchunk = 1;
  while (nchunk * cs < topk) nchunk *= 2;     // power of two so the maps are shifts
  int64_t want = T * nchunk;
  if (want < T * HH) want = T * HH;
  int64_t grid = want < cap ? want : cap;
  grid = (grid / CLU) * CLU;                  // the grid must divide by the cluster size
  if (grid < CLU) grid = CLU;
  return {grid, (int64_t)sm, occ, nthr};
}

void dsa_run(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv, torch::Tensor kpe,
             torch::Tensor idx, torch::Tensor out, torch::Tensor lse, torch::Tensor po,
             torch::Tensor pml, int64_t gen, double sm_scale2, int64_t cs,
             int64_t nth, int64_t grid) {
  const int T = (int)q_nope.size(0);
  const int topk = (int)idx.size(1);
  int64_t nchk = 1;
  while (nchk * cs < topk) nchk *= 2;
  const int *ix = (const int *)idx.data_ptr();
  // the 16 B index-row load needs both the row stride and the base aligned
  const bool fullrow = (topk == 2048) && (((uintptr_t)ix & 15) == 0);
  // the SIMPLE combine packs HH*PARTS units into nchk slots
  const int64_t units = fullrow ? 2 * (int64_t)HH : (int64_t)HH;
  const bool simple = (grid >= T * nchk) && (grid >= T * (int64_t)HH) && (nchk >= units);
  TORCH_CHECK(topk <= 2048, "topk too large for this kernel");
  TORCH_CHECK(idx.scalar_type() == torch::kInt32, "sparse_indices must be int32");
  TORCH_CHECK(q_nope.is_contiguous() && q_pe.is_contiguous() && ckv.is_contiguous() &&
                  kpe.is_contiguous() && idx.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(q_nope.size(1) == HH && q_nope.size(2) == DC && q_pe.size(2) == DR,
              "unexpected head/dim layout");
  const int nkvtok = (int)(ckv.size(0) * ckv.size(1));
  const auto *qn = (const __nv_bfloat16 *)q_nope.data_ptr();
  const auto *qp = (const __nv_bfloat16 *)q_pe.data_ptr();
  const auto *kc = (const __nv_bfloat16 *)ckv.data_ptr();
  const auto *kp = (const __nv_bfloat16 *)kpe.data_ptr();
  auto *o = (__nv_bfloat16 *)out.data_ptr();
  auto *ls = (float *)lse.data_ptr();
  auto *pO = (__nv_bfloat16 *)po.data_ptr();
  auto *pM = (unsigned long long *)pml.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define LAUNCH(C, G, X, M, F, D)                                                         \
  {                                                                                      \
    int nchunk = 1;                                                                     \
    while (nchunk * (C) < topk) nchunk *= 2;                                            \
    cudaFuncSetAttribute(dsa_kernel<C, G, X, M, F, D>,                                   \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,                    \
                         (int)smem_for<C, G, X>());                                      \
    int nclog2 = 0;                                                                      \
    while ((1 << nclog2) < nchunk) ++nclog2;                                             \
    TORCH_CHECK((1 << nclog2) == nchunk, "nchunk must be a power of two");              \
    TORCH_CHECK(nchunk <= (X) / 8 && nchunk <= 32, "nchunk too large to combine");       \
    dsa_kernel<C, G, X, M, F, D><<<(int)grid, X, smem_for<C, G, X>(), st>>>(             \
        qn, qp, kc, kp, ix, o, ls, pO, pM, (unsigned int)gen, (float)sm_scale2, T,       \
        topk, nchunk, nclog2, nkvtok);                                                   \
  }
  DISPATCH(LAUNCH)
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &dsa_plan);
  m.def("run", &dsa_run);
}
"""
def _src_path():
    h = hashlib.sha1(_CU_SRC.encode()).hexdigest()[:16]
    d = os.environ.get("DSA_BUILD_ROOT") or os.path.join(tempfile.gettempdir(), "dsa_build_" + h)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "dsa_kernel.cu")
    if not os.path.exists(p) or open(p).read() != _CU_SRC:
        with open(p, "w") as f:
            f.write(_CU_SRC)
    return d, p
_HERE, _CU_PATH = _src_path()
"""Entry point for the hand-written DSA sparse MLA decode kernel.

run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output, lse)

Everything that is reused between calls is shape-derived only: the JIT build, the
launch geometry and the scratch buffers for the split-K partials.  The generation
counter that stamps those partials is bumped per call, so nothing computed from
the input values survives a call.
"""
import torch

_H, _D, _R = 16, 512, 64
_LOG2E = 1.4426950408889634
_MOD = None
_PLAN = {}



def _mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load
        bd = os.environ.get("DSA_BUILD_DIR", os.path.join(_HERE, "_build"))
        os.makedirs(bd, exist_ok=True)
        _MOD = load(
            name="dsa_kernel",
            sources=[_CU_PATH],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a"],
            build_directory=bd,
            verbose=False,
        )
    return _MOD


_CS = int(os.environ.get("DSA_CS", "64"))



def _plan(T, topk, cs, nth, device):
    key = (T, topk, cs, nth, str(device))
    p = _PLAN.get(key)
    if p is None:
        grid = _mod().plan(cs, nth, T, topk)[0]      # grid, smem, occupancy, threads
        nchunk = 1
        while nchunk * cs < topk:
            nchunk *= 2
        po = torch.empty((T * nchunk, _H, _D), dtype=torch.bfloat16, device=device)
        # {gen, lse} publication slots; zeroed once so a fresh buffer can never
        # look like generation 1
        pml = torch.zeros((T, _H, nchunk), dtype=torch.int64, device=device)
        p = [grid, po, pml, 0]
        _PLAN[key] = p
    return p


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T = q_nope.shape[0]
    topk = sparse_indices.shape[-1]
    dev = q_nope.device
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    if T == 0:
        return (torch.empty((0, _H, _D), dtype=torch.bfloat16, device=dev),
                torch.empty((0, _H), dtype=torch.float32, device=dev))
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    sparse_indices = sparse_indices.contiguous().to(torch.int32)
    cs = _CS
    nth = 256
    pl = _plan(T, topk, cs, nth, dev)
    grid, po, pml = pl[0], pl[1], pl[2]
    pl[3] += 1
    out = torch.empty((T, _H, _D), dtype=torch.bfloat16, device=dev)
    lse = torch.empty((T, _H), dtype=torch.float32, device=dev)
    _mod().run(q_nope, q_pe, ckv_cache.reshape(-1, _D), kpe_cache.reshape(-1, _R),
               sparse_indices, out, lse, po, pml, pl[3], sm_scale * _LOG2E, cs, nth, grid)
    return out, lse
