// DSA top-k indexer (DeepSeek-V3.2 sparse attention indexer), written from scratch.
//
// One (cooperative) kernel launch, fixed grid (= #SMs), one work item per CTA, decoded on-device from
// seq_lens. Items are ordered [tail item per long batch][scoring items (one per page)][fill items]:
//   * fill  (seq_len <= TOPK): all tokens are selected -> emit global token ids, pad with -1.
//   * score (one page of a long batch): 64 tokens x 64 heads with mma.sync m16n8k32 (e4m3 in, fp32 acc).
//     The weighted relu-sum becomes a monotonic non-zero uint32 key that is simply stored to scratch
//     (no fence, no flag, no atomics): the CTA exits immediately.
//   * tail  (one per long batch): polls the key scratch of its batch; a key has arrived when it is non-zero.
//     Arriving keys are ingested on the fly into shared-memory histograms (4096 fine bins of key bits
//     [26:15], 64 coarse bins) and min/max, so when the last page lands only its 64 keys remain to be
//     ingested. The threshold bin is located with two warp-level scans, the few candidates of that bin are
//     rank-selected, and the top-k ids are compacted with ballot sweeps over the keys held in registers.
//     If the digit cannot order the keys (min/max differ in bits [31:27]) a register radix select is used.
//     Finally the tail re-zeroes the key scratch for the next call.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int TOPK = 2048;
constexpr int PAGE_TOKENS = 64;
constexpr int HEADS = 64;
constexpr int DIM = 128;
constexpr int PAGE_BYTES = PAGE_TOKENS * (DIM + 4);  // 8448: 64 fp8 rows then 64 fp32 scales
constexpr int SCALE_OFF = PAGE_TOKENS * DIM;         // 8192
constexpr int NBINS = 4096;
constexpr int NCOARSE = 64;
constexpr int HIST_SHIFT = 15;  // fine digit = key bits [26:15]; coarse digit = bits [26:21]
constexpr int NTHREADS = 512;
constexpr int NWARPS = NTHREADS / 32;
constexpr int SCORE_WARPS = 16;  // warps that run the mma in a scoring CTA
constexpr int MAXR = 16;  // keys per thread held in registers (seq_len <= NTHREADS * MAXR = 8192)
constexpr int NVEC = MAXR / 4;  // 16-byte key slots per thread
constexpr int MAX_BT_CACHE = 4096;  // block_table entries cached in smem
constexpr int BT_PER_THREAD = MAX_BT_CACHE / NTHREADS;
constexpr int MAX_CAND = 256;  // candidates of the threshold bin kept in smem (more -> fallback)
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ void mma_e4m3_16832(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// optional per-CTA phase timing (dbg != nullptr): slot 0 = globaltimer at start, others = clock64 deltas
#define TSTAMP(slot)                                                                        \
  do {                                                                                      \
    if (dbg != nullptr && threadIdx.x == 0) dbg[blockIdx.x * 16 + (slot)] = clock64() - t0; \
  } while (0)



// monotonic float -> uint32 map; 0 is reserved as the "not yet written" marker (only -NaN would map to it)
__device__ __forceinline__ uint32_t float_to_key(float f) {
  uint32_t u = __float_as_uint(f);
  u = (u >> 31) ? ~u : (u | 0x80000000u);
  return u == 0u ? 1u : u;
}

__device__ __forceinline__ int warp_incl_scan(int v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(FULL, v, o);
    if (lane >= o) v += y;
  }
  return v;
}

// One warp: 64 bins (2 per lane, lane l owns bins 2l, 2l+1) with counts c0, c1. Finds the bin (from the top)
// where the cumulative count reaches k. Returns bin and need (elements still needed from that bin, >= 1).
__device__ __forceinline__ void warp_find_bin64(int c0, int c1, int k, int& bin, int& need) {
  const int lane = threadIdx.x & 31;
  const int incl = warp_incl_scan(c0 + c1);
  const int total = __shfl_sync(FULL, incl, 31);
  const int above = total - incl;  // keys in bins above this lane's pair
  int mybin = -1, myneed = 0;
  if (above < k && k <= above + c0 + c1) {
    if (above + c1 >= k) {
      mybin = 2 * lane + 1;
      myneed = k - above;
    } else {
      mybin = 2 * lane;
      myneed = k - above - c1;
    }
  }
  const unsigned m = __ballot_sync(FULL, mybin >= 0);
  const int src = __ffs(m) - 1;
  bin = __shfl_sync(FULL, mybin, src);
  need = __shfl_sync(FULL, myneed, src);
}

// Block-wide find over thread-owned bins (fallback radix select). Ends with a barrier.
template <int PER>
__device__ __forceinline__ void find_bin(const int (&loc)[PER], int k, int* s_w, int* found) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int s = 0;
#pragma unroll
  for (int i = 0; i < PER; i++) s += loc[i];
  const int incl_w = warp_incl_scan(s);
  if (lane == 31) s_w[warp] = incl_w;
  __syncthreads();
  int pre = 0, total = 0;
#pragma unroll
  for (int w = 0; w < NWARPS; w++) {
    const int y = s_w[w];
    total += y;
    pre += (w < warp) ? y : 0;
  }
  const int above = total - (incl_w + pre);
  if (above < k && k <= above + s) {
    int running = above;
#pragma unroll
    for (int i = PER - 1; i >= 0; i--) {
      if (running + loc[i] >= k) {
        found[0] = threadIdx.x * PER + i;
        found[1] = k - running;
        break;
      }
      running += loc[i];
    }
  }
  __syncthreads();
}

struct Smem {
  int hist[NBINS];
  int coarse[NCOARSE];
  int bt[MAX_BT_CACHE];
  int2 cand[MAX_CAND];  // (key, token index)
  int outbuf[TOPK];     // staged output row, written back with coalesced 16-byte stores
  float part[4][PAGE_TOKENS];
  int w[NWARPS], w2[NWARPS];
  uint32_t minmax[2 * NWARPS];
  int found[4];
  int cnt;
};

// Fallback: register radix select (12/12/8 bits on range-normalised keys). Returns the raw threshold key.
__device__ uint32_t fallback_threshold(Smem& sm, const uint32_t (&kr)[MAXR], int n, int k, uint32_t kmin,
                                       uint32_t kmax) {
  const int tid = threadIdx.x;
  const uint32_t range = kmax - kmin;
  const int lz = (range == 0u) ? 0 : __clz(range);
  constexpr int PER = NBINS / NTHREADS;
  int loc[PER];
#pragma unroll
  for (int j = 0; j < PER; j++) sm.hist[tid + j * NTHREADS] = 0;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < MAXR; r++) {
    const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
    if (i < n) atomicAdd(&sm.hist[((kr[r] - kmin) << lz) >> 20], 1);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PER; j++) {
    loc[j] = sm.hist[tid * PER + j];
    sm.hist[tid * PER + j] = 0;
  }
  find_bin<PER>(loc, k, sm.w, sm.found);
  const uint32_t b1 = sm.found[0];
  const int need1 = sm.found[1];
#pragma unroll
  for (int r = 0; r < MAXR; r++) {
    const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
    const uint32_t kn = (kr[r] - kmin) << lz;
    if (i < n && (kn >> 20) == b1) atomicAdd(&sm.hist[(kn >> 8) & 0xfffu], 1);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PER; j++) {
    loc[j] = sm.hist[tid * PER + j];
    sm.hist[tid * PER + j] = 0;
  }
  find_bin<PER>(loc, need1, sm.w2, sm.found);
  const uint32_t pre = (b1 << 20) | ((uint32_t)sm.found[0] << 8);
  const int need2 = sm.found[1];
#pragma unroll
  for (int r = 0; r < MAXR; r++) {
    const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
    const uint32_t kn = (kr[r] - kmin) << lz;
    if (i < n && (kn >> 8) == (pre >> 8)) atomicAdd(&sm.hist[kn & 0xffu], 1);
  }
  __syncthreads();
  int loc1[1];
  loc1[0] = (tid < 256) ? sm.hist[tid] : 0;
  find_bin<1>(loc1, need2, sm.w, sm.found);
  return ((pre | (uint32_t)sm.found[0]) >> lz) + kmin;
}

// Warp-uniform decode of work item `it` from seq_lens. Items: [tail per long batch][score pages][fill].
enum ItemKind { KIND_TAIL = 0, KIND_SCORE = 1, KIND_FILL = 2 };

__device__ __forceinline__ bool decode_item(const int* __restrict__ seq_lens, int B, int it, int& b, int& p,
                                            int& kind) {
  const int lane = threadIdx.x & 31;
  if (B <= 32) {
    // one packed scan: pages [31:18], long batches [17:9], short batches [8:0]
    const int n = (lane < B) ? __ldg(seq_lens + lane) : 0;
    const bool big = (lane < B) && n > TOPK;
    const int pages = big ? ((n + 63) >> 6) : 0;
    const int packed = (pages << 18) | ((big ? 1 : 0) << 9) | ((lane < B && !big) ? 1 : 0);
    const int incl = warp_incl_scan(packed);
    const int tot = __shfl_sync(FULL, incl, 31);
    const int ns = tot >> 18, nt = (tot >> 9) & 0x1ff, nf = tot & 0x1ff;
    if (it >= nt + ns + nf) return false;
    int target, mine, incl_f;
    if (it < nt) {
      kind = KIND_TAIL;
      target = it;
      mine = big ? 1 : 0;
      incl_f = (incl >> 9) & 0x1ff;
    } else if (it < nt + ns) {
      kind = KIND_SCORE;
      target = it - nt;
      mine = pages;
      incl_f = incl >> 18;
    } else {
      kind = KIND_FILL;
      target = it - nt - ns;
      mine = (lane < B && !big) ? 1 : 0;
      incl_f = incl & 0x1ff;
    }
    const unsigned m = __ballot_sync(FULL, target < incl_f);
    const int l = __ffs(m) - 1;
    b = l;
    p = target - __shfl_sync(FULL, incl_f - mine, l);
    return true;
  }
  int nt = 0, ns = 0, nf = 0;
  for (int base = 0; base < B; base += 32) {
    const int bb = base + lane;
    const int n = (bb < B) ? __ldg(seq_lens + bb) : 0;
    const bool big = (bb < B) && n > TOPK;
    nt += __reduce_add_sync(FULL, big ? 1 : 0);
    ns += __reduce_add_sync(FULL, big ? ((n + 63) >> 6) : 0);
    nf += __reduce_add_sync(FULL, (bb < B && n <= TOPK) ? 1 : 0);
  }
  if (it >= nt + ns + nf) return false;
  int target;
  if (it < nt) {
    kind = KIND_TAIL;
    target = it;
  } else if (it < nt + ns) {
    kind = KIND_SCORE;
    target = it - nt;
  } else {
    kind = KIND_FILL;
    target = it - nt - ns;
  }
  int acc = 0;
  for (int base = 0; base < B; base += 32) {
    const int bb = base + lane;
    const int n = (bb < B) ? __ldg(seq_lens + bb) : 0;
    const bool big = (bb < B) && n > TOPK;
    const int cnt = (kind == KIND_SCORE) ? (big ? ((n + 63) >> 6) : 0)
                                         : ((kind == KIND_TAIL) ? (big ? 1 : 0) : ((bb < B && !big) ? 1 : 0));
    const int incl = warp_incl_scan(cnt);
    const int ctot = __shfl_sync(FULL, incl, 31);
    if (target < acc + ctot) {
      const unsigned m = __ballot_sync(FULL, target < acc + incl);
      const int l = __ffs(m) - 1;
      b = base + l;
      p = target - (acc + __shfl_sync(FULL, incl - cnt, l));
      return true;
    }
    acc += ctot;
  }
  return false;  // unreachable
}

__global__ void __launch_bounds__(NTHREADS, 1)
dsa_topk_indexer_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ kc, const float* __restrict__ w,
                        const int* __restrict__ seq_lens, const int* __restrict__ bt, int* __restrict__ out,
                        uint32_t* __restrict__ keys, int B, int max_pages, long long kc_bytes,
                        uint64_t* __restrict__ dbg, int* __restrict__ diag) {
  const long long t0 = clock64();
  if (dbg != nullptr && threadIdx.x == 0) dbg[blockIdx.x * 16] = globaltimer_ns();
  __shared__ Smem sm;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bt_n = B * max_pages;

  // block_table -> smem (all loads in flight at once, overlapping the seq_lens load); warm the k cache TLB
#ifdef DSA_NO_BTCACHE
  const bool bt_cached = false;  // measurement variant: block_table read directly from global memory
  if (tid == 0) prefetch_l2(bt + ((blockIdx.x * 32) % bt_n));
#else
  const bool bt_cached = bt_n <= MAX_BT_CACHE;
#endif
  int btv[BT_PER_THREAD];
  if (bt_cached) {
#pragma unroll
    for (int r = 0; r < BT_PER_THREAD; r++) {
      const int i = tid + r * NTHREADS;
      btv[r] = (i < bt_n) ? __ldg(bt + i) : 0;
    }
  }
  if (tid == 0) {
    const long long off = (long long)blockIdx.x << 21;
    if (off < kc_bytes) prefetch_l2(kc + off);
  }
  int b = 0, p = 0, kind = 0;
  const bool has_item = decode_item(seq_lens, B, blockIdx.x, b, p, kind);
  if (diag != nullptr && threadIdx.x == 0) {
    diag[blockIdx.x * 8 + 0] = has_item ? kind + 1 : 0;
    diag[blockIdx.x * 8 + 1] = b;
    diag[blockIdx.x * 8 + 2] = p;
    diag[blockIdx.x * 8 + 3] = 1;  // started
  }
  if (bt_cached) {
#pragma unroll
    for (int r = 0; r < BT_PER_THREAD; r++) {
      const int i = tid + r * NTHREADS;
      if (i < bt_n) sm.bt[i] = btv[r];
    }
  }
  if (tid == 0) sm.cnt = 0;
  __syncthreads();
  if (!has_item) return;  // block-uniform
  const int n = __shfl_sync(FULL, __ldg(seq_lens + b), 0);
  int* out_b = out + (size_t)b * TOPK;
  const int* btrow_g = bt + (size_t)b * max_pages;
  const int* s_btrow = sm.bt + b * max_pages;  // valid only if bt_cached
  TSTAMP(2);

  if (kind == KIND_FILL) {
    // all tokens selected: emit global ids, pad with -1 (4 ids per thread)
    const int i0 = tid * 4;
    int4 v = make_int4(-1, -1, -1, -1);
    if (i0 < n) {
      const int page = bt_cached ? s_btrow[i0 >> 6] : __ldg(btrow_g + (i0 >> 6));
      v.x = page * PAGE_TOKENS + (i0 & 63);
      v.y = (i0 + 1 < n) ? v.x + 1 : -1;
      v.z = (i0 + 2 < n) ? v.x + 2 : -1;
      v.w = (i0 + 3 < n) ? v.x + 3 : -1;
    }
    *reinterpret_cast<int4*>(out_b + i0) = v;
    TSTAMP(12);
    return;
  }

  uint32_t* kb = keys + (size_t)b * max_pages * PAGE_TOKENS;

  if (kind == KIND_SCORE) {
    const int page_id = bt_cached ? s_btrow[p] : __ldg(btrow_g + p);
    const uint8_t* kp = kc + (size_t)page_id * PAGE_BYTES;
    float scale = 0.f;
    if (tid < PAGE_TOKENS) scale = __ldg(reinterpret_cast<const float*>(kp + SCALE_OFF) + tid);
    if (warp < SCORE_WARPS) {
    const int g = lane >> 2, t = lane & 3;
    const int mt = warp & 3, nq = warp >> 2;  // m-tile (16 tokens), head quarter (2 n-tiles = 16 heads)
    const int row0 = mt * 16 + g, row1 = row0 + 8;
    // q / weights only depend on b: issue first
    uint4 Q[2][2];
    float2 wv[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int nt = nq * 2 + j;
      const int head = nt * 8 + g;
      const uint4* qr = reinterpret_cast<const uint4*>(q + ((size_t)b * HEADS + head) * DIM + 32 * t);
      Q[j][0] = __ldg(qr);
      Q[j][1] = __ldg(qr + 1);
      wv[j] = __ldg(reinterpret_cast<const float2*>(w + b * HEADS + nt * 8 + 2 * t));
    }
    // thread t of a quad owns physical dims [32t, 32t+32) of its rows (same permutation for A and B)
    const uint4* kr0 = reinterpret_cast<const uint4*>(kp + row0 * DIM + 32 * t);
    const uint4* kr1 = reinterpret_cast<const uint4*>(kp + row1 * DIM + 32 * t);
    uint4 A0[2], A1[2];
    A0[0] = __ldg(kr0);
    A0[1] = __ldg(kr0 + 1);
    A1[0] = __ldg(kr1);
    A1[1] = __ldg(kr1 + 1);

    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(A0);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(A1);
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const uint32_t* bq = reinterpret_cast<const uint32_t*>(Q[j]);
#pragma unroll
      for (int ks = 0; ks < 4; ks++)
        mma_e4m3_16832(acc[j], a0[2 * ks], a1[2 * ks], a0[2 * ks + 1], a1[2 * ks + 1], bq[2 * ks], bq[2 * ks + 1]);
    }
    float p0 = 0.f, p1 = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      p0 += wv[j].x * fmaxf(acc[j][0], 0.f) + wv[j].y * fmaxf(acc[j][1], 0.f);
      p1 += wv[j].x * fmaxf(acc[j][2], 0.f) + wv[j].y * fmaxf(acc[j][3], 0.f);
    }
    p0 += __shfl_xor_sync(FULL, p0, 1);
    p0 += __shfl_xor_sync(FULL, p0, 2);
    p1 += __shfl_xor_sync(FULL, p1, 1);
    p1 += __shfl_xor_sync(FULL, p1, 2);
    if (t == 0) {
      sm.part[nq][row0] = p0;
      sm.part[nq][row1] = p1;
    }
    }  // warp < SCORE_WARPS
    __syncthreads();
    TSTAMP(4);
    if (tid < PAGE_TOKENS) {
      const int tok = p * PAGE_TOKENS + tid;
      if (tok < n) {
        const float s = (sm.part[0][tid] + sm.part[1][tid] + sm.part[2][tid] + sm.part[3][tid]) * scale;
        // the non-zero key itself is the arrival signal for the tail: use a strong (volatile) store so it is
        // guaranteed to become visible to the polling CTA (a weak store may linger in this SM's caches)
        const uint32_t key = float_to_key(s);
        #ifdef DSA_RELAXED
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(kb + tok), "r"(key) : "memory");
#else
        asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(kb + tok), "r"(key) : "memory");
#endif
      }
    }
    TSTAMP(5);
    if (diag != nullptr && tid == 0) diag[blockIdx.x * 8 + 4] = 2;  // keys stored
    return;
  }

  // ---- tail item ----
#ifdef DSA_NO_TAIL
  return;  // measurement variant: scoring + fill only (output of long batches is left undefined)
#endif
  const int k = n < TOPK ? n : TOPK;
  const int npages_b = (n + 63) >> 6;
  constexpr int PER = NBINS / NTHREADS;
#pragma unroll
  for (int j = 0; j < PER; j++) sm.hist[tid + j * NTHREADS] = 0;
  if (tid < NCOARSE) sm.coarse[tid] = 0;
  const int* s_bt_t = s_btrow;
  if (!bt_cached) {
    for (int i = tid; i < npages_b; i += NTHREADS) sm.bt[i] = __ldg(btrow_g + i);
    s_bt_t = sm.bt;
  }
  __syncthreads();

  // ingest keys as they arrive (non-zero). Each thread privately re-polls only its own missing slots (no
  // block barrier per round); histograms are built incrementally so only the last page's keys remain when
  // the final key lands. The digit ordering is valid iff every key has bits [31:27] == 0b10000
  // (scores in [2, 2^17) or exactly 0); otherwise a radix-select fallback is used.
  // slot r = 4 s + j holds token i = 4 (tid + s NTHREADS) + j: NVEC 16-byte strong loads per thread per poll
  uint32_t kr[MAXR];
  unsigned pending = 0;  // bit r set: slot r valid and not yet ingested
#pragma unroll
  for (int r = 0; r < MAXR; r++) {
    kr[r] = 0u;
    if (4 * (tid + (r >> 2) * NTHREADS) + (r & 3) < n) pending |= 1u << r;
  }
  const int n4 = (n + 3) & ~3;
  const uint4* pa[NVEC];
#pragma unroll
  for (int s2 = 0; s2 < NVEC; s2++) pa[s2] = reinterpret_cast<const uint4*>(kb + min(4 * (tid + s2 * NTHREADS), n4 - 4));
  bool bad_digit = false;
  int rounds = 0;
  while (pending != 0u) {
    uint32_t v[MAXR];
#ifdef DSA_RELAXED
#define DSA_LD "ld.relaxed.gpu.global.v4.u32 "
#else
#define DSA_LD "ld.volatile.global.v4.u32 "
#endif
    asm volatile(
        DSA_LD "{%0,%1,%2,%3}, [%16];\n\t"
        DSA_LD "{%4,%5,%6,%7}, [%17];\n\t"
        DSA_LD "{%8,%9,%10,%11}, [%18];\n\t"
        DSA_LD "{%12,%13,%14,%15}, [%19];\n\t"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
          "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
        : "l"(pa[0]), "l"(pa[1]), "l"(pa[2]), "l"(pa[3])
        : "memory");
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      if ((pending & (1u << r)) && v[r] != 0u) {
        const uint32_t key = v[r];
        kr[r] = key;
        pending &= ~(1u << r);
        atomicAdd(&sm.hist[(key >> HIST_SHIFT) & 0xfffu], 1);
        atomicAdd(&sm.coarse[(key >> (HIST_SHIFT + 6)) & 0x3fu], 1);
        // digit ordering needs bits [31:27] == 0b11000 (scores in [2, 2^17)); an exact zero score
        // (key 0x80000000) is the smallest possible non-negative key and lands in bin 0 consistently
        bad_digit |= ((key >> (HIST_SHIFT + 12)) != 0x18u) && (key != 0x80000000u);
      }
    }
    rounds++;
    if (diag != nullptr && (rounds & 255) == 0 && clock64() - t0 > 6000000LL) {  // ~3 ms watchdog
      if (tid == 0) diag[blockIdx.x * 8 + 5] = -1;
      diag[blockIdx.x * 8 + 6] = __popc(pending);  // some thread's missing-slot count
      diag[blockIdx.x * 8 + 7] = 4 * (tid + (__ffs(pending) - 1 >> 2) * NTHREADS) + ((__ffs(pending) - 1) & 3);
      break;
    }
  }
  bool fast = !__syncthreads_or(bad_digit);
  TSTAMP(7);
  if (dbg != nullptr && tid == 0) dbg[blockIdx.x * 16 + 15] = rounds;
  int gidb[MAXR];

  unsigned mg[MAXR], me[MAXR];
  int cg = 0, ce = 0;
  int need3 = 0;
  bool use_ties = false;  // ties / threshold-bin members written after the sure keys
  const unsigned lt = (1u << lane) - 1u;
  if (fast) {
    // warp 0: two warp-level scans (coarse bin, then fine bin inside it); other warps: global ids of their
    // slots meanwhile (token i = 4 (tid + s NTHREADS) + j lies in page (tid + s NTHREADS) >> 4)
    if (warp == 0) {
      int c1, needc;
      warp_find_bin64(sm.coarse[2 * lane], sm.coarse[2 * lane + 1], k, c1, needc);
      int rel, need1;
      warp_find_bin64(sm.hist[c1 * 64 + 2 * lane], sm.hist[c1 * 64 + 2 * lane + 1], needc, rel, need1);
      if (lane == 0) {
        sm.found[0] = c1 * 64 + rel;
        sm.found[1] = need1;
      }
    }
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int pg = (tid + (r >> 2) * NTHREADS) >> 4;
      gidb[r] = (pg < npages_b) ? s_bt_t[pg] * PAGE_TOKENS + 4 * (tid & 15) + (r & 3) : -1;
    }
    __syncthreads();
    // broadcast through shfl so the compiler can prove these values warp-uniform (avoids collective loops)
    const uint32_t b1 = __shfl_sync(FULL, (uint32_t)sm.found[0], 0);
    const int need1 = __shfl_sync(FULL, sm.found[1], 0);
    const int n_sure = k - need1;
    TSTAMP(8);
    // one sweep: keys in bins above b1 are selected for sure (masks kept in registers); keys in bin b1 are
    // collected as candidates
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
      mg[r] = __ballot_sync(FULL, (i < n) && ((kr[r] >> HIST_SHIFT) & 0xfffu) > b1);
      cg += __popc(mg[r]);
    }
#ifdef DSA_APPROX
    // bin-resolution selection: the need1 keys of the threshold bin are taken in slot order (their scores
    // differ from the exact top-k by at most the fine-bin width, 2^-8 of the octave); no rank select
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
      me[r] = __ballot_sync(FULL, (i < n) && ((kr[r] >> HIST_SHIFT) & 0xfffu) == b1);
      ce += __popc(me[r]);
    }
    if (lane == 0) {
      sm.w[warp] = cg;
      sm.w2[warp] = ce;
    }
    need3 = need1;
    use_ties = true;
    __syncthreads();
    TSTAMP(9);
    const int C = 0;
    (void)n_sure;
    if (false) {
#else
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
      if ((i < n) && ((kr[r] >> HIST_SHIFT) & 0xfffu) == b1) {
        const int pos = atomicAdd(&sm.cnt, 1);  // rare: only the keys of the threshold bin
        if (pos < MAX_CAND) sm.cand[pos] = make_int2((int)kr[r], gidb[r]);
      }
    }
    if (lane == 0) sm.w[warp] = cg;
    __syncthreads();
    const int C = __shfl_sync(FULL, sm.cnt, 0);
    TSTAMP(9);
    if (dbg != nullptr && tid == 0) dbg[blockIdx.x * 16 + 14] = C;
    if (C <= MAX_CAND) {
#endif
      // rank select inside bin b1, one warp per candidate (lanes compare against 32 candidates at a time):
      // the top need1 candidates go to out[n_sure + rank]
      for (int c0 = 0; c0 < C; c0 += NWARPS) {  // uniform trip count for the whole block
        const int c = c0 + warp;
        const int2 me2 = sm.cand[c < C ? c : 0];
        int rank = 0;
        for (int j0 = 0; j0 < C; j0 += 32) {
          const int j = j0 + lane;
          bool above = false;
          if (j < C) {
            const int2 cj = sm.cand[j];
            above = ((uint32_t)cj.x > (uint32_t)me2.x) || (cj.x == me2.x && j < c);
          }
          rank += __popc(__ballot_sync(FULL, above));
        }
        if (c < C && lane == 0 && rank < need1) sm.outbuf[n_sure + rank] = me2.y;
      }
      TSTAMP(13);
    } else {
      fast = false;  // bin b1 too crowded for the candidate list: use the exact threshold instead
      cg = 0;
    }
  }
  if (!fast) {
    // fallback: exact threshold T over range-normalised keys; ties (== T) taken in sweep order
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int pg = (tid + (r >> 2) * NTHREADS) >> 4;
      gidb[r] = (pg < npages_b) ? s_bt_t[pg] * PAGE_TOKENS + 4 * (tid & 15) + (r & 3) : -1;
    }
    uint32_t kmin = 0xffffffffu, kmax = 0u;
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      if (4 * (tid + (r >> 2) * NTHREADS) + (r & 3) < n) {
        kmin = min(kmin, kr[r]);
        kmax = max(kmax, kr[r]);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      kmin = min(kmin, __shfl_xor_sync(FULL, kmin, o));
      kmax = max(kmax, __shfl_xor_sync(FULL, kmax, o));
    }
    if (lane == 0) {
      sm.minmax[warp] = kmin;
      sm.minmax[NWARPS + warp] = kmax;
    }
    __syncthreads();
#pragma unroll
    for (int w2 = 0; w2 < NWARPS; w2++) {
      kmin = min(kmin, sm.minmax[w2]);
      kmax = max(kmax, sm.minmax[NWARPS + w2]);
    }
    const uint32_t T = fallback_threshold(sm, kr, n, k, kmin, kmax);
    use_ties = true;
#pragma unroll
    for (int r = 0; r < MAXR; r++) {
      const int i = 4 * (tid + (r >> 2) * NTHREADS) + (r & 3);
      mg[r] = __ballot_sync(FULL, (i < n) && kr[r] > T);
      me[r] = __ballot_sync(FULL, (i < n) && kr[r] == T);
      cg += __popc(mg[r]);
      ce += __popc(me[r]);
    }
    if (lane == 0) {
      sm.w[warp] = cg;
      sm.w2[warp] = ce;
    }
    __syncthreads();
  }

  // compaction of the "sure" keys: block prefix over warps, then writes from the register masks
  int base_g = 0, base_e = 0, n_gt = 0;
#pragma unroll
  for (int w2 = 0; w2 < NWARPS; w2++) {
    const int y = sm.w[w2];
    n_gt += y;
    base_g += (w2 < warp) ? y : 0;
    if (use_ties) base_e += (w2 < warp) ? sm.w2[w2] : 0;
  }
  if (!fast) need3 = k - n_gt;
#pragma unroll
  for (int r = 0; r < MAXR; r++) {
    if ((mg[r] >> lane) & 1u) sm.outbuf[base_g + __popc(mg[r] & lt)] = gidb[r];
    base_g += __popc(mg[r]);
    if (use_ties) {
      if ((me[r] >> lane) & 1u) {
        const int pos = base_e + __popc(me[r] & lt);
        if (pos < need3) sm.outbuf[n_gt + pos] = gidb[r];
      }
      base_e += __popc(me[r]);
    }
  }
  for (int i = k + tid; i < TOPK; i += NTHREADS) sm.outbuf[i] = -1;
  // reset the key scratch for the next call (whole 16-byte slots; entries >= n are unused)
#pragma unroll
  for (int s2 = 0; s2 < NVEC; s2++)
    if (4 * (tid + s2 * NTHREADS) < n)
      *reinterpret_cast<uint4*>(kb + 4 * (tid + s2 * NTHREADS)) = make_uint4(0u, 0u, 0u, 0u);
  __syncthreads();
  // coalesced write-back of the staged row (4 ids per thread)
  reinterpret_cast<int4*>(out_b)[tid] = reinterpret_cast<const int4*>(sm.outbuf)[tid];
  if (diag != nullptr && tid == 0) diag[blockIdx.x * 8 + 4] = 3;  // tail done
  TSTAMP(10);
}

}  // namespace

void dsa_topk_indexer(torch::Tensor q, torch::Tensor k_cache, torch::Tensor weights, torch::Tensor seq_lens,
                      torch::Tensor block_table, torch::Tensor out, torch::Tensor keys, int64_t grid,
                      c10::optional<torch::Tensor> dbg, c10::optional<torch::Tensor> diag) {
  int B = (int)q.size(0);
  int max_pages = (int)block_table.size(1);
  TORCH_CHECK(q.size(1) == HEADS && q.size(2) == DIM, "q shape");
  TORCH_CHECK(k_cache.size(1) == PAGE_TOKENS && k_cache.size(3) == DIM + 4, "k cache shape");
  TORCH_CHECK(out.size(1) == TOPK, "out shape");
  TORCH_CHECK(max_pages * PAGE_TOKENS <= NTHREADS * MAXR && max_pages <= MAX_BT_CACHE, "sequence too long");
  const uint8_t* q_p = reinterpret_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* kc_p = reinterpret_cast<const uint8_t*>(k_cache.data_ptr());
  const float* w_p = weights.data_ptr<float>();
  const int* sl_p = seq_lens.data_ptr<int>();
  const int* bt_p = block_table.data_ptr<int>();
  int* out_p = out.data_ptr<int>();
  uint32_t* keys_p = reinterpret_cast<uint32_t*>(keys.data_ptr<int>());
  long long kc_bytes = (long long)(k_cache.numel() * k_cache.element_size());
  uint64_t* dbg_p = dbg.has_value() ? reinterpret_cast<uint64_t*>(dbg->data_ptr<int64_t>()) : nullptr;
  int* diag_p = diag.has_value() ? diag->data_ptr<int>() : nullptr;
  void* args[] = {&q_p, &kc_p, &w_p, &sl_p, &bt_p, &out_p, &keys_p, &B, &max_pages, &kc_bytes, &dbg_p, &diag_p};
  // cooperative launch: guarantees all CTAs are co-resident, so the tail CTA may poll for the keys
  cudaError_t err = cudaLaunchCooperativeKernel((const void*)dsa_topk_indexer_kernel, dim3((int)grid),
                                                dim3(NTHREADS), args, 0, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("dsa_topk_indexer", &dsa_topk_indexer); }
