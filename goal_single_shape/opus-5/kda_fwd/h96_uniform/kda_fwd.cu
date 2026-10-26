// Kimi Delta Attention forward, chunked-parallel, SM100 (tcgen05 / UMMA).
//
// Per chunk of C=64 tokens (scalar recurrence: bench/kda_ref.py):
//   l_t = lower_bound*log2(e) * sigmoid(exp(A_log)*(g_t + dt_bias))     [log2 domain]
//   c_t = prefix sum of l ; r = c_{C-1}/2 ; e_t = 2^(c_t-r) ; E = 2^r
//   kt = k_raw*e , kb = k_raw/e , qt = q_raw*e        (l2 norms folded into
//   nk = 1/||k_raw||, nq = 1/||q_raw||                 the small matrices)
//   [Gk;Gq]   = [kt;qt] kb^T                           one M=128 MMA
//   P  = tril(diag(beta nk) Gk diag(nk),-1) ; T = (I+P)^-1 ; T' = diag(nk) T
//   [KtS;QtS] = [kt;qt] S                              one M=128 MMA
//   Xb = diag(beta)(V - diag(nk) KtS) ; U' = T' Xb
//   O  = diag(nq*scale) ( QtS + tril(Gq,0) U' )
//   S^T = E * ( E*S^T + kb^T U' )                      state kept in TMEM
//
// The mid-point reference r halves the exponent range of e and 1/e relative to
// referencing the chunk start, which doubles the tolerable decay rate.
//
// Shared memory is addressed with the closed-form 128B swizzle in kda_common.cuh
// so that every store is a 16-byte vector store; CuTe is used only to build the
// UMMA descriptors.
#include <cstdio>
#include "kda_common.cuh"

using MmaKK64  = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16,bf16,float,128, 64,UMMA::Major::K,UMMA::Major::K >{}));
using MmaKM64  = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16,bf16,float,128, 64,UMMA::Major::K,UMMA::Major::MN>{}));
using MmaKM128 = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16,bf16,float,128,128,UMMA::Major::K,UMMA::Major::MN>{}));
// MMA 5 takes kb^T with the *channel* (M) index contiguous.  That is kb's
// natural [token][channel] order, so the prologue can store the bf16x4 it has
// already packed -- no register gather -- and consecutive lanes land 8 bytes
// apart instead of 512 (the K-major form gave a 16-way store bank conflict).
using MmaMM128 = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16,bf16,float,128,128,UMMA::Major::MN,UMMA::Major::MN>{}));
// Narrow TMEM slices: fewer live registers per thread, which is what caps occupancy.
#ifdef TCOLS_OVR
#define TCOLS TCOLS_OVR
#else
#define TCOLS 16            // TMEM columns moved per 128-thread group step
#endif
#define TJ    (TCOLS / 8)   // 8-element vector stores per step
using MmaN32   = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<bf16,bf16,float,128, TCOLS,UMMA::Major::K,UMMA::Major::K >{}));

template <class Mma,int M,int K> using LA_K  = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<bf16>{},  partition_shape_A(Mma{},make_shape(Int<M>{},Int<K>{}))));
template <class Mma,int N,int K> using LB_K  = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<bf16>{},  partition_shape_B(Mma{},make_shape(Int<N>{},Int<K>{}))));
template <class Mma,int N,int K> using LB_MN = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<bf16>{}, partition_shape_B(Mma{},make_shape(Int<N>{},Int<K>{}))));
template <class Mma,int M,int K> using LA_MN = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_MN_SW128_Atom<bf16>{}, partition_shape_A(Mma{},make_shape(Int<M>{},Int<K>{}))));

using L_KQ  = LA_K <MmaKK64,  128, 128>;   // [kt;qt]            A K-major
using L_KB  = LB_K <MmaKK64,   64, 128>;   // kb   (N=s,K=d)     B K-major
using L_KBT = LA_MN<MmaMM128, 128,  64>;   // kb^T (M=Kdim,K=s)  A MN-major
using L_SB  = LB_MN<MmaKM128, 128, 128>;   // S^T  (N=V,K=Kdim)  B MN-major
using L_VB  = LB_MN<MmaKM128, 128,  64>;   // Xb/U'(N=V,K=s)     B MN-major
using L_A64 = LA_K <MmaKM128, 128,  64>;   // [X;0]              A K-major
using L_B64 = LB_MN<MmaKM64,   64,  64>;   // X    (N=64,K=64)   B MN-major

struct Shm {
  // B1, B2 and nd live inside KQ.  KQ holds [kt;qt] and is read only by MMA 1
  // (the Gram) and MMA 2 (KQS), both of which have completed at the single
  // ctl.wait() before the Gram post; it is not written again until the next
  // chunk's prologue, after barrier 11.  B1 is written in the Gram post and
  // read by dot1, nd likewise and read by the inverse, B2 written by the
  // inverse and read by dot2 -- all strictly inside that dead window.
  //   B1 at KQ+0, B2 at KQ+4096, nd at KQ+8192  (24,832 B of 32,768)
  // The nd row stride of 33 floats is unchanged: the padding that fixed the
  // 32-way bank conflict is a property of the stride, not of the base.
  // 32KB + 8KB + 8KB + 8.4KB -> 32KB.
  alignas(128) bf16 KQ [cosize_v<L_KQ>];
  // A1 lives inside KB.  KB holds kb and is read only by MMA 1 (the Gram),
  // so it is dead from the same ctl.wait() onward; A1 is first written by the
  // inverse (after barrier 5) and last read by MMA 3, entirely inside that
  // window.  16KB saved.
  alignas(128) bf16 KB [cosize_v<L_KB>];
  // Placement probe: -DTPAD=n inserts n*128 bytes before KBT, shifting every
  // tile after it.  Footprint grows by at most 4KB, staying inside the same
  // carve-out, so this isolates absolute tile placement.
#ifdef TPAD
  char tpad[TPAD * 128];
#endif
  alignas(128) bf16 KBT[cosize_v<L_KBT>];
  // SB spans XB and A2, which are declared adjacently below -- two 16KB
  // tiles, both 128-aligned, so no padding between them: 32,768 B exactly,
  // the cosize of L_SB.  SB is written by the state rescale between barriers
  // 3 and 4 and read by MMA 2, and is dead at the ctl.wait() that follows.
  // Everything living in those 32KB begins after that wait -- AH in the Gram
  // post, A2 in the A2 stage, XB in the Xb stage, UU after MMA 3.  The one
  // earlier tenant is `tot`, finished between barriers 1 and 2, before the
  // rescale writes SB.  32KB saved.
  // UU shares XB.  Their lifetimes are disjoint within a chunk: the Xb stage
  // writes XB, MMA 3 consumes it and completes at its own ctl.wait(), and
  // only then does the U' dump write UU.  MMA 4 and MMA 5 read UU, the
  // output stage reuses it for coalescing (write then re-read, both before
  // barrier 11), and the next chunk's Xb write comes after that chunk's
  // barrier 3.  Same layout (L_VB) and same cursor, so sUU just points at
  // sh.XB.  32KB -> 16KB.
  alignas(128) bf16 XB [cosize_v<L_VB>];
  // A2 and AH share one tile.  Both are 128x64 K-major A operands, and each
  // writes only half the rows: the A2 stage writes rows 0..63 (guarded by
  // rr0 < CB) and the Gram post writes AH's rows 64..127 (the r >= CB
  // branch).  offK<128>(r,c) = sw128(r*64 + c) for K = 64, so those are the
  // disjoint element ranges 0..4095 and 4096..8191.  Each MMA reads all 128
  // rows, but dot2 discards the products of rows 64..127 and MMA 4 discards
  // those of rows 0..63 -- exactly the halves the other tile owns -- so
  // reading the neighbour's real data there is as harmless as reading the
  // uninitialised memory that was there before.  AH is written first (Gram
  // post) and read last (MMA 4); A2 is written and read entirely in between,
  // and never touches AH's rows.  48KB -> 32KB for the three A tiles.
  alignas(128) bf16 A2 [cosize_v<L_A64>];
  // Controlled test of the shared-memory/L1 partition: -DSHPAD_KB=n inflates
  // the footprint by n KB without changing a single instruction, so any
  // timing difference is the carve-out and nothing else.
#ifdef SHPAD_KB
  char shpad[SHPAD_KB * 1024];
#endif
#define NBAR 8
  alignas(16)  uint64_t bar[NBAR];
  alignas(16)  uint32_t tmem_ptr;
  // Padded to 33: the Gram post writes nd[b][r & 31][...] and consecutive
  // threads have consecutive r, so an unpadded row stride of 32 floats
  // (128 B) puts every thread in a warp on the same bank -- a 32-way
  // conflict on every write.  33 floats staggers them by one bank.
  // tot lives inside XB (which UU already shares).  It is written and read
  // only between barriers 1 and 2 of the prologue; XB is written after
  // barrier 7 and UU after MMA 3, and the previous chunk's output staging
  // finished reading UU before barrier 11.  8KB saved.
  alignas(16) float hfv[DD];
  float nk[CB], nq[CB], bs[CB];
  float Ev[DD], Eprev[DD];
};

#define TM_STATE   0
#define TM_GRAM  128
#define TM_KQS   192
#define TM_U     320
#define TM_INV   448

CUTE_DEVICE auto cid32() {
  MmaN32 m;
  return m.get_slice(0).partition_C(make_identity_tensor(make_shape(Int<128>{}, Int<TCOLS>{})));
}
CUTE_DEVICE auto acc32(uint32_t addr) {
  MmaN32 m; Tensor t = m.get_slice(0).make_fragment_C(cid32()); t.data() = addr; return t;
}
template <int N> CUTE_DEVICE auto accN(uint32_t addr) {
  using M = conditional_t<N == 64, MmaKK64, MmaKM128>;
  M m; Tensor t = m.get_slice(0).make_fragment_C(
      m.get_slice(0).partition_C(make_identity_tensor(make_shape(Int<128>{}, Int<N>{}))));
  t.data() = addr; return t;
}

__device__ __forceinline__ void st8(bf16* p, const float* x) {
  *reinterpret_cast<bf16x8*>(p) = pack8(x);
}
__device__ __forceinline__ void st8z(bf16* p) {
  bf16x8 w;
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) w.v[i] = bf16(0.f);
  *reinterpret_cast<bf16x8*>(p) = w;
}

// Rotating mbarriers: with NBAR of them a barrier is only reused after NBAR
// MMAs, so the leader cannot flip a phase twice before every thread has
// observed it.  That removes the __syncthreads() that used to guard each MMA --
// barrier stalls were the single largest source of idle issue slots.
struct Ctl {
  // `ph` must be a bitmask, not `int ph[NBAR]`.  Indexed by the runtime slot
  // `idx`, an array cannot be register-allocated, so ptxas put the whole Ctl
  // in local memory: a 48-byte frame zeroed at entry and a dependent
  // LDL/STL read-modify-write of ph[idx] immediately before *every* MMA wait
  // (13 of the kernel's 19 local-memory instructions were this one line).
  // One bit per slot keeps the same rotation in three register ALU ops.
  uint64_t* bar; int idx; uint32_t ph;
  CUTE_DEVICE uint64_t* cur() { return bar + idx; }
  CUTE_DEVICE void wait() {
    cute::wait_barrier(bar[idx], int((ph >> idx) & 1u));
    ph ^= (1u << idx);
    idx = (idx + 1) & (NBAR - 1);
  }
};

template <class Mma, class TA, class TB, class TC>
CUTE_DEVICE void issue_mma(Mma mma, TA const& a, TB const& b, TC& c, bool acc) {
  mma.accumulate_ = acc ? UMMA::ScaleOut::One : UMMA::ScaleOut::Zero;
  CUTE_UNROLL
  for (int k = 0; k < size<2>(a); ++k) { gemm(mma, a(_,_,k), b(_,_,k), c); mma.accumulate_ = UMMA::ScaleOut::One; }
}
template <class Mma, class TA, class TB, class TC>
CUTE_DEVICE void run_mma(Mma mma, TA const& a, TB const& b, TC& c, bool acc, bool lead, Ctl& ctl) {
  if (lead) { issue_mma(mma, a, b, c, acc); cutlass::arch::umma_arrive(ctl.cur()); }
  ctl.wait();
}

#ifdef FORCE2
__global__ __launch_bounds__(NTHR, 2) void kda_fwd_kernel(
#else
__global__ __launch_bounds__(NTHR) void kda_fwd_kernel(
#endif
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ G, const bf16* __restrict__ BETA,
    const float* __restrict__ A_LOG, const float* __restrict__ DT_BIAS,
    bf16* __restrict__ STATE, bf16* __restrict__ OUT, const long* __restrict__ CU,
    int H, float lb2, float scale) {
  extern __shared__ char smem_raw[];
  Shm& sh = *reinterpret_cast<Shm*>(smem_raw);
  bf16* const shSB = sh.XB;          // SB spans XB + A2 (adjacent, 32KB)
  bf16* const shA1 = sh.KB;
  auto& shTOT = *reinterpret_cast<float (*)[NPG][DD]>(sh.XB);
  bf16* const shB1 = sh.KQ;
  bf16* const shB2 = sh.KQ + cosize_v<L_B64>;
  auto& shND = *reinterpret_cast<float (*)[2][32][33]>(sh.KQ + 2 * cosize_v<L_B64>);
  const int h = blockIdx.x, nseq = blockIdx.y;
  const int bos = (int)CU[nseq], L = (int)CU[nseq + 1] - bos;
  const int tid = threadIdx.x, tg = tid / CGR, cg = tid % CGR;
  const int d0 = cg * CPT, t0 = tg * TPT;
  const bool lead = (tid < 32);

  MmaKK64 mmaG; MmaKM64 mmaI; MmaKM128 mmaB; MmaMM128 mmaS;
  Tensor sKQ  = make_tensor(make_smem_ptr(sh.KQ),  L_KQ{});
  Tensor sKB  = make_tensor(make_smem_ptr(sh.KB),  L_KB{});
  Tensor sKBT = make_tensor(make_smem_ptr(sh.KBT), L_KBT{});
  Tensor sSB  = make_tensor(make_smem_ptr(shSB),  L_SB{});
  Tensor sXB  = make_tensor(make_smem_ptr(sh.XB),  L_VB{});

  Tensor sUU  = make_tensor(make_smem_ptr(sh.XB),  L_VB{});   // shares XB
  Tensor sAH  = make_tensor(make_smem_ptr(sh.A2),  L_A64{});   // shares A2
  Tensor sA1  = make_tensor(make_smem_ptr(shA1),  L_A64{});
  Tensor sA2  = make_tensor(make_smem_ptr(sh.A2),  L_A64{});
  Tensor sB1  = make_tensor(make_smem_ptr(shB1),  L_B64{});
  Tensor sB2  = make_tensor(make_smem_ptr(shB2),  L_B64{});

  using TAlloc = cute::TMEM::Allocator1Sm;
  TAlloc alloc{};
  if (lead) alloc.allocate(TAlloc::Sm100TmemCapacityColumns, &sh.tmem_ptr);
  if (lead && cute::elect_one_sync())
    for (int i = 0; i < NBAR; ++i) cute::initialize_barrier(sh.bar[i], 1);
  smem_sync();
  const uint32_t tb = sh.tmem_ptr;
  Ctl ctl{&sh.bar[0], 0, 0u};

  Tensor tProbe = acc32(tb);
  auto t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b16x{},  tProbe);
  auto r2t = make_tmem_copy(SM100_TMEM_STORE_32dp32b16x{}, tProbe);
  const int tlo = tid & 127, wgrp = tid >> 7;
  auto thrL = t2r.get_slice(tlo);
  auto thrS = r2t.get_slice(tlo);
  auto cCsh = shape(thrL.partition_D(cid32()));
  // each thread holds row `tlo`, 32 consecutive columns (verified in t_layout2.cu)
#define TCHUNK(NCOL) for (int ch = wgrp; ch < (NCOL) / TCOLS; ch += NTG)
#define TLOAD(base) float rr[TCOLS]; { Tensor rt = make_tensor(make_rmem_ptr(&rr[0]), cCsh); \
      copy(t2r, thrL.partition_S(acc32((base) + TCOLS * ch)), rt); \
      cutlass::arch::fence_view_async_tmem_load(); }
// TLOAD issues the tcgen05.ld and immediately blocks on it.  TLOAD_ISSUE
// omits the wait so that work independent of the loaded values can occupy the
// latency window; the caller must TLOAD_WAIT() before reading rr.
#define TLOAD_ISSUE(base) float rr[TCOLS]; { Tensor rt = make_tensor(make_rmem_ptr(&rr[0]), cCsh); \
      copy(t2r, thrL.partition_S(acc32((base) + TCOLS * ch)), rt); }
#define TLOAD_WAIT() cutlass::arch::fence_view_async_tmem_load()
#define TSTORE(base) { Tensor rt = make_tensor(make_rmem_ptr(&rr[0]), cCsh); \
      copy(r2t, rt, thrS.partition_D(acc32((base) + TCOLS * ch))); \
      cutlass::arch::fence_view_async_tmem_store(); }
#define C0 (TCOLS * ch)
// TLOAD issues one tcgen05.ld and immediately blocks on it
// (fence_view_async_tmem_load is tcgen05.wait::ld), so a TCHUNK loop exposes
// the TMEM latency once per iteration with nothing overlapping it.  TLOAD2
// issues both of a two-iteration loop's loads and waits once.
#define NCH(NCOL) ((NCOL) / TCOLS / NTG)            // iterations per thread
// TLOADN generalises TLOAD2 to NCH(NCOL) iterations: every tcgen05.ld is
// issued before the single tcgen05.wait::ld, so the TMEM latency is paid once
// per stage rather than once per iteration, for any NTG (= any NTHR).  At
// NTG == 4 it expands to exactly the two-load form it replaces.
#define TLOADN(base, NCOL) float rn[NCH(NCOL)][TCOLS]; { \
      _Pragma("unroll") \
      for (int q = 0; q < NCH(NCOL); ++q) { \
        Tensor tq = make_tensor(make_rmem_ptr(&rn[q][0]), cCsh); \
        copy(t2r, thrL.partition_S(acc32((base) + TCOLS * (wgrp + q * NTG))), tq); \
      } \
      cutlass::arch::fence_view_async_tmem_load(); }
#define C0N(q) (TCOLS * (wgrp + (q) * NTG))

  // ---- initial state (gmem S[v][k]) -> TMEM as S^T[k][v]
  {
    const bf16* sp = STATE + ((size_t)nseq * H + h) * DD * DD;
    TCHUNK(128) {
      float rr[32];
      CUTE_UNROLL
      for (int i = 0; i < TCOLS; ++i) rr[i] = float(sp[(C0 + i) * DD + tlo]);
      TSTORE(tb + TM_STATE);
    }
    for (int i = tid; i < DD; i += NTHR) sh.Eprev[i] = 1.f;
  }
  smem_sync();

  // MMA operand descriptors and TMEM accumulators are loop invariant
  auto frKQ_G  = mmaG.get_slice(0).make_fragment_A(sKQ);
  auto frKB_G  = mmaG.get_slice(0).make_fragment_B(sKB);
  auto frKQ_B  = mmaB.get_slice(0).make_fragment_A(sKQ);
  auto frSB_B  = mmaB.get_slice(0).make_fragment_B(sSB);
  auto frA1_I  = mmaI.get_slice(0).make_fragment_A(sA1);
  auto frB1_I  = mmaI.get_slice(0).make_fragment_B(sB1);
  auto frA2_I  = mmaI.get_slice(0).make_fragment_A(sA2);
  auto frB2_I  = mmaI.get_slice(0).make_fragment_B(sB2);
  auto frA1_B  = mmaB.get_slice(0).make_fragment_A(sA1);
  auto frXB_B  = mmaB.get_slice(0).make_fragment_B(sXB);
  auto frAH_B  = mmaB.get_slice(0).make_fragment_A(sAH);
  auto frUU_B  = mmaB.get_slice(0).make_fragment_B(sUU);
  auto frUU_S  = mmaS.get_slice(0).make_fragment_B(sUU);
  auto frKBT_B = mmaS.get_slice(0).make_fragment_A(sKBT);
  Tensor tGram  = accN<64>(tb + TM_GRAM);
  Tensor tKQS   = accN<128>(tb + TM_KQS);
  Tensor tInv   = accN<64>(tb + TM_INV);
  Tensor tU     = accN<128>(tb + TM_U);
  Tensor tState = accN<128>(tb + TM_STATE);

  // All shared-memory rows touched by this thread are fixed for the whole
  // kernel, so every swizzle XOR is hoisted here and each store costs one
  // shift/xor/add instead of a full offK/offMN evaluation.
  const MCur<128> cM128(tlo);       // SB, XB, UU   (k = tlo)
  const MCur< 64> cM64 (tlo);       // B1           (k = tlo)
  const KCur<128> cKlo (tlo);       // AH hi half, A1/A2 row rr0 = tlo
  const KCur<128> cKhi (tlo + CB);  // A1/A2 padding row
  const KCur<128> cKt  (tlo - CB);  // AH lo half
  int oKQk[TPT], oKQq[TPT], oKB[TPT], oKBT[TPT];
  CUTE_UNROLL
  for (int i = 0; i < TPT; i++) {
    oKQk[i] = offK<128>(t0 + i, d0);
    oKQq[i] = offK<128>(CB + t0 + i, d0);
    oKB [i] = offK< 64>(t0 + i, d0);
  }
  CUTE_UNROLL
  for (int i = 0; i < TPT; i++) oKBT[i] = offMN<128>(d0, t0 + i);

  // Per-thread global cursors: the chunk loop then advances them by a constant
  // instead of recomputing a 64-bit index (an IMAD.WIDE pair) per load.
  const size_t gstride = (size_t)H * DD;
  const size_t g0off   = (size_t)(bos + t0) * gstride + (size_t)h * DD + d0;
  const bf16* gQ = Q + g0off;
  const bf16* gK = K + g0off;
  const bf16* gG = G + g0off;
  const bf16* gB = BETA + (size_t)bos * H + h;

  const float Aexp = __expf(A_LOG[h]);
  float dtb[CPT];
  CUTE_UNROLL
  for (int c = 0; c < CPT; c++) dtb[c] = DT_BIAS[h * DD + d0 + c];


  for (int c0 = 0; c0 < L; c0 += 32, gQ += 32 * gstride, gK += 32 * gstride,
                                     gG += 32 * gstride, gB += 32 * H) {
#if !defined(MMA_ONLY) && SKIP != 1
    // ============================ prologue ============================
    // Issue every global load for this chunk up front so their latency
    // overlaps (only 8 warps are resident, so dependent loads stall badly).
    // beta's load is issued first and consumed last.  gB is strided by H, so
    // each of the 64 issuing lanes touches its own 32-byte sector and the
    // load is long-latency; it used to sit immediately before barrier 3,
    // where all sixteen warps then waited on it.  Issued here its latency is
    // covered by the whole prologue, and only one register is held.
    const bf16 bv = (tid < 32 && c0 + tid < L) ? gB[(size_t)tid * H] : bf16(0.f);
    float cs[TPT][CPT], run[CPT];
    CUTE_UNROLL
    for (int c = 0; c < CPT; c++) run[c] = 0.f;
    CUTE_UNROLL
    for (int i = 0; i < TPT; i++) {
      CUTE_UNROLL
      {
      const bool valid = t0 + i < 32 && c0 + t0 + i < L;
      const bf16xC gv = valid ? *reinterpret_cast<const bf16xC*>(gG + i * gstride) : bf16xC{};
      CUTE_UNROLL
#if SKIP == 15
      for (int c = 0; c < CPT; c++) { run[c] += float(gv.v[c]); cs[i][c] = run[c]; }
#else
      for (int c = 0; c < CPT; c++) { run[c] += valid ? lb2 * sigmoid_fast(Aexp * (float(gv.v[c]) + dtb[c])) : 0.f; cs[i][c] = run[c]; }
#endif
      }
    }
    CUTE_UNROLL
    for (int j = 0; j < CPT / 4; j++)
      *reinterpret_cast<float4*>(&shTOT[tg][d0 + 4 * j]) =
          make_float4(run[4 * j], run[4 * j + 1], run[4 * j + 2], run[4 * j + 3]);
    BAR(1);
    // one warp turns the per-group totals into an exclusive prefix so that every
    // other thread only has to read its own (pre, half) pair
    if (tg == 0) {
      float run2[CPT];
      CUTE_UNROLL
      for (int c = 0; c < CPT; c++) run2[c] = 0.f;
      CUTE_UNROLL
      for (int w = 0; w < NPG; w++) {
        CUTE_UNROLL
        for (int j = 0; j < CPT / 4; j++) {
          const float4 t4 = *reinterpret_cast<const float4*>(&shTOT[w][d0 + 4 * j]);
          *reinterpret_cast<float4*>(&shTOT[w][d0 + 4 * j]) =
              make_float4(run2[4 * j], run2[4 * j + 1], run2[4 * j + 2], run2[4 * j + 3]);
          run2[4 * j] += t4.x; run2[4 * j + 1] += t4.y;
          run2[4 * j + 2] += t4.z; run2[4 * j + 3] += t4.w;
        }
      }
      CUTE_UNROLL
      for (int j = 0; j < CPT / 4; j++)
        *reinterpret_cast<float4*>(&sh.hfv[d0 + 4 * j]) =
            make_float4(0.5f * run2[4 * j], 0.5f * run2[4 * j + 1],
                        0.5f * run2[4 * j + 2], 0.5f * run2[4 * j + 3]);
      CUTE_UNROLL
      for (int c = 0; c < CPT; c++) sh.Ev[d0 + c] = exp2_fast(0.5f * run2[c]);
    }
    BAR(2);
    float (&e)[TPT][CPT] = cs;     // e overwrites cs in place
    {
      float pre[CPT], hf[CPT];
      CUTE_UNROLL
      for (int j = 0; j < CPT / 4; j++) {
        const float4 p4 = *reinterpret_cast<const float4*>(&shTOT[tg][d0 + 4 * j]);
        const float4 h4 = *reinterpret_cast<const float4*>(&sh.hfv[d0 + 4 * j]);
        pre[4 * j] = p4.x; pre[4 * j + 1] = p4.y; pre[4 * j + 2] = p4.z; pre[4 * j + 3] = p4.w;
        hf [4 * j] = h4.x; hf [4 * j + 1] = h4.y; hf [4 * j + 2] = h4.z; hf [4 * j + 3] = h4.w;
      }
      CUTE_UNROLL
      for (int i = 0; i < TPT; i++)
        CUTE_UNROLL
#if SKIP == 14
        for (int c = 0; c < CPT; c++) e[i][c] = cs[i][c] + pre[c] - hf[c];
#else
        for (int c = 0; c < CPT; c++) e[i][c] = exp2_fast(cs[i][c] + pre[c] - hf[c]);
#endif
    }
    // The l2-norm reductions are done for all TPT tokens at once so the five
    // shuffle rounds have TPT-way ILP instead of forming TPT dependent chains.
    float sk[TPT], sq[TPT];
    {   // k -> kt (KQ rows 0..63), kb (KB), kb^T (KBT)
      CUTE_UNROLL
      for (int i = 0; i < TPT; i++) {
#if SKIP == 21
        bf16xC kv, qv2;
        CUTE_UNROLL
        for (int c = 0; c < CPT; c++) { kv.v[c] = bf16(scale); qv2.v[c] = bf16(lb2); }
#else
        const bf16xC kv = (t0 + i < 32 && c0 + t0 + i < L) ? *reinterpret_cast<const bf16xC*>(gK + i * gstride) : bf16xC{};
        const bf16xC qv2 = (t0 + i < 32 && c0 + t0 + i < L) ? *reinterpret_cast<const bf16xC*>(gQ + i * gstride) : bf16xC{};
#endif
        float s = 0.f, sq2 = 0.f, fa[CPT], fb[CPT], fq[CPT];
        CUTE_UNROLL
        for (int c = 0; c < CPT; c++) {
          const float ei = e[i][c], re = rcp_fast(ei);   // shared by k and q
          const float x = float(kv.v[c]), xq = float(qv2.v[c]);
          s += x * x; sq2 += xq * xq;
          fa[c] = x * ei; fb[c] = x * re; fq[c] = xq * ei;
        }
        sk[i] = s; sq[i] = sq2;
#if SKIP == 20
        { bf16xC z = packC(fq), y = packC(fb), x = packC(fa);
          if (tid == 0xffff) { sh.KQ[0] = z.v[0]; sh.KB[0] = y.v[0]; sh.KBT[0] = x.v[0]; } }
#else
        *reinterpret_cast<bf16xC*>(&sh.KQ[oKQq[i]]) = packC(fq);
        const bf16xC b = packC(fb);
        *reinterpret_cast<bf16xC*>(&sh.KBT[oKBT[i]]) = b;   // channel-contiguous
        *reinterpret_cast<bf16xC*>(&sh.KQ[oKQk[i]]) = packC(fa);
        *reinterpret_cast<bf16xC*>(&sh.KB[oKB[i]]) = b;
#endif
      }
    }
#if SKIP != 12
    CUTE_UNROLL
    for (int off = CGR / 2; off; off >>= 1) {
      CUTE_UNROLL
      for (int i = 0; i < TPT; i++) sk[i] += __shfl_xor_sync(0xffffffffu, sk[i], off);
      CUTE_UNROLL
      for (int i = 0; i < TPT; i++) sq[i] += __shfl_xor_sync(0xffffffffu, sq[i], off);
    }
    if (cg == 0) {
      CUTE_UNROLL
      for (int i = 0; i < TPT; i++) {
        sh.nk[t0 + i] = rsqrt_fast(sk[i] + 1e-12f);
        sh.nq[t0 + i] = rsqrt_fast(sq[i] + 1e-12f) * scale;
      }
    }
#endif
    if (tid < CB) sh.bs[tid] = (tid < 32 && c0 + tid < L) ? sigmoid_fast(float(bv)) : 0.f;   // load issued above
    SBAR(3);

#endif
    // Issue the Gram MMA first: it only needs KQ/KB, so the state rescale below
    // (which produces the operand of the second MMA) overlaps with it.
    if (lead) issue_mma(mmaG, frKQ_G, frKB_G, tGram, false);
#if !defined(MMA_ONLY) && SKIP != 2
    // The previous chunk's MMA 5 wrote tState; this is the first reader, so
    // its deferred barrier is collected here.  Skipped on the first chunk,
    // where no MMA 5 has been issued.
    // ============ scale S^T by Ev*Eprev (row = Kdim), dump to SB ============
    {
      const float fac = sh.Ev[tlo] * sh.Eprev[tlo];
      TCHUNK(128) {
        TLOAD(tb + TM_STATE);
        CUTE_UNROLL
        for (int i = 0; i < TCOLS; ++i) rr[i] *= fac;
        CUTE_UNROLL
        for (int j = 0; j < TJ; ++j) {
          *reinterpret_cast<bf16x8*>(&shSB[cM128(C0 + j * 8)]) = pack8(&rr[j * 8]);
        }
        TSTORE(tb + TM_STATE);
      }
    }
    SBAR(4);
#endif
    for (int i = tid; i < DD; i += NTHR) sh.Eprev[i] = sh.Ev[i];

    // ====== MMA 2: [kt;qt] S ; one wait covers both MMAs ======
    if (lead) { issue_mma(mmaB, frKQ_B, frSB_B, tKQS, false); cutlass::arch::umma_arrive(ctl.cur()); }
#if !defined(MMA_ONLY) && SKIP != 4
    // The V prefetch moved here from after barrier 5.  MMA 2's latency is the
    // one fully exposed wait in the chunk -- the state rescale covers MMA 1
    // (removing it, SKIP=2, is free), but nothing occupied the gap between
    // the MMA 2 issue and this wait.  V is independent of MMA 2, its loads
    // are uncoalesced (thread tlo owns token tlo, so consecutive lanes are
    // H*DD apart) and therefore long-latency, and Xb does not need them
    // until three barriers later.
    bf16x8 vpf[(128 / TCOLS + NTG - 1) / NTG][TJ];
    if (tlo < CB) {
      const bf16* vp0 = V + (size_t)(bos + c0 + tlo) * H * DD + h * DD;
      int t = 0;
      TCHUNK(128) {
        CUTE_UNROLL
        for (int j = 0; j < TJ; ++j) vpf[t][j] = (tlo < 32 && c0 + tlo < L) ? *reinterpret_cast<const bf16x8*>(vp0 + C0 + j * 8) : bf16x8{};
        t++;
      }
    }
#endif
    ctl.wait();

#if !defined(MMA_ONLY) && SKIP != 3
    {   // -> N = -P  (diag blocks to shND, block-lower to B2) and Ah
      const int r = tlo;
      TCHUNK(64) {
        TLOAD_ISSUE(tb + TM_GRAM);
        // beta and the nk factors do not depend on the Gram values, so they
        // are read inside the tcgen05.ld's latency window instead of after
        // the wait.  The Gram post is 7.3% of the kernel with a single
        // exposed TMEM load (one TCHUNK iteration, so nothing to batch it
        // with), which is the same situation the V prefetch exploited at
        // MMA 2.
        // Branch-free: tcgen05.ld and tcgen05.wait::ld are warp-scoped, so no
        // divergence may sit between them.  r >= CB would index nk/bs out of
        // range, so the index is clamped; those lanes' w0 is unused.
        const int rc = (r < CB) ? r : 0;
        const float w0 = sh.bs[rc] * sh.nk[rc];
        float nkv[TCOLS];
        CUTE_UNROLL
        for (int i = 0; i < TCOLS; ++i) nkv[i] = sh.nk[C0 + i];
        TLOAD_WAIT();
        if (r < CB) {
          CUTE_UNROLL
          for (int i = 0; i < TCOLS; ++i) {
            int s2 = C0 + i;
            float v = (r > s2) ? -(w0 * nkv[i] * rr[i]) : 0.f;
            if ((r >> 5) == (s2 >> 5)) { shND[r >> 5][r & 31][s2 & 31] = v; rr[i] = 0.f; }
            else rr[i] = v;
          }
          CUTE_UNROLL
          for (int j = 0; j < TJ; ++j) {   // B MN-major: [k=r][n=s2]
            *reinterpret_cast<bf16x8*>(&shB1[cM64(C0 + j * 8)]) = pack8(&rr[j * 8]);
          }
        } else {
          const int t = r - CB;
          CUTE_UNROLL
          for (int j = 0; j < TJ; ++j) {
            float d1[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; ++i) d1[i] = (t >= C0 + j * 8 + i) ? rr[j * 8 + i] : 0.f;
            // AH's lower 64 rows are not zeroed.  MMA 4 multiplies them into
            // tKQS rows 0..63, which held KtS -- already consumed by Xb and
            // never read again (the output stage reads rows 64..127, and the
            // next chunk's MMA 2 overwrites the whole accumulator with
            // acc=false).  Whatever garbage is there is harmless.
            st8 (&sh.A2[cKlo(C0 + j * 8)], d1);   // AH rows 64..127
          }
        }
      }
    }
    // nd is published to *threads* (the inverse); B1 and AH are read by MMAs
    // only at dot1 and MMA 4, and the fenced barriers before those cover
    // them.  No proxy fence needed here.
    BAR(5);
#endif

// SKIP=4 covers the V prefetch (now issued in MMA 2's window, above, under a
// second region with the same number) together with the inverse and Xb below,
// so its ablation measures all three plus barrier 6 -- not a single stage.
#if !defined(MMA_ONLY) && SKIP != 4
    // ---- warp-local inverse of the two 32x32 diagonal blocks: Dm = (I-Nd)^-1
    //      warp b owns block b; lane j owns column j of that block.
    // The zero-fill runs *first*.  col[32] is private to 2 of the 8 warps but
    // registers are allocated uniformly across the CTA, so letting its live
    // range span the zero-fill and its barrier charged every thread 32
    // registers.  Confining it to a barrier-free scope is what makes the
    // 128-register budget of a second resident CTA reachable.
    {
      for (int r = tid; r < CB; r += NTHR) {             // only the 64 real rows
        const KCur<128> ca(r);
        const MCur< 64> cb(r);
        const int dlo = (r < CB) ? ((r >> 5) << 5) : CB;   // diagonal block of row r
        CUTE_UNROLL
        for (int j = 0; j < CB; j += 8) {
          if (j >= dlo && j < dlo + 32) continue;          // the inverse fills this
          st8z(&shA1[ca(j)]);
          st8z(&shB2[cb(j)]);
        }
      }
      // The inverse needs 64 threads: two 32x32 diagonal blocks, one lane per
      // column.  It used the prologue's tg/cg, which encoded warp and lane
      // only because that map had CGR = 32; it must index tid directly.
      if (tid < 64) {
        const int b = tid >> 5, lane = tid & 31;
        float col[32];
#if SKIP == 9
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) col[i] = (i == lane) ? 1.f : 0.f;
#else
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) col[i] = (i == lane) ? 1.f : 0.f;
        CUTE_UNROLL
        for (int i = 1; i < 32; ++i) {
          float a0 = col[i], a1 = 0.f;
          CUTE_UNROLL
          for (int j = 0; j < 32; ++j)
            if (j < i) { if (j & 1) a1 += shND[b][i][j] * col[j]; else a0 += shND[b][i][j] * col[j]; }
          col[i] = a0 + a1;
        }
#endif
        CUTE_UNROLL
        for (int i = 0; i < 32; ++i) {
          const int r = b * 32 + i, c = b * 32 + lane;
          shA1[offK<128>(r, c)] = bf16(col[i]);
          shB2[offMN<64>(c, r)] = bf16(col[i]);
        }
      }
    }
#if !defined(MMA_ONLY) && SKIP != 6
    // Xb = diag(beta)(V - diag(nk) KtS)  -> B MN-major [k=s][n=V]
    {
      const int r = tlo;
      // Xb runs here, inside the block inverse's window.  The inverse's
      // substitution loop is 3.4% of the kernel (SKIP=9) on 2 of 16 warps,
      // so fourteen warps were idle through it; Xb reads tKQS (MMA 2, long
      // done) and the prefetched V, so it is independent of the inverse, and
      // two barriers separated them so ptxas could not move it itself.
      // Moving it earlier also *shortens* vpf's live range instead of
      // extending it, which is why this costs no registers.  Both of its
      // TMEM reads are issued before the single wait.
      if (r < CB) {
        TLOADN(tb + TM_KQS, 128);
        const float br = sh.bs[r], nkr = sh.nk[r];
        CUTE_UNROLL
        for (int q = 0; q < NCH(128); ++q) {
          CUTE_UNROLL
          for (int j = 0; j < TJ; ++j) {
            float d[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; ++i)
              d[i] = br * (float(vpf[q][j].v[i]) - nkr * rn[q][j * 8 + i]);
            st8(&sh.XB[cM128(C0N(q) + j * 8)], d);
          }
        }
      }
    }
#endif
    SBAR(6);
#endif

    const int rr0 = tlo;
    // dot1: M = Dm @ No   (block-strictly-lower with two blocks, so M^2 = 0)
    run_mma(mmaI, frA1_I, frB1_I, tInv, false, lead, ctl);
#if !defined(MMA_ONLY) && SKIP != 5
    if (rr0 < CB) TCHUNK(64) {
      TLOAD(tb + TM_INV);
      float d[8];
      CUTE_UNROLL
      for (int j = 0; j < TJ; ++j) {
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) d[i] = rr[j * 8 + i] + ((rr0 == C0 + j * 8 + i) ? 1.f : 0.f);
        st8(&sh.A2[cKlo(C0 + j * 8)], d);
        // A2 rows 64..127 are not zeroed: dot2 puts them in tInv rows
        // 64..127, and the A1 = T' stage reads only rr0 < CB.
      }
    }
    SBAR(7);
#endif
    // dot2: T = (I+M) @ Dm
    run_mma(mmaI, frA2_I, frB2_I, tInv, false, lead, ctl);

#if !defined(MMA_ONLY) && SKIP != 6
    { const float nkr = (tlo < CB) ? sh.nk[tlo] : 0.f;
      if (rr0 < CB) TCHUNK(64) {
        TLOAD(tb + TM_INV);
        float d[8];
        CUTE_UNROLL
        for (int j = 0; j < TJ; ++j) {
          CUTE_UNROLL
          for (int i = 0; i < 8; ++i) d[i] = nkr * rr[j * 8 + i];
          st8(&shA1[cKlo(C0 + j * 8)], d);
          // likewise A1 rows 64..127 -> tU rows 64..127, and the U' dump
          // reads only rr0 < CB.
        }
      }
    }
    SBAR(8);
#endif

    // ======================= MMA 3: U' = T' Xb =======================
    run_mma(mmaB, frA1_B, frXB_B, tU, false, lead, ctl);
#if !defined(MMA_ONLY) && SKIP != 7
    if (rr0 < CB) {
      TLOADN(tb + TM_U, 128);
      CUTE_UNROLL
      for (int q = 0; q < NCH(128); ++q) {
        CUTE_UNROLL
        for (int j = 0; j < TJ; ++j) st8(&sh.XB[cM128(C0N(q) + j * 8)], rn[q] + j * 8);
      }
    }
    SBAR(9);
#endif
    // ============ MMA 4: KQS += Ah U' ; MMA 5: S^T += kb^T U' ============
    // One wait covers both.  MMA 5's result (tState) is not read until the
    // next chunk's rescale, so its wait can be deferred a chunk onto its own
    // barrier -- tried, and it is 1% *slower*: MMA 5 then writes tState while
    // the output stage reads tKQS, and they contend for TMEM, whereas the
    // bundled wait serialises them for free.
    if (lead) {
      issue_mma(mmaB, frAH_B,  frUU_B, tKQS,   true);
      issue_mma(mmaS, frKBT_B, frUU_S, tState, true);
      cutlass::arch::umma_arrive(ctl.cur());
    }
    ctl.wait();
#if !defined(MMA_ONLY) && SKIP != 8
    // O = diag(nq*scale) KQS[64:128], staged through A2/AH, whose last reader
    // is MMA 4 -- already waited on just above.  It used to stage through UU,
    // but UU is MMA 5's B operand and MMA 5's wait is now deferred to the
    // next chunk, so overwriting UU here would race with it.  A2 is free at
    // this point (dot2 read it, and the next chunk's Gram post does not write
    // AH until after barrier 11) and costs no extra shared memory.
    // Writing straight to global from this thread map is uncoalesced: thread
    // tlo owns token tlo-CB, so consecutive lanes are H*DD apart and every
    // lane lands in its own 32-byte sector -- 32 transactions per warp.
    // Staging through shared memory and re-reading with 16 lanes per token
    // makes each token's 256 bytes one contiguous store: 2 transactions.
    {
      const int r = tlo;
      if (r >= CB) {
        const float nqr = sh.nq[r - CB];
        const MCur<128> cO(r - CB);
        TCHUNK(128) {
          TLOAD(tb + TM_KQS);
          CUTE_UNROLL
          for (int j = 0; j < TJ; ++j) {
            float d[8];
            CUTE_UNROLL
            for (int i = 0; i < 8; ++i) d[i] = nqr * rr[j * 8 + i];
            st8(&sh.A2[cO(C0 + j * 8)], d);
          }
        }
      }
    }
    BAR(10);  // A2 staging is thread -> thread; no MMA reads it here
    {   // 16 lanes per token: one 256-byte contiguous store per token
      const int lane = tid & 15, tok0 = tid >> 4;
      CUTE_UNROLL
      for (int t = tok0; t < 32 && c0 + t < L; t += NTHR / 16) {
        const MCur<128> cO(t);
        const bf16x8 w = *reinterpret_cast<const bf16x8*>(&sh.A2[cO(lane * 8)]);
        *reinterpret_cast<bf16x8*>(OUT + (size_t)(bos + c0 + t) * H * DD
                                       + h * DD + lane * 8) = w;
      }
    }
    // This barrier is not needed for correctness -- the output stage only
    // reads TMEM and sh.nq, neither of which the next prologue touches before
    // its own first two barriers.  It is kept because removing it costs 8%:
    // without it the warps drift apart across the chunk boundary and the next
    // chunk's global loads lose locality (long_scoreboard 2.84 -> 3.87).
    // No fence: the next MMA to read anything is MMA 1, behind the prologue's
    // own fenced barrier.
    BAR(11);
#endif
  }

  // ---- final state: S[v][k] = Eprev[k] * S^T[k][v]
  {
    bf16* sp = STATE + ((size_t)nseq * H + h) * DD * DD;
    const float ep = sh.Eprev[tlo];
    TCHUNK(128) {
      TLOAD(tb + TM_STATE);
      CUTE_UNROLL
      for (int i = 0; i < TCOLS; ++i) sp[(C0 + i) * DD + tlo] = bf16(rr[i] * ep);
    }
  }
  smem_sync();
  if (lead) { alloc.release_allocation_lock(); alloc.free(sh.tmem_ptr, TAlloc::Sm100TmemCapacityColumns); }
}

extern "C" {
int kda_smem_bytes() { return (int)sizeof(Shm); }
void kda_fwd_launch(const void* Q, const void* K, const void* V, const void* G, const void* B,
                    const float* A, const float* DT, void* STATE, void* OUT, const void* CU,
                    int H, int N, float lower_bound, float scale, cudaStream_t st,
                    int stop_at, void* DBG) {
  (void)stop_at; (void)DBG;
  static bool init = false;
  size_t sm = sizeof(Shm);
  if (!init) { cudaFuncSetAttribute(kda_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); init = true; }
  kda_fwd_kernel<<<dim3(H, N), NTHR, sm, st>>>((const bf16*)Q, (const bf16*)K, (const bf16*)V,
      (const bf16*)G, (const bf16*)B, A, DT, (bf16*)STATE, (bf16*)OUT, (const long*)CU, H,
      lower_bound * 1.4426950408889634f, scale);
}
}
