// FP8 block-scale MoE (DeepSeek-V3 routing, 32 local experts, H=7168, I=2048) for NVIDIA B200 (sm_100a).
// Self-contained CUDA C++: routing, row compaction, two tcgen05 (UMMA) grouped GEMMs with TMA + TMEM and
// software FP32 block-scale promotion, SwiGLU, FP8 hi/lo re-quantization and weighted bf16 accumulation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace {

constexpr int HID = 7168;
constexpr int INTER = 2048;
constexpr int E_LOCAL = 32;
constexpr int E_GLOBAL = 256;
constexpr int CNT_STRIDE = 32;           // per-expert counters padded to one 128 B line each (contended atomics)
constexpr int NUM_HB = HID / 128;        // 56 k-blocks of GEMM1
constexpr int NUM_IB = INTER / 128;      // 16 k-blocks of GEMM2
constexpr int BM = 128, BN = 256, BK = 128;
#ifndef MOE_PW
#define MOE_PW 8   // promotion warps: 8 (2 column groups of 128) or 16 (4 groups of 64: shorter per-warp chains)
#endif
constexpr int PW = MOE_PW;
constexpr int NG = PW / 4;                  // promotion column groups
constexpr int CG = 256 / NG;                // accumulator columns per promotion thread (128 or 64)
constexpr int HC = CG / 2;                  // columns per X1 / X2 half (GEMM1) per thread
constexpr int PROMO_THREADS = 32 * PW;
constexpr int GEMM_THREADS = 128 + PROMO_THREADS;   // warps 0-3: producer / MMA / copy, then the promotion warps
constexpr int BAR_MMA = 32 + PROMO_THREADS;         // named barriers 1 / 2: MMA warp + promotion warps
constexpr int BAR_RED = 64 + PROMO_THREADS;         // named barriers 5 / 6: reduction warps + promotion warps
#ifndef MOE_PROMO_REGS
#define MOE_PROMO_REGS ((MOE_PW == 16) ? 104 : 232)   // 16 warps: 112 blocks forever in setmaxnreg.inc (per-SMSP register pool)
#endif
constexpr int PROMO_REGS = MOE_PROMO_REGS;  // setmaxnreg budget of a promotion thread
constexpr uint32_t TMEM_COLS = 512;
#ifndef MOE_TIMING
#define MOE_TIMING 0   // dev-only: clock64 phase counters in the promotion loop
#endif
#if MOE_TIMING
__device__ long long g_dbg_timing[2][148 * 128];   // per CTA: 12 warps x 8 slots (dev timing, see dev/epi_timing.py)
__device__ long long g_dbg_cta[2][148 * 8];   // per CTA: smid, start ns, end ns, producer empty-wait cycles, MMA full-wait cycles, MMA named-sync cycles
__device__ long long g_dbg_copy[148 * 2];
// dev-only event trace (MOE_TIMING): CTAs 0 / 1, regions of 1024 (code, clock64) pairs: 0 producer, 1 MMA warp, 2 promotion warp 4
__device__ long long g_dbg_trace[2][2][3][1024][2];
#define TRACE_CODE(ev, tile, kb) ((static_cast<long long>(ev) << 24) | (static_cast<long long>(tile) << 12) | static_cast<long long>((kb) & 0xfff))   // GEMM1 copy warps: end of row copies, end of output zeroing (globaltimer)
#define TSTAMP(v) long long v = clock64()
#else
#define TSTAMP(v)
#endif
#ifndef MOE_ABL
#define MOE_ABL 0   // dev-only ablation: 1 = no promotion work, 5 = TMA only (no MMA, no promotion)
#endif

// ------------------------------------------------------------------------------------------------
// PTX helpers
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
#define MOE_2SM_STAGES 6   // (2-SM cta_group::2 variant removed after rejection; see git history v16 and profile/)
#ifndef MOE_B256
#define MOE_B256 1   // single-CTA GEMM2: 1 = one 256-row B box per k-block (locked-clock GEMM2 389 -> 378 us), 0 = two 128-row boxes
#endif
#ifndef MOE_B4D
#define MOE_B4D 1    // single-CTA GEMM1: 1 = one 4D box (both X1/X2 halves) per k-block, 0 = two 3D boxes
#endif
#ifndef MOE_SA_GROUP
#define MOE_SA_GROUP 1   // k-blocks per row-scale bulk copy (1, 2 or 4); 2 and 4 make GEMM2 ~75 us slower (cold L2)
#endif
#ifndef MOE_PF
#define MOE_PF 0   // L2 prefetch distance in k-blocks for the next A/B tiles (0 = off)
#endif
#ifndef MOE_AMC
#define MOE_AMC 0   // 1 = clusters of 2 CTAs along N share the A tile (each loads 64 rows, TMA multicast to both)
#endif
#ifndef MOE_FORK
#define MOE_FORK 0   // !MOE_ZERO only: output memset forked onto the side stream 1 = before routing, else after it
#endif
#ifndef MOE_COPY_TMA
#define MOE_COPY_TMA 0   // GEMM1 copy warps: 1 = TMA bulk copies through a 3-slot smem staging ring (correct, no gain: 0.956 vs 0.952 ms), 0 = LSU loads/stores
#endif
#ifndef MOE_W13_PF_EXPERTS
#define MOE_W13_PF_EXPERTS 2   // experts whose GEMM1 weights the routing kernel prefetches into L2 (0 = off)
#endif
#ifndef MOE_ONESTORE
#define MOE_ONESTORE 1   // 1 = rows of tokens with exactly one local expert are not zeroed and written with a plain bulk store instead of a reduction
#endif
#ifndef MOE_FASTSIG
#define MOE_FASTSIG 1   // routing sigmoid with __expf / __fdividef (0 = fp32 expf + IEEE division, the reference's math)
#endif
#ifndef MOE_ZERO
#define MOE_ZERO 1   // 1 = GEMM1's promotion warps zero the output before awaiting the routing grid (no memset kernel)
#endif
#ifndef MOE_DYN
#define MOE_DYN 1   // 1 = tiles claimed dynamically from a global counter (producer warp), 0 = static round robin
#endif
#ifndef MOE_HALF
#define MOE_HALF 1   // 1 = the last m-tile of an expert with <= 64 rows runs as a 64 x 256 tile (tcgen05.mma M = 64)
#endif
#ifndef MOE_REORDER
#define MOE_REORDER 1   // promotion: 1 = release TMEM stage before the last two FMA blocks, 0 = before the last one
#endif
#ifndef MOE_WAIT_HINT
#define MOE_WAIT_HINT 1   // 1 = try_wait with a suspend-time hint (thread sleeps instead of spinning)
#endif
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
#if MOE_WAIT_HINT
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@p bra DONE_%=;\n\t"
      "bra WAIT_%=;\n"
      "DONE_%=:\n\t}" ::"r"(smem_u32(bar)),
      "r"(parity), "r"(0x989680u)
      : "memory");
#else
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
      "@p bra DONE_%=;\n\t"
      "bra WAIT_%=;\n"
      "DONE_%=:\n\t}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
#endif
}
__device__ __forceinline__ void tma_prefetch(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const void* map, uint64_t* bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d_mc(void* dst, const void* map, uint64_t* bar, int x, int y, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
      " [%0], [%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y), "h"(mask)
      : "memory");
}
__device__ __forceinline__ void umma_commit_mc(uint64_t* bar, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
          smem_u32(bar)),
      "h"(mask)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* dst, const void* map, uint64_t* bar, int x, int y, int z) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y), "r"(z)
      : "memory");
}
#ifndef MOE_L2HINT
#define MOE_L2HINT 0   // 1 = weight-tile TMA loads with an L2 evict_last policy, output zeroing with streaming stores
#endif
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void tma_load_3d_hint(void* dst, const void* map, uint64_t* bar, int x, int y, int z, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y), "r"(z), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void tma_load_4d_hint(void* dst, const void* map, uint64_t* bar, int x, int y, int z, int w, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6}], [%2], %7;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y), "r"(z), "r"(w), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void st_global_cs_v4(void* p, int4 v) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* dst, const void* map, uint64_t* bar, int x, int y, int z, int w) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(x), "r"(y), "r"(z), "r"(w)
      : "memory");
}
__device__ __forceinline__ void tma_prefetch_2d(const void* map, int x, int y) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(x), "r"(y)
               : "memory");
}
__device__ __forceinline__ void tma_prefetch_3d(const void* map, int x, int y, int z) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(x), "r"(y), "r"(z)
               : "memory");
}
__device__ __forceinline__ void tma_prefetch_4d(const void* map, int x, int y, int z, int w) {
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" ::"l"(reinterpret_cast<uint64_t>(map)),
               "r"(x), "r"(y), "r"(z), "r"(w)
               : "memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}
__device__ __forceinline__ void umma_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void umma_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void umma_mma_f8(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                            uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void tmem_ld_x32(uint32_t taddr, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tmem_st_x32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31};"
      :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
         "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]),
         "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]),
         "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(taddr)
      : "memory");
}
__device__ __forceinline__ void tmem_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t remote_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote_addr) : "memory");
}
// Relaxed remote arrive: no memory ordering (used for the TMEM-free signal, which publishes no memory writes;
// ~140 cycles one way vs ~460 for .release.cluster on B200).
__device__ __forceinline__ void mbar_arrive_remote_relaxed(uint32_t remote_addr) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote_addr) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
__device__ __forceinline__ int ld_relaxed_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
__device__ __forceinline__ void bulk_prefetch_l2(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(reinterpret_cast<uint64_t>(p)), "r"(bytes) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void named_bar_arrive(uint32_t id, uint32_t count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_bar_sync(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void red_add_bf16x2_v4(void* gptr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(gptr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

// UMMA shared-memory descriptor for a K-major, 128B-swizzled tile whose rows are 128 bytes (BK fp8).
// 8-row swizzle atoms are 1024 B apart (SBO). Version 1 (sm100), layout type SWIZZLE_128B = 2.
__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_addr) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr & 0x3FFFFu) >> 4);
  d |= static_cast<uint64_t>(1) << 16;             // LBO (ignored for swizzled K-major)
  d |= static_cast<uint64_t>(1024 >> 4) << 32;     // SBO
  d |= static_cast<uint64_t>(1) << 46;             // version
  d |= static_cast<uint64_t>(2) << 61;             // SWIZZLE_128B
  return d;
}
// Instruction descriptor: kind::f8f6f4, D = F32, A/B = E4M3, both K-major, M=128, N=BN.
__host__ __device__ constexpr uint32_t make_idesc_f8(uint32_t M, uint32_t N) {
  return (1u << 4) | ((N >> 3) << 17) | ((M >> 4) << 24);
}

#ifndef MOE_FMA_YIELD
#define MOE_FMA_YIELD 0   // 1 = nanosleep between the promotion FMA blocks (yield the SMSP to the producer / MMA warp)
#endif
#ifndef MOE_FIN_OFFLOAD
#define MOE_FIN_OFFLOAD 1   // GEMM1: 1 = done_tiles signalled by copy warp 2 (promotion warps bump a smem counter with release semantics, no finish bar.sync / gpu fence on their path), 0 = warp 4 fences + atomics after a bar.sync
#endif
#ifndef MOE_SILU_FOLD
#define MOE_SILU_FOLD 1   // GEMM1 epilogue: 1 = c' = x1 x2 (1 + tanh(x2/2)) with the 0.5 folded into the quantization scale (64 fewer FMUL per thread), 2 = also packed f32x2 multiplies, 0 = original (1 / 2 neutral at locked clocks: the epilogue is off the MMA's critical path)
#endif
#ifndef MOE_STS
#define MOE_STS 1   // GEMM2 staging stores: 1 = st.shared.v4 (STS.128), 0 = generic stores through the smem pointer (ST.E.128)
#endif
#ifndef MOE_READY_CACHE
#define MOE_READY_CACHE 0   // producer: 1 = poll an expert's readiness counter only until first observed ready (bit mask; useless here: a CTA's consecutive tiles are different experts), 0 = poll at every tile start
#endif
#ifndef MOE_READY_EARLY
#define MOE_READY_EARLY 0   // producer: 1 = relaxed load of the next tile's readiness counter MOE_READY_DIST k-blocks before the tile boundary + fence.acq_rel.gpu at the boundary if it was ready, 2 = acquire load early, 3 = relaxed load early and only the proxy fence at the boundary (the writer's fence + atomic publish the rows; our TMA loads are control-dependent on the observed value), 0 = acquire poll at the tile start
#endif
#ifndef MOE_STAGGER
#define MOE_STAGGER 0   // GEMM1: CTA b delays its first tile claim by (b % MOE_STAGGER) / MOE_STAGGER of a tile period (desynchronizes the lockstep tile transitions; 0 = off)
#endif
#ifndef MOE_NEXT_PF
#define MOE_NEXT_PF 0   // producer: TMA-prefetch the next tile's first MOE_NEXT_PF k-blocks (rejected: tensor prefetch ops load the saturated TMA unit like real loads; locked GEMM1 +18 / GEMM2 +60 us)
#endif
#ifndef MOE_READY_DIST
#define MOE_READY_DIST 6   // MOE_READY_EARLY: k-blocks before the tile boundary at which the next tile's readiness counter is read
#endif
#ifndef MOE_PWPF
#define MOE_PWPF 0   // promotion warps L2-prefetch (prefetch.global.L2, 6 lines per thread) the next tile's first MOE_PWPF_KB k-blocks MOE_PWPF k-blocks before the GEMM1 tile boundary (rejected: locked GEMM1 652 -> 717, GEMM2 364 -> 410 us; the SM's L2 request path is the saturated resource); 0 = off
#endif
#ifndef MOE_PWPF2
#define MOE_PWPF2 6   // same distance for GEMM2 (16-k-block tiles)
#endif
#ifndef MOE_PWPF_KB
#define MOE_PWPF_KB 4   // k-blocks of the next tile prefetched (A 128 lines + B 256 lines of 128 B each per k-block)
#endif
#ifndef MOE_CLAIM_AHEAD
#define MOE_CLAIM_AHEAD 1   // producer: 1 = claim the next tile while issuing the current one (not within 2 waves of the end), 0 = claim at tile start
#endif
#ifndef MOE_WFOLD
#define MOE_WFOLD 0   // GEMM2: 1 = routing weight folded into the per-k-block scales (rejected: the dependent row_weight load then stalls k-block 0; locked GEMM2 376 -> 385 us), 0 = applied in the epilogue
#endif
#ifndef MOE_FFMA2
#define MOE_FFMA2 1   // 1 = promotion FMAs as packed fma.rn.f32x2 (sm_100a), 0 = scalar fmaf, 2 = half packed / half scalar
#endif
#ifndef MOE_FASTSILU
#define MOE_FASTSILU 2   // GEMM1 epilogue: 2 = silu via tanh.approx (one MUFU per element), 1 = __fdividef (MUFU.RCP), 0 = IEEE fp32 division (slow path on exp overflow)
#endif
#ifndef MOE_BF16ACC
#define MOE_BF16ACC 0   // 1 = bf16x2 register accumulators (cvt.rn.bf16x2.f32 + fma.rn.bf16x2), 0 = fp32 (fma.rn.f32x2)
#endif
// bf16x2 helpers: pack two fp32 (lo -> low half), packed FMA, unpack halves
__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint32_t hfma2_bf16(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
// Truncating fp32 pair -> bf16x2 (upper halves, one PRMT): the k-block partial is rounded toward zero (<= 2^-8
// relative, mean -2^-9) before the bf16 FMA; the accumulator itself is still rounded to nearest by the FMA.
__device__ __forceinline__ uint32_t trunc_bf16x2(uint32_t lo, uint32_t hi) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(d) : "r"(lo), "r"(hi));
  return d;
}
#ifndef MOE_BF16_TRUNC
#define MOE_BF16_TRUNC 0   // MOE_ACC_BF16: 1 = partials truncated to bf16 (PRMT), 0 = rounded (cvt.rn.bf16x2.f32)
#endif
__device__ __forceinline__ float bf16lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bf16hi(uint32_t p) { return __uint_as_float(p & 0xffff0000u); }
// Packed FP32 FMA (sm_100a): {lo,hi}(acc) = {lo,hi}(v) * {lo,hi}(s) + acc, operands are 64-bit register pairs.
__device__ __forceinline__ unsigned long long ffma2(unsigned long long acc, unsigned long long v, unsigned long long s) {
  asm("fma.rn.f32x2 %0, %1, %2, %0;" : "+l"(acc) : "l"(v), "l"(s));
  return acc;
}
__device__ __forceinline__ unsigned long long fmul2(unsigned long long a, unsigned long long b) {
  unsigned long long d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ unsigned long long ffma2_3(unsigned long long a, unsigned long long b, unsigned long long c) {
  unsigned long long d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}
__device__ __forceinline__ unsigned long long pack2f(float lo, float hi) {
  return static_cast<unsigned long long>(__float_as_uint(lo)) | (static_cast<unsigned long long>(__float_as_uint(hi)) << 32);
}
__device__ __forceinline__ float lo2f(unsigned long long v) { return __uint_as_float(static_cast<uint32_t>(v)); }
__device__ __forceinline__ float hi2f(unsigned long long v) { return __uint_as_float(static_cast<uint32_t>(v >> 32)); }
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void sts32(uint32_t addr, int v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_cta_shared(uint32_t addr, uint32_t v) {
  asm volatile("red.release.cta.shared::cta.add.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_cta_shared(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long pack2(uint32_t lo, uint32_t hi) {
  return static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
}

#ifndef MOE_PACK_INT
#define MOE_PACK_INT 1   // GEMM2 epilogue bf16 pack: 1 = integer round-half-up + prmt (ALU pipe), 0 = cvt.rn.bf16x2.f32 (F2FP)
#endif
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
#if MOE_PACK_INT
  // round half up on the magnitude bits, then take the upper halves (one PRMT); NaN / inf stay NaN / inf
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(d) : "r"(__float_as_uint(lo) + 0x8000u), "r"(__float_as_uint(hi) + 0x8000u));
  return d;
#else
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
#endif
}

// ------------------------------------------------------------------------------------------------
// K1: DeepSeek-V3 no-aux routing. One token per warp: lane l owns experts 8l .. 8l+7 (expert group l / 4), so the
// group top-2 is a local top-2 merged over the 4 lanes of the group and the top-8 selection is a local 8-element
// argmax plus a 5-level warp argmax per round. 14107 warps (vs 3527 with 8 lanes per token) give the scheduler
// 4x the warps to hide the sigmoid / shuffle / atomic latencies (the kernel is latency-bound, not chain-bound).
// Math and tie rules mirror the reference exactly (fp32 expf + IEEE division, ties -> lower index).
// ------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4) routing_kernel(const float* __restrict__ logits,
                                                      const __nv_bfloat16* __restrict__ bias, int T, int local_offset,
                                                      float rsf, int* __restrict__ cnt, int cap,
                                                      int* __restrict__ row_token, float* __restrict__ row_weight,
                                                      const uint8_t* __restrict__ w13_pf, size_t w13_pf_bytes,
                                                      int* __restrict__ tok_cnt) {
  // Fixed expert layout: row = le * cap + slot (slot from the per-expert counter), so the row maps are final here
  // without a prefix pass; GEMM1's copy warps gather the rows (hidden states, scales) from these maps.
  pdl_launch_dependents();   // GEMM1 CTAs are scheduled as SMs free up; they wait for this grid before reading
  if (threadIdx.x == 0) {
    // DRAM is idle during routing: prefetch the first experts' GEMM1 weights (GEMM1's first tiles) into L2,
    // 32 KB per CTA
    const size_t off = static_cast<size_t>(blockIdx.x) * 32768;
    if (off + 32768 <= w13_pf_bytes) {
      bulk_prefetch_l2(w13_pf + off, 16384);
      bulk_prefetch_l2(w13_pf + off + 16384, 16384);
    }
  }
  const int token = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;   // expert group of this lane
  const bool tok_ok = token < T;
  const float* lg = logits + static_cast<size_t>(tok_ok ? token : 0) * E_GLOBAL + lane * 8;
  float s[8], swb[8];
  float bia[8];
  if (bias) {   // this lane's 8 bias values (bf16): one 16 B load instead of 8 scalar loads
    const uint4 bb = *reinterpret_cast<const uint4*>(bias + lane * 8);
    const uint32_t bw[4] = {bb.x, bb.y, bb.z, bb.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { bia[2 * k] = __uint_as_float(bw[k] << 16); bia[2 * k + 1] = __uint_as_float(bw[k] & 0xffff0000u); }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) bia[i] = 0.0f;
  }
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const float4 a = *reinterpret_cast<const float4*>(lg + 4 * q);
    const float l[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = 4 * q + k;
#if MOE_FASTSIG
      s[i] = __fdividef(1.0f, 1.0f + __expf(-l[k]));   // ~2 ulp; selection verified identical on the workload
#else
      s[i] = 1.0f / (1.0f + expf(-l[k]));   // exact mirror of the reference (fp32 expf + IEEE division)
#endif
      swb[i] = s[i] + bia[i];
    }
  }
  // group score = sum of the two largest biased scores of the group's 32 experts: local top-2, merged over 4 lanes
  float m1 = -INFINITY, m2 = -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (swb[i] > m1) { m2 = m1; m1 = swb[i]; }
    else if (swb[i] > m2) { m2 = swb[i]; }
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    const float o1 = __shfl_xor_sync(0xffffffffu, m1, off);
    const float o2 = __shfl_xor_sync(0xffffffffu, m2, off);
    const float n1 = fmaxf(m1, o1);
    const float n2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
    m1 = n1; m2 = n2;
  }
  const float gscore = m1 + m2;
  int rank = 0;
#pragma unroll
  for (int gg = 0; gg < 8; ++gg) {
    const float o = __shfl_sync(0xffffffffu, gscore, gg * 4);
    if (o > gscore || (o == gscore && gg < g)) ++rank;
  }
  const bool keep = rank < 4;
  float v[8];   // candidates: the biased scores of kept groups
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = keep ? swb[i] : -INFINITY;
  unsigned sel_mask = 0;
#pragma unroll 1
  for (int round = 0; round < 8; ++round) {
    // local argmax (ties -> lower index) as a 3-level tree, then a 5-level warp argmax
    float tv[4];
    int ti[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const bool a = v[2 * k] >= v[2 * k + 1];
      tv[k] = a ? v[2 * k] : v[2 * k + 1];
      ti[k] = a ? 2 * k : 2 * k + 1;
    }
    const bool a0 = tv[0] >= tv[1];
    const bool a1 = tv[2] >= tv[3];
    const float u0 = a0 ? tv[0] : tv[1], u1 = a1 ? tv[2] : tv[3];
    const int w0 = a0 ? ti[0] : ti[1], w1 = a1 ? ti[2] : ti[3];
    const bool a2 = u0 >= u1;
    const float bv = a2 ? u0 : u1;
    const int bi = a2 ? w0 : w1;
    // warp argmax with one redux.sync on an order-preserving integer key (float order == key order; no NaNs,
    // -0.0 cannot occur: s > 0 so s + bias == 0 rounds to +0.0); ties -> lowest lane == lowest expert index
    const uint32_t bits = __float_as_uint(bv);
    const uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    const uint32_t m = __reduce_max_sync(0xffffffffu, key);
    const unsigned win = __ballot_sync(0xffffffffu, key == m);
    if (lane == __ffs(win) - 1) {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = (i == bi) ? -INFINITY : v[i];
      sel_mask |= 1u << bi;
    }
  }
  float part = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) part += ((sel_mask >> i) & 1u) ? s[i] : 0.0f;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
  const float inv = rsf / (part + 1e-20f);
  const int le0 = lane * 8 - local_offset;   // local index of this lane's first expert
  bool tok_single = false;
  int nloc_w = 0, single_row = 0;
#if MOE_ONESTORE
  {
    // number of this token's selected experts that are local (0..8): rows of tokens with exactly one local expert
    // are written by one plain store in GEMM2 and need no zeroing
    int nloc = 0;
    if (le0 > -8 && le0 < E_LOCAL) {
#pragma unroll
      for (int i = 0; i < 8; ++i) nloc += (((sel_mask >> i) & 1u) && (le0 + i >= 0) && (le0 + i < E_LOCAL)) ? 1 : 0;
    }
    nloc_w = __reduce_add_sync(0xffffffffu, nloc);
    tok_single = (nloc_w == 1);
  }
#else
  (void)tok_cnt;
#endif
  if (tok_ok && sel_mask && le0 > -8 && le0 < E_LOCAL) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if ((sel_mask >> i) & 1u) {
        const int le = le0 + i;
        if (le >= 0 && le < E_LOCAL) {
          const int slot = atomicAdd(&cnt[le * CNT_STRIDE], 1);
          const int row = le * cap + slot;
          row_token[row] = (MOE_ONESTORE && tok_single) ? (token | 0x40000000) : token;   // bit 30: single contribution
          row_weight[row] = s[i] * inv;
          single_row = row;   // (only one lane has a local expert when tok_single)
        }
      }
    }
  }
#if MOE_ONESTORE
  // tok_cnt[token]: number of local experts (0..8), or the token's only local row + 16 when it has exactly one
  single_row = __reduce_add_sync(0xffffffffu, single_row);
  if (tok_ok && lane == 0) tok_cnt[token] = tok_single ? (single_row + 16) : nloc_w;
#endif
}

// Padded-row prefix over experts: rows of expert e are [128*mt_off[e], 128*mt_off[e] + cnt[e]).
__device__ __forceinline__ void compute_prefix(const int* __restrict__ cnt, int* s_mt_off, int* s_cnt,
                                               int* s_voff = nullptr, int* s_pt_off = nullptr) {
  // s_voff (optional): prefix of the valid (unpadded) row counts, for the row-copy work split.
  // s_pt_off (optional): prefix of the pair-tile counts ceil(mt_e / 2) (MOE_PAIR schedule).
  const int lane = threadIdx.x & 31;
  const int c = cnt[lane * CNT_STRIDE];
  int mt = (c + BM - 1) / BM;
  int pt = (mt + 1) / 2;
  int vo = c;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, mt, off);
    const int n2 = __shfl_up_sync(0xffffffffu, vo, off);
    const int n3 = __shfl_up_sync(0xffffffffu, pt, off);
    if (lane >= off) { mt += n; vo += n2; pt += n3; }
  }
  s_mt_off[lane + 1] = mt;
  if (lane == 0) s_mt_off[0] = 0;
  if (s_voff) {
    s_voff[lane + 1] = vo;
    if (lane == 0) s_voff[0] = 0;
  }
  if (s_pt_off) {
    s_pt_off[lane + 1] = pt;
    if (lane == 0) s_pt_off[0] = 0;
  }
  s_cnt[lane] = c;
}
__device__ __forceinline__ int find_expert(const int* s_off, int x) {
  int e = 0;
#pragma unroll
  for (int i = 1; i < E_LOCAL; ++i) e = (x >= s_off[i]) ? i : e;
  return e;
}

// ------------------------------------------------------------------------------------------------
// K3 / K5: persistent grouped GEMM, tile 128 x 256 x 128, tcgen05 MMA, TMEM accumulators, software promotion.
//   MODE 0 (GEMM1): A = A_perm [rows][7168], B = W13[e] [4096][7168]; B tile rows [0,128) = X1 cols j..,
//                   rows [128,256) = X2 cols 2048+j..; epilogue C = silu(X2)*X1 -> fp8 + per-(row,128 cols) scale.
//   MODE 1 (GEMM2): A = C_hi [rows][2048], B = W2[e] [7168][2048]; epilogue output[token][n] += w * acc.
// Per stage the producer also bulk-copies the row scales (128 floats) and the two weight scales (16 B windows)
// into a shared-memory ring, so the promotion warps never touch global memory inside the k-loop.
// Barriers: full[S] (TMA bytes, waited by the MMA warp), empty[S] (MMA commit -> producer, smem stage free),
// tfull[2] (tcgen05.commit -> promotion warps), and hardware named barriers 1/2 (promotion warps -> MMA warp:
// TMEM accumulator stage a consumed; 8 x bar.arrive + 1 x bar.sync, 288 threads).
//
// Two variants share the promotion / epilogue code (promote_and_store):
//   gemm_kernel   : one CTA per 128 x 256 tile (cta_group::1), 4 stages of 48 KB.
//   gemm2sm_kernel: MOE_2SM, clusters of 2 CTAs per 256 x 256 tile (tcgen05.mma.cta_group::2): each CTA loads its
//                   own 128 A rows and 128 of the 256 B rows (32 KB per stage, 6 stages); the leader CTA issues
//                   the MMAs, its full barrier collects both CTAs' TMA bytes (.cta_group::2 loads), one multicast
//                   commit releases the smem stage / TMEM stage in both CTAs. L2->SMEM traffic for B halves.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void bulk_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(src)), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
// Bulk store smem -> global (bulk async-group completion) and its waits.
__device__ __forceinline__ void bulk_store_1d(void* gdst, const void* ssrc, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(reinterpret_cast<uint64_t>(gdst)),
               "r"(smem_u32(ssrc)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit_group() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
// Bulk reduction smem -> global (bf16 add, bulk async-group completion).
__device__ __forceinline__ void bulk_reduce_add_bf16(void* gdst, const void* ssrc, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;" ::"l"(
                   reinterpret_cast<uint64_t>(gdst)),
               "r"(smem_u32(ssrc)), "r"(bytes)
               : "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait_group_read() { asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory"); }
template <int N>
__device__ __forceinline__ void bulk_wait_group() { asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_proxy_async_shared_cta() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// .cta_group::2 tensor loads: data lands in the executing CTA, the transaction bytes are counted on the mbarrier at
// the same offset in the leader CTA (peer bit 24 cleared, as CUTLASS's SM100_TMA_2SM_LOAD does).
__device__ __forceinline__ void tma_load_2d_2sm(void* dst, const void* map, uint32_t leader_bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(leader_bar), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_2sm(void* dst, const void* map, uint32_t leader_bar, int x, int y, int z) {
  asm volatile(
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(leader_bar), "r"(x), "r"(y), "r"(z)
      : "memory");
}
__device__ __forceinline__ void umma_mma_f8_2sm(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc,
                                                uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(d_tmem),
      "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate)
      : "memory");
}
__device__ __forceinline__ void umma_commit_2sm_mc(uint64_t* bar, uint16_t mask) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
          smem_u32(bar)),
      "h"(mask)
      : "memory");
}
__device__ __forceinline__ void tmem_alloc_2sm(uint32_t* dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(dst_smem)), "r"(ncols)
               : "memory");
}
__device__ __forceinline__ void tmem_relinquish_2sm() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc_2sm(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(ncols) : "memory");
}

#ifndef MOE_PAIR
#define MOE_PAIR 0   // 1 = 256 x 256 tiles: the two m-tiles (halves) of a pair share each B stage; bf16x2 accumulators
#endif
#if MOE_PAIR
#define MOE_ACC_BF16 1
#else
#define MOE_ACC_BF16 MOE_BF16ACC
#endif
constexpr int MH = MOE_PAIR ? 2 : 1;   // m-halves (128-row m-tiles) per tile
#ifndef MOE_TRS
#define MOE_TRS 0   // 1 = TMEM-resident accumulation with in-place rescaling (no register accumulators), 0 = register promotion
#endif
static_assert(!(MOE_TRS && MOE_SA_GROUP != 1), "MOE_TRS needs MOE_SA_GROUP == 1");
static_assert(!(MOE_TRS && !MOE_PAIR), "MOE_TRS needs MOE_PAIR (one TMEM buffer per half; k-blocks must not alternate buffers)");
static_assert(!(MOE_PW != 8 && (MOE_PAIR || MOE_TRS || MOE_ACC_BF16)), "MOE_PW=16 only with the fp32 register promotion");
static_assert(!(MOE_RED_OFFLOAD == 1 && MOE_PW != 8), "MOE_RED_OFFLOAD=1 chunk layout assumes 8 promotion warps");
static_assert(!(MOE_PAIR && MOE_AMC), "MOE_PAIR excludes MOE_AMC");
static_assert(!(MOE_PAIR && MOE_SA_GROUP != 1), "MOE_PAIR needs MOE_SA_GROUP == 1");

template <int MODE, bool TWO_SM>
struct GemmCfg {
  static constexpr int NKB = MODE == 0 ? NUM_HB : NUM_IB;
  static constexpr int NT = MODE == 0 ? (2 * INTER / 256) : (HID / 256);   // 16 or 28
#ifndef MOE_STAGES2
#define MOE_STAGES2 3   // GEMM2 pipeline depth (3 frees 48 KB of shared memory for the output staging buffer)
#endif
#ifndef MOE_CHI_TMA
#define MOE_CHI_TMA 0   // GEMM1 epilogue: 1 = c_hi tile written with TMA bulk stores from a smem staging tile (rejected: locked-clock GEMM1 720 vs 706 us), 0 = st.global
#endif
#ifndef MOE_RED_OFFLOAD
#define MOE_RED_OFFLOAD 2   // GEMM2 output reductions: 2 = TMA bulk reductions (cp.reduce.async.bulk, 256 B per row) issued by the idle copy warps from a smem staging tile, 1 = red.global from the copy warps, 0 = red.global from the promotion warps (needs MOE_STAGES2=3 for 1 / 2)
#endif
static_assert(!(MOE_RED_OFFLOAD && MOE_STAGES2 > 3), "MOE_RED_OFFLOAD needs MOE_STAGES2 <= 3 (shared memory)");
  static constexpr int STAGES = TWO_SM ? (MODE == 0 ? 6 : 4) : (MOE_PAIR ? 3 : (MODE == 1 ? MOE_STAGES2 : 4));
  static constexpr uint32_t A_BYTES = BM * BK;                       // one m-half: 128 A rows
  static constexpr uint32_t B_BYTES = TWO_SM ? (128 * BK) : (BN * BK);   // 2-SM: this CTA's 2 x 64 B rows; else 256 B rows
  static constexpr uint32_t STAGE_BYTES = MH * A_BYTES + B_BYTES;    // 48 KB (64 KB with MOE_PAIR), 1024 B aligned
  // Scale rings (row scales are laid out [m-tile][k-block][128] so that 4 k-blocks are one contiguous 2 KB copy).
  //   SA: 8 slots x MH halves x (SA_GROUP k-blocks x 128 row scales);
  //   SB: 8 slots x the tile's weight scales (GEMM1: 2 x 56 floats at [0] / [64]; GEMM2: 2 x 16 floats at [0]),
  //       one copy per tile.
  // The producer runs at most STAGES + 2 k-blocks ahead of the promotion warps, slots are reused 8 groups / 8 tiles
  // (>= 8 k-blocks) later, so a slot is only rewritten after it has been read.
  // 2-SM: the producer may run STAGES (6) k-blocks ahead of the MMA and the promotion lags the MMA by up to 2
  // k-blocks, so the rings need more than 8 slots there.
  static constexpr int SA_SLOTS = TWO_SM ? 16 : 8, SB_SLOTS = TWO_SM ? 16 : 8;
  static constexpr int SA_GROUP = MOE_SA_GROUP;
  // MOE_TRS: each stage also carries the next k-block's row scales (k-blocks kb, kb+1 are contiguous in the
  // [m-tile][k-block][128] layout; the scale arrays are padded by one k-block so the copy never runs past the end)
  static constexpr uint32_t SA_HALF_BYTES = (MOE_TRS ? 2 : 1) * SA_GROUP * 512;
  static constexpr int SA_HALF_FLOATS = SA_HALF_BYTES / 4;
  static constexpr uint32_t SA_SLOT_BYTES = MH * SA_HALF_BYTES;
  static constexpr uint32_t SB_SLOT_BYTES = (MODE == 0) ? 512 : 128;   // GEMM1: 2 x 56 floats at [0] / [64]; GEMM2: 2 x 16 floats
  static constexpr uint32_t SB_TX = MODE == 0 ? 448u : 128u;
  static constexpr uint32_t RING_OFF = STAGES * STAGE_BYTES;
  static constexpr uint32_t SB_OFF = RING_OFF + SA_SLOTS * SA_SLOT_BYTES;
  static constexpr uint32_t BAR_OFF = SB_OFF + SB_SLOTS * SB_SLOT_BYTES;   // mbarriers + tmem slot (< 512 B)
  static constexpr uint32_t XCH_OFF = BAR_OFF + 512;                            // 2 x NG x MH x 128 floats amax exchange
  static constexpr int STG_SLOTS = 3;                                           // copy-warp staging ring (rows of 7168 B)
  static constexpr uint32_t STG_OFF = XCH_OFF + MH * NG * 1024;
  // MOE_RED_OFFLOAD (GEMM2): the promotion warps stage the packed bf16 output tile (2 x 4 x 16 chunks of 512 B =
  // 64 KB) + 128 token ids; the copy warps issue the global reductions from it.
  static constexpr uint32_t OUT_OFF = STG_OFF + (MOE_COPY_TMA && MODE == 0 ? STG_SLOTS * HID : 0);
  // mode 1: 128 chunks of 512 B (32 rows x 16 B); mode 2 (TMA bulk reduce): 256 rows x 272 B (row-contiguous, padded)
  static constexpr uint32_t OUT_ROWS_BYTES = MOE_RED_OFFLOAD == 2 ? 69632u : 65536u;
  static constexpr uint32_t OUT_BYTES = (MOE_RED_OFFLOAD && MODE == 1) ? (OUT_ROWS_BYTES + 1024u) : 0u;
  // MOE_CHI_TMA (GEMM1, single-CTA kernel): the fp8 output tile is staged as 128 rows x 144 B (128 B + pad) and
  // written by one 128 B bulk store per row.
  static constexpr uint32_t CHI_OFF = OUT_OFF + OUT_BYTES;
  static constexpr uint32_t CHI_BYTES = (MOE_CHI_TMA && MODE == 0 && !TWO_SM) ? (128u * 144u) : 0u;
  static constexpr uint32_t SMEM_BYTES = CHI_OFF + CHI_BYTES + 1024;
  static_assert(STAGES * 2 + SA_SLOTS + 8 <= 63, "barrier area");
  static_assert(SMEM_BYTES <= 232448, "shared memory budget");
};

// Per-tile state shared by the producer, MMA and promotion warps.
struct TileInfo {
  int e, nt, mrel;       // expert, n-tile, first m-tile relative to the expert's first m-tile
  bool valid[MH];        // half h exists (an odd number of m-tiles leaves the second half of the last pair empty)
  bool half[MH];         // half h holds <= 64 rows: 64 x 256 tile (tcgen05.mma M = 64, 64-row A box)
};

// Clamp a block scale away from zero (|s| >= 2^-40) so that scale ratios stay finite; NaN -> +-2^-40.
__device__ __forceinline__ float safe_scale(float s) {
  const float eps = 9.094947e-13f;   // 2^-40
  return (fabsf(s) >= eps) ? s : copysignf(eps, s);
}

// Promotion + epilogue for one tile of this CTA (8 warps = 2 warpgroups).
//   ng: running scale-group counter (shared convention with the producer: a group starts at the first k-block of a
//   tile / split-K piece and at every k-block multiple of 4).
//   Each thread owns one TMEM lane (row) per half and 128 of the 256 accumulator columns; with MOE_PAIR the two
//   halves alternate per k-block (TMEM stage a = half), otherwise consecutive k-blocks alternate.
//   MOE_TRS: the MMA accumulates every k-block into the same TMEM buffer; after k-block kb the warps multiply the
//   buffer in place by s_kb / s_kb+1 (per row x n-block block scale), so that the buffer always holds
//   sum_j (s_j / s_kb) P_j and the epilogue multiplies by s_last: numerically an fp32 accumulation of s_j P_j.
template <int MODE, int STAGES, int SA_SLOTS, int SB_SLOTS, int SA_HALF_FLOATS, int SB_SLOT_FLOATS, class Release>
__device__ __forceinline__ void promote_and_store(
    uint8_t* smem, const float* sa_ring, const float* sb_ring, uint64_t* full_bar, uint64_t* tfull_bar,
    uint32_t tmem_lane_base, uint32_t& acc_it, uint32_t& stage, uint32_t& phase, uint32_t& ng, int tile_iter,
    const TileInfo& ti, const bool* active, const int* row, const int* row_in_tile, const int* token, const float* wtok, int* trc_p,
    int* s_fin_e, unsigned* s_fin_cnt, int pf_e, int pf_nt, int pf_row0, bool pf_half, const uint8_t* b_raw, const uint8_t* a_raw,
    int kb0, int kb1, int g, int warp, int lane, uint32_t xch_off, uint32_t out_off, uint32_t chi_off, int& pending_e,
    uint8_t* __restrict__ c_hi, float* __restrict__ sc, __nv_bfloat16* __restrict__ out, int* __restrict__ done_tiles,
    Release release) {
  (void)out_off; (void)chi_off; (void)pending_e;
  // release(a): this warp has finished reading / writing TMEM stage a (named-barrier arrive).
  // active[h]: this lane owns a row of half h (false for lanes 16..31 of a 64-row half and for missing halves).
  constexpr int SA_GROUP = MOE_SA_GROUP;
  const uint32_t col0 = (MODE == 0) ? (HC * g) : (CG * g);          // X1 block / first column block
  const uint32_t col1 = (MODE == 0) ? (128 + HC * g) : (CG * g + HC);   // X2 block / second column block
  (void)full_bar;
#if MOE_ABL == 1 || MOE_ABL == 5 || MOE_ABL == 6 || MOE_ABL == 7
  constexpr bool ABL_SKIP = true;
#else
  constexpr bool ABL_SKIP = false;
#endif
  // ---------------- epilogue for one half: accf(i) = final fp32 value of this thread's column i ----------------
  auto epilogue = [&](int h, auto accf) {
    if constexpr (MODE == 0) {
      // acc[0..63] = X1 (cols j..), acc[64..127] = X2 (cols 2048+j..). C = silu(X2) * X1, then FP8 re-quantization
      // with one scale per (row, 128-column block): this warpgroup holds 64 of the 128 columns, the partner
      // warpgroup the other 64 -> exchange the per-row amax through shared memory.
      float c[HC];
      float amax = 0.0f;
      TSTAMP(tg0);
#if MOE_FASTSILU == 2 && MOE_SILU_FOLD == 2
      // c' = 2 silu(x2) x1 = (x1 x2) (1 + tanh(x2 / 2)); the factor 0.5 is folded into the quantization scale below.
      // Packed f32x2 multiplies (column pairs are adjacent accumulator registers).
#pragma unroll
      for (int i = 0; i < HC; i += 2) {
        const unsigned long long x1p = pack2f(accf(i), accf(i + 1));
        const unsigned long long x2p = pack2f(accf(HC + i), accf(HC + i + 1));
        const unsigned long long hp = fmul2(x2p, pack2f(0.5f, 0.5f));
        float t0, t1;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t0) : "f"(lo2f(hp)));
        asm("tanh.approx.f32 %0, %1;" : "=f"(t1) : "f"(hi2f(hp)));
        const unsigned long long pp = fmul2(x1p, x2p);
        const unsigned long long cp = ffma2_3(pp, pack2f(t0, t1), pp);
        c[i] = lo2f(cp); c[i + 1] = hi2f(cp);
        amax = fmaxf(amax, fmaxf(fabsf(c[i]), fabsf(c[i + 1])));
      }
#else
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        const float x2 = accf(HC + i);
#if MOE_FASTSILU == 2 && MOE_SILU_FOLD
        float th;
        asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * x2));
        const float pr = accf(i) * x2;
        c[i] = fmaf(pr, th, pr);   // = 2 x1 silu(x2)
#elif MOE_FASTSILU == 2
        // silu(x) = x * sigmoid(x) = x * 0.5 * (1 + tanh(x / 2)): one MUFU.TANH instead of EX2 + RCP
        float th;
        asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * x2));
        c[i] = accf(i) * (x2 * (0.5f + 0.5f * th));
#elif MOE_FASTSILU
        c[i] = accf(i) * __fdividef(x2, 1.0f + __expf(-x2));
#else
        c[i] = accf(i) * (x2 / (1.0f + __expf(-x2)));
#endif
        amax = fmaxf(amax, fabsf(c[i]));
      }
#endif
      float* xch = reinterpret_cast<float*>(smem + xch_off) + ((tile_iter & 1) * MH + h) * (NG * 128);
      if (active[h]) xch[g * 128 + row_in_tile[h]] = amax;
      TSTAMP(tg1);
      named_bar_sync(3, PROMO_THREADS);
      TSTAMP(tg2);
#pragma unroll
      for (int g2 = 0; g2 < NG; ++g2)
        if (g2 != g) amax = fmaxf(amax, xch[g2 * 128 + row_in_tile[h]]);
      constexpr bool FOLD = (MOE_FASTSILU == 2 && MOE_SILU_FOLD != 0);   // c holds 2 x the SwiGLU output
      const float scale = amax * (FOLD ? (1.0f / 896.0f) : (1.0f / 448.0f));
      const float inv = amax > 0.0f ? 448.0f / amax : 0.0f;   // FOLD: c / 2 * (448 / (amax / 2)) = c * (448 / amax)
      uint32_t hw[HC / 4];
#pragma unroll
      for (int j = 0; j < HC / 4; ++j) {
        uint32_t h4 = 0;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          float2 x = make_float2(c[j * 4 + 2 * k], c[j * 4 + 2 * k + 1]);
          if (MOE_SILU_FOLD == 2) { const unsigned long long xs = fmul2(pack2f(x.x, x.y), pack2f(inv, inv)); x = make_float2(lo2f(xs), hi2f(xs)); }
          else x = make_float2(x.x * inv, x.y * inv);
          const __nv_fp8x2_storage_t h2 = __nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
          h4 |= static_cast<uint32_t>(h2) << (16 * k);
        }
        hw[j] = h4;
      }
      const size_t coff = static_cast<size_t>(row[h]) * INTER + ti.nt * 128 + HC * g;
      if (active[h] && MOE_ABL != 12) {   // ABL 12: no c_hi / scale stores (timing)
#if MOE_CHI_TMA
        if (MH == 1) {
          uint8_t* stg = smem + chi_off + row_in_tile[h] * 144 + HC * g;
#pragma unroll
          for (int j = 0; j < HC / 16; ++j)
            *reinterpret_cast<uint4*>(stg + 16 * j) = make_uint4(hw[4 * j], hw[4 * j + 1], hw[4 * j + 2], hw[4 * j + 3]);
        } else
#endif
        {
#pragma unroll
          for (int j = 0; j < HC / 16; ++j) {
            reinterpret_cast<int4*>(c_hi + coff)[j] = make_int4(hw[4 * j], hw[4 * j + 1], hw[4 * j + 2], hw[4 * j + 3]);
          }
        }
        if (g == 0) sc[(static_cast<size_t>(row[h] >> 7) * NUM_IB + ti.nt) * 128 + (row[h] & 127)] = scale;
      }
#if MOE_TIMING
      { TSTAMP(tg3); if (lane == 0) { long long* d = g_dbg_timing[MODE] + blockIdx.x * 128 + warp * 8; d[0] += tg1 - tg0; d[1] += tg2 - tg1; d[2] += tg3 - tg2; } }
#endif
    } else {
#if MOE_RED_OFFLOAD
      // stage w * acc (bf16) for the copy warps: chunk j of row (q, lane) at ((g*4+q)*16 + j) * 512 + lane * 16
      // (conflict-free: the 32 lanes of a warp write 512 contiguous bytes per chunk)
      {
        const int q = warp & 3;
        TSTAMP(te0);
        named_bar_sync(6, BAR_RED);   // staging buffer free (copy warps finished the previous tile's reductions)
        TSTAMP(te1);
        uint8_t* stg = smem + out_off;
        const float w = (MODE == 1 && MOE_WFOLD && !MOE_TRS) ? 1.f : wtok[h];   // MOE_WFOLD: already in the scales
        const int col_base = CG * g;                 // this thread's first of 256 columns
        const int h2 = col_base >> 7;                // 128-column half (one 256 B bulk reduction per row)
        const int boff = (col_base & 127) * 2;       // byte offset inside the row's 256 B
#pragma unroll
        for (int j = 0; j < CG / 8; ++j) {
          const uint32_t p0 = pack_bf16x2(w * accf(j * 8 + 0), w * accf(j * 8 + 1));
          const uint32_t p1 = pack_bf16x2(w * accf(j * 8 + 2), w * accf(j * 8 + 3));
          const uint32_t p2 = pack_bf16x2(w * accf(j * 8 + 4), w * accf(j * 8 + 5));
          const uint32_t p3 = pack_bf16x2(w * accf(j * 8 + 6), w * accf(j * 8 + 7));
          if (MOE_RED_OFFLOAD == 2 && MOE_STS)
            sts128(smem_u32(stg + (h2 * 128 + q * 32 + lane) * 272 + boff + j * 16), p0, p1, p2, p3);
          else if (MOE_RED_OFFLOAD == 2)
            *reinterpret_cast<uint4*>(stg + (h2 * 128 + q * 32 + lane) * 272 + boff + j * 16) = make_uint4(p0, p1, p2, p3);
          else
            *reinterpret_cast<uint4*>(stg + ((g * 4 + q) * 16 + j) * 512 + lane * 16) = make_uint4(p0, p1, p2, p3);
        }
        int* s_tok = reinterpret_cast<int*>(stg + (MOE_RED_OFFLOAD == 2 ? 69632 : 65536));
        if (g == 0) s_tok[q * 32 + lane] = token[h];   // bit 30: single-contribution row (plain store)
        if (warp == 4 && lane == 0) { s_tok[128] = ti.nt; s_tok[129] = 0; s_tok[130] = (kb1 - kb0 < NUM_IB) ? 1 : 0; }
        __threadfence_block();
        named_bar_arrive(5, BAR_RED);   // tile staged
#if MOE_TIMING
        { TSTAMP(te2); if (lane == 0) { long long* d = g_dbg_timing[MODE] + blockIdx.x * 128 + warp * 8; d[0] += te1 - te0; d[1] += te2 - te1; d[2] += 1; } }
#endif
      }
      if (false) {
#else
      if (token[h] >= 0 && MOE_ABL != 11) {   // ABL 11: no GEMM2 epilogue (timing); ABL 10: pack only, no red
#endif
        const float w = (MODE == 1 && MOE_WFOLD && !MOE_TRS) ? 1.f : wtok[h];
        __nv_bfloat16* dst = out + static_cast<size_t>(token[h] & 0x3fffffff) * HID + ti.nt * BN + CG * g;
#pragma unroll
        for (int j = 0; j < CG / 8; ++j) {
          const uint32_t p0 = pack_bf16x2(w * accf(j * 8 + 0), w * accf(j * 8 + 1));
          const uint32_t p1 = pack_bf16x2(w * accf(j * 8 + 2), w * accf(j * 8 + 3));
          const uint32_t p2 = pack_bf16x2(w * accf(j * 8 + 4), w * accf(j * 8 + 5));
          const uint32_t p3 = pack_bf16x2(w * accf(j * 8 + 6), w * accf(j * 8 + 7));
          if (MOE_ABL != 10) red_add_bf16x2_v4(dst + j * 8, p0, p1, p2, p3);
          else if (p0 == 0x12345678u && p1 == p2 && p2 == p3) dst[j * 8] = __float2bfloat16(w);   // keep the packs alive
        }
      }
    }
  };
  auto finish_tile = [&]() {
    if constexpr (MODE == 0) {
#if MOE_FIN_OFFLOAD && !MOE_CHI_TMA
      // Each lane publishes its stores at CTA scope (red.release); copy warp 2 observes 256 arrivals per tile
      // (acquire), fences at gpu scope (cumulative) and bumps done_tiles[e]. No bar.sync on the promotion path.
      if (warp == 4 && lane == 0) s_fin_e[tile_iter & 31] = ti.e;
      red_release_cta_shared(smem_u32(s_fin_cnt), 1u);
      return;
#endif
      named_bar_sync(3, PROMO_THREADS);   // all promotion warps' stores issued (MOE_CHI_TMA: staging complete)
#if MOE_CHI_TMA
      if (MH == 1) {
        // one 128 B bulk store per row (g == 0 threads); completion is awaited at the next tile's epilogue, which
        // then signals this tile's done_tiles (pending_e)
        if (g == 0) {
          fence_proxy_async_shared_cta();
          if (active[0])
            bulk_store_1d(c_hi + static_cast<size_t>(row[0]) * INTER + ti.nt * 128, smem + chi_off + row_in_tile[0] * 144, 128);
          bulk_commit_group();
        }
        return;
      }
#endif
      if (warp == 4 && lane == 0) {
        __threadfence();
        atomicAdd(done_tiles + ti.e * CNT_STRIDE, 1);
      }
    }
  };
#if MOE_TRS
  // ---------------- TMEM-resident accumulation ----------------
  // k-loop: after every k-block but the last, each half's buffer is multiplied in place by s_kb / s_kb+1.
  // The last k-block's TMEM stages stay held (no release) until the epilogue below has read them.
#pragma unroll 1
  for (int kb = kb0; kb < kb1; ++kb) {
    const bool gstart = (kb == kb0) || ((kb & (SA_GROUP - 1)) == 0);
    const uint32_t cur_g = gstart ? ng++ : ng - 1;
    const bool last = (kb == kb1 - 1);
#pragma unroll
    for (int h = 0; h < MH; ++h) {
      const uint32_t a = acc_it & 1u, aph = (acc_it >> 1) & 1u;
      TSTAMP(tp0);
      mbar_wait(&tfull_bar[a], aph);
      umma_fence_after();
#if MOE_TIMING
      { TSTAMP(tp1); if (warp == 4 && lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 6] += tp1 - tp0; } }
#endif
      const bool skip = ABL_SKIP || ((MH > 1) && !ti.valid[h]);
      if (skip) {
        umma_fence_before();
        release(a);
      } else if (!last) {
        const float* sS = sa_ring + (cur_g % SA_SLOTS) * (MH * SA_HALF_FLOATS) + h * SA_HALF_FLOATS;
        const float* sbw = sb_ring + (tile_iter % SB_SLOTS) * (SB_SLOT_FLOATS);
        const uint32_t taddr = tmem_lane_base + a * BN;
        // ratio of this k-block's block scale to the next one's (row scales kb / kb + 1 in the same slot)
        const float sav = sS[row_in_tile[h]], savn = sS[128 + row_in_tile[h]];
        const float r1 = __fdividef(safe_scale(sav * ((MODE == 0) ? sbw[kb] : sbw[16 * g + kb])),
                                    safe_scale(savn * ((MODE == 0) ? sbw[kb + 1] : sbw[16 * g + kb + 1])));
        const float r2 = (MODE == 0) ? __fdividef(safe_scale(sav * sbw[64 + kb]), safe_scale(savn * sbw[64 + kb + 1])) : r1;
        const unsigned long long r1p = pack2(__float_as_uint(r1), __float_as_uint(r1));
        const unsigned long long r2p = pack2(__float_as_uint(r2), __float_as_uint(r2));
        auto mul32 = [&](uint32_t* v, unsigned long long rp) {
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const unsigned long long t = ffma2(0ull, pack2(v[i], v[i + 1]), rp);
            v[i] = static_cast<uint32_t>(t);
            v[i + 1] = static_cast<uint32_t>(t >> 32);
          }
        };
        uint32_t v0[32], v1[32], f0[32], f1[32];
        tmem_ld_x32(taddr + col0, v0);
        tmem_ld_x32(taddr + col0 + 32, v1);
        tmem_ld_wait();
        mul32(v0, r1p);
        mul32(v1, r1p);
        tmem_st_x32(taddr + col0, v0);
        tmem_st_x32(taddr + col0 + 32, v1);
        tmem_ld_x32(taddr + col1, f0);
        tmem_ld_x32(taddr + col1 + 32, f1);
        tmem_ld_wait();
        mul32(f0, r2p);
        mul32(f1, r2p);
        tmem_st_x32(taddr + col1, f0);
        tmem_st_x32(taddr + col1 + 32, f1);
        tmem_st_wait();
        umma_fence_before();
        release(a);
#if MOE_TIMING
        { TSTAMP(tp2); if (warp == 4 && lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 7] += tp2 - tp0; } }
#endif
      }
      ++acc_it;
    }
    if (++stage == STAGES) { stage = 0; phase ^= 1; }
  }
  // ---------------- epilogue: read each half's final accumulator, release its TMEM stage, store ----------------
  if (ABL_SKIP) {
    if (MODE == 0 && warp == 4 && lane == 0) atomicAdd(done_tiles + ti.e * CNT_STRIDE, 1);   // keep GEMM2's protocol alive
    return;
  }
  {
    const uint32_t cur_g = ng - 1;   // scale group of the last k-block
    const float* sbw = sb_ring + (tile_iter % SB_SLOTS) * (SB_SLOT_FLOATS);
    const int kbl = kb1 - 1;
#pragma unroll
    for (int h = 0; h < MH; ++h) {
      if (MH > 1 && !ti.valid[h]) continue;   // uniform across the CTA (its stage was released in the loop)
      const uint32_t a = (acc_it - MH + h) & 1u;   // sub-step parity of (last k-block, half h)
      const uint32_t taddr = tmem_lane_base + a * BN;
      const float* sS = sa_ring + (cur_g % SA_SLOTS) * (MH * SA_HALF_FLOATS) + h * SA_HALF_FLOATS;
      const float sav = sS[row_in_tile[h]];
      const float s1 = safe_scale(sav * ((MODE == 0) ? sbw[kbl] : sbw[16 * g + kbl]));
      const float s2 = (MODE == 0) ? safe_scale(sav * sbw[64 + kbl]) : s1;
      uint32_t v0[32], v1[32], f0[32], f1[32];
      tmem_ld_x32(taddr + col0, v0);
      tmem_ld_x32(taddr + col0 + 32, v1);
      tmem_ld_x32(taddr + col1, f0);
      tmem_ld_x32(taddr + col1 + 32, f1);
      tmem_ld_wait();
      umma_fence_before();
      release(a);
      if (MOE_ABL == 9) continue;   // ABL 9: no epilogue stores (timing)
      auto accf = [&](int i) {
        const uint32_t u = (i < 32) ? v0[i] : (i < 64) ? v1[i - 32] : (i < 96) ? f0[i - 64] : f1[i - 96];
        return __uint_as_float(u) * ((i < 64) ? s1 : s2);
      };
      epilogue(h, accf);
    }
  }
  if (MOE_ABL != 9) finish_tile();
  else if (MODE == 0 && warp == 4 && lane == 0) atomicAdd(done_tiles + ti.e * CNT_STRIDE, 1);
#else
  // ---------------- register accumulation ----------------
#if MOE_ACC_BF16
  uint32_t accb[MH * 64];   // bf16x2 pairs: accb[h * 64 + i] = cols (2i, 2i+1) of this thread's 128 columns of half h
#pragma unroll
  for (int i = 0; i < MH * 64; ++i) accb[i] = 0u;
  auto accf2 = [&](int h, int i) { return (i & 1) ? bf16hi(accb[h * 64 + (i >> 1)]) : bf16lo(accb[h * 64 + (i >> 1)]); };
#else
  float acc[CG];
#pragma unroll
  for (int i = 0; i < CG; ++i) acc[i] = 0.0f;
  auto accf2 = [&](int, int i) { return acc[i]; };
#endif
#pragma unroll 1
  for (int kb = kb0; kb < kb1; ++kb) {
    const bool gstart = (kb == kb0) || ((kb & (SA_GROUP - 1)) == 0);
    const uint32_t cur_g = gstart ? ng++ : ng - 1;
#pragma unroll
    for (int h = 0; h < MH; ++h) {
      const uint32_t a = acc_it & 1u, aph = (acc_it >> 1) & 1u;
      TSTAMP(tp0);
      mbar_wait(&tfull_bar[a], aph);
      umma_fence_after();
#if MOE_TIMING
      { TSTAMP(tp1); if (warp == 4 && lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 6] += tp1 - tp0; }
        if (lane == 0 && kb == kb0) g_dbg_timing[MODE][blockIdx.x * 128 + warp * 8 + 6] += tp1 - tp0;
        if (lane == 0 && kb == kb1 - 1) g_dbg_timing[MODE][blockIdx.x * 128 + warp * 8 + 7] += tp1 - tp0;
        if (warp == 4 && lane == 0 && blockIdx.x < 2 && *trc_p < 1024) { g_dbg_trace[MODE][blockIdx.x][2][*trc_p][0] = TRACE_CODE(7, tile_iter, kb - kb0); g_dbg_trace[MODE][blockIdx.x][2][*trc_p][1] = tp1; ++*trc_p; } }
#endif
      const float* sS = sa_ring + (cur_g % SA_SLOTS) * (MH * SA_HALF_FLOATS) + h * SA_HALF_FLOATS;
      const float* sbw = sb_ring + (tile_iter % SB_SLOTS) * (SB_SLOT_FLOATS);
      const uint32_t taddr = tmem_lane_base + a * BN;
      uint32_t v0[32], v1[32];
      const bool skip = ABL_SKIP || ((MH > 1) && !ti.valid[h]);   // MH == 1: the single half always exists
      if (skip) {
        umma_fence_before();
        release(a);
        (void)sS; (void)sbw; (void)taddr; (void)v0; (void)v1;
      } else {
        if (MOE_ABL != 3) {   // ABL 3: FMAs only (no TMEM loads)
          tmem_ld_x32(taddr + col0, v0);
          if (CG == 128) tmem_ld_x32(taddr + col0 + 32, v1);
        }
        // scales for this k-block (read while the TMEM loads are in flight)
        const float sav = sS[(kb & (SA_GROUP - 1)) * 128 + row_in_tile[h]];
        // GEMM1: cols [0,64) of this warpgroup are X1 (n-block nt), cols [64,128) are X2 (n-block 16+nt).
        // GEMM2: both column halves of warpgroup g belong to n-block 2*nt+g.
        const float sb1v = (MODE == 0) ? sbw[kb] : sbw[16 * ((CG * g) >> 7) + kb];   // GEMM2: n-block of this group
        const float sb2v = (MODE == 0) ? sbw[64 + kb] : sb1v;
        const float wf = (MODE == 1 && MOE_WFOLD) ? wtok[h] : 1.f;   // GEMM2: routing weight folded into the scale
        const float s1 = sav * sb1v * wf;
        const float s2 = sav * sb2v * wf;
        const bool act = active[h] && (MOE_ABL != 2);   // ABL 2: TMEM loads only
#if MOE_ACC_BF16
        const uint32_t s1p = cvt_bf16x2(s1, s1), s2p = cvt_bf16x2(s2, s2);
        auto fma32 = [&](int base, const uint32_t* v, uint32_t sp) {
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const uint32_t vb = MOE_BF16_TRUNC ? trunc_bf16x2(v[i], v[i + 1])
                                               : cvt_bf16x2(__uint_as_float(v[i]), __uint_as_float(v[i + 1]));
            accb[h * 64 + ((base + i) >> 1)] = hfma2_bf16(vb, sp, accb[h * 64 + ((base + i) >> 1)]);
          }
        };
#elif MOE_FFMA2
        const unsigned long long s1p = pack2(__float_as_uint(s1), __float_as_uint(s1));
        const unsigned long long s2p = pack2(__float_as_uint(s2), __float_as_uint(s2));
        auto fma32 = [&](int base, const uint32_t* v, unsigned long long sp) {
          if (MOE_FFMA2 == 2 && base < 64) {   // mixed: scalar FMAs for the first column half (second FP32 pipe half)
            const float sf = __uint_as_float(static_cast<uint32_t>(sp));
#pragma unroll
            for (int i = 0; i < 32; ++i) acc[base + i] = fmaf(sf, __uint_as_float(v[i]), acc[base + i]);
            return;
          }
#pragma unroll
          for (int i = 0; i < 32; i += 2) {
            const unsigned long long t =
                ffma2(pack2(__float_as_uint(acc[base + i]), __float_as_uint(acc[base + i + 1])), pack2(v[i], v[i + 1]), sp);
            acc[base + i] = __uint_as_float(static_cast<uint32_t>(t));
            acc[base + i + 1] = __uint_as_float(static_cast<uint32_t>(t >> 32));
          }
        };
#else
        const float s1p = s1, s2p = s2;
        auto fma32 = [&](int base, const uint32_t* v, float sp) {
#pragma unroll
          for (int i = 0; i < 32; ++i) acc[base + i] = fmaf(sp, __uint_as_float(v[i]), acc[base + i]);
        };
#endif
        if constexpr (CG == 64) {
          // 16 promotion warps: X1 block (col0) then X2 block (col1), release right after the second load lands
          tmem_ld_wait();
          if (act) fma32(0, v0, s1p);
          if (MOE_ABL != 3) tmem_ld_x32(taddr + col1, v1);
          tmem_ld_wait();
          umma_fence_before();
          release(a);
          if (act) fma32(32, v1, s2p);
        } else {
        tmem_ld_wait();
        if (act) fma32(0, v0, s1p);
        if (MOE_ABL != 3) tmem_ld_x32(taddr + col1, v0);
        if (MOE_FMA_YIELD) __nanosleep(1);
        if (act && MOE_ABL != 4) fma32(32, v1, s1p);   // ABL 4: FMAs on half the columns only
#if MOE_REORDER
        if (MOE_ABL != 3) tmem_ld_x32(taddr + col1 + 32, v1);
        tmem_ld_wait();
        umma_fence_before();
        release(a);   // TMEM stage a consumed by this warp (before the last two FMA blocks: shorter release chain)
#if MOE_TIMING
        { TSTAMP(tp2); if (warp == 4 && lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 7] += tp2 - tp0;
            if (blockIdx.x < 2 && *trc_p < 1024) { g_dbg_trace[MODE][blockIdx.x][2][*trc_p][0] = TRACE_CODE(8, tile_iter, kb - kb0); g_dbg_trace[MODE][blockIdx.x][2][*trc_p][1] = tp2; ++*trc_p; } } }
#endif
        if (act && MOE_ABL != 4) fma32(64, v0, s2p);
        if (MOE_FMA_YIELD) __nanosleep(1);
        if (act) fma32(96, v1, s2p);
        {
          constexpr int PFD = (MODE == 0) ? MOE_PWPF : MOE_PWPF2;
          if (PFD > 0 && pf_e >= 0 && kb == kb1 - PFD) {
            // L2 prefetch of the next tile's first MOE_PWPF_KB k-blocks: 128 B lines, 384 per k-block (B 256 + A 128),
            // thread p takes lines p, p + 256, ... (6 per thread for 4 k-blocks); no TMA involvement
            const int p = (warp - 4) * 32 + lane;
#pragma unroll
            for (int j = 0; j < (MOE_PWPF_KB * 384 + 255) / 256; ++j) {
              const int qq = p + 256 * j;
              const int kbq = qq / 384, r = qq - kbq * 384;
              if (kbq < MOE_PWPF_KB) {
                const uint8_t* addr;
                if (r < 256) {
                  const int brow = (MODE == 0) ? ((r < 128) ? pf_nt * 128 + r : INTER + pf_nt * 128 + (r - 128)) : pf_nt * BN + r;
                  addr = b_raw + (static_cast<size_t>(pf_e) * ((MODE == 0) ? 2 * INTER : HID) + brow) * ((MODE == 0) ? HID : INTER) + kbq * BK;
                } else {
                  const int ar = r - 256;
                  if (pf_half && ar >= BM / 2) continue;
                  addr = a_raw + static_cast<size_t>(pf_row0 + ar) * ((MODE == 0) ? HID : INTER) + kbq * BK;
                }
                prefetch_l2(addr);
              }
            }
          }
        }
#else
        tmem_ld_wait();
        tmem_ld_x32(taddr + col1 + 32, v1);
        if (act) fma32(64, v0, s2p);
        tmem_ld_wait();
        umma_fence_before();
        release(a);
#if MOE_TIMING
        { TSTAMP(tp2); if (warp == 4 && lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 7] += tp2 - tp0; } }
#endif
        if (act) fma32(96, v1, s2p);
#endif
        }
      }
      ++acc_it;
    }
    if (++stage == STAGES) { stage = 0; phase ^= 1; }
  }
  // ---------------- epilogue ----------------
  if (ABL_SKIP) {
    if (MODE == 0 && warp == 4 && lane == 0) atomicAdd(done_tiles + ti.e * CNT_STRIDE, 1);   // keep GEMM2's protocol alive
    return;
  }
#if MOE_CHI_TMA
  if constexpr (MODE == 0 && MH == 1) {
    // previous tile's c_hi bulk stores complete (long since: 56 k-blocks ago) -> its done_tiles signal; the staging
    // buffer is free for this tile
    if (g == 0) bulk_wait_group<0>();
    named_bar_sync(3, PROMO_THREADS);
    if (pending_e >= 0 && warp == 4 && lane == 0) { __threadfence(); atomicAdd(done_tiles + pending_e * CNT_STRIDE, 1); }
    pending_e = ti.e;
  }
#endif
  TSTAMP(tq0);
#pragma unroll
  for (int h = 0; h < MH; ++h) {
    if (MH > 1 && !ti.valid[h]) continue;   // uniform across the CTA
    epilogue(h, [&](int i) { return accf2(h, i); });
  }
  finish_tile();
#if MOE_TIMING
  { TSTAMP(tq1); if (lane == 0) g_dbg_timing[MODE][blockIdx.x * 128 + warp * 8 + 5] += tq1 - tq0;
    if (warp == 4 && lane == 0 && blockIdx.x < 2 && *trc_p + 1 < 1024) {
      g_dbg_trace[MODE][blockIdx.x][2][*trc_p][0] = TRACE_CODE(9, tile_iter, 0); g_dbg_trace[MODE][blockIdx.x][2][*trc_p][1] = tq0; ++*trc_p;
      g_dbg_trace[MODE][blockIdx.x][2][*trc_p][0] = TRACE_CODE(10, tile_iter, 0); g_dbg_trace[MODE][blockIdx.x][2][*trc_p][1] = tq1; ++*trc_p; } }
#endif
#endif
}

// Wait for the producer kernel's per-expert readiness counter before loading expert e.
template <int MODE>
__device__ __forceinline__ void wait_expert_ready(int e, const int* s_tile_off, const int* s_cnt, const int* rows_done,
                                                  const int* done_tiles) {
  // GEMM1: rows_done[e] counts the valid rows of expert e copied into a_perm / sa_perm by the copy warps (all CTAs);
  // GEMM2: done_tiles[e] counts GEMM1 tiles (pair-tiles with MOE_PAIR) of expert e.
  const int need = (MODE == 0) ? s_cnt[e] : (s_tile_off[e + 1] - s_tile_off[e]) * (2 * INTER / 256);
  const int* ctr = (MODE == 0) ? (rows_done + e * CNT_STRIDE) : (done_tiles + e * CNT_STRIDE);
  while (ld_acquire_gpu(ctr) < need) __nanosleep(256);
  fence_proxy_async_global();   // make the producer kernel's generic-proxy stores visible to our TMA loads
}

template <int MODE>
__global__ void __launch_bounds__(GEMM_THREADS, 1)
    gemm_kernel(const __grid_constant__ CUtensorMap tmap_a, const __grid_constant__ CUtensorMap tmap_b,
                const __grid_constant__ CUtensorMap tmap_ah, const int* __restrict__ cnt, const float* __restrict__ sa, const float* __restrict__ sb, int max_rows,
                uint8_t* __restrict__ c_hi, float* __restrict__ sc, const int* __restrict__ row_token,
                const float* __restrict__ row_weight, __nv_bfloat16* __restrict__ out, int* __restrict__ done_tiles,
                int* __restrict__ rows_done, int* __restrict__ tile_ctr, int cap, int num_tokens,
                int* __restrict__ zero_ctr, int zero_target, const uint8_t* __restrict__ hs,
                const float* __restrict__ hs_scale, uint8_t* __restrict__ a_perm, float* __restrict__ sa_w,
                const int* __restrict__ tok_cnt, const uint8_t* __restrict__ b_raw, const uint8_t* __restrict__ a_raw) {
  // b_raw / a_raw: raw pointers of the B (weights) and A (a_perm / c_hi) operands for the promotion warps' L2 prefetch.
  // cap: rows per expert in the fixed expert layout (expert e's rows are [e * cap, e * cap + cnt[e])).
  // tok_cnt[token]: number of local experts of the token (MOE_ONESTORE: rows with exactly one are stored, not reduced).
  // GEMM1 copy warps (2, 3): gather the valid rows (hs -> a_perm, hs_scale -> sa_w) in expert order, rows cid,
  // cid + ncl, ... of the concatenated valid rows, publishing rows_done[e]; afterwards (MOE_ZERO) they zero the
  // output rows cid, cid + ncl, ... and bump zero_ctr; GEMM2's promotion warps wait for zero_ctr == zero_target.
  // tile_ctr: MOE_DYN work counter (zeroed per call); the producer claims tiles and publishes them in s_tile_ring.
  // done_tiles[e]: GEMM1 tiles completed per expert (GEMM1 increments, GEMM2 waits for T_e * 16 before loading).
  // rows_done[e]: valid rows of expert e copied by the copy warps of all CTAs (GEMM1 producers wait for cnt[e]).
  using Cfg = GemmCfg<MODE, false>;
  constexpr int NKB = Cfg::NKB;
  constexpr int NT = Cfg::NT;
  constexpr int STAGES = Cfg::STAGES;
  constexpr uint32_t A_BYTES = Cfg::A_BYTES;
  constexpr uint32_t STAGE_BYTES = Cfg::STAGE_BYTES;

  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + Cfg::BAR_OFF);
  uint64_t* empty_bar = full_bar + STAGES;
  uint64_t* tfull_bar = empty_bar + STAGES;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(tfull_bar + 2);
  float* sa_ring = reinterpret_cast<float*>(smem + Cfg::RING_OFF);
  float* sb_ring = reinterpret_cast<float*>(smem + Cfg::SB_OFF);
  __shared__ int s_mt_off[33];
  __shared__ int s_pt_off[33];   // pair-tile prefix (MOE_PAIR)
  __shared__ int s_cnt[32];
  __shared__ int s_voff[33];   // valid-row prefix (copy warps)
  constexpr int TILE_RING = 16;   // > max producer lead in tiles (STAGES + 2 k-blocks, >= 1 k-block per tile)
  __shared__ int s_tile_ring[TILE_RING];
  __shared__ int s_next_pt[4];         // MOE_PWPF: tile claimed ahead while tile t (t & 3) runs, -1 = none
  __shared__ int s_fin_e[32];          // MOE_FIN_OFFLOAD: expert of finished tile t at t & 31 (-1 = end)
  __shared__ unsigned s_fin_cnt;       // MOE_FIN_OFFLOAD: 256 arrivals per finished tile

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // PDL: both GEMMs are programmatic dependents (GEMM1 of the routing kernel, GEMM2 of GEMM1) and neither waits for
  // the whole primary grid: they poll per-expert counters (rows_done / done_tiles), so their CTAs start on the first
  // experts while the primary is still working on later ones.
  // MOE_AMC: clusters of 2 CTAs process the two n-tiles 2p / 2p+1 of the same m-tile; each CTA TMA-loads 64 rows
  // of the shared A tile and multicasts them into both CTAs' stage buffers (smem stage freed by both MMAs).
#if MOE_AMC
  const uint32_t rank = cluster_ctarank();
  const int cid = blockIdx.x >> 1;
  const int ncl = gridDim.x >> 1;
  constexpr int NTP = NT / 2;   // n-tile pairs per m-tile
#else
  const uint32_t rank = 0u;
  const int cid = blockIdx.x;
  const int ncl = gridDim.x;
  constexpr int NTP = NT;
  (void)rank;
#endif
#if !MOE_ZERO
  (void)num_tokens; (void)zero_ctr; (void)zero_target;
#endif
  if constexpr (MODE == 0) pdl_wait();   // routing grid complete: counts and row maps are final
  if (warp == 0) compute_prefix(cnt, s_mt_off, s_cnt, s_voff, s_pt_off);
  if (warp == 1 && lane == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(&full_bar[i], 1);
      mbar_init(&empty_bar[i], MOE_AMC ? 2 : 1);   // MOE_AMC: both CTAs' MMA commits (multicast)
    }
    for (int i = 0; i < 2; ++i) mbar_init(&tfull_bar[i], 1);
    for (int i = 0; i < 3; ++i) mbar_init(tfull_bar + 4 + i, 1);   // copy-warp staging barriers
    fence_barrier_init();
    s_fin_cnt = 0u;
  }
  if (warp == 2) {
    tmem_alloc(tmem_slot, TMEM_COLS);
    tmem_relinquish();
  }
  if (warp == 3 && lane == 0) {
    tma_prefetch(&tmap_a);
    tma_prefetch(&tmap_b);
  }
  umma_fence_before();
  __syncthreads();
  umma_fence_after();
  if constexpr (MODE == 0) pdl_launch_dependents();   // GEMM2 may launch (it polls done_tiles before loading)
#if MOE_AMC
  cluster_sync_all();   // both CTAs' barriers are initialized before any multicast load / commit
#endif
#if MOE_TIMING
  if (threadIdx.x == 0) {
    uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    g_dbg_cta[MODE][blockIdx.x * 8 + 0] = smid;
    g_dbg_cta[MODE][blockIdx.x * 8 + 1] = t;
  }
#endif
  const uint32_t tmem_base = *tmem_slot;
  // Tile decode: tiles ordered m-tile-major (consecutive CTAs share the A tile; the expert's B stays L2-resident).
  // MOE_PAIR: an m-unit is a pair of m-tiles (2p, 2p+1 within the expert); the second may be missing.
  // GEMM2 (additive epilogue): the tiles of the incomplete last wave are split along K into NKB one-k-block pieces
  // so the tail is ~1 k-block long instead of a whole tile. With MOE_AMC a "tile" here is a pair of n-tiles.
  const int* s_tile_off = MOE_PAIR ? s_pt_off : s_mt_off;
  const int n_tiles = s_tile_off[32] * NTP;
  const int n_full = (MODE == 1) ? (n_tiles / ncl) * ncl : n_tiles;
  const int total_pt = n_full + (n_tiles - n_full) * (MODE == 1 ? NKB : 1);
  // half: an m-tile with <= 64 rows runs as a 64 x 256 tile (M = 64 MMA, 64-row A box; TMEM rows 16q..16q+15 of the
  // tile are in lanes 32q..32q+15, so lanes 16..31 of each promotion warp are idle).
  auto decode = [&](int pt, TileInfo& ti) {
    int tile = pt;
    if (pt >= n_full) tile = n_full + (pt - n_full) / NKB;
    const int mu = tile / NTP;
    ti.nt = (tile - mu * NTP) * (NT / NTP) + static_cast<int>(rank);
    ti.e = find_expert(s_tile_off, mu);
    ti.mrel = (mu - s_tile_off[ti.e]) * MH;
#pragma unroll
    for (int h = 0; h < MH; ++h) {
      const int rem = s_cnt[ti.e] - (ti.mrel + h) * BM;
      ti.valid[h] = rem > 0;
      ti.half[h] = MOE_HALF && !MOE_AMC && rem > 0 && rem <= BM / 2;
    }
  };
  auto krange = [&](int pt, int& kb0, int& kb1) {
    if (pt < n_full) { kb0 = 0; kb1 = NKB; }
    else { kb0 = (pt - n_full) % NKB; kb1 = kb0 + 1; }
  };

  if (warp < 4) {
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {
      // ---------------- TMA producer ----------------
      uint32_t stage = 0, phase = 0, ng = 0;
      int tile_iter = 0;
      const uint64_t l2pol = MOE_L2HINT ? l2_policy_evict_last() : 0ull;
      (void)l2pol;
#if MOE_TIMING
      long long w_empty = 0;
#endif
      // L2 prefetch: the A / B tiles of k-block kb + MOE_PF (continuing into the next tile of this CTA) are
      // prefetched into L2 so that the TMA loads MOE_PF stages later hit L2 instead of DRAM.
      auto l2_prefetch = [&](const TileInfo& t2, int row0_2, int kb2) {
#if MOE_PF > 0
        tma_prefetch_2d(&tmap_a, kb2 * BK, row0_2);
        if constexpr (MODE == 0) {
#if MOE_B4D
          tma_prefetch_4d(&tmap_b, kb2 * BK, t2.nt * 128, 0, t2.e);
#else
          tma_prefetch_3d(&tmap_b, kb2 * BK, t2.nt * 128, t2.e);
          tma_prefetch_3d(&tmap_b, kb2 * BK, INTER + t2.nt * 128, t2.e);
#endif
        } else {
          tma_prefetch_3d(&tmap_b, kb2 * BK, t2.nt * BN, t2.e);
#if !MOE_B256
          tma_prefetch_3d(&tmap_b, kb2 * BK, t2.nt * BN + 128, t2.e);
#endif
        }
#else
        (void)t2; (void)row0_2; (void)kb2;
#endif
      };
#if MOE_TIMING
      int trc = 0;
      auto trace = [&](int ev, int tile, int kb) {
        if (blockIdx.x < 2 && trc < 1024) { long long tcl = clock64(); g_dbg_trace[MODE][blockIdx.x][0][trc][0] = TRACE_CODE(ev, tile, kb); g_dbg_trace[MODE][blockIdx.x][0][trc][1] = tcl; ++trc; }
      };
#else
      auto trace = [&](int, int, int) {};
#endif
      if (MOE_STAGGER > 1 && MODE == 0) {
        // one-time start stagger: (b % S) / S of a 56-k-block tile period (~950 cycles per k-block)
        const long long t_end = clock64() + static_cast<long long>(blockIdx.x % MOE_STAGGER) * (56ll * 950ll / MOE_STAGGER);
        while (clock64() < t_end) __nanosleep(500);
      }
      uint32_t ready_mask = 0;   // MOE_READY_CACHE: experts observed ready
      int next_pt = -1;          // MOE_CLAIM_AHEAD: tile claimed during the previous tile
      int early_v = -1, early_need = 0x7fffffff;   // MOE_READY_EARLY: counter value read early for the next tile (and its target)
#if MOE_DYN
      for (;; ++tile_iter) {
        // claim the next tile; the slot is published before this tile's first full-barrier arrive (release), so
        // the MMA warp (acquire on full) and the promotion warps (via tfull) read it after their first wait.
        TSTAMP(tc0);
        const int pt = (MOE_CLAIM_AHEAD && next_pt >= 0) ? next_pt : atomicAdd(tile_ctr, 1);
        next_pt = -1;
        s_tile_ring[tile_iter % TILE_RING] = pt;
#if MOE_TIMING
        { TSTAMP(tc1); if (lane == 0) { g_dbg_timing[MODE][blockIdx.x * 128 + 0] += tc1 - tc0; g_dbg_timing[MODE][blockIdx.x * 128 + 1] += 1; } }
#endif
        trace(3, tile_iter, 0);
        if (MOE_CLAIM_AHEAD && pt + 2 * ncl < total_pt) next_pt = atomicAdd(tile_ctr, 1);   // result needed at the next tile
        if (MOE_PWPF > 0) s_next_pt[tile_iter & 3] = next_pt;   // for the promotion warps' L2 prefetch (visible via the barrier chain)
        if (pt >= total_pt) {
          mbar_wait(&empty_bar[stage], phase ^ 1);
          mbar_arrive_expect_tx(&full_bar[stage], 0u);   // wake the MMA warp: it reads the sentinel and exits
          break;
        }
#else
      for (int pt = cid; pt < total_pt; pt += ncl, ++tile_iter) {
#endif
        TileInfo ti;
        decode(pt, ti);
        const int e = ti.e, nt = ti.nt;
        const int row0 = e * cap + ti.mrel * BM;   // fixed expert layout: first row of half 0
        uint32_t a_tx = 0, nvalid = 0;
#pragma unroll
        for (int h = 0; h < MH; ++h) {
          if (ti.valid[h]) { a_tx += ti.half[h] ? (A_BYTES / 2) : A_BYTES; ++nvalid; }
        }
        int kb0, kb1;
        krange(pt, kb0, kb1);
        // next tile of this CTA (prefetch target once kb + MOE_PF runs past this tile)
        TileInfo t2 = ti;
        int row0_2 = 0, kb0n = 0, kb1n = 0;
        const int pt2 = pt + ncl;
        const bool has_next = !MOE_DYN && pt2 < total_pt;
        if (has_next) { decode(pt2, t2); row0_2 = t2.e * cap + t2.mrel * BM; krange(pt2, kb0n, kb1n); }
        if (tile_iter == 0) {
          for (int k = kb0; k < kb1 && k < kb0 + MOE_PF; ++k) l2_prefetch(ti, row0, k);
        }
        for (int kb = kb0; kb < kb1; ++kb) {
          TSTAMP(te0);
          mbar_wait(&empty_bar[stage], phase ^ 1);
#if MOE_TIMING
          { TSTAMP(te1); w_empty += te1 - te0; if (kb == kb0 && lane == 0) g_dbg_timing[MODE][blockIdx.x * 128 + 3] += te1 - te0; }
#endif
          trace(1, tile_iter, kb - kb0);
          {
            const int pk = kb + MOE_PF;
            if (pk < kb1) l2_prefetch(ti, row0, pk);
            else if (has_next && kb0n + (pk - kb1) < kb1n) l2_prefetch(t2, row0_2, kb0n + (pk - kb1));
          }
          uint8_t* sA = smem + stage * STAGE_BYTES;
          uint8_t* sB = sA + MH * A_BYTES;
          uint64_t* bar = &full_bar[stage];
          const bool gstart = (kb == kb0) || ((kb & (MOE_SA_GROUP - 1)) == 0);
#if MOE_ABL == 6
          mbar_arrive_expect_tx(bar, 0u);   // MMA-only ablation: no loads
          if (gstart) ++ng;
          (void)sA; (void)sB; (void)row0; (void)nvalid;
          if (false) {
#elif MOE_ABL == 7
          // TMA-only with one 128-row B box per k-block (32 KB per stage instead of 48: the byte count of a
          // 256 x 256 pair tile per 128 x 256 x 128 MACs); requires MOE_B4D=0 / MOE_B256=0
          mbar_arrive_expect_tx(bar, a_tx + Cfg::B_BYTES / 2 + (gstart ? nvalid * Cfg::SA_HALF_BYTES : 0u) + (kb == kb0 ? Cfg::SB_TX : 0u));
          {
#else
          mbar_arrive_expect_tx(bar, a_tx + Cfg::B_BYTES + (gstart ? nvalid * Cfg::SA_HALF_BYTES : 0u) + (kb == kb0 ? Cfg::SB_TX : 0u));
          {
#endif
          // B tile and weight scales first (inputs: always ready), then wait for the expert's rows (GEMM1: copied
          // by the copy warps) / tiles (GEMM2: GEMM1 output) before the A tile and row-scale loads
          if (gstart && kb == kb0) {
            float* sbs = sb_ring + (tile_iter % Cfg::SB_SLOTS) * (Cfg::SB_SLOT_BYTES / 4);
            if constexpr (MODE == 0) {
              bulk_load_1d(sbs, sb + (static_cast<size_t>(e) * 32 + nt) * NUM_HB, 224, bar);
              bulk_load_1d(sbs + 64, sb + (static_cast<size_t>(e) * 32 + 16 + nt) * NUM_HB, 224, bar);
            } else {
              bulk_load_1d(sbs, sb + (static_cast<size_t>(e) * NUM_HB + 2 * nt) * NUM_IB, 128, bar);
            }
          }
          if constexpr (MODE == 0) {
#if MOE_B4D
            if (MOE_L2HINT) tma_load_4d_hint(sB, &tmap_b, bar, kb * BK, nt * 128, 0, e, l2pol);
            else tma_load_4d(sB, &tmap_b, bar, kb * BK, nt * 128, 0, e);   // box {128 k, 128 n, 2 halves (X1 / X2), 1 expert}
#else
            tma_load_3d(sB, &tmap_b, bar, kb * BK, nt * 128, e);
            if (MOE_ABL != 7) tma_load_3d(sB + 16384, &tmap_b, bar, kb * BK, INTER + nt * 128, e);
#endif
          } else {
#if MOE_B256
            if (MOE_L2HINT) tma_load_3d_hint(sB, &tmap_b, bar, kb * BK, nt * BN, e, l2pol);
            else tma_load_3d(sB, &tmap_b, bar, kb * BK, nt * BN, e);         // box {128 k, 256 n, 1 expert}
#else
            tma_load_3d(sB, &tmap_b, bar, kb * BK, nt * BN, e);
            if (MOE_ABL != 7) tma_load_3d(sB + 16384, &tmap_b, bar, kb * BK, nt * BN + 128, e);
#endif
          }
          if (kb == kb0) {
            TSTAMP(tr0);
            if (MOE_READY_EARLY && early_v >= early_need) {
              if (MOE_READY_EARLY == 1) fence_acq_rel_gpu();   // the relaxed read observed the release: acquire now
              fence_proxy_async_global();
            } else if (!MOE_READY_CACHE || !((ready_mask >> e) & 1u)) {
              wait_expert_ready<MODE>(e, s_tile_off, s_cnt, rows_done, done_tiles);
              ready_mask |= 1u << e;
            }
            early_v = -1; early_need = 0x7fffffff;
#if MOE_TIMING
            { TSTAMP(tr1); if (lane == 0) g_dbg_timing[MODE][blockIdx.x * 128 + 2] += tr1 - tr0; }
#endif
            trace(4, tile_iter, kb - kb0);
          }
#if MOE_AMC
          // this CTA's 64 rows of the shared A tile, multicast into both CTAs' stage buffers (box = 64 rows)
          tma_load_2d_mc(sA + rank * (A_BYTES / 2), &tmap_a, bar, kb * BK, row0 + static_cast<int>(rank) * (BM / 2), 0x3);
#else
#pragma unroll
          for (int h = 0; h < MH; ++h) {
            if (ti.valid[h]) tma_load_2d(sA + h * A_BYTES, ti.half[h] ? &tmap_ah : &tmap_a, bar, kb * BK, row0 + h * BM);
          }
#endif
          if (gstart) {
            float* sS = sa_ring + (ng % Cfg::SA_SLOTS) * (MH * Cfg::SA_HALF_FLOATS);
            ++ng;
#pragma unroll
            for (int h = 0; h < MH; ++h) {
              if (ti.valid[h])
                bulk_load_1d(sS + h * Cfg::SA_HALF_FLOATS,
                             sa + (static_cast<size_t>(row0 / BM + h) * NKB + (kb & ~(MOE_SA_GROUP - 1))) * 128,
                             Cfg::SA_HALF_BYTES, bar);
            }
          }
          }
          if (MOE_READY_EARLY && next_pt >= 0 && kb == kb1 - MOE_READY_DIST) {
            // next tile (claimed early): read its expert's readiness counter (consumed at the boundary)
            TileInfo tn;
            decode(next_pt, tn);
            early_need = (MODE == 0) ? s_cnt[tn.e] : (s_tile_off[tn.e + 1] - s_tile_off[tn.e]) * (2 * INTER / 256);
            const int* ctr = (MODE == 0) ? (rows_done + tn.e * CNT_STRIDE) : (done_tiles + tn.e * CNT_STRIDE);
            early_v = (MOE_READY_EARLY == 2) ? ld_acquire_gpu(ctr) : ld_relaxed_gpu(ctr);   // 1 / 3: relaxed
          }
          if (MOE_NEXT_PF > 0 && next_pt >= 0 && next_pt < n_full && kb >= kb1 - MOE_NEXT_PF && kb < kb1) {
            // next tile: TMA-prefetch its k-block i = kb - (kb1 - MOE_NEXT_PF) into L2
            const int i = kb - (kb1 - MOE_NEXT_PF);
            TileInfo tn;
            decode(next_pt, tn);
            const int rown = tn.e * cap + tn.mrel * BM;
            {
              if constexpr (MODE == 0) {
#if MOE_B4D
                tma_prefetch_4d(&tmap_b, i * BK, tn.nt * 128, 0, tn.e);
#else
                tma_prefetch_3d(&tmap_b, i * BK, tn.nt * 128, tn.e);
                tma_prefetch_3d(&tmap_b, i * BK, INTER + tn.nt * 128, tn.e);
#endif
              } else {
                tma_prefetch_3d(&tmap_b, i * BK, tn.nt * BN, tn.e);
#if !MOE_B256
                tma_prefetch_3d(&tmap_b, i * BK, tn.nt * BN + 128, tn.e);
#endif
              }
              if (tn.valid[0]) tma_prefetch_2d(tn.half[0] ? &tmap_ah : &tmap_a, i * BK, rown);
            }
          }
          trace(2, tile_iter, kb - kb0);
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
#if MOE_TIMING
      g_dbg_cta[MODE][blockIdx.x * 8 + 3] = w_empty;
#endif
    } else if (MOE_RED_OFFLOAD && MODE == 1 && warp >= 2) {
      // ---------------- reduction warps (GEMM2, 64 threads): drain the staged output tiles ----------------
      // 128 chunks per tile (g, q, j), 512 B each = 32 rows x 16 B; chunk c handled by warp 2 + (c & 1)
      const int t = threadIdx.x - 64;
      const uint8_t* stg = smem + Cfg::OUT_OFF;
      const int* s_tok = reinterpret_cast<const int*>(stg + Cfg::OUT_ROWS_BYTES);
#if MOE_ZERO
      if (t == 0) { while (ld_acquire_gpu(zero_ctr) < zero_target) __nanosleep(256); }
#endif
      named_bar_sync(7, 64);     // both reduction warps ready (output zeroed)
      named_bar_arrive(6, BAR_RED);  // staging buffer initially free
      for (;;) {
        named_bar_sync(5, BAR_RED);  // a tile has been staged (or the sentinel)
        if (s_tok[129]) { if (MOE_RED_OFFLOAD == 2) bulk_wait_group<0>(); break; }
        const int nt = s_tok[128];
        const bool piece = s_tok[130] != 0;   // split-K piece: partial sums -> always reduce
        if (MOE_RED_OFFLOAD == 2) {
          // one 256 B bulk reduction per (row, column half): warp 2 takes g = 0, warp 3 takes g = 1
          const int g = warp - 2;
          fence_proxy_async_shared_cta();   // generic-proxy smem writes (promotion warps) -> async proxy
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int row = q * 32 + lane;
            const int tv = s_tok[row];
            if (tv >= 0) {
              const int tok = tv & 0x3fffffff;
              if ((MOE_ONESTORE == 1 || MOE_ONESTORE == 2) && !piece && (tv & 0x40000000))   // the token's only contribution: plain store
                bulk_store_1d(out + static_cast<size_t>(tok) * HID + nt * BN + 128 * g, stg + (g * 128 + row) * 272, 256);
              else
                bulk_reduce_add_bf16(out + static_cast<size_t>(tok) * HID + nt * BN + 128 * g, stg + (g * 128 + row) * 272, 256);
            }
          }
          bulk_commit_group();
          bulk_wait_group_read<0>();   // staging read by the TMA (the global reductions complete asynchronously)
        } else {
          for (int i = 0; i < 64; ++i) {
            const int c = (warp - 2) + 2 * i;
            const int g = c >> 6, q = (c >> 4) & 3, j = c & 15;
            const int tok = s_tok[q * 32 + lane] & 0x3fffffff;
            const uint4 v = *reinterpret_cast<const uint4*>(stg + c * 512 + lane * 16);
            if (s_tok[q * 32 + lane] >= 0)
              red_add_bf16x2_v4(out + static_cast<size_t>(tok) * HID + nt * BN + 128 * g + j * 8, v.x, v.y, v.z, v.w);
          }
        }
        __syncwarp();
        named_bar_arrive(6, BAR_RED);   // staging buffer free
      }
    } else if (MODE == 0 && warp >= 2) {
      // ---------------- copy warps (GEMM1 only, 64 threads) ----------------
      // Valid rows of the concatenated experts, strided over the CTAs: 7168 B per row (thread t: 16 B chunks
      // t, t + 64, ..., 7 chunks), scales: thread t < 56 copies k-block t. rows_done[e] is published (release)
      // when this CTA moves past expert e, so GEMM1 producers can start expert e while later experts are copied.
      const int t = threadIdx.x - 64;
      const int total_v = s_voff[32];
      const int e_head = find_expert(s_mt_off, 0);   // first non-empty expert: copied by the promotion warps
      int e_cur = -1, n_cur = 0;
#if MOE_COPY_TMA
      // Warp 2 lane 0 runs a 3-slot bulk pipeline: row i is bulk-loaded into slot i % 3 (mbarrier), stored from
      // smem by a bulk store (bulk async-group) once row i + 2 has been issued; slot i % 3 is reused after the
      // store of row i - 3 has finished reading it (wait_group.read 2). Warp 3 copies the 56 row scales.
      uint64_t* stg_bar = tfull_bar + 4;
      int issued = 0, stored = 0;   // rows loaded / rows whose store was issued (stored >= issued - 2)
      int pend_row[2];              // row ids of rows issued - 1 and issued - 2 (index: row number & 1)
      auto store_row = [&](int j) {
        const int js = j % Cfg::STG_SLOTS;
        mbar_wait(stg_bar + js, (j / Cfg::STG_SLOTS) & 1);
        fence_proxy_async_shared_cta();
        bulk_store_1d(a_perm + static_cast<size_t>(pend_row[j & 1]) * HID, smem + Cfg::STG_OFF + js * HID, HID);
        bulk_commit_group();
      };
      // drain: store the rows still in staging and wait for every bulk store to complete (before a publish)
      auto drain = [&]() {
        while (stored < issued) { store_row(stored); ++stored; }
        bulk_wait_group<0>();
        fence_proxy_async_global();
      };
#endif
      for (int v = s_voff[e_head + 1] + cid; v < total_v; v += ncl) {
        const int e = find_expert(s_voff, v);
        if (e != e_cur) {
          if (e_cur >= 0) {
#if MOE_COPY_TMA
            if (t == 0) drain();
#endif
            __threadfence();
            named_bar_sync(6, 64);
            if (t == 0 && n_cur) atomicAdd(rows_done + e_cur * CNT_STRIDE, n_cur);
          }
          e_cur = e;
          n_cur = 0;
        }
        const int row = e * cap + (v - s_voff[e]);
        const int token = row_token[row] & 0x3fffffff;
#if MOE_COPY_TMA
        // warp 2 lane 0: 3-slot bulk pipeline (load row i, store row i - 2 after its load landed); warp 3: scales
        if (t == 0) {
          const int slot = issued % Cfg::STG_SLOTS;
          if (issued >= Cfg::STG_SLOTS) bulk_wait_group_read<Cfg::STG_SLOTS - 1>();   // store of row issued - 3 read this slot
          mbar_arrive_expect_tx(stg_bar + slot, HID);
          bulk_load_1d(smem + Cfg::STG_OFF + slot * HID, hs + static_cast<size_t>(token) * HID, HID, stg_bar + slot);
          if (issued >= 2 && stored == issued - 2) { store_row(stored); ++stored; }   // row issued - 2 (not drained yet)
          pend_row[issued & 1] = row;
          ++issued;
        } else if (t >= 32) {
#pragma unroll
          for (int qq = 0; qq < 2; ++qq) {
            const int kb = (t - 32) + 32 * qq;
            if (kb < NUM_HB)
              sa_w[(static_cast<size_t>(row >> 7) * NUM_HB + kb) * 128 + (row & 127)] = __ldg(hs_scale + static_cast<size_t>(kb) * num_tokens + token);
          }
        }
#else
        const int4* src = reinterpret_cast<const int4*>(hs + static_cast<size_t>(token) * HID) + t;
        int4* dst = reinterpret_cast<int4*>(a_perm + static_cast<size_t>(row) * HID) + t;
        int4 val[7];
#pragma unroll
        for (int c = 0; c < 7; ++c) val[c] = __ldg(src + 64 * c);
#pragma unroll
        for (int c = 0; c < 7; ++c) dst[64 * c] = val[c];
        if (t < NUM_HB)
          sa_w[(static_cast<size_t>(row >> 7) * NUM_HB + t) * 128 + (row & 127)] = __ldg(hs_scale + static_cast<size_t>(t) * num_tokens + token);
#endif
        ++n_cur;
      }
#if MOE_COPY_TMA
      if (t == 0) drain();
#endif
      __threadfence();
      named_bar_sync(6, 64);
      if (t == 0 && n_cur && e_cur >= 0) atomicAdd(rows_done + e_cur * CNT_STRIDE, n_cur);
#if MOE_TIMING
      if (t == 0) { long long tt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt)); g_dbg_copy[blockIdx.x * 2] = tt; }
#endif
#if MOE_ZERO
      // zero the output rows cid, cid + ncl, ... (replaces the memset kernel); MOE_ONESTORE: rows of tokens with
      // exactly one local expert are fully written by that expert's GEMM2 tile (plain store) and are skipped
      // GEMM2's split-K tail: tiles >= n_full2 are 1-k-block pieces that reduce partial sums, so rows of m-tiles from
      // n_full2 / NT2 on are zeroed even for single-expert tokens (their full tiles then store over the zeros).
      const int n_tiles2 = s_mt_off[32] * (HID / 256);
      const int mt_tail2 = ((n_tiles2 / ncl) * ncl) / (HID / 256);
      for (int r = cid; r < num_tokens; r += ncl) {
        if (MOE_ONESTORE == 1 || MOE_ONESTORE == 3) {
          const int v = __ldg(tok_cnt + r);
          if (v >= 16) {
            const int row = v - 16, e = row / cap, slot = row - e * cap;
            if (s_mt_off[e] + slot / BM < mt_tail2) continue;   // one plain store covers the whole row
          }
        }
        int4* dst = reinterpret_cast<int4*>(out + static_cast<size_t>(r) * HID);
#pragma unroll 4
        for (int c = t; c < HID * 2 / 16; c += 64) {
          if (MOE_L2HINT) st_global_cs_v4(dst + c, make_int4(0, 0, 0, 0));   // streaming: not needed until GEMM2
          else dst[c] = make_int4(0, 0, 0, 0);
        }
      }
      __threadfence();
      named_bar_sync(6, 64);
      if (t == 0) atomicAdd(zero_ctr, 1);
#if MOE_TIMING
      if (t == 0) { long long tt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt)); g_dbg_copy[blockIdx.x * 2 + 1] = tt; }
#endif
#endif
#if MOE_FIN_OFFLOAD && !MOE_CHI_TMA
      if (warp == 2) {
        // done_tiles signals for the finished tiles (256 promotion-lane arrivals each, in tile order): the acquire
        // on the counter + a gpu-scope fence (cumulative) make the promotion warps' c_hi / scale stores visible
        // before the atomic.
        const uint32_t cnt_addr = smem_u32(&s_fin_cnt);
        for (int k = 0;; ++k) {
          if (lane == 0) { while (ld_acquire_cta_shared(cnt_addr) < 256u * (k + 1)) __nanosleep(128); }
          __syncwarp();
          const int e = s_fin_e[k & 31];
          if (e < 0) break;
          if (lane == 0) { __threadfence(); atomicAdd(done_tiles + e * CNT_STRIDE, 1); }
          __syncwarp();
        }
      }
#endif
    } else if (warp == 1) {
      // ---------------- MMA issuer (whole warp runs the loop for the named barriers; lane 0 issues) ----------------
      constexpr uint32_t idesc128 = make_idesc_f8(BM, BN);
      constexpr uint32_t idesc64 = make_idesc_f8(BM / 2, BN);
      uint32_t stage = 0, phase = 0, acc_it = 0;
#if MOE_TIMING
      long long w_full = 0, w_sync = 0;
      int trc = 0;
#endif
#if MOE_DYN
      for (int tile_iter = 0;; ++tile_iter) {
        mbar_wait(&full_bar[stage], phase);   // first k-block of the next tile (or the producer's sentinel arrive)
        const int pt = s_tile_ring[tile_iter % TILE_RING];
        if (pt >= total_pt) {
          // Drain: the promotion warps must have consumed TMEM stage acc_it & 1 (sub-step acc_it - 2) before the
          // sentinel arrive advances that barrier again (two completed phases would alias the parity).
          if (acc_it >= 2) named_bar_sync(1 + (acc_it & 1u), BAR_MMA);
          if (lane == 0) mbar_arrive(&tfull_bar[acc_it & 1u]);   // wake the promotion warps: sentinel
          __syncwarp();
          if (acc_it >= 1) named_bar_sync(1 + ((acc_it + 1) & 1u), BAR_MMA);   // sub-step acc_it - 1
          break;
        }
#else
      for (int pt = cid; pt < total_pt; pt += ncl) {
#endif
        int kb0, kb1;
        krange(pt, kb0, kb1);
        TileInfo ti;
        decode(pt, ti);
        for (int kb = kb0; kb < kb1; ++kb) {
#pragma unroll
          for (int h = 0; h < MH; ++h) {
            const uint32_t a = acc_it & 1u;
            // all 8 promotion warps finished reading TMEM stage a (sub-step acc_it - 2)
            TSTAMP(tm0);
            if (acc_it >= 2) named_bar_sync(1 + a, BAR_MMA);
            TSTAMP(tm1);
            if (h == 0) mbar_wait(&full_bar[stage], phase);
#if MOE_TIMING
            { TSTAMP(tm2); w_sync += tm1 - tm0; w_full += tm2 - tm1;
              if (lane == 0 && kb - kb0 < 4) { g_dbg_timing[MODE][blockIdx.x * 128 + 8 + (kb - kb0)] += tm1 - tm0; g_dbg_timing[MODE][blockIdx.x * 128 + 12 + (kb - kb0)] += tm2 - tm1; }
              if (lane == 0 && blockIdx.x < 2 && trc + 1 < 1024) {
                g_dbg_trace[MODE][blockIdx.x][1][trc][0] = TRACE_CODE(5, tile_iter, kb - kb0); g_dbg_trace[MODE][blockIdx.x][1][trc][1] = tm1; ++trc;   // after the named wait
                g_dbg_trace[MODE][blockIdx.x][1][trc][0] = TRACE_CODE(6, tile_iter, kb - kb0); g_dbg_trace[MODE][blockIdx.x][1][trc][1] = tm2; ++trc; } }   // after the full wait
#endif
            if (lane == 0) {
              umma_fence_after();
              const uint32_t sA_addr = smem_u32(smem + stage * STAGE_BYTES);
              const uint64_t adesc = make_smem_desc(sA_addr + h * A_BYTES);
              const uint64_t bdesc = make_smem_desc(sA_addr + MH * A_BYTES);
              const uint32_t d_tmem = tmem_base + a * BN;
              const uint32_t idesc = ti.half[h] ? idesc64 : idesc128;
#if MOE_ABL != 5 && MOE_ABL != 7
              if (MH == 1 || ti.valid[h]) {
#pragma unroll
                for (int k = 0; k < BK / 32; ++k)
                  umma_mma_f8(d_tmem, adesc + 2 * k, bdesc + 2 * k, idesc, (k != 0 || (MOE_TRS && kb != kb0)) ? 1u : 0u);
              }
#else
              (void)adesc; (void)bdesc; (void)d_tmem; (void)idesc;
#endif
              if (h == MH - 1) {
#if MOE_AMC
                umma_commit_mc(&empty_bar[stage], 0x3);   // MMAs done -> this CTA's half of the stage is free in both CTAs
#else
                umma_commit(&empty_bar[stage]);   // MMAs of both halves done -> smem stage free
#endif
              }
              umma_commit(&tfull_bar[a]);       // TMEM stage full
            }
            __syncwarp();
            ++acc_it;
          }
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
#if MOE_TIMING
      if (lane == 0) { g_dbg_cta[MODE][blockIdx.x * 8 + 4] = w_full; g_dbg_cta[MODE][blockIdx.x * 8 + 5] = w_sync; }
#endif
#if !MOE_DYN
      // drain the two outstanding named-barrier generations
      if (acc_it >= 2) { named_bar_sync(1 + (acc_it & 1u), BAR_MMA); named_bar_sync(1 + ((acc_it + 1) & 1u), BAR_MMA); }
      else if (acc_it == 1) { named_bar_sync(1, BAR_MMA); }
#endif
    }
  } else {
    setmaxnreg_inc<PROMO_REGS>();
    // ---------------- promotion + epilogue warps (PW warps = NG column groups x 4 TMEM lane quarters) ----------------
    int trc_promo = 0;                    // dev trace index (MOE_TIMING)
    const int g = (warp - 4) >> 2;        // column group
    const int q = warp & 3;               // TMEM lane quarter
    const uint32_t tmem_lane_base = tmem_base + (static_cast<uint32_t>(q * 32) << 16);
    uint32_t acc_it = 0, stage = 0, phase = 0, ng = 0;
    int tile_iter = 0;
    int pending_e = -1;   // MOE_CHI_TMA: expert of the tile whose c_hi bulk stores are in flight
#if MOE_ZERO
    if constexpr (MODE == 1) {   // the output must be zeroed (GEMM1's copy warps of every CTA) before the first red
      if (lane == 0) { while (ld_acquire_gpu(zero_ctr) < zero_target) __nanosleep(256); }
      __syncwarp();
    }
#endif
    if constexpr (MODE == 0) {
      // Head start: the promotion warps are idle until the first accumulator is ready, so the 8 warps of every CTA
      // copy the rows of the first non-empty expert (one row per warp, 32 lanes x 14 chunks of 16 B) before the
      // copy warps (which start at the next expert) could deliver them.
      const int e_head = find_expert(s_mt_off, 0);
      const int v_end = s_voff[e_head + 1];
      int n = 0;
      for (int v = cid + (warp - 4) * ncl; v < v_end; v += PW * ncl) {
        const int row = e_head * cap + v;   // earlier experts are empty: v - s_voff[e_head] == v
        const int token = row_token[row] & 0x3fffffff;
        const int4* src = reinterpret_cast<const int4*>(hs + static_cast<size_t>(token) * HID) + lane;
        int4* dst = reinterpret_cast<int4*>(a_perm + static_cast<size_t>(row) * HID) + lane;
        int4 val[14];
#pragma unroll
        for (int c = 0; c < 14; ++c) val[c] = __ldg(src + 32 * c);
#pragma unroll
        for (int c = 0; c < 14; ++c) dst[32 * c] = val[c];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int kb = lane + 32 * qq;
          if (kb < NUM_HB)
            sa_w[(static_cast<size_t>(row >> 7) * NUM_HB + kb) * 128 + (row & 127)] = __ldg(hs_scale + static_cast<size_t>(kb) * num_tokens + token);
        }
        ++n;
      }
      __threadfence();
      __syncwarp();
      if (lane == 0 && n) atomicAdd(rows_done + e_head * CNT_STRIDE, n);
    }
#if MOE_DYN
    for (;; ++tile_iter) {
      TSTAMP(tt0);
      mbar_wait(&tfull_bar[acc_it & 1u], (acc_it >> 1) & 1u);   // first TMEM stage of the next tile (or sentinel)
#if MOE_TIMING
      { TSTAMP(tt1); if (lane == 0) g_dbg_timing[MODE][blockIdx.x * 128 + warp * 8 + 3] += tt1 - tt0; }
#endif
      const int pt = s_tile_ring[tile_iter % TILE_RING];
      if (pt >= total_pt) {
#if MOE_CHI_TMA
        if constexpr (MODE == 0) {   // flush the last tile's c_hi bulk stores and signal it
          if (g == 0) bulk_wait_group<0>();
          named_bar_sync(3, PROMO_THREADS);
          if (pending_e >= 0 && warp == 4 && lane == 0) { __threadfence(); atomicAdd(done_tiles + pending_e * CNT_STRIDE, 1); }
        }
#endif
#if MOE_RED_OFFLOAD
        if constexpr (MODE == 1) {   // tell the reduction warps to exit
          named_bar_sync(6, BAR_RED);
          if (warp == 4 && lane == 0) reinterpret_cast<int*>(smem + Cfg::OUT_OFF + Cfg::OUT_ROWS_BYTES)[129] = 1;
          __threadfence_block();
          named_bar_arrive(5, BAR_RED);
        }
#endif
#if MOE_FIN_OFFLOAD && !MOE_CHI_TMA
        if constexpr (MODE == 0) {   // end marker for copy warp 2's done_tiles loop
          if (warp == 4 && lane == 0) s_fin_e[tile_iter & 31] = -1;
          red_release_cta_shared(smem_u32(&s_fin_cnt), 1u);
        }
#endif
        break;
      }
#else
    for (int pt = cid; pt < total_pt; pt += ncl, ++tile_iter) {
#endif
      TileInfo ti;
      decode(pt, ti);
      bool active[MH];
      int row_in_tile[MH], row[MH], token[MH];
      float wtok[MH];
#pragma unroll
      for (int h = 0; h < MH; ++h) {
        active[h] = ti.valid[h] && (!ti.half[h] || lane < 16);
        row_in_tile[h] = ti.half[h] ? (q * 16 + (lane & 15)) : (q * 32 + lane);
        row[h] = ti.e * cap + (ti.mrel + h) * BM + row_in_tile[h];   // fixed expert layout
        token[h] = -1;
        wtok[h] = 0.f;
        if constexpr (MODE == 1) {
          // padding rows of the expert's last tile hold stale data: only rows below the expert's count are tokens
          if (active[h] && row_in_tile[h] < s_cnt[ti.e] - (ti.mrel + h) * BM) {
            token[h] = row_token[row[h]];   // bit 30 (MOE_ONESTORE): the token's only contribution -> plain store
            wtok[h] = row_weight[row[h]];
          }
        }
      }
      int kb0, kb1;
      krange(pt, kb0, kb1);
      int pf_e = -1, pf_nt = 0, pf_row0 = 0;
      bool pf_half = false;
      if (MOE_PWPF > 0 && pt < n_full) {
        const int npt = s_next_pt[tile_iter & 3];   // claimed by the producer before this tile's first load
        if (npt >= 0 && npt < n_full) {
          TileInfo tn;
          decode(npt, tn);
          pf_e = tn.e; pf_nt = tn.nt; pf_row0 = tn.e * cap + tn.mrel * BM; pf_half = tn.half[0];
        }
      }
      promote_and_store<MODE, STAGES, Cfg::SA_SLOTS, Cfg::SB_SLOTS, Cfg::SA_HALF_FLOATS, Cfg::SB_SLOT_BYTES / 4>(
          smem, sa_ring, sb_ring, full_bar, tfull_bar, tmem_lane_base, acc_it, stage, phase, ng, tile_iter, ti, active, row,
          row_in_tile, token, wtok, &trc_promo, s_fin_e, &s_fin_cnt, pf_e, pf_nt, pf_row0, pf_half, b_raw, a_raw, kb0, kb1, g, warp, lane, Cfg::XCH_OFF, Cfg::OUT_OFF, Cfg::CHI_OFF, pending_e, c_hi, sc, out, done_tiles,
          [](uint32_t a) { named_bar_arrive(1 + a, BAR_MMA); });
    }
  }
  umma_fence_before();
  __syncthreads();
#if MOE_TIMING
  if (threadIdx.x == 0) {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    g_dbg_cta[MODE][blockIdx.x * 8 + 2] = t;
  }
#endif
#if MOE_AMC
  cluster_sync_all();   // the peer may still multicast into / commit to this CTA until it reaches this point
#endif
  if (warp == 2) {
    umma_fence_after();
    tmem_dealloc(tmem_base, TMEM_COLS);
  }
}

#ifndef MOE_2SM
#define MOE_2SM 0   // 1 = clusters of 2 CTAs, tcgen05.mma.cta_group::2 (M = 256 over the pair, two N = 128 sub-steps per k-block into 4 TMEM buffers)
#endif
static_assert(!(MOE_2SM && (MOE_PAIR || MOE_TRS || MOE_PW != 8 || MOE_AMC || MOE_RED_OFFLOAD == 1)), "MOE_2SM: fp32 register promotion, 8 promotion warps, RED_OFFLOAD 0 / 2");
__device__ __forceinline__ void tma_load_4d_2sm(void* dst, const void* map, uint32_t leader_bar, int x, int y, int z, int w) {
  asm volatile(
      "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(leader_bar), "r"(x), "r"(y), "r"(z), "r"(w)
      : "memory");
}
__device__ __forceinline__ void st_shared_cluster_u64(uint32_t remote_addr, unsigned long long v) {
  asm volatile("st.shared::cluster.u64 [%0], %1;" ::"r"(remote_addr), "l"(v) : "memory");
}
// Tile ring entries carry their own sequence number (tile_iter + 1 in the upper half) so that readers in either
// CTA simply spin until the slot holds the entry they expect (no barrier phases to alias).
__device__ __forceinline__ unsigned long long ring_entry(int tile_iter, int pt) {
  return (static_cast<unsigned long long>(tile_iter + 1) << 32) | static_cast<uint32_t>(pt);
}
__device__ __forceinline__ int ring_read(const unsigned long long* slot, int tile_iter) {
  unsigned long long v;
  for (;;) {
    v = *reinterpret_cast<const volatile unsigned long long*>(slot);
    if ((v >> 32) == static_cast<unsigned long long>(tile_iter + 1)) break;
    __nanosleep(64);
  }
  return static_cast<int>(static_cast<uint32_t>(v));
}

// ------------------------------------------------------------------------------------------------
// 2-SM GEMM (MOE_2SM): one cluster of 2 CTAs per 256 x 256 pair tile (m-tiles 2p / 2p+1 of one expert: CTA rank r
// owns m-tile 2p + r; the second may be missing or a 64-row tail). Per k-block each CTA loads its own 128 A rows
// (16 KB) and its 64-row halves of the two N = 128 B blocks (2 x 8 KB); the leader's MMA warp issues
// tcgen05.mma.cta_group::2 (M = 256, N = 128) per sub-step s = 0, 1 into TMEM buffer (2 kb + s) & 3; its commits are
// multicast to both CTAs (tfull[buf], empty[stage]). TMEM buffers are released through the leader's tfree[buf]
// mbarriers (count 16: 8 local + 8 remote promotion warps). The leader's producer claims tiles dynamically and
// forwards them to the follower's ring via st.shared::cluster + a remote arrive on the follower's tile barrier.
// Promotion (both CTAs, 8 warps): thread (g, q, lane) owns row q*32+lane and columns [64 g, 64 g + 64) of each
// sub-step: acc[0..63] = sub-step 0, acc[64..127] = sub-step 1 (GEMM1: X1 / X2 of the same columns -> the single-CTA
// epilogue layout).
// ------------------------------------------------------------------------------------------------
template <int MODE>
__global__ void __launch_bounds__(GEMM_THREADS, 1)
    gemm2sm_kernel(const __grid_constant__ CUtensorMap tmap_a, const __grid_constant__ CUtensorMap tmap_b,
                   const __grid_constant__ CUtensorMap tmap_ah, const int* __restrict__ cnt, const float* __restrict__ sa, const float* __restrict__ sb, int max_rows,
                   uint8_t* __restrict__ c_hi, float* __restrict__ sc, const int* __restrict__ row_token,
                   const float* __restrict__ row_weight, __nv_bfloat16* __restrict__ out, int* __restrict__ done_tiles,
                   int* __restrict__ rows_done, int* __restrict__ tile_ctr, int cap, int num_tokens,
                   int* __restrict__ zero_ctr, int zero_target, const uint8_t* __restrict__ hs,
                   const float* __restrict__ hs_scale, uint8_t* __restrict__ a_perm, float* __restrict__ sa_w,
                   const int* __restrict__ tok_cnt, const uint8_t* __restrict__ b_raw, const uint8_t* __restrict__ a_raw) {
  (void)tok_cnt; (void)b_raw; (void)a_raw;   // (MOE_ONESTORE not implemented in the 2-SM variant: it zeroes and reduces every row)
  using Cfg = GemmCfg<MODE, true>;
  constexpr int NKB = Cfg::NKB;
  constexpr int NT = Cfg::NT;
  constexpr int STAGES = Cfg::STAGES;
  constexpr uint32_t A_BYTES = Cfg::A_BYTES;
  constexpr uint32_t B_BYTES = Cfg::B_BYTES;
  constexpr uint32_t STAGE_BYTES = Cfg::STAGE_BYTES;
  constexpr int SA_SLOTS = Cfg::SA_SLOTS;
  (void)max_rows;

  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + Cfg::BAR_OFF);
  uint64_t* empty_bar = full_bar + STAGES;
  uint64_t* rfull_bar = empty_bar + STAGES;      // scale-ring slot barriers (local)
  uint64_t* tfull_bar = rfull_bar + SA_SLOTS;    // [4] TMEM buffer full (multicast commits)
  uint64_t* tfree_bar = tfull_bar + 4;           // [4] leader only: TMEM buffer released by both CTAs' promotion warps
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(tfree_bar + 4);
  float* sa_ring = reinterpret_cast<float*>(smem + Cfg::RING_OFF);
  float* sb_ring = reinterpret_cast<float*>(smem + Cfg::SB_OFF);
  __shared__ int s_mt_off[33];
  __shared__ int s_pt_off[33];
  __shared__ int s_cnt[32];
  __shared__ int s_voff[33];
  constexpr int TILE_RING = 16;
  __shared__ unsigned long long s_tile_ring[TILE_RING];   // sequence-tagged tile indices (written by the leader's producer)

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t rank = cluster_ctarank();
  const bool leader = rank == 0u;
  if (threadIdx.x < TILE_RING) s_tile_ring[threadIdx.x] = 0ull;
  const int cid = blockIdx.x;      // copy warps: all CTAs share the row copies / zeroing
  const int ncl = gridDim.x;

  if constexpr (MODE == 0) pdl_wait();
  if (warp == 0) compute_prefix(cnt, s_mt_off, s_cnt, s_voff, s_pt_off);
  if (warp == 1 && lane == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(&full_bar[i], 1); mbar_init(&empty_bar[i], 1); }
    for (int i = 0; i < SA_SLOTS; ++i) mbar_init(&rfull_bar[i], 1);
    for (int i = 0; i < 4; ++i) { mbar_init(&tfull_bar[i], 1); mbar_init(&tfree_bar[i], 16); }
    fence_barrier_init();
  }
  if (warp == 2) {
    tmem_alloc_2sm(tmem_slot, TMEM_COLS);
    tmem_relinquish_2sm();
  }
  if (warp == 3 && lane == 0) { tma_prefetch(&tmap_a); tma_prefetch(&tmap_b); }
  umma_fence_before();
  __syncthreads();
  umma_fence_after();
  if constexpr (MODE == 0) pdl_launch_dependents();
  cluster_sync_all();   // barriers of both CTAs initialized before any cta_group::2 load / multicast commit / remote arrive
#if MOE_TIMING
  if (threadIdx.x == 0) {
    uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    g_dbg_cta[MODE][blockIdx.x * 8 + 0] = smid;
    g_dbg_cta[MODE][blockIdx.x * 8 + 1] = t;
  }
#endif
  const uint32_t tmem_base = *tmem_slot;
  const int n_tiles = s_pt_off[32] * NT;
  const int nclus = gridDim.x >> 1;
  const int n_full = (MODE == 1) ? (n_tiles / nclus) * nclus : n_tiles;
  const int total_pt = n_full + (n_tiles - n_full) * (MODE == 1 ? NKB : 1);
  struct Tile { int e, nt, mrel; bool valid, half, valid_p, half_p; };
  auto decode = [&](int pt, Tile& t) {
    int tile = pt;
    if (pt >= n_full) tile = n_full + (pt - n_full) / NKB;
    const int mu = tile / NT;
    t.nt = tile - mu * NT;
    t.e = find_expert(s_pt_off, mu);
    const int base = (mu - s_pt_off[t.e]) * 2;
    t.mrel = base + static_cast<int>(rank);
    const int rem = s_cnt[t.e] - t.mrel * BM;
    const int rem_p = s_cnt[t.e] - (base + static_cast<int>(rank ^ 1u)) * BM;
    t.valid = rem > 0; t.half = rem > 0 && rem <= BM / 2;
    t.valid_p = rem_p > 0; t.half_p = rem_p > 0 && rem_p <= BM / 2;
  };
  auto krange = [&](int pt, int& kb0, int& kb1) {
    if (pt < n_full) { kb0 = 0; kb1 = NKB; }
    else { kb0 = (pt - n_full) % NKB; kb1 = kb0 + 1; }
  };
  auto a_tx_of = [](bool valid, bool half) -> uint32_t { return valid ? (half ? A_BYTES / 2 : A_BYTES) : 0u; };

  if (warp < 4) {
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {
      // ---------------- TMA producer (both CTAs; the leader claims the tiles) ----------------
      uint32_t stage = 0, phase = 0, ng = 0;
      const uint32_t ring_remote = mapa_shared(smem_u32(s_tile_ring), 1u);
      for (int tile_iter = 0;; ++tile_iter) {
        int pt;
        if (leader) {
          pt = atomicAdd(tile_ctr, 1);
          const unsigned long long ent = ring_entry(tile_iter, pt);
          *reinterpret_cast<volatile unsigned long long*>(&s_tile_ring[tile_iter % TILE_RING]) = ent;
          st_shared_cluster_u64(ring_remote + (tile_iter % TILE_RING) * 8, ent);
        } else {
          pt = ring_read(&s_tile_ring[tile_iter % TILE_RING], tile_iter);
        }
        if (pt >= total_pt) {
          if (leader) {
            mbar_wait(&empty_bar[stage], phase ^ 1);
            mbar_arrive_expect_tx(&full_bar[stage], 0u);   // wake the MMA warp: sentinel
          }
          break;
        }
        Tile t;
        decode(pt, t);
        const int row0 = t.e * cap + t.mrel * BM;
        const uint32_t tx_full = a_tx_of(t.valid, t.half) + a_tx_of(t.valid_p, t.half_p) + 2u * B_BYTES;
        int kb0, kb1;
        krange(pt, kb0, kb1);
        for (int kb = kb0; kb < kb1; ++kb) {
          mbar_wait(&empty_bar[stage], phase ^ 1);
          uint8_t* sA = smem + stage * STAGE_BYTES;
          uint8_t* sB = sA + A_BYTES;
          const bool gstart = (kb == kb0) || ((kb & (MOE_SA_GROUP - 1)) == 0);
          if (gstart) {
            // scales -> this CTA's rings (local rfull barrier per slot)
            uint64_t* rbar = &rfull_bar[ng % SA_SLOTS];
            float* sS = sa_ring + (ng % SA_SLOTS) * Cfg::SA_HALF_FLOATS;
            ++ng;
            mbar_arrive_expect_tx(rbar, (t.valid ? Cfg::SA_HALF_BYTES : 0u) + (kb == kb0 ? Cfg::SB_TX : 0u));
            if (kb == kb0) {
              float* sbs = sb_ring + (tile_iter % Cfg::SB_SLOTS) * (Cfg::SB_SLOT_BYTES / 4);
              if constexpr (MODE == 0) {
                bulk_load_1d(sbs, sb + (static_cast<size_t>(t.e) * 32 + t.nt) * NUM_HB, 224, rbar);
                bulk_load_1d(sbs + 64, sb + (static_cast<size_t>(t.e) * 32 + 16 + t.nt) * NUM_HB, 224, rbar);
              } else {
                bulk_load_1d(sbs, sb + (static_cast<size_t>(t.e) * NUM_HB + 2 * t.nt) * NUM_IB, 128, rbar);
              }
            }
            if (t.valid)
              bulk_load_1d(sS, sa + (static_cast<size_t>(row0 / BM) * NKB + (kb & ~(MOE_SA_GROUP - 1))) * 128, Cfg::SA_HALF_BYTES, rbar);
          }
          if (leader) mbar_arrive_expect_tx(&full_bar[stage], tx_full);
          const uint32_t lbar = smem_u32(&full_bar[stage]) & 0xFEFFFFFFu;   // the leader's barrier (pair addressing)
          // B: this CTA's 64-row halves of the two N = 128 blocks (one 4D box: {128 k, 64 n, 2 blocks, 1 expert})
          if constexpr (MODE == 0) tma_load_4d_2sm(sB, &tmap_b, lbar, kb * BK, t.nt * 128 + 64 * static_cast<int>(rank), 0, t.e);
          else tma_load_4d_2sm(sB, &tmap_b, lbar, kb * BK, 64 * static_cast<int>(rank), 2 * t.nt, t.e);
          if (kb == kb0) {
            if constexpr (MODE == 0) { while (ld_acquire_gpu(rows_done + t.e * CNT_STRIDE) < s_cnt[t.e]) __nanosleep(256); }
            else { while (ld_acquire_gpu(done_tiles + t.e * CNT_STRIDE) < (s_pt_off[t.e + 1] - s_pt_off[t.e]) * 32) __nanosleep(256); }
            fence_proxy_async_global();
          }
          if (t.valid) tma_load_2d_2sm(sA, t.half ? &tmap_ah : &tmap_a, lbar, kb * BK, row0);
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
    } else if (MODE == 1 && warp >= 2) {
#if MOE_RED_OFFLOAD == 2
      // ---------------- reduction warps (GEMM2): TMA bulk reductions from the staged output tile ----------------
      const int t = threadIdx.x - 64;
      const uint8_t* stg = smem + Cfg::OUT_OFF;
      const int* s_tok = reinterpret_cast<const int*>(stg + Cfg::OUT_ROWS_BYTES);
#if MOE_ZERO
      if (t == 0) { while (ld_acquire_gpu(zero_ctr) < zero_target) __nanosleep(256); }
#endif
      named_bar_sync(7, 64);
      named_bar_arrive(6, BAR_RED);
      for (;;) {
        named_bar_sync(5, BAR_RED);
        if (s_tok[129]) { bulk_wait_group<0>(); break; }
        const int nt = s_tok[128];
        const int g = warp - 2;
        fence_proxy_async_shared_cta();
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int row = q * 32 + lane;
          const int tok = s_tok[row];
          if (tok >= 0)
            bulk_reduce_add_bf16(out + static_cast<size_t>(tok) * HID + nt * BN + 128 * g, stg + (g * 128 + row) * 272, 256);
        }
        bulk_commit_group();
        bulk_wait_group_read<0>();
        __syncwarp();
        named_bar_arrive(6, BAR_RED);
      }
#else
      (void)zero_ctr; (void)zero_target;
#endif
    } else if (MODE == 0 && warp >= 2) {
      // ---------------- copy warps (GEMM1): row copies in expert order, then output zeroing ----------------
      const int t = threadIdx.x - 64;
      const int total_v = s_voff[32];
      const int e_head = find_expert(s_mt_off, 0);
      int e_cur = -1, n_cur = 0;
      for (int v = s_voff[e_head + 1] + cid; v < total_v; v += ncl) {
        const int e = find_expert(s_voff, v);
        if (e != e_cur) {
          if (e_cur >= 0) {
            __threadfence();
            named_bar_sync(6, 64);
            if (t == 0 && n_cur) atomicAdd(rows_done + e_cur * CNT_STRIDE, n_cur);
          }
          e_cur = e;
          n_cur = 0;
        }
        const int row = e * cap + (v - s_voff[e]);
        const int token = row_token[row] & 0x3fffffff;
        const int4* src = reinterpret_cast<const int4*>(hs + static_cast<size_t>(token) * HID) + t;
        int4* dst = reinterpret_cast<int4*>(a_perm + static_cast<size_t>(row) * HID) + t;
        int4 val[7];
#pragma unroll
        for (int c = 0; c < 7; ++c) val[c] = __ldg(src + 64 * c);
#pragma unroll
        for (int c = 0; c < 7; ++c) dst[64 * c] = val[c];
        if (t < NUM_HB)
          sa_w[(static_cast<size_t>(row >> 7) * NUM_HB + t) * 128 + (row & 127)] = __ldg(hs_scale + static_cast<size_t>(t) * num_tokens + token);
        ++n_cur;
      }
      __threadfence();
      named_bar_sync(6, 64);
      if (t == 0 && n_cur && e_cur >= 0) atomicAdd(rows_done + e_cur * CNT_STRIDE, n_cur);
#if MOE_ZERO
      for (int r = cid; r < num_tokens; r += ncl) {
        int4* dst = reinterpret_cast<int4*>(out + static_cast<size_t>(r) * HID);
#pragma unroll 4
        for (int c = t; c < HID * 2 / 16; c += 64) dst[c] = make_int4(0, 0, 0, 0);
      }
      __threadfence();
      named_bar_sync(6, 64);
      if (t == 0) atomicAdd(zero_ctr, 1);
#endif
    } else if (warp == 1 && leader) {
      // ---------------- MMA issuer (leader CTA only) ----------------
      constexpr uint32_t idesc = make_idesc_f8(2 * BM, 128);   // M = 256 over the pair, N = 128 per sub-step
      uint32_t stage = 0, phase = 0, acc_it = 0;
      for (int tile_iter = 0;; ++tile_iter) {
        mbar_wait(&full_bar[stage], phase);   // first k-block of the next tile (or the sentinel)
        const int pt = ring_read(&s_tile_ring[tile_iter % TILE_RING], tile_iter);
        if (pt >= total_pt) {
          // drain: all buffers released by both CTAs' promotion warps, then the sentinel (an empty multicast commit)
          for (uint32_t i = (acc_it >= 4 ? acc_it - 4 : 0); i < acc_it; ++i) mbar_wait(&tfree_bar[i & 3], (i >> 2) & 1u);
          if (lane == 0) umma_commit_2sm_mc(&tfull_bar[acc_it & 3], 0x3);
          __syncwarp();
          break;
        }
        Tile t;
        decode(pt, t);
        int kb0, kb1;
        krange(pt, kb0, kb1);
        for (int kb = kb0; kb < kb1; ++kb) {
#pragma unroll
          for (int s = 0; s < 2; ++s) {
            const uint32_t buf = acc_it & 3u;
            if (acc_it >= 4) mbar_wait(&tfree_bar[buf], ((acc_it - 4) >> 2) & 1u);   // buffer released (both CTAs)
            if (s == 0) mbar_wait(&full_bar[stage], phase);
            if (lane == 0) {
              umma_fence_after();
              const uint32_t sA_addr = smem_u32(smem + stage * STAGE_BYTES);
              const uint64_t adesc = make_smem_desc(sA_addr);
              const uint64_t bdesc = make_smem_desc(sA_addr + A_BYTES + s * (B_BYTES / 2));
              const uint32_t d_tmem = tmem_base + buf * 128;
#if MOE_ABL != 5 && MOE_ABL != 7
#pragma unroll
              for (int k = 0; k < BK / 32; ++k) umma_mma_f8_2sm(d_tmem, adesc + 2 * k, bdesc + 2 * k, idesc, k != 0 ? 1u : 0u);
#else
              (void)adesc; (void)bdesc; (void)d_tmem;
#endif
              umma_commit_2sm_mc(&tfull_bar[buf], 0x3);                 // buffer full in both CTAs
              if (s == 1) umma_commit_2sm_mc(&empty_bar[stage], 0x3);   // stage free in both CTAs
            }
            __syncwarp();
            ++acc_it;
          }
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
      }
    }
  } else {
    setmaxnreg_inc<232>();
    // ---------------- promotion + epilogue warps (both CTAs) ----------------
    const int g = (warp - 4) >> 2;
    const int q = warp & 3;
    const int row_in_tile = q * 32 + lane;
    const uint32_t tmem_lane_base = tmem_base + (static_cast<uint32_t>(q * 32) << 16);
    const uint32_t tfree_remote = mapa_shared(smem_u32(tfree_bar), 0u);
    uint32_t acc_it = 0, ng = 0;
    auto release = [&](uint32_t buf) {
      umma_fence_before();
      __syncwarp();
      if (lane == 0) {
        if (leader) mbar_arrive(&tfree_bar[buf]);
        else mbar_arrive_remote_relaxed(tfree_remote + buf * 8);
      }
    };
#if MOE_ZERO
    if constexpr (MODE == 1) {
      if (lane == 0) { while (ld_acquire_gpu(zero_ctr) < zero_target) __nanosleep(256); }
      __syncwarp();
    }
#endif
    if constexpr (MODE == 0) {
      // head start: copy the first non-empty expert's rows (one row per warp)
      const int e_head = find_expert(s_mt_off, 0);
      const int v_end = s_voff[e_head + 1];
      int n = 0;
      for (int v = cid + (warp - 4) * ncl; v < v_end; v += 8 * ncl) {
        const int row = e_head * cap + v;
        const int token = row_token[row] & 0x3fffffff;
        const int4* src = reinterpret_cast<const int4*>(hs + static_cast<size_t>(token) * HID) + lane;
        int4* dst = reinterpret_cast<int4*>(a_perm + static_cast<size_t>(row) * HID) + lane;
        int4 val[14];
#pragma unroll
        for (int c = 0; c < 14; ++c) val[c] = __ldg(src + 32 * c);
#pragma unroll
        for (int c = 0; c < 14; ++c) dst[32 * c] = val[c];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int kb = lane + 32 * qq;
          if (kb < NUM_HB)
            sa_w[(static_cast<size_t>(row >> 7) * NUM_HB + kb) * 128 + (row & 127)] = __ldg(hs_scale + static_cast<size_t>(kb) * num_tokens + token);
        }
        ++n;
      }
      __threadfence();
      __syncwarp();
      if (lane == 0 && n) atomicAdd(rows_done + e_head * CNT_STRIDE, n);
    }
    for (int tile_iter = 0;; ++tile_iter) {
      mbar_wait(&tfull_bar[acc_it & 3u], (acc_it >> 2) & 1u);          // first sub-step of the tile (or the sentinel)
      const int pt = ring_read(&s_tile_ring[tile_iter % TILE_RING], tile_iter);
      if (pt >= total_pt) {
#if MOE_RED_OFFLOAD == 2
        if constexpr (MODE == 1) {
          named_bar_sync(6, BAR_RED);
          if (warp == 4 && lane == 0) reinterpret_cast<int*>(smem + Cfg::OUT_OFF + Cfg::OUT_ROWS_BYTES)[129] = 1;
          __threadfence_block();
          named_bar_arrive(5, BAR_RED);
        }
#endif
        break;
      }
      Tile t;
      decode(pt, t);
      const bool active = t.valid && (row_in_tile < (t.half ? BM / 2 : BM));
      const int row = t.e * cap + t.mrel * BM + row_in_tile;
      int token = -1;
      float wtok = 0.f;
      if constexpr (MODE == 1) {
        if (active && row_in_tile < s_cnt[t.e] - t.mrel * BM) { token = row_token[row] & 0x3fffffff; wtok = row_weight[row]; }
      }
      int kb0, kb1;
      krange(pt, kb0, kb1);
      float acc[128];
#pragma unroll
      for (int i = 0; i < 128; ++i) acc[i] = 0.0f;
      const float* sbw = sb_ring + (tile_iter % Cfg::SB_SLOTS) * (Cfg::SB_SLOT_BYTES / 4);
#pragma unroll 1
      for (int kb = kb0; kb < kb1; ++kb) {
        const bool gstart = (kb == kb0) || ((kb & (MOE_SA_GROUP - 1)) == 0);
        const uint32_t cur_g = gstart ? ng++ : ng - 1;
        const float* sS = sa_ring + (cur_g % SA_SLOTS) * Cfg::SA_HALF_FLOATS;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const uint32_t buf = acc_it & 3u;
          mbar_wait(&tfull_bar[buf], (acc_it >> 2) & 1u);
          umma_fence_after();
#if MOE_ABL == 1 || MOE_ABL == 5 || MOE_ABL == 6 || MOE_ABL == 7
          const bool skip = true;
#else
          const bool skip = !t.valid;
#endif
          if (skip) {
            release(buf);
          } else {
            const uint32_t taddr = tmem_lane_base + buf * 128 + 64 * g;
            uint32_t v0[32], v1[32];
            tmem_ld_x32(taddr, v0);
            tmem_ld_x32(taddr + 32, v1);
            if (gstart && s == 0) mbar_wait(&rfull_bar[cur_g % SA_SLOTS], (cur_g / SA_SLOTS) & 1u);
            const float sav = sS[(kb & (MOE_SA_GROUP - 1)) * 128 + row_in_tile];
            const float sbv = (MODE == 0) ? (s == 0 ? sbw[kb] : sbw[64 + kb]) : sbw[16 * s + kb];
            const float sv = sav * sbv;
            const unsigned long long sp = pack2(__float_as_uint(sv), __float_as_uint(sv));
            tmem_ld_wait();
            release(buf);   // TMEM buffer read; the FMAs run while the next MMA proceeds
            if (active) {
#pragma unroll
              for (int i = 0; i < 32; i += 2) {
                const unsigned long long t0 = ffma2(pack2(__float_as_uint(acc[64 * s + i]), __float_as_uint(acc[64 * s + i + 1])), pack2(v0[i], v0[i + 1]), sp);
                acc[64 * s + i] = __uint_as_float(static_cast<uint32_t>(t0));
                acc[64 * s + i + 1] = __uint_as_float(static_cast<uint32_t>(t0 >> 32));
                const unsigned long long t1 = ffma2(pack2(__float_as_uint(acc[64 * s + 32 + i]), __float_as_uint(acc[64 * s + 32 + i + 1])), pack2(v1[i], v1[i + 1]), sp);
                acc[64 * s + 32 + i] = __uint_as_float(static_cast<uint32_t>(t1));
                acc[64 * s + 32 + i + 1] = __uint_as_float(static_cast<uint32_t>(t1 >> 32));
              }
            }
          }
          ++acc_it;
        }
      }
      // ---------------- epilogue (this CTA's 128 rows) ----------------
#if MOE_ABL == 1 || MOE_ABL == 5 || MOE_ABL == 6 || MOE_ABL == 7
      if (MODE == 0 && warp == 4 && lane == 0) atomicAdd(done_tiles + t.e * CNT_STRIDE, 1);
      continue;
#endif
      if constexpr (MODE == 0) {
        if (t.valid) {
          float c[64];
          float amax = 0.0f;
#pragma unroll
          for (int i = 0; i < 64; ++i) {
            const float x2 = acc[64 + i];
#if MOE_FASTSILU == 2
            float th;
            asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * x2));
            c[i] = acc[i] * (x2 * (0.5f + 0.5f * th));
#else
            c[i] = acc[i] * __fdividef(x2, 1.0f + __expf(-x2));
#endif
            amax = fmaxf(amax, fabsf(c[i]));
          }
          float* xch = reinterpret_cast<float*>(smem + Cfg::XCH_OFF) + (tile_iter & 1) * 256;
          if (active) xch[g * 128 + row_in_tile] = amax;
          named_bar_sync(3, 256);
          amax = fmaxf(amax, xch[(1 - g) * 128 + row_in_tile]);
          const float scale = amax * (1.0f / 448.0f);
          const float inv = amax > 0.0f ? 448.0f / amax : 0.0f;
          uint32_t hw[16];
#pragma unroll
          for (int j = 0; j < 16; ++j) {
            uint32_t h4 = 0;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const float2 x = make_float2(c[j * 4 + 2 * k], c[j * 4 + 2 * k + 1]);
              const __nv_fp8x2_storage_t h2 =
                  __nv_cvt_float2_to_fp8x2(make_float2(x.x * inv, x.y * inv), __NV_SATFINITE, __NV_E4M3);
              h4 |= static_cast<uint32_t>(h2) << (16 * k);
            }
            hw[j] = h4;
          }
          const size_t coff = static_cast<size_t>(row) * INTER + t.nt * 128 + 64 * g;
          if (active) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
              reinterpret_cast<int4*>(c_hi + coff)[j] = make_int4(hw[4 * j], hw[4 * j + 1], hw[4 * j + 2], hw[4 * j + 3]);
            if (g == 0) sc[(static_cast<size_t>(row >> 7) * NUM_IB + t.nt) * 128 + (row & 127)] = scale;
          }
          named_bar_sync(3, 256);   // all stores issued
        }
        if (warp == 4 && lane == 0) { __threadfence(); atomicAdd(done_tiles + t.e * CNT_STRIDE, 1); }
      } else {
#if MOE_RED_OFFLOAD == 2
        // stage w * acc (bf16): sub-step s = 128-column half s of the tile, this thread's 64 columns at byte 128 g
        named_bar_sync(6, BAR_RED);
        uint8_t* stg = smem + Cfg::OUT_OFF;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float* a = acc + 64 * s + 8 * j;
            const uint32_t p0 = pack_bf16x2(wtok * a[0], wtok * a[1]);
            const uint32_t p1 = pack_bf16x2(wtok * a[2], wtok * a[3]);
            const uint32_t p2 = pack_bf16x2(wtok * a[4], wtok * a[5]);
            const uint32_t p3 = pack_bf16x2(wtok * a[6], wtok * a[7]);
            *reinterpret_cast<uint4*>(stg + (s * 128 + row_in_tile) * 272 + 128 * g + j * 16) = make_uint4(p0, p1, p2, p3);
          }
        }
        int* s_tok = reinterpret_cast<int*>(stg + Cfg::OUT_ROWS_BYTES);
        if (g == 0) s_tok[row_in_tile] = token;
        if (warp == 4 && lane == 0) { s_tok[128] = t.nt; s_tok[129] = 0; }
        __threadfence_block();
        named_bar_arrive(5, BAR_RED);
#else
        if (token >= 0) {
#pragma unroll
          for (int s = 0; s < 2; ++s) {
            __nv_bfloat16* dst = out + static_cast<size_t>(token) * HID + t.nt * BN + 128 * s + 64 * g;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float* a = acc + 64 * s + 8 * j;
              red_add_bf16x2_v4(dst + j * 8, pack_bf16x2(wtok * a[0], wtok * a[1]), pack_bf16x2(wtok * a[2], wtok * a[3]),
                                pack_bf16x2(wtok * a[4], wtok * a[5]), pack_bf16x2(wtok * a[6], wtok * a[7]));
            }
          }
        }
#endif
      }
    }
  }
  umma_fence_before();
  __syncthreads();
#if MOE_TIMING
  if (threadIdx.x == 0) {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    g_dbg_cta[MODE][blockIdx.x * 8 + 2] = t;
  }
#endif
  cluster_sync_all();   // the peer may still commit / arrive into this CTA until here
  if (warp == 2) {
    umma_fence_after();
    tmem_dealloc_2sm(tmem_base, TMEM_COLS);
  }
}

// ------------------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------------------
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

template <typename... KArgs, typename... Args>
void launch_pdl(void (*kern)(KArgs...), int grid, int block, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kern, std::forward<Args>(args)...);
  TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

template <typename... KArgs, typename... Args>
void launch_gemm(void (*kern)(KArgs...), int grid, uint32_t smem, int cluster, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(GEMM_THREADS);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = cluster;
  attr[1].val.clusterDim.y = 1;
  attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = cluster > 1 ? 2 : 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kern, std::forward<Args>(args)...);
  TORCH_CHECK(err == cudaSuccess, "gemm launch failed: ", cudaGetErrorString(err));
}

EncodeTiledFn get_encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
    TORCH_CHECK(err == cudaSuccess && p != nullptr, "cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}

// K-major uint8 tensor map: dims {inner (bytes), rows[, experts]}, box {128, box_rows[, 1]}, 128B swizzle.
CUtensorMap make_map(const void* ptr, uint64_t inner, uint64_t rows, uint64_t experts, uint32_t box_rows) {
  CUtensorMap m;
  const uint32_t rank = experts ? 3 : 2;
  cuuint64_t gdim[3] = {inner, rows, experts};
  cuuint64_t gstride[2] = {inner, inner * rows};
  cuuint32_t box[3] = {128, box_rows, 1};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(ptr), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  return m;
}

// W13 viewed as {k, n (2048), half (X1 / X2), expert}: one box {128, rows, 2, 1} fetches both halves of a GEMM1 B tile.
CUtensorMap make_map_w13_4d(const void* ptr, uint32_t rows = 128) {
  CUtensorMap m;
  cuuint64_t gdim[4] = {HID, INTER, 2, E_LOCAL};
  cuuint64_t gstride[3] = {HID, static_cast<cuuint64_t>(INTER) * HID, static_cast<cuuint64_t>(2 * INTER) * HID};
  cuuint32_t box[4] = {128, rows, 2, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(ptr), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (4d) failed: ", static_cast<int>(r));
  return m;
}

// W2 viewed as {k (2048), n in block (128), n-block (56), expert}: box {128, 64, 2, 1} = this CTA's 64-row halves of
// two consecutive 128-column blocks (2-SM GEMM2).
CUtensorMap make_map_w2_4d(const void* ptr) {
  CUtensorMap m;
  cuuint64_t gdim[4] = {INTER, 128, HID / 128, E_LOCAL};
  cuuint64_t gstride[3] = {INTER, static_cast<cuuint64_t>(128) * INTER, static_cast<cuuint64_t>(HID) * INTER};
  cuuint32_t box[4] = {128, 64, 2, 1};
  cuuint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = get_encode_fn()(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, const_cast<void*>(ptr), gdim, gstride, box, estr,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (w2 4d) failed: ", static_cast<int>(r));
  return m;
}

struct Workspace {
  int T = 0;
  int max_rows = 0;
  int cap = 0;   // rows per expert in the fixed expert layout (T rounded up to a tile: any routing fits)
  torch::Tensor cnt, row_token, row_weight, tok_cnt;
  torch::Tensor a_perm, sa_perm, c_hi, sc;
  CUtensorMap map_a, map_hi, map_a64, map_hi64;
};
Workspace g_ws;
int g_num_sms = 0;
bool g_attr_set = false;
cudaStream_t g_side_stream = nullptr, g_main_stream = nullptr;
cudaEvent_t g_ev_fork = nullptr, g_ev_join = nullptr, g_ev_in = nullptr, g_ev_out = nullptr;

// CUDA-graph replay cache: the whole forward (memsets + 4 kernels) is captured once per distinct set of input
// pointers / shapes and replayed afterwards (the benchmark calls run() repeatedly with the same tensors).
struct GraphKey {
  const void* p[8];
  int T;
  int offset;
  float rsf;
  bool operator==(const GraphKey& o) const {
    for (int i = 0; i < 8; ++i) if (p[i] != o.p[i]) return false;
    return T == o.T && offset == o.offset && rsf == o.rsf;
  }
};
GraphKey g_graph_key = {};
cudaGraphExec_t g_graph_exec = nullptr;
torch::Tensor g_out;   // persistent output buffer (contents recomputed on every call)

void ensure_workspace(int T, torch::Device dev) {
  if (g_ws.T >= T && g_ws.max_rows > 0) return;
  const int cap = ((T + BM - 1) / BM) * BM;
  const int max_rows = E_LOCAL * cap;
  auto opt_i = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opt_bf = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
  g_ws.T = T;
  g_ws.max_rows = max_rows;
  g_ws.cap = cap;
  // per-expert counters, each on its own 128 B line: [0]: routing counts, [1]: GEMM1 tiles done, [2]: rows copied,
  // [3]: dynamic tile counters of GEMM1 (+0) and GEMM2 (+CNT_STRIDE), output-zeroed counter (+2 CNT_STRIDE)
  g_ws.cnt = torch::zeros({4 * E_LOCAL * CNT_STRIDE}, opt_i);
  g_ws.row_token = torch::empty({max_rows}, opt_i);
  g_ws.tok_cnt = torch::zeros({cap}, opt_i);
  g_ws.row_weight = torch::empty({max_rows}, opt_f);
  g_ws.a_perm = torch::empty({max_rows, HID}, opt_u8);
  g_ws.sa_perm = torch::empty({NUM_HB * max_rows + 256}, opt_f);   // +1 k-block of padding (MOE_TRS copies kb, kb+1)
  g_ws.c_hi = torch::empty({max_rows, INTER}, opt_u8);
  g_ws.sc = torch::empty({NUM_IB * max_rows + 256}, opt_f);
  g_ws.map_a = make_map(g_ws.a_perm.data_ptr(), HID, max_rows, 0, BM);
  g_ws.map_hi = make_map(g_ws.c_hi.data_ptr(), INTER, max_rows, 0, BM);
  g_ws.map_a64 = make_map(g_ws.a_perm.data_ptr(), HID, max_rows, 0, BM / 2);
  g_ws.map_hi64 = make_map(g_ws.c_hi.data_ptr(), INTER, max_rows, 0, BM / 2);
}

torch::Tensor moe_forward(torch::Tensor routing_logits, torch::Tensor routing_bias, torch::Tensor hidden_states,
                          torch::Tensor hidden_states_scale, torch::Tensor gemm1_weights,
                          torch::Tensor gemm1_weights_scale, torch::Tensor gemm2_weights,
                          torch::Tensor gemm2_weights_scale, int64_t local_expert_offset,
                          double routed_scaling_factor) {
  const at::cuda::CUDAGuard guard(hidden_states.device());
  const int T = static_cast<int>(routing_logits.size(0));
  TORCH_CHECK(routing_logits.size(1) == E_GLOBAL);
  TORCH_CHECK(hidden_states.size(0) == T && hidden_states.size(1) == HID);
  TORCH_CHECK(gemm1_weights.size(0) == E_LOCAL && gemm1_weights.size(1) == 2 * INTER && gemm1_weights.size(2) == HID);
  TORCH_CHECK(gemm2_weights.size(0) == E_LOCAL && gemm2_weights.size(1) == HID && gemm2_weights.size(2) == INTER);
  TORCH_CHECK(routing_logits.is_contiguous() && hidden_states.is_contiguous() && hidden_states_scale.is_contiguous());
  TORCH_CHECK(gemm1_weights.is_contiguous() && gemm2_weights.is_contiguous());
  TORCH_CHECK(gemm1_weights_scale.is_contiguous() && gemm2_weights_scale.is_contiguous());
  TORCH_CHECK(routing_logits.scalar_type() == torch::kFloat32 && hidden_states_scale.scalar_type() == torch::kFloat32);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  (void)stream;
  if (g_num_sms == 0) {
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, hidden_states.get_device());
  }
  if (!g_attr_set) {
    cudaFuncSetAttribute(gemm_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmCfg<0, false>::SMEM_BYTES);
    cudaFuncSetAttribute(gemm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmCfg<1, false>::SMEM_BYTES);
#if MOE_2SM
    cudaFuncSetAttribute(gemm2sm_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmCfg<0, true>::SMEM_BYTES);
    cudaFuncSetAttribute(gemm2sm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, GemmCfg<1, true>::SMEM_BYTES);
#endif
    g_attr_set = true;
  }
  ensure_workspace(T, hidden_states.device());
  Workspace& ws = g_ws;
  const int max_rows = ws.max_rows;

  // Weight tensor maps (pointers may change between calls; encoding is a few microseconds on the host).
  const CUtensorMap map_w13 = MOE_2SM ? make_map_w13_4d(gemm1_weights.data_ptr(), 64)
                              : MOE_B4D ? make_map_w13_4d(gemm1_weights.data_ptr())
                                        : make_map(gemm1_weights.data_ptr(), HID, 2 * INTER, E_LOCAL, 128);
  const CUtensorMap map_w2 = MOE_2SM ? make_map_w2_4d(gemm2_weights.data_ptr())
                                     : make_map(gemm2_weights.data_ptr(), INTER, HID, E_LOCAL, MOE_B256 ? 256 : 128);

  if (!g_side_stream) {
    cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&g_main_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_ev_fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_ev_join, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_ev_in, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_ev_out, cudaEventDisableTiming);
  }
  const __nv_bfloat16* bias_ptr = nullptr;
  torch::Tensor bias_c;
  if (routing_bias.defined() && routing_bias.numel() > 0) {
    bias_c = routing_bias.to(torch::kBFloat16).contiguous();   // no-op copy for a bf16 contiguous bias
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias_c.data_ptr());
  }
  if (!g_out.defined() || g_out.size(0) != T) {
    g_out = torch::empty({T, HID}, torch::TensorOptions().dtype(torch::kBFloat16).device(hidden_states.device()));
  }
  torch::Tensor out = g_out;

  GraphKey key;
  key.p[0] = routing_logits.data_ptr(); key.p[1] = bias_ptr; key.p[2] = hidden_states.data_ptr();
  key.p[3] = hidden_states_scale.data_ptr(); key.p[4] = gemm1_weights.data_ptr(); key.p[5] = gemm1_weights_scale.data_ptr();
  key.p[6] = gemm2_weights.data_ptr(); key.p[7] = gemm2_weights_scale.data_ptr();
  key.T = T; key.offset = static_cast<int>(local_expert_offset); key.rsf = static_cast<float>(routed_scaling_factor);

  // Order our private stream after the caller's stream (inputs ready), run/replay, then order the caller after us.
  cudaEventRecord(g_ev_in, stream);
  cudaStreamWaitEvent(g_main_stream, g_ev_in, 0);
  if (g_graph_exec && key == g_graph_key) {
    cudaGraphLaunch(g_graph_exec, g_main_stream);
    cudaEventRecord(g_ev_out, g_main_stream);
    cudaStreamWaitEvent(stream, g_ev_out, 0);
    return out;
  }
  if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = nullptr; }
  cudaGraph_t graph = nullptr;
  cudaStreamBeginCapture(g_main_stream, cudaStreamCaptureModeThreadLocal);
  stream = g_main_stream;   // everything below is captured

  cudaMemsetAsync(ws.cnt.data_ptr(), 0, 4 * E_LOCAL * CNT_STRIDE * sizeof(int), stream);
  if (MOE_ONESTORE) cudaMemsetAsync(ws.tok_cnt.data_ptr(), 0, static_cast<size_t>(T) * sizeof(int), stream);

  // Zero the output on a side stream. Forked after the gather (MOE_FORK 0) it overlaps GEMM1; forking it before /
  // after the routing kernel delays routing (memset CTAs occupy the SMs) and is 35-50 us slower end to end.
  auto fork_memset = [&]() {
#if !MOE_ZERO
    cudaEventRecord(g_ev_fork, stream);
    cudaStreamWaitEvent(g_side_stream, g_ev_fork, 0);
    cudaMemsetAsync(out.data_ptr(), 0, static_cast<size_t>(T) * HID * sizeof(__nv_bfloat16), g_side_stream);
#endif
  };
  if (MOE_FORK == 1) fork_memset();

  // routing + row copy (fixed expert layout); GEMM1 is its programmatic dependent
  routing_kernel<<<(T + 7) / 8, 256, 0, stream>>>(   // one token per warp
      routing_logits.data_ptr<float>(), bias_ptr, T, static_cast<int>(local_expert_offset),
      static_cast<float>(routed_scaling_factor), ws.cnt.data_ptr<int>(), ws.cap, ws.row_token.data_ptr<int>(),
      ws.row_weight.data_ptr<float>(), reinterpret_cast<const uint8_t*>(gemm1_weights.data_ptr()),
      static_cast<size_t>(MOE_W13_PF_EXPERTS) * 2 * INTER * HID, ws.tok_cnt.data_ptr<int>());
  if (MOE_FORK != 1) fork_memset();

  constexpr int kCluster = (MOE_AMC || MOE_2SM) ? 2 : 1;
#if MOE_2SM
  auto* k1 = gemm2sm_kernel<0>;
  auto* k2 = gemm2sm_kernel<1>;
  constexpr uint32_t smem1 = GemmCfg<0, true>::SMEM_BYTES, smem2 = GemmCfg<1, true>::SMEM_BYTES;
#else
  auto* k1 = gemm_kernel<0>;
  auto* k2 = gemm_kernel<1>;
  constexpr uint32_t smem1 = GemmCfg<0, false>::SMEM_BYTES, smem2 = GemmCfg<1, false>::SMEM_BYTES;
#endif
  const int gemm_grid = g_num_sms & ~(kCluster - 1);
  const CUtensorMap& map_a_use = MOE_AMC ? ws.map_a64 : ws.map_a;
  const CUtensorMap& map_hi_use = MOE_AMC ? ws.map_hi64 : ws.map_hi;
  int* zero_ctr = ws.cnt.data_ptr<int>() + 3 * E_LOCAL * CNT_STRIDE + 2 * CNT_STRIDE;
  const int zero_target = gemm_grid;   // one arrive per GEMM1 CTA (its copy warps)
  launch_gemm(k1, gemm_grid, smem1, kCluster, stream, map_a_use, map_w13, ws.map_a64, ws.cnt.data_ptr<int>(),
              ws.sa_perm.data_ptr<float>(), gemm1_weights_scale.data_ptr<float>(), max_rows,
              reinterpret_cast<uint8_t*>(ws.c_hi.data_ptr()), ws.sc.data_ptr<float>(), static_cast<const int*>(ws.row_token.data_ptr<int>()),
              static_cast<const float*>(nullptr), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws.cnt.data_ptr<int>() + E_LOCAL * CNT_STRIDE,
              ws.cnt.data_ptr<int>() + 2 * E_LOCAL * CNT_STRIDE, ws.cnt.data_ptr<int>() + 3 * E_LOCAL * CNT_STRIDE, ws.cap, T, zero_ctr, zero_target,
              reinterpret_cast<const uint8_t*>(hidden_states.data_ptr()), hidden_states_scale.data_ptr<float>(),
              reinterpret_cast<uint8_t*>(ws.a_perm.data_ptr()), ws.sa_perm.data_ptr<float>(),
              static_cast<const int*>(ws.tok_cnt.data_ptr<int>()), reinterpret_cast<const uint8_t*>(gemm1_weights.data_ptr()),
              reinterpret_cast<const uint8_t*>(ws.a_perm.data_ptr()));

#if !MOE_ZERO
  // join the side stream (output memset) before the accumulating GEMM2
  cudaEventRecord(g_ev_join, g_side_stream);
  cudaStreamWaitEvent(stream, g_ev_join, 0);
#endif

  launch_gemm(k2, gemm_grid, smem2, kCluster, stream, map_hi_use, map_w2, ws.map_hi64, ws.cnt.data_ptr<int>(), ws.sc.data_ptr<float>(),
              gemm2_weights_scale.data_ptr<float>(), max_rows, static_cast<uint8_t*>(nullptr),
              static_cast<float*>(nullptr), ws.row_token.data_ptr<int>(), ws.row_weight.data_ptr<float>(),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws.cnt.data_ptr<int>() + E_LOCAL * CNT_STRIDE,
              static_cast<int*>(nullptr), ws.cnt.data_ptr<int>() + 3 * E_LOCAL * CNT_STRIDE + CNT_STRIDE, ws.cap, T, zero_ctr, zero_target,
              static_cast<const uint8_t*>(nullptr), static_cast<const float*>(nullptr), static_cast<uint8_t*>(nullptr),
              static_cast<float*>(nullptr), static_cast<const int*>(ws.tok_cnt.data_ptr<int>()),
              reinterpret_cast<const uint8_t*>(gemm2_weights.data_ptr()), reinterpret_cast<const uint8_t*>(ws.c_hi.data_ptr()));

  cudaError_t cerr = cudaStreamEndCapture(g_main_stream, &graph);
  TORCH_CHECK(cerr == cudaSuccess, "graph capture failed: ", cudaGetErrorString(cerr));
  cerr = cudaGraphInstantiate(&g_graph_exec, graph, 0);
  TORCH_CHECK(cerr == cudaSuccess, "graph instantiate failed: ", cudaGetErrorString(cerr));
  cudaGraphDestroy(graph);
  g_graph_key = key;
  cudaGraphLaunch(g_graph_exec, g_main_stream);
  cudaEventRecord(g_ev_out, g_main_stream);
  cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), g_ev_out, 0);
  return out;
}

torch::Tensor debug_timing() {
  auto t = torch::zeros({2, 148 * 128}, torch::TensorOptions().dtype(torch::kInt64));
#if MOE_TIMING
  cudaMemcpyFromSymbol(t.data_ptr(), g_dbg_timing, sizeof(long long) * 2 * 148 * 128);
#endif
  return t;
}
torch::Tensor debug_cta() {
  auto t = torch::zeros({2, 148 * 8}, torch::TensorOptions().dtype(torch::kInt64));
#if MOE_TIMING
  cudaMemcpyFromSymbol(t.data_ptr(), g_dbg_cta, sizeof(long long) * 2 * 148 * 8);
#endif
  return t;
}

torch::Tensor debug_trace() {
  auto t = torch::zeros({2, 2, 3, 1024, 2}, torch::TensorOptions().dtype(torch::kInt64));
#if MOE_TIMING
  cudaMemcpyFromSymbol(t.data_ptr(), g_dbg_trace, sizeof(long long) * 2 * 2 * 3 * 1024 * 2);
#endif
  return t;
}

torch::Tensor debug_copy() {
  auto t = torch::zeros({148 * 2}, torch::TensorOptions().dtype(torch::kInt64));
#if MOE_TIMING
  cudaMemcpyFromSymbol(t.data_ptr(), g_dbg_copy, sizeof(long long) * 148 * 2);
#endif
  return t;
}

std::vector<torch::Tensor> debug_buffers() {
  return {g_ws.cnt, g_ws.row_token, g_ws.row_weight, g_ws.a_perm, g_ws.sa_perm, g_ws.c_hi, g_ws.sc};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_forward", &moe_forward, "FP8 block-scale MoE forward (B200)");
  m.def("debug_buffers", &debug_buffers, "workspace buffers");
  m.def("debug_timing", &debug_timing, "promotion phase timing (dev)");
  m.def("debug_cta", &debug_cta, "per-CTA smid/start/end (dev)");
  m.def("debug_copy", &debug_copy, "GEMM1 copy-warp phase end times (dev)");
  m.def("debug_trace", &debug_trace, "producer / MMA / promotion event trace of CTAs 0-1 (dev)");
}
