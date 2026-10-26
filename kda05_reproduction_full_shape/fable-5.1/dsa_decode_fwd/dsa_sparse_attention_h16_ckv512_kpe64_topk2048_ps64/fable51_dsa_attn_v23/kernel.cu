// DSA sparse attention (DeepSeek V3.2 style) decode kernel for NVIDIA B200 (sm_100a).  v2
//
// Grid = num_tokens x NUM_SEG segments x CLUSTER_SIZE CTAs, one thread-block cluster of 8 CTAs per
// (segment, token), segment-major order. Each CTA owns TILE_KEYS = 64 top-k slots (one tile):
//   * every CTA reads the token's full 2048-slot index row (one ballot per warp) and derives the
//     activity of all 32 CTAs of the token, so inactive CTAs exit at once and the merge topology
//     is known everywhere without communication;
//   * cp.async gather with zero-fill for invalid slots, XOR-swizzled smem tile;
//   * QK^T with mma.sync m16n8k16 (M = 16 heads), Q fragments held in registers (each warp owns a
//     144-dim K-slice x 32-key half), S partials reduced in smem; softmax in fp32 (exp2);
//   * PV with the same smem tile (ldmatrix.trans), each warp owns 64 output dims;
//   * merge level 1 inside the cluster through fp16 DSMEM pushes (only among active ranks);
//   * merge level 2 across segments through a global workspace of tagged 8-byte words: a relaxed
//     per-(token, slice) counter elects the last-arriving owner, which polls the other segments'
//     words until they carry this call's tag (no release fence, no acquire reload).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <math.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace cg = cooperative_groups;

namespace {

constexpr int NUM_HEADS = 16;
constexpr int D_CKV = 512;
constexpr int D_KPE = 64;
constexpr int D_QK = D_CKV + D_KPE;  // 576
constexpr int TOPK = 2048;
constexpr int PAGE_SIZE = 64;

constexpr int CLUSTER_SIZE = 8;
constexpr int TILE_KEYS = 64;                          // slots per CTA
constexpr int SEG_KEYS = CLUSTER_SIZE * TILE_KEYS;     // 512 slots per cluster
constexpr int NUM_SEG = TOPK / SEG_KEYS;               // 4 segments per token
constexpr int CTAS_PER_TOKEN = NUM_SEG * CLUSTER_SIZE; // 32
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;            // 256
constexpr int NUM_SLICES = 8;                          // output dims split in 8 x 64
constexpr int SLICE_DIMS = D_CKV / NUM_SLICES;         // 64

constexpr int ROW_BYTES = D_QK * 2;                    // 1152 bytes per key row (ckv ++ kpe)
constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;         // 72 16-byte chunks
constexpr int CKV_CHUNKS = D_CKV * 2 / 16;             // 64
constexpr int TILE_BYTES = TILE_KEYS * ROW_BYTES;      // 73728
#ifndef DSA_QTMA
#define DSA_QTMA 1  // Q tile through cp.async.bulk (TMA engine) into padded rows (one mbarrier) instead of 1152 cp.async
#endif
#ifndef DSA_QTMA_LATE
#define DSA_QTMA_LATE 0  // 1: the Q bulk copies are issued after the tile ballots (active CTAs only)
#endif
// Q tile layout. DSA_QTMA: [16 heads][QN_STRIDE B] q_nope rows (1024 B + 16 B pad: the 8 rows of an ldmatrix 8x8
// matrix hit 8 distinct 16-B bank groups) followed by [16 heads][QP_STRIDE B] q_pe rows (128 + 16 B); otherwise
// [9 blocks][16 heads][128 B] with XOR-swizzled chunks.
constexpr int QN_STRIDE = D_CKV * 2 + 16;              // 1040
constexpr int QP_STRIDE = D_KPE * 2 + 16;              // 144
constexpr int QP_OFF = NUM_HEADS * QN_STRIDE;          // 16640
constexpr int Q_BYTES = DSA_QTMA ? (QP_OFF + NUM_HEADS * QP_STRIDE) : NUM_HEADS * ROW_BYTES;  // 18944 / 18432
constexpr int P_STRIDE = TILE_KEYS + 8;                // bf16 elements (144 B rows) for ldmatrix
constexpr int NUM_BLOCKS = D_QK / 64;                  // 9 dim-blocks of 64 bf16 (128 B per key row)
constexpr int KV_BLOCK_BYTES = TILE_KEYS * 128;        // 8192: [64 keys][128 B]
constexpr int Q_BLOCK_BYTES = NUM_HEADS * 128;         // 2048: [16 heads][128 B]
constexpr int S_SLICES = NUM_BLOCKS;                   // one S partial per dim-block (warp w -> block w, all -> block 8)
constexpr int S_STRIDE = TILE_KEYS + 8;                // fp32 elements per S-partial row (288 B: the fragment float2 stores of 8 rows x 4 lanes hit 32 distinct banks)
constexpr int RECV_SLOTS = 14;                         // max (7/n)*n + n-1 over n = 2..8 is 13
constexpr int STAGE_STRIDE = SLICE_DIMS + 8;           // fp16 elements per padded epilogue row (144 B)
constexpr int EPI_SLICE_BYTES = NUM_HEADS * STAGE_STRIDE * 2;  // 2304 bytes per padded epilogue slice
constexpr int SLICE_BYTES = NUM_HEADS * SLICE_DIMS * 2;  // 2048 bytes per exchanged slice: [16 heads][128 B], 16-B chunks XOR-swizzled by head % 8

// Shared memory layout (bytes)
constexpr int SMEM_KV_OFF = 0;                                   // KV tile
constexpr int SMEM_STAGE_OFF = 36864;                            // [8 slices][2048 B] fp16 = 16384 (KV blocks 4-5, dead after PV)
constexpr int SMEM_Q_OFF = SMEM_KV_OFF + TILE_BYTES;             // Q tile [9 blocks][16][128 B]
constexpr int SMEM_P_OFF = SMEM_Q_OFF + Q_BYTES;                 // [16][P_STRIDE] bf16
constexpr int SMEM_IDX_OFF = SMEM_P_OFF + NUM_HEADS * P_STRIDE * 2;  // [64] int32
constexpr int SMEM_MISC_OFF = SMEM_IDX_OFF + TILE_KEYS * 4;
// misc: s_scale[16], s_m[16], s_l[16] (f32), s_bits[8] (u32), s_old[8] (int)
// +256: merge mbarrier (8) +16 pad; +272: trace[16] (128); +400: gather mbarriers [9] (72)
constexpr int SMEM_GBAR_OFF = SMEM_MISC_OFF + 400;
constexpr int SMEM_MISC_BYTES = 400 + NUM_BLOCKS * 8;
constexpr int SMEM_S_OFF = (SMEM_MISC_OFF + SMEM_MISC_BYTES + 127) / 128 * 128;  // [9][16][S_STRIDE] f32
// Dedicated receive region (not aliasing the KV tile): remote pushes may land while this CTA is
// still computing, so the level-1 push needs no cluster barrier on the critical path.
constexpr int SMEM_RECV_O_OFF = (SMEM_S_OFF + S_SLICES * NUM_HEADS * S_STRIDE * 4 + 127) / 128 * 128;  // [RECV_SLOTS][2048 B] fp16 (swizzled)
constexpr int SMEM_RECV_ML_OFF = SMEM_RECV_O_OFF + RECV_SLOTS * SLICE_BYTES;  // [8 src][16][2] f32 = 1 KB
constexpr int SMEM_TOTAL = SMEM_RECV_ML_OFF + CLUSTER_SIZE * NUM_HEADS * 2 * 4;
static_assert(SMEM_STAGE_OFF + NUM_SLICES * SLICE_BYTES <= TILE_BYTES, "stage must fit in the KV tile");
static_assert(NUM_HEADS * D_CKV * 4 <= TILE_BYTES, "fp32 O staging must fit in the dead KV tile");
static_assert(SMEM_TOTAL <= 227 * 1024, "smem budget");
static_assert(SMEM_GBAR_OFF % 8 == 0, "mbarrier alignment");
constexpr int SMEM_LAUNCH = SMEM_TOTAL > 120 * 1024 ? SMEM_TOTAL : 120 * 1024;  // 1 CTA/SM

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gptr));
}
// src_bytes = 16 copies, src_bytes = 0 writes zeros without reading global memory.
__device__ __forceinline__ void cp_async_16_zfill(uint32_t smem_addr, const void* gptr, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gptr), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// wait_group with a (small) runtime count: resolved to an immediate once the callers' loops are unrolled
__device__ __forceinline__ void cp_async_wait_rt(int n) {
  switch (n) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    case 7: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 8;\n" ::: "memory"); break;
  }
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
// Store four 8x8 b16 matrices from mma C-fragment layout into smem rows (lane groups of 8 give row addresses).
__device__ __forceinline__ void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
// D = A(16x16 bf16) * B(16x8 bf16) + D, fp32 accumulate
__device__ __forceinline__ void mma_bf16_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack_half2(uint32_t v) {
  return __half22float2(*reinterpret_cast<const __half2*>(&v));
}
// Byte offset of 16-byte chunk `chunk` (0..71) of key row `row` in the [9 blocks][64 rows][128 B]
// layout (16-byte chunks XOR-swizzled by row % 8 inside each 128-byte row: ldmatrix conflict-free).
__device__ __forceinline__ uint32_t kv_off(int row, int chunk) {
  return static_cast<uint32_t>((chunk >> 3) * KV_BLOCK_BYTES + row * 128 + (((chunk & 7) ^ (row & 7)) << 4));
}
// Byte offset of 16-byte chunk `chunk` (0..7) of head row `h` in an exchanged slice ([16][128 B], chunks
// XOR-swizzled by h % 8: conflict-free stmatrix writes and row reads).
#ifndef DSA_NTMAJOR
#define DSA_NTMAJOR 0  // config-4: gather issued n-tile-major (a row's 4 blocks back to back = DRAM page locality, one commit group per n-tile), QK per n-tile with register-resident Q
#endif
#ifndef DSA_ROWMAJOR
#define DSA_ROWMAJOR 0  // experiment: config-4 gather issued row-major (a row's 4 blocks back to back, one commit group; QK waits for all)
#endif
#ifndef DSA_QREG
#define DSA_QREG 0  // config-4 QK: the warp's 16 Q k-step fragments are preloaded into registers (64 regs) before the block loop
#endif
#ifndef DSA_FUSED4
#define DSA_FUSED4 0  // config-4 tiles (5-8 n-tiles): key-group warp pairs exchange one S partial, fused softmax
#endif
#ifndef DSA_FUSED_NT
#define DSA_FUSED_NT 4  // tiles with <= this many n-tiles use the fused per-warp QK + softmax path (0 = off)
#endif
#ifndef DSA_WARP_GATHER
#define DSA_WARP_GATHER 1  // each warp copies exactly the (keys x dim-blocks) sub-tile its QK consumes (cp.async groups, no gather mbarriers)
#endif
#ifndef DSA_PV_SPLIT
#define DSA_PV_SPLIT 1  // PV in this many dim parts per warp, each part pushed as soon as it is ready (2 measured +0.18 us: PV 0.45 -> 0.7-1.0 us, pipeline drain at the part boundary)
#endif
// With DSA_PV_SPLIT parts, the slice is stored part-major: [part][16 heads][CPP chunks] (each part is a
// contiguous SLICE_BYTES / PV_SPLIT block that can be pushed on its own), chunks XOR-swizzled by
// (h / (8 / CPP)) % CPP so that the 8 rows of an stmatrix 8x8 matrix hit 8 distinct bank groups.
constexpr int PV_SPLIT = DSA_PV_SPLIT;
constexpr int SLICE_CPP = 8 / PV_SPLIT;                   // 16-byte chunks per head row per part
constexpr int PART_BYTES = SLICE_BYTES / PV_SPLIT;        // 2048 / PV_SPLIT
static_assert(PV_SPLIT == 1 || PV_SPLIT == 2 || PV_SPLIT == 4, "PV_SPLIT");
__device__ __forceinline__ uint32_t slice_off(int h, int chunk) {
  const int part = chunk / SLICE_CPP, c = chunk % SLICE_CPP;
  const int sw = (h / (8 / SLICE_CPP)) & (SLICE_CPP - 1);
  return static_cast<uint32_t>(part * PART_BYTES + h * (SLICE_CPP * 16) + ((c ^ sw) << 4));
}
__device__ __forceinline__ uint32_t q_off(int row, int chunk) {
  return static_cast<uint32_t>((chunk >> 3) * Q_BLOCK_BYTES + row * 128 + (((chunk & 7) ^ (row & 7)) << 4));
}
// Arrive on `bar` once all prior cp.async of this thread have completed (counts as one arrival).
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT2:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT2;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_init_fence_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_parity_cluster(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT;\n"
      "}\n" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
#ifndef DSA_NO_CLUSTER
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
#ifndef DSA_NO_CLUSTER
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#endif
}
// shared::cta address -> shared::cluster address of the same offset in CTA `rank`
__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
// 16-byte / 8-byte asynchronous stores into another CTA's smem, signalling its mbarrier (tx bytes)
// 8-byte asynchronous store into another CTA's smem, signalling its mbarrier (8 tx bytes).
__device__ __forceinline__ void st_async_v2f32(uint32_t raddr, float a, float b, uint32_t rbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n"
               ::"r"(raddr), "f"(a), "f"(b), "r"(rbar) : "memory");
}
// Bulk copy local smem -> global (bulk async-group completion), and the group wait.
__device__ __forceinline__ void bulk_copy_s2g(void* gdst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(gdst), "r"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_commit_group() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_group_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
// Wait until the smem sources of all committed bulk groups have been read (not for global completion).
__device__ __forceinline__ void bulk_wait_group_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
// 16-byte asynchronous store into another CTA's smem, signalling its mbarrier (16 tx bytes).
__device__ __forceinline__ void st_async_v4b32(uint32_t raddr, uint4 v, uint32_t rbar) {
  asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               ::"r"(raddr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(rbar) : "memory");
}
// Bulk copy from local smem into another CTA's smem (shared::cluster address), completion signalled
// on that CTA's mbarrier (tx bytes). Sizes / addresses must be multiples of 16.
__device__ __forceinline__ void bulk_copy_s2s(uint32_t rdst, uint32_t src, uint32_t bytes, uint32_t rbar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(rdst), "r"(src), "r"(bytes), "r"(rbar) : "memory");
}
// Bulk copy global -> local smem (shared::cluster form with a local address), completion on `bar`.
__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// Prefetch `bytes` (multiple of 16) of global memory into L2 (fire-and-forget, no smem).
__device__ __forceinline__ void prefetch_l2_bulk(const void* gptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(gptr), "r"(bytes) : "memory");
}
__device__ __forceinline__ void fence_proxy_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// 16-byte global accesses made of two 8-byte scalar words (each word is single-copy atomic at gpu
// scope, so a (data, tag) pair is either entirely old or entirely new).
__device__ __forceinline__ void st_relaxed_v2u64(void* gptr, uint4 v) {
  const uint64_t lo = (static_cast<uint64_t>(v.y) << 32) | v.x;
  const uint64_t hi = (static_cast<uint64_t>(v.w) << 32) | v.z;
  asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1, %2};\n" ::"l"(gptr), "l"(lo), "l"(hi) : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_v2u64(const void* gptr) {
  uint64_t lo, hi;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];\n" : "=l"(lo), "=l"(hi) : "l"(gptr) : "memory");
  return make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32), static_cast<uint32_t>(hi),
                    static_cast<uint32_t>(hi >> 32));
}

// 16-byte atomic read through the L2 atomic units (CAS with an impossible expected value: a published
// word always carries a non-zero tag, so nothing is ever written); predicated like the load form.
__device__ __forceinline__ void atom_cas128_pred(uint4& v, void* gptr, bool pred) {
  uint64_t lo, hi;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b128 cur, cmp;\n"
      "setp.ne.b32 p, %3, 0;\n"
      "mov.b128 cmp, {0, 0};\n"
      "mov.b64 %0, 0;\n"
      "mov.b64 %1, 0;\n"
      "@p atom.relaxed.gpu.global.cas.b128 cur, [%2], cmp, cmp;\n"
      "@p mov.b128 {%0, %1}, cur;\n"
      "}\n"
      : "=l"(lo), "=l"(hi)
      : "l"(gptr), "r"(static_cast<int>(pred))
      : "memory");
  v = make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32), static_cast<uint32_t>(hi),
                 static_cast<uint32_t>(hi >> 32));
}
// Predicated form: v is zeroed when pred is false (no branch, so consecutive loads pipeline).
__device__ __forceinline__ void ld_relaxed_v2u64_pred(uint4& v, const void* gptr, bool pred) {
  uint64_t lo, hi;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %3, 0;\n"
      "@p ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];\n"
      "@!p mov.b64 %0, 0;\n"
      "@!p mov.b64 %1, 0;\n"
      "}\n"
      : "=l"(lo), "=l"(hi)
      : "l"(gptr), "r"(static_cast<int>(pred))
      : "memory");
  v = make_uint4(static_cast<uint32_t>(lo), static_cast<uint32_t>(lo >> 32), static_cast<uint32_t>(hi),
                 static_cast<uint32_t>(hi >> 32));
}

// ex2.approx.ftz (2 ulp, exp2(-inf) = +0): the full-range exp2f adds a denormal-result fix-up branch
// per call; results feed bf16 P values / fp16-weighted merges, so the approximation is far below the
// output precision. lg2.approx is used only on l >= 1 (the max key contributes exp2(0) = 1).
__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float fast_log2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Warp-level fp32 max over the lanes of `mask` (sm_100a redux.sync.f32; all lanes of the mask must
// execute it together).
__device__ __forceinline__ float redux_max_f32(float x, unsigned mask) {
  float y;
  asm volatile("redux.sync.max.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "r"(mask));
  return y;
}

__device__ __forceinline__ unsigned long long globaltimer_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Byte offset of 16-byte chunk `chunk` (0..7) of row `row` inside one [rows][128 B] block (KV blocks and
// Q blocks share the swizzle: chunk ^ (row % 8)).
__device__ __forceinline__ uint32_t blk_off(int row, int chunk) {
  return static_cast<uint32_t>(row * 128 + ((chunk ^ (row & 7)) << 4));
}
// smem address of 16-byte chunk `chunk` (0..7) of head row `row` in Q dim-block `b` (0..7 ckv, 8 kpe)
__device__ __forceinline__ uint32_t q_addr(uint32_t q_base, int b, int row, int chunk) {
#if DSA_QTMA
  return b < NUM_BLOCKS - 1 ? q_base + row * QN_STRIDE + b * 128 + chunk * 16
                            : q_base + QP_OFF + row * QP_STRIDE + chunk * 16;
#else
  return q_base + b * Q_BLOCK_BYTES + blk_off(row, chunk);
#endif
}

#ifndef DSA_KG_BIG
#define DSA_KG_BIG 4  // key groups for tiles with >= 5 n-tiles (4: two n-tiles x 4 blocks per warp; 8 measured neutral: single mma chain + 2x Q ldmatrix traffic)
#endif
static_assert(S_STRIDE == 64 + 8, "S_STRIDE");
constexpr int S_STRIDE_ = S_STRIDE;
// QK^T over NT n-tiles (keys nt0*8 ..) x the 8/NT ckv dim-blocks starting at b0, waiting for each block's
// mbarrier before using it; the n-tiles beyond n_valid are skipped. acc starts from `init` (NT == 1: the
// warp's kpe partial). Result stored as fp32 S partial rows r0 / r0+8 at `dst` (a [16][S_STRIDE] slot).
template <int NT, bool QREG = false>
__device__ __forceinline__ void qk_ckv(uint32_t kv_base, uint32_t q_base, uint32_t gbar0, int lane, int nt0,
                                       int n_valid, int b0, const float* init, float* dst,
                                       float* acc_out = nullptr, const float* init1 = nullptr) {
  constexpr int NB = 8 / NT;  // ckv blocks handled by this warp
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_chunk = lane >> 4;
  const int b_key = (lane & 7) + (lane >> 4) * 8;
  const int b_chunk = (lane >> 3) & 1;
  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = (i == 0) ? init[j] : ((i == 1 && init1 != nullptr) ? init1[j] : 0.f);
  }
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};  // NT == 1: second accumulation chain (odd k-steps)
  uint32_t qa[QREG ? NB * 4 : 1][4];     // QREG: the warp's Q fragments (k-step bi*4+j), loaded once (Q has landed)
  if constexpr (QREG) {
#pragma unroll
    for (int bi = 0; bi < NB; ++bi) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        ldmatrix_x4(q_addr(q_base, b0 + bi, a_row, 2 * j + a_chunk), qa[bi * 4 + j][0], qa[bi * 4 + j][1], qa[bi * 4 + j][2], qa[bi * 4 + j][3]);
    }
  }
#pragma unroll
  for (int bi = 0; bi < NB; ++bi) {
    const int b = b0 + bi;
#if DSA_WARP_GATHER
    if (DSA_ROWMAJOR && NT == 2) cp_async_wait<0>(); else cp_async_wait_rt(NB - 1 - bi);  // this warp's copies of block b have landed
    __syncwarp();
#elif !defined(DSA_DIAG_NOARRIVE)
    mbar_wait_parity(gbar0 + 8 * b, 0);  // block b is complete (all threads' copies)
#endif
    const uint32_t kvb = kv_base + b * KV_BLOCK_BYTES;
    if constexpr (NT == 1) {
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {  // two k-steps per B ldmatrix (chunks 4jp .. 4jp+3 of keys nt0*8..)
        uint32_t bb[4];
        ldmatrix_x4(kvb + blk_off(nt0 * 8 + (lane & 7), 4 * jp + (lane >> 3)), bb[0], bb[1], bb[2], bb[3]);
        uint32_t a[4];
        ldmatrix_x4(q_addr(q_base, b, a_row, 4 * jp + a_chunk), a[0], a[1], a[2], a[3]);
        mma_bf16_16816(acc[0], a, bb);
        ldmatrix_x4(q_addr(q_base, b, a_row, 4 * jp + 2 + a_chunk), a[0], a[1], a[2], a[3]);
        mma_bf16_16816(acc1, a, bb + 2);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // k-steps of the block
        uint32_t a[4];
        if constexpr (QREG) {
          a[0] = qa[bi * 4 + j][0]; a[1] = qa[bi * 4 + j][1]; a[2] = qa[bi * 4 + j][2]; a[3] = qa[bi * 4 + j][3];
        } else {
          ldmatrix_x4(q_addr(q_base, b, a_row, 2 * j + a_chunk), a[0], a[1], a[2], a[3]);
        }
#pragma unroll
        for (int np = 0; np < NT / 2; ++np) {  // n-tile pairs
          if (2 * np >= n_valid) break;
          uint32_t bb[4];
          ldmatrix_x4(kvb + blk_off((nt0 + 2 * np) * 8 + b_key, 2 * j + b_chunk), bb[0], bb[1], bb[2], bb[3]);
          mma_bf16_16816(acc[2 * np], a, bb);
          mma_bf16_16816(acc[2 * np + 1], a, bb + 2);
        }
      }
    }
  }
  const int r0 = lane >> 2;
  const int c0 = (lane & 3) * 2;
  if constexpr (NT == 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[0][j] += acc1[j];
  }
  if (acc_out != nullptr) {  // fused path: the fragments stay in registers ([NT][4])
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc_out[nt * 4 + j] = acc[nt][j];
    }
    return;
  }
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    if (nt >= n_valid) break;
    const int key = (nt0 + nt) * 8 + c0;
    *reinterpret_cast<float2*>(dst + r0 * S_STRIDE_ + key) = make_float2(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<float2*>(dst + (r0 + 8) * S_STRIDE_ + key) = make_float2(acc[nt][2], acc[nt][3]);
  }
}

// Config-4, n-tile-major: S of n-tiles nt0, nt0+1 (keys 8nt0 ..) over the 4 ckv blocks b0 .. b0+3, one
// cp.async commit group per n-tile (n-tile 0's rows land first); Q fragments of the 16 k-steps are loaded
// into registers once (the Q tile has landed), each n-tile runs two accumulation chains (even / odd k-steps).
__device__ __forceinline__ void qk_ckv_nt(uint32_t kv_base, uint32_t q_base, int lane, int nt0, int n_valid, int b0,
                                          float* dst) {
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int a_chunk = lane >> 4;
  uint32_t qa[16][4];
#pragma unroll
  for (int bi = 0; bi < 4; ++bi) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      ldmatrix_x4(q_addr(q_base, b0 + bi, a_row, 2 * j + a_chunk), qa[bi * 4 + j][0], qa[bi * 4 + j][1], qa[bi * 4 + j][2], qa[bi * 4 + j][3]);
  }
  const int r0 = lane >> 2;
  const int c0 = (lane & 3) * 2;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    if (nt >= n_valid) break;
    cp_async_wait_rt(1 - nt);  // this n-tile's rows (my commit group) have landed
    __syncwarp();
    float accA[4] = {0.f, 0.f, 0.f, 0.f}, accB[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int bi = 0; bi < 4; ++bi) {
      const uint32_t kvb = kv_base + (b0 + bi) * KV_BLOCK_BYTES;
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {  // two k-steps per ldmatrix.x4 (chunks 4jp .. 4jp+3 of the 8 keys)
        uint32_t bb[4];
        ldmatrix_x4(kvb + blk_off((nt0 + nt) * 8 + (lane & 7), 4 * jp + (lane >> 3)), bb[0], bb[1], bb[2], bb[3]);
        mma_bf16_16816(accA, qa[bi * 4 + 2 * jp], bb);
        mma_bf16_16816(accB, qa[bi * 4 + 2 * jp + 1], bb + 2);
      }
    }
    const int key = (nt0 + nt) * 8 + c0;
    *reinterpret_cast<float2*>(dst + r0 * S_STRIDE_ + key) = make_float2(accA[0] + accB[0], accA[1] + accB[1]);
    *reinterpret_cast<float2*>(dst + (r0 + 8) * S_STRIDE_ + key) = make_float2(accA[2] + accB[2], accA[3] + accB[3]);
  }
}

template <int N> struct IntC { static constexpr int value = N; };


struct Params {
  const __nv_bfloat16* __restrict__ q_nope;  // [T,16,512]
  const __nv_bfloat16* __restrict__ q_pe;    // [T,16,64]
  const __nv_bfloat16* __restrict__ ckv;     // [P*64,512]
  const __nv_bfloat16* __restrict__ kpe;     // [P*64,64]
  const int32_t* __restrict__ indices;       // [T,2048]
  __nv_bfloat16* __restrict__ out;           // [T,16,512]
  float* __restrict__ lse;                   // [T,16]
  int* __restrict__ counters;                // [T][8] per-(token, slice) arrival counters (self-reset)
  uint4* __restrict__ ws_o;                  // [T][NUM_SEG][8 slices][16 heads][11]: (half2, half2, half2, tag) = dims 6j..6j+5
  uint4* __restrict__ ws_ml;                 // [T][NUM_SEG][16 heads]: (M bits, tag, L bits, tag)
  float scale_log2;                          // sm_scale * log2(e)
  int num_tokens;
  uint32_t tag;                              // per-call tag (never 0; the workspace starts zeroed)
  unsigned long long* trace;                 // dev-only: [grid][TRACE_SLOTS] globaltimer stamps
};

#ifndef DSA_ORDER
#define DSA_ORDER 3
#endif
#ifndef FULL_TOKENS
#define FULL_TOKENS 3
#endif
#ifndef DSA_REST
#define DSA_REST 1
#endif
#ifndef DSA_EPI_TMA
#define DSA_EPI_TMA 3
#endif
#ifndef DSA_PUSH_SPLIT
#define DSA_PUSH_SPLIT 3  // slices (of 8) pushed with st.async from all threads instead of bulk copies
#endif
#ifndef DSA_POLL_TMA
#define DSA_POLL_TMA 0
#endif
#ifndef DSA_POLL_LD
#define DSA_POLL_LD 1
#endif
#ifndef DSA_WS_PREFETCH
#define DSA_WS_PREFETCH 1  // merger CTAs prefetch their level-2 poll lines into L2 during the gather latency
#endif
#ifndef DSA_POLL_STAGGER_NS
#define DSA_POLL_STAGGER_NS 0  // > 0: a second level-2 poll issued this long after the first without waiting (measured +0.2 us: the two polls compete for the outstanding-load budget)
#endif
#ifndef DSA_CWAIT_EARLY
#define DSA_CWAIT_EARLY 0  // 1: phase-1 cluster wait before the kpe block wait (measured +0.2..0.45 us: an acquire wait with the gather in flight stalls)
#endif
#ifndef DSA_ATOM_SEM
#define DSA_ATOM_SEM 0
#endif
#ifndef DSA_EPI_BULK_WARPS
#define DSA_EPI_BULK_WARPS 0  // single-CTA epilogue: warps >= 8 - N store their slice with per-row cp.async.bulk instead of st.global
#endif
#ifndef DSA_LSE_FIRST
#define DSA_LSE_FIRST 0  // single-CTA epilogue: lse store issued before the output stores
#endif
#ifndef DSA_LOCAL_SLICE
#define DSA_LOCAL_SLICE 1  // level-1: a rank's own slices are read from its staging area instead of being DSMEM-copied to itself
#endif
#ifndef DSA_ABL_POLL_HEADS
#define DSA_ABL_POLL_HEADS 16  // ablation only (wrong results): the merger polls / checks only heads < N
#endif
#ifndef DSA_SUM_UNROLL
#define DSA_SUM_UNROLL 0  // softmax: S partial slots loaded at once per kg configuration (measured neutral, +0.02 us all-23)
#endif
#ifndef DSA_REDUX
#define DSA_REDUX 0  // softmax: 16-lane max with redux.sync.max.f32 (measured +0.07 us all-23: slower than 4 shuffles)
#endif
constexpr int TRACE_SLOTS = 16;
constexpr int POLL_LIMIT = 1 << 20;  // safety net for the level-2 polling loop (never reached in practice)
#ifndef DSA_EARLY_EXIT
#define DSA_EARLY_EXIT 1  // inactive CTAs (other than CTA 0 of the token) exit right after their tile ballots
#endif
#ifndef DSA_WS4
#define DSA_WS4 0  // 1: level-2 words = 4 fp16 + pad + tag (16 per head row, one per thread, no smem re-map; measured +0.1..0.2 us on big workloads); 0 = 6 fp16 + tag (11 per row)
#endif
// 16-byte words per (slice, head) row: 16 (4 fp16 + 32-bit pad + 32-bit tag) or 11 (6 fp16 + 32-bit tag)
constexpr int WS_O_STRIDE = DSA_WS4 ? SLICE_DIMS / 4 : (SLICE_DIMS + 5) / 6;
constexpr int WS_O_THREADS = WS_O_STRIDE;  // publishing / polling threads per head
#ifdef DSA_TRACE
// Stamps are kept in shared memory (no global stores that a release fence would have to wait for)
// and flushed by thread 0 right before the CTA exits.
#define TRACE_STAMP(slot)                                                      \
  do {                                                                         \
    if (threadIdx.x == 0) s_trace[(slot)] = globaltimer_ns();                  \
  } while (0)
#define TRACE_FLUSH()                                                                                  \
  do {                                                                                                 \
    if (p.trace != nullptr && threadIdx.x == 0) {                                                      \
      _Pragma("unroll") for (int _i = 0; _i < TRACE_SLOTS; ++_i)                                       \
        p.trace[blockIdx.x * TRACE_SLOTS + _i] = s_trace[_i];                                          \
    }                                                                                                  \
  } while (0)
#define TRACE_STAMP_W7(slot)                                                      \
  do {                                                                            \
    if (warp == NUM_WARPS - 1 && lane == 0) s_trace[(slot)] = globaltimer_ns();   \
  } while (0)
#else
#define TRACE_STAMP_W7(slot) \
  do {                       \
  } while (0)
#define TRACE_STAMP(slot) \
  do {                    \
  } while (0)
#define TRACE_FLUSH() \
  do {                \
  } while (0)
#endif

#ifdef DSA_NO_CLUSTER  // measurement only: launch overhead without clusters (single-CTA tokens only)
__global__ void __launch_bounds__(NUM_THREADS, 1) dsa_sparse_attn_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int rank = static_cast<int>(blockIdx.x & (CLUSTER_SIZE - 1));
#else
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
    dsa_sparse_attn_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
#endif
  const int cluster_id = blockIdx.x / CLUSTER_SIZE;
#if DSA_ORDER == 0
  // segment-major: segment 0 of every token launches first
  const int token = cluster_id % p.num_tokens;
  const int seg = cluster_id / p.num_tokens;
#elif DSA_ORDER == 1
  // token-major: the four segment clusters of a token are adjacent in the launch queue, so the
  // big tokens (all four segments active) are placed in the earliest possible hardware wave
  const int token = cluster_id / NUM_SEG;
  const int seg = cluster_id % NUM_SEG;
#elif DSA_ORDER == 3
  // shape-group specialized: segment 0 of every token first (always active unless the token is empty),
  // then segments 1-3 token by token in a per-T priority order (the tokens observed to be large in the
  // workload set: T=8 -> 2,5,6,4; T=7 -> 2,6,5,4; T=6 -> 2,4,5,3), so that with 15 co-resident clusters
  // the full segments of the large tokens start in the first hardware wave. Division-free.
  int token, seg;
  if (cluster_id < p.num_tokens) {
    token = cluster_id;
    seg = 0;
  } else {
    const int r = cluster_id - p.num_tokens;  // 0 .. 3T-1 < 32
    const int k = (r * 11) >> 5;              // r / 3 for r < 32
    seg = 1 + (r - 3 * k);
    const unsigned prio = (p.num_tokens == 8) ? 0x73104652u : (p.num_tokens == 7) ? 0x3104562u
                        : (p.num_tokens == 6) ? 0x103542u : 0x76543210u;  // nibble k = k-th token
    token = static_cast<int>((prio >> (4 * k)) & 15u);
  }
#else
  // hybrid: the first FULL_TOKENS tokens token-major (all their segments in the first hardware
  // wave of 15 clusters); for the remaining tokens segment 0 (always active) first, then segment 1,
  // then (segment 2, segment 3) pairs per token (DSA_REST 1) or plain segment-major (DSA_REST 0).
  // No runtime division: the launch order is on the critical path of the index-row load.
  const int n_full = p.num_tokens < FULL_TOKENS ? p.num_tokens : FULL_TOKENS;
  int token, seg;
  if (cluster_id < n_full * NUM_SEG) {
    token = cluster_id >> 2;
    seg = cluster_id & 3;
  } else {
    const int r = cluster_id - n_full * NUM_SEG;
    const int rest = p.num_tokens - n_full;
#if DSA_REST == 0
    seg = (r >= rest) + (r >= 2 * rest) + (r >= 3 * rest);
    token = n_full + r - seg * rest;
#else
    if (r < 2 * rest) {
      seg = (r >= rest) ? 1 : 0;
      token = n_full + r - seg * rest;
    } else {
      const int r2 = r - 2 * rest;
      seg = 2 + (r2 & 1);
      token = n_full + (r2 >> 1);
    }
#endif
  }
#endif
  const int cit = seg * CLUSTER_SIZE + rank;  // CTA index within the token (0..31)
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  int32_t* s_idx = reinterpret_cast<int32_t*>(smem + SMEM_IDX_OFF);               // [64]
  float* s_S = reinterpret_cast<float*>(smem + SMEM_S_OFF);                         // [9][16][S_STRIDE]
  __nv_bfloat16* s_P = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_P_OFF);         // [16][P_STRIDE]
  float* s_scale = reinterpret_cast<float*>(smem + SMEM_MISC_OFF);                  // [16]
  float* s_m = s_scale + 16;                                                        // [16]
  float* s_l = s_m + 16;                                                            // [16]
  unsigned* s_bits = reinterpret_cast<unsigned*>(s_l + 16);                         // [8] warp ballots
  // (s_bits + 8: 8 ints formerly used for atomic results; unused, layout kept)
  uint64_t* s_mbar = reinterpret_cast<uint64_t*>(smem + SMEM_MISC_OFF + 16 * 4 * 3 + 8 * 4 + 8 * 4);  // merge mbarrier
  uint64_t* s_mbar_ml = s_mbar + 1;  // (m, l) exchange mbarrier
  unsigned long long* s_trace = reinterpret_cast<unsigned long long*>(smem + SMEM_MISC_OFF + 16 * 4 * 3 + 8 * 4 + 8 * 4 + 16);
  const uint32_t gbar0 = smem_u32(smem + SMEM_GBAR_OFF);  // 9 gather mbarriers, 8 B apart
#if DSA_QTMA
  static_assert(DSA_WARP_GATHER, "DSA_QTMA reuses the first gather mbarrier slot");
  const uint32_t qbar = gbar0;  // Q tile mbarrier (expect_tx 18432 B; every warp waits before its first Q read)
#endif
#ifdef DSA_TRACE
  if (tid < TRACE_SLOTS) s_trace[tid] = 0ull;
#endif
  TRACE_STAMP(0);
  if (tid == 0) {
    mbar_init(smem_u32(s_mbar), 1);
    mbar_init(smem_u32(s_mbar_ml), 1);
#if DSA_QTMA
    mbar_init(qbar, 1);
    mbar_arrive_expect_tx(qbar, NUM_HEADS * ROW_BYTES);  // 18432 B of payload (the row padding is never written)
#endif
#if !DSA_WARP_GATHER
#pragma unroll
    for (int b = 0; b < NUM_BLOCKS; ++b) mbar_init(gbar0 + 8 * b, NUM_THREADS);
#endif
    mbar_init_fence_cluster();
  }

  // ---- 1. index loads: my tile's 64 slots per warp (lane l holds slots l and l + 32) and the
  //         token's full 2048-slot row (thread t holds slots [8t, 8t+8)) for the activity of all
  //         32 CTAs of the token. The tile slots feed the gather issue directly (no smem hop, no
  //         CTA barrier); the row is consumed afterwards, in the shadow of the gather's DRAM latency.
  const int32_t* tok_idx = p.indices + static_cast<size_t>(token) * TOPK;
  const int32_t my_lo = tok_idx[cit * TILE_KEYS + lane];
  const int32_t my_hi = tok_idx[cit * TILE_KEYS + 32 + lane];
  const int4* idx_row = reinterpret_cast<const int4*>(tok_idx);
  const int4 ia = idx_row[tid * 2];
  const int4 ib = idx_row[tid * 2 + 1];
  // ---- 2. Q tile (16 x 576) -> smem while the index loads are in flight ------------------------
#if DSA_QTMA
  // Bulk copies (TMA engine, no LSU slots): lane 0 of warp w copies head rows 2w, 2w+1 of q_nope (1024 B)
  // and q_pe (128 B) into the padded rows; completion on qbar. The barrier init must be visible first.
  __syncthreads();
  auto issue_q = [&]() {
    if (lane == 0) {
      const uint32_t q_base = smem_u32(smem + SMEM_Q_OFF);
#pragma unroll
      for (int r = 2 * warp; r < 2 * warp + 2; ++r) {
        bulk_copy_g2s(q_base + r * QN_STRIDE, p.q_nope + (static_cast<size_t>(token) * NUM_HEADS + r) * D_CKV, D_CKV * 2, qbar);
        bulk_copy_g2s(q_base + QP_OFF + r * QP_STRIDE, p.q_pe + (static_cast<size_t>(token) * NUM_HEADS + r) * D_KPE, D_KPE * 2, qbar);
      }
    }
  };
#if !DSA_QTMA_LATE
  issue_q();
#endif
#else
  {
    const uint32_t q_base = smem_u32(smem + SMEM_Q_OFF);
    constexpr int TOTAL = NUM_HEADS * CHUNKS_PER_ROW;  // 1152
#ifdef DSA_DIAG_NOQ
    for (int c = tid; c < 0; c += NUM_THREADS) {  // diag: no Q loads (wrong results)
#else
    for (int c = tid; c < TOTAL; c += NUM_THREADS) {
#endif
      const int h = c / CHUNKS_PER_ROW;
      const int chunk = c - h * CHUNKS_PER_ROW;
      const char* src;
      if (chunk < CKV_CHUNKS) {
        src = reinterpret_cast<const char*>(p.q_nope) +
              (static_cast<size_t>(token) * NUM_HEADS + h) * D_CKV * 2 + chunk * 16;
      } else {
        src = reinterpret_cast<const char*>(p.q_pe) +
              (static_cast<size_t>(token) * NUM_HEADS + h) * D_KPE * 2 + (chunk - CKV_CHUNKS) * 16;
      }
      cp_async_16(q_base + q_off(h, chunk), src);
    }
#if DSA_WARP_GATHER
    cp_async_commit();  // group 0: the Q tile (complete before any later group)
#endif
  }
#endif
#ifdef DSA_DIAG_STAMPS
  TRACE_STAMP(4);  // diag: Q loads issued
#endif
  // ---- 2b. TLB warm-up: the benchmark flushes L2 before every call, so the first touch of the
  //          small pages used only at the end (workspace, output, lse) would pay a page walk on the
  //          critical path. Load one word of each now; the results are consumed after the softmax. --
  int warm = 0;
#ifndef DSA_NO_WARM
  if (tid < 4) {
    const void* wp = (tid == 0) ? static_cast<const void*>(p.ws_ml + static_cast<size_t>(token) * NUM_SEG * NUM_HEADS)
                   : (tid == 1) ? static_cast<const void*>(p.ws_o + static_cast<size_t>(token) * NUM_SEG * NUM_SLICES * NUM_HEADS * WS_O_STRIDE)
                   : (tid == 2) ? static_cast<const void*>(p.out + static_cast<size_t>(token) * NUM_HEADS * D_CKV)
                                : static_cast<const void*>(p.lse + token * NUM_HEADS);
    warm = __ldcg(static_cast<const int*>(wp));
  }
#endif
#if !DSA_QTMA
  __syncthreads();  // mbarrier inits visible to every thread before the gather arrives on them
#endif
  // ---- 3. gather the tile (zero-fill invalid slots) as soon as this warp's tile slots land:
  //         dim-block-major so that block b completes early and warp b can start its QK while the
  //         remaining blocks are still in flight. Each thread handles keys tid/8 and 32 + tid/8. ----
  const unsigned m_lo = __ballot_sync(0xffffffffu, my_lo >= 0);  // bit k: slot k valid
  const unsigned m_hi = __ballot_sync(0xffffffffu, my_hi >= 0);  // bit k: slot 32 + k valid
#ifdef DSA_DIAG_STAMPS
  TRACE_STAMP(8);  // diag: tile slots landed (ballots done)
#endif
#if DSA_EARLY_EXIT
  // Inactive CTA (no valid slot in its tile; CTA-uniform: every warp loaded the same 64 slots): arrive on
  // the cluster barrier (exited CTAs count as arrived in the later phases) and leave as soon as the Q
  // copies have landed. CTA 0 of the token keeps the full-row path: it writes the zero output of a
  // token without any valid key.
  if ((m_lo | m_hi) == 0u && cit != 0) {
    cluster_arrive_relaxed();
    cp_async_wait<0>();
#if DSA_QTMA && !DSA_QTMA_LATE
    mbar_wait_parity(qbar, 0);  // the Q bulk copies target this CTA's smem: they must land before the exit
#endif
    TRACE_STAMP(15);
    TRACE_FLUSH();
    return;
  }
#endif
#if DSA_QTMA && DSA_QTMA_LATE
  issue_q();
#endif
  const int n_rows = m_hi ? (64 - __clz(m_hi)) : (m_lo ? (32 - __clz(m_lo)) : 0);  // highest valid slot + 1
  const int n_ksteps = (n_rows + 15) >> 4;           // PV k-steps actually needed
  const int n_ntiles = (n_rows + 7) >> 3;
  // CTA-uniform QK config: 8 = fused per-warp QK + softmax (warp w = n-tile w over all 576 dims)
  const int kg_cfg = (n_ntiles <= DSA_FUSED_NT) ? 8 : (n_ntiles <= 2) ? 1 : ((n_ntiles <= 4) ? 2 : DSA_KG_BIG);
  auto slot_of = [&](int key) -> int {  // top-k slot of key 0..63 (both shuffles executed: uniform control flow)
    const int lo = __shfl_sync(0xffffffffu, my_lo, key & 31);
    const int hi = __shfl_sync(0xffffffffu, my_hi, key & 31);
    return key < 32 ? lo : hi;
  };
#if DSA_WARP_GATHER
  // Per-warp gather ownership: warp w copies the kpe rows of n-tile w (keys 8w..8w+7) and the ckv sub-tile
  // its QK consumes (config 1: rows 0..15 of block w; config 2: rows 32kg..+32 of blocks 2dg, 2dg+1;
  // config 4: rows 16kg..+16 of blocks 4dg..+4), one cp.async commit group per block (kpe first), so the
  // QK waits only on this warp's own groups (cp.async.wait_group + __syncwarp): no gather mbarriers, no
  // per-thread arrives. The PV reads the whole tile after cp_async_wait<0> + __syncthreads.
  {
    const uint32_t dst_base = smem_u32(smem + SMEM_KV_OFF);
    const int n_pad = n_ksteps * 16;  // rows up to the next multiple of 16 are loaded or zero-filled
    const char* ckv_c = reinterpret_cast<const char*>(p.ckv);
    const char* kpe_c = reinterpret_cast<const char*>(p.kpe);
    // Lane mapping keeps every warp instruction on 4 rows x 128 contiguous bytes (lanes 8r..8r+7 = the 8
    // chunks of one row), so each instruction is 4 coalesced line requests, as in the original gather.
    // Per row: one global base, one smem base and the zero-fill size, then immediate offsets per block.
    const int cc = lane & 7;   // chunk within the 128-B block
    const int rg = lane >> 3;  // row within the instruction's group of 4
    struct RowSrc { const char* g; uint32_t s; uint32_t sz; };
    auto row_src = [&](int key, int blk0) -> RowSrc {  // key row, ckv block blk0 (chunk cc), smem block blk0
      const int slot = slot_of(key);
      const bool v = slot >= 0;
      RowSrc r;
      r.g = ckv_c + (v ? static_cast<size_t>(slot) : 0) * (D_CKV * 2) + blk0 * 128 + cc * 16;
      r.s = dst_base + blk0 * KV_BLOCK_BYTES + key * 128 + ((cc ^ (key & 7)) << 4);
      r.sz = v ? 16u : 0u;
      return r;
    };
#if DSA_FUSED4
    if (kg_cfg == 4) {  // kpe rows of key group kg = warp (warps 0-3; keys 16kg .. +16, 4 instructions)
      if (warp < 4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int k = warp * 16 + 4 * j + rg;
          const int sk = slot_of(k);
          if (k < n_pad)
            cp_async_16_zfill(dst_base + (NUM_BLOCKS - 1) * KV_BLOCK_BYTES + k * 128 + ((cc ^ (k & 7)) << 4),
                              kpe_c + (sk >= 0 ? static_cast<size_t>(sk) : 0) * (D_KPE * 2) + cc * 16, sk >= 0 ? 16u : 0u);
        }
      }
      cp_async_commit();
    } else
#endif
    {  // kpe rows of n-tile `warp` (keys 8w .. 8w+7): two instructions of 4 rows
      const int k0 = warp * 8 + rg, k1 = k0 + 4;
      const int s0 = slot_of(k0), s1 = slot_of(k1);
      if (k0 < n_pad)
        cp_async_16_zfill(dst_base + (NUM_BLOCKS - 1) * KV_BLOCK_BYTES + k0 * 128 + ((cc ^ (k0 & 7)) << 4),
                          kpe_c + (s0 >= 0 ? static_cast<size_t>(s0) : 0) * (D_KPE * 2) + cc * 16, s0 >= 0 ? 16u : 0u);
      if (k1 < n_pad)
        cp_async_16_zfill(dst_base + (NUM_BLOCKS - 1) * KV_BLOCK_BYTES + k1 * 128 + ((cc ^ (k1 & 7)) << 4),
                          kpe_c + (s1 >= 0 ? static_cast<size_t>(s1) : 0) * (D_KPE * 2) + cc * 16, s1 >= 0 ? 16u : 0u);
      cp_async_commit();
    }
    if (kg_cfg == 1) {  // block `warp`, rows 0..15 (4 instructions; n_pad == 16)
      RowSrc r[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) r[j] = row_src(4 * j + rg, warp);
#pragma unroll
      for (int j = 0; j < 4; ++j) cp_async_16_zfill(r[j].s, r[j].g, r[j].sz);
      cp_async_commit();
    } else if (kg_cfg == 2) {  // rows 32kg..+32 of blocks 2dg, 2dg+1 (8 instructions per block)
      const int kg = warp >> 2, dg = warp & 3;  // kg 0 = warps 0-3: one warp per SM sub-partition
      RowSrc r[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) r[j] = row_src(32 * kg + 4 * j + rg, 2 * dg);
#pragma unroll
      for (int bb = 0; bb < 2; ++bb) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (32 * kg + 4 * j + rg < n_pad) cp_async_16_zfill(r[j].s + bb * KV_BLOCK_BYTES, r[j].g + bb * 128, r[j].sz);
        cp_async_commit();
      }
    } else if (kg_cfg == 8) {  // rows 8w..8w+7 of blocks 0..7 (2 instructions per block)
      RowSrc r[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) r[j] = row_src(8 * warp + 4 * j + rg, 0);
#pragma unroll
      for (int bb = 0; bb < 8; ++bb) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (8 * warp + 4 * j + rg < n_pad) cp_async_16_zfill(r[j].s + bb * KV_BLOCK_BYTES, r[j].g + bb * 128, r[j].sz);
        cp_async_commit();
      }
    } else {  // rows 16kg..+16 of blocks 4dg..+4 (4 instructions per block)
      const int kg = warp & 3, dg = warp >> 2;
      RowSrc r[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) r[j] = row_src(16 * kg + 4 * j + rg, 4 * dg);
#if DSA_NTMAJOR
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // rows of n-tile 0 (j = 0, 1), then n-tile 1 (j = 2, 3)
#pragma unroll
        for (int bb = 0; bb < 4; ++bb)
          if (16 * kg + 4 * j + rg < n_pad) cp_async_16_zfill(r[j].s + bb * KV_BLOCK_BYTES, r[j].g + bb * 128, r[j].sz);
        if (j == 1 || j == 3) cp_async_commit();
      }
#elif DSA_ROWMAJOR
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int bb = 0; bb < 4; ++bb)
          if (16 * kg + 4 * j + rg < n_pad) cp_async_16_zfill(r[j].s + bb * KV_BLOCK_BYTES, r[j].g + bb * 128, r[j].sz);
      }
      cp_async_commit();
#else
#pragma unroll
      for (int bb = 0; bb < 4; ++bb) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
          if (16 * kg + 4 * j + rg < n_pad) cp_async_16_zfill(r[j].s + bb * KV_BLOCK_BYTES, r[j].g + bb * 128, r[j].sz);
        cp_async_commit();
      }
#endif
    }
  }
#else
  {
    const uint32_t dst_base = smem_u32(smem + SMEM_KV_OFF);
    const int n_pad = n_ksteps * 16;  // rows up to the next multiple of 16 are loaded or zero-filled
    const int kq = tid >> 3;          // key_a = kq, key_b = 32 + kq
    const int key_a = kq, key_b = 32 + kq;
    const int cib = tid & 7;
    const int idx_a = __shfl_sync(0xffffffffu, my_lo, kq);
    const int idx_b = __shfl_sync(0xffffffffu, my_hi, kq);
    const bool va = idx_a >= 0, vb = idx_b >= 0;  // a valid key is always < n_rows <= n_pad
    const size_t row_a = va ? static_cast<size_t>(idx_a) : 0, row_b = vb ? static_cast<size_t>(idx_b) : 0;
#pragma unroll
    for (int bb = 0; bb < NUM_BLOCKS; ++bb) {
      const int b = (bb == 0) ? (NUM_BLOCKS - 1) : (bb - 1);  // kpe block first, then blocks 0..7
      const int chunk = b * 8 + cib;  // global chunk index 0..71
      const char* src_a;
      const char* src_b;
      if (chunk < CKV_CHUNKS) {
        src_a = reinterpret_cast<const char*>(p.ckv) + row_a * (D_CKV * 2) + chunk * 16;
        src_b = reinterpret_cast<const char*>(p.ckv) + row_b * (D_CKV * 2) + chunk * 16;
      } else {
        src_a = reinterpret_cast<const char*>(p.kpe) + row_a * (D_KPE * 2) + (chunk - CKV_CHUNKS) * 16;
        src_b = reinterpret_cast<const char*>(p.kpe) + row_b * (D_KPE * 2) + (chunk - CKV_CHUNKS) * 16;
      }
      if (key_a < n_pad) cp_async_16_zfill(dst_base + kv_off(key_a, chunk), src_a, va ? 16u : 0u);
      if (key_b < n_pad) cp_async_16_zfill(dst_base + kv_off(key_b, chunk), src_b, vb ? 16u : 0u);
#ifndef DSA_DIAG_NOARRIVE
      cp_async_mbar_arrive(gbar0 + 8 * b);  // fires when this thread's copies so far have landed
#endif
    }
  }
  cp_async_commit();
#endif
  TRACE_STAMP(6);  // gather issued
  // ---- 3a. activity of all 32 CTAs of the token from the full index row (identical everywhere) --
  {
    const bool any_valid = (ia.x >= 0) | (ia.y >= 0) | (ia.z >= 0) | (ia.w >= 0) |
                           (ib.x >= 0) | (ib.y >= 0) | (ib.z >= 0) | (ib.w >= 0);
    const unsigned bal = __ballot_sync(0xffffffffu, any_valid);
    if (lane == 0) s_bits[warp] = bal;
    // my 64 slots live in warp cit/4, lanes 8*(cit%4) .. +8 (8 slots each): keep them for the softmax mask
    const bool my_group = (warp == (cit >> 2)) && ((lane >> 3) == (cit & 3));
    if (my_group) {
      int4* dst = reinterpret_cast<int4*>(s_idx + (lane & 7) * 8);
      dst[0] = ia;
      dst[1] = ib;
    }
  }
  __syncthreads();
  // Cluster barrier phase 1 (arrive only): makes the mbarrier inits visible cluster-wide; the wait
  // happens right before the level-1 push, long after every rank has arrived (inactive ranks exit
  // after arriving and count as arrived in later phases).
  cluster_arrive_relaxed();
  // Activity of all 32 CTAs of the token (identical in every CTA of the token).
  unsigned cta_act = 0u;  // bit c = CTA c of the token has a valid slot
#pragma unroll
  for (int w = 0; w < NUM_WARPS; ++w) {
    const unsigned b = s_bits[w];
#pragma unroll
    for (int j = 0; j < 4; ++j) cta_act |= (((b >> (8 * j)) & 0xffu) ? 1u : 0u) << (4 * w + j);
  }
  const unsigned active_mask = (cta_act >> (seg * CLUSTER_SIZE)) & 0xffu;  // my cluster
  unsigned seg_mask = 0u;
#pragma unroll
  for (int s = 0; s < NUM_SEG; ++s) seg_mask |= (((cta_act >> (s * CLUSTER_SIZE)) & 0xffu) ? 1u : 0u) << s;
  const int n_active = __popc(active_mask);
  const int n_seg = __popc(seg_mask);
  const bool i_am_active = (cta_act >> cit) & 1u;
  // Level-2 merger segment: the active segment with the most active ranks (ties -> highest index),
  // so its owners hold as few slices as possible and it is launched no earlier than the others.
  int mseg = 0;
  {
    int mbest = 0;
#pragma unroll
    for (int s = 0; s < NUM_SEG; ++s) {
      const int n = __popc((cta_act >> (s * CLUSTER_SIZE)) & 0xffu);
      if (n > 0 && n >= mbest) {
        mbest = n;
        mseg = s;
      }
    }
  }
  const bool i_merge = (seg == mseg);  // my segment does the level-2 merge (if n_seg >= 2)
  TRACE_STAMP(1);

  if (!i_am_active) {
    cp_async_wait<0>();
#if DSA_QTMA
    mbar_wait_parity(qbar, 0);
#endif
    if (n_seg == 0 && cit == 0) {
      // Token with no valid key: reference gives zeros and lse = -inf.
      for (int i = tid; i < NUM_HEADS * D_CKV / 8; i += NUM_THREADS)
        reinterpret_cast<uint4*>(p.out + static_cast<size_t>(token) * NUM_HEADS * D_CKV)[i] =
            make_uint4(0u, 0u, 0u, 0u);
      if (tid < NUM_HEADS) p.lse[token * NUM_HEADS + tid] = -INFINITY;
    }
    TRACE_STAMP(15);
    TRACE_FLUSH();
    return;
  }

  // ---- 3b. merge topology, receive slots, expected bytes and remote addresses (overlaps the
  //          DRAM latency of the gather). Only meaningful when n_active >= 2. ---------------------
  const int r0 = lane >> 2;          // fragment rows r0 / r0+8
  const int c0 = (lane & 3) * 2;     // fragment column pair
  const int q = lane & 3;            // quad lane: owns dims 16q..16q+15 of a slice after transpose
  const int nn = n_active > 0 ? n_active : 1;
  const int my_pos = __popc(active_mask & ((1u << rank) - 1u));
  unsigned act_list = 0u;            // active ranks, 4 bits per position
  {
    int n = 0;
#pragma unroll
    for (int r = 0; r < CLUSTER_SIZE; ++r) {
      if ((active_mask >> r) & 1u) {
        act_list |= static_cast<unsigned>(r) << (4 * n);
        ++n;
      }
    }
  }
  unsigned owners = 0u;              // owner rank of each slice, 4 bits per slice
  unsigned slot_bases = 0u;          // (slice / n) * n of each slice, 4 bits per slice
  unsigned owned = 0u;               // bit i: I own slice i
  int n_owned = 0;
  {
    int pos = 0, base = 0;
#pragma unroll
    for (int i = 0; i < NUM_SLICES; ++i) {
      const int o = (act_list >> (4 * pos)) & 15;
      owners |= static_cast<unsigned>(o) << (4 * i);
      slot_bases |= static_cast<unsigned>(base) << (4 * i);
      if (o == rank) {
        owned |= 1u << i;
        ++n_owned;
      }
      ++pos;
      if (pos == nn) {
        pos = 0;
        base += nn;
      }
    }
  }
#if DSA_WS_PREFETCH
  // Level-2 merger: pull the workspace lines this CTA will poll (the other segments' records of its
  // owned slices) into L2 now, during the gather's DRAM latency. The benchmark flushes L2 before every
  // call, so otherwise the publishers' stores and the merger's polls meet on lines that are being
  // fetched from DRAM (a poll = a DRAM round trip, 1.0-1.5 us, instead of an L2 hit).
  if (i_merge && n_seg >= 2 && warp == 0) {
    const unsigned pending = seg_mask & ~(1u << seg);
    const int s = lane >> 3, k = lane & 7;  // lane -> (segment s, owned slice k); lanes 0..31 cover 4 x 8
    const size_t tok_base_pf = static_cast<size_t>(token) * NUM_SEG;
    if (((pending >> s) & 1u) && k < n_owned) {
      const int i = (my_pos + k * nn) & 7;
      prefetch_l2_bulk(p.ws_o + (((tok_base_pf + s) * NUM_SLICES + i) * NUM_HEADS) * WS_O_STRIDE, NUM_HEADS * WS_O_STRIDE * 16);
    }
    if (((pending >> s) & 1u) && k == 0)
      prefetch_l2_bulk(p.ws_ml + (tok_base_pf + s) * NUM_HEADS, NUM_HEADS * 16);
  }
#endif
  const uint32_t mbar_local = smem_u32(s_mbar);
  float* s_recv_ml = reinterpret_cast<float*>(smem + SMEM_RECV_ML_OFF);  // [8 src][16][2]
  const uint32_t mbar_ml_local = smem_u32(s_mbar_ml);
  if (n_active >= 2 && tid == 0) {
    // Owner side: every active rank (including myself) bulk-copies my owned slices; every other
    // active rank sends its (m, l) pairs right after its softmax (st.async, 8 B per head).
    mbar_arrive_expect_tx(mbar_local, static_cast<uint32_t>((n_active - DSA_LOCAL_SLICE) * n_owned * SLICE_BYTES));
    mbar_arrive_expect_tx(mbar_ml_local, static_cast<uint32_t>((n_active - 1) * NUM_HEADS * 8));
  }

  // ---- 4. QK^T partials. The 8 warps split the (n-tiles x ckv dim-blocks) work as KG key groups x
  //         DG dim groups (KG * DG = 8): warp w = (kg = w % KG, dg = w / KG) computes S over the
  //         n-tiles [kg * 8/KG, +8/KG) and the dim-blocks [dg * KG, +KG), each block as soon as it
  //         lands, into partial slot dg. Big tiles use KG = DSA_KG_BIG so that only NT = 8/KG n-tiles
  //         x 4 k-steps remain after the last block lands (and few partials); small tiles keep the
  //         dim-split (KG = 1) so their k-steps spread over all warps. The kpe block (lands first) is
  //         computed per n-tile = warp into slot 8 (folded into the warp's own accumulator for KG = 8).
  const int n_part = 8 / kg_cfg;  // ckv partial slots 0 .. n_part-1 (+ slot 8 unless folded)
  float s_frag[8];  // fused paths: S[r0][c0, c0+1], S[r0+8][c0, c0+1] per owned n-tile (1 or 2)
  {
    const uint32_t kv_base = smem_u32(smem + SMEM_KV_OFF);
    const uint32_t q_base = smem_u32(smem + SMEM_Q_OFF);
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    // kpe block (issued first): keys 8*warp .. +8 over 4 k-steps (two ldmatrix.x4, two k-steps each)
    float accpe[4] = {0.f, 0.f, 0.f, 0.f};
    s_frag[0] = s_frag[1] = s_frag[2] = s_frag[3] = 0.f;
#if DSA_CWAIT_EARLY
    if (n_active >= 2) cluster_wait_acquire();
#endif
#if DSA_WARP_GATHER
    if (DSA_ROWMAJOR && kg_cfg == 4) cp_async_wait<0>();
    else cp_async_wait_rt((DSA_NTMAJOR && kg_cfg == 4) ? 2 : kg_cfg);  // my kpe rows (and the Q tile) have landed
    __syncwarp();
#if DSA_QTMA
    mbar_wait_parity(qbar, 0);  // the Q tile (bulk copies) has landed
#endif
#elif defined(DSA_DIAG_NOARRIVE)
    cp_async_wait<0>();
    __syncthreads();
#else
    mbar_wait_parity(gbar0 + 8 * (NUM_BLOCKS - 1), 0);
#endif
    TRACE_STAMP(9);
    auto qk_kpe = [&](int nt, float* acc) {  // kpe partial of n-tile nt (keys 8nt .. 8nt+7), 4 k-steps
      const uint32_t kvb = kv_base + (NUM_BLOCKS - 1) * KV_BLOCK_BYTES;
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        uint32_t b[4];  // lanes 0-7: chunk 4jp, 8-15: +1, 16-23: +2, 24-31: +3
        ldmatrix_x4(kvb + blk_off(nt * 8 + (lane & 7), 4 * jp + (lane >> 3)), b[0], b[1], b[2], b[3]);
        uint32_t a[4];
        ldmatrix_x4(q_addr(q_base, NUM_BLOCKS - 1, a_row, 4 * jp + (lane >> 4)), a[0], a[1], a[2], a[3]);
        mma_bf16_16816(acc, a, b);
        ldmatrix_x4(q_addr(q_base, NUM_BLOCKS - 1, a_row, 4 * jp + 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
        mma_bf16_16816(acc, a, b + 2);
      }
    };
#if DSA_FUSED4
    if (kg_cfg == 4) {
      // Config 4 fused: warp (kg = w & 3, dg = w >> 2). dg 0 computes the kpe partials of n-tiles 2kg, 2kg+1
      // and blocks 0-3, dg 1 blocks 4-7; dg 1 hands its fragments to dg 0 through smem (named barrier
      // per key group), so dg 0 holds the complete S of keys 16kg .. +16 for the fused softmax.
      const int kg = warp & 3, dg = warp >> 2;
      const int nv = n_ntiles - kg * 2;
      float accpe1[4] = {0.f, 0.f, 0.f, 0.f};
      if (dg == 0 && nv > 0) {
        qk_kpe(2 * kg, accpe);
        if (nv > 1) qk_kpe(2 * kg + 1, accpe1);
      }
      s_frag[4] = s_frag[5] = s_frag[6] = s_frag[7] = 0.f;
      if (nv > 0) qk_ckv<2>(kv_base, q_base, gbar0, lane, kg * 2, nv < 2 ? nv : 2, dg * 4, accpe, nullptr, s_frag, accpe1);
      float* xbuf = s_S + kg * (32 * 8);  // [4 kg][32 lanes][8 floats]
      if (dg == 1) {
        *reinterpret_cast<float4*>(xbuf + lane * 8) = make_float4(s_frag[0], s_frag[1], s_frag[2], s_frag[3]);
        *reinterpret_cast<float4*>(xbuf + lane * 8 + 4) = make_float4(s_frag[4], s_frag[5], s_frag[6], s_frag[7]);
      }
      asm volatile("bar.sync %0, 64;\n" ::"r"(1 + kg) : "memory");
      if (dg == 0) {
        const float4 u0 = *reinterpret_cast<const float4*>(xbuf + lane * 8);
        const float4 u1 = *reinterpret_cast<const float4*>(xbuf + lane * 8 + 4);
        s_frag[0] += u0.x; s_frag[1] += u0.y; s_frag[2] += u0.z; s_frag[3] += u0.w;
        s_frag[4] += u1.x; s_frag[5] += u1.y; s_frag[6] += u1.z; s_frag[7] += u1.w;
      }
    } else
#endif
    if (warp < n_ntiles) qk_kpe(warp, accpe);  // keys 8w..8w+7 hold a valid key (others: masked, S garbage)
    const int r0 = lane >> 2;
    const int c0 = (lane & 3) * 2;
    if (kg_cfg == 8) {
      // KG = 8: warp w = n-tile w over all 8 ckv blocks, kpe folded in: the S fragment stays in registers
      if (warp < n_ntiles) qk_ckv<1>(kv_base, q_base, gbar0, lane, warp, 1, 0, accpe, nullptr, s_frag);
    } else if (!(DSA_FUSED4 && kg_cfg == 4)) {  // (config 4 handled above when fused)
      float* sPe = s_S + (NUM_BLOCKS - 1) * (NUM_HEADS * S_STRIDE);
      *reinterpret_cast<float2*>(sPe + r0 * S_STRIDE + warp * 8 + c0) = make_float2(accpe[0], accpe[1]);
      *reinterpret_cast<float2*>(sPe + (r0 + 8) * S_STRIDE + warp * 8 + c0) = make_float2(accpe[2], accpe[3]);
      const float zero4[4] = {0.f, 0.f, 0.f, 0.f};
      if (kg_cfg == 1) {
        qk_ckv<8>(kv_base, q_base, gbar0, lane, 0, n_ntiles, warp, zero4, s_S + warp * (NUM_HEADS * S_STRIDE));
      } else if (kg_cfg == 2) {
        const int kg = warp >> 2, dg = warp & 3;  // (matches the gather ownership)
        const int nv = n_ntiles - kg * 4;
        if (nv > 0) qk_ckv<4>(kv_base, q_base, gbar0, lane, kg * 4, nv < 4 ? nv : 4, dg * 2, zero4, s_S + dg * (NUM_HEADS * S_STRIDE));
      } else {  // 4
        const int kg = warp & 3, dg = warp >> 2;
        const int nv = n_ntiles - kg * 2;
#if DSA_NTMAJOR
        if (nv > 0) qk_ckv_nt(kv_base, q_base, lane, kg * 2, nv < 2 ? nv : 2, dg * 4, s_S + dg * (NUM_HEADS * S_STRIDE));
#else
        if (nv > 0) qk_ckv<2, DSA_QREG != 0>(kv_base, q_base, gbar0, lane, kg * 2, nv < 2 ? nv : 2, dg * 4, zero4, s_S + dg * (NUM_HEADS * S_STRIDE));
#endif
      }
    }
  }
  float m_run = 0.f, l_run = 0.f;  // per-head (head tid/16) softmax statistics: only needed by the merge paths
  float fr_M[2], fr_L[2], fr_inv[2];  // fragment-layout statistics of rows r0 / r0 + 8 (every warp)
  const bool solo = (n_active == 1 && n_seg == 1);  // CTA-uniform: single-CTA token (no merge, no exchange)
  // ---- 5f. fused softmax on register fragments (CTA-uniform paths): warps 0-3 contribute NTW n-tiles
  //          each (keys keybase + 8nt + c0, +1): per-head max / sum within the quad, across the four
  //          warps through smem (two CTA barriers; the first one also completes the gather for the
  //          PV). Every warp then holds M / L / 1/L of rows r0, r0 + 8 in registers (the PV epilogue
  //          layout); the head-group layout (m_run, l_run) is produced only for merge paths (third
  //          barrier). Contributing warps write their P columns (zeros for masked keys).
  auto fused_softmax = [&](auto ntw_c, int keybase) {
    constexpr int NTW = decltype(ntw_c)::value;
    const bool contrib = warp < 4;
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;
    const int h = tid >> 4;
    auto valid = [&](int key) -> bool { return ((key < 32 ? (m_lo >> key) : (m_hi >> (key - 32))) & 1u) != 0u; };
    float sc[NTW][4];
    float mA = -INFINITY, mB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
      const int key0 = keybase + nt * 8 + c0;
      const bool v0 = contrib && valid(key0), v1 = contrib && valid(key0 + 1);
      sc[nt][0] = v0 ? s_frag[nt * 4 + 0] * p.scale_log2 : -INFINITY;
      sc[nt][1] = v1 ? s_frag[nt * 4 + 1] * p.scale_log2 : -INFINITY;
      sc[nt][2] = v0 ? s_frag[nt * 4 + 2] * p.scale_log2 : -INFINITY;
      sc[nt][3] = v1 ? s_frag[nt * 4 + 3] * p.scale_log2 : -INFINITY;
      mA = fmaxf(mA, fmaxf(sc[nt][0], sc[nt][1]));
      mB = fmaxf(mB, fmaxf(sc[nt][2], sc[nt][3]));
    }
    mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 1));
    mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 2));
    mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 1));
    mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 2));
    float* s_mx = s_S + 1024;  // [4 warps][16 heads] (after the config-4 exchange buffer)
    float* s_ls = s_S + 1088;  // [4 warps][16 heads]
    if (contrib && (lane & 3) == 0) {
      s_mx[warp * 16 + r0] = mA;
      s_mx[warp * 16 + r0 + 8] = mB;
    }
    cp_async_wait<0>();
    __syncthreads();
#ifdef DSA_DIAG_STAMPS
    TRACE_STAMP(4);  // diag: fused barrier 1 passed
#endif
    float MA = -INFINITY, MB = -INFINITY;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      MA = fmaxf(MA, s_mx[w * 16 + r0]);
      MB = fmaxf(MB, s_mx[w * 16 + r0 + 8]);
    }
    float lA = 0.f, lB = 0.f;
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt) {
      const float p0 = fast_exp2(sc[nt][0] - MA), p1 = fast_exp2(sc[nt][1] - MA);
      const float p2 = fast_exp2(sc[nt][2] - MB), p3 = fast_exp2(sc[nt][3] - MB);
      lA += p0 + p1;
      lB += p2 + p3;
      if (contrib) {
        const int key0 = keybase + nt * 8 + c0;
        *reinterpret_cast<uint32_t*>(s_P + r0 * P_STRIDE + key0) = pack_bf16x2(p0, p1);
        *reinterpret_cast<uint32_t*>(s_P + (r0 + 8) * P_STRIDE + key0) = pack_bf16x2(p2, p3);
      }
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    if (contrib && (lane & 3) == 0) {
      s_ls[warp * 16 + r0] = lA;
      s_ls[warp * 16 + r0 + 8] = lB;
    }
    __syncthreads();
#ifdef DSA_DIAG_STAMPS
    TRACE_STAMP(8);  // diag: fused barrier 2 passed
#endif
    float LA = 0.f, LB = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      LA += s_ls[w * 16 + r0];
      LB += s_ls[w * 16 + r0 + 8];
    }
    fr_M[0] = MA; fr_M[1] = MB;
    fr_L[0] = LA; fr_L[1] = LB;
    fr_inv[0] = __fdividef(1.f, LA); fr_inv[1] = __fdividef(1.f, LB);
    if (!solo) {  // merge paths need the head-group layout
      if (warp == 0 && (lane & 3) == 0) {
        s_m[r0] = MA; s_m[r0 + 8] = MB;
        s_l[r0] = LA; s_l[r0 + 8] = LB;
      }
      __syncthreads();
      m_run = s_m[h];
      l_run = s_l[h];
    }
  };
  const bool fused_path = (kg_cfg == 8) || (DSA_FUSED4 && kg_cfg == 4);  // CTA-uniform
  if (kg_cfg == 8) {
    TRACE_STAMP(2);
    fused_softmax(IntC<1>{}, warp * 8);
#if DSA_FUSED4
  } else if (kg_cfg == 4) {
    TRACE_STAMP(2);
    fused_softmax(IntC<2>{}, (warp & 3) * 16);
#endif
  } else {
  cp_async_wait<0>();
  __syncthreads();
  TRACE_STAMP(2);

  // ---- 5. softmax: thread -> head tid/16, keys 4*(tid%16) .. +4 ---------------------------------
  {
    const int h = tid >> 4;
    const int k0 = (tid & 15) * 4;
    float4 sv = *reinterpret_cast<const float4*>(s_S + h * S_STRIDE + k0);
    if (kg_cfg != 8) {
#if DSA_SUM_UNROLL
      // All partial slots of the configuration (n_part ckv partials + the kpe slot) loaded at once, so
      // the smem loads pipeline instead of forming a chain of dependent round trips (unroll 1 loop:
      // ~40 cycles per slot).
      auto sum_parts = [&](auto np_c) {
        constexpr int NP = decltype(np_c)::value;
        float4 v[NP];
#pragma unroll
        for (int sl = 1; sl < NP; ++sl) v[sl] = *reinterpret_cast<const float4*>(s_S + sl * (NUM_HEADS * S_STRIDE) + h * S_STRIDE + k0);
        v[0] = *reinterpret_cast<const float4*>(s_S + (NUM_BLOCKS - 1) * (NUM_HEADS * S_STRIDE) + h * S_STRIDE + k0);
#pragma unroll
        for (int sl = 0; sl < NP; ++sl) {
          sv.x += v[sl].x; sv.y += v[sl].y; sv.z += v[sl].z; sv.w += v[sl].w;
        }
      };
      if (kg_cfg == 1) sum_parts(IntC<8>{});
      else if (kg_cfg == 2) sum_parts(IntC<4>{});
      else sum_parts(IntC<2>{});
#else
      const float4 t = *reinterpret_cast<const float4*>(s_S + (NUM_BLOCKS - 1) * (NUM_HEADS * S_STRIDE) + h * S_STRIDE + k0);
      sv.x += t.x; sv.y += t.y; sv.z += t.z; sv.w += t.w;
#pragma unroll 1
      for (int sl = 1; sl < n_part; ++sl) {
        const float4 u = *reinterpret_cast<const float4*>(s_S + sl * (NUM_HEADS * S_STRIDE) + h * S_STRIDE + k0);
        sv.x += u.x; sv.y += u.y; sv.z += u.z; sv.w += u.w;
      }
#endif
    }
    const int4 iv = *reinterpret_cast<const int4*>(s_idx + k0);
    float s[4] = {iv.x >= 0 ? sv.x * p.scale_log2 : -INFINITY, iv.y >= 0 ? sv.y * p.scale_log2 : -INFINITY,
                  iv.z >= 0 ? sv.z * p.scale_log2 : -INFINITY, iv.w >= 0 ? sv.w * p.scale_log2 : -INFINITY};
    float mx = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
#if DSA_REDUX
    mx = redux_max_f32(mx, 0xffffu << (lane & 16));  // one instruction over the 16 lanes of the head
#else
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
#endif
    // the tile has >= 1 valid key (CTA is active) -> mx is finite
    float pv[4];
    float ps = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      pv[j] = fast_exp2(s[j] - mx);
      ps += pv[j];
    }
    ps += __shfl_xor_sync(0xffffffffu, ps, 8);
    ps += __shfl_xor_sync(0xffffffffu, ps, 4);
    ps += __shfl_xor_sync(0xffffffffu, ps, 2);
    ps += __shfl_xor_sync(0xffffffffu, ps, 1);
    m_run = mx;
    l_run = ps;
    *reinterpret_cast<uint2*>(s_P + h * P_STRIDE + k0) =
        make_uint2(pack_bf16x2(pv[0], pv[1]), pack_bf16x2(pv[2], pv[3]));
    if ((tid & 15) == 0) {
      s_m[h] = mx;
      s_l[h] = ps;
      s_scale[h] = __fdividef(1.f, ps);
    }
  }
  }
  {
    const int h = tid >> 4;
    const float mx = m_run, ps = l_run;
    if (n_active >= 2) {
#if !DSA_CWAIT_EARLY
      // Phase 1 of the cluster barrier (arrived in the prologue): every rank's mbarriers are
      // initialized, so (m, l) can be sent now and the merge weights computed during the PV.
      TRACE_STAMP(4);
      cluster_wait_acquire();
      TRACE_STAMP(8);
#endif
      {
        // every thread of the head group holds (mx, ps): lane j sends to the j-th active rank
        const int j = tid & 15;
        if (j < n_active) {
          const int r = (act_list >> (4 * j)) & 15;
          if (r != rank)
            st_async_v2f32(mapa_shared(smem_u32(s_recv_ml + (rank * NUM_HEADS + h) * 2), r), mx, ps,
                           mapa_shared(mbar_ml_local, r));
        }
      }
    }
  }
  if (!fused_path) {
    __syncthreads();
    fr_M[0] = s_m[r0]; fr_M[1] = s_m[r0 + 8];
    fr_L[0] = s_l[r0]; fr_L[1] = s_l[r0 + 8];
    fr_inv[0] = s_scale[r0]; fr_inv[1] = s_scale[r0 + 8];
  }
  asm volatile("" ::"r"(warm));  // sink of the TLB warm-up loads (long complete by now)
  // lse of a single-CTA token, written from the fragment layout by the quad leaders of warp 0
  auto write_lse_frag = [&]() {
    if (warp == 0 && (lane & 3) == 0) {
      p.lse[token * NUM_HEADS + r0] = fr_M[0] + fast_log2(fr_L[0]);
      p.lse[token * NUM_HEADS + r0 + 8] = fr_M[1] + fast_log2(fr_L[1]);
    }
  };

  TRACE_STAMP(10);  // softmax done (P in smem)
  // ---- 6. PV: O[16][64w..64w+64) = P[16][64] * V[64][64w..] -----------------------------------
  float o_acc[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) o_acc[i][j] = 0.f;
  }
  // Level-1 push of part `pt` of my slice (slice j = warp, dims 64j + pt * 64 / PV_SPLIT ..): stage it as
  // fp16 with stmatrix (own slice: straight into my receive slot; others: the dead Q region) and push
  // it to the owner's receive slot (slice/n)*n + my_pos: slices < 8 - SPLIT as a bulk copy (lane 0),
  // the rest as 16-byte st.async stores from the 32 lanes. Called right after the part is computed, so
  // the transfer of part 0 overlaps the PV of part 1.
  __half* s_recv_o = reinterpret_cast<__half*>(smem + SMEM_RECV_O_OFF);   // [RECV_SLOTS][2048 B] swizzled
  auto push_part = [&](auto pt_c) {
    constexpr int PT = decltype(pt_c)::value;
    static_assert(NUM_SLICES * SLICE_BYTES <= Q_BYTES, "exchange staging must fit in the Q region");
    const int j = warp;
    const int owner = (owners >> (4 * j)) & 15;
    const int slot = ((slot_bases >> (4 * j)) & 15) + my_pos;
#if DSA_LOCAL_SLICE
    const bool local = (owner == rank);
    const uint32_t st_base = local ? smem_u32(s_recv_o) + slot * SLICE_BYTES : smem_u32(smem + SMEM_Q_OFF) + warp * SLICE_BYTES;
#else
    const uint32_t st_base = smem_u32(smem + SMEM_Q_OFF) + warp * SLICE_BYTES;
#endif
    const int st_row = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = PT * (4 / PV_SPLIT); k < (PT + 1) * (4 / PV_SPLIT); ++k) {  // n-tiles 2k, 2k+1 -> chunks 2k, 2k+1
      stmatrix_x4(st_base + slice_off(st_row, 2 * k + (lane >> 4)),
                  pack_half2(o_acc[2 * k][0], o_acc[2 * k][1]), pack_half2(o_acc[2 * k][2], o_acc[2 * k][3]),
                  pack_half2(o_acc[2 * k + 1][0], o_acc[2 * k + 1][1]), pack_half2(o_acc[2 * k + 1][2], o_acc[2 * k + 1][3]));
    }
    const uint32_t rdst = mapa_shared(smem_u32(s_recv_o) + slot * SLICE_BYTES + PT * PART_BYTES, owner);
    const uint32_t rbar = mapa_shared(mbar_local, owner);
#if DSA_LOCAL_SLICE
    if (local) {
    } else
#endif
    if (j < NUM_SLICES - DSA_PUSH_SPLIT) {
      fence_proxy_async_smem();  // this warp's staging writes -> visible to the bulk-copy engine
      __syncwarp();
      if (lane == 0) bulk_copy_s2s(rdst, st_base + PT * PART_BYTES, PART_BYTES, rbar);
    } else {
      __syncwarp();
#pragma unroll
      for (int k = 0; k < PART_BYTES / 16 / 32; ++k) {
        const int off = (k * 32 + lane) * 16;
        const uint4 v = *reinterpret_cast<const uint4*>(smem + SMEM_Q_OFF + warp * SLICE_BYTES + PT * PART_BYTES + off);
        st_async_v4b32(rdst + off, v, rbar);
      }
    }
  };
  const bool multi_rank = (n_active >= 2);  // CTA-uniform
  {
    const uint32_t kv_base = smem_u32(smem + SMEM_KV_OFF);
    const uint32_t p_base = smem_u32(s_P);
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_col = (lane >> 4) * 8;
    const int b_key_lo = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int b_chunk_lo = lane >> 4;
    auto pv_part = [&](auto pt_c) {
      constexpr int PT = decltype(pt_c)::value;
#pragma unroll
      for (int kstep = 0; kstep < TILE_KEYS / 16; ++kstep) {
        if (kstep >= n_ksteps) break;  // rows beyond n_rows were never loaded (P is 0 there anyway)
        uint32_t a[4];
        ldmatrix_x4(p_base + (a_row * P_STRIDE + kstep * 16 + a_col) * 2, a[0], a[1], a[2], a[3]);
        const int key = kstep * 16 + b_key_lo;
#pragma unroll
        for (int nt2 = PT * (4 / PV_SPLIT); nt2 < (PT + 1) * (4 / PV_SPLIT); ++nt2) {
          uint32_t b[4];
          const int chunk = warp * 8 + nt2 * 2 + b_chunk_lo;
          ldmatrix_x4_trans(kv_base + kv_off(key, chunk), b[0], b[1], b[2], b[3]);
          mma_bf16_16816(o_acc[nt2 * 2], a, b);
          mma_bf16_16816(o_acc[nt2 * 2 + 1], a, b + 2);
        }
      }
      if (multi_rank) push_part(pt_c);
    };
    pv_part(IntC<0>{});
    if constexpr (PV_SPLIT >= 2) pv_part(IntC<1>{});
    if constexpr (PV_SPLIT >= 4) {
      pv_part(IntC<2>{});
      pv_part(IntC<3>{});
    }
  }
#ifdef DSA_TRACE
  __syncthreads();
#endif
  TRACE_STAMP(3);


  // ---- 7a. single active CTA in the whole token: normalize, stage bf16 [16][512] in the dead Q
  //          region (exact global layout) and write it with one 16 KB bulk copy ----------------
  if (n_active == 1 && n_seg == 1) {
    const float inv_lo = fr_inv[0];
    const float inv_hi = fr_inv[1];
    __nv_bfloat16* out_t = p.out + static_cast<size_t>(token) * NUM_HEADS * D_CKV;
#if DSA_EPI_TMA == 3
    // Per-warp: stmatrix the 16 x 64 bf16 slice into a padded staging area in the dead Q region
    // (no CTA barrier), read it back as 16-byte chunks and store 4 full 128-byte rows per warp
    // instruction.
    {
      static_assert(NUM_WARPS * EPI_SLICE_BYTES <= Q_BYTES, "per-warp staging must fit in the Q region");
      const uint32_t st_base = smem_u32(smem + SMEM_Q_OFF) + warp * EPI_SLICE_BYTES;  // [16 rows][144 B]
      const uint32_t st_addr = st_base + ((lane & 7) + ((lane >> 3) & 1) * 8) * (STAGE_STRIDE * 2) + (lane >> 4) * 16;
#pragma unroll
      for (int k = 0; k < 4; ++k) {  // n-tiles 2k, 2k+1: matrices (lo, hi, lo, hi)
        stmatrix_x4(st_addr + k * 32,
                    pack_bf16x2(o_acc[2 * k][0] * inv_lo, o_acc[2 * k][1] * inv_lo),
                    pack_bf16x2(o_acc[2 * k][2] * inv_hi, o_acc[2 * k][3] * inv_hi),
                    pack_bf16x2(o_acc[2 * k + 1][0] * inv_lo, o_acc[2 * k + 1][1] * inv_lo),
                    pack_bf16x2(o_acc[2 * k + 1][2] * inv_hi, o_acc[2 * k + 1][3] * inv_hi));
      }
      __syncwarp();
      TRACE_STAMP(12);  // staged
#if DSA_LSE_FIRST
      write_lse_frag();
#endif
      const uint8_t* st_ptr = smem + SMEM_Q_OFF + warp * EPI_SLICE_BYTES;
      if (warp < NUM_WARPS - DSA_EPI_BULK_WARPS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int row = k * 4 + (lane >> 3);  // head
          const int chunk = lane & 7;           // 16-byte chunk of the 128-byte row slice
          const uint4 v = *reinterpret_cast<const uint4*>(st_ptr + row * (STAGE_STRIDE * 2) + chunk * 16);
          *reinterpret_cast<uint4*>(out_t + row * D_CKV + warp * SLICE_DIMS + chunk * 8) = v;
        }
      } else {
        // Bulk-copy path (TMA engine, in parallel with the st.global path of the other warps): one
        // 128-byte row of the slice per lane 0..15, completion waited only for the smem reads.
        fence_proxy_async_smem();  // this warp's staging writes -> visible to the bulk-copy engine
        __syncwarp();
        if (lane < NUM_HEADS) {
          bulk_copy_s2g(out_t + lane * D_CKV + warp * SLICE_DIMS, st_base + lane * (STAGE_STRIDE * 2), 128);
          bulk_commit_group();
        }
      }
      TRACE_STAMP(13);  // output stores issued
#if !DSA_LSE_FIRST
      write_lse_frag();
#endif
      if (warp >= NUM_WARPS - DSA_EPI_BULK_WARPS && lane < NUM_HEADS) bulk_wait_group_read0();
    }
#elif DSA_EPI_TMA
    static_assert(NUM_HEADS * D_CKV * 2 <= Q_BYTES, "output staging must fit in the Q region");
    __nv_bfloat16* stage_o = reinterpret_cast<__nv_bfloat16*>(smem + SMEM_Q_OFF);  // [16][512] bf16
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = warp * SLICE_DIMS + i * 8 + c0;
      *reinterpret_cast<uint32_t*>(stage_o + r0 * D_CKV + d) = pack_bf16x2(o_acc[i][0] * inv_lo, o_acc[i][1] * inv_lo);
      *reinterpret_cast<uint32_t*>(stage_o + (r0 + 8) * D_CKV + d) = pack_bf16x2(o_acc[i][2] * inv_hi, o_acc[i][3] * inv_hi);
    }
#if DSA_EPI_TMA == 1
    // one 16 KB bulk copy (measured +0.4 us on small tokens: the completion wait is on the critical path)
    fence_proxy_async_smem();
    __syncthreads();
    TRACE_STAMP(12);  // staged
    if (tid == 0) {
      bulk_copy_s2g(out_t, smem_u32(stage_o), NUM_HEADS * D_CKV * 2);
      bulk_commit_group();
    }
    write_lse_frag();
    TRACE_STAMP(13);  // bulk store issued
    if (tid == 0) bulk_wait_group_all();
#else
    // transpose through smem: every thread stores four coalesced 16-byte chunks
    __syncthreads();
    TRACE_STAMP(12);  // staged
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int c = tid + k * NUM_THREADS;  // 16-byte chunk 0..1023 of the [16][512] bf16 tile
      const uint4 v = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(stage_o) + c * 16);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out_t) + c * 16) = v;
    }
    TRACE_STAMP(13);  // output stores issued
    write_lse_frag();
#endif
#else
    // Quad transpose so every lane holds 16 contiguous dims per row and writes 16-byte vectors:
    // lane q of a quad gathers n-tiles 2q, 2q+1 (dims 16q .. 16q+15) from all four lanes.
    uint32_t lo[8], hi[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      lo[i] = pack_bf16x2(o_acc[i][0] * inv_lo, o_acc[i][1] * inv_lo);
      hi[i] = pack_bf16x2(o_acc[i][2] * inv_hi, o_acc[i][3] * inv_hi);
    }
    uint32_t alo[4], blo[4], ahi[4], bhi[4];  // [src lane q']: n-tile 2q (a) and 2q+1 (b)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int dst_q = (q - k) & 3;   // the lane that will read my value this round
      const int src_q = (q + k) & 3;   // the lane I read from
      uint32_t s_alo = lo[0], s_blo = lo[1], s_ahi = hi[0], s_bhi = hi[1];
#pragma unroll
      for (int j = 1; j < 4; ++j) {
        s_alo = (dst_q == j) ? lo[2 * j] : s_alo;
        s_blo = (dst_q == j) ? lo[2 * j + 1] : s_blo;
        s_ahi = (dst_q == j) ? hi[2 * j] : s_ahi;
        s_bhi = (dst_q == j) ? hi[2 * j + 1] : s_bhi;
      }
      const int src_lane = (lane & ~3) | src_q;
      const uint32_t r_alo = __shfl_sync(0xffffffffu, s_alo, src_lane);
      const uint32_t r_blo = __shfl_sync(0xffffffffu, s_blo, src_lane);
      const uint32_t r_ahi = __shfl_sync(0xffffffffu, s_ahi, src_lane);
      const uint32_t r_bhi = __shfl_sync(0xffffffffu, s_bhi, src_lane);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        alo[j] = (src_q == j) ? r_alo : alo[j];
        blo[j] = (src_q == j) ? r_blo : blo[j];
        ahi[j] = (src_q == j) ? r_ahi : ahi[j];
        bhi[j] = (src_q == j) ? r_bhi : bhi[j];
      }
    }
    TRACE_STAMP(12);  // transpose done
    const int d0 = warp * SLICE_DIMS + 16 * q;
    *reinterpret_cast<uint4*>(out_t + r0 * D_CKV + d0) = make_uint4(alo[0], alo[1], alo[2], alo[3]);
    *reinterpret_cast<uint4*>(out_t + r0 * D_CKV + d0 + 8) = make_uint4(blo[0], blo[1], blo[2], blo[3]);
    *reinterpret_cast<uint4*>(out_t + (r0 + 8) * D_CKV + d0) = make_uint4(ahi[0], ahi[1], ahi[2], ahi[3]);
    *reinterpret_cast<uint4*>(out_t + (r0 + 8) * D_CKV + d0 + 8) = make_uint4(bhi[0], bhi[1], bhi[2], bhi[3]);
    TRACE_STAMP(13);  // output stores issued
    write_lse_frag();
#endif
    TRACE_STAMP(15);
    TRACE_FLUSH();
    return;
  }

  // ---- 7b. merge level 1 inside the cluster (n_active >= 2) -------------------------------------
  // Slice i (dims 64i..64i+64) is owned by the (i mod n_active)-th active rank. Each warp pushes its
  // fp16 slice into the owner's receive slot (slice/n)*n + my_pos with a bulk copy; (m, l) go to every
  // active rank. The owner waits on its mbarrier for the expected bytes (no second cluster barrier),
  // then merges its owned slices one at a time (a dynamic loop: the work is proportional to the number
  // of owned slices) and consumes each merged slice at once: output store when this is the only
  // active segment, tagged publish + fp32 copy in the dead S region otherwise.
  const int h = tid >> 4;
  const int dl = (tid & 15) * 4;
  const int jw = tid & 15;  // level-2 word index of this thread (jw < WS_O_STRIDE: dims 6jw .. 6jw+5)
  float M_seg = m_run, L_seg = l_run;  // per-head merge state after level 1 (replicated per head group)
  float* s_mo = reinterpret_cast<float*>(smem + SMEM_S_OFF);  // [16 heads][512] fp32 merged slices (S partials are dead)
  const uint32_t tag = p.tag;
  const size_t tok_base = static_cast<size_t>(token) * NUM_SEG;
  __nv_bfloat16* out_t = p.out + static_cast<size_t>(token) * NUM_HEADS * D_CKV;
  // Publish slice i of this segment from its fp32 copy in s_mo (row h, dims i*64 ..): thread jw < 11
  // packs dims 6jw .. 6jw+5 (the 11th word carries 2 padding dims) with the call tag.
  // Publish slice i of this segment. DSA_WS4: thread jw (= dl / 4) publishes its own 4 dims (passed in, no
  // smem hop); otherwise thread jw < 11 packs dims 6jw .. 6jw+5 from the fp32 copy in s_mo.
#if DSA_WS4
  auto publish_o = [&](int i, float4 o) {
    st_relaxed_v2u64(p.ws_o + (((tok_base + seg) * NUM_SLICES + i) * NUM_HEADS + h) * WS_O_STRIDE + jw,
                     make_uint4(pack_half2(o.x, o.y), pack_half2(o.z, o.w), 0u, tag));
  };
  auto publish_smem = [&](int i) { publish_o(i, *reinterpret_cast<const float4*>(s_mo + h * D_CKV + i * SLICE_DIMS + dl)); };
#else
  auto publish_smem = [&](int i) {
    if (jw < WS_O_STRIDE) {
      const float* src = s_mo + h * D_CKV + i * SLICE_DIMS + 6 * jw;
      const float2 a = *reinterpret_cast<const float2*>(src);
      const float2 b = *reinterpret_cast<const float2*>(src + 2);
      const float2 c = *reinterpret_cast<const float2*>(src + 4);
      st_relaxed_v2u64(p.ws_o + (((tok_base + seg) * NUM_SLICES + i) * NUM_HEADS + h) * WS_O_STRIDE + jw,
                       make_uint4(pack_half2(a.x, a.y), pack_half2(b.x, b.y), pack_half2(c.x, c.y), tag));
    }
  };
#endif
  if (n_active >= 2) {
    // (slices were staged and pushed part by part right after each PV part)
    TRACE_STAMP(11);  // copies issued
    {
      // Merge weights from the (m, l) pairs sent after the softmax (own pair in registers), while
      // the slice copies are in flight. Everything is indexed by position among the active ranks
      // (NA = 2 / 4 / 8 positions unrolled, positions >= n_active masked), so a 2-rank token does not
      // run 8-source loops: receive slot (slice/n)*n + position holds position q's copy.
      mbar_wait_parity_cluster(mbar_ml_local, 0);
      const uint8_t* recv_b = reinterpret_cast<const uint8_t*>(s_recv_o);
      const uint32_t off0 = slice_off(h, dl >> 3) + (dl & 7) * 2;  // my (head, dims) inside a 2 KB slot
      float inv = 1.f;
      // consume one merged slice: output store (single segment), tagged publish, or fp32 copy (merger)
      auto consume = [&](int i, float4 o) {
        if (n_seg == 1) {
          *reinterpret_cast<uint2*>(out_t + h * D_CKV + i * SLICE_DIMS + dl) =
              make_uint2(pack_bf16x2(o.x * inv, o.y * inv), pack_bf16x2(o.z * inv, o.w * inv));
        } else if (!i_merge) {
#if DSA_WS4
          publish_o(i, o);
#else
          // re-map 4 -> 6 dims per thread through smem (the 16 threads of a head are half a warp)
          *reinterpret_cast<float4*>(s_mo + h * D_CKV + i * SLICE_DIMS + dl) = o;
          __syncwarp();
          publish_smem(i);
#endif
        } else {
          *reinterpret_cast<float4*>(s_mo + h * D_CKV + i * SLICE_DIMS + dl) = o;
        }
      };
      auto merge = [&](auto na_c) {
        constexpr int NA = decltype(na_c)::value;
        float2 ml[NA];
#pragma unroll
        for (int q = 0; q < NA; ++q) {
          const int r = (act_list >> (4 * q)) & 15;
          ml[q] = (q >= n_active) ? make_float2(-INFINITY, 0.f)
                : (r == rank) ? make_float2(m_run, l_run)
                              : *reinterpret_cast<const float2*>(s_recv_ml + (r * NUM_HEADS + h) * 2);
        }
        float M = -INFINITY;
#pragma unroll
        for (int q = 0; q < NA; ++q) M = fmaxf(M, ml[q].x);
        float w[NA];
        float L = 0.f;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
          w[q] = (ml[q].x > -INFINITY) ? fast_exp2(ml[q].x - M) : 0.f;
          L += w[q] * ml[q].y;
        }
        M_seg = M;
        L_seg = L;
        inv = (n_seg == 1) ? ((L > 0.f) ? __fdividef(1.f, L) : 0.f) : 1.f;
        mbar_wait_parity_cluster(mbar_local, 0);  // all slice copies into my receive buffers landed
#if DSA_LOCAL_SLICE
        __syncthreads();  // every warp's own slice (staged into its receive slot) is visible to the merging threads
#endif
        // Phase 2 (arrive now, wait before exit): once every active rank passed its wait above, all
        // bulk copies have completed, so the staging sources may be released.
        cluster_arrive_relaxed();
        TRACE_STAMP(5);
        auto load_merge = [&](int i) -> float4 {
          const uint32_t base = static_cast<uint32_t>(((slot_bases >> (4 * i)) & 15) * SLICE_BYTES) + off0;
          uint2 v[NA];
#pragma unroll
          for (int q = 0; q < NA; ++q) {
            const uint2 t = *reinterpret_cast<const uint2*>(recv_b + base + q * SLICE_BYTES);
            v[q] = (q < n_active) ? t : make_uint2(0u, 0u);  // never-written slots may hold NaN patterns
          }
          float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (int q = 0; q < NA; ++q) {
            const float2 a0 = unpack_half2(v[q].x), a1 = unpack_half2(v[q].y);
            o.x += w[q] * a0.x; o.y += w[q] * a0.y; o.z += w[q] * a1.x; o.w += w[q] * a1.y;
          }
          return o;
        };
        // my owned slices are i = my_pos + k * n_active, k = 0 .. n_owned-1 (round-robin ownership);
        // two per iteration (independent chains) plus a single tail (the common case: one owned slice)
        int k = 0;
        for (; k + 1 < n_owned; k += 2) {
          const int i0 = my_pos + k * nn, i1 = i0 + nn;
          const float4 o0 = load_merge(i0);
          const float4 o1 = load_merge(i1);
          consume(i0, o0);
          consume(i1, o1);
        }
        if (k < n_owned) consume(my_pos + k * nn, load_merge(my_pos + k * nn));
      };
      if (n_active <= 2) merge(IntC<2>{});
      else if (n_active <= 4) merge(IntC<4>{});
      else merge(IntC<8>{});
    }
    TRACE_STAMP(7);
    if (n_seg == 1) {
      // ---- 7c. single active segment: output written above; lse from the owner of slice 0 ---------
      if ((owned & 1u) && (tid & 15) == 0) p.lse[token * NUM_HEADS + h] = (L_seg > 0.f) ? (M_seg + fast_log2(L_seg)) : -INFINITY;
      cluster_wait_acquire();  // phase 2: all bulk copies out of my staging buffer are done
      TRACE_STAMP(15);
      TRACE_FLUSH();
      return;
    }
  } else {
    // Single active rank in this segment but other segments are active: I own all 8 slices.
    // Move my fragments into the (h, dl) thread mapping through smem and publish them all.
    __syncthreads();  // the S region is dead (softmax done everywhere)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = warp * SLICE_DIMS + i * 8 + c0;
      *reinterpret_cast<float2*>(s_mo + r0 * D_CKV + d) = make_float2(o_acc[i][0], o_acc[i][1]);
      *reinterpret_cast<float2*>(s_mo + (r0 + 8) * D_CKV + d) = make_float2(o_acc[i][2], o_acc[i][3]);
    }
    __syncthreads();
    if (!i_merge) {
#pragma unroll
      for (int i = 0; i < NUM_SLICES; ++i) publish_smem(i);
    }
    owned = 0xffu;  // all slices are mine
    TRACE_STAMP(7);
  }

  // ---- 7d. merge level 2 across segments (n_seg >= 2): tagged publish, elected merger ----------
  // Every published 16-byte vector is two (data, tag) 8-byte words (8-byte scalar accesses are
  // single-copy atomic), so the data validates itself: no fence, no atomic, no counters. The owners
  // of the merger segment poll the other segments' words for their owned slices with gpu-scope
  // relaxed loads until every tag equals this call's tag, then merge them with their own fp32 slices.
  if (!i_merge) {
    if ((tid & 15) == 0)  // this segment's (M, L) record (every active rank writes identical values)
      st_relaxed_v2u64(p.ws_ml + (tok_base + seg) * NUM_HEADS + h,
                       make_uint4(__float_as_uint(M_seg), tag, __float_as_uint(L_seg), tag));
    TRACE_STAMP(12);  // workspace stores issued
    if (n_active >= 2) cluster_wait_acquire();  // phase 2 (see above)
    TRACE_STAMP(15);
    TRACE_FLUSH();
    return;
  }
  {
    const unsigned pending = seg_mask & ~(1u << seg);
    // One owned slice per batch (the merger segment has all 8 ranks active whenever a level-2 merge
    // exists on prefix-packed top-k rows, so n_owned == 1 in practice).
    for (int b = 0; b < n_owned; ++b) {
      const int i = (my_pos + b * nn) & 7;
      uint4 v[NUM_SEG], mlw[NUM_SEG];
      // every (predicated) load first, checks afterwards: one L2 round trip per poll
      // The (M, L) record of a head is loaded by one lane of the head group (lanes 0 / 16 of the warp)
      // and broadcast with shuffles: the poll is bounded by the number of outstanding load
      // transactions of the CTA (16 x 16-B loads of the same record per head would double it).
      const bool ml_lane = (lane & 15) == 0;
      auto issue = [&](uint4* vv, uint4* mm) {
#pragma unroll
        for (int s = 0; s < NUM_SEG; ++s) {
          const bool ps = (pending >> s) & 1u;
          ld_relaxed_v2u64_pred(mm[s], p.ws_ml + (tok_base + s) * NUM_HEADS + h, ps && ml_lane && h < DSA_ABL_POLL_HEADS);
          ld_relaxed_v2u64_pred(vv[s], p.ws_o + (((tok_base + s) * NUM_SLICES + i) * NUM_HEADS + h) * WS_O_STRIDE + jw, ps && jw < WS_O_STRIDE && h < DSA_ABL_POLL_HEADS);
        }
      };
      auto bcast = [&](uint4* mm) {
#pragma unroll
        for (int s = 0; s < NUM_SEG; ++s) {
          mm[s].x = __shfl_sync(0xffffffffu, mm[s].x, lane & 16);
          mm[s].y = __shfl_sync(0xffffffffu, mm[s].y, lane & 16);
          mm[s].z = __shfl_sync(0xffffffffu, mm[s].z, lane & 16);
          mm[s].w = __shfl_sync(0xffffffffu, mm[s].w, lane & 16);
        }
      };
      auto valid = [&](const uint4* vv, uint4* mm) -> bool {
        bcast(mm);
        bool ok = true;
#pragma unroll
        for (int s = 0; s < NUM_SEG; ++s) {
          if (!((pending >> s) & 1u)) continue;
          ok &= (h >= DSA_ABL_POLL_HEADS) | ((mm[s].y == tag) & (mm[s].w == tag) & ((vv[s].w == tag) | (jw >= WS_O_STRIDE)));
        }
        return __all_sync(0xffffffffu, ok);
      };
      int spins = 0;
#if DSA_POLL_STAGGER_NS > 0
      // Staggered double poll: the publishers' stores usually become visible 0.3-0.6 us after this
      // CTA's own level-1 merge, so a second poll issued DSA_POLL_STAGGER_NS after the first (without
      // waiting for it) returns the fresh data one round trip earlier than a sequential re-poll.
      {
        uint4 v2[NUM_SEG], mlw2[NUM_SEG];
        issue(v, mlw);
        const unsigned long long t0 = globaltimer_ns();
        while (globaltimer_ns() - t0 < DSA_POLL_STAGGER_NS) {
        }
        issue(v2, mlw2);
        if (!valid(v, mlw)) {
          spins = 1;
          if (valid(v2, mlw2)) {
#pragma unroll
            for (int s = 0; s < NUM_SEG; ++s) {
              v[s] = v2[s];
              mlw[s] = mlw2[s];
            }
          } else {
            for (spins = 2; spins < POLL_LIMIT; ++spins) {
              issue(v, mlw);
              if (valid(v, mlw)) break;
            }
          }
        }
      }
#else
      for (; spins < POLL_LIMIT; ++spins) {
        issue(v, mlw);
        if (valid(v, mlw)) break;
      }
#endif
#ifdef DSA_TRACE
      if (tid == 0 && b == 0) s_trace[14] = static_cast<unsigned long long>(spins + 1);  // poll iterations
      if (b == 0) TRACE_STAMP(13);
#endif
      // merge state over the segments: own (M_seg, L_seg) in registers, the others from the records
      float m_s[NUM_SEG], l_s[NUM_SEG], w_s[NUM_SEG];
      float M = M_seg;
#pragma unroll
      for (int s = 0; s < NUM_SEG; ++s) {
        const bool act = (pending >> s) & 1u;
        m_s[s] = act ? __uint_as_float(mlw[s].x) : -INFINITY;
        l_s[s] = act ? __uint_as_float(mlw[s].z) : 0.f;
        M = fmaxf(M, m_s[s]);
      }
      const float w_own = fast_exp2(M_seg - M);
      float L = w_own * L_seg;
#pragma unroll
      for (int s = 0; s < NUM_SEG; ++s) {
        w_s[s] = (m_s[s] > -INFINITY) ? fast_exp2(m_s[s] - M) : 0.f;
        L += w_s[s] * l_s[s];
      }
      const float inv = (L > 0.f) ? __fdividef(1.f, L) : 0.f;
#if DSA_WS4
      {
        // my 4 dims (dl .. dl+3): own fp32 slice from s_mo, the other segments' fp16 words
        const float4 a = *reinterpret_cast<const float4*>(s_mo + h * D_CKV + i * SLICE_DIMS + dl);
        float o[4] = {a.x * w_own, a.y * w_own, a.z * w_own, a.w * w_own};
#pragma unroll
        for (int s = 0; s < NUM_SEG; ++s) {
          const float2 v0 = unpack_half2(v[s].x);
          const float2 v1 = unpack_half2(v[s].y);
          o[0] += w_s[s] * v0.x; o[1] += w_s[s] * v0.y;
          o[2] += w_s[s] * v1.x; o[3] += w_s[s] * v1.y;
        }
        *reinterpret_cast<uint2*>(out_t + h * D_CKV + i * SLICE_DIMS + dl) =
            make_uint2(pack_bf16x2(o[0] * inv, o[1] * inv), pack_bf16x2(o[2] * inv, o[3] * inv));
      }
#else
      if (jw < WS_O_STRIDE) {
        const float* src = s_mo + h * D_CKV + i * SLICE_DIMS + 6 * jw;
        const float2 a = *reinterpret_cast<const float2*>(src);
        const float2 b = *reinterpret_cast<const float2*>(src + 2);
        const float2 c = *reinterpret_cast<const float2*>(src + 4);
        float o[6] = {a.x * w_own, a.y * w_own, b.x * w_own, b.y * w_own, c.x * w_own, c.y * w_own};
#pragma unroll
        for (int s = 0; s < NUM_SEG; ++s) {
          const float2 v0 = unpack_half2(v[s].x);
          const float2 v1 = unpack_half2(v[s].y);
          const float2 v2 = unpack_half2(v[s].z);
          o[0] += w_s[s] * v0.x; o[1] += w_s[s] * v0.y;
          o[2] += w_s[s] * v1.x; o[3] += w_s[s] * v1.y;
          o[4] += w_s[s] * v2.x; o[5] += w_s[s] * v2.y;
        }
        __nv_bfloat16* dst = out_t + h * D_CKV + i * SLICE_DIMS + 6 * jw;  // 4-byte aligned
        *reinterpret_cast<uint32_t*>(dst) = pack_bf16x2(o[0] * inv, o[1] * inv);
        *reinterpret_cast<uint32_t*>(dst + 2) = pack_bf16x2(o[2] * inv, o[3] * inv);
        if (jw < WS_O_STRIDE - 1) *reinterpret_cast<uint32_t*>(dst + 4) = pack_bf16x2(o[4] * inv, o[5] * inv);
      }
#endif
      if (i == 0 && jw == 0) p.lse[token * NUM_HEADS + h] = (L > 0.f) ? (M + fast_log2(L)) : -INFINITY;
    }
  }
  if (n_active >= 2) cluster_wait_acquire();  // phase 2 (see above)
  TRACE_STAMP(15);
  TRACE_FLUSH();
}

}  // namespace

static unsigned long long* g_trace_ptr = nullptr;
void dsa_set_trace(tvm::ffi::TensorView buf) {
  g_trace_ptr = buf.numel() > 0 ? static_cast<unsigned long long*>(buf.data_ptr()) : nullptr;
}

void dsa_run(tvm::ffi::TensorView q_nope, tvm::ffi::TensorView q_pe, tvm::ffi::TensorView ckv_cache,
             tvm::ffi::TensorView kpe_cache, tvm::ffi::TensorView sparse_indices, double sm_scale,
             tvm::ffi::TensorView output, tvm::ffi::TensorView lse, tvm::ffi::TensorView counters,
             tvm::ffi::TensorView ws_o, tvm::ffi::TensorView ws_ml, int64_t tag) {
  const int num_tokens = static_cast<int>(q_nope.shape()[0]);
  TVM_FFI_ICHECK_EQ(q_nope.shape()[1], NUM_HEADS);
  TVM_FFI_ICHECK_EQ(q_nope.shape()[2], D_CKV);
  TVM_FFI_ICHECK_EQ(q_pe.shape()[2], D_KPE);
  TVM_FFI_ICHECK_EQ(ckv_cache.shape()[1], PAGE_SIZE);
  TVM_FFI_ICHECK_EQ(sparse_indices.shape()[1], TOPK);
  TVM_FFI_ICHECK_GE(counters.numel(), static_cast<int64_t>(num_tokens) * NUM_SLICES);
  TVM_FFI_ICHECK_GE(ws_o.numel() * static_cast<int64_t>(ws_o.dtype().bits / 8),
                    static_cast<int64_t>(num_tokens) * NUM_SEG * NUM_SLICES * NUM_HEADS * WS_O_STRIDE * 16);
  TVM_FFI_ICHECK_GE(ws_ml.numel() * static_cast<int64_t>(ws_ml.dtype().bits / 8),
                    static_cast<int64_t>(num_tokens) * NUM_SEG * NUM_HEADS * 16);
  TVM_FFI_ICHECK_NE(tag, 0);

  Params p;
  p.q_nope = static_cast<const __nv_bfloat16*>(q_nope.data_ptr());
  p.q_pe = static_cast<const __nv_bfloat16*>(q_pe.data_ptr());
  p.ckv = static_cast<const __nv_bfloat16*>(ckv_cache.data_ptr());
  p.kpe = static_cast<const __nv_bfloat16*>(kpe_cache.data_ptr());
  p.indices = static_cast<const int32_t*>(sparse_indices.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(output.data_ptr());
  p.lse = static_cast<float*>(lse.data_ptr());
  p.counters = static_cast<int*>(counters.data_ptr());
  p.ws_o = static_cast<uint4*>(ws_o.data_ptr());
  p.ws_ml = static_cast<uint4*>(ws_ml.data_ptr());
  p.tag = static_cast<uint32_t>(tag);
  p.scale_log2 = static_cast<float>(sm_scale * 1.4426950408889634);
  p.num_tokens = num_tokens;
  p.trace = g_trace_ptr;

  DLDevice dev = q_nope.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(dsa_sparse_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_LAUNCH);
    attr_set = true;
  }
  if (num_tokens == 0) return;
  dim3 grid(num_tokens * CTAS_PER_TOKEN);
  dsa_sparse_attn_kernel<<<grid, NUM_THREADS, SMEM_LAUNCH, stream>>>(p);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << "dsa_sparse_attn_kernel launch failed: " << cudaGetErrorString(err);
  }
}
