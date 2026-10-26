// KDA forward v21: tcgen05 (5th-gen tensor cores, TMEM accumulators) + TMA, warp-specialised, 1 task per CTA, 2 CTAs/SM. Tiles built in place in the TMA stage; normalisation folded into the solve and the output read-out.
// Math: chunked delta rule with per-channel gates, sub-chunks of 16 tokens, state kept transposed S^T[v][k] (fp32 in TMEM).
// Written from the KDA definition; no reference kernel code used.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

#define BT 32
#define D 128
#define NST 4
#define NTHREADS 768
#ifndef PF_AHEAD
#define PF_AHEAD 4   // L2 prefetch distance beyond the smem stages
#endif

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; d |= (uint64_t)swz << 61; return d;
}
__device__ __forceinline__ uint32_t make_idesc_tf32(int M, int N, int transA, int transB) {
    return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)transA << 15) | ((uint32_t)transB << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void mma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, int enable_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(enable_d));
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"
                 :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                    "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]),
                    "r"(r[16]), "r"(r[17]), "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]),
                    "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]), "r"(taddr) : "memory");
}
__device__ __forceinline__ uint32_t make_idesc(int M, int N, int transA, int transB) {
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)transA << 15) | ((uint32_t)transB << 16) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void mbar_init(uint32_t mbar, int count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mbar), "r"(count)); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t mbar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t parity) {
    asm volatile("{\n.reg .pred p;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra LAB_WAIT;\n}\n" ::"r"(mbar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* tmap, int c0, int c1, int c2, uint32_t mbar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                 ::"r"(dst), "l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* tmap, int c0, int c1, int c2, uint32_t src) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];\n"
                 ::"l"(reinterpret_cast<uint64_t>(tmap)), "r"(c0), "r"(c1), "r"(c2), "r"(src) : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void mma_ss(uint32_t d_tmem, uint64_t adesc, uint64_t bdesc, uint32_t idesc, int enable_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(enable_d));
}
__device__ __forceinline__ void mma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t bdesc, uint32_t idesc, int enable_d) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                 ::"r"(d_tmem), "r"(a_tmem), "l"(bdesc), "r"(idesc), "r"(enable_d));
}
__device__ __forceinline__ void mma_commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(mbar) : "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
template <int ID> __device__ __forceinline__ void bar_sync_t(int cnt) { asm volatile("bar.sync %0, %1;" ::"n"(ID), "r"(cnt) : "memory"); }
template <int ID> __device__ __forceinline__ void bar_arrive_t(int cnt) { asm volatile("bar.arrive %0, %1;" ::"n"(ID), "r"(cnt) : "memory"); }
template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }
__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_ld16_nw(uint32_t taddr, uint32_t* r) {   // no wait: caller issues tmem_wait_ld()
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void tmem_st16(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};\n"
                 :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                    "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_st8(uint32_t taddr, const uint32_t* r) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ float2 unpack_bf16(uint32_t u) { __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u); return __bfloat1622float2(v); }
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
    __nv_bfloat162 z = __hmul2(x, y); return *reinterpret_cast<uint32_t*>(&z);
}
__device__ __forceinline__ float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) { __half2 v = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void c2a(uint32_t* a, const float (*c)[4]) {
    a[0] = pack_f16(c[0][0], c[0][1]); a[1] = pack_f16(c[0][2], c[0][3]); a[2] = pack_f16(c[1][0], c[1][1]); a[3] = pack_f16(c[1][2], c[1][3]);
}
__device__ __forceinline__ void c2b(uint32_t* b, const float (*c)[4]) {
    b[0] = pack_f16(c[0][0], c[0][1]); b[1] = pack_f16(c[1][0], c[1][1]); b[2] = pack_f16(c[0][2], c[0][3]); b[3] = pack_f16(c[1][2], c[1][3]);
}
__device__ __forceinline__ void mm16(float (*out)[4], const float (*X)[4], const float (*YT)[4]) {
    uint32_t a[4], b[4]; c2a(a, X); c2b(b, YT);
    out[0][0] = out[0][1] = out[0][2] = out[0][3] = 0.f; out[1][0] = out[1][1] = out[1][2] = out[1][3] = 0.f;
    mma_f16(out[0], a, b[0], b[1]); mma_f16(out[1], a, b[2], b[3]);
}

// swizzled TMA tile [16 rows][128 elems] bf16 as 2 boxes of [16][64] (128B rows, SW128): byte offset of (row r, elem d)
__device__ __forceinline__ int swz_off(int r, int d) { int box = d >> 6, dd = d & 63; return box * 2048 + r * 128 + (((dd >> 3) ^ (r & 7)) << 4) + ((dd & 7) << 1); }
// K-major core-matrix tile with 8-row groups (SBO=128B) and 8-col k-blocks (LBO): byte offset of (row r, col k)
__device__ __forceinline__ int km_off(int r, int k, int lbo) { return (k >> 3) * lbo + (r >> 3) * 128 + ((r & 7) << 4) + ((k & 7) << 1); }
// fp32 K-major core-matrix tile: (row r, col k) with 4 k per 16B chunk; LBO = k-block(4) stride
__device__ __forceinline__ int km32_off(int r, int k, int lbo, int sbo = 128) { return (k >> 2) * lbo + (r >> 3) * sbo + ((r & 7) << 4) + ((k & 3) << 2); }
// MN-major (N contiguous) core-matrix tile: rows = K index, cols = N index; k-block (8 rows) stride LBO, n-group (8 cols) stride 128B
__device__ __forceinline__ int mn_off(int krow, int n, int lbo) { return (krow >> 3) * lbo + (n >> 3) * 128 + ((krow & 7) << 4) + ((n & 7) << 1); }

#define BSYNC(id, cnt, code) do { DBG(code); bar_sync_t<(id)>((cnt)); } while (0)
#define BARRIVE(id, cnt, code) do { DBG(code); bar_arrive_t<(id)>((cnt)); } while (0)
#define MWAIT(mb, par, code) do { DBG(code); mbar_wait((mb), (par)); } while (0)
__device__ __forceinline__ float rcp_approx(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void tma_prefetch_3d(const CUtensorMap* tmap, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];\n" ::"l"(tmap), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t mbar, uint32_t parity) {
    uint32_t ok; asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mbar), "r"(parity) : "memory"); return ok != 0;
}
__device__ __forceinline__ void nsleep(unsigned ns) { asm volatile("nanosleep.u32 %0;" ::"r"(ns)); }
__device__ __forceinline__ void mbar_arrive(uint32_t mbar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(mbar) : "memory"); }


// ---------------------------------------------------------------- v41: BT = 32 as two 16-token half-solves, 1 CTA per SM
// warps: 0-3 TMEM warps, 4-7 gate group 0 (even chunks), 8-11 gate group 1 (odd chunks), 12 solver half 0, 13 norms + solver half 1 (+M_k),
//        14 MMA issuer, 15 TMA producer
#define BAR_GRP0 1
#define BAR_GRP1 2
#define BAR_C2A 3      // TMEM warps arrive 128 (UB0 ready) + issuer sync 32 -> 160
#define BAR_C2B 8      // TMEM warps arrive 128 (S scaled, UB1 ready) + issuer sync 32 -> 160
#define BAR_C3 4       // TMEM warps arrive 128 (A_S(j) ready, O(j-1) read) + issuer sync 32 -> 160
#define BAR_SFREE 5    // +p: q/k/g rows of stage j%NST free (after mma(2b)): TMEM warps arrive 128 + TMA warp sync 32 (ids 5,6)
#define BAR_VFREE 9    // +p: v rows free (after the output copy-out) (ids 9,10)
#define BAR_END 0

#define TM_S 0
#define TM_O0 128
#define TM_R0 144
#define TM_O1 160
#define TM_R1 176
#define TM_AS 192
#define TM_UB 256      // UB0 256-263, UB1 264-271
#define NFULL 192
#define BF_LBO 3072    // bf16 K-major [192 n][32 k]: 8-token k-block stride = 24 n-groups * 128 B
#define M_LBO 512      // bf16 K-major [32 n][16 k]
#define SUB 2048       // 16 rows x 128 B (TMA box {64 ch, 1 head, 16 tok})
#define STAGE_BYTES (16 * SUB)
// stage layout per channel half hi (base hi*8*SUB): sub 0 q rows 0-15, 1 k rows 0-15, 2 q rows 16-31, 3 k rows 16-31, 4 g rows 0-15, 5 g rows 16-31, 6 v rows 0-15, 7 v rows 16-31
// after the gate pass: q -> qt, k -> kt (full gauge after rescale), g -> kinv (half gauge) then g rows 0-15 -> khat0^(16)

struct __align__(1024) Smem {
    unsigned char in[NST][STAGE_BYTES];
    unsigned char bfull[NST][4 * BF_LBO];    // [192 n][32 k] bf16: n<128 khat[t][n]; 128-143 Bl11; 144-159 zero; 160-175 Bl22; 176-191 zero
    unsigned char mcor[NST][2 * M_LBO];      // M01 = [qt1; kt'1] . khat0^(16)^T, bf16 K-major [32 n][16 k]
    unsigned char tvb[2][2][1024];           // [parity][half]: T_V hi | lo, bf16 K-major [16 n][16 k]
    __half xs[NST][2][BT / 2][BT / 2 + 2];   // X11, X22 (fp16, pre-scaled, masked)
    float nk[NST][BT / 2][BT / 2 + 1];       // N_k = kt1 . khat0^(16)^T
    float eend[NST][D];
    float rq[NST][BT], rk[NST][BT], bsig[NST][BT];
    float2 xch[2][4][64];
    uint64_t mbar_in[NST], mbar_v[NST], mbar_tiles[NST], mbar_norm[NST], mbar_solve[NST];
    uint64_t mbar_m1[2], mbar_m2a[2], mbar_m2b[2], mbar_m3[2];
    uint32_t tmem_base;
};

// 16x16 Neumann inverse of (I - X) for strictly lower-triangular X given in C-fragment layout (X, XT) -> Tm (C-fragment layout)
__device__ __forceinline__ void neumann16(const float (*X)[4], const float (*XT)[4], float (*Tm)[4], int gid, int t4) {
    float P[2][4], X2[2][4], X2T[2][4], T3[2][4];
    mm16(X2, X, XT); mm16(X2T, XT, X);
    mm16(T3, X2, XT);
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int rr = gid + 8 * (jj >> 1), cc = nt * 8 + 2 * t4 + (jj & 1);
            P[nt][jj] = ((rr == cc) ? 1.f : 0.f) + X[nt][jj] + X2[nt][jj] + T3[nt][jj];
        }
    float X4[2][4], X4T[2][4], X8T[2][4], P4[2][4];
    mm16(X4, X2, X2T); mm16(X4T, X2T, X2);
    mm16(X8T, X4T, X4);
#pragma unroll
    for (int nt = 0; nt < 2; nt++)
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int rr = gid + 8 * (jj >> 1), cc = nt * 8 + 2 * t4 + (jj & 1);
            float id = (rr == cc) ? 1.f : 0.f;
            X4T[nt][jj] += id; X8T[nt][jj] += id;
        }
    mm16(P4, P, X4T);
    mm16(Tm, P4, X8T);
}
// kt'_h = (-T_R,h) . kt_h for one 16-row half: A fragments (hi/lo), B = kt rows via ldmatrix.trans; results into registers c[8][2][4]
__device__ __forceinline__ void ktprime_half(const unsigned char* kt, const uint32_t* ahi, const uint32_t* alo, int mo, int r, float (*c)[2][4]) {
#pragma unroll
    for (int g2 = 0; g2 < 8; g2++) {
        c[g2][0][0] = c[g2][0][1] = c[g2][0][2] = c[g2][0][3] = 0.f; c[g2][1][0] = c[g2][1][1] = c[g2][1][2] = c[g2][1][3] = 0.f;
        const int cc = 2 * (g2 & 3) + (mo >> 1);
        uint32_t b[4];
        ldsm_x4_trans(b[0], b[1], b[2], b[3], kt + (g2 >> 2) * 8 * SUB + r * 128 + ((cc ^ (r & 7)) << 4));
        mma_bf16(c[g2][0], ahi, b[0], b[1]); mma_bf16(c[g2][0], alo, b[0], b[1]);
        mma_bf16(c[g2][1], ahi, b[2], b[3]); mma_bf16(c[g2][1], alo, b[2], b[3]);
    }
}
__device__ __forceinline__ void ktprime_store(unsigned char* kt, const float (*c)[2][4]) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int g2 = 0; g2 < 8; g2++)
#pragma unroll
        for (int w = 0; w < 2; w++) {
            const int nt = 2 * g2 + w, chunk = nt & 7, box = nt >> 3;
            *reinterpret_cast<uint32_t*>(kt + box * 8 * SUB + gid * 128 + ((chunk ^ gid) << 4) + 4 * t4) = pack_bf16(c[g2][w][0], c[g2][w][1]);
            *reinterpret_cast<uint32_t*>(kt + box * 8 * SUB + (gid + 8) * 128 + ((chunk ^ gid) << 4) + 4 * t4) = pack_bf16(c[g2][w][2], c[g2][w][3]);
        }
}

__global__ void __launch_bounds__(NTHREADS, 1)
kda_fwd_kernel49(const __grid_constant__ CUtensorMap tm_q, const __grid_constant__ CUtensorMap tm_k, const __grid_constant__ CUtensorMap tm_v,
                 const __grid_constant__ CUtensorMap tm_g, const __grid_constant__ CUtensorMap tm_o,
                 const __nv_bfloat16* __restrict__ beta, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                 __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, const int64_t* __restrict__ cu_seqlens,
                 int H, float scale, float lower_bound, long long* __restrict__ tbuf, int* dbg_ptr) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    unsigned char* smem_al = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
    Smem& sm = *reinterpret_cast<Smem*>(smem_al);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int j = -1; (void)j;
    const int gid = lane >> 2, t4 = lane & 3;
    const int th = blockIdx.x;
    const int n = th / H, h = th % H;
    const int64_t bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
    const int64_t row_stride = (int64_t)H * D;
    const int nsub = (int)((eos - bos + BT - 1) / BT);
    __nv_bfloat16* sp = state + ((int64_t)(n * H + h)) * D * D;
    const float Aexp = ex2_approx(A_log[h] * 1.4426950408889634f);
    const float hlb2 = 0.5f * lower_bound * 1.4426950408889634f;
#ifdef KDA_DEBUG
#define DBG(code) do { if (lane == 0 && dbg_ptr != nullptr && blockIdx.x < 64) { ((volatile int*)dbg_ptr)[blockIdx.x * 16 + warp] = (code); } } while (0)
#else
#define DBG(code) do {} while (0)
#endif
#ifdef KDA_TIMING
#define TSTAMP(ph) do { if (tbuf != nullptr && blockIdx.x == 100 && lane == 0 && j >= 8 && j < 16) tbuf[(warp * 8 + (j - 8)) * 8 + (ph)] = clock64(); } while (0)
#define TSTAMP_DEP(ph) do { int dmy_; asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(dmy_) : "r"(smem_u32(&sm.rq[0][0]))); \
    if (tbuf != nullptr && blockIdx.x == 100 && lane == 0 && j >= 8 && j < 16) tbuf[(warp * 8 + (j - 8)) * 8 + (ph)] = clock64() + (dmy_ & 0); } while (0)
    if (tid == 0 && tbuf != nullptr) { unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); tbuf[1536 + 4 * blockIdx.x] = smid; tbuf[1536 + 4 * blockIdx.x + 1] = clock64(); }
#else
#define TSTAMP(ph) do {} while (0)
#define TSTAMP_DEP(ph) do {} while (0)
#endif

    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(&sm.tmem_base)), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
    }
    if (tid == 0) {
        for (int s = 0; s < NST; s++) { mbar_init(smem_u32(&sm.mbar_in[s]), 1); mbar_init(smem_u32(&sm.mbar_v[s]), 1); mbar_init(smem_u32(&sm.mbar_tiles[s]), 256); mbar_init(smem_u32(&sm.mbar_norm[s]), 32); mbar_init(smem_u32(&sm.mbar_solve[s]), 64); }
        for (int p = 0; p < 2; p++) { mbar_init(smem_u32(&sm.mbar_m1[p]), 1); mbar_init(smem_u32(&sm.mbar_m2a[p]), 1); mbar_init(smem_u32(&sm.mbar_m2b[p]), 1); mbar_init(smem_u32(&sm.mbar_m3[p]), 1); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    // static zeros of bfull rows 128..191 (all k) for every stage: the Bl blocks written later only cover their own (half, half) quadrant
    for (int i = tid; i < NST * 4 * 64; i += NTHREADS) {   // stage, k-block, 64 rows -> one 16 B chunk each
        const int s = i / (4 * 64), rem = i % (4 * 64);
        const int kb = rem / 64, nrow = 128 + rem % 64;
        *reinterpret_cast<uint4*>(sm.bfull[s] + kb * BF_LBO + (nrow >> 3) * 128 + (nrow & 7) * 16) = make_uint4(0u, 0u, 0u, 0u);
    }
    fence_proxy_async();
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = sm.tmem_base;
    auto issue_tma = [&](int i, int part) {   // full warp; part 0: q/k/g boxes (12 x 2 KB) on mbar_in, part 1: v boxes (4 x 2 KB) on mbar_v
        if (i < nsub) {
            int s = i % NST;
            uint32_t mb = smem_u32(part ? &sm.mbar_v[s] : &sm.mbar_in[s]);
            if (lane == 0) mbar_expect_tx(mb, part ? 4 * SUB : 12 * SUB);
            __syncwarp();
            int t0 = (int)(bos + (int64_t)i * BT);
            const int nb = part ? 4 : 12;
            if (lane < nb) {
                const int hi = part ? (lane >> 1) : (lane / 6), li = part ? (lane & 1) : (lane % 6);
                const int sub = part ? (6 + li) : li;
                const int x = (sub < 4) ? (sub & 1) : ((sub < 6) ? 2 : 3);
                const int hf = (sub < 4) ? (sub >> 1) : (sub & 1);
                const CUtensorMap* mp = (x == 0) ? &tm_q : (x == 1) ? &tm_k : (x == 2) ? &tm_g : &tm_v;
                tma_load_3d(smem_u32(sm.in[s] + hi * 8 * SUB + sub * SUB), mp, hi * 64, h, t0 + 16 * hf, mb);
            }
            __syncwarp();
        }
    };
    if (warp == 23) { for (int i = 0; i < NST; i++) { issue_tma(i, 0); issue_tma(i, 1); } }

    if (warp >= 4 && warp < 20) {
        // ================================ GATE GROUPS (2 x 8 warps; group g: chunks j = g mod 2): thread = channel pair cp, token quarter qq (8 tokens) ================================
        const int grp = (warp >= 12) ? 1 : 0;
        const int gl = tid - 128 - 256 * grp;            // 0..255
        const int qq = gl >> 6, cp = gl & 63, gw = (gl >> 5) & 7;
        const int hf = qq >> 1, rb = 8 * (qq & 1);        // half and row base within the half's 16-row sub-buffer
        const int d = 2 * cp;
        const int box = d >> 6, c0s = ((d & 63) >> 3) << 4, wb = (d & 7) * 2;
        const int qb = box * 8 * SUB + (hf ? 2 : 0) * SUB + rb * 128 + wb;   // + i*128 + xo[i]  (token 16 hf + rb + i, i < 8)
        const int kb = qb + SUB, gb = box * 8 * SUB + (4 + hf) * SUB + rb * 128 + wb;
        uint32_t xo[8];
#pragma unroll
        for (int m = 0; m < 8; m++) xo[m] = (uint32_t)(c0s ^ (((rb + m) & 7) << 4));
        const int bf_off = qq * BF_LBO + (d >> 3) * 128 + (d & 7) * 16;       // khat chunk (8 tokens of this quarter) of channel d
        const int bf_sw = ((lane >> 2) & 1) ? 16 : 0;
        const float2 dtb = *reinterpret_cast<const float2*>(dt_bias + h * D + d);
        const float hA = 0.5f * Aexp, hAdx = hA * dtb.x, hAdy = hA * dtb.y;
        const int mo = lane >> 3, rr8 = (lane & 7) + 8 * (mo & 1);
        uint32_t braw = 0x0000u; bool bok = false;
        auto load_beta = [&](int i) {
            if (gw == 2) {
                int64_t tok = bos + (int64_t)i * BT + lane;
                bok = (i < nsub) && (tok < eos);
                const uint16_t* src = reinterpret_cast<const uint16_t*>(beta + (bok ? tok * H + h : 0));
                asm volatile("ld.global.nc.u16 %0, [%1];" : "=r"(braw) : "l"(src));
            }
        };
        auto gbar = [&](int code) { if (grp) BSYNC(BAR_GRP1, 256, code); else BSYNC(BAR_GRP0, 256, code); };
        load_beta(grp);
        for (j = grp; j < nsub; j += 2) {
            const int s = j % NST;
            const int64_t t0 = bos + (int64_t)j * BT;
            const int nvalid = (int)min((int64_t)BT, eos - t0);
            MWAIT(smem_u32(&sm.mbar_in[s]), (j / NST) & 1, 1100 * 1000 + (int)(j & 1023));
            TSTAMP(0);
            const unsigned char* st = sm.in[s];
            unsigned char* stw = sm.in[s];
            const unsigned char* gq = st + qb; const unsigned char* gk = st + kb; const unsigned char* gg = st + gb;
            unsigned char* wq = stw + qb; unsigned char* wk = stw + kb; unsigned char* wg = stw + gb;
            float ex[8], ey[8];
            float tx = 1.f, ty = 1.f;
            if (nvalid >= BT) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float2 gv = unpack_bf16(*reinterpret_cast<const uint32_t*>(gg + i * 128 + xo[i]));
                    const float e0 = ex2_approx(fmaf(hlb2, tanh_approx(fmaf(hA, gv.x, hAdx)), hlb2));
                    const float e1 = ex2_approx(fmaf(hlb2, tanh_approx(fmaf(hA, gv.y, hAdy)), hlb2));
                    ex[i] = e0; ey[i] = e1; tx *= e0; ty *= e1;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float2 gv = unpack_bf16(*reinterpret_cast<const uint32_t*>(gg + i * 128 + xo[i]));
                    const bool ok = (8 * qq + i) < nvalid;
                    const float e0 = ok ? ex2_approx(fmaf(hlb2, tanh_approx(fmaf(hA, gv.x, hAdx)), hlb2)) : 1.f;
                    const float e1 = ok ? ex2_approx(fmaf(hlb2, tanh_approx(fmaf(hA, gv.y, hAdy)), hlb2)) : 1.f;
                    ex[i] = e0; ey[i] = e1; tx *= e0; ty *= e1;
                }
            }
            TSTAMP(3);
            sm.xch[grp][qq][cp] = make_float2(tx, ty);
            if (gw == 2) {
                const float bv = bok ? __bfloat162float(__ushort_as_bfloat16((unsigned short)braw)) : -1e30f;
                sm.bsig[s][lane] = fmaf(0.5f, tanh_approx(0.5f * bv), 0.5f);
                load_beta(j + 2);
            }
            {   // q/k norms (FMA): token tn = 4 gw + (lane & 3), channel eighth ce = lane >> 2 (16 channels = 2 chunks of 8)
                const int tn = 4 * gw + (lane & 3), ce = lane >> 2;
                const int rowoff = ((tn >> 4) ? 2 : 0) * SUB + (tn & 15) * 128;
                float sq = 0.f, sk = 0.f;
#pragma unroll
                for (int ch = 0; ch < 2; ch++) {
                    const int cidx = 2 * ce + ch;
                    const int off = (cidx >> 3) * 8 * SUB + rowoff + (((cidx & 7) ^ (tn & 7)) << 4);
                    const uint4 qa = *reinterpret_cast<const uint4*>(st + off);
                    const uint4 ka = *reinterpret_cast<const uint4*>(st + SUB + off);
                    const uint32_t qw[4] = {qa.x, qa.y, qa.z, qa.w}, kw[4] = {ka.x, ka.y, ka.z, ka.w};
#pragma unroll
                    for (int u = 0; u < 4; u++) { const float2 a2 = unpack_bf16(qw[u]), b2 = unpack_bf16(kw[u]); sq = fmaf(a2.x, a2.x, fmaf(a2.y, a2.y, sq)); sk = fmaf(b2.x, b2.x, fmaf(b2.y, b2.y, sk)); }
                }
                sq += __shfl_xor_sync(0xffffffff, sq, 4); sq += __shfl_xor_sync(0xffffffff, sq, 8); sq += __shfl_xor_sync(0xffffffff, sq, 16);
                sk += __shfl_xor_sync(0xffffffff, sk, 4); sk += __shfl_xor_sync(0xffffffff, sk, 8); sk += __shfl_xor_sync(0xffffffff, sk, 16);
                if (ce == 0) { sm.rq[s][tn] = rsqrtf(sq + 1e-12f) * scale; sm.rk[s][tn] = rsqrtf(sk + 1e-12f); }
            }
            gbar(1101 * 1000 + (int)(j & 1023));
            const float2 T0 = sm.xch[grp][0][cp], T1 = sm.xch[grp][1][cp], T2 = sm.xch[grp][2][cp], T3 = sm.xch[grp][3][cp];
            TSTAMP_DEP(4);
            // half-local prefix / suffix scales: half h = quarters 2h, 2h+1
            const float pinx = (qq & 1) ? (hf ? T2.x : T0.x) : 1.f, piny = (qq & 1) ? (hf ? T2.y : T0.y) : 1.f;     // product of the earlier quarter in this half
            const float soutx = (qq & 1) ? 1.f : (hf ? T3.x : T1.x), souty = (qq & 1) ? 1.f : (hf ? T3.y : T1.y);   // product of the later quarter in this half
            const float Thx = hf ? T2.x * T3.x : T0.x * T1.x, Thy = hf ? T2.y * T3.y : T0.y * T1.y;                // half total
            const float coutx = hf ? 1.f : (T2.x * T3.x), couty = hf ? 1.f : (T2.y * T3.y);                        // later half total
            const float E16x = T0.x * T1.x, E16y = T0.y * T1.y;
            const float Ex = E16x * T2.x * T3.x, Ey = E16y * T2.y * T3.y;
            if (qq == 0) *reinterpret_cast<float2*>(&sm.eend[s][d]) = make_float2(Ex, Ey);
            const float rtx = rcp_approx(Thx), rty = rcp_approx(Thy);            // 1 / half total
            {   // suffix pass: kinv^(h) = k S_half / T_half (over g) ; khat = k S_half cout (-> bfull)
                float Sx = soutx, Sy = souty;
                uint32_t kh[8];
#pragma unroll
                for (int i = 7; i >= 0; i--) {
                    const uint32_t kp = *reinterpret_cast<const uint32_t*>(gk + i * 128 + xo[i]);
                    *reinterpret_cast<uint32_t*>(wg + i * 128 + xo[i]) = hmul2_bf16(kp, pack_bf16(Sx * rtx, Sy * rty));
                    kh[i] = hmul2_bf16(kp, pack_bf16(Sx * coutx, Sy * couty));
                    Sx *= ex[i]; Sy *= ey[i];
                }
                uint32_t khx[4], khy[4];
#pragma unroll
                for (int i = 0; i < 8; i += 2) { khx[i >> 1] = __byte_perm(kh[i], kh[i + 1], 0x5410); khy[i >> 1] = __byte_perm(kh[i], kh[i + 1], 0x7632); }
                unsigned char* bf = sm.bfull[s] + bf_off;
                const uint4 vx = make_uint4(khx[0], khx[1], khx[2], khx[3]), vy = make_uint4(khy[0], khy[1], khy[2], khy[3]);
                *reinterpret_cast<uint4*>(bf + bf_sw) = bf_sw ? vy : vx;
                *reinterpret_cast<uint4*>(bf + (bf_sw ^ 16)) = bf_sw ? vx : vy;
            }
            TSTAMP(5);
            {   // prefix pass (half gauge): qt^(h) = q P_half, kt^(h) = k P_half
                float Px = pinx, Py = piny;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    Px *= ex[i]; Py *= ey[i];
                    const uint32_t pk = pack_bf16(Px, Py);
                    const uint32_t qp = *reinterpret_cast<const uint32_t*>(gq + i * 128 + xo[i]);
                    const uint32_t kp = *reinterpret_cast<const uint32_t*>(gk + i * 128 + xo[i]);
                    *reinterpret_cast<uint32_t*>(wq + i * 128 + xo[i]) = hmul2_bf16(qp, pk);
                    *reinterpret_cast<uint32_t*>(wk + i * 128 + xo[i]) = hmul2_bf16(kp, pk);
                }
            }
            gbar(1103 * 1000 + (int)(j & 1023));
            TSTAMP(1);
            if (gw < 4) {   // smalls (half gauge): gw 0/1: A_hh = kt_h . kinv_h^T -> X_hh ; gw 2/3: B_hh = qt_h . kinv_h^T -> Bl_hh   (h = gw & 1)
                const int hh = gw & 1;
                const unsigned char* abase = st + (hh ? 2 : 0) * SUB + ((gw < 2) ? SUB : 0);
                const unsigned char* bbase = st + (4 + hh) * SUB;
                float acc[2][4] = {};
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    uint32_t a[4], b[4];
                    const int c = 2 * (ks & 3) + (mo >> 1);
                    const int off = (ks >> 2) * 8 * SUB + rr8 * 128 + ((c ^ (rr8 & 7)) << 4);
                    ldsm_x4(a[0], a[1], a[2], a[3], abase + off);
                    ldsm_x4(b[0], b[1], b[2], b[3], bbase + off);
                    mma_bf16(acc[0], a, b[0], b[2]); mma_bf16(acc[1], a, b[1], b[3]);
                }
                if (gw < 2) {
                    const int r0 = 16 * hh + gid, r1 = r0 + 8;
                    const float w0 = -sm.bsig[s][r0] * sm.rk[s][r0], w1 = -sm.bsig[s][r1] * sm.rk[s][r1];
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int jj = 0; jj < 4; jj++) {
                            const int rl = gid + 8 * (jj >> 1), cl = nt * 8 + 2 * t4 + (jj & 1);
                            sm.xs[s][hh][rl][cl] = __float2half((cl < rl) ? acc[nt][jj] * ((jj >> 1) ? w1 : w0) * sm.rk[s][16 * hh + cl] : 0.f);
                        }
                } else {
                    unsigned char* bf = sm.bfull[s];
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int rb2 = 0; rb2 < 2; rb2++) {
                            const int tl = gid + 8 * rb2, il = nt * 8 + 2 * t4;
                            *reinterpret_cast<uint32_t*>(bf + km_off((hh ? 160 : 128) + tl, 16 * hh + il, BF_LBO)) = pack_bf16((il <= tl) ? acc[nt][2 * rb2] : 0.f, (il + 1 <= tl) ? acc[nt][2 * rb2 + 1] : 0.f);
                        }
                }
            }
            gbar(1107 * 1000 + (int)(j & 1023));   // smalls done reading the half-gauge tiles
            TSTAMP(6);
            if (hf == 1) {   // rescale qt/kt rows 16-31 to the full gauge (x E_16)
                const uint32_t ce = pack_bf16(E16x, E16y);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int off = i * 128 + xo[i];
                    *reinterpret_cast<uint32_t*>(wq + off) = hmul2_bf16(*reinterpret_cast<const uint32_t*>(gq + off), ce);
                    *reinterpret_cast<uint32_t*>(wk + off) = hmul2_bf16(*reinterpret_cast<const uint32_t*>(gk + off), ce);
                }
            }
            gbar(1108 * 1000 + (int)(j & 1023));   // full-gauge tiles complete
            if (gw >= 4 && gw < 6) {   // N = [qt1 ; kt1] (full gauge, rows 16-31) . kinv0^T : gw 4 -> N_q -> mcor rows 0-15 ; gw 5 -> N_k -> nk
                const unsigned char* abase = st + 2 * SUB + ((gw == 5) ? SUB : 0);
                const unsigned char* bbase = st + 4 * SUB;   // kinv0 rows 0-15
                float acc[2][4] = {};
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    uint32_t a[4], b[4];
                    const int c = 2 * (ks & 3) + (mo >> 1);
                    const int off = (ks >> 2) * 8 * SUB + rr8 * 128 + ((c ^ (rr8 & 7)) << 4);
                    ldsm_x4(a[0], a[1], a[2], a[3], abase + off);
                    ldsm_x4(b[0], b[1], b[2], b[3], bbase + off);
                    mma_bf16(acc[0], a, b[0], b[2]); mma_bf16(acc[1], a, b[1], b[3]);
                }
                if (gw == 4) {
                    unsigned char* mc = sm.mcor[s];
#pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        *reinterpret_cast<uint32_t*>(mc + km_off(gid, nt * 8 + 2 * t4, M_LBO)) = pack_bf16(acc[nt][0], acc[nt][1]);
                        *reinterpret_cast<uint32_t*>(mc + km_off(gid + 8, nt * 8 + 2 * t4, M_LBO)) = pack_bf16(acc[nt][2], acc[nt][3]);
                    }
                } else {
#pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        sm.nk[s][gid][nt * 8 + 2 * t4] = acc[nt][0]; sm.nk[s][gid][nt * 8 + 2 * t4 + 1] = acc[nt][1];
                        sm.nk[s][gid + 8][nt * 8 + 2 * t4] = acc[nt][2]; sm.nk[s][gid + 8][nt * 8 + 2 * t4 + 1] = acc[nt][3];
                    }
                }
            }
            fence_proxy_async();
            TSTAMP(2);
            DBG(1102 * 1000 + (int)(j & 1023));
            mbar_arrive(smem_u32(&sm.mbar_tiles[s]));
        }
        BSYNC(BAR_END, NTHREADS, 1104 * 1000 + (int)(j & 1023));
    } else if (warp >= 20) {
        const int hw = warp - 20;
        if (hw == 2) {
            // ---------------- MMA issuer (warp 14)
            const uint32_t idesc_ro = make_idesc(128, 64, 0, 0);
            const uint32_t idesc_v = make_idesc(128, 16, 1, 0);
            const uint32_t idesc_c = make_idesc(128, 32, 0, 0);
            const uint32_t idesc_full = make_idesc(128, NFULL, 0, 0);
            for (j = 0; j < nsub; j++) {
                const int p = j & 1, s = j % NST;
                const uint32_t par = (j >> 1) & 1, spar = (j / NST) & 1;
                if (lane == 0) { MWAIT(smem_u32(&sm.mbar_tiles[s]), spar, 1200 * 1000 + (int)(j & 1023)); MWAIT(smem_u32(&sm.mbar_solve[s]), spar, 1202 * 1000 + (int)(j & 1023)); }
                __syncwarp();
                BSYNC(BAR_C3, 160, 1201 * 1000 + (int)(j & 1023));
                TSTAMP(0);
                const uint32_t stg = smem_u32(sm.in[s]);
                if (lane == 0) {
                    tc_fence_after();
#pragma unroll
                    for (int kk = 0; kk < 8; kk++)
                        mma_ts(tmem + TM_O0, tmem + TM_AS + 8 * kk, make_sdesc(stg + (kk >> 2) * 8 * SUB + (kk & 3) * 32, 16, 1024, 2), idesc_ro, kk > 0);
                    mma_commit(smem_u32(&sm.mbar_m1[p]));
                    MWAIT(smem_u32(&sm.mbar_v[s]), spar, 1207 * 1000 + (int)(j & 1023));
                    MWAIT(smem_u32(&sm.mbar_m1[p]), par, 1203 * 1000 + (int)(j & 1023));
                    tc_fence_after();
                    const uint32_t tv = smem_u32(sm.tvb[p][0]);
                    mma_ss(tmem + TM_R0, make_sdesc(stg + 6 * SUB, 8 * SUB, 1024, 2), make_sdesc(tv, 256, 128, 0), idesc_v, 1);
                    mma_ss(tmem + TM_R0, make_sdesc(stg + 6 * SUB, 8 * SUB, 1024, 2), make_sdesc(tv + 512, 256, 128, 0), idesc_v, 1);
                    mma_commit(smem_u32(&sm.mbar_m2a[p]));
                }
                __syncwarp();
                TSTAMP(1);
                BSYNC(BAR_C2A, 160, 1204 * 1000 + (int)(j & 1023));   // UB0 ready
                if (lane == 0) {
                    tc_fence_after();
                    mma_ts(tmem + TM_O1, tmem + TM_UB, make_sdesc(smem_u32(sm.mcor[s]), M_LBO, 128, 0), idesc_c, 1);   // [O1|R1] += UB0^T . M01^T
                    const uint32_t tv = smem_u32(sm.tvb[p][1]);
                    mma_ss(tmem + TM_R1, make_sdesc(stg + 7 * SUB, 8 * SUB, 1024, 2), make_sdesc(tv, 256, 128, 0), idesc_v, 1);
                    mma_ss(tmem + TM_R1, make_sdesc(stg + 7 * SUB, 8 * SUB, 1024, 2), make_sdesc(tv + 512, 256, 128, 0), idesc_v, 1);
                    mma_commit(smem_u32(&sm.mbar_m2b[p]));
                }
                __syncwarp();
                BSYNC(BAR_C2B, 160, 1205 * 1000 + (int)(j & 1023));   // S scaled, UB1 ready
                TSTAMP(2);
                if (lane == 0) {
                    tc_fence_after();
                    const uint32_t bf_s = smem_u32(sm.bfull[s]);
#pragma unroll
                    for (int ks = 0; ks < 2; ks++)
                        mma_ts(tmem + TM_S, tmem + TM_UB + 8 * ks, make_sdesc(bf_s + ks * 2 * BF_LBO, BF_LBO, 128, 0), idesc_full, 1);
                    mma_commit(smem_u32(&sm.mbar_m3[p]));
                }
                __syncwarp();
                TSTAMP(3);
            }
            BSYNC(BAR_END, NTHREADS, 1206 * 1000 + (int)(j & 1023));
        } else if (hw == 3) {
            // ---------------- TMA producer (warp 15)
            for (j = 0; j < nsub; j++) {
                const int p = j & 1;
                if (p) BSYNC(BAR_SFREE + 1, 160, 1300 * 1000 + (int)(j & 1023)); else BSYNC(BAR_SFREE, 160, 1300 * 1000 + (int)(j & 1023));
                TSTAMP_DEP(0);
                issue_tma(j + NST, 0);
                TSTAMP(1);
                if (p) BSYNC(BAR_VFREE + 1, 160, 1304 * 1000 + (int)(j & 1023)); else BSYNC(BAR_VFREE, 160, 1304 * 1000 + (int)(j & 1023));
                issue_tma(j + NST, 1);
                TSTAMP(2);
            }
            BSYNC(BAR_END, NTHREADS, 1301 * 1000 + (int)(j & 1023));
        } else {
            // ---------------- warps 12 (half 0) / 13 (half 1): 16-token solves; warp 13 also computes the norms at landing and M_k
            const int mo = lane >> 3, r = (lane & 7) + 8 * (mo & 1);
            for (j = 0; j < nsub; j++) {
                const int s = j % NST, p = j & 1;
                MWAIT(smem_u32(&sm.mbar_tiles[s]), (j / NST) & 1, 1500 * 1000 + (int)(j & 1023));
                if (j >= 2) MWAIT(smem_u32(hw ? &sm.mbar_m2b[p] : &sm.mbar_m2a[p]), ((j - 2) >> 1) & 1, 1503 * 1000 + (int)(j & 1023));   // tvb[p][hw] consumed
                TSTAMP(0);
                float X[2][4], XT[2][4], Tm[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; nt++)
#pragma unroll
                    for (int jj = 0; jj < 4; jj++) {
                        int rr = gid + 8 * (jj >> 1), cc = nt * 8 + 2 * t4 + (jj & 1);
                        X[nt][jj] = __half2float(sm.xs[s][hw][rr][cc]);
                        XT[nt][jj] = __half2float(sm.xs[s][hw][cc][rr]);
                    }
                neumann16(X, XT, Tm, gid, t4);
                TSTAMP(1);
                // T_V = diag(rk) T diag(bsig) -> tvb[p][hw] (bf16 hi/lo) ; -T_R = -T_V diag(rk) -> A fragments
                uint32_t ahi[4], alo[4];
                const int r0 = 16 * hw + gid, r1 = r0 + 8;
                const float rkr0 = sm.rk[s][r0], rkr1 = sm.rk[s][r1];
#pragma unroll
                for (int nt = 0; nt < 2; nt++)
#pragma unroll
                    for (int rb = 0; rb < 2; rb++) {
                        const int rl = gid + 8 * rb, cl = nt * 8 + 2 * t4, cg = 16 * hw + cl;
                        const float rkr = rb ? rkr1 : rkr0;
                        const float tv0 = rkr * Tm[nt][2 * rb] * sm.bsig[s][cg], tv1 = rkr * Tm[nt][2 * rb + 1] * sm.bsig[s][cg + 1];
                        const uint32_t hi = pack_bf16(tv0, tv1);
                        const float2 hf2 = unpack_bf16(hi);
                        const uint32_t lo = pack_bf16(tv0 - hf2.x, tv1 - hf2.y);
                        const int ko = km_off(rl, cl, 256);
                        *reinterpret_cast<uint32_t*>(&sm.tvb[p][hw][ko]) = hi;
                        *reinterpret_cast<uint32_t*>(&sm.tvb[p][hw][512 + ko]) = lo;
                        const float tr0 = -tv0 * sm.rk[s][cg], tr1 = -tv1 * sm.rk[s][cg + 1];
                        const uint32_t rhi = pack_bf16(tr0, tr1);
                        const float2 rf = unpack_bf16(rhi);
                        ahi[nt * 2 + rb] = rhi; alo[nt * 2 + rb] = pack_bf16(tr0 - rf.x, tr1 - rf.y);
                    }
                TSTAMP(2);
                if (hw == 1) {   // M_k = (-T_R1) N_k -> mcor rows 16..31
                    uint32_t b[4];
                    b[0] = pack_bf16(sm.nk[s][2 * t4][gid], sm.nk[s][2 * t4 + 1][gid]);
                    b[1] = pack_bf16(sm.nk[s][2 * t4 + 8][gid], sm.nk[s][2 * t4 + 9][gid]);
                    b[2] = pack_bf16(sm.nk[s][2 * t4][gid + 8], sm.nk[s][2 * t4 + 1][gid + 8]);
                    b[3] = pack_bf16(sm.nk[s][2 * t4 + 8][gid + 8], sm.nk[s][2 * t4 + 9][gid + 8]);
                    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_bf16(c0, ahi, b[0], b[1]); mma_bf16(c0, alo, b[0], b[1]);
                    mma_bf16(c1, ahi, b[2], b[3]); mma_bf16(c1, alo, b[2], b[3]);
                    unsigned char* mc = sm.mcor[s];
                    *reinterpret_cast<uint32_t*>(mc + km_off(16 + gid, 2 * t4, M_LBO)) = pack_bf16(c0[0], c0[1]);
                    *reinterpret_cast<uint32_t*>(mc + km_off(24 + gid, 2 * t4, M_LBO)) = pack_bf16(c0[2], c0[3]);
                    *reinterpret_cast<uint32_t*>(mc + km_off(16 + gid, 8 + 2 * t4, M_LBO)) = pack_bf16(c1[0], c1[1]);
                    *reinterpret_cast<uint32_t*>(mc + km_off(24 + gid, 8 + 2 * t4, M_LBO)) = pack_bf16(c1[2], c1[3]);
                }
                {   // kt'_h = (-T_R,h) kt_h (rows of this half, in place)
                    unsigned char* kt = sm.in[s] + (hw ? 3 : 1) * SUB;
                    float cres[8][2][4];
                    ktprime_half(kt, ahi, alo, mo, r, cres);
                    __syncwarp();
                    ktprime_store(kt, cres);
                }
                fence_proxy_async();
                DBG(1501 * 1000 + (int)(j & 1023));
                mbar_arrive(smem_u32(&sm.mbar_solve[s]));
                TSTAMP(5);
            }
            BSYNC(BAR_END, NTHREADS, 1502 * 1000 + (int)(j & 1023));
        }
    } else {
        // ================================ TMEM WARPS (0-3) ================================
        const int v = lane + 32 * warp;
        const uint32_t lane_base = (uint32_t)(32 * warp) << 16;
        {
            const uint4* srow = reinterpret_cast<const uint4*>(sp + (int64_t)v * D);
#pragma unroll 1
            for (int c = 0; c < 8; c++) {
                uint4 w0 = srow[2 * c], w1 = srow[2 * c + 1];
                uint32_t pk[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                uint32_t f[16];
#pragma unroll
                for (int i = 0; i < 8; i++) { float2 t2 = unpack_bf16(pk[i]); f[2 * i] = __float_as_uint(t2.x); f[2 * i + 1] = __float_as_uint(t2.y); }
                tmem_st16(tmem + lane_base + TM_S + 16 * c, f);
                tmem_st8(tmem + lane_base + TM_AS + 8 * c, pk);
            }
            tmem_wait_st();
        }
        tc_fence_before();
        BARRIVE(BAR_C3, 160, 1600 * 1000);
        for (j = 0; j < nsub; j++) {
            const int p = j & 1, s = j % NST;
            const int64_t t0 = bos + (int64_t)j * BT;
            const int nvalid = (int)min((int64_t)BT, eos - t0);
            const uint32_t par = (j >> 1) & 1;
            TSTAMP(0);
            MWAIT(smem_u32(&sm.mbar_tiles[s]), (j / NST) & 1, 1601 * 1000 + (int)(j & 1023));
            TSTAMP_DEP(6);
            MWAIT(smem_u32(&sm.mbar_m2a[p]), par, 1608 * 1000 + (int)(j & 1023));
            tc_fence_after();
            {   // UB0 = bf16(R0)
                uint32_t rr[16], pk[8];
                tmem_ld16(tmem + lane_base + TM_R0, rr);
#pragma unroll
                for (int u = 0; u < 8; u++) pk[u] = pack_bf16(__uint_as_float(rr[2 * u]), __uint_as_float(rr[2 * u + 1]));
                tmem_st8(tmem + lane_base + TM_UB, pk);
            }
            tmem_wait_st();
            tc_fence_before();
            BARRIVE(BAR_C2A, 160, 1610 * 1000 + (int)(j & 1023));
            const float* ep = sm.eend[s];
#pragma unroll 1
            for (int c = 0; c < 8; c += 2) {   // S *= E(j), two 16-column chunks per batch
                uint32_t r0[16], r1[16];
                tmem_ld16_nw(tmem + lane_base + TM_S + 16 * c, r0);
                tmem_ld16_nw(tmem + lane_base + TM_S + 16 * c + 16, r1);
                const float4* e4 = reinterpret_cast<const float4*>(ep + 16 * c);
                tmem_wait_ld();
#pragma unroll
                for (int u = 0; u < 4; u++) { const float4 e = e4[u]; r0[4 * u] = __float_as_uint(__uint_as_float(r0[4 * u]) * e.x); r0[4 * u + 1] = __float_as_uint(__uint_as_float(r0[4 * u + 1]) * e.y); r0[4 * u + 2] = __float_as_uint(__uint_as_float(r0[4 * u + 2]) * e.z); r0[4 * u + 3] = __float_as_uint(__uint_as_float(r0[4 * u + 3]) * e.w); }
#pragma unroll
                for (int u = 0; u < 4; u++) { const float4 e = e4[4 + u]; r1[4 * u] = __float_as_uint(__uint_as_float(r1[4 * u]) * e.x); r1[4 * u + 1] = __float_as_uint(__uint_as_float(r1[4 * u + 1]) * e.y); r1[4 * u + 2] = __float_as_uint(__uint_as_float(r1[4 * u + 2]) * e.z); r1[4 * u + 3] = __float_as_uint(__uint_as_float(r1[4 * u + 3]) * e.w); }
                tmem_st16(tmem + lane_base + TM_S + 16 * c, r0);
                tmem_st16(tmem + lane_base + TM_S + 16 * c + 16, r1);
            }
            tmem_wait_st();
            TSTAMP(1);
            MWAIT(smem_u32(&sm.mbar_m2b[p]), par, 1611 * 1000 + (int)(j & 1023));
            tc_fence_after();
            {   // UB1 = bf16(R1)
                uint32_t rr[16], pk[8];
                tmem_ld16(tmem + lane_base + TM_R1, rr);
#pragma unroll
                for (int u = 0; u < 8; u++) pk[u] = pack_bf16(__uint_as_float(rr[2 * u]), __uint_as_float(rr[2 * u + 1]));
                tmem_st8(tmem + lane_base + TM_UB + 8, pk);
            }
            tmem_wait_st();
            tc_fence_before();
            BARRIVE(BAR_C2B, 160, 1602 * 1000 + (int)(j & 1023));
            if (p) BARRIVE(BAR_SFREE + 1, 160, 1605 * 1000 + (int)(j & 1023)); else BARRIVE(BAR_SFREE, 160, 1605 * 1000 + (int)(j & 1023));   // q/k/g rows free
            TSTAMP(2);
            MWAIT(smem_u32(&sm.mbar_m3[p]), par, 1603 * 1000 + (int)(j & 1023));
            tc_fence_after();
            TSTAMP(4);
            {   // O read-out (x rq_t) -> otile [t][v]  (both halves loaded first)
                unsigned char* ob = sm.in[s] + 6 * SUB + v * 2;   // staged in the dead v rows: t < 16 -> half-0 v subs, t >= 16 -> half-1 v subs
                uint32_t ro[2][16];
                tmem_ld16_nw(tmem + lane_base + TM_O0, ro[0]);
                tmem_ld16_nw(tmem + lane_base + TM_O1, ro[1]);
                tmem_wait_ld();
#pragma unroll
                for (int u2 = 0; u2 < 2; u2++)
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const int tb = 16 * u2 + 8 * hh;
                        const float4 q0 = *reinterpret_cast<const float4*>(&sm.rq[s][tb]), q1 = *reinterpret_cast<const float4*>(&sm.rq[s][tb + 4]);
                        const float rqv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                        for (int t = 0; t < 8; t++)
                            *reinterpret_cast<__nv_bfloat16*>(ob + u2 * 8 * SUB + (8 * hh + t) * 256) = __float2bfloat16(__uint_as_float(ro[u2][8 * hh + t]) * rqv[t]);
                    }
            }
            TSTAMP(5);
#pragma unroll 1
            for (int c = 0; c < 8; c += 4) {   // A_S = bf16(S), four chunks per batch
                uint32_t rr[4][16];
#pragma unroll
                for (int u = 0; u < 4; u++) tmem_ld16_nw(tmem + lane_base + TM_S + 16 * (c + u), rr[u]);
                tmem_wait_ld();
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    uint32_t pk[8];
#pragma unroll
                    for (int w = 0; w < 8; w++) pk[w] = pack_bf16(__uint_as_float(rr[u][2 * w]), __uint_as_float(rr[u][2 * w + 1]));
                    tmem_st8(tmem + lane_base + TM_AS + 8 * (c + u), pk);
                }
            }
            tmem_wait_st();
            tc_fence_before();
            TSTAMP(7);
            BARRIVE(BAR_C3, 160, 1604 * 1000 + (int)(j & 1023));
            __syncwarp();
            {
                const int cb = 2 * (lane & 1);
#pragma unroll
                for (int u2 = 0; u2 < 2; u2++) {
                    const int rt = (lane >> 1) + 16 * u2;
                    const unsigned char* orow = sm.in[s] + 6 * SUB + u2 * 8 * SUB + (lane >> 1) * 256 + 64 * warp;
                    const uint4 c0 = *reinterpret_cast<const uint4*>(orow + 16 * cb);
                    const uint4 c1 = *reinterpret_cast<const uint4*>(orow + 16 * (cb + 1));
                    if (rt < nvalid) {
                        uint4* dst = reinterpret_cast<uint4*>(out + (t0 + rt) * row_stride + (int64_t)h * D + 32 * warp + 8 * cb);
                        dst[0] = c0; dst[1] = c1;
                    }
                }
                __syncwarp();
            }
            if (p) BARRIVE(BAR_VFREE + 1, 160, 1612 * 1000 + (int)(j & 1023)); else BARRIVE(BAR_VFREE, 160, 1612 * 1000 + (int)(j & 1023));   // v rows free
            TSTAMP(3);
        }
        {
            uint4* srow = reinterpret_cast<uint4*>(sp + (int64_t)v * D);
#pragma unroll 1
            for (int c = 0; c < 8; c++) {
                uint32_t rr[16], pk[8];
                tmem_ld16(tmem + lane_base + TM_S + 16 * c, rr);
#pragma unroll
                for (int u = 0; u < 8; u++) pk[u] = pack_bf16(__uint_as_float(rr[2 * u]), __uint_as_float(rr[2 * u + 1]));
                srow[2 * c] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
                srow[2 * c + 1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
            }
        }
        tc_fence_before();
        BSYNC(BAR_END, NTHREADS, 1606 * 1000 + (int)(j & 1023));
        if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tmem), "r"(512));
#ifdef KDA_TIMING
        if (tid == 0 && tbuf != nullptr) tbuf[1536 + 4 * blockIdx.x + 2] = clock64();
#endif
    }
}
// ------------------------------------------------------------------ host side
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                                  CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn get_encode() {
    static EncodeTiledFn fn = nullptr;
    if (!fn) {
        void* p = nullptr; cudaDriverEntryPointQueryResult q;
        cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(e == cudaSuccess && p != nullptr, "cuTensorMapEncodeTiled not found");
        fn = (EncodeTiledFn)p;
    }
    return fn;
}
static CUtensorMap make_map(void* base, int64_t T, int64_t H) {
    CUtensorMap m;
    cuuint64_t gdim[3] = {128, (cuuint64_t)H, (cuuint64_t)T};
    cuuint64_t gstride[2] = {(cuuint64_t)128 * 2, (cuuint64_t)H * 128 * 2};
    cuuint32_t box[3] = {64, 1, 16};
    cuuint32_t estr[3] = {1, 1, 1};
    CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, gdim, gstride, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed ", (int)r);
    return m;
}

static int* g_dbg_host = nullptr; static int* g_dbg_dev = nullptr;
static void ensure_dbg() {
    if (!g_dbg_host) {
        cudaHostAlloc((void**)&g_dbg_host, 64 * 16 * sizeof(int), cudaHostAllocMapped);
        memset(g_dbg_host, 0, 64 * 16 * sizeof(int));
        cudaHostGetDevicePointer((void**)&g_dbg_dev, g_dbg_host, 0);
    }
}
std::vector<int64_t> dbg_read() { ensure_dbg(); std::vector<int64_t> v(64 * 16); for (int i = 0; i < 64 * 16; i++) v[i] = ((volatile int*)g_dbg_host)[i]; return v; }
torch::Tensor kda_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, torch::Tensor A_log,
                      torch::Tensor dt_bias, torch::Tensor state, torch::Tensor cu_seqlens, double scale, double lower_bound,
                      int64_t nv, c10::optional<torch::Tensor> out_opt, c10::optional<torch::Tensor> tbuf_opt) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous());
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && state.dtype() == torch::kBFloat16 && state.is_contiguous());
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt64 && q.size(0) == 1 && q.size(3) == D);
    TORCH_CHECK(A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kFloat32 && dt_bias.is_contiguous());
    torch::Tensor out = out_opt.has_value() ? out_opt.value() : torch::empty_like(q);
    TORCH_CHECK(out.is_contiguous());
    int64_t T = q.size(1); int H = q.size(2);
    int N = cu_seqlens.numel() - 1;
    CUtensorMap tq = make_map(q.data_ptr(), T, H), tk = make_map(k.data_ptr(), T, H), tv = make_map(v.data_ptr(), T, H), tg = make_map(g.data_ptr(), T, H), to = make_map(out.data_ptr(), T, H);
    size_t smem = sizeof(Smem) + 1024;
    if (getenv("KDA_SMEM_PAD")) smem += atoi(getenv("KDA_SMEM_PAD"));
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t ae = cudaFuncSetAttribute(kda_fwd_kernel49, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(ae == cudaSuccess, "smem attr failed: ", cudaGetErrorString(ae), " smem=", smem);
        if (getenv("KDA_VERBOSE")) {
            int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kda_fwd_kernel49, NTHREADS, smem);
            cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, kda_fwd_kernel49);
            printf("kda v34: dynamic smem = %zu bytes (sizeof(Smem)=%zu), blocks/SM = %d | numRegs %d static smem %zu local %zu maxThreads %d\n", smem, sizeof(Smem), nb, fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, fa.maxThreadsPerBlock);
            for (size_t s2 : {(size_t)65536, (size_t)98304, (size_t)110592, (size_t)113664}) { cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kda_fwd_kernel49, NTHREADS, s2); printf("   occupancy at %zu B: %d blocks/SM\n", s2, nb); }
        }
        attr_set = true;
    }
    int* dbgp = nullptr;
#ifdef KDA_DEBUG
    ensure_dbg(); dbgp = g_dbg_dev;
#endif
    kda_fwd_kernel49<<<N * H, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        tq, tk, tv, tg, to,
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(state.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        cu_seqlens.data_ptr<int64_t>(), H, (float)scale, (float)lower_bound,
        tbuf_opt.has_value() ? reinterpret_cast<long long*>(tbuf_opt.value().data_ptr<int64_t>()) : nullptr, dbgp);
    cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("kda_fwd", &kda_fwd, "KDA forward v34 (tcgen05)"); m.def("dbg_read", &dbg_read); }
