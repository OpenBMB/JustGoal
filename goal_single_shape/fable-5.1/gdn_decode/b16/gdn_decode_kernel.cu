// GDN decode v41: v10 structure + L2 bulk prefetch (PD) + optional 32B vectors (VEC) + coalesced out stores (OG).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
namespace {
template <int VEC> struct Vec;
template <> struct Vec<4> {
    float v[4];
    __device__ __forceinline__ void load(const float* p) { asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "l"(p)); }
    __device__ __forceinline__ void store(float* p) const { asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory"); }
};
template <> struct Vec<8> {
    float v[8];
    __device__ __forceinline__ void load(const float* p) {
        asm volatile("ld.global.cs.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]) : "l"(p)); }
    __device__ __forceinline__ void store(float* p) const {
        asm volatile("st.global.cs.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(p), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"f"(v[4]),"f"(v[5]),"f"(v[6]),"f"(v[7]) : "memory"); }
};
__device__ __forceinline__ uint2 ld_u2(const void* p) { uint2 r; asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r; }
__device__ __forceinline__ uint2 ld_u2h(const void* p) { uint2 r; asm volatile("ld.global.nc.L2::256B.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r; }
__device__ __forceinline__ unsigned ld_u16h(const void* p) { unsigned short r; asm volatile("ld.global.nc.L2::256B.u16 %0, [%1];" : "=h"(r) : "l"(p)); return r; }
__device__ __forceinline__ float ld_f32h(const float* p) { float r; asm volatile("ld.global.nc.L2::256B.f32 %0, [%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ unsigned ld_u32h(const void* p) { unsigned r; asm volatile("ld.global.nc.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(p)); return r; }
typedef unsigned long long u64;
__device__ __forceinline__ u64 pk2(float lo, float hi) { u64 d; asm("mov.b64 %0, {%1,%2};" : "=l"(d) : "f"(lo), "f"(hi)); return d; }
__device__ __forceinline__ float2 up2(u64 d) { float2 r; asm("mov.b64 {%0,%1}, %2;" : "=f"(r.x), "=f"(r.y) : "l"(d)); return r; }
__device__ __forceinline__ u64 fma2(u64 a, u64 b, u64 c) { u64 d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ void pf_l2(const void* p, unsigned bytes) { asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory"); }
__device__ __forceinline__ uint4 ld_u4(const void* p) { uint4 r; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ unsigned ld_u16(const void* p) { unsigned short r; asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(r) : "l"(p)); return r; }
__device__ __forceinline__ float ld_f32(const float* p) { float r; asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ float rcp_newton(float d) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d)); return fmaf(r, fmaf(-d, r, 1.f), r); }
__device__ __forceinline__ float bf_lo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float gsum16(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 8); x += __shfl_xor_sync(0xffffffffu, x, 4);
    x += __shfl_xor_sync(0xffffffffu, x, 2); x += __shfl_xor_sync(0xffffffffu, x, 1); return x;
}

template <int VEC, int PD, int OG, int PSM, int L2H, int TRED, int PK>
__global__ void __launch_bounds__(128)
gdn_decode_qk4_v8_d128_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
               const __nv_bfloat16* __restrict__ v, float* __restrict__ state,
               const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ a,
               const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ b,
               __nv_bfloat16* __restrict__ out, float scale, int pd)
{
    constexpr int HV = 8, HQ = 4, K = 128, V = 128, NCH = 8 / VEC, NG = 2;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int group = lane >> 4, sub = lane & 15;
    const int bh_ = blockIdx.x >> 3, bi_ = bh_ >> 3, kh_ = (bh_ & 7) >> 1;
    if (PSM == 1 && tid == 0 && blockIdx.x < 7) {   // blocks 0..6: one bulk prefetch each for the whole small-operand set
        const int B = (int)gridDim.x >> 6;
        if (blockIdx.x == 0) pf_l2(q, B * 1024); else if (blockIdx.x == 1) pf_l2(k, B * 1024);
        else if (blockIdx.x == 2) pf_l2(v, B * 1024); else if (blockIdx.x == 3) pf_l2((const char*)v + B * 1024, B * 1024);
        else if (blockIdx.x == 4) pf_l2(a, B * 16); else if (blockIdx.x == 5) pf_l2(b, B * 16); else { pf_l2(A_log, 32); pf_l2(dt_bias, 32); }
    }
    if (PSM == 2 && tid == 0) {   // every block prefetches the whole small-operand set (L2 merges)
        const int B = (int)gridDim.x >> 6;
        pf_l2(q, B * 1024); pf_l2(k, B * 1024); pf_l2(v, B * 2048); pf_l2(a, B * 16); pf_l2(b, B * 16); pf_l2(A_log, 32); pf_l2(dt_bias, 32);
    }
    if (PSM == 3 && tid == 0 && blockIdx.x < 64) {   // first 64 blocks: 1KB slices of q/k/v (64KB) + scalars by block 0
        const int B = (int)gridDim.x >> 6; const int i = blockIdx.x;
        if (i < B) pf_l2((const char*)q + i * 1024, 1024); else if (i < 2 * B) pf_l2((const char*)k + (i - B) * 1024, 1024); else if (i < 4 * B) pf_l2((const char*)v + (i - 2 * B) * 1024, 1024);
        if (i == 0) { pf_l2(a, B * 16); pf_l2(b, B * 16); pf_l2(A_log, 32); pf_l2(dt_bias, 32); }
    }
    if (PSM == 4 && tid == 0) {   // every block prefetches ITS head's q/k rows, its v rows and the scalar lines
        pf_l2(q + (size_t)(bi_ * HQ + kh_) * K, 256); pf_l2(k + (size_t)(bi_ * HQ + kh_) * K, 256); pf_l2(v + (size_t)bh_ * V + ((blockIdx.x & 7) << 4), 32);
        pf_l2(a + bi_ * HV, 16); pf_l2(b + bi_ * HV, 16); pf_l2(A_log, 32); pf_l2(dt_bias, 32);
    }
    if (PD) {   // prefetch the 8KB region of a block pd ahead into L2 (one bulk request per block)
        if (tid == 0) {
            const int tb = (int)blockIdx.x + pd;
            if (tb < (int)gridDim.x) { const float* p = state + (size_t)tb * (16 * K); asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(16 * K * 4) : "memory"); }
        }
    }
    const int bh = blockIdx.x >> 3;
    const int row0 = ((blockIdx.x & 7) << 4) + (warp << 2) + group;
    const int bi = bh >> 3, h = bh & 7, kh = h >> 1;
    float* sbase = state + (size_t)blockIdx.x * (16 * K) + (warp << 2 | group) * K + sub * VEC;
    const __nv_bfloat16* qb = q + (size_t)(bi * HQ + kh) * K + sub * VEC;
    const __nv_bfloat16* kb = k + (size_t)(bi * HQ + kh) * K + sub * VEC;
    const __nv_bfloat16* vb = v + (size_t)bh * V + row0;

    const unsigned aw = ld_u32h(reinterpret_cast<const unsigned*>(a) + ((bi * HV + h) >> 1)), bw = ld_u32h(reinterpret_cast<const unsigned*>(b) + ((bi * HV + h) >> 1));
    const unsigned araw = (h & 1) ? (aw >> 16) : (aw & 0xffffu), braw = (h & 1) ? (bw >> 16) : (bw & 0xffffu);
    const float Al = L2H ? ld_f32h(A_log + h) : ld_f32(A_log + h), db = L2H ? ld_f32h(dt_bias + h) : ld_f32(dt_bias + h);
    unsigned qraw[4], kraw[4];
    if (VEC == 8) { uint4 t = ld_u4(qb); qraw[0] = t.x; qraw[1] = t.y; qraw[2] = t.z; qraw[3] = t.w; uint4 u = ld_u4(kb); kraw[0] = u.x; kraw[1] = u.y; kraw[2] = u.z; kraw[3] = u.w; }
    else { uint2 t0 = ld_u2h(qb), t1 = ld_u2h(qb + 64); qraw[0] = t0.x; qraw[1] = t0.y; qraw[2] = t1.x; qraw[3] = t1.y;
           uint2 u0 = ld_u2h(kb), u1 = ld_u2h(kb + 64); kraw[0] = u0.x; kraw[1] = u0.y; kraw[2] = u1.x; kraw[3] = u1.y; }
    const uint2 vw = ld_u2h(v + (size_t)bh * V + (row0 - group));
    const unsigned vr0 = group ? (vw.x >> 16) : (vw.x & 0xffffu), vr1 = group ? (vw.y >> 16) : (vw.y & 0xffffu);
    Vec<VEC> s[NG][NCH];
    #pragma unroll
    for (int gi = 0; gi < NG; ++gi)
        #pragma unroll
        for (int c = 0; c < NCH; ++c) s[gi][c].load(sbase + gi * 2 * K + c * 64);

    const float x = bf_lo(araw) + db;
    const float sp = (x > 20.f) ? x : log1pf(expf(x));
    const float g = expf(-expf(Al) * sp);
    const float beta = rcp_newton(1.f + expf(-bf_lo(braw)));
    const float nbg = -beta * g;
    const float bv0 = beta * bf_lo(vr0), bv1 = beta * bf_lo(vr1);
    float qq[8], kk[8];
    #pragma unroll
    for (int e = 0; e < 4; ++e) { qq[2*e] = bf_lo(qraw[e]); qq[2*e+1] = bf_hi(qraw[e]); kk[2*e] = bf_lo(kraw[e]); kk[2*e+1] = bf_hi(kraw[e]); }
    float qk = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) qk = fmaf(qq[e], kk[e], qk);
    qk = gsum16(qk);

    float sv[NG][8];
    #pragma unroll
    for (int gi = 0; gi < NG; ++gi)
        #pragma unroll
        for (int c = 0; c < NCH; ++c)
            #pragma unroll
            for (int e = 0; e < VEC; ++e) sv[gi][c * VEC + e] = s[gi][c].v[e];
    float ks[NG], qs[NG];
    #pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
        float pk0 = 0.f, pk1 = 0.f, pq0 = 0.f, pq1 = 0.f;
        #pragma unroll
        for (int e = 0; e < 8; e += 2) { pk0 = fmaf(sv[gi][e], kk[e], pk0); pk1 = fmaf(sv[gi][e+1], kk[e+1], pk1); pq0 = fmaf(sv[gi][e], qq[e], pq0); pq1 = fmaf(sv[gi][e+1], qq[e+1], pq1); }
        ks[gi] = pk0 + pk1; qs[gi] = pq0 + pq1;
    }
    float gs[NG][8];
    #pragma unroll
    for (int gi = 0; gi < NG; ++gi)
        #pragma unroll
        for (int e = 0; e < 8; ++e) gs[gi][e] = g * sv[gi][e];
    float o[NG];
    if (TRED) {
        const bool b3 = (sub & 8) != 0, b2 = (sub & 4) != 0;
        const float A = ks[0], B = qs[0], C = ks[1], D = qs[1];
        const float r0 = __shfl_xor_sync(0xffffffffu, b3 ? A : C, 8), r1 = __shfl_xor_sync(0xffffffffu, b3 ? B : D, 8);
        const float X = (b3 ? C : A) + r0, Y = (b3 ? D : B) + r1;
        const float r2 = __shfl_xor_sync(0xffffffffu, b2 ? X : Y, 4);
        float Z = (b2 ? Y : X) + r2;
        Z += __shfl_xor_sync(0xffffffffu, Z, 2); Z += __shfl_xor_sync(0xffffffffu, Z, 1);
        const int base = group * 16;
        ks[0] = __shfl_sync(0xffffffffu, Z, base); ks[1] = __shfl_sync(0xffffffffu, Z, base + 8);
        qs[0] = __shfl_sync(0xffffffffu, Z, base + 4); qs[1] = __shfl_sync(0xffffffffu, Z, base + 12);
    } else {
        ks[0] = gsum16(ks[0]); ks[1] = gsum16(ks[1]);
        qs[0] = gsum16(qs[0]); qs[1] = gsum16(qs[1]);
    }
    #pragma unroll
    for (int gi = 0; gi < NG; ++gi) {
        const float delta = fmaf(nbg, ks[gi], gi == 0 ? bv0 : bv1);
        Vec<VEC> t[NCH];
        if (PK) {
            const u64 d2 = pk2(delta, delta);
            #pragma unroll
            for (int c = 0; c < NCH; ++c) {
                #pragma unroll
                for (int e = 0; e < VEC; e += 2) {
                    const u64 r = fma2(pk2(kk[c * VEC + e], kk[c * VEC + e + 1]), d2, pk2(gs[gi][c * VEC + e], gs[gi][c * VEC + e + 1]));
                    const float2 f = up2(r); t[c].v[e] = f.x; t[c].v[e + 1] = f.y;
                }
                t[c].store(sbase + gi * 2 * K + c * 64);
            }
        } else {
            #pragma unroll
            for (int c = 0; c < NCH; ++c) {
                #pragma unroll
                for (int e = 0; e < VEC; ++e) t[c].v[e] = fmaf(kk[c * VEC + e], delta, gs[gi][c * VEC + e]);
                t[c].store(sbase + gi * 2 * K + c * 64);
            }
        }
        o[gi] = scale * fmaf(g, qs[gi], qk * delta);
    }
    if (OG) {
        // rows of this warp: row0w + {0,1,2,3}; lane 0 holds rows +0 (o[0]), +2 (o[1]); lane 16 holds +1, +3.
        const float o1 = __shfl_sync(0xffffffffu, o[0], 16), o3 = __shfl_sync(0xffffffffu, o[1], 16);
        if (lane == 0) {
            __nv_bfloat162 p0 = __floats2bfloat162_rn(o[0], o1), p1 = __floats2bfloat162_rn(o[1], o3);
            uint2 pk; pk.x = *reinterpret_cast<unsigned*>(&p0); pk.y = *reinterpret_cast<unsigned*>(&p1);
            *reinterpret_cast<uint2*>(out + (size_t)bh * V + row0) = pk;
        }
    } else {
        if (sub == 0) { out[(size_t)bh * V + row0] = __float2bfloat16(o[0]); out[(size_t)bh * V + row0 + 2] = __float2bfloat16(o[1]); }
    }
}

template <int VEC, int PD, int OG, int PSM, int L2H, int TRED, int PK>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state, torch::Tensor A_log,
            torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b, torch::Tensor out, double scale, int pd)
{
    const int B = q.size(0);
    gdn_decode_qk4_v8_d128_kernel<VEC, PD, OG, PSM, L2H, TRED, PK><<<B * 64, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()), state.data_ptr<float>(), A_log.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (float)scale, pd);
}
} // namespace
// In-place state update; `out` must be a preallocated [B,1,8,128] bf16 tensor.
// pd: state prefetch distance in blocks (0 = off). psm: 1 = prefetch small-operand set (needs 16B-aligned tensors, even B).
void gdn_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state, torch::Tensor A_log,
                torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b, torch::Tensor out, double scale, int64_t pd, int64_t psm)
{
    TORCH_CHECK(q.dim() == 4 && q.size(1) == 1 && q.size(2) == 4 && q.size(3) == 128, "q must be [B,1,4,128]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q");
    TORCH_CHECK(v.dim() == 4 && v.size(1) == 1 && v.size(2) == 8 && v.size(3) == 128, "v must be [B,1,8,128]");
    TORCH_CHECK(state.dim() == 4 && state.size(0) == q.size(0) && state.size(1) == 8 && state.size(2) == 128 && state.size(3) == 128, "state must be [B,8,128,128]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16, "q/k/v bf16");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 && a.numel() == q.size(0) * 8 && b.numel() == q.size(0) * 8, "a/b bf16 [B,1,8]");
    TORCH_CHECK(state.scalar_type() == at::kFloat && A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat, "state/A_log/dt_bias fp32");
    TORCH_CHECK(A_log.numel() == 8 && dt_bias.numel() == 8, "A_log/dt_bias [8]");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == v.sizes(), "out must be bf16 [B,1,8,128]");
    bool aligned = true;
    for (auto& t : {q, k, v, state, A_log, a, dt_bias, b, out}) { TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "tensors must be CUDA + contiguous"); aligned &= ((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0); }
    const int B = q.size(0);
    const bool use_psm = psm != 0 && aligned && B >= 2 && (B % 2) == 0;
    const int dist = (int)pd;
    if (dist > 0) { if (use_psm) launch<4, 1, 1, 3, 1, 1, 0>(q, k, v, state, A_log, a, dt_bias, b, out, scale, dist); else launch<4, 1, 1, 0, 1, 1, 0>(q, k, v, state, A_log, a, dt_bias, b, out, scale, dist); }
    else          { if (use_psm) launch<4, 0, 1, 3, 1, 1, 0>(q, k, v, state, A_log, a, dt_bias, b, out, scale, 0);    else launch<4, 0, 1, 0, 1, 1, 0>(q, k, v, state, A_log, a, dt_bias, b, out, scale, 0); }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn_decode", &gdn_decode, "GDN decode qk4/v8/d128 k-last (in-place state)"); }
