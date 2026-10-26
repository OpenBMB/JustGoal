// Host-side launcher (torch extension binding).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include "moe_small.cuh"
#include "moe_route.cuh"
#include "moe_v3.cuh"

using namespace moe;

static int g_num_sms = -1;
static int g_small_grid[16] = {0};  // per config (index by NTOK/16)

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn g_encode = nullptr;

static EncodeTiledFn get_encode() {
    if (!g_encode) {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult q;
        cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
        TORCH_CHECK(e == cudaSuccess && fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
        g_encode = (EncodeTiledFn)fn;
    }
    return g_encode;
}

// 2D uint8 tensor [rows][cols_bytes] (row-major), box [box_rows][128 B], 128B swizzle
static void make_tmap(CUtensorMap* m, const void* ptr, uint64_t rows, uint64_t cols_bytes, uint32_t box_rows) {
    cuuint64_t dims[2] = {cols_bytes, rows};
    cuuint64_t strides[1] = {cols_bytes};
    cuuint32_t box[2] = {128, box_rows};
    cuuint32_t estr[2] = {1, 1};
    CUresult r = get_encode()(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
}

template <int NTOK, int NCW, int SCL_ROWS, int NSTAGE, int QN, int NBUF = 512 / NTOK, int MINB = 1, int CPS = 1, int NTEAM = 1>
static void launch_small(SmallParams& p, cudaStream_t stream, const void* act_src, int64_t act_rows, const void* cbuf, int64_t c_rows) {
    constexpr size_t smem = small_smem_bytes<NTOK, NCW, SCL_ROWS, NSTAGE, QN, NBUF, MINB, CPS, NTEAM>();
    static int grid = 0;    // per template instantiation
    auto kern = moe_small_kernel<NTOK, NCW, SCL_ROWS, NSTAGE, QN, NBUF, MINB, CPS, NTEAM>;
    make_tmap(&p.tmap_act, act_src, (uint64_t)act_rows, H, NTOK);
    make_tmap(&p.tmap_c, cbuf, (uint64_t)c_rows, I, NTOK);
    if (grid == 0) {
        cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(ea == cudaSuccess, "set max dynamic smem failed (", smem, "): ", cudaGetErrorString(ea));
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        (void)cudaGetLastError();
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NCW * 32 + 128, smem);
        TORCH_CHECK(per_sm >= MINB, "kernel does not fit (smem ", smem, ", per_sm ", per_sm, ")");
        grid = g_num_sms * MINB;
    }
    kern<<<grid, NCW * 32 + 128, smem, stream>>>(p);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "main kernel launch failed (NTOK=", NTOK, ", smem=", smem, "): ", cudaGetErrorString(e));
}
#define CHECK_LAUNCH(name) do { cudaError_t e_ = cudaGetLastError(); TORCH_CHECK(e_ == cudaSuccess, name, " launch failed: ", cudaGetErrorString(e_)); } while (0)

static int g_v3_grid[8] = {0};
template <int NGRP>
static void launch_v3(const moe3::SmallParams& p, cudaStream_t stream) {
    constexpr size_t smem = moe3::small_smem_bytes<NGRP>();
    auto kern = moe3::moe_small_kernel<NGRP>;
    if (g_v3_grid[NGRP] == 0) {
        cudaError_t ea = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        TORCH_CHECK(ea == cudaSuccess, "v3 smem attr failed: ", cudaGetErrorString(ea));
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        (void)cudaGetLastError();
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, moe3::SMALL_THREADS, smem);
        TORCH_CHECK(per_sm >= 1, "v3 kernel does not fit");
        g_v3_grid[NGRP] = per_sm * g_num_sms;
    }
    kern<<<g_v3_grid[NGRP], moe3::SMALL_THREADS, smem, stream>>>(p);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "v3 launch failed: ", cudaGetErrorString(e));
}

int64_t small_grid_size(int64_t ntok) { return g_small_grid[ntok / 16]; }

void moe_forward(torch::Tensor logits, torch::Tensor bias, torch::Tensor A, torch::Tensor A_scale,
                 torch::Tensor W13, torch::Tensor W13s, torch::Tensor W2, torch::Tensor W2s,
                 int64_t local_offset, double scaling, torch::Tensor out,
                 torch::Tensor P, torch::Tensor Cbuf, torch::Tensor Cs, torch::Tensor out_acc, torch::Tensor counters,
                 int64_t ks1, int64_t ks2, torch::Tensor trace,
                 torch::Tensor r_ids, torch::Tensor r_w, torch::Tensor r_cnt /*[96]: count, fill, start*/, torch::Tensor r_npairs,
                 torch::Tensor r_tok, torch::Tensor r_wl, torch::Tensor act_stage,
                 torch::Tensor Ag, torch::Tensor obuf, torch::Tensor tok_pairs, int64_t large_ntok,
                 torch::Tensor As_g, torch::Tensor Cs_t) {
    const at::cuda::OptionalCUDAGuard guard(A.device());
    if (g_num_sms < 0) cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, A.get_device());
    const int T = (int)A.size(0);
    TORCH_CHECK(A.size(1) == H && W13.size(0) == E_LOCAL && W13.size(1) == N1 && W13.size(2) == H);
    TORCH_CHECK(W2.size(1) == H && W2.size(2) == I && logits.size(1) == E_GLOBAL);
    TORCH_CHECK(A_scale.size(0) == KB1 && A_scale.size(1) == T);
    TORCH_CHECK(W13.is_contiguous() && W2.is_contiguous() && A.is_contiguous());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

    SmallParams p;
    make_tmap(&p.tmap_w13, W13.data_ptr(), (uint64_t)E_LOCAL * N1, H, G1_ROWS);
    make_tmap(&p.tmap_w2, W2.data_ptr(), (uint64_t)E_LOCAL * H, I, G2_ROWS);
    p.logits = logits.data_ptr<float>();
    p.bias = reinterpret_cast<const bf16_t*>(bias.data_ptr());
    p.A = reinterpret_cast<const fp8_t*>(A.data_ptr());
    p.A_scale = A_scale.data_ptr<float>();
    p.W13s = W13s.data_ptr<float>();
    p.W2s = W2s.data_ptr<float>();
    p.T = T;
    p.local_offset = (int)local_offset;
    p.scaling = (float)scaling;
    p.P = P.data_ptr<float>();
    p.ks1 = (int)ks1; p.ks2 = (int)ks2;
    p.Cbuf = reinterpret_cast<fp8_t*>(Cbuf.data_ptr());
    p.Cs = Cs.data_ptr<float>();
    p.out_acc = out_acc.data_ptr<float>();
    p.out = reinterpret_cast<bf16_t*>(out.data_ptr());
    p.counters = counters.data_ptr<int>();
    p.trace = trace.numel() > 0 ? reinterpret_cast<long long*>(trace.data_ptr<int64_t>()) : nullptr;
    TORCH_CHECK(counters.numel() >= CNT_TOTAL);
    TORCH_CHECK((ks1 == 0 || ks1 == 1 || ks1 == 2 || ks1 == 4 || ks1 == 8) && (ks2 == 0 || ks2 == 1 || ks2 == 2 || ks2 == 4));
    p.lists_global = T > TMAX_SMALL ? 1 : 0;
    p.obuf = reinterpret_cast<bf16_t*>(obuf.data_ptr());
    p.As_g = As_g.data_ptr<float>();
    p.Cs_t = Cs_t.data_ptr<float>();
    p.act_stage = reinterpret_cast<fp8_t*>(act_stage.data_ptr());
    p.g_rcount = r_cnt.data_ptr<int>();
    p.g_rstart = r_cnt.data_ptr<int>() + 64;
    p.g_tok_list = r_tok.data_ptr<int>();
    p.g_w_list = r_wl.data_ptr<float>();
    p.g_rzero = r_cnt.data_ptr<int>();
    if (p.lists_global) {
        route_kernel<<<(T + 7) / 8, 256, 0, stream>>>(p.logits, p.bias, T, p.scaling, p.local_offset,
                                                     r_ids.data_ptr<uint8_t>(), r_w.data_ptr<float>(), r_cnt.data_ptr<int>(), r_npairs.data_ptr<int>());
        scatter_kernel<<<(T * TOP_K + 255) / 256, 256, 0, stream>>>(r_ids.data_ptr<uint8_t>(), r_w.data_ptr<float>(), T, p.local_offset,
                                                                    r_cnt.data_ptr<int>(), r_cnt.data_ptr<int>() + 32, r_cnt.data_ptr<int>() + 64,
                                                                    r_tok.data_ptr<int>(), r_wl.data_ptr<float>(), tok_pairs.data_ptr<int>());
        CHECK_LAUNCH("route/scatter");
        gather_a_kernel<<<(T * TOP_K + 7) / 8, 256, 0, stream>>>(p.A, r_tok.data_ptr<int>(), r_cnt.data_ptr<int>(), reinterpret_cast<fp8_t*>(Ag.data_ptr()),
                                                                 p.A_scale, T, As_g.data_ptr<float>());
        CHECK_LAUNCH("gather_a");
        TORCH_CHECK((ks1 == 1 || ks1 == 0) && (ks2 == 1 || ks2 == 0), "large-T path expects ks1=ks2=1 (or auto)");
        const int ln = large_ntok > 0 ? (int)large_ntok : (T <= 2048 ? 64 : 128);
        // default: per-tile scale tables (SCL_ROWS=56); SCL_ROWS=1 variants stream the scales with the stages (experimental)
        if (ln == 64) launch_small<64, 8, 56, 7, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));   // 7-stage ring (latency-bound at T~901)
        else if (ln == 33) launch_small<32, 8, 56, 8, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 70) launch_small<64, 8, 56, 6, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 69) launch_small<64, 8, 56, 3, 2, 8, 1, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 133) launch_small<128, 8, 56, 2, 2, 4, 1, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 34) launch_small<32, 8, 56, 9, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 66) launch_small<64, 8, 1, 8, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (ln == 130) launch_small<128, 8, 1, 5, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else launch_small<128, 8, 56, 4, 2>(p, stream, Ag.data_ptr(), Ag.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        combine_kernel<<<(T + 3) / 4, 256, 0, stream>>>(p.obuf, tok_pairs.data_ptr<int>(), T, p.out);
        CHECK_LAUNCH("combine");
    } else if (large_ntok == 0 || large_ntok == 3) {
        // small T: v3 streaming core (TMA -> smem -> ldmatrix -> fp16 mma.sync), 2 CTAs/SM
        moe3::SmallParams q;
        make_tmap(&q.tmap_w13, W13.data_ptr(), (uint64_t)E_LOCAL * N1, H, moe3::G1_ROWS);
        make_tmap(&q.tmap_w2, W2.data_ptr(), (uint64_t)E_LOCAL * H, I, moe3::G2_ROWS);
        q.logits = p.logits; q.bias = p.bias; q.A = p.A; q.A_scale = p.A_scale; q.W13s = p.W13s; q.W2s = p.W2s;
        q.T = T; q.local_offset = p.local_offset; q.scaling = p.scaling;
        q.ks1 = (int)ks1; q.ks2 = (int)ks2; q.trace = p.trace;
        q.P = p.P; q.Cbuf = p.Cbuf; q.Cs = p.Cs; q.out_acc = p.out_acc; q.out = p.out; q.counters = p.counters;
        q.r_ids = r_ids.data_ptr<uint8_t>(); q.r_w = r_w.data_ptr<float>();
        TORCH_CHECK(r_ids.numel() >= T * moe3::TOP_K && r_w.numel() >= T * moe3::TOP_K);
        TORCH_CHECK(Cs.size(1) >= moe3::CS_PER_ROW && counters.numel() >= moe3::CNT_TOTAL);
        launch_v3<2>(q, stream);
    } else {
        TORCH_CHECK(act_stage.size(0) >= (int64_t)g_num_sms * 3 * 32, "act_stage too small");
        const int sn = large_ntok > 0 ? (int)large_ntok : 16;
        if (sn == 16) launch_small<16, 4, 64, 11, 2>(p, stream, act_stage.data_ptr(), act_stage.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (sn == 17) launch_small<16, 8, 64, 10, 2, 32, 1, 1, 2>(p, stream, act_stage.data_ptr(), act_stage.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else if (sn == 18) launch_small<16, 8, 64, 5, 2, 32, 1, 2, 2>(p, stream, act_stage.data_ptr(), act_stage.size(0), Cbuf.data_ptr(), Cbuf.size(0));
        else launch_small<32, 4, 64, 9, 2>(p, stream, act_stage.data_ptr(), act_stage.size(0), Cbuf.data_ptr(), Cbuf.size(0));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_forward", &moe_forward, "fused fp8 block-scale MoE");
    m.def("small_grid_size", &small_grid_size, "grid size used by the small kernel");
}
