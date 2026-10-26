"""Build + expose the from-scratch kernel as a torch op via cpp_extension."""
import os, sys, torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))

CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
extern "C" cudaError_t gqa_paged_decode_launch(const void*, const void*, const void*, const int32_t*, const int32_t*,
                                               void*, float*, int, int, int, float, cudaStream_t, unsigned long long*);
std::vector<torch::Tensor> gqa_paged_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                            torch::Tensor kv_indptr, torch::Tensor kv_indices, double sm_scale,
                                            c10::optional<torch::Tensor> tbuf) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous());
  TORCH_CHECK(k_cache.size(1) == 1, "page_size must be 1");
  TORCH_CHECK(k_cache.size(3) == 128 && q.size(2) == 128);
  TORCH_CHECK(kv_indptr.dtype() == torch::kInt32 && kv_indices.dtype() == torch::kInt32);
  const int B = q.size(0), HQ = q.size(1), HKV = k_cache.size(2);
  TORCH_CHECK(HQ == HKV * 4, "kernel specialised for GQA group size 4");
  auto out = torch::empty_like(q);
  auto lse = torch::empty({B, HQ}, q.options().dtype(torch::kFloat32));
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t e = gqa_paged_decode_launch(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
                                          kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(),
                                          out.data_ptr(), lse.data_ptr<float>(), B, HKV, (int)kv_indices.numel(), (float)sm_scale, st.stream(),
                                          tbuf.has_value() ? (unsigned long long*)tbuf->data_ptr() : nullptr);
  TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
  return {out, lse};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gqa_paged_decode", &gqa_paged_decode); }
"""

def build(splits=8, threads=128, rounds=2, verbose=False, timing=False, extra=()):
    name = f"gqa_decode_s{splits}_t{threads}_r{rounds}" + ("_timing" if timing else "") + "".join("_" + e.strip("-").replace("=", "") for e in extra)
    bdir = os.path.join(HERE, "build", name)
    os.makedirs(bdir, exist_ok=True)
    cpp_path = os.path.join(bdir, "ext.cpp")
    if not os.path.exists(cpp_path) or open(cpp_path).read() != CPP:
        open(cpp_path, "w").write(CPP)
    mod = load(name=name, sources=[cpp_path, os.path.join(HERE, "gqa_decode_kernel.cu")],
               extra_cuda_cflags=["-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a",
                                  f"-DGQA_SPLITS={splits}", f"-DGQA_THREADS={threads}", f"-DGQA_ROUNDS={rounds}",
                                  "--use_fast_math", "-lineinfo"] + (["-DGQA_TIMING"] if timing else []) + list(extra),
               extra_cflags=["-O3", "-std=c++17"],
               build_directory=bdir, verbose=verbose)
    return mod


CPP_TC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
extern "C" cudaError_t gqa_paged_decode_tc_launch(const void*, const void*, const void*, const int32_t*, const int32_t*,
                                                  void*, float*, int, int, int, float, cudaStream_t);
std::vector<torch::Tensor> gqa_paged_decode_tc(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                                               torch::Tensor kv_indptr, torch::Tensor kv_indices, double sm_scale,
                                               int64_t max_kv_len) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous());
  TORCH_CHECK(k_cache.size(1) == 1, "page_size must be 1");
  TORCH_CHECK(k_cache.size(3) == 128 && q.size(2) == 128);
  TORCH_CHECK(kv_indptr.dtype() == torch::kInt32 && kv_indices.dtype() == torch::kInt32);
  const int B = q.size(0), HQ = q.size(1), HKV = k_cache.size(2);
  TORCH_CHECK(HQ == HKV * 4, "kernel specialised for GQA group size 4");
  if (max_kv_len <= 0) max_kv_len = kv_indices.numel();   // conservative bound without a device sync
  auto out = torch::empty_like(q);
  auto lse = torch::empty({B, HQ}, q.options().dtype(torch::kFloat32));
  auto st = at::cuda::getCurrentCUDAStream();
  cudaError_t e = gqa_paged_decode_tc_launch(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
                                             kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(),
                                             out.data_ptr(), lse.data_ptr<float>(), B, HKV, (int)max_kv_len,
                                             (float)sm_scale, st.stream());
  TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
  return {out, lse};
}
extern "C" cudaError_t gqa_paged_decode_tc_read_timing(unsigned long long*, int);
torch::Tensor gqa_tc_timing() {
  auto t = torch::zeros({64 * 16}, torch::dtype(torch::kInt64));
  gqa_paged_decode_tc_read_timing((unsigned long long*)t.data_ptr(), 64 * 16);
  return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gqa_paged_decode_tc", &gqa_paged_decode_tc); m.def("gqa_tc_timing", &gqa_tc_timing); }
"""

def build_tc(verbose=False, timing=False):
    name = "gqa_decode_tc" + ("_timing" if timing else "")
    bdir = os.path.join(HERE, "build", name)
    os.makedirs(bdir, exist_ok=True)
    cpp_path = os.path.join(bdir, "ext.cpp")
    if not os.path.exists(cpp_path) or open(cpp_path).read() != CPP_TC:
        open(cpp_path, "w").write(CPP_TC)
    return load(name=name, sources=[cpp_path, os.path.join(HERE, "gqa_decode_tc_kernel.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "--use_fast_math", "-lineinfo"] + (["-DGQA_TIMING"] if timing else []),
                extra_cflags=["-O3", "-std=c++17"], build_directory=bdir, verbose=verbose)
