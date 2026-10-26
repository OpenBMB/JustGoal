"""MoE FP8 block-scale (DeepSeek-V3 routing, top-8, 8 groups/4 kept) forward for B200.
run(...) -> output bf16 [T, 7168]. All data-dependent work happens inside run() (two kernel launches)."""
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _load():
    global _ext
    if _ext is not None:
        return _ext
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    cuda_bin = os.path.join(os.environ["CUDA_HOME"], "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    from torch.utils.cpp_extension import load
    build_dir = os.path.join(_HERE, "build" + os.environ.get("MOE_BUILD_TAG", ""))
    os.makedirs(build_dir, exist_ok=True)
    _ext = load(
        name="moe_fp8_blockscale_b200" + os.environ.get("MOE_BUILD_TAG", ""),
        sources=[os.path.join(_HERE, "moe.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17", "-gencode", "arch=compute_100a,code=sm_100a", "--use_fast_math"]
                          + os.environ.get("MOE_EXTRA_FLAGS", "").split(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda"],
        build_directory=build_dir,
        verbose=False,
    )
    return _ext


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor):
    ext = _load()
    return ext.run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                   gemm2_weights, gemm2_weights_scale, int(local_expert_offset), float(routed_scaling_factor))
