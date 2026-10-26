"""FP8 block-scale MoE for NVIDIA B200 (sm_100a).

Entry point (destination-passing style): run(...) writes into `output`.
The CUDA source (moe_sm100.cu) is compiled at import time with sm_100a flags.
"""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

_VERBOSE = os.environ.get("MOE_BUILD_VERBOSE", "0") == "1"
_cuda_flags = [
    "-O3",
    "-std=c++17",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "--expt-relaxed-constexpr",
    "-lineinfo",
]
if _VERBOSE:
    _cuda_flags += ["-Xptxas", "-v", "-Xptxas", "--warn-on-spills"]
_extra = os.environ.get("MOE_EXTRA_FLAGS", "").split()
_cuda_flags += _extra
_suffix = ("_v" if _VERBOSE else "") + ("_" + "".join(c for c in "".join(_extra) if c.isalnum()) if _extra else "")
_mod = load(
    name="moe_fp8_block_scale_sm100" + _suffix,
    sources=[os.path.join(_HERE, "moe_sm100.cu")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=_cuda_flags,
    verbose=_VERBOSE,
)


_USE_GRAPH = os.environ.get("MOE_NO_GRAPH", "0") != "1"
_graphs = {}        # key: tuple of data pointers -> (CUDAGraph, keepalive tensors)
_seen = {}          # key -> number of eager calls so far
_MAX_GRAPHS = 8


def _key(tensors, scalars, output):
    return tuple(t.data_ptr() for t in tensors) + tuple(scalars) + (output.data_ptr(),)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
):
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    tensors = (routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
               gemm2_weights, gemm2_weights_scale)
    args = tensors + (int(local_expert_offset), float(routed_scaling_factor), output)
    if not _USE_GRAPH:
        _mod.run(*args)
        return
    k = _key(tensors, (int(local_expert_offset), float(routed_scaling_factor)), output)
    g = _graphs.get(k)
    if g is not None:
        g[0].replay()
        return
    n = _seen.get(k, 0)
    if n == 0:
        # first call for this pointer set: eager (allocates the workspace, builds descriptors)
        _seen[k] = 1
        _mod.run(*args)
        return
    # second call: capture the launch sequence into a graph and replay it
    if len(_graphs) >= _MAX_GRAPHS:
        _graphs.clear()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            _mod.run(*args)
    torch.cuda.current_stream().wait_stream(s)
    _graphs[k] = (graph, tensors, output)
    graph.replay()


def debug_buffers():
    return _mod.debug_buffers()


def debug_trace():
    return _mod.debug_trace()


def debug_times():
    return _mod.debug_times()
