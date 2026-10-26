"""FP8 block-scale MoE (DeepSeek-V3 routing) -- from-scratch CUDA implementation for NVIDIA B200.

Entry point: run(...) -> output [T, 7168] bf16.
The CUDA extension is built once (keyed by a hash of the sources) into a persistent cache directory and
re-used by later processes without invoking the build system again.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCES = ["moe_host.cu", "moe_common.cuh", "moe_small.cuh", "moe_route.cuh", "moe_v3.cuh"]
_EXT_NAME = "moe_fp8_blockscale_ext"
_ext = None
_ws = {}


def _cache_root():
    return os.environ.get("MOE_BUILD_DIR", os.path.join(os.path.expanduser("~"), ".cache", "moe_fp8_blockscale"))


def _source_hash():
    h = hashlib.sha256()
    for f in _SOURCES:
        with open(os.path.join(_HERE, f), "rb") as fh:
            h.update(f.encode())
            h.update(fh.read())
    h.update(torch.__version__.encode())
    h.update(b"trace" if os.environ.get("MOE_TRACE_N") else b"notrace")
    h.update(b"stream" if os.environ.get("MOE_STREAM_ONLY") else b"full")
    h.update(os.environ.get("MOE_DEFS", "").encode())
    return h.hexdigest()[:16]


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext
    build_dir = os.path.join(_cache_root(), _source_hash())
    so_path = os.path.join(build_dir, _EXT_NAME + ".so")
    if os.path.exists(so_path):
        loader = importlib.machinery.ExtensionFileLoader(_EXT_NAME, so_path)
        spec = importlib.util.spec_from_file_location(_EXT_NAME, so_path, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _ext = mod
        return _ext
    os.makedirs(build_dir, exist_ok=True)
    for f in _SOURCES:
        shutil.copy(os.path.join(_HERE, f), os.path.join(build_dir, f))
    from torch.utils.cpp_extension import load
    _ext = load(
        name=_EXT_NAME,
        sources=[os.path.join(build_dir, "moe_host.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17",
                           "--use_fast_math", "-lineinfo"] + (["-DMOE_TRACE"] if os.environ.get("MOE_TRACE_N") else []) + (["-DMOE_STREAM_ONLY"] if os.environ.get("MOE_STREAM_ONLY") else []) + [d for d in os.environ.get("MOE_DEFS", "").split() if d],
        extra_cflags=["-O3"],
        build_directory=build_dir,
        verbose=bool(int(os.environ.get("MOE_VERBOSE", "0"))),
    )
    return _ext


def _workspace(T, device):
    """Shape-only (data independent) workspace; allocated once per T. All counters/scratch are self-cleaning."""
    key = (T, str(device))
    w = _ws.get(key)
    if w is None:
        w = dict(
            P=torch.zeros(T * 8, 4096, dtype=torch.float32, device=device),
            Cbuf=torch.empty(T * 8, 2048, dtype=torch.uint8, device=device),
            Cs=torch.empty(T * 8, 64, dtype=torch.float32, device=device),
            out_acc=torch.zeros(T, 7168, dtype=torch.float32, device=device),
            counters=torch.zeros(4096, dtype=torch.int32, device=device),
            trace=torch.zeros(int(os.environ.get("MOE_TRACE_N", "0")), dtype=torch.int64, device=device),
            r_ids=torch.empty(T * 8, dtype=torch.uint8, device=device),
            r_w=torch.empty(T * 8, dtype=torch.float32, device=device),
            r_cnt=torch.zeros(96, dtype=torch.int32, device=device),
            r_npairs=torch.empty(T, dtype=torch.int32, device=device),
            r_tok=torch.empty(T * 8, dtype=torch.int32, device=device),
            r_wl=torch.empty(T * 8, dtype=torch.float32, device=device),
            act_stage=torch.empty(296 * 3 * 32 if T <= 128 else 1, 7168, dtype=torch.uint8, device=device),   # grid * QN * NTOK_max rows
            Ag=torch.empty(T * 8 if T > 128 else 1, 7168, dtype=torch.uint8, device=device),
            obuf=torch.empty(T * 8 if T > 128 else 1, 7168, dtype=torch.bfloat16, device=device),
            tok_pairs=torch.empty(T * 8, dtype=torch.int32, device=device),
            As_g=torch.empty(56 * (T * 8 if T > 128 else 1) + 16, dtype=torch.float32, device=device),   # +16: bulk-copy slack
            Cs_t=torch.empty(32 * (T * 8 if T > 128 else 1) + 16, dtype=torch.float32, device=device),
        )
        _ws[key] = w
    return w


LARGE_NTOK = int(os.environ.get("MOE_LARGE_NTOK", "0"))
KS1 = int(os.environ.get("MOE_KS1", "0"))   # 0 = chosen in-kernel from the active expert count
KS2 = int(os.environ.get("MOE_KS2", "0"))


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor):
    ext = _get_ext()
    T = hidden_states.shape[0]
    dev = hidden_states.device
    ws = _workspace(T, dev)
    out = torch.empty(T, 7168, dtype=torch.bfloat16, device=dev)
    ks1, ks2 = (KS1, KS2) if T <= 128 else (1, 1)
    ext.moe_forward(routing_logits.contiguous(), routing_bias.contiguous(), hidden_states, hidden_states_scale.contiguous(),
                    gemm1_weights, gemm1_weights_scale.contiguous(), gemm2_weights, gemm2_weights_scale.contiguous(),
                    int(local_expert_offset), float(routed_scaling_factor), out,
                    ws["P"], ws["Cbuf"], ws["Cs"], ws["out_acc"], ws["counters"], ks1, ks2, ws["trace"],
                    ws["r_ids"], ws["r_w"], ws["r_cnt"], ws["r_npairs"], ws["r_tok"], ws["r_wl"], ws["act_stage"],
                    ws["Ag"], ws["obuf"], ws["tok_pairs"], LARGE_NTOK, ws["As_g"], ws["Cs_t"])
    return out
