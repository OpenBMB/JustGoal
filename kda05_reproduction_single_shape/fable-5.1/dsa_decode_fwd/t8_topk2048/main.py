"""DSA sparse attention (h16, ckv512, kpe64, topk2048, page_size 64) for NVIDIA B200.

Entry point: ``run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)``
returns ``(output, lse)``. The whole operator is a single hand-written CUDA kernel in
``kernel.cu`` (same directory), JIT-compiled from source with torch.utils.cpp_extension.
"""

import hashlib
import os
from pathlib import Path

import torch

_MODULE = None
_WORKSPACE = {}

# workspace geometry must match kernel.cu
_NTILES = 32
_H = 16
_DC = 512
_ML_FLOATS_PER_TILE = 32
_CTR_STRIDE = 32


def _build():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from torch.utils.cpp_extension import load

    src_path = Path(os.environ.get("DSA_KERNEL") or (Path(__file__).resolve().parent / "kernel.cu"))   # DSA_KERNEL: A/B experiments
    src = src_path.read_text()
    trace = os.environ.get("DSA_TRACE") == "1"
    extra_defs = os.environ.get("DSA_DEFS", "").split()  # A/B experiment knobs
    tag = hashlib.sha1((src + ("T" if trace else "") + " ".join(extra_defs)).encode()).hexdigest()[:12]
    # sm_100a is required for the Blackwell-specific instructions; torch picks the flags from this env var.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
    _MODULE = load(
        name=f"dsa_sparse_attn_{tag}",
        sources=[str(src_path)],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
        + (["-DDSA_TRACE"] if trace else []) + extra_defs,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("DSA_VERBOSE") == "1",
    )
    return _MODULE


def _workspace(num_tokens, device):
    """Persistent per-(device, T) scratch: per-tile partials ((m, l) records and fp16 O slices) and the
    per-(token, slice) arrival counters of the fused kernel. The counters are never reset: every call adds
    one per tile and the kernel waits for 32 * epoch; the workspace is re-created before the epoch could wrap."""
    key = (str(device), num_tokens)
    ws = _WORKSPACE.get(key)
    if ws is None or ws[3][0] >= (1 << 23):
        ws_ml = torch.zeros(num_tokens * _NTILES * _ML_FLOATS_PER_TILE, dtype=torch.float32, device=device)
        ws_o = torch.zeros(num_tokens * _NTILES * _H * _DC, dtype=torch.float16, device=device)
        counters = torch.zeros(num_tokens * _NTILES * _CTR_STRIDE, dtype=torch.int32, device=device)
        ws = (ws_ml, ws_o, counters, [0])
        _WORKSPACE[key] = ws
    ws[3][0] += 1
    return ws


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    mod = _build()
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    key = (str(q_nope.device), q_nope.shape[0])
    ws_ml, ws_o, counters, epoch = _workspace(q_nope.shape[0], q_nope.device)
    try:
        out, lse = mod.dsa_sparse_attention(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), ws_ml, ws_o, counters, epoch[0]
        )
    except Exception:
        _WORKSPACE.pop(key, None)   # the arrival counters may be out of step: start a fresh workspace
        raise
    return out, lse
