"""DSA sparse attention (h16, ckv512, kpe64, topk2048, page64) for NVIDIA B200.

Python-language solution that JIT-compiles the CUDA kernel in `kernel.cu` with
`-arch=sm_100a` through tvm-ffi at import time. Entry point `run` is destination-passing style.
"""

import hashlib
import os
from pathlib import Path

import torch
import tvm_ffi.cpp

_HERE = Path(__file__).resolve().parent
_SRC = (_HERE / "kernel.cu").read_text()
_EXTRA = os.environ.get("DSA_NVCC_EXTRA", "").split()  # dev-only knobs, e.g. -DDSA_TRACE
_TAG = hashlib.sha1((_SRC + " ".join(_EXTRA)).encode()).hexdigest()[:12]
_BUILD_DIR = Path(os.environ.get("DSA_BUILD_DIR", Path.home() / ".cache" / "dsa_sparse_attn")) / _TAG
_BUILD_DIR.mkdir(parents=True, exist_ok=True)
# tvm-ffi adds -gencode for the detected compute capability (plain sm_100, which lacks tcgen05);
# restrict code generation to the architecture-specific target.
os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

_mod = tvm_ffi.cpp.load_inline(
    name=f"dsa_sparse_attn_{_TAG}",
    cuda_sources=_SRC,
    functions=["dsa_run", "dsa_set_trace"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"] + _EXTRA,
    build_directory=str(_BUILD_DIR),
)
_dsa_run = _mod.dsa_run

# Persistent workspace for the cross-segment merge: tagged 16-byte records (two 8-byte (data, tag)
# words each). It starts zeroed (tag 0 is never used) and is rewritten by every call; the per-call tag
# tells the merging CTAs which words already belong to the current call.
_NUM_SEG, _NUM_SLICES, _HEADS, _SLICE = 4, 8, 16, 64
_ws = {}
_seq = 0


def _workspace(device, num_tokens):
    key = (str(device), num_tokens)
    w = _ws.get(key)
    if w is None:
        cap = max(num_tokens, 16)
        # One allocation (one set of pages) for the counters, the (M, L) records and the slices.
        n_ctr = cap * _NUM_SLICES // 2            # int32 counters, as int64 words
        n_ml = cap * _NUM_SEG * _HEADS * 2
        n_o = cap * _NUM_SEG * _NUM_SLICES * _HEADS * (_SLICE // 4) * 2
        buf = torch.zeros(n_ctr + n_ml + n_o, dtype=torch.int64, device=device)
        counters = buf[:n_ctr].view(torch.int32)
        ws_ml = buf[n_ctr:n_ctr + n_ml]
        ws_o = buf[n_ctr + n_ml:]
        w = (counters, ws_o, ws_ml)
        _ws[key] = w
    return w


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    global _seq
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    _seq = (_seq % 0x7FFFFFFF) + 1  # 1 .. 2^31-1, never 0
    counters, ws_o, ws_ml = _workspace(q_nope.device, q_nope.shape[0])
    _dsa_run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), output, lse,
             counters, ws_o, ws_ml, _seq)
