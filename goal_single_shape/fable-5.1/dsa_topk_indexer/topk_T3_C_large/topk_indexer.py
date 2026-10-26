"""Python entry for the from-scratch DSA top-k indexer kernel (dsa_topk_kernel.cu)."""
import os

import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
TOPK = 2048

_ext = None
_scratch = {}
_grid = None


def _load():
    global _ext, _grid
    if _ext is None:
        variant = os.environ.get("DSA_VARIANT", "")  # "" (volatile) or "relaxed" (ld/st.relaxed.gpu)
        _ext = load(
            name="dsa_topk_indexer_scratch" + ("_" + variant if variant else ""),
            sources=[os.path.join(ROOT, "dsa_topk_kernel.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas", "-v"]
            + (["-DDSA_RELAXED"] if variant == "relaxed" else [])
            + (["-DDSA_NO_TAIL"] if variant == "notail" else []),
            verbose=True,
        )
        _grid = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _ext


def _get_scratch(B, max_pages, device):
    key = (B, max_pages, str(device))
    if key not in _scratch:
        # key scratch: zero means "not written yet"; the tail re-zeroes its batch after every call
        _scratch[key] = torch.zeros(B * max_pages * 64, dtype=torch.int32, device=device)
    return _scratch[key]


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    ext = _load()
    B, max_pages = block_table.shape
    out = torch.empty((B, TOPK), dtype=torch.int32, device=q_index_fp8.device)
    keys = _get_scratch(B, max_pages, q_index_fp8.device)
    # one work item per CTA: #long batches + their #pages + #short batches must be <= #SMs (true for this
    # workload; a persistent item loop would be the upgrade path for bigger batches)
    ext.dsa_topk_indexer(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, out, keys, _grid, None,
                         None)
    return out


def run_diag(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """Debug: run with a polling watchdog and per-CTA diagnostics (grid, 8) int32."""
    ext = _load()
    B, max_pages = block_table.shape
    out = torch.empty((B, TOPK), dtype=torch.int32, device=q_index_fp8.device)
    keys = _get_scratch(B, max_pages, q_index_fp8.device)
    diag = torch.zeros((_grid, 8), dtype=torch.int32, device=q_index_fp8.device)
    ext.dsa_topk_indexer(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, out, keys, _grid, None,
                         diag)
    return out, diag, keys


def run_timed(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """Same as run, but also returns a (grid, 16) int64 tensor of per-CTA phase timestamps (debug only)."""
    ext = _load()
    B, max_pages = block_table.shape
    out = torch.empty((B, TOPK), dtype=torch.int32, device=q_index_fp8.device)
    keys = _get_scratch(B, max_pages, q_index_fp8.device)
    dbg = torch.zeros((_grid, 16), dtype=torch.int64, device=q_index_fp8.device)
    ext.dsa_topk_indexer(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, out, keys, _grid, dbg,
                         None)
    return out, dbg
