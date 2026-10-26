"""MLA paged prefill causal (H=16, d_ckv=512, d_kpe=64, page_size=1) for B200."""
import os, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_mod = None

def _get():
    global _mod
    if _mod is None:
        d = "/root/mla18/build_final"
        os.makedirs(d, exist_ok=True)
        _mod = load(name="mla_final",
                    sources=[os.path.join(_HERE, "mla_kernel.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math",
                                       "-gencode", "arch=compute_100a,code=sm_100a"],
                    extra_cflags=["-O3"],
                    extra_ldflags=["-L/usr/local/cuda-13.1/targets/x86_64-linux/lib", "-lcudart"],
                    build_directory=d, verbose=False)
    return _mod

_SCRATCH = None

def mla_paged_prefill(q_nope, q_pe, ckv_cache, kpe_cache,
                      qo_indptr, kv_indptr, kv_indices, sm_scale,
                      out=None, lse=None):
    """q_nope (Nq,H,512) bf16, q_pe (Nq,H,64) bf16,
       ckv_cache (P,1,512) bf16, kpe_cache (P,1,64) bf16,
       qo_indptr (2,) i32, kv_indptr (2,) i32, kv_indices (kv,) i32  ->
       out (Nq,H,512) bf16, lse (Nq,H) f32.   Batch size 1."""
    global _SCRATCH
    m = _get()
    qlen, H, _ = q_nope.shape
    kvlen = int(kv_indices.numel())
    if out is None:
        out = torch.empty(qlen, H, 512, device=q_nope.device, dtype=torch.bfloat16)
    if lse is None:
        lse = torch.empty(qlen, H, device=q_nope.device, dtype=torch.float32)
    if _SCRATCH is None or _SCRATCH.numel() < qlen * 8:
        _SCRATCH = torch.zeros(qlen * 8, device=q_nope.device, dtype=torch.int64)
    m.mla_prefill_v6(q_nope, q_pe, ckv_cache, kpe_cache, kv_indices,
                     out, lse, _SCRATCH, sm_scale, qlen, kvlen)
    return out, lse


def run(inp, out=None, lse=None):
    return mla_paged_prefill(inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"],
                             inp["qo_indptr"], inp["kv_indptr"], inp["kv_indices"],
                             inp["sm_scale"], out, lse)
