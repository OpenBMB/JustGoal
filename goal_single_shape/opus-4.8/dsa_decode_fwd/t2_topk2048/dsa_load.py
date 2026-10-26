import os
import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="dsa_ext",
    sources=[os.path.join(_dir, "dsa.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def dsa_decode_fwd(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    out, lse = _ext.dsa_forward(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    return out, lse
