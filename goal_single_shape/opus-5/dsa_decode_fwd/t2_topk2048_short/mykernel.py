"""Build + thin wrapper around the custom DSA sparse MLA decode kernel."""
import os
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
HERE = os.path.dirname(os.path.abspath(__file__))

_mod = load(
    name="dsa_custom",
    sources=[os.path.join(HERE, "dsa_kernel.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math", "-lineinfo"],
    extra_cflags=["-O3"],
    build_directory=os.path.join(HERE, "build"),
    verbose=True,
)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    return _mod.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
