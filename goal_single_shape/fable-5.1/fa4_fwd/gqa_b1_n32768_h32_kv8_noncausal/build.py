import os, sys; os.environ["TORCH_CUDA_ARCH_LIST"]="10.0a"; os.environ["CUDA_HOME"]="/usr/local/cuda"
import torch
from torch.utils.cpp_extension import load
def build(verbose=False, extra=[], name="fa_fwd", src=None):
    if src is None:
        src = os.environ.get("FA_SRC", "/root/work/fa/fa_fwd.cu")
        if "FA_SRC" in os.environ: name = name + "_" + os.path.basename(src).split(".")[0]
    return load(name=name, sources=[src],
                extra_cuda_cflags=["-gencode","arch=compute_100a,code=sm_100a","-O3","-std=c++17","--expt-relaxed-constexpr","-lineinfo"] + extra,
                extra_ldflags=["-lcuda","-L/usr/local/cuda/lib64"], build_directory=(os.makedirs("/root/work/build_"+name, exist_ok=True) or "/root/work/build_"+name), verbose=verbose)
