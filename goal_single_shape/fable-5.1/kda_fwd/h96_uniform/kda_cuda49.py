import os, torch
from torch.utils.cpp_extension import load
_here = os.path.dirname(os.path.abspath(__file__))
_mod = None
def _get():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
        _mod = load(name="kda_cuda49_ext", sources=[os.path.join(_here, "kda_cuda49.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-DPF_AHEAD=0"] + (["-DKDA_TIMING"] if os.environ.get("KDA_TIMING") else []) + (["-DKDA_DEBUG"] if os.environ.get("KDA_DEBUG") else []),
                    extra_include_paths=[], verbose=False, build_directory=os.path.join(_here, "build49t" if os.environ.get("KDA_TIMING") else ("build49d" if os.environ.get("KDA_DEBUG") else "build49")))
    return _mod
def kda_fwd(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens, scale, lower_bound=-5.0, NV=1, out=None, tbuf=None, **kw):
    return _get().kda_fwd(q, k, v, g, beta, A_log, dt_bias, state, cu_seqlens, float(scale), float(lower_bound), NV, out, tbuf)
