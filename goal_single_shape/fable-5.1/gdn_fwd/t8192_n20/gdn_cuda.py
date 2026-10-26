import os, torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_defs = os.environ.get("GDN_DEFINES", "")
_build = os.path.join(_here, "build_cuda" + ("_" + _defs.replace("-D", "").replace(" ", "_") if _defs else ""))
os.makedirs(_build, exist_ok=True)
_ext = load(name="gdn_cuda_ext", sources=[os.path.join(_here, "gdn_cuda.cu")], build_directory=_build,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++17", "-lineinfo"] + _defs.split(),
            verbose=False)


_empty = torch.empty(0, dtype=torch.int64, device="cuda")


MODE = int(os.environ.get("GDN_MODE", "0"))


def gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, dbg=None):
    return _ext.gdn_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, float(scale), _empty if dbg is None else dbg, MODE)
