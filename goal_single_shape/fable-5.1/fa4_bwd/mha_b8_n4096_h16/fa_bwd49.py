"""ctypes wrapper for the from-scratch B200 causal MHA backward kernel."""
import ctypes, os, subprocess, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.environ.get("FA_BWD49_SO", os.path.join(_HERE, "libfa_bwd49.so"))
_lib = None


def build(force=False):
    src = os.path.join(_HERE, "fa_bwd49.cu")
    if force or not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(src):
        cmd = ["nvcc", "-O3", "-gencode", "arch=compute_100a,code=sm_100a", "-Xptxas", "-v", "-shared", "-Xcompiler", "-fPIC",
               "-o", _SO, src]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


def lib():
    global _lib
    if _lib is None:
        build()
        _lib = ctypes.CDLL(_SO)
        _lib.fa_bwd_launch.restype = ctypes.c_int
        _lib.fa_bwd_launch.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                                  ctypes.c_void_p, ctypes.c_int]
    return _lib


class Workspace:
    def __init__(self, B, N, H, D, device):
        self.delta = torch.empty(B, H, N, device=device, dtype=torch.float32)
        self.dq_accum = torch.zeros(B, H, D, N, device=device, dtype=torch.float16)   # transposed fp16 accumulator; zero between calls
        self.dq = torch.empty(B, N, H, D, device=device, dtype=torch.bfloat16)
        self.dk = torch.empty(B, N, H, D, device=device, dtype=torch.bfloat16)
        self.dv = torch.empty(B, N, H, D, device=device, dtype=torch.bfloat16)
        self.counters = torch.zeros(B * H, device=device, dtype=torch.int32)


def fa_bwd(q, k, v, o, do, lse, scale, ws: Workspace, skip_pre=False):
    B, N, H, D = q.shape
    assert D == 128 and N % 128 == 0
    for t in (q, k, v, o, do):
        assert t.dtype == torch.bfloat16 and t.is_contiguous()
    assert lse.dtype == torch.float32 and lse.shape == (B, H, N) and lse.is_contiguous()
    err = lib().fa_bwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), do.data_ptr(), lse.data_ptr(),
                              ws.delta.data_ptr(), ws.dq_accum.data_ptr(), ws.dq.data_ptr(), ws.dk.data_ptr(), ws.dv.data_ptr(),
                              ws.counters.data_ptr(), B, N, H, scale, torch.cuda.current_stream().cuda_stream, int(skip_pre))
    if err != 0:
        raise RuntimeError(f"fa_bwd_launch cuda error {err}")
    return ws.dq, ws.dk, ws.dv
