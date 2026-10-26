"""PyTorch wrapper for the from-scratch GDN decode kernel (gdn_decode_qk4_v8_d128_k_last, B200).

    output, new_state = gdn_decode(q, k, v, state, A_log, a, dt_bias, b, scale)

Same argument order and semantics as the flashinfer-trace definition; new_state is a fresh tensor (state is not modified).
"""
import ctypes, math, os, subprocess, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def _load():
    global _lib
    if _lib is None:
        src = os.path.join(_HERE, "gdn_decode_final.cu")
        so = os.path.join(_HERE, "build", "gdn_decode_final.so")
        os.makedirs(os.path.dirname(so), exist_ok=True)
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.run(["nvcc", "-O3", "-std=c++17", "-arch=sm_100a", "--shared", "-Xcompiler", "-fPIC", "-o", so, src], check=True)
        _lib = ctypes.CDLL(so)
        _lib.gdn_decode_launch.restype = ctypes.c_int
        _lib.gdn_decode_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_float] + [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p]
    return _lib


@torch.no_grad()
def gdn_decode(q, k, v, state, A_log, a, dt_bias, b, scale=None, output=None, new_state=None):
    B, T, HQ, K = q.shape
    assert T == 1 and HQ == 4 and K == 128 and v.shape[2] == 8 and v.shape[3] == 128
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert state.dtype == torch.float32 and state.shape == (B, 8, 128, 128)
    assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32 and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    q, k, v, state, a, b = (t.contiguous() for t in (q, k, v, state, a, b))
    A_log, dt_bias = A_log.contiguous(), dt_bias.contiguous()
    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    if output is None:
        output = torch.empty(B, 1, 8, 128, dtype=torch.bfloat16, device=q.device)
    if new_state is None:
        new_state = torch.empty_like(state)
    rc = _load().gdn_decode_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr(), A_log.data_ptr(), a.data_ptr(),
                                   dt_bias.data_ptr(), b.data_ptr(), float(scale), output.data_ptr(), new_state.data_ptr(), B,
                                   torch.cuda.current_stream(q.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"gdn_decode_launch failed: cuda error {rc}")
    return output, new_state


# flashinfer-bench style entry point
def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    return gdn_decode(q, k, v, state, A_log, a, dt_bias, b, scale)
