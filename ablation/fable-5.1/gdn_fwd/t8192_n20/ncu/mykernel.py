import ctypes, os, math
import numpy as np
import torch

_lib = None
def lib():
    global _lib
    if _lib is None:
        path = os.environ.get("GDN_LIB", "/root/gdn/build/libgdn_fwd.so")
        _lib = ctypes.CDLL(path)
        _lib.gdn_fwd_launch.restype = ctypes.c_int
        _lib.gdn_fwd_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
        _lib.gdn_fwd_workspace_bytes.restype = ctypes.c_int64
        _lib.gdn_fwd_workspace_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return _lib

_ws = {}
def gdn_fwd(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale=None, out=None, new_state=None, cu_host=None, dbg=None):
    T = q.shape[0]; N = cu_seqlens.numel() - 1
    if scale is None or scale == 0.0: scale = 1.0 / math.sqrt(128)
    if cu_host is None: cu_host = cu_seqlens.cpu().numpy().astype(np.int64)
    cu_ptr = cu_host.ctypes.data
    L = lib()
    nbytes = L.gdn_fwd_workspace_bytes(cu_ptr, N)
    key = (q.device.index, nbytes)
    if key not in _ws or _ws[key].numel() < nbytes:
        _ws[key] = torch.empty(nbytes, dtype=torch.uint8, device=q.device)
    ws = _ws[key]
    if out is None: out = torch.empty((T, 8, 128), dtype=torch.bfloat16, device=q.device)
    if new_state is None: new_state = torch.empty((N, 8, 128, 128), dtype=torch.float32, device=q.device)
    nc = ctypes.c_int(0)
    stream = torch.cuda.current_stream().cuda_stream
    rc = L.gdn_fwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), 0 if state is None else state.data_ptr(),
                          A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(), cu_ptr, N, T, float(scale),
                          out.data_ptr(), new_state.data_ptr(), ws.data_ptr(), nbytes, stream, ctypes.byref(nc), 0 if dbg is None else dbg.data_ptr())
    if rc != 0: raise RuntimeError("gdn_fwd_launch rc=%d" % rc)
    return out, new_state
