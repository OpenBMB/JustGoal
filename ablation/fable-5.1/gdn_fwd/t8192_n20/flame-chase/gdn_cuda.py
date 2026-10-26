"""ctypes wrapper around libgdn.so (src/gdn_kernel.cu). Single C call launches both kernels."""
import ctypes, os
import torch

_lib = None


def load(path=None):
    global _lib
    if _lib is None:
        path = path or os.environ.get("GDN_LIB", os.path.join(os.path.dirname(__file__), "libgdn.so"))
        _lib = ctypes.CDLL(path)
        _lib.gdn_scratch_bytes.restype = ctypes.c_size_t
        _lib.gdn_scratch_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
        _lib.gdn_forward.restype = ctypes.c_int
        _lib.gdn_forward.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
                                                             ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return _lib


_scratch_cache = {}


_dbg = None


def gdn_forward(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, dbg=False):
    """Full op: raw inputs -> (output bf16 [T,8,128], new_state f32 [N,8,128,128])."""
    lib = load()
    T = q.shape[0]
    N = cu_seqlens.numel() - 1
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert A_log.dtype == torch.float32 and dt_bias.dtype == torch.float32
    assert cu_seqlens.dtype == torch.int64
    for t in (q, k, v, a, b, A_log, dt_bias, cu_seqlens):
        assert t.is_contiguous() and t.is_cuda
    if state is not None:
        assert state.dtype == torch.float32 and state.is_contiguous() and state.is_cuda
    if scale is None or scale == 0.0:
        scale = 1.0 / (128 ** 0.5)
    out = torch.empty((T, 8, 128), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((N, 8, 128, 128), dtype=torch.float32, device=q.device)
    key = (T, N, q.device.index)
    if key not in _scratch_cache:
        _scratch_cache[key] = torch.zeros(lib.gdn_scratch_bytes(T, N), dtype=torch.uint8, device=q.device)   # flags start at 0 (epochs are >= 1)
    scratch = _scratch_cache[key]
    stream = torch.cuda.current_stream(q.device).cuda_stream
    global _dbg
    dbg_ptr = None
    if dbg:
        _dbg = torch.zeros(64, dtype=torch.int64, device=q.device)
        dbg_ptr = _dbg.data_ptr()
    rc = lib.gdn_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), state.data_ptr() if state is not None else None,
                         A_log.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), b.data_ptr(), cu_seqlens.data_ptr(),
                         float(scale), out.data_ptr(), new_state.data_ptr(), scratch.data_ptr(), T, N, stream, dbg_ptr)
    if rc != 0:
        raise RuntimeError("gdn_forward failed")
    return out, new_state
