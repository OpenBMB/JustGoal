"""ctypes wrapper for our from-scratch FA4-style varlen backward (B200)."""
import ctypes, os, torch

_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfa4bwd.so"))
        _lib.fa4_bwd_launch.restype = ctypes.c_int
        _lib.fa4_bwd_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float] + \
            [ctypes.c_void_p] * 3 + [ctypes.c_void_p] * 7 + [ctypes.c_int, ctypes.c_void_p]
    return _lib

_ws = {}
def workspace(T, H, D, n_docs, device):
    key = (T, H, D, n_docs, str(device))
    if key not in _ws:
        _ws[key] = dict(
            dq_accum=torch.empty(T, H, D, dtype=torch.float16, device=device),   # fp16: f16x2 reduce-adds are 3.4x faster than f32
            # augmentation slices (H, Tp, 16) bf16 (per-doc 64-padded rows): [-lse/scale (hi, lo), 0...] and [-rowsum(dO*O) (hi, lo), 0...]
            qaug=torch.zeros(H, (T + 63) // 64 * 64 + 64 * n_docs, 16, dtype=torch.bfloat16, device=device),
            doaug=torch.zeros(H, (T + 63) // 64 * 64 + 64 * n_docs, 16, dtype=torch.bfloat16, device=device),
            pad_start=torch.zeros(n_docs + 1, dtype=torch.int32, device=device),
            items=torch.empty((T // 128 + n_docs + 1), dtype=torch.int32, device=device),
            n_pairs=torch.zeros(1, dtype=torch.int32, device=device),
            counter=torch.zeros(1, dtype=torch.int32, device=device),
        )
    return _ws[key]

_num_sms = None
def fa4_bwd(q, k, v, o, do, lse, cu_seqlens, softmax_scale, dq=None, dk=None, dv=None, num_sms=None):
    """q,k,v,o,do: (T,H,128) bf16 contiguous; lse: (H,T) fp32; cu_seqlens: int32 (n_docs+1). Returns dq,dk,dv bf16."""
    global _num_sms
    T, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    for x in (q, k, v, o, do):
        assert x.is_contiguous() and x.shape == (T, H, D)
    assert lse.shape == (H, T) and lse.dtype == torch.float32 and lse.is_contiguous()
    n_docs = cu_seqlens.numel() - 1
    ws = workspace(T, H, D, n_docs, q.device)
    if dq is None: dq = torch.empty_like(q)
    if dk is None: dk = torch.empty_like(k)
    if dv is None: dv = torch.empty_like(v)
    if num_sms is None:
        if _num_sms is None:
            _num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_sms = _num_sms
    rc = lib().fa4_bwd_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), do.data_ptr(), lse.data_ptr(),
                              cu_seqlens.data_ptr(), n_docs, T, H, float(softmax_scale),
                              dq.data_ptr(), dk.data_ptr(), dv.data_ptr(),
                              ws["dq_accum"].data_ptr(), ws["qaug"].data_ptr(), ws["doaug"].data_ptr(),
                              ws["items"].data_ptr(), ws["n_pairs"].data_ptr(), ws["counter"].data_ptr(), ws["pad_start"].data_ptr(), num_sms,
                              torch.cuda.current_stream(q.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"fa4_bwd_launch failed rc={rc}")
    return dq, dk, dv
