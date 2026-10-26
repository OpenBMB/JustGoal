"""ctypes harness for the from-scratch DSA sparse MLA decode kernel."""
import ctypes
import os
import torch

_LIB = None
TOPK = 2048
HEADS = 16
DCKV = 512
NBAL = 128  # must match NBAL in dsa_kernel.cu


def lib(path="/root/dsa_work/libdsa.so"):
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(path)
        _LIB.dsa_launch.restype = ctypes.c_int
        _LIB.dsa_launch.argtypes = [ctypes.c_void_p] * 10 + [
            ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        _LIB.dsa_nsplit.restype = ctypes.c_int
        _LIB.dsa_nsplit.argtypes = [ctypes.c_int]
    return _LIB


class Runner:
    def __init__(self, T, variant=0, so="/root/dsa_work/libdsa.so"):
        self.L = lib(so)
        self.T = T
        self.variant = variant
        self.nsplit = self.L.dsa_nsplit(variant)
        dev = "cuda"
        # A balanced variant indexes the partials by a global block slot, so the
        # scratch has to cover NBAL slots however few tokens there are.
        slots = max(T * self.nsplit, NBAL)
        self.o_part = torch.empty(slots * HEADS * DCKV, dtype=torch.bfloat16, device=dev)
        self.ml_part = torch.empty(slots * 2 * HEADS, dtype=torch.float32, device=dev)
        self.out = torch.empty(T, HEADS, DCKV, dtype=torch.bfloat16, device=dev)
        self.lse = torch.empty(T, HEADS, dtype=torch.float32, device=dev)
        # arrival flags, one per 128 B line (see CNT_STRIDE), past FLAG_BASE
        self.cnt = torch.zeros(1024 + max(T * self.nsplit, NBAL) * 32,
                               dtype=torch.int32, device=dev)
        torch.cuda.synchronize()

    def __call__(self, q_nope, q_pe, ckv, kpe, idx, sm_scale):
        rc = self.L.dsa_launch(
            ctypes.c_void_p(q_nope.data_ptr()), ctypes.c_void_p(q_pe.data_ptr()),
            ctypes.c_void_p(ckv.data_ptr()), ctypes.c_void_p(kpe.data_ptr()),
            ctypes.c_void_p(idx.data_ptr()), ctypes.c_void_p(self.out.data_ptr()),
            ctypes.c_void_p(self.lse.data_ptr()), ctypes.c_void_p(self.o_part.data_ptr()),
            ctypes.c_void_p(self.ml_part.data_ptr()), ctypes.c_void_p(self.cnt.data_ptr()),
            ctypes.c_int(self.T), ctypes.c_float(sm_scale), ctypes.c_int(self.variant),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"dsa_launch failed rc={rc}")
        return self.out, self.lse
