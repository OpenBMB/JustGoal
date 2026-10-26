"""Entry point: causal GQA attention backward on B200 (CuTe DSL, tcgen05/TMEM/TMA).

Destination-passing style: run(q, k, v, out, dout, lse, sm_scale, dq, dk, dv).
"""
import os
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

try:  # harness imports this file as <package>.kernel; dev runs import it top-level
    from .bwd_sm100 import AttnBwdSm100
except ImportError:
    from bwd_sm100 import AttnBwdSm100

_CACHE = {}


def _wrap(t):
    return from_dlpack(t, assumed_align=16)


def _get(q, k):
    B, S, H, D = q.shape
    HKV = k.shape[2]
    key = (B, S, H, HKV, D, q.device.index)
    ent = _CACHE.get(key)
    if ent is None:
        dev = q.device
        ws = {
            "dq_acc": torch.empty(B, S, H, D, dtype=(torch.float16 if int(os.environ.get("ABWD_DQ_F16", 1)) else torch.float32), device=dev),
            "lse2": torch.empty(B, H, S, dtype=torch.float32, device=dev),
            "ndelta": torch.empty(B, H, S, dtype=torch.float32, device=dev),
        }
        kernel = AttnBwdSm100(B, S, H, HKV, D)
        ent = {"ws": ws, "kernel": kernel, "compiled": None}
        _CACHE[key] = ent
    return ent


def run(q, k, v, out, dout, lse, sm_scale, dq, dk, dv):
    assert q.dtype == torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert out.is_contiguous() and dout.is_contiguous() and lse.is_contiguous()
    assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()
    ent = _get(q, k)
    ws = ent["ws"]
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    args = [_wrap(q), _wrap(k), _wrap(v), _wrap(out), _wrap(dout), _wrap(lse),
            _wrap(dq), _wrap(dk), _wrap(dv), _wrap(ws["dq_acc"]), _wrap(ws["lse2"]), _wrap(ws["ndelta"]),
            cutlass.Float32(float(sm_scale)), stream]
    if ent["compiled"] is None:
        import os
        opts = os.environ.get("ABWD_COMPILE_OPTS", "")   # dev only, e.g. "--keep-cubin --generate-line-info"
        if opts:
            ent["compiled"] = cute.compile(ent["kernel"], *args, options=opts)
        else:
            ent["compiled"] = cute.compile(ent["kernel"], *args)
    ent["compiled"](*args)
    return dq, dk, dv
