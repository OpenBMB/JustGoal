"""FIB solution entry point for attention_bwd_causal_h16_d128 and attention_bwd_varlen_causal_h16_d128.

run(...) returns (dq, dk, dv). The kernel lives in attn_bwd_kernel.py next to this file (Gluon / Blackwell).
The runner passes inputs positionally in definition order:
  dense : q, k, v, out, dout, lse, sm_scale
  varlen: q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale
"""
import os
import sys
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_kernel_module():
    name = "attn_bwd_kernel"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_kernel = _load_kernel_module()


def _scalar(x, default=None):
    if x is None:
        return default
    if hasattr(x, "item"):
        return x.item()
    return x


def run(q, k, v, out, dout, lse, *args, **kwargs):
    sm_scale = kwargs.get("sm_scale", None)
    cu_seqlens = kwargs.get("cu_seqlens", None)
    max_seqlen = kwargs.get("max_seqlen", None)
    if len(args) == 1:
        sm_scale = args[0]
    elif len(args) == 2:
        cu_seqlens, max_seqlen = args
    elif len(args) >= 3:
        cu_seqlens, max_seqlen, sm_scale = args[0], args[1], args[2]
    sm_scale = float(_scalar(sm_scale, q.shape[-1] ** -0.5))
    if cu_seqlens is None:
        return _kernel.attention_bwd(q, k, v, out, dout, lse, sm_scale)
    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)
    return _kernel.attention_bwd(q, k, v, out, dout, lse, sm_scale, cu_seqlens=cu_seqlens.contiguous(),
                                 max_seqlen=int(_scalar(max_seqlen)))
