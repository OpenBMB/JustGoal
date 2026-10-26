import ctypes
import os
import subprocess

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "hand224.cu")
_SO = os.path.join(_DIR, "hand224_k.so")
_TL = "/opt/sglang/lib/python3.12/site-packages/tilelang"
_NVCC = "/usr/local/cuda-13.0/bin/nvcc"

if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(_SRC):
    cmd = [
        _NVCC, "-gencode", "arch=compute_100a,code=sm_100a", "-std=c++17",
        "-O3", "--use_fast_math", "-DENABLE_BF16",
        f"-I{_TL}/src", f"-I{_TL}/3rdparty/cutlass/include",
        "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-lcuda",
        _SRC, "-o", _SO,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join(l for l in r.stderr.splitlines()
                    if "registers" in l.lower() or "spill" in l.lower()
                    or "error" in l.lower()))
    if r.returncode != 0:
        raise SystemExit("nvcc failed")

lib = ctypes.CDLL(_SO)
lib.hand224_launch.argtypes = [ctypes.c_void_p] * 11 + [
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.hand224_launch.restype = ctypes.c_int


def attn_bwd(q, k, v, out, do, lse, softmax_scale, causal=True, cfg=None):
    B, N, H, D = q.shape
    H_KV = k.shape[2]
    # k-outer: a CTA owns (b, h_kv, k_tile) and walks all H/H_KV q heads, so
    # dK/dV stay in TMEM for the entire walk and are stored once as bf16 --
    # no accumulator buffer, no memset, no cast pass.  dQ is the only thing
    # reduced (2.21 GB, half of hand63's), straight into its final layout.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    # delta[b, h, n] = rowsum(O * dO); the kernel changes (h, q tile) every
    # iteration so it cannot recompute it inline the way hand63 does.
    delta = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    # dQ reduces into (B, H, D//8, N, 8) so a warp's lanes stay contiguous;
    # hand224_post transposes it back into dq.
    dq_acc = torch.empty((B, H, D // 8, N, 8), device=q.device, dtype=q.dtype)

    rc = lib.hand224_launch(
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()), ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(dk.data_ptr()),
        ctypes.c_void_p(do.data_ptr()), ctypes.c_void_p(dq.data_ptr()),
        ctypes.c_void_p(dv.data_ptr()), ctypes.c_void_p(delta.data_ptr()),
        ctypes.c_void_p(dq_acc.data_ptr()),
        B * H_KV, N // 128,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"hand224_launch failed: {rc}")

    return dq, dk, dv
