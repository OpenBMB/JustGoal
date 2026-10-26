"""FP8 block-scale MoE with DeepSeek-V3 no-aux routing.

Host side for the hand-written CUDA kernels in moe_kernel.cu.  Everything that
depends on the input data is computed on the GPU inside the three kernels; the
only things kept between calls are the compiled module and shape-derived
scratch buffers.
"""
import os
import torch
from torch.utils.cpp_extension import load

H = 7168
I = 2048
E_GLOBAL = 256

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def _ensure_toolchain():
    """Make sure nvcc is reachable (it is not on the default PATH here)."""
    import shutil
    if shutil.which("nvcc"):
        return
    for root in ("/usr/local/cuda", os.environ.get("CUDA_HOME", "")):
        if root and os.path.exists(os.path.join(root, "bin", "nvcc")):
            os.environ.setdefault("CUDA_HOME", root)
            os.environ["PATH"] = os.path.join(root, "bin") + os.pathsep + \
                os.environ.get("PATH", "")
            return


def _ext():
    global _EXT
    if _EXT is None:
        _ensure_toolchain()
        flags = [f for f in ("MOE_NOMMA", "MOE_CPA_CA", "MOE_STRIP_A",
                             "MOE_STRIP_SC", "MOE_STRIP_EPI", "MOE_SKIP_LDM",
                             "MOE_SKIP_MMA", "MOE_SKIP_PROMO")
                 if os.environ.get(f)]
        nm = "moe_fp8_bs" + "".join("_" + f[4:].lower() for f in flags)
        if os.environ.get("MOE_MINBLK"):
            nm += "_mb" + os.environ["MOE_MINBLK"]
        for k in ("MOE_JU1", "MOE_JU2"):
            if os.environ.get(k):
                nm += "_" + k[-3:].lower() + os.environ[k]
        _EXT = load(
            name=nm,
            sources=[os.path.join(_HERE, "moe_kernel.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100a", "--use_fast_math"]
            + ["-D" + f for f in flags]
            + (["-DMOE_MINBLK=" + os.environ["MOE_MINBLK"]]
               if os.environ.get("MOE_MINBLK") else [])
            + ["-D%s=%s" % (k, os.environ[k]) for k in ("MOE_JU1", "MOE_JU2")
               if os.environ.get(k)],
            verbose=bool(os.environ.get("MOE_VERBOSE")),
        )
    return _EXT


_WS = {}
_PAR = [0]

# Tuned (bm, cfg1, cfg2, nc1, nc2) per seq_len bucket.  Everything here is a
# function of the shape only, never of the data.
# use_ffma: 0 = tensor-core path, 1 = CUDA-core path with MAXC=16,
# 2 = CUDA-core path with MAXC=32.  The CUDA-core path is memory bound (and has
# no m-padding) whenever (expert,token) pairs stay well under ~5x the number of
# active experts, which holds for every short sequence.
_TUNE = [
    # (T_max, bm, cfg1, cfg2, nc1, nc2, use_ffma)
    (128, 16, 0, 0, 592, 592, 1),
    (256, 32, 0, 0, 296, 296, 0),
    (1024, 64, 0, 0, 296, 296, 0),
    (1 << 30, 128, 0, 0, 296, 296, 0),
]


def _pick(T):
    for tmax, bm, c1, c2, n1, n2, ff in _TUNE:
        if T <= tmax:
            break
    bm = int(os.environ.get("MOE_BM", 0)) or bm
    n1 = int(os.environ.get("MOE_NC1", 0)) or n1
    n2 = int(os.environ.get("MOE_NC2", 0)) or n2
    if os.environ.get("MOE_CFG1") is not None:
        c1 = int(os.environ["MOE_CFG1"])
    if os.environ.get("MOE_CFG2") is not None:
        c2 = int(os.environ["MOE_CFG2"])
    if os.environ.get("MOE_FFMA") is not None:
        ff = int(os.environ["MOE_FFMA"])
    return bm, n1, n2, c1, c2, ff


def _ws(T, dev):
    w = _WS.get(T)
    if w is None:
        bm, nc1, nc2, cfg1, cfg2, ffma = _pick(T)
        pairs_cap = min(8 * T, 32 * T) + 32 * bm
        tp = (T + 3) & ~3
        w = dict(
            bm=bm, nc1=nc1, nc2=nc2, cfg1=cfg1, cfg2=cfg2, ffma=ffma,
            pairs_cap=pairs_cap,
            cnt=torch.zeros((2, 32), dtype=torch.int32, device=dev),
            # one extra row of slack: the pipeline may prefetch a few ints past
            # the last expert's bucket.
            tokids=torch.zeros((33, tp), dtype=torch.int32, device=dev),
            tokw=torch.zeros((33, tp), dtype=torch.float32, device=dev),
            Cq=torch.zeros((pairs_cap, I), dtype=torch.float8_e4m3fn, device=dev),
            Cs=torch.zeros((pairs_cap, 32), dtype=torch.float32, device=dev),
            Cf=(torch.zeros((pairs_cap, I), dtype=torch.float32, device=dev)
                if ffma else torch.zeros(1, dtype=torch.float32, device=dev)),
            ascg=torch.zeros((33, tp, 56), dtype=torch.float32, device=dev),
        )
        _WS[T] = w
    return w


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    T = routing_logits.shape[0]
    dev = hidden_states.device
    if routing_logits.dtype != torch.float32:
        routing_logits = routing_logits.to(torch.float32)
    routing_logits = routing_logits.contiguous()
    if routing_bias is None:
        routing_bias = torch.zeros(E_GLOBAL, dtype=torch.bfloat16, device=dev)
    elif routing_bias.dtype != torch.bfloat16:
        routing_bias = routing_bias.to(torch.bfloat16)
    routing_bias = routing_bias.reshape(-1).contiguous()
    hidden_states_scale = hidden_states_scale.to(torch.float32).contiguous()
    gemm1_weights_scale = gemm1_weights_scale.to(torch.float32).contiguous()
    gemm2_weights_scale = gemm2_weights_scale.to(torch.float32).contiguous()
    hidden_states = hidden_states.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    if torch.is_tensor(local_expert_offset):
        local_expert_offset = int(local_expert_offset.item())
    if torch.is_tensor(routed_scaling_factor):
        routed_scaling_factor = float(routed_scaling_factor.item())

    w = _ws(T, dev)
    par = _PAR[0]
    _PAR[0] = 1 - par
    return _ext().moe_run(
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        int(local_expert_offset), float(routed_scaling_factor),
        w["cnt"][par], w["cnt"][1 - par], w["tokids"], w["tokw"], w["Cq"], w["Cs"],
        w["ascg"], w["Cf"], w["pairs_cap"], w["bm"], w["nc1"], w["nc2"], w["cfg1"],
        w["cfg2"], w["ffma"])


@torch.no_grad()
def routing_only(routing_logits, routing_bias, hidden_states_scale, cnt, tokids,
                 tokw, ascg, nloc, tslot, local_expert_offset,
                 routed_scaling_factor, EL, want_ascg=0, kind=1):
    """Launch just the routing kernel (used by the Triton GEMM path)."""
    _ext().moe_routing(routing_logits, routing_bias, hidden_states_scale, cnt, tokids,
                       tokw, ascg, nloc, tslot, int(local_expert_offset),
                       float(routed_scaling_factor), int(EL), int(want_ascg),
                       int(kind))
