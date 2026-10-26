"""MoE FP8 block-scale (DeepSeek-V3 routing) for B200.

Python entry point that JIT-compiles the self-contained CUDA source `kernel.cu` with nvcc
(`-arch=sm_100a`) and launches it through ctypes.  Destination-passing style: `output` is the
last argument.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "kernel.cu")
_LIB = None
_LOCK = threading.Lock()

H = 7168
I = 2048
TOPK = 8
NLOCAL = 32
ROUTE_WARPS = 8
LARGE_T = int(os.environ.get("MOE_LARGE_T", "512"))
TINY_T = int(os.environ.get("MOE_TINY_T", "32"))
TINY_LO = int(os.environ.get("MOE_TINY_LO", "33"))  # tiny 1-CTA/SM config retired (v42: fused 2-CTA/SM config wins)
LARGE_MODE = int(os.environ.get("MOE_LARGE_MODE", "1"))
ROUTE_MAX_BLOCKS = int(os.environ.get("MOE_ROUTE_MAX_BLOCKS", "560"))
ROUTE_ONE_BLOCK_T = int(os.environ.get("MOE_ROUTE_ONE_BLOCK_T", "16"))  # v43b: T <= this routes in one block (no look-back)  # 1 = tcgen05, 3 = mma.sync MT=64, 4 = mma.sync MT=128
SPLIT_MULT = int(os.environ.get("MOE_SPLIT_MULT", "1"))
FIXED_CAP_T = int(os.environ.get("MOE_FIXED_CAP_T", "0"))
FUSED_SMALL = int(os.environ.get("MOE_FUSED_SMALL", "1"))  # v42: fused GEMM1+GEMM2 small-path kernel
FUSED_TINY = int(os.environ.get("MOE_FUSED_TINY", "0"))
U2_ENABLE = int(os.environ.get("MOE_U2", "0"))
SCHED_TARGET = int(os.environ.get("MOE_SCHED_TARGET", "0"))
ROUTE_INKERNEL = int(os.environ.get("MOE_ROUTE_INKERNEL", "1"))  # v58: fused small path routes in its prologue
ROUTE_INKERNEL_T = int(os.environ.get("MOE_ROUTE_INKERNEL_T", "511"))  # ... for T <= this (v60b: cooperative routing for every fused-path T)
COMBINE_ATOMIC = int(os.environ.get("MOE_COMBINE_ATOMIC", "1"))
ROUTE_COOP = int(os.environ.get("MOE_ROUTE_COOP", "1"))
ROUTE_SPLIT = int(os.environ.get("MOE_ROUTE_SPLIT", "1"))  # v63: large path routes with top-k / prefix / scatter kernels (one token per warp)
SCHED_WC = int(os.environ.get("MOE_SCHED_WC", "0"))  # v65: work-conserving two-queue scheduler (1 + 4 * reserve in grids)
S1_MULT = int(os.environ.get("MOE_S1_MULT", "0"))  # v65: GEMM1 split-K target (sub-units >= mult * grid), 0 = legacy
S2_MULT = int(os.environ.get("MOE_S2_MULT", "0"))
ROUTE_CTA = int(os.environ.get("MOE_ROUTE_CTA", "1"))  # v66: CTA-wide routing of one token (8 warps = 8 groups)
EARLY_W = int(os.environ.get("MOE_EARLY_W", "1"))  # v67: first unit's weight stages issued during the table build
SPEC_PF = int(os.environ.get("MOE_SPEC_PF", "0"))  # v68: K-blocks of first-wave GEMM1 weight tiles prefetched into L2 before routing
SPEC_PF_T = int(os.environ.get("MOE_SPEC_PF_T", "32"))  # ... for T >= this
A_BOX = int(os.environ.get("MOE_A_BOX", "1"))
G2_PF = int(os.environ.get("MOE_G2_PF", "0"))  # v73: GEMM2 units a waiting GEMM2 CTA prefetches into L2 (0 = off)
PF_AHEAD = int(os.environ.get("MOE_PF_AHEAD", "0"))  # v73: K-blocks prefetched into L2 ahead of the smem ring (0 = off)
PF_POL = int(os.environ.get("MOE_PF_POL", "0"))  # v73: 0 normal, 1 evict_last, 2 evict_first
G2_N128 = int(os.environ.get("MOE_G2_N128", "0"))  # v71: 128x128 UMMA GEMM2 tiles with 4 TMEM buffers / 6 stages  # v69: UMMA A tiles with a 32/64/128-row TMA box chosen per unit
A_GATHER = int(os.environ.get("MOE_A_GATHER", "0"))  # v62 (rejected, 30% slower at 14107): UMMA GEMM1 gathers activation rows itself via cp.async; routing-side copy is faster  # v60: cooperative in-kernel routing (grid barrier) instead of per-CTA redundant routing  # v59: GEMM2 epilogues reduce multi-pair rows into the output (no combine kernel)  # v56d: -1 = contiguous static ranges (no atomics); >= 0 = dynamic fetch, split target (0 = legacy split rule)  # v53: 2-SM UMMA clusters for T >= 2048 (validated, but 8 % slower than the 1-SM kernel -> off)  # fusion loses with 1 CTA/SM (a waiting CTA idles its SM)  # fixed-capacity routing (v35) measured neutral; off by default


def _find_nvcc():
    cands = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _TCH

        if _TCH:
            cands.append(os.path.join(_TCH, "bin", "nvcc"))
    except Exception:
        pass
    w = shutil.which("nvcc")
    if w:
        cands.append(w)
    cands += ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13/bin/nvcc"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise RuntimeError("nvcc not found")


def _build():
    src = open(_SRC, "rb").read()
    extra = [f for f in os.environ.get("MOE_NVCC_EXTRA", "").split() if f]
    tag = hashlib.sha1(src + b"v73" + " ".join(extra).encode()).hexdigest()[:16]
    cache_root = os.environ.get("MOE_KERNEL_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "fib_moe_kernel"
    )
    try:
        os.makedirs(cache_root, exist_ok=True)
    except Exception:
        cache_root = os.path.join(tempfile.gettempdir(), "fib_moe_kernel")
        os.makedirs(cache_root, exist_ok=True)
    so = os.path.join(cache_root, f"moe_{tag}.so")
    if not os.path.isfile(so):
        nvcc = _find_nvcc()
        tmp = so + f".{os.getpid()}.tmp"
        cmd = [
            nvcc,
            "-O3",
            "-std=c++17",
            "-gencode",
            "arch=compute_100a,code=sm_100a",
            "-lineinfo",
            "-DNDEBUG",
            "-Xcompiler",
            "-fPIC",
            "-shared",
            *extra,
            _SRC,
            "-o",
            tmp,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.moe_run.restype = ctypes.c_int
    lib.moe_run.argtypes = (
        [ctypes.c_void_p] * 8
        + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        + [ctypes.c_void_p] * 15
        + [ctypes.c_int] * 7
        + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        + [ctypes.c_int] * 11
        + [ctypes.c_void_p]
    )
    return lib


def _lib():
    global _LIB
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                _LIB = _build()
    return _LIB


_WS = {}


def _workspace(T, device):
    key = (T, str(device))
    ws = _WS.get(key)
    if ws is not None:
        return ws
    # one warp per token where possible; the routing kernel's look-back requires every block to be
    # co-resident, so cap the block count at ROUTE_MAX_BLOCKS (<= 4 blocks/SM guaranteed by launch bounds)
    tpw = max(1, (T + ROUTE_MAX_BLOCKS * ROUTE_WARPS - 1) // (ROUTE_MAX_BLOCKS * ROUTE_WARPS))
    if T <= ROUTE_ONE_BLOCK_T:
        tpw = (T + ROUTE_WARPS - 1) // ROUTE_WARPS
    tpw = min(16, tpw)
    tok_per_blk = ROUTE_WARPS * tpw
    route_blocks = (T + tok_per_blk - 1) // tok_per_blk
    # fixed-capacity routing (no inter-block look-back) for T <= FIXED_CAP_T: expert e owns rows
    # [e*T, (e+1)*T) of the pair arrays
    fixed_cap = T if T <= FIXED_CAP_T else 0
    p_cap = NLOCAL * T if fixed_cap else T * TOPK
    ws = dict(
        fixed_cap=fixed_cap,
        expert_count=torch.zeros(2 * NLOCAL + 16, dtype=torch.int32, device=device),  # [72] v65 GEMM2 work counter  # [0:32] counts, [32:64] GEMM1 done, [64] arrival ctr, [65] done flag
        blk_pre=torch.zeros(max(route_blocks, (T + ROUTE_WARPS - 1) // ROUTE_WARPS) * NLOCAL, dtype=torch.int32, device=device),
        token_npair=torch.zeros(T, dtype=torch.int32, device=device),
        multi_list=torch.zeros(T, dtype=torch.int32, device=device),
        route_scratch=torch.zeros(T * TOPK * 4, dtype=torch.int32, device=device),
        tpw=tpw,
        route_blocks=route_blocks,
        p_cap=p_cap,
        epoch=0,
        blk_count=torch.zeros(max(route_blocks, (T + ROUTE_WARPS - 1) // ROUTE_WARPS) * NLOCAL, dtype=torch.int32, device=device),
        flags=torch.zeros(route_blocks, dtype=torch.int32, device=device),
        expert_off=torch.empty(NLOCAL + 1, dtype=torch.int32, device=device),
        mtile_prefix=torch.empty(NLOCAL + 1, dtype=torch.int32, device=device),
        pair_token=torch.empty(p_cap, dtype=torch.int32, device=device),
        pair_w=torch.empty(p_cap, dtype=torch.float32, device=device),
        token_pairs=torch.empty(T * TOPK, dtype=torch.int32, device=device),
        act=torch.zeros((p_cap + 128) * I, dtype=torch.uint8, device=device),
        act_in=torch.zeros((p_cap + 128) * H, dtype=torch.uint8, device=device) if (T >= LARGE_T and not A_GATHER) else None,
        act_in_scale=torch.zeros(56 * p_cap, dtype=torch.float32, device=device) if (T >= LARGE_T and not A_GATHER) else None,
        act_scale=torch.empty((I // 64) * p_cap, dtype=torch.float32, device=device),
        pair_out=torch.empty(p_cap * H, dtype=torch.bfloat16, device=device),
        partial=torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=device),
        counters=torch.zeros(4096, dtype=torch.int32, device=device),
        trace=torch.zeros(2 * 64 * 16 + 16, dtype=torch.int64, device=device) if os.environ.get("MOE_TRACE") else None,
    )
    _WS[key] = ws
    return ws


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset,
    routed_scaling_factor,
    output: torch.Tensor,
):
    lib = _lib()
    T = routing_logits.shape[0]
    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    dev = hidden_states.device
    ws = _workspace(T, dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    ws["epoch"] = (ws["epoch"] % 1000000000) + 1

    rl = routing_logits if routing_logits.is_contiguous() else routing_logits.contiguous()
    rb = routing_bias if routing_bias.is_contiguous() else routing_bias.contiguous()
    hs = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    hss = hidden_states_scale if hidden_states_scale.is_contiguous() else hidden_states_scale.contiguous()

    rc = lib.moe_run(
        rl.data_ptr(),
        rb.data_ptr(),
        hs.data_ptr(),
        hss.data_ptr(),
        gemm1_weights.data_ptr(),
        gemm1_weights_scale.data_ptr(),
        gemm2_weights.data_ptr(),
        gemm2_weights_scale.data_ptr(),
        int(T),
        int(local_expert_offset),
        float(routed_scaling_factor),
        output.data_ptr(),
        ws["blk_count"].data_ptr(),
        ws["flags"].data_ptr(),
        ws["expert_off"].data_ptr(),
        ws["mtile_prefix"].data_ptr(),
        ws["pair_token"].data_ptr(),
        ws["pair_w"].data_ptr(),
        ws["token_pairs"].data_ptr(),
        ws["act"].data_ptr(),
        ws["act_scale"].data_ptr(),
        ws["pair_out"].data_ptr(),
        ws["partial"].data_ptr(),
        ws["counters"].data_ptr(),
        ws["trace"].data_ptr() if ws["trace"] is not None else 0,
        ws["act_in"].data_ptr() if ws["act_in"] is not None else 0,
        ws["act_in_scale"].data_ptr() if ws["act_in_scale"] is not None else 0,
        int(ws["p_cap"]),
        int(ws["tpw"]),
        int(ws["route_blocks"]),
        LARGE_MODE if T >= LARGE_T else (2 if TINY_LO <= T <= TINY_T else 0),
        int(ws["epoch"]),
        SPLIT_MULT,
        int(ws["fixed_cap"]),
        ws["expert_count"].data_ptr(),
        (FUSED_TINY if (TINY_LO <= T <= TINY_T) else FUSED_SMALL) if T < LARGE_T else 0,
        ws["blk_pre"].data_ptr(),
        ws["token_npair"].data_ptr(),
        ws["multi_list"].data_ptr(),
        U2_ENABLE,
        SCHED_TARGET,
        ROUTE_INKERNEL if T <= ROUTE_INKERNEL_T else 0,
        COMBINE_ATOMIC if (FUSED_SMALL or T >= LARGE_T) else 0,
        ROUTE_COOP,
        ws["route_scratch"].data_ptr(),
        A_GATHER,
        ROUTE_SPLIT,
        SCHED_WC,
        S1_MULT,
        S2_MULT,
        ROUTE_CTA,
        EARLY_W,
        SPEC_PF if T >= SPEC_PF_T else 0,
        A_BOX,
        G2_N128,
        G2_PF,
        PF_AHEAD,
        PF_POL,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"moe_run failed with CUDA error {rc}")
    return output
