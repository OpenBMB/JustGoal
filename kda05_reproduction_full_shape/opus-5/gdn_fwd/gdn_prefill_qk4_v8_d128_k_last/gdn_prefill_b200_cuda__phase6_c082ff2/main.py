"""Gated Delta Net prefill (k-last state layout) for NVIDIA B200.

definition: gdn_prefill_qk4_v8_d128_k_last

The kernel itself is `gdn_prefill.cu`, compiled here at first use. It is built
from source rather than through the flashinfer-bench CUDA builders because those
compile with `-gencode=arch=compute_100,code=sm_100` and accept no extra flags,
while `tcgen05.*` needs `.target sm_100a` (`ptxas` rejects `tcgen05.alloc`
under `sm_100`). FAQ.md prescribes exactly this workaround. A plain `extern "C"`
entry point called through ctypes keeps `torch/extension.h` out of the build,
so a rebuild takes seconds.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile

DESTINATION_PASSING_STYLE = True

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "gdn_prefill.cu")
_SPLIT_MIN_T = int(os.environ.get("GDN_SPLIT_MIN_T", "245"))

HV, HEAD_DIM = 8, 128

# Programmatic dependent launch for the split path's second kernel.  A launch
# attribute, so it can be flipped per call (dev/x_pdlrt.sh does a paired A/B).
_PDL = int(os.environ.get("GDN_PDL", "1"))

# Dev-only switch for the triangular Gram skip (a warp whose whole output block
# is in the strict upper triangle writes zeros whatever the gemm produces, so a
# quarter of the W/P work is provably dead).  Read per call and applied as a
# +10000 offset on the config id, which the kernel decodes -- so one process can
# alternate it and dev/pairab.py can measure it with the paired protocol.
_TRI = int(os.environ.get("GDN_TRI", "1"))

_lib = None
_ws = None  # persistent workspace for the split path, grown on demand


def _nvcc() -> str:
    for cand in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        if cand and os.path.exists(os.path.join(cand, "bin", "nvcc")):
            return os.path.join(cand, "bin", "nvcc")
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "bin", "nvcc")):
            return os.path.join(CUDA_HOME, "bin", "nvcc")
    except Exception:
        pass
    from shutil import which

    p = which("nvcc")
    if p:
        return p
    raise RuntimeError("nvcc not found")


def _build():
    src = open(_SRC, "rb").read()
    abl = os.environ.get("GDN_ABL")
    cpst = os.environ.get("GDN_CPST")
    flags = [
        *((f"-DGDN_ABL={int(abl)}",) if abl else ()),
        *((f"-DGDN_CPST={int(cpst)}",) if cpst is not None else ()),
        *((f"-DGDN_XPAD={int(os.environ['GDN_XPAD'])}",)
          if os.environ.get("GDN_XPAD") else ()),
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "-lineinfo",
        "--use_fast_math",
        "-Xptxas=-v",
        "--shared",
        "-Xcompiler",
        "-fPIC",
    ]
    key = hashlib.sha256(src + "|".join(flags).encode()).hexdigest()[:16]
    cache = os.environ.get("GDN_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "gdn_prefill_build"
    )
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"gdn_prefill_{key}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        cmd = [_nvcc(), *flags, _SRC, "-o", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                "nvcc failed:\n" + " ".join(cmd) + "\n" + r.stdout + "\n" + r.stderr
            )
        if os.environ.get("GDN_VERBOSE"):
            print(r.stderr, file=sys.stderr)
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.gdn_prefill_launch.restype = ctypes.c_int
    lib.gdn_prefill_launch.argtypes = [ctypes.c_void_p] * 11 + [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.gdn_prefill_smem.restype = ctypes.c_int
    lib.gdn_prefill_smem.argtypes = [ctypes.c_int]
    lib.gdn_prefill_ws_bytes.restype = ctypes.c_long
    lib.gdn_prefill_ws_bytes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return lib


def _pick_cfg(total_seq_len: int, num_seqs: int) -> int:
    """cfg -> (C, VB, num_warps, min_ctas); see GDN_CFGS in gdn_prefill.cu.

    The split-path ids are the `PTF = 1` ones (805/825/855/801): the pre-kernel
    publishes PT = P.Tm, so the main kernel's `U = Tm.rhs` and `O += P.U` fuse
    into a single stage that shares its B-operand fragments, one barrier shorter
    on the per-chunk chain.  Paired A/B against the 7xx equivalents: 1.029
    (705/805), 1.011 (725/825), 1.037 (755/855), 1.033 (701/801).  The 7xx ids
    stay instantiated so that comparison can be re-run (docs/draft.md 20.7).

    The V dimension of the state is independent, so VB trades duplicated
    per-chunk [C,C] work against the CTA count (num_seqs * 8 * 128/VB).
    Chosen from the measured sweep in dev/cfg_sweep.py.
    """
    env = int(os.environ.get("GDN_CFG", "-1"))
    if env >= 0:
        return env
    # Split path (two kernels from one C entry point, so only ~6us of extra
    # span): the chunk body shrinks from ~16 barrier regions to ~5, which is a
    # ~3x shorter critical path for the longest sequence. It only pays once the
    # chunk chain is long enough to amortise the second launch.
    if total_seq_len < _SPLIT_MIN_T:
        # Too short to amortise the split path's second kernel launch (~6 us):
        # one fused kernel, VB=16 (VB=16 maximises the CTA count, which is what
        # fills the GPU when there are few sequences).
        #
        # C is then a pure chunk-count question.  With one sequence the host
        # knows the length exactly, and C=16 covers T <= 16 in one chunk whose
        # [C,C] work is a quarter of C=32's: 4.79 -> 3.9 us at T=6, and seven of
        # the 100 workloads are num_seqs = 1 with T <= 16.  (With several
        # sequences the host cannot see max_seqlen, and T/num_seqs is too poor a
        # proxy to risk a second chunk: 28/2 is 14+14 but could have been 27+1.)
        #
        # C=64 wins when it halves the chunk count rather than saving just one
        # chunk, i.e. exactly when ceil(T/32) is even -- but only with 8 warps:
        # at 4 warps the C=64 instantiation needs 255 registers and spills 68
        # bytes, and the win collapses to 1-2%.  At 8 warps the Gram
        # accumulators halve (Tile<64,64,8> gives each warp 4 tiles, not 8),
        # there are no spills, and the paired A/B is 1.035x at T=35, 1.064x at
        # 49, 1.065x at 61, 1.091x at 97, 1.106x at 116 -- against 0.90x at
        # T=67 and 92, where ceil(T/32) is odd and C=64 saves only one chunk of
        # three.
        if num_seqs == 1 and total_seq_len <= 16:
            return 7
        if num_seqs == 1 and -(-total_seq_len // 32) % 2 == 0:
            return 8
        return 0
    if num_seqs <= 2:
        # Nothing to reorder, so skip the ORDER=1 CTA remap: it costs a
        # dependent cu_seqlens round trip ahead of the state load (~1.5%).
        return 855
    if num_seqs >= 14 and total_seq_len < 350 * num_seqs:
        # Enough sequences, and balanced enough, that the single longest one is
        # not the whole critical path.  Here the cost is dominated by how much
        # total CTA-chunk work the SMs have to chew through, so the win comes
        # from doing less duplicated per-chunk [C,C] work: VB=32 halves the
        # number of V-slice CTAs that each re-derive the same chunk operands.
        # C is then decided by whether the longest sequence is long enough for
        # C=64's shorter chunk chain to beat C=32's halved per-token [C,C] work.
        # Measured over the whole 8192 family, that boundary is *exactly*
        # max_seqlen ~ 2300: 2287, 2247, 2262, 1894, 1296 want C=32 and 2822,
        # 2891, 2921, 3319, 3371 want C=64, with 2300 a tie.  max_seqlen lives
        # in cu_seqlens on the device, and reading it would cost the D2H sync
        # that makes the baseline 167 us on a 6-token workload, so num_seqs has
        # to stand in for it -- imperfectly, since max_seqlen is not monotone in
        # num_seqs here (8192/38 has max 2039, 8192/43 has max 2822).  The
        # boundary that minimises the total mispredict cost over the family is
        # num_seqs >= 38 (it gives up 7% on the two 8192/43 workloads; moving it
        # to 44 instead gives up 12% on the two 8192/38 ones).
        # total_seq_len < 5000 with this many sequences means no sequence can be
        # long (4124/15 has max 912), so C=32 there too.
        return 801 if (num_seqs >= 38 or total_seq_len < 5000) else 825
    # Otherwise the single longest sequence is the critical path: C=64 halves
    # the number of chunk steps, and its much more expensive triangular inverse
    # is now paid in the fully parallel pre-kernel rather than on the chain.
    return 805


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    global _lib
    if _lib is None:
        _lib = _build()

    import torch

    num_seqs = cu_seqlens.shape[0] - 1
    total_seq_len = q.shape[0]
    if scale is None or scale == 0.0:
        scale = 1.0 / (HEAD_DIM ** 0.5)
    cfg = _pick_cfg(total_seq_len, num_seqs)
    if not _TRI:
        cfg += 10000

    # Persistent scratch for the split path. Allocated once (during the
    # harness's warm-up) and only ever grown; its contents are fully rewritten
    # by the pre-kernel on every call.
    global _ws
    need = _lib.gdn_prefill_ws_bytes(cfg % 1000 if cfg >= 1000 else cfg, num_seqs,
                                     total_seq_len)
    if need and (_ws is None or _ws.numel() < need):
        # Round up to 8 MB so a run over many shapes reallocates a handful of
        # times instead of once per new shape; the buffer's address turns out to
        # matter (the split path's cost is bimodal per process, and the only
        # process-level state that moves is this allocation).
        _ws = torch.empty(-(-need // (8 << 20)) * (8 << 20), dtype=torch.uint8,
                          device=q.device)
    ws_ptr = _ws.data_ptr() if need else 0

    rc = _lib.gdn_prefill_launch(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(state.data_ptr() if state is not None else 0),
        ctypes.c_void_p(A_log.data_ptr()),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(dt_bias.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(cu_seqlens.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(new_state.data_ptr()),
        num_seqs,
        total_seq_len,
        float(scale),
        1 if state is not None else 0,
        cfg,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ctypes.c_void_p(ws_ptr),
        _PDL,
    )
    if rc != 0:
        raise RuntimeError(f"gdn_prefill_launch failed: cuda error {rc}")
