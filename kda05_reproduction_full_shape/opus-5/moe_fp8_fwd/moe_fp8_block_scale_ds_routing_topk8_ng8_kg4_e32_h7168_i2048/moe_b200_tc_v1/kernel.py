"""MoE FP8 block-scale kernel for B200 — JIT build wrapper.

The kernel itself is CUDA C++ in ``moe_kernel.cu``; this module exists only to
compile it with the architecture flag it needs.

Why a Python entry point for a CUDA kernel: the FlashInfer-Bench CUDA builders
compile with a fixed ``-gencode arch=compute_100,code=sm_100``, and ``ptxas``
rejects every ``tcgen05`` instruction for ``sm_100`` ("Instruction
'tcgen05.alloc' not supported on .target 'sm_100'").  Blackwell's native FP8
tensor cores are only reachable from ``sm_100a``.  The contest FAQ names this
exact situation and its workaround:

    Q: Can I pass custom compile flags for CUDA C++ submissions?
    The builders currently do not support custom compile flags.  As a
    workaround, submit a Python solution and compile the CUDA kernel yourself
    within the code (using torch.utils.cpp_extension.load() or
    tvm_ffi.cpp.load()).

    Q: What SM architecture is used?
    The evaluation machine supports sm_100a.  If your CUDA kernel needs
    sm_100a-specific features, specify the architecture explicitly in your
    build flags.

So the solution is still a hand-written CUDA C++ kernel; only the build moved
in-process.  Everything is compiled from source at evaluation time -- no
pre-built binaries are shipped.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

_SRC_NAME = "moe_kernel.cu"
_MODULE = None


def _cache_root() -> Path:
    """A writable, stable directory for the JIT artefacts."""
    env = os.environ.get("MOE_JIT_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or "/tmp"
    return Path(base) / "moe_fp8_b200_jit"


def _build() -> object:
    """Compile moe_kernel.cu for sm_100a and return the loaded module.

    The sources are content-hashed into a stable path.  This matters: the
    FlashInfer-Bench Python builder writes the solution sources to a fresh
    temporary directory in every process, and ``--use-isolated-runner`` starts
    one process per workload.  Keying the source path and the ninja build
    directory on the content hash -- and *not* rewriting the file when it is
    already there -- means ninja sees an unchanged input and the 2nd..19th
    workload just dlopen the existing library instead of recompiling it.
    """
    import tvm_ffi.cpp

    src = (Path(__file__).resolve().parent / _SRC_NAME).read_bytes()
    key = hashlib.sha256(src).hexdigest()[:16]

    root = _cache_root() / key
    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cu = root / _SRC_NAME

    # Write once, atomically; never touch the mtime afterwards.
    if not cu.exists() or cu.read_bytes() != src:
        tmp = root / f".{_SRC_NAME}.{os.getpid()}"
        tmp.write_bytes(src)
        os.replace(tmp, cu)

    # compute_100a / sm_100a.  tvm_ffi prepends this gencode itself (default:
    # whatever nvidia-smi reports, i.e. compute_100), and splits the value on
    # ".", so the "a" has to ride along in the minor field.  Passing
    # -arch=sm_100a via extra_cuda_cflags does *not* work: both targets end up
    # on the command line and the compute_100 one fails.
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"

    flags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-Xptxas=-v",
        "--threads=8",
    ]

    lock = root / ".build.lock"
    for _ in range(600):  # ~10 min worst case; only the first process builds
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            import time

            time.sleep(1.0)
            so = list(build_dir.glob("*.so"))
            if so:
                break
            continue
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        break

    try:
        return tvm_ffi.cpp.load(
            f"moe_fp8_b200_{key}",
            cuda_files=[str(cu)],
            extra_cuda_cflags=flags,
            build_directory=str(build_dir),
        )
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _module() -> object:
    global _MODULE
    if _MODULE is None:
        _MODULE = _build()
    return _MODULE


# Compile at import time.  The benchmark imports the solution before it starts
# timing, and its measurement window only spans CUPTI kernel activity, so the
# build cost is never charged to a measured iteration -- but doing it here also
# means a build failure surfaces as a clean build error rather than as a
# first-iteration outlier.
try:
    _module()
except Exception as _e:  # pragma: no cover - diagnostics only
    print(f"[moe] JIT build at import failed: {_e}", file=sys.stderr)


def moe_run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    output,
):
    """Destination-passing entry point; see the definition for the contract."""
    _module().moe_run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        int(local_expert_offset),
        float(routed_scaling_factor),
        output,
    )
