"""ctypes bindings for the shared libraries built from dsa_kernel.cu by job.sh.

`libdsa.so` is the kernel under test; `libdsa_<variant>.so` are tuning variants
built with different -D flags (see job.sh) and benchmarked alongside it.
"""
import ctypes
import glob
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed: cudaError {err}")


class DsaLib:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)[len("libdsa"):-len(".so")].lstrip("_") or "new"
        lib = ctypes.CDLL(path)
        lib.dsa_launch.restype = ctypes.c_int
        lib.dsa_launch.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.dsa_config.restype = ctypes.c_char_p
        self.has_tools = hasattr(lib, "dsa_marker_launch")  # marker / probes / flush live in dsa_kernel.cu builds
        if self.has_tools:
            lib.dsa_marker_launch.restype = ctypes.c_int
            lib.dsa_marker_launch.argtypes = [ctypes.c_void_p]
            lib.dsa_probe_launch.restype = ctypes.c_int
            lib.dsa_probe_launch.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
            lib.dsa_l2_flush_launch.restype = ctypes.c_int
            lib.dsa_l2_flush_launch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p]
        self.has_mma_probe = hasattr(lib, "dsa_mma_probe_launch")
        if self.has_mma_probe:
            lib.dsa_mma_probe_launch.restype = ctypes.c_int
            lib.dsa_mma_probe_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        if hasattr(lib, "dsa_timing_read"):
            lib.dsa_timing_read.restype = ctypes.c_int
            lib.dsa_timing_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib = lib
        self.config = lib.dsa_config().decode()
        self.is_timing = "timing=1" in self.config
        self.is_diagnostic = self.is_timing or "skip_compute=1" in self.config

    def timing_read(self):
        """Timing build only: (32 CTAs x 7 stamps) globaltimer ns and SM cycles as int64 arrays."""
        buf = torch.zeros((32, 14), dtype=torch.int64)
        torch.cuda.synchronize()
        _check(self._lib.dsa_timing_read(buf.data_ptr(), buf.numel() * 8), "dsa_timing_read")
        return buf[:, :7], buf[:, 7:]

    def fwd(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """Same contract as the definition's run(): returns (output bf16, lse fp32)."""
        T, H, D = q_nope.shape
        assert (H, D) == (16, 512) and q_pe.shape == (T, H, 64), (q_nope.shape, q_pe.shape)
        assert ckv_cache.shape[1:] == (64, 512) and kpe_cache.shape[1:] == (64, 64)
        assert sparse_indices.shape == (T, 2048) and sparse_indices.dtype == torch.int32
        assert q_nope.dtype == q_pe.dtype == ckv_cache.dtype == kpe_cache.dtype == torch.bfloat16
        for x in (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
            assert x.is_contiguous() and x.is_cuda
        out = torch.empty((T, H, D), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((T, H), dtype=torch.float32, device=q_nope.device)
        _check(self._lib.dsa_launch(q_nope.data_ptr(), q_pe.data_ptr(), ckv_cache.data_ptr(),
                                    kpe_cache.data_ptr(), sparse_indices.data_ptr(), float(sm_scale),
                                    out.data_ptr(), lse.data_ptr(), T, _stream()), "dsa_launch")
        return out, lse

    def marker(self):
        _check(self._lib.dsa_marker_launch(_stream()), "dsa_marker_launch")

    def probe(self, chain, sparse_indices, ckv_cache, sink):
        _check(self._lib.dsa_probe_launch(chain, sparse_indices.data_ptr(), ckv_cache.data_ptr(),
                                          sink.data_ptr(), _stream()), "dsa_probe_launch")

    def mma_probe(self, chain, sink):
        _check(self._lib.dsa_mma_probe_launch(chain, sink.data_ptr(), _stream()), "dsa_mma_probe_launch")

    def l2_flush(self, buf, mode=0, sink=None):
        """mode 0: overwrite the buffer (L2 full of dirty lines); mode 1: stream-read it (clean lines)."""
        _check(self._lib.dsa_l2_flush_launch(buf.data_ptr(), buf.numel() * buf.element_size(), mode,
                                             0 if sink is None else sink.data_ptr(), _stream()),
               "dsa_l2_flush_launch")


def load_all():
    """The kernel under test first, then the tuning variants (sorted by name).
    DSA_LIBS_ONLY / DSA_LIBS_EXCLUDE (comma-separated variant names) filter the variants."""
    main = os.path.join(_HERE, "libdsa.so")
    variants = sorted(p for p in glob.glob(os.path.join(_HERE, "libdsa_*.so")))
    only = [x for x in os.environ.get("DSA_LIBS_ONLY", "").split(",") if x]
    exclude = [x for x in os.environ.get("DSA_LIBS_EXCLUDE", "").split(",") if x]
    libs = [DsaLib(p) for p in [main] + variants if os.path.exists(p)]
    if only:
        libs = [lib for lib in libs if lib.name in only or lib.name == "new"]
    return [lib for lib in libs if lib.name not in exclude]
