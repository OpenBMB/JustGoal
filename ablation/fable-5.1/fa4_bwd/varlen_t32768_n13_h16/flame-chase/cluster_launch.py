"""Launch Triton/Gluon kernels in thread-block clusters WITHOUT compiling with num_ctas>1.

Triton's launcher takes the cluster dimension from the packed kernel metadata (num_ctas).  Compiling with
num_ctas>1 forces cluster-aware (CGA) layouts everywhere, which we do not want: our kernels keep plain
single-CTA layouts and talk to the peer CTA explicitly (inline PTX: %cluster_ctarank, mapa, st.shared::cluster).
So we patch pack_metadata to report the requested cluster size for registered kernel names."""
from triton.backends.nvidia.compiler import CUDABackend

_CLUSTER = {}
_orig_pack = CUDABackend.pack_metadata


def _patched_pack(self, metadata):
    nw, nc, sh = _orig_pack(self, metadata)
    nc = _CLUSTER.get(metadata.name, nc)
    return (nw, nc, sh)


CUDABackend.pack_metadata = _patched_pack


def register_cluster(kernel_name: str, size: int):
    _CLUSTER[kernel_name] = size
