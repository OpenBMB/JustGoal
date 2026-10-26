"""My DSA sparse MLA decode kernel (loads the CUDA extension in dsa_decode.cu)."""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "build")
os.makedirs(_BUILD, exist_ok=True)
_FLAGS = ["-O3", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo"]
TIMING = os.environ.get("DSA_TIMING", "0") == "1"
COOPERATIVE = os.environ.get("DSA_COOP", "1") == "1"
if TIMING:
    _bd = os.path.join(_BUILD, "timed"); os.makedirs(_bd, exist_ok=True)
    _ext = load(name="dsa_decode_ext_timed", sources=[os.path.join(_HERE, "dsa_decode.cu")],
                build_directory=_bd, extra_cuda_cflags=_FLAGS + ["-DDSA_TIMING"], verbose=False)
else:
    _ext = load(name="dsa_decode_ext", sources=[os.path.join(_HERE, "dsa_decode.cu")],
                build_directory=_BUILD, extra_cuda_cflags=_FLAGS, verbose=False)

NC = _ext.num_ctas()
NSTAMP = 32
_state = {"epoch": 0, "ws": {}, "stamps": None}


def _workspace(T, device):
    key = (T, str(device))
    if key not in _state["ws"]:
        _state["ws"][key] = torch.zeros(T * NC * _ext.rec_bytes(), dtype=torch.uint8, device=device)
    return _state["ws"][key]


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T = q_nope.shape[0]
    ws = _workspace(T, q_nope.device)
    # 16-bit epoch cycles through 1..65535; every workspace is cleared when it wraps so a stale
    # record can never carry the current epoch.
    _state["epoch"] = _state["epoch"] % 65535 + 1
    if _state["epoch"] == 1:
        for w in _state["ws"].values():
            w.zero_()
    stamps = (None, None)
    if TIMING:
        if _state["stamps"] is None:
            _state["stamps"] = (torch.zeros(T * NC * NSTAMP, dtype=torch.int64, device="cuda"),
                                torch.zeros(T * NC * NSTAMP, dtype=torch.int64, device="cuda"))
        stamps = _state["stamps"]
    return tuple(_ext.run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale), ws,
                          _state["epoch"], COOPERATIVE, stamps[0], stamps[1]))


def report_stamps(T):
    """Per-phase durations (us) of the last call (median over CTAs; clock64 / measured SM clock)."""
    c = _state["stamps"][0].view(T, NC, NSTAMP).cpu().double()
    g = _state["stamps"][1].view(T, NC, NSTAMP).cpu().double()
    names = ["idx+compact", "Q landed", "K landed", "QK mma", "softmax+publish", "V landed", "spin batch0", "M+consume", "merge+store"]
    last = 9
    dc = (c[:, :, last] - c[:, :, 0]); dg = (g[:, :, last] - g[:, :, 0]).clamp(min=1)
    ghz = (dc / dg).mean().item()
    print(f"  est SM clock {ghz:.3f} GHz; span first start -> last end = {(g[:, :, last].max() - g[:, :, 0].min()).item()/1000:.3f} us; "
          f"start skew = {(g[:, :, 0].max() - g[:, :, 0].min()).item()/1000:.3f} us")
    for t in range(T):
        d = (c[t, :, 1:last + 1] - c[t, :, 0:last]) / ghz / 1000.0
        tot = (c[t, :, last] - c[t, :, 0]) / ghz / 1000
        print(f"  token {t}: " + "  ".join(f"{nm}={d[:, i].median():.2f}" for i, nm in enumerate(names)) +
              f"  | total median={tot.median():.2f} us (max {tot.max():.2f})")
