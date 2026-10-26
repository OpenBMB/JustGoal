"""DSA sparse paged attention (decode) — custom CUDA kernel v16 (split-KV, 32-row chunks, progressive combine).
run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale) -> (output bf16 [T,16,512], lse fp32 [T,16], log2-based)
"""
import math
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_CPP_DECL = """
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> dsa_forward(torch::Tensor q_nope, torch::Tensor q_pe, torch::Tensor ckv_cache,
                                       torch::Tensor kpe_cache, torch::Tensor sparse_indices, double sm_scale,
                                       torch::Tensor zero_row, torch::Tensor part, torch::Tensor rec, torch::Tensor masks,
                                       torch::Tensor dws, torch::Tensor perm, int64_t gen, torch::Tensor trace);
torch::Tensor dsa_probe(int64_t nblocks);
int dsa_max_blocks();
"""

_ext = None
_ws = {}
_gen = [0]
_maxb = [None]
NCHUNK = 64
H = 16
MASK_STRIDE = 16


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        with open(os.path.join(_HERE, "dsa_kernel.cu")) as f:
            src = f.read()
        trace = os.environ.get("DSA_TRACE", "") == "1"
        extra = []
        tag = ""
        for env in sorted(k for k in os.environ if k.startswith("DSA_") and k not in ("DSA_TRACE", "DSA_VERBOSE", "DSA_NOFLUSH", "DSA_FLUSH_MODE")):
            extra.append("-D" + env + "=" + os.environ[env]); tag += "_" + env[4:].lower() + os.environ[env]
        name = "dsa_sparse_attn_v16" + ("_trace" if trace else "") + tag
        build_dir = os.path.join(_HERE, "_build", name)
        os.makedirs(build_dir, exist_ok=True)
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_DECL,
            cuda_sources=src,
            functions=["dsa_forward", "dsa_max_blocks", "dsa_probe"],
            extra_cuda_cflags=["-O3", "-lineinfo"] + (["-DDSA_TRACE"] if trace else []) + extra,
            build_directory=build_dir,
            verbose=False,
        )
    return _ext


def _build_perm(smids, T):
    """Input-independent placement-aware chunk permutation. SMs are ordered with the ones hosting fewer blocks first; the
    k-th block of every SM takes the next chunk of the k-th slab of the c-major order (chunk id = c * T + t, likely-active
    low chunk indices first). Each SM then holds chunks of two tokens, each at two well separated chunk indices."""
    NB = len(smids)
    by_sm = {}
    for b, sm in enumerate(smids):
        by_sm.setdefault(int(sm), []).append(b)
    sms = sorted(by_sm, key=lambda sm: (len(by_sm[sm]), sm))
    maxk = max(len(v) for v in by_sm.values())
    order = []
    for k in range(maxk):
        for sm in sms:
            if k < len(by_sm[sm]):
                order.append(by_sm[sm][k])
    assert len(order) == NB
    perm = [0] * NB
    for j, b in enumerate(order):
        perm[b] = j
    assert sorted(perm) == list(range(NB))
    return perm


_perm = {}


def _get_perm(ext, T, device):
    key = (T, str(device))
    p = _perm.get(key)
    if p is None:
        nb = T * NCHUNK
        smids = ext.dsa_probe(nb).cpu().tolist()
        smids2 = ext.dsa_probe(nb).cpu().tolist()
        if os.environ.get("DSA_VERBOSE"):
            same = sum(1 for a, b in zip(smids, smids2) if a == b)
            print(f"dsa: placement probe: {len(set(smids))} SMs, deterministic matches {same}/{nb}")
        p = torch.tensor(_build_perm(smids, T), dtype=torch.int32, device=device)
        _perm[key] = p
    return p


def _workspace(T, device):
    """Input-independent one-time buffers: zero row (TMA source for invalid slots), partial / record scratch (fully
    rewritten every call before being read), completion masks (double-buffered by launch parity, reset by the kernel)."""
    key = (T, device)
    w = _ws.get(key)
    if w is None:
        zero_row = torch.zeros(1024, dtype=torch.bfloat16, device=device)
        part = torch.zeros(T * NCHUNK * 4 * 16 * 64, dtype=torch.int32, device=device)
        rec = torch.zeros(T * NCHUNK * H * 2, dtype=torch.float32, device=device)
        masks = torch.zeros(2 * T * MASK_STRIDE, dtype=torch.int64, device=device)
        dws = torch.zeros(2 * (256 + 2 + 512), dtype=torch.int32, device=device)
        ntr = T * NCHUNK * 16 if os.environ.get("DSA_TRACE", "") == "1" else 0
        trace = torch.zeros(ntr, dtype=torch.int64, device=device)
        w = (zero_row, part, rec, masks, trace, dws)
        _ws[key] = w
    return w


@torch.no_grad()
def _fallback(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Plain fp32 torch implementation (used only when the grid would not fit co-resident, e.g. huge num_tokens)."""
    T, Hh, Dd = q_nope.shape
    Kc_all = ckv_cache.reshape(-1, Dd)
    Kp_all = kpe_cache.reshape(-1, q_pe.shape[-1])
    out = torch.zeros((T, Hh, Dd), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.full((T, Hh), -float("inf"), dtype=torch.float32, device=q_nope.device)
    for t in range(T):
        idx = sparse_indices[t]
        valid = idx[idx != -1].to(torch.long)
        if valid.numel() == 0:
            continue
        Kc = Kc_all[valid].float(); Kp = Kp_all[valid].float()
        logits = (q_nope[t].float() @ Kc.T + q_pe[t].float() @ Kp.T) * sm_scale
        lse[t] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        out[t] = (torch.softmax(logits, dim=-1) @ Kc).to(torch.bfloat16)
    return out, lse


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    ext = _load()
    T = q_nope.shape[0]
    if sparse_indices.dtype != torch.int32:
        sparse_indices = sparse_indices.to(torch.int32)
    if _maxb[0] is None:
        _maxb[0] = ext.dsa_max_blocks()
    if T * NCHUNK > _maxb[0]:
        return _fallback(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    zero_row, part, rec, masks, trace, dws = _workspace(T, q_nope.device)
    perm = _get_perm(ext, T, q_nope.device)
    gen = _gen[0]
    _gen[0] = gen + 1
    out, lse = ext.dsa_forward(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, float(sm_scale),
                               zero_row, part, rec, masks, dws, perm, gen, trace)
    return out, lse


def get_trace():
    for w in _ws.values():
        return w[4]
