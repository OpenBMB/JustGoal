"""GQA paged decode (page_size=1, bf16) - host side plan + launcher.

plan() is host-side scheduling done once per (batch layout, kv lengths), exactly
like flashinfer's wrapper.plan(); only run() is on the timed path.
"""
import math, os, torch
from torch.utils.cpp_extension import load

_mod = None
def _module():
    global _mod
    if _mod is None:
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
        here = os.path.dirname(os.path.abspath(__file__))
        _mod = load(name="gqa_decode_ext", sources=[os.path.join(here, "gqa_decode.cu")],
                    extra_cuda_cflags=["-O3", "--use_fast_math",
                                       "-gencode=arch=compute_100a,code=sm_100a"],
                    verbose=False)
    return _mod

TPW, WARPS = 16, 4          # tuned: 16 tokens per warp, 4 warps -> 64-token chunks

class Plan:
    def __init__(self, kv_lens, num_kv_heads, tpw=TPW, warps=WARPS, device="cuda"):
        self.tpw, self.warps = tpw, warps
        self.chunk = chunk = tpw * warps
        B, G = len(kv_lens), len(kv_lens) * num_kv_heads
        self.gstride = num_kv_heads        # chunk-major: the 8 kv heads of a token range are adjacent
        indptr = [0]
        for L in kv_lens: indptr.append(indptr[-1] + L)
        assert min(kv_lens) >= 1, "every request needs at least one kv token"
        tasks, gf, gn = [], [0]*G, [0]*G
        for b, L in enumerate(kv_lens):
            nch = max(1, (L + chunk - 1)//chunk)
            bounds = [(L*i)//nch for i in range(nch+1)]
            base = len(tasks)
            for h in range(num_kv_heads):
                g = b*num_kv_heads + h
                gf[g], gn[g] = base + h, nch
            for c in range(nch):
                for h in range(num_kv_heads):
                    tasks.append((b, h, indptr[b] + bounds[c], bounds[c+1] - bounds[c]))
        assert max(t[3] for t in tasks) <= chunk
        self.n = len(tasks)
        self.tasks = torch.tensor(tasks, dtype=torch.int32, device=device).contiguous()
        self.tasks2 = torch.tensor([(gn[b*num_kv_heads+h], gf[b*num_kv_heads+h], b*num_kv_heads+h, 0)
                                    for (b, h, _, _) in tasks],
                                   dtype=torch.int32, device=device).contiguous()
        self.part_acc = torch.empty(self.n, 4*128, dtype=torch.float32, device=device)
        self.part_ml = torch.empty(self.n, 4, 2, dtype=torch.float32, device=device)
        self.counters = torch.zeros(G*32, dtype=torch.int32, device=device)

def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale, plan, out=None, lse=None):
    """out (B,HQ,D) bf16 and lse (B,HQ) f32 (natural log). Single kernel launch."""
    B, HQ, D = q.shape
    if out is None: out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=q.device)
    if lse is None: lse = torch.empty(B, HQ, dtype=torch.float32, device=q.device)
    _module().gqa_decode(q, k_cache, v_cache, kv_indices, plan.tasks, plan.tasks2,
                         plan.part_acc, plan.part_ml, plan.counters, out, lse,
                         sm_scale, plan.tpw, plan.warps, plan.gstride)
    return out, lse
