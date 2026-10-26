import os as _os
import math, time, sys, os
_DQ_BF16 = "dqf32" not in os.environ.get("FA_DBG", "")
import numpy as np
import torch, triton, triton.language as tl
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import fa_bwd_kernel as K

BLK = 128

@triton.jit
def _delta_kernel(o_ptr, do_ptr, delta_ptr, T, H, D: tl.constexpr):
    pid = tl.program_id(0)  # one program per (t, h) group of ROWS rows
    ROWS: tl.constexpr = 8
    rows = pid * ROWS + tl.arange(0, ROWS)
    t = rows // H
    h = rows % H
    d = tl.arange(0, D)
    offs = rows[:, None] * D + d[None, :]
    mask = rows[:, None] < T * H
    o = tl.load(o_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + h * T + t, s, mask=rows < T * H)

def compute_delta(o, do):
    T, H, D = o.shape
    delta = torch.empty(H, T, device=o.device, dtype=torch.float32)
    _delta_kernel[(triton.cdiv(T * H, 8),)](o, do, delta, T, H, D=D, num_warps=4)
    return delta

def build_schedule(cu_seqlens_cpu, H, grid):
    """Pair scheduling for 2-CTA clusters: a slot = (item for rank0, item for rank1, paired flag).
    Paired slot: adjacent kv blocks (kb, kb+1) of the same (doc, head) -> dQ partials are exchanged.
    Returns items[(n_slots*2, 4)] = (b, kb, h, paired) with b=-1 for an empty item, and cluster offsets."""
    import heapq
    cu = [int(x) for x in cu_seqlens_cpu]
    n_clusters = grid // 2
    slots = []   # (cost, item0, item1, paired)
    singles = []
    for b in range(len(cu) - 1):
        L = cu[b + 1] - cu[b]
        nqb = (L + BLK - 1) // BLK
        for h in range(H):
            kb = 0
            while kb < nqb:
                if kb + 1 < nqb:
                    slots.append((nqb - kb, (b, kb, h, 1), (b, kb + 1, h, 1), 1))
                    kb += 2
                else:
                    singles.append((nqb - kb, (b, kb, h, 0)))
                    kb += 1
    singles.sort(key=lambda x: -x[0])
    for i in range(0, len(singles), 2):
        a = singles[i]
        if i + 1 < len(singles):
            c = singles[i + 1]
            slots.append((max(a[0], c[0]), a[1], c[1], 0))
        else:
            slots.append((a[0], a[1], (-1, 0, 0, 0), 0))
    order = os.environ.get("FA_ORDER", "locality")
    if order == "cost":
        slots.sort(key=lambda x: -x[0])
    else:
        # locality order: longest documents first, then head, then kv block ascending (expensive first within a group)
        L_of = {b: cu[b + 1] - cu[b] for b in range(len(cu) - 1)}
        def key(s):
            i0 = s[1]
            return (-L_of[i0[0]] if i0[0] >= 0 else 0, i0[2], i0[1])
        slots.sort(key=key)
    heap = [(0.0, g) for g in range(n_clusters)]
    heapq.heapify(heap)
    buckets = [[] for _ in range(n_clusters)]
    for cost, i0, i1, paired in slots:
        load, g = heapq.heappop(heap)
        buckets[g].append((i0, i1))
        heapq.heappush(heap, (load + cost + 0.5, g))
    flat = []
    offsets = [0]
    for g in range(n_clusters):
        for i0, i1 in buckets[g]:
            flat.append(i0)
            flat.append(i1)
        offsets.append(len(flat) // 2)
    return torch.tensor(flat, dtype=torch.int32).view(-1, 4), torch.tensor(offsets, dtype=torch.int32), max(len(b) for b in buckets)

class FaBwd:
    def __init__(self, q, k, v, o, do, lse, cu_seqlens, grid=148):
        self.T, self.H, self.D = q.shape
        self.grid = grid
        self.scale = 1.0 / math.sqrt(self.D)
        self.items, self.offsets, _ = build_schedule(cu_seqlens.cpu().tolist(), self.H, grid)
        if "nopair" in os.environ.get("FA_DBG", ""):
            self.items[:, 3] = 0
        self.items = self.items.cuda(); self.offsets = self.offsets.cuda()
        self.dq_acc = torch.zeros(self.T, self.H, self.D, device=q.device, dtype=torch.float32)
        self.dk = torch.empty_like(k); self.dv = torch.empty_like(v); self.dq = torch.empty_like(q)
        self.delta = torch.empty(self.H, self.T, device=q.device, dtype=torch.float32)
        self.compiled = None
        self.dbg = torch.zeros(max(grid * 16, 6 * 128 * 128), device=q.device, dtype=torch.float32)
        self.q, self.k, self.v, self.o, self.do, self.lse, self.cu = q, k, v, o, do, lse, cu_seqlens

    def _cute_args(self):
        def t3(x):  # (T,H,D) -> cute tensor viewed as (T, D, H)
            return from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mQ = t3(self.q); mK = t3(self.k); mV = t3(self.v); mdO = t3(self.do)
        mLSE = from_dlpack(self.lse, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mDelta = from_dlpack(self.delta, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        mDQ = from_dlpack(self.dq if _DQ_BF16 else self.dq_acc, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mDK = from_dlpack(self.dk, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        mDV = from_dlpack(self.dv, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        cu = from_dlpack(self.cu, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        items = from_dlpack(self.items, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        offs = from_dlpack(self.offsets, assumed_align=4).mark_layout_dynamic(leading_dim=0)
        return [mQ, mK, mV, mdO, mLSE, mDelta, mDQ, mDK, mDV, cu, items, offs,
                cutlass.Float32(self.scale * 1.4426950408889634), cutlass.Float32(self.scale),
                cutlass.Int32(self.T), cutlass.Int32(self.H), cutlass.Int32(self.grid), from_dlpack(self.dbg, assumed_align=16).mark_layout_dynamic(leading_dim=0)]

    def compile(self):
        args = self._cute_args()
        self.compiled = cute.compile(K.host_launch, *args)
        self.args = args

    def run(self):
        # preprocess
        compute_delta_into(self.o, self.do, self.delta)
        if _DQ_BF16:
            self.dq.zero_()                      # dQ accumulated in place with bf16 atomics (scale folded in the kernel)
            self.compiled(*self.args)
        else:
            self.dq_acc.zero_()
            self.compiled(*self.args)
            convert_dq(self.dq_acc, self.dq, self.scale)
        return self.dq, self.dk, self.dv

@triton.jit
def _convert_kernel(src_ptr, dst_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(src_ptr + offs, mask=m, other=0.0)
    tl.store(dst_ptr + offs, (x * scale).to(tl.bfloat16), mask=m)

def convert_dq(dq_acc, dq, scale):
    n = dq_acc.numel()
    _convert_kernel[(triton.cdiv(n, 4096),)](dq_acc, dq, n, scale, BLOCK=4096, num_warps=8)

def compute_delta_into(o, do, delta):
    T, H, D = o.shape
    _delta_kernel[(triton.cdiv(T * H, 8),)](o, do, delta, T, H, D=D, num_warps=4)
