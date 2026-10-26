"""Entry point: attention_bwd_varlen_causal_h16_d128 (value-returning style)."""

import os
import sys

import torch
import triton

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bwd_triton import attention_bwd_triton, _bwd_preprocess, _bwd_convert_dq, _bwd_convert_dq_t, _bwd_convert_dq_tiled, _get_dq_accum, _get_xblk, _build_tile_table  # noqa: E402

_IMPL = os.environ.get("ATT_IMPL", "dsl3")
_CFG = dict(
    BLOCK_M=int(os.environ.get("ATT_BM", "64")),
    BLOCK_N=int(os.environ.get("ATT_BN", "64")),
    num_warps=int(os.environ.get("ATT_WARPS", "8")),
    num_stages=int(os.environ.get("ATT_STAGES", "2")),
)

DQ_T = False
DQ_TILED = False
CVT_INK = False
DQ_DIRECT = False
DQ_DIRECT_F16 = False
_ACC_CLEAN = {}      # id(accumulator) -> True when the buffer is known to be all zeros (in-kernel conversion re-zeroes it)
_CNT_CACHE = {}
if _IMPL == "dsl":
    from bwd_dsl import attention_bwd_prepare, attention_bwd_launch, TILE, DQ_F16  # noqa: E402
elif _IMPL == "dsl2":
    from bwd_dsl2 import attention_bwd_prepare, attention_bwd_launch, TILE  # noqa: E402
    DQ_F16 = True
    DQ_T = True      # dQ accumulator stored transposed (H, D, T)
elif _IMPL == "dsl3":
    from bwd_dsl3 import attention_bwd_prepare, attention_bwd_launch, TILE, DQ_TILED, CVT_INK, DQ_DIRECT, DQ_DIRECT_F16  # noqa: E402
    DQ_F16 = True
    DQ_T = not DQ_DIRECT          # DQ_DIRECT: dq itself is the (bf16) reduction target; no accumulator, no convert

_SCHED_CACHE = {}


def _get_sched(nseq_pad, device):
    """Persistent kernel scheduling buffers: per-sequence padded accumulator starts and the work-item counter."""
    key = (nseq_pad, str(device))
    r = _SCHED_CACHE.get(key)
    if r is None:
        r = (torch.zeros(nseq_pad, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device))
        _SCHED_CACHE[key] = r
    return r

_TABLE_CACHE = {}


def run_dsl(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    T, H, D = q.shape
    device = q.device
    num_seqs = cu_seqlens.numel() - 1
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    if DQ_DIRECT and DQ_DIRECT_F16:
        dq_accum = _get_dq_accum(T, H, D, device, torch.float16, transposed=False, num_seqs=num_seqs, tiled=False)   # (T, H, D) f16
    elif DQ_DIRECT:
        dq_accum = dq
    else:
        dq_accum = _get_dq_accum(T, H, D, device, torch.float16 if DQ_F16 else torch.float32, transposed=DQ_T, num_seqs=num_seqs, tiled=DQ_TILED)
    T_cap = dq_accum.numel() // (H * D) if DQ_T else T
    blk_cnt = cvt_q = None
    zero_sep = os.environ.get("ATT_ZERO_SEP", "0") == "1"      # experiment: zero the accumulator with a separate fill kernel
    if zero_sep and DQ_T and not CVT_INK:
        dq_accum.zero_()
    if CVT_INK:
        if not _ACC_CLEAN.get(id(dq_accum), False):
            dq_accum.zero_()                      # first use, or a previous call did not complete
        _ACC_CLEAN[id(dq_accum)] = False
        nblk = T_cap // 64
        blk_cnt = _CNT_CACHE.get((H, nblk, str(device)))
        if blk_cnt is None:
            blk_cnt = torch.zeros(H * nblk + 1024 * 8, dtype=torch.int32, device=device)   # + per-CTA debug counters
            _CNT_CACHE[(H, nblk, str(device))] = blk_cnt
        cvt_q = _CNT_CACHE.get(("q", str(device)))
        if cvt_q is None:
            cvt_q = torch.zeros(1024 * 64 * 6, dtype=torch.int32, device=device)          # per-CTA conversion job rings
            _CNT_CACHE[("q", str(device))] = cvt_q
    delta = torch.empty(H, T, dtype=torch.float32, device=device)
    max_tiles = (T + TILE - 1) // TILE + num_seqs
    order = int(os.environ.get("ATT_ORDER", "2"))
    table = _TABLE_CACHE.get(max_tiles)
    if table is None:
        table = torch.empty(max_tiles * H, dtype=torch.int32, device=device)
        _TABLE_CACHE[max_tiles] = table
    nseq_pad = max(16, triton.next_power_of_2(num_seqs + 1))
    seq_acc = counter = None
    # host-side DSL argument preparation first (not on the GPU timeline)
    if _IMPL == "dsl3":
        xq, xdo, cb = _get_xblk(T, H, device)
        seq_acc, counter = _get_sched(nseq_pad, device)
        prepared = attention_bwd_prepare(q, k, v, dout, dq_accum, dk, dv, cu_seqlens, table, max_tiles, float(sm_scale), xq, xdo, cb, seq_acc, counter,
                                         dq=dq, blk_cnt=blk_cnt, cvt_q=cvt_q)
    elif DQ_T:
        xq, xdo, cb = _get_xblk(T, H, device)
        prepared = attention_bwd_prepare(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, table, max_tiles, float(sm_scale), xq, xdo, cb)
    else:
        prepared = attention_bwd_prepare(q, k, v, dout, lse, delta, dq_accum, dk, dv, cu_seqlens, table, max_tiles, float(sm_scale))
    BLOCK_T = int(os.environ.get("ATT_PRE_BT", "8" if os.environ.get("ATT_PRE_ALLH", "0") == "1" else "16"))
    pre_warps = int(os.environ.get("ATT_PRE_WARPS", "4"))
    T_zero = T_cap
    pre_hfast = os.environ.get("ATT_PRE_HFAST", "1") == "1"
    pre_grid = (H, triton.cdiv(T_zero, BLOCK_T) + 1) if pre_hfast else (triton.cdiv(T_zero, BLOCK_T) + 1, H)
    _bwd_preprocess[pre_grid](
        out, dout, delta, dq_accum, T, H=H, D=D, BLOCK_T=BLOCK_T,
        cu_seqlens=cu_seqlens, num_seqs=num_seqs, table=table, max_tiles=max_tiles, BLOCK_N=TILE,
        MAX_NBLK=(int(max_seqlen) + TILE - 1) // TILE, NSEQ_PAD=nseq_pad, num_warps=pre_warps,
        DQ_T=DQ_T, T_pad=T_cap if DQ_T else 0, T_grid=T_zero,
        LSE=lse, XQ=xq if (DQ_T or DQ_DIRECT) else None, XDO=xdo if (DQ_T or DQ_DIRECT) else None, inv_scale=1.0 / float(sm_scale), XBLK=DQ_T or DQ_DIRECT,
        SEQ_ACC=seq_acc, COUNTER=counter, SEQ_MAJOR=order, ALL_HEADS=os.environ.get("ATT_PRE_ALLH", "0") == "1",   # experiment: 125 us vs 67
        ZERO_ACC=not (CVT_INK or zero_sep), BLK_CNT=blk_cnt, NBLK=(T_cap // 64) if CVT_INK else 0, HFAST=pre_hfast,
        HG=int(os.environ.get("ATT_HGROUP", "1")))
    attention_bwd_launch(prepared)
    if DQ_DIRECT and DQ_DIRECT_F16:
        n_elem = T * H * D
        _bwd_convert_dq[(triton.cdiv(n_elem, 4096),)](dq_accum, dq, n_elem, float(sm_scale), BLOCK=4096, num_warps=8)
    elif DQ_DIRECT:
        pass                                      # dq was reduce-added in place (bf16); nothing to convert
    elif CVT_INK:
        _ACC_CLEAN[id(dq_accum)] = True           # the kernel converted and re-zeroed every block
    elif DQ_TILED:
        cvt_hfast = os.environ.get("ATT_CVT_HFAST", "1") == "1"
        cvt_grid = (H, T_cap // 64) if cvt_hfast else (T_cap // 64, H)
        _bwd_convert_dq_tiled[cvt_grid](dq_accum, dq, T, cu_seqlens, num_seqs, float(sm_scale), H=H, D=D, NBLK=T_cap // 64,
                                        NSEQ_PAD=nseq_pad, EVICT=int(os.environ.get("ATT_CVT_EVICT", "0")), HFAST=cvt_hfast,
                                        num_warps=int(os.environ.get("ATT_CVT_WARPS", "4")))
    elif DQ_T:
        BT = int(os.environ.get("ATT_CVT_BT", "64"))
        BD = int(os.environ.get("ATT_CVT_BD", "128"))
        _bwd_convert_dq_t[(triton.cdiv(T + 64 * num_seqs, BT), H, D // BD)](dq_accum, dq, T, dq_accum.shape[-1], cu_seqlens, num_seqs, float(sm_scale), H=H, D=D, BT=BT,
                                                                           NSEQ_PAD=nseq_pad, MODE=int(os.environ.get("ATT_CVT_MODE", "0")), BD=BD, num_warps=int(os.environ.get("ATT_CVT_WARPS", "4")))
    else:
        n_elem = T * H * D
        BLOCK = 4096
        _bwd_convert_dq[(triton.cdiv(n_elem, BLOCK),)](dq_accum, dq, n_elem, float(sm_scale), BLOCK=BLOCK, num_warps=8)
    return dq, dk, dv


def run(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale):
    if _IMPL in ("dsl", "dsl2", "dsl3"):
        return run_dsl(q, k, v, out, dout, lse, cu_seqlens, max_seqlen, sm_scale)
    return attention_bwd_triton(q, k, v, out, dout, lse, cu_seqlens, int(max_seqlen), float(sm_scale), **_CFG)
