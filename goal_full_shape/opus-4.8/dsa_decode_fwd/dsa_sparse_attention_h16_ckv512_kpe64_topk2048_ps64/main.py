import math
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Phase-1 flash kernel: program (t, s) streams the contiguous range
# [s*SPLIT_N, (s+1)*SPLIT_N) of token t's sparse_indices, sharing gathered
# KV across all H heads, producing a partial (m, l, acc).
# ---------------------------------------------------------------------------
@triton.jit
def _phase1(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    po_ptr, pm_ptr, pl_ptr,
    scale2, topk, num_splits,
    stride_qn_t, stride_qn_h, stride_qp_t, stride_qp_h, stride_idx_t,
    stride_po_t, stride_po_h, stride_po_s, stride_pm_t, stride_pm_h, stride_pm_s,
    SPLIT_N: tl.constexpr, BLOCK_N: tl.constexpr,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr, SKIP: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // num_splits
    s = pid % num_splits
    start = s * SPLIT_N
    offs_h = tl.arange(0, H)
    offs_c = tl.arange(0, D_CKV)
    offs_p = tl.arange(0, D_KPE)

    qn = tl.load(q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_c[None, :]).to(tl.bfloat16)
    qp = tl.load(q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_p[None, :]).to(tl.bfloat16)

    m_i = tl.full((H,), -float("inf"), tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, D_CKV), tl.float32)

    for off in range(0, SPLIT_N, BLOCK_N):
        cols = start + off + tl.arange(0, BLOCK_N)
        cmask = cols < topk
        ids = tl.load(idx_ptr + t * stride_idx_t + cols, mask=cmask, other=-1)
        valid = ids != -1
        go = (not SKIP) or (tl.sum(valid.to(tl.int32)) > 0)
        if go:
            ids_safe = tl.where(valid, ids, 0)
            kc = tl.load(ckv_ptr + ids_safe[:, None] * D_CKV + offs_c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            kp = tl.load(kpe_ptr + ids_safe[:, None] * D_KPE + offs_p[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            sc = tl.dot(qn, tl.trans(kc))
            sc += tl.dot(qp, tl.trans(kp))
            sc = sc * scale2
            sc = tl.where(valid[None, :], sc, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_sub = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp2(sc - m_sub[:, None])
            alpha = tl.exp2(m_i - m_sub)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
            m_i = m_new

    tl.store(po_ptr + t * stride_po_t + offs_h[:, None] * stride_po_h + s * stride_po_s + offs_c[None, :], acc)
    tl.store(pm_ptr + t * stride_pm_t + offs_h * stride_pm_h + s * stride_pm_s, m_i)
    tl.store(pl_ptr + t * stride_pm_t + offs_h * stride_pm_h + s * stride_pm_s, l_i)


@triton.jit
def _fused2(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    po_ptr, pm_ptr, pl_ptr, cnt_ptr, out_ptr, lse_ptr,
    scale2, topk, num_splits, num_splits_T,
    stride_qn_t, stride_qn_h, stride_qp_t, stride_qp_h, stride_idx_t,
    stride_po_t, stride_po_h, stride_po_s, stride_pm_t, stride_pm_h, stride_pm_s,
    stride_out_t, stride_out_h, stride_lse_t,
    SPLIT_N: tl.constexpr, BLOCK_N: tl.constexpr,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr, NS: tl.constexpr,
    CHUNK_C: tl.constexpr, SKIP: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // num_splits
    s = pid % num_splits
    start = s * SPLIT_N
    offs_h = tl.arange(0, H)
    offs_c = tl.arange(0, D_CKV)
    offs_p = tl.arange(0, D_KPE)
    qn = tl.load(q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_c[None, :]).to(tl.bfloat16)
    qp = tl.load(q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_p[None, :]).to(tl.bfloat16)
    m_i = tl.full((H,), -float("inf"), tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, D_CKV), tl.float32)
    for off in range(0, SPLIT_N, BLOCK_N):
        cols = start + off + tl.arange(0, BLOCK_N)
        cmask = cols < topk
        ids = tl.load(idx_ptr + t * stride_idx_t + cols, mask=cmask, other=-1)
        valid = ids != -1
        nv = tl.sum(valid.to(tl.int32))
        go = (not SKIP) or (nv > 0)
        if go:
            ids_safe = tl.where(valid, ids, 0)
            kc = tl.load(ckv_ptr + ids_safe[:, None] * D_CKV + offs_c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            kp = tl.load(kpe_ptr + ids_safe[:, None] * D_KPE + offs_p[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            sc = tl.dot(qn, tl.trans(kc))
            sc += tl.dot(qp, tl.trans(kp))
            sc = sc * scale2
            sc = tl.where(valid[None, :], sc, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_sub = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp2(sc - m_sub[:, None])
            alpha = tl.exp2(m_i - m_sub)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
            m_i = m_new
    tl.store(po_ptr + t * stride_po_t + offs_h[:, None] * stride_po_h + s * stride_po_s + offs_c[None, :], acc)
    tl.store(pm_ptr + t * stride_pm_t + offs_h * stride_pm_h + s * stride_pm_s, m_i)
    tl.store(pl_ptr + t * stride_pm_t + offs_h * stride_pm_h + s * stride_pm_s, l_i)

    if tl.max(l_i) > 0.0:
        tl.atomic_max(cnt_ptr + num_splits_T + t, s + 1, sem="relaxed")
    last = tl.atomic_add(cnt_ptr + t, 1, sem="acq_rel")
    if last == NS - 1:
        nact = tl.load(cnt_ptr + num_splits_T + t)
        mg = tl.full((H,), -float("inf"), tl.float32)
        for s2 in range(0, nact):
            ms = tl.load(pm_ptr + t * stride_pm_t + offs_h * stride_pm_h + s2 * stride_pm_s)
            mg = tl.maximum(mg, ms)
        msub = tl.where(mg == -float("inf"), 0.0, mg)
        accf = tl.zeros((H, D_CKV), tl.float32)
        lg = tl.zeros((H,), tl.float32)
        for s2 in range(0, nact):
            ms = tl.load(pm_ptr + t * stride_pm_t + offs_h * stride_pm_h + s2 * stride_pm_s)
            ls = tl.load(pl_ptr + t * stride_pm_t + offs_h * stride_pm_h + s2 * stride_pm_s)
            sca = tl.exp2(ms - msub)
            oo = tl.load(po_ptr + t * stride_po_t + offs_h[:, None] * stride_po_h + s2 * stride_po_s + offs_c[None, :])
            accf += oo * sca[:, None]
            lg += ls * sca
        denom = tl.where(lg == 0.0, 1.0, lg)
        tl.store(out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_c[None, :],
                 (accf / denom[:, None]).to(out_ptr.dtype.element_ty))
        lse = tl.where(lg == 0.0, -float("inf"), mg + tl.log2(lg))
        tl.store(lse_ptr + t * stride_lse_t + offs_h, lse)
        tl.store(cnt_ptr + t, 0)
        tl.store(cnt_ptr + num_splits_T + t, 0)


@triton.jit
def _single(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    out_ptr, lse_ptr,
    scale2, topk,
    stride_qn_t, stride_qn_h, stride_qp_t, stride_qp_h, stride_idx_t,
    stride_out_t, stride_out_h, stride_lse_t,
    BLOCK_N: tl.constexpr, H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
):
    t = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_c = tl.arange(0, D_CKV)
    offs_p = tl.arange(0, D_KPE)
    qn = tl.load(q_nope_ptr + t * stride_qn_t + offs_h[:, None] * stride_qn_h + offs_c[None, :]).to(tl.bfloat16)
    qp = tl.load(q_pe_ptr + t * stride_qp_t + offs_h[:, None] * stride_qp_h + offs_p[None, :]).to(tl.bfloat16)

    m_i = tl.full((H,), -float("inf"), tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, D_CKV), tl.float32)
    for off in range(0, topk, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        cmask = cols < topk
        ids = tl.load(idx_ptr + t * stride_idx_t + cols, mask=cmask, other=-1)
        valid = ids != -1
        if tl.sum(valid.to(tl.int32)) > 0:
            ids_safe = tl.where(valid, ids, 0)
            kc = tl.load(ckv_ptr + ids_safe[:, None] * D_CKV + offs_c[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            kp = tl.load(kpe_ptr + ids_safe[:, None] * D_KPE + offs_p[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            sc = tl.dot(qn, tl.trans(kc))
            sc += tl.dot(qp, tl.trans(kp))
            sc = sc * scale2
            sc = tl.where(valid[None, :], sc, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_sub = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp2(sc - m_sub[:, None])
            alpha = tl.exp2(m_i - m_sub)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
            m_i = m_new

    denom = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / denom[:, None]
    tl.store(out_ptr + t * stride_out_t + offs_h[:, None] * stride_out_h + offs_c[None, :],
             out.to(out_ptr.dtype.element_ty))
    lse = tl.where(l_i == 0.0, -float("inf"), m_i + tl.log2(l_i))
    tl.store(lse_ptr + t * stride_lse_t + offs_h, lse)


@triton.jit
def _combine(
    po_ptr, pm_ptr, pl_ptr, out_ptr, lse_ptr, num_splits,
    stride_po_t, stride_po_h, stride_po_s, stride_pm_t, stride_pm_h, stride_pm_s,
    stride_out_t, stride_out_h, stride_lse_t,
    H: tl.constexpr, D_CKV: tl.constexpr, NS: tl.constexpr, CHUNK_C: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    cchunk = tl.program_id(2)
    offs_c = cchunk * CHUNK_C + tl.arange(0, CHUNK_C)
    offs_s = tl.arange(0, NS)

    m = tl.load(pm_ptr + t * stride_pm_t + h * stride_pm_h + offs_s * stride_pm_s)
    l = tl.load(pl_ptr + t * stride_pm_t + h * stride_pm_h + offs_s * stride_pm_s)
    o = tl.load(po_ptr + t * stride_po_t + h * stride_po_h + offs_s[:, None] * stride_po_s + offs_c[None, :])

    m_g = tl.max(m, 0)
    m_sub = tl.where(m_g == -float("inf"), 0.0, m_g)
    scale = tl.exp2(m - m_sub)
    l_g = tl.sum(l * scale, 0)
    acc = tl.sum(o * scale[:, None], 0)
    denom = tl.where(l_g == 0.0, 1.0, l_g)
    out = acc / denom
    tl.store(out_ptr + t * stride_out_t + h * stride_out_h + offs_c, out.to(out_ptr.dtype.element_ty))
    if cchunk == 0:
        lse = tl.where(l_g == 0.0, -float("inf"), m_g + tl.log2(l_g))
        tl.store(lse_ptr + t * stride_lse_t + h, lse)


_CFG = {"num_splits": None, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2,
        "cw": 4, "CHUNK_C": 512, "mode": "auto", "fw": 8, "skip": True, "po_bf16": True}


def _num_splits_for(T):
    if _CFG["num_splits"] is not None:
        return _CFG["num_splits"]
    if T <= 2:
        return 8
    return 16


_WS = {}


def _workspace(T, num_splits, device):
    key = (T, num_splits, device)
    buf = _WS.get(key)
    if buf is None:
        _dt = torch.bfloat16 if _CFG.get("po_bf16") else torch.float32
        po = torch.empty((T, 16, num_splits, 512), dtype=_dt, device=device)  # [T,H,NS,Dc]
        pm = torch.empty((T, 16, num_splits), dtype=torch.float32, device=device)        # [T,H,NS]
        pl = torch.empty((T, 16, num_splits), dtype=torch.float32, device=device)
        cnt = torch.zeros((2 * T,), dtype=torch.int32, device=device)  # [:T]=completion, [T:]=n_active
        _WS[key] = buf = (po, pm, pl, cnt)
    return buf


def _launch_fused2(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                   scale2, topk, T, H, D_CKV, D_KPE, device,
                   num_splits, BLOCK_N, num_warps, num_stages, skip):
    SPLIT_N = (topk + num_splits - 1) // num_splits
    po, pm, pl, cnt = _workspace(T, num_splits, device)
    _fused2[(T * num_splits,)](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        po, pm, pl, cnt, output, lse,
        scale2, topk, num_splits, T,
        q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1), sparse_indices.stride(0),
        po.stride(0), po.stride(1), po.stride(2), pm.stride(0), pm.stride(1), pm.stride(2),
        output.stride(0), output.stride(1), lse.stride(0),
        SPLIT_N=SPLIT_N, BLOCK_N=BLOCK_N, H=H, D_CKV=D_CKV, D_KPE=D_KPE, NS=num_splits,
        CHUNK_C=128, SKIP=skip, num_warps=num_warps, num_stages=num_stages,
    )


def _launch_two(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                scale2, topk, T, H, D_CKV, D_KPE, device,
                num_splits, BLOCK_N, num_warps, num_stages, skip, cw, CHUNK_C):
    SPLIT_N = (topk + num_splits - 1) // num_splits
    po, pm, pl, cnt = _workspace(T, num_splits, device)
    _phase1[(T * num_splits,)](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        po, pm, pl, scale2, topk, num_splits,
        q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1), sparse_indices.stride(0),
        po.stride(0), po.stride(1), po.stride(2), pm.stride(0), pm.stride(1), pm.stride(2),
        SPLIT_N=SPLIT_N, BLOCK_N=BLOCK_N, H=H, D_CKV=D_CKV, D_KPE=D_KPE, SKIP=skip,
        num_warps=num_warps, num_stages=num_stages,
    )
    nchunks = D_CKV // CHUNK_C
    _combine[(T, H, nchunks)](
        po, pm, pl, output, lse, num_splits,
        po.stride(0), po.stride(1), po.stride(2), pm.stride(0), pm.stride(1), pm.stride(2),
        output.stride(0), output.stride(1), lse.stride(0),
        H=H, D_CKV=D_CKV, NS=num_splits, CHUNK_C=CHUNK_C, num_warps=cw,
    )


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    T, H, D_CKV = q_nope.shape
    D_KPE = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    device = q_nope.device
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    scale2 = sm_scale * LOG2E

    ckv_flat = ckv_cache.view(-1, D_CKV)
    kpe_flat = kpe_cache.view(-1, D_KPE)
    output = torch.empty((T, H, D_CKV), dtype=torch.bfloat16, device=device)
    lse = torch.empty((T, H), dtype=torch.float32, device=device)

    mode = _CFG["mode"]
    if mode == "auto":
        # Single-launch fused kernel (no inter-kernel gap), pipelined phase-1.
        _launch_fused2(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                       scale2, topk, T, H, D_CKV, D_KPE, device,
                       num_splits=16, BLOCK_N=128, num_warps=8, num_stages=2, skip=False)
        return output, lse
    if mode == "hyb":
        if T <= 2:
            _launch_fused2(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                           scale2, topk, T, H, D_CKV, D_KPE, device,
                           num_splits=16, BLOCK_N=64, num_warps=8, num_stages=2, skip=False)
        else:
            _launch_two(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                        scale2, topk, T, H, D_CKV, D_KPE, device,
                        num_splits=16, BLOCK_N=64, num_warps=4, num_stages=2,
                        skip=False, cw=4, CHUNK_C=128)
        return output, lse

    num_splits = _num_splits_for(T)
    BLOCK_N = _CFG["BLOCK_N"]
    if mode == "fused2":
        _launch_fused2(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                       scale2, topk, T, H, D_CKV, D_KPE, device,
                       num_splits, BLOCK_N, _CFG["num_warps"], _CFG["num_stages"], _CFG["skip"])
    elif mode == "single":
        _single[(T,)](
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            output, lse, scale2, topk,
            q_nope.stride(0), q_nope.stride(1), q_pe.stride(0), q_pe.stride(1), sparse_indices.stride(0),
            output.stride(0), output.stride(1), lse.stride(0),
            BLOCK_N=BLOCK_N, H=H, D_CKV=D_CKV, D_KPE=D_KPE,
            num_warps=_CFG["num_warps"], num_stages=_CFG["num_stages"],
        )
    else:
        _launch_two(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse,
                    scale2, topk, T, H, D_CKV, D_KPE, device,
                    num_splits, BLOCK_N, _CFG["num_warps"], _CFG["num_stages"],
                    _CFG["skip"], _CFG["cw"], _CFG["CHUNK_C"])
    return output, lse
