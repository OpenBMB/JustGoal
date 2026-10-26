# SPDX-License-Identifier: BSD-3-Clause
"""Hand-built tcgen05.mma issue for the backward kernel's MMA warp (D23).

The DSL's cute.gemm rebuilds both UMMA shared-memory descriptors per MMA in regular
registers (~20 instructions per tcgen05.mma in SASS). Here each k-loop becomes one inline
PTX block: the descriptor of k-block 0 is built once by the DSL
(cute.nvgpu.tcgen05.smem_descriptor_to_int of the fragment iterator); the k-block and
pipeline-stage offsets are static immediates (16-byte units) read from the fragment layout
at trace time and added to the low word of the descriptor (the 14-bit start-address field
never carries into the LBO field for the 227 KB of shared memory in use).
"""
from typing import List, Optional, Sequence, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm


def make_idesc(m: int, n: int, a_mn_major: bool, b_mn_major: bool) -> int:
    """PTX tcgen05 instruction descriptor for kind::f16 with BF16 A/B and an F32 accumulator."""
    v = 1 << 4  # D: F32
    v |= 1 << 7  # A: BF16
    v |= 1 << 10  # B: BF16
    v |= int(a_mn_major) << 15
    v |= int(b_mn_major) << 16
    v |= (n >> 3) << 17
    v |= (m >> 4) << 24
    return v


def static_offsets(frag: cute.Tensor, coords: Sequence[tuple]) -> List[int]:
    """Offsets (descriptor units, i.e. 16 B, for smem fragments; elements for TMEM
    fragments) of the given coordinates in the fragment layout; must fold at trace time."""
    out = []
    for c in coords:
        o = cute.crd2idx(c, frag.layout)
        if not isinstance(o, int):
            raise RuntimeError(f"umma_raw: non-static fragment offset for {c}: {o}")
        out.append(o)
    return out


def desc_parts(frag: cute.Tensor) -> Tuple[Int32, Int32]:
    """(lo, hi) words of the UMMA smem descriptor of the fragment's first element."""
    d = tcgen05.smem_descriptor_to_int(frag.iterator)
    lo = Int32(d & Int64(0xFFFFFFFF))
    hi = Int32(d >> Int64(32))
    return lo, hi


def mma_group(
    d_tmem: Int32,
    a_lo: Int32,
    a_hi: Optional[Int32],
    b_lo: Int32,
    b_hi: Int32,
    idesc: int,
    a_offs: Sequence[int],
    b_offs: Sequence[int],
    acc_first: Int32,
    a_in_tmem: bool = False,
) -> None:
    """Issue len(a_offs) tcgen05.mma.cta_group::1.kind::f16 into d_tmem. The first MMA
    accumulates iff acc_first != 0, the following ones always accumulate. A is a smem
    descriptor (a_lo/a_hi) or a TMEM address (a_lo, a_in_tmem). Must run under
    cute.arch.elect_one()."""
    n = len(a_offs)
    assert len(b_offs) == n
    lines = [
        "{",
        ".reg .pred p;",
        ".reg .b64 ad, bd;",
        ".reg .b32 alo, blo;",
        "setp.ne.b32 p, $6, 0;",
    ]
    for k in range(n):
        if k == 1:
            lines.append("setp.ne.b32 p, $7, 0;")
        lines.append(f"add.u32 alo, $1, {a_offs[k]};")
        lines.append(f"add.u32 blo, $3, {b_offs[k]};")
        lines.append("mov.b64 bd, {blo, $4};")
        if a_in_tmem:
            lines.append(f"tcgen05.mma.cta_group::1.kind::f16 [$0], [alo], bd, $5, p;")
        else:
            lines.append("mov.b64 ad, {alo, $2};")
            lines.append(f"tcgen05.mma.cta_group::1.kind::f16 [$0], ad, bd, $5, p;")
    lines.append("}")
    asm = "\n\t".join(lines) + "\n"
    a_hi_v = a_hi if a_hi is not None else Int32(0)
    operands = [
        Int32(d_tmem).ir_value(),
        Int32(a_lo).ir_value(),
        Int32(a_hi_v).ir_value(),
        Int32(b_lo).ir_value(),
        Int32(b_hi).ir_value(),
        Int32(idesc).ir_value(),
        Int32(acc_first).ir_value(),
        Int32(1).ir_value(),
    ]
    llvm.inline_asm(
        None,
        operands,
        asm,
        "r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def red_add_v4_16bit(addr: Int64, words: Sequence[Int32], packed: str) -> None:
    """16-byte vector reduction into global memory: red.global.add.noftz.v4.<packed> with
    packed = 'f16x2' | 'bf16x2' (four 32-bit words = eight 16-bit elements)."""
    assert len(words) == 4 and packed in ("f16x2", "bf16x2")
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value()] + [Int32(w).ir_value() for w in words],
        f"red.global.add.noftz.v4.{packed} [$0], {{$1, $2, $3, $4}};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def globaltimer() -> Int64:
    """Nanosecond %globaltimer (for in-kernel instrumentation)."""
    from cutlass._mlir import ir

    v = llvm.inline_asm(
        ir.IntegerType.get_signless(64),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int64(v)


def nanosleep(ns: int) -> None:
    """nanosleep.u32 with an immediate (approximate, for polling loops)."""
    llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {int(ns)};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
