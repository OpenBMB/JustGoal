"""FP8 block-scale DeepSeek-V3 MoE for B200 — Python entry that JIT-compiles the self-contained CUDA kernels."""
import hashlib
import os

import torch

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "moe_kernel.cu")
    with open(src, "rb") as f:
        digest = hashlib.sha1(f.read()).hexdigest()[:12]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    abl = os.environ.get("MOE_ABL", "0")  # dev-only ablation switch; production builds use 0
    timing = os.environ.get("MOE_TIMING", "0")  # dev-only instrumentation; production builds use 0
    hint = os.environ.get("MOE_WAIT_HINT", "1")
    f2 = os.environ.get("MOE_FFMA2", "1")
    b256 = os.environ.get("MOE_B256", "1")
    b4d = os.environ.get("MOE_B4D", "1")
    reo = os.environ.get("MOE_REORDER", "1")
    sag = os.environ.get("MOE_SA_GROUP", "1")
    pf = os.environ.get("MOE_PF", "0")
    amc = os.environ.get("MOE_AMC", "0")
    fork = os.environ.get("MOE_FORK", "0")
    half = os.environ.get("MOE_HALF", "1")
    dyn = os.environ.get("MOE_DYN", "1")
    zero = os.environ.get("MOE_ZERO", "1")
    fsig = os.environ.get("MOE_FASTSIG", "1")
    ctma = os.environ.get("MOE_COPY_TMA", "0")
    wpf = os.environ.get("MOE_W13_PF_EXPERTS", "2")
    bfa = os.environ.get("MOE_BF16ACC", "0")
    pair = os.environ.get("MOE_PAIR", "0")
    btr = os.environ.get("MOE_BF16_TRUNC", "0")
    trs = os.environ.get("MOE_TRS", "0")
    fsilu = os.environ.get("MOE_FASTSILU", "2")
    pki = os.environ.get("MOE_PACK_INT", "1")
    st2 = os.environ.get("MOE_STAGES2", "3")
    rof = os.environ.get("MOE_RED_OFFLOAD", "2")
    pw = os.environ.get("MOE_PW", "8")
    two = os.environ.get("MOE_2SM", "0")
    chi = os.environ.get("MOE_CHI_TMA", "0")
    l2h = os.environ.get("MOE_L2HINT", "0")
    fy = os.environ.get("MOE_FMA_YIELD", "0")
    ones = os.environ.get("MOE_ONESTORE", "1")
    wfold = os.environ.get("MOE_WFOLD", "0")
    rdc = os.environ.get("MOE_READY_CACHE", "0")
    rde = os.environ.get("MOE_READY_EARLY", "0")
    npf = os.environ.get("MOE_NEXT_PF", "0")
    rdd = os.environ.get("MOE_READY_DIST", "6")
    pwpf = os.environ.get("MOE_PWPF", "0")
    pwpf2 = os.environ.get("MOE_PWPF2", "6")
    pwkb = os.environ.get("MOE_PWPF_KB", "4")
    stg = os.environ.get("MOE_STAGGER", "0")
    cla = os.environ.get("MOE_CLAIM_AHEAD", "1")
    fin = os.environ.get("MOE_FIN_OFFLOAD", "1")
    sfo = os.environ.get("MOE_SILU_FOLD", "1")
    sts = os.environ.get("MOE_STS", "1")
    pregs = os.environ.get("MOE_PROMO_REGS", "")
    pregs_flag = [f"-DMOE_PROMO_REGS={pregs}"] if pregs else []
    _EXT = load(
        name=f"moe_fp8_ds_b200_{digest}_a{abl}_t{timing}_h{hint}_f{f2}_b{b256}{b4d}{reo}_g{sag}_p{pf}_amc{amc}_f{fork}_h{half}_d{dyn}_z{zero}_fs{fsig}_ct{ctma}_pf{wpf}_bfa{bfa}_pr{pair}_tr{btr}_trs{trs}_fs{fsilu}_pk{pki}_s{st2}_ro{rof}_pw{pw}_pr{pregs}_2sm{two}_chi{chi}_l2{l2h}_fy{fy}_os{ones}_wf{wfold}_rc{rdc}{rde}{npf}{rdd}_pw{pwpf}{pwpf2}{pwkb}_sg{stg}_ca{cla}_fi{fin}_sf{sfo}_st{sts}",
        sources=[src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr", f"-DMOE_ABL={abl}", f"-DMOE_TIMING={timing}", f"-DMOE_WAIT_HINT={hint}", f"-DMOE_FFMA2={f2}", f"-DMOE_B256={b256}", f"-DMOE_B4D={b4d}", f"-DMOE_REORDER={reo}", f"-DMOE_SA_GROUP={sag}", f"-DMOE_PF={pf}", f"-DMOE_AMC={amc}", f"-DMOE_FORK={fork}", f"-DMOE_HALF={half}", f"-DMOE_DYN={dyn}", f"-DMOE_ZERO={zero}", f"-DMOE_FASTSIG={fsig}", f"-DMOE_COPY_TMA={ctma}", f"-DMOE_W13_PF_EXPERTS={wpf}", f"-DMOE_BF16ACC={bfa}", f"-DMOE_PAIR={pair}", f"-DMOE_BF16_TRUNC={btr}", f"-DMOE_TRS={trs}", f"-DMOE_FASTSILU={fsilu}", f"-DMOE_PACK_INT={pki}", f"-DMOE_STAGES2={st2}", f"-DMOE_RED_OFFLOAD={rof}", f"-DMOE_PW={pw}", f"-DMOE_2SM={two}", f"-DMOE_CHI_TMA={chi}", f"-DMOE_L2HINT={l2h}", f"-DMOE_FMA_YIELD={fy}", f"-DMOE_ONESTORE={ones}", f"-DMOE_WFOLD={wfold}", f"-DMOE_READY_CACHE={rdc}", f"-DMOE_READY_EARLY={rde}", f"-DMOE_NEXT_PF={npf}", f"-DMOE_READY_DIST={rdd}", f"-DMOE_PWPF={pwpf}", f"-DMOE_PWPF2={pwpf2}", f"-DMOE_PWPF_KB={pwkb}", f"-DMOE_STAGGER={stg}", f"-DMOE_CLAIM_AHEAD={cla}", f"-DMOE_FIN_OFFLOAD={fin}", f"-DMOE_SILU_FOLD={sfo}", f"-DMOE_STS={sts}"] + pregs_flag,
        verbose=False,
    )
    return _EXT


def _scalar(x, cast):
    if isinstance(x, torch.Tensor):
        return cast(x.item())
    return cast(x)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset,
    routed_scaling_factor,
):
    ext = _load_ext()
    if routing_bias is None:
        routing_bias = torch.zeros(routing_logits.shape[1], dtype=torch.bfloat16, device=routing_logits.device)
    return ext.moe_forward(
        routing_logits.contiguous().float(),
        routing_bias,
        hidden_states.contiguous(),
        hidden_states_scale.contiguous().float(),
        gemm1_weights.contiguous(),
        gemm1_weights_scale.contiguous().float(),
        gemm2_weights.contiguous(),
        gemm2_weights_scale.contiguous().float(),
        _scalar(local_expert_offset, int),
        _scalar(routed_scaling_factor, float),
    )
