# MoE FP8 block-scale (DeepSeek-V3 routing) for B200.
# Pipeline: routing -> meta -> gemm1 -> swiglu+quant -> gemm2 -> combine
import torch

H = 7168
I = 2048
N1 = 2 * I            # 4096
KB1 = H // 128        # 56
KB2 = I // 128        # 16
NB1 = N1 // 128       # 32
NB2 = H // 128        # 56
E_GLB = 256
E_LOC = 32
MAXC_BASE = 1024      # max tokens per local expert slot table (>= T)
MAXTILES_BASE = 128
BM = 128



# ---------------------------------------------------------------- workspace
_WS = {}


def _ws(T, device):
    key = (T, device)
    w = _WS.get(key)
    if w is None:
        maxc = max(MAXC_BASE, (T + 127) // 128 * 128)
        mtiles = max(MAXTILES_BASE, E_LOC + (T * 8 + BM - 1) // BM + 8)
        mtl = E_LOC + (T * 8 + BM - 1) // BM
        maxrows = T * 8 + BM + 128
        maxrows = (maxrows + 127) // 128 * 128
        w = dict(
            cnt=torch.zeros(E_LOC, dtype=torch.int32, device=device),
            slot_tok=torch.empty(E_LOC * maxc, dtype=torch.int32, device=device),
            slot_w=torch.empty(E_LOC * maxc, dtype=torch.float32, device=device),
            tok_e=torch.empty(T * 8, dtype=torch.int32, device=device),
            tok_j=torch.empty(T * 8, dtype=torch.int32, device=device),
            tok_n=torch.empty(T, dtype=torch.int32, device=device),
            padoffs=torch.empty(E_LOC, dtype=torch.int32, device=device),
            tile_e=torch.zeros(mtiles, dtype=torch.int32, device=device),
            tile_base=torch.zeros(mtiles, dtype=torch.int32, device=device),
            tile_j0=torch.zeros(mtiles, dtype=torch.int32, device=device),
            ntiles=torch.zeros(4, dtype=torch.int32, device=device),
            Ap=torch.zeros((maxrows, H), dtype=torch.float8_e4m3fn, device=device),
            Asp=torch.zeros((KB1, maxrows), dtype=torch.float32, device=device),
            G1=torch.empty((maxrows, N1), dtype=torch.bfloat16, device=device),
            Cq=torch.empty((maxrows, I), dtype=torch.float8_e4m3fn, device=device),
            Cs=torch.empty((KB2, maxrows), dtype=torch.float32, device=device),
            P=torch.empty((maxrows, H), dtype=torch.bfloat16, device=device),
            scal=torch.zeros(2, dtype=torch.float32, device=device),
            scal_h=torch.zeros(2, dtype=torch.float32).pin_memory(),
            maxrows=maxrows, maxc=maxc, mtiles=mtiles, mtl=mtl,
        )
        w['ptrs'] = [w[k].data_ptr() for k in ('Ap','Asp','cnt','slot_tok','slot_w','tok_e','tok_j',
                                               'tok_n','padoffs','tile_e','tile_base','tile_j0','ntiles','G1','Cq','Cs','P')]
        _WS[key] = w
    return w


_TUNE = dict(CFG1=3 * 4096 + 3 * 256 + 2 * 16 + 2, CFG2=3 * 4096 + 3 * 256 + 1 * 16 + 2, L2W=0)
NPROG = 148


def _build_cu():
    import os
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    cutlass = os.environ.get('CUTLASS_DIR', '/root/cutlass/include')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '10.0a')
    return load(name='moe_kernels_v1',
                sources=[os.path.join(here, 'moe_kernels.cu')],
                extra_cuda_cflags=['-O3', '-gencode=arch=compute_100a,code=sm_100a',
                                   '--expt-relaxed-constexpr', '--expt-extended-lambda',
                                   '-I' + cutlass, '-DNDEBUG', '--use_fast_math'],
                extra_cflags=['-O3', '-I' + cutlass],
                extra_ldflags=['-lcuda'],
                verbose=bool(int(os.environ.get('MOE_VERBOSE', '0'))))


_cu = _build_cu()


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    T = routing_logits.shape[0]
    dev = hidden_states.device
    w = _ws(T, dev)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)

    MR = w['maxrows']
    P = w['ptrs']
    _cu.moe_all(routing_logits.data_ptr(), routing_bias.data_ptr(), hidden_states.data_ptr(),
                hidden_states_scale.data_ptr(), gemm1_weights.data_ptr(),
                gemm1_weights_scale.data_ptr(), gemm2_weights.data_ptr(),
                gemm2_weights_scale.data_ptr(), out.data_ptr(),
                P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10], P[11],
                P[12], P[13], P[14], P[15], P[16], T, MR, w['maxc'], w['mtl'],
                int(local_expert_offset), float(routed_scaling_factor), NPROG,
                _TUNE['CFG1'], _TUNE['CFG2'], _TUNE['L2W'])

    return out
