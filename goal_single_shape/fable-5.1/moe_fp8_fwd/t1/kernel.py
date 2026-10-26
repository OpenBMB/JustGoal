"""T=1 DeepSeek-V3 MoE FP8 block-scale decode. Hand-written CUDA kernel (moe_kernel.cu), built
once with nvcc and loaded through ctypes. run() computes everything (routing, GEMM1, SwiGLU,
GEMM2, weighted sum) from the raw inputs on every call; only the compiled binary, SM count and a
zeroed scratch workspace are reused across calls."""
import ctypes, hashlib, os, subprocess, sys
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, 'moe_kernel.cu')


def _find_nvcc():
    for c in (os.environ.get('CUDA_HOME', '') + '/bin/nvcc', '/usr/local/cuda/bin/nvcc', 'nvcc'):
        if c.startswith('/') and os.path.exists(c):
            return c
    return 'nvcc'


def _build():
    src = open(_SRC, 'rb').read()
    dbg = os.environ.get('MOE_DEBUG', '0')
    mode = os.environ.get('MOE_MODE', '0')
    extra = os.environ.get('MOE_EXTRA', '')
    tag = hashlib.sha1(src + dbg.encode() + mode.encode() + extra.encode()).hexdigest()[:16]
    cache = os.environ.get('MOE_KERNEL_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'moe_decode_kernel'))
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f'libmoe_{tag}.so')
    if not os.path.exists(so):
        nvcc = _find_nvcc()
        cuda_home = os.path.dirname(os.path.dirname(nvcc)) if nvcc.startswith('/') else '/usr/local/cuda'
        tmp = so + f'.{os.getpid()}.tmp'
        cmd = [nvcc, '-O3', '-std=c++17', '-arch=sm_100a', '--shared', '-Xcompiler', '-fPIC',
               f'-DMOE_DEBUG={dbg}', f'-DMOE_MODE={mode}'] + ([extra] if extra else []) + ['-o', tmp, _SRC, f'-L{cuda_home}/lib64/stubs', '-lcuda']
        subprocess.run(cmd, check=True)
        os.replace(tmp, so)
    return so


_lib = ctypes.CDLL(_build())
_lib.moe_ws_bytes.restype = ctypes.c_size_t
_lib.moe_num_sms.restype = ctypes.c_int
_lib.moe_init.restype = ctypes.c_int
_lib.moe_run.restype = ctypes.c_int
_lib.moe_run.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
_lib.moe_dbg_per_block.restype = ctypes.c_int
DEBUG_BUF = None  # set to an int64 cuda tensor of nsm*moe_dbg_per_block() to collect a timeline

_state = {}


def _prep(device):
    key = device.index if device.index is not None else torch.cuda.current_device()
    st = _state.get(key)
    if st is None:
        with torch.cuda.device(key):
            rc = _lib.moe_init()
            if rc != 0:
                raise RuntimeError(f'moe_init failed: {rc}')
            ws = torch.zeros(int(_lib.moe_ws_bytes()), dtype=torch.uint8, device=device)
            st = [ws, int(_lib.moe_num_sms()), 0]   # workspace, SM count, launch epoch (tags stale records)
        _state[key] = st
    return st


@torch.no_grad()
def run(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset,
        routed_scaling_factor):
    assert routing_logits.shape == (1, 256) and routing_logits.dtype == torch.float32
    assert routing_bias.dtype == torch.bfloat16 and routing_bias.numel() == 256
    assert hidden_states.shape == (1, 7168) and hidden_states.dtype == torch.float8_e4m3fn
    assert hidden_states_scale.numel() == 56 and hidden_states_scale.dtype == torch.float32
    assert gemm1_weights.shape == (32, 4096, 7168) and gemm1_weights.dtype == torch.float8_e4m3fn
    assert gemm1_weights_scale.shape == (32, 32, 56) and gemm1_weights_scale.dtype == torch.float32
    assert gemm2_weights.shape == (32, 7168, 2048) and gemm2_weights.dtype == torch.float8_e4m3fn
    assert gemm2_weights_scale.shape == (32, 56, 16) and gemm2_weights_scale.dtype == torch.float32
    for x in (routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
              gemm1_weights_scale, gemm2_weights, gemm2_weights_scale):
        assert x.is_contiguous() and x.is_cuda
    dev = hidden_states.device
    st = _prep(dev)
    ws, nsm = st[0], st[1]
    st[2] = (st[2] + 1) & 0x7fffffff or 1     # per-launch epoch; never 0 (fresh workspace holds zeros)
    epoch = st[2]
    out = torch.empty(1, 7168, dtype=torch.bfloat16, device=dev)
    stream = torch.cuda.current_stream(dev).cuda_stream
    rc = _lib.moe_run(stream, gemm1_weights.data_ptr(), gemm2_weights.data_ptr(), routing_logits.data_ptr(),
                      routing_bias.data_ptr(), hidden_states.data_ptr(), hidden_states_scale.data_ptr(),
                      gemm1_weights_scale.data_ptr(), gemm2_weights_scale.data_ptr(),
                      int(local_expert_offset), float(routed_scaling_factor), ws.data_ptr(), out.data_ptr(), nsm, epoch,
                      DEBUG_BUF.data_ptr() if DEBUG_BUF is not None else None)
    if rc != 0:
        raise RuntimeError(f'moe_run failed: {rc}')
    return out
