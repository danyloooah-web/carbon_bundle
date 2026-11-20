from __future__ import annotations
import contextlib
import logging
import torch
import torch.nn.functional as F
from sglang.kernels.ops.gemm import bf16_fp32
from sglang.srt.layers.quantization import unquant
from glm53f_core import gemm_plan_table
NODE = "layer"
_STOCK_DISPATCH = unquant._bf16_gemm_dispatch_impl
_STOCK_FP32 = bf16_fp32.linear_bf16_fp32
_STOCK_LINEAR = F.linear
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
ALGO_BYTES = 64
_LIVE: dict = {}
_STATE: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf GEMM plans: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf GEMM plans: %s", detail)


def _module():
    
    if "module" not in _STATE:
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

        _STATE["module"] = gen_mm_bf16_cublaslt_module().build_and_load()
    return _STATE["module"]


def _rules_cublas(m: int, n: int, k: int) -> bool:
    if m <= unquant._BF16_LOW_M_AUTOTUNE_MAX_M:
        return False
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        return False
    if unquant._use_hopper_bf16_tiny_gemv is not None or unquant._use_hopper_bf16_gemv is not None:
        return False
    return not (unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k))


def _stock_cublas(m: int, n: int, k: int) -> bool:
    
    return not (unquant._mm_bf16 is not None and unquant._flashinfer_autotuner.is_tuning_mode) and _rules_cublas(m, n, k)


def _key(x, w, dtype: str):
    
    if not (isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and x.is_cuda and x.dim() == 2
            and w.dim() == 2 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.shape[1] == w.shape[1]
            and x.is_contiguous() and w.is_contiguous() and x.shape[0] > 0):
        return None
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        return None
    return x.shape[0], w.shape[0], w.shape[1], dtype

def _run(x, w, out, algo) -> None:
    
    _module().mm_bf16_cublaslt_run_with_algo(x, w, None, out, _STATE["workspace"], torch.cuda.current_blas_handle(),
                                             algo, 0)


def _stock(x, w, dtype: str):
    return _STOCK_LINEAR(x, w) if dtype == "bf16" else _STOCK_FP32(x, w)

def _probe(key, hexed: str):
    
    rows, n, k, dtype = key
    _require(dtype in _DTYPES and len(hexed) == 2 * ALGO_BYTES, f"row {key}: a malformed table entry")
    _require(dtype == "fp32" or _rules_cublas(rows, n, k), f"row {key}: stock does not run cuBLAS at these rows "
             "(a table written for another dispatch)")
    algo = bytes.fromhex(hexed)
    buf = torch.frombuffer(bytearray(algo), dtype=torch.uint8)
    gen = torch.Generator(device="cuda").manual_seed(184)
    with torch.no_grad():
        for trial in range(2):
            x = torch.randn((rows, k), generator=gen, device="cuda").to(torch.bfloat16)
            w = (torch.randn((n, k), generator=gen, device="cuda") * k ** -0.5).to(torch.bfloat16)
            if trial:
                w = torch.nn.Parameter(w)
            ref = _stock(x, w, dtype)
            out = torch.full((rows, n), float("nan"), dtype=_DTYPES[dtype], device="cuda")
            _run(x, w, out, buf)
            view = torch.int16 if dtype == "bf16" else torch.int32
            same = out.view(view) == ref.view(view)
            _require(bool(same.all()), f"row {key}: the plan's bytes differ from stock's in {int((~same).sum())} of "
                     f"{same.numel()} elements (trial {trial}): not a pinnable plan here")
    return buf


def prepare(layer) -> None:
    if _STATE.get("probed") or not gemm_plan_table.PLANS:
        return
    _require(not torch.cuda.is_current_stream_capturing(), "prepare inside a CUDA-graph capture")
    _STATE["workspace"] = torch.empty(0, dtype=torch.uint8, device="cuda")
    for key, hexed in sorted(gemm_plan_table.PLANS.items()):
        _LIVE[key] = _probe(key, hexed)
    _STATE["probed"] = True
    _say("table", f"{len(_LIVE)} plans probed, every one stock's bytes ({gemm_plan_table.SOURCE})")


def _dispatch(x, weight, bias, addend=None):
    key = _key(x, weight, "bf16") if bias is None and addend is None and F.linear is _STOCK_LINEAR else None
    algo = _LIVE.get(key) if key is not None else None
    if algo is None or not _stock_cublas(*key[:3]):
        return _STOCK_DISPATCH(x, weight, bias, addend=addend)
    out = torch.empty((key[0], key[1]), dtype=torch.bfloat16, device=x.device)
    _run(x, weight, out, algo)
    _say(f"bf16 {key[1:3]}", f"pinned plans serve BF16 [{key[1]}, {key[2]}] (first: {key[0]} rows)")
    return out


def _linear_fp32(x, y, *, hpc_kernel_min_m=None):
    
    key = _key(x, y, "fp32") if hpc_kernel_min_m is None and bf16_fp32._linear_bf16_fp32_algo == "cublas" else None
    algo = _LIVE.get(key) if key is not None else None
    if algo is None:
        return _STOCK_FP32(x, y, hpc_kernel_min_m=hpc_kernel_min_m)
    out = torch.empty((key[0], key[1]), dtype=torch.float32, device=x.device)
    _run(x, y, out, algo)
    _say(f"fp32 {key[1:3]}", f"pinned plans serve BF16 -> FP32 [{key[1]}, {key[2]}] (first: {key[0]} rows)")
    return out


@contextlib.contextmanager
def routed(layer, forward_batch):
    if not _LIVE:
        yield
        return
    _require(unquant._bf16_gemm_dispatch_impl is _STOCK_DISPATCH and bf16_fp32.linear_bf16_fp32 is _STOCK_FP32,
             "the BF16 dispatch is not stock's at a layer's call (a nested call, or another route holds it)")
    unquant._bf16_gemm_dispatch_impl, bf16_fp32.linear_bf16_fp32 = _dispatch, _linear_fp32
    try:
        yield
    finally:
        unquant._bf16_gemm_dispatch_impl, bf16_fp32.linear_bf16_fp32 = _STOCK_DISPATCH, _STOCK_FP32
