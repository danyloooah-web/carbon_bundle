from __future__ import annotations
import contextlib
import logging
import time
import torch
import torch.nn.functional as F
from sglang.kernels.ops.gemm import fp8_kernel as gemm_fp8
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization import fp8_utils, unquant
from sglang.srt.runtime_context import get_disagg
NODE = "proj"
DECODE = False
MIN_ROWS = 4096
BLOCK = [128, 128]
N_ALIGN = 64
WARM_ROWS = (4096, 4097, 5000, 6143, 8192, 12345, 16384, 24571, 32768)
DECODE_MIN_ROWS = 129
DECODE_MAX_ROWS = 768
WARM_ROWS_DECODE = tuple(range(DECODE_MIN_ROWS, DECODE_MAX_ROWS + 1))
_STOCK_APPLY = unquant.UnquantizedLinearMethod.apply
_FP8: dict = {}
_WARM: set = set()
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf FP8 linear: {why}")
_DEST: dict = {}
def _into(entry, x2d, bias, dest) -> bool:
    weight, scale, n = entry[0], entry[1], entry[2]
    if not (n == weight.shape[0] and bias is None and entry[6] is None and dest.dtype == torch.bfloat16
            and dest.is_contiguous() and dest.device == x2d.device and tuple(dest.shape) == (x2d.shape[0], n)
            and x2d.is_contiguous() and not fp8_utils._use_fp8_block_gemv(x2d, weight, scale, BLOCK)
            and fp8_utils._deepgemm_block_fp8_supported(tuple(weight.shape), BLOCK, torch.bfloat16)
            and not fp8_utils._ue8m0_deepgemm_skips_block32(BLOCK, scale, None)):
        _say(("dest", n), f"the {n}-column projection's rows ({x2d.shape[0]}) did not go to the given output "
                          f"{tuple(dest.shape)}: its own output serves")
        return False
    q, s = fp8_utils.sglang_per_token_group_quant_fp8(
        x2d, BLOCK[1], column_major_scales=True, scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
    gemm_fp8.deep_gemm_fp8_fp8_bf16_nt(q, s, weight, scale, dest, BLOCK[0], BLOCK[1])
    return True
def _say(tag, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf FP8 linear: %s", detail)


def _projection(node):
    
    layer = node if isinstance(node, LinearBase) else getattr(node, "fused_qkvbfg_a_proj", None)
    _require(layer is not None, f"the claimed projection is {type(node).__name__}, not an unquantized linear")
    return layer


def prepare(node) -> None:
    
    layer = _projection(node)
    mode = get_disagg().disaggregation_mode

    in_prefill, in_decode = mode != "decode", mode != "prefill" and DECODE
    if not (in_prefill or in_decode) or id(layer) in _FP8:
        return
    serve_linear(layer, WARM_ROWS if in_prefill else (), in_decode, mode, _gate_columns(layer))

def _gate_columns(layer) -> tuple:
    
    sizes = getattr(layer, "output_partition_sizes", None)
    shape = tuple(getattr(getattr(layer, "weight", None), "shape", ()))
    _require(isinstance(sizes, (list, tuple)) and len(sizes) == 6 and all(isinstance(s, int) and s > 0 for s in sizes)
             and sizes[0] == sizes[1] == sizes[2] and sizes[4] == sizes[5] and shape[:1] == (sum(sizes),),
             f"the claimed projection's packed shards are {sizes}, not q, k, v, b, f_a and g_a of its {shape} weight")
    return sum(sizes[:3]), sum(sizes)
def serve_linear(layer, prefill_rows: tuple, in_decode: bool, mode: str, exact: tuple | None = None) -> None:
    
    _require(deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
             "DeepGEMM with power-of-two scales is not enabled on this engine")
    _require(not torch.cuda.is_current_stream_capturing(), "prepare ran under graph capture")
    _require(isinstance(layer, LinearBase) and type(layer.quant_method) is unquant.UnquantizedLinearMethod,
             f"the claimed projection is {type(layer).__name__}, not an unquantized linear")
    weight = layer.weight.data
    _require(weight.dim() == 2 and weight.dtype == torch.bfloat16 and weight.is_cuda and weight.shape[1] % 128 == 0,
             f"the claimed projection's weight is {weight.dtype} {tuple(weight.shape)} on {weight.device}")
    n, source = weight.shape[0], weight
    _require(exact is None or 0 <= exact[0] < exact[1] <= n,
             f"exact columns {exact} lie outside the projection's {n} output columns")
    if n % N_ALIGN:
        source = weight.new_zeros((n + N_ALIGN - n % N_ALIGN, weight.shape[1]))
        source[:n].copy_(weight)
    quantized, scale = fp8_utils.quant_weight_ue8m0(weight_dequant=source, weight_block_size=BLOCK)
    window = (prefill_rows[0], prefill_rows[-1]) if prefill_rows else None
    entry = _FP8[id(layer)] = (quantized, fp8_utils.transform_scale_ue8m0(scale, mn=quantized.shape[-2]), n,
                               window is not None, in_decode, window, exact)
    shape = (*quantized.shape, tuple(prefill_rows), in_decode, exact)
    if shape in _WARM:
        return
    began = time.perf_counter()
    compile_utils._INITIALIZATION_DICT[
        (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, quantized.shape[0], quantized.shape[1], 1)] = True
    rows = tuple(prefill_rows) + (WARM_ROWS_DECODE if in_decode else ())
    x = torch.zeros((max(rows), quantized.shape[1]), dtype=torch.bfloat16, device=weight.device)
    for m in rows:
        fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
            input=x[:m], weight=entry[0], block_size=BLOCK, weight_scale=entry[1], input_scale=None, bias=None)
        if exact is not None:
            F.linear(x[:m], weight[exact[0]:exact[1]])
    del x
    torch.cuda.synchronize()
    _WARM.add(shape)
    logging.getLogger(__name__).info(
        "glmf FP8 linear on a %s engine: weights of %d x %d rows as FP8 %d x %d, output columns %s stock's BF16, "
        "prefill rows %s, decode rows %s, %.1f s for %d warm calls (DeepGEMM's compile included)", mode, n,
        weight.shape[1], quantized.shape[0], quantized.shape[1], exact, window, in_decode,
        time.perf_counter() - began, len(rows))


def _apply(self, layer, x, bias=None):
    
    entry = _FP8.get(id(layer))
    if entry is None or not x.shape[-1]:
        return _STOCK_APPLY(self, layer, x, bias)
    m = x.numel() // x.shape[-1]
    if not ((entry[3] and entry[5][0] <= m <= entry[5][1])
            or (entry[4] and DECODE_MIN_ROWS <= m <= DECODE_MAX_ROWS)):
        return _STOCK_APPLY(self, layer, x, bias)
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-1] == entry[0].shape[1]):
        why = (f"{x.dtype} input {'on the GPU' if x.is_cuda else 'off the GPU'}, "
               f"{'contiguous' if x.is_contiguous() else 'strided'}, {x.shape[-1]} wide")
        _say((entry[2], why), f"stock's BF16 GEMM serves the {entry[2]}-column projection's calls of {why} "
                              f"(first: {m} rows)")
        return _STOCK_APPLY(self, layer, x, bias)
    n, wide = entry[2], entry[0].shape[0]
    x2d = x.reshape(-1, x.shape[-1])
    dest = _DEST.pop(id(layer), None)
    if dest is not None and _into(entry, x2d, bias, dest):
        return dest.view(*x.shape[:-1], n)
    out = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback(
        input=x2d, weight=entry[0], block_size=BLOCK, weight_scale=entry[1],
        input_scale=None, bias=bias if n == wide else None)
    if n != wide:
        out = out[:, :n]
        if bias is not None:
            out.add_(bias)
    if entry[6] is not None:
        first, end = entry[6]
        out[:, first:end].copy_(F.linear(x2d, layer.weight[first:end], None if bias is None else bias[first:end]))
    return out.view(*x.shape[:-1], n)


@contextlib.contextmanager
def routed(node, forward_batch):
    
    if id(_projection(node)) not in _FP8:
        yield
        return
    unquant.UnquantizedLinearMethod.apply = _apply
    try:
        yield
    finally:
        unquant.UnquantizedLinearMethod.apply = _STOCK_APPLY
