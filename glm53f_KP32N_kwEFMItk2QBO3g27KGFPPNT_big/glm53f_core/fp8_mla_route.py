from __future__ import annotations
import contextlib
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route
NODE = "layer"
WARM_ROWS = (8193, 12345, 16384, 24571, 32768)
def _targets(layer) -> list:
    attention = layer.self_attn
    if not isinstance(attention, DeepseekV2AttentionMLA):
        return []
    found = [getattr(attention, "q_b_proj", None), getattr(attention, "kv_b_proj", None)]
    fp8_linear_route._require(all(target is not None for target in found),
                              f"sparse layer {getattr(layer, 'layer_id', '?')} lacks q_b_proj or kv_b_proj")
    return found
def prepare(layer) -> None:
    mode = get_disagg().disaggregation_mode
    if mode == "decode":
        return
    for target in _targets(layer):
        if id(target) not in fp8_linear_route._FP8:
            fp8_linear_route.serve_linear(target, WARM_ROWS, False, mode)


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if not any(id(target) in fp8_linear_route._FP8 for target in _targets(layer)):
        yield
        return
    unquant.UnquantizedLinearMethod.apply = fp8_linear_route._apply
    try:
        yield
    finally:
        unquant.UnquantizedLinearMethod.apply = fp8_linear_route._STOCK_APPLY
