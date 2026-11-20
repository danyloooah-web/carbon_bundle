from __future__ import annotations
import contextlib
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route
NODE = "layer"
WARM_ROWS = (8193, 12345, 16384, 24571, 32768)
def _targets(layer) -> list:
    mlp = layer.mlp
    if not isinstance(mlp, DeepseekV2MoE):
        return []
    shared = getattr(mlp, "shared_experts", None)
    found = [getattr(shared, "gate_up_proj", None), getattr(shared, "down_proj", None)]
    fp8_linear_route._require(all(target is not None for target in found),
                              f"MoE layer {getattr(layer, 'layer_id', '?')} holds no separate shared expert with "
                              "gate_up_proj and down_proj")
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
