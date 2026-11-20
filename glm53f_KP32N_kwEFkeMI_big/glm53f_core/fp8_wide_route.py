from __future__ import annotations
import contextlib
from sglang.srt.layers.quantization import unquant
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route
NODE = "layer"
WARM_ROWS = (8193, 12345, 16384, 24571, 32768)
O_PROJ_ONLY = True
def _targets(layer) -> list:
    attention = layer.self_attn
    found = [getattr(attention, "o_proj", None)]
    if not O_PROJ_ONLY:
        shared = getattr(layer.mlp, "shared_experts", None)
        found += [getattr(attention, "fused_qkv_a_proj_with_mqa", None),
                  getattr(shared, "gate_up_proj", None) if shared is not None else None]
    return [target for target in found if target is not None]
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
