from __future__ import annotations
import logging
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route
NODE = "proj"
_SAID: set = set()
def prepare(node) -> None:
    layer = fp8_linear_route._projection(node)
    mode = get_disagg().disaggregation_mode
    in_prefill, in_decode = mode != "decode", mode != "prefill" and fp8_linear_route.DECODE
    if not (in_prefill or in_decode):
        return
    entry = fp8_linear_route._FP8.get(id(layer))
    if entry is not None:
        fp8_linear_route._require(entry[6] is None,
                                  "the KDA input projection holds fp8_linear_route's copy with its gate columns kept: "
                                  "fp8_gates_route must be prepared first (its place in make_bundles.py's "
                                  "COMPONENTS_SECOND)")
        return
    gates = fp8_linear_route._gate_columns(layer)
    fp8_linear_route.serve_linear(layer, fp8_linear_route.WARM_ROWS if in_prefill else (), in_decode, mode, None)
    if mode not in _SAID:
        _SAID.add(mode)
        logging.getLogger(__name__).info(
            "glmf FP8 gates: the KDA input projection's beta, f_a and g_a columns %s are FP8 on this %s engine",
            gates, mode)

routed = fp8_linear_route.routed
