from __future__ import annotations
import contextlib
import torch
from sglang.srt.layers.moe.moe_runner import base as moe_base
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_disagg
from glm53f_core import e31_fused_route, e31_stage_route
NODE = "layer"
LOW, HIGH, STEP = 8192, 16384, 1024
_HIDDEN = 4096
_LAYERS: dict = {}
_DEV: dict = {}
_state: dict = {"capture": False, "out": None}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf MoE graph: {why}")


def _capturing() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.cuda.is_current_stream_capturing() or bool(get_is_capture_mode())


def prepare(layer) -> None:
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE) or get_disagg().disaggregation_mode != "prefill":
        return
    _LAYERS[id(layer)] = {}


def _device(device) -> dict:
    d = _DEV.get(device.index)
    if d is None:
        d = _DEV[device.index] = {
            "in": torch.zeros(HIGH, _HIDDEN, dtype=torch.bfloat16, device=device),
            "out": torch.zeros(HIGH, _HIDDEN, dtype=torch.bfloat16, device=device),
            "pool": torch.cuda.graph_pool_handle(),
            "stream": torch.cuda.Stream(device=device),
        }
    return d


def input_for(layer, rows: int, device):
    
    if id(layer) not in _LAYERS or not LOW < rows <= HIGH or _capturing():
        return None
    return _device(device)["in"][:rows]


def _stage(rows: int):
    
    if not (e31_fused_route._state["live"] and e31_fused_route._state["depth"] > 0
            and e31_stage_route._state["bufs"] is not None and e31_stage_route._armed(rows) is None):
        return None
    parity = (e31_stage_route._state["epoch"] + 1) % e31_stage_route._SLOTS
    return e31_stage_route._state["bufs"]["slots"][parity], parity


def _capture(mlp, stock, forward_batch, d: dict, bucket: int, out):
    x = d["in"][:bucket]
    current, side = torch.cuda.current_stream(), d["stream"]
    saved = forward_batch.num_token_non_padded
    forward_batch.num_token_non_padded = None
    graph = torch.cuda.CUDAGraph()
    try:
        side.wait_stream(current)
        with torch.cuda.stream(side):
            _state["out"] = out
            try:
                with moe_base.moe_output_buffer_ctx(out):
                    stock(mlp, x, forward_batch, None)
                graph.capture_begin(pool=d["pool"], capture_error_mode="thread_local")
                _state["capture"] = True
                try:
                    with moe_base.moe_output_buffer_ctx(out):
                        y = stock(mlp, x, forward_batch, None)
                finally:
                    _state["capture"] = False
                    graph.capture_end()
            finally:
                _state["out"] = None
        current.wait_stream(side)
    finally:
        forward_batch.num_token_non_padded = saved
    _require(isinstance(y, torch.Tensor) and y.data_ptr() == out.data_ptr() and tuple(y.shape) == (bucket, _HIDDEN),
             f"the captured MoE call of {bucket} rows did not write the output buffer")
    return graph


def _serving(mlp, graphs: dict):
    stock = type(mlp).forward

    def forward(hidden_states, forward_batch=None, gemm_output_zero_allocator=None, *rest, **kwargs):
        rows = hidden_states.shape[0]
        if (rest or kwargs or gemm_output_zero_allocator is not None or forward_batch is None
                or not LOW < rows <= HIGH or _capturing() or hidden_states.dtype != torch.bfloat16
                or hidden_states.dim() != 2 or hidden_states.shape[1] != _HIDDEN
                or not hidden_states.is_contiguous()):
            return stock(mlp, hidden_states, forward_batch, gemm_output_zero_allocator, *rest, **kwargs)
        bucket = -(-rows // STEP) * STEP
        d = _device(hidden_states.device)
        stage = _stage(rows)
        full, key = (d["out"], bucket) if stage is None else (stage[0], (bucket, stage[1]))
        if hidden_states.data_ptr() != d["in"].data_ptr():
            d["in"][:rows].copy_(hidden_states)
        graph = graphs.get(key)
        if graph is None:
            graph = graphs[key] = _capture(mlp, stock, forward_batch, d, bucket, full[:bucket])
        graph.replay()
        return full[:rows]

    return forward


@contextlib.contextmanager
def routed(layer, forward_batch):
    graphs = _LAYERS.get(id(layer))
    if graphs is None or not forward_batch.forward_mode.is_extend() or _capturing():
        yield
        return
    mlp = layer.mlp
    mlp.forward = _serving(mlp, graphs)
    try:
        yield
    finally:
        del mlp.forward
