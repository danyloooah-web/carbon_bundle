from __future__ import annotations
import contextlib
import logging
import moe_finalize_native
import torch
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route, moe_finalize_route, moe_graph_route
NODE = "layer"
MIN_ROWS, MAX_ROWS = 193, 32768
ISSUE_FIRST = False
FORK_AFTER_ROUTED = False
SPLIT = True
SEED = True
_MOE = deepseek_v2.DeepseekV2MoE
_STOCK_NORMAL = _MOE.forward_normal
_STOCK_SHARED = _MOE._forward_shared_experts
_LAYERS: set = set()
_STREAMS: dict = {}
_CALL: dict = {}
_SET: list = []
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf MoE shared overlap: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf MoE shared overlap: %s", detail)
def prepare(layer) -> None:
    _require(not (ISSUE_FIRST and FORK_AFTER_ROUTED), "ISSUE_FIRST forks at the input, not after the routed experts")
    _require(not (SEED and FORK_AFTER_ROUTED), "SEED needs the shared rows before GEMM2, not after the routed experts")
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, _MOE) or get_disagg().disaggregation_mode == "decode":
        return
    if not (mlp.shared_experts is not None and mlp.num_fused_shared_experts == 0 and not mlp._shared_expert_tp1
            and not mlp._fuse_shared_experts_inside_sbo and not mlp._enable_a2a_moe
            and mlp.experts.moe_runner_config.inplace is False):
        _say("layer", f"layer {getattr(layer, 'layer_id', '?')}'s MoE does not defer a separate TP-sharded shared "
                      "expert on the standard dispatch: stock's schedule serves it")
        return
    device = torch.cuda.current_device()
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device=device)
    _LAYERS.add(id(layer))


def _outside(moe, hidden_states, kw: dict) -> tuple[str, str] | None:
    rows = hidden_states.shape[0] if isinstance(hidden_states, torch.Tensor) else -1
    if kw.get("skip_shared_experts"):
        return "skip", "a call that skips the shared expert"
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return f"rows {rows < MIN_ROWS}", f"{rows} rows, outside {MIN_ROWS} to {MAX_ROWS}"
    if not moe_graph_route._state["capture"] and (
            deepseek_v2.get_is_capture_mode() or torch.cuda.is_current_stream_capturing()
            or deepseek_v2.is_in_breakable_cuda_graph() or deepseek_v2.is_in_tc_piecewise_cuda_graph()):
        return "graph", "a call under capture or in a breakable or piecewise graph"
    if hidden_states.device.index not in _STREAMS:
        return "device", f"a call on device {hidden_states.device.index}, which prepare never saw"
    if moe._moe_quant_once_enabled():
        return "quant-once", "a quant-once call (the shared expert takes the routed path's FP8 pair)"
    return None


def _issue(moe, hidden_states, gemm_output_zero_allocator) -> None:
    
    side = _STREAMS[hidden_states.device.index]
    side.wait_event(_CALL["fork"])
    down = moe.shared_experts.down_proj
    dest = moe_graph_route._state["out"] if _CALL.get("seed") else None
    if not (isinstance(dest, torch.Tensor) and dest.dtype == torch.bfloat16 and dest.is_contiguous()
            and dest.device == hidden_states.device and tuple(dest.shape) == tuple(hidden_states.shape)):
        dest = None
    if dest is not None:
        fp8_linear_route._DEST[id(down)] = dest
    try:
        with torch.cuda.stream(side):
            rows = _STOCK_SHARED(moe, hidden_states, gemm_output_zero_allocator)
            done = torch.cuda.Event()
            done.record(side)
    finally:
        fp8_linear_route._DEST.pop(id(down), None)
    hidden_states.record_stream(side)
    owned = dest is not None and isinstance(rows, torch.Tensor) and rows.data_ptr() == dest.data_ptr()
    _CALL.update(rows=rows, done=done, owned=owned)


def _forward_normal(self, hidden_states, gemm_output_zero_allocator=None, *args, **kw):
    
    why = _outside(self, hidden_states, kw)
    if why is not None:
        _say(why[0], f"stock's schedule serves a call ({why[1]})")
        return _STOCK_NORMAL(self, hidden_states, gemm_output_zero_allocator, *args, **kw)
    _require(not _CALL, "a MoE call inside another's")
    fork = torch.cuda.Event()
    fork.record(torch.cuda.current_stream())
    split = SPLIT and not kw.get("return_moe_output") and not kw.get("leave_output_sum")
    seed = SEED and split
    _CALL.update(moe=self, fork=fork, input=hidden_states, zero=gemm_output_zero_allocator, split=split, seed=seed,
                 asked=False)
    try:
        if ISSUE_FIRST or seed:
            _issue(self, hidden_states, gemm_output_zero_allocator)
        out = _STOCK_NORMAL(self, hidden_states, gemm_output_zero_allocator, *args, **kw)
        asked, owed, seeded = _CALL["asked"], "owed" in _CALL, _CALL.get("seeded") is not None
    finally:
        _CALL.clear()
    _require(asked, "forward_normal returned without asking for the shared rows")
    _require(not owed, "forward_normal returned with the split shared rows never added")
    _say("ours", f"the shared expert ran beside the routed experts (first: {hidden_states.shape[0]} rows; issued "
                 f"{'first' if ISSUE_FIRST or seed else 'at the ask'}, fork "
                 f"{'after the routed experts' if FORK_AFTER_ROUTED else 'at the input'}, split finalize {split})")
    if seeded:
        _say("seeded" + ("-graph" if moe_graph_route._state["capture"] else ""),
             f"GEMM2 summed onto the shared rows, no zero and no add (first: {hidden_states.shape[0]} rows"
             f"{', inside moe_graph_route capture' if moe_graph_route._state['capture'] else ', eager'})")
    return out


def _shared(self, hidden_states, gemm_output_zero_allocator=None, pre_quant_input=None):
    
    if not _CALL:
        return _STOCK_SHARED(self, hidden_states, gemm_output_zero_allocator, pre_quant_input=pre_quant_input)
    _require(_CALL["moe"] is self and not _CALL["asked"], "a second or foreign ask for the shared rows in one call")
    _require(pre_quant_input is None, "a quant-once pair for the shared expert")
    _require(hidden_states is _CALL["input"] and gemm_output_zero_allocator is _CALL["zero"],
             "the shared expert asked for on another input than the call's")
    _CALL["asked"] = True
    main = torch.cuda.current_stream()
    if "done" not in _CALL:
        if FORK_AFTER_ROUTED:
            _CALL["fork"].record(main)
        _issue(self, hidden_states, gemm_output_zero_allocator)
    rows = _CALL["rows"]
    _require(isinstance(rows, torch.Tensor), "the shared expert returned no rows")
    if not _CALL.get("owned"):
        rows.record_stream(main)
    if _CALL["split"]:
        _CALL["owed"] = rows
    else:
        main.wait_event(_CALL["done"])
    return rows


def seed(out: torch.Tensor) -> bool:
    
    if not _CALL.get("seed") or "done" not in _CALL or _CALL.get("seeded") is not None:
        return False
    rows = _CALL["rows"]
    if not (isinstance(rows, torch.Tensor) and tuple(rows.shape) == tuple(out.shape) and rows.dtype == out.dtype
            and rows.is_contiguous() and out.is_contiguous()):
        _say("unseeded", f"shared rows {tuple(getattr(rows, 'shape', ()))} do not fit the MoE output "
                         f"{tuple(out.shape)}: the zeroed output and the add serve")
        return False
    main = torch.cuda.current_stream()
    main.wait_event(_CALL["done"])
    if rows.data_ptr() != out.data_ptr():
        if not _CALL.get("owned"):
            rows.record_stream(main)
        out.copy_(rows)
    _CALL["seeded"] = out.data_ptr()
    return True


def _add(experts, routed, shared, routed_scaling_factor):
    
    if not (_CALL.get("owed") is not None and shared is _CALL["owed"]):
        return _SET[2](experts, routed, shared, routed_scaling_factor)
    if _CALL.get("seeded") is not None:
        _require(not moe_finalize_route._PENDING, "a finalize owed in a seeded call")
        _require(isinstance(routed, torch.Tensor) and routed.data_ptr() == _CALL["seeded"],
                 "the seeded output is not the rows handed to the add")
        del _CALL["owed"]
        return routed
    owed = moe_finalize_route._PENDING
    if owed:
        out, gemm2, weights, permuted, pdl = owed[0]
        _require(isinstance(routed, torch.Tensor) and routed.data_ptr() == out.data_ptr()
                 and tuple(routed.shape) == tuple(out.shape) and routed.is_contiguous(),
                 "the split finalize's owed output is not the rows handed to the add")
        owed.pop()
        moe_finalize_native.routed(gemm2, permuted, weights, routed, pdl)
    torch.cuda.current_stream().wait_event(_CALL["done"])
    del _CALL["owed"]
    return _SET[2](experts, routed, shared, routed_scaling_factor)

@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if id(layer) not in _LAYERS:
        yield
        return
    _require(not _SET, "a layer's call inside another's")
    _SET[:] = [_MOE.forward_normal, _MOE._forward_shared_experts, deepseek_v2.maybe_fuse_routed_scale_and_shared_add]
    _MOE.forward_normal, _MOE._forward_shared_experts = _forward_normal, _shared
    deepseek_v2.maybe_fuse_routed_scale_and_shared_add = _add
    try:
        yield
    finally:
        _MOE.forward_normal, _MOE._forward_shared_experts = _SET[0], _SET[1]
        deepseek_v2.maybe_fuse_routed_scale_and_shared_add = _SET[2]
        _SET.clear()
