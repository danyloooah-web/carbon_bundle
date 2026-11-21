from __future__ import annotations
import contextlib
import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState, _FfnUpdate
from sglang.srt.models import glm5_next
from glm53f_core import e31_fused_route as fused
from glm53f_core import mhc_ar_route as fused_ar
NODE = "layer"
_HC, _HIDDEN, _MIXES = 4, 4096, 24
MIN_ROWS, MAX_ROWS = 6, 384
UPPER_ROWS = (385, 768)
_RMS_EPS, _HC_EPS, _ITERS, _POST_MULT, _NORM_EPS = 1e-5, 1e-6, 20, 2.0, 1e-5
_LAYERS: dict = {}
_STACK: dict = {}
_KEPT: list = [None]
_state = {"native": None, "limits": {}, "said": set(), "tally": {}, "side": {}}
_SPLIT: list = []
def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-mhcfi] {detail}", flush=True)
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-mhcfi] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"FlashInfer mHC: {why}")
def _load_native():
    import mhc_fi_native
    return mhc_fi_native
def _dense(t, dtype, shape: tuple, align: int = 16) -> bool:
    return (isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == dtype and tuple(t.shape) == shape
            and t.is_contiguous() and t.data_ptr() % align == 0)
def _facts(stage_name: str, fn, scale, base) -> tuple:
    _require(_dense(fn, torch.float32, (_MIXES, _HC * _HIDDEN), 32) and _dense(scale, torch.float32, (3,))
             and _dense(base, torch.float32, (_MIXES,)),
             f"{stage_name} mix parameters {getattr(fn, 'dtype', None)} {tuple(getattr(fn, 'shape', ()))}, "
             f"{tuple(getattr(scale, 'shape', ()))}, {tuple(getattr(base, 'shape', ()))}")
    return fn, scale, base
def prepare(layer) -> None:
    
    if id(layer) in _LAYERS:
        return
    _require(not torch.cuda.is_current_stream_capturing(), "prepare ran under graph capture")
    config = layer.config
    _require(bool(config.mhc) and config.hc_mult == _HC and config.hidden_size == _HIDDEN,
             f"a layer of {config.hc_mult} streams of {config.hidden_size} (mhc {config.mhc})")
    _require((config.rms_norm_eps, config.hc_eps, config.hc_sinkhorn_iters) == (_RMS_EPS, _HC_EPS, _ITERS),
             f"config rms_norm_eps {config.rms_norm_eps}, hc_eps {config.hc_eps}, hc_sinkhorn_iters "
             f"{config.hc_sinkhorn_iters}: the kernel is built for {_RMS_EPS}, {_HC_EPS}, {_ITERS}")
    _require(float(glm5_next._MHC_POST_MULT_VALUE) == _POST_MULT,
             f"post scale {glm5_next._MHC_POST_MULT_VALUE}: the kernel is built for {_POST_MULT}")
    device = layer.hc_ffn_fn.data.device
    if _state["native"] is None:
        native = _load_native()
        _require(callable(getattr(native, "prepare", None)) and callable(getattr(native, "limits", None))
                 and callable(getattr(native, "run", None)) and callable(getattr(native, "sinkhorn", None)),
                 "mhc_fi_native lacks prepare / limits / run / sinkhorn")
        _state["native"] = native
    if device.index not in _state["limits"]:
        _state["native"].prepare(device.index)
        limits = tuple(_state["native"].limits(device.index))
        _require(limits[0] == 1 and min(limits[1:]) >= 1, f"no group of 16 CTAs fits cuda:{device.index} ({limits})")
        _state["limits"][device.index] = limits
        _state["side"][device.index] = torch.cuda.Stream(device=device)
        _tally("prepared", lambda: f"cuda:{device.index}, co-resident 16-CTA clusters (decode, stream, short, "
                                   f"crossover, long) {limits[1:]}; rows {MIN_ROWS}-{MAX_ROWS} and "
                                   f"{UPPER_ROWS[0]}-{UPPER_ROWS[1]}")
    norm = layer.input_layernorm
    _LAYERS[id(layer)] = (
        _facts("FFN", layer.hc_ffn_fn.data, layer.hc_ffn_scale.data, layer.hc_ffn_base.data),
        _facts("attention", layer.hc_attn_fn.data, layer.hc_attn_scale.data, layer.hc_attn_base.data),
        (norm.weight.data, float(norm.variance_epsilon)))
    _STACK[(id(config), bool(layer.is_nextn), int(layer.layer_id))] = layer


def _outside(state, hidden_states, residual, weight, eps):
    
    if fused._state["pending"] is not None:
        return "pending", "E31 holds this boundary's sum for its own post (sum + post in one kernel)"
    rows = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
    if not (MIN_ROWS <= rows <= MAX_ROWS or UPPER_ROWS[0] <= rows <= UPPER_ROWS[1]):
        return "rows", f"{rows} tokens, outside {MIN_ROWS} to {MAX_ROWS} and {UPPER_ROWS[0]} to {UPPER_ROWS[1]}"
    if weight is None:
        return "norm", "no input norm to fuse"
    if not (_dense(weight, torch.bfloat16, (_HIDDEN,)) and eps == _NORM_EPS):
        return "norm layout", f"an input norm of {weight.dtype} {tuple(weight.shape)}, epsilon {eps}"
    h_post, h_res = state.h_post, state.h_res
    if h_res is None or h_post is None:
        return "mixes", "no mixes from the stage's read"
    if not (_dense(hidden_states, torch.bfloat16, (rows, _HIDDEN)) and _dense(residual, torch.bfloat16, tuple(residual.shape))
            and residual.numel() == rows * _HC * _HIDDEN and _dense(h_post, torch.float32, tuple(h_post.shape))
            and h_post.numel() == rows * _HC and _dense(h_res, torch.float32, tuple(h_res.shape))
            and h_res.numel() == rows * _HC * _HC):
        return "layout", (f"output {hidden_states.dtype} {tuple(hidden_states.shape)}, streams {residual.dtype} "
                          f"{tuple(residual.shape)}, mixes {h_post.dtype} {tuple(h_post.shape)} / {h_res.dtype} "
                          f"{tuple(h_res.shape)} (dense at 16-byte addresses wanted)")
    return None


def _owed(hidden_states, why):
    
    partial = fused_ar.pending()
    if partial is None:
        return hidden_states
    _require(hidden_states is partial, "a boundary was handed another tensor while a partial is pending (the sum's "
                                       "result moved or was replaced before its boundary)")
    return hidden_states if why is None else fused_ar.complete(partial)


def _join() -> None:
    
    for index in {t.device.index for t in _SPLIT}:
        torch.cuda.current_stream(index).wait_stream(_state["side"][index])
    _SPLIT.clear()

def _run(hidden_states, residual, state, facts, weight) -> tuple:
    
    rows, dev = hidden_states.shape[0], hidden_states.device
    fn, scale, base = facts
    streams = torch.empty((rows, _HC * _HIDDEN), dtype=torch.bfloat16, device=dev)
    post = torch.empty((rows, _HC), dtype=torch.float32, device=dev)
    comb = torch.empty((rows, _HC * _HC), dtype=torch.float32, device=dev)
    with mhc.use_symmetric_memory(mhc.get_parallel().tp_group, disabled=not mhc.is_allocation_symmetric()):
        y = torch.empty((rows, _HIDDEN), dtype=torch.bfloat16, device=dev)
    args = (hidden_states, residual.view(rows, _HC, _HIDDEN), state.h_post.view(rows, _HC),
            state.h_res.view(rows, _HC, _HC), fn, scale, base, weight, streams.view(rows, _HC, _HIDDEN), post,
            comb.view(rows, _HC, _HC), y)
    split = 1 if torch.cuda.is_current_stream_capturing() else 0
    if fused_ar.pending() is hidden_states:
        _state["native"].run_ar(*args, *fused_ar.take(hidden_states), split)
    else:
        _state["native"].run(*args, split)
    if split:
        side = _state["side"][dev.index]
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            _state["native"].sinkhorn(comb)
        _SPLIT.append(comb)
    return y, streams, post, comb


def _middle(found, facts):
    

    def update_and_read_ffn_input(state, hidden_states, residual, out_norm=None):
        _join()
        weight = None if out_norm is None else out_norm.weight.data
        eps = None if out_norm is None else float(out_norm.variance_epsilon)
        why = _outside(state, hidden_states, residual, weight, eps)
        hidden_states = _owed(hidden_states, why)
        if why is not None:
            _say(why[0], f"the method found serves a middle boundary: {why[1]}")
            return found(state, hidden_states, residual, out_norm=out_norm)
        y, streams, post, comb = _run(hidden_states, residual, state, facts, weight)
        state.h_res, state.h_post = comb, post
        _tally("middle", lambda: f"{streams.shape[0]} rows")
        return y, streams

    return update_and_read_ffn_input


def _exiting(found):
    

    def update(update_, hidden_states, residual):
        _join()
        state = update_.state
        owner = getattr(state.hc_post, "__self__", None)
        following = None
        if id(owner) in _LAYERS:
            after = _STACK.get((id(owner.config), bool(owner.is_nextn), int(owner.layer_id) + 1))
            following = None if after is None else _LAYERS.get(id(after))
        if state.is_last_layer:
            why = ("last", "the last layer's exit also contracts the streams")
        elif following is None:
            why = ("next", "the next layer is not prepared by this bundle")
        else:
            why = _outside(state, hidden_states, residual, *following[2])
        hidden_states = _owed(hidden_states, why)
        if why is not None:
            _say("exit " + why[0], f"the method found serves a layer's exit: {why[1]}")
            return found(update_, hidden_states, residual)
        _, attn_facts, (weight, eps) = following
        y, streams, post, comb = _run(hidden_states, residual, state, attn_facts, weight)
        state.clear_coefficients()
        _KEPT[0] = (streams, y, comb, post, weight, eps)
        _tally("exit", lambda: f"{streams.shape[0]} rows")
        return streams
    return update


def _reading(found):
    

    def read_attn_input(state, hidden_states, out_norm=None):
        kept, _KEPT[0] = _KEPT[0], None
        if kept is not None:
            streams, y, comb, post, weight, eps = kept
            if (streams is hidden_states and out_norm is not None
                    and out_norm.weight.data.data_ptr() == weight.data_ptr()
                    and float(out_norm.variance_epsilon) == eps):
                state.h_res, state.h_post = comb, post
                return y, hidden_states
            _say("read other", "the method found serves a layer handed other streams or another norm than the "
                               "previous exit prepared")
        return found(state, hidden_states, out_norm)

    return read_attn_input


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    entry = _LAYERS.get(id(layer))
    if entry is None:
        yield
        return
    found = (MHCState.update_and_read_ffn_input, MHCState.read_attn_input, _FfnUpdate.update)
    MHCState.update_and_read_ffn_input = _middle(found[0], entry[0])
    MHCState.read_attn_input = _reading(found[1])
    _FfnUpdate.update = _exiting(found[2])
    fused_ar.boundaries(True)
    try:
        yield
    finally:
        fused_ar.boundaries(False)
        MHCState.update_and_read_ffn_input, MHCState.read_attn_input, _FfnUpdate.update = found
