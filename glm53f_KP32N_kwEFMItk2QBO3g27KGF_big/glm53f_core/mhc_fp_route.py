from __future__ import annotations
import contextlib
import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState, _FfnUpdate
from sglang.srt.models import glm5_next
from glm53f_core import e31_fused_route as fused
from glm53f_core import e31_stage_route as stage
from glm53f_core import moe_graph_route
NODE = "layer"
_HC, _HIDDEN, _MIXES = 4, 4096, 24
MIN_ROWS, MAX_ROWS = 8193, 16384
_TILE, _TILES_A_CLUSTER = 16, 20
_RMS_EPS, _HC_EPS, _ITERS, _POST_MULT, _NORM_EPS = 1e-5, 1e-6, 20, 2.0, 1e-5
_LAYERS: dict = {}
_STACK: dict = {}
_KEPT: list = [None]
_state = {"native": None, "clusters": {}, "said": set(), "tally": {}}
def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-mhcfp] {detail}", flush=True)
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-mhcfp] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"FlashInfer fused-peer mHC: {why}")
def _load_native():
    import mhc_fp_native

    return mhc_fp_native


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
    groups = fused._state["groups"]
    _require(groups is not None, "e31_layer_route's prepare has not run before this component's")
    if not groups:
        _say("inactive", f"E31 is inactive on this engine ({fused._state['why']}): the method found serves every "
                         "boundary")
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
                 and callable(getattr(native, "run_peer", None)), "mhc_fp_native lacks prepare / limits / run_peer")
        _state["native"] = native
    if device.index not in _state["clusters"]:
        _state["native"].prepare(device.index)
        limits = tuple(_state["native"].limits(device.index))

        _require(limits[0] == 1 and 1 <= limits[1] and _TILES_A_CLUSTER * limits[1] <= -(-MIN_ROWS // _TILE),
                 f"cuda:{device.index}: {limits[1]} co-resident 16-CTA clusters of the kernel; {MIN_ROWS} rows need "
                 f"1 to {-(-MIN_ROWS // _TILE) // _TILES_A_CLUSTER} for FlashInfer's long-stream regime")
        _state["clusters"][device.index] = limits[1]
        _tally("prepared", lambda: f"cuda:{device.index}, {limits[1]} co-resident 16-CTA clusters, {limits[2]} B of "
                                   f"shared memory a CTA; rows {MIN_ROWS}-{MAX_ROWS} in E31's slot form")
    norm = layer.input_layernorm
    _LAYERS[id(layer)] = (
        _facts("FFN", layer.hc_ffn_fn.data, layer.hc_ffn_scale.data, layer.hc_ffn_base.data),
        _facts("attention", layer.hc_attn_fn.data, layer.hc_attn_scale.data, layer.hc_attn_base.data),
        (norm.weight.data, float(norm.variance_epsilon)))
    _STACK[(id(config), bool(layer.is_nextn), int(layer.layer_id))] = layer

def _outside(state, hidden_states, residual, weight, eps):
    
    pending = fused._state["pending"]
    if pending is None:
        return "complete", "the sum is complete (no sum pending)"
    if stage._state["pending"] is not pending:
        return "stage copy", "E31's stage copy holds the sum (E31's post serves it)"
    if hidden_states is not pending[0]:
        return "other pending", "E31 holds the sum of another tensor (its post raises on this)"
    rows = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return "rows", f"{rows} tokens, outside {MIN_ROWS} to {MAX_ROWS}"
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
        return "layout", (f"partial {hidden_states.dtype} {tuple(hidden_states.shape)}, streams {residual.dtype} "
                          f"{tuple(residual.shape)}, mixes {h_post.dtype} {tuple(h_post.shape)} / {h_res.dtype} "
                          f"{tuple(h_res.shape)} (dense at 16-byte addresses wanted)")
    return None


def _run(residual, state, facts, weight, into=None) -> tuple:
    
    partial, epoch = fused._state["pending"]
    b = stage._state["bufs"]
    rows, dev = partial.shape[0], partial.device
    fn, scale, base = facts
    streams = torch.empty((rows, _HC * _HIDDEN), dtype=torch.bfloat16, device=dev)
    post = torch.empty((rows, _HC), dtype=torch.float32, device=dev)
    comb = torch.empty((rows, _HC * _HC), dtype=torch.float32, device=dev)
    if into is not None and _dense(into, torch.bfloat16, (rows, _HIDDEN)) and into.device == dev:
        y = into
    else:
        with mhc.use_symmetric_memory(mhc.get_parallel().tp_group, disabled=not mhc.is_allocation_symmetric()):
            y = torch.empty((rows, _HIDDEN), dtype=torch.bfloat16, device=dev)
    _state["native"].run_peer(partial, b["peer_slots"][epoch % stage._SLOTS], residual.view(rows, _HC, _HIDDEN),
                              state.h_post.view(rows, _HC), state.h_res.view(rows, _HC, _HC), fn, scale, base, weight,
                              streams.view(rows, _HC, _HIDDEN), post, comb.view(rows, _HC, _HC), y, b["flags"],
                              b["peer_flags"], b["published"], epoch)
    fused._state["pending"] = stage._state["pending"] = None
    return y, streams, post, comb


def _middle(found, facts, layer):

    def update_and_read_ffn_input(state, hidden_states, residual, out_norm=None):
        weight = None if out_norm is None else out_norm.weight.data
        eps = None if out_norm is None else float(out_norm.variance_epsilon)
        why = _outside(state, hidden_states, residual, weight, eps)
        if why is not None:
            _say(why[0], f"the method found serves a middle boundary: {why[1]}")
            return found(state, hidden_states, residual, out_norm=out_norm)
        into = moe_graph_route.input_for(layer, hidden_states.shape[0], hidden_states.device)
        y, streams, post, comb = _run(residual, state, facts, weight, into)
        state.h_res, state.h_post = comb, post
        _tally("middle, fused peer sum" + (" into the MoE graph input" if y is into else ""),
               lambda: f"{streams.shape[0]} rows")
        return y, streams

    return update_and_read_ffn_input


def _exiting(found):
    

    def update(update_, hidden_states, residual):
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
        if why is not None:
            _say("exit " + why[0], f"the method found serves a layer's exit: {why[1]}")
            return found(update_, hidden_states, residual)
        _, attn_facts, (weight, eps) = following
        y, streams, post, comb = _run(residual, state, attn_facts, weight)
        state.clear_coefficients()
        _KEPT[0] = (streams, y, comb, post, weight, eps)
        _tally("exit, fused peer sum", lambda: f"{streams.shape[0]} rows")
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
    MHCState.update_and_read_ffn_input = _middle(found[0], entry[0], layer)
    MHCState.read_attn_input = _reading(found[1])
    _FfnUpdate.update = _exiting(found[2])
    try:
        yield
    finally:
        MHCState.update_and_read_ffn_input, MHCState.read_attn_input, _FfnUpdate.update = found
