from __future__ import annotations
import contextlib
import logging
import time
import deep_gemm
import torch
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState, _FfnUpdate
from sglang.srt.model_executor.runner_utils.pool import get_or_create_global_graph_capture_stream
from sglang.srt.models import glm5_next
from glm53f_core import e31_fused_route as fused
NODE = "layer"
HC = 4
MIN_ROWS = 385
MAX_ROWS = 32768
_STOCK = MHCState.update_and_read_ffn_input
_STOCK_READ = MHCState.read_attn_input
_STOCK_EXIT = _FfnUpdate.update
_LAYERS: dict = {}
_STACK: dict = {}
_KEPT: list = [None]
_WARM: set = set()
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf Mega mHC: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf Mega mHC: %s", detail)
def _mega(x, residual, post, comb, weight, eps, facts) -> tuple:
    fn, scales, bases, rms_eps, hc_eps, iterations, post_scale = facts
    rows, hidden = x.shape
    new_residual = torch.empty((rows, HC, hidden), dtype=torch.bfloat16, device=x.device)
    new_post = torch.empty((rows, HC, 1), dtype=torch.float32, device=x.device)
    new_comb = torch.empty((rows, HC, HC), dtype=torch.float32, device=x.device)
    y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device)
    engine_mode = deep_gemm.get_deterministic_algorithms()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.mega_mhc(
            x=x, residual=residual, shifted_prev_mix=None, post_mix=post, comb_res_mix=comb, fn=fn,
            mix_scales=scales, mix_bases=bases, hc_mult=HC, hc_norm_eps=rms_eps, hc_pre_eps=hc_eps,
            hc_post_scale=post_scale, sinkhorn_eps=hc_eps, num_sinkhorn_iters=iterations, rmsnorm_weight=weight,
            rmsnorm_eps=eps, rmsnorm_scale=1.0, new_residual=new_residual, new_prev_mix=None,
            new_post_mix=new_post, new_comb_res_mix=new_comb, y_bf16=y)
    finally:
        deep_gemm.use_deterministic_algorithms(engine_mode)
    return y, new_residual, new_post, new_comb


def _mix_facts(config, stage: str, fn, scales, bases) -> tuple:
    
    mixes = HC * (HC + 2)
    _require(fn.dtype == torch.float32 and tuple(fn.shape) == (mixes, HC * config.hidden_size) and fn.is_contiguous()
             and scales.dtype == torch.float32 and tuple(scales.shape) == (3,)
             and bases.dtype == torch.float32 and tuple(bases.shape) == (mixes,) and fn.is_cuda,
             f"{stage} mix parameters {fn.dtype} {tuple(fn.shape)}, {tuple(scales.shape)}, {tuple(bases.shape)} on "
             f"{fn.device}")
    return (fn, scales, bases, float(config.rms_norm_eps), float(config.hc_eps), int(config.hc_sinkhorn_iters),
            float(glm5_next._MHC_POST_MULT_VALUE))


def prepare(layer) -> None:
    if id(layer) in _LAYERS:
        return
    _require(callable(getattr(deep_gemm, "mega_mhc", None)), "this engine's deep_gemm has no mega_mhc")
    config = layer.config
    _require(bool(config.mhc) and config.hc_mult == HC, f"a layer of {config.hc_mult} streams (mhc {config.mhc})")
    hidden = config.hidden_size
    _require(hidden % 1024 == 0, f"hidden size {hidden}: the kernel splits it into 16 runs of 64-wide blocks")
    _require(not torch.cuda.is_current_stream_capturing(), "prepare ran under graph capture")
    facts = _mix_facts(config, "FFN", layer.hc_ffn_fn.data, layer.hc_ffn_scale.data, layer.hc_ffn_base.data)
    opening = _mix_facts(config, "attention", layer.hc_attn_fn.data, layer.hc_attn_scale.data,
                         layer.hc_attn_base.data)
    norm = layer.input_layernorm
    _LAYERS[id(layer)] = (facts, opening, norm.weight.data, float(norm.variance_epsilon))
    _STACK[(id(config), bool(layer.is_nextn), int(layer.layer_id))] = layer
    began = time.perf_counter()
    warmed = []
    for stream in (torch.cuda.current_stream(), get_or_create_global_graph_capture_stream()):
        if stream.cuda_stream in _WARM:
            continue
        with torch.cuda.stream(stream):
            x = torch.zeros((MIN_ROWS, hidden), dtype=torch.bfloat16, device=facts[0].device)
            _mega(x, torch.zeros((MIN_ROWS, HC, hidden), dtype=torch.bfloat16, device=x.device),
                  torch.zeros((MIN_ROWS, HC, 1), dtype=torch.float32, device=x.device),
                  torch.zeros((MIN_ROWS, HC, HC), dtype=torch.float32, device=x.device),
                  torch.ones((hidden,), dtype=torch.bfloat16, device=x.device), float(config.rms_norm_eps), facts)
        torch.cuda.synchronize()
        _WARM.add(stream.cuda_stream)
        warmed.append(stream.cuda_stream)
    if warmed:
        logging.getLogger(__name__).info(
            "glmf Mega mHC: the kernel ran on %d stream(s) in %.1f s (DeepGEMM's compile included); it serves "
            "boundaries of %d to %d tokens", len(warmed), time.perf_counter() - began, MIN_ROWS, MAX_ROWS)


def _outside(state, hidden_states, residual, has_norm: bool):
    
    rows = hidden_states.shape[0]
    if fused._state["pending"] is not None:
        return "pending", "the fused all-reduce holds this boundary's sum for its own post"
    if not has_norm:
        return "norm", "no input norm to fuse"
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return "rows", f"{rows} tokens, outside {MIN_ROWS} to {MAX_ROWS}"
    if state.h_res is None or state.h_post is None:
        return "mixes", "no mixes from the stage's read"
    if not (hidden_states.dtype == torch.bfloat16 and hidden_states.dim() == 2 and hidden_states.is_contiguous()
            and residual.dtype == torch.bfloat16 and residual.is_contiguous()
            and residual.numel() == rows * HC * hidden_states.shape[1]
            and state.h_post.dtype == torch.float32 and state.h_post.is_contiguous()
            and state.h_post.numel() == rows * HC and state.h_res.dtype == torch.float32
            and state.h_res.is_contiguous() and state.h_res.numel() == rows * HC * HC):
        return "layout", (f"output {hidden_states.dtype} {tuple(hidden_states.shape)}, streams {residual.dtype} "
                          f"{tuple(residual.shape)}")
    return None
def _boundary(facts):
    

    def update_and_read_ffn_input(state, hidden_states, residual, out_norm=None):
        why = _outside(state, hidden_states, residual, out_norm is not None)
        if why is not None:
            _say(why[0], f"stock's chain serves a middle boundary: {why[1]}")
            return _STOCK(state, hidden_states, residual, out_norm=out_norm)
        rows, hidden = hidden_states.shape
        y, new_residual, new_post, new_comb = _mega(
            hidden_states, residual.view(rows, HC, hidden), state.h_post.view(rows, HC, 1),
            state.h_res.view(rows, HC, HC), out_norm.weight.data, float(out_norm.variance_epsilon), facts)
        state.h_res = new_comb.view(rows, HC * HC)
        state.h_post = new_post.view(rows, HC)
        _say("ours", f"one kernel serves middle boundaries (first: {rows} tokens)")
        return y, new_residual.view(rows, HC * hidden)

    return update_and_read_ffn_input


def _update(update_, hidden_states, residual):
    
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
        why = _outside(state, hidden_states, residual, True)
    if why is not None:
        _say("exit " + why[0], f"stock's post serves a layer's exit: {why[1]}")
        return _STOCK_EXIT(update_, hidden_states, residual)
    _, attn_facts, weight, eps = following
    rows, hidden = hidden_states.shape
    y, new_residual, new_post, new_comb = _mega(
        hidden_states, residual.view(rows, HC, hidden), state.h_post.view(rows, HC, 1),
        state.h_res.view(rows, HC, HC), weight, eps, attn_facts)
    state.clear_coefficients()
    streams = new_residual.view(rows, HC * hidden)
    _KEPT[0] = (streams, y, new_comb.view(rows, HC * HC), new_post.view(rows, HC), weight, eps)
    _say("exit ours", f"one kernel serves a layer's exit and the next layer's read (first: {rows} tokens)")
    return streams


def _read(state, hidden_states, out_norm=None):
    
    kept, _KEPT[0] = _KEPT[0], None
    if kept is not None:
        streams, y, comb, post, weight, eps = kept
        if (streams is hidden_states and out_norm is not None
                and out_norm.weight.data.data_ptr() == weight.data_ptr()
                and float(out_norm.variance_epsilon) == eps):
            state.h_res, state.h_post = comb, post
            return y, hidden_states
        _say("read other", "stock's read serves a layer that was handed other streams or another norm than the "
                           "previous exit prepared")
    return _STOCK_READ(state, hidden_states, out_norm)


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    entry = _LAYERS.get(id(layer))
    if entry is None:
        yield
        return
    MHCState.update_and_read_ffn_input = _boundary(entry[0])
    MHCState.read_attn_input = _read
    _FfnUpdate.update = _update
    try:
        yield
    finally:
        MHCState.update_and_read_ffn_input = _STOCK
        MHCState.read_attn_input = _STOCK_READ
        _FfnUpdate.update = _STOCK_EXIT
