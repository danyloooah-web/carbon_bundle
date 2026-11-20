from __future__ import annotations
import contextlib
import logging
import time
import torch
from sglang.srt.layers.attention.linear.kda_flashinfer_prefill_graph import FlashInferKDAPrefillGraph
from sglang.srt.runtime_context import get_disagg, get_exec
NODE = "kda"
_GRAPH = FlashInferKDAPrefillGraph
_STOCK_RUN = _GRAPH.run
_COMPILED: dict = {}
_FI: dict = {}
_LIVE = [False]
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA graph padding: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA graph padding: %s", detail)
def prepare(node) -> None:
    _require(_GRAPH.run is _STOCK_RUN, "FlashInferKDAPrefillGraph.run is not stock's")
    if get_disagg().disaggregation_mode == "decode":
        _say("decode", "a PD decode server captures no prefill graph: nothing compiled, its calls stay stock's")
        return
    bound = getattr(getattr(node, "attn", None), "lower_bound", None)
    if bound is None:
        _say("unbounded", f"{type(node).__name__} has no gate lower bound: FlashInfer's BT16 does not serve it")
        return
    track = bool(get_exec().mamba.enable_mamba_extra_buffer)
    key = (float(bound), track)
    if key in _COMPILED:
        return
    from flashinfer import kda_prefill_cute
    from flashinfer.kda_kernels import kda_chunked_bt16

    from glm53f_core import kda_graph_pad_kernel
    lookup = kda_prefill_cute._get_compiled_cute_dsl_kda
    _require(lookup.__module__ == kda_prefill_cute.__name__ and lookup is _FI.get("lookup", lookup),
             "kda_prefill_cute._get_compiled_cute_dsl_kda is not FlashInfer's own")
    began = time.perf_counter()
    _COMPILED[key] = kda_graph_pad_kernel.build(float(bound), track)
    _FI.update(bt16=kda_chunked_bt16, adapter=kda_prefill_cute, lookup=lookup)
    _say(f"compile {key}", f"the guarded BT16 engine kernel compiled for gate bound {key[0]}, tracking {track}, in "
                           f"{time.perf_counter() - began:.1f} s")


def _served(graph, layer, q):
    
    if not _COMPILED:
        _say("none", "nothing compiled in this process: its graph calls are stock's")
        return None
    bound = getattr(layer, "lower_bound", None)
    key = (None if bound is None else float(bound), bool(graph.track))
    _require(key in _COMPILED, f"a graph call at gate bound {key[0]}, tracking {key[1]}; prepare compiled "
                               f"{sorted(_COMPILED)}")
    slots, heads = graph.cu_seqlens.numel() - 1, q.shape[2]
    if _FI["bt16"]._route_for_workspace(slots, heads, q.device, "auto") != "engine":
        _say(f"decomp {slots} {heads}", f"a bucket of {slots} sequence slots at {heads} heads takes FlashInfer's "
                                        "decomposition: stock's")
        return None
    _say("served", f"graph calls run the guarded engine kernel (first: {slots} sequence slots, {heads} heads, "
                   f"{q.shape[1]} tokens, tracking {key[1]})")
    return _COMPILED[key]


def _run(self, *, layer, q, k, v, g, beta, ssm_states, h_track):
    
    args = dict(layer=layer, q=q, k=k, v=v, g=g, beta=beta, ssm_states=ssm_states, h_track=h_track)
    call = _served(self, layer, q)
    if call is None:
        return _STOCK_RUN(self, **args)
    adapter, lookup = _FI["adapter"], _FI["lookup"]
    _require(adapter._get_compiled_cute_dsl_kda is lookup, "FlashInfer's compiled-kernel lookup is already replaced")
    want = (float(layer.lower_bound), torch.float32, True, True, bool(self.track), True, bool(self.track))

    def compiled(*, lower_bound, state_dtype, has_state_in, has_state_out, has_state_ckpt, has_state_indices,
                 has_checkpoint_state_indices):
        
        got = (float(lower_bound), state_dtype, has_state_in, has_state_out, has_state_ckpt, has_state_indices,
               has_checkpoint_state_indices)
        _require(got == want, f"FlashInfer asks for the build {got}, this graph call was compiled as {want}")
        return call
    adapter._get_compiled_cute_dsl_kda = compiled
    try:
        return _STOCK_RUN(self, **args)
    finally:
        adapter._get_compiled_cute_dsl_kda = lookup


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if mode.is_draft_extend_v2() or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False):
        yield
        return
    _require(not _LIVE[0] and _GRAPH.run is _STOCK_RUN, "FlashInferKDAPrefillGraph.run is already replaced (a block's "
             "call inside another)")
    _LIVE[0] = True
    _GRAPH.run = _run
    try:
        yield
    finally:
        _GRAPH.run = _STOCK_RUN
        _LIVE[0] = False
