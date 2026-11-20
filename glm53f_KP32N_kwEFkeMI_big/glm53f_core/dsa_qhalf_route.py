from __future__ import annotations
import contextlib
import logging
import flashinfer.decode
import torch
import torch.nn.functional as F
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_disagg
from glm53f_core import dsa_rows_route as row_owned
NODE = "dsa"
HEADS, WIDTH = row_owned.HEADS, row_owned.WIDTH
FP8 = torch.float8_e4m3fn
_STOCK_QB = DeepseekV2AttentionMLA.q_b_proj_forward
_WEIGHTS: dict = {}
_CALL: dict = {}
_SET: list = []
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA q-half: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA q-half: %s", detail)
class _QueryHere:
    def __init__(self, native):
        self.native = native
    def push(self, src, dsts, pads, rank, mode, *rest):
        return 0 if mode == 0 else self.native.push(src, dsts, pads, rank, mode, *rest)
def _not_linear(layer, m: int) -> str | None:
    n, k = layer.weight.shape
    if getattr(layer, "small_m_rowmajor", False) or getattr(layer, "small_m_tiled_weight", None) is not None:
        return "gfx95's small-M GEMM"
    gemm = unquant.get_bf16_gemm_backend()
    if not (gemm.is_cutedsl() or gemm.is_gemv()):
        return None
    if unquant._mm_bf16 is not None and 0 < m <= unquant._BF16_LOW_M_AUTOTUNE_MAX_M:
        return "FlashInfer's autotuned CuTe DSL GEMM"
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        return "a tuned split-K GEMM"
    if any(rule is not None and rule(m, n, k) for rule in (unquant._use_hopper_bf16_tiny_gemv,
                                                           unquant._use_hopper_bf16_gemv)):
        return "a Hopper GEMV"
    if unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k):
        return "the CuTe DSL TGV GEMM"
    return None


def prepare(node) -> None:
    
    _require("group" in row_owned._STATE, "dsa_rows_route is not prepared before this route (the components' order)")
    if id(node) in _WEIGHTS:
        return
    group = row_owned._STATE["group"]
    why = ("A1 is inactive" if group is None else "the MTP draft layer" if getattr(node, "is_nextn", False)
           else "a PD prefill server" if get_disagg().disaggregation_mode == "prefill" else None)
    if why is not None:
        _WEIGHTS[id(node)] = None
        _say("inactive " + why, f"{why}: no replicas, A1's queries")
        return
    world, linear, state = row_owned._STATE["world"], node.q_b_proj, row_owned._STATE
    heads, qb, kc, step = HEADS // world, node.q_b_proj.weight, node.w_kc, row_owned.NEXT_N * world
    for ok, why in ((type(node).q_b_proj_forward is _STOCK_QB, "q_b_proj_forward is overridden"),
                    (node._q_b_proj_verified_shape is False, "q_b_proj takes the min-latency fused-A GEMM"),
                    (type(linear.quant_method) is unquant.UnquantizedLinearMethod and linear.bias is None,
                     f"q_b_proj's method is {type(linear.quant_method)!r} or it has a bias"),
                    (not node.use_deep_gemm_bmm, "w_kc takes DeepGEMM's bmm")):
        _require(ok, f"stock's queries are not the F.linear and torch.bmm this route replicates: {why}")
    served = range(-(-row_owned.MIN_ROWS // step) * step, state["capacity"] + 1, step)
    off = [(m, why) for m in served for why in (_not_linear(linear, m),) if why is not None]
    _require(not off, f"stock's q_b_proj {tuple(qb.shape)} runs {off[0][1] if off else ''} at "
                      f"{off[0][0] if off else 0} rows ({len(off)} served row counts), not the F.linear it replicates")
    nope = node.qk_nope_head_dim
    _require(node.num_local_heads == heads and node.qk_rope_head_dim == 0 and node.kv_lora_rank == WIDTH
             and qb.dtype == kc.dtype == torch.bfloat16 and tuple(qb.shape) == (heads * nope, node.q_lora_rank)
             and tuple(kc.shape) == (heads, nope, WIDTH) and qb.is_contiguous() and kc.transpose(1, 2).is_contiguous(),
             f"q_b_proj {qb.dtype} {tuple(qb.shape)} and w_kc {tuple(kc.shape)} strides {tuple(kc.stride())} are not "
             f"{heads} heads' BF16 shards in stock's layout")
    _require(not torch.cuda.is_current_stream_capturing(),
             "prepare inside a CUDA-graph capture: the gather is a collective")
    full_q = torch.empty((HEADS * nope, qb.shape[1]), dtype=qb.dtype, device=qb.device)
    full_k = torch.empty((HEADS, WIDTH, nope), dtype=kc.dtype, device=kc.device)
    for full, shard in ((full_q, qb), (full_k, kc.transpose(1, 2))):
        torch.distributed.all_gather_into_tensor(full, shard, group=group.device_group)
    _WEIGHTS[id(node)] = (full_q, full_k.transpose(1, 2))
    _say("replicas", f"rank {row_owned._STATE['rank']} of {world}: q_b_proj and w_kc of all {HEADS} heads replicated, "
                     f"{(full_q.numel() + full_k.numel()) * 2 >> 20} MiB a layer")
def _q_b_proj_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
    
    if self is not _CALL["node"]:
        return _STOCK_QB(self, q_lora)
    _require("made" not in _CALL, "a second q_b_proj_forward in one layer's call")
    n = q_lora.shape[0]
    rows = row_owned._cut(n)
    if rows is None:
        _CALL["made"] = None
        _say(f"rows {n}", f"{n} rows, outside A1's cut: stock's queries and A1's push")
        return _STOCK_QB(self, q_lora)
    qb, kc = _CALL["weights"]
    _require(q_lora.dim() == 2 and q_lora.dtype == qb.dtype and q_lora.shape[1] == qb.shape[1],
             f"q_lora {q_lora.dtype} {tuple(q_lora.shape)} for a q_b_proj of {qb.shape[1]} inputs")
    q = F.linear(q_lora[rows], qb).view(rows.stop - rows.start, HEADS, -1)
    _CALL["made"] = (n, torch.bmm(q.transpose(0, 1), kc).transpose(0, 1))
    return q_lora.new_empty((0, self.num_local_heads, self.qk_head_dim))


def _quantize(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim):
    made = _CALL.get("made")
    if made is None:
        return _SET[0](q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
    n, query = made
    _require("query" not in _CALL and q_nope.shape[0] == 0 and q_rope.shape[0] == 0,
             f"a quantize of {q_nope.shape[0]} query rows where this route handed stock's chain none")
    q, k, k_rope = _SET[0](query, query[..., :0], k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
    _require(q.dtype == FP8 and q.is_contiguous() and q.shape == query.shape,
             f"the quantize returned a {q.dtype} {tuple(q.shape)} query for {tuple(query.shape)}")
    _CALL["query"] = q
    return q.view(n, q_nope.shape[1], q.shape[2]), k, k_rope


def _attention(*args, **kwargs):
    
    query = _CALL.pop("query", None)
    _CALL["attended"] = True
    if query is None:
        return _SET[1](*args, **kwargs)
    n = _CALL["made"][0]
    why = ("positional", "arguments passed by position") if args else row_owned._outside(kwargs)
    _require(why is None, f"this rank's rows' queries were made for A1's call of {n} rows, but the attention call is "
                          f"outside A1's domain: {why[1] if why else ''}")
    q = kwargs["query"]
    _require(q.data_ptr() == query.data_ptr() and q.shape[0] == n,
             "the attention call's query is not the one this route's quantize returned")
    state = row_owned._buffers(q.shape[2], q.device)
    held = state["qv"], state["native"]
    state["qv"], state["native"] = query, _QueryHere(held[1])
    try:
        out = _SET[1](**kwargs)
    finally:
        state["qv"], state["native"] = held
    _CALL["served"] = True
    _say("ours", f"queries made on their rows' owners (first: {n} rows, {query.shape[0]} a rank of "
                 f"{row_owned._STATE['world']}, {HEADS} heads)")
    return out


@contextlib.contextmanager
def routed(node, forward_batch):
    mode = forward_batch.forward_mode
    if not mode.is_target_verify():
        name = getattr(mode, "name", str(mode))
        _say(f"mode {name}", f"A1's queries serve {name} calls")
        yield
        return
    _require(id(node) in _WEIGHTS, "a verify call of a module this route never prepared")
    weights = _WEIGHTS[id(node)]
    if weights is None:
        yield
        return
    _require(flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is row_owned._attention,
             "A1 is not set (the components' order)")
    _require(not _SET and DeepseekV2AttentionMLA.q_b_proj_forward is _STOCK_QB,
             "a sparse layer's call inside another's, or another route holds q_b_proj_forward")
    _SET[:] = [dsa_backend.mla_quantize_for_fp8_no_rope, row_owned._attention]
    _CALL.update(node=node, weights=weights)
    DeepseekV2AttentionMLA.q_b_proj_forward = _q_b_proj_forward
    dsa_backend.mla_quantize_for_fp8_no_rope = _quantize
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _attention
    try:
        yield
        _require("made" in _CALL or "attended" not in _CALL,
                 "an attention call without this route's q_b_proj_forward: stock's query path is not the one replaced")
        _require(not _CALL.get("made") or _CALL.get("served"),
                 "this rank's rows' queries were made; no attention call took them")
    finally:
        DeepseekV2AttentionMLA.q_b_proj_forward = _STOCK_QB
        dsa_backend.mla_quantize_for_fp8_no_rope, flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _SET
        _SET.clear()
        _CALL.clear()
