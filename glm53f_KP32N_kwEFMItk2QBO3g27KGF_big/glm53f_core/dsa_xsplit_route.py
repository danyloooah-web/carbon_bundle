from __future__ import annotations
import contextlib
import logging
import sys
import time
import torch
from sglang.kernels.ops.gemm import fp8_kernel
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.graph_utils import padded_extend_real_tokens
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_disagg
from glm53f_core import dsa_split_route as token_split
from glm53f_core import e31_fused_route as fused
from glm53f_core import e31_stage_route as stage
from glm53f_core import fp8_linear_route
NODE = "layer"
MIN_ROWS, MAX_ROWS = 8193, 16384
BLOCK = [128, 128]
WARM_ROWS = (4097, 5000, 6143, 8192)
NEG_ZERO = -0.0
BF16 = torch.bfloat16
_STOCK_QB = DeepseekV2AttentionMLA.q_b_proj_forward
_MIXIN = forward_mla.DeepseekMLAForwardMixin
_STOCK_PV = _MIXIN.project_mla_values
_STOCK_FINISH = forward_mla._finish_absorb_output
_WEIGHTS: dict = {}
_WARM: set = set()
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA exchange-free split: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA exchange-free split: %s", detail)


def plan(rows: int, rank: int) -> tuple[int, int, int]:
    
    first = rows // 2
    return (0, first, rows - first) if rank == 0 else (first, rows - first, rows - first)


def as_stock(buffer: torch.Tensor, rows: int, heads: int, width: int) -> torch.Tensor:
    
    return buffer.view(-1)[:rows * heads * width].view(rows, heads, width)


def halves(own, peer, rank: int) -> tuple:
    
    return (own, peer) if rank == 0 else (peer, own)


def _gemm(x: torch.Tensor, weight: tuple, out: torch.Tensor) -> torch.Tensor:
    
    q, scale = fp8_utils.sglang_per_token_group_quant_fp8(
        x, BLOCK[1], column_major_scales=True, scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
    fp8_kernel.deep_gemm_fp8_fp8_bf16_nt(q, scale, weight[0], weight[1], out, BLOCK[0], BLOCK[1])
    return out

def _fp8_copy(linear, what: str) -> tuple:
    
    entry = fp8_linear_route._FP8.get(id(linear))
    _require(entry is not None, f"{what} has no FP8 copy (fp8_mla_route / fp8_wide_route held?): version 179's "
                                f"{what} is not the FP8 linear this route mirrors")
    weight, scale, n, prefill, _, window, exact = entry
    _require(prefill and window[0] <= MIN_ROWS and MAX_ROWS <= window[1] and exact is None and n == weight.shape[0]
             and fp8_utils._deepgemm_block_fp8_supported(tuple(weight.shape), BLOCK, BF16),
             f"{what}'s FP8 copy serves rows {window}, exact columns {exact}, {n} of {tuple(weight.shape)} rows: not "
             f"DeepGEMM at every row of {MIN_ROWS}-{MAX_ROWS}")
    return weight, scale


def _quantized(bf16: torch.Tensor, n: int) -> tuple:
    
    weight, scale = fp8_utils.quant_weight_ue8m0(weight_dequant=bf16, weight_block_size=BLOCK)
    return weight, fp8_utils.transform_scale_ue8m0(scale, mn=n)

def _gathered(shard: torch.Tensor, group, rank: int) -> torch.Tensor:
    
    shard = shard.contiguous()
    both = torch.empty((2, *shard.shape), dtype=shard.dtype, device=shard.device)
    torch.distributed.all_gather_into_tensor(both, shard, group=group.device_group)
    return both[1 - rank].clone()
def _inactive(layer) -> str | None:
    attention = getattr(layer, "self_attn", None)
    group = token_split._PAIR.get("group")
    if not isinstance(attention, DeepseekV2AttentionMLA):
        return "a linear-attention layer"
    if getattr(layer, "is_nextn", False) or getattr(attention, "is_nextn", False):
        return "the MTP draft layer"
    if group is None:
        return "the token split is inactive (attention TP is not two ranks)"
    if not fused._state["groups"]:
        return f"E31 is inactive ({fused._state['why']})"
    if get_disagg().disaggregation_mode == "decode":
        return "a PD decode server (no eager prefill forward)"
    if id(layer) not in stage._state["projections"]:
        return "o_proj's sum does not reach E31's slot form here (e31_stage_route left it stock's)"
    return None

def prepare(layer) -> None:
    if id(layer) in _WEIGHTS:
        return
    why = _inactive(layer)
    if why is not None:
        _WEIGHTS[id(layer)] = None
        _say("inactive " + why, f"{why}: the layer keeps version 179's path")
        return
    node, group = layer.self_attn, token_split._PAIR["group"]
    rank, heads, rank_kv = group.rank_in_group, node.num_local_heads, node.kv_lora_rank
    for ok, clause in (
            (type(node).q_b_proj_forward is _STOCK_QB and node._q_b_proj_verified_shape is False,
             "q_b_proj_forward is overridden or takes the min-latency fused-A GEMM"),
            (not node.use_deep_gemm_bmm and node.dcp_size == 1, "DeepGEMM bmms or decode context parallelism"),
            (node.qk_rope_head_dim == 0 and node.q_lora_rank is not None, "a RoPE part or no q_lora"),
            (forward_mla.kv_b_lora_correction(node) is None and not forward_mla._SGLANG_EXPERIMENTAL_LORA_OPTI
             and not forward_mla._ENABLE_DSA_Q8KV8_BORN_FP8_Q and not node.needs_unfused_mla_qkv,
             "a LoRA correction, the born-FP8 query or an unfused-query override"),
            (getattr(node.o_proj, "bias", None) is None and node.o_proj.reduce_results is False,
             "o_proj has a bias or reduces its own result")):
        _require(ok, f"stock's absorbed chain is not the one this route mirrors: {clause}")
    qb, kc, vc, o = node.q_b_proj.weight, node.w_kc, node.w_vc, node.o_proj.weight
    nope, vdim = node.qk_nope_head_dim, node.v_head_dim
    _require(qb.dtype == kc.dtype == vc.dtype == o.dtype == BF16 and tuple(qb.shape) == (heads * nope, node.q_lora_rank)
             and tuple(kc.shape) == (heads, nope, rank_kv) and kc.transpose(1, 2).is_contiguous()
             and tuple(vc.shape) == (heads, rank_kv, vdim) and vc.transpose(1, 2).is_contiguous()
             and o.dim() == 2 and o.shape == (fused._HIDDEN, heads * vdim) and o.is_contiguous() and qb.is_contiguous(),
             f"q_b_proj {tuple(qb.shape)}, w_kc {tuple(kc.shape)} {tuple(kc.stride())}, w_vc {tuple(vc.shape)} "
             f"{tuple(vc.stride())}, o_proj {tuple(o.shape)} are not {heads} heads' BF16 shards in stock's layout")
    own_q, own_o = _fp8_copy(node.q_b_proj, "q_b_proj"), _fp8_copy(node.o_proj, "o_proj")
    _require(not torch.cuda.is_current_stream_capturing(), "prepare under graph capture: the gather is a collective")
    began = time.perf_counter()
    weights = {"q": _quantized(_gathered(qb, group, rank), qb.shape[0]),

               "kc": _gathered(kc.transpose(1, 2), group, rank).transpose(1, 2),
               "vc": _gathered(vc.transpose(1, 2), group, rank).transpose(1, 2),
               "o": _quantized(_gathered(o, group, rank), o.shape[0]), "own_q": own_q, "own_o": own_o}
    for linear, (w, s) in (("q_b_proj", weights["q"]), ("o_proj", weights["o"])):
        _require(fp8_utils._deepgemm_block_fp8_supported(tuple(w.shape), BLOCK, BF16) and w.dtype == own_q[0].dtype,
                 f"the peer's {linear} half {w.dtype} {tuple(w.shape)} is not a DeepGEMM FP8 weight")
    for w in (own_q, own_o):
        if tuple(w[0].shape) in _WARM:
            continue
        x = torch.zeros((max(WARM_ROWS), w[0].shape[1]), dtype=BF16, device=qb.device)
        out = torch.empty((max(WARM_ROWS), w[0].shape[0]), dtype=BF16, device=qb.device)
        for m in WARM_ROWS:
            _gemm(x[:m], w, out[:m])
        _WARM.add(tuple(w[0].shape))
    torch.cuda.synchronize()
    _WEIGHTS[id(layer)] = weights
    mib = sum(t.numel() * t.element_size() for k in ("q", "kc", "vc", "o") for t in (
        weights[k] if isinstance(weights[k], tuple) else (weights[k],))) >> 20
    _say("prepared", f"rank {rank} of the pair: the peer's q_b_proj / w_kc / w_vc / o_proj halves held, {mib} MiB a "
                     f"layer (first in {time.perf_counter() - began:.2f} s); rows {MIN_ROWS}-{MAX_ROWS} served")


def _outside(self, q_lora: torch.Tensor) -> str | None:
    
    rows = q_lora.shape[0] if q_lora.dim() == 2 else -1
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return f"{rows} rows, outside {MIN_ROWS} to {MAX_ROWS}"
    weight = _CALL["weights"]["own_q"][0]
    if not (q_lora.dtype == BF16 and q_lora.is_contiguous() and q_lora.shape[1] == weight.shape[1]):
        return f"q_lora {q_lora.dtype} {tuple(q_lora.shape)}, not the contiguous BF16 rows q_b_proj's FP8 linear takes"
    if padded_extend_real_tokens(rows, _CALL["batch"]) is not None:
        return "MLP-sync padded rows"
    armed = stage._armed(rows)
    if armed is not None:
        return f"E31's slot is not armed: {armed}"
    return None


def _q_b_proj_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
    
    if self is not _CALL.get("node") or not _CALL.get("entered"):
        return _STOCK_QB(self, q_lora)
    _require("made" not in _CALL, "a second q_b_proj_forward in one layer's call")
    why = _outside(self, q_lora)
    if why is not None:
        _CALL["made"] = None
        _say("front " + why.split(",")[0], f"version 179's path serves a call: {why}")
        return _STOCK_QB(self, q_lora)
    rows, rank, weights = q_lora.shape[0], _CALL["rank"], _CALL["weights"]
    start, count, held = plan(rows, rank)
    heads, nope = self.num_local_heads, self.qk_nope_head_dim
    x = q_lora[start:start + held]
    q, scale = fp8_utils.sglang_per_token_group_quant_fp8(
        x, BLOCK[1], column_major_scales=True, scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
    absorbed = torch.empty((2 * heads, held, self.kv_lora_rank), dtype=BF16, device=x.device)
    for h, (w, kc) in enumerate(zip(halves(weights["own_q"], weights["q"], rank), halves(self.w_kc, weights["kc"], rank))):
        half = torch.empty((held, w[0].shape[0]), dtype=BF16, device=x.device)
        fp8_kernel.deep_gemm_fp8_fp8_bf16_nt(q, scale, w[0], w[1], half, BLOCK[0], BLOCK[1])
        torch.bmm(half.view(held, heads, nope).transpose(0, 1), kc, out=absorbed[h * heads:(h + 1) * heads])
    _CALL["made"] = (rows, start, count, held, absorbed.transpose(0, 1))
    return q_lora.new_empty((0, heads, self.qk_head_dim))


def _quantizing(found):
    

    def quantize(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim):
        made = _CALL.get("made")
        if not made or "query" in _CALL:
            return found(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
        rows, _, _, held, query = made
        _require(q_nope.shape[0] == 0 and q_rope.shape[0] == 0 and q_nope.shape[1] * 2 == query.shape[1],
                 f"a quantize of {q_nope.shape[0]} query rows where this route handed stock's chain none")
        q, k, rope = found(query, query[..., :0], k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
        _require(q.dtype == torch.float8_e4m3fn and q.is_contiguous() and q.shape == query.shape,
                 f"the quantize returned a {q.dtype} {tuple(q.shape)} query for {tuple(query.shape)}")
        _CALL["query"], _CALL["made"] = q, made[:4] + (None,)
        return as_stock(q, rows, q_nope.shape[1], q.shape[2]), k, rope

    return quantize

def enter(node):
    
    if _CALL.get("node") is not node or _CALL.get("entered"):
        return None
    found = dsa_backend.mla_quantize_for_fp8_no_rope
    dsa_backend.mla_quantize_for_fp8_no_rope = _quantizing(found)
    _CALL["entered"] = True
    return found


def leave(found) -> None:
    
    dsa_backend.mla_quantize_for_fp8_no_rope = found
    _CALL["entered"] = False

_COUNTERS: dict = {}


def _counter(rows: int, heads: int, device) -> torch.Tensor:
    
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    need = get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, heads, sms)
    buf = _COUNTERS.get(device)
    if buf is None:
        most = get_trtllm_gen_multi_ctas_kv_counter_bytes(MAX_ROWS - MAX_ROWS // 2, heads, sms)
        buf = _COUNTERS[device] = torch.zeros(max(need, most), dtype=torch.uint8, device=device)
    _require(buf.numel() >= need, f"the counter buffer holds {buf.numel()} bytes, a call of {rows} rows x {heads} heads "
                                  f"needs {need}")
    return buf


def owns(kwargs: dict) -> bool:
    
    return "query" in _CALL and "attended" not in _CALL


def serve(args: tuple, kwargs: dict, stock) -> torch.Tensor:
    
    rows, start, count, held, _ = _CALL["made"]
    query, q = _CALL["query"], kwargs.get("query")
    why = ("positional", "arguments passed by position") if args else token_split._outside(kwargs)
    _require(why is None, f"this rank's rows' queries were made, but the attention call is outside the token split's "
                          f"domain: {why[1] if why else ''}")
    heads = q.shape[-2]
    _require(q.data_ptr() == query.data_ptr() and q.shape[0] == rows and 2 * heads == query.shape[1],
             "the attention call's query is not the one this route's quantize returned")
    out = torch.empty((held, 1, query.shape[1], kwargs["kv_lora_rank"]), dtype=BF16, device=query.device)
    call = dict(kwargs)
    call.update(query=query[:count].unsqueeze(1), block_tables=kwargs["block_tables"][start:start + count],
                seq_lens=kwargs["seq_lens"][start:start + count], out=out[:count])
    if kwargs.get("sparse_mla_top_k_lens") is not None:
        call["sparse_mla_top_k_lens"] = kwargs["sparse_mla_top_k_lens"][start:start + count]
    if kwargs.get("multi_ctas_kv_counter_buffer") is not None:
        call["multi_ctas_kv_counter_buffer"] = _counter(count, query.shape[1], query.device)
    got = stock(**call)
    got = got[0] if isinstance(got, tuple) else got
    _require(got.dtype == BF16 and got.data_ptr() == out.data_ptr() and got.numel() == count * out[0].numel(),
             f"stock's kernel returned {got.dtype} {tuple(got.shape)} rows of its own, not the output it was handed")
    _CALL["attended"] = out
    _say("served", f"rows {start}-{start + count} of {rows} here, one kernel call of {query.shape[1]} heads, no exchange")
    return as_stock(out, rows, heads, out.shape[-1]).unsqueeze(1)

def _project_mla_values(self, attn_output, zero_allocator):
    
    attended = _CALL.get("attended")
    if self is not _CALL.get("node") or attended is None:
        return _STOCK_PV(self, attn_output, zero_allocator)
    _require("values" not in _CALL and attn_output.data_ptr() == attended.data_ptr(),
             "the V up-projection is handed another tensor than this route's attention output")
    _, _, count, _, _ = _CALL["made"]
    heads, vdim = self.num_local_heads, self.v_head_dim
    values = []
    for h, vc in enumerate(halves(self.w_vc, _CALL["weights"]["vc"], _CALL["rank"])):
        rows_h = attended[:count, 0, h * heads:(h + 1) * heads]
        v = torch.empty((count, heads * vdim), dtype=BF16, device=attended.device)
        torch.bmm(rows_h.transpose(0, 1), vc, out=v.view(-1, heads, vdim).transpose(0, 1))
        values.append(v)
    _CALL["values"] = values
    return values[0]
def _finish_absorb_output(module, attn_bmm_output, attention_output_gate, topk_indices):
    values = _CALL.get("values")
    if module is not _CALL.get("node") or values is None:
        return _STOCK_FINISH(module, attn_bmm_output, attention_output_gate, topk_indices)
    _require("finished" not in _CALL and attn_bmm_output is values[0] and attention_output_gate is None,
             "o_proj is handed another tensor than this route's V up-projection, or an output gate")
    rows, start, count, _, _ = _CALL["made"]
    armed = stage._armed(rows)
    _require(armed is None, f"E31's slot is not armed at the call's o_proj: {armed}")
    slot = stage._next_slot(rows, values[0].device)
    rank, weights = _CALL["rank"], _CALL["weights"]
    mine = slot[start:start + count]
    scratch = torch.empty_like(mine)
    for h, w in enumerate(halves(weights["own_o"], weights["o"], rank)):
        _gemm(values[h], w, mine if h == rank else scratch)
    mine.add_(scratch)
    slot[:start].fill_(NEG_ZERO)
    slot[start + count:].fill_(NEG_ZERO)
    _CALL["finished"] = True
    if module.next_skip_topk is None:
        return slot
    return slot, topk_indices if module.next_skip_topk else None

@contextlib.contextmanager
def routed(layer, forward_batch):
    
    _require(id(layer) in _WEIGHTS, "a call of a layer this route never prepared")
    weights = _WEIGHTS[id(layer)]
    mode = forward_batch.forward_mode
    if weights is None or not fused._state["live"] or not mode.is_extend_or_draft_extend_or_mixed():
        yield
        return
    _require(not _CALL, "a layer's call inside another's")
    found = (DeepseekV2AttentionMLA.q_b_proj_forward, _MIXIN.project_mla_values, forward_mla._finish_absorb_output)
    _require(found == (_STOCK_QB, _STOCK_PV, _STOCK_FINISH),
             "another route holds q_b_proj_forward, project_mla_values or _finish_absorb_output")
    _CALL.update(node=layer.self_attn, layer=layer, weights=weights, batch=forward_batch,
                 rank=token_split._PAIR["group"].rank_in_group)
    DeepseekV2AttentionMLA.q_b_proj_forward = _q_b_proj_forward
    _MIXIN.project_mla_values = _project_mla_values
    forward_mla._finish_absorb_output = _finish_absorb_output
    try:
        yield
        reached = [step for step in ("query", "attended", "values", "finished") if step in _CALL]
        _require(not _CALL.get("made") or "finished" in _CALL,
                 f"this rank's rows' queries were made, the call's chain reached only {reached or ['the front']}")
    finally:
        DeepseekV2AttentionMLA.q_b_proj_forward = _STOCK_QB
        _MIXIN.project_mla_values = _STOCK_PV
        forward_mla._finish_absorb_output = _STOCK_FINISH
        _CALL.clear()

if not token_split._XSPLIT:
    token_split._XSPLIT.append(sys.modules[__name__])
