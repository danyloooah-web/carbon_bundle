from __future__ import annotations
import contextlib
import logging
import flashinfer.decode
import torch
from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import recommended_max_tokens
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from glm53f_core import idx_nextn_route
NODE = "dsa"
NEXT_N = idx_nextn_route.NEXT_N
HEADS, WIDTH = 64, 512
MAX_RANKS = 8
MIN_ROWS = 168
FLOOR_ROWS = 768
MAX_GRID = 128
FP8 = torch.float8_e4m3fn
_STOCK = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
_STOCK_EXCHANGE = idx_nextn_route._exchange
_UNSERVED = ("out", "lse", "cum_seq_lens_q", "max_q_len", "sinks", "return_lse", "enable_dcp")
_STATE: dict = {}
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA rows: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA rows: %s", detail)
def prepare(node) -> None:
    
    _require(callable(_STOCK), "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla missing")
    if "group" in _STATE:
        return
    group = get_attn_tp_group()
    world = group.world_size
    if not 2 <= world <= MAX_RANKS or HEADS % world:
        _STATE["group"] = None
        _say("inactive", f"attention TP is {world} rank(s) on this engine: stock's calls")
        return
    import rows_push_native

    _require(callable(getattr(rows_push_native, "push", None)) and callable(getattr(rows_push_native, "pad_words", None)),
             "rows_push_native lacks push or pad_words")
    rows = recommended_max_tokens(include_prefill=False, floor=FLOOR_ROWS)
    _STATE.update(group=group, rank=group.rank_in_group, world=world, native=rows_push_native,
                  capacity=-(-rows // (NEXT_N * world)) * NEXT_N * world)
def _cut(n: int) -> slice | None:
    
    world = _STATE["world"]
    if n < MIN_ROWS or n > _STATE["capacity"] or n % (NEXT_N * world):
        return None
    share, rank = n // world, _STATE["rank"]
    return slice(rank * share, (rank + 1) * share)

def _outside(kwargs: dict) -> tuple[str, str] | None:
    q, tables, seq_lens = kwargs.get("query"), kwargs.get("block_tables"), kwargs.get("seq_lens")
    if not (isinstance(q, torch.Tensor) and isinstance(tables, torch.Tensor) and isinstance(seq_lens, torch.Tensor)):
        return "arguments", "query, block tables or sequence lengths not passed as tensors by keyword"
    world = _STATE["world"]
    if (q.dtype != FP8 or q.dim() != 4 or q.shape[1] != 1 or q.shape[2] * world != HEADS or q.shape[3] != WIDTH
            or not q.is_contiguous() or q.data_ptr() % 16):
        return "query", f"the query is {q.dtype} {tuple(q.shape)}, not contiguous FP8 [rows, 1, {HEADS // world}, {WIDTH}]"
    n = q.shape[0]
    if _cut(n) is None:
        return "rows", (f"{n} rows: not {world} shares of whole requests from {MIN_ROWS} to the buffers' "
                        f"{_STATE['capacity']}")
    if (tables.dtype != torch.int32 or tables.dim() != 3 or tuple(tables.shape[:2]) != (n, 1)
            or tables.stride(2) != 1 or seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (n,)):
        return "tables", (f"block tables {tables.dtype} {tuple(tables.shape)} and lengths {seq_lens.dtype} "
                          f"{tuple(seq_lens.shape)} are not int32 [rows, 1, width] and [rows]")
    lens = kwargs.get("sparse_mla_top_k_lens")
    if lens is not None and (not isinstance(lens, torch.Tensor) or tuple(lens.shape) != (n,)):
        return "lengths", "top-k lengths that are not one a row"
    for name in _UNSERVED:
        if kwargs.get(name) is not None and kwargs.get(name) is not False:
            return name, f"{name} is set"
    if kwargs.get("kv_lora_rank") != WIDTH or kwargs.get("qk_rope_head_dim") != 0:
        return "rank", f"latent rank {kwargs.get('kv_lora_rank')} with a RoPE width of {kwargs.get('qk_rope_head_dim')}"
    return None


def _buffers(heads: int, device) -> dict:
    state = _STATE
    if "res" in state:
        _require(state["heads"] == heads, f"a call of {heads} heads a rank after buffers for {state['heads']}")
        return state
    _require(not torch.cuda.is_current_stream_capturing(),
             "the first served call is inside a CUDA-graph capture (no eager warm-up preceded it)")
    import torch.distributed._symmetric_memory as symmetric_memory

    group, native, world, rows = state["group"], state["native"], state["world"], state["capacity"]
    share = rows // world
    qbuf = symmetric_memory.empty(share * HEADS * WIDTH, dtype=torch.uint8, device=device)
    res = symmetric_memory.empty(rows * heads * WIDTH, dtype=torch.bfloat16, device=device)
    pads = symmetric_memory.empty(native.pad_words(), dtype=torch.int32, device=device)
    pads.zero_()
    handles = [symmetric_memory.rendezvous(t, group.device_group) for t in (qbuf, res, pads)]
    for handle in handles:
        _require(handle.rank == state["rank"] and handle.world_size == world,
                 f"a symmetric handle of rank {handle.rank} of {handle.world_size} on rank {state['rank']} of {world}")
    torch.cuda.synchronize(device)
    group.barrier()
    state.update(heads=heads, handles=handles, qbuf=qbuf, pads_buf=pads,
                 qv=qbuf.view(FP8).view(share, HEADS, WIDTH),
                 out=torch.empty((share, HEADS, WIDTH), dtype=torch.bfloat16, device=device),
                 res=res.view(rows, heads, WIDTH),
                 q_dsts=[int(p) for p in handles[0].buffer_ptrs], r_dsts=[int(p) for p in handles[1].buffer_ptrs],
                 pads=[int(p) for p in handles[2].buffer_ptrs])
    _say("buffers", f"rank {state['rank']} of {world}: buffers for {rows} rows ({share} owned) exchanged")
    return state


def _served(kwargs: dict, rows: slice) -> torch.Tensor:
    
    q = kwargs["query"]
    n, heads = q.shape[0], q.shape[2]
    state = _buffers(heads, q.device)
    world, rank, native, share = state["world"], state["rank"], state["native"], n // state["world"]
    item = heads * WIDTH
    native.push(q, state["q_dsts"], state["pads"], rank, 0, n, item, state["qv"].shape[0], MAX_GRID)
    out = state["out"][:share]
    call = dict(kwargs)
    call.update(query=state["qv"][:share].unsqueeze(1), block_tables=kwargs["block_tables"][rows],
                seq_lens=kwargs["seq_lens"][rows], out=out.unsqueeze(1))
    lens = kwargs.get("sparse_mla_top_k_lens")
    if lens is not None:
        call["sparse_mla_top_k_lens"] = lens[rows]
    o = _STOCK(**call)
    o = o[0] if isinstance(o, tuple) else o
    _require(o.dtype == torch.bfloat16 and o.data_ptr() == out.data_ptr() and o.numel() == out.numel(),
             f"stock's kernel returned {o.dtype} {tuple(o.shape)} rows of its own, not the output it was handed")
    native.push(out, state["r_dsts"], state["pads"], rank, 1, n, item * 2, state["res"].shape[0], MAX_GRID)
    _say("ours", f"rows served by their owners (first: {n} rows, {share} a rank of {world}, {world * heads} heads)")
    return state["res"][:n].unsqueeze(1)


def _attention(*args, **kwargs):
    
    skipped = _CALL.pop("skipped", None)
    why = ("positional", "arguments passed by position") if args else _outside(kwargs)
    if why is not None:
        _require(skipped is None, f"this layer's indexer kept only this rank's picks (rows {skipped}) but the "
                                  f"attention call is outside the route: {why[1]}")
        _say(why[0], f"stock's kernel serves a call: {why[1]}")
        return _STOCK(*args, **kwargs)
    n = kwargs["query"].shape[0]
    rows = _cut(n)
    _require(skipped is None or skipped == (n, rows.start, rows.stop),
             f"the indexer kept rows {skipped} of its call, the attention call owns rows {(n, rows.start, rows.stop)}")
    return _served(kwargs, rows)

def _own_picks(mine: torch.Tensor, rows: slice, n: int, out_rows: int) -> torch.Tensor:
    
    cut = _cut(n) if _CALL.get("skip") and out_rows == n else None
    if cut is None or (cut.start, cut.stop) != (rows.start, rows.stop):
        return _STOCK_EXCHANGE(mine, rows, n, out_rows)
    out = torch.full((out_rows, mine.shape[1]), -1, dtype=mine.dtype, device=mine.device)
    out[rows].copy_(mine)
    _CALL["skipped"] = (n, rows.start, rows.stop)
    _say("picks", f"the verify picks stay on their owners (first: rows {rows.start}-{rows.stop} of {n})")
    return out

def _picks_read_elsewhere(node, forward_batch) -> str | None:
    
    if getattr(node, "next_skip_topk", None):
        return "the next layer reuses this layer's top-k"
    if getattr(forward_batch, "reuse_dsa_topk_indices", False):
        return "the batch reuses top-k indices across layers"
    if getattr(getattr(forward_batch, "spec_info", None), "dsa_seed_topk_capture", None) is not None:
        return "the batch captures a draft seed from the picks"
    if get_global_indexer_capturer() is not None:
        return "the indexer top-k capturer is on"
    return None


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if _STATE.get("group") is None or not (mode.is_target_verify() or mode.is_draft_extend_v2()):
        yield
        return
    current = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
    _require(current is _STOCK, f"another route holds trtllm_batch_decode_with_kv_cache_mla ({current!r})")
    _require(idx_nextn_route._exchange is _STOCK_EXCHANGE, "idx_nextn_route's exchange is already replaced")
    skip = False
    if mode.is_target_verify():
        elsewhere = _picks_read_elsewhere(node, forward_batch)
        skip = elsewhere is None
        if not skip:
            _say("exchange " + elsewhere, f"the verify picks are exchanged: {elsewhere}")
    _CALL.clear()
    _CALL["skip"] = skip
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _attention
    if skip:
        idx_nextn_route._exchange = _own_picks
    try:
        yield
        _require("skipped" not in _CALL, f"this layer's indexer kept only this rank's picks (rows {_CALL.get('skipped')}) "
                                         "and no attention call followed")
    finally:
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _STOCK
        idx_nextn_route._exchange = _STOCK_EXCHANGE
        _CALL.clear()
