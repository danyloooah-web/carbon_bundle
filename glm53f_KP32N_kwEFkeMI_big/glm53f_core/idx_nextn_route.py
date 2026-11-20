from __future__ import annotations
import contextlib
import logging
import deep_gemm
import torch
from sglang.kernels.ops.attention.dsa.paged_mqa_logits import deepgemm_paged_mqa_logits_native
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
NODE = "dsa"
NEXT_N = 6
HEADS, DIM = 32, 128
BLOCK = 64
MIN_ROWS = 240
EXCHANGE = "all_reduce"
EXACT_SLOTS = 1 << 24
STAGE_BYTES = 8 << 20
_STOCK = IndexerKPool._get_topk_paged
_SAID: set = set()
_INDEXERS: dict = {}
_HELD: set = set()
_FORWARD: dict = {}
_SPLIT: dict = {}
_STAGES: dict = {}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf index logits next_n: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf index logits next_n: %s", detail)
def _attention_tp() -> dict:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return {"group": None, "plan": None}
    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import build_kpool_topk_v2_plan
    group = get_attn_tp_group()
    _say("group", f"the row split's group: attention TP of {group.world_size} ranks, this rank {group.rank_in_group}")
    return {"group": group, "plan": build_kpool_topk_v2_plan}
def prepare(node) -> None:
    
    if id(node) in _INDEXERS:
        return
    indexer = getattr(node, "indexer", None)
    _require(isinstance(indexer, IndexerKPool), f"the claimed attention module's indexer is {type(indexer).__name__}")
    _require(callable(getattr(deep_gemm, "fp8_paged_mqa_logits", None))
             and callable(getattr(deep_gemm, "get_paged_mqa_logits_metadata", None)),
             "deep_gemm lacks fp8_paged_mqa_logits or get_paged_mqa_logits_metadata")
    _require(indexer.n_heads == HEADS and indexer.head_dim == DIM, f"indexer heads {indexer.n_heads} x {indexer.head_dim}")
    _require(EXCHANGE in ("all_reduce", "all_gather"), f"EXCHANGE is {EXCHANGE!r}")
    _INDEXERS[id(node)] = indexer
    if "group" not in _SPLIT:
        _SPLIT.update(_attention_tp())
def _outside(indexer, q_fp8, weights, n_real: int) -> tuple[str, str] | None:
    if indexer.paged_mqa_logits_backend.is_aiter():
        return "aiter", "the AITER paged-MQA backend serves it"
    if indexer._should_use_tilelang_paged_mqa_logits(q_fp8):
        return "tilelang", "the TileLang paged-MQA kernel serves it"
    if q_fp8.dtype != torch.float8_e4m3fn or q_fp8.dim() != 3 or tuple(q_fp8.shape[1:]) != (HEADS, DIM):
        return "query", f"the query is {q_fp8.dtype} {tuple(q_fp8.shape)}, not FP8 [rows, {HEADS}, {DIM}]"
    if weights.dim() != 3 or tuple(weights.shape[1:]) != (HEADS, 1):
        return "weights", f"the weights are {tuple(weights.shape)}"
    if n_real < NEXT_N or n_real % NEXT_N:
        return "rows", f"{n_real} real rows are not {NEXT_N} a request"
    return None

def _rows(n: int, world: int, rank: int) -> slice | None:
    
    if world < 2 or n < MIN_ROWS or n % (NEXT_N * world) or n % 4:
        return None
    share = n // world
    return slice(rank * share, (rank + 1) * share)

def _split(indexer, pool, n: int) -> slice | None:
    
    group = _SPLIT["group"]
    if group is None or group.world_size < 2:
        return None
    rows = _rows(n, group.world_size, group.rank_in_group)
    if rows is None:
        why = ("rows", f"{n} rows are under {MIN_ROWS} or not {group.world_size} shares of whole requests")
    elif EXCHANGE == "all_reduce" and pool.size + pool.page_size > EXACT_SLOTS:
        why = ("slots", f"the pool's {pool.size + pool.page_size} slots are not all below float32's 2^24")
    elif EXCHANGE == "all_reduce" and n * indexer.topk_output_width * 4 > STAGE_BYTES:
        why = ("stage", f"a {n}-row float32 stage is over the push all-reduce's {STAGE_BYTES} bytes")
    else:
        _say("split", f"rows split over {group.world_size} ranks, the picks exchanged by {EXCHANGE} (first: {n} rows, "
                      f"rows {rows.start}-{rows.stop} here)")
        return rows
    _say("whole " + why[0], f"every rank scores every row of a call: {why[1]}")
    return None


def _schedule(forward_batch, layer_id: int, lens_2d: torch.Tensor, sm_count: int, plan_lens=None):
    capturing = lens_2d.is_cuda and torch.cuda.is_current_stream_capturing()
    held = _FORWARD
    if (held and held["batch"] is forward_batch and layer_id > held["layer"] and held["capturing"] == capturing
            and held["sms"] == sm_count and held["lens"].data_ptr() == lens_2d.data_ptr()
            and held["lens"].shape == lens_2d.shape and (held["plan"] is None) == (plan_lens is None)):
        held["layer"] = layer_id
        return held["schedule"], held["plan"]
    plan = None if plan_lens is None else _SPLIT["plan"](plan_lens)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lens_2d.clamp(min=1), BLOCK, sm_count)
    _FORWARD.update(batch=forward_batch, layer=layer_id, capturing=capturing, sms=sm_count, lens=lens_2d,
                    schedule=schedule, plan=plan)
    return schedule, plan


def _stage(n: int, mine: torch.Tensor) -> torch.Tensor:
    key = (n, mine.shape[1], mine.device)
    stage = _STAGES.get(key)
    if stage is None:
        capturing = mine.is_cuda and torch.cuda.is_current_stream_capturing()
        _require(not capturing, f"a {n}-row stage would be made under CUDA-graph capture: no eager call of it ran")
        stage = _STAGES[key] = torch.zeros((n, mine.shape[1]), dtype=torch.float32, device=mine.device)
    return stage


def _exchange(mine: torch.Tensor, rows: slice, n: int, out_rows: int) -> torch.Tensor:
    
    group = _SPLIT["group"]
    out = mine.new_empty((out_rows, mine.shape[1]))
    if out_rows > n:
        out[n:].fill_(-1)
    if EXCHANGE == "all_gather":
        group.all_gather_into_tensor(out[:n], mine)
        return out
    stage = _stage(n, mine)
    stage[rows].copy_(mine)
    summed = group.all_reduce(stage)
    _require(summed.data_ptr() != stage.data_ptr(),
             "the all-reduce summed the stage in place, so its other ranks' rows are no longer zero")
    out[:n].copy_(summed)
    return out
def _split_call(indexer, forward_batch, layer_id, rows: slice, q_real, w_real, kv_cache_fp8, seqlens_32,
                pool_seqlens, pool_context_lens, pool_block_tables, topk_v2_plan, metadata, out_rows: int):
    
    share = rows.stop - rows.start
    lens_2d = pool_context_lens.reshape(-1)[rows].view(share // NEXT_N, NEXT_N)
    schedule, plan = _schedule(forward_batch, layer_id, lens_2d, indexer.sm_count,
                               None if topk_v2_plan is None else pool_seqlens[rows])
    logits = deepgemm_paged_mqa_logits_native(
        deep_gemm.fp8_paged_mqa_logits, q_real[rows], kv_cache_fp8, w_real[rows], lens_2d, pool_block_tables[rows],
        schedule, pool_block_tables.shape[1] * BLOCK, q_offset=share, B=share // NEXT_N, next_n=NEXT_N)
    page_table_1, topk_offsets, _ = indexer._kpool_fused_topk_mapping(metadata)
    mine = indexer._topk_from_kpool_logits(
        logits=logits, pool_lens=pool_seqlens[rows], topk_v2_plan=plan, seq_lens=seqlens_32[rows],
        page_table=None if page_table_1 is None else page_table_1[rows],
        topk_offsets=None if topk_offsets is None else topk_offsets[rows])
    return _exchange(mine.contiguous(), rows, seqlens_32.shape[0], out_rows)


def served(indexer, forward_batch, layer_id, q_fp8, weights, metadata, kind: str):
    pool = dsa_indexer_kpool.get_token_to_kv_pool()
    _require(pool.index_page_size == BLOCK, f"the index cache's page is {pool.index_page_size}")
    seqlens_32 = metadata.get_seqlens_expanded()
    num_q_padded, n_real = q_fp8.shape[0], seqlens_32.shape[0]
    why = _outside(indexer, q_fp8, weights, n_real)
    if why is not None:
        _say(why[0], f"stock's call serves a {kind} step: {why[1]}")
        return _STOCK(indexer, forward_batch, layer_id, q_fp8, weights, metadata)
    block_tables = metadata.get_page_table_64()
    kv_cache_fp8 = indexer._get_index_k_read_buffer(pool, layer_id)
    _require(kv_cache_fp8.dim() == 2, f"the index cache is {tuple(kv_cache_fp8.shape)}")
    kv_cache_fp8 = kv_cache_fp8.view(kv_cache_fp8.shape[0], BLOCK, 1, DIM + 4)
    q_real, w_real = q_fp8[:n_real], weights[:n_real].squeeze(2)
    pool_seqlens, pool_context_lens, pool_block_tables, _, topk_v2_plan = indexer._get_kpool_decode_metadata(
        metadata=metadata, block_tables=block_tables, seqlens_32=seqlens_32, blocksize=BLOCK,
        build_schedule_metadata=False)
    requests = n_real // NEXT_N
    _require(pool_block_tables.shape[0] >= n_real and pool_context_lens.numel() >= n_real,
             f"pooled tables {tuple(pool_block_tables.shape)} / lengths {tuple(pool_context_lens.shape)} for {n_real} rows")
    rows = _split(indexer, pool, n_real)
    if rows is not None:
        return _split_call(indexer, forward_batch, layer_id, rows, q_real, w_real, kv_cache_fp8, seqlens_32,
                           pool_seqlens, pool_context_lens, pool_block_tables, topk_v2_plan, metadata, num_q_padded)
    lens_2d = pool_context_lens.reshape(-1)[:n_real].view(requests, NEXT_N)
    schedule, _ = _schedule(forward_batch, layer_id, lens_2d, indexer.sm_count)
    logits = deepgemm_paged_mqa_logits_native(
        deep_gemm.fp8_paged_mqa_logits, q_real, kv_cache_fp8, w_real, lens_2d, pool_block_tables[:n_real], schedule,
        pool_block_tables.shape[1] * BLOCK, q_offset=n_real, B=requests, next_n=NEXT_N)
    page_table_1, topk_offsets, _ = indexer._kpool_fused_topk_mapping(metadata)
    _say("ours " + kind, f"one next_n = {NEXT_N} call serves {kind} steps (first: {requests} requests)")
    return indexer._topk_from_kpool_logits(
        logits=logits, pool_lens=pool_seqlens, topk_v2_plan=topk_v2_plan, seq_lens=seqlens_32,
        page_table=page_table_1, topk_offsets=topk_offsets, out_rows=num_q_padded if num_q_padded != n_real else None)


def _get_topk_paged(indexer, forward_batch, layer_id, q_fp8, weights, metadata):
    
    if not forward_batch.forward_mode.is_target_verify():
        return _STOCK(indexer, forward_batch, layer_id, q_fp8, weights, metadata)
    return served(indexer, forward_batch, layer_id, q_fp8, weights, metadata, "verify")
@contextlib.contextmanager
def installed(indexer, entry):
    
    _require(id(indexer) not in _HELD, "the indexer's _get_topk_paged is already a route's")
    _HELD.add(id(indexer))
    indexer._get_topk_paged = lambda *a, **k: entry(indexer, *a, **k)
    try:
        yield
    finally:
        del indexer._get_topk_paged
        _HELD.discard(id(indexer))

@contextlib.contextmanager
def routed(node, forward_batch):
    
    if not forward_batch.forward_mode.is_target_verify():
        yield
        return
    with installed(_INDEXERS[id(node)], _get_topk_paged):
        yield
