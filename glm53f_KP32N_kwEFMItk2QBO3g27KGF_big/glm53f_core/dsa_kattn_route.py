from __future__ import annotations
import contextlib
import logging
import sys
import flashinfer.decode
import torch
from sglang.srt.layers.attention.dsa import kpool_fp8_index
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import is_in_breakable_cuda_graph
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import is_in_tc_piecewise_cuda_graph
NODE = "dsa"
K_ATTN = 1024
K_DRAFT = K_ATTN
DECODE = False
SKIP = frozenset({19, 23, 31})
DRAFT = False
POOL, TOPK = 4, 2048
GROUPS = (128, 160, 192, 224, 256, 512)
_STOCK_SELECT = IndexerKPool._topk_from_kpool_logits
_NODES: dict = {}
_PEERS: dict = {}
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA K' attention: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA K' attention: %s", detail)
def _peer(name: str):
    return sys.modules.get(__name__.rsplit(".", 1)[0] + "." + name)
def prepare(node) -> None:
    if id(node) in _NODES:
        return
    indexer = getattr(node, "indexer", None)
    _require(isinstance(indexer, IndexerKPool), f"the claimed attention module's indexer is {type(indexer).__name__}")
    _require(indexer.index_kpool == POOL and indexer.index_topk == TOPK,
             f"the indexer pools {indexer.index_kpool} tokens and keeps {indexer.index_topk}, not {POOL} and {TOPK}")
    _require(IndexerKPool._topk_from_kpool_logits is _STOCK_SELECT,
             "another route holds IndexerKPool._topk_from_kpool_logits")
    draft = bool(getattr(node, "is_nextn", False))
    k = K_DRAFT if draft else K_ATTN
    _require(k % POOL == 0 and k // POOL in GROUPS and k < TOPK, f"K' {k} is not {POOL} x one of {GROUPS} under {TOPK}")
    if not _PEERS:
        import dsa_pagetable_native

        _require(callable(getattr(dsa_pagetable_native, "table", None)), "dsa_pagetable_native lacks table")
        _PEERS.update(native=dsa_pagetable_native, split=_peer("idx_split_attn_route"), node=_peer("idx_split_node"),
                      nextn=_peer("idx_nextn_route"), rows=_peer("dsa_rows_route"))
    layer = indexer.layer_id
    kept = (("draft", "the draft layer keeps its table in every call (DRAFT False)") if draft and not DRAFT
            else (f"skip {layer}", f"layer {layer} keeps its table in every call (SKIP)") if layer in SKIP else None)
    _NODES[id(node)] = (indexer, k, kept)
    _say(f"prepared {draft}", f"{'the draft layer' if draft else 'target layers'}: a row reads its best {k} picks")


def _outside(forward_batch) -> tuple[str, str] | None:
    
    if not DECODE:
        mode = forward_batch.forward_mode
        if not mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed():
            return "decode", "decode-side calls keep their table (verify, draft extend, decode, the draft decode steps)"
    if is_in_breakable_cuda_graph() or is_in_tc_piecewise_cuda_graph():
        return "graph", "prefill graph calls (their indexer runs in the graph's eager break) keep their picks' table"
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    if getattr(backend, "use_mha", False):
        return "mha", "dense MHA prefill calls have no sparse attention"
    if (getattr(backend, "use_fused_topk", None) is not True or getattr(backend, "attn_dcp_size", 1) != 1
            or getattr(backend, "hisparse_coordinator", None) is not None):
        return "unfused", "calls whose top-k is not fused into the page table (or with DCP / hisparse) keep their table"
    return None


def _kprime(indexer, k: int, logits, pool_lens, seq_lens=None, page_table=None, topk_offsets=None, row_starts=None,
            out_rows=None, page_table_row_index=None, topk_v2_plan=None) -> torch.Tensor:
    
    n = logits.shape[0]
    if page_table is not None and page_table_row_index is None and page_table.shape[0] != n:
        page_table = page_table[:n]
    if topk_offsets is not None and topk_offsets.shape[0] != n:
        topk_offsets = topk_offsets[:n]
    if page_table_row_index is not None and page_table_row_index.shape[0] != n:
        page_table_row_index = page_table_row_index[:n]
    return kpool_fp8_index.topk_from_pooled_history_logits(
        logits=logits, group_lengths=pool_lens, kpool=indexer.index_kpool, topk=k, page_table=page_table,
        topk_offsets=topk_offsets, seq_lens=seq_lens, row_starts=row_starts, out_rows=out_rows,
        page_table_row_index=page_table_row_index, topk_v2_plan=topk_v2_plan)


def _select(args: tuple, kw: dict) -> torch.Tensor:
    _require(not args, "IndexerKPool._topk_from_kpool_logits called with positional arguments")
    indexer = _CALL["indexer"]
    picks = _STOCK_SELECT(indexer, **kw)
    _CALL["parts"].append(_kprime(indexer, _CALL["k"], **kw))
    if not _CALL["armed"]:
        _require(flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is _CALL["entry"],
                 "the attention kernel's entry changed inside the layer's call before its picks were ranked")
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _attention
        _CALL["armed"] = True
    return picks


def _take() -> list:
    parts, _CALL["parts"] = _CALL["parts"], []
    return parts


def _stack(parts: list, rows: int | None) -> torch.Tensor:
    _require(bool(parts), "picks laid out with no K' selection made for them")
    picks = parts[0] if len(parts) == 1 else torch.cat(parts)
    if rows is not None and picks.shape[0] != rows:
        _require(picks.shape[0] < rows, f"{picks.shape[0]} K' rows for a table of {rows}")
        picks = torch.cat([picks, picks.new_full((rows - picks.shape[0], picks.shape[1]), -1)])
    return picks

def _width(picks: torch.Tensor) -> int:
    return (picks.shape[1] + 3) // 4 * 4

def _lay(picks: torch.Tensor, table: torch.Tensor, counts: torch.Tensor) -> None:
    
    _PEERS["native"].table(picks, table, counts)


def _padded(state, kw: dict) -> torch.Tensor:
    split = _PEERS["split"]
    _require(not _CALL["parts"] and _CALL["record"] is None, "a second selection layout in one layer's call")
    skipped = split._CALL["skipped"]
    table = _CALL["padded"](state, kw)
    if split._CALL["skipped"] == skipped:
        return table
    n = kw["row_chunks"][-1].stop
    h = n // 2
    part = _stack(_take(), h)
    _require(2 * h == n and table.shape[0] == n, f"PAD-ZC took a call of {n} rows with a table of {table.shape[0]}")
    out = part.new_empty((n, _width(part)))
    counts = part.new_empty((n,))
    for at in (state.rank * h, (1 - state.rank) * h):
        _lay(part, out[at:at + h], counts[at:at + h])
    _CALL["record"] = ("table", out, counts)
    _say("pad-zc", f"PAD-ZC prefill calls: this rank's K' half in both halves (first: {n} rows)")
    return table


def _gather(both: torch.Tensor, mine: torch.Tensor, group) -> None:
    
    _CALL["gather"](both, mine, group)
    _require(_CALL["record"] is None, "a second selection layout in one layer's call")
    h = mine.shape[0]
    part = _stack(_take(), None)
    _require(both.shape[0] == 2 * h and h - 1 <= part.shape[0] <= h,
             f"a gather of {h}-row halves after {part.shape[0]} K' rows")
    halves = part.new_empty((2 * h, part.shape[1]))
    _CALL["gather"](halves, _stack([part], h).contiguous(), group)
    _CALL["record"] = ("gathered", halves)
    _say("gathered", f"indexer-split calls with their all-gather: the K' halves gathered too (first: {2 * h} rows)")


def _exchange(mine: torch.Tensor, rows: slice, n: int, out_rows: int) -> torch.Tensor:
    
    out = _CALL["exchange"](mine, rows, n, out_rows)
    parts = _take()
    _require(len(parts) == 1 and parts[0].shape[0] == rows.stop - rows.start and _CALL["record"] is None,
             f"the row split's exchange of rows {rows.start}-{rows.stop} after {len(parts)} K' selection(s)")
    _CALL["record"] = ("own", parts[0], rows, n)
    return out


def _owned(kwargs: dict, cut: slice, n: int, rows: int) -> bool:
    
    a1 = _PEERS["rows"]
    if a1 is None or a1._STATE.get("group") is None or rows != n or a1._outside(kwargs) is not None:
        return False
    owned = a1._cut(n)
    return owned is not None and (owned.start, owned.stop) == (cut.start, cut.stop)


def _attention(*args, **kwargs):
    
    entry, record, parts = _CALL["entry"], _CALL["record"], _CALL["parts"]
    if record is None and not parts:
        _say("again", "a second kernel call in one layer's call keeps its table")
        return entry(*args, **kwargs)
    _CALL["record"], _CALL["parts"] = None, []
    _require(not args, "the attention kernel called with positional arguments")
    tables, lens = kwargs.get("block_tables"), kwargs.get("sparse_mla_top_k_lens")
    _require(isinstance(tables, torch.Tensor) and tables.dim() == 3 and tables.shape[1] == 1
             and isinstance(lens, torch.Tensor) and tuple(lens.shape) == tuple(tables.shape[:1]),
             "the kernel call's block tables and top-k lengths are not stock's [rows, 1, width] and [rows]")
    rows = tables.shape[0]
    if rows == 0:
        return entry(**kwargs)
    kind = "rows" if record is None else record[0]
    if kind == "table":
        table, counts = record[1], record[2]
        _require(table.shape[0] == rows, f"a K' table of {table.shape[0]} rows for a call of {rows}")
    elif kind == "own":
        picks, cut, n = record[1], record[2], record[3]
        if not _owned(kwargs, cut, n, rows):
            _say("own elsewhere", f"row-split calls A1 does not take on this rank's rows keep their table (first: "
                                  f"{rows} rows, K' rows {cut.start}-{cut.stop} of {n})")
            return entry(**kwargs)
        table, counts = picks.new_empty((rows, _width(picks))), picks.new_empty((rows,))
        _lay(picks, table[cut], counts[cut])
        _say("own", f"row-split calls: this rank's K' rows (first: rows {cut.start}-{cut.stop} of {n})")
    else:
        picks = _stack(parts, None) if record is None else record[1]
        if kind == "gathered":
            _require(picks.shape[0] - 1 <= rows <= picks.shape[0], f"gathered K' halves of {picks.shape[0]} rows "
                                                                   f"for a call of {rows}")
            picks = picks[:rows]
        picks = _stack([picks], rows)
        table, counts = picks.new_empty((rows, _width(picks))), picks.new_empty((rows,))
        _lay(picks, table, counts)
        _say(kind, f"{kind} K' tables (first: {rows} rows, {table.shape[1]} wide)")
    return entry(**dict(kwargs, block_tables=table.unsqueeze(1), sparse_mla_top_k=table.shape[1],
                        sparse_mla_top_k_lens=counts))
@contextlib.contextmanager
def routed(node, forward_batch):
    
    _require(id(node) in _NODES, "a call of a module this route never prepared")
    indexer, k, kept = _NODES[id(node)]
    why = kept if kept is not None else _outside(forward_batch)
    if why is not None:
        _say(why[0], why[1])
        yield
        return
    _require(not _CALL, "a sparse layer's call inside another's")
    split, node_module, nextn = _PEERS["split"], _PEERS["node"], _PEERS["nextn"]
    held = {"entry": flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla}
    if split is not None:
        held["padded"] = split._padded
    if node_module is not None:
        held["gather"] = node_module._gather
    if nextn is not None:
        held["exchange"] = nextn._exchange
    _CALL.update(held, indexer=indexer, k=k, parts=[], record=None, armed=False)
    indexer._topk_from_kpool_logits = lambda *args, **kw: _select(args, kw)
    if split is not None:
        split._padded = _padded
    if node_module is not None:
        node_module._gather = _gather
    if nextn is not None:
        nextn._exchange = _exchange
    try:
        yield
        _require(not _CALL["parts"] and _CALL["record"] is None,
                 "this layer's picks were ranked to K' and no kernel call of the layer took them")
    finally:
        if _CALL.get("armed"):
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = held["entry"]
        if split is not None:
            split._padded = held["padded"]
        if node_module is not None:
            node_module._gather = held["gather"]
        if nextn is not None:
            nextn._exchange = held["exchange"]
        del indexer._topk_from_kpool_logits
        _CALL.clear()
