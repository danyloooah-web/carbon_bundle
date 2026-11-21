from __future__ import annotations
import contextlib
import logging
import sys
from sglang.srt.layers.attention import dsa_backend
from glm53f_core import idx_split_node
NODE = "dsa"
_STATES: dict = {}
_TOKEN_SPLIT: list = []
_CALL = {"skip": False, "skipped": 0, "rank": 0}
_STOCK_LENS = dsa_backend.prepare_trtllm_nope_sparse_metadata
_KEPT: dict = {}
def _token_split():
    if not _TOKEN_SPLIT:
        found = sys.modules.get(__name__.rsplit(".", 1)[0] + ".dsa_split_route")
        _TOKEN_SPLIT.append(found)
        logging.getLogger(__name__).info(
            "glmf indexer split: %s", "the top-k exchange is skipped where the token split serves the layer"
            if found is not None else "no token split in this bundle: every split call keeps the exchange")
    return _TOKEN_SPLIT[0]
def prepare(node) -> None:
    if id(node) in _STATES:
        return
    indexer = getattr(node, "indexer", None)
    if indexer is None:
        raise RuntimeError("glmf indexer split: the claimed attention module holds no indexer")
    _STATES[id(node)] = idx_split_node.prepare(indexer)
def _padded(state, kw: dict):
    module, rank = state.module, state.rank
    stock = type(module)._kpool_topk_by_row_chunks
    n = kw["row_chunks"][-1].stop if kw["row_chunks"] else 0
    why = idx_split_node._why_stock(kw, n)
    if why is not None:
        idx_split_node._tally(f"whole rows: {why}")
        return stock(module, **kw)
    h = n // 2
    lo, hi = (0, h) if rank == 0 else (h, n)
    part = stock(module, **idx_split_node._half(kw, lo, hi))
    if part.shape[0] != h:
        raise RuntimeError(f"glm-idxsplit: stock returned {part.shape[0]} rows for a half of {h}")
    table = _kept_rows(part, n, lo, hi)
    _CALL["skipped"] += 1
    idx_split_node._tally("split rows at stock's pad width, no exchange (PAD-ZC)",
                          lambda: f"{n} rows x {table.shape[1]} columns, rows {lo}-{hi} here")
    return table


def _skipping(state, token_split):

    def call(**kw):
        n = kw["row_chunks"][-1].stop if kw["row_chunks"] else 0
        _CALL["rank"] = state.rank
        _CALL["skip"] = (token_split.flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is token_split._attention
                         and n % 2 == 0 and token_split.MIN_ROWS <= n <= token_split.MAX_ROWS)
        try:
            return _padded(state, kw) if _CALL["skip"] else state.split(**kw)
        finally:
            _CALL["skip"] = False

    return call


def _shares_topk(forward_batch) -> bool:
    spec = getattr(forward_batch, "spec_info", None)
    return (bool(getattr(forward_batch, "reuse_dsa_topk_indices", False))
            or getattr(spec, "dsa_seed_topk_capture", None) is not None)


def _kept_rows(part, n: int, lo: int, hi: int):
    
    kept = _KEPT.get(part.device)
    if kept is None:
        import dsa_pagetable_native

        rows, width = _token_split().MAX_ROWS, part.shape[1]
        kept = _KEPT[part.device] = {"native": dsa_pagetable_native, "width": width, "lens": part.new_ones((rows,)),
                                     "table": part.new_full((rows, (width + 3) // 4 * 4), -1)}
        logging.getLogger(__name__).info(
            "glmf indexer split: a kept top-k table of %d x %d for the calls that skip the exchange (PTONCE)",
            rows, kept["table"].shape[1])
    if part.shape[1] != kept["width"] or n > kept["table"].shape[0]:
        raise RuntimeError(f"glmf indexer split: a top-k of {n} rows x {part.shape[1]} for a kept table of "
                           f"{tuple(kept['table'].shape)} made for {kept['width']} columns")


    if dsa_backend.prepare_trtllm_nope_sparse_metadata is not _lengths:
        if dsa_backend.prepare_trtllm_nope_sparse_metadata is not _STOCK_LENS:
            raise RuntimeError("glmf indexer split: another route holds the backend's sparse metadata name in a call "
                               "that skips the exchange (PTONCE needs stock's prepare_trtllm_nope_sparse_metadata)")
        dsa_backend.prepare_trtllm_nope_sparse_metadata = _lengths
    kept["native"].table(part, kept["table"][lo:hi], kept["lens"][lo:hi])
    table = kept["table"][:n]
    _CALL["kept"] = (table, kept["lens"][:n])
    return table


def _lengths(page_table):
    
    kept = _CALL.get("kept")
    if kept is None or page_table is not kept[0]:
        return _STOCK_LENS(page_table)
    _CALL["fed"] += 1
    return kept[1]


@contextlib.contextmanager
def routed(node, forward_batch):
    state = _STATES[id(node)]
    if state.group is None:
        yield
        return
    token_split = _token_split()
    if token_split is None or _shares_topk(forward_batch):
        state.module._kpool_topk_by_row_chunks = state.split
        try:
            yield
        finally:
            del state.module._kpool_topk_by_row_chunks
        return
    state.module._kpool_topk_by_row_chunks = _skipping(state, token_split)
    _CALL["skipped"] = 0
    served = token_split._SERVED[0]
    found, _CALL["fed"] = dsa_backend.prepare_trtllm_nope_sparse_metadata, 0
    try:
        yield
        if _CALL["skipped"] and token_split._SERVED[0] - served < _CALL["skipped"]:
            raise RuntimeError(
                f"glmf indexer split: {_CALL['skipped']} top-k call(s) skipped the exchange but the token split served "
                f"{token_split._SERVED[0] - served} attention call(s) of this layer: the peer's rows were never there")
        if _CALL["fed"] < _CALL["skipped"]:
            raise RuntimeError(
                f"glmf indexer split: {_CALL['skipped']} kept top-k table(s) of this layer, the sparse metadata asked "
                f"for {_CALL['fed']}: another reader took a table whose peer rows are not this call's (PTONCE)")
    finally:
        _CALL["skip"] = False
        _CALL.pop("kept", None)
        if dsa_backend.prepare_trtllm_nope_sparse_metadata is _lengths:
            dsa_backend.prepare_trtllm_nope_sparse_metadata = found
        del state.module._kpool_topk_by_row_chunks
