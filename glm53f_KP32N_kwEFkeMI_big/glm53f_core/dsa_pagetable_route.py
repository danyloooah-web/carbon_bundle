from __future__ import annotations
import contextlib
import logging
import flashinfer.decode
import torch
from sglang.srt.layers.attention import dsa_backend
from glm53f_core import dsa_rows_route as row_owned
from glm53f_core import idx_nextn_route
NODE = "dsa"
_BACKEND = dsa_backend.DeepseekSparseAttnBackend
_STOCK_PAD = _BACKEND._pad_trtllm_sparse_page_table
_STOCK_LENS = dsa_backend.prepare_trtllm_nope_sparse_metadata
_STATE: dict = {}
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA page table: {why}")


def _say(tag: str, detail: str) -> None:
    
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA page table: %s", detail)


def prepare(node) -> None:
    
    if "native" in _STATE:
        return
    _require("group" in row_owned._STATE, "dsa_rows_route is not prepared before this route (the components' order)")
    if row_owned._STATE["group"] is None:
        _STATE["native"] = None
        _say("inactive", "row-owned attention is inactive on this engine: stock's pad and sparse metadata")
        return
    import dsa_pagetable_native
    _require(callable(getattr(dsa_pagetable_native, "table", None)), "dsa_pagetable_native lacks table")
    _STATE["native"] = dsa_pagetable_native


def _write(picks: torch.Tensor, cut: slice) -> torch.Tensor:
    
    state = _STATE
    if "table" not in state:
        _require(not torch.cuda.is_current_stream_capturing(),
                 "the first served call is inside a CUDA-graph capture (no eager warm-up preceded it)")
        rows, width = row_owned._STATE["capacity"], picks.shape[1]
        state.update(width=width, lens=torch.ones(rows, dtype=torch.int32, device=picks.device),
                     table=torch.full((rows, (width + 3) // 4 * 4), -1, dtype=torch.int32, device=picks.device))
        _say("table", f"a table of {rows} rows x {state['table'].shape[1]} for {width} picks a row")
    _require(picks.shape[1] == state["width"] and picks.device == state["table"].device, "another width or device")
    state["native"].table(picks, state["table"][cut], state["lens"][cut])
    return state["table"]


def _own_table(mine: torch.Tensor, rows: slice, n: int, out_rows: int) -> torch.Tensor:
    
    cut = row_owned._cut(n) if row_owned._CALL.get("skip") and out_rows == n else None
    if cut is None or (cut.start, cut.stop) != (rows.start, rows.stop):
        return row_owned._own_picks(mine, rows, n, out_rows)
    table = _write(mine, cut)
    row_owned._CALL["skipped"] = (n, cut.start, cut.stop)
    view = _CALL["view"] = table[:n, : _STATE["width"]]
    _say("own", f"the verify picks written into the table (first: rows {cut.start}-{cut.stop} of {n})")
    return view


def _pad(backend, page_table_1: torch.Tensor):
    
    view, n = _CALL.pop("view", None), page_table_1.shape[0]
    if view is not None:
        _require(page_table_1 is view, "this layer's picks stayed on their owners, but the pad got another table")
        table = _STATE["table"][:n]
    else:
        _require("skipped" not in row_owned._CALL, f"rows {row_owned._CALL.get('skipped')} only are in flight")
        if page_table_1.dim() == 2 and page_table_1.dtype == torch.int32 and 0 < n < row_owned.MIN_ROWS:
            table = _write(page_table_1, slice(0, n))[:n]
            _say("low", f"every row of whole picks written under A1's {row_owned.MIN_ROWS} rows (first: {n} rows)")
            _CALL["table"] = table
            return table, table.shape[1]
        if _CALL["decode"]:
            _CALL["table"] = None
            _say(f"stock decode {n}", f"stock's pad and sparse metadata serve a draft decode of {n} rows")
            return _STOCK_PAD(backend, page_table_1)

        rows = backend.forward_metadata.dsa_cache_seqlens_int32.shape[0]
        whole = page_table_1.dim() == 2 and page_table_1.dtype == torch.int32
        cut = row_owned._cut(n) if n == rows and whole else None
        if cut is None:
            _CALL["table"] = None
            _say(f"stock {n}", f"stock's pad and sparse metadata serve a call of {n} rows")
            return _STOCK_PAD(backend, page_table_1)
        table = _write(page_table_1[cut], cut)[:n]
        row_owned._CALL["skipped"] = (n, cut.start, cut.stop)
        _say("whole", f"this rank's rows of whole picks written (first: rows {cut.start}-{cut.stop} of {n})")
    _CALL["table"] = table
    return table, table.shape[1]


def _lengths(page_table: torch.Tensor) -> torch.Tensor:
    table = _CALL.get("table")
    if table is None:
        return _STOCK_LENS(page_table)
    _require(page_table is table, "the sparse metadata was asked for another table than the one the pad returned")
    return _STATE["lens"][: table.shape[0]]


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    a1_mode = mode.is_target_verify() or mode.is_draft_extend_v2()
    if _STATE.get("native") is None or not (a1_mode or mode.is_decode()):
        yield
        return
    decode = not a1_mode
    _require(decode or flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is row_owned._attention,
             "A1 is not set (order)")
    _require(_BACKEND._pad_trtllm_sparse_page_table is _STOCK_PAD and dsa_backend.prepare_trtllm_nope_sparse_metadata
             is _STOCK_LENS, "another route holds the page table's pad or the sparse metadata")
    own = idx_nextn_route._exchange is row_owned._own_picks
    _CALL.clear()
    _CALL["decode"] = decode
    if own:
        idx_nextn_route._exchange = _own_table
    _BACKEND._pad_trtllm_sparse_page_table, dsa_backend.prepare_trtllm_nope_sparse_metadata = _pad, _lengths
    try:
        yield
    finally:
        _BACKEND._pad_trtllm_sparse_page_table = _STOCK_PAD
        dsa_backend.prepare_trtllm_nope_sparse_metadata = _STOCK_LENS
        if own:
            idx_nextn_route._exchange = row_owned._own_picks
        _CALL.clear()
