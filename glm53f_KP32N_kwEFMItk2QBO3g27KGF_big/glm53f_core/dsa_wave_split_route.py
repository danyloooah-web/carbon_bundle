from __future__ import annotations
import contextlib
import logging
import flashinfer.decode
import torch
from glm53f_core import dsa_rows_route as row_owned
NODE = "dsa"
_STOCK = row_owned._STOCK
_STATE: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA wave split: {why}")
def _say(tag: str, detail: str) -> None:
    
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA wave split: %s", detail)


def prepare(node) -> None:
    
    if "active" in _STATE:
        return
    _require("group" in row_owned._STATE, "dsa_rows_route is not prepared before this route (the components' order)")
    _STATE["active"] = row_owned._STATE["group"] is not None
    if not _STATE["active"]:
        _say("inactive", "row-owned attention is inactive on this engine: nothing to split")


def _cut(rows: int, device) -> int | None:
    
    if "sms" not in _STATE:
        _STATE["sms"] = torch.cuda.get_device_properties(device).multi_processor_count
    sms = _STATE["sms"]
    return sms if sms < rows <= sms + sms // 2 else None

def _rows(*args, **kwargs):
    
    state, q, out = row_owned._STATE, kwargs.get("query"), kwargs.get("out")
    if args or not (isinstance(q, torch.Tensor) and isinstance(out, torch.Tensor) and "qv" in state
                    and q.data_ptr() == state["qv"].data_ptr() and out.data_ptr() == state["out"].data_ptr()):
        _say("other", "a call that is not A1's on its own rows: stock's kernel")
        return _STOCK(*args, **kwargs)
    rows = q.shape[0]
    cut = _cut(rows, q.device)
    if cut is None:
        _say(f"one {rows}", f"A1's call of {rows} rows is one call ({_STATE['sms']} SMs)")
        return _STOCK(**kwargs)
    tables, lengths, lens = kwargs["block_tables"], kwargs["seq_lens"], kwargs.get("sparse_mla_top_k_lens")
    _require(tables.shape[0] == lengths.shape[0] == out.shape[0] == rows and (lens is None or lens.shape[0] == rows),
             f"A1's call of {rows} rows with {tables.shape[0]} block tables, {lengths.shape[0]} lengths, "
             f"{out.shape[0]} output rows")
    for part in (slice(0, cut), slice(cut, rows)):
        call = dict(kwargs, query=q[part], block_tables=tables[part], seq_lens=lengths[part], out=out[part])
        if lens is not None:
            call["sparse_mla_top_k_lens"] = lens[part]
        o = _STOCK(**call)
        o = o[0] if isinstance(o, tuple) else o
        _require(o.data_ptr() == call["out"].data_ptr() and o.numel() == call["out"].numel(),
                 "stock's kernel returned rows of its own, not the output it was handed")
    _say("split", f"A1's rows as two calls (first: {cut} + {rows - cut} rows, {_STATE['sms']} SMs)")
    return out


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if not _STATE.get("active") or not (mode.is_target_verify() or mode.is_draft_extend_v2()):
        yield
        return
    _require(flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is row_owned._attention, "A1 is not set (order)")
    _require(row_owned._STOCK is _STOCK, f"another route holds the kernel A1 runs its rows with ({row_owned._STOCK!r})")
    row_owned._STOCK = _rows
    try:
        yield
    finally:
        row_owned._STOCK = _STOCK
