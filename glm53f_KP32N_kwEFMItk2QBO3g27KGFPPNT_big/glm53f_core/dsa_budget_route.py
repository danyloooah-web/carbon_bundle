from __future__ import annotations
import contextlib
import weakref
import torch
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.model_executor.runner_utils import capture_mode
NODE = "dsa"
_STOCK = dsa_indexer_kpool._mqa_logits_row_chunks


_HELD: dict = {"batch": None, "chunks": {}}


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA logits budget: {why}")


def prepare(node) -> None:
    _require(callable(_STOCK), "dsa_indexer_kpool._mqa_logits_row_chunks is missing")
def _row_chunks(*, num_rows: int, num_cols: int, device: torch.device) -> tuple:
    
    if capture_mode.is_capture_mode or torch.cuda.is_current_stream_capturing():
        return _STOCK(num_rows=num_rows, num_cols=num_cols, device=device)
    key = (num_rows, num_cols, device.index)
    chunks = _HELD["chunks"].get(key)
    if chunks is None:
        chunks = _HELD["chunks"][key] = _STOCK(num_rows=num_rows, num_cols=num_cols, device=device)
    return chunks


@contextlib.contextmanager
def routed(node, forward_batch):
    _require(dsa_indexer_kpool._mqa_logits_row_chunks is _STOCK,
             "another route holds dsa_indexer_kpool._mqa_logits_row_chunks")
    held = _HELD["batch"]
    if held is None or held() is not forward_batch:
        _HELD["batch"], _HELD["chunks"] = weakref.ref(forward_batch), {}
    dsa_indexer_kpool._mqa_logits_row_chunks = _row_chunks
    try:
        yield
    finally:
        dsa_indexer_kpool._mqa_logits_row_chunks = _STOCK
