from __future__ import annotations
import dataclasses
import hashlib
import os
import torch
_MIN_ROWS = 4096
_MIN_KEYS = 4096
_ENGINES = {
    "314bce7c": {
        "srt/layers/attention/dsa/dsa_indexer_kpool.py":
            "db386335724accbbbeaaf5396be8af3b22fa71b5a9cbeb57f5b4198e0c9a0efc",
        "srt/models/deepseek_v2.py": "314e533c1606278d8294334edf13a23f082f1687a06f5e9bdf92a4958775ea83",
        "srt/distributed/parallel_state.py": "719134fd60c858df12bb002884e270f26b5d7249ecf7ac1340da710ec5c52d2d",
    },
}
_state = {"differs": None, "tally": {}}
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-idxsplit] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)

def _engine_differences() -> list:
    if _state["differs"] is None:
        import sglang

        root = os.path.dirname(os.path.abspath(sglang.__file__))
        best = None
        for digests in _ENGINES.values():
            differing = []
            for relative, expected in digests.items():
                try:
                    found = hashlib.sha256(_read(os.path.join(root, relative))).hexdigest()
                except OSError:
                    found = None
                if found != expected:
                    differing.append(relative)
            if best is None or len(differing) < len(best):
                best = differing
        _state["differs"] = best
    return _state["differs"]


def _capturing() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.cuda.is_current_stream_capturing() or bool(get_is_capture_mode())


def _gather(both: torch.Tensor, mine: torch.Tensor, group) -> None:
    torch.distributed.all_gather_into_tensor(both, mine, group=group.device_group)


def _cut(tensor, rows: slice):
    return None if tensor is None else tensor[rows]


def _why_stock(kw: dict, n: int):
    
    chunks, kv = kw["row_chunks"], kw["kv_fp8"]
    if kv is None:
        return "no pooled keys"
    if n < _MIN_ROWS:
        return f"under {_MIN_ROWS} rows"
    if kv[0].shape[0] < _MIN_KEYS:
        return f"under {_MIN_KEYS} pooled keys"
    if kw["out_rows"] not in (None, n) or kw["q_fp8"].shape[0] != n:
        return "padding rows"
    if chunks[0].start != 0 or any(a.stop != b.start for a, b in zip(chunks, chunks[1:])):
        return "row chunks that do not tile the rows"
    if _capturing():
        return "CUDA-graph capture"
    return None


def _half(kw: dict, lo: int, hi: int) -> dict:
    
    rows = slice(lo, hi)
    half = dict(kw)
    half.update(
        q_fp8=kw["q_fp8"][rows], weights=kw["weights"][rows], logits_starts=_cut(kw["logits_starts"], rows),
        logits_ends=kw["logits_ends"][rows], pool_lens=kw["pool_lens"][rows], seq_lens=_cut(kw["seq_lens"], rows),

        page_table=kw["page_table"] if kw["page_table_row_index"] is not None else _cut(kw["page_table"], rows),
        page_table_row_index=_cut(kw["page_table_row_index"], rows), topk_offsets=_cut(kw["topk_offsets"], rows),
        topk_row_starts=_cut(kw["topk_row_starts"], rows), out_rows=hi - lo,
        row_chunks=tuple(slice(max(c.start, lo) - lo, min(c.stop, hi) - lo)
                         for c in kw["row_chunks"] if c.stop > lo and c.start < hi),
    )
    return half


@dataclasses.dataclass
class Prepared:
    

    module: object
    group: object
    rank: int
    split: object = None
    why: str = ""


def _splitting(prepared: Prepared):
    module, group, rank = prepared.module, prepared.group, prepared.rank
    stock = type(module)._kpool_topk_by_row_chunks

    def split(**kw):
        n = kw["row_chunks"][-1].stop if kw["row_chunks"] else 0
        why = _why_stock(kw, n)
        if why is not None:
            _tally(f"whole rows: {why}")
            return stock(module, **kw)
        h = (n + 1) // 2
        lo, hi = (0, h) if rank == 0 else (h, n)
        part = stock(module, **_half(kw, lo, hi))
        if part.shape[0] != hi - lo:
            raise RuntimeError(f"glm-idxsplit: stock returned {part.shape[0]} rows for a half of {hi - lo}")
        mine = part if hi - lo == h else torch.cat([part, part.new_full((h - (hi - lo), part.shape[1]), -1)])
        both = part.new_empty((2 * h, part.shape[1]))
        _gather(both, mine.contiguous(), group)
        _tally("split rows", lambda: f"{n} rows x {kw['kv_fp8'][0].shape[0]} pooled keys, rows {lo}-{hi} here")
        return both[:n]

    return split

def prepare(module) -> Prepared:
    
    differs = _engine_differences()
    if differs:
        raise RuntimeError(f"glm-idxsplit: this engine is not one this node mirrors ({', '.join(sorted(_ENGINES))}); "
                           f"differing: {', '.join(differs)}")
    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

    if type(module) is not IndexerKPool:
        raise RuntimeError(f"glm-idxsplit serves IndexerKPool, got {type(module).__name__}")
    method = module._kpool_topk_by_row_chunks
    if getattr(method, "__func__", None) is not IndexerKPool._kpool_topk_by_row_chunks:
        raise RuntimeError("glm-idxsplit: the indexer already carries its own _kpool_topk_by_row_chunks")
    group = get_attn_tp_group()
    if group.world_size != 2:
        why = f"attention TP is {group.world_size} ranks"
        _tally("prepared stock-only", lambda: why)
        return Prepared(module, None, 0, why=why)
    prepared = Prepared(module, group, group.rank_in_group)
    prepared.split = _splitting(prepared)
    _tally("prepared", lambda: f"rank {prepared.rank} of 2, split from {_MIN_ROWS} rows and {_MIN_KEYS} pooled keys")
    return prepared


def forward(prepared: Prepared, *args, **kwargs):
    
    module = prepared.module
    if prepared.group is None:
        return module.forward(*args, **kwargs)
    module._kpool_topk_by_row_chunks = prepared.split
    try:
        return module.forward(*args, **kwargs)
    finally:
        del module._kpool_topk_by_row_chunks
