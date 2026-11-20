from __future__ import annotations
import contextlib
import logging
import flashinfer.decode
import torch
from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.model_executor.runner import get_is_capture_mode
NODE = "dsa"
MIN_ROWS = 4096
MAX_ROWS = 16384
CUTS_SMALL = (0.5, 0.5)
CUTS_LARGE = (0.15, 0.7, 0.15)
FP8 = torch.float8_e4m3fn
_STOCK = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
_UNSERVED = ("out", "lse", "cum_seq_lens_q", "max_q_len", "sinks", "return_lse", "enable_dcp")
_PAIR: dict = {}
_SERVED = [0]
_SAID: set = set()
_XSPLIT: list = []
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA token split: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf DSA token split: %s", detail)
def prepare(node) -> None:
    _require(callable(_STOCK), "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla missing")
    if "group" in _PAIR:
        return
    group = get_attn_tp_group()
    if group.world_size != 2:
        _PAIR["group"] = None
        _say("inactive", f"attention TP is {group.world_size} ranks on this engine: stock's call")
        return
    import torch.distributed._symmetric_memory as symmetric_memory
    from cuda.bindings import runtime
    _require(callable(getattr(symmetric_memory, "empty", None))
             and callable(getattr(symmetric_memory, "rendezvous", None)), "torch's symmetric memory lacks empty or rendezvous")
    _require(callable(getattr(runtime, "cudaMemcpy2DAsync", None)), "cuda.bindings.runtime lacks cudaMemcpy2DAsync")
    _PAIR.update(group=group, symmetric=symmetric_memory, runtime=runtime)
def _buffers(heads: int, width: int, out_width: int, device) -> dict:
    state = _PAIR.get("state")
    if state is not None:
        _require(state["dims"] == (heads, width, out_width),
                 f"a call of {heads} heads x {width} -> {out_width} after buffers for {state['dims']}")
        return state
    group, symmetric = _PAIR["group"], _PAIR["symmetric"]
    half = (MAX_ROWS + 1) // 2

    qbuf = symmetric.empty(half * 2 * heads * width, dtype=torch.uint8, device=device)
    obuf = symmetric.empty(MAX_ROWS * 2 * heads * out_width, dtype=torch.bfloat16, device=device)
    hq = symmetric.rendezvous(qbuf, group.device_group)
    ho = symmetric.rendezvous(obuf, group.device_group)
    rank = group.rank_in_group
    peer = 1 - rank
    state = _PAIR["state"] = {
        "dims": (heads, width, out_width), "rank": rank, "peer": peer, "qbuf": qbuf, "obuf": obuf, "hq": hq, "ho": ho,
        "qwords": qbuf.view(torch.int64).view(half, 2 * heads * width // 8),
        "qv": qbuf.view(FP8).view(half, 2 * heads, width),
        "ov": obuf.view(MAX_ROWS, 2 * heads, out_width),
        "peer_q": hq.get_buffer(peer, (half * 2 * heads * width,), torch.uint8, 0),
        "peer_o": ho.get_buffer(peer, (MAX_ROWS * 2 * heads * out_width,), torch.bfloat16, 0),
        "side": torch.cuda.Stream(device=device), "bytes": torch.empty((), dtype=torch.bfloat16).element_size()}
    _say("pair", f"rank {rank} of the TP2 pair: buffers for {half} tokens' queries and {MAX_ROWS} tokens' outputs "
                 "exchanged")
    return state

def _cuts(n: int, cuts: tuple) -> list:
    
    out, acc = [0], 0.0
    for fraction in cuts:
        acc += fraction
        out.append(min(n, int(round(acc * n))))
    out[-1] = n
    return out

def _push_rows(state: dict, dst: int, dst_pitch: int, src: int, src_pitch: int, width: int, m: int) -> None:
    
    runtime = _PAIR["runtime"]
    err, = runtime.cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, m,
                                     runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice, state["side"].cuda_stream)
    _require(err == runtime.cudaError_t.cudaSuccess, f"the pitched copy to the peer failed: {err}")


def _outside(kwargs: dict) -> tuple[str, str] | None:
    
    q, tables, seq_lens = kwargs.get("query"), kwargs.get("block_tables"), kwargs.get("seq_lens")
    if not (isinstance(q, torch.Tensor) and isinstance(tables, torch.Tensor) and isinstance(seq_lens, torch.Tensor)):
        return "arguments", "query, block tables or sequence lengths not passed as tensors by keyword"
    if q.dtype != FP8 or q.dim() != 4 or q.shape[1] != 1 or not q.is_contiguous():
        return "query", f"the query is {q.dtype} {tuple(q.shape)}, not contiguous FP8 [tokens, 1, heads, width]"
    rows = q.shape[0]
    if rows < MIN_ROWS or rows > MAX_ROWS:
        return "rows", f"{rows} tokens, outside {MIN_ROWS} to {MAX_ROWS}"
    if tables.shape[0] != rows or seq_lens.shape[0] != rows:
        return "tables", f"{tables.shape[0]} block-table rows and {seq_lens.shape[0]} sequence lengths for {rows} tokens"
    lens = kwargs.get("sparse_mla_top_k_lens")
    if lens is not None and (not isinstance(lens, torch.Tensor) or lens.shape[0] != rows):
        return "lengths", "top-k lengths that are not one a token"
    for name in _UNSERVED:
        if kwargs.get(name) is not None and kwargs.get(name) is not False:
            return name, f"{name} is set"
    if not isinstance(kwargs.get("kv_lora_rank"), int):
        return "rank", "no integer latent rank"
    return None


def _split(kwargs: dict) -> torch.Tensor:
    
    q = kwargs["query"]
    rows, heads, width = q.shape[0], q.shape[-2], q.shape[-1]
    out_width = kwargs["kv_lora_rank"]
    state = _buffers(heads, width, out_width, q.device)
    rank, peer, hq, ho, side = state["rank"], state["peer"], state["hq"], state["ho"], state["side"]
    first = rows // 2
    count, start = (first, rows - first), (0, first)
    my_start, my_count, peer_start, peer_count = start[rank], count[rank], start[peer], count[peer]
    cuts = CUTS_LARGE if rows >= 8192 else CUTS_SMALL
    my_cuts, peer_cuts = _cuts(my_count, cuts), _cuts(peer_count, cuts)
    chunks = len(cuts)
    qf = q.view(rows, heads, width)
    qv, ov, e = state["qv"], state["ov"], state["bytes"]
    q_row, o_row = 2 * heads * width, 2 * heads * out_width * e
    tables, seq_lens, lens = kwargs["block_tables"], kwargs["seq_lens"], kwargs.get("sparse_mla_top_k_lens")
    main = torch.cuda.current_stream()


    hq.barrier(channel=chunks)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        for c in range(chunks):
            a, b = peer_cuts[c], peer_cuts[c + 1]
            _push_rows(state, state["peer_q"].data_ptr() + a * q_row + rank * heads * width, q_row,
                       qf.data_ptr() + (peer_start + a) * heads * width, heads * width, heads * width, b - a)
            hq.put_signal(peer, channel=c)



    words = heads * width // 8
    state["qwords"][:my_count, rank * words:(rank + 1) * words].copy_(
        qf.view(rows, heads * width)[my_start:my_start + my_count].view(torch.int64))
    for c in range(chunks):
        a, b = my_cuts[c], my_cuts[c + 1]
        m = b - a
        hq.wait_signal(peer, channel=c)
        mine = ov[my_start + a:my_start + b]
        call = dict(kwargs)
        call.update(query=qv[a:b].unsqueeze(1), block_tables=tables[my_start + a:my_start + b],
                    seq_lens=seq_lens[my_start + a:my_start + b], out=mine.unsqueeze(1))
        if lens is not None:
            call["sparse_mla_top_k_lens"] = lens[my_start + a:my_start + b]
        o = _STOCK(**call)
        o = o[0] if isinstance(o, tuple) else o
        _require(o.dtype == torch.bfloat16 and o.data_ptr() == mine.data_ptr() and o.numel() == mine.numel(),
                 f"stock's kernel returned {o.dtype} {tuple(o.shape)} rows of its own, not the output it was handed")
        done = torch.cuda.Event()
        done.record(main)
        with torch.cuda.stream(side):
            side.wait_event(done)
            at = (my_start + a) * o_row + peer * heads * out_width * e
            _push_rows(state, state["peer_o"].data_ptr() + at, o_row, state["obuf"].data_ptr() + at, o_row,
                       heads * out_width * e, m)
            ho.put_signal(peer, channel=c)
    for c in range(chunks):
        ho.wait_signal(peer, channel=c)
    main.wait_stream(side)
    _SERVED[0] += 1
    _say("ours", f"the pair serves eager prefill calls (first: {rows} tokens, {chunks} chunks a rank)")
    return ov[:rows, rank * heads:(rank + 1) * heads].unsqueeze(1)


def _attention(*args, **kwargs):
    
    if _XSPLIT and _XSPLIT[0].owns(kwargs):
        out = _XSPLIT[0].serve(args, kwargs, _STOCK)
        _SERVED[0] += 1
        return out
    if args:
        _say("positional", "stock's kernel serves a call: arguments passed by position")
        return _STOCK(*args, **kwargs)
    why = _outside(kwargs)
    if why is not None:
        _say(why[0], f"stock's kernel serves a call: {why[1]}")
        return _STOCK(**kwargs)
    return _split(kwargs)


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if (_PAIR.get("group") is None or mode.is_draft_extend_v2()
            or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False)
            or torch.cuda.is_current_stream_capturing() or get_is_capture_mode()):
        yield
        return
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _attention
    held = _XSPLIT[0].enter(node) if _XSPLIT else None
    try:
        yield
    finally:
        if held is not None:
            _XSPLIT[0].leave(held)
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _STOCK
