from __future__ import annotations
import contextlib
import logging
import sys
import rotq_native
import torch
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.quantization import unquant
from sglang.srt.runtime_context import get_disagg
from glm53f_core import dsa_rotq, dsa_rotq_route, idx_nextn_route
NODE = "dsa"
HEADS, DIM, NEXT_N = idx_nextn_route.HEADS, idx_nextn_route.DIM, idx_nextn_route.NEXT_N
FLOOR = 168
SEED = 159
FP8 = torch.float8_e4m3fn
_STOCK_TV = IndexerKPool._forward_cuda_target_verify
_STOCK_SPLIT = idx_nextn_route._split
_PEERS: dict = {}
_SHAPES: dict = {}
_NODES: dict = {}
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf indexer own-row query: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf indexer own-row query: %s", detail)
def _peer(name: str):
    return sys.modules.get(__name__.rsplit(".", 1)[0] + "." + name)
def _not_linear(layer, m: int) -> str | None:
    n, k = layer.weight.shape
    if getattr(layer, "small_m_rowmajor", False) or getattr(layer, "small_m_tiled_weight", None) is not None:
        return "gfx95's small-M GEMM"
    gemm = unquant.get_bf16_gemm_backend()
    if not (gemm.is_cutedsl() or gemm.is_gemv()):
        return None
    if unquant._mm_bf16 is not None and 0 < m <= unquant._BF16_LOW_M_AUTOTUNE_MAX_M:
        return "FlashInfer's autotuned CuTe DSL GEMM"
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        return "a tuned split-K GEMM"
    if any(rule is not None and rule(m, n, k) for rule in (unquant._use_hopper_bf16_tiny_gemv,
                                                           unquant._use_hopper_bf16_gemv)):
        return "a Hopper GEMV"
    if unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k):
        return "the CuTe DSL TGV GEMM"
    return None


def _own_wqb(q: torch.Tensor, weight: torch.Tensor, rows: slice, n: int) -> torch.Tensor:
    out = q.new_empty((n, weight.shape[0]))
    torch.mm(q[rows], weight.t(), out=out[rows])
    return out


def _own_project(indexer, x: torch.Tensor, rows: slice) -> torch.Tensor:
    gate = _PEERS["gate"]
    _require(IndexerKPool._apply_q_scale_and_softmax_scale is gate._scale,
             "dsa_gate_route's gate methods are set without its scaling (WSCALE)")
    raw = gate._raw(indexer, x[rows])
    dsa_rotq_route.arm_gates(raw, indexer.n_heads ** -0.5, indexer.softmax_scale)
    return raw


def _own_logits_gate(indexer, x: torch.Tensor, q_scale: torch.Tensor, rows: slice, n: int) -> torch.Tensor:
    
    weights = _PEERS["gate"]._raw(indexer, x[rows]) * indexer.n_heads ** -0.5
    out = q_scale.new_empty((n, indexer.n_heads, 1))
    torch.mul(weights.unsqueeze(-1) * q_scale[rows], indexer.softmax_scale, out=out[rows])
    return out


def _own_rotq(x: torch.Tensor, rows: slice) -> tuple:
    
    out, y = torch.empty_like(x), torch.empty_like(x, dtype=FP8)
    s = x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)
    gates = dsa_rotq_route.take_gates(x[rows])
    if gates is None:
        rotq_native.rotate_quant(x[rows], out[rows], y[rows], s[rows], dsa_rotq.D ** -0.5, dsa_rotq.GRID_PER_SM)
        return out, y, s
    final = torch.empty_like(s)
    rotq_native.rotate_quant_gated(x[rows], out[rows], y[rows], s[rows], gates[0], final[rows], dsa_rotq.D ** -0.5,
                                   gates[1], gates[2], dsa_rotq.GRID_PER_SM)
    dsa_rotq_route._SCALED[:] = [(gates[0], s, final)]
    return out, y, s


def _cut(n: int, rank: int, world: int) -> slice:
    share = n // world
    return slice(rank * share, (rank + 1) * share)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    
    kind = torch.int16 if a.element_size() == 2 else torch.int32
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(kind), b.view(kind))


def _probe(node, indexer) -> None:
    
    _require(not torch.cuda.is_current_stream_capturing(), "the probe would run inside a CUDA-graph capture")
    group, mode = idx_nextn_route._SPLIT["group"], get_disagg().disaggregation_mode
    shapes = []
    if group is not None and group.world_size >= 2 and mode != "prefill":
        step, a1 = NEXT_N * group.world_size, _PEERS["rows"]
        top = max(768, (a1._STATE.get("capacity") or 0) if a1 is not None else 0)
        shapes += [("decode", n) for n in range(-(-FLOOR // step) * step, top + 1, step)]
    _SHAPES["probed"] = True
    if not shapes:
        _say("no shapes", "no split on this engine: nothing to probe")
        return
    weight, gate = indexer.wq_b.weight, _PEERS["gate"]
    rank, world = group.rank_in_group, group.world_size
    gen = torch.Generator(device=weight.device).manual_seed(SEED)
    top = max(n for _, n in shapes)
    q = torch.randn((top, weight.shape[1]), generator=gen, device=weight.device, dtype=torch.bfloat16)
    flags = []
    for phase, n in shapes:
        rows = _cut(n, rank, world)
        flags.append(_not_linear(indexer.wq_b, n) is None
                     and _same(torch.nn.functional.linear(q[:n], weight)[rows], _own_wqb(q[:n], weight, rows, n)[rows]))
    del q
    x = torch.randn((top, indexer.hidden_size), generator=gen, device=weight.device, dtype=torch.bfloat16)
    for phase, n in shapes:
        rows = _cut(n, rank, world)
        flags.append(gate is not None and id(indexer) in gate._WEIGHT
                     and _same(gate._raw(indexer, x[:n])[rows], gate._raw(indexer, x[:n][rows])))
    voted = torch.tensor(flags, dtype=torch.int32, device=weight.device)
    torch.distributed.all_reduce(voted, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
    verdicts = voted.tolist()
    for i, shape in enumerate(shapes):
        _SHAPES[shape] = (bool(verdicts[i]), bool(verdicts[len(shapes) + i]))
    off = [f"{p} {n}: {'' if q_ok else 'wq_b '}{'' if g_ok else 'gates'}" for (p, n), (q_ok, g_ok) in
           ((s, _SHAPES[s]) for s in shapes) if not (q_ok and g_ok)]
    _say("probe", f"rank {rank} of {world}: {len(shapes) - len(off)} of {len(shapes)} shapes this route's GEMMs serve "
                  f"bit for bit; v156's GEMMs at {off or 'none'}")


def prepare(node) -> None:
    
    if id(node) in _NODES:
        return
    if not _PEERS:
        _PEERS.update(rows=_peer("dsa_rows_route"), gate=_peer("dsa_gate_route"), merge=_peer("qb_wqb_merge_route"))
        _require(_PEERS["rows"] is None or _PEERS["rows"].MIN_ROWS == FLOOR,
                 f"A1 serves from {_PEERS['rows'].MIN_ROWS if _PEERS['rows'] else 0} rows, this route's floor is {FLOOR}")
    indexer = getattr(node, "indexer", None)
    why = ("no pooled indexer" if not isinstance(indexer, IndexerKPool)
           else "idx_nextn_route did not prepare it" if idx_nextn_route._INDEXERS.get(id(node)) is not indexer
           else "the indexers' scale format is not ue8m0 (rotq_native)" if dsa_rotq_route._ROUND != [True]
           else "the indexer applies RoPE" if not indexer.skip_rope and indexer.rope_head_dim > 0 else None)
    if why is not None:
        _NODES[id(node)] = None
        _say("inactive " + why, f"{why}: v156's query")
        return
    lin, w = indexer.wq_b, indexer.wq_b.weight
    _require(type(lin.quant_method) is unquant.UnquantizedLinearMethod and lin.bias is None and w.dtype == torch.bfloat16
             and tuple(w.shape) == (HEADS * DIM, indexer.q_lora_rank) and w.is_contiguous() and not w.requires_grad
             and not getattr(lin, "small_m_rowmajor", False) and getattr(lin, "small_m_tiled_weight", None) is None
             and indexer.n_heads == HEADS and indexer.head_dim == DIM and indexer.block_size == DIM,
             f"wq_b {w.dtype} {tuple(w.shape)} ({type(lin.quant_method).__name__}) of {indexer.n_heads} x "
             f"{indexer.head_dim} heads is not stock's plain BF16 linear of this checkpoint's indexer")
    if "probed" not in _SHAPES:
        _probe(node, indexer)
        _SHAPES["weight"] = tuple(w.shape)
    _require(_SHAPES["weight"] == tuple(w.shape), f"a wq_b of {tuple(w.shape)} after the probe's {_SHAPES['weight']}")
    _NODES[id(node)] = indexer




def _keeps_picks(mode, n: int, q_rows: int) -> bool:
    
    a1 = _PEERS["rows"]
    return (mode.is_target_verify() and a1 is not None and a1._STATE.get("group") is not None
            and bool(a1._CALL.get("skip")) and q_rows == n and a1._cut(n) is not None)


def _split_at(indexer, pool, n: int) -> slice | None:
    
    group = idx_nextn_route._SPLIT["group"]
    if group is None or group.world_size < 2 or n < FLOOR or n % (NEXT_N * group.world_size) or n % 4:
        return None
    if idx_nextn_route.EXCHANGE == "all_reduce" and (pool.size + pool.page_size > idx_nextn_route.EXACT_SLOTS
                                                     or n * indexer.topk_output_width * 4 > idx_nextn_route.STAGE_BYTES):
        return None
    _say("floor", f"the split from {FLOOR} rows where A1 keeps the picks (first: {n} rows)")
    return _cut(n, group.rank_in_group, group.world_size)


def _verify_rows(indexer, batch, metadata, n: int, q_rows: int) -> slice | None:
    
    mode = batch.forward_mode
    if id(indexer) not in idx_nextn_route._HELD:
        return None
    if mode.is_draft_extend_v2():
        plan = metadata.attn_metadata.kpool_write_plan
        if plan is None or plan.num_draft_tokens != NEXT_N:
            return None
    elif not mode.is_target_verify():
        return None
    probe = torch.empty((0, HEADS, DIM), dtype=FP8, device=indexer.wq_b.weight.device)
    if (indexer.paged_mqa_logits_backend.is_aiter() or indexer._should_use_tilelang_paged_mqa_logits(probe)
            or n < NEXT_N or n % NEXT_N):
        return None
    pool = dsa_indexer_kpool.get_token_to_kv_pool()
    if n < idx_nextn_route.MIN_ROWS and _keeps_picks(mode, n, q_rows):
        return _split_at(indexer, pool, n)
    return _STOCK_SPLIT(indexer, pool, n)


def _split(indexer, pool, n: int) -> slice | None:
    
    held = _CALL.get("split")
    _require(held is not None and held[0] is indexer and held[1] == n and not _CALL["taken"],
             f"served() asked to split {n} rows of a call this route did not decide (or a second time)")
    _CALL["taken"] = True
    return held[2]


def _verify_entry(indexer, **kw):
    
    x, q_lora, metadata = kw["x"], kw["q_lora"], kw["metadata"]
    n = metadata.get_seqlens_expanded().shape[0]
    rows = _verify_rows(indexer, kw["forward_batch"], metadata, n, x.shape[0])
    _CALL.update(split=(indexer, n, rows), taken=False)
    try:
        if rows is None or not kw.get("return_indices", True) or not x.shape[0] == q_lora.shape[0] == n:
            return _STOCK_TV(indexer, **kw)
        wqb_ok, gate_ok = _SHAPES.get(("decode", n), (False, False))
        kind = "draft-extend" if kw["forward_batch"].forward_mode.is_draft_extend_v2() else "verify"
        with _query(indexer, kind, rows, n, x, q_lora, wqb_ok, gate_ok):
            out = _STOCK_TV(indexer, **kw)
        _require(_CALL["taken"], "this rank's rows' query was made and served() never asked for the split")
        return out
    finally:
        _CALL.clear()



def _wqb(indexer, inp, own: bool):
    
    call = _CALL["query"]
    _require(not call["wqb"] and inp is call["q_lora"], "a second wq_b call, or one on another input than q_lora")
    call["wqb"] = 1
    merge = _PEERS["merge"]
    made = merge._CALL.get("made") if merge is not None else None
    own = own and inp.dim() == 2 and inp.is_contiguous()
    tuning = unquant._mm_bf16 is not None and unquant._flashinfer_autotuner.is_tuning_mode
    if own and not isinstance(made, list) and not tuning:
        out, bias = _own_wqb(inp, indexer.wq_b.weight, call["rows"], call["n"]), None
    else:
        out, bias = type(indexer.wq_b).forward(indexer.wq_b, inp)
    call["ptr"] = out.data_ptr()
    return out, bias


def _rotate(x: torch.Tensor) -> torch.Tensor:
    
    call = _CALL["query"]
    _require(call["wqb"] and not call["rot"] and x.data_ptr() == call["ptr"]
             and tuple(x.shape) == (call["n"], HEADS, DIM) and x.dtype == torch.bfloat16 and x.is_contiguous(),
             f"rotate_activation of {x.dtype} {tuple(x.shape)}: not the query this call's wq_b made, or a second one")
    call["rot"] = 1
    out, y, s = _own_rotq(x, call["rows"])
    dsa_rotq_route._PENDING[:] = [(out, out._version, y, s)]
    return out

def _gated(indexer, inp, q_scale=None):
    
    call = _CALL["query"]
    _require(not call["gate"] and inp is call["x"], "a second head-gate call, or one on another input than x")
    call["gate"] = 1
    if not inp.is_contiguous():
        return (IndexerKPool._project_and_scale_head_gates(indexer, inp) if q_scale is None
                else IndexerKPool._get_logits_head_gate(indexer, inp, q_scale))
    if q_scale is None:
        return _own_project(indexer, inp, call["rows"])
    return _own_logits_gate(indexer, inp, q_scale, call["rows"], call["n"])


@contextlib.contextmanager
def _query(indexer, kind: str, rows: slice, n: int, x, q_lora, wqb_own: bool, gate_own: bool):
    
    rotate, gate = dsa_indexer_kpool.rotate_activation, _PEERS["gate"]
    _require(rotate is dsa_rotq_route._rotate, "dsa_rotq_route's rotate_activation is not set for the call")
    for bound, func in ((indexer.wq_b.forward, type(indexer.wq_b).forward),
                        (indexer._project_and_scale_head_gates, type(indexer)._project_and_scale_head_gates),
                        (indexer._get_logits_head_gate, type(indexer)._get_logits_head_gate)):
        _require(getattr(bound, "__func__", None) is func, f"{func.__name__} is already set on the module by another route")
    gates = (gate_own and gate is not None and IndexerKPool._project_and_scale_head_gates is gate._project
             and IndexerKPool._get_logits_head_gate is gate._gate)
    _CALL["query"] = dict(rows=rows, n=n, x=x, q_lora=q_lora, wqb=0, rot=0, gate=0, ptr=None)
    indexer.wq_b.forward = lambda inp: _wqb(indexer, inp, wqb_own)
    dsa_indexer_kpool.rotate_activation = _rotate
    if gates:
        indexer._project_and_scale_head_gates = lambda inp: _gated(indexer, inp)
        indexer._get_logits_head_gate = lambda inp, q_scale: _gated(indexer, inp, q_scale)
    try:
        yield
        call = _CALL["query"]
        _require(call["wqb"] and call["rot"] and not dsa_rotq_route._PENDING,
                 "the query's wq_b, rotation or quantize did not run as stock's verify / extend call makes them")
        _say(f"ours {kind} {wqb_own} {gates}", f"{kind} queries made on this rank's rows (first: {n} rows, rows "
                                              f"{rows.start}-{rows.stop}; wq_b {'here' if wqb_own else 'v156'}, gates "
                                              f"{'here' if gates else 'v156'})")
    finally:
        del indexer.wq_b.forward
        dsa_indexer_kpool.rotate_activation = rotate
        if gates:
            del indexer._project_and_scale_head_gates
            del indexer._get_logits_head_gate
        _CALL.pop("query", None)


@contextlib.contextmanager
def routed(node, forward_batch):
    
    _require(id(node) in _NODES, "a call of a module this route never prepared")
    indexer, mode = _NODES[id(node)], forward_batch.forward_mode
    if indexer is not None and (mode.is_target_verify() or mode.is_draft_extend_v2()):
        _require(idx_nextn_route._split is _STOCK_SPLIT and not _CALL, "another route holds idx_nextn_route._split")
        indexer._forward_cuda_target_verify = lambda **kw: _verify_entry(indexer, **kw)
        idx_nextn_route._split = _split
        try:
            yield
        finally:
            del indexer._forward_cuda_target_verify
            idx_nextn_route._split = _STOCK_SPLIT
            _CALL.clear()
    else:
        yield
