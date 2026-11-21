"""r60 ERP2 (11-20): ids passed as a fresh aligned copy -> the prepare JIT covers every served call (no lazy Triton
specialization on an offset ids view inside the timed window; 033 showed a decaying early stall at load 160).
embed_repl_route.py: r60 RIDERS ERP = EMBED-REPL (11-20) -- a replicated TP2 embedding table for the v32 king's
prefill eager calls of >= MIN_IDS ids. BITWISE (the lookup of the one owning shard + an exact zero), Triton only.

Stock VocabParallelEmbedding.forward on a TP2 prefill rank: the own vocabulary half's rows (the other half's ids
zero-masked), then a tensor-parallel all-reduce of [ids, hidden] bf16 (134 MB at 16K ids): x + (+0) on every element.
Here each rank holds the peer's half too (gathered once at prepare: +per-rank vocab/2 x hidden bf16 = ~0.63 GB), and
one Triton launch writes bf16(fp32(own row or +0) + fp32(peer row or +0)) = the same words with no collective (x + 0 is
exact; -0 + +0 = +0 as the all-reduce gives). Port of our FINALL10s layerar/embrepl.py (CHK41 op 49: voted build, 2
voted checks, served bytes == stock on both ranks, 71.5 vs 557.2 us a 16K-id call), Python 3.10-clean.

How. nodes.py serves model.embed_tokens (op "embed": prepare_embed / forward_embed, the prepare_logits / forward_logits
pattern; components with NODE "embed"). prepare (the FIRST embedding module prepared in the process; a later one, such
as the MTP draft's, stays stock) checks the facts (VocabParallelEmbedding, unquantized bf16, TP2 exactly, the two
unpadded halves, no added vocabulary, the module's TP rank = the coordinator's) and VOTES them over the TP group, then
gathers the peer half in CHUNK_ROWS pieces (all_gather_into_tensor), checks it (int64 checksums of both copies, voted)
and compiles the kernel for int32 / int64 ids. routed() sets VocabParallelEmbedding.forward to _served for the op's call
(restored on exit): an eager call of >= MIN_IDS integer ids on CUDA with the attention-TP input not scattered is ours;
its first CHECKS calls compute ours AND stock's (stock's bytes are returned), compare words and vote; a miss / error on
any rank -> OFF on every rank. Every decision is a rank-uniform fact (the same ids on both TP ranks), so the two ranks
never split between our path (no collective) and stock's (an all-reduce). Decode engines (TP4) and captures: stock.
Census (WARNING; this logger keeps WARNING under the arena's log_level error): "embed-repl: ..."."""
from __future__ import annotations

import contextlib
import logging
import time
import types

import torch
import triton
import triton.language as tl
from sglang.srt.distributed import parallel_state
from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.runtime_context import get_disagg

NODE = "embed"
EMBED_REPL = True     # False = this component does nothing (the base bundle's behaviour exactly)
MIN_IDS = 8192        # eager calls of >= this many ids (a 16K cold chunk / an 8K+ warm extend)
CHECKS = 2            # voted bitwise checks (calls) before ours serves
CHUNK_ROWS = 8192     # peer-half gather piece (rows): 2 x 8192 x hidden bf16 scratch
SUM_ROWS = 2048       # checksum piece (rows)
BH, WARPS = 1024, 4   # kernel: hidden elements a program, warps
LOG_AT = (1, 34, 340)

_VPE = vpe.VocabParallelEmbedding
_STOCK_FORWARD = _VPE.forward
_STATES: dict = {}
_ST = {"first": None, "said": set()}
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    if msg not in _ST["said"]:
        _ST["said"].add(msg)
        _log.warning("%s", msg)


@triton.jit
def _er_kernel(ids_ptr, own_ptr, peer_ptr, out_ptr, own_lo, own_hi, peer_lo, peer_hi, H, BH: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BH + tl.arange(0, BH)
    m = c < H
    t = tl.load(ids_ptr + r).to(tl.int64)
    a_ok = (t >= own_lo) & (t < own_hi)
    b_ok = (t >= peer_lo) & (t < peer_hi)
    a = tl.load(own_ptr + (t - own_lo) * H + c, mask=m & a_ok, other=0.0).to(tl.float32)
    b = tl.load(peer_ptr + (t - peer_lo) * H + c, mask=m & b_ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + r * H + c, (a + b).to(tl.bfloat16), mask=m)


def _dev(t) -> bool:
    """ERP serves CUDA tensors only (one seam: the CPU suite runs the kernel under minitriton)."""
    return t.is_cuda


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no CUDA state: treat as a capture (stock's path)
        return True


def _vote(tp, why):
    """Every TP rank's verdict (None = pass), all_gather_object on the coordinator's CPU group: the first failing
    rank's reason, or None -- the same answer on every rank."""
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    return f"rank {bad[0][0]}: {bad[0][1]}" if bad else None


def _facts(module, tp):
    """(own_lo, own_hi, peer_lo, peer_hi, hidden) or the reason this module stays stock."""
    w = getattr(module, "weight", None)
    si = getattr(module, "shard_indices", None)
    qm = getattr(module, "quant_method", None)
    if type(module) is not _VPE:
        return f"module is {type(module).__name__}, not VocabParallelEmbedding"
    if not (torch.is_tensor(w) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous() and _dev(w)):
        return f"weight {getattr(w, 'dtype', None)} {tuple(getattr(w, 'shape', ()))}"
    if type(qm).__name__ != "UnquantizedEmbeddingMethod" or module._forward_hooks or module._forward_pre_hooks:
        return f"quant method {type(qm).__name__} / module hooks"
    rank = tp.rank_in_group
    mrank = getattr(module, "tp_rank", rank)
    if getattr(module, "tp_size", None) != 2 or mrank != rank:
        return f"module tp {mrank}/{getattr(module, 'tp_size', None)} != the coordinator's {rank}/2"
    if getattr(module, "use_attn_tp_group", False):
        if parallel_state.get_attn_tp_group().world_size != tp.world_size:
            return "the module sums over an attention-TP group other than the TP group"
    if getattr(module, "output_dtype", None) is not None:
        return f"output dtype {module.output_dtype}"
    per = w.shape[0]
    lo, hi = getattr(si, "org_vocab_start_index", None), getattr(si, "org_vocab_end_index", None)
    pad_lo, pad_hi = getattr(si, "padded_org_vocab_start_index", None), getattr(si, "padded_org_vocab_end_index", None)
    add_lo, add_hi = getattr(si, "added_vocab_start_index", None), getattr(si, "added_vocab_end_index", None)
    if not (lo == pad_lo == rank * per and hi == pad_hi == (rank + 1) * per and add_lo == add_hi
            and getattr(module, "org_vocab_size", None) == 2 * per
            and getattr(module, "num_embeddings_per_partition", per) == per):
        return (f"shards: rows {per}, org [{lo}, {hi}) padded [{pad_lo}, {pad_hi}) added [{add_lo}, {add_hi}) "
                f"org_vocab {getattr(module, 'org_vocab_size', None)}")
    peer = 1 - rank
    return (lo, hi, peer * per, (peer + 1) * per, w.shape[1])


def _sum(t):
    """(sum of the int16 words, position-weighted sum) of a bf16 tensor, in SUM_ROWS pieces (int64 scratch bounded)."""
    x = t.reshape(-1).view(torch.int16)
    step = SUM_ROWS * (t.shape[-1] if t.dim() > 1 else 1)
    s1, s2 = 0, 0
    for a in range(0, x.numel(), step):
        c = x[a:a + step].to(torch.int64)
        w = torch.arange(a, a + c.numel(), device=c.device, dtype=torch.int64) % 65521 + 1
        s1 += int(c.sum())
        s2 += int((c * w).sum())
        del c, w
    return s1, s2


def _off(st, why) -> None:
    st.off = why
    _say(f"embed-repl: OFF on every rank (stock's forward from here): {why}")


def _build(st) -> None:
    """The peer half gathered (voted allocation, chunked all_gather_into_tensor), both copies checksummed and voted."""
    w = st.module.weight
    per, h = w.shape
    me, dev = st.tp.rank_in_group, w.device
    why, peer, buf = None, None, None
    try:
        peer = torch.empty((per, h), dtype=w.dtype, device=dev)
        buf = torch.empty((2, min(CHUNK_ROWS, per), h), dtype=w.dtype, device=dev)
    except Exception as e:  # noqa: BLE001 - voted below
        why = f"{type(e).__name__}: {str(e)[:300]}"
    bad = _vote(st.tp, why)
    if bad is not None:
        return _off(st, f"build: {bad}")
    for a in range(0, per, CHUNK_ROWS):
        b = min(per, a + CHUNK_ROWS)
        out = buf.view(-1)[:2 * (b - a) * h].view(2, b - a, h)
        st.tp.all_gather_into_tensor(out.view(-1), w[a:b].reshape(-1))
        peer[a:b].copy_(out[1 - me])
    del buf
    sums = None
    try:
        sums = (_sum(w), _sum(peer))
    except Exception as e:  # noqa: BLE001 - voted below
        why = f"{type(e).__name__}: {str(e)[:300]}"
    votes = st.tp.all_gather_object((why, sums))
    bad = [(r, v[0]) for r, v in enumerate(votes) if v[0] is not None]
    if not bad:
        bad = [(r, "its copy of the peer half differs from the peer's own") for r, v in enumerate(votes)
               if v[1][1] != votes[1 - r][1][0]]
    if bad:
        return _off(st, f"build: rank {bad[0][0]} {bad[0][1]}")
    st.peer = peer
    try:
        free = int(torch.cuda.mem_get_info(dev)[0]) >> 20
    except Exception:  # noqa: BLE001 - census only
        free = -1
    _say(f"embed-repl: built tp{me}/2: peer half {per} x {h} gathered ({peer.numel() * 2 / 2 ** 20:.1f} MiB), "
         f"checksums agree; device free {free} MiB")


def _run(st, ids):
    own_lo, own_hi, peer_lo, peer_hi, h = st.facts
    t = ids.numel()
    out = torch.empty((t, h), dtype=torch.bfloat16, device=ids.device)
    if t:
        _er_kernel[(t, triton.cdiv(h, BH))](ids.clone(memory_format=torch.contiguous_format), st.module.weight, st.peer, out, own_lo, own_hi, peer_lo,
                                            peer_hi, h, BH=BH, num_warps=WARPS)
    return out


def _warm(st) -> None:
    """The kernel's JIT at prepare for int32 and int64 ids (one id of this rank's own half): no compile in a served
    call of the same specialization class."""
    dev, t0 = st.module.weight.device, time.monotonic()
    try:
        for dt in (torch.int64, torch.int32):
            _run(st, torch.full((1,), st.facts[0], dtype=dt, device=dev))
        torch.cuda.synchronize(dev)
    except Exception as e:  # noqa: BLE001 - the first check call compiles (and votes a refusal OFF)
        _say(f"embed-repl: kernel compile at prepare failed ({type(e).__name__}: {str(e)[:200]}); the first check "
             "compiles")
        return
    _say(f"embed-repl: kernel compiled in {time.monotonic() - t0:.2f} s (int64 / int32 ids)")


def _ids_of(args, kwargs):
    return args[0] if args else kwargs.get("input_")


def _takes(st, ids) -> bool:
    return (st.armed and st.off is None and torch.is_tensor(ids) and ids.dim() == 1 and ids.numel() >= MIN_IDS
            and not ids.is_floating_point() and ids.dtype != torch.bool and _dev(ids) and not _capturing()
            and not get_attn_tp_context().input_scattered)


def _check(st, module, args, kwargs, ids):
    why, got = None, None
    try:
        got = _run(st, ids)
    except Exception as e:  # noqa: BLE001 - voted below
        why = f"{type(e).__name__}: {str(e)[:300]}"
    want = _STOCK_FORWARD(module, *args, **kwargs)
    nword = -1
    if why is None:
        if not (torch.is_tensor(want) and want.shape == got.shape and want.dtype == got.dtype):
            why = f"stock output {getattr(want, 'dtype', None)} {tuple(getattr(want, 'shape', ()))}"
        else:
            nword = int((want.view(torch.int16) != got.view(torch.int16)).sum().item())
            if nword:
                why = f"words differ ({nword} of {want.numel()})"
    bad = _vote(st.tp, why)
    if bad is not None:
        _off(st, f"check ids={ids.numel()}: {bad}")
        return want
    st.checks += 1
    _say(f"embed-repl: check {st.checks}/{CHECKS} tp{st.tp.rank_in_group}/2 ids={ids.numel()}: words equal on every "
         f"rank (differing words {nword}); " + ("serving from the next call" if st.checks >= CHECKS else "checking"))
    return want


def _served(self, *args, **kwargs):
    """VocabParallelEmbedding.forward while routed() is live for the op's call."""
    st = _STATES.get(id(self))
    ids = _ids_of(args, kwargs)
    if st is None or st.module is not self or not _takes(st, ids):
        return _STOCK_FORWARD(self, *args, **kwargs)
    if st.checks < CHECKS:
        return _check(st, self, args, kwargs, ids)
    out = _run(st, ids)
    st.n += 1
    if st.n in LOG_AT or st.n % 4096 == 0:
        _log.warning("embed-repl: serving n=%d ids=%d tp%d/2", st.n, ids.numel(), st.tp.rank_in_group)
    return out


def prepare(module) -> None:
    if not EMBED_REPL or id(module) in _STATES:
        return
    st = types.SimpleNamespace(module=module, armed=False, off=None, facts=None, peer=None, checks=0, n=0, tp=None)
    _STATES[id(module)] = st
    if get_disagg().disaggregation_mode == "decode":
        _say("embed-repl: decode: stock (the replicated table serves TP2 prefill calls only)")
        return
    if _ST["first"] is not None:
        _say(f"embed-repl: a second embedding module ({type(module).__name__}) stays stock (one table copy a process)")
        return
    _ST["first"] = id(module)
    tp = parallel_state.get_tp_group()
    if tp.world_size != 2:
        _say(f"embed-repl: stock (TP world {tp.world_size}: the replicated table serves the TP2 engines only)")
        return
    try:
        got = _facts(module, tp)
    except Exception as e:  # noqa: BLE001 - voted below
        got = f"{type(e).__name__}: {str(e)[:300]}"
    bad = _vote(tp, got if isinstance(got, str) else None)
    if bad is not None:
        _say(f"embed-repl: stock on every rank ({bad})")
        return
    st.tp, st.facts = tp, got
    _build(st)
    if st.peer is None:
        return
    st.armed = True
    _warm(st)
    _say(f"embed-repl: armed tp{tp.rank_in_group}/2: eager calls of >= {MIN_IDS} ids from the replicated table (no "
         f"all-reduce) after {CHECKS} voted checks")


@contextlib.contextmanager
def routed(node, forward_batch):
    """VocabParallelEmbedding.forward = _served for the op's call when this module is armed; restored on exit."""
    st = _STATES.get(id(node))
    if st is None or not st.armed or st.off is not None:
        yield
        return
    if _VPE.forward is not _STOCK_FORWARD:
        _say("embed-repl: VocabParallelEmbedding.forward is already replaced: stock's call")
        yield
        return
    _VPE.forward = _served
    try:
        yield
    finally:
        _VPE.forward = _STOCK_FORWARD
