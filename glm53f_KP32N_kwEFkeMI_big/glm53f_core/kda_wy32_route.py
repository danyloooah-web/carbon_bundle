"""r59 KW (our lever on this family): the decode engine's KDA target-verify calls inside kda_v2_route's domain served by
kda_wy32.py (KDA-WY32 + WY-ONORM: the WY-form fp32 verify with the gated output norm in its epilogue), proved against
stock's kernel at boot and voted over the TP group; their path otherwise.

What runs without this component (SGLang 314bce7c + this family): the TP4 decode engine's TARGET_VERIFY forward runs, in
each of the 34 KDA layers, KDAAttnBackend._forward_target_verify (kda_verify_route.py's transcription) ->
_run_dspark_cutedsl_mtp, which imports kda_decode_mtp.fused_kda_decode_mtp_dspark at call time and calls it with the
four ReplaySSM rings and, under kda_verify_norm_route (C3), the layer's output norm (onorm_gate / onorm_weight /
onorm_eps); kda_v2_route sets that module name to its own function for the call (their v2 kernel inside its domain,
stock's kernel outside it).

What this component does: it is listed in components.ACTIVE right after kda_v2_route, so its route is entered inside
theirs and sets the same module name, for one target-verify call of a claimed linear-attention block, to ``_serve``,
which keeps the value it replaced (their function: "their path") and puts it back after the call. ``_serve`` hands a
call to kda_wy32 when kda_wy32.wy32_call accepts it (the fused output norm present, the rings without per-step
snapshots, an fp32 state pool, six tokens a request (1 bonus + 5 drafts), above one wave: H * N > SMs, the decay
product inside fp32's normal range, 16-byte-aligned tensors on one device), when its compile variant is proved and no
check has failed; every other call is their path's, said once a kind in the engine log. The decode engine's TP4 calls
(16 heads a rank) and the audit engine's TP2 verify calls (32 heads a rank) are both served, so the audit sees it.

Exactness contract: the ring stores and the conv-window stores are stock's statements (bitwise); each output differs
from stock's (+ its fused norm) by fp32 reassociation (~1e-6 relative), then stock's norm text and one BF16 rounding.
Proof (fail closed to their path, never to garbage):
  * boot proof, at the first EAGER call of each compile variant (kda_wy32's key: H, plan, eps, scale, lower bound,
    layouts), every TP rank at the same layer call (variants and domains are shape / dtype / layout facts): stock's
    kernel (_STOCK, captured at import) on the call's own inputs, then ours with its write set NaN-poisoned first ->
    rings + conv windows bitwise (when the call's slots and scratch rows are distinct: graph warm-up batches may repeat
    a slot, whose racing writers the seeded case covers), outputs by _cmp; then a seeded case on the call's own buffers
    (same request count and layouts = the same compile variants of both kernels): min(N, 16) requests on distinct random
    slots / scratch rows, request 1 and every request past them pads (slot -1), random fp32 state / conv state /
    activations, one NaN q channel, gates with NaN / +-inf / +-0 / +-40 / -100 entries; the touched state, conv, ring
    and window rows saved first and restored after: rings + windows bitwise, outputs by _cmp, the NaN propagated, pads
    zero;
  * then KW_LIVE live checks: the next eager served calls run stock beside ours the same way;
  * every verdict (an exception included: a JIT / launch error is a verdict) is all-gathered over the TP group (CPU
    group, eager only: never inside a capture); any miss on any rank -> OFF on every rank: their path re-serves that
    call (rewriting its rows) and every later call;
  * inside a CUDA-graph capture a variant is served ONLY once it is proved AND the KW_LIVE live checks have passed, all
    in eager calls (r59 review F5: SGLang's runner warms every graph size eagerly first; the proof is the first eager
    call of a variant and the live checks the next eager served calls -- the 2nd and 3rd KDA layers of the same warm-up
    forward -- so the first capture normally finds both done); a capture that meets a variant before that runs their
    path in that graph (logged once). A miss voted AFTER any capture served KW (a new variant's proof) raises on every
    rank (RuntimeError "glmf KDA WY32: OFF after a capture served KW ..."): captured graphs would keep replaying wy32 on
    a path the process has ruled out, so the process stops loudly instead.
_cmp (bf16 outputs): NaN where NaN and the same-signed inf where inf (payload free); over finite elements the differing
fraction <= KW_NEQ, the aggregate rel-L2 <= KW_AGG and every (row, head) rel-L2 <= KW_ROW (a zero reference row needs a
zero row). Why not 1e-4 a row: a ~1e-6 fp32 reorder flips ~1e-3 of the BF16 outputs by one ulp, and one flip of a
row's largest element is up to 2^-7 of it (C22's rule and literals; r59/kw/tests/t_emu_kw.py measures both kernels'
forms on the CPU and refuses every mutant there).
Switches: WY32 False = this component does nothing (the family's behaviour exactly); WY_ONORM False = the same (this
domain always carries the fused norm, which only the fold can serve). Census: "kw: on ...", "kw: OFF on every rank:
...", "kw: live check ...", "kw: served N" (Python-level launches: eager calls + graph captures).
"""

from __future__ import annotations

import contextlib
import logging

import torch
from sglang.kernels.ops.attention import kda_decode_mtp
from sglang.srt.distributed import parallel_state

from glm53f_core import kda_wy32

NODE = "kda"  # the claimed node this component's route is set at (nodes.py): a linear-attention layer's block
WY32 = True                  # False = this component does nothing (the family's behaviour exactly)
WY_ONORM = True              # the norm folded into WY32's epilogue (False = nothing served: see the docstring)
KW_LIVE = 2                  # live checks a process after the first proof (eager served calls, voted)
KW_NEQ = 1e-2                # _cmp: max fraction of differing finite bf16 outputs vs stock's
KW_AGG = 1e-3                # _cmp: max aggregate rel-L2 vs stock's
KW_ROW = 1e-2                # _cmp: max per-(row, head) rel-L2 (bf16: one 1-ulp flip <= 2^-7 = 7.8e-3)
KW_SEED = 20261119           # seeded case: the generator seed (+ the request count)
KW_SEED_REQS = 16            # seeded case: live requests (the rest of the call's N are pads)
_STOCK = kda_decode_mtp.fused_kda_decode_mtp_dspark
_RINGS = ("replayssm_rawv", "replayssm_rawk", "replayssm_g", "replayssm_beta")
_WINDOWS = ("intermediate_conv_q", "intermediate_conv_k", "intermediate_conv_v")
_SET: list = []              # the module name's value this route replaced for the call in flight (their path)
_KW = {"off": None, "proved": set(), "live": 0, "served": 0, "captured": 0}  # captured: Python-level launches in captures
_SAID: set = set()
# r59 K339 logger fix: the arena's log_level "error" (SGLang configure_logger: basicConfig(level=ERROR,
# force=True)) sets the root logger to ERROR; this module's census lines are WARNING
logging.getLogger(__name__).setLevel(logging.WARNING)


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA WY32: {why}")


def _say(tag: str, detail: str) -> None:
    """One log line a kind of event."""
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("kw: %s", detail)


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


def _sync(t) -> None:
    if t.is_cuda:
        torch.cuda.synchronize(t.device)


def prepare(node) -> None:
    """The name this route sets is stock's function, and the block's chain module is the 128-wide head form (fail at
    boot, not in a forward). Nothing compiles here: a variant compiles inside its proof."""
    if not (WY32 and WY_ONORM):
        return
    _require(callable(_STOCK) and kda_decode_mtp.fused_kda_decode_mtp_dspark is _STOCK,
             "kda_decode_mtp.fused_kda_decode_mtp_dspark is not stock's")
    layer = getattr(node, "attn", None)
    _require(layer is not None and (layer.head_q_dim, layer.head_k_dim, layer.head_v_dim) == (128, 128, 128),
             f"{type(node).__name__}'s chain module is not the verify kernel's 128-wide head form")


def _bits(a, b) -> bool:
    """Bitwise equality of two tensors of one dtype (NaN payloads included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    v = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(v), b.contiguous().view(v)))


def _cmp(got, want):
    """(ok, text) for two bf16 outputs of one call (the docstring's rule)."""
    if got.shape != want.shape:
        return False, f"shapes {tuple(got.shape)} vs {tuple(want.shape)}"
    g = got.reshape(-1, got.shape[-1]).double()
    w = want.reshape(-1, want.shape[-1]).double()
    gn, wn_, gi, wi = torch.isnan(g), torch.isnan(w), torch.isinf(g), torch.isinf(w)
    cls = bool(torch.equal(gn, wn_)) and bool(torch.equal(gi, wi)) and bool(torch.equal(g[gi], w[wi]))
    fin = ~(gn | wn_ | gi | wi)
    gz, wz = torch.where(fin, g, 0.0), torch.where(fin, w, 0.0)
    nfin = max(int(fin.sum()), 1)
    neq = float(((gz != wz) & fin).sum()) / nfin
    dn, wn = float((gz - wz).norm()), float(wz.norm())
    agg = dn / wn if wn > 0 else (0.0 if dn == 0 else float("inf"))
    rd, rn = (gz - wz).norm(dim=1), wz.norm(dim=1)
    row = torch.where(rn > 0, rd / rn.clamp_min(1e-300),
                      torch.where(rd > 0, torch.full_like(rd, float("inf")), torch.zeros_like(rd)))
    rmax = float(row.max()) if row.numel() else 0.0
    ok = cls and neq <= KW_NEQ and agg <= KW_AGG and rmax <= KW_ROW
    return ok, (f"neq {neq:.1e} agg {agg:.1e} row max {rmax:.1e} non-finite {int((~fin).sum())} "
                f"({'same classes' if cls else 'CLASSES DIFFER'})")


def _writes(kw):
    """The call's write set: (slots, scratch rows) of its live requests (slot >= 0; pads write nothing but their
    output rows) and whether both are distinct."""
    n = kw["cu_seqlens"].numel() - 1
    ci = kw["ssm_state_indices"][:n].long()
    live = ci >= 0
    slots, rows = ci[live], kw["intermediate_state_indices"][:n].long()[live]
    distinct = slots.unique().numel() == slots.numel() and rows.unique().numel() == rows.numel()
    return slots, rows, distinct


def _rows(kw, slots, rows):
    """Copies of every row the kernels write at these slots / scratch rows: the four rings' first six steps and the
    three conv windows' first six steps."""
    out = [kw[n].index_select(0, slots)[:, :, :kda_wy32.TOKENS].clone() for n in _RINGS]
    out += [kw[n].index_select(0, rows)[:, :kda_wy32.TOKENS].clone() for n in _WINDOWS]
    return out


def _poison(kw, slots, rows) -> None:
    """NaN into the kernels' write set (a skipped write cannot read back the other kernel's bytes)."""
    for n in _RINGS:
        kw[n][slots, :, :kda_wy32.TOKENS] = float("nan")
    for n in _WINDOWS:
        kw[n][rows, :kda_wy32.TOKENS] = float("nan")


def _leg(kw, call):
    """Stock's kernel, then ours (write set poisoned) on the same inputs: (ours' output, ok, text)."""
    slots, rows, distinct = _writes(kw)
    want = _STOCK(**kw)
    ref = _rows(kw, slots, rows)
    _poison(kw, slots, rows)
    got = kda_wy32.wy32_launch(call)
    now = _rows(kw, slots, rows)
    _sync(got)
    same = all(_bits(a, b) for a, b in zip(ref, now)) if distinct else None
    ok, txt = _cmp(got, want)
    rows_txt = {True: "bitwise", False: "DIFFER", None: "not compared (repeated slots / rows: racing writers)"}[same]
    return got, ok and same is not False, f"rings + windows {rows_txt}, out {txt}"


def _seeded(kw, call):
    """The proof's seeded case on the call's own buffers (the docstring): (ok, text)."""
    n_req = kw["cu_seqlens"].numel() - 1
    x_v, pool = kw["x_v"], kw["recurrent_state"]
    dev = x_v.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(KW_SEED + n_req)
    win = [kw[n] for n in _WINDOWS]
    n = min(n_req, KW_SEED_REQS, pool.shape[0], win[0].shape[0])
    slots = torch.randperm(pool.shape[0], generator=gen, device=dev)[:n]
    rows = torch.randperm(win[0].shape[0], generator=gen, device=dev)[:n]
    ci = torch.full_like(kw["ssm_state_indices"], -1)
    ci[:n] = slots.to(ci.dtype)
    if n >= 3:
        ci[1] = -1
    isi = kw["intermediate_state_indices"].clone()
    isi[:n] = rows.to(isi.dtype)
    live = ci[:n] >= 0
    ls, lr = slots[live], rows[live]
    convs = [kw[n] for n in ("cs_q", "cs_k", "cs_v")]
    rings = [kw[n] for n in _RINGS]
    keep = [pool[ls].clone()] + [c[ls].clone() for c in convs] + [r[ls].clone() for r in rings] + \
        [w[lr].clone() for w in win]

    def rnd(like, scale):
        t = torch.empty_strided(tuple(like.shape), tuple(like.stride()), dtype=like.dtype, device=dev)
        t.copy_(torch.randn(tuple(like.shape), generator=gen, device=dev).mul_(scale))
        return t

    try:
        pool[ls] = torch.randn(tuple(pool[ls].shape), generator=gen, device=dev).mul_(0.5)
        for c in convs:
            c[ls] = torch.randn(tuple(c[ls].shape), generator=gen, device=dev).to(c.dtype)
        skw = dict(kw, x_q=rnd(kw["x_q"], 1.0), x_k=rnd(kw["x_k"], 1.0), x_v=rnd(x_v, 1.0), g=rnd(kw["g"], 2.0),
                   beta=rnd(kw["beta"], 1.0), onorm_gate=rnd(kw["onorm_gate"], 3.0), ssm_state_indices=ci,
                   intermediate_state_indices=isi,
                   cu_seqlens=torch.arange(n_req + 1, device=dev, dtype=kw["cu_seqlens"].dtype) * kda_wy32.TOKENS)
        skw["x_q"][0, min(2, kda_wy32.TOKENS - 1), 0, 5] = float("nan")   # request 0, token 2: one q channel
        gate = skw["onorm_gate"]
        for j, v in enumerate((float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 40.0, -40.0, -100.0)):
            gate[0, j % gate.shape[1], j % gate.shape[2], 7 + 13 * j] = v
        why, scall = kda_wy32.wy32_call(**skw)
        if why is not None or scall[0] != call[0]:
            return False, f"seeded case: not the live call's variant ({why or 'another compile key'})"
        _poison(skw, ls, lr)
        want = _STOCK(**skw)
        ref = _rows(skw, ls, lr)
        _poison(skw, ls, lr)
        got = kda_wy32.wy32_launch(scall)
        now = _rows(skw, ls, lr)
        _sync(got)
        same = all(_bits(a, b) for a, b in zip(ref, now))
        ok, txt = _cmp(got, want)
        nan_seen = bool(torch.isnan(want[0, :kda_wy32.TOKENS].float()).any())
        pad_req = ci[:n_req] < 0
        pad_tok = pad_req.repeat_interleave(kda_wy32.TOKENS)
        pad_ok = all(bool((o[0][pad_tok].view(torch.int16) == 0).all()) for o in (got, want))
        n_pad = int(pad_req.sum())
    finally:
        pool[ls] = keep[0]
        for c, k in zip(convs, keep[1:4]):
            c[ls] = k
        for r, k in zip(rings, keep[4:8]):
            r[ls] = k
        for w, k in zip(win, keep[8:11]):
            w[lr] = k
    return same and ok and nan_seen and pad_ok, (
        f"seeded {n} of {n_req} requests ({n_pad} pads, NaN q, special gates): rings + windows "
        f"{'bitwise' if same else 'DIFFER'}, out {txt}, NaN propagated {nan_seen}, pads zero {pad_ok}")


def _vote(ok: bool, txt: str):
    """Every TP rank's verdict through the TP group's CPU all-gather: the failing (rank, text) pairs, the same list on
    every rank."""
    tp = parallel_state.get_tp_group()
    got = tp.all_gather_object((bool(ok), str(txt)[:400])) if tp.world_size > 1 else [(bool(ok), str(txt))]
    return [(r, v[1]) for r, v in enumerate(got) if not v[0]]


def _verdict(prev, kw, call, seeded: bool):
    """A proof (seeded) or a live check (not seeded) of this eager call, voted: ours' output, or their path's after OFF
    on every rank."""
    out, ok, txt = None, False, ""
    try:
        out, ok, txt = _leg(kw, call)
        if seeded:
            ok_s, txt_s = _seeded(kw, call)
            ok, txt = ok and ok_s, f"{txt}; {txt_s}"
    except Exception as exc:  # noqa: BLE001 - a JIT / launch error is a verdict (voted), never a silent fallback
        ok, txt = False, f"{type(exc).__name__}: {exc}"[:400]
    bad = _vote(ok, txt)
    n, H = call[3], call[4]
    if not bad:
        if seeded:
            _KW["proved"].add(call[0])
            logging.getLogger(__name__).warning("kw: on (variant %d of this process: H %d, %d requests, plan %s): %s",
                                                len(_KW["proved"]), H, n, call[8], txt)
        else:
            logging.getLogger(__name__).warning("kw: live check %d/%d (H %d, %d requests): %s", _KW["live"],
                                                KW_LIVE, H, n, txt)
        return _count(out)
    _KW["off"] = f"rank {bad[0][0]}: {bad[0][1]}"
    logging.getLogger(__name__).warning("kw: OFF on every rank: %s (H %d, %d requests); their path from here",
                                        _KW["off"], H, n)
    # r59 review F5: graphs captured with KW inside keep replaying it; the voted miss is the same on every rank, so every
    # rank stops here together instead of leaving those graphs on a path the process has ruled out
    _require(_KW["captured"] == 0, f"OFF after a capture served KW ({_KW['captured']} launches inside captures): "
                                   f"{_KW['off']}")
    return prev(**kw)


def _count(out):
    _KW["served"] += 1
    if _KW["served"] in (1, 34, 340) or _KW["served"] % 4096 == 0:
        logging.getLogger(__name__).warning("kw: served %d (Python-level launches: eager calls + graph captures)",
                                            _KW["served"])
    return out


def _serve(**kw):
    """kda_decode_mtp.fused_kda_decode_mtp_dspark's arguments and result during a routed call: ours for a proved
    variant inside the domain, their path (the value this route replaced) for everything else."""
    prev = _SET[-1]
    if _KW["off"] is not None:
        return prev(**kw)
    try:
        why, call = kda_wy32.wy32_call(**kw)
    except TypeError as exc:     # a keyword set kda_wy32 does not know: outside
        why, call = f"keywords ({exc})", None
    if why is not None:
        _say("out " + "".join(c for c in why if not c.isdigit())[:48], f"their path serves a verify call: {why}")
        return prev(**kw)
    if _capturing():
        if call[0] in _KW["proved"] and _KW["live"] >= KW_LIVE:
            _KW["captured"] += 1
            return _count(kda_wy32.wy32_launch(call))
        _say("capture", "a variant met inside a CUDA-graph capture before its proof and the live checks passed in eager "
                        "calls: their path in that graph")
        return prev(**kw)
    if call[0] not in _KW["proved"]:
        return _verdict(prev, kw, call, seeded=True)
    if _KW["live"] < KW_LIVE:
        _KW["live"] += 1
        return _verdict(prev, kw, call, seeded=False)
    return _count(kda_wy32.wy32_launch(call))


@contextlib.contextmanager
def routed(node, forward_batch):
    """Ours for one target-verify call of a claimed linear-attention block (in capture too), inside kda_v2_route's
    route; the replaced value again after it."""
    if not (WY32 and WY_ONORM) or not forward_batch.forward_mode.is_target_verify():
        yield
        return
    _require(not _SET, "a linear-attention block's verify call inside another's")
    _SET.append(kda_decode_mtp.fused_kda_decode_mtp_dspark)
    kda_decode_mtp.fused_kda_decode_mtp_dspark = _serve
    try:
        yield
    finally:
        kda_decode_mtp.fused_kda_decode_mtp_dspark = _SET.pop()
