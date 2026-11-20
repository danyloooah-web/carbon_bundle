"""kda_track_route.py: r60 RIDERS KTG = KDA-TRACK-GLUE (11-20) -- the KDA prefix-track glue left on the v32 king's TP2
eager prefill path as single Triton row-copy launches. BITWISE (a word move), Triton only, no native unit.

The king's kda_once_route installs, for an eager extend call only, its own _track_mamba_state_extend (stock's function on
int64-widened indices) and _convolve_prefill (stock's conv with has_initial_state made once a forward) on
KDAAttnBackend. Stock still pays, per KDA layer and per tracked call (hybrid_linear_attn_backend.py:1142-1167 and
kda_backend.py:1146-1154 of the 314bce7c engine):
  M1 TRACK-FUSE  ``ssm_states[h_dst] = h_track_buf[h_batch_src]`` and ``ssm_states[final_dst] = ssm_states[final_src]``:
                 a gather into a temp + an index_put each (2 MiB rows);
  M2 CONV-TRACK  ``conv[0][conv_states_mask_indices] = mixed_qkv[track_conv_indices]``: a gather + an index_put.
Here each is ONE launch when ONE row moves (n == 1: source and destination rows are disjoint or the same row, each word
read and written by the same lane, so stock's temp is not needed), same dtype, rows contiguous; anything else is
stock's statement. Ported from our bC1tFINALbEf1 P2 (layerar/node.py:10305-10352, 10500-10740), served there in-engine.

How (nothing stock is edited at prepare; every swap is undone on exit):
  * prepare (the first PREFILL KDA node; kda_track_route sits right after kda_once_route in ACTIVE) rebinds two module
    globals of kda_once_route that its routed() installs on the class at call time: _track_mamba_state_extend ->
    _track (stock's function body statement for statement on kda_once's _widened(meta), the h / final copies behind
    _rows) and _convolve_prefill -> _convolve (the M2 snapshot at stock's program point, then kda_once's own
    _convolve_prefill unchanged).
  * M2's vehicle: while kda_once's routed is live (KDAAttnBackend._convolve_prefill is _convolve), this route's routed()
    sets KDAAttnBackend.forward_extend to _forward_extend for the layer call: on an M2 call (one tracked request) it
    hands stock's forward_extend a shallow copy of forward_metadata whose has_mamba_track_mask is False (stock skips
    the snapshot statement; every other read before _convolve_prefill is the same object), and _convolve puts the live
    metadata back FIRST, then takes the snapshot (one launch, or stock's statement), then kda_once's conv.
Safety: stock's two functions are pinned by the sha256 of their source text (PIN_TRACK / PIN_EXTEND, the 314bce7c
engine = imgsrc 93a5b07); kda_once_route's globals must be the king's own (apply_ktg.py also pins the file by content);
any drift -> "kda-track: OFF" with kda_once_route untouched (or just that switch off). Proofs: the first eager call of
each form ((form, dtype, index dtypes, same tensor)) runs our launch on the destination row, saves it, restores the
row, then stock's statement (stock's bytes are live), compares integer words, and votes over the TP group
(all_gather_object, CPU group): a miss or error on any rank -> that switch OFF on every rank for the process. Triggers
are forward-metadata facts (rank-uniform). Under capture, decode or draft-extend: kda_once's path exactly.
Census (WARNING; this logger keeps WARNING under the arena's log_level error): "kda-track: ..."."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import logging
import time

import torch
import triton
import triton.language as tl
from sglang.srt.distributed import parallel_state
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.runtime_context import get_disagg

from glm53f_core import kda_once_route

NODE = "kda"
TRACK_FUSE = True    # M1: the track row copies as one launch each (False = kda_once's path for them)
CONV_TRACK = True    # M2: the conv-window snapshot as one launch (False = stock's statement)
ROW_BLOCK, ROW_WARPS = 4096, 8   # M1: words a program, warps (a copy: any blocking is bitwise)
CT_BLOCK, CT_WARPS = 2048, 4     # M2: elements a program, warps
# sha256 of inspect.getsource() of the stock functions this route replicates / steers (engine 314bce7c, imgsrc 93a5b07)
PIN_TRACK = "bd64a34eda887c62954e8c266f8c81ff7c8d8daeb2c83384b25d3413eec93271"   # MambaAttnBackendBase._track_mamba_state_extend
PIN_EXTEND = "26a6020c7c267a1178e8033db4738f3fde2e50a34989fa7d1b45e93dd0fe996a"  # KDAAttnBackend.forward_extend
LOG_AT = (1, 64, 4096)

_BACKEND = kda_backend.KDAAttnBackend
_STOCK_EXTEND = _BACKEND.forward_extend
_ST = {"done": False, "tf_on": False, "ct_on": False, "tf": set(), "tf_off": None, "ct": set(), "ct_off": None,
       "pend": [], "warm": set(), "said": set(), "king_track": None, "king_conv": None, "n_tf": 0, "n_ct": 0}
_CELL: list = []
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _say(msg: str) -> None:
    if msg not in _ST["said"]:
        _ST["said"].add(msg)
        _log.warning("%s", msg)


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no CUDA state: treat as a capture (stock's path)
        return True


def _dev(t) -> bool:
    """KTG serves CUDA tensors only (one seam: the CPU suite runs the kernels under minitriton)."""
    return t.is_cuda


def _words(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32) if t.element_size() == 4 else t


def _rowc(t) -> bool:
    """True iff every row t[i] is one contiguous run (rows may be strided)."""
    want, s = [], 1
    for d in reversed(t.shape[1:]):
        want.append(s)
        s *= d
    return t.dim() >= 1 and tuple(t.stride()[1:]) == tuple(reversed(want))


def _idx(t, n) -> bool:
    return torch.is_tensor(t) and t.dim() == 1 and t.numel() == n and t.dtype in (torch.int32, torch.int64)


def _src_sha(fn) -> str:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    except Exception as e:  # noqa: BLE001 - no source: the pin cannot hold
        return f"unreadable ({type(e).__name__})"


@triton.jit(do_not_specialize=["n_src", "n_dst"])
def _kt_rowcopy_kernel(src, dst, src_idx, dst_idx, src_rs, dst_rs, n_src, n_dst, ROW, BLOCK: tl.constexpr):
    # M1: dst[dst_idx[i]] = src[src_idx[i]] (whole rows; a word move). Negative indices wrap as ATen's; an index out of
    # range writes nothing (ATen would raise a device assert there).
    i = tl.program_id(0)
    s = tl.load(src_idx + i).to(tl.int64)
    d = tl.load(dst_idx + i).to(tl.int64)
    s = tl.where(s < 0, s + n_src, s)
    d = tl.where(d < 0, d + n_dst, d)
    ok = (s >= 0) & (s < n_src) & (d >= 0) & (d < n_dst)
    offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = (offs < ROW) & ok
    v = tl.load(src + s * src_rs + offs, mask=m)
    tl.store(dst + d * dst_rs + offs, v, mask=m)


def _rowcopy(dst_t, dst_idx, src_t, src_idx) -> None:
    """M1 launch: dst_t[dst_idx] = src_t[src_idx] for 1-D index tensors of n rows (callers: n == 1)."""
    n = src_idx.numel()
    if n == 0:
        return
    row = src_t.numel() // max(1, src_t.shape[0])
    _kt_rowcopy_kernel[(n, triton.cdiv(row, ROW_BLOCK))](src_t, dst_t, src_idx, dst_idx, src_t.stride(0),
                                                        dst_t.stride(0), src_t.shape[0], dst_t.shape[0], row,
                                                        BLOCK=ROW_BLOCK, num_warps=ROW_WARPS)


@triton.jit(do_not_specialize=["n_x", "n_slots"])
def _kt_convtrack_kernel(x, x_rs, tok_idx, conv, slot_idx, c_ss, c_rs, n_x, n_slots, DIM, R: tl.constexpr,
                         BLOCK: tl.constexpr):
    # M2: conv[slot_idx[i], r, :] = x[tok_idx[i, r], :] (rows a word move; x a row-strided view).
    p = tl.program_id(0)
    i = p // R
    r = p % R
    t = tl.load(tok_idx + p).to(tl.int64)
    s = tl.load(slot_idx + i).to(tl.int64)
    t = tl.where(t < 0, t + n_x, t)
    s = tl.where(s < 0, s + n_slots, s)
    ok = (t >= 0) & (t < n_x) & (s >= 0) & (s < n_slots)
    offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = (offs < DIM) & ok
    v = tl.load(x + t * x_rs + offs, mask=m)
    tl.store(conv + s * c_ss + r * c_rs + offs, v, mask=m)


def _convtrack(conv0, slot_idx, x, tok_idx) -> None:
    """M2 launch: conv0[slot_idx] = x[tok_idx] for slot_idx [n], tok_idx [n, R] contiguous, x [T, DIM] unit-stride rows."""
    n, r = tok_idx.shape
    if n == 0:
        return
    dim = x.shape[1]
    _kt_convtrack_kernel[(n * r, triton.cdiv(dim, CT_BLOCK))](x, x.stride(0), tok_idx, conv0, slot_idx, conv0.stride(0),
                                                             conv0.stride(1), x.shape[0], conv0.shape[0], dim, R=r,
                                                             BLOCK=CT_BLOCK, num_warps=CT_WARPS)


def _vote(why) -> list:
    """Every TP rank's verdict (None = pass) through the coordinator's all_gather_object (CPU group): the failing
    (rank, reason) pairs, the same list on every rank."""
    tp = parallel_state.get_tp_group()
    got = tp.all_gather_object(why) if tp.world_size > 1 else [why]
    return [(r, w) for r, w in enumerate(got) if w is not None]


def _rows_why(dst_t, dst_idx, src_t, src_idx):
    """None if M1 may serve dst_t[dst_idx] = src_t[src_idx] (pure Python facts: shapes, dtypes, strides)."""
    if not (torch.is_tensor(dst_t) and torch.is_tensor(src_t) and _idx(dst_idx, 1) and _idx(src_idx, 1)):
        return "not one row"
    if not (dst_t.dtype == src_t.dtype and dst_t.dim() == src_t.dim() >= 2 and dst_t.shape[1:] == src_t.shape[1:]
            and dst_t.shape[0] > 0 and src_t.shape[0] > 0 and _dev(dst_t) and src_t.device == dst_t.device
            and dst_idx.device == dst_t.device and src_idx.device == dst_t.device):
        return "dtype / shape / device"
    if not (_rowc(dst_t) and _rowc(src_t)):
        return "rows not contiguous"
    return None


def _rows(form, dst_t, dst_idx, src_t, src_idx) -> bool:
    """M1: True = dst_t[dst_idx] = src_t[src_idx] done by one launch (a proved form); False = the caller runs stock's
    statement (not served, or this form's proof call: our launch's row saved, the row restored, then stock's statement;
    _settle compares and votes)."""
    if _ST["tf_off"] is not None:
        return False
    why = _rows_why(dst_t, dst_idx, src_t, src_idx)
    if why is not None:
        _say(f"kda-track track-fuse: {form}: stock's statement ({why})")
        return False
    key = (form, dst_t.dtype, dst_idx.dtype, src_idx.dtype, dst_t is src_t)
    if key in _ST["tf"]:
        _rowcopy(dst_t, dst_idx, src_t, src_idx)
        _ST["n_tf"] += 1
        if _ST["n_tf"] in LOG_AT:
            _log.warning("kda-track: served track-fuse x%d, conv-track x%d", _ST["n_tf"], _ST["n_ct"])
        return True
    keep = dst_t[dst_idx].clone()
    try:
        _rowcopy(dst_t, dst_idx, src_t, src_idx)
        got = dst_t[dst_idx].clone()
    except Exception as e:  # noqa: BLE001 - a compile / launch refusal: voted OFF in _settle
        got = e
    dst_t[dst_idx] = keep
    _ST["pend"].append(("tf", key, dst_t, dst_idx, got))
    return False


def _ct_why(conv0, slot_idx, x, tok_idx):
    """None if M2 may serve conv0[slot_idx] = x[tok_idx] (pure Python facts)."""
    if not (torch.is_tensor(conv0) and torch.is_tensor(x) and _idx(slot_idx, 1) and torch.is_tensor(tok_idx)
            and tok_idx.dim() == 2 and tok_idx.shape[0] == 1 and tok_idx.dtype in (torch.int32, torch.int64)
            and tok_idx.is_contiguous()):
        return "not one row"
    if not (conv0.dim() == 3 and x.dim() == 2 and conv0.shape[1] == tok_idx.shape[1] and conv0.shape[2] == x.shape[1]
            and conv0.stride(2) == 1 and x.stride(1) == 1 and conv0.dtype == x.dtype and x.shape[0] > 0
            and _dev(conv0) and x.device == conv0.device and slot_idx.device == conv0.device
            and tok_idx.device == conv0.device):
        return "dtype / shape / layout"
    return None


def _ct(conv0, slot_idx, x, tok_idx) -> bool:
    """M2: True = conv0[slot_idx] = x[tok_idx] done by one launch (a proved form); False = the caller runs stock's
    statement (as _rows: the proof call saves our row, restores it, and lets stock's statement run)."""
    if _ST["ct_off"] is not None:
        return False
    why = _ct_why(conv0, slot_idx, x, tok_idx)
    if why is not None:
        _say(f"kda-track conv-track: stock's statement ({why})")
        return False
    key = ("conv", conv0.dtype, slot_idx.dtype, tok_idx.dtype)
    if key in _ST["ct"]:
        _convtrack(conv0, slot_idx, x, tok_idx)
        _ST["n_ct"] += 1
        if _ST["n_ct"] in LOG_AT:
            _log.warning("kda-track: served track-fuse x%d, conv-track x%d", _ST["n_tf"], _ST["n_ct"])
        return True
    keep = conv0[slot_idx].clone()
    try:
        _convtrack(conv0, slot_idx, x, tok_idx)
        got = conv0[slot_idx].clone()
    except Exception as e:  # noqa: BLE001 - a compile / launch refusal: voted OFF in _settle
        got = e
    conv0[slot_idx] = keep
    _ST["pend"].append(("ct", key, conv0, slot_idx, got))
    return False


def _settle() -> None:
    """The pending proofs of this call (stock's statement has run): our saved row vs the live row, integer words;
    verdicts voted over the TP group (every rank holds the same pending list: the triggers are rank-uniform)."""
    while _ST["pend"]:
        kind, key, dst_t, dst_idx, got = _ST["pend"].pop(0)
        why = None
        try:
            if isinstance(got, Exception):
                raise got
            want = dst_t[dst_idx]
            if not (got.shape == want.shape and got.dtype == want.dtype
                    and torch.equal(_words(got), _words(want))):
                why = f"{key[0]}: our launch's row differs from stock's statement"
        except Exception as e:  # noqa: BLE001 - voted below; stock's bytes are live
            why = f"{type(e).__name__}: {str(e)[:200]}"
        bad = _vote(why)
        name = "track-fuse" if kind == "tf" else "conv-track"
        if bad:
            _ST[kind + "_off"] = f"rank {bad[0][0]}: {bad[0][1]}"
            _log.warning("kda-track %s: OFF on every rank (stock's statements): %s", name, bad[:2])
        else:
            _ST[kind].add(key)
            _log.warning("kda-track %s: %s (%s, index %s / %s) one launch == stock's statement bitwise on every rank: "
                         "on", name, key[0], key[1], key[2], key[3])


def _track(self, forward_batch, h, ssm_states, forward_metadata, track_states=None, *, h_track_buf=None):
    """kda_once_route._track_mamba_state_extend (M1): stock's MambaAttnBackendBase._track_mamba_state_extend
    (hybrid_linear_attn_backend.py:1123-1167, PIN_TRACK) statement for statement on kda_once's _widened(meta), the
    h-snapshot and final-state copies behind _rows (one launch when served, else stock's statement); this call's
    pending proofs settled at the end."""
    if _ST["tf_off"] is not None or _capturing():
        return _ST["king_track"](self, forward_batch, h, ssm_states, forward_metadata, track_states,
                                 h_track_buf=h_track_buf)
    meta = kda_once_route._widened(forward_metadata)
    try:
        if meta.has_mamba_track_mask:
            if meta.track_ssm_h_src.numel() > 0:
                if h_track_buf is not None:
                    if not _rows("h", ssm_states, meta.track_ssm_h_dst, h_track_buf, meta.track_ssm_h_batch_src):
                        ssm_states[meta.track_ssm_h_dst] = h_track_buf[
                            meta.track_ssm_h_batch_src
                        ].to(ssm_states.dtype, copy=False)
                else:
                    assert h is not None
                    h = h.squeeze(0)
                    ssm_states[meta.track_ssm_h_dst] = h[
                        meta.track_ssm_h_src
                    ].to(ssm_states.dtype, copy=False)
            if (
                meta.track_ssm_recompute_dst is not None
                and meta.track_ssm_recompute_dst.numel() > 0
            ):
                assert track_states is not None
                ssm_states[meta.track_ssm_recompute_dst] = (
                    track_states.squeeze(0).to(ssm_states.dtype, copy=False)
                )
            if meta.track_ssm_final_src.numel() > 0:
                if not _rows("final", ssm_states, meta.track_ssm_final_dst, ssm_states, meta.track_ssm_final_src):
                    ssm_states[meta.track_ssm_final_dst] = ssm_states[
                        meta.track_ssm_final_src
                    ]
    except BaseException:
        del _ST["pend"][:]
        raise
    _settle()


def _ct_takes(self, fm, forward_batch) -> bool:
    """M2's per-call predicate (forward metadata facts only, rank-uniform): a normal eager extend of one tracked
    request's conv window (stock's forward_extend reaches the snapshot statement, then _convolve_prefill)."""
    if (_ST["ct_off"] is not None or forward_batch is None or _capturing()
            or forward_batch.forward_mode.is_target_verify()
            or type(self) is not _BACKEND or getattr(self, "prefill_graph_metadata", None) is not None
            or getattr(self._convolve_prefill, "__func__", None) is not _convolve
            or not getattr(fm, "has_mamba_track_mask", False)):
        return False
    mi, ti = getattr(fm, "conv_states_mask_indices", None), getattr(fm, "track_conv_indices", None)
    return (torch.is_tensor(mi) and torch.is_tensor(ti) and mi.dim() == 1 and mi.numel() == 1 and ti.dim() == 2
            and ti.shape[0] == 1)


def _forward_extend(self, *args, **kwargs):
    """KDAAttnBackend.forward_extend while this route's routed() is live (M2's vehicle): on an M2 call stock's
    forward_extend runs on a forward_metadata copy whose has_mamba_track_mask is False until _convolve puts the live
    one back; every other call is stock's exactly."""
    fm = self.forward_metadata
    forward_batch = kwargs["forward_batch"] if "forward_batch" in kwargs else (args[1] if len(args) > 1 else None)
    if _CELL or not _ct_takes(self, fm, forward_batch):
        return _STOCK_EXTEND(self, *args, **kwargs)
    fm2 = copy.copy(fm)
    fm2.has_mamba_track_mask = False
    _CELL[:] = [self, fm]
    self.forward_metadata = fm2
    try:
        out = _STOCK_EXTEND(self, *args, **kwargs)
    except BaseException:
        if _CELL:
            self.forward_metadata = fm
            del _CELL[:]
        raise
    if _CELL:
        # cannot happen on the pinned source (PIN_EXTEND: the snapshot statement is followed by _convolve_prefill)
        self.forward_metadata = fm
        del _CELL[:]
        _ST["ct_off"] = "forward_extend returned without calling _convolve_prefill"
        raise RuntimeError("kda-track conv-track: forward_extend returned without calling _convolve_prefill on an M2 "
                           "call (the conv snapshot was skipped): engine drift past PIN_EXTEND")
    return out


def _convolve(self, layer, forward_batch, mixed_qkv, conv_states, cache_indices):
    """kda_once_route._convolve_prefill (M2): on an M2 call the live forward_metadata back FIRST, the snapshot at
    stock's program point (kda_backend.py:1146-1154, right before its _convolve_prefill call) behind _ct, then
    kda_once's own _convolve_prefill exactly; every other call is kda_once's."""
    if _CELL and _CELL[0] is self:
        fm = _CELL[1]
        del _CELL[:]
        self.forward_metadata = fm
        conv0 = conv_states.transpose(-1, -2)        # == mamba_cache_params.conv[0] (forward_extend's conv_states view)
        try:
            if not _ct(conv0, fm.conv_states_mask_indices, mixed_qkv, fm.track_conv_indices):
                conv0[fm.conv_states_mask_indices] = mixed_qkv[fm.track_conv_indices]
        except BaseException:
            del _ST["pend"][:]
            raise
        _settle()
    return _ST["king_conv"](self, layer, forward_batch, mixed_qkv, conv_states, cache_indices)


def _drift() -> tuple:
    """(why the whole route is off, why M1 is off, why M2 is off): kda_once_route / stock not the forms this route
    was written for (empty lists = they are)."""
    k, why, tf, ct = kda_once_route, [], [], []
    for name, fn in (("_track_mamba_state_extend", getattr(k, "_track_mamba_state_extend", None)),
                     ("_convolve_prefill", getattr(k, "_convolve_prefill", None))):
        if not (callable(fn) and getattr(fn, "__module__", None) == k.__name__
                and getattr(fn, "__name__", None) == name):
            why.append(f"kda_once_route.{name} is not the king's own")
    if getattr(k, "_BACKEND", None) is not _BACKEND:
        why.append("kda_once_route._BACKEND is not KDAAttnBackend")
    if not callable(getattr(k, "_widened", None)):
        why.append("kda_once_route has no _widened")
    live = getattr(k, "_LIVE", None)
    if not (isinstance(live, list) and live == [False]):
        why.append("kda_once_route is mid-call or has no _LIVE")
    stock_track = getattr(k, "_STOCK_TRACK", None)
    if stock_track is not _BACKEND._track_mamba_state_extend:
        tf.append("kda_once_route._STOCK_TRACK is not KDAAttnBackend's inherited _track_mamba_state_extend")
    elif _src_sha(stock_track) != PIN_TRACK:
        tf.append(f"stock _track_mamba_state_extend source {_src_sha(stock_track)[:12]} != {PIN_TRACK[:12]}")
    if _BACKEND.forward_extend is not _STOCK_EXTEND or getattr(_STOCK_EXTEND, "__qualname__", None) != \
            "KDAAttnBackend.forward_extend":
        ct.append("KDAAttnBackend.forward_extend is not stock's own")
    elif _src_sha(_STOCK_EXTEND) != PIN_EXTEND:
        ct.append(f"stock forward_extend source {_src_sha(_STOCK_EXTEND)[:12]} != {PIN_EXTEND[:12]}")
    if getattr(k, "_STOCK_CONVOLVE", None) is not _BACKEND._convolve_prefill:
        ct.append("kda_once_route._STOCK_CONVOLVE is not KDAAttnBackend._convolve_prefill")
    return why, tf, ct


def _warm(node) -> None:
    """The two kernels' JIT once a device at prepare (eager; never inside a capture): every index dtype pair the forms
    can carry, on scratch tensors whose strides fall in the serving specialization class (multiples of 16)."""
    w = getattr(getattr(node, "o_proj", None), "weight", None)
    if not torch.is_tensor(w) or not _dev(w) or _capturing():
        return
    dev = w.device
    if dev.index in _ST["warm"]:
        return
    _ST["warm"].add(dev.index)
    t0, idt = time.monotonic(), (torch.int32, torch.int64)
    try:
        if _ST["tf_on"]:
            pool = torch.zeros((2, 16), dtype=torch.float32, device=dev)
            for si in idt:
                for di in idt:
                    _rowcopy(pool, torch.zeros((1,), dtype=di, device=dev), pool, torch.ones((1,), dtype=si, device=dev))
        if _ST["ct_on"]:
            xw = torch.zeros((4, 32), dtype=torch.bfloat16, device=dev)
            cw = torch.zeros((2, 3, 32), dtype=torch.bfloat16, device=dev)
            for si in idt:
                for ti in idt:
                    _convtrack(cw, torch.zeros((1,), dtype=si, device=dev), xw, torch.zeros((1, 3), dtype=ti, device=dev))
        torch.cuda.synchronize(dev)
    except Exception as e:  # noqa: BLE001 - the first proof call compiles (and votes a refusal OFF)
        _say(f"kda-track: kernel compile at prepare failed ({type(e).__name__}: {str(e)[:200]}); first use compiles")
        return
    _say(f"kda-track: kernels compiled on cuda:{dev.index} in {time.monotonic() - t0:.2f} s (track-fuse "
         f"{_ST['tf_on']}, conv-track {_ST['ct_on']})")


def prepare(node) -> None:
    if not (TRACK_FUSE or CONV_TRACK):
        return
    if not _ST["done"]:
        _ST["done"] = True
        if get_disagg().disaggregation_mode == "decode":
            _say("kda-track: decode: nothing (the KDA track copies and the conv snapshot are eager prefill glue)")
            return
        why, tf, ct = _drift()
        if why:
            _log.warning("kda-track: OFF (kda_once_route untouched): %s", "; ".join(why))
            return
        _ST["king_track"], _ST["king_conv"] = kda_once_route._track_mamba_state_extend, kda_once_route._convolve_prefill
        if TRACK_FUSE and not tf:
            kda_once_route._track_mamba_state_extend = _track
            _ST["tf_on"] = True
        if CONV_TRACK and not ct:
            kda_once_route._convolve_prefill = _convolve
            _ST["ct_on"] = True
        off = [f"{n} off ({'; '.join(w)})" for n, w in (("track-fuse", tf), ("conv-track", ct)) if w]
        if not (_ST["tf_on"] or _ST["ct_on"]):
            _log.warning("kda-track: OFF (kda_once_route untouched): %s", "; ".join(off))
            return
        _log.warning("kda-track: on (first prepared KDA node %s): track-fuse %s, conv-track %s%s; per-form voted proofs "
                     "on the first eager call", type(node).__name__, _ST["tf_on"], _ST["ct_on"],
                     (" -- " + "; ".join(off)) if off else "")
    if _ST["tf_on"] or _ST["ct_on"]:
        _warm(node)


@contextlib.contextmanager
def routed(node, forward_batch):
    """M2's vehicle for a layer call: KDAAttnBackend.forward_extend = _forward_extend while kda_once's routed() has
    our _convolve installed (an eager extend call); restored on exit."""
    if not _ST["ct_on"] or _ST["ct_off"] is not None or _BACKEND._convolve_prefill is not _convolve:
        yield
        return
    if _BACKEND.forward_extend is not _STOCK_EXTEND:
        _say("kda-track conv-track: KDAAttnBackend.forward_extend is already replaced: stock's statement this call")
        yield
        return
    _BACKEND.forward_extend = _forward_extend
    try:
        yield
    finally:
        _BACKEND.forward_extend = _STOCK_EXTEND
        if _CELL:
            _CELL[0].forward_metadata = _CELL[1]
            del _CELL[:]
