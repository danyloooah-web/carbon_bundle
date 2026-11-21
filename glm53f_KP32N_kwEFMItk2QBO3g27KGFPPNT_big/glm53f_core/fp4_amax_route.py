"""r61 NaN-CONFINE (ours; byte-identical on finite inputs): the king's fp4_prefill_route quantizes the activations of
its "tensor"-scaled NVFP4 GEMMs (ACT_SCALE o_proj = G2a, which OUR fp4_cover_route switches on; down_proj = G1, off in our
trees) with ONE dynamic scale per call, amax = vector_norm(x, inf) over every row of the batch (fp4_prefill_route._fp4).
A single non-finite activation in any row therefore makes amax, the global scale and alpha non-finite for ALL rows of the
16K prefill batch -> every request in the batch is poisoned; chunked prefill chains the poison into the next request's
first chunk and the session radix cache keeps the poisoned KV (r61/garbage/ROOTCAUSE.md, H1: box12 042 = one prefill
engine all token 0 for the whole boot). The king ships G2a off, so in its tree one NaN costs one request.

This component (NODE "layer", prefill / audit engines; listed right AFTER fp4_cover_route) rebinds the king module's
global _fp4 ONCE, at the first prefill layer's prepare, to _fp4_confined: the same steps in the same order with the same
calls, except that a dynamic amax goes through amax = where(isfinite(raw), raw, last_good) (last_good = one device
element per NVFP4 copy, updated to the amax actually used). On finite inputs where() returns raw itself: the same scale,
the same quantized bytes, the same alpha, the same GEMM -> byte-identical. A non-finite raw takes the copy's last finite
amax, so the other rows of the batch keep a finite scale and only the poisoned request's own rows stay non-finite. The
first dynamic call of a copy has no history (last_good starts as that call's raw value). Cost: isfinite + where + copy_
(three one-element kernels) a dynamic call, no host sync. A device counter of non-finite amax values is read (one sync)
every SAY_EVERY dynamic calls and logged when it moved ("fp4n: ...").
Refusals (rank-uniform, nothing changed, "fp4n: OFF <why>"): the king module's _fp4 is not the function this mirrors
(its name, or a global it uses missing), a different _fp4 already installed, NANC False, the decode engine.
Census (engine log, WARNING; this module's logger keeps WARNING under the arena's log_level error): "fp4n: ..."."""
from __future__ import annotations

import contextlib
import logging

import torch
from sglang.srt.runtime_context import get_disagg

from glm53f_core import fp4_prefill_route

NODE = "layer"   # the claimed node this component's prepare runs on (nodes.py): a decoder layer
NANC = True      # False = this component does nothing (the base bundle's behaviour exactly)
SAY_EVERY = 4096  # dynamic calls between two reads of the non-finite counter (one host sync each)

_ST = {"done": False, "on": False, "king": None, "calls": 0, "seen": 0}
_LAST: dict = {}   # id(copy) -> one-element float32 device tensor: the copy's last amax used
_BAD: dict = {}    # device index -> one-element int32 device counter of non-finite raw amax values
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _confine(copy, raw: torch.Tensor) -> torch.Tensor:
    """The amax this call uses: raw itself when finite, else the copy's last one (no host sync)."""
    good = _LAST.get(id(copy))
    if good is None:
        good = _LAST[id(copy)] = raw.clone()
    ok = torch.isfinite(raw)
    bad = _BAD.get(raw.device.index)
    if bad is None:
        bad = _BAD[raw.device.index] = torch.zeros((1,), dtype=torch.int32, device=raw.device)
    bad.add_((~ok).to(torch.int32))
    amax = torch.where(ok, raw, good)
    good.copy_(amax)
    return amax


def _say(device) -> None:
    _ST["calls"] += 1
    if _ST["calls"] % SAY_EVERY or torch.cuda.is_current_stream_capturing():
        return
    bad = _BAD.get(device.index)
    n = int(bad.item()) if bad is not None else 0
    if n != _ST["seen"]:
        _ST["seen"] = n
        _log.warning("fp4n: %d non-finite activation amax value(s) confined so far on cuda:%s (each kept its batch's other "
                     "rows on the last finite scale)", n, device.index)


def _fp4_confined(copy, x: torch.Tensor, out: torch.Tensor, key=None) -> None:
    """fp4_prefill_route._fp4 with the dynamic amax confined (module docstring); every other step is the king's."""
    fp = fp4_prefill_route
    made = fp._CALL["quantized"].get(key) if key is not None else None
    if made is None:
        if copy.gs is not None:
            amax, gs = None, copy.gs
        else:
            raw = torch.linalg.vector_norm(x, float("inf")).float().clamp_min_(fp.TINY).reshape(1)
            amax = _confine(copy, raw)
            num = fp._FI["num"].get(x.device)
            if num is None:
                num = fp._FI["num"][x.device] = torch.full((1,), fp.SCALE_NUM, dtype=torch.float32, device=x.device)
            gs = torch.div(num, amax)
            _say(x.device)
        made = (*fp._quantize(x, gs), amax)
        if key is not None:
            fp._CALL["quantized"][key] = made
    xq, xs, amax = made
    fp._run(copy, xq, xs, copy.alpha if amax is None else amax * copy.coef, out, fp._TACTIC[(copy.n, copy.k)])


def _refusal(fp) -> str:
    king = fp._fp4
    if king is _fp4_confined:
        return ""
    if getattr(king, "__name__", None) != "_fp4" or getattr(king, "__module__", None) != fp.__name__:
        return f"the module's _fp4 is {king!r}, not the king's function"
    have = (("_CALL", hasattr(fp, "_CALL")), ("_FI", hasattr(fp, "_FI")), ("_quantize", hasattr(fp, "_quantize")),
            ("_run", hasattr(fp, "_run")), ("_TACTIC", hasattr(fp, "_TACTIC")), ("TINY", hasattr(fp, "TINY")),
            ("SCALE_NUM", hasattr(fp, "SCALE_NUM")))
    missing = [n for n, ok in have if not ok]
    if missing:
        return f"the module lacks {missing}"
    return ""


def prepare(layer) -> None:
    if _ST["done"] or not NANC:
        return
    _ST["done"] = True
    if get_disagg().disaggregation_mode == "decode":
        _log.warning("fp4n: decode: nothing (the king's NVFP4 prefill groups are prefill-only)")
        return
    fp = fp4_prefill_route
    why = _refusal(fp)
    if why:
        _log.warning("fp4n: OFF (the king's _fp4 untouched): %s", why)
        return
    if fp._fp4 is not _fp4_confined:
        _ST["king"] = fp._fp4
        fp._fp4 = _fp4_confined
    _ST["on"] = True
    _log.warning("fp4n: on (first prepared layer %s): the dynamic NVFP4 activation amax (ACT_SCALE 'tensor': G2a o_proj) is "
                 "confined to finite values per copy; byte-identical on finite inputs", getattr(layer, "layer_id", "?"))


@contextlib.contextmanager
def routed(layer, forward_batch):
    """Nothing per call: the king's fp4_prefill_route serves through the rebound _fp4."""
    yield
