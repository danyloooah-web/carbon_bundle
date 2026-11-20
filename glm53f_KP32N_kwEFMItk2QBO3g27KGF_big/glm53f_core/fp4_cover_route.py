"""fp4_cover_route.py: r60 FP4C (11-20) -- the v32 king's dormant NVFP4 prefill groups switched on. LOSSY (W4A4):
an audit (cacheon check) + the quality screen come before any submission.

  G2a = the DSA attention o_proj (the king's fp4_prefill_route SHAPES["G2a"]["o_proj"], own + peer halves served through
        dsa_xsplit_route at 8,193-16,384 rows) on the 7 DSA layers our AMAP audit admitted on the old line: SKIP {3, 19,
        23, 35}. The king ships GROUPS = ("G3",), so G2a is off on every layer.
  G3  = the KDA fused in-proj, now also on layers 1 and 2: the king's SKIP["G3"] = {0, 1, 2} -> {0} (our X12 audit:
        L1 / L2 margin 1.0).

How. This component is listed right BEFORE fp4_prefill_route in components.ACTIVE, so on every decoder layer it prepares
first (nodes.prepare_layer prepares a layer's components in ACTIVE order). At the first PREFILL layer's prepare it
rebinds the king module's globals GROUPS and SKIP (fp4_prefill_route._targets reads GROUPS, and its prepare reads SKIP,
at prepare time) and pins the G2a shape's tactic to the cutlass runner's default (-1) in _TACTIC, so the king's _sweep
returns early for that shape instead of choosing a non-default tactic (r57: a non-default cutlass tactic writing into
the E31 slot view gave NaN DSA_O output in-engine). The G3 shape keeps the king's sweep and probe.
Safety: every literal is checked against the v32 king's before anything changes (apply_fp4c.py also pins the file by
content); any drift, a G2a tactic already set, or a layer the king prepared before this component -> "fp4c: OFF" with
the reason and the king's groups untouched. No collective here (the king's own prepare all-gathers the G2a peer
halves); the same literals on every rank -> rank-uniform. Decode engines: nothing (the king's route is prefill-only).
Census (engine log, WARNING; this module's logger keeps WARNING under the arena's log_level error): "fp4c: ..."."""
from __future__ import annotations

import contextlib
import logging

from sglang.srt.runtime_context import get_disagg

from glm53f_core import fp4_prefill_route

NODE = "layer"   # the claimed node this component's prepare runs on (nodes.py): a decoder layer
FP4C = True      # False = this component does nothing (the base bundle's behaviour exactly)
G2A_SKIP = frozenset({3, 19, 23, 35})  # the DSA layers our AMAP audit did NOT admit for G2a
G3_SKIP = frozenset({0})               # G3 now also on layers 1 and 2
G2A_KEY = (4096, 8192)                 # fp4_prefill_route.SHAPES["G2a"]["o_proj"] = its _TACTIC key (n, k), both halves
G2A_TACTIC = -1                        # the cutlass FP4 runner's default config (CutlassFp4GemmRunner.forward's default)
KING_GROUPS = ("G3",)
KING_SKIP = {"G1": frozenset({3}), "G2a": frozenset({3}), "G3": frozenset({0, 1, 2}), "G4": frozenset({3})}

_ST = {"done": False, "on": False}
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _drift(fp) -> list:
    """Why the king's fp4_prefill_route is not the v32 king's form this component was written for (empty = it is)."""
    why = []
    if fp.GROUPS != KING_GROUPS:
        why.append(f"GROUPS {fp.GROUPS!r} != {KING_GROUPS!r}")
    if not isinstance(fp.SKIP, dict) or {k: frozenset(v) for k, v in fp.SKIP.items()} != KING_SKIP:
        why.append(f"SKIP {fp.SKIP!r} != the king's")
    shapes = fp.SHAPES.get("G2a", {}) if isinstance(fp.SHAPES, dict) else {}
    if tuple(shapes.get("o_proj", ())) != G2A_KEY:
        why.append(f"SHAPES G2a o_proj {shapes.get('o_proj')!r} != {G2A_KEY}")
    if not isinstance(fp._TACTIC, dict) or G2A_KEY in fp._TACTIC:
        why.append("the G2a tactic is already set")
    if fp._LAYERS:
        why.append(f"{len(fp._LAYERS)} layer(s) prepared before this component")
    return why


def prepare(layer) -> None:
    if _ST["done"] or not FP4C:
        return
    _ST["done"] = True
    if get_disagg().disaggregation_mode == "decode":
        _log.warning("fp4c: decode: nothing (the king's NVFP4 prefill groups are prefill-only)")
        return
    fp = fp4_prefill_route
    why = _drift(fp)
    if why:
        _log.warning("fp4c: OFF (the king's groups untouched): %s", "; ".join(why))
        return
    skip = dict(fp.SKIP)
    skip["G2a"] = G2A_SKIP
    skip["G3"] = G3_SKIP
    fp.SKIP = skip
    fp.GROUPS = KING_GROUPS + ("G2a",)
    fp._TACTIC[G2A_KEY] = G2A_TACTIC
    _ST["on"] = True
    _log.warning("fp4c: on (first prepared layer %s): GROUPS %s; SKIP G2a %s, G3 %s; G2a [%d x %d] tactic pinned %d (the "
                 "cutlass default)", getattr(layer, "layer_id", "?"), fp.GROUPS, sorted(G2A_SKIP), sorted(G3_SKIP),
                 G2A_KEY[0], G2A_KEY[1], G2A_TACTIC)


@contextlib.contextmanager
def routed(layer, forward_batch):
    """Nothing per call: the king's fp4_prefill_route serves the groups this component switched on."""
    yield
