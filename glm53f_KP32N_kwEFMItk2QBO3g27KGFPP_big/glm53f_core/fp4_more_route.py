"""fp4_more_route.py: r60 FP4M (11-20) -- two more of the v32 king's dormant NVFP4 prefill groups switched on, on top of
OUR FP4C (fp4_cover_route). LOSSY (W4A4): an audit (cacheon check) + the quality screen come before any submission.
r60 G4S27 + G1OFF form: only G4 is switched on (ADD), on 6 DSA layers (G4_SKIP adds 27); G1 stays on the king's FP8
copy (ADD's comment: its gain is hidden by the king's shared-expert side stream). The G1 text below is the FP4M design.

  G1 = the MoE layers' shared expert (gate_up_proj [2048 x 4096] + down_proj [4096 x 1024] a TP2 rank): the king's own
       SKIP {3} kept -> served on MoE layers 4..44.
  G4 = the DSA layers' q_b_proj [8192 x 1536] (own + peer halves through dsa_xsplit_route, as the king wrote it) and
       kv_b_proj [16384 x 512], on the same 7 DSA layers FP4C serves G2a on (SKIP {3, 19, 23, 35}: our AMAP-admitted set;
       the king's own SKIP is {3}).
Why it is wired: the king ships the FP8 prefill copies both groups require (fp8_shared_route: the shared expert's
gate_up / down; fp8_mla_route: q_b / kv_b; both over 8,193-32,768 rows, both in ACTIVE before fp4_prefill_route) and
the whole G1 / G4 path in fp4_prefill_route (_targets, _peers, the xsplit q_b halves, _sweep's tactic choice + its
0.25 BF16 probe); only GROUPS = ("G3",) keeps them off. Measured 11-20 (box11): the 224 / 160 TTFT queue is
PREFILL-bound, so prefill GEMM time is the lever.

How. Listed right AFTER fp4_cover_route and right BEFORE fp4_prefill_route in components.ACTIVE: on the first PREFILL
layer's prepare (after FP4C rebound GROUPS / SKIP) it appends G1 and G4 to the king module's GROUPS and sets their SKIP;
the king's _sweep then picks each new shape's tactic (G1 / G4 never write the E31 slot view that made FP4C pin G2a's
tactic). Safety: every literal is checked first (GROUPS must be FP4C's ("G3", "G2a"), SKIP FP4C's, the G1 / G4 SHAPES
the king's, no tactic set for their shapes, no layer prepared yet); any drift -> "fp4m: OFF" with the reason and the
groups untouched. No collective here; the same literals on every rank. Decode engines: nothing.
Census (engine log, WARNING): "fp4m: ..."."""
from __future__ import annotations

import contextlib
import logging

from sglang.srt.runtime_context import get_disagg

from glm53f_core import fp4_prefill_route

NODE = "layer"   # the claimed node this component's prepare runs on (nodes.py): a decoder layer
FP4M = True      # False = this component does nothing (the base bundle's behaviour exactly)
ADD = ("G4",)   # r60 G1OFF: G1 (the shared expert) NOT switched on: on prefill the king's moe_shared_overlap_route runs
                #   the shared expert on a side stream beside the routed experts (193-32,768 rows), so its NVFP4 speedup
                #   is off the critical path (box11: FP4C-only -> +FP4M cold -0.35..-0.47 % = G4's share alone) while it
                #   adds W4A4 error on 41 MoE layers; down_proj NVFP4 is even slower than FP8 (042 FP4-BENCH x1.02-1.24)
G1_SKIP = frozenset({3})                # the king's own
G4_SKIP = frozenset({3, 19, 23, 27, 35})  # FP4C's G2a set + 27 (r60 G4S27: chk-KP32NkwEFMIk2 layer 27 worst 0.845 with G4 on; G2a kept)
FP4C_GROUPS = ("G3", "G2a")
FP4C_SKIP = {"G1": frozenset({3}), "G2a": frozenset({3, 19, 23, 35}), "G3": frozenset({0}), "G4": frozenset({3})}
SHAPES = {"G1": {"gate_up_proj": (2048, 4096), "down_proj": (4096, 1024)},
          "G4": {"q_b_proj": (8192, 1536), "kv_b_proj": (16384, 512)}}

_ST = {"done": False, "on": False}
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _keys() -> set:
    """The _TACTIC keys (n, k) of the G1 / G4 shapes."""
    return {tuple(s) for g in ADD for s in SHAPES[g].values()}


def _drift(fp) -> list:
    why = []
    if fp.GROUPS != FP4C_GROUPS:
        why.append(f"GROUPS {fp.GROUPS!r} != FP4C's {FP4C_GROUPS!r} (fp4_cover_route must have switched G2a on first)")
    if not isinstance(fp.SKIP, dict) or {k: frozenset(v) for k, v in fp.SKIP.items()} != FP4C_SKIP:
        why.append(f"SKIP {fp.SKIP!r} != FP4C's")
    shapes = fp.SHAPES if isinstance(fp.SHAPES, dict) else {}
    for g in ADD:
        got = {n: tuple(v) for n, v in shapes.get(g, {}).items()}
        if got != SHAPES[g]:
            why.append(f"SHAPES {g} {got!r} != {SHAPES[g]!r}")
    if not isinstance(fp._TACTIC, dict) or _keys() & set(fp._TACTIC):
        why.append("a G1 / G4 tactic is already set")
    if fp._LAYERS:
        why.append(f"{len(fp._LAYERS)} layer(s) prepared before this component")
    return why


def prepare(layer) -> None:
    if _ST["done"] or not FP4M:
        return
    _ST["done"] = True
    if get_disagg().disaggregation_mode == "decode":
        _log.warning("fp4m: decode: nothing (the king's NVFP4 prefill groups are prefill-only)")
        return
    fp = fp4_prefill_route
    why = _drift(fp)
    if why:
        _log.warning("fp4m: OFF (the king's groups untouched): %s", "; ".join(why))
        return
    skip = dict(fp.SKIP)
    skip["G1"] = G1_SKIP
    skip["G4"] = G4_SKIP
    fp.SKIP = skip
    fp.GROUPS = FP4C_GROUPS + ADD
    _ST["on"] = True
    _log.warning("fp4m: on (first prepared layer %s): GROUPS %s; SKIP G1 %s, G4 %s; G1 / G4 tactics by the king's sweep",
                 getattr(layer, "layer_id", "?"), fp.GROUPS, sorted(G1_SKIP), sorted(G4_SKIP))


@contextlib.contextmanager
def routed(layer, forward_batch):
    """Nothing per call: the king's fp4_prefill_route serves the groups this component switched on."""
    yield
