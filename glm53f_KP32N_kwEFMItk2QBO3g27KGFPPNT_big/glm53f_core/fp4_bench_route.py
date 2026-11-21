"""fp4_bench_route.py: r60 FP4-BENCH (11-20) -- a BOOT-TIME, DIAGNOSTIC-ONLY timing of the king's NVFP4 prefill groups
(FP4C / FP4M levers: G1 shared expert, G3 KDA in-proj, G4 q_b / kv_b) against the FP8 DeepGEMM path they replace, on the
REAL weight copies of this engine, so one lifetime's log shows each FP4 group's effect per shape. It changes nothing that
serves: no global is swapped, no state the real calls read is written.

Why at boot: the engine stderr keeps only its first 16 MiB and the validator's CACHEON-EXECUTION-SUMMARY dumps fill it as
the timed loads begin (box11 036: kept log ends 05:44Z, first request 05:42:55Z) -> only boot-time lines survive.

How (NODE "layer", prefill engines only, listed right AFTER fp4_prefill_route so a layer's prepare here runs after the
king's): when the king has prepared FP4 copies for a layer (fp4_prefill_route._LAYERS[id(layer)]["copies"]), each new
(group, name) shape (once a process; o_proj / G2a skipped: its FP4 path may write the E31 stage slot) is timed at
BENCH_ROWS with fresh random bf16 inputs: FP8 = the king's saved stock linear (fp4_prefill_route._STOCK_LINEAR =
fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback on the copy's FP8 weight / scale) vs NVFP4 = the king's own
_linear on the same FP8 weight object (its _BY_FP8 lookup -> activation quantize + the swept cutlass tactic). CUDA
events, median of REPS after 2 warm calls; never under a capture. Census (WARNING), one line a shape:
"fp4-bench: <group> <name> [rows x k] -> n: FP8 a us / NVFP4 b us (x r) ..."."""
from __future__ import annotations

import contextlib
import logging
import statistics

import torch
from sglang.srt.runtime_context import get_disagg

from glm53f_core import fp4_prefill_route

NODE = "layer"
FP4BENCH = True
BENCH_ROWS = (8193, 16384)
REPS = 7
SKIP_NAMES = frozenset({"o_proj"})

_ST = {"done": set(), "off": None, "n": 0}
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _time_us(fn) -> float:
    fn()
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


def _bench(copy) -> str:
    fp = fp4_prefill_route
    weight, scale = copy.fp8, getattr(copy, "fp8_scale", None)
    entry = None
    for e in getattr(fp.fp8_linear_route, "_FP8", {}).values():
        if e[0] is weight:
            entry = e
            break
    if entry is None:
        return "no FP8 entry for this copy"
    scale = entry[1]
    parts = []
    for rows in BENCH_ROWS:
        x = torch.randn((rows, copy.k), device=weight.device, dtype=torch.bfloat16)
        f8 = _time_us(lambda: fp._STOCK_LINEAR(input=x, weight=weight, block_size=[128, 128], weight_scale=scale))
        f4 = _time_us(lambda: fp._linear(x, weight, [128, 128], scale))
        parts.append(f"[{rows} x {copy.k}] -> {copy.n}: FP8 {f8:.1f} us / NVFP4 {f4:.1f} us (x{f4 / max(f8, 1e-9):.3f})")
        del x
    return "; ".join(parts)


def prepare(layer) -> None:
    if not FP4BENCH or _ST["off"] is not None:
        return
    if get_disagg().disaggregation_mode == "decode":
        if _ST["off"] is None:
            _ST["off"] = "decode"
            _log.warning("fp4-bench: decode: nothing (the NVFP4 prefill groups are prefill-only)")
        return
    if torch.cuda.is_current_stream_capturing():
        return
    held = getattr(fp4_prefill_route, "_LAYERS", {}).get(id(layer))
    if not held:
        return
    for copy in held.get("copies", ()):
        key = (copy.group, copy.name, copy.n, copy.k)
        if key in _ST["done"] or copy.name in SKIP_NAMES:
            continue
        _ST["done"].add(key)
        try:
            msg = _bench(copy)
        except Exception as e:  # noqa: BLE001 - diagnostic only: report and go on
            msg = f"unavailable ({type(e).__name__}: {str(e)[:160]})"
        _log.warning("fp4-bench: %s %s (layer %s): %s", copy.group, copy.name, held.get("layer", "?"), msg)
    torch.cuda.synchronize()


@contextlib.contextmanager
def routed(layer, forward_batch):
    """Nothing per call (diagnostic only)."""
    yield
