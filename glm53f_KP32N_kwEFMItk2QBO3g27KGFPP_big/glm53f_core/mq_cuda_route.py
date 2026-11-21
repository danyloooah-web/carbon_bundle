"""mq_cuda_route.py: r60 MQ-CUDA (11-20) -- the PREFILL MoE input NVFP4 quantize served by FlashInfer's backend="cuda"
kernel instead of the backend="cute-dsl" one SGLang forces on sm100, ONLY where the two are BYTE-EQUAL and cuda is faster
(proven and timed at prepare on this GPU, voted across the TP group). BITWISE by construction of the gate.

Where. SGLang's standard FP4 MoE path (flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4, which the king's
moe_dsl_route intercepts downstream at trtllm_fp4_block_scale_moe) quantizes the MoE input through
flashinfer_trtllm.quantize_hidden_states_fp4 -> the module global flashinfer_trtllm.fp4_quantize (= fp4_utils.fp4_quantize,
whose backend fp4_utils._flashinfer_fp4_quantize_backend is "cute-dsl" on sm100), linear scale layout
(fp4_quantize(x, input_scale_quant, 16, False, False)). The king's own G3 quantize already runs backend "cuda"
(fp4_prefill_route.QUANT_BACKEND). This component (NODE "layer", prefill engines only) swaps that ONE module global for
_cuda_quant for the duration of a prefill layer call (restored in a finally; the king's swap pattern), so eager calls and
the king's captured MoE graphs (moe_graph_route) take the cuda kernel.

Gate (once a process, at the first MoE layer's prepare, eager, never under capture): the call form is pinned
(sha256 of quantize_hidden_states_fp4's source == PIN_QHS; the backend literal is "cute-dsl"; the module global is
fp4_utils's); PROBE: for rows in PROBE_ROWS and three input families (normal, outlier columns x64, zero rows) the stock
call and _cuda_quant are compared BYTE FOR BYTE (packed values and scales) with the layer's real global scale
(experts.w13_input_scale_quant); TIMING: CUDA events, median of TIME_REPS at TIME_ROWS for both backends (the per-lever
in-engine timing the big-arm plan logs); VOTE: every rank's (equal, stock us, cuda us, why) through all_gather_object on
the TP group: ON only if every rank is byte-equal and cuda <= (1 - MIN_GAIN) x stock on every rank. Census (WARNING):
"mq-cuda: on|OFF ... (cute-dsl X us vs cuda Y us at N rows, rank r)"."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import statistics

import torch
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.runtime_context import get_disagg

NODE = "layer"
MQCUDA = True          # False = this component does nothing
PROBE_ROWS = (8193, 12345, 16384)
TIME_ROWS, TIME_REPS = 16384, 9
MIN_GAIN = 0.03        # ON only if cuda is >= 3 % faster than cute-dsl on every rank
HIDDEN = 4096
PIN_QHS = "85b7f0532de1516b62e741de7d30457fc121c1676e5f0399eeb42e6b1e72e3b2"  # quantize_hidden_states_fp4 (314bce7c)

_ST = {"done": False, "on": False, "stock": None, "served": 0}
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _cuda_quant(input, global_scale=None, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True,
                is_sf_8x4_layout=False, enable_pdl=None):
    """fp4_utils.fp4_quantize's signature, served by FlashInfer's backend="cuda" kernel."""
    from flashinfer import fp4_quantize
    return fp4_quantize(input=input, global_scale=global_scale, sf_vec_size=sf_vec_size, sf_use_ue8m0=sf_use_ue8m0,
                        is_sf_swizzled_layout=is_sf_swizzled_layout, is_sf_8x4_layout=is_sf_8x4_layout,
                        enable_pdl=enable_pdl, backend="cuda")


def _src_sha(fn) -> str:
    try:
        return hashlib.sha256((inspect.getsource(fn).rstrip("\n") + "\n").encode()).hexdigest()
    except Exception as e:  # noqa: BLE001 - no source: the pin cannot hold
        return f"unreadable ({type(e).__name__})"


def _group():
    from sglang.srt.distributed import get_tp_group
    return get_tp_group()


def _families(rows: int, device) -> list:
    g = torch.Generator(device=device).manual_seed(20261120 + rows)
    x = torch.randn((rows, HIDDEN), generator=g, device=device, dtype=torch.float32)
    out = x.clone()
    out[:, ::97] *= 64.0                     # outlier columns (activation spikes)
    zero = x.clone()
    zero[: max(1, rows // 7)] = 0.0          # all-zero rows (a zero block max)
    return [x.to(torch.bfloat16), out.to(torch.bfloat16), zero.to(torch.bfloat16)]


def _equal(a, b) -> bool:
    if not (isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == len(b) == 2):
        return False
    return all(x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape)
               and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x, y in zip(a, b))


def _time_us(fn, *args) -> float:
    """Median CUDA-event time (us) of fn(*args) over TIME_REPS (two warm calls first)."""
    fn(*args)
    fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(TIME_REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(*args)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


def _probe(stock, gs, device) -> tuple:
    """(why or None, cases compared, stock us, cuda us) on this rank."""
    cases = 0
    for rows in PROBE_ROWS:
        for x in _families(rows, device):
            if not _equal(stock(x, gs, 16, False, False), _cuda_quant(x, gs, 16, False, False)):
                return f"{rows} rows: cuda != cute-dsl bytes (family {cases % 3})", cases, None, None
            cases += 1
    x = _families(TIME_ROWS, device)[0]
    t_stock = _time_us(stock, x, gs, 16, False, False)
    t_cuda = _time_us(_cuda_quant, x, gs, 16, False, False)
    return None, cases, t_stock, t_cuda


def prepare(layer) -> None:
    if _ST["done"] or not MQCUDA:
        return
    if get_disagg().disaggregation_mode == "decode":
        _ST["done"] = True
        _log.warning("mq-cuda: decode: nothing (the decode MoE input quantize is the king's fused entry kernel)")
        return
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    gs = getattr(experts, "w13_input_scale_quant", None)
    if not isinstance(gs, torch.Tensor) or torch.cuda.is_current_stream_capturing():
        return  # not an FP4 MoE layer (or under a capture): the next layer's prepare probes
    _ST["done"] = True
    stock = flashinfer_trtllm.fp4_quantize
    why = None
    if getattr(fp4_utils, "_flashinfer_fp4_quantize_backend", None) != "cute-dsl":
        why = f"fp4_utils' backend is {getattr(fp4_utils, '_flashinfer_fp4_quantize_backend', None)!r}, not 'cute-dsl'"
    elif stock is not getattr(fp4_utils, "fp4_quantize", None):
        why = "flashinfer_trtllm.fp4_quantize is not fp4_utils.fp4_quantize (another route holds it)"
    elif _src_sha(flashinfer_trtllm.quantize_hidden_states_fp4) != PIN_QHS:
        why = f"quantize_hidden_states_fp4's source {_src_sha(flashinfer_trtllm.quantize_hidden_states_fp4)[:12]} != pin"
    rec = (why, 0, None, None)
    if why is None:
        try:
            rec = _probe(stock, gs, gs.device)
        except Exception as e:  # noqa: BLE001 - voted below: a fault on any rank -> OFF everywhere
            rec = (f"probe raised {type(e).__name__}: {str(e)[:160]}", 0, None, None)
    grp = _group()
    votes = grp.all_gather_object(rec) if grp.world_size > 1 else [rec]
    bad = [(r, v[0]) for r, v in enumerate(votes) if v[0] is not None]
    slow = [(r, v[2], v[3]) for r, v in enumerate(votes) if v[0] is None and not v[3] <= (1.0 - MIN_GAIN) * v[2]]
    times = ", ".join(f"rank {r}: cute-dsl {v[2]:.1f} us vs cuda {v[3]:.1f} us" for r, v in enumerate(votes)
                      if v[2] is not None)
    if bad:
        _log.warning("mq-cuda: OFF on every rank (rank %d: %s); %s", bad[0][0], bad[0][1], times or "no timing")
        return
    if slow:
        _log.warning("mq-cuda: OFF on every rank (byte-equal on %d cases, but cuda is not >= %.0f %% faster at %d rows: "
                     "%s)", votes[0][1], MIN_GAIN * 100, TIME_ROWS, times)
        return
    _ST["on"], _ST["stock"] = True, stock
    _log.warning("mq-cuda: on (first MoE layer %s): cuda == cute-dsl bytes on %d cases x every rank (%s rows x 3 "
                 "families); %d rows: %s", getattr(layer, "layer_id", "?"), votes[0][1], "/".join(map(str, PROBE_ROWS)),
                 TIME_ROWS, times)


@contextlib.contextmanager
def routed(layer, forward_batch):
    if not _ST["on"] or flashinfer_trtllm.fp4_quantize is not _ST["stock"]:
        yield
        return
    flashinfer_trtllm.fp4_quantize = _cuda_quant
    try:
        yield
    finally:
        flashinfer_trtllm.fp4_quantize = _ST["stock"]
    _ST["served"] += 1
    n = _ST["served"]
    if n in (1, 42, 420) or n % 42000 == 0:
        _log.warning("mq-cuda: served %d prefill layer calls with the cuda quantize", n)
