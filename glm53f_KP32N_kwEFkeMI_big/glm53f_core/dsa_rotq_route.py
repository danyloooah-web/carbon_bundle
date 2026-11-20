from __future__ import annotations
import contextlib
import torch
from sglang.kernels.ops.quantization import act_quant_triton
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_prefill_cuda_graph
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
from glm53f_core import dsa_rotq
NODE = "dsa"
_STOCK_ROTATE = dsa_indexer_kpool.rotate_activation
_STOCK_QUANT = act_quant_triton.act_quant
_STOCK_BREAK = kpool_prefill_cuda_graph.bcg_kpool_indexer_prefill_with_output
_ROUND: list = []
_PENDING: list = []
_GATES: list = []
_SCALED: list = []
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA query rotate+quant: {why}")
def prepare(model) -> None:
    indexers = [m for m in model.modules() if isinstance(m, dsa_indexer_kpool.IndexerKPool)]
    _require(len(indexers) > 0, "no IndexerKPool in the stack")
    shapes = {(m.head_dim, m.block_size, m.scale_fmt) for m in indexers}
    _require(len(shapes) == 1, f"indexers differ: {sorted(shapes, key=str)}")
    head_dim, block_size, scale_fmt = shapes.pop()
    _require(head_dim == dsa_rotq.D and block_size == dsa_rotq.D, f"head dim {head_dim}, block {block_size}")
    _require(callable(_STOCK_ROTATE) and callable(_STOCK_QUANT), "rotate_activation or act_quant missing")
    _require(callable(getattr(_STOCK_BREAK, "run_eager", None)),
             "bcg_kpool_indexer_prefill_with_output is not an eager_on_graph callable")
    _ROUND[:] = [scale_fmt is not None]


def arm_gates(raw: torch.Tensor, c1: float, ss: float) -> None:
    
    _require(not _GATES and not _SCALED, "head gates armed while another call's are still held")
    _GATES[:] = [(raw, raw._version, c1, ss)]


def take_gates(x: torch.Tensor):
    if not _GATES:
        return None
    raw, version, c1, ss = _GATES.pop()
    _require(_ROUND == [True] and raw._version == version and raw.dtype == torch.float32 and raw.is_contiguous()
             and raw.device == x.device and tuple(raw.shape) == tuple(x.shape[:-1]),
             f"head gates {raw.dtype} {tuple(raw.shape)} armed for a rotation of {tuple(x.shape)} (scale format "
             f"{_ROUND})")
    return raw, c1, ss


def scaled(weights: torch.Tensor, q_scale: torch.Tensor):
    _require(not any(weights is raw for raw, *_ in _GATES), "armed head gates were scaled before a rotation took them")
    if not _SCALED or weights is not _SCALED[0][0]:
        return None
    raw, s, gates = _SCALED.pop()
    _require(q_scale is s, "the head gates' q_scale is not the scales of the rotation that scaled them")
    return gates


def _rotate(x: torch.Tensor) -> torch.Tensor:
    
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() >= 2 and x.shape[-1] == dsa_rotq.D
            and x.is_contiguous() and x.data_ptr() % 16 == 0):
        _require(not _GATES, f"head gates armed for a rotation of {x.dtype} {tuple(x.shape)}, which the fused "
                             "kernel does not take")
        return _STOCK_ROTATE(x)
    gates = take_gates(x)
    if gates is None:
        out, y, s = dsa_rotq.rotate_quant(x, dsa_rotq.D ** -0.5, _ROUND[0])
    else:
        out, y, s, final = dsa_rotq.rotate_quant_gated(x, gates[0], dsa_rotq.D ** -0.5, gates[1], gates[2])
        _SCALED[:] = [(gates[0], s, final)]
    _PENDING[:] = [(out, out._version, y, s)]
    return out


def _quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None):
    
    if _PENDING:
        out, version, y, s = _PENDING[0]
        if x is out and x._version == version and block_size == dsa_rotq.D and (scale_fmt is not None) == _ROUND[0]:
            _PENDING.clear()
            return y, s
    return _STOCK_QUANT(x, block_size, scale_fmt)

@contextlib.contextmanager
def _installed():
    saved = dsa_indexer_kpool.rotate_activation, act_quant_triton.act_quant
    dsa_indexer_kpool.rotate_activation, act_quant_triton.act_quant = _rotate, _quant
    try:
        yield
    finally:
        dsa_indexer_kpool.rotate_activation, act_quant_triton.act_quant = saved
        _PENDING.clear()
        held = len(_GATES) + len(_SCALED)
        _GATES.clear()
        _SCALED.clear()
    _require(not held, "head gates armed for the rotation's scaling were never scaled by the call's head-gate route")


@eager_on_graph
def _indexer_break(indexer, x, q_lora, positions, output, layer_id) -> None:
    with _installed():
        _STOCK_BREAK.run_eager(indexer, x, q_lora, positions, output, layer_id)


@contextlib.contextmanager
def routed(model, forward_batch):
    
    kpool_prefill_cuda_graph.bcg_kpool_indexer_prefill_with_output = _indexer_break
    try:
        with _installed():
            yield
    finally:
        kpool_prefill_cuda_graph.bcg_kpool_indexer_prefill_with_output = _STOCK_BREAK
