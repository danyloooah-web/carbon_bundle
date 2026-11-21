from __future__ import annotations
import contextlib
import torch
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool
from glm53f_core import dsa_rotq_route
NODE = "dsa"
_INDEXER = dsa_indexer_kpool.IndexerKPool
_STOCK_GATE = _INDEXER._get_logits_head_gate
_STOCK_PROJECT = _INDEXER._project_and_scale_head_gates
_STOCK_SCALE = _INDEXER._apply_q_scale_and_softmax_scale
_WEIGHT: dict = {}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA head gate: {why}")
def prepare(node) -> None:
    indexers = [m for m in node.modules() if isinstance(m, _INDEXER)]
    _require(len(indexers) == 1, f"the claimed attention holds {len(indexers)} pooled indexers")
    indexer = indexers[0]
    if id(indexer) in _WEIGHT:
        return
    weight = indexer.weights_proj.weight.data
    _require(weight.dtype == torch.float32 and weight.dim() == 2 and weight.is_cuda
             and weight.shape[0] == indexer.n_heads,
             f"the head-gate projection is {weight.dtype} {tuple(weight.shape)} for {indexer.n_heads} heads")
    _require(callable(_STOCK_GATE) and callable(_STOCK_PROJECT) and callable(_STOCK_SCALE),
             "the indexer's head-gate methods are missing")
    half = weight.to(torch.bfloat16)
    _require(bool(torch.equal(half.float(), weight)), "the head-gate projection holds values BF16 cannot carry")
    _WEIGHT[id(indexer)] = half.contiguous()

def _raw(indexer, x: torch.Tensor):
    
    weight = _WEIGHT.get(id(indexer))
    if weight is None or not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] > 0
                              and x.shape[1] == weight.shape[1]):
        return None
    return torch.mm(x, weight.t(), out_dtype=torch.float32)


def _gate(self, x: torch.Tensor, q_scale: torch.Tensor):
    
    raw = _raw(self, x)
    if raw is None:
        return _STOCK_GATE(self, x, q_scale)
    weights = raw * self.n_heads**-0.5
    return weights.unsqueeze(-1) * q_scale * self.softmax_scale


def _project(self, x: torch.Tensor) -> torch.Tensor:
    
    raw = _raw(self, x)
    if raw is None:
        return _STOCK_PROJECT(self, x)
    if (dsa_indexer_kpool.rotate_activation is dsa_rotq_route._rotate and dsa_rotq_route._ROUND == [True]
            and _INDEXER._apply_q_scale_and_softmax_scale is _scale):
        dsa_rotq_route.arm_gates(raw, self.n_heads**-0.5, self.softmax_scale)
        return raw
    return raw * self.n_heads**-0.5


def _scale(self, weights: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    final = dsa_rotq_route.scaled(weights, q_scale)
    return _STOCK_SCALE(self, weights, q_scale) if final is None else final


@contextlib.contextmanager
def routed(node, forward_batch):
    
    _INDEXER._get_logits_head_gate, _INDEXER._project_and_scale_head_gates = _gate, _project
    _INDEXER._apply_q_scale_and_softmax_scale = _scale
    try:
        yield
    finally:
        _INDEXER._get_logits_head_gate, _INDEXER._project_and_scale_head_gates = _STOCK_GATE, _STOCK_PROJECT
        _INDEXER._apply_q_scale_and_softmax_scale = _STOCK_SCALE
