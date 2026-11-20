from __future__ import annotations
import contextlib
import logging
import torch
from sglang.srt.layers.quantization import unquant
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_disagg
NODE = "dsa"
_STOCK_APPLY = unquant.UnquantizedLinearMethod.apply
_STOCK_QB = DeepseekV2AttentionMLA.q_b_proj_forward
_WEIGHTS: dict = {}
_CALL: dict = {}
_TAKEN = ("taken",)
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf q_b_proj + wq_b merge: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf q_b_proj + wq_b merge: %s", detail)
def _stock_kernel(m: int, n: int, k: int):
    
    backend = unquant.get_bf16_gemm_backend()
    if not (backend.is_cutedsl() or backend.is_gemv()):
        return None
    if unquant._mm_bf16 is not None and (m <= unquant._BF16_LOW_M_AUTOTUNE_MAX_M
                                         or unquant._flashinfer_autotuner.is_tuning_mode):
        return "stock runs FlashInfer's low-M CuTe DSL GEMM there, or tunes it"
    if unquant._enable_bf16_splitk_gemm and unquant.use_bf16_splitk_gemm(m, n, k):
        return "stock runs a split-K table's kernel"
    if unquant._use_hopper_bf16_tiny_gemv is not None or unquant._use_hopper_bf16_gemv is not None:
        return "stock runs a GEMV backend"
    if unquant._use_cutedsl_bf16_gemm is not None and unquant._use_cutedsl_bf16_gemm(m, n, k):
        return "stock runs TGV there"
    return None

def prepare(node) -> None:
    
    if id(node) in _WEIGHTS:
        return
    qb, wqb = node.q_b_proj, getattr(getattr(node, "indexer", None), "wq_b", None)
    why = ("the MTP draft layer" if getattr(node, "is_nextn", False)
           else "a PD prefill server" if get_disagg().disaggregation_mode == "prefill"
           else "no indexer of its own (skip_topk)" if wqb is None or node.skip_topk
           else f"q_b_proj's {qb.weight.shape[0]} and wq_b's {wqb.weight.shape[0]} output columns"
           if qb.weight.shape != wqb.weight.shape else None)
    if why is not None:
        _WEIGHTS[id(node)] = None
        _say("inactive " + why, f"{why}: stock's two GEMMs")
        return
    for linear in (qb, wqb):
        w = linear.weight
        _require(type(linear.quant_method) is unquant.UnquantizedLinearMethod and linear.bias is None
                 and w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous() and not w.requires_grad
                 and not getattr(linear, "small_m_rowmajor", False)
                 and getattr(linear, "small_m_tiled_weight", None) is None,
                 f"a {tuple(w.shape)} {w.dtype} projection that is not stock's plain BF16 unquantized linear")
    _require(type(node).q_b_proj_forward is _STOCK_QB and not node._q_b_proj_verified_shape,
             "q_b_proj_forward is not stock's plain self.q_b_proj call")
    _require(not torch.cuda.is_current_stream_capturing(), "prepare inside a CUDA-graph capture")
    n, k = qb.weight.shape
    _WEIGHTS[id(node)] = torch.cat([qb.weight, wqb.weight]).view(2, n, k).transpose(1, 2)
    _say("weights", f"[q_b_proj; wq_b] as one [{2 * n}, {k}] BF16 weight: {4 * n * k >> 20} MiB a sparse layer")


def _apply(self, layer, x, bias=None):
    
    if layer is _CALL["qb"]:
        return _first(self, layer, x, bias)
    if layer is _CALL["wqb"]:
        return _second(self, layer, x, bias)
    return _STOCK_APPLY(self, layer, x, bias)


def _identity(x: torch.Tensor) -> tuple:
    return x.data_ptr(), tuple(x.shape), x.stride(), x._version, torch.cuda.current_stream()


def _first(self, layer, x, bias):
    _require("made" not in _CALL, "a second q_b_proj call in one layer's call")
    weights = _CALL["weights"]
    k, n = weights.shape[1], weights.shape[2]
    m = x.shape[0] if x.dim() == 2 else -1
    ok = x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous()
    why = _stock_kernel(m, n, k) if ok and bias is None else "an input other than a contiguous BF16 GPU [rows, k]"
    if why is not None:
        _CALL["made"] = None
        _say(f"declined {why}", f"stock's two GEMMs: {why} (first: {m} rows, {x.dtype}, {tuple(x.stride())})")
        return _STOCK_APPLY(self, layer, x, bias)
    both = torch.bmm(x.expand(2, -1, -1), weights)
    _CALL["made"] = [_identity(x), both[1]]
    _say("ours", f"q_b_proj and wq_b as one launch (first: {m} rows, 2 x {n} columns)")
    return both[0]


def _second(self, layer, x, bias):
    if "made" not in _CALL:
        _require(DeepseekV2AttentionMLA.q_b_proj_forward is not _STOCK_QB,
                 "wq_b's call came before q_b_proj's or q_b_proj's GEMM left stock's apply: this route serves "
                 "q_b_proj, then wq_b")
        _CALL["made"] = None
        _say("q-half", "q_b_proj made by another route's q_b_proj_forward (C1 q-half): stock's wq_b alone")
        return _STOCK_APPLY(self, layer, x, bias)
    made = _CALL["made"]
    if made is None:
        return _STOCK_APPLY(self, layer, x, bias)
    _require(made is not _TAKEN, "a second wq_b call in one layer's call")
    _require(bias is None and _identity(x) == made[0],
             f"wq_b's input {_identity(x)[:4]} is not the tensor q_b_proj's call read {made[0][:4]} on the same stream")
    _CALL["made"] = _TAKEN
    return made[1]


@contextlib.contextmanager
def routed(node, forward_batch):
    
    _require(id(node) in _WEIGHTS and not _CALL, "a call of a module this route never prepared, or inside another's")
    weights, mode = _WEIGHTS[id(node)], forward_batch.forward_mode
    held = unquant.UnquantizedLinearMethod.apply is not _STOCK_APPLY
    if weights is None or held or not mode.is_target_verify():
        if weights is not None:
            name = "held" if held else getattr(mode, "name", str(mode))
            _say(f"mode {name}", "another route holds UnquantizedLinearMethod.apply: its two calls" if held
                 else f"stock's two GEMMs serve {name} calls")
        yield
        return
    _CALL.update(qb=node.q_b_proj, wqb=node.indexer.wq_b, weights=weights)
    unquant.UnquantizedLinearMethod.apply = _apply
    try:
        yield
        made = _CALL.get("made")
        _require(made is None or made is _TAKEN, "q_b_proj's call made wq_b's output and no wq_b call took it")
    finally:
        unquant.UnquantizedLinearMethod.apply = _STOCK_APPLY
        _CALL.clear()
