from __future__ import annotations
import contextlib
import logging
import flashinfer.fused_moe as fused_moe
import moe_finalize_native
import torch
from flashinfer.autotuner import AutoTuner
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.models import deepseek_v2
NODE = "layer"
HIDDEN, TOP_K = 4096, 8
MIN_ROWS, MAX_ROWS = 193, 16384
_LAYERS: set = set()
_PENDING: list = []
_SET: list = []
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf MoE finalize: {why}")
def _say(tag: str, detail: str) -> None:
    
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf MoE finalize: %s", detail)


def prepare(layer) -> None:
    
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE):
        return
    experts = mlp.experts
    if not (isinstance(experts.quant_method, ModelOptNvFp4FusedMoEMethod) and experts.top_k == TOP_K
            and experts.hidden_size == HIDDEN and not experts.reduce_results and mlp.num_fused_shared_experts == 0
            and not mlp._shared_expert_tp1 and mlp.shared_experts is not None):
        _say("layer", f"layer {getattr(layer, 'layer_id', '?')}'s MoE is not the NVFP4 top-{TOP_K} MoE with a "
                      f"TP-sharded shared expert ({type(experts.quant_method).__name__}): stock finalizes it")
        return
    _require(callable(fused_moe.trtllm_fp4_block_scale_moe), "flashinfer.fused_moe.trtllm_fp4_block_scale_moe missing")
    _require(callable(deepseek_v2.maybe_fuse_routed_scale_and_shared_add),
             "deepseek_v2.maybe_fuse_routed_scale_and_shared_add missing")
    _LAYERS.add(id(layer))


def _moe(*args, **kw):
    
    out, hs = kw.get("output"), kw.get("hidden_states")
    rows = hs.shape[0] if isinstance(hs, torch.Tensor) else -1
    if args or kw.get("do_finalize") is not True or not isinstance(out, torch.Tensor) or kw.get("top_k") != TOP_K:
        tag, why = "form", "a call not of SGLang's finalizing top-8 form (stock's deferred finalize at 192 rows or fewer)"
    elif not MIN_ROWS <= rows <= MAX_ROWS:
        tag, why = f"rows {rows < MIN_ROWS}", f"{rows} rows, outside {MIN_ROWS} to {MAX_ROWS}"
    elif not (out.is_cuda and out.dtype == torch.bfloat16 and tuple(out.shape) == (rows, HIDDEN)
              and out.is_contiguous()):
        tag, why = "output", f"an output of {out.dtype} {tuple(out.shape)}"
    elif AutoTuner.get().is_tuning_mode:
        tag, why = "tuning", "the AutoTuner's tuning"
    else:
        tag = None
    if tag is not None:
        _say(tag, f"FlashInfer's finalize and stock's add serve a call ({why})")
        return _SET[0](*args, **kw)
    _require(not _PENDING, "a MoE call before the previous one's add")
    result = _SET[0](**dict(kw, do_finalize=False))
    _require(isinstance(result, (list, tuple)) and len(result) == 3,
             f"FlashInfer's unfinalized call returned {type(result).__name__} of {len(result)}")
    _PENDING.append((out, *result, kw.get("enable_pdl") is True))
    return [out]


def _add(experts, routed, shared, routed_scaling_factor):
    
    if not _PENDING or not isinstance(routed, torch.Tensor) or routed.data_ptr() != _PENDING[0][0].data_ptr():
        _say("add", "stock's add serves a call without an owed finalize")
        return _SET[1](experts, routed, shared, routed_scaling_factor)
    out, gemm2, weights, permuted, pdl = _PENDING.pop()
    _require(tuple(routed.shape) == tuple(out.shape) and routed.is_contiguous(),
             f"the owed output came back as {tuple(routed.shape)}, contiguous {routed.is_contiguous()}")
    _require(isinstance(shared, torch.Tensor), "the owed add has no shared-expert rows")
    moe_finalize_native.fuse(gemm2, permuted, weights, shared, routed, pdl)
    _say("ours", f"one launch finalizes and adds the shared rows (first: {routed.shape[0]} rows, PDL {pdl})")
    return routed


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if id(layer) not in _LAYERS:
        yield
        return
    if deepseek_v2.is_in_breakable_cuda_graph() or deepseek_v2.is_in_tc_piecewise_cuda_graph():
        _say("graph", "stock finalizes in breakable and piecewise graphs")
        yield
        return
    _require(not _SET, "a layer's call inside another's")
    _SET[:] = [fused_moe.trtllm_fp4_block_scale_moe, deepseek_v2.maybe_fuse_routed_scale_and_shared_add]
    fused_moe.trtllm_fp4_block_scale_moe, deepseek_v2.maybe_fuse_routed_scale_and_shared_add = _moe, _add
    try:
        yield
    finally:
        fused_moe.trtllm_fp4_block_scale_moe, deepseek_v2.maybe_fuse_routed_scale_and_shared_add = _SET
        _SET.clear()
        owed = len(_PENDING)
        _PENDING.clear()
    _require(not owed, "a MoE call's add never came: its output rows were never written")
