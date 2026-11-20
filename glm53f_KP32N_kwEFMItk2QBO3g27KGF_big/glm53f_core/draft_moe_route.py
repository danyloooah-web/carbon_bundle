from __future__ import annotations
import contextlib
import logging
import time
import torch
from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
    align_mxfp8_moe_weights_for_flashinfer_trtllm,
    get_activation_type,
)
from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
NODE = "dmoe"
CHUNK = 32
_TILE_M = 128
_STOCK_APPLY = unquant.UnquantizedFusedMoEMethod.apply
_INFO: dict = {}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf draft MoE: {why}")


def _canonical_rows(param: torch.Tensor, gated: bool) -> tuple[torch.Tensor, int]:
    
    first = param[0].view(torch.uint8)
    rows = first.shape[1]
    tile = first.permute(1, 0, 2).reshape(rows, -1)
    perm = (_maybe_get_cached_w3_w1_permute_indices({}, tile, _TILE_M, is_gated_act_gemm=True) if gated
            else get_w2_permute_indices_with_cache({}, tile, _TILE_M)).to(param.device)
    _require(torch.equal(perm.sort().values, torch.arange(rows, device=param.device)),
             "FlashInfer's row permutation of the BF16 experts is not a bijection")
    return torch.argsort(perm), rows


def _canonical(param: torch.Tensor, lo: int, hi: int, inverse: torch.Tensor, rows: int) -> torch.Tensor:
    
    blocked = param[lo:hi].view(torch.uint8)
    tiles = blocked.permute(0, 2, 1, 3).reshape(hi - lo, rows, -1)
    return tiles[:, inverse].contiguous().view(torch.bfloat16)
def _mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, m, k = weight.shape
    values, scale = flashinfer_mxfp8_quantize(weight.reshape(-1, k).contiguous(), False)
    return values.view(n, m, k), scale.view(torch.uint8).contiguous().view(n, m, k // 32)


def _experts(node):
    layer = getattr(node, "experts", None)
    _require(layer is not None and hasattr(layer, "quant_method"), f"the claimed node is {type(node).__name__}, "
             "not a MoE with fused experts")
    return layer
def prepare(node) -> None:
    
    layer = _experts(node)
    if id(layer) in _INFO:
        return
    began = time.perf_counter()
    method, config = layer.quant_method, layer.moe_runner_config
    _require(type(method) is unquant.UnquantizedFusedMoEMethod and method.use_flashinfer_trtllm_moe
             and method.runner.runner_backend.is_flashinfer_trtllm(),
             f"the experts run {type(method).__name__} on {method.runner.runner_backend}, not unquantized experts "
             "on the FlashInfer TRT-LLM runner")
    _require(not torch.cuda.is_current_stream_capturing(), "prepare ran under graph capture")
    _require(config.is_gated and config.activation == "silu" and not config.no_combine
             and not config.num_fused_shared_experts,
             f"the experts' runner config is not gated SiLU with a combined output ({config})")
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    experts, inter, hidden = layer.num_local_experts, config.intermediate_size_per_partition, w2.shape[2]
    _require(w13.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16 and w13.is_cuda
             and tuple(w13.shape) == (experts, hidden // 64, 2 * inter, 64)
             and tuple(w2.shape) == (experts, inter // 64, hidden, 64),
             f"the experts' weights are {w13.dtype} {tuple(w13.shape)} and {w2.dtype} {tuple(w2.shape)}, not "
             f"stock's BF16 block layout for {experts} experts of {hidden} x {inter}")
    order13, order2 = _canonical_rows(w13, True), _canonical_rows(w2, False)
    out = None
    for lo in range(0, experts, CHUNK):
        hi = min(experts, lo + CHUNK)
        holder = torch.nn.Module()
        holder.moe_runner_config = config
        values13, scale13 = _mxfp8(_canonical(w13, lo, hi, *order13))
        values2, scale2 = _mxfp8(_canonical(w2, lo, hi, *order2))
        holder.w13_weight = torch.nn.Parameter(values13, requires_grad=False)
        holder.w13_weight_scale_inv = torch.nn.Parameter(scale13, requires_grad=False)
        holder.w2_weight = torch.nn.Parameter(values2, requires_grad=False)
        holder.w2_weight_scale_inv = torch.nn.Parameter(scale2, requires_grad=False)
        align_mxfp8_moe_weights_for_flashinfer_trtllm(holder)
        part = (holder.w13_weight.data, holder.w13_weight_scale_inv.data, holder.w2_weight.data,
                holder.w2_weight_scale_inv.data)
        if out is None:
            out = tuple(torch.empty((experts, *t.shape[1:]), dtype=t.dtype, device=t.device) for t in part)
        for whole, t in zip(out, part):
            whole[lo:hi].copy_(t)
    _require(tuple(out[0].shape) == (experts, 2 * inter, hidden) and tuple(out[2].shape) == (experts, hidden, inter)
             and out[0].dtype == torch.float8_e4m3fn and out[1].dtype == torch.uint8,
             f"the aligned MXFP8 experts are {out[0].dtype} {tuple(out[0].shape)} and {tuple(out[2].shape)}")
    _INFO[id(layer)] = FlashInferTrtllmFp8MoeQuantInfo(
        w13_weight=out[0], w2_weight=out[2], global_num_experts=int(layer.num_experts),
        local_expert_offset=int(layer.moe_ep_rank) * experts, local_num_experts=experts, intermediate_size=inter,
        routing_method_type=int(config.routing_method_type), block_quant=True, use_mxfp8=True, weight_block_k=32,
        w13_weight_scale_inv=out[1], w2_weight_scale_inv=out[3],

        gemm1_clamp_limit=None,
        activation_type=get_activation_type(config.activation, is_gated=config.is_gated))
    torch.cuda.synchronize()
    logging.getLogger(__name__).info(
        "glmf draft MoE: %d experts of %d x %d a rank as MXFP8 (%.2f GB beside the BF16 experts) in %.1f s",
        experts, hidden, inter, sum(t.numel() * t.element_size() for t in out) / 2**30,
        time.perf_counter() - began)

def _apply(self, layer, dispatch_output):
    
    info = _INFO.get(id(layer))
    x = dispatch_output.hidden_states
    if info is None or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[0] == 0:
        return _STOCK_APPLY(self, layer, dispatch_output)
    return self.runner.run(dispatch_output, info)


@contextlib.contextmanager
def routed(node, forward_batch):
    
    _require(id(_experts(node)) in _INFO, "the claimed MoE was not prepared")
    unquant.UnquantizedFusedMoEMethod.apply = _apply
    try:
        yield
    finally:
        unquant.UnquantizedFusedMoEMethod.apply = _STOCK_APPLY
