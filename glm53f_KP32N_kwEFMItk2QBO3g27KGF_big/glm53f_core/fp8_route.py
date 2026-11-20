from __future__ import annotations
import contextlib
import fp8_pack
import torch
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
NODE = "dsa"
RANK = 512
GRID_PER_SM = 8
FP8 = torch.float8_e4m3fn
_STOCK = dsa_backend.mla_quantize_for_fp8_no_rope
_MIXIN = forward_mla.DeepseekMLAForwardMixin


_STOCK_BREAK = _MIXIN._eager_bmm_attention


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf FP8 query pack: {why}")


def prepare(model) -> None:
    
    _require(callable(_STOCK), "dsa_backend.mla_quantize_for_fp8_no_rope missing")
    _require(callable(getattr(_STOCK_BREAK, "run_eager", None)),
             "DeepseekMLAForwardMixin._eager_bmm_attention is not an eager_on_graph callable")
def _rows_ok(x: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == RANK and x.stride(-1) == 1
            and x.data_ptr() % 16 == 0)

def _quantize(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim):
    
    q_len, num_heads = q_rope.shape[:2]
    if not (qk_rope_head_dim == 0 and kv_lora_rank == RANK and q_len > 0 and q_nope.dim() == 3
            and tuple(q_nope.shape) == (q_len, num_heads, RANK) and _rows_ok(q_nope)
            and q_nope.stride(0) % 8 == 0 and q_nope.stride(1) % 8 == 0
            and k_nope.dim() == 2 and k_nope.shape[0] > 0 and k_nope.is_contiguous() and _rows_ok(k_nope)):
        return _STOCK(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
    q_out = q_rope.new_empty(q_len, num_heads, kv_lora_rank, dtype=FP8)
    k_out = k_nope.new_empty(k_nope.shape, dtype=FP8)
    fp8_pack.pack(q_nope, q_out, k_nope, k_out, GRID_PER_SM)
    return q_out, k_out, k_rope.to(FP8)


@eager_on_graph
def _bmm_attention(self, *args, **kwargs) -> None:
    
    saved = dsa_backend.mla_quantize_for_fp8_no_rope
    dsa_backend.mla_quantize_for_fp8_no_rope = _quantize
    try:
        _STOCK_BREAK.run_eager(self, *args, **kwargs)
    finally:
        dsa_backend.mla_quantize_for_fp8_no_rope = saved


@contextlib.contextmanager
def routed(model, forward_batch):
    
    dsa_backend.mla_quantize_for_fp8_no_rope = _quantize
    _MIXIN._eager_bmm_attention = _bmm_attention
    try:
        yield
    finally:
        dsa_backend.mla_quantize_for_fp8_no_rope = _STOCK
        _MIXIN._eager_bmm_attention = _STOCK_BREAK
