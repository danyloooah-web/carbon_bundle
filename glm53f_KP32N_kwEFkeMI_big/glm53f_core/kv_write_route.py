from __future__ import annotations
import contextlib
import logging
import kv_write
import torch
from sglang.kernels.ops.kvcache import mla_buffer
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
NODE = "dsa"
RANK = 512
GRID_PER_SM = 8
FP8 = torch.float8_e4m3fn
_SAID: set = set()
_PENDING: list = []
_SET: list = []
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KV write: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf KV write: %s", detail)
def prepare(node) -> None:
    _require(getattr(node, "kv_lora_rank", None) == RANK and getattr(node, "qk_rope_head_dim", None) == 0,
             f"the attention module is not a NoPE MLA of rank {RANK}: kv_lora_rank "
             f"{getattr(node, 'kv_lora_rank', None)}, qk_rope_head_dim {getattr(node, 'qk_rope_head_dim', None)}")
    _require(callable(dsa_backend.mla_quantize_for_fp8_no_rope), "dsa_backend.mla_quantize_for_fp8_no_rope missing")
    _require(callable(mla_buffer._set_mla_kv_buffer_impl), "mla_buffer._set_mla_kv_buffer_impl missing")

def _rows_ok(x: torch.Tensor) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == RANK and x.stride(-1) == 1
            and x.data_ptr() % 16 == 0)


def _quantize(q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim):
    _require(not _PENDING, "a quantize before the previous one's store")
    q_len, num_heads = q_rope.shape[:2]
    if not (qk_rope_head_dim == 0 and kv_lora_rank == RANK and q_len > 0 and q_nope.dim() == 3
            and tuple(q_nope.shape) == (q_len, num_heads, RANK) and _rows_ok(q_nope)
            and q_nope.stride(0) % 8 == 0 and q_nope.stride(1) % 8 == 0 and q_nope.stride(0) >= 0
            and q_nope.stride(1) >= 0 and k_nope.dim() == 2 and k_nope.shape[0] > 0 and k_nope.is_contiguous()
            and _rows_ok(k_nope) and k_rope.numel() == 0):
        _say("call", f"the quantize before this route serves a call: q {tuple(q_nope.shape)} {q_nope.dtype}, k "
                     f"{tuple(k_nope.shape)}, rank {kv_lora_rank}, RoPE {qk_rope_head_dim}")
        return _SET[0](q_nope, q_rope, k_nope, k_rope, kv_lora_rank, qk_rope_head_dim)
    q_out = q_rope.new_empty(q_len, num_heads, kv_lora_rank, dtype=FP8)
    k_out = k_nope.new_empty(k_nope.shape, dtype=FP8)
    _PENDING.append((q_nope, q_out, k_nope, k_out))
    return q_out, k_out, k_rope.to(FP8)


def _impl(kv_buffer, loc, cache_k_nope, cache_k_rope=None, *, reserved_skip_index, dcp_world_size, dcp_rank):
    
    if not _PENDING or cache_k_nope.data_ptr() != _PENDING[0][3].data_ptr() \
            or tuple(cache_k_nope.shape) != tuple(_PENDING[0][3].shape):
        return _SET[1](kv_buffer, loc, cache_k_nope, cache_k_rope, reserved_skip_index=reserved_skip_index,
                       dcp_world_size=dcp_world_size, dcp_rank=dcp_rank)
    q_nope, q_out, k_nope, k_out = _PENDING.pop()
    rows = k_out.shape[0]
    _require(cache_k_nope.element_size() == 1 and (cache_k_rope is None or cache_k_rope.numel() == 0),
             f"the store of the pending k is {cache_k_nope.dtype} with RoPE bytes "
             f"{None if cache_k_rope is None else tuple(cache_k_rope.shape)}")
    _require(dcp_world_size == 1, f"the store of the pending k is DCP-sharded ({dcp_world_size} ranks)")
    _require(loc.is_cuda and loc.dtype in (torch.int64, torch.int32) and loc.dim() == 1 and loc.is_contiguous()
             and loc.numel() == rows, f"the store's loc is {loc.dtype} {tuple(loc.shape)} for {rows} k rows")
    _require(kv_buffer.is_cuda and kv_buffer.element_size() == 1 and kv_buffer.dim() >= 2 and kv_buffer.shape[0] > 0
             and kv_buffer[0].is_contiguous() and kv_buffer[0].numel() == RANK and kv_buffer.stride(0) >= RANK
             and kv_buffer.stride(0) % 16 == 0 and kv_buffer.data_ptr() % 16 == 0,
             f"the layer's cache is {kv_buffer.dtype} {tuple(kv_buffer.shape)} strides {tuple(kv_buffer.stride())}")
    kv_write.write(q_nope, q_out, k_nope, k_out, kv_buffer, loc, GRID_PER_SM)
    _say("ours", f"one launch writes the FP8 query and the KV rows (first: {q_nope.shape[0]} x {q_nope.shape[1]} query "
                 f"rows, {rows} cache rows, loc {loc.dtype})")
    return None


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2()):
        name = getattr(mode, "name", str(mode))
        _say(f"mode {name}", f"stock's KV write serves {name} forwards")
        yield
        return
    if getattr(get_token_to_kv_pool(), "dsa_kv_cache_store_fp8", None) is not False:
        _say("layout", "the KV pool is not the raw FP8 layout (dsa_kv_cache_store_fp8 not off): stock's KV write "
                       "serves every call")
        yield
        return
    _require(not _SET, "a sparse layer's call inside another's")
    _SET[:] = [dsa_backend.mla_quantize_for_fp8_no_rope, mla_buffer._set_mla_kv_buffer_impl]
    dsa_backend.mla_quantize_for_fp8_no_rope, mla_buffer._set_mla_kv_buffer_impl = _quantize, _impl
    try:
        yield
    finally:
        dsa_backend.mla_quantize_for_fp8_no_rope, mla_buffer._set_mla_kv_buffer_impl = _SET
        _SET.clear()
        owed = len(_PENDING)
        _PENDING.clear()
    _require(not owed, "a quantize's store never came: its query and k were never written")
