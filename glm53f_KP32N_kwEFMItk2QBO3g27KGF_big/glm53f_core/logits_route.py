from __future__ import annotations
import contextlib
import logging
import torch
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import recommended_max_tokens
from sglang.srt.layers.logits_processor import LogitsProcessor, should_apply_lm_head_quant_method
from sglang.srt.runtime_context import LoRABatchLayout, get_forward
NODE = "logits"
MAX_RANKS = 8
SIDE_WORDS = 12288
MAX_GRID = 128
FLOOR_ROWS = 768
_STOCK = LogitsProcessor._get_logits
_STATE: dict = {}
_SAID: set = set()
_HELD: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf target logits gather: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf target logits gather: %s", detail)
def _on_gpu(x: torch.Tensor) -> bool:
    return x.is_cuda
def _static_outside(node) -> str | None:
    for ok, why in ((node.do_tensor_parallel_all_gather, "it gathers no logits across the TP group"),
                    (not node.use_attn_tp_group, "its lm_head is on the attention TP group (dp-lm-head)"),
                    (not node.do_tensor_parallel_all_gather_dp_attn, "DP attention gathers its hidden states"),
                    (not node.use_tp_lm_head_all_to_all, "the TP lm_head all-to-all is on"),
                    (node.logit_scale is None, "it scales the logits"),
                    (node.final_logit_softcapping is None, "it softcaps the logits"),
                    (not node.use_fp32_lm_head and not node.use_fp32_lm_head_output, "its lm_head runs in FP32"),
                    (node.rl_on_policy_target is None, "an RL on-policy target is set"),
                    (not node.return_full_logits, "it returns full logits")):
        if not ok:
            return why
    return None

def prepare(node) -> None:
    _require(type(node) is LogitsProcessor, f"the claimed node is {type(node).__name__}, not LogitsProcessor")
    if "active" in _STATE:
        _require(_STATE["node"] == id(node), "a second logits processor in one process")
        return
    group = get_tp_group()
    capacity = recommended_max_tokens(include_prefill=False, floor=FLOOR_ROWS)
    why = _static_outside(node)
    if why is None and not 2 <= group.world_size <= MAX_RANKS:
        why = f"the TP group is {group.world_size} rank(s)"
    if why is None and capacity * group.world_size > SIDE_WORDS:
        why = f"a stage of {capacity} rows x {group.world_size} ranks has more side words than the pull's {SIDE_WORDS}"
    if why is not None:
        _STATE.update(active=False, node=id(node))
        _say("inactive", f"stock's gather serves every call on this engine: {why}")
        return
    import logits_native

    _require(callable(getattr(logits_native, "gather", None)) and callable(getattr(logits_native, "pad_words", None)),
             "logits_native lacks gather or pad_words")
    _STATE.update(active=True, node=id(node), group=group, rank=group.rank_in_group, world=group.world_size,
                  native=logits_native, capacity=capacity)


def _outside(node, hidden_states, lm_head, logits_metadata, embedding_bias) -> tuple[str, str] | None:
    
    mode = logits_metadata.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "mode", f"a {mode.name} call"
    if embedding_bias is not None:
        return "bias", "an embedding bias"
    if (not _on_gpu(hidden_states) or hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2
            or not hidden_states.is_contiguous() or hidden_states.shape[0] < 1):
        return "hidden", (f"hidden states {hidden_states.dtype} {tuple(hidden_states.shape)}, not contiguous BF16 "
                          "[rows, hidden]")
    weight = getattr(lm_head, "weight", None)
    if (not torch.is_tensor(weight) or weight.dtype != torch.bfloat16 or weight.dim() != 2
            or weight.shape[1] != hidden_states.shape[1] or not weight.is_contiguous()):
        return "lm_head", "an lm_head without a contiguous BF16 [vocab slice, hidden] weight"
    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        return "lora", "a LoRA lm_head"
    if should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None)):
        return "quant", f"a quantized lm_head ({type(getattr(lm_head, 'quant_method', None)).__name__})"
    world, rank, vr = _STATE["world"], _STATE["rank"], weight.shape[0]
    shard = getattr(lm_head, "shard_indices", None)
    if (getattr(lm_head, "tp_size", None) != world or getattr(lm_head, "use_attn_tp_group", False) or shard is None
            or shard.org_vocab_start_index != rank * vr or shard.num_org_elements != vr
            or shard.num_added_elements != 0 or node.vocab_size != world * vr or vr % 16):
        return "vocab", (f"the lm_head's slice is not rank {rank}'s {vr} of {node.vocab_size} in rank order, unpadded, "
                         "a multiple of 16")
    if hidden_states.shape[0] > _STATE["capacity"]:
        return "rows", f"{hidden_states.shape[0]} rows, over the stage's {_STATE['capacity']}"
    return None


def _buffers(vr: int, device) -> dict:
    
    state = _STATE
    if "stage" in state:
        _require(state["vr"] == vr, f"a call with a {vr}-column slice after a stage of {state['vr']}")
        return state
    _require(not torch.cuda.is_current_stream_capturing(),
             "the first served call is inside a CUDA-graph capture (no eager warm-up preceded it)")
    import torch.distributed._symmetric_memory as symmetric_memory
    group, native, capacity = state["group"], state["native"], state["capacity"]
    stage = torch.empty(capacity * vr, dtype=torch.bfloat16, device=device)
    wire = symmetric_memory.empty(capacity * vr // 4 + capacity, dtype=torch.int32, device=device)
    pads = symmetric_memory.empty(native.pad_words(), dtype=torch.int32, device=device)
    pads.zero_()
    hw = symmetric_memory.rendezvous(wire, group.device_group)
    hp = symmetric_memory.rendezvous(pads, group.device_group)
    for handle in (hw, hp):
        _require(handle.rank == state["rank"] and handle.world_size == state["world"],
                 f"a symmetric handle of rank {handle.rank} of {handle.world_size} on rank {state['rank']} of "
                 f"{state['world']}")
    torch.cuda.synchronize(device)
    group.barrier()
    state.update(stage=stage, wire=wire, pads_buf=pads, handles=(hw, hp), vr=vr,
                 wires=[int(p) for p in hw.buffer_ptrs], pads=[int(p) for p in hp.buffer_ptrs])
    _say("buffers", f"rank {state['rank']} of {state['world']}: a wire of {capacity} rows x {vr} codes exchanged")
    return state


def _get_logits(node, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
    
    why = _outside(node, hidden_states, lm_head, logits_metadata, embedding_bias)
    if why is not None:
        _say(why[0], f"stock's gather serves a call: {why[1]}")
        return _STOCK(node, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
    rows, vr = hidden_states.shape[0], lm_head.weight.shape[0]
    state = _buffers(vr, hidden_states.device)
    stage = state["stage"][: rows * vr].view(rows, vr)

    with get_forward().scoped(lora_batch_layout=LoRABatchLayout.DP_LOCAL):
        logits = node._compute_lm_head(hidden_states, lm_head, embedding_bias, out=stage)
    _require(logits.data_ptr() == stage.data_ptr() and tuple(logits.shape) == (rows, vr)
             and logits.dtype == torch.bfloat16,
             f"stock's lm_head did not write its result into the stage it was handed ({logits.dtype} "
             f"{tuple(logits.shape)})")
    vocab = state["world"] * vr

    buffer = logits_metadata.next_token_logits_buffer if use_logits_buffer else None
    if buffer is not None and tuple(buffer.shape) == (rows, vocab):
        _require(buffer.dtype == torch.float32 and buffer.stride(1) == 1 and (rows == 1 or buffer.stride(0) >= vocab),
                 f"the logits buffer is {buffer.dtype} with strides {tuple(buffer.stride())}")
        out = buffer
    else:
        out = torch.empty((rows, vocab), dtype=torch.float32, device=hidden_states.device)
    grid = state["native"].gather(out, stage, state["wires"], state["pads"], state["rank"], vr, state["capacity"],
                                  MAX_GRID)
    _say("ours", f"a pack and one pull serve the gather (first: {rows} rows, {state['world']} ranks, {grid} CTAs)")
    return out


@contextlib.contextmanager
def routed(node, forward_batch):
    mode = forward_batch.forward_mode
    if not _STATE.get("active") or not (mode.is_decode() or mode.is_target_verify()):
        yield
        return

    _require(id(node) not in _HELD, "the logits processor's _get_logits is already this route's")
    _HELD.add(id(node))
    node._get_logits = lambda *a, **k: _get_logits(node, *a, **k)
    try:
        yield
    finally:
        del node._get_logits
        _HELD.discard(id(node))
