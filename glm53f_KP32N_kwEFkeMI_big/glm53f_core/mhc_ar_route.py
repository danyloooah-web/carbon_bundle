from __future__ import annotations
import contextlib
import torch
from glm53f_core import e31_fused_route as fused
NODE = "layer"
_HIDDEN, _RANKS = 4096, 4
ROWS = (6, 80)
_SLOT_BYTES = 2 * _RANKS * ROWS[1] * _HIDDEN * 2
_MISSING = object()
_state = {"checked": False, "why": "", "groups": (), "native": None, "keep": None, "slots": None, "sync": None,
          "rank": 0, "none": None, "pending": None, "live": False, "boundaries": False, "said": set(), "tally": {}}
def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-mhcar] {detail}", flush=True)
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-mhcar] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"fused all-reduce: {why}")
def _load_native():
    import mhc_fi_native
    return mhc_fi_native
def _groups():
    from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_moe_tp_group, get_tp_group

    return get_tp_group(), get_attn_tp_group(), get_moe_tp_group()


def prepare(layer) -> None:
    
    if _state["checked"]:
        return
    _require(not torch.cuda.is_current_stream_capturing(), "prepare ran under graph capture")
    unique = []
    for group in _groups():
        if all(group is not seen for seen in unique):
            unique.append(group)
    ranks = [list(g.ranks) for g in unique]
    if any(g.world_size != _RANKS for g in unique) or any(r != ranks[0] for r in ranks):
        _state.update(checked=True, why=f"the sum groups are not one {_RANKS}-rank set (ranks {ranks})")
        _say("inactive", f"inactive on this engine: {_state['why']}; every sum is stock's")
        return
    native = _load_native()
    _require(callable(getattr(native, "prepare_ar", None)) and callable(getattr(native, "run_ar", None)),
             "mhc_fi_native lacks prepare_ar / run_ar (a module from before version 150)")
    import torch.distributed._symmetric_memory as symmetric_memory

    group, device = unique[0], torch.device("cuda", torch.cuda.current_device())
    native.prepare_ar(device.index)
    area = symmetric_memory.empty(_SLOT_BYTES, dtype=torch.uint8, device=device)
    area.zero_()
    handle = symmetric_memory.rendezvous(area, group.device_group)
    _require(handle.rank == group.rank_in_group and handle.world_size == _RANKS,
             f"a symmetric handle of rank {handle.rank} of {handle.world_size} on rank {group.rank_in_group}")
    sync = torch.zeros(2, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    group.barrier()
    _state.update(checked=True, groups=tuple(unique), native=native, keep=(area, handle),
                  slots=[int(p) for p in handle.buffer_ptrs], sync=sync, rank=group.rank_in_group,
                  none=torch.empty(0, dtype=torch.bfloat16, device=device))
    _tally("prepared", lambda: f"rank {group.rank_in_group} of {ranks[0]}, {len(unique)} group object(s), rows "
                               f"{ROWS[0]}-{ROWS[1]}, slots {_SLOT_BYTES / 2 ** 20:.0f} MiB a rank")

def _stock_one_shot(group, x: torch.Tensor):
    
    if x.is_cpu:
        return "cpu", "a CPU tensor"
    if group.hpu_communicator is not None and not group.hpu_communicator.disabled:
        return "hpu", "the group's HPU communicator"
    if group.xpu_communicator is not None and not group.xpu_communicator.disabled:
        return "xpu", "the group's XPU communicator"
    if torch.compiler.is_compiling():
        return "compile", "torch.compile's out-of-place op"
    pynccl = group.get_available_pynccl_comm()
    mscclpp = group.pymscclpp_comm is not None and group.pymscclpp_comm.should_mscclpp_allreduce(x)
    ca = group.ca_comm
    custom = ca is not None and not ca.disabled and ca.should_custom_ar(x)
    if pynccl is not None and group.is_symmetric_memory_enabled() and not mscclpp and not custom:
        return "symm", "NCCL's symmetric-memory sum"
    if group._can_use_flashinfer_allreduce(x):
        return "flashinfer", "FlashInfer's all-reduce"
    method = group._resolve_outplace_all_reduce_method(input_=x, should_use_pymscclpp_allreduce=mscclpp)
    if method != "ca":
        return f"method {method}", f"SGLang's {method or 'in-place NCCL'} all-reduce"
    if not ca.uses_one_shot_push(x.numel() * x.element_size()):
        return "algo", "its custom all-reduce, not with the one-shot push at this size and mode"
    return None


def boundaries(installed: bool) -> None:
    
    _state["boundaries"] = installed


def _declined(group, x: torch.Tensor):
    
    if not _state["boundaries"]:
        return "boundaries", "this layer's boundaries are not mhc_fi_route's (nothing would take the partial)"
    if fused._state["pending"] is not None:
        return "e31", "E31 holds a pending partial"
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == _HIDDEN and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        return "layout", f"a partial {x.dtype} {tuple(x.shape)}"
    if not ROWS[0] <= x.shape[0] <= ROWS[1]:
        return "rows", f"{x.shape[0]} rows, outside {ROWS[0]} to {ROWS[1]}"
    return _stock_one_shot(group, x)


def _deferring(group, before):
    

    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        _require(_state["pending"] is None, "a sum was asked for while another partial is pending (its boundary "
                                            "never took it)")
        why = _declined(group, input_)
        if why is not None:
            _say(why[0], f"the sum it wraps serves a call: {why[1]}")
            return before(input_)
        _state["pending"] = (input_, before)
        return input_

    return all_reduce


def pending():
    
    held = _state["pending"]
    return None if held is None else held[0]


def complete(x: torch.Tensor) -> torch.Tensor:
    
    held = _state["pending"]
    _require(held is not None and held[0] is x, "complete() of a tensor that is not the pending partial")
    _state["pending"] = None
    _tally("completed by stock", lambda: f"{x.shape[0]} rows")
    return held[1](x)


def take(x: torch.Tensor) -> tuple:
    
    held = _state["pending"]
    _require(held is not None and held[0] is x, "take() of a tensor that is not the pending partial")
    _state["pending"] = None
    _tally("fused", lambda: f"{x.shape[0]} rows{', captured' if torch.cuda.is_current_stream_capturing() else ''}")
    return _state["slots"], _state["sync"], _state["rank"], _state["none"]


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if _state["native"] is None:
        yield
        return
    _require(not _state["live"], "already routed (a layer's call inside another)")
    wrapped = []
    _state["live"] = True
    _state["pending"] = None
    try:
        for group in _state["groups"]:


            current = group.all_reduce
            own = not (getattr(current, "__self__", None) is group
                       and getattr(current, "__func__", None) is type(group).all_reduce)
            wrapped.append((group, current if own else _MISSING))
            group.all_reduce = _deferring(group, current)
        yield
        _require(_state["pending"] is None, "the layer's call returned with a partial pending (no boundary took it)")
    finally:
        for group, kept in reversed(wrapped):
            if kept is _MISSING:
                del group.all_reduce
            else:
                group.all_reduce = kept
        _state["pending"] = None
        _state["live"] = False
