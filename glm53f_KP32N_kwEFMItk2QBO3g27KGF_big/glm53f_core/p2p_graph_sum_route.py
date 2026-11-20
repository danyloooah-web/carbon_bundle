from __future__ import annotations
import contextlib
import torch
from glm53f_core import e31_fused_route as fused
NODE = "layer"
_HIDDEN = 4096
MAX_ROWS = 8192
_SLOT = MAX_ROWS * _HIDDEN // 8
_MISSING = object()
_state = {"checked": False, "why": "", "groups": (), "native": None, "owner": None, "peer": 0, "mine": 0,
          "parity": None, "live": False, "said": set(), "tally": {}}
def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-p2ps] {detail}", flush=True)
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-p2ps] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()

def _load_native():
    import p2p_graph_sum_native

    return p2p_graph_sum_native


def _allocator():

    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import _allocate_symmetric_memory

    return _allocate_symmetric_memory

def prepare(layer) -> None:
    
    if _state["checked"]:
        return
    groups = fused._state["groups"]
    if groups is None:
        raise RuntimeError("p2p graph sum: e31_layer_route's prepare has not run before this component's")
    if not groups:
        _state.update(checked=True, why=f"E31 found no two-rank sum groups ({fused._state['why']})")
        _say("inactive", f"inactive on this engine: {_state['why']}; every sum is stock's")
        return
    if _capturing():
        raise RuntimeError("p2p graph sum: prepare ran under graph capture (the workspace must exist before any capture)")
    native = _load_native()
    if not (callable(getattr(native, "sum", None)) and callable(getattr(native, "layout", None))):
        raise RuntimeError("p2p graph sum: p2p_graph_sum_native lacks sum / layout")
    grid, threads = (int(v) for v in native.layout())
    group = groups[0]
    rank, device = group.rank_in_group, torch.device("cuda", torch.cuda.current_device())
    nbytes = 2 * _SLOT * 16
    owner, slabs, _ = _allocator()(nbytes, device, group.cpu_group)
    if len(slabs) != 2 or any(s.numel() != nbytes for s in slabs):
        raise RuntimeError(f"p2p graph sum: the workspace's rendezvous handed {len(slabs)} buffers")
    slabs[rank].fill_(0xFF)
    torch.cuda.synchronize(device)
    torch.distributed.barrier(group=group.cpu_group)
    _state.update(checked=True, groups=tuple(groups), native=native, owner=(owner, slabs),
                  peer=int(slabs[1 - rank].data_ptr()), mine=int(slabs[rank].data_ptr()),
                  parity=torch.zeros(grid, dtype=torch.int32, device=slabs[rank].device))
    _tally("prepared", lambda: f"rank {rank} of {list(group.ranks)}, {grid} CTAs of {threads} threads, receive slots "
                               f"of 2 x {MAX_ROWS} x {_HIDDEN} BF16 a rank")


def _stock_path(group, x: torch.Tensor):
    
    if group.world_size != 2:
        return "world", f"a group of {group.world_size} ranks"
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
    if method is not None:
        return method, f"SGLang's out-of-place '{method}' all-reduce"
    return None


def _declined(group, x: torch.Tensor):
    
    if fused._state["pending"] is not None:
        return "pending", "E31 holds a pending partial (its own checks decide)"
    if fused._state["live"] and fused._eligible(x):
        return "e31", "E31 sums it into its post"
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == _HIDDEN and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        return "layout", f"a partial {x.dtype} {tuple(x.shape)}"
    if not 1 <= x.shape[0] <= MAX_ROWS:
        return "rows", f"{x.shape[0]} rows, outside 1 to {MAX_ROWS}"
    found = _stock_path(group, x)
    if found is not None:
        return found[0], f"SGLang's all_reduce serves it with {found[1]}"
    return None
def _summing(group, before):
    

    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        why = _declined(group, input_)
        if why is not None:
            _say(why[0], f"the sum it wraps serves a call: {why[1]}")
            return before(input_)
        _state["native"].sum(input_, _state["peer"], _state["mine"], _state["parity"], _SLOT)
        _tally("sum", lambda: f"{input_.shape[0]} rows{', captured' if _capturing() else ''}")
        return input_

    return all_reduce


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if _state["native"] is None:
        yield
        return
    if _state["live"]:
        raise RuntimeError("p2p graph sum: already routed (a layer's call inside another)")
    summed = []
    _state["live"] = True
    try:
        for group in _state["groups"]:


            current = group.all_reduce
            own = not (getattr(current, "__self__", None) is group
                       and getattr(current, "__func__", None) is type(group).all_reduce)
            summed.append((group, current if own else _MISSING))
            group.all_reduce = _summing(group, current)
        yield
    finally:
        for group, kept in reversed(summed):
            if kept is _MISSING:
                del group.all_reduce
            else:
                group.all_reduce = kept
        _state["live"] = False
