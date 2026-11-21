from __future__ import annotations
import contextlib
import hashlib
import os
import torch
NODE = "root"
_HIDDEN = 4096
_HC = 4
_MIN_ROWS = 4096
_MAX_ROWS = 16384
_STAGE_GRID = 592
_VARIANT = 1
_FTZ = False
_ENGINES = {
    "314bce7c": {
        "srt/distributed/parallel_state.py": "719134fd60c858df12bb002884e270f26b5d7249ecf7ac1340da710ec5c52d2d",
        "srt/distributed/communication_op.py": "f93114fdfe4583bcb93ce80505476691453b19aec51ec67c9c0207834afe97b9",
        "srt/layers/layer_boundary/ops.py": "7cfebd26331b11e346053d2b3d05bcecbf71a156137c0307bdb181c8dcace248",
        "srt/layers/layer_boundary/output.py": "52162c22b80e0f426c1b9148863262823bfd95679ed5edf6b6a329aa12310cef",
        "srt/layers/layer_boundary/prepare.py": "1a6d490c8f57ed2b756539aac2abac0234a721b45a51754fc3ea035e06fa9171",
        "srt/layers/layer_boundary/exit.py": "1b3dc89f589689afc98bd6b29db843ed797ec5253fb7ef96d5f2a9e1d8387018",
        "srt/layers/layer_boundary/residual/mhc.py": "41aaeccd29786e142da51a2100bfbddb6bc27799c23c7c900a7ffbff513bfea3",
        "srt/layers/moe/utils.py": "264c7a2bdbc88ba132697155a054c03d052b49626850c62d400a43d2acaca435",
        "srt/models/glm5_next.py": "82868ef8dc9d803aec2c4c24d412125b894d6518ee602e9780f64b899171c5f4",
        "kernels/ops/layernorm/mhc.py": "8191428b57592d0374cf0a4110f4829635f92371ad7035adb5ca9c0822cf9ffc",
    },
}
_state = {"groups": None, "why": "", "native": None, "glm": None, "layers": (), "depth": 0, "bufs": None,
          "pending": None, "live": False, "tally": {}}

def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-e31f] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _read(path: str) -> bytes:
    
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _engine_differences() -> list:
    import sglang

    root = os.path.dirname(os.path.abspath(sglang.__file__))
    best = None
    for digests in _ENGINES.values():
        differing = []
        for relative, expected in digests.items():
            try:
                found = hashlib.sha256(_read(os.path.join(root, relative))).hexdigest()
            except OSError:
                found = None
            if found != expected:
                differing.append(relative)
        if best is None or len(differing) < len(best):
            best = differing
    return best


def _groups():
    from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_moe_tp_group, get_tp_group

    return get_tp_group(), get_attn_tp_group(), get_moe_tp_group()


def _load_native():
    import e31_fused_native

    return e31_fused_native


def _glm_module():
    import sglang.srt.models.glm5_next as glm5_next

    return glm5_next


def _symmetric(numel: int, dtype, device, name: str):
    import torch.distributed._symmetric_memory as symm_mem

    buf = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(buf, name)
    return buf, handle


def _setup(device) -> dict:
    
    group = _state["groups"][0]
    name = group.device_group.group_name
    peer = 1 - group.rank_in_group
    stage, hs = _symmetric(_MAX_ROWS * _HIDDEN, torch.bfloat16, device, name)
    flags, hf = _symmetric(2, torch.int32, device, name)
    flags.zero_()
    torch.cuda.synchronize(device)
    group.barrier()
    _tally("buffers set up", lambda: f"rank {group.rank_in_group}, stage {_MAX_ROWS} x {_HIDDEN} BF16, variant "
                                     f"{_VARIANT}{' ftz' if _FTZ else ''}")
    return {"stage": stage, "peer_stage": int(hs.buffer_ptrs[peer]), "flags": flags,
            "peer_flags": int(hf.buffer_ptrs[peer]), "state": torch.zeros(3, dtype=torch.int32, device=device),
            "epoch": 0}


def prepare(model) -> None:
    
    differs = _engine_differences()
    if differs:
        raise RuntimeError(f"E31 fused: this engine is not one this route mirrors ({', '.join(sorted(_ENGINES))}); "
                           f"differing: {', '.join(differs)}")
    unique = []
    for group in _groups():
        if all(group is not seen for seen in unique):
            unique.append(group)
    ranks = [list(g.ranks) for g in unique]
    if any(g.world_size != 2 for g in unique) or any(r != ranks[0] for r in ranks):
        _state.update(groups=(), why=f"sum groups are not one two-rank set (ranks {ranks})")
        _tally("inactive", lambda: _state["why"])
        return
    native = _load_native()
    if not (callable(getattr(native, "stage", None)) and callable(getattr(native, "post", None))):
        raise RuntimeError("E31 fused: e31_fused_native lacks stage / post")
    glm = _glm_module()
    if not callable(getattr(glm, "_hc_post_fn", None)):
        raise RuntimeError("E31 fused: glm5_next has no _hc_post_fn to route")



    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise RuntimeError("E31 fused: the model node has no decoder layers")
    shadowed = [i for i, layer in enumerate(layers)
                if getattr(layer.forward, "__func__", None) is not type(layer).forward]
    if shadowed:
        raise RuntimeError(f"E31 fused: decoder layers {shadowed[:4]} already carry their own forward")
    _state.update(groups=tuple(unique), why="", native=native, glm=glm, layers=layers)
    _tally("prepared", lambda: f"{len(unique)} group object(s) over ranks {ranks[0]}, rows {_MIN_ROWS}-{_MAX_ROWS}, "
                               f"variant {_VARIANT}{' ftz' if _FTZ else ''}")


def _on_gpu(x: torch.Tensor) -> bool:
    return x.is_cuda


def _capturing() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.cuda.is_current_stream_capturing() or bool(get_is_capture_mode())


def _eligible(x: torch.Tensor) -> bool:
    
    return (_on_gpu(x) and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == _HIDDEN and x.is_contiguous()
            and _MIN_ROWS <= x.shape[0] <= _MAX_ROWS and not _capturing())
def _deferring(stock):
    

    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        if _state["pending"] is not None:
            raise RuntimeError("E31 fused: a sum was asked for while another partial is pending (its post never ran)")
        if _state["depth"] == 0:
            _tally("stock sum outside the decoder layers", lambda: f"{tuple(input_.shape)} {input_.dtype}")
            return stock(input_)
        if not _eligible(input_):
            _tally("stock sum", lambda: f"{tuple(input_.shape)} {input_.dtype}")
            return stock(input_)
        if _state["bufs"] is None:
            _state["bufs"] = _setup(input_.device)
        b = _state["bufs"]
        b["epoch"] += 1
        _state["native"].stage(input_, b["stage"], b["peer_stage"], b["flags"], b["peer_flags"], b["state"],
                               b["epoch"], _STAGE_GRID)
        _state["pending"] = (input_, b["epoch"])
        return input_

    return all_reduce

def _posting(stock):
    

    def hc_post(x, residual, h_post, h_res, hc_mult):
        pending = _state["pending"]
        if pending is None:
            return stock(x=x, residual=residual, h_post=h_post, h_res=h_res, hc_mult=hc_mult)
        partial, epoch = pending
        if x is not partial:
            raise RuntimeError("E31 fused: the post received another tensor while a partial is pending (the sum's "
                               "result was moved or replaced before its post)")
        rows = x.shape[0]
        if (hc_mult != _HC or residual.dtype != torch.bfloat16 or not residual.is_contiguous()
                or residual.numel() != rows * _HC * _HIDDEN or h_post.dtype != torch.float32
                or not h_post.is_contiguous() or h_post.numel() != rows * _HC or h_res.dtype != torch.float32
                or not h_res.is_contiguous() or h_res.numel() != rows * _HC * _HC):
            raise RuntimeError(f"E31 fused: an mHC post this kernel does not mirror (hc_mult {hc_mult}, residual "
                               f"{residual.dtype} {tuple(residual.shape)}, h_post {h_post.dtype} {tuple(h_post.shape)}, "
                               f"h_res {h_res.dtype} {tuple(h_res.shape)})")
        b = _state["bufs"]
        out = torch.empty((rows, _HC * _HIDDEN), dtype=torch.bfloat16, device=x.device)
        _state["native"].post(partial, residual, h_post, h_res, out, b["stage"], b["peer_stage"], b["flags"],
                              b["peer_flags"], b["state"], epoch, _VARIANT, _FTZ)
        _state["pending"] = None
        _tally("fused sum + post", lambda: f"{rows} rows")
        return out

    return hc_post

def _in_layer(layer):
    
    stock = type(layer).forward

    def forward(*args, **kwargs):
        _state["depth"] += 1
        try:
            return stock(layer, *args, **kwargs)
        finally:
            _state["depth"] -= 1

    return forward


@contextlib.contextmanager
def routed(model, forward_batch):
    
    if not _state["groups"] or not forward_batch.forward_mode.is_extend() or _capturing():
        yield
        return
    if _state["live"]:
        raise RuntimeError("E31 fused: already routed (a nested route)")
    glm = _state["glm"]
    stock_post = glm._hc_post_fn
    installed, counted = [], []
    _state["live"] = True
    _state["pending"] = None
    _state["depth"] = 0
    try:
        for group in _state["groups"]:
            group.all_reduce = _deferring(group.all_reduce)
            installed.append(group)
        for layer in _state["layers"]:
            layer.forward = _in_layer(layer)
            counted.append(layer)
        glm._hc_post_fn = _posting(stock_post)
        yield
        if _state["pending"] is not None:
            raise RuntimeError("E31 fused: the forward returned with a partial still pending (its post never ran)")
    finally:
        glm._hc_post_fn = stock_post
        for layer in counted:
            del layer.forward
        for group in installed:
            del group.all_reduce
        _state["pending"] = None
        _state["depth"] = 0
        _state["live"] = False
