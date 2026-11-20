from __future__ import annotations
import contextlib
from glm53f_core import e31_fused_route as fused
NODE = "layer"
def prepare(layer) -> None:
    state = fused._state
    if state["groups"] is not None:
        return
    differs = fused._engine_differences()
    if differs:
        raise RuntimeError(f"E31 fused: this engine is not one this route mirrors ({', '.join(sorted(fused._ENGINES))}); "
                           f"differing: {', '.join(differs)}")
    unique = []
    for group in fused._groups():
        if all(group is not seen for seen in unique):
            unique.append(group)
    ranks = [list(g.ranks) for g in unique]
    if any(g.world_size != 2 for g in unique) or any(r != ranks[0] for r in ranks):
        state.update(groups=(), why=f"sum groups are not one two-rank set (ranks {ranks})")
        fused._tally("inactive", lambda: state["why"])
        return
    native = fused._load_native()
    if not (callable(getattr(native, "stage", None)) and callable(getattr(native, "post", None))):
        raise RuntimeError("E31 fused: e31_fused_native lacks stage / post")
    glm = fused._glm_module()
    if not callable(getattr(glm, "_hc_post_fn", None)):
        raise RuntimeError("E31 fused: glm5_next has no _hc_post_fn to route")
    state.update(groups=tuple(unique), why="", native=native, glm=glm, layers=())
    fused._tally("prepared for a layer's call", lambda: (
        f"{len(unique)} group object(s) over ranks {ranks[0]}, rows {fused._MIN_ROWS}-{fused._MAX_ROWS}, variant "
        f"{fused._VARIANT}{' ftz' if fused._FTZ else ''}"))


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    state = fused._state
    if not state["groups"] or not forward_batch.forward_mode.is_extend() or fused._capturing():
        yield
        return
    if state["live"]:
        raise RuntimeError("E31 fused: already routed (a layer's call inside another)")
    glm = state["glm"]
    stock_post = glm._hc_post_fn
    installed = []
    state["live"] = True
    state["pending"] = None
    state["depth"] = 1
    try:
        for group in state["groups"]:
            group.all_reduce = fused._deferring(group.all_reduce)
            installed.append(group)
        glm._hc_post_fn = fused._posting(stock_post)
        yield
        if state["pending"] is not None:
            raise RuntimeError("E31 fused: the layer returned with a partial still pending (its post never ran)")
    finally:
        glm._hc_post_fn = stock_post
        for group in installed:
            del group.all_reduce
        state["pending"] = None
        state["depth"] = 0
        state["live"] = False
