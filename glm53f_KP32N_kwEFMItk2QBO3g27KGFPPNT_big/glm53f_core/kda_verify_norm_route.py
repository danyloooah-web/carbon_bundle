from __future__ import annotations
import contextlib
import inspect
import logging
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.models import glm5_next
NODE = "kda"
D = 128
_BLOCK = glm5_next.Glm5NextLinearAttention
_BACKEND = kda_backend.KDAAttnBackend
_STOCK_FUSED = _BLOCK.forward_qkvbfg_fused
_STOCK_PLAIN = _BLOCK.forward_qkvbfg
_STOCK_RUN = _BACKEND._run_dspark_cutedsl_mtp
_STOCK_RUN_SLOT = inspect.getattr_static(_BACKEND, "_run_dspark_cutedsl_mtp")
_STOCK_NORM = FusedRMSNormGated.forward
_ARGS: dict = {}
_NORMS: dict = {}
_GATE: dict = {}
_DONE: dict = {}
_LIVE = [False]
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA verify norm: {why}")


def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf KDA verify norm: %s", detail)


def prepare(node) -> None:
    _require(type(node) is _BLOCK, f"the block is {type(node).__name__}")
    norm = node.o_norm
    _require(type(norm) is FusedRMSNormGated and norm.activation == "sigmoid" and norm.bias is None
             and norm.weight is not None and tuple(norm.weight.shape) == (D,) and node.head_dim == D,
             f"o_norm is not the sigmoid-gated RMSNorm over one {D}-wide head ({type(norm).__name__})")
    _require(_BLOCK.forward_qkvbfg_fused is _STOCK_FUSED and _BLOCK.forward_qkvbfg is _STOCK_PLAIN
             and _BACKEND._run_dspark_cutedsl_mtp is _STOCK_RUN and FusedRMSNormGated.forward is _STOCK_NORM,
             "a method this route sets is not stock's")
    _NORMS[id(norm)] = node.attn
    _ARGS[id(node.attn)] = (None,) * 5 + (norm.weight.data.float().contiguous(), float(norm.eps))

def _offering(stock):
    def project(self, hidden_states, forward_batch):
        out = stock(self, hidden_states, forward_batch)
        _GATE[id(self.attn)] = out[3]
        return out
    return project


_FUSED, _PLAIN = _offering(_STOCK_FUSED), _offering(_STOCK_PLAIN)


def _run(**kw):
    
    layer = kw["layer"]
    _require(id(layer) in _ARGS, "the verify kernel ran for a chain module this route did not prepare")
    gate = _GATE.pop(id(layer), None)
    rows = kw["mixed_qkv"].shape[0]
    _require(gate is not None and gate.shape[0] >= rows, "no output-norm gate of the call's rows was offered")
    stock_args = layer._k3_fused_decode_args
    layer._k3_onorm_gate, layer._k3_onorm_consumed = gate[:rows], False
    layer._k3_fused_decode_args = _ARGS[id(layer)]
    try:
        out = _STOCK_RUN(**kw)
    finally:
        layer._k3_onorm_gate, layer._k3_fused_decode_args = None, stock_args
    _require(layer._k3_onorm_consumed is True, "the verify kernel did not take the output norm")
    _DONE[id(layer)] = out
    return out


def _norm(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    
    attn = _NORMS.get(id(self))
    done = None if attn is None else _DONE.pop(id(attn), None)
    if done is None:
        if attn is not None:
            _GATE.pop(id(attn), None)
            _say("stock", "a target-verify call ran another kernel than the CuTe verify: stock's o_norm after it")
        return _STOCK_NORM(self, x, g, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)
    _require(residual is None and not prenorm, "o_norm called with a residual")
    rows = x.shape[1] if x.dim() == 4 else -1
    _require(x.shape[2:] == done.shape[2:] and x.shape[0] == 1 and (
        x.data_ptr() == done.data_ptr() if rows == done.shape[1] else rows > done.shape[1]),
        "o_norm's input is not the normed kernel output (or it with stock's pad rows)")
    return x


@contextlib.contextmanager
def routed(node, forward_batch):
    
    if not forward_batch.forward_mode.is_target_verify():
        yield
        return
    _require(not _LIVE[0], "already set (a block's call inside another)")
    _LIVE[0] = True
    _GATE.clear(), _DONE.clear()
    _BLOCK.forward_qkvbfg_fused, _BLOCK.forward_qkvbfg = _FUSED, _PLAIN
    _BACKEND._run_dspark_cutedsl_mtp = staticmethod(_run)
    FusedRMSNormGated.forward = _norm
    _say("on", "target-verify calls apply each KDA layer's gated output norm inside the CuTe verify kernel")
    try:
        yield
    finally:
        _BLOCK.forward_qkvbfg_fused, _BLOCK.forward_qkvbfg = _STOCK_FUSED, _STOCK_PLAIN
        _BACKEND._run_dspark_cutedsl_mtp = _STOCK_RUN_SLOT
        FusedRMSNormGated.forward = _STOCK_NORM
        _GATE.clear(), _DONE.clear()
        _LIVE[0] = False
