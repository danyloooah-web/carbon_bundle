from __future__ import annotations
import contextlib
import logging
import torch
from sglang.kernels.ops.attention import kda_decode_mtp
from glm53f_core import kda_v2_verify
NODE = "kda"
TOKENS = 6
_STOCK = kda_decode_mtp.fused_kda_decode_mtp_dspark
_SET: list = []
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA verify v2: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf KDA verify v2: %s", detail)
def prepare(node) -> None:
    
    _require(callable(_STOCK) and kda_decode_mtp.fused_kda_decode_mtp_dspark is _STOCK,
             "kda_decode_mtp.fused_kda_decode_mtp_dspark is not stock's")
    layer = getattr(node, "attn", None)
    _require(layer is not None and (layer.head_q_dim, layer.head_k_dim, layer.head_v_dim) == (128, 128, 128),
             f"{type(node).__name__}'s chain module is not the verify kernel's 128-wide head form")


def _outside(kw: dict) -> tuple[str, str] | None:
    
    if any(kw.get(name) is None for name in ("onorm_gate", "onorm_weight", "onorm_eps")):
        return "onorm", "no fused output norm (kda_verify_norm_route did not offer it)"
    if kw.get("intermediate_ssm") is not None:
        return "snapshots", "per-step state snapshots are asked for (no ReplaySSM rings)"
    if any(kw.get(name) is None for name in ("replayssm_rawv", "replayssm_rawk", "replayssm_g", "replayssm_beta")):
        return "rings", "the ReplaySSM rings are missing"
    if kw["recurrent_state"].dtype != torch.float32:
        return "state", f"the state pool is {kw['recurrent_state'].dtype}"
    tokens, requests = kw["x_q"].shape[1], kw["cu_seqlens"].numel() - 1
    if requests <= 0 or tokens != requests * TOKENS:
        return "tokens", f"{tokens} tokens for {requests} requests is not {TOKENS} a request"
    return None


def _fused(**kw):
    
    why = _outside(kw)
    if why is not None:
        _say(why[0], f"stock's kernel serves a verify call: {why[1]}")
        return _STOCK(**kw)
    kw.pop("intermediate_ssm")
    out = kda_v2_verify.fused_kda_v2_verify(**kw)
    _say("ours", f"the v2 kernel serves verify calls (first: {kw['x_q'].shape[1]} tokens, "
                 f"{kw['x_q'].shape[2]} heads)")
    return out


@contextlib.contextmanager
def routed(node, forward_batch):
    
    if not forward_batch.forward_mode.is_target_verify():
        yield
        return
    _require(not _SET, "a linear-attention block's verify call inside another's")
    _SET.append(kda_decode_mtp.fused_kda_decode_mtp_dspark)
    kda_decode_mtp.fused_kda_decode_mtp_dspark = _fused
    try:
        yield
    finally:
        kda_decode_mtp.fused_kda_decode_mtp_dspark = _SET.pop()
