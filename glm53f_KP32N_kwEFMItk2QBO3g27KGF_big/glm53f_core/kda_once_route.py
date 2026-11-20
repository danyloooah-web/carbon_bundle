from __future__ import annotations
import contextlib
import copy
import logging
import weakref
import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
NODE = "kda"
_BACKEND = kda_backend.KDAAttnBackend
_STOCK_CONVOLVE = _BACKEND._convolve_prefill
_STOCK_TRACK = MambaAttnBackendBase._track_mamba_state_extend
_MASK: dict = {"batch": None, "source": None, "version": None, "mask": None}
_META: dict = {"meta": None, "read": None, "copy": None}
_LIVE = [False]
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA once a forward: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA once a forward: %s", detail)


def prepare(node) -> None:
    
    _require(callable(_STOCK_CONVOLVE) and callable(_STOCK_TRACK), "KDAAttnBackend lacks _convolve_prefill or "
             "_track_mamba_state_extend")

    _require(_BACKEND._track_mamba_state_extend is _STOCK_TRACK,
             "KDAAttnBackend carries its own _track_mamba_state_extend")

def _prefix_mask(forward_batch):
    
    lens = forward_batch.extend_prefix_lens
    if not isinstance(lens, torch.Tensor):
        return lens > 0
    held, source = _MASK["batch"], _MASK["source"]
    if (held is not None and held() is forward_batch and source is not None and source() is lens
            and _MASK["version"] == lens._version):
        return _MASK["mask"]
    mask = lens > 0
    _MASK.update(batch=weakref.ref(forward_batch), source=weakref.ref(lens), version=lens._version, mask=mask)
    _say("mask", "the conv's has_initial_state is made once a forward batch")
    return mask


def _convolve_prefill(self, layer, forward_batch, mixed_qkv, conv_states, cache_indices):
    
    split_size = (
        layer.q_dim
        if self.kernel_dispatcher.prefill_backend.is_flashinfer()
        and layer.q_dim == layer.k_dim == layer.v_dim
        else 0
    )
    conv_output = kda_backend.causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=_prefix_mask(forward_batch),
        cache_indices=cache_indices,
        query_start_loc=self.forward_metadata.query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        output_split_size=split_size,
    )
    if split_size:
        return conv_output.unbind(0)
    return conv_output.transpose(0, 1).split(
        [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
    )


def _read(meta) -> tuple:
    
    return (meta.has_mamba_track_mask, meta.track_ssm_h_src, meta.track_ssm_h_dst, meta.track_ssm_h_batch_src,
            meta.track_ssm_recompute_dst, meta.track_ssm_final_src, meta.track_ssm_final_dst)
def _narrow(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype == torch.int32 and t.is_cuda


def _widened(meta):
    
    read = _read(meta)
    held = _META["meta"]
    if held is not None and held() is meta:
        kept = _META["read"]
        if kept[0] == read[0] and all(a is b and (b is None or a._version == version)
                                      for (a, version), b in zip(kept[1:], read[1:])):
            return meta if _META["copy"] is None else _META["copy"]
    narrow = [t for t in read[1:] if _narrow(t)]
    wide = None
    if narrow:
        wide = copy.copy(meta)
        (wide.track_ssm_h_src, wide.track_ssm_h_dst, wide.track_ssm_h_batch_src, wide.track_ssm_recompute_dst,
         wide.track_ssm_final_src, wide.track_ssm_final_dst) = (
            t.to(torch.int64) if _narrow(t) else t for t in read[1:])
        _say("track", f"the track indices are made int64 once a forward ({len(narrow)} int32 index tensor(s))")
    _META.update(meta=weakref.ref(meta), read=(read[0], *((t, None if t is None else t._version) for t in read[1:])),
                 copy=wide)
    return meta if wide is None else wide


def _track_mamba_state_extend(self, forward_batch, h, ssm_states, forward_metadata, track_states=None, *,
                              h_track_buf=None):
    
    return _STOCK_TRACK(self, forward_batch, h, ssm_states, _widened(forward_metadata), track_states,
                        h_track_buf=h_track_buf)


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if (mode.is_draft_extend_v2() or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False)
            or torch.cuda.is_current_stream_capturing() or get_is_capture_mode()):
        yield
        return
    _require(not _LIVE[0], "already set (a block's call inside another)")
    _LIVE[0] = True
    _BACKEND._convolve_prefill = _convolve_prefill
    _BACKEND._track_mamba_state_extend = _track_mamba_state_extend
    try:
        yield
    finally:
        _BACKEND._convolve_prefill = _STOCK_CONVOLVE
        del _BACKEND._track_mamba_state_extend
        _LIVE[0] = False
