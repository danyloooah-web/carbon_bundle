from __future__ import annotations
import contextlib
import hashlib
import logging
import os
import weakref
import torch
from sglang.kernels.ops.mamba import causal_conv1d_triton
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import FlashInferKDAPrefillKernel
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
NODE = "kda"
CONV_VARIANT = 8
_CONV_PIN = "6c05f0534a3330ccfddb75378a8e13a2d08631f80eb77108473f060dbe2b217d"
_CONV_FORM = "width 4, SiLU, no bias"
_STOCK_CONV = kda_backend.causal_conv1d_fn
_NATIVE: list = [None]
_SHARED: dict = {"batch": None, "source": None, "version": None, "int64": None}
_SAID: set = set()
_TALLY: dict = {}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA launch: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA launch: %s", detail)


def _tally(route: str, describe=None) -> None:
    calls = _TALLY[route] = _TALLY.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glmf-conv] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)


def _sha256(path: str) -> str:
    digest, fd = hashlib.sha256(), os.open(path, os.O_RDONLY)
    try:
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(fd)


def prepare(node) -> None:
    _require(_STOCK_CONV is causal_conv1d_triton.causal_conv1d_fn,
             "the KDA backend's causal_conv1d_fn is not kernels.ops.mamba.causal_conv1d_triton's")
    _require(callable(getattr(causal_conv1d_triton, "_causal_conv1d_fwd_kernel", None)),
             "causal_conv1d_triton has no _causal_conv1d_fwd_kernel")
    if _NATIVE[0] is not None:
        return
    found = _sha256(causal_conv1d_triton.__file__)
    _require(found == _CONV_PIN, f"causal_conv1d_triton.py is not SGLang 314bce7c's (sha256 {found[:16]}): "
                                 "kda_conv_native reproduces that file's launch")
    import kda_conv_native

    rows = kda_conv_native.variants()
    names = [row[1] for row in rows]
    _require(0 <= CONV_VARIANT < len(rows) and not rows[CONV_VARIANT][7],
             f"kda_conv_native variant {CONV_VARIANT} is not a compiled fixed form (compiled: {names})")
    _NATIVE[0] = kda_conv_native
    _tally("prepared", lambda: f"conv kernel variant {CONV_VARIANT} ({names[CONV_VARIANT]})")


def _conv(x, weight, bias, conv_states, query_start_loc, seq_lens_cpu, cache_indices=None, has_initial_state=None,
          activation="silu", pad_slot_id=causal_conv1d_triton.PAD_SLOT_ID, validate_data=False, block_table=None,
          output_split_size=0, **kwargs):
    if (kwargs or block_table is not None or validate_data or conv_states is None or x.dim() != 2
            or not seq_lens_cpu or output_split_size <= 0 or x.shape[0] % output_split_size
            or conv_states.dim() != 3 or conv_states.shape[1] != x.shape[0]
            or weight.shape[1] - 1 > conv_states.shape[2] or torch.cuda.is_current_stream_capturing()):
        _tally("conv: stock (another call form)")
        return _STOCK_CONV(x, weight, bias, conv_states, query_start_loc, seq_lens_cpu, cache_indices=cache_indices,
                           has_initial_state=has_initial_state, activation=activation, pad_slot_id=pad_slot_id,
                           validate_data=validate_data, block_table=block_table,
                           output_split_size=output_split_size, **kwargs)
    silu = activation is True or activation in ("silu", "swish")
    _require(bias is None and weight.dim() == 2 and weight.shape[1] == 4 and silu,
             f"the KDA conv kernel serves {_CONV_FORM}; this packed call has bias {bias is not None}, weight "
             f"{tuple(weight.shape)}, activation {activation!r}")
    native = _NATIVE[0]
    _require(native is not None, "the packed conv was called before prepare imported kda_conv_native")
    dim, total_tokens = x.shape
    out = torch.empty((dim // output_split_size, total_tokens, output_split_size), dtype=x.dtype, device=x.device)
    native.conv(CONV_VARIANT, x, weight, conv_states, query_start_loc, cache_indices, has_initial_state, out,
                len(seq_lens_cpu), max(seq_lens_cpu), pad_slot_id)
    _tally(f"conv cuda variant {CONV_VARIANT}",
           lambda: f"{dim} channels x {total_tokens} tokens, split {output_split_size}")
    return out


def _int64_once(offsets: torch.Tensor, batch) -> torch.Tensor:
    held, source = _SHARED["batch"], _SHARED["source"]
    if (held is not None and held() is batch and source is not None and source() is offsets
            and _SHARED["version"] == offsets._version):
        return _SHARED["int64"]
    copy = offsets.to(torch.int64)
    _SHARED.update(batch=weakref.ref(batch), source=weakref.ref(offsets), version=offsets._version, int64=copy)
    return copy


def _one_plan(stock, batch):

    def recurrent_kda(*args, **kwargs):
        offsets = kwargs.get("cu_seqlens")
        if isinstance(offsets, torch.Tensor) and offsets.dtype == torch.int32 and offsets.is_cuda:
            kwargs["cu_seqlens"] = _int64_once(offsets, batch)
            _say("plan", "stock's KDA chain gets one int64 offsets tensor a forward")
        return stock(*args, **kwargs)

    return recurrent_kda


def _prefill_kernel():
    linear = getattr(get_attn_backend(), "linear_attn_backend", None)
    kernel = getattr(getattr(linear, "kernel_dispatcher", None), "extend_kernel", None)
    if type(kernel) is not FlashInferKDAPrefillKernel:
        _say("kernel", f"the KDA prefill kernel is {type(kernel).__name__}: stock's chain keeps its own offsets")
        return None
    return kernel


@contextlib.contextmanager
def routed(node, forward_batch):
    mode = forward_batch.forward_mode
    if (mode.is_draft_extend_v2() or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False)
            or torch.cuda.is_current_stream_capturing() or get_is_capture_mode()):
        yield
        return
    kernel = _prefill_kernel()
    entry = None if kernel is None else kernel._recurrent_kda
    kda_backend.causal_conv1d_fn = _conv
    if kernel is not None:
        kernel._recurrent_kda = _one_plan(entry, forward_batch)
    try:
        yield
    finally:
        kda_backend.causal_conv1d_fn = _STOCK_CONV
        if kernel is not None:
            kernel._recurrent_kda = entry
