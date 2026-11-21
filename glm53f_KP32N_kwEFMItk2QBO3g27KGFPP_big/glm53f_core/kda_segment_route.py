from __future__ import annotations
import contextlib
import logging
import time
import weakref
import torch
from sglang.kernels.ops.attention.kda_input_pack import contiguous_kda_inputs
from sglang.srt.layers.attention.linear.kernels import kda_flashinfer_prefill
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_disagg
NODE = "kda"
SEGMENTS = 3
GROWTH = 1.5
MIN_TOKENS = 4096
MAX_STALLS = 4
BT = 16
D = 128
_TVM_FFI = "--enable-tvm-ffi"
_KERNEL = kda_flashinfer_prefill.FlashInferKDAPrefillKernel
_STOCK = _KERNEL.extend
_COMPILED: dict = {}
_STREAMS: dict = {}
_EVENTS: dict = {}
_PLAN: dict = {"key": None, "plan": None}
_FP32: dict = {}
_SLOTS: dict = {"batch": None, "of": None, "int32": None}
_BATCH = [None]
_UNDER_LAST = [None]
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA segments: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA segments: %s", detail)
def _compile(bound: float) -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from flashinfer.kda_kernels import kda_chunked_bt16 as bt16

    from glm53f_core import kda_chain_kernel

    began = time.perf_counter()
    gate = bound * bt16.LOG2_E
    plain = bt16._unified_fakes(cutlass.BFloat16, cutlass.Float32, True, True, False, True, False, cutlass.BFloat16)
    (q, k, v, raw_gate, a_log, dt_bias, beta, cu, _order, _gen_order, slots, cu_chunks, _gen_chunks, kd, qd, w, qk,
     diag, state, out, final), _, _, _ = plain
    prep = cute.compile(bt16.host_prep, q, k, raw_gate, a_log, dt_bias, beta, cu, cu_chunks, kd, qd, w, qk, diag,
                        make_fake_stream(), 0, 0, 0, 0, True, gate, bt16.THREADS_PER_CTA, cutlass.BFloat16,
                        options=_TVM_FFI)
    chain = cute.compile(kda_chain_kernel.host_chain_dv2, v, cu, cu_chunks, kd, qd, w, qk, diag, slots, state, out,
                         final, make_fake_stream(), 0, 0, bt16.DEFAULT_SCALE, None, None, None, 0,
                         bt16.THREADS_PER_CTA, options=_TVM_FFI)
    tracked = bt16._unified_fakes(cutlass.BFloat16, cutlass.Float32, True, True, True, True, True, cutlass.BFloat16)
    (_, _, v, _, _, _, _, cu, _, _, slots, cu_chunks, _, kd, qd, w, qk, diag, state, out, final), ckpt, cu_ckpts, \
        ckpt_slots = tracked
    chain_ckpt = cute.compile(kda_chain_kernel.host_chain_dv2, v, cu, cu_chunks, kd, qd, w, qk, diag, slots, state,
                              out, final, make_fake_stream(), 0, 0, bt16.DEFAULT_SCALE, ckpt, cu_ckpts, ckpt_slots, 0,
                              bt16.THREADS_PER_CTA, options=_TVM_FFI)
    _COMPILED[bound] = (bt16, prep, chain, chain_ckpt)
    _say("compile", f"BT16's prep and the chains with the left-half wait (kda_chain_kernel) compiled apart for gate "
                    f"bound {bound} in {time.perf_counter() - began:.1f} s")


def prepare(node) -> None:
    _require(callable(_STOCK) and callable(contiguous_kda_inputs), "FlashInferKDAPrefillKernel.extend or "
             "contiguous_kda_inputs missing")
    bound = getattr(getattr(node, "attn", None), "lower_bound", None)
    if get_disagg().disaggregation_mode == "decode":
        _say("decode", "a PD decode server runs no eager prefill: nothing compiled, its calls stay stock's")
        return
    if bound is None:
        _say("unbounded", f"{type(node).__name__} has no gate lower bound: FlashInfer's chain does not serve it")
        return
    if float(bound) not in _COMPILED:
        _compile(float(bound))


def _bounds(tokens: int, tracked: int | None) -> tuple[list, tuple | None]:
    
    chunks = -(-tokens // BT)
    weights = [GROWTH ** i for i in range(SEGMENTS)]
    edge, bounds = 0.0, [0]
    for weight in weights[:-1]:
        edge += weight
        bound = round(edge / sum(weights) * chunks) * BT
        if bounds[-1] < bound < tokens:
            bounds.append(bound)
    bounds.append(tokens)
    if tracked is None:
        return bounds, None
    k = next(i for i in range(len(bounds) - 1) if bounds[i] < tracked <= bounds[i + 1])
    interval = tracked - bounds[k]
    if (bounds[k + 1] - bounds[k]) // interval > MAX_STALLS:
        bounds.insert(k + 1, tracked)
    return bounds, (k, interval // BT, (bounds[k + 1] - bounds[k]) // interval)


def _plan(bt16, tokens: int, tracked: int | None, heads: int, device: torch.device) -> tuple:
    key = (device.index, tokens, tracked, heads)
    if _PLAN["key"] == key:
        return _PLAN["plan"]
    bounds, tracked_at = _bounds(tokens, tracked)
    segments, offset = [], 0
    for start, end in zip(bounds, bounds[1:]):
        n = end - start
        chunks = -(-n // BT)
        nbytes = bt16._decomp_ws_bytes(heads, chunks)
        ctas, per_cta = bt16._k1_grid(chunks, heads)
        segments.append((start, end, chunks, torch.arange(0, n + 1, n, dtype=torch.int64, device=device),
                         torch.arange(0, chunks + 1, chunks, dtype=torch.int32, device=device), offset, nbytes,
                         ctas, per_cta))
        offset += nbytes
    checkpoint = None
    if tracked_at is not None:
        k, stride, stops = tracked_at
        rows = torch.full((stops,), -1, dtype=torch.int32, device=device)
        rows[:1].fill_(0)
        checkpoint = (k, stride, torch.arange(0, stops + 1, stops, dtype=torch.int64, device=device), rows)
    plan = (segments, checkpoint, offset)
    _PLAN.update(key=key, plan=plan)
    _say("plan", f"first served call: {tokens} tokens in segments {bounds}"
                 + (f", the snapshot at {tracked} in segment {tracked_at[0]}" if tracked is not None else ""))
    return plan


def _stream(device: torch.device) -> torch.cuda.Stream:
    
    held = _STREAMS.get(device.index)
    if held is None:
        held = _STREAMS[device.index] = torch.cuda.Stream(device=device,
                                                          priority=min(torch.cuda.Stream.priority_range()))
    return held


def _events(device: torch.device, count: int) -> list:
    held = _EVENTS.setdefault(device.index, [])
    while len(held) < count:
        held.append(torch.cuda.Event())
    return held


def _tracked_boundary(tokens: int, checkpoints: int, interval: int) -> int:
    
    batch = _BATCH[0]
    lens, track = batch.extend_seq_lens_cpu, batch.mamba_track_seqlens_cpu
    prefix, mask = batch.extend_prefix_lens_cpu, batch.mamba_prefill_track_mask_cpu
    _require(checkpoints == 1 and interval > 0 and interval % 32 == 0,
             f"a one-sequence call carries {checkpoints} snapshots every {interval} tokens")
    _require(track is not None and prefix is not None and mask is not None and list(lens) == [tokens],
             "the forward batch carries no host tracking lists for its one row")
    relative = track[0] - prefix[0]
    done = relative // interval
    _require(bool(mask[0]) and relative % interval != 0 and 0 < done <= tokens // interval,
             f"stock's plan maps no snapshot for row 0 (tracked {mask[0]}, at {relative} of {tokens} tokens)")
    return done * interval
def _segmented(compiled: tuple, q, k, v, g, beta, a_log, dt_bias, pool, slots, tracked, checkpoints) -> torch.Tensor:
    
    bt16, prep, chain, chain_ckpt = compiled
    device, tokens, heads = q.device, q.shape[1], q.shape[2]
    segments, checkpoint, nbytes = _plan(bt16, tokens, tracked, heads, device)
    out = torch.empty_like(q)
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    dt_bias = dt_bias.reshape(heads, D)
    caller, chains = torch.cuda.current_stream(device), _stream(device)
    n = len(segments)
    events = _events(device, 2 * n + 1)
    ready, prepped, walked = events[0], events[1:n + 1], events[n + 1:2 * n + 1]
    spaces = [bt16._partition_workspace(workspace.narrow(0, offset, size), heads, chunks)
              for _, _, chunks, _, _, offset, size, _, _ in segments]

    def launch_prep(i: int, stream) -> None:
        start, end, _, cu, cu_chunks, _, _, ctas, per_cta = segments[i]
        ws = spaces[i]
        prep(q[:, start:end], k[:, start:end], g[:, start:end], a_log, dt_bias, beta[:, start:end], cu, cu_chunks,
             ws["kd"], ws["qd"], ws["w"], ws["qk"], ws["diag"], stream.cuda_stream, ctas, per_cta, 0, heads)
        prepped[i].record(stream)

    def launch_chain(i: int) -> None:
        start, end, _, cu, cu_chunks, _, _, _, _ = segments[i]
        ws = spaces[i]
        args = (v[:, start:end], cu, cu_chunks, ws["kd"], ws["qd"], ws["w"], ws["qk"], ws["diag"], slots, pool,
                out[:, start:end], pool, chains.cuda_stream, 0, heads, D ** -0.5)
        if checkpoint is not None and checkpoint[0] == i:
            chain_ckpt(*args, checkpoints, checkpoint[2], checkpoint[3], checkpoint[1])
        else:
            chain(*args, None, None, None, 0)
        walked[i].record(chains)

    late = _UNDER_LAST[0](out, segments[-1][0]) if _UNDER_LAST[0] is not None and n > 1 else None
    ready.record(caller)
    chains.wait_event(ready)
    try:
        launch_prep(0, chains)
        launch_chain(0)
        for i in range(1, n):
            caller.wait_event(prepped[0] if i == 1 else walked[i - 2])
            launch_prep(i, caller)
            chains.wait_event(prepped[i])
            launch_chain(i)
        if late is not None:
            caller.wait_event(walked[n - 2])
            late()
    finally:
        caller.wait_stream(chains)
    return out


def _flat32(gate):
    
    if not isinstance(gate, torch.nn.Parameter):
        return gate.reshape(-1).float().contiguous()
    key = (gate.data_ptr(), gate.numel(), gate.dtype)
    flat = _FP32.get(key)
    if flat is None:
        flat = _FP32[key] = gate.detach().reshape(-1).float().contiguous()
    return flat

def _slots32(cache_indices):
    
    if cache_indices.dtype == torch.int32:
        return cache_indices
    batch, tensor = _SLOTS["batch"], _SLOTS["of"]
    if batch is None or batch() is not _BATCH[0] or tensor() is not cache_indices:
        _SLOTS.update({"batch": weakref.ref(_BATCH[0]), "of": weakref.ref(cache_indices),
                       "int32": cache_indices.to(torch.int32)})
    return _SLOTS["int32"]


def _outside(compiled, q, pool, query_start_loc) -> tuple | None:
    
    if compiled is None:
        return "bound", "no chain compiled for this gate bound in this process: its calls are stock's"
    if torch.cuda.is_current_stream_capturing():
        return "capture", "a call under graph capture is stock's"
    if query_start_loc.numel() != 2:
        return "several", "a call of two or more sequences is stock's"
    if pool.dtype != torch.float32:
        return "pool", f"a {pool.dtype} state pool is stock's"
    if q.dim() != 4 or q.shape[0] != 1 or q.shape[3] != D or q.dtype != torch.bfloat16:
        return "inputs", f"inputs {q.dtype} {tuple(q.shape)} are stock's"
    if q.shape[1] < MIN_TOKENS:
        return "short", f"a call under {MIN_TOKENS} tokens is stock's (first: {q.shape[1]})"
    if 2 * q.shape[2] > torch.cuda.get_device_properties(q.device).multi_processor_count:
        return "heads", f"{q.shape[2]} heads take stock's packed engine"
    if list(_BATCH[0].extend_seq_lens_cpu or ()) != [q.shape[1]]:
        return "rows", "a call whose rows are not its batch's one sequence is stock's"
    return None


def _extend(self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, A_log, dt_bias, lower_bound,
            return_intermediate_states=False, state_checkpoint_cu_starts=None, num_state_checkpoints=0,
            state_checkpoint_every_n_tokens=0, state_checkpoint_indices=None, track_ssm_h_batch_src=None, **kwargs):
    
    compiled = None if lower_bound is None else _COMPILED.get(float(lower_bound))
    why = _outside(compiled, q, ssm_states, query_start_loc)
    if why is not None:
        _say(*why)
        return _STOCK(self, q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,
                      query_start_loc=query_start_loc, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                      return_intermediate_states=return_intermediate_states,
                      state_checkpoint_cu_starts=state_checkpoint_cu_starts, num_state_checkpoints=num_state_checkpoints,
                      state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
                      state_checkpoint_indices=state_checkpoint_indices, track_ssm_h_batch_src=track_ssm_h_batch_src,
                      **kwargs)
    needs_checkpoint = return_intermediate_states and kwargs.get("track_state") is not None
    tracked, checkpoints = None, None
    if needs_checkpoint:
        tracked = _tracked_boundary(q.shape[1], num_state_checkpoints, state_checkpoint_every_n_tokens)
        checkpoints = ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
    q, k, v, g, beta = contiguous_kda_inputs(q, k, v, g[:, : q.shape[1]], beta[:, : q.shape[1]])
    output = _segmented(compiled, q, k, v, g, beta, _flat32(A_log), _flat32(dt_bias), ssm_states,
                        _slots32(cache_indices), tracked, checkpoints)
    if needs_checkpoint:
        kwargs["track_state"][track_ssm_h_batch_src] = checkpoints.float()
    return (output, None) if return_intermediate_states else output


@contextlib.contextmanager
def routed(node, forward_batch):
    
    mode = forward_batch.forward_mode
    if (mode.is_draft_extend_v2() or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False)
            or torch.cuda.is_current_stream_capturing() or get_is_capture_mode()):
        yield
        return
    _require(_KERNEL.extend is _STOCK, "FlashInferKDAPrefillKernel.extend is already replaced (kda_eager_route "
             "beside this route, or a block's call inside another)")
    _KERNEL.extend = _extend
    _BATCH[0] = forward_batch
    try:
        yield
    finally:
        _KERNEL.extend = _STOCK
        _BATCH[0] = None
