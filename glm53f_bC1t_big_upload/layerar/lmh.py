import copy
import logging
import time
import types

import torch
import triton
import triton.language as tl

LMH_MODE = 2
LMH_FALLBACK_MODE1 = True
LMH_GEMM_OUT = True
LMH_MAX_ROWS = 768
LMH_BLOCK = 1024
LMH_WARPS = 4
LMH_PULL_ADDR = "delta"
LMH_TIMEOUT_MS = 5000
LMH_PROOF_SEED = 20261113
LMH_EDGE_ROWS = (6, 48, 96, 384, 576, 624, 768)
LMH_SKEW_S = 0.0005
LMH_MEM_MARGIN = 2 << 30
LMH_EMPTY_CACHE = True
LMH_WORLD = 4
LMH_PDL = False

_LMH_LOG = logging.getLogger(__name__)
_LMH_LOG.setLevel(logging.WARNING)


def _lmh_need(ok, what):
    if not ok:
        raise RuntimeError(f"lmh-gather precondition failed: {what}")


def _lmh_capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _lmh_sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _lmh_free(dev):
    return torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 1 << 62


def _lmh_pdl():
    return {"PDL": False}



@triton.jit
def _lmh_pull_kernel(p0, p1, p2, p3, out_ptr, S, out_stride,
                     RANK: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    k = tl.program_id(0)
    cb = tl.program_id(1)
    row = tl.program_id(2).to(tl.int64)
    r = (k + RANK) % W
    if PDL:
        tl.extra.cuda.gdc_wait()
    if r == 0:
        src = p0
    elif r == 1:
        src = p1
    elif r == 2:
        src = p2
    else:
        src = p3
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    m = cols < S
    x = tl.load(src + row * S + cols, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * out_stride + r * S + cols, x.to(tl.float32), mask=m)


@triton.jit
def _lmh_pull_kernel_d(stage_ptr, delta_ptr, out_ptr, S, out_stride,
                       RANK: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    k = tl.program_id(0)
    cb = tl.program_id(1)
    row = tl.program_id(2).to(tl.int64)
    r = (k + RANK) % W
    if PDL:
        tl.extra.cuda.gdc_wait()
    d = tl.multiple_of(tl.load(delta_ptr + r), 8)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    m = cols < S
    x = tl.load(stage_ptr + d + row * S + cols, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * out_stride + r * S + cols, x.to(tl.float32), mask=m)


class _LmhExchange:

    def __init__(self, rank, world, S, max_rows, peers, sync, owner=None):
        _lmh_need(world == LMH_WORLD and 0 <= rank < world and len(peers) == world, "exchange world / rank")
        for p in peers:
            _lmh_need(p.dtype == torch.bfloat16 and tuple(p.shape) == (max_rows, S) and p.is_contiguous()
                      and p.data_ptr() % 16 == 0, f"stage layout {tuple(p.shape)} {p.dtype}")
        self.rank, self.world, self.S, self.max_rows = rank, world, S, max_rows
        self.peers, self.stage, self.sync, self.owner = peers, peers[rank], sync, owner
        gaps = [p.data_ptr() - peers[rank].data_ptr() for p in peers]
        _lmh_need(all(g % 16 == 0 for g in gaps), f"stage byte offsets {gaps} are not 16-byte multiples")
        self.delta = torch.tensor([g // 2 for g in gaps], dtype=torch.int64, device=peers[rank].device)

    def pull(self, out, rows):
        S = self.S
        _lmh_need(0 < rows <= self.max_rows and out.dtype == torch.float32 and out.dim() == 2
                  and tuple(out.shape) == (rows, self.world * S) and out.stride(1) == 1
                  and out.stride(0) >= self.world * S and out.data_ptr() % 16 == 0, "pull output layout")
        grid = (self.world, triton.cdiv(S, LMH_BLOCK), rows)
        if LMH_PULL_ADDR == "delta":
            _lmh_pull_kernel_d[grid](self.stage, self.delta, out, S, out.stride(0),
                                     RANK=self.rank, W=self.world, BLOCK=LMH_BLOCK, num_warps=LMH_WARPS,
                                     **_lmh_pdl())
        else:
            _lmh_need(LMH_PULL_ADDR == "peer", f"LMH_PULL_ADDR {LMH_PULL_ADDR!r}")
            p = self.peers
            _lmh_pull_kernel[grid](p[0], p[1], p[2], p[3], out, S, out.stride(0),
                                   RANK=self.rank, W=self.world, BLOCK=LMH_BLOCK, num_warps=LMH_WARPS,
                                   **_lmh_pdl())
        return out


def _lmh_build_exchange(st, dev):
    tp, W, S, n = st.tp, st.W, st.S, LMH_MAX_ROWS * st.S
    buf, why = None, None
    try:
        from torch._C._distributed_c10d import _SymmetricMemory
        buf = _SymmetricMemory.empty_strided_p2p((n,), [1], torch.bfloat16, dev, tp.cpu_group.group_name)
    except Exception as exc:
        why = f"symmetric allocation: {type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    if bad:
        return None, f"rank {bad[0][0]} {bad[0][1]} (voted before the rendezvous: no rank entered it)"
    try:
        hdl = _SymmetricMemory.rendezvous(buf)
        if hdl.world_size != W or hdl.rank != st.rank:
            return None, f"symmetric handle rank {hdl.rank}/{hdl.world_size} is not TP rank {st.rank}/{W}"
        off = buf.data_ptr() - int(hdl.buffer_ptrs[st.rank])
        if off < 0 or off % 2:
            return None, f"stage offset {off} inside its symmetric block"
        peers = [hdl.get_buffer(r, [n], torch.bfloat16, off // 2).view(LMH_MAX_ROWS, S) for r in range(W)]
        if peers[st.rank].data_ptr() != buf.data_ptr():
            return None, "this rank's mapped stage is not the allocated tensor"

        def sync(channel):
            hdl.barrier(channel=channel, timeout_ms=LMH_TIMEOUT_MS)

        return _LmhExchange(st.rank, W, S, LMH_MAX_ROWS, peers, sync, owner=(buf, hdl)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


_LMH_BUILD = [_lmh_build_exchange]



def _lmh_sg():
    from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
    from sglang.srt.environ import envs
    from sglang.srt.layers import logits_processor as lp
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.runtime_context import LoRABatchLayout, get_disagg, get_forward, get_spec
    return types.SimpleNamespace(lp=lp, ps=ps, envs=envs, ForwardBatch=ForwardBatch, LoRABatchLayout=LoRABatchLayout,
                                 get_disagg=get_disagg, get_forward=get_forward, get_spec=get_spec,
                                 gathered_softcap_logits=gathered_softcap_logits,
                                 is_symmetric_memory_enabled=is_symmetric_memory_enabled)


def _lmh_reach(module, sg, tp):
    if type(module) is not sg.lp.LogitsProcessor:
        return f"the node is {type(module).__name__}, not stock LogitsProcessor"
    role = sg.get_disagg().disaggregation_mode
    role = str(getattr(role, "value", role))
    if tp.world_size != LMH_WORLD or role != "decode":
        return f"tp {tp.world_size} role {role} (served only on the TP{LMH_WORLD} decode engine)"
    return None


def _lmh_static_gap(module, sg):
    if not (module.do_tensor_parallel_all_gather and not module.use_attn_tp_group
            and not module.do_tensor_parallel_all_gather_dp_attn and not module.use_tp_lm_head_all_to_all):
        return "the head is not the plain TP vocab gather (DP-attention / attention-TP / all-to-all LM head)"
    if module.final_logit_softcapping is not None or module.logit_scale is not None:
        return "final_logit_softcapping or logit_scale is set (stock would fuse them)"
    if module.use_fp32_lm_head or module.use_fp32_lm_head_output or module.rl_on_policy_target is not None:
        return "an fp32 / RL on-policy LM head"
    if module.return_full_logits:
        return "return_full_logits"
    if sg.envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
        return "SGLANG_TRACE_LOGITS_E2E_SYNC synchronizes inside the head"
    if sg.is_symmetric_memory_enabled():
        return "symmetric-memory NCCL allocation is on (stock's gather would register its buffers)"
    n = sg.get_spec().speculative_num_draft_tokens
    if not (type(n) is int and n >= 1 and LMH_MAX_ROWS % n == 0):
        return f"speculative_num_draft_tokens {n!r}"
    return None


def prepare_logits(module):
    sg = _lmh_sg()
    stock_forward = type(module).forward
    wrapped = getattr(module.forward, "__wrapped__", None)
    if wrapped is not None:
        _lmh_need(getattr(wrapped, "__func__", None) is stock_forward and getattr(wrapped, "__self__", None) is module,
                  "the forward the dispatcher wraps is not the class forward bound to this module")
    tp = sg.ps.get_tp_group()
    st = types.SimpleNamespace(module=module, sg=sg, tp=tp, stock_forward=stock_forward,
                               stock_get_logits=type(module)._get_logits, W=tp.world_size, rank=tp.rank_in_group,
                               N=0, S=0, H=0, V=0, gap=None, armed=False, proved=False, proven={}, ex=None,
                               lm_ok=None, said=set())
    reach = f"switched off (LMH_MODE {LMH_MODE!r})" if LMH_MODE not in (1, 2) else _lmh_reach(module, sg, tp)
    if reach is not None:
        _lmh_say(st, f"off ({reach}); every call is stock LogitsProcessor.forward")
        return st
    st.armed, st.gap = True, _lmh_static_gap(module, sg)
    if st.gap is None:
        st.N = sg.get_spec().speculative_num_draft_tokens
    _lmh_say(st, f"armed: mode {LMH_MODE} (fallback mode 1: {LMH_FALLBACK_MODE1}); static gap {st.gap}; proof at the "
                 f"first eager target-verify call; served only inside its CUDA-graph captures")
    return st


def _lmh_say(st, msg):
    if msg not in st.said:
        st.said.add(msg)
        _LMH_LOG.warning("lmh-gather tp%d/%d: %s", st.rank, st.W, msg)


def _lmh_head_gap(st, lm_head):
    w = getattr(lm_head, "weight", None)
    key = (id(lm_head), None if w is None else w.data_ptr())
    if st.lm_ok is not None and st.lm_ok[0] == key:
        return st.lm_ok[1]
    gap = None
    lp, si = st.sg.lp, getattr(lm_head, "shard_indices", None)
    if not isinstance(lm_head, lp.VocabParallelEmbedding) or w is None or si is None:
        gap = f"lm_head {type(lm_head).__name__} is not a VocabParallelEmbedding with a weight"
    elif hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        gap = "a LoRA lm_head"
    elif lp.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None)):
        gap = "a quantized lm_head (stock's quant_method.apply branch)"
    elif not (w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous()):
        gap = f"lm_head weight {w.dtype} {tuple(w.shape)}"
    else:
        S = w.shape[0]
        if not (si.org_vocab_start_index == st.rank * S and si.num_org_elements == S and si.num_elements_padded == S
                and si.num_added_elements_padded == 0 and S % 8 == 0 and st.W * S == st.module.vocab_size):
            gap = (f"vocab shard [{si.org_vocab_start_index}, {si.org_vocab_end_index}) of {S} rows, padded "
                   f"{si.num_elements_padded}, vocab {st.module.vocab_size}: not rank-ordered unpadded W*S")
    st.lm_ok = (key, gap)
    if gap is None:
        st.S, st.H, st.V = w.shape[0], w.shape[1], st.W * w.shape[0]
    return gap



def _lmh_logits(st, mod, mode, hidden_states, lm_head, md, use_logits_buffer):
    sg = st.sg
    hidden_states, _local = mod._gather_dp_attn_hidden_states(hidden_states, md)
    T, W, S = hidden_states.shape[0], st.W, st.S
    ex = st.ex if mode == 2 else None
    _lmh_need(mode in (1, 2) and (mode == 1 or ex is not None) and 0 < T <= LMH_MAX_ROWS, f"mode {mode} rows {T}")
    with sg.get_forward().scoped(lora_batch_layout=sg.LoRABatchLayout.DP_LOCAL):
        if ex is not None:
            ex.sync(0)
            stage = ex.stage[:T]
            logits = mod._compute_lm_head(hidden_states, lm_head, out=stage if LMH_GEMM_OUT else None)
        else:
            logits = mod._compute_lm_head(hidden_states, lm_head)
    _lmh_need(logits.dtype == torch.bfloat16 and tuple(logits.shape) == (T, S) and logits.is_contiguous(),
              f"lm_head output {logits.dtype} {tuple(logits.shape)} (expected bf16 [{T}, {S}])")
    if ex is not None:
        if LMH_GEMM_OUT:
            _lmh_need(logits.data_ptr() == stage.data_ptr(), "the lm_head GEMM did not write the stage (out=)")
        else:
            stage.copy_(logits)
    buf = md.next_token_logits_buffer if use_logits_buffer else None
    if buf is not None and tuple(buf.shape) == (T, W * S):
        _lmh_need(buf.dtype == torch.float32 and buf.stride(1) == 1, "next_token_logits_buffer layout")
        out = buf
    else:
        out = torch.empty((T, W * S), dtype=torch.float32, device=logits.device)
    if ex is not None:
        ex.sync(1)
        return ex.pull(out, T)
    gathered = torch.empty((W * T, S), dtype=logits.dtype, device=logits.device)
    st.tp.all_gather_into_tensor(gathered, logits)
    return sg.gathered_softcap_logits(gathered.view(W, T, S), None, out)


def _lmh_get_logits(st, self, hidden_states, lm_head, logits_metadata, embedding_bias=None, use_logits_buffer=True):
    T = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
    mode = st.proven.get(T)
    if (mode is None or embedding_bias is not None or self.sharded_verify or self.defer_verify_gather
            or not logits_metadata.forward_mode.is_target_verify()):
        return st.stock_get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias, use_logits_buffer)
    gap = _lmh_head_gap(st, lm_head)
    _lmh_need(gap is None and hidden_states.dtype == torch.bfloat16 and hidden_states.shape[1] == st.H,
              f"proven rows {T}: {gap or 'hidden states ' + str(hidden_states.dtype)}")
    return _lmh_logits(st, self, mode, hidden_states, lm_head, logits_metadata, use_logits_buffer)


def forward_logits(st, *args, **kwargs):
    mod = st.module
    if st.armed:
        fb = args[3] if len(args) > 3 else kwargs.get("logits_metadata")
        if isinstance(fb, st.sg.ForwardBatch) and fb.forward_mode.is_target_verify():
            if _lmh_capturing():
                if st.proven:
                    proxy = copy.copy(mod)

                    def bound(self, *a, **k):
                        return _lmh_get_logits(st, self, *a, **k)

                    object.__setattr__(proxy, "_get_logits", types.MethodType(bound, proxy))
                    return st.stock_forward(proxy, *args, **kwargs)
                if not st.proved:
                    st.proved = True
                    _lmh_say(st, "off (the first target-verify call was already a capture: no proof point); stock")
            elif not st.proved:
                st.proved = True
                _lmh_prove(st, fb, args, kwargs)
    return st.stock_forward(mod, *args, **kwargs)



def _lmh_families(T, H, real, gen, dev, edge):
    fams = [("randn", torch.randn(T, H, generator=gen, device=dev).to(torch.bfloat16))]
    if not edge:
        return fams
    if torch.is_tensor(real) and real.dim() == 2 and real.shape[1] == H and real.shape[0] > 0:
        idx = torch.arange(T, device=dev) % real.shape[0]
        fams.append(("real", real.to(torch.bfloat16).index_select(0, idx).contiguous()))
    e = torch.randn(T, H, generator=gen, device=dev) * 16
    e[0::7] = 0.0
    e[1::7] = -0.0
    e[2::7] *= 1e36
    e[3::7] *= 1e-39
    e[4::7, ::5] = float("nan")
    fams.append(("edge", e.to(torch.bfloat16)))
    return fams


def _lmh_same(ref, got):
    return (tuple(ref.shape) == tuple(got.shape) and ref.dtype == got.dtype == torch.float32
            and torch.equal(ref.view(torch.int32), got.view(torch.int32)))


def _lmh_prove(st, fb, args, kwargs):
    t0 = time.perf_counter()
    mod, tp, sg = st.module, st.tp, st.sg
    lm_head = args[2] if len(args) > 2 else kwargs.get("lm_head")
    real = args[1] if len(args) > 1 else kwargs.get("hidden_states")
    gap = st.gap or _lmh_head_gap(st, lm_head)
    dev = lm_head.weight.device if gap is None else torch.device("cpu")
    want2 = LMH_MODE == 2 and gap is None and _lmh_free(dev) >= LMH_MAX_ROWS * st.S * 2 + LMH_MEM_MARGIN
    votes = tp.all_gather_object((gap, want2))
    gaps = [g for g, _ in votes if g is not None]
    if gaps:
        _lmh_say(st, f"off ({gaps[0]}; voted on every rank); every call is stock")
        return
    use2 = all(w for _, w in votes)
    why2 = None if use2 else "not every rank has the memory / mode 2 is switched off"
    if use2:
        st.ex, why2 = _LMH_BUILD[0](st, dev)
        reasons = tp.all_gather_object(why2)
        use2 = all(r is None for r in reasons)
        why2 = next((r for r in reasons if r is not None), None)
    use1 = LMH_MODE == 1 or LMH_FALLBACK_MODE1
    if not use2:
        st.ex = None
    gen = torch.Generator(device=dev).manual_seed(LMH_PROOF_SEED)
    md_ref = types.SimpleNamespace(forward_mode=fb.forward_mode, next_token_logits_buffer=None)
    rows = [st.N * k for k in range(1, LMH_MAX_ROWS // st.N + 1)]
    mine, cases = {}, 0
    for T in rows:
        edge = T in LMH_EDGE_ROWS
        ok2, ok1 = use2, use1
        for name, h in _lmh_families(T, st.H, real, gen, dev, edge):
            ref = st.stock_get_logits(mod, h, lm_head, md_ref, None, False)
            if use2:
                if edge and name == "randn":
                    _lmh_sync(dev)
                    time.sleep(st.rank * LMH_SKEW_S)
                buf = torch.full((T, st.V), float("nan"), dtype=torch.float32, device=dev)
                got = _lmh_logits(st, mod, 2, h, lm_head, types.SimpleNamespace(
                    forward_mode=fb.forward_mode, next_token_logits_buffer=buf), True)
                ok2 = ok2 and got is buf and _lmh_same(ref, got)
                if T == rows[-1] and name == "randn":
                    h2 = torch.randn(T, st.H, generator=gen, device=dev).to(torch.bfloat16)
                    ref2 = st.stock_get_logits(mod, h2, lm_head, md_ref, None, False)
                    a = _lmh_logits(st, mod, 2, h, lm_head, md_ref, False)
                    b = _lmh_logits(st, mod, 2, h2, lm_head, md_ref, False)
                    ok2 = ok2 and _lmh_same(ref, a) and _lmh_same(ref2, b)
                    del h2, ref2, a, b
                del buf, got
            if use1:
                buf = torch.full((T, st.V), float("nan"), dtype=torch.float32, device=dev)
                got = _lmh_logits(st, mod, 1, h, lm_head, types.SimpleNamespace(
                    forward_mode=fb.forward_mode, next_token_logits_buffer=buf), True)
                ok1 = ok1 and got is buf and _lmh_same(ref, got)
                del buf, got
            cases += 1
            del ref
        mine[T] = (bool(ok2), bool(ok1))
    _lmh_sync(dev)
    every = tp.all_gather_object(mine)
    for T in rows:
        if use2 and all(v[T][0] for v in every):
            st.proven[T] = 2
        elif use1 and all(v[T][1] for v in every):
            st.proven[T] = 1
    if LMH_EMPTY_CACHE and dev.type == "cuda":
        torch.cuda.empty_cache()
    two = sorted(T for T, m in st.proven.items() if m == 2)
    one = sorted(T for T, m in st.proven.items() if m == 1)
    stock = [T for T in rows if T not in st.proven]
    span = lambda v: f"{v[0]}-{v[-1]} ({len(v)})" if v else "- (0)"
    _lmh_say(st, f"{'ON' if st.proven else 'off'}: mode 2 rows {span(two)}, mode 1 rows {span(one)}, stock rows "
                 f"{span(stock)}; mode 2 {'built' if use2 else 'not built: ' + str(why2)}; {cases} proof cases "
                 f"x (2: {use2}, 1: {use1}) in {time.perf_counter() - t0:.2f} s; S {st.S} H {st.H} V {st.V}")
