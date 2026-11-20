from __future__ import annotations
import contextlib
import logging
import sys
import flashinfer.decode
import torch
NODE = "dsa"
WIDTH = 512
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
EXACT_ROWS = 512
SERVE = {64: (193, 16384), 32: (1024, 16384)}
VALID_PREFIX = True
PREFILL_PLANS = {64: (1024, 2048, 4096, 8192), 32: (1024, 2048, 4096)}
PREFILL_WIDTHS = (1028, 2052)
_TRTLLM = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla
_PEERS = ("dsa_rows_route", "dsa_wave_split_route", "dsa_split_route")
_USED = frozenset({"query", "kv_cache", "block_tables", "sparse_mla_top_k", "sparse_mla_top_k_lens", "out",
                   "bmm1_scale", "bmm2_scale", "kv_lora_rank", "qk_rope_head_dim"})
_IGNORED = frozenset({"workspace_buffer", "qk_nope_head_dim", "seq_lens", "max_seq_len",
                      "multi_ctas_kv_counter_buffer", "enable_pdl", "backend", "cute_dsl_impl"})
_DEFAULTS = (("sinks", None), ("skip_softmax_threshold_scale_factor", None), ("lse", None), ("return_lse", False),
             ("cum_seq_lens_q", None), ("max_q_len", None), ("enable_dcp", False), ("cp_world", 1), ("cp_rank", 0),
             ("causal_seqlens_kv_global", None), ("use_fp16_softmax", None), ("return_lse_base", None),
             ("kv_scale_format", "auto"), ("uses_shared_paged_kv_idx", True), ("is_var_seq", True))
_STATE: dict = {}
_PLANS: dict = {}
_SET: list = []
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf DSA TS attention: {why}")
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf DSA TS attention: %s", detail)

def _peer(name: str):
    
    return sys.modules.get(__name__.rsplit(".", 1)[0] + "." + name)


def _plan(rows: int, heads: int, width: int, device) -> tuple:
    
    packed = rows > EXACT_ROWS
    capacity = 1 << (rows - 1).bit_length() if packed else rows
    key = (capacity, heads, width, packed, device.index)
    hit = _PLANS.get(key)
    if hit is not None:
        return hit
    _require(not torch.cuda.is_current_stream_capturing(),
             f"no plan for {rows} rows x {heads} heads over {width} slots inside a CUDA-graph capture (no eager call "
             "of the shape preceded it)")
    wrapper = _STATE["wrapper"]()
    wrapper.plan(device, 1 if packed else rows, heads, max_topk=width, max_seq_len_q=capacity if packed else 1,
                 packed_query=packed, q_data_type=FP8, qk_rope_head_dim=0, kv_cache_format="native",
                 return_lse=False, assume_valid_prefix=VALID_PREFIX)
    offsets = torch.zeros(2, dtype=torch.int32, device=device) if packed else None
    _PLANS[key] = (wrapper, offsets)
    _say(f"plan {key}", f"planned {'packed ' if packed else ''}{capacity} rows x {heads} heads over {width} slots")
    return wrapper, offsets


def _outside(kwargs: dict) -> tuple[str, str] | None:
    
    extra = set(kwargs) - _USED - _IGNORED - {name for name, _ in _DEFAULTS}
    if extra:
        return "arguments", f"arguments this route does not take: {sorted(extra)}"
    for name, default in _DEFAULTS:
        value = kwargs.get(name, default)
        if isinstance(value, torch.Tensor) or value != default:
            return name, f"{name} is {value!r}"
    if kwargs.get("backend", "auto") not in ("auto", "trtllm-gen"):
        return "backend", f"backend {kwargs.get('backend')!r}"
    q, kv, tables = kwargs.get("query"), kwargs.get("kv_cache"), kwargs.get("block_tables")
    lens, out = kwargs.get("sparse_mla_top_k_lens"), kwargs.get("out")
    if not (isinstance(q, torch.Tensor) and isinstance(kv, torch.Tensor) and isinstance(tables, torch.Tensor)
            and isinstance(lens, torch.Tensor)):
        return "tensors", "query, KV pool, block tables or top-k lengths not passed as tensors"
    if kwargs.get("kv_lora_rank") != WIDTH or kwargs.get("qk_rope_head_dim") != 0:
        return "rank", f"latent rank {kwargs.get('kv_lora_rank')} with a RoPE width of {kwargs.get('qk_rope_head_dim')}"
    if (q.dtype != FP8 or q.dim() != 4 or q.shape[1] != 1 or q.shape[3] != WIDTH or not q.is_contiguous()
            or q.data_ptr() % 16):
        return "query", f"the query is {q.dtype} {tuple(q.shape)}, not contiguous FP8 [rows, 1, heads, {WIDTH}]"
    rows, heads = q.shape[0], q.shape[2]
    window = SERVE.get(heads)
    if window is None or not window[0] <= rows <= window[1]:
        return f"shape {heads}", f"{rows} rows x {heads} heads, outside the served windows {SERVE}"
    width = kwargs.get("sparse_mla_top_k")
    if (not isinstance(width, int) or width <= 0 or tables.dtype != torch.int32 or tables.dim() != 3
            or tuple(tables.shape) != (rows, 1, width) or tables.stride(2) != 1):
        return "tables", f"block tables {tables.dtype} {tuple(tables.shape)} for a top-k of {width!r}"
    if lens.dtype != torch.int32 or tuple(lens.shape) != (rows,) or not lens.is_contiguous():
        return "lengths", f"top-k lengths {lens.dtype} {tuple(lens.shape)}"
    if (kv.dtype not in (FP8, torch.uint8) or kv.shape[-1] != WIDTH or not kv.is_contiguous()
            or kv.data_ptr() % 16):
        return "pool", f"a KV pool of {kv.dtype} {tuple(kv.shape)}"
    if out is not None and (out.dtype != BF16 or tuple(out.shape) != (rows, 1, heads, WIDTH)
                            or not out.is_contiguous() or out.data_ptr() % 16):
        return "out", f"an output of {out.dtype} {tuple(out.shape)}"
    for name in ("bmm1_scale", "bmm2_scale"):
        value = kwargs.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            return name, f"{name} is {type(value).__name__} {value!r}, not a positive host scalar"
    return None


def _kernel(*args, **kwargs):
    
    why = ("positional", "arguments passed by position") if args else _outside(kwargs)
    if why is not None:
        _say(why[0], f"trtllm-gen serves a call: {why[1]}")
        return _TRTLLM(*args, **kwargs)
    q = kwargs["query"]
    rows, heads = q.shape[0], q.shape[2]
    tables = kwargs["block_tables"]
    width = tables.shape[2]
    wrapper, offsets = _plan(rows, heads, width, q.device)
    out = kwargs.get("out")
    if out is None:
        out = torch.empty((rows, 1, heads, WIDTH), dtype=BF16, device=q.device)
    pool = kwargs["kv_cache"]
    pool = (pool.view(FP8) if pool.dtype == torch.uint8 else pool).view(-1, WIDTH)
    scales = dict(softmax_scale=float(kwargs.get("bmm1_scale", 1.0)), output_scale=float(kwargs.get("bmm2_scale", 1.0)),
                  validate=False)
    if offsets is None:
        wrapper.run(q, pool, tables[:, 0], kwargs["sparse_mla_top_k_lens"], out=out, **scales)
    else:
        offsets[1:].fill_(rows)
        wrapper.run(q.view(rows, heads, WIDTH), pool, tables[:, 0], kwargs["sparse_mla_top_k_lens"],
                    qo_indptr=offsets, out=out.view(rows, heads, WIDTH), **scales)
    _say(f"ours {heads}", f"the TS kernel serves {heads}-head calls (first: {rows} rows over {width} slots)")
    return out


def prepare(node) -> None:
    
    _require(getattr(node, "kv_lora_rank", None) == WIDTH and getattr(node, "qk_rope_head_dim", None) == 0,
             f"the attention module's latent is {getattr(node, 'kv_lora_rank', None)} wide with "
             f"{getattr(node, 'qk_rope_head_dim', None)} RoPE dims, not {WIDTH} and 0")
    if "wrapper" in _STATE:
        return
    from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

    device = torch.device("cuda", torch.cuda.current_device())
    _require(torch.cuda.get_device_capability(device) in ((10, 0), (10, 3)),
             f"the TS sparse MLA needs SM100 or SM103, this is {torch.cuda.get_device_capability(device)}")
    peers = [m for m in (_peer(name) for name in _PEERS) if m is not None]
    _STATE.update(wrapper=BatchSparseMLADecodePagedTSWrapper, peers=peers)
    from sglang.srt.distributed import get_attn_tp_group
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode == "prefill" and get_attn_tp_group().world_size == 2:
        _require(not torch.cuda.is_current_stream_capturing(), "prepare inside a CUDA-graph capture")
        for heads, capacities in PREFILL_PLANS.items():
            for capacity in capacities:
                for width in PREFILL_WIDTHS:
                    _plan(capacity, heads, width, device)
    _say("prepared", f"peers holding the kernel: {[m.__name__.rsplit('.', 1)[-1] for m in peers]}")

@contextlib.contextmanager
def routed(node, forward_batch):
    
    _require(not _SET, "a sparse layer's call inside another's")
    _SET.extend((module, _TRTLLM) for module in _STATE["peers"] if module._STOCK is _TRTLLM)
    try:
        for module, _ in _SET:
            module._STOCK = _kernel
        yield
    finally:
        for module, held in _SET:
            module._STOCK = held
        _SET.clear()
