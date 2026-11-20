from __future__ import annotations
import contextlib
import logging
import time
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.gemm import fp8_kernel
from sglang.kernels.ops.quantization.fp8_kernel import create_per_token_group_quant_fp8_output_scale
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.quantization import fp8_utils, unquant
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_disagg
from glm53f_core import e31_fused_route, e31_stage_route, fp8_linear_route, fp8_wide_route
NODE = "kda"
D, BT, GROUP = 128, 32, 128
_STOCK_QUANT = fp8_utils.sglang_per_token_group_quant_fp8
EARLY_GEMM = True
CHAIN_CTAS_A_HEAD = 2
_STOCK_GEMM = fp8_kernel.deep_gemm_fp8_fp8_bf16_nt
_BLOCKS: dict = {}
_CALL: dict = {}
_SEGMENT = [None]
_WARM: set = set()
_GEMM_WARM: set = set()
_SAID: set = set()
@triton.jit
def _onorm_quant_kernel(x, g, q, s, w, eps, T, row0, s_stride, D: tl.constexpr, BT: tl.constexpr,
                        BD: tl.constexpr, HEADS: tl.constexpr):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]
    p_x = x + o_t[:, None] * D + o_d[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_xbar = tl.where(m_d[None, :], b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_x_hat = b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :]
    p_g = g + o_t[:, None] * D + o_d[None, :]
    b_g = tl.load(p_g, mask=m_x, other=0.0).to(tl.float32)
    b_y = b_y * tl.sigmoid(b_g)
    b_y = b_y.to(tl.bfloat16).to(tl.float32)
    b_e = (tl.maximum(tl.max(tl.abs(b_y), axis=1), 1e-10) * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    b_e = (b_e >> 23) + ((b_e & 0x7FFFFF) != 0).to(tl.int32)
    b_s = ((254 - b_e) << 23).to(tl.float32, bitcast=True)
    o_q = tl.max_contiguous(o_d, 8)
    tl.store(q + o_t[:, None] * D + o_q[None, :], (b_y * b_s[:, None]).to(tl.float8e4nv), mask=m_x)
    o_r = row0 + o_t
    tl.store(s + o_r % HEADS // 4 * s_stride + o_r // HEADS * 4 + o_r % HEADS % 4, b_e.to(tl.uint8), mask=m_t)
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA o_norm quant: {why}")


def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA o_norm quant: %s", detail)


def buffers(tokens: int, width: int, device) -> tuple:
    q = torch.empty((tokens, width), dtype=torch.float8_e4m3fn, device=device)
    scale = create_per_token_group_quant_fp8_output_scale(x_shape=(tokens, width), device=device, group_size=GROUP,
                                                          column_major_scales=True, scale_tma_aligned=True,
                                                          scale_ue8m0=True)
    return q, scale, torch.empty(0, dtype=torch.uint8, device=device).set_(scale.untyped_storage())


def quantize(x, g, weight, eps: float, q, scale, scale_bytes, first: int, end: int) -> None:
    heads = x.shape[1]
    rows = (end - first) * heads
    if rows > 0:
        _onorm_quant_kernel[(triton.cdiv(rows, BT),)](
            x.view(-1, D)[first * heads:], g.view(-1, D)[first * heads:], q[first:], scale_bytes, weight, eps, rows,
            first * heads, scale.stride(1) * 4, D=D, BT=BT, BD=D, HEADS=heads, num_warps=4)


def _outside(x, g, block) -> str | None:
    entry = fp8_linear_route._FP8.get(id(block.o_proj))
    if entry is None or not entry[3] or unquant.UnquantizedLinearMethod.apply is not fp8_linear_route._apply:
        return "o_proj is not FP8 in this call"
    if not (x.dim() == 4 and x.shape[0] == 1 and x.shape[3] == D and x.dtype == torch.bfloat16 and x.is_cuda
            and x.is_contiguous() and x.shape[2] % 4 == 0 and x.shape[2] * D == entry[0].shape[1]):
        return f"input {x.dtype} {tuple(x.shape)}"
    if not entry[5][0] <= x.shape[1] <= entry[5][1]:
        return f"{x.shape[1]} rows outside o_proj's FP8 rows {entry[5]}"
    if tuple(g.shape) != tuple(x.shape[1:]) or g.dtype != torch.bfloat16 or not g.is_contiguous():
        return f"gate {g.dtype} {tuple(g.shape)}"
    return None


def _idle_sms(device, heads: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count - CHAIN_CTAS_A_HEAD * heads


def _early_gemm(block, q, scale, tokens: int, start: int, heads: int):
    entry = fp8_linear_route._FP8.get(id(block.o_proj))
    if not EARLY_GEMM or entry is None:
        return None
    if entry[2] != entry[0].shape[0] or entry[6] is not None:
        _say("gemm columns", "o_proj's FP8 weight is padded or has exact columns: its GEMM stays one call")
        return None
    projected = block.o_proj.forward
    if not (e31_fused_route._state["live"]
            and getattr(projected, "__module__", None) == e31_stage_route.__name__
            and getattr(projected, "__qualname__", "").startswith("_projecting.")):
        _say("gemm stage", "o_proj's output is not E31's stage slot in this call: its GEMM stays one call")
        return None
    why = e31_stage_route._armed(tokens)
    if why is not None:
        _say("gemm armed", f"o_proj's GEMM stays one call ({why})")
        return None
    if start != _SEGMENT[0]._bounds(tokens, None)[0][-2]:
        _say("gemm short", "a tracked call's last segment is not the plan's: o_proj's GEMM stays one call")
        return None
    _require(scale.shape[0] >= tokens and scale.stride(0) == 1, f"the scales' rows are not their first axis: "
             f"{tuple(scale.shape)} strides {scale.stride()} for {tokens} rows")
    slot = e31_stage_route._next_slot(tokens, q.device)
    sms, (block_n, block_k) = _idle_sms(q.device, heads), fp8_linear_route.BLOCK

    def launch() -> None:
        with deep_gemm_wrapper.configure_deep_gemm_num_sms(sms):
            _STOCK_GEMM(q[:start], scale[:start], entry[0], entry[1], slot[:start], block_n, block_k)

    return launch, (q.data_ptr(), slot.data_ptr(), tokens, start, scale, entry[0])


def _early(out, start: int):
    block, gate = _CALL.get("block"), _CALL.get("gate")
    if gate is None or start <= 0 or not gate.is_contiguous() or gate.numel() != out[0].numel():
        return None
    gate = gate.view(out.shape[1:])
    if _outside(out, gate, block) is not None:
        return None
    q, scale, scale_bytes = buffers(out.shape[1], out.shape[2] * D, out.device)
    gemm = _early_gemm(block, q, scale, out.shape[1], start, out.shape[2])

    def launch() -> None:
        o_norm = _BLOCKS[id(block)]
        quantize(out[0], gate, o_norm.weight, float(o_norm.eps), q, scale, scale_bytes, 0, start)
        _CALL["early"] = (out.data_ptr(), gate.data_ptr(), out.shape[1], start, q, scale, scale_bytes)
        if gemm is not None:
            _require(_CALL.get("gemm") is None, "o_proj's early rows ran twice in one call")
            gemm[0]()
            _CALL["gemm"] = gemm[1]

    return launch


def _norming(o_norm, block):
    stock_forward = type(o_norm).forward

    def forward(x, g, residual=None, prenorm=False, residual_in_fp32=False):
        why = (_outside(x, g, block) if residual is None and not prenorm and not residual_in_fp32
               else "a residual or prenorm")
        if why is not None:
            _require("early" not in _CALL, f"rows were done under the chain, then o_norm's call is not ours ({why})")
            _say(why.split(" ")[0], f"stock's o_norm and quantize serve a call ({why})")
            return stock_forward(o_norm, x, g, residual, prenorm, residual_in_fp32)
        tokens, early = x.shape[1], _CALL.pop("early", None)
        if early is None:
            first, (q, scale, scale_bytes) = 0, buffers(tokens, x.shape[2] * D, x.device)
        else:
            _require(early[:3] == (x.data_ptr(), g.data_ptr(), tokens), "rows normed under the chain belong to "
                     "another output or gate than o_norm's")
            first, q, scale, scale_bytes = early[3:]
        quantize(x[0], g, o_norm.weight, float(o_norm.eps), q, scale, scale_bytes, first, tokens)
        _require(_CALL.get("held") is None, "o_norm ran again before o_proj took its operand")
        _CALL["held"] = (x.data_ptr(), (tokens, x.shape[2] * D), q, scale)
        return x

    return forward


def _quantizing(x, group_size, *args, **kwargs):
    held = _CALL.get("held")
    if held is None or x.data_ptr() != held[0] or tuple(x.shape) != held[1] or x.dtype != torch.bfloat16:
        return _STOCK_QUANT(x, group_size, *args, **kwargs)
    _require(group_size == GROUP and not args and kwargs == dict(column_major_scales=True, scale_tma_aligned=True,
                                                                 scale_ue8m0=True),
             f"o_proj quantizes o_norm's rows by group {group_size}, {args} {kwargs}, not DeepGEMM's UE8M0 form")
    _CALL["held"] = None
    return held[2], held[3]


def _gemming(A, As, B, Bs, C, block_n=128, block_k=128):
    done = _CALL.get("gemm")
    if done is None or A.data_ptr() != done[0]:
        return _STOCK_GEMM(A, As, B, Bs, C, block_n, block_k)
    _, slot, tokens, start, scale, weight = done
    _require(C.data_ptr() == slot and A.shape[0] == C.shape[0] == tokens and As.data_ptr() == scale.data_ptr()
             and As.stride() == scale.stride() and B is weight and [block_n, block_k] == fp8_linear_route.BLOCK,
             "o_proj's GEMM on the held operand is not into the slot its early rows went to")
    _CALL["gemm"] = None
    return _STOCK_GEMM(A[start:], As[start:], B, Bs, C[start:], block_n, block_k)


def _warm_gemm(node, heads: int) -> None:
    weight = node.o_proj.weight
    sms = _idle_sms(weight.device, heads)
    key = (weight.device.index, tuple(weight.shape), sms)
    if key in _GEMM_WARM:
        return
    first = max(fp8_wide_route.WARM_ROWS[0], _SEGMENT[0].MIN_TOKENS, e31_fused_route._MIN_ROWS)
    last = min(fp8_wide_route.WARM_ROWS[-1], e31_fused_route._MAX_ROWS)
    plans = {t: _SEGMENT[0]._bounds(t, None)[0][-2] for t in range(first, last + 1)}
    starts, rest = sorted(set(plans.values())), sorted({t - start for t, start in plans.items()})
    _require(starts[0] > 0, f"a plan of {first} to {last} tokens has no segment before its last")
    w, ws = fp8_utils.quant_weight_ue8m0(weight_dequant=torch.zeros_like(weight.data),
                                         weight_block_size=fp8_linear_route.BLOCK)
    ws = fp8_utils.transform_scale_ue8m0(ws, mn=w.shape[-2])
    rows = max(starts[-1], rest[-1])
    q, scale, _ = buffers(rows, weight.shape[1], weight.device)
    out = torch.empty((rows, weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
    compile_utils._INITIALIZATION_DICT[
        (compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16, w.shape[0], w.shape[1], 1)] = True
    began = time.perf_counter()
    with deep_gemm_wrapper.configure_deep_gemm_num_sms(sms):
        for m in starts:
            _STOCK_GEMM(q[:m], scale[:m], w, ws, out[:m], *fp8_linear_route.BLOCK)
    for m in rest:
        _STOCK_GEMM(q[:m], scale[:m], w, ws, out[:m], *fp8_linear_route.BLOCK)
    torch.cuda.synchronize(weight.device)
    _GEMM_WARM.add(key)
    logging.getLogger(__name__).info(
        "glmf KDA o_norm quant: o_proj's split GEMM built for %d early row counts (%d to %d) at %d idle SMs and %d "
        "kept ones (%d to %d) in %.1f s", len(starts), starts[0], starts[-1], sms, len(rest), rest[0], rest[-1],
        time.perf_counter() - began)


def prepare(node) -> None:
    from glm53f_core import components

    if get_disagg().disaggregation_mode == "decode":
        _say("decode", "a PD decode server runs no eager prefill: nothing served")
        return
    _SEGMENT[0] = next((c for c in components.ACTIVE if c.__name__.rsplit(".", 1)[-1] == "kda_segment_route"), None)
    o_norm = getattr(node, "o_norm", None)
    weight = getattr(o_norm, "weight", None)
    if not (getattr(o_norm, "activation", None) == "sigmoid" and getattr(o_norm, "bias", 0) is None
            and isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16 and tuple(weight.shape) == (D,)
            and getattr(o_norm.forward, "__func__", None) is type(o_norm).forward):
        _say("o_norm", f"{type(o_norm).__name__} is not the sigmoid RMS norm of 128 this route was written for")
        return
    _BLOCKS[id(node)] = o_norm
    heads = node.o_proj.weight.shape[1] // D
    if (weight.device.index, heads) not in _WARM:
        x = torch.zeros((4, heads, D), dtype=torch.bfloat16, device=weight.device)
        quantize(x, x, weight, 1e-5, *buffers(4, heads * D, weight.device), 0, 4)
        torch.cuda.synchronize(weight.device)
        _WARM.add((weight.device.index, heads))
    if EARLY_GEMM and _SEGMENT[0] is not None:
        _require(deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM, "o_proj's early GEMM needs DeepGEMM's JIT (its SM count)")
        _warm_gemm(node, heads)


@contextlib.contextmanager
def routed(node, forward_batch):
    mode = forward_batch.forward_mode
    if (id(node) not in _BLOCKS or mode.is_draft_extend_v2()
            or not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=False)
            or torch.cuda.is_current_stream_capturing() or get_is_capture_mode()):
        yield
        return
    _require(not _CALL and fp8_utils.sglang_per_token_group_quant_fp8 is _STOCK_QUANT
             and fp8_kernel.deep_gemm_fp8_fp8_bf16_nt is _STOCK_GEMM,
             "the quantize or DeepGEMM's entry is already replaced, or a call's state is left over")
    segment = _SEGMENT[0] if node.do_fuse_qkvbfg else None
    o_norm, stock_gate = _BLOCKS[id(node)], type(node).forward_qkvbfg_fused

    def gating(*args, **kwargs):
        made = stock_gate(node, *args, **kwargs)
        _CALL["gate"] = made[3]
        return made

    _CALL["block"] = node
    o_norm.forward = _norming(o_norm, node)
    fp8_utils.sglang_per_token_group_quant_fp8 = _quantizing
    fp8_kernel.deep_gemm_fp8_fp8_bf16_nt = _gemming
    if segment is not None:
        node.forward_qkvbfg_fused = gating
        segment._UNDER_LAST[0] = _early
    try:
        yield
    finally:
        if segment is not None:
            segment._UNDER_LAST[0] = None
            del node.forward_qkvbfg_fused
        fp8_kernel.deep_gemm_fp8_fp8_bf16_nt = _STOCK_GEMM
        fp8_utils.sglang_per_token_group_quant_fp8 = _STOCK_QUANT
        del o_norm.forward
        held, gemm = _CALL.get("held"), _CALL.get("gemm")
        _CALL.clear()
    _require(held is None, "o_norm's operand was held but o_proj never quantized it (its rows were never normed)")
    _require(gemm is None, "o_proj's early rows ran under the chain but o_proj's GEMM never took the rest")
