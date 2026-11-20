from __future__ import annotations
import contextlib
import hashlib
import os
import torch
import torch.nn.functional as F
from sglang.kernels.ops.gemm import fp8_kernel
from sglang.srt.layers.moe.moe_runner import base as moe_base
from sglang.srt.layers.quantization import fp8_utils
from glm53f_core import e31_fused_route as fused
NODE = "layer"
_SLOTS = 2
_PINS = {
    "srt/models/deepseek_v2.py": "314e533c1606278d8294334edf13a23f082f1687a06f5e9bdf92a4958775ea83",
    "srt/models/deepseek_common/attention_forward_methods/forward_mla.py":
        "4b20261de805fc7d0ea823c98a53ab648745a5dab385b858bdea966b1b7fdbc1",
    "srt/layers/linear.py": "ec5fe0360562bd056de8b07e496c822444dcc8d5831fab2ecc589b15b44145a5",
    "srt/layers/quantization/unquant.py": "0dbc392662a9afd328c5f9b1db2d8d8a606b1b99cf1ac44cb36c1c373cc5cadd",
    "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py":
        "2221b8cfdefafa191ce3eb09116a977adef4befb8c29315abab811423300658b",
    "srt/layers/moe/moe_runner/base.py": "b9b7f20001db6b1aefa679e86e7ae4439fca9509385048be1457381efd91cf72",
    "srt/layers/moe/moe_runner/flashinfer_trtllm.py": "7afaeb2ac929c30cefb9958e940ece6ebb940a92e7e5f827380b7f0307ff5ba1",
    "srt/layers/moe/fused_moe_triton/layer.py": "0f05f451217b776910ca7d8bfbfacac8b47c00fefedab84e5f08dce86c6b0b36",
    "srt/layers/moe/token_dispatcher/standard.py":
        "3d150846965821de4d481d5bdee7674bd4bb7a3ff63970373be58592b4bb27d6",
    "srt/layers/quantization/fp8_utils.py": "9be248e927dcbf2e0811b2e49486de548233cf7870e4979999ab68fd90b3fc23",
    "kernels/ops/gemm/fp8_kernel.py": "2ae64e0d28e9851751c6e30fa8153e0d90c754e7b46f6c2fa703a1c6a376f4fd",
}
_state = {"checked": False, "projections": {}, "bufs": None, "epoch": 0, "pending": None, "said": set(), "tally": {}}
def _say(tag: str, detail: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        print(f"[glm-e31s] {detail}", flush=True)
def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-e31s] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)
def _engine_differences() -> list:
    import sglang
    root = os.path.dirname(os.path.abspath(sglang.__file__))
    differing = []
    for relative, expected in _PINS.items():
        try:
            found = hashlib.sha256(fused._read(os.path.join(root, relative))).hexdigest()
        except OSError:
            found = None
        if found != expected:
            differing.append(relative)
    return differing
def _batch_invariant() -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    return bool(is_batch_invariant_mode_enabled())
def prepare(layer) -> None:
    state = fused._state
    if state["groups"] is None:
        raise RuntimeError("E31 stage: e31_layer_route's prepare has not run before this component's")
    if not state["groups"]:
        _say("inactive", f"E31 is inactive on this engine ({state['why']}): nothing is staged")
        return
    if not _state["checked"]:
        differs = _engine_differences()
        if differs:
            raise RuntimeError(f"E31 stage: this engine is not SGLang 314bce7c; differing: {', '.join(differs)}")
        if not callable(getattr(state["native"], "post_slot", None)):
            raise RuntimeError("E31 stage: e31_fused_native lacks post_slot")
        _state["checked"] = True
    o_proj = getattr(layer.self_attn, "o_proj", None)
    weight = getattr(o_proj, "weight", None)
    if _batch_invariant():
        _say("batch-invariant", "SGLang's batch-invariant mode is on: o_proj is stock's and its sum keeps E31's copy")
        return
    if not (isinstance(o_proj, torch.nn.Module) and isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16 and weight.dim() == 2 and weight.shape[0] == fused._HIDDEN
            and getattr(o_proj, "bias", None) is None and getattr(o_proj, "reduce_results", True) is False
            and getattr(o_proj.forward, "__func__", None) is type(o_proj).forward):
        _say("o_proj", f"layer {getattr(layer, 'layer_id', '?')}'s o_proj is not a BF16 projection to {fused._HIDDEN} "
                       "without a bias or its own forward: its sum keeps E31's copy")
        return
    _state["projections"][id(layer)] = (o_proj, weight)


def _setup(device) -> dict:
    
    group = fused._state["groups"][0]
    name = group.device_group.group_name
    peer = 1 - group.rank_in_group
    rows, hidden = fused._MAX_ROWS, fused._HIDDEN
    slots, hs = fused._symmetric(_SLOTS * rows * hidden, torch.bfloat16, device, name)
    flags, hf = fused._symmetric(2, torch.int32, device, name)
    flags.zero_()
    published = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    group.barrier()
    step = rows * hidden * slots.element_size()
    base = int(hs.buffer_ptrs[peer])
    _tally("slots set up", lambda: f"rank {group.rank_in_group}, {_SLOTS} x {rows} x {hidden} BF16")
    return {"slots": slots.view(_SLOTS, rows, hidden), "peer_slots": tuple(base + k * step for k in range(_SLOTS)),
            "flags": flags, "peer_flags": int(hf.buffer_ptrs[peer]), "published": published}


def _next_slot(rows: int, device) -> torch.Tensor:
    
    if _state["bufs"] is None:
        _state["bufs"] = _setup(device)
    return _state["bufs"]["slots"][(_state["epoch"] + 1) % _SLOTS][:rows]


def _armed(rows: int) -> str | None:
    
    if not fused._MIN_ROWS <= rows <= fused._MAX_ROWS:
        return f"{rows} rows, outside {fused._MIN_ROWS} to {fused._MAX_ROWS}"
    if fused._capturing():
        return "under capture"
    if fused._state["pending"] is not None:
        return "a sum is pending (its post has not run)"
    return None


def _projecting(o_proj, weight):
    
    stock_forward = type(o_proj).forward

    def forward(*args, **kwargs):
        before, before_fp8 = F.linear, fp8_utils.w8a8_block_fp8_matmul_deepgemm

        def linear(input, weight_, bias=None):
            if weight_ is not weight:
                return before(input, weight_, bias)
            why = None
            if bias is not None:
                why = "a bias"
            elif not (fused._on_gpu(input) and input.dtype == torch.bfloat16 and input.dim() == 2
                      and input.shape[1] == weight.shape[1]):
                why = f"input {input.dtype} {tuple(input.shape)}"
            else:
                why = _armed(input.shape[0])
            if why is not None:
                _say("o_proj " + why.split(",")[0], f"stock's o_proj serves a call ({why}): its sum keeps E31's copy")
                return before(input, weight_, bias)
            slot = _next_slot(input.shape[0], input.device)
            torch.mm(input, weight.t(), out=slot)
            _tally("o_proj into the stage", lambda: f"{input.shape[0]} rows")
            return slot

        def fp8_matmul(A, B, As, Bs, block_size, output_dtype):
            if not (tuple(B.shape) == tuple(weight.shape) and output_dtype == torch.bfloat16 and fused._on_gpu(A)
                    and A.dim() == 2 and A.shape[1] == weight.shape[1] and A.is_contiguous()):
                why = f"an FP8 GEMM of {tuple(A.shape)} by {tuple(B.shape)} into {output_dtype}"
            else:
                why = _armed(A.shape[0])
            if why is not None:
                _say("fp8 o_proj " + why.split(",")[0], f"stock's FP8 o_proj serves a call ({why}): its sum keeps "
                                                        "E31's copy")
                return before_fp8(A, B, As, Bs, block_size, output_dtype)
            slot = _next_slot(A.shape[0], A.device)
            fp8_kernel.deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, slot, block_size[0], block_size[1])
            _tally("FP8 o_proj into the stage", lambda: f"{A.shape[0]} rows")
            return slot

        F.linear = linear
        fp8_utils.w8a8_block_fp8_matmul_deepgemm = fp8_matmul
        try:
            return stock_forward(o_proj, *args, **kwargs)
        finally:
            fp8_utils.w8a8_block_fp8_matmul_deepgemm = before_fp8
            F.linear = before

    return forward


def _buffering(stock_buffer):
    

    def get_fp4_moe_output_buffer(num_tokens, hidden_size, dtype, device, symmetric):
        request = dict(num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype, device=device, symmetric=symmetric)
        if not (hidden_size == fused._HIDDEN and dtype == torch.bfloat16):
            tag, why = "moe output", f"an output of {num_tokens} x {hidden_size} {dtype}"
        else:
            why = _armed(num_tokens)
            tag = "moe " + str(why).split(",")[0]
        if why is not None:
            _say(tag, f"the MoE's output is stock's ({why}): its sum keeps E31's copy")
            return stock_buffer(**request)
        slot = _next_slot(num_tokens, device)
        with moe_base.moe_output_buffer_ctx(slot):
            out = stock_buffer(**request)
        if out is not slot:
            _say("moe declined", "stock's buffer rule declined the slot (symmetric memory on): the MoE's sum keeps "
                                 "E31's copy")
            return out
        _tally("MoE output into the stage", lambda: f"{num_tokens} rows")
        return out

    return get_fp4_moe_output_buffer


def _deferring(e31_sum):
    

    def all_reduce(input_: torch.Tensor) -> torch.Tensor:
        state = fused._state
        if state["pending"] is not None:
            raise RuntimeError("E31 fused: a sum was asked for while another partial is pending (its post never ran)")
        b = _state["bufs"]
        if b is not None and state["depth"] > 0 and fused._eligible(input_):
            epoch = _state["epoch"] + 1
            if input_.data_ptr() == b["slots"][epoch % _SLOTS].data_ptr():
                _state["epoch"] = epoch
                state["pending"] = _state["pending"] = (input_, epoch)
                _tally("sum without a copy", lambda: f"{tuple(input_.shape)}")
                return input_
        return e31_sum(input_)

    return all_reduce


def _posting(e31_post):
    

    def hc_post(x, residual, h_post, h_res, hc_mult):
        pending = fused._state["pending"]
        if pending is None or pending is not _state["pending"]:
            return e31_post(x=x, residual=residual, h_post=h_post, h_res=h_res, hc_mult=hc_mult)
        partial, epoch = pending
        if x is not partial:
            raise RuntimeError("E31 fused: the post received another tensor while a partial is pending (the sum's "
                               "result was moved or replaced before its post)")
        rows = x.shape[0]
        if (hc_mult != fused._HC or residual.dtype != torch.bfloat16 or not residual.is_contiguous()
                or residual.numel() != rows * fused._HC * fused._HIDDEN or h_post.dtype != torch.float32
                or not h_post.is_contiguous() or h_post.numel() != rows * fused._HC or h_res.dtype != torch.float32
                or not h_res.is_contiguous() or h_res.numel() != rows * fused._HC * fused._HC):
            raise RuntimeError(f"E31 fused: an mHC post this kernel does not mirror (hc_mult {hc_mult}, residual "
                               f"{residual.dtype} {tuple(residual.shape)}, h_post {h_post.dtype} {tuple(h_post.shape)}, "
                               f"h_res {h_res.dtype} {tuple(h_res.shape)})")
        b = _state["bufs"]
        out = torch.empty((rows, fused._HC * fused._HIDDEN), dtype=torch.bfloat16, device=x.device)
        fused._state["native"].post_slot(partial, residual, h_post, h_res, out, b["peer_slots"][epoch % _SLOTS],
                                         b["flags"], b["peer_flags"], b["published"], epoch, fused._VARIANT, fused._FTZ)
        fused._state["pending"] = _state["pending"] = None
        _tally("post without a copy", lambda: f"{rows} rows")
        return out

    return hc_post


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    state = fused._state
    if not state["live"]:
        yield
        return
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

    glm = state["glm"]
    e31_post = glm._hc_post_fn
    stock_buffer = flashinfer_trtllm._get_fp4_moe_output_buffer
    projection = _state["projections"].get(id(layer))
    summed, posted, buffered, projected = [], False, False, False
    try:
        for group in state["groups"]:
            e31_sum = group.all_reduce
            group.all_reduce = _deferring(e31_sum)
            summed.append((group, e31_sum))
        glm._hc_post_fn = _posting(e31_post)
        posted = True
        flashinfer_trtllm._get_fp4_moe_output_buffer = _buffering(stock_buffer)
        buffered = True
        if projection is not None:
            projection[0].forward = _projecting(*projection)
            projected = True
        yield
    finally:
        if projected:
            del projection[0].forward
        if buffered:
            flashinfer_trtllm._get_fp4_moe_output_buffer = stock_buffer
        if posted:
            glm._hc_post_fn = e31_post
        for group, e31_sum in summed:
            group.all_reduce = e31_sum
        _state["pending"] = None
