from __future__ import annotations
import contextlib
import logging
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.layer_boundary.residual import mhc as residual
NODE = "layer"
STREAMS, BLOCK = 4, 1024
MAX_ROWS = 16384
_STOCK_EXPAND, _STOCK_CONTRACT = residual.hc_expand, residual.hc_contract
_SAID: set = set()
def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf mHC ends: %s", detail)
@triton.jit
def _contract_kernel(x_ptr, out_ptr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    src = x_ptr + row * (4 * H) + cols
    acc = 0.0 + tl.load(src).to(tl.float32)
    acc = acc + tl.load(src + H).to(tl.float32)
    acc = acc + tl.load(src + 2 * H).to(tl.float32)
    acc = acc + tl.load(src + 3 * H).to(tl.float32)
    tl.store(out_ptr + row * H + cols, (acc * 0.25).to(tl.bfloat16))


def _expand(x: torch.Tensor, n: int) -> torch.Tensor:
    
    if not (x.is_cuda and x.dim() == 2 and x.stride(-1) == 1 and 1 <= x.shape[0] <= MAX_ROWS):
        _say("expand", f"stock's hc_expand serves a {x.dim()}-D input {tuple(x.shape)} of strides {tuple(x.stride())} "
                       f"on {x.device}")
        return _STOCK_EXPAND(x, n)
    return mhc.hc_broadcast(x, n).view(x.shape[0], n * x.shape[1])


def _contract(x: torch.Tensor, n: int) -> torch.Tensor:
    rows = x.shape[0] if x.dim() == 2 else 0
    if not (n == STREAMS and 1 <= rows <= MAX_ROWS and x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.shape[1] % (STREAMS * BLOCK) == 0):
        _say("contract", f"stock's hc_contract serves {n} streams of a {x.dtype} {tuple(x.shape)} input "
                         f"(contiguous {x.is_contiguous()}) on {x.device}")
        return _STOCK_CONTRACT(x, n)
    hidden = x.shape[1] // STREAMS
    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device)
    _contract_kernel[(rows, hidden // BLOCK)](x, out, H=hidden, BLOCK=BLOCK, num_warps=4)
    return out


def prepare(layer) -> None:
    pass


@contextlib.contextmanager
def routed(layer, forward_batch):
    found = (residual.hc_expand, residual.hc_contract)
    if found != (_STOCK_EXPAND, _STOCK_CONTRACT):
        raise RuntimeError("glmf mHC ends: another route holds the residual module's hc_expand or hc_contract")
    residual.hc_expand, residual.hc_contract = _expand, _contract
    try:
        yield
    finally:
        residual.hc_expand, residual.hc_contract = found
