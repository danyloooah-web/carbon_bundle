from __future__ import annotations
import rotq_native
import torch
import triton
import triton.language as tl
D = 128
BLOCK_M = 8
GRID_PER_SM = 3
@triton.jit
def _stage(x, BM: tl.constexpr, S: tl.constexpr):
    v = tl.permute(tl.reshape(x, (BM, 128 // (2 * S), 2, S)), (0, 1, 3, 2))
    a, b = tl.split(v)
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (BM, 128))
@triton.jit
def _rotate_quant_kernel(X, OUT, Y, S_OUT, M, scale, round_scale: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    row_mask = rows < M
    offs = rows[:, None].to(tl.int64) * 128 + cols[None, :]
    mask = row_mask[:, None] & (cols[None, :] < 128)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    x = _stage(x, BM, 1)
    x = _stage(x, BM, 2)
    x = _stage(x, BM, 4)
    x = _stage(x, BM, 8)
    x = _stage(x, BM, 16)
    x = _stage(x, BM, 32)
    x = _stage(x, BM, 64)
    rotated = (x * scale).to(tl.bfloat16)
    tl.store(OUT + offs, rotated, mask=mask)
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    xq = tl.where(mask, rotated.to(tl.float32), 0.0)
    x_abs = tl.abs(xq)
    amax = tl.max(x_abs, axis=1)
    amax = tl.maximum(amax, 1e-4)
    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        q_scale = tl.exp2(log_ceil)
    else:
        q_scale = amax * fp8_max_inv
    y = xq / q_scale[:, None]
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
    tl.store(Y + offs, y, mask=mask)
    tl.store(S_OUT + rows, q_scale, mask=row_mask)


def rotate_quant(x: torch.Tensor, scale: float, round_scale: bool):
    
    rows = x.numel() // D
    out = torch.empty_like(x)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.shape[:-1], 1, dtype=torch.float32)
    if rows and round_scale:
        rotq_native.rotate_quant(x, out, y, s, scale, GRID_PER_SM)
    elif rows:
        _rotate_quant_kernel[(triton.cdiv(rows, BLOCK_M),)](x, out, y, s, rows, scale, round_scale=round_scale,
                                                            BM=BLOCK_M, num_warps=4)
    return out, y, s


def rotate_quant_gated(x: torch.Tensor, raw: torch.Tensor, scale: float, c1: float, ss: float):
    
    out = torch.empty_like(x)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.shape[:-1], 1, dtype=torch.float32)
    g = torch.empty_like(s)
    if x.numel():
        rotq_native.rotate_quant_gated(x, out, y, s, raw, g, scale, c1, ss, GRID_PER_SM)
    return out, y, s, g
