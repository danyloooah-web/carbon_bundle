"""KDA-WY32 + WY-ONORM target-verify kernel (r59 KW): kda_wy32_route.py serves it inside kda_v2_route's domain.

A bundle-owned fork of SGLang 314bce7c kernels/ops/attention/kda_decode_mtp.py (Apache-2.0): stock's DSpARK target-
verify KDA kernel (fused_kda_decode_mtp_dspark, CuTe DSL) in its ReplaySSM + fused-output-norm form (CACHE_RING,
APPLY_ONORM). Written by r59/kw/src/mkkw.py from our C25 layerar/kdadyn.py (kda_wy_verify_kernel: r57 KDA-WY32 +
WY-ONORM) and stock's file; never edit by hand.

Stock's text: phase 1 (convolutions, SiLU, L2 norms, gate, beta, the output-norm gate / weight staging, every ReplaySSM
ring store and conv-window store: the side outputs are stock's statements), the slot -1 pad rows, the norm epilogue
(kda_decode_mtp.py :827-:846, verbatim: sum of squares, rsqrt, raw * rms * weight * gate, one BF16 rounding) and the
host helpers _ld_p1, _tensor_layout_key, _fits_32bit_stride, _cute_tensor.
What differs: phase 2. Stock walks the six tokens through the register-resident state one by one (two 256-thread CTAs
an SM); here the WY form in fp32 (r57/wy/DESIGN_WY32.md): per (request, head) the decay products P_t, kh = k P,
qh = q P, kt = k / P, the Gram dots M = strict_lower(kh kt^T) and L = tril(qh kt^T), Tm = (I + diag(beta) M)^-1
diag(beta) by forward substitution, QT = L Tm, A = qh - QT kh, W = QT V; then ONE streamed pass over the committed
state rows (16-byte loads, an L2 prefetch before the PDL wait, no smem stage): o[t, j] = A[t] . S0[j] + W[t, j] (fp32,
mul.rn / add.rn / fma.rn in r57/wy/wy_emu.py's order), staged in smem for stock's epilogue. 128 threads a CTA, grid
(H, n_req, 1), a 64-register bound (8 CTAs an SM); the request count is a runtime argument and the tensors whose
extents follow it are compiled with dynamic layouts, the state pool with 64-bit offsets: one compile per (H, static
layouts) serves every batch size and every layer.
Rounding: each fp32 output differs from stock's by reassociation (~1e-6 relative, r59/kw/tests/t_emu_kw.py), then
stock's norm text and one BF16 rounding.

This file does no namespace reflection of any kind and binds ``cute`` exactly once (the cutlass import below), so the
validator's engine_tree admits its compile call (cacheon/dsl_jit_policy.py decides per file).
"""

# SPDX-License-Identifier: Apache-2.0  (fork of SGLang source, sgl-project/sglang 314bce7c)

import time

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass._mlir.dialects import nvvm
import torch

from sglang.kernels.arch import current_arch

KW_WY_THREADS = 128          # block threads (128 + MINB 8 = 64 registers, 8 CTAs an SM, no spill on sm_103a)
KW_WY_R = 4                  # state rows a warp loads before one reduce-scatter (1 / 2 / 4)
KW_WY_PF = 1                 # 1 = cp.async.bulk.prefetch.L2 of the CTA's state rows before the PDL wait
KW_WY_MINB = 8               # min CTAs / SM launch bound (0 = the DSL default, unbounded)
KW_WY_MAX_LOGDECAY = 80.0    # product form only while 6 |lower_bound| <= this (exp(-80) is fp32-normal: P = prod a_t
                             # >= exp(6 lower_bound) by phase 1's gate, k / P finite)
WY_LANES = 32                # q / k / g smem tiles in plain channel order (phase 1's P2_LANES_K = 32)

WARP_SIZE = 32
TILE_K = 128
KERNEL_WIDTH = 4
# Phase 1 conv jobs: q, k, g. Each gets its own equal-sized warp group; the
# v-conv gets whatever is left over.
P1_NUM_JOBS = 3
# Share of the block's warps handed to the v-conv, as a divisor: 4 -> a quarter.
P1_V_WARP_DIVISOR = 4
# Stock's narrow block width: phase 1 loads with ld.global.nc only at this width (never at 128 threads).
BLOCK_THREADS_NARROW = 256
HEAD_DIM = TILE_K
VEC_SIZE = HEAD_DIM // WARP_SIZE


def _ld_p1(invariant, x, coord):
    """Load one phase-1 input element, optionally as ld.global.nc."""
    if not invariant:
        return cutlass.Float32(x[coord])
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        x.element_type,
        num_bits_per_copy=x.element_type.width,
        invariant=True,
    )
    src = cute.make_tensor(
        x.iterator + cute.crd2idx(coord, x.layout), cute.make_layout((1,))
    )
    dst = cute.make_rmem_tensor(cute.make_layout((1,)), x.element_type)
    cute.copy(atom, src, dst)
    return cutlass.Float32(dst[0])


def _wy_mul(x, y):
    """r57 KDA-WY32: mul.rn.f32 (never contracted into an fma)."""
    return cute.math.mul(x, y, rounding=cute.math.RoundingMode.NEAREST_EVEN)


def _wy_add(x, y):
    """r57 KDA-WY32: add.rn.f32 (never contracted into an fma)."""
    return cute.math.add(x, y, rounding=cute.math.RoundingMode.NEAREST_EVEN)


def _wy_sub(x, y):
    """r57 KDA-WY32: x - y as add.rn.f32 of -y (IEEE: the same result, signed zeros included)."""
    return cute.math.add(x, -y, rounding=cute.math.RoundingMode.NEAREST_EVEN)


def _wy_smem_bytes(t_loop, rv, onorm=False):
    """r57 KDA-WY32: kda_wy_verify_kernel's smem (each tile 16-byte aligned) + 256 bytes of slack (r57 WY-ONORM: +
    sOnormGate / sOnormWeight when ``onorm``)."""
    total = 256
    for n in (t_loop * TILE_K, t_loop * TILE_K, t_loop * TILE_K, t_loop, t_loop * TILE_K, t_loop * rv,
              2 * t_loop * t_loop, t_loop * t_loop) + ((t_loop * TILE_K, TILE_K) if onorm else ()):
        total += -(-n * 4 // 16) * 16
    return total


def _wy_s4(s, row, lane):
    """r57 KDA-WY32: an fp32 [T, 128] smem tile's channels 4 lane .. 4 lane + 3 of ``row`` as one 16-byte load."""
    src = cute.make_tensor(
        (s.iterator + (row * HEAD_DIM + lane * 4)).align(16), cute.make_layout((4,))
    )
    dst = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    cute.autovec_copy(src, dst)
    return dst


def _wy_g4(gs, row, lane):
    """r57 KDA-WY32: state row ``row`` (a [V, K] fp32 gmem view), channels 4 lane .. 4 lane + 3: one 16-byte load."""
    src = cute.make_tensor(
        (gs.iterator + cute.crd2idx((row, lane * 4), gs.layout)).align(16), cute.make_layout((4,))
    )
    dst = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=128)
    cute.copy(atom, src, dst)
    return dst


@cute.kernel
def kda_wy_verify_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    P2_LANES_K: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    VS: cutlass.Constexpr[int],
    WY_R: cutlass.Constexpr[int],
    WY_PF: cutlass.Constexpr[int],
):
    """r57 KDA-WY32: C21's phase 1 verbatim (q / k / g / beta / v into smem, every ReplaySSM ring store, every
    conv-window scratch store), then the WY-form phase 2 in fp32: one streamed pass over the CTA's value rows of the
    committed state (ReplaySSM ring mode only: no state is written). Grid (H, n_req, VS)."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, i_vs = cute.arch.block_idx()
    head_off = i_hv * HEAD_DIM
    T_LOOP = 1 + NUM_SPEC
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P1_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_JOB_WARPS = (NUM_WARPS - P1_V_WARPS) // P1_NUM_JOBS
    P1_QKG_WARPS = P1_NUM_JOBS * P1_JOB_WARPS
    V_CH_PER_THREAD = TILE_K // (P1_V_WARPS * WARP_SIZE)
    P2_VEC = TILE_K // P2_LANES_K
    P1_NC = BLOCK_THREADS == BLOCK_THREADS_NARROW
    # WY phase 2: this CTA's WY_RV value rows; each warp streams WY_RPW of them in groups of WY_R rows, lane l owning
    # channels 4l..4l+3 of a row (one 16-byte load); a group's 6 * WY_R lane partials meet in a reduce-scatter whose
    # WY_H halving steps leave 3 sums a lane, then the butterfly over the remaining 5 - WY_H offsets.
    WY_RV = HEAD_DIM // VS
    WY_RPW = WY_RV // NUM_WARPS
    WY_NG = WY_RPW // WY_R
    WY_H = {1: 1, 2: 2, 4: 3}[WY_R]
    WY_SUB = 1 << (5 - WY_H)
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sVall = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
    )
    sOall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * WY_RV,)), 16)
    sGram = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((2 * T_LOOP * T_LOOP,)), 16
    )
    sQT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * T_LOOP,)), 16)
    if cutlass.const_expr(APPLY_ONORM):  # r57 WY-ONORM: phase 1's own staging targets (VS 1 only)
        sOnormGate = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
        sOnormWeight = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((HEAD_DIM,)), 16
        )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((2 * (KERNEL_WIDTH - 1) * VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    r_wq = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH * VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    r_w4 = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH,), stride=(1,)), cutlass.Float32
    )

    slot = ssm_state_indices[i_n]
    # CUDA-graph padding rows use slot == -1.
    if slot < 0:
        cute.arch.griddepcontrol_wait()
        pad_bos = cu_seqlens[i_n]
        for i_t in cutlass.range_constexpr(T_LOOP):
            if tidx < HEAD_DIM:
                o[0, pad_bos + i_t, i_hv, tidx] = cutlass.BFloat16(0.0)
        cute.arch.griddepcontrol_launch_dependents()
        # nvvm.exit, not `return`: the DSL rejects an early return out of a
        # staged if (UNSUP_EARLY_EXIT).
        nvvm.exit()

    # int64: `slot * stride` overflows int32 on envelope-strided pools.
    slot = cutlass.Int64(slot)

    # q/k/g each run on P1_JOB_WARPS warps split by token parity and the v-conv
    # takes the rest. Each token's conv is an independent window over globals,
    # so the split needs no cross-warp communication.
    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS
    # Only the q-conv warps read the q window and only the k-conv warps the k
    # window; the g and v warps overwrite r_state in phase 2 without reading it.
    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[w * VEC_SIZE + i] = cutlass.Float32(
                        cs_q[slot, head_off + k_idx, w]
                    )
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i] = (
                        cutlass.Float32(cs_k[slot, head_off + k_idx, w])
                    )

    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                cute.autovec_copy(w_q[(head_off + i * 32 + in_warp_tid, None)], r_w4)
                for w in range(KERNEL_WIDTH):
                    r_wq[w * VEC_SIZE + i] = r_w4[w]
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                cute.autovec_copy(w_k[(head_off + i * 32 + in_warp_tid, None)], r_w4)
                for w in range(KERNEL_WIDTH):
                    r_wq[w * VEC_SIZE + i] = r_w4[w]

    # r57 KDA-WY32: stock's pre-PDL state preload, as an L2 prefetch of this CTA's state rows (no smem staging).
    if cutlass.const_expr(WY_PF == 1):
        if tidx == 0:
            wy_pf = h0.iterator + cute.crd2idx((slot, i_hv, i_vs * WY_RV, 0), h0.layout)
            nvvm.cp_async_bulk_prefetch(
                wy_pf.llvm_ptr, cutlass.Int32(WY_RV * HEAD_DIM * 4).ir_value()
            )

    cute.arch.griddepcontrol_wait()

    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]
    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)

    # Private weights remove the initial cross-warp dependency.

    if warp_idx < P1_QKG_WARPS:
        if p1_job == 2:
            r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)
            # The gate warps do not use conv-window registers. Hold invariant
            # bias values there until phase 2 overwrites the state registers.
            for i in range(VEC_SIZE):
                r_state[i] = cutlass.Float32(
                    dt_bias[i_hv * HEAD_DIM + i * WARP_SIZE + in_warp_tid]
                )
        # Warp starting at token p1_par needs its window advanced that
        # many steps; the g path is pointwise and needs none.
        if p1_job == 0:
            for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                if _pi < cutlass.min(p1_par, n_tok):
                    for i in range(VEC_SIZE):
                        _xn = _ld_p1(
                            P1_NC, x_q, (0, bos + _pi, i_hv, i * 32 + in_warp_tid)
                        )
                        r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                        r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                        r_state[2 * VEC_SIZE + i] = _xn
        elif p1_job == 1:
            for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                if _pi < cutlass.min(p1_par, n_tok):
                    for i in range(VEC_SIZE):
                        _xn = _ld_p1(
                            P1_NC, x_k, (0, bos + _pi, i_hv, i * 32 + in_warp_tid)
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                            r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                        )
                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = _xn
        # One straight-line block per job, so the warp's token chains interleave.
        for _job in cutlass.range_constexpr(P1_NUM_JOBS):
            if p1_job == _job:
                for producer_step in cutlass.range_constexpr(
                    (T_LOOP + P1_JOB_WARPS - 1) // P1_JOB_WARPS
                ):
                    i_t = p1_par + producer_step * P1_JOB_WARPS
                    if cutlass.const_expr(T_LOOP % P1_JOB_WARPS == 0) or i_t < T_LOOP:
                        token = bos + cutlass.min(i_t, n_tok - 1)
                        if cutlass.const_expr(_job == 0):
                            for i_pair in range(VEC_SIZE // 2):
                                i0 = i_pair * 2
                                i1 = i_pair * 2 + 1
                                k_idx0 = i0 * 32 + in_warp_tid
                                k_idx1 = i1 * 32 + in_warp_tid
                                r_conv_0 = 0.0
                                r_conv_1 = 0.0
                                for w in range(KERNEL_WIDTH - 1):
                                    r_conv_0 += (
                                        r_state[w * VEC_SIZE + i0]
                                        * r_wq[w * VEC_SIZE + i0]
                                    )
                                    r_conv_1 += (
                                        r_state[w * VEC_SIZE + i1]
                                        * r_wq[w * VEC_SIZE + i1]
                                    )
                                r_xq_0 = _ld_p1(P1_NC, x_q, (0, token, i_hv, k_idx0))
                                r_xq_1 = _ld_p1(P1_NC, x_q, (0, token, i_hv, k_idx1))
                                _cwq_last_0 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i0]
                                _cwq_last_1 = r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i1]
                                r_conv_0 += r_xq_0 * _cwq_last_0
                                r_conv_1 += r_xq_1 * _cwq_last_1
                                e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                e1 = cute.math.exp(-r_conv_1, fastmath=True)
                                sig_0 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                                sig_1 = cute.arch.rcp_approx(cutlass.Float32(1.0) + e1)
                                r_q[i0] = r_conv_0 * sig_0
                                r_q[i1] = r_conv_1 * sig_1
                                r_state[0 * VEC_SIZE + i0] = r_state[1 * VEC_SIZE + i0]
                                r_state[0 * VEC_SIZE + i1] = r_state[1 * VEC_SIZE + i1]
                                r_state[1 * VEC_SIZE + i0] = r_state[2 * VEC_SIZE + i0]
                                r_state[1 * VEC_SIZE + i1] = r_state[2 * VEC_SIZE + i1]
                                r_state[2 * VEC_SIZE + i0] = r_xq_0
                                r_state[2 * VEC_SIZE + i1] = r_xq_1
                            sum_q = 0.0
                            for i in range(VEC_SIZE):
                                sum_q += r_q[i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_q += cute.arch.shuffle_sync_bfly(
                                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_q_scaled = (
                                cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                            )
                            for i in range(VEC_SIZE):
                                r_q[i] = r_q[i] * rnorm_q_scaled
                            for i in range(VEC_SIZE):
                                k_idx = i * WARP_SIZE + in_warp_tid
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sQ[
                                    i_t,
                                    qk_grp,
                                    qk_j // 4,
                                    qk_j % 4,
                                ] = r_q[i]
                        elif cutlass.const_expr(_job == 1):
                            r_b_raw = cutlass.Float32(0.0)
                            if in_warp_tid == 0:
                                r_b_raw = _ld_p1(P1_NC, beta, (0, token, i_hv))
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                r_conv = (
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i
                                    ]
                                    * r_wq[0 * VEC_SIZE + i]
                                )
                                r_conv += (
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                    ]
                                    * r_wq[1 * VEC_SIZE + i]
                                )
                                r_conv += (
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                    ]
                                    * r_wq[2 * VEC_SIZE + i]
                                )
                                r_xk = _ld_p1(P1_NC, x_k, (0, token, i_hv, k_idx))
                                r_conv += r_xk * r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i]
                                r_conv = r_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0)
                                    + cute.math.exp(-r_conv, fastmath=True)
                                )
                                r_k[i] = r_conv
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i
                                ] = r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                ]
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                ] = r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                ]
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                ] = r_xk
                            # Complete independent key chains before raw-ring stores.
                            if cutlass.const_expr(CACHE_RING):
                                for i in range(VEC_SIZE):
                                    k_idx = i * WARP_SIZE + in_warp_tid
                                    ring_rawk[slot, i_hv, i_t, k_idx] = (
                                        cutlass.BFloat16(r_k[i])
                                    )
                            sum_k = 0.0
                            for i in range(VEC_SIZE):
                                sum_k += r_k[i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_k += cute.arch.shuffle_sync_bfly(
                                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                            for i in range(VEC_SIZE):
                                r_k[i] = r_k[i] * rnorm_k
                            for i in range(VEC_SIZE):
                                k_idx = i * WARP_SIZE + in_warp_tid
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sK[
                                    i_t,
                                    qk_grp,
                                    qk_j // 4,
                                    qk_j % 4,
                                ] = r_k[i]
                            if in_warp_tid == 0:
                                sBeta[i_t] = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0)
                                    + cute.math.exp(-r_b_raw, fastmath=True)
                                )
                                if cutlass.const_expr(CACHE_RING):
                                    ring_beta[slot, i_hv, i_t] = sBeta[i_t]
                        else:
                            # Gate warps do not otherwise use r_q in this phase.
                            for i in range(VEC_SIZE):
                                r_q[i] = _ld_p1(
                                    P1_NC, g, (0, token, i_hv, i * 32 + in_warp_tid)
                                )
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                r_g_raw = r_q[i] + r_state[i]
                                exp_A_x = r_exp_A * r_g_raw
                                sigmoid_val = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0)
                                    + cute.math.exp(-exp_A_x, fastmath=True)
                                )
                                r_gk = lower_bound * sigmoid_val
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sG[
                                    i_t,
                                    qk_grp,
                                    qk_j // 4,
                                    qk_j % 4,
                                ] = cute.math.exp(r_gk, fastmath=True)
                                if cutlass.const_expr(CACHE_RING):
                                    ring_g[slot, i_hv, i_t, k_idx] = r_gk
                        if cutlass.const_expr(_job == 0):
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                for w in range(KERNEL_WIDTH - 1):
                                    intermediate_conv_q[
                                        scratch_row, i_t, head_off + k_idx, w
                                    ] = cutlass.BFloat16(r_state[w * VEC_SIZE + i])
                        elif cutlass.const_expr(_job == 1):
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                for w in range(KERNEL_WIDTH - 1):
                                    intermediate_conv_k[
                                        scratch_row, i_t, head_off + k_idx, w
                                    ] = cutlass.BFloat16(
                                        r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + w * VEC_SIZE
                                            + i
                                        ]
                                    )
                        # Reach token i_t + P1_JOB_WARPS. The conv body already
                        # advanced one step; clamp the rest so the final iteration's
                        # unread advance stays in bounds at the last request.
                        if cutlass.const_expr(_job == 0):
                            for _a in range(P1_JOB_WARPS - 1):
                                _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                                for i in range(VEC_SIZE):
                                    _xn = _ld_p1(
                                        P1_NC,
                                        x_q,
                                        (0, _nx, i_hv, i * 32 + in_warp_tid),
                                    )
                                    r_state[0 * VEC_SIZE + i] = r_state[
                                        1 * VEC_SIZE + i
                                    ]
                                    r_state[1 * VEC_SIZE + i] = r_state[
                                        2 * VEC_SIZE + i
                                    ]
                                    r_state[2 * VEC_SIZE + i] = _xn
                        elif cutlass.const_expr(_job == 1):
                            for _a in range(P1_JOB_WARPS - 1):
                                _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                                for i in range(VEC_SIZE):
                                    _xn = _ld_p1(
                                        P1_NC,
                                        x_k,
                                        (0, _nx, i_hv, i * 32 + in_warp_tid),
                                    )
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i
                                    ] = r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                    ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                    ] = r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                    ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                    ] = _xn
    else:
        for _c in range(V_CH_PER_THREAD):
            _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (HEAD_DIM // V_CH_PER_THREAD)
            _csv0 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 2])
            cute.autovec_copy(w_v[(head_off + _v_idx, None)], r_w4)
            _wv = [r_w4[w] for w in range(KERNEL_WIDTH)]
            # The v warps finish precompute early. Use their spare interval
            # to stage the independent onorm inputs before the existing barrier.
            if cutlass.const_expr(APPLY_ONORM):
                sOnormWeight[_v_idx] = cutlass.Float32(onorm_weight[_v_idx])
            # Sliding conv window, oldest -> newest.
            _win = [_csv0, _csv1, _csv2]
            for _t in cutlass.range_constexpr(T_LOOP):
                _win.append(
                    cutlass.Float32(
                        x_v[0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx]
                    )
                )
            for _t in cutlass.range_constexpr(T_LOOP):
                _vconv = _win[_t] * _wv[0]
                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    _vconv += _win[_t + _w] * _wv[_w]
                _vconv = _vconv * cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                )
                sVall[_t * HEAD_DIM + _v_idx] = _vconv
                if cutlass.const_expr(APPLY_ONORM):
                    gate_raw = _ld_p1(
                        P1_NC,
                        onorm_g,
                        (0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx),
                    )
                    sOnormGate[_t * HEAD_DIM + _v_idx] = cute.arch.rcp_approx(
                        cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True)
                    )
                if cutlass.const_expr(CACHE_RING):
                    ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, head_off + _v_idx, _w] = (
                        cutlass.BFloat16(_win[_t + 1 + _w])
                    )

    # r57 KDA-WY32 phase 2 (WY form, fp32; op order = r57/wy/wy_emu.py wy_p2).
    cute.arch.barrier()
    sQf = cute.make_tensor(sQ.iterator, cute.make_layout((T_LOOP, HEAD_DIM), stride=(HEAD_DIM, 1)))
    sKf = cute.make_tensor(sK.iterator, cute.make_layout((T_LOOP, HEAD_DIM), stride=(HEAD_DIM, 1)))
    sGf = cute.make_tensor(sG.iterator, cute.make_layout((T_LOOP, HEAD_DIM), stride=(HEAD_DIM, 1)))
    sVf = cute.make_tensor(sVall.iterator, cute.make_layout((T_LOOP, HEAD_DIM), stride=(HEAD_DIM, 1)))
    sOf = cute.make_tensor(sOall.iterator, cute.make_layout((T_LOOP, WY_RV), stride=(WY_RV, 1)))
    # 1. Per channel: decay products P_t (running product of phase 1's multipliers a_t = sG), kh = k P (into sK),
    #    qh = q P (into sQ), kt = k / P (into sG).
    if tidx < HEAD_DIM:
        wy_p = sGf[0, tidx]
        for t in cutlass.range_constexpr(T_LOOP):
            if cutlass.const_expr(t > 0):
                wy_p = _wy_mul(wy_p, sGf[t, tidx])
            wy_k = sKf[t, tidx]
            sKf[t, tidx] = _wy_mul(wy_k, wy_p)
            sQf[t, tidx] = _wy_mul(sQf[t, tidx], wy_p)
            sGf[t, tidx] = wy_k / wy_p
    cute.arch.barrier()
    # 2. Gram dots: M[i][j] = kh_i . kt_j (j < i) at sGram[i T + j], L[i][j] = qh_i . kt_j (j <= i) at
    #    sGram[T^2 + i T + j]; warp w takes dots w, w + NUM_WARPS, ...; lane l: channels 4l..4l+3, then the butterfly.
    wy_dots = []
    for i in cutlass.range_constexpr(T_LOOP):
        for j in cutlass.range_constexpr(i):
            wy_dots.append((0, i, j))
    for i in cutlass.range_constexpr(T_LOOP):
        for j in cutlass.range_constexpr(i + 1):
            wy_dots.append((1, i, j))
    for w in cutlass.range_constexpr(NUM_WARPS):
        if warp_idx == w:
            # This warp's dots: all lane partials first, then the butterflies interleaved, then lane 0's stores.
            wy_mine = [wy_dots[d] for d in range(w, len(wy_dots), NUM_WARPS)]
            wy_accs = []
            for d in cutlass.range_constexpr(len(wy_mine)):
                if cutlass.const_expr(wy_mine[d][0] == 0):
                    wy_x = _wy_s4(sKf, wy_mine[d][1], in_warp_tid)
                else:
                    wy_x = _wy_s4(sQf, wy_mine[d][1], in_warp_tid)
                wy_y = _wy_s4(sGf, wy_mine[d][2], in_warp_tid)
                wy_acc = _wy_mul(wy_x[0], wy_y[0])
                for e in cutlass.range_constexpr(1, 4):
                    wy_acc = cute.math.fma(wy_x[e], wy_y[e], wy_acc)
                wy_accs.append(wy_acc)
            for offset in [16, 8, 4, 2, 1]:
                for d in cutlass.range_constexpr(len(wy_mine)):
                    wy_accs[d] = _wy_add(
                        wy_accs[d],
                        cute.arch.shuffle_sync_bfly(wy_accs[d], offset=offset, mask=-1, mask_and_clamp=31),
                    )
            if in_warp_tid == 0:
                for d in cutlass.range_constexpr(len(wy_mine)):
                    sGram[
                        wy_mine[d][0] * T_LOOP * T_LOOP + wy_mine[d][1] * T_LOOP + wy_mine[d][2]
                    ] = wy_accs[d]
    cute.arch.barrier()
    # 3. Warp 0, lane c < T: column c of Tm by forward substitution (rhs = beta_i e_i), then column c of QT = L Tm.
    if warp_idx == 0:
        if in_warp_tid < T_LOOP:
            wy_tm = []
            for i in cutlass.range_constexpr(T_LOOP):
                wy_acc = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(i):
                    wy_bm = _wy_mul(sBeta[i], sGram[i * T_LOOP + j])
                    wy_acc = cute.math.fma(wy_bm, wy_tm[j], wy_acc)
                wy_rhs = cutlass.Float32(
                    cutlass.select_(in_warp_tid == i, sBeta[i], cutlass.Float32(0.0))
                )
                wy_tm.append(_wy_sub(wy_rhs, wy_acc))
            for t in cutlass.range_constexpr(T_LOOP):
                wy_acc = _wy_mul(sGram[T_LOOP * T_LOOP + t * T_LOOP], wy_tm[0])
                for s in cutlass.range_constexpr(1, t + 1):
                    wy_acc = cute.math.fma(sGram[T_LOOP * T_LOOP + t * T_LOOP + s], wy_tm[s], wy_acc)
                sQT[t * T_LOOP + in_warp_tid] = wy_acc
    cute.arch.barrier()
    # 4. Per channel c: A[t][c] = qh[t][c] - sum_{s<=t} QT[t][s] kh[s][c] (into sQ). Per value row j of this CTA:
    #    W[t][j] = sum_{s<=t} QT[t][s] v[s][j] (into sVall, same place).
    if tidx < HEAD_DIM:
        wy_qh = [sQf[t, tidx] for t in range(T_LOOP)]
        wy_kh = [sKf[t, tidx] for t in range(T_LOOP)]
        for t in cutlass.range_constexpr(T_LOOP):
            wy_acc = _wy_mul(sQT[t * T_LOOP], wy_kh[0])
            for s in cutlass.range_constexpr(1, t + 1):
                wy_acc = cute.math.fma(sQT[t * T_LOOP + s], wy_kh[s], wy_acc)
            sQf[t, tidx] = _wy_sub(wy_qh[t], wy_acc)
    if cutlass.const_expr(BLOCK_THREADS >= HEAD_DIM + WY_RV):
        wy_wj = tidx - HEAD_DIM
    else:
        wy_wj = tidx
    if wy_wj >= 0:
        if wy_wj < WY_RV:
            wy_j = i_vs * WY_RV + wy_wj
            wy_v = [sVf[t, wy_j] for t in range(T_LOOP)]
            for t in cutlass.range_constexpr(T_LOOP):
                wy_acc = _wy_mul(sQT[t * T_LOOP], wy_v[0])
                for s in cutlass.range_constexpr(1, t + 1):
                    wy_acc = cute.math.fma(sQT[t * T_LOOP + s], wy_v[s], wy_acc)
                sVf[t, wy_j] = wy_acc
    cute.arch.barrier()
    # 5. Stream this CTA's state rows: lane l holds A[t][4l..4l+3] (24 registers); per group of WY_R rows, one 16-byte
    #    load a row, 6 dot partials a row (packed fma pairs over tokens), reduce-scatter to the lanes, sums to sOall.
    wy_a = [_wy_s4(sQf, t, in_warp_tid) for t in range(T_LOOP)]
    # Every global read but this CTA's state rows has landed (phase 1); the state is ReplaySSM-frozen (not written by
    # the preceding kernels in the stream: stock preloads it before griddepcontrol_wait).
    cute.arch.griddepcontrol_launch_dependents()
    wy_g = h0[(slot, i_hv, None, None)]
    for wy_grp in cutlass.range(WY_NG):
        wy_row0 = warp_idx * WY_RPW + wy_grp * WY_R
        wy_s = [_wy_g4(wy_g, i_vs * WY_RV + wy_row0 + r, in_warp_tid) for r in range(WY_R)]
        wy_cur = []
        for r in cutlass.range_constexpr(WY_R):
            for tp in cutlass.range_constexpr(T_LOOP // 2):
                wy_p0, wy_p1 = cute.arch.mul_packed_f32x2(
                    (wy_a[2 * tp][0], wy_a[2 * tp + 1][0]), (wy_s[r][0], wy_s[r][0])
                )
                for e in cutlass.range_constexpr(1, 4):
                    wy_p0, wy_p1 = cute.arch.fma_packed_f32x2(
                        src_a=(wy_a[2 * tp][e], wy_a[2 * tp + 1][e]),
                        src_b=(wy_s[r][e], wy_s[r][e]),
                        src_c=(wy_p0, wy_p1),
                    )
                wy_cur.append(wy_p0)
                wy_cur.append(wy_p1)
        for h in cutlass.range_constexpr(WY_H):
            wy_bit = (in_warp_tid >> (4 - h)) & 1
            wy_half = len(wy_cur) // 2
            wy_nxt = []
            for i in cutlass.range_constexpr(wy_half):
                wy_send = cutlass.Float32(
                    cutlass.select_(wy_bit == 0, wy_cur[i + wy_half], wy_cur[i])
                )
                wy_keep = cutlass.Float32(
                    cutlass.select_(wy_bit == 0, wy_cur[i], wy_cur[i + wy_half])
                )
                wy_nxt.append(
                    _wy_add(
                        wy_keep,
                        cute.arch.shuffle_sync_bfly(
                            wy_send, offset=16 >> h, mask=-1, mask_and_clamp=31
                        ),
                    )
                )
            wy_cur = wy_nxt
        for h in cutlass.range_constexpr(WY_H, 5):
            wy_nxt = []
            for i in cutlass.range_constexpr(3):
                wy_nxt.append(
                    _wy_add(
                        wy_cur[i],
                        cute.arch.shuffle_sync_bfly(
                            wy_cur[i], offset=16 >> h, mask=-1, mask_and_clamp=31
                        ),
                    )
                )
            wy_cur = wy_nxt
        wy_q = in_warp_tid & (WY_SUB - 1)
        if wy_q < 3:
            wy_vi = 3 * (in_warp_tid >> (5 - WY_H)) + wy_q
            wy_r = wy_vi // T_LOOP
            wy_dot = cutlass.Float32(
                cutlass.select_(
                    wy_q == 0,
                    wy_cur[0],
                    cutlass.Float32(cutlass.select_(wy_q == 1, wy_cur[1], wy_cur[2])),
                )
            )
            if cutlass.const_expr(APPLY_ONORM):  # r59 KW: o = dot + W (fp32), staged for stock's norm epilogue
                wy_dot = _wy_add(wy_dot, sVf[wy_vi - wy_r * T_LOOP, i_vs * WY_RV + wy_row0 + wy_r])
            sOf[wy_vi - wy_r * T_LOOP, wy_row0 + wy_r] = wy_dot
    # 6. Epilogue: one warp per verify row. APPLY_ONORM: stock's norm on the staged fp32 o (K3), one BF16 rounding; else o = dot + W -> bf16 (C22's coalesced row stores).
    cute.arch.barrier()
    if cutlass.const_expr(APPLY_ONORM):  # r59 KW: stock's APPLY_ONORM epilogue (kda_decode_mtp.py :827-:846, verbatim) on o
        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            sumsq = cutlass.Float32(0.0)
            for i in range(VEC_SIZE):
                _o = sOall[i_t * HEAD_DIM + i * 32 + in_warp_tid]
                sumsq += _o * _o
            for offset in [16, 8, 4, 2, 1]:
                sumsq += cute.arch.shuffle_sync_bfly(
                    sumsq, offset=offset, mask=-1, mask_and_clamp=31
                )
            rms = cute.math.rsqrt(
                sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True
            )
            for i in range(VEC_SIZE):
                v_idx = i * 32 + in_warp_tid
                raw_o = sOall[i_t * HEAD_DIM + v_idx]
                gate = sOnormGate[i_t * HEAD_DIM + v_idx]
                weight = sOnormWeight[v_idx]
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                        raw_o * rms * weight * gate
                    )
    else:
        for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
            for i in range(WY_RV // WARP_SIZE):
                wy_jl = i * WARP_SIZE + in_warp_tid
                if i_t < n_tok:
                    o[0, bos + i_t, i_hv, i_vs * WY_RV + wy_jl] = cutlass.BFloat16(
                        _wy_add(sOf[i_t, wy_jl], sVf[i_t, i_vs * WY_RV + wy_jl])
                    )


@cute.jit
def _run_kda_wy(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    n_req: cutlass.Int32,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    CACHE_RING: cutlass.Constexpr[bool],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    VS: cutlass.Constexpr[int],
    WY_R: cutlass.Constexpr[int],
    WY_PF: cutlass.Constexpr[int],
    WY_MINB: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """r59 KW: stock's launcher arguments in ring mode (no intermediate_ssm; one compile per variant serves every N), launching kda_wy_verify_kernel
    on grid (H, n_req, VS). Smem: q / k / g tiles in plain [token, channel] order (32 lanes: P2_VEC 4), beta, v, the
    staged outputs of this CTA's rows, the Gram dots and QT (~17 KB at T 6, VS 1); no state stage."""
    t_loop = 1 + NUM_SPEC
    smem_qk_layout = cute.make_layout(
        (t_loop, WY_LANES, 1, 4), stride=(TILE_K, 1, 4 * WY_LANES, WY_LANES)
    )
    if cutlass.const_expr(APPLY_ONORM and VS != 1):  # r57 WY-ONORM: a CTA must own the whole head
        raise ValueError(f"WY-ONORM needs VS 1 (got VS {VS})")
    smem_bytes = _wy_smem_bytes(t_loop, TILE_K // VS, APPLY_ONORM)
    kda_wy_verify_kernel(
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ring_rawv,
        ring_rawk,
        ring_g,
        ring_beta,
        onorm_g,
        onorm_weight,
        smem_qk_layout,
        ssm_state_indices,
        cu_seqlens,
        scale,
        NUM_SPEC,
        BLOCK_THREADS,
        WY_LANES,
        lower_bound,
        CACHE_RING,
        APPLY_ONORM,
        onorm_eps,
        VS,
        WY_R,
        WY_PF,
    ).launch(
        grid=(H, n_req, VS),
        block=[BLOCK_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        use_pdl=True,
        min_blocks_per_mp=WY_MINB,
    )


def _tensor_layout_key(tensor, fits_32bit_stride):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        fits_32bit_stride,
    )


def _fits_32bit_stride(tensor):
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        if abs(int(stride)) > 2**31 - 1:
            return False
        if size:
            max_offset += (int(size) - 1) * abs(int(stride))
            if max_offset > 2**31 - 1:
                return False
    return True


def _cute_tensor(tensor, *, dynamic=True, fits_32bit_stride=None):
    from cutlass.cute.runtime import from_dlpack

    if tensor.requires_grad:
        tensor = tensor.detach()
    if fits_32bit_stride is None:
        fits_32bit_stride = _fits_32bit_stride(tensor)
    value = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=fits_32bit_stride,
    )
    leading_dim = next(
        (dim for dim, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if not dynamic or leading_dim is None:
        return value
    return value.mark_layout_dynamic(leading_dim)


_COMPILED = {}
# The launcher's tensors whose extents follow the request count (by position in a call's args): compiled with dynamic
# layouts, so one compile serves every request count. The rest keep static layouts; the state pool always with 64-bit
# offsets, so every layer's view of SGLang's shared pool is one compile (kda_v2_verify.py does the same).
_DYNAMIC = frozenset((1, 2, 3, 11, 13, 14, 15, 19, 20, 25))
_OUT = 14                    # the output's position in a call's args (allocated by wy32_launch)
_JIT = []                    # (H, plan, seconds) of every compile this process made
TOKENS = 6                   # 1 bonus + 5 drafts a request: the one verify form phase 2 is written for


def _plan(*, H, N, lower_bound, variant):
    """(block threads, VS, R, PF, MINB) for a call, or why this kernel declines it (a str). Default: the shipped
    literals at VS 1 (the norm epilogue needs one CTA to own a head's 128 values) above one wave (H * N > SMs); at or
    below one wave the caller's path serves. ``variant`` (kits only): that 5-tuple at any N, VS 1."""
    if not -float(lower_bound) * TOKENS <= KW_WY_MAX_LOGDECAY:
        return f"lower_bound {lower_bound}: the decay product could leave fp32's normal range"
    if variant is None:
        sms = current_arch().num_sms
        if H * N <= sms:
            return f"one wave ({H} heads x {N} requests <= {sms} SMs)"
        variant = (KW_WY_THREADS, 1, KW_WY_R, KW_WY_PF, KW_WY_MINB)
    bt, vs, r, pf, minb = (int(x) for x in variant)
    rpw = (TILE_K // vs) // (bt // WARP_SIZE) if bt in (128, 256, 512) and vs == 1 else 0
    if r not in (1, 2, 4) or pf not in (0, 1) or minb < 0 or rpw < r or rpw % r:
        raise ValueError(f"KDA-WY32: bad variant {variant} (VS 1 only: the norm epilogue owns a whole head)")
    return bt, vs, r, pf, minb


def _aligned(tensor) -> bool:
    return tensor.data_ptr() % 16 == 0


def wy32_call(*, x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v, g, beta, A_log, dt_bias, recurrent_state,
              intermediate_ssm, intermediate_state_indices, intermediate_conv_q, intermediate_conv_k,
              intermediate_conv_v, ssm_state_indices, cu_seqlens, lower_bound, scale=None, replayssm_rawv=None,
              replayssm_rawk=None, replayssm_g=None, replayssm_beta=None, onorm_gate=None, onorm_weight=None,
              onorm_eps=None, variant=None):
    """fused_kda_decode_mtp_dspark's keywords -> (None, call) when this kernel serves the call, else (why, None).
    Host facts only (shapes, dtypes, strides, alignment, devices, the SM count): nothing is allocated, compiled or
    launched. The served form: the fused output norm (all three onorm_* given), the four ReplaySSM rings without
    per-step snapshots, an fp32 [slots, H, 128, 128] state pool with 16-byte-aligned slots, six dense tokens a request,
    above one wave, the decay product inside fp32's normal range, every tensor 16-byte aligned on one device."""
    rings = (replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta)
    if onorm_gate is None or onorm_weight is None or onorm_eps is None:
        return "no fused output norm", None
    if intermediate_ssm is not None:
        return "per-step state snapshots asked for", None
    if any(ring is None for ring in rings):
        return "no ReplaySSM rings", None
    H, T, N = x_q.shape[2], x_q.shape[1], cu_seqlens.numel() - 1
    shape = (1, T, H, TILE_K)
    if (tuple(x_q.shape) != shape or tuple(x_k.shape) != shape or tuple(x_v.shape) != shape
            or tuple(g.shape) != shape or tuple(beta.shape) != (1, T, H) or tuple(onorm_gate.shape) != shape):
        return "q / k / v / g / beta / gate shapes", None
    if N <= 0 or T != TOKENS * N:
        return f"{T} tokens for {N} requests (not {TOKENS} a request)", None
    s = recurrent_state
    if (s.dtype != torch.float32 or tuple(s.shape[1:]) != (H, TILE_K, TILE_K)
            or tuple(s.stride()[-3:]) != (TILE_K * TILE_K, TILE_K, 1) or s.stride(0) % 4 or s.storage_offset() % 4):
        return "the state pool is not fp32 [slots, H, 128, 128] with 16-byte-aligned slots", None
    ring_len = replayssm_rawv.shape[2]
    if (ring_len < TOKENS or tuple(replayssm_rawv.shape[1:]) != (H, ring_len, TILE_K)
            or tuple(replayssm_rawk.shape[1:]) != (H, ring_len, TILE_K)
            or tuple(replayssm_g.shape[1:]) != (H, ring_len, TILE_K) or tuple(replayssm_beta.shape[1:]) != (H, ring_len)
            or replayssm_rawv.dtype != torch.bfloat16 or replayssm_rawk.dtype != torch.bfloat16
            or replayssm_g.dtype != torch.float32 or replayssm_beta.dtype != torch.float32):
        return "ReplaySSM ring layouts / dtypes", None
    if (onorm_gate.dtype != torch.bfloat16 or onorm_weight.dtype != torch.float32
            or tuple(onorm_weight.shape) != (TILE_K,) or tuple(onorm_weight.stride()) != (1,)
            or not 0.0 < float(onorm_eps) < float("inf")):
        return "output-norm gate / weight / eps", None
    if (ssm_state_indices.dtype != torch.int32 or cu_seqlens.dtype != torch.int32
            or intermediate_state_indices.dtype not in (torch.int32, torch.int64) or ssm_state_indices.numel() < N
            or intermediate_state_indices.numel() < N):
        return "index dtypes / lengths", None
    if scale is None:
        scale = TILE_K**-0.5
    plan = _plan(H=H, N=N, lower_bound=lower_bound, variant=variant)
    if isinstance(plan, str):
        return plan, None
    args = (recurrent_state, x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v, A_log, g, dt_bias, beta, None,
            intermediate_state_indices, intermediate_conv_q, intermediate_conv_k, intermediate_conv_v,
            ssm_state_indices, cu_seqlens, replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta, onorm_gate,
            onorm_weight)
    real = [t for i, t in enumerate(args) if i != _OUT]
    if any(t.device != x_v.device for t in real) or not all(_aligned(t) for t in real):
        return "devices / 16-byte alignment", None
    out_fits = T * H * TILE_K - 1 <= 2**31 - 1          # torch.empty_like(x_v): a contiguous [1, T, H, 128]
    fits = tuple(False if i == 0 else out_fits if i == _OUT else _fits_32bit_stride(t) for i, t in enumerate(args))
    key = (
        current_arch().cuda.capability,
        H,
        plan,
        float(onorm_eps),
        float(scale),
        float(lower_bound),
        *(
            (x_v.device, x_v.dtype, 4, fits[i], 3) if i == _OUT
            else (t.device, t.dtype, t.dim(), fits[i], next((d for d, st in enumerate(t.stride()) if st == 1), None))
            if i in _DYNAMIC
            else _tensor_layout_key(t, fits[i])
            for i, t in enumerate(args)
        ),
    )
    return None, (key, args, fits, N, H, float(scale), float(lower_bound), float(onorm_eps), plan, x_v)


def wy32_compiled(key) -> bool:
    return key in _COMPILED


def wy32_launch(call):
    """One served call (its variant compiled at its first launch): the output, allocated as stock's
    (torch.empty_like(x_v)), written on the current stream by one PDL launch; no other allocation, no host sync."""
    key, args, fits, N, H, scale, lower_bound, eps, plan, x_v = call
    out = torch.empty_like(x_v)
    if out.stride(-1) != 1 or _fits_32bit_stride(out) != fits[_OUT]:
        raise RuntimeError("KDA-WY32: the output's layout is not the one its compile key holds")
    args = args[:_OUT] + (out,) + args[_OUT + 1:]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _COMPILED.get(key)
    if compiled is None:
        t0 = time.monotonic()
        compiled = cute.compile(
            _run_kda_wy,
            *(_cute_tensor(t, dynamic=i in _DYNAMIC, fits_32bit_stride=f) for i, (t, f) in enumerate(zip(args, fits))),
            N,
            scale=scale,
            H=H,
            NUM_SPEC=TOKENS - 1,
            BLOCK_THREADS=plan[0],
            lower_bound=lower_bound,
            CACHE_RING=True,
            APPLY_ONORM=True,
            onorm_eps=eps,
            VS=plan[1],
            WY_R=plan[2],
            WY_PF=plan[3],
            WY_MINB=plan[4],
            stream=stream,
        )
        _COMPILED[key] = compiled
        _JIT.append((H, plan, round(time.monotonic() - t0, 2)))
    compiled(*(_cute_tensor(t, fits_32bit_stride=f) for t, f in zip(args, fits)), N, stream)
    return out
