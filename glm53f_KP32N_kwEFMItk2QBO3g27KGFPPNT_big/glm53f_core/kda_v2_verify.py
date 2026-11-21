from __future__ import annotations
import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync
from sglang.kernels.arch import current_arch
from sglang.kernels.jit.cute_aot_cache import get_jit_cache
from sglang.kernels.ops.attention import kda_decode_mtp as mtp
WARP_SIZE = mtp.WARP_SIZE
TILE_K = mtp.TILE_K
HEAD_DIM = mtp.HEAD_DIM
KERNEL_WIDTH = mtp.KERNEL_WIDTH
VEC_SIZE = mtp.VEC_SIZE
P1_NUM_JOBS = mtp.P1_NUM_JOBS
P1_V_WARP_DIVISOR = mtp.P1_V_WARP_DIVISOR
P2_LANES_K = mtp.P2_LANES_RESIDENT
_ld_p1 = mtp._ld_p1
ROW_BLOCKS = HEAD_DIM // WARP_SIZE
CHUNK = 16
STAGES = 2
STAGE_ELEMS = WARP_SIZE * CHUNK
def _issue_stage(atom, gStateV, sStageV, lane, row_block, chunk, stage) -> None:
    for i in range(WARP_SIZE // 8):
        row = i * 8 + lane // 4
        cute.copy(atom, gStateV[(None, lane % 4, chunk, row_block * WARP_SIZE + row)],
                  sStageV[(None, (lane % 4) ^ ((row // 2) % 4), row, stage)])
    cute.arch.cp_async_commit_group()
def _ia(t: int, s: int) -> int:
    return t * (t - 1) // 2 + s
def _ib(t: int, s: int, T: int) -> int:
    return T * (T - 1) // 2 + _ia(t, s)
def _ic(t: int, T: int) -> int:
    return T * (T - 1) + t
def _coef_plan(t_loop: int, warps: int) -> list:
    units = [(t, s) for t in range(t_loop - 1, -1, -1) for s in [None] + list(range(t - 1, -1, -1))]
    cap = max(2, -(-t_loop * t_loop // warps))
    while True:
        plan, cur, used = [], [], 0
        for t, s in units:
            size = 1 if s is None else 2
            if used + size > cap:
                plan.append(cur)
                cur, used = [], 0
            cur.append((t, s))
            used += size
        plan.append(cur)
        if len(plan) <= warps:
            break
        cap += 1
    groups = []
    for warp_units in plan + [[] for _ in range(warps - len(plan))]:
        mine = []
        for t, s in warp_units:
            if mine and mine[-1][0] == t:
                mine[-1][2].append(s)
            else:
                mine.append((t, int(s is None), [] if s is None else [s]))
        groups.append(mine)
    return groups


@cute.kernel
def kda_v2_verify_kernel(
    h0: cute.Tensor, x_q: cute.Tensor, x_k: cute.Tensor, x_v: cute.Tensor,
    w_q: cute.Tensor, w_k: cute.Tensor, w_v: cute.Tensor, cs_q: cute.Tensor, cs_k: cute.Tensor, cs_v: cute.Tensor,
    A_log: cute.Tensor, g: cute.Tensor, dt_bias: cute.Tensor, beta: cute.Tensor, o: cute.Tensor,
    intermediate_state_indices: cute.Tensor, intermediate_conv_q: cute.Tensor, intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor, ring_rawv: cute.Tensor, ring_rawk: cute.Tensor, ring_g: cute.Tensor,
    ring_beta: cute.Tensor, onorm_g: cute.Tensor, onorm_weight: cute.Tensor, smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor, cu_seqlens: cute.Tensor,
    scale: cutlass.Constexpr[float], NUM_SPEC: cutlass.Constexpr[int], BLOCK_THREADS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float], onorm_eps: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, _ = cute.arch.block_idx()
    head_off = i_hv * HEAD_DIM
    T_LOOP = 1 + NUM_SPEC
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P1_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_JOB_WARPS = (NUM_WARPS - P1_V_WARPS) // P1_NUM_JOBS
    P1_QKG_WARPS = P1_NUM_JOBS * P1_JOB_WARPS
    V_CH_PER_THREAD = TILE_K // (P1_V_WARPS * WARP_SIZE)
    P2_VEC = TILE_K // P2_LANES_K
    KSPLIT = NUM_WARPS // ROW_BLOCKS
    LOADS = TILE_K // CHUNK // KSPLIT
    P1_NC = BLOCK_THREADS == mtp.BLOCK_THREADS_NARROW
    smem = cutlass.utils.SmemAllocator()
    sStage = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS * STAGES * STAGE_ELEMS,)), 128)
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16)
    sOall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16)
    sOnormGate = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16)
    sOnormWeight = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HEAD_DIM,)), 16)
    sQp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * TILE_K,)), 16)
    sKp = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * TILE_K,)), 16)
    sCoef = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * T_LOOP,)), 16)
    sGw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_WARPS * P2_LANES_K * P2_LANES_K,)), 16)
    sZ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP * TILE_K,)), 16)
    sPart = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((KSPLIT - 1) * T_LOOP * HEAD_DIM,)), 16)
    r_q = cute.make_rmem_tensor(cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32)

    R_STATE_ELEMS = max(CHUNK, 2 * (KERNEL_WIDTH - 1) * VEC_SIZE)
    r_state = cute.make_rmem_tensor(cute.make_layout((R_STATE_ELEMS,), stride=(1,)), cutlass.Float32)
    r_wq = cute.make_rmem_tensor(cute.make_layout((KERNEL_WIDTH * VEC_SIZE,), stride=(1,)), cutlass.Float32)
    r_w4 = cute.make_rmem_tensor(cute.make_layout((KERNEL_WIDTH,), stride=(1,)), cutlass.Float32)


    N_STEPS = (T_LOOP + P1_JOB_WARPS - 1) // P1_JOB_WARPS
    r_xpre = cute.make_rmem_tensor(cute.make_layout((max(1, P1_JOB_WARPS - 1) * VEC_SIZE,), stride=(1,)),
                                   cutlass.Float32)
    r_xtok = cute.make_rmem_tensor(cute.make_layout((N_STEPS * VEC_SIZE,), stride=(1,)), cutlass.Float32)
    r_xadv = cute.make_rmem_tensor(cute.make_layout((max(1, (N_STEPS - 1) * (P1_JOB_WARPS - 1)) * VEC_SIZE,),
                                                    stride=(1,)), cutlass.Float32)
    r_bet = cute.make_rmem_tensor(cute.make_layout((N_STEPS,), stride=(1,)), cutlass.Float32)

    slot = ssm_state_indices[i_n]



    bos = cu_seqlens[i_n]
    eos = cu_seqlens[i_n + 1]

    if slot < 0:
        cute.arch.griddepcontrol_wait()
        for i_t in cutlass.range_constexpr(T_LOOP):
            if tidx < HEAD_DIM:
                o[0, bos + i_t, i_hv, tidx] = cutlass.BFloat16(0.0)
        cute.arch.griddepcontrol_launch_dependents()
        nvvm.exit()

    slot = cutlass.Int64(slot)



    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS
    if warp_idx < P1_QKG_WARPS:
        if p1_job == 0:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[w * VEC_SIZE + i] = cutlass.Float32(cs_q[slot, head_off + k_idx, w])
        elif p1_job == 1:
            for i in range(VEC_SIZE):
                k_idx = i * 32 + in_warp_tid
                for w in range(KERNEL_WIDTH - 1):
                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i] = (
                        cutlass.Float32(cs_k[slot, head_off + k_idx, w]))

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



    row_block = warp_idx % ROW_BLOCKS
    kq = warp_idx // ROW_BLOCKS
    state_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.Float32,
                                     num_bits_per_copy=128)
    gStateV = cute.make_tensor(h0[(slot, i_hv, None, None)].iterator, cute.make_layout(
        (4, 4, TILE_K // CHUNK, HEAD_DIM), stride=(1, 4, CHUNK, TILE_K)))
    sStageV = cute.make_tensor(sStage.iterator, cute.make_layout(
        (4, 4, WARP_SIZE, NUM_WARPS * STAGES), stride=(1, 4, CHUNK, STAGE_ELEMS)))
    for s_l in cutlass.range_constexpr(min(STAGES, LOADS)):
        _issue_stage(state_atom, gStateV, sStageV, in_warp_tid, row_block, s_l * KSPLIT + kq, warp_idx * STAGES + s_l)

    cute.arch.griddepcontrol_wait()

    n_tok = eos - bos
    scratch_row = intermediate_state_indices[i_n]
    r_exp_A = cutlass.Float32(0.0)
    if warp_idx < P1_QKG_WARPS:
        for _job in cutlass.range_constexpr(P1_NUM_JOBS):
            if p1_job == _job:
                _src = x_q if _job == 0 else (x_k if _job == 1 else g)
                _o = (KERNEL_WIDTH - 1) * VEC_SIZE if _job == 1 else 0



                if cutlass.const_expr(_job == 2):
                    _a_log = cutlass.Float32(A_log[i_hv])
                    for i in range(VEC_SIZE):
                        r_state[i] = cutlass.Float32(dt_bias[i_hv * HEAD_DIM + i * WARP_SIZE + in_warp_tid])
                if cutlass.const_expr(_job < 2):
                    for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                        for i in range(VEC_SIZE):
                            r_xpre[_pi * VEC_SIZE + i] = _ld_p1(P1_NC, _src, (0, bos + _pi, i_hv, i * 32 + in_warp_tid))
                for producer_step in cutlass.range_constexpr(N_STEPS):
                    i_t = p1_par + producer_step * P1_JOB_WARPS
                    token = bos + cutlass.min(i_t, n_tok - 1)
                    for i in range(VEC_SIZE):
                        r_xtok[producer_step * VEC_SIZE + i] = _ld_p1(
                            P1_NC, _src, (0, token, i_hv, i * 32 + in_warp_tid))
                    if cutlass.const_expr(_job == 1):
                        r_bet[producer_step] = _ld_p1(P1_NC, beta, (0, token, i_hv))
                    if cutlass.const_expr(_job < 2 and producer_step + 1 < N_STEPS):
                        for _a in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                            _nx = bos + cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                            for i in range(VEC_SIZE):
                                r_xadv[(producer_step * (P1_JOB_WARPS - 1) + _a) * VEC_SIZE + i] = _ld_p1(
                                    P1_NC, _src, (0, _nx, i_hv, i * 32 + in_warp_tid))

                if cutlass.const_expr(_job == 2):
                    r_exp_A = cute.math.exp(_a_log, fastmath=True)
                if cutlass.const_expr(_job < 2):
                    for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                        if _pi < cutlass.min(p1_par, n_tok):
                            for i in range(VEC_SIZE):
                                _xn = r_xpre[_pi * VEC_SIZE + i]
                                r_state[_o + 0 * VEC_SIZE + i] = r_state[_o + 1 * VEC_SIZE + i]
                                r_state[_o + 1 * VEC_SIZE + i] = r_state[_o + 2 * VEC_SIZE + i]
                                r_state[_o + 2 * VEC_SIZE + i] = _xn
                for producer_step in cutlass.range_constexpr(N_STEPS):
                    i_t = p1_par + producer_step * P1_JOB_WARPS
                    if cutlass.const_expr(T_LOOP % P1_JOB_WARPS == 0) or i_t < T_LOOP:
                        if cutlass.const_expr(_job == 0):
                            for i_pair in range(VEC_SIZE // 2):
                                i0 = i_pair * 2
                                i1 = i_pair * 2 + 1
                                r_conv_0 = 0.0
                                r_conv_1 = 0.0
                                for w in range(KERNEL_WIDTH - 1):
                                    r_conv_0 += r_state[w * VEC_SIZE + i0] * r_wq[w * VEC_SIZE + i0]
                                    r_conv_1 += r_state[w * VEC_SIZE + i1] * r_wq[w * VEC_SIZE + i1]
                                r_xq_0 = r_xtok[producer_step * VEC_SIZE + i0]
                                r_xq_1 = r_xtok[producer_step * VEC_SIZE + i1]
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
                                sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                            rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                            for i in range(VEC_SIZE):
                                r_q[i] = r_q[i] * rnorm_q_scaled
                            for i in range(VEC_SIZE):
                                k_idx = i * WARP_SIZE + in_warp_tid
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sQ[i_t, qk_grp, qk_j // 4, qk_j % 4] = r_q[i]
                        elif cutlass.const_expr(_job == 1):
                            r_b_raw = cutlass.Float32(0.0)
                            if in_warp_tid == 0:
                                r_b_raw = r_bet[producer_step]
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                r_conv = (r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i]
                                          * r_wq[0 * VEC_SIZE + i])
                                r_conv += (r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i]
                                           * r_wq[1 * VEC_SIZE + i])
                                r_conv += (r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i]
                                           * r_wq[2 * VEC_SIZE + i])
                                r_xk = r_xtok[producer_step * VEC_SIZE + i]
                                r_conv += r_xk * r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i]
                                r_conv = r_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True))
                                r_k[i] = r_conv
                                r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i] = (
                                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i])
                                r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i] = (
                                    r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i])
                                r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i] = r_xk

                            for i in range(VEC_SIZE):
                                k_idx = i * WARP_SIZE + in_warp_tid
                                ring_rawk[slot, i_hv, i_t, k_idx] = cutlass.BFloat16(r_k[i])
                            sum_k = 0.0
                            for i in range(VEC_SIZE):
                                sum_k += r_k[i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                            rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                            for i in range(VEC_SIZE):
                                r_k[i] = r_k[i] * rnorm_k
                            for i in range(VEC_SIZE):
                                k_idx = i * WARP_SIZE + in_warp_tid
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sK[i_t, qk_grp, qk_j // 4, qk_j % 4] = r_k[i]
                            if in_warp_tid == 0:
                                sBeta[i_t] = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-r_b_raw, fastmath=True))
                                ring_beta[slot, i_hv, i_t] = sBeta[i_t]
                        else:

                            for i in range(VEC_SIZE):
                                r_q[i] = r_xtok[producer_step * VEC_SIZE + i]
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                r_g_raw = r_q[i] + r_state[i]
                                exp_A_x = r_exp_A * r_g_raw
                                sigmoid_val = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True))
                                r_gk = lower_bound * sigmoid_val
                                qk_grp = k_idx % P2_LANES_K
                                qk_j = k_idx // P2_LANES_K
                                sG[i_t, qk_grp, qk_j // 4, qk_j % 4] = cute.math.exp(r_gk, fastmath=True)
                                ring_g[slot, i_hv, i_t, k_idx] = r_gk
                        if cutlass.const_expr(_job == 0):
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                for w in range(KERNEL_WIDTH - 1):
                                    intermediate_conv_q[scratch_row, i_t, head_off + k_idx, w] = cutlass.BFloat16(
                                        r_state[w * VEC_SIZE + i])
                        elif cutlass.const_expr(_job == 1):
                            for i in range(VEC_SIZE):
                                k_idx = i * 32 + in_warp_tid
                                for w in range(KERNEL_WIDTH - 1):
                                    intermediate_conv_k[scratch_row, i_t, head_off + k_idx, w] = cutlass.BFloat16(
                                        r_state[(KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i])


                        if cutlass.const_expr(_job < 2 and producer_step + 1 < N_STEPS):
                            for _a in range(P1_JOB_WARPS - 1):
                                for i in range(VEC_SIZE):
                                    _xn = r_xadv[(producer_step * (P1_JOB_WARPS - 1) + _a) * VEC_SIZE + i]
                                    r_state[_o + 0 * VEC_SIZE + i] = r_state[_o + 1 * VEC_SIZE + i]
                                    r_state[_o + 1 * VEC_SIZE + i] = r_state[_o + 2 * VEC_SIZE + i]
                                    r_state[_o + 2 * VEC_SIZE + i] = _xn
    else:

        _vin = []
        for _c in cutlass.range_constexpr(V_CH_PER_THREAD):
            _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (HEAD_DIM // V_CH_PER_THREAD)
            _csv0 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 0])
            _csv1 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 1])
            _csv2 = cutlass.Float32(cs_v[slot, head_off + _v_idx, 2])
            cute.autovec_copy(w_v[(head_off + _v_idx, None)], r_w4)
            _wv = [r_w4[w] for w in range(KERNEL_WIDTH)]
            _onw = cutlass.Float32(onorm_weight[_v_idx])

            _win = [_csv0, _csv1, _csv2]
            for _t in cutlass.range_constexpr(T_LOOP):
                _win.append(cutlass.Float32(x_v[0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx]))
            _gates = []
            for _t in cutlass.range_constexpr(T_LOOP):
                _gates.append(_ld_p1(P1_NC, onorm_g, (0, bos + cutlass.min(_t, n_tok - 1), i_hv, _v_idx)))
            _vin.append((_v_idx, _wv, _onw, _win, _gates))
        for _c in cutlass.range_constexpr(V_CH_PER_THREAD):
            _v_idx, _wv, _onw, _win, _gates = _vin[_c]


            sOnormWeight[_v_idx] = _onw
            for _t in cutlass.range_constexpr(T_LOOP):
                _vconv = _win[_t] * _wv[0]
                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                    _vconv += _win[_t + _w] * _wv[_w]
                _vconv = _vconv * cute.arch.rcp_approx(cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True))
                sVall[_t * HEAD_DIM + _v_idx] = _vconv
                gate_raw = _gates[_t]
                sOnormGate[_t * HEAD_DIM + _v_idx] = cute.arch.rcp_approx(
                    cutlass.Float32(1.0) + cute.math.exp(-gate_raw, fastmath=True))
                ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                    intermediate_conv_v[scratch_row, _t, head_off + _v_idx, _w] = cutlass.BFloat16(_win[_t + 1 + _w])



    swz = (in_warp_tid // 2) % 4
    r_s4 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_first = cute.make_rmem_tensor(cute.make_layout((CHUNK,), stride=(1,)), cutlass.Float32)
    if cutlass.const_expr(LOADS > STAGES):



        gStateV = cute.make_tensor(h0[(cutlass.Int64(ssm_state_indices[i_n]), i_hv, None, None)].iterator,
                                   cute.make_layout((4, 4, TILE_K // CHUNK, HEAD_DIM), stride=(1, 4, CHUNK, TILE_K)))
        cute.arch.cp_async_wait_group(STAGES - 1)
        cute.arch.sync_warp()
        for c in cutlass.range_constexpr(4):
            cute.autovec_copy(sStageV[(None, swz ^ c, in_warp_tid, warp_idx * STAGES)], r_s4)
            for e in cutlass.range_constexpr(4):
                r_first[4 * c + e] = r_s4[e]
        cute.arch.sync_warp()
        _issue_stage(state_atom, gStateV, sStageV, in_warp_tid, row_block, STAGES * KSPLIT + kq, warp_idx * STAGES)
    cute.arch.barrier()

    flat = cute.make_layout((T_LOOP * TILE_K,))
    sQf = cute.make_tensor(sQ.iterator, flat)
    sKf = cute.make_tensor(sK.iterator, flat)
    sGf = cute.make_tensor(sG.iterator, flat)

    if tidx < HEAD_DIM:
        r_cp = cutlass.Float32(1.0)
        for t in cutlass.range_constexpr(T_LOOP):
            if cutlass.const_expr(t == 0):
                r_cp = sGf[tidx]
            else:
                r_cp = r_cp * sGf[t * TILE_K + tidx]
            sQp[t * TILE_K + tidx] = r_cp * sQf[t * TILE_K + tidx]
            sKp[t * TILE_K + tidx] = r_cp * sKf[t * TILE_K + tidx]



    lanes_of = cute.make_layout((4, WARP_SIZE, T_LOOP), stride=(1, 4, TILE_K))
    sQv = cute.make_tensor(sQ.iterator, lanes_of)
    sKv = cute.make_tensor(sK.iterator, lanes_of)
    sGv = cute.make_tensor(sG.iterator, lanes_of)
    r_a = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_b = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_p = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_ks = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_cf = cute.make_rmem_tensor(cute.make_layout((T_LOOP * T_LOOP,), stride=(1,)), cutlass.Float32)
    coef_plan = _coef_plan(T_LOOP, NUM_WARPS)
    for c_w in cutlass.range_constexpr(NUM_WARPS):
        if cutlass.const_expr(len(coef_plan[c_w]) > 0):
            if warp_idx == c_w:
                c_n = 0
                c_dst = []
                for c_g in cutlass.range_constexpr(len(coef_plan[c_w])):
                    c_t, c_with, c_run = coef_plan[c_w][c_g]
                    cute.autovec_copy(sKv[(None, in_warp_tid, c_t)], r_a)
                    cute.autovec_copy(sQv[(None, in_warp_tid, c_t)], r_b)
                    cute.autovec_copy(sGv[(None, in_warp_tid, c_t)], r_p)
                    if cutlass.const_expr(c_with):
                        r_cf[c_n] = r_a[0] * r_b[0] + r_a[1] * r_b[1] + r_a[2] * r_b[2] + r_a[3] * r_b[3]
                        c_dst.append((_ic(c_t, T_LOOP), c_n))
                        c_n += 1
                    if cutlass.const_expr(len(c_run) > 0):
                        for c_s in cutlass.range_constexpr(c_t - 1, c_run[0], -1):
                            cute.autovec_copy(sGv[(None, in_warp_tid, c_s)], r_ks)
                            for j in cutlass.range_constexpr(4):
                                r_p[j] = r_p[j] * r_ks[j]
                    for c_r in cutlass.range_constexpr(len(c_run)):
                        c_s = c_run[c_r]
                        cute.autovec_copy(sKv[(None, in_warp_tid, c_s)], r_ks)
                        acc_a = cutlass.Float32(0.0)
                        acc_b = cutlass.Float32(0.0)
                        for j in cutlass.range_constexpr(4):
                            term = r_ks[j] * r_p[j]
                            acc_a = acc_a + term * r_a[j]
                            acc_b = acc_b + term * r_b[j]
                        r_cf[c_n] = acc_a
                        r_cf[c_n + 1] = acc_b
                        c_dst.append((_ia(c_t, c_s), c_n))
                        c_dst.append((_ib(c_t, c_s, T_LOOP), c_n + 1))
                        c_n += 2
                        if cutlass.const_expr(c_s > 0 and c_r + 1 < len(c_run)):
                            cute.autovec_copy(sGv[(None, in_warp_tid, c_s)], r_ks)
                            for j in cutlass.range_constexpr(4):
                                r_p[j] = r_p[j] * r_ks[j]
                for i in cutlass.range_constexpr(c_n):
                    for offset in [16, 8, 4, 2, 1]:
                        r_cf[i] = r_cf[i] + cute.arch.shuffle_sync_bfly(r_cf[i], offset=offset, mask=-1,
                                                                        mask_and_clamp=31)
                if in_warp_tid == 0:
                    for c_k in cutlass.range_constexpr(len(c_dst)):
                        sCoef[c_dst[c_k][0]] = r_cf[c_dst[c_k][1]]
    cute.arch.barrier()

    r_m = cute.make_rmem_tensor(cute.make_layout((T_LOOP,), stride=(1,)), cutlass.Float32)
    for t in cutlass.range_constexpr(T_LOOP):
        acc_m = cutlass.Float32(cutlass.select_(in_warp_tid == t, cutlass.Float32(1.0), cutlass.Float32(0.0)))
        for s in cutlass.range_constexpr(t):
            acc_m = acc_m - sCoef[_ia(t, s)] * r_m[s]
        r_m[t] = acc_m * sBeta[t]
    if in_warp_tid < P2_LANES_K:
        for t in cutlass.range_constexpr(T_LOOP):
            acc_g = sCoef[_ic(t, T_LOOP)] * r_m[t]
            for s in cutlass.range_constexpr(t):
                acc_g = acc_g + sCoef[_ib(t, s, T_LOOP)] * r_m[s]
            sGw[(warp_idx * P2_LANES_K + t) * P2_LANES_K + in_warp_tid] = acc_g
    cute.arch.sync_warp()
    if tidx < HEAD_DIM:
        for t in cutlass.range_constexpr(T_LOOP):
            acc_z = sQp[t * TILE_K + tidx]
            for s in cutlass.range_constexpr(t + 1):
                acc_z = acc_z - sGw[(warp_idx * P2_LANES_K + t) * P2_LANES_K + s] * sKp[s * TILE_K + tidx]
            sZ[t * TILE_K + tidx] = acc_z
    cute.arch.barrier()



    sZb = cute.make_tensor(sZ.iterator, cute.make_layout((4, TILE_K // 4, T_LOOP), stride=(1, 4, TILE_K)))
    r_z4 = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), cutlass.Float32)
    r_acc = cute.make_rmem_tensor(cute.make_layout((2 * T_LOOP,), stride=(1,)), cutlass.Float32)
    for s_l in cutlass.range_constexpr(LOADS):
        if cutlass.const_expr(s_l == 0 and LOADS > STAGES):
            for j in cutlass.range_constexpr(CHUNK):
                r_state[j] = r_first[j]
        else:


            cute.arch.cp_async_wait_group(min(STAGES + s_l, LOADS) - 1 - s_l)
            cute.arch.sync_warp()
            for c in cutlass.range_constexpr(4):
                cute.autovec_copy(sStageV[(None, swz ^ c, in_warp_tid, warp_idx * STAGES + s_l % STAGES)], r_s4)
                for e in cutlass.range_constexpr(4):
                    r_state[4 * c + e] = r_s4[e]
        if cutlass.const_expr(s_l > 0 and s_l + STAGES < LOADS):
            cute.arch.sync_warp()
            _issue_stage(state_atom, gStateV, sStageV, in_warp_tid, row_block, (s_l + STAGES) * KSPLIT + kq,
                         warp_idx * STAGES + s_l % STAGES)
        for c in cutlass.range_constexpr(4):
            for t in cutlass.range_constexpr(T_LOOP):
                cute.autovec_copy(sZb[(None, (s_l * KSPLIT + kq) * 4 + c, t)], r_z4)
                if cutlass.const_expr(s_l == 0 and c == 0):
                    dot_e, dot_o = cute.arch.mul_packed_f32x2((r_state[0], r_state[1]), (r_z4[0], r_z4[1]))
                else:
                    dot_e, dot_o = cute.arch.fma_packed_f32x2(
                        src_a=(r_state[4 * c], r_state[4 * c + 1]), src_b=(r_z4[0], r_z4[1]),
                        src_c=(r_acc[2 * t], r_acc[2 * t + 1]))
                dot_e, dot_o = cute.arch.fma_packed_f32x2(
                    src_a=(r_state[4 * c + 2], r_state[4 * c + 3]), src_b=(r_z4[2], r_z4[3]), src_c=(dot_e, dot_o))
                r_acc[2 * t] = dot_e
                r_acc[2 * t + 1] = dot_o
                if cutlass.const_expr(BLOCK_THREADS == mtp.BLOCK_THREADS_NARROW and t % 2 == 1):


                    cute.arch.sync_warp()


    row = row_block * WARP_SIZE + in_warp_tid
    for t in cutlass.range_constexpr(T_LOOP):
        r_acc[2 * t] = r_acc[2 * t] + r_acc[2 * t + 1]
    for p in cutlass.range_constexpr(1, KSPLIT):
        if kq == p:
            for t in cutlass.range_constexpr(T_LOOP):
                sPart[((p - 1) * T_LOOP + t) * HEAD_DIM + row] = r_acc[2 * t]
    cute.arch.barrier()
    cute.arch.griddepcontrol_launch_dependents()
    if kq == 0:
        for t in cutlass.range_constexpr(T_LOOP):
            acc_o = r_acc[2 * t]
            for p in cutlass.range_constexpr(1, KSPLIT):
                acc_o = acc_o + sPart[((p - 1) * T_LOOP + t) * HEAD_DIM + row]
            for s in cutlass.range_constexpr(t + 1):
                acc_o = acc_o + sGw[(warp_idx * P2_LANES_K + t) * P2_LANES_K + s] * sVall[s * HEAD_DIM + row]
            sOall[t * HEAD_DIM + row] = acc_o


    cute.arch.barrier()
    for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
        sumsq = cutlass.Float32(0.0)
        for i in range(VEC_SIZE):
            _o = sOall[i_t * HEAD_DIM + i * 32 + in_warp_tid]
            sumsq += _o * _o
        for offset in [16, 8, 4, 2, 1]:
            sumsq += cute.arch.shuffle_sync_bfly(sumsq, offset=offset, mask=-1, mask_and_clamp=31)
        rms = cute.math.rsqrt(sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True)
        for i in range(VEC_SIZE):
            v_idx = i * 32 + in_warp_tid
            raw_o = sOall[i_t * HEAD_DIM + v_idx]
            gate = sOnormGate[i_t * HEAD_DIM + v_idx]
            weight = sOnormWeight[v_idx]
            if i_t < n_tok:
                o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(raw_o * rms * weight * gate)



def _smem_bytes(t_loop: int, block_threads: int) -> int:
    
    warps = block_threads // WARP_SIZE
    sizes = [t_loop * TILE_K * 4] * 3 + [t_loop * 4, t_loop * HEAD_DIM * 4]
    sizes += [t_loop * HEAD_DIM * 4, t_loop * HEAD_DIM * 4, HEAD_DIM * 4]
    sizes += [t_loop * TILE_K * 4] * 2 + [t_loop * t_loop * 4]
    sizes += [warps * P2_LANES_K * P2_LANES_K * 4, t_loop * TILE_K * 4]
    sizes += [(warps // (HEAD_DIM // WARP_SIZE) - 1) * t_loop * HEAD_DIM * 4]
    return warps * STAGES * STAGE_ELEMS * 4 + sum((n + 15) // 16 * 16 for n in sizes) + 256


@cute.jit
def _launch(
    h0: cute.Tensor, x_q: cute.Tensor, x_k: cute.Tensor, x_v: cute.Tensor, w_q: cute.Tensor, w_k: cute.Tensor,
    w_v: cute.Tensor, cs_q: cute.Tensor, cs_k: cute.Tensor, cs_v: cute.Tensor, A_log: cute.Tensor, g: cute.Tensor,
    dt_bias: cute.Tensor, beta: cute.Tensor, out: cute.Tensor, intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor, intermediate_conv_k: cute.Tensor, intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor, cu_seqlens: cute.Tensor, ring_rawv: cute.Tensor, ring_rawk: cute.Tensor,
    ring_g: cute.Tensor, ring_beta: cute.Tensor, onorm_g: cute.Tensor, onorm_weight: cute.Tensor,
    n_req: cutlass.Int32, scale: cutlass.Constexpr[float], H: cutlass.Constexpr[int], NUM_SPEC: cutlass.Constexpr[int],
    BLOCK_THREADS: cutlass.Constexpr[int], lower_bound: cutlass.Constexpr[float], onorm_eps: cutlass.Constexpr[float],
    stream: cuda.CUstream,
):
    
    p2_vec = TILE_K // P2_LANES_K
    qk_stride, _ = mtp._qk_smem(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    smem_qk_layout = cute.make_layout((1 + NUM_SPEC, P2_LANES_K, p2_vec // 4, 4), stride=qk_stride)
    kda_v2_verify_kernel(
        h0, x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v, A_log, g, dt_bias, beta, out,
        intermediate_state_indices, intermediate_conv_q, intermediate_conv_k, intermediate_conv_v,
        ring_rawv, ring_rawk, ring_g, ring_beta, onorm_g, onorm_weight, smem_qk_layout,
        ssm_state_indices, cu_seqlens, scale, NUM_SPEC, BLOCK_THREADS, lower_bound, onorm_eps,
    ).launch(
        grid=(H, n_req, 1),
        block=[BLOCK_THREADS, 1, 1],
        smem=_smem_bytes(1 + NUM_SPEC, BLOCK_THREADS),
        stream=stream,
        use_pdl=True,



        min_blocks_per_mp=3 if BLOCK_THREADS == mtp.BLOCK_THREADS_NARROW else 0,
        preferred_smem_carveout=100 if BLOCK_THREADS == mtp.BLOCK_THREADS_NARROW else None,
    )


_COMPILED = get_jit_cache("glmf_kda_verify_rows", source_paths=(__file__,), enable_tvm_ffi=False)



DYNAMIC = frozenset((1, 2, 3, 11, 13, 14, 15, 19, 20, 25))


def _leading(tensor) -> int | None:
    return next((dim for dim, stride in enumerate(tensor.stride()) if stride == 1), None)


def fused_kda_v2_verify(*, x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v, g, beta, A_log, dt_bias, recurrent_state,
                        intermediate_state_indices, intermediate_conv_q, intermediate_conv_k, intermediate_conv_v,
                        ssm_state_indices, cu_seqlens, lower_bound, scale, replayssm_rawv, replayssm_rawk,
                        replayssm_g, replayssm_beta, onorm_gate, onorm_weight, onorm_eps):
    
    H, N, T = x_q.shape[2], cu_seqlens.numel() - 1, x_q.shape[1]
    expected_shape = (1, T, H, TILE_K)
    if tuple(x_q.shape) != expected_shape or tuple(x_k.shape) != expected_shape:
        raise ValueError(f"expected q/k shape {expected_shape}")
    if tuple(x_v.shape) != expected_shape or tuple(g.shape) != expected_shape:
        raise ValueError(f"expected v/g shape {expected_shape}")
    if tuple(beta.shape) != (1, T, H):
        raise ValueError(f"expected beta shape {(1, T, H)}")
    if N <= 0 or T % N != 0 or T // N < 2 or (T // N) % 2 or T // N > P2_LANES_K:
        raise ValueError(f"the v2 verify needs an even 1 + num_spec in 2..{P2_LANES_K} dense tokens a request (K3's "
                         f"form, the route's six; the head's G solve takes a lane a column); got T={T}, N={N}")
    num_spec = T // N - 1
    if (recurrent_state.dtype != torch.float32 or tuple(recurrent_state.shape[1:]) != (H, TILE_K, TILE_K)
            or tuple(recurrent_state.stride()[-3:]) != (TILE_K * TILE_K, TILE_K, 1)
            or recurrent_state.stride(0) % 4 != 0 or recurrent_state.storage_offset() % 4 != 0):
        raise ValueError("the v2 verify needs the fp32 contiguous [slots, H, V=128, K=128] state with 16-byte-aligned "
                         "slot offsets")
    ring_len = replayssm_rawv.shape[2]
    if (ring_len < 1 + num_spec or replayssm_rawv.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_rawk.shape[1:] != (H, ring_len, TILE_K) or replayssm_g.shape[1:] != (H, ring_len, TILE_K)
            or replayssm_beta.shape[1:] != (H, ring_len)):
        raise ValueError(f"expected ReplaySSM ring layouts [slots, H={H}, L>={1 + num_spec}, {TILE_K}] / [slots, H, L]")
    if (replayssm_rawv.dtype != torch.bfloat16 or replayssm_rawk.dtype != torch.bfloat16
            or replayssm_g.dtype != torch.float32 or replayssm_beta.dtype != torch.float32):
        raise ValueError("expected ReplaySSM ring dtypes rawv/rawk=bf16, g/beta=fp32")
    if tuple(onorm_gate.shape) != expected_shape or tuple(onorm_weight.shape) != (TILE_K,):
        raise ValueError(f"expected output-norm gate {expected_shape} and weight {(TILE_K,)}")
    if onorm_gate.dtype != torch.bfloat16 or onorm_weight.dtype != torch.float32:
        raise ValueError("expected output-norm gate=bf16 and weight=fp32")
    block_threads = mtp._block_threads(H=H, N=N)
    out = torch.empty_like(x_v)
    args = (recurrent_state, x_q, x_k, x_v, w_q, w_k, w_v, cs_q, cs_k, cs_v, A_log, g, dt_bias, beta, out,
            intermediate_state_indices, intermediate_conv_q, intermediate_conv_k, intermediate_conv_v,
            ssm_state_indices, cu_seqlens, replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta, onorm_gate,
            onorm_weight)
    fits_32bit = tuple(False if i == 0 else mtp._fits_32bit_stride(tensor) for i, tensor in enumerate(args))
    key = (current_arch().cuda.capability, H, num_spec, block_threads, float(onorm_eps), float(scale),
           float(lower_bound), *((tensor.device, tensor.dtype, tensor.dim(), fits, _leading(tensor)) if i in DYNAMIC
                                 else mtp._tensor_layout_key(tensor, fits)
                                 for i, (tensor, fits) in enumerate(zip(args, fits_32bit))))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _COMPILED[key] if key in _COMPILED else None
    if compiled is None:
        compiled = cute.compile(
            _launch,
            *(mtp._cute_tensor(tensor, dynamic=i in DYNAMIC, fits_32bit_stride=fits)
              for i, (tensor, fits) in enumerate(zip(args, fits_32bit))),
            cutlass.Int32(N), scale=float(scale), H=H, NUM_SPEC=num_spec, BLOCK_THREADS=block_threads,
            lower_bound=float(lower_bound), onorm_eps=float(onorm_eps), stream=stream,
        )
        _COMPILED[key] = compiled
    compiled(*(mtp._cute_tensor(tensor, fits_32bit_stride=fits) for tensor, fits in zip(args, fits_32bit)),
             cutlass.Int32(N), stream)
    return out
