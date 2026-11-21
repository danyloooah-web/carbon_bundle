import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims
from flashinfer.kda_kernels.kda_chunked_bt16 import (
    BT, DIAG_REC_ELEMS, DK, DV, DV_HALF, K2_O_SMEM_STAGE_SIZE, K2_O_SMEM_TILE_SIZE, K2_QSTATE_STAGE_COUNT,
    K2_RAW_STAGE_COUNT, K2_TMA_MBAR_STAGE_COUNT, K2_TX_BYTES, KDA_CG1_REGS, KDA_SERVICE_REGS,
    KDA_TMEM_STATE_COL_OFFSET, O_TMA_SWIZZLE_ALIGNMENT_BYTES, O_TMA_SWIZZLE_ELEMS, QK_REC_ELEMS,
    RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES, RAW_F16_TMA_SWIZZLE_ELEMS, ROLES, TCGEN05_F16_K_ATOM, THREADS_PER_WARP,
    TILE_ELEMS, TMEM_ALLOC_COLS, V_TILE_ELEMS, advance_ring_stage, checkpoint_read_done_arrive,
    checkpoint_read_done_wait, cta_sync, epilogue_chain_wait_and_store_final_output,
    epilogue_chain_wait_and_store_full_output, final_state_stored_arrive, final_state_stored_wait,
    is_compute_group0_warp, is_compute_group1_warp, is_service_warpgroup, is_tmem_user_warp, output_consumed_wait,
    output_ready_arrive, output_ready_wait, raw_consumed_arrive, raw_consumed_wait, raw_ready_arrive,
    raw_ready_wait, state_input_ready_wait, tcgen05_chain_issue_delta_half_mma, tcgen05_chain_issue_qkv_mma,
    tcgen05_chain_load_qstate_output_tmem, tcgen05_chain_stage_vmx_input_tmem, tcgen05_commit,
    tcgen05_issue_state_k_mma, tcgen05_issue_state_projection_mma, tcgen05_qstate_acc_tmem_col_offset,
    tcgen05_rescale_state_dv2_half_regs, tcgen05_shared_input_tmem_col_offset,
    tcgen05_stage_state_input_dv2_half_tmem, tcgen05_store_final_state_tmem, tcgen05_store_initial_state_tmem,
    tcgen05_wait_acc_buffer_ready, tma_chain_stage_load_inputs, tma_transfer_wait, tmem_user_sync,
    update_ready_arrive, update_ready_wait,
)
@cute.kernel
def kernel_chain_dv2_lwait(
    tma_desc_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_diag: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_qk: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    head_base: cutlass.Int32,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    dv_half = bidy % 2
    bidy = head_base + bidy // 2
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % THREADS_PER_WARP
    sequence_start = cutlass.Int32(cu_seqlens[bidx])
    sequence_end = cutlass.Int32(cu_seqlens[bidx + 1])
    seqlen = sequence_end - sequence_start
    num_chunks = cute.ceil_div(seqlen, BT)
    chunk_base = cutlass.Int32(cu_chunks[bidx])
    input_dtype = v.element_type
    tma_mbar = cutlass.Array(
        cutlass.Int64,
        K2_TMA_MBAR_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    raw_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        K2_RAW_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    state_input_ready_l_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    state_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    u_input_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    update_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    shared_acc_ready_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_l_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    k_restore_consumed_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    qstate_acc_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    stateq_done_mbar = cutlass.Array(
        cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8
    )
    output_ready_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    output_consumed_mbar = cutlass.Array(
        cutlass.Int64,
        K2_QSTATE_STAGE_COUNT,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    final_state_stored_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    checkpoint_read_done_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
    )

    kd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    w_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    qd_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    v_raw_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * V_TILE_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=RAW_F16_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )
    diag_raw_smem = cutlass.Array(
        cutlass.Float32,
        K2_RAW_STAGE_COUNT * DIAG_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    qk_smem = cutlass.Array(
        input_dtype,
        K2_RAW_STAGE_COUNT * QK_REC_ELEMS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    o_smem = cutlass.Array(
        out.element_type,
        K2_O_SMEM_TILE_SIZE,
        space=cutlass.AddressSpace.smem,
        alignment=O_TMA_SWIZZLE_ALIGNMENT_BYTES,
    )

    if warp_idx == ROLES.tma_load:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_TMA_MBAR_STAGE_COUNT):
                prims.mbarrier_init(tma_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(K2_RAW_STAGE_COUNT):
                prims.mbarrier_init(raw_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(
                    raw_consumed_mbar.subview(stage),
                    5,
                )
    elif warp_idx == ROLES.tcgen05_mma:
        if prims.elect_sync():
            prims.mbarrier_init(state_input_ready_l_mbar, 4)
            prims.mbarrier_init(state_input_ready_mbar, 4)
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(u_input_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(shared_acc_ready_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_l_mbar.subview(stage), 1)
                prims.mbarrier_init(k_restore_consumed_mbar.subview(stage), 1)
                prims.mbarrier_init(stateq_done_mbar.subview(stage), 1)
            for stage in cutlass.range_constexpr(2):
                prims.mbarrier_init(update_ready_mbar.subview(stage), 8)
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(qstate_acc_ready_mbar.subview(stage), 1)
            prims.mbarrier_init(final_state_stored_mbar, 8)
            if cutlass.const_expr(state_ckpt is not None):
                prims.mbarrier_init(checkpoint_read_done_mbar, 4)
    elif warp_idx == ROLES.epilogue:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT):
                prims.mbarrier_init(output_ready_mbar.subview(stage), 4)
                prims.mbarrier_init(output_consumed_mbar.subview(stage), 1)
    prims.fence_mbarrier_init()
    cta_sync()

    if is_tmem_user_warp(warp_idx):
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_alloc(tmem_ptr_i32, TMEM_ALLOC_COLS, group="cta_1")
        tmem_user_sync()
        if warp_idx == ROLES.tcgen05_mma:
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        tmem_user_sync()
    cta_sync()

    if is_service_warpgroup(warp_idx):
        prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)

    if warp_idx == ROLES.tma_load:
        raw_stage = cutlass.Int32(0)
        raw_consumed_phase = cutlass.Int32(1)
        issue_mbar_slot = cutlass.Int32(0)
        wait_mbar_slot = cutlass.Int32(0)
        ready_stage = cutlass.Int32(0)
        tma_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            ws_chunk = chunk_base + chunk
            ws_row_start = ws_chunk * cutlass.Int32(BT)
            v_row_start = sequence_start + chunk * cutlass.Int32(BT)
            raw_consumed_wait(
                raw_consumed_mbar.subview(raw_stage),
                raw_consumed_phase,
            )
            tma_chain_stage_load_inputs(
                tma_desc_kd,
                tma_desc_w,
                tma_desc_qd,
                tma_desc_v,
                tma_desc_diag,
                tma_desc_qk,
                kd_smem.subview(raw_stage * TILE_ELEMS),
                w_smem.subview(raw_stage * TILE_ELEMS),
                qd_smem.subview(raw_stage * TILE_ELEMS),
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS),
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                bidy,
                dv_half,
                ws_row_start,
                ws_chunk,
                v_row_start,
                tma_mbar.subview(issue_mbar_slot),
                K2_TX_BYTES,
            )
            issue_mbar_slot, _ = advance_ring_stage(
                issue_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            if chunk >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1):
                tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
                wait_mbar_slot, wait_wrapped = advance_ring_stage(
                    wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
                )
                tma_phase = tma_phase ^ wait_wrapped
                raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
                ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)
            raw_stage, raw_wrapped = advance_ring_stage(
                raw_stage, 1, K2_RAW_STAGE_COUNT
            )
            raw_consumed_phase = raw_consumed_phase ^ raw_wrapped
        tma_full = cutlass.Int32(
            num_chunks >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
        )
        tma_tail = (
            tma_full * cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)
            + (cutlass.Int32(1) - tma_full) * num_chunks
        )
        for _tail in cutlass.range(tma_tail, unroll=1):
            tma_transfer_wait(tma_mbar.subview(wait_mbar_slot), tma_phase)
            wait_mbar_slot, wait_wrapped = advance_ring_stage(
                wait_mbar_slot, 1, K2_TMA_MBAR_STAGE_COUNT
            )
            tma_phase = tma_phase ^ wait_wrapped
            raw_ready_arrive(raw_ready_mbar.subview(ready_stage))
            ready_stage, _ = advance_ring_stage(ready_stage, 1, K2_RAW_STAGE_COUNT)

    elif warp_idx == ROLES.super_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_phase12 = cutlass.Int32(0)
        si_l_phase12 = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            acc_stage = chunk % 2
            if chunk > 0:
                prev12 = chunk - cutlass.Int32(1)
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(prev12 % K2_QSTATE_STAGE_COUNT),
                    (prev12 // K2_QSTATE_STAGE_COUNT) % 2,
                )
                raw_consumed_arrive(
                    raw_consumed_mbar.subview(prev12 % K2_RAW_STAGE_COUNT)
                )
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            si_l_phase12 = state_input_ready_wait(state_input_ready_l_mbar, si_l_phase12)
            si_phase12 = state_input_ready_wait(state_input_ready_mbar, si_phase12)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            output_ready_wait(
                output_ready_mbar.subview(qstate_stage),
                (chunk // K2_QSTATE_STAGE_COUNT + cutlass.Int32(1)) % 2,
            )
            tcgen05_issue_state_projection_mma(
                qd_smem.subview(raw_stage * TILE_ELEMS),
                tmem_raw_addr,
                stateq_done_mbar.subview(kr_stage),
                tcgen05_qstate_acc_tmem_col_offset(qstate_stage),
                input_dtype,
                0,
                DK // TCGEN05_F16_K_ATOM,
                False,
                True,
                DV_HALF,
            )
            update_ready_wait(update_ready_mbar.subview(chunk % 2), (chunk // 2) % 2)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            xpack_col12 = tcgen05_shared_input_tmem_col_offset(acc_stage)
            tcgen05_chain_issue_qkv_mma(
                qk_smem.subview(raw_stage * QK_REC_ELEMS),
                tmem_raw_addr,
                xpack_col12,
                qstate_stage,
                qstate_acc_ready_mbar.subview(qstate_stage),
                input_dtype,
                True,
            )

    elif warp_idx == ROLES.tcgen05_mma:
        tmem_raw_addr = tmem_ptr_i32.load()
        si_l_phase = cutlass.Int32(0)
        si_phase = cutlass.Int32(0)
        for chunk in cutlass.range(num_chunks, unroll=1):
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            raw_phase = (chunk // K2_RAW_STAGE_COUNT) % 2
            acc_stage = chunk % 2
            qstate_stage = chunk % K2_QSTATE_STAGE_COUNT
            kr_stage = chunk % 2
            kd_stage_smem = kd_smem.subview(raw_stage * TILE_ELEMS)
            w_stage_smem = w_smem.subview(raw_stage * TILE_ELEMS)

            raw_ready_wait(raw_ready_mbar.subview(raw_stage), raw_phase)

            si_l_phase = state_input_ready_wait(state_input_ready_l_mbar, si_l_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                0,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                False,
                False,
                DV_HALF,
            )
            si_phase = state_input_ready_wait(state_input_ready_mbar, si_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_issue_state_k_mma(
                kd_stage_smem,
                tmem_raw_addr,
                shared_acc_ready_mbar.subview(acc_stage),
                acc_stage,
                input_dtype,
                (DK // TCGEN05_F16_K_ATOM) // 2,
                DK // TCGEN05_F16_K_ATOM,
                True,
                True,
                DV_HALF,
            )

            xpack_col = tcgen05_shared_input_tmem_col_offset(acc_stage)
            update_ready_wait(update_ready_mbar.subview(chunk % 2), (chunk // 2) % 2)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem, tmem_raw_addr, xpack_col, input_dtype, 0
            )
            tcgen05_commit(k_restore_consumed_l_mbar.subview(kr_stage))
            tcgen05_chain_issue_delta_half_mma(
                w_stage_smem, tmem_raw_addr, xpack_col, input_dtype, 1
            )
            tcgen05_commit(k_restore_consumed_mbar.subview(kr_stage))

        final_state_stored_wait(final_state_stored_mbar, cutlass.Int32(0))
        tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
        prims.tcgen05_dealloc(tmem_ptr, TMEM_ALLOC_COLS, group="cta_1")

    elif warp_idx == ROLES.epilogue:
        for chunk in cutlass.range(num_chunks, unroll=1):
            if chunk > 0:
                epilogue_chain_wait_and_store_full_output(
                    tma_desc_o,
                    o_smem,
                    output_ready_mbar,
                    output_consumed_mbar,
                    sequence_start,
                    bidy,
                    dv_half,
                    chunk - cutlass.Int32(1),
                    K2_QSTATE_STAGE_COUNT,
                )
        if num_chunks > 0:
            epilogue_chain_wait_and_store_final_output(
                tma_desc_o,
                out,
                o_smem,
                output_ready_mbar,
                output_consumed_mbar,
                sequence_start,
                bidy,
                dv_half,
                seqlen,
                num_chunks - cutlass.Int32(1),
                lane,
                K2_QSTATE_STAGE_COUNT,
            )

    elif is_compute_group1_warp(warp_idx):
        prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
        tmem_raw_addr = tmem_ptr_i32.load()
        state_slot = bidx
        if cutlass.const_expr(state_indices is not None):
            state_slot = cutlass.Int32(state_indices[bidx])
        ckpt_slot = cutlass.Int32(0)
        if cutlass.const_expr(state_ckpt is not None):
            ckpt_stride = checkpoint_stride_chunks
            ckpt_next = ckpt_stride
            ckpt_slot = cutlass.Int32(cu_ckpts[bidx])
        if cutlass.const_expr(checkpoint_state_indices is not None):
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                None,
                cutlass.Int32(0),
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                lane,
                HALF=True,
            )
        else:
            tcgen05_store_initial_state_tmem(
                tmem_raw_addr,
                initial_state,
                state_ckpt,
                ckpt_slot,
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                lane,
                HALF=True,
            )
        if cutlass.const_expr(
            state_ckpt is not None and checkpoint_state_indices is None
        ):
            ckpt_slot += cutlass.Int32(1)
        if num_chunks > 0:
            diag_raw_stage = diag_raw_smem.subview(0)
            state_left = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                0,
            )
            raw_ready_wait(raw_ready_mbar.subview(0), 0)
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_left,
                state_input_ready_l_mbar,
                0,
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(shared_acc_ready_mbar.subview(0), 0)
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(0),
                warp_idx,
                vmx_lane,
                0,
                0,
                input_dtype,
            )
            post_scale_lane = cute.arch.lane_idx()
            update_ready_arrive(update_ready_mbar.subview(0))
            if cutlass.const_expr(state_ckpt is not None):
                checkpoint_due = ckpt_next == cutlass.Int32(1)
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (cutlass.Int32(BT) <= seqlen)
                else:
                    checkpoint_due = checkpoint_due & (cutlass.Int32(1) < num_chunks)
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(0), 0
                    )
                    tcgen05_wait_acc_buffer_ready(k_restore_consumed_mbar.subview(0), 0)
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_STATE_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            dv_half,
                            warp_idx,
                            post_scale_lane,
                            HALF=True,
                        )
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride
        for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
            prev = chunk - cutlass.Int32(1)
            raw_stage = chunk % K2_RAW_STAGE_COUNT
            diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
            acc_stage = chunk % 2
            prev_kr_phase = (prev // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                stateq_done_mbar.subview(prev % 2), prev_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(prev % 2), prev_kr_phase
            )
            state_right = tcgen05_stage_state_input_dv2_half_tmem(
                tmem_raw_addr,
                warp_idx,
                input_dtype,
                1,
            )
            raw_consumed_arrive(raw_consumed_mbar.subview(prev % K2_RAW_STAGE_COUNT))
            raw_ready_wait(
                raw_ready_mbar.subview(raw_stage),
                (chunk // K2_RAW_STAGE_COUNT) % 2,
            )
            tcgen05_rescale_state_dv2_half_regs(
                tmem_raw_addr,
                diag_raw_stage,
                warp_idx,
                state_right,
                state_input_ready_mbar,
                1,
            )
            tcgen05_wait_acc_buffer_ready(
                shared_acc_ready_mbar.subview(acc_stage), (chunk // 2) % 2
            )
            vmx_lane = cute.arch.lane_idx()
            tcgen05_chain_stage_vmx_input_tmem(
                tmem_raw_addr,
                v_raw_smem.subview(raw_stage * V_TILE_ELEMS),
                warp_idx,
                vmx_lane,
                acc_stage,
                acc_stage,
                input_dtype,
            )
            post_scale_lane = cute.arch.lane_idx()
            update_ready_arrive(update_ready_mbar.subview(chunk % 2))
            if cutlass.const_expr(state_ckpt is not None):
                checkpoint_due = chunk + cutlass.Int32(1) == ckpt_next
                if cutlass.const_expr(checkpoint_state_indices is not None):
                    checkpoint_due = checkpoint_due & (
                        (chunk + cutlass.Int32(1)) * cutlass.Int32(BT) <= seqlen
                    )
                else:
                    checkpoint_due = checkpoint_due & (
                        chunk + cutlass.Int32(1) < num_chunks
                    )
                if checkpoint_due:
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_l_mbar.subview(chunk % 2),
                        (chunk // 2) % 2,
                    )
                    tcgen05_wait_acc_buffer_ready(
                        k_restore_consumed_mbar.subview(chunk % 2),
                        (chunk // 2) % 2,
                    )
                    checkpoint_output_slot = ckpt_slot
                    if cutlass.const_expr(checkpoint_state_indices is not None):
                        checkpoint_output_slot = cutlass.Int32(
                            checkpoint_state_indices[ckpt_slot]
                        )
                    if (checkpoint_output_slot >= cutlass.Int32(0)) & (
                        checkpoint_output_slot < cutlass.Int32(state_ckpt.shape[0])
                    ):
                        tcgen05_store_final_state_tmem(
                            tmem_raw_addr,
                            KDA_TMEM_STATE_COL_OFFSET,
                            state_ckpt,
                            checkpoint_output_slot,
                            bidy,
                            dv_half,
                            warp_idx,
                            post_scale_lane,
                            HALF=True,
                        )
                    checkpoint_read_done_arrive(checkpoint_read_done_mbar)
                    ckpt_slot += cutlass.Int32(1)
                    ckpt_next += ckpt_stride
        if num_chunks > 0:
            last = num_chunks - cutlass.Int32(1)
            last_kr_phase = (last // 2) % 2
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_l_mbar.subview(last % 2), last_kr_phase
            )
            tcgen05_wait_acc_buffer_ready(
                k_restore_consumed_mbar.subview(last % 2), last_kr_phase
            )
        if cutlass.const_expr(final_state is not None):
            final_lane = cute.arch.lane_idx()
            tcgen05_store_final_state_tmem(
                tmem_raw_addr,
                KDA_TMEM_STATE_COL_OFFSET,
                final_state,
                state_slot,
                bidy,
                dv_half,
                warp_idx,
                final_lane,
                HALF=True,
            )
        final_state_stored_arrive(final_state_stored_mbar)
    elif is_compute_group0_warp(warp_idx):
        if warp_idx < 4:
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            for chunk in cutlass.range(num_chunks, unroll=1):
                o_stage = chunk % K2_QSTATE_STAGE_COUNT
                o_phase = (chunk // K2_QSTATE_STAGE_COUNT) % 2
                output_consumed_wait(
                    output_consumed_mbar.subview(o_stage),
                    (o_phase + 1) % 2,
                )
                tcgen05_wait_acc_buffer_ready(
                    qstate_acc_ready_mbar.subview(o_stage), o_phase
                )
                tcgen05_chain_load_qstate_output_tmem(
                    tmem_raw_addr,
                    o_smem,
                    warp_idx,
                    lane,
                    o_stage * K2_O_SMEM_STAGE_SIZE,
                    o_stage,
                    SCALE,
                    out.element_type,
                )
                output_ready_arrive(output_ready_mbar.subview(o_stage))
            final_state_stored_arrive(final_state_stored_mbar)
        else:
            prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)
            tmem_raw_addr = tmem_ptr_i32.load()
            if cutlass.const_expr(state_ckpt is not None):
                cg0_ckpt_stride = checkpoint_stride_chunks
            if num_chunks > 0:
                state_input_ready_wait(
                    state_input_ready_l_mbar,
                    cutlass.Int32(0),
                )
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                update_ready_arrive(update_ready_mbar.subview(0))
            for chunk in cutlass.range(1, num_chunks, 1, unroll=1):
                prev = chunk - cutlass.Int32(1)
                raw_stage = chunk % K2_RAW_STAGE_COUNT
                prev_kr_phase = (prev // 2) % 2
                tcgen05_wait_acc_buffer_ready(
                    stateq_done_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                tcgen05_wait_acc_buffer_ready(
                    k_restore_consumed_l_mbar.subview(prev % 2),
                    prev_kr_phase,
                )
                if cutlass.const_expr(state_ckpt is not None):
                    if chunk % cg0_ckpt_stride == 0:
                        checkpoint_read_done_wait(
                            checkpoint_read_done_mbar,
                            ((chunk // cg0_ckpt_stride) + cutlass.Int32(1)) % 2,
                        )
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                state_left = tcgen05_stage_state_input_dv2_half_tmem(
                    tmem_raw_addr,
                    warp_idx,
                    input_dtype,
                    0,
                )
                raw_ready_wait(
                    raw_ready_mbar.subview(raw_stage),
                    (chunk // K2_RAW_STAGE_COUNT) % 2,
                )
                diag_raw_stage = diag_raw_smem.subview(raw_stage * DIAG_REC_ELEMS)
                tcgen05_rescale_state_dv2_half_regs(
                    tmem_raw_addr,
                    diag_raw_stage,
                    warp_idx,
                    state_left,
                    state_input_ready_l_mbar,
                    0,
                )
                update_ready_arrive(update_ready_mbar.subview(chunk % 2))


@cute.jit
def host_chain_dv2(
    v: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_chunks: cute.Tensor,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    head_base: cutlass.Int32,
    launch_heads: cutlass.Int32,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    THREADS: cutlass.Constexpr,
) -> None:

    num_sequences = cu_seqlens.shape[0] - 1
    seqlen = v.shape[1]
    heads = v.shape[2]
    ws_rows = ws_kd.shape[2]
    num_chunks_total = ws_qk.shape[2]
    tile_head_stride = DK
    diag_head_stride = DIAG_REC_ELEMS
    qk_head_stride = QK_REC_ELEMS
    if heads != cutlass.Int32(1):
        tile_head_stride = DK * ws_rows
        diag_head_stride = DIAG_REC_ELEMS * num_chunks_total
        qk_head_stride = QK_REC_ELEMS * num_chunks_total
    tile_layout = cute.make_layout(
        (DK, ws_rows, heads, 1),
        stride=(1, DK, tile_head_stride, DK),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, 1),
        stride=(1, DV * heads, DV, DV * seqlen * heads),
    )
    diag_layout = cute.make_layout(
        (DIAG_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            DIAG_REC_ELEMS,
            diag_head_stride,
            DIAG_REC_ELEMS,
        ),
    )
    qk_layout = cute.make_layout(
        (QK_REC_ELEMS, num_chunks_total, heads, 1),
        stride=(
            1,
            QK_REC_ELEMS,
            qk_head_stride,
            QK_REC_ELEMS,
        ),
    )
    f16_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_kd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_kd.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_w = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_w.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_qd = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qd.iterator, tile_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, v_layout),
        box_dims=f16_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_diag = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_diag.iterator, diag_layout),
        box_dims=(DIAG_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_qk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(ws_qk.iterator, qk_layout),
        box_dims=(QK_REC_ELEMS, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(out.iterator, v_layout),
        box_dims=(O_TMA_SWIZZLE_ELEMS, BT, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel_chain_dv2_lwait(
        tma_desc_kd,
        tma_desc_w,
        tma_desc_qd,
        tma_desc_v,
        tma_desc_diag,
        tma_desc_qk,
        tma_desc_o,
        v,
        cu_seqlens,
        cu_chunks,
        state_indices,
        initial_state,
        out,
        final_state,
        head_base,
        SCALE,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
    ).launch(
        grid=(num_sequences, launch_heads * 2, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        preferred_smem_carveout=100,
    )
