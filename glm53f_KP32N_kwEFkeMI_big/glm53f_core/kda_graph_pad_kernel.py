from __future__ import annotations
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_fake_stream
from flashinfer.kda_kernels import kda_chunked_bt16 as bt16
from flashinfer.kda_kernels.kda_chunked_bt16 import (
    BT,
    DK,
    DV,
    O_TMA_SWIZZLE_ELEMS,
    PACKED_METADATA_THREADS,
    RAW_F16_TMA_SWIZZLE_ELEMS,
    RAW_F32_TMA_SWIZZLE_ELEMS,
    gate_dtype_is_f32,
    packed_metadata_kernel,
)
_BODY = bt16.kernel.__wrapped__
_TVM_FFI = "--enable-tvm-ffi"
@cute.kernel
def kernel_live(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_v: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_gate: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_beta: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_o: cutlass.GridConstant[cuda.TensorMap],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    
    _, sequence_slot, _ = cute.arch.block_idx()
    sequence = cutlass.Int32(seq_order[sequence_slot])
    if cu_seqlens[sequence + 1] > cu_seqlens[sequence]:
        _BODY(
            tma_desc_q,
            tma_desc_k,
            tma_desc_v,
            tma_desc_gate,
            tma_desc_beta,
            tma_desc_o,
            q,
            k,
            v,
            raw_gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            seq_order,
            state_indices,
            initial_state,
            out,
            final_state,
            SCALE,
            state_ckpt,
            cu_ckpts,
            checkpoint_state_indices,
            checkpoint_stride_chunks,
            SAFE_GATE,
            GATE_SCALE_LOG2,
            gate_dtype,
        )


@cute.jit
def host_live(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    state_indices: cute.Tensor | None,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream,
    SCALE: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    checkpoint_stride_chunks: cutlass.Int32,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    
    packed_batch = q.shape[0]
    num_sequences = cu_seqlens.shape[0] - 1
    seqlen = q.shape[1]
    heads = q.shape[2]







    qk_batch_stride = DK
    v_batch_stride = DV
    if packed_batch != cutlass.Int32(1):
        qk_batch_stride = DK * seqlen * heads
        v_batch_stride = DV * seqlen * heads
    qk_layout = cute.make_layout(
        (DK, seqlen, heads, packed_batch),
        stride=(1, DK * heads, DK, qk_batch_stride),
    )
    v_layout = cute.make_layout(
        (DV, seqlen, heads, packed_batch),
        stride=(1, DV * heads, DV, v_batch_stride),
    )
    q_tma = cute.make_tensor(q.iterator, qk_layout)
    k_tma = cute.make_tensor(k.iterator, qk_layout)
    v_tma = cute.make_tensor(v.iterator, v_layout)
    gate_tma = cute.make_tensor(raw_gate.iterator, qk_layout)
    out_tma = cute.make_tensor(out.iterator, v_layout)
    raw_f16_tma_box = (RAW_F16_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    raw_f32_tma_box = (RAW_F32_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    o_tma_box = (O_TMA_SWIZZLE_ELEMS, BT, 1, 1)
    tma_desc_q = cuda.create_tensor_map_tiled_from_view(
        q_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_k = cuda.create_tensor_map_tiled_from_view(
        k_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_desc_v = cuda.create_tensor_map_tiled_from_view(
        v_tma,
        box_dims=raw_f16_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


    gate_tma_box = raw_f32_tma_box if gate_dtype_is_f32(gate_dtype) else raw_f16_tma_box
    tma_desc_gate = cuda.create_tensor_map_tiled_from_view(
        gate_tma,
        box_dims=gate_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    beta_heads = cutlass.Int32(beta.shape[2])
    beta_lsb = beta_heads & (-beta_heads)
    beta_g = cutlass.Int32(8) // cutlass.min(beta_lsb, cutlass.Int32(8))
    beta_is_pair = cutlass.Int32(1)
    if beta_g == cutlass.Int32(1):
        beta_is_pair = cutlass.Int32(0)
    beta_rows = (cutlass.Int32(beta.shape[1]) + beta_g - cutlass.Int32(1)) // beta_g
    beta_inner = beta_g * beta_heads
    beta_box_inner = (
        cutlass.Int32(8) * (cutlass.Int32(1) - beta_is_pair) + beta_inner * beta_is_pair
    )
    beta_box_outer = (
        cutlass.Int32(BT) * (cutlass.Int32(1) - beta_is_pair)
        + (cutlass.Int32(BT) // beta_g + cutlass.Int32(1)) * beta_is_pair
    )
    beta_tma_view = cute.make_tensor(
        beta.iterator,
        cute.make_layout((beta_inner, beta_rows), stride=(1, beta_inner)),
    )
    tma_desc_beta = cuda.create_tensor_map_tiled_from_view(
        beta_tma_view,
        box_dims=(beta_box_inner, beta_box_outer),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_desc_o = cuda.create_tensor_map_tiled_from_view(
        out_tma,
        box_dims=o_tma_box,
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    kernel_live(
        tma_desc_q,
        tma_desc_k,
        tma_desc_v,
        tma_desc_gate,
        tma_desc_beta,
        tma_desc_o,
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seq_order,
        state_indices,
        initial_state,
        out,
        final_state,
        SCALE,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
        SAFE_GATE,
        GATE_SCALE_LOG2,
        gate_dtype,
    ).launch(
        grid=(heads, num_sequences, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def unified_live(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    seq_order: cute.Tensor,
    generate_seq_order: cutlass.Int32,
    state_indices: cute.Tensor | None,
    cu_chunks: cute.Tensor,
    generate_cu_chunks: cutlass.Int32,
    ws_kd: cute.Tensor,
    ws_qd: cute.Tensor,
    ws_w: cute.Tensor,
    ws_qk: cute.Tensor,
    ws_diag: cute.Tensor,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    stream_a: cuda_driver.CUstream,
    seq_route: cutlass.Int32,
    sm_count: cutlass.Int32,
    checkpoint_stride_chunks: cutlass.Int32,
    prep_num_ctas: cutlass.Int32,
    prep_cpc: cutlass.Int32,
    scale: cutlass.Float32,
    state_ckpt: cute.Tensor | None,
    cu_ckpts: cute.Tensor | None,
    checkpoint_state_indices: cute.Tensor | None,
    SAFE_GATE: cutlass.Constexpr,
    GATE_SCALE_LOG2: cutlass.Constexpr,
    THREADS: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    gate_dtype: cutlass.Constexpr,
) -> None:
    
    z = cutlass.Int32(0)
    if (generate_seq_order != z) | (generate_cu_chunks != z):



        packed_metadata_kernel(
            cu_seqlens,
            seq_order,
            cu_chunks,
            generate_seq_order,
            generate_cu_chunks,
        ).launch(
            grid=(1, 1, 1),
            block=(PACKED_METADATA_THREADS, 1, 1),
            stream=stream_a,
        )
    host_live(
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        seq_order,
        state_indices,
        initial_state,
        out,
        final_state,
        stream_a,
        scale,
        state_ckpt,
        cu_ckpts,
        checkpoint_state_indices,
        checkpoint_stride_chunks,
        SAFE_GATE,
        GATE_SCALE_LOG2,
        THREADS,
        gate_dtype,
    )


_ENGINE = "engine"


def build(lower_bound: float, track: bool):
    

    BaseDSL._preprocess_and_replace_code(_BODY)
    fakes, ckpt, cu_ckpts, ckpt_slots = bt16._unified_fakes(cutlass.BFloat16, cutlass.Float32, True, True, track,
                                                             True, track, cutlass.BFloat16)
    unified = cute.compile(unified_live, *fakes, make_fake_stream(), 0, 0, 0, 0, 0, bt16.DEFAULT_SCALE, ckpt,
                           cu_ckpts, ckpt_slots, True, lower_bound * bt16.LOG2_E, bt16.THREADS_PER_CTA, _ENGINE,
                           cutlass.BFloat16, options=_TVM_FFI)
    spec = dict(dtype=cutlass.BFloat16, state_dtype=cutlass.Float32, gate_dtype=cutlass.BFloat16, safe_gate=True,
                gate_lower_bound=lower_bound, has_state_in=True, has_state_out=True, has_state_ckpt=track,
                has_state_indices=True, has_checkpoint_state_indices=track, mode=_ENGINE)
    return bt16._make_call(unified, spec)
