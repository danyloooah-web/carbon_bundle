from __future__ import annotations
import contextlib
import logging
import torch
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear import utils as linear_utils
NODE = "kda"
_BACKEND = kda_backend.KDAAttnBackend
_STOCK = _BACKEND._forward_target_verify
_LIVE = [False]
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf KDA verify: {why}")


def _say(tag: str, detail: str) -> None:
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).info("glmf KDA verify: %s", detail)


def prepare(node) -> None:
    
    _require(callable(_STOCK), "KDAAttnBackend lacks _forward_target_verify")
    _require(_BACKEND._forward_target_verify is _STOCK, "KDAAttnBackend._forward_target_verify is not stock's")
    _require(callable(kda_backend.select_verify_intermediate_state_indices),
             "kda_backend lacks select_verify_intermediate_state_indices")






def _forward_target_verify(
    self,
    layer: kda_backend.RadixLinearAttention,
    forward_batch: kda_backend.ForwardBatch,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    
    fm = self.forward_metadata
    seq_len = mixed_qkv.shape[0]
    query_start_loc = fm.query_start_loc
    cache_indices = fm.mamba_cache_indices
    retrieve_next_token = fm.retrieve_next_token
    retrieve_next_sibling = fm.retrieve_next_sibling
    retrieve_parent_token = fm.retrieve_parent_token
    mamba_cache_params = self.state_layer_cache(layer.layer_id)
    conv_states = mamba_cache_params.conv[0]
    ssm_states = mamba_cache_params.temporal
    intermediate_state_cache = getattr(mamba_cache_params, "intermediate_ssm", None)



    replayssm_rawv = getattr(mamba_cache_params, "replayssm_rawv", None)
    replayssm_on = replayssm_rawv is not None
    if intermediate_state_cache is None and not replayssm_on:
        raise RuntimeError(
            "KDA target_verify requires a speculative mamba cache "
            "(MambaPool.SpeculativeState); none found."
        )




    replayssm_rawk = replayssm_g = replayssm_beta = None
    ring_kwargs = {}
    if replayssm_on:
        replayssm_rawk = mamba_cache_params.replayssm_rawk
        replayssm_g = mamba_cache_params.replayssm_g
        replayssm_beta = mamba_cache_params.replayssm_beta
        ring_kwargs = dict(
            cache_ring=True,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
        )



        if self.state_pool is not None and self.state_pool.replayssm_request_ring:
            ring_indices = fm.replayssm_ring_indices
            if ring_indices is not None:
                ring_kwargs["ring_indices"] = ring_indices
    intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
    intermediate_state_indices = kda_backend.select_verify_intermediate_state_indices(
        self.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        None,
        self.req_to_token_pool.size,
    )

    draft_token_num = forward_batch.spec_info.draft_token_num
    ragged_layout = forward_batch.ragged_verify_layout
    if self._can_run_dspark_cutedsl_mtp(
        layer=layer,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        draft_token_num=draft_token_num,
        ragged_layout=ragged_layout,
        conv_states=conv_states,
        ssm_states=ssm_states,
        intermediate_state_cache=intermediate_state_cache,
        intermediate_conv_window_cache=intermediate_conv_window_cache,
        retrieve_parent_token=retrieve_parent_token,
        replayssm_rawv=replayssm_rawv,
    ):
        return self._run_dspark_cutedsl_mtp(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            conv_states=conv_states,
            ssm_states=ssm_states,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_conv_window_cache=intermediate_conv_window_cache,
            intermediate_state_indices=intermediate_state_indices,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=mamba_cache_params.replayssm_rawk,
            replayssm_g=mamba_cache_params.replayssm_g,
            replayssm_beta=mamba_cache_params.replayssm_beta,
        )
    if ragged_layout is None:
        batch_size = seq_len // draft_token_num
        conv_state_indices = cache_indices[:batch_size]




        if self._can_run_fused_chain_verify(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            draft_token_num=draft_token_num,
            conv_states=conv_states,
            ssm_states=ssm_states,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_conv_window_cache=intermediate_conv_window_cache,
            cache_indices=conv_state_indices,
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
        ):
            return self._fused_chain_verify_fn(
                mixed_qkv=mixed_qkv,
                conv_weight=layer.conv_weights,
                conv_bias=layer.bias,



                conv_state=conv_states.transpose(-1, -2),
                conv_state_indices=conv_state_indices,
                intermediate_conv_window=(
                    intermediate_conv_window_cache.transpose(-1, -2)
                ),
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                ssm_states=ssm_states,
                cache_indices=conv_state_indices,
                intermediate_states_buffer=intermediate_state_cache,
                scale=layer.head_k_dim**-0.5,
                T=draft_token_num,
                num_q_heads=layer.num_q_heads,
                num_v_heads=layer.num_v_heads,
                head_k_dim=layer.head_k_dim,
                head_v_dim=layer.head_v_dim,
                lower_bound=layer.lower_bound,
                **ring_kwargs,
            )
        dense_token_indices = None
        mixed_qkv_dense = mixed_qkv.view(batch_size, draft_token_num, -1)
    else:






        batch_size = query_start_loc.shape[0] - 1
        num_dense_tokens = batch_size * draft_token_num
        dense_token_indices = kda_backend.ragged_verify_dense_scatter_indices(
            query_start_loc=query_start_loc,
            seq_len=seq_len,
            draft_token_num=draft_token_num,
        )
        dense = mixed_qkv.new_zeros(num_dense_tokens + 1, mixed_qkv.shape[-1])
        dense.index_copy_(0, dense_token_indices, mixed_qkv)
        mixed_qkv_dense = dense[:num_dense_tokens].view(
            batch_size, draft_token_num, -1
        )



    mixed_qkv_reshaped = mixed_qkv_dense.transpose(1, 2)
    mixed_qkv_processed = kda_backend.causal_conv1d_update(
        mixed_qkv_reshaped,
        conv_states.transpose(-1, -2),
        layer.conv_weights,
        layer.bias,
        activation="silu",
        conv_state_indices=cache_indices[:batch_size],
        intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
        intermediate_state_indices=intermediate_state_indices[:batch_size],
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
    )
    mixed_qkv_flat = mixed_qkv_processed.transpose(1, 2).reshape(
        batch_size * draft_token_num, -1
    )
    if dense_token_indices is None:
        mixed_qkv = mixed_qkv_flat
    else:

        padded_flat = mixed_qkv_flat.new_zeros(
            batch_size * draft_token_num + 1, mixed_qkv_flat.shape[-1]
        )
        padded_flat[: batch_size * draft_token_num] = mixed_qkv_flat
        mixed_qkv = padded_flat[dense_token_indices]

    q, k, v = mixed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
    q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

    core_attn_out = self.kernel_dispatcher.target_verify(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        intermediate_states_buffer=intermediate_state_cache,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=draft_token_num,
        retrieve_parent_token=retrieve_parent_token,
        lower_bound=layer.lower_bound,
        **ring_kwargs,
        **(
            dict(
                zip(
                    ("fused_accept_state_indices", "fused_accept_num_accepted"),
                    self._fused_accept_indices(
                        cache_indices=cache_indices,
                        query_start_loc=query_start_loc,
                        intermediate_state_cache=intermediate_state_cache,
                        draft_token_num=draft_token_num,
                    ),
                )
            )
            if self.accept_lens_pool is not None
            else {}
        ),
    )
    if dense_token_indices is not None:



        covered = dense_token_indices < (batch_size * draft_token_num)
        core_attn_out = torch.where(covered.view(1, -1, 1, 1), core_attn_out, 0.0)
    return core_attn_out


@contextlib.contextmanager
def routed(node, forward_batch):
    
    if not forward_batch.forward_mode.is_target_verify() or linear_utils.pp_spec_stable_rows_enabled():
        yield
        return
    _require(not _LIVE[0], "already set (a block's call inside another)")
    _LIVE[0] = True
    _BACKEND._forward_target_verify = _forward_target_verify
    _say("on", "verify calls run without the unread cache_indices >= 0 compare (one launch a layer)")
    try:
        yield
    finally:
        _BACKEND._forward_target_verify = _STOCK
        _LIVE[0] = False
