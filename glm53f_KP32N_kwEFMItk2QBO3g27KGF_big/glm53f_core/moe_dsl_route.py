from __future__ import annotations
import contextlib
import math
import flashinfer.fused_moe as fused_moe
import torch
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_disagg
from glm53f_core import moe_shared_overlap_route
NODE = "layer"
HIDDEN, TOP_K = 4096, 8
LARGE = ((128, 256), (1, 1), (128, 192), (1, 2), 128)
SMALL = ((128, 128), (1, 1), (128, 128), (1, 1), 128)
SPLIT = 5120
CHECKS = ((1, 1.0), (19, 1.0), (2048, 1.0), (8192, 1.0), (16384, 2.0))
KEYS = frozenset((
    "routing_logits", "routing_bias", "hidden_states", "hidden_states_scale", "gemm1_weights", "gemm1_weights_scale",
    "gemm1_bias", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale",
    "gemm2_bias", "output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar", "per_token_scale",
    "num_experts", "top_k", "n_group", "topk_group", "intermediate_size", "local_expert_offset", "local_num_experts",
    "routed_scaling_factor", "routing_method_type", "do_finalize", "activation_type", "tune_max_num_tokens",
    "enable_pdl", "output"))
_ENTRIES: dict = {}
_PLANS: dict = {}
_OPS: dict = {}
_SET: list = []
_CALL: dict = {}
_TALLY: dict = {}
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf MoE DSL: {why}")
def _tally(route: str, describe=None) -> None:
    
    calls = _TALLY[route] = _TALLY.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glmf-moedsl] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)


def _ops() -> dict:
    
    if not _OPS:
        from flashinfer import nvfp4_block_scale_interleave
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices, get_w2_permute_indices_with_cache
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4)
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            blockscaled_contiguous_grouped_gemm_finalize_fusion)
        from flashinfer.fused_moe.cute_dsl.moe_utils import moe_output_memset_inplace, moe_sort
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization.utils import swizzle_blockscale

        _OPS.update(weave=nvfp4_block_scale_interleave, mma=convert_sf_to_mma_layout,
                    p13=_maybe_get_cached_w3_w1_permute_indices, p2=get_w2_permute_indices_with_cache,
                    gemm1=blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
                    gemm2=blockscaled_contiguous_grouped_gemm_finalize_fusion, memset=moe_output_memset_inplace,
                    sort=moe_sort, gate=moe_fused_gate, standard=StandardTopKOutput, swizzle=swizzle_blockscale)
    return _OPS


def _inverse(perm: torch.Tensor, device) -> torch.Tensor:
    perm = perm.to(device=device, dtype=torch.int64).reshape(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)
    return inv
def _halves(rows: int, device) -> torch.Tensor:
    
    half = rows // 2
    base = torch.arange(half, device=device).view(-1, 64)
    return torch.stack((base, base + half), dim=1).reshape(-1)


def _unweave(ops: dict, rows: int, cols: int, device) -> torch.Tensor:
    
    _require(rows * cols < 1 << 24, f"a {rows}x{cols} scale matrix is too large to trace its interleave")
    idx = torch.arange(rows * cols, device=device, dtype=torch.int64)
    src = torch.zeros_like(idx)
    for byte in range(3):
        plane = (idx >> 8 * byte & 255).to(torch.uint8).view(rows, cols)
        src |= ops["weave"](plane).reshape(-1).to(torch.int64) << 8 * byte
    _require(torch.equal(torch.sort(src).values, idx),
             f"the block-scale interleave of a {rows}x{cols} matrix is not a permutation")
    return _inverse(src, device)


def _plan(ops: dict, w13, s13, w2, s2) -> dict:
    
    count, rows, packed = w13.shape
    hidden, inter = w2.shape[1], w2.shape[2] * 2
    device = w13.device
    key = (count, rows, packed, hidden, inter, str(device))
    plan = _PLANS.get(key)
    if plan is None:
        cache = {}
        p13 = ops["p13"](cache, w13[0], 128, is_gated_act_gemm=True)
        q13 = ops["p13"](cache, s13[0].view(torch.uint8), 128, num_elts_per_sf=16, is_gated_act_gemm=True)
        p2 = ops["p2"](cache, w2[0], 128)
        q2 = ops["p2"](cache, s2[0].view(torch.uint8), 128, num_elts_per_sf=16)
        order = _halves(rows, device)
        plan = _PLANS[key] = dict(w13=_inverse(p13, device)[order], w2=_inverse(p2, device),
                                  s13=(_unweave(ops, rows, hidden // 16, device), _inverse(q13, device)[order]),
                                  s2=(_unweave(ops, hidden, inter // 16, device), _inverse(q2, device)))
    return plan


def _scales(shuffled: torch.Tensor, index: tuple) -> torch.Tensor:
    
    count, rows, cols = shuffled.shape
    flat = shuffled.view(torch.uint8).reshape(count, rows * cols)
    return flat.index_select(1, index[0]).view(count, rows, cols).index_select(1, index[1]).view(torch.float8_e4m3fn)

def _layout(experts, plan: dict) -> dict:
    
    ops = _ops()
    w13, s13 = experts.w13_weight.data, experts.w13_weight_scale.data
    w2, s2 = experts.w2_weight.data, experts.w2_weight_scale.data
    count, rows, _ = w13.shape
    hidden, inter = w2.shape[1], w2.shape[2] * 2
    return dict(
        w13=w13.index_select(1, plan["w13"]), w2=w2.index_select(1, plan["w2"]),
        s13=ops["mma"](ops["swizzle"](_scales(s13, plan["s13"])).contiguous().view(torch.uint8).reshape(-1),
                       m=rows, k=hidden, num_groups=count, sf_vec_size=16),
        s2=ops["mma"](ops["swizzle"](_scales(s2, plan["s2"])).contiguous().view(torch.uint8).reshape(-1),
                      m=hidden, k=inter, num_groups=count, sf_vec_size=16))


def _refusal(experts) -> str:
    
    method, config = experts.quant_method, experts.moe_runner_config
    backend = getattr(method, "_moe_runner_backend", None)
    if type(method).__name__ != "ModelOptNvFp4FusedMoEMethod":
        return f"the experts' quant method is {type(method).__name__}"
    if backend is None or not backend.is_flashinfer_trtllm():
        return f"the MoE runner backend is {backend}"
    if config.activation != "silu" or not config.is_gated or config.gemm1_alpha is not None:
        return f"the activation is {config.activation} (gated {config.is_gated}, gemm1_alpha {config.gemm1_alpha})"
    if int(experts.num_local_experts) != int(experts.num_experts) or int(getattr(experts, "moe_ep_size", 1)) != 1:
        return "the experts are split across expert-parallel ranks"
    if getattr(experts, "trtllm_padded_hidden_size", None) is not None or int(experts.hidden_size) != HIDDEN:
        return f"the hidden size is {experts.hidden_size} (padded {getattr(experts, 'trtllm_padded_hidden_size', None)})"
    if int(experts.top_k) != TOP_K:
        return f"the experts route top {experts.top_k}"
    if method.quant_config.use_per_token_activation:
        return "activations use per-token scales"
    return ""


def _convert(experts, resident: bool) -> dict:
    
    ops = _ops()
    w13, s13 = experts.w13_weight.data, experts.w13_weight_scale.data
    w2, s2 = experts.w2_weight.data, experts.w2_weight_scale.data
    count, rows, packed = w13.shape
    hidden, inter = w2.shape[1], w2.shape[2] * 2
    _require(w13.dtype == w2.dtype == torch.uint8 and s13.dtype == s2.dtype == torch.float8_e4m3fn
             and rows == 2 * inter and packed * 2 == hidden and inter % 128 == 0
             and tuple(s13.shape) == (count, rows, hidden // 16) and tuple(s2.shape) == (count, hidden, inter // 16)
             and int(experts.intermediate_size_per_partition) == inter,
             f"unexpected FP4 expert layout w13 {tuple(w13.shape)} {w13.dtype}, scales {tuple(s13.shape)} {s13.dtype}, "
             f"w2 {tuple(w2.shape)}, scales {tuple(s2.shape)}")
    plan = _plan(ops, w13, s13, w2, s2)
    config = experts.moe_runner_config
    gate = experts.g1_alphas.data.float().reshape(-1)
    up = getattr(experts, "g1_alphas_up", None)

    _require(up is None or torch.equal(up.data.float().reshape(-1), gate),
             "gate and up GEMM1 scales differ after the stock layout's shared rescale")
    fc2 = experts.w2_input_scale_quant.data.float().reshape(-1)
    down = experts.g2_alphas.data.float().reshape(-1)
    _require(gate.numel() == count and down.numel() == count and fc2.numel() == 1,
             f"scale vectors g1 {gate.numel()}, g2 {down.numel()}, fc2 {fc2.numel()} for {count} experts")
    limit = config.gemm1_clamp_limit or config.swiglu_limit
    clamp = getattr(experts, "gemm1_clamp_limit", None)

    _require((limit is None) == (clamp is None)
             and (clamp is None or torch.allclose(clamp.data.float().reshape(-1) * gate,
                                                  torch.full_like(gate, float(limit)), rtol=1e-4)),
             f"the stock clamp does not match a {limit} SwiGLU limit on the GEMM1 scales")
    return dict(plan=plan, experts=experts, layout=_layout(experts, plan) if resident else None, a1=gate.contiguous(),
                a2=down.contiguous(), fc2=fc2.reshape(1).contiguous(), count=count, inter=inter,
                glu={} if limit is None else {"swiglu_limit": float(limit)},
                rsf=1.0 if config.routed_scaling_factor is None else float(config.routed_scaling_factor),
                freed=False, layer="?")


def _route(entry: dict, logits, bias, groups, topk_group) -> tuple:
    
    weights, ids = _ops()["gate"](logits, bias, TOP_K, scoring_func="sigmoid", renormalize=True,
                                  routed_scaling_factor=entry["rsf"], apply_routed_scaling_factor_on_output=True,
                                  num_expert_group=groups or 1, topk_group=topk_group or 1)
    return (weights if weights.dtype == torch.float32 else weights.float(),
            ids if ids.dtype == torch.int32 else ids.to(torch.int32))


def _form(args: tuple, kw: dict, entry: dict) -> str:
    
    if args or frozenset(kw) != KEYS:
        return "a call not of SGLang's finalizing from-logits form"
    if not (kw["do_finalize"] is True and kw["per_token_scale"] is None and kw["gemm1_bias"] is None
            and kw["gemm2_bias"] is None and kw["gemm1_alpha"] is None and kw["gemm1_beta"] is None):
        return "an unfinalized, per-token-scaled or biased call"
    hs, scale, out, logits = kw["hidden_states"], kw["hidden_states_scale"], kw["output"], kw["routing_logits"]
    rows = hs.shape[0]
    if not (rows > 0 and hs.dtype == torch.uint8 and tuple(hs.shape) == (rows, HIDDEN // 2)
            and scale.dtype == torch.float8_e4m3fn and tuple(scale.shape) == (rows, HIDDEN // 16)
            and isinstance(logits, torch.Tensor) and tuple(logits.shape) == (rows, entry["count"])):
        return f"an input of {hs.dtype} {tuple(hs.shape)}, scales {tuple(scale.shape)}"
    if not (isinstance(out, torch.Tensor) and out.dtype == torch.bfloat16 and tuple(out.shape) == (rows, HIDDEN)
            and out.is_contiguous() and out.device == hs.device):
        return f"an output of {getattr(out, 'dtype', None)} {tuple(getattr(out, 'shape', ()))}"
    scaling = kw["routed_scaling_factor"]
    if not (kw["top_k"] == TOP_K and kw["num_experts"] == kw["local_num_experts"] == entry["count"]
            and kw["local_expert_offset"] == 0 and kw["intermediate_size"] == entry["inter"]
            and (1.0 if scaling is None else float(scaling)) == entry["rsf"]):
        return "another expert shape or routed scale"
    if kw["gemm1_weights"].data_ptr() != entry["experts"].w13_weight.data_ptr():
        return "another layer's experts"
    return ""


def _dsl(entry: dict, kw: dict) -> None:
    ops = _ops()
    x, x_sf, out = kw["hidden_states"], kw["hidden_states_scale"], kw["output"]
    rows = x.shape[0]
    layout = entry["layout"] if entry["layout"] is not None else _layout(entry["experts"], entry["plan"])
    weights, ids = _route(entry, kw["routing_logits"], kw["routing_bias"], kw["n_group"], kw["topk_group"])
    tile1, cluster1, tile2, cluster2, tokens = LARGE if rows >= SPLIT else SMALL
    pdl = bool(kw["enable_pdl"])
    t2e, bound, _, p2e, _, live = ops["sort"](
        token_selected_experts=ids, token_final_scales=weights, num_experts=entry["count"], top_k=TOP_K,
        local_expert_offset=0, num_local_experts=entry["count"], tile_tokens_dim=tokens)
    inter, inter_sf = ops["gemm1"](
        a=x, b=layout["w13"], a_scale=x_sf, b_scale=layout["s13"], alpha=entry["a1"], tile_idx_to_expert_idx=t2e,
        tile_idx_to_mn_limit=bound, token_id_mapping=p2e, num_non_exiting_tiles=live, out=None, out_scale=None,
        global_scale=entry["fc2"], topk=TOP_K, c_dtype="float4_e2m1fn", mma_tiler_mn=tile1, cluster_shape_mn=cluster1,
        enable_pdl=pdl, raster_along_m=rows < SPLIT, gated=True, **entry["glu"])
    if not moe_shared_overlap_route.seed(out):
        ops["memset"](out)
    ops["gemm2"](
        a=inter, b=layout["w2"], a_scale=inter_sf, b_scale=layout["s2"], alpha=entry["a2"], tile_idx_to_expert_idx=t2e,
        num_non_exiting_tiles=live, tile_idx_to_mn_limit=bound, permuted_idx_to_expanded_idx=p2e,
        token_final_scales=weights, out=out, a_dtype="float4_e2m1fn", b_dtype="float4_e2m1fn", mma_tiler_mn=tile2,
        cluster_shape_mn=cluster2, enable_pdl=pdl, use_fused_finalize=True, raster_along_m=True)
    _tally("DSL MoE", lambda: f"{rows} rows, {'large' if rows >= SPLIT else 'small'} tiles, layout "
                              f"{'resident' if entry['layout'] is not None else 'converted for this call'}")


def _moe(*args, **kw):
    
    entry = _CALL["entry"]
    why = _form(args, kw, entry)
    _require(not why, f"layer {entry['layer']}: {why} reached the FP4 MoE inside this route's domain")
    _dsl(entry, kw)
    return [kw["output"]]


def _routed_moe(*args, **kw):
    
    _require(False, f"layer {_CALL['entry']['layer']}: a materialized top-k reached the FP4 MoE inside this route's "
                    "domain")


@contextlib.contextmanager
def _serving(entry: dict):
    
    _require(not _SET, "a layer's call inside another's")
    _SET[:] = [fused_moe.trtllm_fp4_block_scale_moe, fused_moe.trtllm_fp4_block_scale_routed_moe]
    _CALL["entry"] = entry
    fused_moe.trtllm_fp4_block_scale_moe, fused_moe.trtllm_fp4_block_scale_routed_moe = _moe, _routed_moe
    try:
        yield
    finally:
        fused_moe.trtllm_fp4_block_scale_moe, fused_moe.trtllm_fp4_block_scale_routed_moe = _SET
        _SET.clear()
        _CALL.clear()


def _gap(got: torch.Tensor, want: torch.Tensor) -> tuple:
    
    size = want.norm(dim=1)
    gap = (got - want).norm(dim=1)
    return float(gap.norm() / size.norm().clamp_min(1e-12)), gap / size.clamp_min(1e-12)


def _check(mlp, entry: dict) -> str:
    
    ops = _ops()
    device = entry["a1"].device
    found = []
    for rows, sigma in CHECKS:
        gen = torch.Generator(device=device)
        gen.manual_seed(7 + rows)
        x = (torch.randn(rows, HIDDEN, generator=gen, device=device) * sigma).to(torch.bfloat16)
        logits = mlp.gate(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        topk = mlp.topk(x, logits)
        config = topk.topk_config
        chosen = ops["standard"](*_route(entry, topk.router_logits, config.correction_bias, config.num_expert_group,
                                         config.topk_group), logits)
        stock = mlp.experts(x.clone(), topk).float()
        routed_ = mlp.experts(x.clone(), chosen).float()
        with _serving(entry):
            dsl = mlp.experts(x.clone(), topk).float()
        rel, row = _gap(dsl, routed_)
        worst = float(row.max())
        moved, shift = _gap(routed_, stock)
        flips = int((shift > 0.1).sum())
        found.append(f"{rows} rows rel {rel:.4f} worst {worst:.4f}, router rel {moved:.4f} flips {flips}")
        _require(math.isfinite(rel) and math.isfinite(worst) and rel <= 0.03 and worst <= 0.15
                 and math.isfinite(moved) and flips <= max(2, rows // 100),
                 f"layer {entry['layer']}'s CuTe DSL experts differ from stock: {'; '.join(found)}")
    return "; ".join(found)


def _free(experts) -> float:
    device = experts.w13_weight.device
    before = torch.cuda.memory_allocated(device)
    for param in (experts.w13_weight, experts.w13_weight_scale, experts.w2_weight, experts.w2_weight_scale):
        param.data = torch.empty(1, dtype=param.dtype, device=device).expand(param.shape)
    torch.cuda.empty_cache()
    return (before - torch.cuda.memory_allocated(device)) / 2 ** 30


def _role() -> str:
    mode = get_disagg().disaggregation_mode
    return str(getattr(mode, "value", mode))


def _capturing() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.cuda.is_current_stream_capturing() or bool(get_is_capture_mode())


def prepare(layer) -> None:
    role = _role()
    if role == "decode":
        return
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE):
        return
    number = getattr(layer, "layer_id", "?")
    experts = mlp.experts
    why = _refusal(experts)
    _require(not why, f"layer {number}: {why}")
    resident = role == "prefill"
    entry = _convert(experts, resident)
    entry["layer"] = number
    report = _check(mlp, entry)
    freed = _free(experts) if resident else 0.0
    entry["freed"] = resident
    _ENTRIES[id(layer)] = entry
    print(f"[glmf-moedsl] layer {number} ({role}): {entry['count']} experts on CuTe DSL, limit {entry['glu']}; "
          f"{report}; freed {freed:.2f} GiB; layout {'resident' if resident else 'converted per call'}", flush=True)


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    entry = _ENTRIES.get(id(layer))
    if entry is None or not (entry["freed"] or (forward_batch.forward_mode.is_extend() and not _capturing())):
        yield
        return
    with _serving(entry):
        yield
