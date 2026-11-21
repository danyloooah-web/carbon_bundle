from __future__ import annotations
import contextlib
import logging
import flashinfer.fused_moe as fused_moe
import moe_entry_native
import moe_wide_native
import torch
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import core as moe_core
from flashinfer.quantization import nvfp4_quantization_utils as nvfp4_utils
from flashinfer.quantization.kernels import nvfp4_quantize as nvfp4_kernels
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as moe_runner
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models import deepseek_v2
NODE = "layer"
OP = "flashinfer::trtllm_fp4_block_scale_moe"
HIDDEN, TOP_K, EXPERTS = 4096, 8, 288
MIN_ROWS, MAX_ROWS = 17, 768
KERNEL_ROWS, CAND, CNT_WORDS = 128, 72, 17
DEFERRED_MAX = 192
WIDE_ROWS, WIDE_CLUSTERS = 768, 12
FP4_TILES = (8, 16, 32, 64, 128, 192, 256)
PINS: dict = {576: (64, 18), 624: (64, 18), 672: (64, 18), 720: (64, 19), 768: (64, 19)}
PASSED = frozenset((
    "routing_logits", "routing_bias", "hidden_states", "hidden_states_scale", "gemm1_weights", "gemm1_weights_scale",
    "gemm1_bias", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale",
    "gemm2_bias", "output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar", "per_token_scale",
    "num_experts", "top_k", "n_group", "topk_group", "intermediate_size", "local_expert_offset", "local_num_experts",
    "routed_scaling_factor", "routing_method_type", "do_finalize", "activation_type", "tune_max_num_tokens",
    "enable_pdl"))
WEIGHTS = ("gemm1_weights", "gemm1_weights_scale", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "gemm2_weights",
           "gemm2_weights_scale", "output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar")
SCALARS = ("num_experts", "n_group", "topk_group", "intermediate_size", "local_expert_offset", "local_num_experts",
           "routed_scaling_factor", "routing_method_type", "activation_type", "tune_max_num_tokens", "enable_pdl")
_LAYERS: dict = {}
_SHAPES: dict = {}
_BUFFERS: dict = {}
_WORKSPACE: list = []
_SET: list = []
_CALL: dict = {}
_SAID: set = set()
def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf MoE entry: {why}")


def _say(tag: str, detail: str) -> None:
    
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf MoE entry: %s", detail)


def prepare(layer) -> None:
    
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE):
        return
    experts = mlp.experts
    if not (isinstance(experts.quant_method, ModelOptNvFp4FusedMoEMethod) and experts.top_k == TOP_K
            and experts.hidden_size == HIDDEN):
        _say("layer", f"layer {getattr(layer, 'layer_id', '?')}'s MoE is not the NVFP4 top-{TOP_K} MoE "
                      f"({type(experts.quant_method).__name__}): stock routes it")
        return
    gate = mlp.gate
    w, bias, gs = gate.weight, mlp.topk.topk_config.correction_bias, experts.w13_input_scale_quant
    _require(isinstance(gate, deepseek_v2.MoEGate) and w.dtype == torch.bfloat16 and tuple(w.shape) == (EXPERTS, HIDDEN)
             and w.is_contiguous() and isinstance(bias, torch.Tensor) and bias.dtype == torch.float32
             and tuple(bias.shape) == (EXPERTS,) and bias.is_contiguous(),
             f"the gate is {type(gate).__name__} with a {w.dtype} {tuple(w.shape)} weight and a "
             f"{getattr(bias, 'dtype', None)} bias, not the BF16 [288, 4096] router with an FP32 bias")
    _require(isinstance(gs, torch.Tensor) and gs.dtype == torch.float32 and gs.numel() == 1 and gs.is_contiguous()
             and not experts.quant_method.quant_config.use_per_token_activation
             and experts.trtllm_padded_hidden_size is None,
             "the MoE input is not quantized as quantize_hidden_states_fp4 does with one FP32 global scale")
    sm107 = torch.cuda.get_device_capability(w.device) == (10, 7)
    _require(fp4_utils._flashinfer_fp4_quantize_backend == "cute-dsl"
             and nvfp4_utils.resolve_nvfp4_4over6(nvfp4_utils._UNSET) is None
             and not nvfp4_kernels._env_flag_enabled("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH")
             and not any(nvfp4_kernels._should_use_tma(m, HIDDEN, torch.bfloat16, is_sm107=sm107,
                                                        sf_layout=nvfp4_kernels.SF_LAYOUT_LINEAR)
                         for m in range(MIN_ROWS, MAX_ROWS + 1)),
             "FlashInfer's quantize here is not the CuTe DSL linear kernel in its fast-math, no-4over6 form the "
             "kernel transcribes")
    _require(all(callable(f) for f in (
        moe_core.allocate_trtllm_moe_canonical_routing,
        moe_core.get_trtllm_moe_sm100_module, moe_core._alloc_trtllm_moe_output, moe_core._torch_view_of_ffi_tensor,
        moe_core.MoeRunnerInputs, moe_core.TrtllmMoERoutingMetadataSlot, moe_runner.quantize_hidden_states_fp4)),
        "FlashInfer or SGLang lacks a piece of the route")
    scale = experts.moe_runner_config.routed_scaling_factor
    _require(isinstance(scale, (int, float)), f"the routed scaling factor is {scale!r}")
    cap = getattr(mlp, "_deferred_finalize_max_tokens", DEFERRED_MAX)
    _require(cap == DEFERRED_MAX, f"SGLang's deferred-finalize cap is {cap!r}, not the {DEFERRED_MAX} rows the route's "
                                  "form split is written for")
    if not _LAYERS:
        moe_entry_native.prepare()
    _LAYERS[id(layer)] = dict(w=w, bias=bias, gs=gs, scale=float(scale),
                              local=(experts.moe_ep_rank * experts.num_local_experts, experts.num_local_experts))


def _outside(args: tuple, kw: dict) -> tuple[str, str] | None:
    
    rows = kw["hidden_states"].shape[0] if isinstance(kw.get("hidden_states"), torch.Tensor) else -1


    form = PASSED if rows <= DEFERRED_MAX else PASSED | {"output"}
    if args or set(kw) != form or kw["do_finalize"] is not False:
        return "form", ("a call not of SGLang's deferred from-logits form" if rows <= DEFERRED_MAX else
                        "a call above 192 rows not of A4's unfinalized form (A4 must wrap outside this route)")
    hidden, logits = kw["hidden_states"], kw["routing_logits"]
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return f"rows {rows > MAX_ROWS}", f"{rows} rows, outside {MIN_ROWS} to {MAX_ROWS}"
    if not (kw["top_k"] == TOP_K and kw["gemm1_bias"] is None and kw["gemm2_bias"] is None
            and kw["per_token_scale"] is None and hidden.dtype == torch.uint8 and hidden.shape[-1] * 2 == HIDDEN
            and isinstance(kw["hidden_states_scale"], torch.Tensor) and isinstance(kw["enable_pdl"], bool)
            and isinstance(logits, torch.Tensor) and tuple(logits.shape) == (rows, kw["num_experts"])
            and logits.is_contiguous()):
        return "tensors", "not NVFP4 top-8 activations routed from contiguous logits without biases"
    if AutoTuner.get().is_tuning_mode:
        return "tuning", "the AutoTuner is in tuning mode"
    return None

def _key(kw: dict) -> tuple:
    tensors = [kw["hidden_states"], kw["hidden_states_scale"], kw["routing_logits"], kw["routing_bias"]]
    tensors += [kw[name] for name in WEIGHTS]
    return tuple(None if t is None else (t.dtype, tuple(t.shape)) for t in tensors) + tuple(kw[n] for n in SCALARS)


def _replaced_with_tactic(kw: dict):
    
    tuner = AutoTuner.get()
    stock_choose, seen = tuner.choose_one, []
    pin = PINS.get(kw["hidden_states"].shape[0])

    def choose_one(custom_op, runners, tuning_config, inputs, **more):
        chosen = stock_choose(custom_op, runners, tuning_config, inputs, **more)
        if custom_op == OP:
            if pin is not None:
                _require(len(runners) == 1 and not tuner.is_tuning_mode
                         and list(pin) in [list(t) for t in runners[0].get_valid_tactics(inputs, None)],
                         f"the pinned tactic {pin} is not a valid FP4 MoE tactic for "
                         f"{kw['hidden_states'].shape[0]} rows here")
                chosen = (runners[0], list(pin))
            seen.append(chosen[1])
        return chosen

    tuner.choose_one = choose_one
    try:
        result = _SET[0](**kw)
    finally:
        del tuner.choose_one
    _require(len(seen) == 1, f"stock's FP4 MoE asked the AutoTuner {len(seen)} times, not once")
    return result, seen[0]


def _inputs(shape: dict, kw: dict) -> list:
    
    record = shape["record"]
    return moe_core.MoeRunnerInputs(
        output=shape["output"], routing_logits=None, topk_ids=record.packed_router_scratch,
        expert_weights=record.expert_weights, hidden_states=kw["hidden_states"],
        hidden_states_scale=kw["hidden_states_scale"], gemm1_lora_delta=None, per_token_scale=None).to_list()


def _body_kwargs(kw: dict) -> dict:
    
    return dict({name: kw[name] for name in WEIGHTS}, routing_input_mode=moe_core.RoutingInputMode.UnpackedPrecomputed,
                num_experts=kw["num_experts"], routing_bias=None, gemm1_bias=None, gemm2_bias=None, n_group=None,
                topk_group=None, local_expert_offset=kw["local_expert_offset"], routed_scaling_factor=None,
                routing_method_type=moe_core.RoutingMethodType.TopK.value, num_fused_shared_experts=0,
                norm_topk_prob=True, routing_replay_out=None, valid_hidden_size=None, valid_intermediate_size=None,
                enable_pdl=kw["enable_pdl"], activation_type=kw["activation_type"], do_finalize=False)


def _buffer(name: str, rows: int, width: int, dtype, device) -> torch.Tensor:
    
    if name not in _BUFFERS:
        _BUFFERS[name] = torch.empty((KERNEL_ROWS if name == "logits" else WIDE_ROWS, width), dtype=dtype,
                                     device=device)
    _require(rows <= _BUFFERS[name].shape[0], f"{rows} rows over the {name} buffer's {_BUFFERS[name].shape[0]}")
    return _BUFFERS[name][:rows]


def default_tile(rows: int, top_k: int, local_experts: int) -> int:
    
    average = float(rows) * float(top_k) / local_experts
    i = next((k for k, tile in enumerate(FP4_TILES) if not tile < average), len(FP4_TILES) - 1)
    return min({FP4_TILES[i], *FP4_TILES[i + 1:i + 3], *((FP4_TILES[i - 1],) if i > 0 else ())})

def _prepare(kw: dict, tactic) -> dict:
    
    pair = None
    logits, rows = kw["routing_logits"], kw["hidden_states"].shape[0]
    if isinstance(tactic, int) and tactic == -1:
        pair = (default_tile(rows, kw["top_k"], kw["local_num_experts"]), -1)
    elif not isinstance(tactic, (int, str, bytes)) and hasattr(tactic, "__getitem__") and hasattr(tactic, "__len__"):
        pair = (int(tactic[0]), int(tactic[1])) if len(tactic) == 2 else None
        pair = pair if pair is not None and pair[0] > 0 and pair[1] >= 0 else None
    _require(pair is not None,
             f"the AutoTuner's tactic for {rows} rows is {tactic!r}, neither a concrete [tile_N, config] nor -1")
    record = moe_core.allocate_trtllm_moe_canonical_routing(logits, top_k=TOP_K, tile_n=pair[0])
    per_expert, padded, e2p, p2t, histogram, batch, limit, ctas = record.scratch
    slot = moe_core.TrtllmMoERoutingMetadataSlot(pair[0], padded, e2p, p2t, record.expert_weights, histogram,
                                                 per_expert, batch, limit, ctas)
    sm100 = moe_core.get_trtllm_moe_sm100_module()
    runner = sm100.MoERunner(
        sm100.moe_op, top_k=TOP_K, num_local_experts=kw["local_num_experts"],
        dtype_act=moe_core.deduce_trtllm_gen_tensor_dtype(kw["hidden_states"], kw["hidden_states_scale"]),
        dtype_weights=moe_core.deduce_trtllm_gen_tensor_dtype(kw["gemm1_weights"], kw["gemm1_weights_scale"]),
        fp8_quantization_type=moe_core.Fp8QuantizationType.NoneFp8, hidden_size=HIDDEN, hidden_size_output=HIDDEN,
        intermediate_size=kw["intermediate_size"], activation_type=kw["activation_type"],
        weight_layout=moe_core.WeightLayout.MajorK, use_shuffled_weight=True, use_per_token_scaling=False,
        num_experts=kw["num_experts"], num_fused_shared_experts=0)
    dev = logits.device
    wide = rows > KERNEL_ROWS
    shape = dict(tactic=tactic, tile=pair[0], key=_key(kw), record=record, slot=slot, runner=runner, wide=wide,
                 output=moe_core._alloc_trtllm_moe_output(rows, HIDDEN, False, dev),
                 logits=None if wide else _buffer("logits", rows, EXPERTS, torch.float32, dev),
                 xq=_buffer("xq", rows, HIDDEN // 2, torch.uint8, dev),
                 xsf=_buffer("xsf", rows, HIDDEN // 16, torch.uint8, dev))
    if wide and "wide" not in _BUFFERS:
        _BUFFERS["wide"] = [torch.zeros(WIDE_CLUSTERS * EXPERTS, dtype=torch.int32, device=dev),
                            torch.zeros(WIDE_ROWS * TOP_K, dtype=torch.int32, device=dev),
                            torch.zeros(WIDE_ROWS * TOP_K, dtype=torch.int32, device=dev),
                            torch.zeros(1, dtype=torch.int32, device=dev)]
    if "scratch" not in _BUFFERS:
        _BUFFERS["scratch"] = [torch.zeros(KERNEL_ROWS * CAND, dtype=torch.int64, device=dev),
                               torch.zeros(KERNEL_ROWS * CAND, dtype=torch.float32, device=dev),
                               torch.zeros(KERNEL_ROWS * TOP_K, dtype=torch.int32, device=dev),
                               torch.zeros(CNT_WORDS, dtype=torch.int32, device=dev)]
    fresh = [moe_core._torch_view_of_ffi_tensor(t) for t in runner.forward(
        _inputs(shape, kw), tactic=tactic, do_preparation=True, da_routing_metadata=list(slot.tensors()),
        **_body_kwargs(kw))]
    if len(fresh) != len(_WORKSPACE) or any(f.dtype != w.dtype or f.numel() > w.numel()
                                            for f, w in zip(fresh, _WORKSPACE)):
        _WORKSPACE[:] = fresh
    shape["workspace"] = [w.view(-1)[:f.numel()].view(f.shape) if f.numel() else f for f, w in zip(fresh, _WORKSPACE)]
    return shape


def _gate(layer, stock):
    
    consts = _LAYERS[id(layer)]
    def forward(hidden_states, *args, **kwargs):
        x, rows = hidden_states, hidden_states.shape[0]
        why = None
        if not (x.dim() == 2 and x.shape[1] == HIDDEN and x.dtype == torch.bfloat16 and x.is_contiguous()
                and x.data_ptr() % 16 == 0):
            why = ("input", f"a {x.dtype} {tuple(x.shape)} gate input")
        elif not MIN_ROWS <= rows <= MAX_ROWS:
            why = (f"gate rows {rows > MAX_ROWS}", f"{rows} rows, outside {MIN_ROWS} to {MAX_ROWS}")
        elif AutoTuner.get().is_tuning_mode:
            why = ("gate tuning", "the AutoTuner is in tuning mode")
        if why is not None:
            _say(why[0], f"stock serves the whole MoE call: {why[1]}")
            return stock(hidden_states, *args, **kwargs)
        shape = _SHAPES.get(rows)
        if shape is None:
            _require(not torch.cuda.is_current_stream_capturing(), f"a captured call of {rows} rows its warm-up never "
                                                                   "made")
            _CALL["first"] = rows
            return stock(hidden_states, *args, **kwargs)
        _require("pending" not in _CALL and "served" not in _CALL, "a second gate call in one layer call")
        if shape["wide"]:
            logits = stock(hidden_states, *args, **kwargs)
            _require(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
                     and tuple(logits.shape) == (rows, EXPERTS) and logits.is_contiguous(),
                     f"stock's gate returned {getattr(logits, 'dtype', type(logits))} "
                     f"{tuple(getattr(logits, 'shape', ()))}, not contiguous FP32 [{rows}, {EXPERTS}] logits")
            moe_wide_native.route(logits, x, consts["bias"], consts["gs"], consts["scale"], shape["tile"],
                                  consts["local"][0], consts["local"][1], shape["xq"], shape["xsf"],
                                  shape["record"].tensors(), _BUFFERS["wide"], True)
        else:
            logits = shape["logits"]
            moe_entry_native.entry(x, consts["w"], consts["bias"], consts["gs"], consts["scale"], shape["tile"],
                                   consts["local"][0], consts["local"][1], None, logits, shape["xq"],
                                   shape["xsf"], shape["record"].tensors(), _BUFFERS["scratch"], True)
        _CALL["pending"] = (x, shape, consts, logits)
        return logits

    return forward

def _quantize(hidden_states, input_scale_quant):
    
    pending = _CALL.get("pending")
    if pending is None:
        return _SET[1](hidden_states, input_scale_quant)
    x, shape, consts, _ = pending
    _require(not _CALL.get("quantized"), "a second quantize of the MoE input in one layer call")
    _require(hidden_states.data_ptr() == x.data_ptr() and hidden_states.shape == x.shape
             and hidden_states.stride() == x.stride() and hidden_states.dtype == x.dtype,
             "the MoE call quantizes another tensor than the gate's input")
    _require(input_scale_quant.data_ptr() == consts["gs"].data_ptr(),
             "the MoE call quantizes with another global scale than the launch read")
    _CALL["quantized"] = True
    return shape["xq"], shape["xsf"].view(torch.float8_e4m3fn)


def _served(*args, **kw):
    
    pending = _CALL.pop("pending", None)
    if pending is None:
        why = _outside(args, kw)
        rows = kw["hidden_states"].shape[0] if "hidden_states" in kw else None
        if why is None and _CALL.get("first") == rows and rows not in _SHAPES:
            _require(not torch.cuda.is_current_stream_capturing(), f"a captured call of {rows} rows its warm-up "
                                                                   "never made")
            result, tactic = _replaced_with_tactic(kw)
            _SHAPES[rows] = _prepare(kw, tactic)
            _say(f"first {rows}", f"{rows} rows: a shape's first call is stock's (tactic {tactic!r} kept)")
            return result
        _say(why[0] if why else "gate", f"the entry this route replaced serves a call: "
                                        f"{why[1] if why else 'its gate was stock'}")
        return _SET[0](*args, **kw)
    x, shape, consts, logits = pending
    _CALL["served"] = True
    why = _outside(args, kw)
    _require(why is None, f"the MoE call the launch routed is not of the form it serves: {why[1] if why else ''}")
    _require(_CALL.get("quantized"), "the MoE call's activations are not the codes the launch wrote")
    _require(_key(kw) == shape["key"] and kw["routing_logits"].data_ptr() == logits.data_ptr()
             and kw["hidden_states"].data_ptr() == shape["xq"].data_ptr()
             and kw["hidden_states_scale"].data_ptr() == shape["xsf"].data_ptr()
             and isinstance(kw["routing_bias"], torch.Tensor)
             and kw["routing_bias"].data_ptr() == consts["bias"].data_ptr()
             and kw["routed_scaling_factor"] is not None and float(kw["routed_scaling_factor"]) == consts["scale"]
             and (kw["local_expert_offset"], kw["local_num_experts"]) == consts["local"]
             and kw["routing_method_type"] == moe_core.RoutingMethodType.DeepSeekV3.value
             and kw["n_group"] in (None, 0, 1) and kw["topk_group"] in (None, 0, 1) and kw["num_experts"] == EXPERTS,
             f"the {x.shape[0]}-row MoE call is not routed as the launch routed it (DeepSeekV3 at n_group 1 from its "
             "logits, bias, scale and experts)")
    result = shape["runner"].forward(_inputs(shape, kw), tactic=shape["tactic"], da_body_workspace=shape["workspace"],
                                     da_routing_metadata=list(shape["slot"].tensors()), **_body_kwargs(kw))
    _require(len(result) == 3, f"FlashInfer's deferred body returned {len(result)} tensors, not 3")
    _say("ours", "the gate launch routes and quantizes; FlashInfer's FP4 body runs from its record")
    return [moe_core._torch_view_of_ffi_tensor(t) for t in result]


@contextlib.contextmanager
def routed(layer, forward_batch):
    
    if id(layer) not in _LAYERS or not get_is_capture_mode():
        yield
        return
    if deepseek_v2.is_in_breakable_cuda_graph() or deepseek_v2.is_in_tc_piecewise_cuda_graph():
        _say("graph", "stock routes in breakable and piecewise graphs")
        yield
        return
    _require(not _SET, "a layer's call inside another's")
    gate = layer.mlp.gate
    _SET[:] = [fused_moe.trtllm_fp4_block_scale_moe, moe_runner.quantize_hidden_states_fp4]
    _CALL.clear()
    fused_moe.trtllm_fp4_block_scale_moe = _served
    moe_runner.quantize_hidden_states_fp4 = _quantize
    gate.forward = _gate(layer, gate.forward)
    try:
        yield
        _require("pending" not in _CALL, "the layer call returned with the gate launch's MoE call never made")
    finally:
        del gate.forward
        fused_moe.trtllm_fp4_block_scale_moe, moe_runner.quantize_hidden_states_fp4 = _SET
        _SET.clear()
        _CALL.clear()
