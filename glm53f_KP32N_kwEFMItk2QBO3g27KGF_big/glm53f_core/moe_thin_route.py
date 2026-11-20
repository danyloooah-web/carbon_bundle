from __future__ import annotations
import contextlib
import hashlib
import os
import torch
_OP = "flashinfer::trtllm_fp4_block_scale_moe"
NODE = "layer"
_PINS = {
    ("sglang", "srt/layers/moe/moe_runner/flashinfer_trtllm.py"):
        "7afaeb2ac929c30cefb9958e940ece6ebb940a92e7e5f827380b7f0307ff5ba1",
    ("flashinfer", "fused_moe/core.py"): "8ab7daf96b8a29fd2a6f81639529f23fa5e9da7147f02f07c9eaa3a4a6b6a4c6",
    ("flashinfer", "fused_moe/shared/inputs.py"): "0077c34120818720a79d44802c7c76b63eb3b35dc4aec8d382fb3dc91eddf8b5",
    ("flashinfer", "fused_moe/__init__.py"): "1177c9cca4efcf9d110210333da140414af770cdb5dd39df62157beb742eaaa5",
}
_DEFERRED = frozenset((
    "routing_logits", "routing_bias", "hidden_states", "hidden_states_scale", "gemm1_weights", "gemm1_weights_scale",
    "gemm1_bias", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "gemm2_weights", "gemm2_weights_scale",
    "gemm2_bias", "output1_scale_scalar", "output1_scale_gate_scalar", "output2_scale_scalar", "per_token_scale",
    "num_experts", "top_k", "n_group", "topk_group", "intermediate_size", "local_expert_offset", "local_num_experts",
    "routed_scaling_factor", "routing_method_type", "do_finalize", "activation_type", "tune_max_num_tokens",
    "enable_pdl"))
_FORMS = {_DEFERRED: (False,), _DEFERRED | {"output"}: (True, False)}
_state = {"core": None, "package": None, "moe_op": None, "tactics": {}, "live": False, "tally": {}}


def _tally(route: str, describe=None) -> None:
    tally = _state["tally"]
    calls = tally[route] = tally.get(route, 0) + 1
    if calls & (calls - 1) == 0:
        print(f"[glm-moethin] {route} x{calls}" + (f": {describe()}" if describe is not None else ""), flush=True)


def _read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _roots() -> dict:
    import flashinfer
    import sglang

    return {"sglang": os.path.dirname(os.path.abspath(sglang.__file__)),
            "flashinfer": os.path.dirname(os.path.abspath(flashinfer.__file__))}


def _differences() -> list:
    roots = _roots()
    differing = []
    for (package, relative), expected in _PINS.items():
        try:
            found = hashlib.sha256(_read(os.path.join(roots[package], relative))).hexdigest()
        except OSError:
            found = None
        if found != expected:
            differing.append(f"{package}/{relative}")
    return differing


def _modules():
    import flashinfer.fused_moe as package
    import flashinfer.fused_moe.core as core

    return package, core


def prepare(model) -> None:
    differs = _differences()
    if differs:
        raise RuntimeError(f"MoE thin: SGLang / FlashInfer are not the versions this route mirrors (314bce7c / 0.7.3); "
                           f"differing: {', '.join(differs)}")
    package, core = _modules()
    if package.trtllm_fp4_block_scale_moe is not core.trtllm_fp4_block_scale_moe:
        raise RuntimeError("MoE thin: flashinfer.fused_moe.trtllm_fp4_block_scale_moe is not core's (already replaced)")
    moe_op = core.get_trtllm_moe_sm100_module().moe_op
    if not callable(getattr(moe_op, "trtllm_fp4_block_scale_moe", None)):
        raise RuntimeError("MoE thin: the loaded TRT-LLM MoE module has no trtllm_fp4_block_scale_moe")
    _state.update(core=core, package=package, moe_op=moe_op)
    _tally("prepared", lambda: f"moe op {type(moe_op).__name__}, {len(_state['tactics'])} recorded tactics")


def _capturing() -> bool:
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.cuda.is_current_stream_capturing() or bool(get_is_capture_mode())


def _thinnable(args: tuple, kwargs: dict) -> str:
    forms = None if args else _FORMS.get(frozenset(kwargs))
    if forms is None:
        return "another argument form"
    finalize = kwargs["do_finalize"]
    hs, scale = kwargs["hidden_states"], kwargs["hidden_states_scale"]
    if not (isinstance(finalize, bool) and finalize in forms and kwargs["per_token_scale"] is None
            and kwargs["gemm1_bias"] is None
            and kwargs["gemm2_bias"] is None and kwargs["routing_logits"] is not None and scale is not None
            and isinstance(kwargs["enable_pdl"], bool) and hs.dtype == torch.uint8 and hs.dim() == 2):
        return "another call form"
    if finalize:
        out = kwargs["output"]
        if not (isinstance(out, torch.Tensor) and out.dtype == torch.bfloat16 and out.device == hs.device
                and out.dim() == 2 and out.shape[0] == hs.shape[0] and out.shape[1] <= hs.shape[-1] * 2):
            return "an output stock would refuse"
    if _state["core"].AutoTuner.get().is_tuning_mode:
        return "the tuner is tuning"
    return ""


def _key(kw: dict) -> tuple:
    return (tuple(kw["hidden_states"].shape), tuple(kw["hidden_states_scale"].shape),
            str(kw["hidden_states_scale"].dtype), tuple(kw["routing_logits"].shape), str(kw["routing_logits"].dtype),
            tuple(kw["gemm1_weights"].shape), str(kw["gemm1_weights"].dtype), tuple(kw["gemm1_weights_scale"].shape),
            str(kw["gemm1_weights_scale"].dtype), tuple(kw["gemm2_weights"].shape), kw["num_experts"], kw["top_k"],
            kw["n_group"], kw["topk_group"], kw["intermediate_size"], kw["local_expert_offset"],
            kw["local_num_experts"], int(kw["routing_method_type"]), int(kw["activation_type"]),
            kw["tune_max_num_tokens"], kw["do_finalize"], tuple(kw["output"].shape) if "output" in kw else None)


def _recorded(stock, kwargs: dict):
    tuner = _state["core"].AutoTuner.get()
    seen = []

    def choose_one(custom_op, runners, tuning_config, inputs, **kw):
        runner, tactic = type(tuner).choose_one(tuner, custom_op, runners, tuning_config, inputs, **kw)
        seen.append((custom_op, tactic))
        return runner, tactic

    tuner.choose_one = choose_one
    try:
        result = stock(**kwargs)
    finally:
        del tuner.choose_one
    if len(seen) != 1 or seen[0][0] != _OP:
        raise RuntimeError(f"MoE thin: stock's launch asked the tuner {seen} (expected one {_OP} lookup)")
    return result, seen[0][1]


def _launch(kw: dict, tactic):
    core = _state["core"]
    hs = kw["hidden_states"]
    finalize = kw["do_finalize"]
    num_tokens = hs.shape[0]
    hidden_size = hs.shape[-1] * 2
    nvfp4_code = core._trtllm_fp4_nvfp4_4over6_code(core._UNSET, None, hs)
    nfse = core._validate_fused_shared_experts(None, kw["local_expert_offset"], kw["local_num_experts"],
                                               kw["num_experts"], kw["routing_method_type"])
    total = kw["top_k"] + nfse
    topk_ids = torch.empty(num_tokens, total, dtype=torch.int32, device=hs.device)
    topk_weights = torch.empty(num_tokens, total, dtype=torch.bfloat16, device=hs.device)
    output = kw["output"] if "output" in kw else core._alloc_trtllm_moe_output(
        num_tokens, core._trtllm_moe_output_width(hidden_size, None), False, hs.device)
    intermediate = _state["moe_op"].trtllm_fp4_block_scale_moe(
        core.RoutingInputMode.FromLogits, kw["routing_logits"], topk_ids, topk_weights, kw["routing_bias"], hs,
        kw["hidden_states_scale"], kw["gemm1_weights"], kw["gemm1_weights_scale"], None, None, kw["gemm1_alpha"],
        kw["gemm1_beta"], kw["gemm1_clamp_limit"], kw["gemm2_weights"], kw["gemm2_weights_scale"], None,
        kw["output1_scale_scalar"], kw["output1_scale_gate_scalar"], kw["output2_scale_scalar"], None,
        kw["num_experts"], kw["top_k"], nfse, kw["n_group"], kw["topk_group"], kw["intermediate_size"],
        kw["local_expert_offset"], kw["local_num_experts"], kw["routed_scaling_factor"], kw["routing_method_type"],
        finalize, kw["enable_pdl"], kw["activation_type"], output, [-1, -1] if tactic == -1 else tactic, True, None,
        [], [], False, None, None, None, False, nvfp4_code)
    return core._unpack_trtllm_moe_output(intermediate, output, finalize, None, topk_weights)


def _thin(stock):

    def trtllm_fp4_block_scale_moe(*args, **kwargs):
        why = _thinnable(args, kwargs)
        if why:
            _tally("stock launch", lambda: why)
            return stock(*args, **kwargs)
        key = _key(kwargs)
        tactic = _state["tactics"].get(key)
        if tactic is None:
            result, chosen = _recorded(stock, kwargs)
            _state["tactics"][key] = (list(chosen) if isinstance(chosen, list) else chosen,)
            _tally("tactic recorded", lambda: f"{key[0][0]} rows: {chosen}")
            return result
        _tally("thin launch", lambda: f"{key[0][0]} rows, tactic {tactic[0]}")
        return _launch(kwargs, tactic[0])

    return trtllm_fp4_block_scale_moe


@contextlib.contextmanager
def routed(model, forward_batch):
    if _state["moe_op"] is None or not forward_batch.forward_mode.is_extend() or _capturing():
        yield
        return
    if _state["live"]:
        raise RuntimeError("MoE thin: already routed (a nested route)")
    package, core = _state["package"], _state["core"]
    if package.trtllm_fp4_block_scale_moe is not core.trtllm_fp4_block_scale_moe:
        raise RuntimeError("MoE thin: flashinfer.fused_moe.trtllm_fp4_block_scale_moe was replaced by someone else")
    _state["live"] = True
    package.trtllm_fp4_block_scale_moe = _thin(core.trtllm_fp4_block_scale_moe)
    try:
        yield
    finally:
        package.trtllm_fp4_block_scale_moe = core.trtllm_fp4_block_scale_moe
        _state["live"] = False
