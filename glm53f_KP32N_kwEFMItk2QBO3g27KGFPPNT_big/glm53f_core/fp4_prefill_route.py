from __future__ import annotations
import contextlib
import logging
import statistics
import time
import torch
import torch.nn.functional as F
from sglang.kernels.ops.gemm import fp8_kernel
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MoE
from sglang.srt.runtime_context import get_disagg
from glm53f_core import fp8_linear_route
NODE = "layer"
GROUPS = ("G3",)
MIN_ROWS, MAX_ROWS = 8193, 32768
SKIP = {"G1": frozenset({3}), "G2a": frozenset({3}), "G3": frozenset({0, 1, 2}), "G4": frozenset({3})}
ACT_SCALE = {"gate_up_proj": "norm", "down_proj": "tensor", "o_proj": "tensor", "fused_qkvbfg_a_proj": "norm",
             "q_b_proj": "norm", "kv_b_proj": "norm"}
NORM_MARGIN = 2.0
BACKEND = "cutlass"
QUANT_BACKEND = "cuda"
E2M1_MAX, E4M3_MAX = 6.0, 448.0
SCALE_NUM = E2M1_MAX * E4M3_MAX
TINY = 1e-30
PROBE_ROWS, PROBE_BAR = 256, 0.25
WORKSPACE = 40 << 20
BF16 = torch.bfloat16
SHAPES = {"G1": {"gate_up_proj": (2048, 4096), "down_proj": (4096, 1024)}, "G2a": {"o_proj": (4096, 8192)},
          "G3": {"fused_qkvbfg_a_proj": (12576, 4096)}, "G4": {"q_b_proj": (8192, 1536), "kv_b_proj": (16384, 512)}}
SWEEP_ROWS = {"gate_up_proj": 16384, "down_proj": 16384, "o_proj": 8192, "fused_qkvbfg_a_proj": 16384,
              "q_b_proj": 8192, "kv_b_proj": 16384}
_STOCK_LINEAR = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
_STOCK_QUANT = fp8_utils.sglang_per_token_group_quant_fp8
_STOCK_DG = fp8_kernel.deep_gemm_fp8_fp8_bf16_nt
_BY_FP8: dict = {}
_LAYERS: dict = {}
_TACTIC: dict = {}
_FI: dict = {}
_CALL: dict = {}
_SAID: set = set()
_TALLY: dict = {}
class _Copy:
    __slots__ = ("group", "name", "linear", "fp8", "weight_t", "scale_t", "inv_gs", "n", "k", "gs", "alpha", "coef")
    def __init__(self, group: str, name: str, linear, fp8, weight, scale, inv_gs: float, act_gs):
        self.group, self.name, self.linear, self.fp8 = group, name, linear, fp8
        self.weight_t, self.scale_t = weight.T, scale.T
        self.n, self.k, self.inv_gs = weight.shape[0], 2 * weight.shape[1], inv_gs
        self.coef = inv_gs / SCALE_NUM
        self.gs = self.alpha = None
        if act_gs is not None:
            self.gs = torch.full((1,), act_gs, dtype=torch.float32, device=weight.device)
            self.alpha = torch.full((1,), inv_gs / float(self.gs), dtype=torch.float32, device=weight.device)


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(f"glmf FP4 prefill: {why}")


def _say(tag: str, detail: str) -> None:
    
    if tag not in _SAID:
        _SAID.add(tag)
        logging.getLogger(__name__).warning("glmf FP4 prefill: %s", detail)
def _tally(tag: str, rows: int) -> None:
    calls = _TALLY[tag] = _TALLY.get(tag, 0) + 1
    if calls & (calls - 1) == 0:
        logging.getLogger(__name__).warning("glmf FP4 prefill: %s x%d (last %d rows)", tag, calls, rows)
def _component(name: str):
    
    from glm53f_core import components
    return next((c for c in components.ACTIVE if c.__name__.rsplit(".", 1)[-1] == name), None)

def _load(device) -> None:
    if _FI:
        return
    from flashinfer import fp4_quantize
    from flashinfer.gemm.gemm_base import get_cutlass_fp4_gemm_module

    _require(BACKEND == "cutlass", f"backend {BACKEND!r}: this route runs mm_fp4's cutlass runner")
    major, minor = torch.cuda.get_device_capability(device)
    _FI.update(quantize=fp4_quantize, runner=get_cutlass_fp4_gemm_module(major, minor).cutlass_fp4_gemm_runner(),
               workspace=torch.empty(WORKSPACE, dtype=torch.uint8, device=device), num={})
def _quantize(x: torch.Tensor, gs: torch.Tensor) -> tuple:
    return _FI["quantize"](x, gs, sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=True,
                           is_sf_8x4_layout=False, backend=QUANT_BACKEND, nvfp4_4over6=None)


def _run(copy: _Copy, xq, xs, alpha, out, tactic: int) -> None:
    _FI["runner"].forward([xq, copy.weight_t, xs, copy.scale_t, alpha, BF16, out, 16, True, _FI["workspace"]],
                          tactic=tactic)

def _fp4(copy: _Copy, x: torch.Tensor, out: torch.Tensor, key=None) -> None:
    
    made = _CALL["quantized"].get(key) if key is not None else None
    if made is None:
        if copy.gs is not None:
            amax, gs = None, copy.gs
        else:
            amax = torch.linalg.vector_norm(x, float("inf")).float().clamp_min_(TINY).reshape(1)
            num = _FI["num"].get(x.device)
            if num is None:
                num = _FI["num"][x.device] = torch.full((1,), SCALE_NUM, dtype=torch.float32, device=x.device)
            gs = torch.div(num, amax)
        made = (*_quantize(x, gs), amax)
        if key is not None:
            _CALL["quantized"][key] = made
    xq, xs, amax = made
    _run(copy, xq, xs, copy.alpha if amax is None else amax * copy.coef, out, _TACTIC[(copy.n, copy.k)])


def _act_gs(layer, name: str, k: int):
    
    kind = ACT_SCALE[name]
    if kind == "tensor":
        return None
    if kind != "norm":
        _require(isinstance(kind, float) and kind > 0, f"{name}'s ACT_SCALE {kind!r} is no kind of scale")
        return kind
    attention = layer.self_attn
    norm = {"gate_up_proj": getattr(layer, "post_attention_layernorm", None),
            "fused_qkvbfg_a_proj": getattr(layer, "input_layernorm", None),
            "q_b_proj": getattr(attention, "q_a_layernorm", None),
            "kv_b_proj": getattr(attention, "kv_a_layernorm", None)}.get(name)
    weight = getattr(norm, "weight", None)
    _require(isinstance(weight, torch.Tensor) and tuple(weight.shape) == (k,),
             f"{name}'s input norm holds no weight of {k}: no norm scale")
    bound = k ** 0.5 * float(weight.float().abs().amax())
    _require(bound > 0, f"{name}'s input norm weight is all zeros")
    return SCALE_NUM / (NORM_MARGIN * bound)


def _copy(group: str, name: str, linear, bf16: torch.Tensor, fp8, act_gs) -> _Copy:
    _require(bf16.dtype == BF16 and bf16.dim() == 2 and bf16.is_cuda and bf16.is_contiguous()
             and bf16.shape[0] % 32 == 0 and bf16.shape[1] % 32 == 0,
             f"{group} {name}'s weight is {bf16.dtype} {tuple(bf16.shape)}, not a contiguous CUDA BF16 matrix of "
             "multiples of 32 (the FP4 GEMM's alignment)")
    amax = float(bf16.abs().amax())
    _require(amax > 0, f"{group} {name}'s weight is all zeros")
    gs = torch.full((1,), SCALE_NUM / amax, dtype=torch.float32, device=bf16.device)
    weight, scale = _quantize(bf16, gs)
    return _Copy(group, name, linear, fp8, weight, scale, 1.0 / float(gs), act_gs)


def _sweep(copy: _Copy, bf16: torch.Tensor, rows: int) -> None:
    key = (copy.n, copy.k)
    if key in _TACTIC:
        return
    gen = torch.Generator(device=bf16.device).manual_seed(310)
    x = torch.randn((rows, copy.k), generator=gen, device=bf16.device).to(BF16)
    gs = torch.full((1,), SCALE_NUM / float(x.abs().amax()), dtype=torch.float32, device=x.device)
    xq, xs = _quantize(x, gs)
    alpha = torch.full((1,), copy.inv_gs / float(gs), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, copy.n), dtype=BF16, device=x.device)
    runner, timed, refused = _FI["runner"], [], []
    inputs = [xq, copy.weight_t, xs, copy.scale_t, alpha, BF16, out, 16, True, _FI["workspace"]]
    for tactic in runner.get_valid_tactics(inputs, None):
        try:

            _run(copy, xq, xs, alpha, out, tactic)
            torch.cuda.synchronize()
        except Exception as exc:
            refused.append((tactic, f"{type(exc).__name__}: {str(exc)[:80]}"))
            continue
        times = []
        for _ in range(3):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            _run(copy, xq, xs, alpha, out, tactic)
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1e3)
        timed.append((statistics.median(times), tactic))
    _require(timed, f"no tactic of {BACKEND} runs [{rows} x {copy.k}] x [{copy.n}]: {refused[:3]}")
    us, best = min(timed)
    _run(copy, xq, xs, alpha, out, best)
    ref = F.linear(x[:PROBE_ROWS], bf16).float()
    rel = float((out[:PROBE_ROWS].float() - ref).norm() / ref.norm().clamp_min(TINY))
    _require(rel < PROBE_BAR, f"tactic {best}'s [{rows} x {copy.k}] x [{copy.n}] is {rel:.3f} from the BF16 product "
                              f"(bar {PROBE_BAR}): not this route's operand layout")
    _TACTIC[key] = best
    logging.getLogger(__name__).info(
        "glmf FP4 prefill: [%d x %d] x [%d] tactic %d of %d (%.1f us; slowest %.1f; %d refused), %.3f from BF16",
        rows, copy.k, copy.n, best, len(timed) + len(refused), us, max(timed)[0], len(refused), rel)


def _targets(layer) -> list:
    
    attention, mlp = layer.self_attn, getattr(layer, "mlp", None)
    found = []
    if "G1" in GROUPS and isinstance(mlp, DeepseekV2MoE):
        shared = getattr(mlp, "shared_experts", None)
        found += [("G1", "gate_up_proj", getattr(shared, "gate_up_proj", None)),
                  ("G1", "down_proj", getattr(shared, "down_proj", None))]
    if isinstance(attention, DeepseekV2AttentionMLA):
        if "G2a" in GROUPS:
            found.append(("G2a", "o_proj", getattr(attention, "o_proj", None)))
        if "G4" in GROUPS:
            found += [("G4", "q_b_proj", getattr(attention, "q_b_proj", None)),
                      ("G4", "kv_b_proj", getattr(attention, "kv_b_proj", None))]
    elif "G3" in GROUPS:
        found.append(("G3", "fused_qkvbfg_a_proj", getattr(attention, "fused_qkvbfg_a_proj", None)))
    missing = [f"{g} {n}" for g, n, linear in found if linear is None]
    _require(not missing, f"layer {getattr(layer, 'layer_id', '?')} lacks {missing}")
    return found


def _peers(layer, attention, held: dict) -> None:
    
    xs = _component("dsa_xsplit_route")
    if xs is None:
        return
    held["xs"], held["xs_gemm"] = xs, xs._gemm
    pair = xs.token_split._PAIR.get("group")
    if (pair is None or not xs.fused._state["groups"] or getattr(layer, "is_nextn", False)
            or getattr(attention, "is_nextn", False)):
        return
    rank = pair.rank_in_group
    for group, name, linear in (("G2a", "o_proj", attention.o_proj), ("G4", "q_b_proj", attention.q_b_proj)):
        if group not in held["groups"]:
            continue
        shard = linear.weight.data.contiguous()
        both = torch.empty((2, *shard.shape), dtype=shard.dtype, device=shard.device)
        torch.distributed.all_gather_into_tensor(both, shard, group=pair.device_group)

        held["peers"][name] = _copy(group, name, None, both[1 - rank].contiguous(), None,
                                    _act_gs(layer, name, shard.shape[1]))
        del both

def prepare(layer) -> None:
    
    if id(layer) in _LAYERS:
        return
    if get_disagg().disaggregation_mode == "decode":
        _LAYERS[id(layer)] = None
        _say("decode", "a PD decode server runs no eager prefill forward: nothing served")
        return
    serve = []
    for group, name, linear in _targets(layer):
        if layer.layer_id in SKIP.get(group, ()):
            _say(f"skip {group} {layer.layer_id}", f"layer {layer.layer_id}: {group} keeps version 194's path (SKIP)")
            continue
        shape = tuple(linear.weight.shape)
        if shape != SHAPES[group][name]:
            _say(f"shape {name}", f"{group} {name} is {shape}, not a TP2 rank's {SHAPES[group][name]}: not served")
            continue
        serve.append((group, name, linear))
    if not serve:
        _LAYERS[id(layer)] = None
        return
    _require(not torch.cuda.is_current_stream_capturing(), "prepare under graph capture")
    began = time.perf_counter()
    weight = serve[0][2].weight
    _load(weight.device)
    held = {"copies": [], "peers": {}, "xs": None, "xs_gemm": None, "mapped": False,
            "groups": {group for group, _, _ in serve}, "layer": getattr(layer, "layer_id", "?")}
    for group, name, linear in serve:
        entry = fp8_linear_route._FP8.get(id(linear))
        _require(entry is not None and entry[3], f"{group} {name} holds no prefill FP8 copy: this route takes over the "
                                                  "calls fp8_linear_route's apply serves (its group's FP8 route ships)")
        _require(entry[5][0] <= MIN_ROWS and MAX_ROWS <= entry[5][1],
                 f"{group} {name}'s FP8 rows {entry[5]} do not cover {MIN_ROWS}-{MAX_ROWS}")
        copy = _copy(group, name, linear, linear.weight.data, entry[0], _act_gs(layer, name, linear.weight.shape[1]))
        _BY_FP8[id(entry[0])] = copy
        held["copies"].append(copy)
    if isinstance(layer.self_attn, DeepseekV2AttentionMLA) and held["groups"] & {"G2a", "G4"}:
        _peers(layer, layer.self_attn, held)
    for copy in held["copies"]:
        _sweep(copy, copy.linear.weight.data, SWEEP_ROWS[copy.name])
    _require(all((p.n, p.k) in _TACTIC for p in held["peers"].values()), "a peer half of a shape with no tactic")
    torch.cuda.synchronize()
    _LAYERS[id(layer)] = held
    made = sum(c.weight_t.numel() + c.scale_t.numel() for c in held["copies"] + list(held["peers"].values()))
    _tally(f"prepared ({'+'.join(sorted(held['groups']))}, {made / 2**20:.1f} MiB a layer)", 0)
    logging.getLogger(__name__).info("glmf FP4 prefill: layer %s %s, %.1f MiB of NVFP4 copies in %.2f s",
                                     held["layer"], sorted(held["groups"]), made / 2**20, time.perf_counter() - began)
def _slot(copy: _Copy, rows: int, device):
    if copy.name != "o_proj":
        return None
    stage = _component("e31_stage_route")
    projected = getattr(copy.linear, "forward", None)
    if (stage is None or not stage.fused._state["live"] or getattr(projected, "__module__", None) != stage.__name__
            or not getattr(projected, "__qualname__", "").startswith("_projecting.")):
        return None
    if stage._armed(rows) is not None:
        return None
    return stage._next_slot(rows, device)


def _linear(input, weight, block_size, weight_scale, input_scale=None, bias=None, act_scale_ue8m0=False,
            weight_bf16=None):
    
    copy = _BY_FP8.get(id(weight))
    rows = input.numel() // input.shape[-1] if input.dim() and input.shape[-1] else 0
    if (copy is None or copy.fp8 is not weight or not MIN_ROWS <= rows <= MAX_ROWS
            or torch.cuda.is_current_stream_capturing()):
        return _STOCK_LINEAR(input, weight, block_size, weight_scale, input_scale=input_scale, bias=bias,
                             act_scale_ue8m0=act_scale_ue8m0, weight_bf16=weight_bf16)
    _require(input_scale is None and bias is None and weight_bf16 is None and list(block_size) == [128, 128]
             and input.dtype == BF16 and input.is_cuda and input.is_contiguous() and input.shape[-1] == copy.k,
             f"{copy.group} {copy.name}: a call of {rows} rows ({input.dtype}, input scale {input_scale is not None}, "
             f"bias {bias is not None}, block {block_size}) this route does not mirror")
    x = input.view(-1, copy.k)
    out = _slot(copy, rows, x.device)
    tag = f"{copy.group} {copy.name}" + (" into E31's slot" if out is not None else "")
    if out is None:
        out = torch.empty((rows, copy.n), dtype=BF16, device=x.device)
    _fp4(copy, x, out)
    _tally(tag, rows)
    return out.view(*input.shape[:-1], copy.n)


def _xs_gemm(x, weight, out):
    
    copy = _BY_FP8.get(id(weight[0]))
    if copy is None or copy.fp8 is not weight[0]:
        return _CALL["xs_gemm"](x, weight, out)
    _require(copy.name == "o_proj" and x.dtype == BF16 and x.dim() == 2 and x.is_contiguous() and x.shape[1] == copy.k
             and out.dtype == BF16 and out.is_contiguous() and tuple(out.shape) == (x.shape[0], copy.n),
             f"xsplit's GEMM of {tuple(x.shape)} into {tuple(out.shape)} on {copy.group} {copy.name}: not o_proj's half")
    _fp4(copy, x, out)
    _tally(f"G2a o_proj {'peer' if copy.linear is None else 'own'} half in xsplit", x.shape[0])
    return out


def _seen_quant(x, group_size, *args, **kwargs):
    
    q, scale = _STOCK_QUANT(x, group_size, *args, **kwargs)
    if x.dim() == 2 and x.dtype == BF16 and x.shape[1] == SHAPES["G4"]["q_b_proj"][1]:
        _CALL["seen"][q.data_ptr()] = x
    return q, scale


def _dg(A, As, B, Bs, C, block_n=128, block_k=128):
    
    copy = _BY_FP8.get(id(B))
    if copy is None or copy.fp8 is not B or copy.name != "q_b_proj":
        return _STOCK_DG(A, As, B, Bs, C, block_n, block_k)
    x = _CALL["seen"].get(A.data_ptr())
    _require(x is not None and x.shape[0] == A.shape[0] == C.shape[0] and C.dtype == BF16 and C.is_contiguous()
             and C.shape[1] == copy.n,
             f"a DeepGEMM call of q_b_proj's half on {tuple(A.shape)} rows this route did not see quantized")
    _fp4(copy, x, C, key=A.data_ptr())
    _tally(f"G4 q_b_proj {'peer' if copy.linear is None else 'own'} half in xsplit", x.shape[0])
    return C


def _map_peers(held: dict, weights: dict) -> None:
    
    if held["mapped"]:
        return
    for group, name, own, peer in (("G2a", "o_proj", "own_o", "o"), ("G4", "q_b_proj", "own_q", "q")):
        if group not in held["groups"]:
            continue
        mine = next((c for c in held["copies"] if c.name == name), None)
        _require(mine is not None and weights[own][0] is mine.fp8 and name in held["peers"],
                 f"xsplit's {name} halves of layer {held['layer']} are not the FP8 copies this route took over")
        copy = held["peers"][name]
        _require((mine.gs is None) == (copy.gs is None) and (mine.gs is None or float(mine.gs) == float(copy.gs)),
                 f"{name}'s halves hold two activation scales: they read one activation")
        copy.fp8 = weights[peer][0]
        _BY_FP8[id(copy.fp8)] = copy
    held["mapped"] = True


@contextlib.contextmanager
def routed(layer, forward_batch):
    held = _LAYERS.get(id(layer))
    if not held:
        yield
        return
    _require(not _CALL, "a layer's call inside another's")
    xs = held["xs"]
    weights = xs._WEIGHTS.get(id(layer)) if xs is not None else None
    halves = qb = False

    if (weights and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            and not torch.cuda.is_current_stream_capturing()):
        _map_peers(held, weights)
        halves, qb = "G2a" in held["groups"], "G4" in held["groups"]
    elif xs is not None and weights is None and held["peers"]:
        _say(f"unused peers {held['layer']}", f"xsplit holds nothing for layer {held['layer']}: its peer halves' FP4 "
                                              "copies serve no call")
    _require(fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback is _STOCK_LINEAR
             and (not halves or xs._gemm is held["xs_gemm"])
             and (not qb or (fp8_utils.sglang_per_token_group_quant_fp8 is _STOCK_QUANT
                             and fp8_kernel.deep_gemm_fp8_fp8_bf16_nt is _STOCK_DG)),
             "the FP8 linear's entry, xsplit's GEMM or the quantize / DeepGEMM entry is already replaced at a layer's "
             "call (a nested call, or another route holds it)")
    _CALL.update(seen={}, quantized={}, xs_gemm=held["xs_gemm"])
    fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback = _linear
    try:
        if halves:
            xs._gemm = _xs_gemm
        if qb:
            fp8_utils.sglang_per_token_group_quant_fp8, fp8_kernel.deep_gemm_fp8_fp8_bf16_nt = _seen_quant, _dg
        yield
    finally:
        if qb:
            fp8_utils.sglang_per_token_group_quant_fp8, fp8_kernel.deep_gemm_fp8_fp8_bf16_nt = _STOCK_QUANT, _STOCK_DG
        if halves:
            xs._gemm = held["xs_gemm"]
        fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback = _STOCK_LINEAR
        _CALL.clear()
