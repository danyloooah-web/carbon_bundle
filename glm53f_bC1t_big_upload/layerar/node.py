
import copy
import functools
import logging
import time
import types

import torch
import triton
import triton.language as tl

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)
_FP8 = torch.float8_e4m3fn
PF_MIN_ROWS = 8192
BLOCK = 128
FP8_MAX = 448.0
SH_SELF_CHECK_MAX = 0.15
SELF_CHECK_SEED = 20261113
AMAX_FLOOR = 1e-4
SELF_CHECK_MAX = 0.12
PROBE_SEED = 20261113
AR_STUB_BYTES = 1024
AR_CTAS = 64
M1_PULL_BLOCKS = 48
AR2_N64_MIN_ROWS = 12288
AR2_COPY_MIN_ROWS = 16384
M1_PROBE_ROWS = 2304
ROWSHARD = True
FASTPOST = True
MEGA_DEC = False
MEGA_ZERO_PAD = False
DEC_MIN_ROWS = 16
DEC_MAX_ROWS = 768
MEGA_MIN_ROWS = 17
RS_MIN_BS = 32
RS_PROBE_BS = 96
RS_PROBE_CTX = (99000, 157000)
FP_BH, FP_WARPS = 1024, 4
FP_PROBE_ROWS = 768
MEGA_PROBE_ROWS = (24, 192, 384, 576, 624, 672, 768)
KDA_EAGER = 1
KDAE_STREAMS = 4
ROTQ = True
RQ_BM, RQ_WARPS = 4, 2
FP8PACK = 2
PK_R, PK_WARPS = 16, 8
K2_CONV = True
K2_NORMQ = True
CONV_BLOCK_M = 32
CONV_BLOCK_N = 256
CONV_WARPS = 4
NORMQ_BT = 16
MEGA_PF = True
MEGAPF_SMS = 128
MEGAPF_PROBE_ROWS = 4864
MOE_SIDE = False
ZC_AR2 = True


def _need(ok, what):
    if not ok:
        raise RuntimeError(f"layerar precondition failed: {what}")


def pow2_ceil(x):
    bits = x.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)

def sh_quant_block(w):
    if w.dim() != 2 or w.shape[0] % BLOCK or w.shape[1] % BLOCK:
        raise RuntimeError(f"sharedfp8: block quant needs a 2-D 128-aligned weight, got {tuple(w.shape)}")
    n, k = w.shape
    x = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    amax = x.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = pow2_ceil(amax / FP8_MAX)
    q = (x / scale[:, None, :, None]).clamp_(-FP8_MAX, FP8_MAX).to(_FP8)
    return q.reshape(n, k), scale.contiguous()

def dequant_block(q, scale):
    n, k = q.shape
    x = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * scale[:, None, :, None]
    return x.reshape(n, k)

def full_rows(q, s, rank, ranks):
    rows, k = q.shape
    fq = torch.zeros(rows * ranks, k, dtype=q.dtype, device=q.device)
    fs = torch.ones(s.shape[0] * ranks, s.shape[1], dtype=s.dtype, device=s.device)
    fq[rank * rows:(rank + 1) * rows].copy_(q)
    fs[rank * s.shape[0]:(rank + 1) * s.shape[0]].copy_(s)
    return fq, fs

def full_cols(q, s, rank, ranks):
    n, cols = q.shape
    fq = torch.zeros(n, cols * ranks, dtype=q.dtype, device=q.device)
    fs = torch.ones(s.shape[0], s.shape[1] * ranks, dtype=s.dtype, device=s.device)
    fq[:, rank * cols:(rank + 1) * cols].copy_(q)
    fs[:, rank * s.shape[1]:(rank + 1) * s.shape[1]].copy_(s)
    return fq, fs

def row_errors(got, want):
    g, w = got.float(), want.float()
    return (g - w).norm(dim=1) / w.norm(dim=1).clamp_min(1e-12)

class NodeState:
    def __init__(self, module, stock_forward, sg=None):
        self.module = module
        self.stock_forward = stock_forward
        self.sg = sg
        self.proxy = None

def _check_moe(module, sg):
    dsv2 = sg.dsv2
    _need(not dsv2._enable_pcg_dsv2_dual_stream,
          "SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM routes the MoE by layer id to the stock module")
    _need(not sg.moe_utils.get_moe_runner_backend().is_gluon(), "gluon MoE backend owns the layer")
    _need(not module._enable_a2a_moe, "a2a MoE path (forward_deepep) is enabled")
    _need(module.num_fused_shared_experts == 0, "shared expert is fused into the routed MoE")
    se = module.shared_experts
    _need(type(se) is dsv2.DeepseekV2MLP, f"shared_experts is {type(se).__name__}, not DeepseekV2MLP")
    gu, dn = se.gate_up_proj, se.down_proj
    _need(type(gu) is sg.linear.MergedColumnParallelLinear and type(dn) is sg.linear.RowParallelLinear,
          f"shared linears are {type(gu).__name__}/{type(dn).__name__}")
    for name, lin in (("gate_up_proj", gu), ("down_proj", dn)):
        _need(type(lin.quant_method) is sg.UnquantizedLinearMethod,
              f"stock shared {name} quant method is {type(lin.quant_method).__name__}")
        _need(lin.weight.dtype == torch.bfloat16 and lin.weight.dim() == 2 and lin.bias is None,
              f"stock shared {name} weight is {lin.weight.dtype} {tuple(lin.weight.shape)} bias={lin.bias is not None}")
        _need(lin.params_dtype == torch.bfloat16, f"stock shared {name} was built with params_dtype {lin.params_dtype}")
    _need(not getattr(se, "_enable_nvfp4_gemm_swiglu_fusion", False), "stock shared expert uses the NVFP4 fusion")
    hidden, inter = gu.input_size, dn.input_size
    _need(hidden == module.config.hidden_size == dn.output_size, "shared hidden size mismatch")
    _need(list(gu.output_sizes) == [inter, inter], f"gate_up output sizes {gu.output_sizes}, intermediate {inter}")
    _need(gu.parallel_group == dn.parallel_group, "shared linears use different parallel groups")
    _need(gu.prefix.endswith(".gate_up_proj"), f"gate_up prefix {gu.prefix!r}")
    ip = gu.weight.shape[0] // 2
    _need(tuple(gu.weight.shape) == (2 * ip, hidden) and tuple(dn.weight.shape) == (hidden, ip),
          f"shared weights {tuple(gu.weight.shape)} / {tuple(dn.weight.shape)}")
    _need(ip % BLOCK == 0 and hidden % BLOCK == 0, "shared per-rank sizes are not 128-aligned")

def _construct(module, sg):
    se = module.shared_experts
    gu, dn = se.gate_up_proj, se.down_proj
    qcfg = sg.Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                        weight_block_size=[BLOCK, BLOCK])
    with torch.device(gu.weight.device):
        ours = sg.dsv2.DeepseekV2MLP(
            hidden_size=gu.input_size,
            intermediate_size=dn.input_size,
            hidden_act=module.config.hidden_act,
            quant_config=qcfg,
            reduce_results=dn.reduce_results,
            prefix=gu.prefix[:-len(".gate_up_proj")],
            swiglu_limit=se.swiglu_limit,
            allow_fused_down=se.allow_fused_down,
            parallel_group=gu.parallel_group,
        )
    for name, lin, stock in (("gate_up_proj", ours.gate_up_proj, gu), ("down_proj", ours.down_proj, dn)):
        _need(lin.orig_dtype == lin.params_dtype, f"FP8 {name} orig_dtype {lin.orig_dtype} is not its params_dtype")
        lin.params_dtype = lin.orig_dtype = stock.params_dtype
    rank_size = sg.get_group_rank_size
    same = (
        ("gate_up weight shape", tuple(ours.gate_up_proj.weight.shape), tuple(gu.weight.shape)),
        ("down weight shape", tuple(ours.down_proj.weight.shape), tuple(dn.weight.shape)),
        ("gate_up tp", rank_size(ours.gate_up_proj.tp_group), rank_size(gu.tp_group)),
        ("down tp", rank_size(ours.down_proj.tp_group), rank_size(dn.tp_group)),
        ("output_sizes", list(ours.gate_up_proj.output_sizes), list(gu.output_sizes)),
        ("gather_output", ours.gate_up_proj.gather_output, gu.gather_output),
        ("input_is_parallel", ours.down_proj.input_is_parallel, dn.input_is_parallel),
        ("reduce_results", ours.down_proj.reduce_results, dn.reduce_results),
        ("use_dp_attention_reduce", ours.down_proj.use_dp_attention_reduce, dn.use_dp_attention_reduce),
        ("is_replicated", ours.is_replicated, se.is_replicated),
        ("swiglu_limit", ours.swiglu_limit, se.swiglu_limit),
        ("allow_fused_down", ours.allow_fused_down, se.allow_fused_down),
        ("use_fused_clamp_act_mul", ours.use_fused_clamp_act_mul, se.use_fused_clamp_act_mul),
        ("act_fn", type(ours.act_fn), type(se.act_fn)),
    )
    for name, mine, stock in same:
        _need(mine == stock, f"FP8 shared expert {name}={mine!r}, stock {stock!r}")
    gemms = (sg.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
             sg.fp8_utils.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback)
    for name, lin in (("gate_up_proj", ours.gate_up_proj), ("down_proj", ours.down_proj)):
        qm = lin.quant_method
        _need(type(qm) is sg.Fp8LinearMethod and qm.block_quant and not qm.use_mxfp8
              and not qm.block_fp8_as_mxfp8 and not qm.use_marlin and qm.weight_block_size == [BLOCK, BLOCK],
              f"SGLang built {type(qm).__name__} for the FP8 {name}")
        fn = getattr(qm.w8a8_block_fp8_linear, "func", qm.w8a8_block_fp8_linear)
        _need(fn in gemms, f"FP8 {name} block GEMM is {getattr(fn, '__name__', fn)!r}, not DeepGEMM or FlashInfer")
        n, k = lin.weight.shape
        _need(lin.weight.dtype == _FP8 and tuple(lin.weight_scale_inv.shape) == (n // BLOCK, k // BLOCK)
              and lin.weight_scale_inv.dtype == torch.float32 and lin.input_scale is None,
              f"FP8 {name} parameters {lin.weight.dtype} {tuple(lin.weight_scale_inv.shape)}")
    return ours, fn.__name__

def _load(module, ours, sg):
    se = module.shared_experts
    gu, dn = ours.gate_up_proj, ours.down_proj
    rank, ranks = sg.get_group_rank_size(gu.tp_group)
    ip = gu.weight.shape[0] // 2
    stock_gu = se.gate_up_proj.weight.data
    ref_gu = torch.empty(gu.weight.shape, dtype=torch.float32, device=stock_gu.device)
    for shard in (0, 1):
        q, s = sh_quant_block(stock_gu[shard * ip:(shard + 1) * ip])
        fq, fs = full_rows(q, s, rank, ranks)
        gu.weight.weight_loader(gu.weight, fq, shard)
        gu.weight_scale_inv.weight_loader(gu.weight_scale_inv, fs, shard)
        _need(torch.equal(gu.weight.data[shard * ip:(shard + 1) * ip].view(torch.uint8), q.view(torch.uint8))
              and torch.equal(gu.weight_scale_inv.data[shard * ip // BLOCK:(shard + 1) * ip // BLOCK], s),
              f"gate_up shard {shard}: the loader did not place this rank's FP8 rows")
        ref_gu[shard * ip:(shard + 1) * ip] = dequant_block(q, s)
    q, s = sh_quant_block(se.down_proj.weight.data)
    fq, fs = full_cols(q, s, rank, ranks)
    dn.weight.weight_loader(dn.weight, fq)
    dn.weight_scale_inv.weight_loader(dn.weight_scale_inv, fs)
    _need(torch.equal(dn.weight.data.view(torch.uint8), q.view(torch.uint8))
          and torch.equal(dn.weight_scale_inv.data, s), "down_proj: the loader did not place this rank's FP8 columns")
    ref_dn = dequant_block(q, s)
    for name, lin in (("gate_up_proj", gu), ("down_proj", dn)):
        lin.quant_method.process_weights_after_loading(lin)
        _need(lin.weight.dtype == _FP8, f"processed FP8 weight is {lin.weight.dtype}")
        qm = lin.quant_method
        due = (getattr(qm.w8a8_block_fp8_linear, "func", qm.w8a8_block_fp8_linear)
               is sg.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
               and sg.should_deepgemm_weight_requant_ue8m0(
                   weight_block_size=[BLOCK, BLOCK], output_dtype=lin.orig_dtype, weight_shape=lin.weight.shape))
        _need((lin.weight_scale_inv.dtype == torch.int32) == due,
              f"FP8 {name}: requant due={due} but the served scale is {lin.weight_scale_inv.dtype}")
    return ref_gu, ref_dn

def unpack_ue8m0(packed, n, k):
    _need(packed.dim() == 2 and packed.shape[0] == n, f"packed UE8M0 scale shape {tuple(packed.shape)} for N={n}")
    b = torch.stack([(packed >> (8 * i)) & 0xFF for i in range(4)], dim=-1).reshape(n, -1)
    return (b[::BLOCK, :k // BLOCK].to(torch.int32) << 23).view(torch.float32)

def _served_values(lin):
    scale = lin.weight_scale_inv.data
    n, k = lin.weight.shape
    if scale.dtype == torch.int32:
        scale = unpack_ue8m0(scale, n, k)
    _need(scale.dtype == torch.float32 and tuple(scale.shape) == (n // BLOCK, k // BLOCK),
          f"processed scale {scale.dtype} {tuple(scale.shape)}")
    return dequant_block(lin.weight.data, scale)

def sh_proxy(module, ours):
    proxy = copy.copy(module)
    object.__setattr__(proxy, "_modules", dict(module._modules))
    proxy._modules["shared_experts"] = ours
    try:
        object.__delattr__(proxy, "forward")
    except AttributeError:
        pass
    _need(module._modules["shared_experts"] is not ours and proxy.shared_experts is ours, "proxy swap")
    _need(proxy.experts is module.experts and proxy.gate is module.gate and proxy.topk is module.topk,
          "proxy does not share stock's router and routed experts")
    return proxy

def sh_build(state, hidden):
    _need(not torch.cuda.is_current_stream_capturing(),
          "first prefill-sized call is inside a CUDA-graph capture (SGLang's runners warm every shape "
          "eagerly first, so this engine broke that order)")
    sg, module = state.sg, state.module
    t0 = time.perf_counter()
    with torch.no_grad():
        ours, gemm = _construct(module, sg)
        ref_gu, ref_dn = _load(module, ours, sg)
        lossless = (torch.equal(_served_values(ours.gate_up_proj), ref_gu)
                    and torch.equal(_served_values(ours.down_proj), ref_dn))
        del ref_gu, ref_dn
        _need(lossless, "sharedfp8: SGLang's UE8M0 requant changed the shared-expert FP8 values")
        gen = torch.Generator(device=hidden.device)
        gen.manual_seed(SELF_CHECK_SEED)
        x = torch.randn(PF_MIN_ROWS, hidden.shape[1], generator=gen, device=hidden.device, dtype=hidden.dtype)
        err = row_errors(ours(x), module.shared_experts(x))
        p50, p90 = (v.item() for v in torch.quantile(err, torch.tensor([0.5, 0.9], device=err.device)))
        del x, err
    _need(p90 < SH_SELF_CHECK_MAX, f"FP8 shared expert self-check row error p90 {p90:.4f} >= {SH_SELF_CHECK_MAX}")
    proxy = sh_proxy(module, ours)
    gu = ours.gate_up_proj
    nbytes = sum(t.numel() * t.element_size() for t in (gu.weight, gu.weight_scale_inv,
                                                       ours.down_proj.weight, ours.down_proj.weight_scale_inv))
    rank, ranks = sg.get_group_rank_size(gu.tp_group)
    scale = "ue8m0-packed" if gu.weight_scale_inv.dtype == torch.int32 else str(gu.weight_scale_inv.dtype)
    _log.warning("sharedfp8: built %s tp_rank=%d/%d gemm=%s scale=%s lossless=%s selfcheck_p50=%.4f p90=%.4f "
                 "fp8_bytes=%d build_s=%.2f first_rows=%d", gu.prefix[:-len(".gate_up_proj")],
                 rank, ranks, gemm, scale, lossless, p50, p90, nbytes, time.perf_counter() - t0, hidden.shape[0])
    return proxy

def sh_prepare(module):
    from sglang.srt.distributed.utils import get_group_rank_size
    from sglang.srt.layers import linear
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
    from sglang.srt.models import deepseek_v2 as dsv2
    from sglang.srt.runtime_context import get_forward
    stock_forward = type(module).forward
    if type(module) is not dsv2.DeepseekV2MoE:
        _log.warning("sharedfp8: %s node serves stock", type(module).__name__)
        return NodeState(module, stock_forward)
    sg = types.SimpleNamespace(
        dsv2=dsv2, linear=linear, moe_utils=moe_utils, fp8_utils=fp8_utils, Fp8Config=Fp8Config,
        Fp8LinearMethod=Fp8LinearMethod, UnquantizedLinearMethod=UnquantizedLinearMethod,
        get_group_rank_size=get_group_rank_size, get_forward=get_forward,
        should_deepgemm_weight_requant_ue8m0=should_deepgemm_weight_requant_ue8m0)
    _check_moe(module, sg)
    return NodeState(module, stock_forward, sg)

def sh_forward(state, *args, **kwargs):
    hidden = args[0] if args else kwargs["hidden_states"]
    if state.sg is None or hidden.shape[0] < PF_MIN_ROWS:
        return state.stock_forward(state.module, *args, **kwargs)
    _need(hidden.dim() == 2, f"MoE input is {tuple(hidden.shape)}, not [tokens, hidden]")
    _need(not state.sg.get_forward().sp_active, "layernorm sequence parallelism is active")
    if state.proxy is None:
        state.proxy = sh_build(state, hidden)
    if getattr(state, "x", None) is not None and moe_x_ok(state, hidden, args, kwargs):
        gza = args[2] if len(args) > 2 else kwargs.get("gemm_output_zero_allocator")
        if moe_x_variant(state) not in _MOE.get("proved", ()):
            return moe_x_probe(state, args, kwargs, hidden, gza)
        return fp_forward(state, hidden, gza, args, kwargs)
    return state.stock_forward(state.proxy, *args, **kwargs)


def ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)

def pj_quant_block(w):
    if w.dim() != 2 or w.shape[0] % BLOCK or w.shape[1] % BLOCK:
        raise RuntimeError(f"projfp8: block quant needs a 2-D 128-aligned weight, got {tuple(w.shape)}")
    n, k = w.shape
    x = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = ceil_to_ue8m0(x.abs().amax(dim=(1, 3)).clamp(min=AMAX_FLOOR) / FP8_MAX)
    q = (x * (1.0 / scale)[:, None, :, None]).to(_FP8)
    return q.reshape(n, k), scale.contiguous()

def pj_dequant_block(q, scale_inv):
    n, k = q.shape
    x = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * scale_inv[:, None, :, None]
    return x.reshape(n, k)

def round_up(n):
    return (n + BLOCK - 1) // BLOCK * BLOCK

def qkvbfg_layout(qkv_rows, b_rows, fg_rows):
    b_pad = round_up(b_rows)
    return qkv_rows + b_pad + fg_rows, qkv_rows + b_rows, qkv_rows + b_pad, qkv_rows + b_rows + fg_rows

def qkvbfg_fp8_weight(w, qkv_rows, b_rows):
    fg_rows = w.shape[0] - qkv_rows - b_rows
    n8, _, fg_at, _ = qkvbfg_layout(qkv_rows, b_rows, fg_rows)
    out = torch.zeros(n8, w.shape[1], dtype=w.dtype, device=w.device)
    out[:qkv_rows + b_rows].copy_(w[:qkv_rows + b_rows])
    out[fg_at:].copy_(w[qkv_rows + b_rows:])
    return out

def qkvbfg_view(y, qkv_rows, b_rows, fg_rows):
    _, gap_at, fg_at, width = qkvbfg_layout(qkv_rows, b_rows, fg_rows)
    if fg_at != gap_at:
        y[:, gap_at:width].copy_(y[:, fg_at:fg_at + fg_rows].clone())
    return y[:, :width]

QPAD_KWARGS = frozenset({"q_rope", "k_rope", "topk_indices", "cos_sin_cache", "is_neox", "llama_4_scaling"})

def head_major(q):
    return (q.dim() == 3 and q.shape[0] > 1 and q.shape[1] > 1 and q.shape[2] % 8 == 0
            and tuple(q.stride()) == (q.shape[2], q.shape[0] * q.shape[2], 1))

def qpad_q8(q):
    tmp = q.to(_FP8)
    _need(tuple(tmp.stride()) == tuple(q.stride()), f"qpad: the FP8 cast changed q's layout to {tuple(tmp.stride())}")
    out = torch.empty(q.shape, dtype=_FP8, device=q.device)
    out.view(torch.int64).copy_(tmp.view(torch.int64))
    return out

def qpad_skip(env, attn, q, fb, kwargs):
    if not (torch.is_tensor(q) and q.dtype == torch.bfloat16 and head_major(q)
            and q.shape[1] == attn.tp_q_head_num and q.shape[2] == attn.head_dim):
        return "q-layout"
    q_rope, k_rope = kwargs.get("q_rope"), kwargs.get("k_rope")
    if not (set(kwargs) <= QPAD_KWARGS and torch.is_tensor(q_rope) and q_rope.shape[-1] == 0
            and torch.is_tensor(k_rope) and kwargs.get("cos_sin_cache") is None
            and kwargs.get("llama_4_scaling") is None):
        return "attention-args"
    mode = fb.forward_mode
    if not mode.is_extend(include_draft_extend_v2=True) or (mode.is_target_verify() and not FP8PACK):
        return "forward-mode"
    backend = env.get_attn_backend()
    if type(backend) is env.Hybrid:
        if attn.layer_id not in backend.full_attn_layers:
            return "linear-layer"
        backend = backend.full_attn_backend
    if type(backend) is not env.Dsa:
        return "backend-class"
    if not (backend.kv_cache_dtype == _FP8 and backend.qk_rope_head_dim == 0 and backend.kv_lora_rank == q.shape[2]
            and backend.attn_dcp_size == 1 and not backend.dsa_shards_index_k and not attn.is_cross_attention):
        return "backend-config"
    impl = backend.dsa_decode_impl if mode.is_draft_extend_v2() or mode.is_target_verify() else backend.dsa_prefill_impl
    if backend._resolve_kpool_tail_backend(kwargs.get("topk_indices"), impl) != "trtllm" or backend.use_mha:
        return "dsa-impl"
    return None

class Fp8Proj(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, x, *rest, **extra):
        if rest or extra:
            raise RuntimeError("projfp8: projection called with arguments the stock prefill path never passes")
        if not x.is_contiguous():
            x = x.contiguous()
        return self.inner(x)

class QkvbfgFp8(torch.nn.Module):
    def __init__(self, lin, qkv_rows, b_rows, fg_rows):
        super().__init__()
        self.lin = lin
        self.rows = (qkv_rows, b_rows, fg_rows)
    def forward(self, x, *rest, **extra):
        if rest or extra:
            raise RuntimeError("projfp8: fused qkvbfg projection called with unexpected arguments")
        if not x.is_contiguous():
            x = x.contiguous()
        return qkvbfg_view(self.lin(x)[0], *self.rows)

class _QpadEager:
    def __init__(self, owner):
        self.owner = owner
    def run_eager(self, q, k, v, output, save_kv_cache=True, key_value_num_tokens=None, idx_output=None,
                  lse_output=None, **kwargs):
        o = self.owner
        fb = o.env.get_forward_batch()
        n = o.env.query_tokens(fb, fb.global_num_token_non_padded_cpu)
        why = qpad_skip(o.env, o.stock, q, fb, kwargs)
        if why is None and not (n == q.shape[0] and key_value_num_tokens is None and idx_output is None
                                and lse_output is None):
            why = "graph-padded-rows"
        return o.stock._eager_attention.run_eager(o.route(q, why), k, v, output, save_kv_cache, key_value_num_tokens,
                                                  idx_output, lse_output, **kwargs)

class QpadAttn(torch.nn.Module):
    def __init__(self, stock, env, prefix):
        super().__init__()
        object.__setattr__(self, "stock", stock)
        self.env, self.prefix, self.seen = env, prefix, set()
        self._eager_attention = _QpadEager(self)
    @property
    def layer_id(self):
        return self.stock.layer_id
    def route(self, q, why):
        if why not in self.seen:
            self.seen.add(why)
            _log.warning("projfp8: qpad %s %s rows=%s", self.prefix, why or "on",
                         tuple(q.shape) if torch.is_tensor(q) else None)
        return q if why is not None else qpad_q8(q)
    def forward(self, q, k, v, forward_batch, save_kv_cache=True, key_value_num_tokens=None, **kwargs):
        why = qpad_skip(self.env, self.stock, q, forward_batch, kwargs)
        if why is None and (self.env.in_graph_scope() or key_value_num_tokens is not None
                            or self.env.padded_rows(q.shape[0], forward_batch) is not None):
            why = "graph-or-padded"
        return self.stock(self.route(q, why), k, v, forward_batch, save_kv_cache, key_value_num_tokens, **kwargs)

class ProjFp8State:
    def __init__(self, module, kind, hs_pos, stock_forward, plan):
        self.module = module
        self.kind = kind
        self.hs_pos = hs_pos
        self.stock_forward = stock_forward
        self.plan = plan
        self.proxy = None
        self.get_forward = None
        self.rq = self.small = None

class _Env:
    def __init__(self):
        from sglang.srt.layers import linear
        from sglang.srt.layers.quantization import fp8, fp8_utils, unquant
        from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
        from sglang.srt.runtime_context import get_forward
        self.ReplicatedLinear = linear.ReplicatedLinear
        self.Fp8Config = fp8.Fp8Config
        self.Fp8LinearMethod = fp8.Fp8LinearMethod
        self.UnquantizedLinearMethod = unquant.UnquantizedLinearMethod
        self.inverse_transform_scale_ue8m0 = fp8_utils.inverse_transform_scale_ue8m0
        self.deepgemm_linear = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        self.requant_wanted = should_deepgemm_weight_requant_ue8m0
        self.get_forward = get_forward
    def fp8_config(self):
        return self.Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                              weight_block_size=[BLOCK, BLOCK])

class _QpadEnv:
    def __init__(self):
        from sglang.kernels.ops.attention import utils as attn_utils
        from sglang.srt.layers.attention import dsa_backend, hybrid_linear_attn_backend
        from sglang.srt.layers.attention.graph_utils import attention_query_tokens, padded_extend_real_tokens
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
        from sglang.srt.model_executor.runner_utils.prefill_graph import is_in_prefill_graph_scope
        _need(dsa_backend.mla_quantize_for_fp8_no_rope is attn_utils.mla_quantize_for_fp8_no_rope,
              "qpad: dsa_backend's mla_quantize_for_fp8_no_rope is not SGLang's kernels/ops/attention/utils.py one")
        self.Hybrid = hybrid_linear_attn_backend.HybridLinearAttnBackend
        self.Dsa = dsa_backend.DeepseekSparseAttnBackend
        self.get_attn_backend = get_attn_backend
        self.get_forward_batch = get_forward_batch
        self.in_graph_scope = is_in_prefill_graph_scope
        self.padded_rows = padded_extend_real_tokens
        self.query_tokens = attention_query_tokens

def _check_bf16_linear(lin, cls, um, name, *, row_parallel=False, fp8_rows=None):
    _need(type(lin) is cls, f"{name} is {type(lin).__name__}, not {cls.__name__}")
    _need(type(lin.quant_method) is um, f"{name} quant method is {type(lin.quant_method).__name__}")
    w = lin.weight
    _need(w.dtype == torch.bfloat16 and w.dim() == 2, f"{name} weight is {w.dtype} {tuple(w.shape)}")
    _need(lin.params_dtype == torch.bfloat16, f"{name} was built with params_dtype {lin.params_dtype}")
    n8 = w.shape[0] if fp8_rows is None else fp8_rows
    _need(0 < n8 <= w.shape[0] and n8 % BLOCK == 0 and w.shape[1] % BLOCK == 0,
          f"{name} FP8 rows {n8} of weight {tuple(w.shape)} not 128-aligned")
    _need(w.is_contiguous(), f"{name} weight is not contiguous")
    _need(getattr(lin, "bias", None) is None, f"{name} has a bias")
    _need(getattr(lin, "small_m_rowmajor", False) is False and getattr(lin, "small_m_tiled_weight", None) is None,
          f"{name} carries a small-M weight layout")
    if row_parallel:
        _need(lin.reduce_results is False, f"{name} reduces its output inside the linear")
        _need(lin.input_is_parallel is True, f"{name} splits its input")
        _need(getattr(lin, "use_decode_attn_tp", False) is False, f"{name} runs decode attention TP")
    elif hasattr(lin, "gather_output"):
        _need(lin.gather_output is False, f"{name} gathers its output")

def _check_common():
    from sglang.srt.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
    from sglang.srt.runtime_context import get_lora
    _need(not is_symmetric_memory_enabled(), "symmetric-memory GEMM outputs are enabled (RowParallelLinear "
          "would allocate o_proj outputs in the symmetric pool)")
    _need(not (get_lora().enable_lora or get_lora().lora_paths), "LoRA is enabled")

def _plan_kda(module, linear, um):
    if not module.do_fuse_qkvbfg:
        _check_bf16_linear(module.qkv_proj, linear.QKVParallelLinear, um, "qkv_proj")
        _check_bf16_linear(module.o_proj, linear.RowParallelLinear, um, "o_proj", row_parallel=True)
        return [("qkv_proj", None), ("o_proj", None)]
    lin = module.fused_qkvbfg_a_proj
    local = module.local_num_heads * module.head_dim
    _check_bf16_linear(lin, linear.MergedColumnParallelRepeatedLinear, um, "fused_qkvbfg_a_proj",
                       fp8_rows=3 * local)
    sizes = list(lin.output_partition_sizes)
    _need(len(sizes) == 6 and sizes[:3] == [local, local, local],
          f"fused_qkvbfg_a_proj partitions {sizes}, expected q/k/v of {local} rows first")
    _need(list(module.split_sizes) == [3 * local, sizes[3], sizes[4] + sizes[5]],
          f"split_sizes {list(module.split_sizes)} disagree with the fused partitions {sizes}")
    _need(lin.weight.shape[0] == sum(sizes) and lin.weight.shape[1] == module.hidden_size,
          f"fused_qkvbfg_a_proj weight {tuple(lin.weight.shape)}")
    _need(0 < sizes[3] and sizes[4] % BLOCK == 0 and sizes[5] % BLOCK == 0 and sizes[4] + sizes[5] > 0,
          f"b / f_a / g_a partitions {sizes[3:]} (f_a and g_a must be 128-aligned)")
    _check_bf16_linear(module.o_proj, linear.RowParallelLinear, um, "o_proj", row_parallel=True)
    return [("fused_qkvbfg_a_proj", (3 * local, sizes[3], sizes[4] + sizes[5])), ("o_proj", None)]

def _plan_mla(module, linear, um):
    _need(module.q_lora_rank is not None and module.has_q_b_proj, "MLA layer without q_lora / q_b_proj")
    _check_bf16_linear(module.q_b_proj, linear.ColumnParallelLinear, um, "q_b_proj")
    _check_bf16_linear(module.o_proj, linear.RowParallelLinear, um, "o_proj", row_parallel=True)
    _need(module._q_b_proj_verified_shape is False and module._use_min_latency_q_b_gemm is not True,
          "q_b_proj takes the min-latency fused GEMM, which reads its weight directly")
    _need(module.dcp_size == 1 and module.q_b_proj_qrep is None and module.q_b_proj_qrep_weight is None,
          "decode-context-parallel replicated q_b_proj is configured")
    from sglang.srt.layers.radix_attention import RadixAttention
    attn = module.attn_mqa
    _need(type(attn) is RadixAttention and attn.head_dim == module.kv_lora_rank + module.qk_rope_head_dim,
          f"attn_mqa is {type(attn).__name__} with head_dim {getattr(attn, 'head_dim', None)} (QPAD stands in for "
          f"SGLang's RadixAttention over the {module.kv_lora_rank + module.qk_rope_head_dim}-wide latent)")
    return [("q_b_proj", None), ("o_proj", None)]

def pj_prepare(module):
    from sglang.srt.layers import linear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod as um
    from sglang.srt.models import deepseek_v2, glm5_next
    if type(module) is glm5_next.Glm5NextLinearAttention:
        kind, hs_pos, plan = "kda", 0, _plan_kda(module, linear, um)
    elif type(module) is deepseek_v2.DeepseekV2AttentionMLA:
        kind, hs_pos, plan = "mla", 1, _plan_mla(module, linear, um)
    else:
        raise RuntimeError(f"projfp8 precondition failed: node is {type(module).__name__}")
    _check_common()
    stock_forward = type(module).forward
    wrapped = getattr(module.forward, "__wrapped__", None)
    if wrapped is not None:
        _need(getattr(wrapped, "__func__", None) is stock_forward and getattr(wrapped, "__self__", None) is module,
              "the forward the dispatcher wraps is not the class forward bound to this module")
    return ProjFp8State(module, kind, hs_pos, stock_forward, plan)

def _probe_check(env, lin, w_bf16, name, segments):
    gen = torch.Generator(device=w_bf16.device)
    gen.manual_seed(PROBE_SEED)
    x = torch.randn(PF_MIN_ROWS, w_bf16.shape[1], generator=gen, device=w_bf16.device, dtype=torch.float32)
    x = x.to(torch.bfloat16)
    got = lin(x)[0]
    ref = torch.nn.functional.linear(x, w_bf16)
    _need(got.shape == ref.shape and got.dtype == torch.bfloat16, f"{name}: FP8 output {tuple(got.shape)} {got.dtype}")
    rels = {}
    for seg, (lo, hi) in segments.items():
        g, r = got[:, lo:hi].float(), ref[:, lo:hi].float()
        rels[seg] = round(((g - r).norm() / r.norm().clamp_min(1e-30)).item(), 5)
        _need(rels[seg] < SELF_CHECK_MAX, f"{name}[{seg}]: FP8 GEMM probe error {rels[seg]:.4f} >= {SELF_CHECK_MAX}")
    return rels

def _served_equal(env, lin, q, s, name):
    n, k = q.shape
    qm = lin.quant_method
    deepgemm = getattr(qm.w8a8_block_fp8_linear, "func", qm.w8a8_block_fp8_linear) is env.deepgemm_linear
    want_packed = deepgemm and env.requant_wanted([BLOCK, BLOCK], output_dtype=lin.orig_dtype,
                                                  weight_shape=lin.weight.shape)
    w = lin.weight.data
    s_now = lin.weight_scale_inv.data
    packed = s_now.dtype == torch.int32
    _need(packed == want_packed, f"{name}: served scale is {s_now.dtype} but SGLang's requant predicate says "
          f"packed={want_packed} (deepgemm={deepgemm}, orig_dtype={lin.orig_dtype})")
    if packed:
        s_now = env.inverse_transform_scale_ue8m0(s_now, mn=n)
    _need(w.dtype == _FP8 and tuple(w.shape) == (n, k), f"{name}: served weight {w.dtype} {tuple(w.shape)}")
    _need(s_now.dtype == torch.float32 and tuple(s_now.shape) == tuple(s.shape),
          f"{name}: served scale {s_now.dtype} {tuple(s_now.shape)}")
    _need(torch.equal(pj_dequant_block(w, s_now), pj_dequant_block(q, s)),
          f"{name}: the weight SGLang serves is not the weight the bundle quantized")
    return "ue8m0-packed" if packed else "fp32"

def _fp8_linear(env, w_bf16, prefix, params_dtype, segments=None):
    n, k = w_bf16.shape
    _need(params_dtype == torch.bfloat16, f"{prefix}: stock params_dtype {params_dtype}")
    with torch.device(w_bf16.device):
        lin = env.ReplicatedLinear(k, n, bias=False, params_dtype=params_dtype,
                                   quant_config=env.fp8_config(), prefix=prefix)
    qm = lin.quant_method
    _need(type(qm) is env.Fp8LinearMethod and qm.block_quant and not qm.use_marlin and not qm.use_mxfp8
          and not qm.block_fp8_as_mxfp8 and list(qm.weight_block_size) == [BLOCK, BLOCK],
          f"{prefix}: SGLang built {type(qm).__name__} (marlin/mxfp8 or another layout)")
    _need(lin.orig_dtype == torch.bfloat16 and lin.params_dtype == torch.bfloat16,
          f"{prefix}: FP8 linear orig_dtype {getattr(lin, 'orig_dtype', None)}")
    _need(lin.weight.dtype == _FP8 and tuple(lin.weight.shape) == (n, k)
          and lin.weight_scale_inv.dtype == torch.float32
          and tuple(lin.weight_scale_inv.shape) == (n // BLOCK, k // BLOCK), f"{prefix}: FP8 parameter layout")
    q, s = pj_quant_block(w_bf16)
    lin.weight_loader(lin.weight, q)
    lin.weight_loader(lin.weight_scale_inv, s)
    _need(torch.equal(lin.weight.data.view(torch.uint8), q.view(torch.uint8))
          and torch.equal(lin.weight_scale_inv.data, s), f"{prefix}: the loader did not place the FP8 weight")
    qm.process_weights_after_loading(lin)
    fmt = _served_equal(env, lin, q, s, prefix)
    rels = _probe_check(env, lin, w_bf16, prefix, segments or {"all": (0, n)})
    backend = getattr(qm.w8a8_block_fp8_linear, "__name__", None) or getattr(
        getattr(qm.w8a8_block_fp8_linear, "func", None), "__name__", "?")
    return lin, {"name": prefix, "n": n, "k": k, "format": fmt, "probe_rel": rels, "backend": backend}

def pj_proxy(module, swaps):
    proxy = copy.copy(module)
    object.__setattr__(proxy, "_modules", dict(module._modules))
    for name, mod in swaps.items():
        _need(name in module._modules, f"stock node has no child {name}")
        proxy._modules[name] = mod
    try:
        object.__delattr__(proxy, "forward")
    except AttributeError:
        pass
    for name, child in module._modules.items():
        if name in swaps:
            _need(module._modules[name] is child and proxy._modules[name] is swaps[name], f"proxy swap {name}")
        else:
            _need(proxy._modules[name] is child, f"proxy does not share stock child {name}")
    return proxy

def pj_build(state):
    _need(not torch.cuda.is_current_stream_capturing(),
          "the node's first prefill-sized call is inside a CUDA-graph capture; the FP8 copies are built eagerly")
    t0 = time.monotonic()
    env = _Env()
    m = state.module
    prefix = getattr(m, "prefix", None) or f"model.layers.{getattr(m, 'layer_id', getattr(m, 'layer_idx', '?'))}.self_attn"
    swaps, facts = {}, []
    with torch.no_grad():
        for name, spec in state.plan:
            stock = m._modules[name]
            if spec is None:
                lin, fact = _fp8_linear(env, stock.weight.data, f"{prefix}.{name}.projfp8", stock.params_dtype)
                swaps[name] = Fp8Proj(lin)
            else:
                qkv_rows, b_rows, fg_rows = spec
                w8 = qkvbfg_fp8_weight(stock.weight.data, qkv_rows, b_rows)
                _, _, fg_at, _ = qkvbfg_layout(qkv_rows, b_rows, fg_rows)
                segments = {"qkv": (0, qkv_rows), "b": (qkv_rows, qkv_rows + b_rows), "fg": (fg_at, fg_at + fg_rows)}
                lin, fact = _fp8_linear(env, w8, f"{prefix}.{name}.projfp8_qkvbfg", stock.params_dtype, segments)
                del w8
                swaps[name] = QkvbfgFp8(lin, qkv_rows, b_rows, fg_rows)
            facts.append(fact)
        if state.kind == "mla":
            swaps["attn_mqa"] = (sx_attn(m.attn_mqa, _QpadEnv(), prefix, "fp8proxy") if FP8PACK
                                 else QpadAttn(m.attn_mqa, _QpadEnv(), prefix))
            facts.append({"name": f"{prefix}.attn_mqa", "qpad": True, "heads": m.attn_mqa.tp_q_head_num})
        if ROTQ and state.rq is not None:
            swaps["indexer"] = state.rq
        proxy = pj_proxy(m, swaps)
    nbytes = sum(p.numel() * p.element_size() for mod in swaps.values() for p in mod.parameters())
    state.get_forward = env.get_forward
    state.proxy = proxy
    _log.warning("projfp8: built %s kind=%s bytes=%d seconds=%.1f %s", prefix, state.kind, nbytes,
                 time.monotonic() - t0, facts)

def pj_forward(state, *args, **kwargs):
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else (
        args[state.hs_pos] if len(args) > state.hs_pos else None)
    if isinstance(hs, tuple):
        hs = hs[0]
    rows = hs.shape[0] if torch.is_tensor(hs) and hs.dim() >= 1 else 0
    if rows < PF_MIN_ROWS:
        small = state.small if ROTQ or FP8PACK else None
        return state.stock_forward(state.module if small is None else small, *args, **kwargs)
    if state.proxy is None:
        pj_build(state)
    if state.get_forward().sp_active:
        raise RuntimeError("projfp8: sequence-parallel forward reached a swapped node")
    return state.stock_forward(state.proxy, *args, **kwargs)



_PROCESS = {}

def _nccl_limit(gc, tp):
    ca, off = gc.ca_comm, (lambda comm: comm is None or comm.disabled)
    _need(gc.world_size == 2 and list(gc.ranks) == list(tp.ranks) and ca is not None and not ca.disabled
          and gc.pymscclpp_comm is None and gc.qr_comm is None and off(gc.pcie_ipc_comm)
          and off(gc.torch_symm_mem_comm) and off(gc.hpu_communicator) and off(gc.xpu_communicator)
          and gc._fi_workspace_hint is None and not gc.is_symmetric_memory_enabled(),
          f"group {gc.unique_name}: a large all-reduce would not be stock's NCCL sum over the TP ranks")
    return ca.max_size

class Nccl64:
    def __init__(self, sg, tp):
        lib, rank, pw = sg.pw.NCCLLibrary(), sg.dist.get_rank(tp.cpu_group), sg.pw
        uid = lib.ncclGetUniqueId() if rank == 0 else pw.ncclUniqueId()
        tensor = torch.ByteTensor(list(uid.internal))
        sg.dist.broadcast(tensor, src=sg.dist.get_process_group_ranks(tp.cpu_group)[0], group=tp.cpu_group)
        for i, byte in enumerate(tensor.tolist()):
            uid.internal[i] = byte
        conf = pw.ncclConfig_t.create()
        conf.minCTAs = conf.maxCTAs = AR_CTAS
        with torch.cuda.device(tp.device):
            self.comm = lib.ncclCommInitRankConfig(2, uid, rank, conf)
        self.lib, self.pw, self.op = lib, pw, pw.ncclRedOpTypeEnum.from_torch(sg.dist.ReduceOp.SUM)
        self.dtype = pw.ncclDataTypeEnum.from_torch(torch.bfloat16)
    def __call__(self, x):
        ptr = self.pw.buffer_type(x.data_ptr())
        stream = self.pw.cudaStream_t(torch.cuda.current_stream().cuda_stream)
        self.lib.ncclAllReduce(ptr, ptr, x.numel(), self.dtype, self.op, self.comm, stream)
        return x

class ExactAr:
    def __init__(self, sg, tp, others, hidden, rows):
        self.sg, self.tp, self.hidden, self.rows, self.cap = sg, tp, hidden, rows, rows * hidden * 2
        self.v2 = self.buf = self.row = self.n64 = self.side = self.evs = None
        self.plans = {}
        every = (tp, *others)
        self.groups = [gc for i, gc in enumerate(every) if all(gc is not g for g in every[:i])]
        self.low = min(_nccl_limit(gc, tp) for gc in self.groups) // (hidden * 2) + 1
        _need(self.low <= min(rows, PF_MIN_ROWS), f"stock's custom all-reduce covers {self.low - 1} rows")
    def zc(self, x):
        _need(x.data_ptr() == self.buf.data_ptr() and x.is_contiguous() and x.dtype == torch.bfloat16 and x.dim() == 2
              and x.shape[1] == self.hidden and 0 < x.shape[0] <= self.rows,
              f"zero-copy sum {tuple(x.shape)} {x.dtype} is not the head of the registered buffer")
        out = self.sg.car(self.v2.obj, x, algo=self.sg.TWO_SHOT_PULL, graph_params=self.row)
        _need(out.data_ptr() == x.data_ptr() and tuple(out.shape) == tuple(x.shape), "2shot_pull did not run in place")
        return x

    def plan(self, t, k, zc):
        p = self.plans.get((t, k, zc))
        if p is None:
            q = t // 128
            sizes = [(q // k + (1 if i < q % k else 0)) * 128 for i in range(k)]
            sizes[-1] += t - sum(sizes)
            _need(not torch.cuda.is_current_stream_capturing() and min(sizes) > 256 and t <= self.rows,
                  f"M1 chunks {sizes} of {t} rows (> 256 rows: stock's mhc_post block rule; never made in a capture)")
            ends = [sum(sizes[:i + 1]) for i in range(k)]
            p = self.plans[(t, k, zc)] = [(e - s, e, (self.row.view(torch.int64) + (e - s) * self.hidden * 2).view(
                torch.uint64) if zc else None) for s, e in zip(sizes, ends)]
        return p

    def m1(self, x, resid, h_res, h_post, mode):
        t, h = x.shape
        zc = mode != "n64"
        plan = self.plan(t, (4 if t >= 12288 else 3 if t >= 6144 else 2) if mode == "zc" else 3 if zc else 2, zc)
        y = self.buf[:t] if zc else x
        cur = torch.cuda.current_stream()
        out = torch.empty((t, 4, h), dtype=torch.bfloat16, device=x.device)
        self.side.wait_stream(cur)
        with torch.cuda.stream(self.side):
            if zc:
                self.v2.obj.set_pull_blocks(M1_PULL_BLOCKS)
            for k, (a, b, row) in enumerate(plan):
                if mode == "copy":
                    y[a:b].copy_(x[a:b])
                if zc:
                    self.sg.car(self.v2.obj, y[a:b], algo=self.sg.TWO_SHOT_PULL, graph_params=row)
                else:
                    self.n64(x[a:b])
                self.evs[k].record(self.side)
            if zc:
                self.v2.obj.set_pull_blocks(None)
        res3, comb, post = resid.view(t, 4, h), h_res.view(t, 4, 4), h_post.view(t, 4)
        for k, (a, b, _) in enumerate(plan):
            cur.wait_event(self.evs[k])
            self.sg.kmhc.mhc_post_tilelang(comb[a:b], res3[a:b], post[a:b], y[a:b], out[a:b], 4, h, num_h_blocks=1)
        return out.view(t, 4 * h)

def _build_comms(ar):
    _need(not torch.cuda.is_current_stream_capturing(), "the first EXACT-AR2 call is inside a CUDA-graph capture "
          "(SGLang's prefill graph runner warms every bucket eagerly first, so this engine broke that order)")
    t0 = time.perf_counter()
    class Comm(ar.sg.CustomAllReduceV2):
        def close(self):
            pass
    v2 = Comm(ar.tp.cpu_group, ar.tp.device, max_size=AR_STUB_BYTES, max_pull_size=AR_STUB_BYTES,
              max_push_size=AR_STUB_BYTES)
    _need(not v2.disabled and v2.world_size == 2 and v2.pull_enabled and v2._is_graph_mode_supported
          and v2._graph_counter == 0 and not v2._graph_inputs, "CustomAllReduceV2 is disabled or not graph-capable")
    buf = torch.empty(ar.rows, ar.hidden, dtype=torch.bfloat16, device=ar.tp.device)
    v2._graph_inputs.append((buf.data_ptr(), ar.cap))
    v2._register_graph_inputs()
    _need(v2._graph_counter == 1 and not v2._graph_inputs, "the zero-copy buffer did not get pointer-table row 0")
    ar.v2, ar.buf, ar.row = v2, buf, v2.graph_params[0]
    ar.n64 = Nccl64(ar.sg, ar.tp)
    ar.side, ar.evs = torch.cuda.Stream(device=ar.tp.device), [torch.cuda.Event() for _ in range(4)]
    _self_check(ar)
    _m1_check(ar)
    _log.warning("layerar: exact-ar2 built tp_rank=%d/2 cap_rows=%d low_rows=%d pull_blocks=%d nccl_ctas=%d "
                 "selfcheck=bitwise groups=%s ar%dMiB m1=zc,copy,n64:bitwise m1_pull_blocks=%d build_s=%.2f", v2.rank,
                 ar.rows, ar.low, v2.config.num_pull_blocks, AR_CTAS, [gc.unique_name for gc in ar.groups],
                 ar.cap >> 20, M1_PULL_BLOCKS, time.perf_counter() - t0)

def _self_check(ar):
    dev = ar.tp.device
    gen = torch.Generator(device=dev).manual_seed(SELF_CHECK_SEED)
    x0 = torch.randn(ar.rows, ar.hidden, generator=gen, device=dev, dtype=torch.bfloat16)
    x1 = torch.randn(ar.rows, ar.hidden, generator=gen, device=dev, dtype=torch.bfloat16)
    nudge = torch.randint(-3, 4, x1[0::4].shape, generator=gen, device=dev, dtype=torch.int16)
    x1[0::4] = ((-x0[0::4]).view(torch.int16) + nudge).view(torch.bfloat16)
    gap = torch.exp2(-(torch.arange(x1[1::4].shape[0], device=dev) % 24).float()).to(torch.bfloat16)
    x1[1::4] = x0[1::4] * gap[:, None]
    x1[2::4] = -x0[2::4]
    x = x1 if ar.tp.rank_in_group else x0
    del x0, x1, nudge, gap
    n64 = ar.n64(x.clone())
    ar.buf.copy_(x)
    zc = ar.zc(ar.buf)
    for gc in ar.groups:
        want = gc.all_reduce(x.clone()).view(torch.int16)
        _need(torch.equal(want, n64.view(torch.int16)) and torch.equal(want, zc.view(torch.int16)),
              f"EXACT-AR2 self-check: a bundle sum of the {x.numel() * 2} B probe differs from {gc.unique_name}'s")
        del want

def _m1_check(ar):
    t, h, dev = M1_PROBE_ROWS, ar.hidden, ar.tp.device
    gen = torch.Generator(device=dev).manual_seed(SELF_CHECK_SEED + 1)
    res = torch.randn(t, 4 * h, generator=gen, device=dev).to(torch.bfloat16)
    h_res, h_post = torch.randn(t, 16, generator=gen, device=dev), torch.rand(t, 4, generator=gen, device=dev) * 2
    x = [torch.randn(t, h, generator=gen, device=dev).to(torch.bfloat16) for _ in range(2)][ar.tp.rank_in_group]
    want = ar.sg.kmhc.hc_post(ar.tp.all_reduce(x.clone()), res, h_post, h_res, 4).view(torch.int16)
    for mode in ("zc", "copy", "n64"):
        y = ar.buf[:t].copy_(x) if mode == "zc" else x.clone()
        got = ar.m1(y, res, h_res, h_post, mode)
        _need(torch.equal(got.view(torch.int16), want), f"M1 self-check: {mode} sum + chunked mhc_post of the {t}-row "
              f"probe differs from stock's sum + hc_post")

class ArFfn:
    mega = None
    def __init__(self, stock, ar, sg, group1, limit1, limit2, st, au, path):
        self.stock, self.ar, self.sg, self.group1, self.limit1, self.limit2 = stock, ar, sg, group1, limit1, limit2
        self.st, self.au, self.path = st, au, path
    def _takes(self, x, limit):
        if not torch.is_tensor(x) or x.dim() != 2 or x.numel() * x.element_size() <= limit:
            return False
        _need(x.dtype == torch.bfloat16 and x.shape[1] == self.ar.hidden and x.is_contiguous(),
              f"EXACT-AR2 sum {tuple(x.shape)} {x.dtype} is not a contiguous bf16 [tokens, hidden] tensor")
        return True
    def _m1_state(self, forward_batch, t):
        st, res = self.st, forward_batch.residual_stream.residual
        _need(self.stock.plan.path_for(forward_batch) is self.path and res is not None and res.shape[0] == t
              and res.is_contiguous() and st.h_res is not None and st.h_res.dtype == st.h_post.dtype == torch.float32
              and st.h_res.is_contiguous() and st.h_post.is_contiguous(),
              "M1: the FFN path, the residual or the mHC coefficients are not the ones the boundary was checked for")
        return st, res
    def prepare(self, hidden_states, forward_batch):
        stream = forward_batch.residual_stream
        c = stream.pending
        if c is None or not self._takes(c.value, self.limit1):
            return self.stock.prepare(hidden_states, forward_batch)
        _need(type(hidden_states) is self.sg.OwedOutput and hidden_states.contribution is c and stream.transport is None
              and type(c.owed) is self.sg.DeclaredSum and c.owed.group is self.group1 and c.owed.complete_sum is None,
              "the attention output is not the declared plain sum the FFN entry was checked for")
        x = c.value
        if x.shape[0] < PF_MIN_ROWS:
            c.value, c.owed = self.ar.n64(x), None
            return self.stock.prepare(c.value, forward_batch)
        st, res = self._m1_state(forward_batch, x.shape[0])
        _need(c.update is self.au and x.data_ptr() == self.ar.buf.data_ptr(),
              "M1: the attention update or the zero-copy partial is not the one the FFN entry was checked for")
        norm = self.stock.norm
        if self.mega is not None and not (self.sg.in_graph() or torch.cuda.is_current_stream_capturing()):
            new, x, st.h_res, st.h_post = mega_entry(self.ar, x, res, st.h_res, st.h_post, self.mega, norm)
        else:
            new = self.ar.m1(x, res, st.h_res, st.h_post, "zc")
            x, st.h_res, st.h_post, norm_fused = st.hc_ffn_pre(new, norm.weight.data, norm.variance_epsilon)
            if not norm_fused and x.shape[0] != 0:
                x = norm(x)
        stream.write(new)
        return x
    def finish(self, hidden_states, forward_batch):
        if not self._takes(hidden_states, self.limit2):
            return self.stock.finish(hidden_states, forward_batch)
        t = hidden_states.shape[0]
        zc = ZC_AR2 and hidden_states.data_ptr() == self.ar.buf.data_ptr()
        if t < AR2_N64_MIN_ROWS and not zc:
            return self.stock.complete_now(self.ar.n64(hidden_states), forward_batch, already_reduced=True)
        stream = forward_batch.residual_stream
        st, res = self._m1_state(forward_batch, t)
        _need(stream.pending is None and stream.transport is None, "M1: the FFN output's stream is not written")
        new = fp_m1(self.ar, hidden_states, res, st.h_res, st.h_post,
                    "zc" if zc else "copy" if t >= AR2_COPY_MIN_ROWS else "n64")
        st.clear_coefficients()
        if st.is_last_layer:
            new = self.sg.rmhc.hc_contract(new, st.hc_mult)
        stream.residual = None
        return stream.write(new)

def _boundary_facts(layer, sg):
    lb, one, G = sg.lbp, {sg.BatchVariant.ORDINARY}, sg.SumGroup
    attn, ffn = layer.attn_boundary, layer.ffn_boundary
    _need(type(attn) is sg.StageBoundary and type(ffn) is sg.StageBoundary and ffn.kind is sg.StageKind.FFN
          and set(attn.plan.paths) == one and set(ffn.plan.paths) == one and ffn.plan.fusions is None
          and attn.declaration.row_transport is None and ffn.declaration.row_transport is None,
          "layer boundaries are not plain single-variant stages (variants / row transport / fusions)")
    made = attn.plan.paths[sg.BatchVariant.ORDINARY].output
    _need(attn.plan.finishes_directly and made.always_partial and made.sum_impl is None
          and made.group in (G.ATTN_TP, G.TP), f"attention output contract group={made.group} sum={made.sum_impl}")
    path = ffn.plan.paths[sg.BatchVariant.ORDINARY]
    entry = path.entry
    _need(entry.input_move is None and entry.attn_input_adapter is None and type(entry.prepare) is functools.partial
          and entry.prepare.func is lb._run_entry, "FFN entry is not _run_entry")
    kw = entry.prepare.keywords
    step, done, read = kw.get("step"), kw.get("completed_step"), kw.get("read")
    _need(kw.get("expected_sum") is made.group and kw.get("written_step") is None and not kw.get("carried_fusions")
          and not kw.get("carried_read_fusions") and read is not None and read.bind is None
          and not read.completing_fusions, "FFN entry carries fusions, a bound read or another sum")
    sk = step.keywords if type(step) is functools.partial else {}
    _need(type(step) is functools.partial and step.func is lb._reduce_update_read and sk.get("group") is made.group
          and sk.get("gathers_residual") is False and sk.get("fusions") == () and not sk.get("read_fusions")
          and not sk.get("read_fusion_indices") and sk.get("read") is read,
          "FFN entry's sum step is not the plain all-reduce + update_and_read")
    dk = done.keywords if type(done) is functools.partial else {}
    _need(type(done) is functools.partial and done.func is lb._update_read and dk.get("pre_move") is None
          and dk.get("enters_stack") is False and dk.get("read") is read and not dk.get("read_gathers")
          and not dk.get("read_gather_indices"), "FFN entry's completed-sum step is not the plain update_and_read")
    out = path.output
    _need(out.sum_impl is None and out.group in (G.MOE_OUTPUT, G.ATTN_TP, G.TP) and not out.may_defer_to_next
          and out.update.applied_at_exit and not out.update.quantized_sum and out.update.bind is None
          and not path.output_move_completes_sum and not path.returns_over_dp and not path.writes_at_handoff
          and not path.exit.sum_in_reduce_scatter and path.exit.sum_left_to_next_input is None,
          f"FFN exit (group {out.group}) does not complete one plain sum at this layer")
    st, au, L, R = getattr(read, "state", None), made.update, sg.g5.Glm5NextDecoderLayer, sg.rmhc
    fused = getattr(st, "hc_ffn_post_pre", None)
    _need(type(read) is R._FfnReadout and type(st) is R.MHCState and st.hc_mult == 4 and type(au) is R._AttnUpdate
          and au.state is st and au.bind is None and au.is_plain_add is False and kw.get("is_plain_add") is False
          and type(out.update) is R._FfnUpdate and out.update.state is st and out.transform is None
          and path.output_move is sg.keep_output and ffn.plan._publish_lora_layout is False
          and all(getattr(f, "__self__", None) is layer and getattr(f, "__func__", None) is g
                  for f, g in ((st.hc_post, L.hc_post), (st.hc_ffn_pre, L.hc_ffn_pre)))
          and (fused is None or (getattr(fused, "__self__", None) is layer and getattr(fused, "__func__", None)
                                 is L.hc_ffn_post_pre and sg.g5._MHC_FUSED_BOUNDARY_MAX_TOKENS < PF_MIN_ROWS))
          and sg.g5._hc_post_fn is sg.kmhc.hc_post and sg.kmhc._use_tilelang_mhc_post() and not sg.kmhc._use_aiter_mhc(),
          "M1: the FFN boundary's mHC read / updates, move, LoRA publish or hc_post dispatch are not stock's plain ones")
    return made.group, out.group, st, au, path

def _prepare_ar(layer, ps, tp):
    import torch.distributed as dist
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo, custom_all_reduce
    from sglang.kernels.ops.gemm.fp8_kernel import deep_gemm_fp8_fp8_bf16_nt
    from sglang.kernels.ops.layernorm import mhc as kmhc
    from sglang.srt.distributed.device_communicators import pynccl_wrapper
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.layer_boundary import contracts, layout, ops, prepare as lbp, stage
    from sglang.srt.layers.layer_boundary.residual import mhc as rmhc, stream
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, post_experts_output_is_complete
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.models import glm5_next
    from sglang.srt.runtime_context import get_exec, get_schedule, max_prefill_buffer_tokens
    SumGroup = layout.SumGroup
    sg = types.SimpleNamespace(lbp=lbp, BatchVariant=contracts.BatchVariant, StageKind=contracts.StageKind,
                               SumGroup=SumGroup, DeclaredSum=stream.DeclaredSum, OwedOutput=stream.OwedOutput,
                               StageBoundary=stage.StageBoundary, dist=dist, car=custom_all_reduce, pw=pynccl_wrapper,
                               TWO_SHOT_PULL=AllReduceAlgo.TWO_SHOT_PULL, CustomAllReduceV2=CustomAllReduceV2,
                               fp8_utils=fp8_utils, dgw=deep_gemm_wrapper, dg_nt=deep_gemm_fp8_fp8_bf16_nt,
                               kmhc=kmhc, rmhc=rmhc, g5=glm5_next, keep_output=ops.keep_output)
    if MEGA_PF:
        import deep_gemm
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import configure_deep_gemm_num_sms
        from sglang.srt.model_executor.runner_utils.prefill_graph import is_in_prefill_graph_scope
        sg.dg, sg.dg_sms, sg.in_graph = deep_gemm, configure_deep_gemm_num_sms, is_in_prefill_graph_scope
    group1, group2, st, au, path = _boundary_facts(layer, sg)
    if MEGA_PF:
        _megapf_facts(layer, sg)
    _need(not get_exec().comm.enable_quant_communications, "quantized all-reduces are enabled")
    gcs = {SumGroup.ATTN_TP: ps.get_attn_tp_group(), SumGroup.TP: tp}
    if group2 is SumGroup.MOE_OUTPUT:
        _need(ps.get_moe_ep_group().world_size == 1 and get_moe_a2a_backend().is_none()
              and not post_experts_output_is_complete(is_tp_path=True), "MoE output sum is not one MoE-TP all-reduce")
        gcs[SumGroup.MOE_OUTPUT] = ps.get_moe_tp_group()
    ar = _PROCESS.get("ar")
    if ar is None:
        rows = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
        ar = _PROCESS["ar"] = ExactAr(sg, tp, (ps.get_attn_tp_group(), ps.get_moe_tp_group()),
                                      layer.config.hidden_size, rows)
    _need(ar.hidden == layer.config.hidden_size and ar.tp is tp, "layers disagree on hidden size or TP group")
    ffn = ArFfn(layer.ffn_boundary, ar, sg, group1, _nccl_limit(gcs[group1], tp), _nccl_limit(gcs[group2], tp),
                st, au, path)
    ffn.mega = layer if MEGA_PF else None
    return ffn


def mega_entry(ar, x, resid, h_res, h_post, layer, norm):
    t, h = x.shape
    plan, f32, dev = ar.plan(t, 2, True), torch.float32, x.device
    cur = torch.cuda.current_stream()
    new, li = torch.empty((t, 4, h), dtype=torch.bfloat16, device=dev), torch.empty((t, h), dtype=torch.bfloat16, device=dev)
    post, comb = torch.empty((t, 4, 1), dtype=f32, device=dev), torch.empty((t, 4, 4), dtype=f32, device=dev)
    ao_pull(ar, x, plan, cur)
    res3, comb0, post0, cfg = resid.view(t, 4, h), h_res.view(t, 4, 4), h_post.view(t, 4, 1), layer.config
    with ar.sg.dg_sms(MEGAPF_SMS):
        for k, (a, b, _) in enumerate(plan):
            cur.wait_event(ar.evs[k])
            ar.sg.dg.mega_mhc(x[a:b], res3[a:b], None, post0[a:b], comb0[a:b], layer.hc_ffn_fn, layer.hc_ffn_scale,
                              layer.hc_ffn_base, 4, cfg.rms_norm_eps, cfg.hc_eps, 2.0, cfg.hc_eps,
                              cfg.hc_sinkhorn_iters, norm.weight.data, norm.variance_epsilon, 1.0, new[a:b], None,
                              post[a:b], comb[a:b], y_bf16=li[a:b])
    return new.view(t, 4 * h), li, comb.view(t, 16), post.view(t, 4)

def _megapf_check(ar, ffn):
    t, h, dev, st, norm = MEGAPF_PROBE_ROWS, ar.hidden, ar.tp.device, ffn.st, ffn.stock.norm
    _need(torch.cuda.get_device_capability(dev)[0] == 10
          and torch.cuda.get_device_properties(dev).multi_processor_count >= MEGAPF_SMS,
          f"M3: mega_mhc is an sm_10x kernel run on {MEGAPF_SMS} SMs")
    t0 = time.perf_counter()
    gen = torch.Generator(device=dev).manual_seed(SELF_CHECK_SEED + 2)
    res = (torch.randn(t, 4 * h, generator=gen, device=dev) * 0.7).to(torch.bfloat16)
    _, h_res, h_post, _ = st.hc_attn_pre(res, norm.weight.data, norm.variance_epsilon)
    x = [(torch.randn(t, h, generator=gen, device=dev) * 0.5).to(torch.bfloat16) for _ in range(2)][ar.tp.rank_in_group]
    new = ar.sg.kmhc.hc_post(ar.tp.all_reduce(x.clone()), res, h_post, h_res, 4)
    li, comb, post, fused = st.hc_ffn_pre(new, norm.weight.data, norm.variance_epsilon)
    want = (new, li if fused else norm(li), comb, post)
    got = mega_entry(ar, ar.buf[:t].copy_(x), res, h_res, h_post, ffn.mega, norm)
    rel = [round(((a.float() - b.float()).norm() / b.float().norm()).item(), 7) for a, b in zip(got, want)]
    shapes = all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(got, want))
    del want, new, li
    again = mega_entry(ar, ar.buf[:t].copy_(x), res, h_res, h_post, ffn.mega, norm)
    ok = shapes and all(torch.equal(a, b) for a, b in zip(got, again)) and rel[0] < 1e-3 and max(rel[1:]) < 1e-2
    del got, again
    _need(all(ar.tp.all_gather_object(ok)), f"M3 self-check: mega_mhc vs stock's sum + hc_post + hc_ffn_pre on some TP "
          f"rank: shapes / determinism / relative errors (residual, layer input, h_res, h_post) {rel}, bounds 1e-3 / 1e-2")
    _log.warning("mega-pf: probe %d rows on %d SMs: mega_mhc == stock's sum + hc_post + hc_ffn_pre within audit class, "
                 "deterministic; rel (residual, layer input, h_res, h_post) = %s; %.2f s (MEGA_PF=%s)", t, MEGAPF_SMS, rel,
                 time.perf_counter() - t0, MEGA_PF)
    return rel

def _megapf_facts(layer, sg):
    cfg, norm, k, dg = layer.config, layer.ffn_boundary.norm, sg.kmhc, sg.dg
    h, w, fn, sc, bs = cfg.hidden_size, norm.weight, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base
    _need(norm is layer.post_attention_layernorm and sg.g5._hc_pre_fn is k.hc_pre and sg.g5._MHC_POST_MULT_VALUE == 2.0
          and k._use_deep_gemm_hc_prenorm() and k._use_tilelang_mhc_pre() and not k._use_aiter_mhc()
          and getattr(dg, "__version__", None) == "0.2.7" and callable(getattr(dg, "mega_mhc", None))
          and h % 1024 == 0 and int(cfg.hc_sinkhorn_iters) >= 1 and fn.dtype == sc.dtype == bs.dtype == torch.float32
          and tuple(fn.shape) == (24, 4 * h) and tuple(sc.shape) == (3,) and tuple(bs.shape) == (24,)
          and fn.is_contiguous() and sc.is_contiguous() and bs.is_contiguous() and w.dtype == torch.bfloat16
          and tuple(w.shape) == (h,) and w.is_contiguous(),
          "M3: the FFN-entry pre is not stock's DeepGEMM-prenorm TileLang hc_pre with a fused bf16 RMSNorm, or "
          "DeepGEMM 0.2.7's mega_mhc does not take this layer's tensors")

_MOE = {}

def _moe_facts(module, ns, ffn):
    from sglang.srt.layers.moe import utils as mu
    from sglang.srt.layers.moe.gluon_backend import should_use_gluon_moe
    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
    from sglang.srt.layers.moe.mega_moe import should_use_mega_moe
    from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mx
    from sglang.srt.layers.quantization.expert_pack import ExpertPackMoEMethod
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import Mxfp4FlashinferCutlassMoEMethod
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
    from sglang.srt.model_executor.runner_utils import prefill_graph as pg
    from sglang.srt.models import deepseek_v2 as dsv2
    cls, qm = type(module), module.experts.quant_method
    fused = (mx.Mxfp4FlashinferTrtllmMoEMethod, Mxfp4FlashinferCutlassMoEMethod, Mxfp4MarlinMoEMethod, ExpertPackMoEMethod)
    _need(cls is dsv2.DeepseekV2MoE and dsv2.maybe_fuse_routed_scale_and_shared_add is mx.maybe_fuse_routed_scale_and_shared_add
          and module.shared_experts is not None and module.num_fused_shared_experts == 0
          and not module._enable_a2a_moe and not module._fuse_shared_experts_inside_sbo
          and not module._shared_expert_tp1 and not module.reduce_results and not getattr(module, "is_hash", False)
          and getattr(module.gate, "e_score_correction_bias_vl", None) is None
          and not module.experts.moe_runner_config.inplace and not isinstance(qm, fused + (KTEPWrapperMethod,))
          and mu.get_moe_runner_backend().is_flashinfer_trtllm() and dsv2._is_cuda and not dsv2._is_hip,
          "moe-side: the eager MoE call is not stock's deferred-shared forward_normal with a plain routed += shared")
    ns.x = types.SimpleNamespace(ar=ffn.ar, tp=ffn.ar.tp, in_graph=pg.is_in_prefill_graph_scope,
                                 gluon=should_use_gluon_moe, mega=should_use_mega_moe, capture=dsv2.get_is_capture_mode)

def moe_x_ok(state, hidden, args, kwargs):
    x, px = state.x, state.proxy
    flags = ("skip_shared_experts", "return_moe_output", "leave_output_sum")
    if not (len(args) <= 3 and not any(kwargs.get(f) for f in flags)
            and set(kwargs) <= {"hidden_states", "forward_batch", "gemm_output_zero_allocator", *flags}
            and hidden.dim() == 2 and hidden.shape[0] >= PF_MIN_ROWS and hidden.dtype == torch.bfloat16
            and not x.capture() and not x.gluon(px) and not x.mega(px, hidden) and not px._can_dual_stream_graph(hidden)
            and not px._moe_quant_once_enabled()):
        return False
    return ZC_AR2 or not (x.in_graph() or torch.cuda.is_current_stream_capturing())

def moe_x_forward(state, hidden, gza):
    px, x = state.proxy, state.x
    eager = not (x.in_graph() or torch.cuda.is_current_stream_capturing())
    side = None
    cur = torch.cuda.current_stream()
    if side is not None:
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            shared = px._forward_shared_experts(hidden, gza)
        hidden.record_stream(side)
    router_logits = px.gate(hidden, gza)
    topk_output = px.topk(hidden, router_logits)
    routed = px.experts(hidden, topk_output)
    if side is None:
        shared = px._forward_shared_experts(hidden, gza)
    else:
        cur.wait_stream(side)
        shared.record_stream(cur)
    t = hidden.shape[0]
    if ZC_AR2 and t <= x.ar.rows:
        _need(routed.shape == shared.shape == (t, x.ar.hidden) and routed.dtype == shared.dtype == torch.bfloat16,
              f"zc-ar2: routed {tuple(routed.shape)} / shared {tuple(shared.shape)} are not [t, hidden] bf16")
        return torch.add(routed, shared, out=x.ar.buf[:t])
    routed += shared
    return routed

def moe_x_variant(state):
    return "plain"

def moe_x_probe(state, args, kwargs, hidden, gza):
    variant = moe_x_variant(state)
    _need(not torch.cuda.is_current_stream_capturing(), f"moe-side: the {variant} route's first call is inside a capture")
    t0 = time.perf_counter()
    want = state.stock_forward(state.proxy, *args, **kwargs).clone()
    got = moe_x_forward(state, hidden, gza)
    same = got.shape == want.shape and got.dtype == want.dtype and torch.equal(got.view(torch.int16), want.view(torch.int16))
    del want
    _need(all(state.x.tp.all_gather_object(same)), f"moe-side: the {variant} MoE forward (side stream / zero-copy sum) "
          f"differs from stock's forward_normal on the probe call on some TP rank")
    _MOE.setdefault("proved", set()).add(variant)
    _log.warning("moe-side: probe %d rows, route %s: shared expert on a side stream %s, sum into the registered buffer "
                 "(ZC_AR2=%s) == stock's forward_normal, bitwise; %.2f s", hidden.shape[0], variant,
                 variant == "side", ZC_AR2, time.perf_counter() - t0)
    return got


def idx_logits(dg, q, kv, weights, lens, table, max_len):
    sched = ph_sched(dg, lens)
    return dg.fp8_paged_mqa_logits(q, kv, weights, lens, table, sched, max_len, clean_logits=False)

def idx_topk_paged(sh, forward_batch, layer_id, q_fp8, weights, metadata):
    st, sg = sh._s, sh._sg
    plan = metadata.attn_metadata.kpool_write_plan
    seqlens_32 = metadata.get_seqlens_expanded()
    n, rows = 0 if plan is None or plan.pool_seqlens_per_q is None else plan.num_draft_tokens, seqlens_32.shape[0]
    if n < 2 or rows % n or q_fp8.dim() != 3 or not 0 < rows <= q_fp8.shape[0]:
        return st._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)
    pool = sg.ik.get_token_to_kv_pool()
    _need(pool.index_page_size == 64, "idx-nextn: the index-K pool page is not 64 tokens")
    block_tables = metadata.get_page_table_64()
    kv = st._get_index_k_read_buffer(pool, layer_id)
    q_in, w_in, num_q_padded = q_fp8, weights, q_fp8.shape[0]
    if rows < num_q_padded:
        q_fp8, weights = q_fp8[:rows], weights[:rows]
    _need(kv.dim() == 2 and weights.dim() == 3, "idx-nextn: index-K buffer / head weights layout")
    kv = kv.view(kv.shape[0], 64, 1, 132)
    weights = weights.squeeze(2)
    _need(not st.paged_mqa_logits_backend.is_aiter() and not st._should_use_tilelang_paged_mqa_logits(q_fp8),
          "idx-nextn: the stock indexer's paged-MQA logits backend is not DeepGEMM")
    lens, _, table, _, v2plan = st._get_kpool_decode_metadata(metadata=metadata, block_tables=block_tables,
                                                              seqlens_32=seqlens_32, blocksize=64,
                                                              build_schedule_metadata=False)
    b, heads, dim = rows // n, q_fp8.shape[1], q_fp8.shape[2]
    _need(lens.dtype == torch.int32 and tuple(lens.shape) == (rows,) and lens.is_contiguous()
          and lens.data_ptr() == plan.pool_seqlens_per_q.data_ptr() and table.dtype == torch.int32 and table.dim() == 2
          and table.shape[0] == rows and table.stride(1) == 1 and q_fp8.dtype == _FP8 and q_fp8.is_contiguous()
          and weights.dtype == torch.float32 and tuple(weights.shape) == (rows, heads) and weights.stride(1) == 1,
          f"idx-nextn: verify call lens {lens.dtype} {tuple(lens.shape)} table {table.dtype} {tuple(table.shape)} "
          f"q {q_fp8.dtype} {tuple(q_fp8.shape)} weights {weights.dtype} {tuple(weights.shape)} is not the plan's layout")
    dg, capturing = sg.ik.deep_gemm, torch.cuda.is_current_stream_capturing()
    warm = _PROCESS.setdefault("idx_warm", set())
    key = (n, heads, dim, dg.get_num_sms())
    _need(not capturing or key in warm, f"idx-nextn: the first next_n={n} paged-MQA call is inside a CUDA-graph capture "
          "(DeepGEMM would JIT there; SGLang's runner warms every graph eagerly first)")
    page_table_1, topk_offsets, _ = st._kpool_fused_topk_mapping(metadata)
    tp = getattr(sg, "tp", None)
    shard = rs_shard_ok(tp, b, rows)
    ph_begin(forward_batch, layer_id, lens, tp)
    if shard:
        if "rs_probe" not in _PROCESS:
            _need(not capturing, "idx-rowshard: the first sharded call is inside a CUDA-graph capture (its boot probe, "
                  "an eager all-gather, runs at the first eager one; SGLang's runner warms every graph eagerly first)")
            _PROCESS["rs_probe"] = _rs_probe(sh, tp, n, heads, dim, v2plan is not None)
        out = idx_rowshard(sh, tp, q_fp8.view(b, n, heads, dim), kv, weights, lens, table[::n], table.shape[1] * 64,
                           seqlens_32, page_table_1, topk_offsets, v2plan is not None, num_q_padded,
                           fb=forward_batch)
    else:
        logits = idx_logits(dg, q_fp8.view(b, n, heads, dim), kv, weights, lens.view(b, n), table[::n],
                            table.shape[1] * 64)
        out = st._topk_from_kpool_logits(logits=logits, pool_lens=lens, topk_v2_plan=v2plan, seq_lens=seqlens_32,
                                         page_table=page_table_1, topk_offsets=topk_offsets,
                                         out_rows=num_q_padded if num_q_padded != rows else None)
        del logits
    ph_end()
    if not capturing:
        warm.add(key)
        checked = _PROCESS.setdefault("idx_checked", set())
        ck = (b, n, table.shape[1]) + (("rowshard",) if shard else ())
        if ck not in checked:
            want = st._get_topk_paged(forward_batch, layer_id, q_in, w_in, metadata)
            same = want.dtype == out.dtype and want.shape == out.shape and torch.equal(want.sort(1).values, out.sort(1).values)
            _need(all(tp.all_gather_object(same)) if tp is not None else same,
                  f"idx-nextn: layer {layer_id} rows={rows} rowshard={shard}: the packed top-k differs from stock's (row sets)")
            checked.add(ck)
            _log.warning("idx-nextn: layer %d rows=%d B=%d rowshard=%s packed top-k == stock's as row sets (ordered %s)",
                         layer_id, rows, b, shard, torch.equal(want, out))
    return out

def idx_shard(st, sg, q, kv, w, lens, table, max_len, seq, pt, offs, plan, a, e, out_rows):
    n = q.shape[1]
    r0, r1 = a * n, e * n
    p = ph_plan(sg.kif, lens[r0:r1]) if plan else None
    logits = idx_logits(sg.ik.deep_gemm, q[a:e], kv, w[r0:r1], lens[r0:r1].view(e - a, n), table[a:e], max_len)
    return st._topk_from_kpool_logits(logits=logits, pool_lens=lens[r0:r1], topk_v2_plan=p, seq_lens=seq[r0:r1],
                                      page_table=None if pt is None else pt[r0:r1],
                                      topk_offsets=None if offs is None else offs[r0:r1], out_rows=out_rows)

def idx_rowshard(sh, tp, q, kv, w, lens, table, max_len, seq, pt, offs, plan, num_q_padded, fb=None):
    b, n, W = q.shape[0], q.shape[1], tp.world_size
    bq = -(-b // W)
    a, e = min(b, tp.rank_in_group * bq), min(b, (tp.rank_in_group + 1) * bq)
    send = idx_shard(sh._s, sh._sg, q, kv, w, lens, table, max_len, seq, pt, offs, plan, a, e, bq * n)
    return rs_gather(tp, send, sh._s.layer_id, bq * n, b * n, num_q_padded, fb)

def _rs_probe(sh, tp, n, heads, dim, plan):
    st, sg, dev = sh._s, sh._sg, tp.device
    gen = torch.Generator(device=dev).manual_seed(PROBE_SEED + 7)
    b, f8 = RS_PROBE_BS, _FP8
    seq = (torch.randint(*RS_PROBE_CTX, (b, 1), generator=gen, device=dev) + torch.arange(1, n + 1, device=dev)).int()
    lens = (seq // st.index_kpool).int().reshape(-1)
    pages = int(lens.max()) // 64 + 1
    kv = torch.empty(pages, 64 * (dim + 4), dtype=torch.uint8, device=dev)
    kv[:, :64 * dim] = (torch.randn(pages, 64 * dim, generator=gen, device=dev) * 8).to(f8).view(torch.uint8)
    kv[:, 64 * dim:] = (torch.rand(pages, 64, generator=gen, device=dev) * 0.05 + 0.005).view(torch.uint8)
    q = (torch.randn(b, n, heads, dim, generator=gen, device=dev) * 8).to(f8)
    w = torch.randn(b * n, heads, generator=gen, device=dev)
    args = (q, kv.view(pages, 64, 1, dim + 4), w, lens, torch.arange(pages, dtype=torch.int32, device=dev).expand(
        b, pages).contiguous(), pages * 64, seq.reshape(-1), None, None, plan)
    full, again = idx_shard(st, sg, *args, 0, b, None), idx_shard(st, sg, *args, 0, b, None)
    got = idx_rowshard(sh, tp, *args, b * n)
    same = got.shape == full.shape and torch.equal(got.sort(1).values, full.sort(1).values)
    _need(all(tp.all_gather_object(same)), f"idx-rowshard probe: W={tp.world_size} gathered top-k of {b}x{n} rows != the "
          f"full batch's row sets on some rank (raise RS_MIN_BS or set ROWSHARD = False)")
    _log.warning("idx-rowshard: probe B=%d rank=%d: gathered == full top-k as row sets (ordered %s, full rerun %s)", b,
                 tp.rank_in_group, torch.equal(got, full), torch.equal(again, full))
    return True

class _IdxShim:
    alt_stream = property(lambda self: self._s.alt_stream)
    compress_gate_stream = property(lambda self: self._s.compress_gate_stream)
    index_topk = property(lambda self: self._s.index_topk)
    index_kpool = property(lambda self: self._s.index_kpool)
    index_kpool_compress_ape = property(lambda self: self._s.index_kpool_compress_ape)
    scale_fmt = property(lambda self: self._s.scale_fmt)
    block_size = property(lambda self: self._s.block_size)
    def _get_q_k_bf16(self, q_lora, *args, **kwargs):
        return self._s._get_q_k_bf16(rsq_rows(self, q_lora), *args, **kwargs)
    _compute_gate_score_if_missing = property(lambda self: self._s._compute_gate_score_if_missing)
    _rs = None
    def _resolve_head_gate_weights(self, x, q_scale, head_weights):
        return rsq_gates(self, x, q_scale, head_weights)
    def __init__(self, stock, sg):
        self._s, self._sg = stock, sg
    def _forward_cuda_target_verify(self, *args, **kwargs):
        self._rs = rsq_plan(self, args, kwargs)
        try:
            return self._sg.Indexer._forward_cuda_target_verify(self, *args, **kwargs)
        finally:
            self._rs = None
    def _get_topk_paged(self, forward_batch, layer_id, q_fp8, weights, metadata):
        fn = idx_topk_paged if self._rs is None else rsq_topk_paged
        return fn(self, forward_batch, layer_id, q_fp8, weights, metadata)

class IdxNN(torch.nn.Module):
    def __init__(self, stock, sg):
        super().__init__()
        object.__setattr__(self, "stock", stock)
        object.__setattr__(self, "shim", _IdxShim(stock, sg))
        object.__setattr__(self, "sg", sg)
    def forward(self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
        st, sg = self.stock, self.sg
        _need(forward_batch.forward_mode.is_target_verify(), "idx-nextn: a non-verify call reached the verify proxy")
        method = st._forward_method
        if method is None:
            return st(x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch, layer_id=layer_id,
                      return_indices=return_indices)
        _need(getattr(method, "__func__", None) is sg.Indexer.forward_cuda and method.__self__ is st
              and not st.is_torch_compile and sg.fused_op.get_fused_op_backend() is None and not st._forward_hooks
              and not st._forward_pre_hooks, "idx-nextn: the stock indexer does not dispatch to IndexerKPool.forward_cuda "
              "(forced backend, torch.compile mode, a rebound dispatch or module hooks)")
        return sg.Indexer._forward_cuda_impl(self.shim, x, q_lora, positions, forward_batch, layer_id, return_indices)

def _idx_prepare(layer, tp=None, role="verify"):
    attn = layer._modules.get("self_attn")
    idx = getattr(attn, "_modules", {}).get("indexer")
    ext = role == "small"
    has_mqa = FP8PACK and getattr(attn, "_modules", {}).get("attn_mqa") is not None
    pk = pk_attn(attn, layer.layer_id, role) if has_mqa else None
    g3_register(attn, pk, layer.layer_id, role, tp)
    if idx is None or (ext and not ROTQ):
        return None if pk is None else types.SimpleNamespace(
            nn=None, proxy=pj_proxy(layer, {"self_attn": pj_proxy(attn, {"attn_mqa": pk})}))
    from sglang.kernels import fused_op
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as ik
    _need(type(idx) is ik.IndexerKPool and idx.index_kpool > 1 and idx.layer_id == layer.layer_id,
          f"layer {layer.layer_id}: the DSA indexer is {type(idx).__name__}, not this layer's IndexerKPool")
    sg = types.SimpleNamespace(ik=ik, Indexer=ik.IndexerKPool, fused_op=fused_op, tp=tp)
    if ROTQ:
        from sglang.kernels.ops.quantization import act_quant_triton
        from sglang.srt.layers.attention.dsa import dsa_indexer, kpool_prefill_cuda_graph
        sg.di, sg.kpg, sg.rotate = dsa_indexer, kpool_prefill_cuda_graph, ik.rotate_activation
        sg.act_quant = act_quant_triton.act_quant
        rq_facts(idx, sg)
    wtc_facts(idx)
    if tp is not None:
        from sglang.srt.layers.attention.dsa import kpool_fp8_index as kif
        _need(tp.world_size > 1 and callable(kif.build_kpool_topk_v2_plan), f"idx-rowshard: TP world {tp.world_size}")
        sg.kif = kif
        ags_register(idx, layer.layer_id)
    if not ext:
        _PH["ids"].add(layer.layer_id)
    nn = RqNN(idx, sg, ext) if ROTQ else IdxNN(idx, sg)
    swaps = {"indexer": nn} if pk is None else {"indexer": nn, "attn_mqa": pk}
    return types.SimpleNamespace(nn=nn, proxy=pj_proxy(layer, {"self_attn": pj_proxy(attn, swaps)}))


@triton.jit
def _post_coef(a_ptr, c_ptr, t, o: tl.constexpr):
    return (tl.load(c_ptr + t * 4 + o), tl.load(a_ptr + t * 16 + 0 + o), tl.load(a_ptr + t * 16 + 4 + o),
            tl.load(a_ptr + t * 16 + 8 + o), tl.load(a_ptr + t * 16 + 12 + o))

@triton.jit
def _post_out(x_ptr, t, o: tl.constexpr, H: tl.constexpr, offs, c, a0, a1, a2, a3, d, b0, b1, b2, b3):
    acc = tl.fma(c, d, a0 * b0)
    acc = tl.fma(a1, b1, acc)
    acc = tl.fma(a2, b2, acc)
    acc = tl.fma(a3, b3, acc)
    tl.store(x_ptr + (t * 4 + o) * H + offs, acc.to(tl.bfloat16))

@triton.jit
def _post_fast_kernel(a_ptr, b_ptr, c_ptr, d_ptr, x_ptr, H: tl.constexpr, BH: tl.constexpr, PDL: tl.constexpr = False):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    if PDL:
        tl.extra.cuda.gdc_wait()
    d = tl.load(d_ptr + t * H + offs).to(tl.float32)
    b0 = tl.load(b_ptr + (t * 4 + 0) * H + offs).to(tl.float32)
    b1 = tl.load(b_ptr + (t * 4 + 1) * H + offs).to(tl.float32)
    b2 = tl.load(b_ptr + (t * 4 + 2) * H + offs).to(tl.float32)
    b3 = tl.load(b_ptr + (t * 4 + 3) * H + offs).to(tl.float32)
    c0, a00, a10, a20, a30 = _post_coef(a_ptr, c_ptr, t, 0)
    c1, a01, a11, a21, a31 = _post_coef(a_ptr, c_ptr, t, 1)
    c2, a02, a12, a22, a32 = _post_coef(a_ptr, c_ptr, t, 2)
    c3, a03, a13, a23, a33 = _post_coef(a_ptr, c_ptr, t, 3)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    _post_out(x_ptr, t, 0, H, offs, c0, a00, a10, a20, a30, d, b0, b1, b2, b3)
    _post_out(x_ptr, t, 1, H, offs, c1, a01, a11, a21, a31, d, b0, b1, b2, b3)
    _post_out(x_ptr, t, 2, H, offs, c2, a02, a12, a22, a32, d, b0, b1, b2, b3)
    _post_out(x_ptr, t, 3, H, offs, c3, a03, a13, a23, a33, d, b0, b1, b2, b3)

def fast_post(y, res, h_res, h_post):
    t, h = y.shape
    out = torch.empty((t, 4 * h), dtype=torch.bfloat16, device=y.device)
    _need(h % FP_BH == 0 and y.dtype == res.dtype == torch.bfloat16 and h_res.dtype == h_post.dtype == torch.float32
          and res.numel() == 4 * t * h and h_res.numel() == 16 * t and h_post.numel() == 4 * t
          and all(x.is_contiguous() and x.data_ptr() % 16 == 0 for x in (y, res, h_res, h_post, out)),
          f"fastpost: post of {tuple(y.shape)} {y.dtype} onto {tuple(res.shape)} is not the dense layout it serves")
    _post_fast_kernel[(t, h // FP_BH)](h_res, res, h_post, y, out, H=h, BH=FP_BH, num_warps=FP_WARPS,
                                       **_pdl("fastpost"))
    return out

def _dec_check(ffn, dev, h):
    gen, st, norm, bf = torch.Generator(device=dev).manual_seed(SELF_CHECK_SEED + 3), ffn.st, ffn.stock.norm, torch.bfloat16
    rel, warm = {}, set(_PROCESS.setdefault("mega_warm", set()))
    for t, kind in ((FP_PROBE_ROWS, 0), (FP_PROBE_ROWS, 1), (FP_PROBE_ROWS, 2), (192, 1)) if FASTPOST else ():
        rn = lambda *shape, k=1.0: torch.randn(*shape, generator=gen, device=dev) * k
        if kind == 1:
            res = (rn(t, 1, h, k=64.0) * torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev).view(1, 4, 1) + rn(t, 4, h))
            h_res = rn(t, 4, 4, k=8.0)
            h_res[:, 1] = -h_res[:, 0] * (1 + 1e-3 * rn(t, 4))
            h_post, y = torch.rand(t, 4, generator=gen, device=dev) * 4 - 2, rn(t, h, k=32.0).to(bf)
        elif kind == 2:
            y, h_post, h_res = rn(t, h, k=32.0).to(bf), torch.rand(t, 4, generator=gen, device=dev) + 1, rn(t, 4, 4)
            res, h_res[:, 0] = torch.cat([y.float()[:, None], rn(t, 3, h, k=1e-6)], 1), -h_post * (1 + 1e-6 * rn(t, 4))
        else:
            res, h_res, h_post, y = rn(t, 4 * h), rn(t, 16), torch.rand(t, 4, generator=gen, device=dev) * 2, rn(t, h).to(bf)
        res, h_res = res.to(bf).reshape(t, 4 * h).contiguous(), h_res.reshape(t, 16).contiguous()
        want = ffn.sg.kmhc.hc_post(y, res, h_post, h_res, 4)
        _need(torch.equal(fast_post(y, res, h_res, h_post).view(torch.int16), want.view(torch.int16)),
              f"fastpost self-check: the {t}-row {('random', 'cancellation', 'first-FMA')[kind]} probe differs from hc_post")
    _PROCESS["mega_warm"] = warm
    _log.warning("decpack: boot probes passed fastpost=%s (bh=%d warps=%d) mega=%s (neq, fro, max_rel)=%s zero_pad=%s",
                 FASTPOST, FP_BH, FP_WARPS, MEGA_DEC, rel, MEGA_ZERO_PAD)
    return True

class DecFfn:
    def __init__(self, stock, sg, group1, group2, st, au, path, layer):
        self.stock, self.sg, self.group1, self.group2 = stock, sg, group1, group2
        self.st, self.au, self.path, self.layer = st, au, path, layer
    def _state(self, forward_batch, t):
        st, res = self.st, forward_batch.residual_stream.residual
        _need(DEC_MIN_ROWS < t <= DEC_MAX_ROWS and forward_batch.forward_mode.is_target_verify()
              and self.stock.plan.path_for(forward_batch) is self.path and res is not None and res.shape[0] == t
              and res.is_contiguous() and st.h_res is not None and st.h_res.dtype == st.h_post.dtype == torch.float32
              and st.h_res.is_contiguous() and st.h_post.is_contiguous(),
              "decpack: the call, FFN path, residual or mHC coefficients are not the ones the boundary was checked for")
        if "dec_checked" not in _PROCESS:
            _need(not torch.cuda.is_current_stream_capturing(), "decpack: the first decode-proxy call is inside a "
                  "CUDA-graph capture (its boot probes JIT; SGLang's runner warms every graph eagerly first)")
            _PROCESS["dec_checked"] = _dec_check(self, res.device, res.shape[1] // 4)
        return st, res
    def prepare(self, hidden_states, forward_batch):
        stream = forward_batch.residual_stream
        c = stream.pending
        _need(c is not None and type(hidden_states) is self.sg.OwedOutput and hidden_states.contribution is c
              and stream.transport is None and type(c.owed) is self.sg.DeclaredSum and c.owed.group is self.group1
              and c.owed.complete_sum is None and c.update is self.au,
              "decpack: the attention output is not the declared plain sum the FFN entry was checked for")
        st, res = self._state(forward_batch, c.value.shape[0])
        lbp, G = self.sg.lbp, self.sg.SumGroup
        x = (lbp.attn_tp_all_reduce(c.value, forward_batch, may_quantize=self.au.quantized_sum)
             if self.group1 is G.ATTN_TP else lbp.tensor_model_parallel_all_reduce(c.value))
        norm = self.stock.norm
        new = fast_post(x, res, st.h_res, st.h_post) if FASTPOST else st.hc_post(x, res, st.h_res, st.h_post)
        x, st.h_res, st.h_post, norm_fused = st.hc_ffn_pre(new, norm.weight.data, norm.variance_epsilon)
        if not norm_fused and x.shape[0] != 0:
            x = norm(x)
        stream.write(new)
        return x
    def finish(self, hidden_states, forward_batch):
        if not FASTPOST:
            return self.stock.finish(hidden_states, forward_batch)
        stream = forward_batch.residual_stream
        _need(torch.is_tensor(hidden_states) and stream.pending is None and stream.transport is None,
              "decpack: the FFN output is not a complete-able tensor on a written stream")
        st, res = self._state(forward_batch, hidden_states.shape[0])
        y = self.sg.lbe.sum_output(hidden_states, self.group2, forward_batch, may_quantize=False)
        new = fast_post(y, res, st.h_res, st.h_post)
        st.clear_coefficients()
        if st.is_last_layer:
            new = self.sg.rmhc.hc_contract(new, st.hc_mult)
        stream.residual = None
        return stream.write(new)

def _dec_prepare(layer, idx):
    import deep_gemm
    from sglang.kernels.ops.layernorm import mhc as kmhc
    from sglang.srt.layers.layer_boundary import contracts, exit as lbe, layout, ops, prepare as lbp, stage
    from sglang.srt.layers.layer_boundary.residual import mhc as rmhc, stream
    from sglang.srt.models import glm5_next
    sg = types.SimpleNamespace(lbp=lbp, lbe=lbe, BatchVariant=contracts.BatchVariant, StageKind=contracts.StageKind,
                               SumGroup=layout.SumGroup, DeclaredSum=stream.DeclaredSum, OwedOutput=stream.OwedOutput,
                               StageBoundary=stage.StageBoundary, kmhc=kmhc, rmhc=rmhc, g5=glm5_next, dg=deep_gemm,
                               keep_output=ops.keep_output)
    group1, group2, st, au, path = _boundary_facts(layer, sg)
    _need(glm5_next._MHC_FUSED_BOUNDARY_MAX_TOKENS <= DEC_MIN_ROWS < MEGA_MIN_ROWS and lbe.sum_output is ops.sum_output
          and group1 in (layout.SumGroup.ATTN_TP, layout.SumGroup.TP) and au.quantized_sum is False
          and path.output.update.quantized_sum is False and lbp.attn_tp_all_reduce is ops.attn_tp_all_reduce
          and callable(lbp.tensor_model_parallel_all_reduce) and layer.config.hidden_size % FP_BH == 0,
          "decpack: the FFN boundary's sums or its unfused-entry row range are not the ones DecFfn replicates")
    ffn = DecFfn(layer.ffn_boundary, sg, group1, group2, st, au, path, layer)
    proxy = pj_proxy(layer, {"self_attn": idx.proxy._modules["self_attn"]} if idx is not None else {})
    object.__setattr__(proxy, "ffn_boundary", ffn)
    _need(layer.ffn_boundary is not ffn and proxy.attn_boundary is layer.attn_boundary, "decode proxy boundaries")
    return types.SimpleNamespace(ffn=ffn, proxy=proxy)

_K2 = {}

@triton.jit()
def _conv_state_update_w4(x_ptr, conv_states_ptr, x_base, conv_states_base, conv_state_batch_coord,
                          sequence_start_index, seqlen, idx_feats, load_init_state,
                          dim: tl.constexpr, num_cache_lines: tl.constexpr,
                          stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
                          stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr,
                          stride_conv_state_tok: tl.constexpr):
    state_len = 3
    if state_len <= seqlen:
        idx_tokens_last = (seqlen - state_len) + tl.arange(0, 4)
        x_ptrs = (x_ptr + ((sequence_start_index + idx_tokens_last).to(tl.int64) * stride_x_token)[:, None]
                  + (idx_feats * stride_x_dim)[None, :])
        mask_x = ((idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None]
                  & (idx_feats < dim)[None, :])
        new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
        idx_tokens_conv = tl.arange(0, 4)
        conv_states_ptrs_target = (conv_states_base[None, :]
                                   + (idx_tokens_conv * stride_conv_state_tok)[:, None])
        mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.debug_barrier()
        tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        if load_init_state:
            idx_tokens_conv = tl.arange(0, 4)
            conv_states_ptrs_source = (conv_states_ptr + (conv_state_batch_coord * stride_conv_state_seq)
                                       + (idx_feats * stride_conv_state_dim)[None, :]
                                       + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None])
            mask = ((conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None] & (idx_feats < dim)[None, :])
            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None]
                      & (idx_feats < dim)[None, :])
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            tl.debug_barrier()
            new_conv_state = tl.where(mask, conv_state, loaded_x)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            idx_tokens_conv = tl.arange(0, 4)
            VAL = state_len - seqlen
            x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
            mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None]
                      & (idx_feats < dim)[None, :])
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)


@triton.jit()
def _tap4_silu(acc0, a, b, c, d, w0, w1, w2, w3):
    acc = acc0
    acc += a * w0
    acc += b * w1
    acc += c * w2
    acc += d * w3
    return acc / (1 + tl.exp(-acc))


@triton.jit()
def _conv_w4_kernel(
    x_ptr, w_ptr, bias_ptr, conv_states_ptr, cache_indices_ptr, has_initial_states_ptr, query_start_loc_ptr, o_ptr,
    dim: tl.constexpr, total_tokens: tl.int32, num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr, stride_istate_dim: tl.constexpr, stride_istate_token: tl.constexpr,
    stride_o_dim: tl.constexpr, stride_o_token: tl.constexpr, pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr, OUTPUT_SPLIT_SIZE: tl.constexpr,
):
    idx_feats = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    chunk_offset = tl.program_id(1)
    idx_seq = tl.program_id(2)
    if idx_seq == pad_slot_id:
        return
    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = 32 * chunk_offset
    seg = min(32, seqlen - token_offset)
    if seg <= 0:
        return
    x_base = x_ptr + sequence_start_index.to(tl.int64) * stride_x_token + idx_feats * stride_x_dim
    conv_state_batch_coord = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    if conv_state_batch_coord == pad_slot_id:
        return
    conv_states_base = (conv_states_ptr + (conv_state_batch_coord * stride_istate_seq)
                        + (idx_feats * stride_istate_dim))
    mask_w = idx_feats < dim
    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            x_elem_ty = x_ptr.dtype.element_ty
            prior_tokens = conv_states_base + 2 * stride_istate_token
            col2 = tl.load(prior_tokens, mask_w, 0.0).to(x_elem_ty)
            col1 = tl.load(prior_tokens - 1 * stride_istate_token, mask_w, 0.0).to(x_elem_ty)
            col0 = tl.load(prior_tokens - 2 * stride_istate_token, mask_w, 0.0).to(x_elem_ty)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        _conv_state_update_w4(x_ptr, conv_states_ptr, x_base, conv_states_base, conv_state_batch_coord,
                              sequence_start_index, seqlen, idx_feats, load_init_state,
                              dim, num_cache_lines, stride_x_dim, stride_x_token,
                              stride_istate_seq, stride_istate_dim, stride_istate_token)
    else:
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_base = w_ptr + (idx_feats * stride_w_dim)
    w0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    xb = x_base + token_offset.to(tl.int64) * stride_x_token
    first_token = (sequence_start_index + token_offset).to(tl.int64)
    if OUTPUT_SPLIT_SIZE:
        ob = (o_ptr + (idx_feats // OUTPUT_SPLIT_SIZE).to(tl.int64) * total_tokens * OUTPUT_SPLIT_SIZE
              + first_token * OUTPUT_SPLIT_SIZE + idx_feats % OUTPUT_SPLIT_SIZE)
        o_tok = OUTPUT_SPLIT_SIZE
    else:
        ob = o_ptr + first_token * stride_o_token + idx_feats * stride_o_dim
        o_tok = stride_o_token
    x0 = tl.load(xb + 0 * stride_x_token, mask=mask_w & (0 < seg))
    x1 = tl.load(xb + 1 * stride_x_token, mask=mask_w & (1 < seg))
    x2 = tl.load(xb + 2 * stride_x_token, mask=mask_w & (2 < seg))
    x3 = tl.load(xb + 3 * stride_x_token, mask=mask_w & (3 < seg))
    x4 = tl.load(xb + 4 * stride_x_token, mask=mask_w & (4 < seg))
    x5 = tl.load(xb + 5 * stride_x_token, mask=mask_w & (5 < seg))
    x6 = tl.load(xb + 6 * stride_x_token, mask=mask_w & (6 < seg))
    x7 = tl.load(xb + 7 * stride_x_token, mask=mask_w & (7 < seg))
    x8 = tl.load(xb + 8 * stride_x_token, mask=mask_w & (8 < seg))
    x9 = tl.load(xb + 9 * stride_x_token, mask=mask_w & (9 < seg))
    x10 = tl.load(xb + 10 * stride_x_token, mask=mask_w & (10 < seg))
    x11 = tl.load(xb + 11 * stride_x_token, mask=mask_w & (11 < seg))
    x12 = tl.load(xb + 12 * stride_x_token, mask=mask_w & (12 < seg))
    x13 = tl.load(xb + 13 * stride_x_token, mask=mask_w & (13 < seg))
    x14 = tl.load(xb + 14 * stride_x_token, mask=mask_w & (14 < seg))
    x15 = tl.load(xb + 15 * stride_x_token, mask=mask_w & (15 < seg))
    x16 = tl.load(xb + 16 * stride_x_token, mask=mask_w & (16 < seg))
    x17 = tl.load(xb + 17 * stride_x_token, mask=mask_w & (17 < seg))
    x18 = tl.load(xb + 18 * stride_x_token, mask=mask_w & (18 < seg))
    x19 = tl.load(xb + 19 * stride_x_token, mask=mask_w & (19 < seg))
    x20 = tl.load(xb + 20 * stride_x_token, mask=mask_w & (20 < seg))
    x21 = tl.load(xb + 21 * stride_x_token, mask=mask_w & (21 < seg))
    x22 = tl.load(xb + 22 * stride_x_token, mask=mask_w & (22 < seg))
    x23 = tl.load(xb + 23 * stride_x_token, mask=mask_w & (23 < seg))
    x24 = tl.load(xb + 24 * stride_x_token, mask=mask_w & (24 < seg))
    x25 = tl.load(xb + 25 * stride_x_token, mask=mask_w & (25 < seg))
    x26 = tl.load(xb + 26 * stride_x_token, mask=mask_w & (26 < seg))
    x27 = tl.load(xb + 27 * stride_x_token, mask=mask_w & (27 < seg))
    x28 = tl.load(xb + 28 * stride_x_token, mask=mask_w & (28 < seg))
    x29 = tl.load(xb + 29 * stride_x_token, mask=mask_w & (29 < seg))
    x30 = tl.load(xb + 30 * stride_x_token, mask=mask_w & (30 < seg))
    x31 = tl.load(xb + 31 * stride_x_token, mask=mask_w & (31 < seg))
    tl.store(ob + 0 * o_tok, _tap4_silu(acc0, col0, col1, col2, x0, w0, w1, w2, w3), mask=mask_w & (0 < seg))
    tl.store(ob + 1 * o_tok, _tap4_silu(acc0, col1, col2, x0, x1, w0, w1, w2, w3), mask=mask_w & (1 < seg))
    tl.store(ob + 2 * o_tok, _tap4_silu(acc0, col2, x0, x1, x2, w0, w1, w2, w3), mask=mask_w & (2 < seg))
    tl.store(ob + 3 * o_tok, _tap4_silu(acc0, x0, x1, x2, x3, w0, w1, w2, w3), mask=mask_w & (3 < seg))
    tl.store(ob + 4 * o_tok, _tap4_silu(acc0, x1, x2, x3, x4, w0, w1, w2, w3), mask=mask_w & (4 < seg))
    tl.store(ob + 5 * o_tok, _tap4_silu(acc0, x2, x3, x4, x5, w0, w1, w2, w3), mask=mask_w & (5 < seg))
    tl.store(ob + 6 * o_tok, _tap4_silu(acc0, x3, x4, x5, x6, w0, w1, w2, w3), mask=mask_w & (6 < seg))
    tl.store(ob + 7 * o_tok, _tap4_silu(acc0, x4, x5, x6, x7, w0, w1, w2, w3), mask=mask_w & (7 < seg))
    tl.store(ob + 8 * o_tok, _tap4_silu(acc0, x5, x6, x7, x8, w0, w1, w2, w3), mask=mask_w & (8 < seg))
    tl.store(ob + 9 * o_tok, _tap4_silu(acc0, x6, x7, x8, x9, w0, w1, w2, w3), mask=mask_w & (9 < seg))
    tl.store(ob + 10 * o_tok, _tap4_silu(acc0, x7, x8, x9, x10, w0, w1, w2, w3), mask=mask_w & (10 < seg))
    tl.store(ob + 11 * o_tok, _tap4_silu(acc0, x8, x9, x10, x11, w0, w1, w2, w3), mask=mask_w & (11 < seg))
    tl.store(ob + 12 * o_tok, _tap4_silu(acc0, x9, x10, x11, x12, w0, w1, w2, w3), mask=mask_w & (12 < seg))
    tl.store(ob + 13 * o_tok, _tap4_silu(acc0, x10, x11, x12, x13, w0, w1, w2, w3), mask=mask_w & (13 < seg))
    tl.store(ob + 14 * o_tok, _tap4_silu(acc0, x11, x12, x13, x14, w0, w1, w2, w3), mask=mask_w & (14 < seg))
    tl.store(ob + 15 * o_tok, _tap4_silu(acc0, x12, x13, x14, x15, w0, w1, w2, w3), mask=mask_w & (15 < seg))
    tl.store(ob + 16 * o_tok, _tap4_silu(acc0, x13, x14, x15, x16, w0, w1, w2, w3), mask=mask_w & (16 < seg))
    tl.store(ob + 17 * o_tok, _tap4_silu(acc0, x14, x15, x16, x17, w0, w1, w2, w3), mask=mask_w & (17 < seg))
    tl.store(ob + 18 * o_tok, _tap4_silu(acc0, x15, x16, x17, x18, w0, w1, w2, w3), mask=mask_w & (18 < seg))
    tl.store(ob + 19 * o_tok, _tap4_silu(acc0, x16, x17, x18, x19, w0, w1, w2, w3), mask=mask_w & (19 < seg))
    tl.store(ob + 20 * o_tok, _tap4_silu(acc0, x17, x18, x19, x20, w0, w1, w2, w3), mask=mask_w & (20 < seg))
    tl.store(ob + 21 * o_tok, _tap4_silu(acc0, x18, x19, x20, x21, w0, w1, w2, w3), mask=mask_w & (21 < seg))
    tl.store(ob + 22 * o_tok, _tap4_silu(acc0, x19, x20, x21, x22, w0, w1, w2, w3), mask=mask_w & (22 < seg))
    tl.store(ob + 23 * o_tok, _tap4_silu(acc0, x20, x21, x22, x23, w0, w1, w2, w3), mask=mask_w & (23 < seg))
    tl.store(ob + 24 * o_tok, _tap4_silu(acc0, x21, x22, x23, x24, w0, w1, w2, w3), mask=mask_w & (24 < seg))
    tl.store(ob + 25 * o_tok, _tap4_silu(acc0, x22, x23, x24, x25, w0, w1, w2, w3), mask=mask_w & (25 < seg))
    tl.store(ob + 26 * o_tok, _tap4_silu(acc0, x23, x24, x25, x26, w0, w1, w2, w3), mask=mask_w & (26 < seg))
    tl.store(ob + 27 * o_tok, _tap4_silu(acc0, x24, x25, x26, x27, w0, w1, w2, w3), mask=mask_w & (27 < seg))
    tl.store(ob + 28 * o_tok, _tap4_silu(acc0, x25, x26, x27, x28, w0, w1, w2, w3), mask=mask_w & (28 < seg))
    tl.store(ob + 29 * o_tok, _tap4_silu(acc0, x26, x27, x28, x29, w0, w1, w2, w3), mask=mask_w & (29 < seg))
    tl.store(ob + 30 * o_tok, _tap4_silu(acc0, x27, x28, x29, x30, w0, w1, w2, w3), mask=mask_w & (30 < seg))
    tl.store(ob + 31 * o_tok, _tap4_silu(acc0, x28, x29, x30, x31, w0, w1, w2, w3), mask=mask_w & (31 < seg))


def conv1d_w4_fast(x, weight, bias, conv_states, query_start_loc, seq_lens_cpu, cache_indices, has_initial_state,
                   output_split_size=0, pad_slot_id=-1):
    dim, cu_seqlen = x.shape
    if (weight.shape != (dim, 4) or dim % CONV_BLOCK_N or cache_indices is None or has_initial_state is None
            or conv_states is None or conv_states.shape[1] != dim or conv_states.shape[2] < 3):
        raise RuntimeError(f"k2 conv: unsupported call dim={dim} weight={tuple(weight.shape)}")
    if output_split_size:
        if dim % output_split_size or output_split_size % CONV_BLOCK_N:
            raise RuntimeError(f"k2 conv: output_split_size {output_split_size}")
        out = torch.empty((dim // output_split_size, cu_seqlen, output_split_size), dtype=x.dtype, device=x.device)
        so_dim, so_tok = 1, output_split_size
    else:
        out = torch.empty_like(x)
        so_dim, so_tok = out.stride(0), out.stride(1)
    grid = (dim // CONV_BLOCK_N, triton.cdiv(max(seq_lens_cpu), CONV_BLOCK_M), len(seq_lens_cpu))
    _conv_w4_kernel[grid](
        x, weight, bias, conv_states, cache_indices, has_initial_state, query_start_loc, out,
        dim, cu_seqlen, conv_states.size(0), x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2), so_dim, so_tok, pad_slot_id,
        HAS_BIAS=bias is not None, BLOCK_N=CONV_BLOCK_N, OUTPUT_SPLIT_SIZE=output_split_size,
        num_warps=CONV_WARPS, num_stages=1,
    )
    return out


_CONV_WARM = set()

def conv_warm(x, weight, bias, conv_states, cache_indices, query_start_loc, split, where):
    key = (x.device, x.dtype, tuple(x.stride()), x.shape[0], weight.dtype, tuple(weight.stride()), bias is None,
           tuple(conv_states.shape), tuple(conv_states.stride()), cache_indices.dtype, query_start_loc.dtype, split)
    if key in _CONV_WARM:
        return
    _need(split and not torch.cuda.is_current_stream_capturing(), "k2 conv: warm-up inside a capture or without planes")
    t0, dev, dim = time.monotonic(), x.device, x.shape[0]
    qsl = torch.tensor([0, 1], dtype=query_start_loc.dtype, device=dev)
    pad = torch.full((1,), -1, dtype=cache_indices.dtype, device=dev)
    init = torch.zeros((1,), dtype=torch.bool, device=dev)
    out = torch.empty((CONV_BLOCK_N,), dtype=x.dtype, device=dev)
    for total in (CONV_BLOCK_M, CONV_BLOCK_M + 1):
        _conv_w4_kernel[(dim // CONV_BLOCK_N, 1, 1)](
            x, weight, bias, conv_states, pad, init, qsl, out,
            dim, total, conv_states.size(0), x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
            conv_states.stride(0), conv_states.stride(1), conv_states.stride(2), 1, split, -1,
            HAS_BIAS=bias is not None, BLOCK_N=CONV_BLOCK_N, OUTPUT_SPLIT_SIZE=split, num_warps=CONV_WARPS,
            num_stages=1)
    _CONV_WARM.add(key)
    _log.warning("projfp8: k2 conv variants compiled %s in %.2f s for %s", where, time.monotonic() - t0, key[2:])


def _convolve_prefill_k2(backend, node, layer, forward_batch, mixed_qkv, conv_states, cache_indices):
    split_size = (layer.q_dim if backend.kernel_dispatcher.prefill_backend.is_flashinfer()
                  and layer.q_dim == layer.k_dim == layer.v_dim else 0)
    _need(split_size and conv_states.dtype == mixed_qkv.dtype,
          f"k2 conv: plane split {split_size}, conv-state pool {conv_states.dtype} for {mixed_qkv.dtype} input")
    x, qsl = mixed_qkv.transpose(0, 1), backend.forward_metadata.query_start_loc
    if not node.warmed:
        conv_warm(x, layer.conv_weights, layer.bias, conv_states, cache_indices, qsl, split_size,
                  "in serving: the build warm-up missed this key")
        node.warmed = True
    conv_output = conv1d_w4_fast(x, layer.conv_weights, layer.bias, conv_states, qsl,
                                 forward_batch.extend_seq_lens_cpu, cache_indices,
                                 forward_batch.extend_prefix_lens > 0, output_split_size=split_size)
    return conv_output.unbind(0)


def k2_conv_probe(ra, row_stride, pool_dtype, tp):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
    _need(not torch.cuda.is_current_stream_capturing(), "k2 conv: the boot probe is inside a CUDA-graph capture")
    w, dim, split = ra.conv_weights, 3 * ra.q_dim, ra.q_dim
    dev = w.device
    gen = torch.Generator(device=dev).manual_seed(PROBE_SEED + 3)
    lens, slots, init = [2051, 33, 70], [3, 1, -1], [True, False, False]
    x = torch.randn(sum(lens), row_stride, generator=gen, device=dev).to(torch.bfloat16)[:, :dim].t()
    base = torch.randn(6, 3, dim, generator=gen, device=dev).to(pool_dtype)
    p1, p2 = base.clone().transpose(-1, -2), base.clone().transpose(-1, -2)
    qsl = torch.tensor([0, lens[0], lens[0] + lens[1], sum(lens)], dtype=torch.int32, device=dev)
    ci = torch.tensor(slots, dtype=torch.int32, device=dev)
    hi = torch.tensor(init, dtype=torch.bool, device=dev)
    want = causal_conv1d_fn(x, w, ra.bias, activation="silu", conv_states=p1, has_initial_state=hi, cache_indices=ci,
                            query_start_loc=qsl, seq_lens_cpu=lens, output_split_size=split)
    got = conv1d_w4_fast(x, w, ra.bias, p2, qsl, lens, ci, hi, output_split_size=split)
    live = slice(0, lens[0] + lens[1])
    same = (want.shape == got.shape and want.stride() == got.stride()
            and torch.equal(want[:, live].view(torch.int16), got[:, live].view(torch.int16))
            and torch.equal(p1.view(torch.int16), p2.view(torch.int16)))
    del x, base, p1, p2, want, got
    _need(all(tp.all_gather_object(same)) if tp is not None else same,
          "k2 conv: conv1d_w4_fast differs from stock's causal_conv1d_fn on the boot probe (output planes or conv-state "
          "pool) on some TP rank")
    _K2["proved"] = True
    _log.warning("k2 conv: probe dim=%d split=%d %s: conv1d_w4_fast == stock causal_conv1d_fn, planes and pools bitwise "
                 "(K2_CONV=%s)", dim, split, lens, K2_CONV)

@triton.jit
def _norm_gated_quant_kernel(x, g, q_out, s_out, w, eps, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                             G: tl.constexpr, s_group_stride, USE_GDC: tl.constexpr):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_t = o_t < T
    m_x = m_t[:, None] & m_d[None, :]
    b_x = tl.load(x + o_t[:, None] * D + o_d[None, :], mask=m_x, other=0.0).to(tl.float32)
    b_xbar = tl.where(m_d[None, :], b_x, 0.0)
    b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    b_y = (b_x * b_rstd[:, None]) * b_w[None, :]
    b_g = tl.load(g + o_t[:, None] * D + o_d[None, :], mask=m_x, other=0.0).to(tl.float32)
    b_y = b_y * tl.sigmoid(b_g)
    y_f = b_y.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y_f), axis=1), 1e-10)
    bits = (amax * 0.002232142857142857).to(tl.uint32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    qs = ((254 - e) << 23).to(tl.float32, bitcast=True)
    tl.store(q_out + o_t[:, None] * D + tl.max_contiguous(o_d, 8)[None, :], (y_f * qs[:, None]).to(tl.float8e4nv),
             mask=m_x)
    tok = o_t // G
    grp = o_t % G
    tl.store(s_out + (grp // 4) * s_group_stride + tok * 4 + grp % 4, e.to(tl.uint8), mask=m_t)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

class NormQ(torch.nn.Module):

    def __init__(self, stock, slot):
        super().__init__()
        object.__setattr__(self, "stock", stock)
        self.slot = slot
        from sglang.kernels.jit.utils import is_arch_support_pdl
        self.pdl = NORMQ_PDL

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        st = self.stock
        if (residual is not None or prenorm or residual_in_fp32 or st.activation != "sigmoid" or st.bias is not None
                or x.shape[-1] != 128 or x.dtype != torch.bfloat16 or g.dtype != torch.bfloat16):
            raise RuntimeError("k2 normq: call outside the KDA prefill o_norm contract")
        xs = x.reshape(-1, 128)
        gs = g.reshape(-1, 128).contiguous()
        H = x.shape[-2]
        rows = xs.shape[0]
        T = rows // H
        q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
        aligned_mn = (T + 3) // 4 * 4
        full = torch.empty(((H + 3) // 4, aligned_mn), device=x.device, dtype=torch.int)
        sb = full.view(torch.uint8)
        kw = {"USE_GDC": True, "launch_pdl": True} if self.pdl else {"USE_GDC": False}
        _norm_gated_quant_kernel[(triton.cdiv(rows, NORMQ_BT),)](xs, gs, q, sb, st.weight, st.eps, rows, D=128,
                                                                 BT=NORMQ_BT,
                                                                 BD=128, G=H, s_group_stride=sb.stride(0), num_warps=4,
                                                                 **kw)
        self.slot.q, self.slot.s = q, full.transpose(-1, -2)[:T, :]
        return q

def _attach_k2(attn, zc, kdae):
    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    m, prefix = attn.module, getattr(attn.proxy, "prefix", "self_attn")
    ra, on, w, heads = m.attn, m.o_norm, m.attn.conv_weights, m.local_num_heads
    local = heads * m.head_dim
    _need(type(ra) is RadixLinearAttention and m.conv_size == 4 and ra.activation == "silu" and torch.is_tensor(w)
          and w.dtype == torch.float32 and tuple(w.shape) == (3 * local, 4) and w.stride(1) == 1
          and ra.q_dim == ra.k_dim == ra.v_dim == local and local % CONV_BLOCK_N == 0 and type(on) is FusedRMSNormGated
          and on.activation == "sigmoid" and on.bias is None and on.weight is not None
          and on.hidden_size == m.head_dim == 128 and zc.ue8m0 and zc.inner.weight.shape[1] == local and heads % 4 == 0
          and attn.proxy._modules["attn"] is ra and attn.proxy._modules["o_norm"] is on
          and (kdae is not None or not K2_CONV),
          "k2: the KDA conv, o_norm or o_proj input format is not the one K2 was proved on")
    swaps = {}
    if K2_NORMQ:
        nq = NormQ(on, types.SimpleNamespace(q=None, s=None))
        gen = torch.Generator(device=w.device).manual_seed(PROBE_SEED + 1)
        x, g = (torch.randn(shape, generator=gen, device=w.device, dtype=torch.bfloat16)
                for shape in ((1, 4099, heads, 128), (4099, heads, 128)))
        with torch.no_grad():
            y = on(x.clone(), g).squeeze(0).flatten(-2)
            q_ref, s_ref = zc.quant(y, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=zc.ue8m0)
            want = zc(y)[0].clone()
            zc.slot = nq.slot
            q = nq(x, g).squeeze(0).flatten(-2)
            s = nq.slot.s
            got = zc(q)[0]
        _need(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and s.dtype == s_ref.dtype
              and s.shape == s_ref.shape and s.stride() == s_ref.stride() and torch.equal(s, s_ref)
              and torch.equal(got.view(torch.int16), want.view(torch.int16)),
              f"{prefix}: k2 NormQ -> zero-copy o_proj is not stock's o_norm -> quant -> GEMM on the probe")
        swaps["o_norm"] = nq
    if K2_CONV:
        first = attn.proxy._modules[attn.plan[0][0]]
        row_stride = (first.lin if type(first) is QkvbfgFp8 else first.inner).weight.shape[0]
        pool = kdae.warm_build(row_stride)
        if not _K2.get("proved"):
            k2_conv_probe(ra, row_stride, pool.dtype, kdae.tp)
        kdae.k2 = True
    return swaps

_KDAE = {}


def _kdae_eq(a, b):
    f = lambda r: [t if t is None else t.view(torch.int16 if t.element_size() == 2 else torch.int32)
                   for t in (r if isinstance(r, tuple) else (r,))]
    a, b = f(a), f(b)
    return len(a) == len(b) and all(x is y or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))


def _kdae_probe(d, tp, x, qsl, cu, kw):
    pool, idx, tr = kw["ssm_states"], kw["cache_indices"].long(), kw.get("track_state")
    keep = pool[idx].clone()
    want = d.extend(*x, query_start_loc=qsl, **kw)
    ref = (pool[idx].clone(), None if tr is None else tr.clone())
    pool[idx] = keep
    got = d.extend(*x, query_start_loc=cu, **kw)
    same = _kdae_eq(want, got) and _kdae_eq(ref, (pool[idx], tr))
    del want, ref, keep
    _need(all(tp.all_gather_object(same)) if tp is not None else same, "kda-eager: the extend with the forward's shared "
          "int64 cu_seqlens differs from stock's (output / state rows / track snapshots) on some TP rank")
    _KDAE["proved"] = True
    _log.warning("kda-eager: probe %d sequence(s) %d tokens: memo route == stock's extend, bitwise (KDA_EAGER=%d)",
                 qsl.numel() - 1, x[0].shape[1], KDA_EAGER)
    return got


def _kdae_extend(d, fik, tp, q, k, v, g, beta, *, query_start_loc, **kw):
    lens, n = kw.get("extend_seq_lens_cpu"), query_start_loc.numel() - 1
    kern = d.effective_extend_kernel(kw.get("lower_bound"), q.shape[1], n)
    if type(kern) is not fik or lens is None or len(lens) != n:
        return d.extend(q, k, v, g, beta, query_start_loc=query_start_loc, **kw)
    sk = (kern, q.device, torch.cuda.current_stream(q.device).cuda_stream)
    st = _KDAE.get(sk)
    if st is None:
        old = [x for x in _KDAE if isinstance(x, tuple)]
        if len(old) >= KDAE_STREAMS:
            _KDAE.pop(old[0])
        st = _KDAE[sk] = {}
    key = (tuple(lens), query_start_loc.dtype)
    if st.get("src") is not query_start_loc or st.get("key") != key:
        st.update(src=query_start_loc, key=key, cu=query_start_loc.to(torch.int64))
    if not _KDAE.get("proved"):
        return _kdae_probe(d, tp, (q, k, v, g, beta), query_start_loc, st["cu"], kw)
    return kp_extend(d, tp, q, k, v, g, beta, st["cu"], lens, kw)


class KdaeAttn(torch.nn.Module):
    def __init__(self, st, tp):
        super().__init__()
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hy
        from sglang.srt.layers.attention.linear import kda_backend as kb
        from sglang.srt.layers.attention.linear.kernels import kda_flashinfer_prefill as kf
        from sglang.srt.model_executor import forward_context as fc
        from sglang.srt.model_executor.runner_utils import prefill_graph as pg
        _need(st.proxy._modules["attn"] is st.module.attn, "kda-eager: the KDA proxy's attn is not the stock module's")
        object.__setattr__(self, "stock", st.module.attn)
        object.__setattr__(self, "tp", tp)
        self.hyb, self.kda, self.fik = hy.HybridLinearAttnBackend, kb.KDAAttnBackend, kf.FlashInferKDAPrefillKernel
        self.backend, self.in_graph = fc.get_attn_backend, pg.is_in_prefill_graph_scope
        self.k2, self.warmed = False, False
        kp_warm(st.module.attn)

    def forward(self, forward_batch, mixed_qkv, a, b):
        fm, real = forward_batch.forward_mode, forward_batch.global_num_token_non_padded_cpu
        _need(not fm.is_decode(), "kda-eager: a decode batch reached the KDA proxy")
        if (self.in_graph() or torch.cuda.is_current_stream_capturing() or not fm.is_extend() or fm.is_target_verify()
                or forward_batch.logical_forward_mode.is_decode() or (real is not None and real < mixed_qkv.shape[0])):
            return self.stock(forward_batch, mixed_qkv, a, b)
        hyb = self.backend()
        lin = getattr(hyb, "linear_attn_backend", None)
        _need(type(hyb) is self.hyb and type(lin) is self.kda and lin.prefill_graph_metadata is None
              and not hyb._is_full_attn(self.stock) and getattr(lin.forward_extend, "__func__", None) is self.kda.forward_extend
              and (not self.k2 or getattr(lin._convolve_prefill, "__func__", None) is self.kda._convolve_prefill),
              "kda-eager: the attention backend is not the eager KDA extend")
        kp_note(forward_batch)
        bp = copy.copy(lin)
        if KDA_EAGER:
            dp = copy.copy(lin.kernel_dispatcher)
            dp.extend = functools.partial(_kdae_extend, lin.kernel_dispatcher, self.fik, self.tp)
            bp.kernel_dispatcher = dp
        if self.k2:
            bp._convolve_prefill = functools.partial(_convolve_prefill_k2, lin, self)
        return self.kda.forward_extend(bp, q=None, k=None, v=None, layer=self.stock, forward_batch=forward_batch,
                                       save_kv_cache=True, mixed_qkv=mixed_qkv, a=a, b=b)

    def warm_build(self, row_stride):
        hyb = self.backend()
        lin, st = getattr(hyb, "linear_attn_backend", None), self.stock
        _need(type(hyb) is self.hyb and type(lin) is self.kda, f"k2 conv: build-time backend {type(hyb).__name__}")
        conv_states = lin.state_layer_cache(st.layer_id).conv[0].transpose(-1, -2)
        split = st.q_dim if lin.kernel_dispatcher.prefill_backend.is_flashinfer() and st.q_dim == st.k_dim == st.v_dim else 0
        _need(split and conv_states.dtype == torch.bfloat16, f"k2 conv: plane split {split}, pool {conv_states.dtype}")
        x = torch.empty((2, row_stride), dtype=torch.bfloat16, device=conv_states.device)[:, :3 * st.q_dim].t()
        for idt in (torch.int32, torch.int64):
            conv_warm(x, st.conv_weights, st.bias, conv_states, x.new_empty(0, dtype=idt),
                      x.new_empty(0, dtype=torch.int32), split, "at build")
        return conv_states

KDA_PF_BACKEND = ""
KP_PIPE_PIECES = 4
KP_PIPE_EXACT = True
KP_MIN_ROWS = 4096
KP_CUDNN_ROWS = (16384,)
KP_CKPT = True
KP_CHECKS = 2
KP_TOL = 3e-2
KP_STATE_TOL = 1e-2
KP_PLANS = 64
KP_LOG_EVERY = 4096
_KP = {"checks": 0, "off": None, "n": {}, "fn": None, "trk": None, "plans": {}, "warm": None}


def _kp_fns():
    fns = _KP["fn"]
    if fns is None:
        from flashinfer import kda as fkda
        from flashinfer.kda_kernels.persistent import forward as pfwd
        from sglang.kernels.ops.attention.kda_input_pack import contiguous_kda_inputs
        fns = _KP["fn"] = (fkda.recurrent_kda, contiguous_kda_inputs, pfwd.make_plan, pfwd.fwd)
    return fns


def _kp_count(why):
    n = _KP["n"]
    n[why] = n.get(why, 0) + 1
    tot = sum(n.values())
    if tot in (1, 34, 340) or tot % KP_LOG_EVERY == 0:
        _log.warning("kda-pf: route counts %s (backend %r, min rows %d)", dict(sorted(n.items())), KDA_PF_BACKEND,
                     KP_MIN_ROWS)


def kp_note(fb):
    m = getattr(fb, "mamba_prefill_track_mask_cpu", None)
    t, p = getattr(fb, "mamba_track_seqlens_cpu", None), getattr(fb, "extend_prefix_lens_cpu", None)
    _KP["trk"] = (int(t[0]) - int(p[0])) if (m is not None and t is not None and p is not None and len(m) == 1
                                             and len(t) == 1 and len(p) == 1 and m[0]) else None


def _kp_ckpt(kw):
    return bool(kw.get("return_intermediate_states")) and kw.get("track_state") is not None


def kp_split(T, kw):
    cs, rel = kw.get("state_checkpoint_every_n_tokens") or 0, _KP["trk"]
    if (not KP_CKPT or kw.get("num_state_checkpoints") != 1 or rel is None or cs <= 0 or cs % 32 or rel % cs == 0
            or kw.get("track_ssm_h_batch_src") is None):
        return None
    c = rel // cs * cs
    return c if 0 < c < T else None


def kp_warm(attn):
    return


def kp_why(q, lens, kw):
    return "disabled"


def _kp_plan(T, H, dev):
    plans = _KP["plans"]
    key = (T, H, dev.index)
    plan = plans.get(key)
    if plan is None:
        if len(plans) >= KP_PLANS:
            plans.pop(next(iter(plans)))
        plan = plans[key] = _kp_fns()[2]([0, T], H, dev)
    return plan


def _kp_ok(*ts):
    return all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in ts)


def kp_serve(d, q, k, v, g, beta, cu, kw):
    rkda, pack, _, fwd = _kp_fns()
    T, H = q.shape[1], q.shape[2]
    x = pack(q, k, v, g[:, :T], beta[:, :T])
    pool, idx, lb = kw["ssm_states"], kw["cache_indices"], kw["lower_bound"]
    a_log = kw["A_log"].reshape(-1).float().contiguous()
    dt = kw["dt_bias"].reshape(-1).float().contiguous()
    out = torch.empty_like(x[0])
    ckpt = _kp_ckpt(kw)
    c = kp_split(T, kw) if ckpt else None
    parts = ((0, T),) if c is None else ((0, c), (c, T))
    if not _kp_ok(*x, out) or any(a % 32 for a, _ in parts):
        _kp_count("align")
        return d.extend(q, k, v, g, beta, query_start_loc=cu, **kw)
    sl = idx.reshape(1).long()
    st = pool.index_select(0, sl)
    for a, b in parts:
        plan = _kp_plan(b - a, H, q.device)
        fwd(x[0][:, a:b], x[1][:, a:b], x[2][:, a:b], x[3][:, a:b], x[4][:, a:b], 128 ** -0.5, out[:, a:b], a_log, dt,
            float(lb), initial_state=st, final_state=st, cu_seqlens=plan[0], plan=plan)
        if b == c:
            kw["track_state"][kw["track_ssm_h_batch_src"]] = st
    pool.index_copy_(0, sl, st)
    return (out, None) if kw.get("return_intermediate_states") else out


def _kp_rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def kp_check(d, tp, q, k, v, g, beta, cu, kw):
    pool, sl = kw["ssm_states"], kw["cache_indices"].reshape(1).long()
    ck = _kp_ckpt(kw)
    trk, rows = (kw["track_state"], kw.get("track_ssm_h_batch_src")) if ck else (None, None)
    keep = pool.index_select(0, sl)
    tkeep = trk.clone() if ck else None
    want = d.extend(q, k, v, g, beta, query_start_loc=cu, **kw)
    ref, tref = pool.index_select(0, sl), (trk.clone() if ck else None)
    pool.index_copy_(0, sl, keep)
    if ck:
        trk.copy_(tkeep)
    rec = {"rows": q.shape[1], "layer": kw.get("layer_id"), "ckpt": ck}
    try:
        got = kp_serve(d, q, k, v, g, beta, cu, kw)
        wo, go = (want[0], got[0]) if isinstance(want, tuple) else (want, got)
        fin = pool.index_select(0, sl)
        rec.update(out_rel=_kp_rel(go, wo), state_rel=_kp_rel(fin, ref),
                   ckpt_rel=_kp_rel(trk[rows], tref[rows]) if ck and rows is not None else 0.0,
                   finite=bool(torch.isfinite(go).all()) and bool(torch.isfinite(fin).all()),
                   same=int((go.view(torch.int16) == wo.view(torch.int16)).sum()), n=wo.numel(),
                   state_diff=int((fin.view(torch.int32) != ref.view(torch.int32)).sum()))
        rec["ok"] = (rec["finite"] and rec["out_rel"] <= KP_TOL and rec["state_rel"] <= KP_STATE_TOL
                     and rec["ckpt_rel"] <= KP_STATE_TOL)
    except Exception as e:
        got, rec["ok"], rec["error"] = None, False, repr(e)[:300]
    votes = tp.all_gather_object(rec) if tp is not None else [rec]
    _KP["checks"] += 1
    if all(r["ok"] for r in votes):
        _log.warning("kda-pf: check %d/%d %d rows layer %s ckpt %s: %s within its class on every rank %s",
                     _KP["checks"], KP_CHECKS, rec["rows"], rec["layer"], ck, KDA_PF_BACKEND,
                     [(round(r["out_rel"], 5), round(r["state_rel"], 6), round(r["ckpt_rel"], 6), r["same"], r["n"],
                       r["state_diff"])
                      for r in votes])
        return got
    _KP["off"] = f"check {_KP['checks']} failed"
    pool.index_copy_(0, sl, ref)
    if ck:
        trk.copy_(tref)
    _log.warning("kda-pf: OFF on every rank for this process (check %d, %d rows, layer %s): %s", _KP["checks"],
                 rec["rows"], rec["layer"], votes)
    return want


def kp_extend(d, tp, q, k, v, g, beta, cu, lens, kw):
    why = kp_why(q, lens, kw)
    _kp_count(why or (KDA_PF_BACKEND + ("+ckpt" if _kp_ckpt(kw) else "")))
    if why:
        return d.extend(q, k, v, g, beta, query_start_loc=cu, **kw)
    if _KP["checks"] < KP_CHECKS:
        return kp_check(d, tp, q, k, v, g, beta, cu, kw)
    return kp_serve(d, q, k, v, g, beta, cu, kw)



@triton.jit
def _rq_st(x, BM: tl.constexpr, S: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(x, (BM, 64 // S, 2, S)), (0, 1, 3, 2)))
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (BM, 128))


@triton.jit(do_not_specialize=["m"])
def _rq_kernel(x_ptr, r_ptr, y_ptr, s_ptr, m, hs, ROUND: tl.constexpr, QUANT: tl.constexpr, BM: tl.constexpr,
               PDL: tl.constexpr = False):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    ok = rows < m
    offs = rows[:, None] * 128 + tl.arange(0, 128)[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
    v = tl.load(x_ptr + offs, mask=ok[:, None], other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    for k in tl.static_range(7):
        v = _rq_st(v, BM, 1 << k)
    r = (v * hs).to(tl.bfloat16)
    tl.store(r_ptr + offs, r, mask=ok[:, None])
    if QUANT:
        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1.0 / fp8_max
        x = r.to(tl.float32)
        x_abs = tl.abs(x)
        amax = tl.max(x_abs, axis=1)
        amax = tl.maximum(amax, 1e-4)
        if ROUND:
            log_val = tl.log2(amax * fp8_max_inv)
            log_ceil = tl.ceil(log_val)
            scale = tl.exp2(log_ceil)
        else:
            scale = amax * fp8_max_inv
        scale_broadcast = scale[:, None]
        y = x / scale_broadcast
        y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)
        tl.store(y_ptr + offs, y, mask=ok[:, None])
        tl.store(s_ptr + rows, scale, mask=ok)


def rq_run(x, quant, rnd):
    m, r = x.numel() // BLOCK, torch.empty_like(x)
    y = torch.empty_like(x, dtype=_FP8) if quant else r
    s = x.new_empty(*x.shape[:-1], 1, dtype=torch.float32) if quant else r
    if m:
        _rq_kernel[(triton.cdiv(m, RQ_BM),)](x, r, y, s, m, BLOCK ** -0.5, ROUND=rnd, QUANT=quant, BM=RQ_BM,
                                             num_warps=RQ_WARPS, **_pdl("rotq"))
    return r, y, s


_rq_ver = lambda t: None if t.is_inference() else t._version


def rq_call(sh, x):
    st, sg, quant = sh._s, sh._sg, sh._q
    rnd = st.scale_fmt is not None
    _need(x.dtype == torch.bfloat16 and x.dim() == 3 and x.shape[-1] == BLOCK and x.is_contiguous()
          and x.data_ptr() % 16 == 0, f"rotq: indexer query {x.dtype} {tuple(x.shape)} is not [T, H, 128] contiguous BF16")
    warm = _PROCESS.setdefault("rq_warm", set())
    if torch.cuda.is_current_stream_capturing():
        _need(rnd in warm, "rotq: the first fused call is inside a CUDA-graph capture (Triton would JIT there)")
    elif rnd not in warm and x.numel():
        want = sg.rotate(x)
        wy, ws = sg.act_quant(want, BLOCK, st.scale_fmt)
        for q in (False, True):
            r, y, s = rq_run(x, q, rnd)
            _need(torch.equal(r.view(torch.int16), want.view(torch.int16)) and (not q or (torch.equal(
                y.view(torch.uint8), wy.view(torch.uint8)) and torch.equal(s.view(torch.int32), ws.view(torch.int32)))),
                f"rotq: layer {st.layer_id} rows={x.shape[0]} quant={q}: the fused kernel differs from stock's")
        del want, wy, ws, r, y, s
        warm.add(rnd)
        _log.warning("rotq: layer %d rows=%d scale_fmt=%s bm=%d: fused rotate (+ quant) == stock's", st.layer_id,
                     x.shape[0], st.scale_fmt, RQ_BM)
    r, y, s = rq_run(x, quant, rnd)
    sh._p = (r, _rq_ver(r), y, s) if quant else None
    return r


def rq_quant(sh, x, block_size=128, scale_fmt=None):
    p, sh._p = sh._p, None
    _need(p is not None and x is p[0] and _rq_ver(x) == p[1] and block_size == BLOCK
          and (scale_fmt is None) == (sh._s.scale_fmt is None), "rotq: act_quant on a tensor that is not this verify "
          "call's fused query")
    return p[2], p[3]


def rq_get_q_k(sh, q_lora, x, positions, enable_dual_stream, forward_batch, precompute_compress_gate=False,
               precompute_head_gate=False):
    st, gate, hw = sh._s, None, None
    if not enable_dual_stream:
        q = st.wq_b(q_lora)[0]
        q = q.reshape(q.shape[0], -1, st.head_dim)
        k = st.k_norm(st.wk(x)[0])
        return rq_call(sh, q), k, gate, hw
    cur = torch.cuda.current_stream()
    st.alt_stream.wait_stream(cur)
    if precompute_compress_gate:
        st.compress_gate_stream.wait_stream(cur)
    with sh._sg.ik.deep_gemm_wrapper.configure_deep_gemm_num_sms(st.half_device_sm_count):
        q = st.wq_b(q_lora)[0]
        q = q.reshape(q.shape[0], -1, st.head_dim)
    if precompute_head_gate:
        hw = wtc_pre(sh, x)
    q = rq_call(sh, q)
    with torch.cuda.stream(st.alt_stream):
        k = st.k_norm(st.wk(x)[0])
    if precompute_compress_gate:
        with torch.cuda.stream(st.compress_gate_stream):
            gate = torch.nn.functional.linear(x, st.index_kpool_compress_gate)
    cur.wait_stream(st.alt_stream)
    return q, k, gate, hw


class _RqShim(_IdxShim):
    index_kpool_compress = property(lambda self: self._s.index_kpool_compress)
    _forward_cuda_skip_logits = property(lambda self: self._s._forward_cuda_skip_logits)
    _compress_write = property(lambda self: self._s._compress_write)
    _can_overlap_prefill = property(lambda self: self._s._can_overlap_prefill)
    _or = None
    def _get_topk_ragged(self, *args, **kwargs):
        return orp_topk_ragged(self, args, kwargs)
    def _get_topk_ragged_kpool_plan(self, forward_batch, layer_id, q_fp8, weights, metadata):
        return orp_topk_plan(self, forward_batch, layer_id, q_fp8, weights, metadata)
    def _resolve_head_gate_weights(self, x, q_scale, head_weights):
        return orp_gates(self, x, q_scale, head_weights)
    def __init__(self, stock, sg):
        super().__init__(stock, sg)
        self._q, self._p = False, None
    def _get_q_k_bf16(self, q_lora, *args, **kwargs):
        q_lora = orp_rows(self, rsq_rows(self, q_lora))
        return rq_get_q_k(self, q_lora, *args, **kwargs) if ROTQ else self._s._get_q_k_bf16(q_lora, *args, **kwargs)
    def _forward_cuda_impl(self, *args, **kwargs):
        return self._sg.Indexer._forward_cuda_impl(self, *args, **kwargs)
    def _forward_cuda_target_verify(self, *args, **kwargs):
        self._rs = rsq_plan(self, args, kwargs)
        if ROTQ:
            _need(not args and kwargs.get("act_quant") is self._sg.act_quant, "rotq: the verify call does not "
                  "hand over SGLang's Triton act_quant")
            kwargs["act_quant"], self._q = functools.partial(rq_quant, self), bool(kwargs.get("return_indices", True))
        try:
            return self._sg.Indexer._forward_cuda_target_verify(self, *args, **kwargs)
        finally:
            self._q, self._p, self._rs = False, None, None

class RqNN(IdxNN):
    def __init__(self, stock, sg, ext):
        super().__init__(stock, sg)
        object.__setattr__(self, "shim", _RqShim(stock, sg))
        object.__setattr__(self, "ext", ext)
    def forward(self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
        if not self.ext:
            return super().forward(x, q_lora, positions, forward_batch, layer_id, return_indices)
        st, sg = self.stock, self.sg
        method = st._forward_method
        if method is None or not (ROTQ and forward_batch.forward_mode.is_extend_without_speculative()):
            return st(x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch, layer_id=layer_id,
                      return_indices=return_indices)
        _need(getattr(method, "__func__", None) is sg.Indexer.forward_cuda and method.__self__ is st
              and not st.is_torch_compile and sg.fused_op.get_fused_op_backend() is None and not st._forward_hooks
              and not st._forward_pre_hooks, "rotq: the stock indexer does not dispatch to IndexerKPool.forward_cuda "
              "(forced backend, torch.compile mode, a rebound dispatch or module hooks)")
        if sg.ik.is_in_breakable_cuda_graph():
            out = torch.empty((x.shape[0] if return_indices else 0, st.index_topk + st.index_kpool - 1),
                              dtype=torch.int32, device=x.device)
            sg.kpg.bcg_kpool_indexer_prefill_with_output(self.shim, x, q_lora, positions, out, layer_id)
            return out if return_indices else None
        return orp_extend(self, x, q_lora, positions, forward_batch, layer_id, return_indices)

def rq_facts(idx, sg):
    _need(not (not idx.skip_rope and idx.rope_head_dim > 0) and idx.head_dim == BLOCK and idx.block_size == BLOCK
          and not sg.di._is_hip and not sg.di._is_xpu and sg.rotate is sg.di.rotate_activation is sg.ik.rotate_activation
          and callable(sg.act_quant) and callable(getattr(sg.kpg.bcg_kpool_indexer_prefill_with_output, "run_eager", None))
          and callable(sg.ik.is_in_breakable_cuda_graph) and (idx.alt_stream is None or idx.half_device_sm_count > 0),
          f"rotq: layer {idx.layer_id}: the indexer query is not the skip_rope CUDA rotate_activation + Triton act_quant path")

_PK = {}

@triton.jit(do_not_specialize=["nq", "nk", "qb"])
def _pk_kernel(q_ptr, k_ptr, qo_ptr, ko_ptr, nq, nk, H, sqt, sqh, skt, qb, D: tl.constexpr, R: tl.constexpr,
               PDL: tl.constexpr = False):
    pid, c = tl.program_id(0), tl.arange(0, D)[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
    if pid < qb:
        r = pid.to(tl.int64) * R + tl.arange(0, R)
        m = (r < nq)[:, None]
        x = tl.load(q_ptr + ((r // H) * sqt + (r % H) * sqh)[:, None] + c, mask=m)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(qo_ptr + r[:, None] * D + c, x.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=m)
    else:
        r = (pid - qb).to(tl.int64) * R + tl.arange(0, R)
        m = (r < nk)[:, None]
        x = tl.load(k_ptr + r[:, None] * skt + c, mask=m)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(ko_ptr + r[:, None] * D + c, x.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=m)

def pk_pack(q, k):
    T, H, D = q.shape
    q8, k8 = torch.empty((T, H, D), dtype=_FP8, device=q.device), torch.empty((T, 1, D), dtype=_FP8, device=q.device)
    qb = triton.cdiv(T * H, PK_R)
    _pk_kernel[(qb + triton.cdiv(T, PK_R),)](q, k, q8, k8, T * H, T, H, q.stride(0), q.stride(1), k.stride(0), qb, D=D,
                                             R=PK_R, num_warps=PK_WARPS, **_pdl("fp8pack"))
    return q8, k8

def pk_probe(dev, H, D):
    from sglang.srt.layers.attention import dsa_backend
    _need(not torch.cuda.is_current_stream_capturing(), "fp8pack: the first packed call is inside a CUDA-graph capture")
    x = torch.arange(1 << 16, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16).view(-1, D)
    q = x.expand(H, *x.shape).contiguous().transpose(0, 1)
    want, got = dsa_backend.mla_quantize_for_fp8_no_rope(q, q[..., :0], x, x[:, :0], D, 0), pk_pack(q, x.unsqueeze(1))
    _need(torch.equal(want[0].view(torch.uint8), got[0].view(torch.uint8))
          and torch.equal(want[1].view(torch.uint8), got[1].view(-1, D).view(torch.uint8)),
          "fp8pack: the packed FP8 q / k differ from stock's cast + copy on the BF16 pattern table")
    _PK[dev] = True
    _log.warning("fp8pack: probe %s: %d BF16 patterns x %d heads and k, pack == stock's cast + copy", dev, 1 << 16, H)

def pk_skip(env, attn, q, k, fb, kwargs):
    why = qpad_skip(env, attn, q, fb, kwargs)
    if why is None and not (torch.is_tensor(k) and k.dtype == torch.bfloat16 and k.shape[0] == q.shape[0]
                            and k.shape[-1] == q.shape[2] and k.numel() == q.shape[0] * q.shape[2] and k.stride(-1) == 1
                            and k.stride(0) % 16 == 0 and k.data_ptr() % 16 == 0 and q.data_ptr() % 16 == 0):
        why = "k-layout"
    return why

class _PackEager(_QpadEager):
    def run_eager(self, q, k, v, output, save_kv_cache=True, key_value_num_tokens=None, idx_output=None,
                  lse_output=None, **kwargs):
        o, fb = self.owner, self.owner.env.get_forward_batch()
        raw = fb.global_num_token_non_padded_cpu
        n = o.env.query_tokens(fb, raw)
        why = pk_skip(o.env, o.stock, q, k, fb, kwargs)
        if why is None and not (type(n) is int and n == raw and 0 < n <= q.shape[0] and key_value_num_tokens is None
                                and idx_output is None and lse_output is None):
            why = "graph-rows"
        orp_heal(o, kwargs)
        q, k, v, kwargs = o.pack(why, q, k, v, kwargs, n, "run_eager")
        return o.stock._eager_attention.run_eager(q, k, v, output, save_kv_cache, key_value_num_tokens, idx_output,
                                                  lse_output, **kwargs)

class PackAttn(QpadAttn):
    def __init__(self, stock, env, prefix, role):
        super().__init__(stock, env, prefix)
        self.role = role
        self._eager_attention = _PackEager(self)
    def pack(self, why, q, k, v, kwargs, n=None, entry="forward"):
        if (entry, why) not in self.seen:
            self.seen.add((entry, why))
            _log.warning("fp8pack: %s [%s %s] %s rows=%s", self.prefix, self.role, entry,
                         f"on (FP8PACK={FP8PACK})" if why is None else f"stock: {why}",
                         tuple(q.shape) if torch.is_tensor(q) else None)
        if why is not None:
            return q, k, v, kwargs
        if q.device not in _PK:
            pk_probe(q.device, q.shape[1], q.shape[2])
        q8, k8 = pk_pack(q[:n], k[:n])
        return q8, k8, k8, dict(kwargs, q_rope=None, k_rope=kwargs["k_rope"].to(_FP8)) if FP8PACK == 2 else kwargs
    def forward(self, q, k, v, forward_batch, save_kv_cache=True, key_value_num_tokens=None, **kwargs):
        g3_guard(self)
        why = pk_skip(self.env, self.stock, q, k, forward_batch, kwargs)
        if why is None and (self.env.in_graph_scope() or key_value_num_tokens is not None
                            or self.env.padded_rows(q.shape[0], forward_batch) is not None):
            why = "graph-or-padded"
        orp_heal(self, kwargs)
        q, k, v, kwargs = self.pack(why, q, k, v, kwargs)
        return self.stock(q, k, v, forward_batch, save_kv_cache, key_value_num_tokens, **kwargs)

def pk_attn(attn, layer_id, role):
    m = getattr(attn, "_modules", {}).get("attn_mqa")
    return sx_attn(m, _QpadEnv(), f"model.layers.{layer_id}.self_attn", role) if FP8PACK and m is not None else None

ROWSHARD_Q = True
RSQ_PROBE_SEED = 20261160


def rs_shard_ok(tp, b, rows):
    if tp is None or not ROWSHARD:
        return False
    bq = -(-b // tp.world_size)
    return RS_MIN_BS <= b and rows <= DEC_MAX_ROWS and (tp.world_size - 1) * bq < b


def _rsq_same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.view(bits), b.view(bits))


def rsq_plan(sh, args, kw):
    tp = getattr(sh._sg, "tp", None)
    if not ROWSHARD_Q or tp is None or not ROWSHARD:
        return None
    _need(not args and {"x", "q_lora", "positions", "forward_batch", "metadata", "enable_dual_stream",
                        "act_quant"} <= set(kw), "rowshard-q: stock's verify body was not called with its keywords")
    q_lora, meta = kw["q_lora"], kw["metadata"]
    if not kw["enable_dual_stream"] or not kw.get("return_indices", True):
        return None
    plan = meta.attn_metadata.kpool_write_plan
    n = 0 if plan is None or plan.pool_seqlens_per_q is None else plan.num_draft_tokens
    rows, pad = meta.get_seqlens_expanded().shape[0], q_lora.shape[0]
    if n < 2 or rows % n or not 0 < rows <= pad or q_lora.dim() != 2 or not q_lora.is_contiguous():
        return None
    b = rows // n
    if not rs_shard_ok(tp, b, rows):
        return None
    bq = -(-b // tp.world_size)
    a, e = min(b, tp.rank_in_group * bq), min(b, (tp.rank_in_group + 1) * bq)
    key, capturing = (n, b, pad, q_lora.shape[1]), torch.cuda.is_current_stream_capturing()
    ok = _PROCESS.setdefault("rsq_ok", {})
    if key not in ok:
        _need(not capturing, f"rowshard-q: the first {b}x{n} verify call is inside a CUDA-graph capture (its wq_b "
              "probe runs eagerly; SGLang's runner warms every graph eagerly first)")
        ok[key] = rsq_probe(sh, tp, q_lora, a * n, e * n, b)
    if not ok[key]:
        return None
    rs = types.SimpleNamespace(n=n, b=b, rows=rows, pad=pad, a=a, e=e, bq=bq, r0=a * n, r1=e * n, key=key, ref=None,
                               qk=0)
    if not capturing and key not in _PROCESS.setdefault("rsq_checked", set()):
        rs.ref = rsq_ref(sh, kw)
    return rs


def rsq_probe(sh, tp, q_lora, r0, r1, b):
    st = sh._s
    gen = torch.Generator(device=q_lora.device).manual_seed(RSQ_PROBE_SEED + b)
    p = torch.randn(tuple(q_lora.shape), generator=gen, device=q_lora.device).to(q_lora.dtype)
    with sh._sg.ik.deep_gemm_wrapper.configure_deep_gemm_num_sms(st.half_device_sm_count):
        full, part = st.wq_b(p)[0], st.wq_b(p[r0:r1])[0]
    mine = _rsq_same(part, full[r0:r1])
    verdicts = tp.all_gather_object(mine)
    _log.warning("rowshard-q: wq_b probe B=%d rows [%d, %d) of %d: block == whole-batch rows %s (ranks %s)", b, r0, r1,
                 q_lora.shape[0], mine, verdicts)
    return all(verdicts)


def rsq_ref(sh, kw):
    st, x = sh._s, kw["x"]
    q, _, _, hw = st._get_q_k_bf16(kw["q_lora"], x, kw["positions"], enable_dual_stream=True,
                                   forward_batch=kw["forward_batch"], precompute_compress_gate=False,
                                   precompute_head_gate=True)
    q_fp8, q_scale = kw["act_quant"](q, st.block_size, st.scale_fmt)
    return q_fp8, st._resolve_head_gate_weights(x, q_scale, hw)


def rsq_rows(sh, q_lora):
    rs = sh._rs
    if rs is None:
        return q_lora
    rs.qk += 1
    _need(rs.qk == 1 and q_lora.shape[0] == rs.pad and q_lora.is_contiguous(), f"rowshard-q: query call {rs.qk} on "
          f"q_lora {tuple(q_lora.shape)}, planned for {rs.pad} rows")
    return q_lora[rs.r0:rs.r1]


def rsq_gates(sh, x, q_scale, head_weights):
    rs = sh._rs
    if rs is not None:
        blk = getattr(rs, "hw_block", False)
        _need(head_weights is not None and head_weights.shape[0] == (rs.r1 - rs.r0 if blk else rs.pad)
              and q_scale.shape[0] == rs.r1 - rs.r0,
              f"rowshard-q: head gates {None if head_weights is None else tuple(head_weights.shape)} / q scales "
              f"{tuple(q_scale.shape)} are not the batch's / the block's")
        if not blk:
            head_weights = head_weights[rs.r0:rs.r1]
    elif head_weights is None:
        head_weights = wtc_project(sh._s, x)
    return ge_scale(sh._s, x, q_scale, head_weights)


def rsq_block(st, sg, q, kv, w, lens, table, max_len, seq, pt, offs, plan, a, e, out_rows):
    n = q.shape[1]
    r0, r1 = a * n, e * n
    p = ph_plan(sg.kif, lens[r0:r1]) if plan else None
    logits = idx_logits(sg.ik.deep_gemm, q, kv, w, lens[r0:r1].view(e - a, n), table[a:e], max_len)
    return st._topk_from_kpool_logits(logits=logits, pool_lens=lens[r0:r1], topk_v2_plan=p, seq_lens=seq[r0:r1],
                                      page_table=None if pt is None else pt[r0:r1],
                                      topk_offsets=None if offs is None else offs[r0:r1], out_rows=out_rows)


def rsq_topk_paged(sh, forward_batch, layer_id, q_fp8, weights, metadata):
    st, sg, rs = sh._s, sh._sg, sh._rs
    tp, n, m = sg.tp, rs.n, rs.r1 - rs.r0
    plan = metadata.attn_metadata.kpool_write_plan
    seqlens_32 = metadata.get_seqlens_expanded()
    _need(plan is not None and plan.pool_seqlens_per_q is not None and plan.num_draft_tokens == n
          and seqlens_32.shape[0] == rs.rows and q_fp8.dim() == 3 and q_fp8.shape[0] == m and weights.shape[0] == m,
          f"rowshard-q: the verify call is no longer the planned {rs.b}x{n} block [{rs.r0}, {rs.r1}) at its top-k")
    pool = sg.ik.get_token_to_kv_pool()
    _need(pool.index_page_size == 64, "idx-nextn: the index-K pool page is not 64 tokens")
    block_tables = metadata.get_page_table_64()
    kv = st._get_index_k_read_buffer(pool, layer_id)
    _need(kv.dim() == 2 and weights.dim() == 3, "idx-nextn: index-K buffer / head weights layout")
    kv = kv.view(kv.shape[0], 64, 1, 132)
    weights = weights.squeeze(2)
    _need(not st.paged_mqa_logits_backend.is_aiter() and not st._should_use_tilelang_paged_mqa_logits(q_fp8),
          "idx-nextn: the stock indexer's paged-MQA logits backend is not DeepGEMM")
    lens, _, table, _, v2plan = st._get_kpool_decode_metadata(metadata=metadata, block_tables=block_tables,
                                                              seqlens_32=seqlens_32, blocksize=64,
                                                              build_schedule_metadata=False)
    heads, dim = q_fp8.shape[1], q_fp8.shape[2]
    _need(lens.dtype == torch.int32 and tuple(lens.shape) == (rs.rows,) and lens.is_contiguous()
          and lens.data_ptr() == plan.pool_seqlens_per_q.data_ptr() and table.dtype == torch.int32 and table.dim() == 2
          and table.shape[0] == rs.rows and table.stride(1) == 1 and q_fp8.dtype == _FP8 and q_fp8.is_contiguous()
          and weights.dtype == torch.float32 and tuple(weights.shape) == (m, heads) and weights.stride(1) == 1,
          f"rowshard-q: verify call lens {lens.dtype} {tuple(lens.shape)} table {table.dtype} {tuple(table.shape)} q "
          f"{q_fp8.dtype} {tuple(q_fp8.shape)} weights {weights.dtype} {tuple(weights.shape)} is not the plan's layout")
    dg, capturing = sg.ik.deep_gemm, torch.cuda.is_current_stream_capturing()
    warm = _PROCESS.setdefault("idx_warm", set())
    key = (n, heads, dim, dg.get_num_sms())
    _need(not capturing or key in warm, f"idx-nextn: the first next_n={n} paged-MQA call is inside a CUDA-graph capture "
          "(DeepGEMM would JIT there; SGLang's runner warms every graph eagerly first)")
    page_table_1, topk_offsets, _ = st._kpool_fused_topk_mapping(metadata)
    if "rs_probe" not in _PROCESS:
        _need(not capturing, "idx-rowshard: the first sharded call is inside a CUDA-graph capture (its boot probe, an "
              "eager all-gather, runs at the first eager one; SGLang's runner warms every graph eagerly first)")
        _PROCESS["rs_probe"] = _rs_probe(sh, tp, n, heads, dim, v2plan is not None)
    ph_begin(forward_batch, layer_id, lens, tp)
    send = rsq_block(st, sg, q_fp8.view(rs.e - rs.a, n, heads, dim), kv, weights, lens, table[::n], table.shape[1] * 64,
                     seqlens_32, page_table_1, topk_offsets, v2plan is not None, rs.a, rs.e, rs.bq * n)
    ph_end()
    out = rs_gather(tp, send, layer_id, rs.bq * n, rs.rows, rs.pad, forward_batch)
    if not capturing:
        warm.add(key)
        checked = _PROCESS.setdefault("rsq_checked", set())
        if rs.key not in checked:
            _need(rs.ref is not None, f"rowshard-q: the first eager {rs.b}x{n} call carries no stock reference")
            q_ref, w_ref = rs.ref
            rs.ref = None
            same_q = _rsq_same(q_fp8, q_ref[rs.r0:rs.r1]) and wtc_same(weights, w_ref.squeeze(2)[rs.r0:rs.r1])
            want = st._get_topk_paged(forward_batch, layer_id, q_ref, w_ref, metadata)
            same_k = wtc_topk_same(want, out)
            verdicts = tp.all_gather_object((same_q, same_k))
            _need(all(x and y for x, y in verdicts), f"rowshard-q: layer {layer_id} {rs.b}x{n}: the block's query / "
                  f"head weights (bitwise) or the gathered top-k (row sets) differ from stock's on some rank: {verdicts}")
            checked.add(rs.key)
            _log.warning("rowshard-q: layer %d B=%d rank=%d rows [%d, %d): block query == stock's rows, gathered top-k "
                         "== stock's as row sets (ordered %s)", layer_id, rs.b, tp.rank_in_group, rs.r0, rs.r1,
                         torch.equal(want, out))
    return out

IDX_AG_SYMM = True
AGS_LAYER_SLOTS = 48
AGS_MAX_ROWS = 192
AGS_BLOCK, AGS_WARPS = 2048, 4
AGS_TIMEOUT_MS = 5000
AGS_SKEW_S = 0.0005
AGS_PROOF_SEED = 20261171
AGS_PROOF_BS = (32, 90, 96, 128)
_AGS = {}


@triton.jit
def _ags_stage_kernel(src_ptr, buf_ptr, base, n, BLOCK: tl.constexpr, PDL: tl.constexpr):
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = k < n
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(src_ptr + k, mask=m)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(buf_ptr + base + k, x, mask=m)


@triton.jit
def _ags_pull_kernel(buf_ptr, delta_ptr, out_ptr, base, n, rows_n, pad_n, W: tl.constexpr, BLOCK: tl.constexpr,
                     PDL: tl.constexpr):
    r = tl.program_id(0)
    k = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    g = r.to(tl.int64) * n + k
    if PDL:
        tl.extra.cuda.gdc_wait()
    if r < W:
        d = tl.multiple_of(tl.load(delta_ptr + r), 4)
        x = tl.load(buf_ptr + base + d + k, mask=(k < n) & (g < rows_n), other=-1)
        live = (k < n) & (g < pad_n)
    else:
        x = tl.full((BLOCK,), -1, tl.int32)
        live = g < pad_n
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + g, x, mask=live)


class AgsExchange:

    def __init__(self, rank, world, cols, slot, ids, buf, delta, sync, owner):
        self.rank, self.world, self.cols, self.slot, self.buf, self.delta = rank, world, cols, slot, buf, delta
        self.sync, self.owner, self.at, self.last, self.warm = sync, owner, {l: i for i, l in enumerate(ids)}, None, set()
        self.zero = torch.zeros_like(delta)

    def _check(self, send, layer_id, R, rows, pad):
        _need(layer_id in self.at and send.dtype == torch.int32 and send.dim() == 2 and tuple(send.shape) == (R, self.cols)
              and send.is_contiguous() and 0 < R <= AGS_MAX_ROWS and 0 < rows <= self.world * R and rows <= pad,
              f"idx-ag-symm: layer {layer_id} shard {send.dtype} {tuple(send.shape)} (R {R}, rows {rows}, pad {pad}) is "
              f"not a registered slot's [<= {AGS_MAX_ROWS}, {self.cols}] int32 shard")

    def _launch(self, send, base, R, rows, pad, out, delta, dst, dst_base):
        n, C, W = R * self.cols, self.cols, self.world
        tail = pad - W * R
        _ags_stage_kernel[(triton.cdiv(n, AGS_BLOCK),)](send, dst, dst_base, n, BLOCK=AGS_BLOCK, num_warps=AGS_WARPS,
                                                       **_pdl("agsstage"))
        if dst is self.buf:
            self.sync()
        grid = (W + (1 if tail > 0 else 0), triton.cdiv(max(n, tail * C), AGS_BLOCK))
        _ags_pull_kernel[grid](self.buf, delta, out, base, n, rows * C, pad * C, W=W, BLOCK=AGS_BLOCK,
                               num_warps=AGS_WARPS, **_pdl("agspull"))

    def precompile(self, send, layer_id, R, rows, pad):
        self._check(send, layer_id, R, rows, pad)
        scratch = torch.empty((max(pad, R) + 1, self.cols), dtype=torch.int32, device=send.device)
        self._launch(send, self.at[layer_id] * self.slot, R, rows, pad, scratch, self.zero, scratch, 0)
        del scratch

    def gather(self, send, layer_id, R, rows, pad):
        self._check(send, layer_id, R, rows, pad)
        i = self.at[layer_id]
        if self.last == i:
            self.sync()
        out = torch.empty((pad, self.cols), dtype=torch.int32, device=send.device)
        self._launch(send, i * self.slot, R, rows, pad, out, self.delta, self.buf, i * self.slot)
        self.last = i
        return out


def ags_register(idx, layer_id):
    if not IDX_AG_SYMM:
        return
    try:
        from torch._C._distributed_c10d import _SymmetricMemory
        ok = callable(getattr(_SymmetricMemory, "empty_strided_p2p", None)) and callable(
            getattr(_SymmetricMemory, "rendezvous", None))
    except ImportError:
        ok = False
    _need(ok, "idx-ag-symm: this torch has no SymmetricMemory p2p allocation / rendezvous (set IDX_AG_SYMM = False)")
    cols = int(idx.index_topk) + int(idx.index_kpool) - 1
    _need(_AGS.setdefault("cols", cols) == cols, f"idx-ag-symm: layer {layer_id} shard width {cols} != {_AGS['cols']}")
    _need(0 <= layer_id < AGS_LAYER_SLOTS, f"idx-ag-symm: layer id {layer_id} >= AGS_LAYER_SLOTS {AGS_LAYER_SLOTS}")
    _AGS.setdefault("ids", set()).add(layer_id)


def ags_build(tp, dev):
    seen, cols, W, rank = sorted(_AGS.get("ids", ())), _AGS.get("cols"), tp.world_size, tp.rank_in_group
    _need(seen and cols, "idx-ag-symm: no DSA layer registered a slot at prepare")
    ids = list(range(AGS_LAYER_SLOTS))
    slot = -(-AGS_MAX_ROWS * cols // 16) * 16
    n, buf, why = slot * len(ids), None, None
    try:
        from torch._C._distributed_c10d import _SymmetricMemory
        buf = _SymmetricMemory.empty_strided_p2p((n,), [1], torch.int32, dev, tp.cpu_group.group_name)
    except Exception as exc:
        why = f"symmetric allocation: {type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    _need(not bad, f"idx-ag-symm: rank {bad[0][0] if bad else -1} {bad[0][1] if bad else ''} (voted before the "
          f"rendezvous: no rank entered it; set IDX_AG_SYMM = False)")
    ex, why = None, None
    try:
        hdl = _SymmetricMemory.rendezvous(buf)
        off = buf.data_ptr() - int(hdl.buffer_ptrs[rank])
        if hdl.world_size != W or hdl.rank != rank:
            why = f"symmetric handle rank {hdl.rank}/{hdl.world_size} is not TP rank {rank}/{W}"
        elif off < 0 or off % 4 or buf.data_ptr() % 16:
            why = f"block offset {off} / address {buf.data_ptr()} inside its symmetric allocation"
        else:
            peers = [hdl.get_buffer(r, [n], torch.int32, off // 4) for r in range(W)]
            gaps = [p.data_ptr() - buf.data_ptr() for p in peers]
            if peers[rank].data_ptr() != buf.data_ptr() or any(g % 16 for g in gaps):
                why = f"mapped blocks {gaps} are not 16-byte offsets of this rank's block"
            else:
                def sync():
                    hdl.barrier(channel=0, timeout_ms=AGS_TIMEOUT_MS)

                delta = torch.tensor([g // 4 for g in gaps], dtype=torch.int64, device=dev)
                ex = AgsExchange(rank, W, cols, slot, ids, buf, delta, sync, owner=(buf, hdl, peers))
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    _need(not bad, f"idx-ag-symm: rank {bad[0][0] if bad else -1} {bad[0][1] if bad else ''} (set IDX_AG_SYMM = False)")
    _log.warning("idx-ag-symm: tp%d/%d built %d slots x [%d, %d] int32 (%.1f MiB a rank), barrier trap %d ms", rank, W,
                 len(ids), AGS_MAX_ROWS, cols, n * 4 / 2 ** 20, AGS_TIMEOUT_MS)
    return ex


def _ags_nccl(tp, send, R, rows, pad):
    width = tp.world_size * R
    recv = send.new_empty((max(width, pad), send.shape[1]))
    tp.all_gather_into_tensor(recv[:width], send)
    if pad > rows:
        recv[rows:pad].fill_(-1)
    return recv[:pad]


def _ags_live(ex, tp, send, layer_id, R, rows, pad, want, skew=False):
    ex.precompile(send, layer_id, R, rows, pad)
    torch.cuda.synchronize(send.device)
    tp.all_gather_object(None)
    if skew:
        time.sleep(tp.rank_in_group * AGS_SKEW_S)
    got = ex.gather(send, layer_id, R, rows, pad)
    return _rsq_same(got, want)


def ags_prove(ex, tp, dev, n=6):
    gen = torch.Generator(device=dev).manual_seed(AGS_PROOF_SEED + tp.rank_in_group)
    W, ids, ok, cases = tp.world_size, sorted(ex.at), True, 0

    def shard(b):
        bq = -(-b // W)
        a, e = min(b, tp.rank_in_group * bq), min(b, (tp.rank_in_group + 1) * bq)
        s = torch.randint(-1, 1 << 30, (bq * n, ex.cols), generator=gen, device=dev, dtype=torch.int32)
        s[(e - a) * n:] = -1
        return s, bq * n

    for j, b in enumerate(AGS_PROOF_BS):
        for pad in (b * n, b * n + 2 * n):
            s, R = shard(b)
            ok = _ags_live(ex, tp, s, ids[j % len(ids)], R, b * n, pad, _ags_nccl(tp, s, R, b * n, pad), skew=True) and ok
            cases += 1
    b = AGS_PROOF_BS[-1]
    runs = [shard(b) for _ in range(len(ids) + 1)]
    wants = [_ags_nccl(tp, s, R, b * n, b * n) for s, R in runs]
    for s, R in runs[:1]:
        ex.precompile(s, ids[0], R, b * n, b * n)
    torch.cuda.synchronize(dev)
    tp.all_gather_object(None)
    order = ids + [ids[-1]]
    gots = [ex.gather(s, lid, R, b * n, b * n) for (s, R), lid in zip(runs, order)]
    ok = all(_rsq_same(g, w) for g, w in zip(gots, wants)) and ok
    cases += len(gots)
    torch.cuda.synchronize(dev)
    verdicts = tp.all_gather_object(bool(ok))
    ex.last = None
    _need(all(verdicts), f"idx-ag-symm: build proof: the exchange differs from pynccl's gather on some rank {verdicts} "
          "(set IDX_AG_SYMM = False)")
    _log.warning("idx-ag-symm: tp%d/%d build proof %d exchanges == pynccl gather, bitwise (skew drill %.1f ms/rank)",
                 tp.rank_in_group, W, cases, AGS_SKEW_S * 1e3)


def ags_gather(tp, send, layer_id, R, rows, pad):
    if not IDX_AG_SYMM:
        return _ags_nccl(tp, send, R, rows, pad)
    ex, key = _AGS.get("ex"), (R, rows, pad)
    if torch.cuda.is_current_stream_capturing():
        _need(ex is not None and key in ex.warm, f"idx-ag-symm: a {rows}-row ({pad} padded) gather inside a CUDA-graph "
              "capture before its eager check (SGLang's runner warms every graph shape eagerly first)")
        return ex.gather(send, layer_id, R, rows, pad)
    if ex is None:
        ex = _AGS["ex"] = ags_build(tp, send.device)
        ags_prove(ex, tp, send.device)
    out = _ags_nccl(tp, send, R, rows, pad)
    if key not in ex.warm:
        same = _ags_live(ex, tp, send, layer_id, R, rows, pad, out)
        verdicts = tp.all_gather_object(bool(same))
        ex.last = None
        _need(all(verdicts), f"idx-ag-symm: layer {layer_id} {rows} rows ({pad} padded): the exchange of the live shard "
              f"differs from pynccl's gather on some rank {verdicts} (set IDX_AG_SYMM = False)")
        ex.warm.add(key)
        _log.warning("idx-ag-symm: layer %d rows=%d pad=%d shard=%d: live exchange == pynccl gather, bitwise (graph-"
                     "served from here)", layer_id, rows, pad, R)
    return out

KDA_DIRECT = True
_KD = {}


def _kd_note(what, rows):
    if what not in _KD:
        _KD[what] = rows
        _log.warning("kda-direct: first %s verify call (%s rows; KDA_DIRECT=%s)", what, rows, KDA_DIRECT)


def kd_verify(lin, sg, layer, forward_batch, mixed_qkv, a, b):
    rows = mixed_qkv.shape[0]
    if sg.lu.pp_spec_stable_rows_enabled():
        _kd_note("stock (SGLANG_ENABLE_PP_SPEC)", rows)
        return sg.KDA._forward_target_verify(lin, layer, forward_batch, mixed_qkv, a, b)
    fm = lin.forward_metadata
    query_start_loc = fm.query_start_loc
    cache_indices = fm.mamba_cache_indices
    retrieve_parent_token = fm.retrieve_parent_token
    mamba_cache_params = lin.state_layer_cache(layer.layer_id)
    conv_states = mamba_cache_params.conv[0]
    ssm_states = mamba_cache_params.temporal
    intermediate_state_cache = getattr(mamba_cache_params, "intermediate_ssm", None)
    replayssm_rawv = getattr(mamba_cache_params, "replayssm_rawv", None)
    if intermediate_state_cache is None and replayssm_rawv is None:
        return sg.KDA._forward_target_verify(lin, layer, forward_batch, mixed_qkv, a, b)
    intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
    intermediate_state_indices = lin.verify_intermediate_state_indices
    draft_token_num = forward_batch.spec_info.draft_token_num
    ragged_layout = forward_batch.ragged_verify_layout
    if not lin._can_run_dspark_cutedsl_mtp(
            layer=layer, mixed_qkv=mixed_qkv, a=a, b=b, draft_token_num=draft_token_num, ragged_layout=ragged_layout,
            conv_states=conv_states, ssm_states=ssm_states, intermediate_state_cache=intermediate_state_cache,
            intermediate_conv_window_cache=intermediate_conv_window_cache, retrieve_parent_token=retrieve_parent_token,
            replayssm_rawv=replayssm_rawv):
        _kd_note("stock (the DSpARK CuTe route declines)", rows)
        return sg.KDA._forward_target_verify(lin, layer, forward_batch, mixed_qkv, a, b)
    fork = KDA_DYN and getattr(layer, "_k3_onorm_gate", None) is None
    return (kw_run_dspark if fork else sg.KDA._run_dspark_cutedsl_mtp)(
        layer=layer, mixed_qkv=mixed_qkv, a=a, b=b, conv_states=conv_states, ssm_states=ssm_states,
        intermediate_state_cache=intermediate_state_cache, intermediate_conv_window_cache=intermediate_conv_window_cache,
        intermediate_state_indices=intermediate_state_indices, cache_indices=cache_indices,
        query_start_loc=query_start_loc, replayssm_rawv=replayssm_rawv, replayssm_rawk=mamba_cache_params.replayssm_rawk,
        replayssm_g=mamba_cache_params.replayssm_g, replayssm_beta=mamba_cache_params.replayssm_beta)


def _kd_rows(lin, layer_id):
    fm, p = lin.forward_metadata, lin.state_layer_cache(layer_id)
    bs = fm.query_start_loc.shape[0] - 1
    ci = fm.mamba_cache_indices[:bs]
    slots = ci[ci >= 0].long()
    rows = lin.verify_intermediate_state_indices[:bs].long()
    got = [getattr(p, "replayssm_rawv", None), getattr(p, "replayssm_rawk", None), getattr(p, "replayssm_g", None),
           getattr(p, "replayssm_beta", None)]
    out = [None if t is None else t.index_select(0, slots).clone() for t in got]
    out.append(p.intermediate_conv_window[0].index_select(0, rows).clone())
    isc = getattr(p, "intermediate_ssm", None)
    out.append(None if isc is None else isc.index_select(0, rows).clone())
    return out


def kd_probe(attn, lin, bp, fb, mixed_qkv, a, b):
    _need(not torch.cuda.is_current_stream_capturing(), "kda-direct: a boot probe (the first direct verify call, or the "
          "KDA-DYN fork's first launch of a compile variant) is inside a CUDA-graph capture (the probe syncs; SGLang's "
          "runner warms every graph shape eagerly first)")
    s, sg = attn.stock, attn.sg
    want = s(fb, mixed_qkv=mixed_qkv, a=a, b=b)
    ref = _kd_rows(lin, s.layer_id)
    n0 = len(_KW.get("admitted", ()))
    _KW["poison"] = KDA_DYN
    _KW["probing"] = True
    try:
        got = sg.KDA.forward_extend(bp, q=None, k=None, v=None, layer=s, forward_batch=fb, save_kv_cache=True,
                                    mixed_qkv=mixed_qkv, a=a, b=b)
    finally:
        _KW.pop("probing", None)
    now = _kd_rows(lin, s.layer_id)
    _KW.pop("poison", None)
    same = _kdae_eq(want, got) and all(x is y or (x is not None and y is not None and _kdae_eq(x, y))
                                       for x, y in zip(ref, now))
    del want, ref, now
    _need(all(attn.tp.all_gather_object(same)) if attn.tp is not None else same, "kda-direct: the direct verify call "
          "differs from stock's (output / rings / conv-window scratch / intermediate states) on some TP rank")
    _KD["proved"] = True
    _log.warning("kda-direct: probe layer %d, %d verify rows: direct route == stock's call, bitwise (output + written "
                 "state rows; fork variants admitted (width, 64-bit-stride args): %s)", s.layer_id, mixed_qkv.shape[0],
                 _KW.get("admitted", [])[n0:])
    return got


class KdDirect(torch.nn.Module):

    def __init__(self, stock, tp):
        super().__init__()
        from sglang.srt.layers import radix_linear_attention as rla
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hy
        from sglang.srt.layers.attention.linear import kda_backend as kb, utils as lu
        from sglang.srt.model_executor import forward_context as fc
        _need(type(stock) is rla.RadixLinearAttention and kb.select_verify_intermediate_state_indices
              is lu.select_verify_intermediate_state_indices and callable(lu.pp_spec_stable_rows_enabled)
              and callable(getattr(kb.KDAAttnBackend, "_run_dspark_cutedsl_mtp", None)),
              f"kda-direct: the KDA attention is {type(stock).__name__}, or kda_backend's verify helpers are not "
              "linear/utils.py's")
        object.__setattr__(self, "stock", stock)
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "sg", types.SimpleNamespace(rla=rla, Hybrid=hy.HybridLinearAttnBackend,
                                                             KDA=kb.KDAAttnBackend, lu=lu, backend=fc.get_attn_backend))

    def forward(self, forward_batch, mixed_qkv, a, b):
        fb, sg, s = forward_batch, self.sg, self.stock
        _need(fb.forward_mode.is_target_verify(), "kda-direct: a non-verify call reached the KDA verify proxy")
        rows = mixed_qkv.shape[0]
        if (not KDA_DIRECT or sg.rla.is_in_prefill_graph_scope() or fb.forward_mode.is_idle()
                or fb.logical_forward_mode.is_decode()):
            _kd_note("stock (route)", rows)
            return s(fb, mixed_qkv=mixed_qkv, a=a, b=b)
        hyb = sg.backend()
        lin = getattr(hyb, "linear_attn_backend", None)
        _need(type(hyb) is sg.Hybrid and type(lin) is sg.KDA and not hyb._is_full_attn(s)
              and getattr(hyb.forward, "__func__", None) is sg.Hybrid.forward
              and getattr(hyb.forward_extend, "__func__", None) is sg.Hybrid.forward_extend
              and getattr(lin.forward_extend, "__func__", None) is sg.KDA.forward_extend
              and getattr(lin._forward_target_verify, "__func__", None) is sg.KDA._forward_target_verify
              and not s._forward_hooks and not s._forward_pre_hooks,
              "kda-direct: the attention backend is not the hybrid KDA backend stock's verify route runs on (a rebound "
              "dispatch, a full-attention layer id or module hooks)")
        bp = copy.copy(lin)
        bp._forward_target_verify = functools.partial(kd_verify, lin, sg)
        if "proved" not in _KD:
            return kd_probe(self, lin, bp, fb, mixed_qkv, a, b)
        _kd_note("direct", rows)
        try:
            return sg.KDA.forward_extend(bp, q=None, k=None, v=None, layer=s, forward_batch=fb, save_kv_cache=True,
                                         mixed_qkv=mixed_qkv, a=a, b=b)
        except KwFresh:
            return kd_probe(self, lin, bp, fb, mixed_qkv, a, b)


def kd_prepare(layer, tp):
    if not KDA_DIRECT:
        return None
    from sglang.srt.models import glm5_next
    attn = layer._modules.get("self_attn")
    if type(attn) is not glm5_next.Glm5NextLinearAttention:
        return None
    ra = attn._modules.get("attn")
    _need(ra is not None and not attn._forward_hooks and not attn._forward_pre_hooks,
          f"kda-direct: layer {layer.layer_id}: the KDA attention has no ``attn`` child, or module hooks")
    return types.SimpleNamespace(nn=None, proxy=pj_proxy(layer, {"self_attn": pj_proxy(attn, {"attn": KdDirect(ra, tp)})}))

KDA_OPROJ_SP = False
KO_CLASS = "bitwise"
KO_ROW_TOL = 1e-2
KO_CHECKS = 1
KO_STRICT = False
KO_BAR = "side"
KO_GEMM = "direct"
KO_ADD = "torch"
KO_NORM = "one"
KO_SKIP_LAYERS = (0, 1, 2)
KO_MIN_PREFIX = 32768
KO_MIN_ROWS = 8193
_KO = {"seen": set(), "plans": {}}


def ko_attach(state, zc):
    attn, ffn = state.attn, state.ffn
    return


IDX_PLAN_HOIST = True
PH_EAGER_CHECKS = 64
_PH = {"fb": None, "memo": {}, "on": None, "checks": 0, "ids": set()}
_PH_NONE = object()


def ph_begin(fb, layer_id, lens, tp):
    if not IDX_PLAN_HOIST:
        return
    if layer_id == min(_PH["ids"] or {layer_id}) or _PH["fb"] is not fb:
        _PH["fb"], _PH["memo"] = fb, {}
    _PH["on"] = (lens.untyped_storage().data_ptr(), tp, torch.cuda.current_stream(lens.device).cuda_stream)


def ph_end():
    _PH["on"] = None


def _ph_get(kind, t, extra, build, same=None):
    on = _PH["on"]
    if on is None or t.untyped_storage().data_ptr() != on[0] or torch.cuda.current_stream(t.device).cuda_stream != on[2]:
        return build()
    key = (kind, t.data_ptr(), tuple(t.shape), tuple(t.stride()), extra)
    memo = _PH["memo"]
    hit = memo.get(key, _PH_NONE)
    if hit is _PH_NONE:
        hit = memo[key] = build()
        return hit
    if not torch.cuda.is_current_stream_capturing() and _PH["checks"] < PH_EAGER_CHECKS:
        _PH["checks"] += 1
        if same is not None:
            ok = bool(same(hit))
        else:
            fresh = build()
            ok = (fresh is None and hit is None) or (fresh is not None and hit is not None and fresh.dtype == hit.dtype
                                                     and fresh.shape == hit.shape and torch.equal(fresh, hit))
        tp = on[1]
        _need(all(tp.all_gather_object(ok)) if tp is not None else ok, f"idx-plan-hoist: the forward's {kind} for "
              f"lens {tuple(t.shape)} differs from a fresh build on some rank (set IDX_PLAN_HOIST = False)")
    return hit


def ph_sched(dg, lens):
    sms = dg.get_num_sms()
    return _ph_get("dg", lens, sms, lambda: dg.get_paged_mqa_logits_metadata(lens, 64, sms))


def ph_plan(kif, lens):
    return _ph_get("v2", lens, None, lambda: kif.build_kpool_topk_v2_plan(lens), lambda hit: _ph_v2_same(kif, lens, hit))


def _ph_v2_same(kif, lens, hit):
    if hit is None:
        return kif.build_kpool_topk_v2_plan(lens) is None
    ref = torch.full_like(hit, -1)
    kif.refresh_kpool_topk_v2_plan(plan=ref, pool_lens=lens)
    n = int(ref[0, 1])
    if n == -1:
        return True
    if not (0 <= n < ref.shape[0] and torch.equal(ref[0], hit[0])):
        return False
    a, b = ref[1:1 + n], hit[1:1 + n]
    return torch.equal(a[a[:, 0].argsort()], b[b[:, 0].argsort()])

PDL_NODE = False
PDL_KERNELS = ("fastpost", "rotq")
_PDLC = {}


def _pdl(name):
    return {"PDL": False}

IDX_WPROJ_TC = True
WTC_TOL = 1e-4
WTC_W_TOL = 1e-3
WTC_TOPK_MAX = 1e-3
WTC_CHECKS = 8
WTC_STRICT = False
_WTC = {"checks": {}, "facts": {}}


def wtc_facts(idx):
    if not IDX_WPROJ_TC:
        return None
    lid = getattr(idx, "layer_id", None)
    hit = _WTC["facts"].get(lid)
    if hit is not None and hit[0] is idx:
        return hit[1]
    from sglang.srt.layers import linear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    wp = getattr(idx, "weights_proj", None)
    w = getattr(wp, "weight", None)
    ok = (type(wp) is linear.ReplicatedLinear and type(getattr(wp, "quant_method", None)) is UnquantizedLinearMethod
          and getattr(wp, "bias", None) is None and torch.is_tensor(w) and w.dtype == torch.float32
          and tuple(w.shape) == (getattr(idx, "n_heads", None), getattr(idx, "hidden_size", None))
          and not wp._forward_hooks and not wp._forward_pre_hooks)
    why = None if ok else (f"weights_proj {type(wp).__name__} {getattr(w, 'dtype', None)} "
                           f"{tuple(w.shape) if torch.is_tensor(w) else None} is not the unquantized fp32 "
                           f"[{getattr(idx, 'n_heads', None)}, {getattr(idx, 'hidden_size', None)}] ReplicatedLinear "
                           "stock projects with")
    _need(why is None or not WTC_STRICT, f"idx-wproj-tc: layer {lid} {why} (WTC_STRICT)")
    _WTC["facts"][lid] = (idx, why)
    if why is not None:
        _log.warning("idx-wproj-tc: layer %s REFUSED, stock fp32 projection: %s", lid, why)
    return why


def wtc_weight(st):
    hit = _WTC.get(st.layer_id)
    if hit is not None and hit[0] is st:
        return hit[1]
    if wtc_facts(st) is not None:
        return None
    if torch.cuda.is_current_stream_capturing():
        if ("cap", st.layer_id) not in _WTC:
            _WTC[("cap", st.layer_id)] = True
            _log.warning("idx-wproj-tc: layer %d's first head-gate call is inside a CUDA-graph capture: stock projection "
                         "in that graph (the weight copy + boot check run eagerly)", st.layer_id)
        return None
    w = st.weights_proj.weight
    wb = w.detach().to(torch.bfloat16).contiguous()
    ok = bool(torch.equal(wb.float(), w))
    _WTC[st.layer_id] = (st, wb if ok else None)
    _log.warning("idx-wproj-tc: layer %d weights_proj %s %s: bf16-exact %s -> %s", st.layer_id, w.dtype,
                 tuple(w.shape), ok, "tensor-core head gates" if ok else "REFUSED, stock fp32 projection")
    return wb if ok else None


def wtc_project(st, x):
    if not (IDX_WPROJ_TC and "off" not in _WTC and torch.is_tensor(x) and x.dtype == torch.bfloat16 and x.dim() == 2
            and x.shape[1] == st.hidden_size and x.stride(1) == 1):
        return None
    wb = wtc_weight(st)
    if wb is None:
        return None
    hw = torch.mm(x, wb.t(), out_dtype=torch.float32) * st.n_heads ** -0.5
    size = "dec" if x.shape[0] <= DEC_MAX_ROWS else "pf"
    done = _WTC["checks"].get(size, 0)
    if done < WTC_CHECKS and x.shape[0] and not torch.cuda.is_current_stream_capturing():
        _WTC["checks"][size] = done + 1
        ref = st._project_and_scale_head_gates(x)
        bound = WTC_TOL * float(ref.abs().max())
        diff = float((hw - ref).abs().max()) if hw.shape == ref.shape else float("inf")
        if not (hw.shape == ref.shape and hw.dtype == ref.dtype and bool(torch.isfinite(hw).all()) and diff <= bound):
            why = (f"layer {st.layer_id} rows={x.shape[0]}: TC head gates differ from stock's by {diff:.3g} > "
                   f"{bound:.3g}")
            _need(not WTC_STRICT, f"idx-wproj-tc: {why} (WTC_STRICT; set IDX_WPROJ_TC = False)")
            _WTC["off"] = why
            _log.warning("idx-wproj-tc: OFF for this process (stock fp32 projection from here; this call served "
                         "stock's): %s", why)
            return ref
        _log.warning("idx-wproj-tc: check %s/%d layer %d rows=%d: max |TC - stock| %.3g <= %.3g (bitwise %s)", size,
                     done + 1, st.layer_id, x.shape[0], diff, bound, torch.equal(hw, ref))
    return hw


def wtc_pre(sh, x):
    rs, st = sh._rs, sh._s
    if rs is not None:
        _need(x.shape[0] >= rs.r1, f"idx-wproj-tc: x {tuple(x.shape)} does not cover the block [{rs.r0}, {rs.r1})")
        hw = wtc_project(st, x[rs.r0:rs.r1])
        if hw is not None:
            rs.hw_block = True
            return hw
    else:
        hw = wtc_project(st, x)
        if hw is not None:
            return hw
    return st._project_and_scale_head_gates(x)


def wtc_same(a, b):
    if not IDX_WPROJ_TC:
        return _rsq_same(a, b)
    if a.dtype != b.dtype or a.shape != b.shape or not bool(torch.isfinite(a).all()):
        return False
    return float((a - b).abs().max()) <= WTC_W_TOL * float(b.abs().max()) if a.numel() else True


def wtc_topk_same(want, out):
    if want.dtype != out.dtype or want.shape != out.shape:
        return False
    if not IDX_WPROJ_TC:
        return torch.equal(want.sort(1).values, out.sort(1).values)
    rows = torch.arange(want.shape[0], device=want.device, dtype=torch.int64)[:, None] << 33
    ka, kb = rows + want.to(torch.int64) + 1, rows + out.to(torch.int64) + 1
    valid = want >= 0
    miss = int((~torch.isin(ka[valid], kb[out >= 0])).sum())
    return miss <= WTC_TOPK_MAX * max(1, int(valid.sum()))


class _WvShim(_IdxShim):
    _get_topk_paged = property(lambda self: self._s._get_topk_paged)


class WvNN(IdxNN):
    def __init__(self, stock, sg):
        super().__init__(stock, sg)
        object.__setattr__(self, "shim", _WvShim(stock, sg))
    def forward(self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
        st, sg = self.stock, self.sg
        method = st._forward_method
        why = None
        if method is None or not forward_batch.forward_mode.is_target_verify():
            why = "mode"
        elif not (getattr(method, "__func__", None) is sg.Indexer.forward_cuda and method.__self__ is st
                  and not st.is_torch_compile and sg.fused_op.get_fused_op_backend() is None and not st._forward_hooks
                  and not st._forward_pre_hooks):
            why = "dispatch"
        if why is not None:
            if why != "mode" and ("wv", why) not in _WTC:
                _WTC[("wv", why)] = True
                _log.warning("idx-wproj-tc: layer %d TP2 verify: stock's indexer call (its dispatch is not "
                             "IndexerKPool.forward_cuda: forced backend, torch.compile mode or hooks)", layer_id)
            return st(x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch, layer_id=layer_id,
                      return_indices=return_indices)
        return sg.Indexer._forward_cuda_impl(self.shim, x, q_lora, positions, forward_batch, layer_id, return_indices)


def wtc_verify_nn(idx, layer_id):
    if not IDX_WPROJ_TC or idx is None:
        return None
    from sglang.kernels import fused_op
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as ik
    if type(idx) is not ik.IndexerKPool or getattr(idx, "layer_id", None) != layer_id or wtc_facts(idx) is not None:
        if type(idx) is not ik.IndexerKPool or getattr(idx, "layer_id", None) != layer_id:
            _log.warning("idx-wproj-tc: layer %d TP2 verify: indexer %s is not this layer's IndexerKPool: stock's",
                         layer_id, type(idx).__name__)
        return None
    return WvNN(idx, types.SimpleNamespace(ik=ik, Indexer=ik.IndexerKPool, fused_op=fused_op, tp=None))

SPLIT_PF = True
SPLIT_PF_MIN_ROWS = 2048
SPLIT_PF_MAX_ROWS = 16384
SPLIT_PF_ROUND = 8192
SPLIT_HEADS = 64
SPLIT_TIMEOUT_MS = 5000
SPLIT_TIMEOUT_EAGER_MS = 600000
SPLIT_ROW_TOL = 6e-2
SPLIT_BIT_FRAC = 0.0
SPLIT_CHECKS = 2
SPLIT_STRICT = False
SX_BLOCK, SX_WARPS = 4096, 8
SX_PROOF_ROWS = 48
SX_WARM_ROWS = 1024
SX_SKEW_S = 0.0005
SX_PROOF_SEED = 20261180
SX_D = 4 * BLOCK
_SX = {"checks": {}, "seen": set()}


@triton.jit(do_not_specialize=["row0", "nq", "nk", "qb"])
def _sx_pack_kernel(q_ptr, k_ptr, qo_ptr, ko_ptr, row0, nq, nk, H, sqt, sqh, skt, qb, D: tl.constexpr,
                    R: tl.constexpr):
    pid, c = tl.program_id(0), tl.arange(0, D)[None, :]
    if pid < qb:
        r = pid.to(tl.int64) * R + tl.arange(0, R)
        m = (r < nq)[:, None]
        x = tl.load(q_ptr + ((row0 + r // H) * sqt + (r % H) * sqh)[:, None] + c, mask=m)
        tl.store(qo_ptr + r[:, None] * D + c, x.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=m)
    else:
        r = (pid - qb).to(tl.int64) * R + tl.arange(0, R)
        m = (r < nk)[:, None]
        x = tl.load(k_ptr + r[:, None] * skt + c, mask=m)
        tl.store(ko_ptr + r[:, None] * D + c, x.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=m)


@triton.jit
def _sx_qget_kernel(buf_ptr, delta_ptr, dst_ptr, src_off, n, W: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(1)
    f = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = f < n
    d = tl.multiple_of(tl.load(delta_ptr + s), 4)
    x = tl.load(buf_ptr + d + src_off + f, mask=m)
    tl.store(dst_ptr + (f // C) * (W * C) + s * C + f % C, x, mask=m)


@triton.jit
def _sx_oget_kernel(buf_ptr, delta_ptr, dst_ptr, obase, hoff, a0, a1, a2, a3, n0, n1, n2, n3, C: tl.constexpr,
                    RW: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(1)
    a = tl.where(s == 0, a0, tl.where(s == 1, a1, tl.where(s == 2, a2, a3)))
    n = tl.where(s == 0, n0, tl.where(s == 1, n1, tl.where(s == 2, n2, n3)))
    f = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = f < n
    d = tl.multiple_of(tl.load(delta_ptr + s), 4)
    x = tl.load(buf_ptr + d + obase + (f // C) * RW + hoff + f % C, mask=m)
    tl.store(dst_ptr + a + f, x, mask=m)


def sx_pack(q, k, qo, row0, rows, k8=None):
    H, D = q.shape[1], q.shape[2]
    nq, nk = rows * H, (k.shape[0] if k8 is not None else 0)
    qb = triton.cdiv(nq, PK_R)
    grid = qb + triton.cdiv(nk, PK_R)
    if grid:
        _sx_pack_kernel[(grid,)](q, k, qo, qo if k8 is None else k8, row0, nq, nk, H, q.stride(0), q.stride(1),
                                 k.stride(0), qb, D=D, R=PK_R, num_warps=PK_WARPS)


def sx_pack_probe(dev, H, D):
    _need(not torch.cuda.is_current_stream_capturing(), "split-attn: the first pack is inside a CUDA-graph capture")
    x = torch.arange(1 << 16, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16).view(-1, D)
    q = x.expand(H, *x.shape).contiguous().transpose(0, 1)
    want_q, want_k = pk_pack(q, x.unsqueeze(1))
    T, cut = q.shape[0], q.shape[0] // 3
    got, k8 = torch.empty((T, H, D), dtype=_FP8, device=dev), torch.empty((T, 1, D), dtype=_FP8, device=dev)
    sx_pack(q, x.unsqueeze(1), got[cut:], cut, T - cut, k8)
    sx_pack(q, x.unsqueeze(1), got[:cut], 0, cut)
    _need(torch.equal(want_q.view(torch.uint8), got.view(torch.uint8)) and torch.equal(want_k.view(torch.uint8),
          k8.view(torch.uint8)), "split-attn: the stage pack differs from FP8-PACK's pk_pack on the BF16 pattern table")
    _SX.setdefault("pk", set()).add(dev)


class SxExchange:

    def __init__(self, rank, world, hl, rows, buf, delta, sync, counter, owner):
        self.rank, self.world, self.hl, self.O, self.buf, self.delta = rank, world, hl, rows, buf, delta
        self.sync, self.counter, self.owner, self.warm, self.kwarm = sync, counter, owner, set(), set()
        self.cq, self.co = hl * SX_D // 4, hl * SX_D // 2
        self.qwords, self.owords = world * rows * self.cq, rows * SPLIT_HEADS * SX_D // 2
        self.zero = torch.zeros_like(delta)

    def qslot(self, at, n):
        return self.buf[at * self.cq:(at + n) * self.cq].view(_FP8).view(n, self.hl, SX_D)

    def ostage(self, n):
        return self.buf[self.qwords:self.qwords + n * SPLIT_HEADS * SX_D // 2].view(torch.bfloat16).view(
            n, 1, SPLIT_HEADS, SX_D)

    def qget(self, dst, at, n, delta=None):
        N = n * self.cq
        if N:
            _sx_qget_kernel[(triton.cdiv(N, SX_BLOCK), self.world)](
                self.buf, self.delta if delta is None else delta, dst.view(torch.int32), at * self.cq, N,
                W=self.world, C=self.cq, BLOCK=SX_BLOCK, num_warps=SX_WARPS)

    def oget(self, out, rng, delta=None):
        a = [rng[s][0] * self.co if s < self.world else 0 for s in range(4)]
        n = [rng[s][1] * self.co if s < self.world else 0 for s in range(4)]
        if max(n):
            _sx_oget_kernel[(triton.cdiv(max(n), SX_BLOCK), self.world)](
                self.buf, self.delta if delta is None else delta, out.view(torch.int32), self.qwords,
                self.rank * self.co, *a, *n, C=self.co, RW=SPLIT_HEADS * SX_D // 2, BLOCK=SX_BLOCK,
                num_warps=SX_WARPS)

    def precompile(self, dev):
        q64 = torch.empty((2, 1, self.world * self.hl, SX_D), dtype=_FP8, device=dev)
        out = torch.empty((2 * self.world, 1, self.hl, SX_D), dtype=torch.bfloat16, device=dev)
        self.qget(q64, 0, 2, self.zero)
        self.oget(out, [(2 * s, 2) for s in range(self.world)], self.zero)
        del q64, out


def sx_build(tp, dev, hl, rows):
    W, rank = tp.world_size, tp.rank_in_group
    n = W * rows * hl * SX_D // 4 + rows * SPLIT_HEADS * SX_D // 2
    buf, why = None, None
    try:
        from torch._C._distributed_c10d import _SymmetricMemory
        buf = _SymmetricMemory.empty_strided_p2p((n,), [1], torch.int32, dev, tp.cpu_group.group_name)
    except Exception as exc:
        why = f"symmetric allocation of {n * 4 / 2 ** 20:.0f} MiB: {type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    if bad:
        return _sx_refuse(f"rank {bad[0][0]} {bad[0][1]} (voted before the rendezvous: no rank entered it)")
    ex, why = None, None
    try:
        hdl = _SymmetricMemory.rendezvous(buf)
        off = buf.data_ptr() - int(hdl.buffer_ptrs[rank])
        if hdl.world_size != W or hdl.rank != rank:
            why = f"symmetric handle rank {hdl.rank}/{hdl.world_size} is not TP rank {rank}/{W}"
        elif off < 0 or off % 4 or buf.data_ptr() % 16:
            why = f"block offset {off} / address {buf.data_ptr()} inside its symmetric allocation"
        else:
            peers = [hdl.get_buffer(r, [n], torch.int32, off // 4) for r in range(W)]
            gaps = [p.data_ptr() - buf.data_ptr() for p in peers]
            if peers[rank].data_ptr() != buf.data_ptr() or any(g % 16 for g in gaps):
                why = f"mapped blocks {gaps} are not 16-byte offsets of this rank's block"
            else:
                def sync():
                    hdl.barrier(channel=0, timeout_ms=SPLIT_TIMEOUT_MS if torch.cuda.is_current_stream_capturing()
                                else SPLIT_TIMEOUT_EAGER_MS)

                from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
                delta = torch.tensor([g // 4 for g in gaps], dtype=torch.int64, device=dev)
                counter = make_persistent_multi_ctas_kv_counter_buffer(dev, SPLIT_HEADS, rows)
                ex = SxExchange(rank, W, hl, rows, buf, delta, sync, counter, owner=(buf, hdl, peers))
                ex.precompile(dev)
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    if bad:
        return _sx_refuse(f"rank {bad[0][0]} {bad[0][1]}")
    sx_prove(ex, tp, dev)
    _log.warning("split-attn: tp%d/%d built a %d-row stage (Hl %d: q %d + o %d words, %.1f MiB a rank), barrier trap %d "
                 "ms captured / %d ms eager", rank, W, rows, hl, ex.qwords, ex.owords, n * 4 / 2 ** 20, SPLIT_TIMEOUT_MS,
                 SPLIT_TIMEOUT_EAGER_MS)
    return ex


def _sx_refuse(why):
    _need(not SPLIT_STRICT, f"split-attn: exchange build refused: {why} (SPLIT_STRICT; set SPLIT_PF / SPLIT_DEC False)")
    _SX["off"] = why
    _log.warning("split-attn: exchange build REFUSED on every rank (voted), PackAttn's path stays: %s", why)
    return None


def sx_prove(ex, tp, dev):
    W, me, n = ex.world, ex.rank, min(SX_PROOF_ROWS, ex.O)
    gen = torch.Generator(device=dev).manual_seed(SX_PROOF_SEED + me)
    q = torch.randint(-2 ** 31, 2 ** 31 - 1, (W, n, ex.cq), generator=gen, device=dev, dtype=torch.int32)
    o = torch.randint(-2 ** 31, 2 ** 31 - 1, (n, SPLIT_HEADS * SX_D // 2), generator=gen, device=dev, dtype=torch.int32)
    qa, oa = torch.empty((W,) + tuple(q.shape), dtype=torch.int32, device=dev), torch.empty(
        (W,) + tuple(o.shape), dtype=torch.int32, device=dev)
    tp.all_gather_into_tensor(qa.view(-1), q.view(-1))
    tp.all_gather_into_tensor(oa.view(-1), o.view(-1))
    want_q = qa[:, me].permute(1, 0, 2).contiguous()
    want_o = oa.view(W, n, SPLIT_HEADS, SX_D // 2)[:, :, me * ex.hl:(me + 1) * ex.hl].reshape(W * n, ex.co)
    for s in range(W):
        ex.buf[s * n * ex.cq:(s + 1) * n * ex.cq].copy_(q[s].view(-1))
    ex.buf[ex.qwords:ex.qwords + o.numel()].copy_(o.view(-1))
    q64 = torch.empty((n, 1, W * ex.hl, SX_D), dtype=_FP8, device=dev)
    out = torch.empty((W * n, 1, ex.hl, SX_D), dtype=torch.bfloat16, device=dev)
    torch.cuda.synchronize(dev)
    tp.all_gather_object(None)
    time.sleep(me * SX_SKEW_S)
    ex.sync()
    ex.qget(q64, me * n, n)
    ex.sync()
    ex.oget(out, [(s * n, n) for s in range(W)])
    torch.cuda.synchronize(dev)
    ok = _rsq_same(q64.view(torch.int32).view(n, W, ex.cq), want_q) and _rsq_same(out.view(torch.int32).view(W * n,
                                                                                                        ex.co), want_o)
    verdicts = tp.all_gather_object(bool(ok))
    _need(all(verdicts), f"split-attn: build proof: the exchange differs from pynccl's gather on some rank {verdicts} "
          "(set SPLIT_PF / SPLIT_DEC False)")
    _log.warning("split-attn: tp%d/%d build proof: q + o pulls of %d rows == pynccl gather, bitwise (skew %.1f ms/rank)",
                 me, W, n, SX_SKEW_S * 1e3)


def sx_env():
    sg = _SX.get("sg")
    if sg is None:
        import flashinfer.decode
        from sglang.srt.distributed import parallel_state as ps
        from sglang.srt.layers.attention import dsa_backend
        tp = ps.get_tp_group()
        _need(ps.get_attn_tp_group().world_size == tp.world_size and tp.world_size in (2, 4),
              f"split-attn: TP {tp.world_size} / attention TP {ps.get_attn_tp_group().world_size} (attention DP or a "
              "world other than 2 / 4)")
        sg = _SX["sg"] = types.SimpleNamespace(tp=tp, db=dsa_backend, fid=flashinfer.decode, envs=dsa_backend.envs)
    return sg


def sx_backend(env):
    backend = env.get_attn_backend()
    return backend.full_attn_backend if type(backend) is env.Hybrid else backend


SX_PF_ROLES = ("small", "fp8proxy")
SX_DEC_ROLES = ("verify", "verify2", "small", "fp8proxy")


def sx_why(attn, q, k, fb, kvn, kwargs):
    mode = fb.forward_mode
    if mode.is_extend_without_speculative():
        kind, on, lo, hi = "pf", SPLIT_PF and attn.role in SX_PF_ROLES, SPLIT_PF_MIN_ROWS, SPLIT_PF_MAX_ROWS
    elif mode.is_target_verify():
        kind, on = "dec", SPLIT_DEC and attn.role in SX_DEC_ROLES
        lo, hi = sx_dec_rows(attn.role)
    else:
        return "mode", "-"
    if not on:
        return "off", kind
    why = pk_skip(attn.env, attn.stock, q, k, fb, kwargs)
    if why is None and (attn.env.in_graph_scope() or kvn is not None
                        or attn.env.padded_rows(q.shape[0], fb) is not None):
        why = "graph-or-padded"
    if why is not None:
        return why, kind
    M, hl = q.shape[0], q.shape[1]
    if FP8PACK != 2 or q.shape[2] != SX_D:
        return "fp8pack-mode", kind
    if not lo <= M <= hi:
        return "rows", kind
    if "off" in _SX:
        return "exchange-refused", kind
    sg = sx_env()
    if sg.tp.world_size * hl != SPLIT_HEADS:
        return "heads", kind
    backend = sx_backend(attn.env)
    if kind == "dec" and not sx_dec_blocks(backend, M, sg.tp.world_size):
        return "small-block", kind
    md, topk = backend.forward_metadata, kwargs.get("topk_indices")
    seq = getattr(md, "dsa_cache_seqlens_int32", None)
    if not (torch.is_tensor(topk) and topk.dim() == 2 and topk.dtype == torch.int32 and topk.shape[0] == M
            and torch.is_tensor(seq) and seq.shape[0] == M and backend.workspace_buffer is not None
            and getattr(fb, "attn_cp_metadata", None) is None and not sg.db.dsa_use_prefill_cp(fb)
            and not sg.db.get_in_autotune_dummy_run() and torch.is_tensor(kwargs.get("k_rope"))):
        return "backend-rows", kind
    if torch.cuda.is_current_stream_capturing() and not sx_graph_ready(kind, M):
        return "not-warm", kind
    return None, kind


def sx_note(attn, kind, why, q):
    key = (attn.prefix, attn.role, kind, why)
    if why not in ("mode", "off") and key not in _SX["seen"]:
        _SX["seen"].add(key)
        _log.warning("split-attn: %s [%s %s] %s rows=%s", attn.prefix, attn.role, kind,
                     "on" if why is None else f"PackAttn's path: {why}", tuple(q.shape) if torch.is_tensor(q) else None)


def sx_block(backend, M, kind, W):
    n = int(getattr(backend, "speculative_num_draft_tokens", 0) or 0) if kind == "dec" else 0
    if n > 1 and M % n == 0:
        return -(-(M // n) // W) * n
    return -(-M // W)


def sx_exchange(sg, dev, hl, kind):
    ex = _SX.get("ex")
    if ex is None and "off" not in _SX:
        _need(not torch.cuda.is_current_stream_capturing(), "split-attn: the first split call is inside a CUDA-graph "
              "capture (its exchange build is collective and eager)")
        if dev not in _SX.get("pk", ()):
            sx_pack_probe(dev, hl, SX_D)
        W = sg.tp.world_size
        ex = sx_build(sg.tp, dev, hl, SPLIT_PF_ROUND if W == 2 else -(-DEC_MAX_ROWS // W))
        if ex is not None:
            _SX["ex"] = ex
    if ex is not None:
        _need(ex.hl == hl and ex.world == sg.tp.world_size, f"split-attn: a {hl}-head call on a {ex.hl}-head exchange")
    return ex


def sx_fi(sg, backend, st, counter, max_seq):
    k_scale = st.k_scale_float if getattr(st, "k_scale_float", None) is not None else 1.0
    return dict(workspace_buffer=backend.workspace_buffer, qk_nope_head_dim=backend.qk_nope_head_dim,
                kv_lora_rank=backend.kv_lora_rank, qk_rope_head_dim=backend.qk_rope_head_dim, max_seq_len=max_seq,
                bmm1_scale=1.0 * k_scale * st.scaling, backend="trtllm-gen",
                skip_softmax_threshold_scale_factor=sg.envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                multi_ctas_kv_counter_buffer=counter)


def sx_kv(backend, key):
    return key.view(-1, backend.physical_page_size, backend.kv_cache_dim).unsqueeze(1).view(
        -1, 1, backend.physical_page_size, backend.kv_cache_dim)


def sx_call(attn, ex, sg, backend, q, k, fb, save, kwargs, bq):
    st, md, M = attn.stock, backend.forward_metadata, q.shape[0]
    topk = kwargs["topk_indices"]
    k8, kr8 = torch.empty((M, 1, SX_D), dtype=_FP8, device=q.device), kwargs["k_rope"].to(_FP8)
    if backend.use_fused_topk:
        ptab = backend._get_fused_topk_page_table(backend._pad_topk_indices(topk, M))
    else:
        mode = fb.forward_mode
        ptab = sg.db.transform_index_page_table_prefill(
            page_table=md.page_table_1, topk_indices=topk, extend_lens_cpu=md.dsa_extend_seq_lens_list, page_size=1,
            output_num_tokens=M, page_table_is_expanded=mode.is_target_verify() or mode.is_draft_extend_v2(),
            cu_seqlens_q=md.cu_seqlens_q)
    fi = sx_fi(sg, backend, st, ex.counter, md.max_seq_len_k)

    def kv_of():
        if save:
            backend.kv_layers.layer(st.layer_id).store_mla(st, fb.out_cache_loc, k8, kr8)
        return sx_kv(backend, backend.kv_layers.read(st.layer_id).key())

    kind = "dec" if fb.forward_mode.is_target_verify() else "pf"
    return sx_rounds(ex, sg, backend, kind, q, k, k8, ptab, md.dsa_cache_seqlens_int32, fi, bq, kv_of)


def sx_rounds(ex, sg, backend, kind, q, k, k8, ptab, seq, fi, bq, kv_of):
    M, hl, W, me, dev = q.shape[0], q.shape[1], ex.world, ex.rank, q.device
    out = torch.empty((M, 1, hl, SX_D), dtype=torch.bfloat16, device=dev)
    ss = bq if bq <= ex.O else ex.O
    for j in range(-(-bq // ex.O)):
        rng = []
        for s in range(W):
            a = min(M, s * bq + j * ss)
            rng.append((a, max(0, min(M, (s + 1) * bq, s * bq + (j + 1) * ss) - a)))
        if ss == bq:
            sx_pack(q, k[:M], ex.qslot(0, M), 0, M, k8)
        else:
            for s in range(W):
                if rng[s][1] or (j == 0 and s == 0):
                    sx_pack(q, k[:M], ex.qslot(s * ss, rng[s][1]), rng[s][0], rng[s][1],
                            k8 if j == 0 and s == 0 else None)
        if j == 0:
            kv = kv_of()
            if kind not in ex.kwarm and not torch.cuda.is_current_stream_capturing():
                a, n = rng[me]
                n = min(n, SX_WARM_ROWS)
                if n:
                    pt, top_k = backend._pad_trtllm_sparse_page_table(ptab[a:a + n])
                    q0 = torch.zeros((n, 1, SPLIT_HEADS, SX_D), dtype=torch.uint8, device=dev).view(_FP8)
                    sx_kernel(sg, kind, ex, q0, kv, pt, top_k, sg.db.prepare_trtllm_nope_sparse_metadata(pt),
                              seq[a:a + n], torch.empty((n, 1, SPLIT_HEADS, SX_D), dtype=torch.bfloat16, device=dev),
                              fi)
                ex.kwarm.add(kind)
        ex.sync()
        a, n = rng[me]
        if n:
            q64 = torch.empty((n, 1, SPLIT_HEADS, SX_D), dtype=_FP8, device=dev)
            ex.qget(q64, me * ss, n)
            pt, top_k = backend._pad_trtllm_sparse_page_table(ptab[a:a + n])
            lens = sg.db.prepare_trtllm_nope_sparse_metadata(pt)
            sx_kernel(sg, kind, ex, q64, kv, pt, top_k, lens, seq[a:a + n], ex.ostage(n), fi)
        ex.sync()
        ex.oget(out, rng)
    return out


def sx_compare(got, want):
    g, w = got.reshape(-1, got.shape[-1]), want.reshape(-1, want.shape[-1])
    err = (g.float() - w.float()).norm(dim=1) / w.float().norm(dim=1).clamp_min(1e-30)
    same = (g.view(torch.int16) == w.view(torch.int16)).all(dim=1).float().mean()
    return bool(torch.isfinite(g).all()), float(err.max()) if err.numel() else 0.0, float(same) if g.numel() else 1.0


def sx_placeholder(sg, fb, M):
    loc = getattr(fb, "out_cache_loc", None)
    local = not (torch.is_tensor(loc) and loc.dim() == 1 and loc.shape[0] >= M
                 and int(torch.unique(loc[:M]).numel()) == M)
    return any(sg.tp.all_gather_object(local))


def sx_attend(attn, kind, q, k, v, fb, save, kwargs):
    sg, backend = sx_env(), sx_backend(attn.env)
    ex = sx_exchange(sg, q.device, q.shape[1], kind)
    if ex is None:
        return PackAttn.forward(attn, q, k, v, fb, save, None, **kwargs)
    eager = not torch.cuda.is_current_stream_capturing()
    if kind == "dec" and attn.role == "verify" and eager and "dproof" not in _SX:
        sx_dec_proof(attn, sg, backend, ex, q, kwargs)
        if "off" in _SX:
            return PackAttn.forward(attn, q, k, v, fb, save, None, **kwargs)
    M, key = q.shape[0], (kind, q.shape[0])
    bq = sx_block(backend, M, kind, ex.world)
    n_ck = _SX["checks"].get(kind, 0)
    check = eager and (n_ck < SPLIT_CHECKS or (kind == "dec" and key not in ex.warm))
    if check and kind == "dec" and sx_placeholder(sg, fb, M):
        if key not in ex.warm:
            return PackAttn.forward(attn, q, k, v, fb, save, None, **kwargs)
        check = False
    want = PackAttn.forward(attn, q, k, v, fb, save, None, **kwargs) if check else None
    ex = sb_view(attn, kind, fb, M, bq, ex, check)
    ex = sf_view(attn, kind, M, ex, check)
    out = sx_call(attn, ex, sg, backend, q, k, fb, save and want is None, kwargs, bq)
    orp_done(kwargs)
    sf_settle(attn)
    if check:
        fin, err, same = sx_compare(out, want.view(out.shape))
        ok = fin and err <= SPLIT_ROW_TOL and same >= SPLIT_BIT_FRAC
        verdicts = sg.tp.all_gather_object((ok, round(err, 7), round(same, 5)))
        if not all(v[0] for v in verdicts):
            why = (f"{attn.prefix} [{attn.role} {kind}] rows={M} block={bq}: the split attention differs from stock's on "
                   f"some rank (ok, rel-L2 max, bitwise share) {verdicts} (bounds {SPLIT_ROW_TOL}, {SPLIT_BIT_FRAC})")
            _need(not SPLIT_STRICT, f"split-attn: {why} (SPLIT_STRICT; set SPLIT_PF / SPLIT_DEC False)")
            _SX["off"] = why
            _log.warning("split-attn: OFF on every rank (voted live check), PackAttn's path from here; this call "
                         "served stock's: %s", why)
            return want
        _SX["checks"][kind] = n_ck + 1
        _log.warning("split-attn: %s [%s %s] rows=%d block=%d rank=%d: split == stock within audit class (rel-L2 max "
                     "%.3g, bitwise %.4f)", attn.prefix, attn.role, kind, M, bq, ex.rank, err, same)
    if eager:
        ex.warm.add(key)
    return out


def sx_dec_sizes(backend, W):
    n = max(1, int(getattr(backend, "speculative_num_draft_tokens", 0) or 0))
    lo, hi = sx_dec_rows("verify")
    return [M for M in range(-(-lo // n) * n, hi + 1, n) if sx_dec_blocks(backend, M, W)]


def sx_dec_data(attn, backend, ex, q, kwargs, rows):
    dev, hl, S = q.device, q.shape[1], SX_DEC_PROOF_SLOTS
    st, top = attn.stock, kwargs.get("topk_indices")
    kc = backend.kv_layers.read(st.layer_id).key()
    _need(kc.element_size() == 1 and backend.kv_cache_dim == SX_D and S % backend.physical_page_size == 0
          and bool(backend.use_fused_topk) and torch.is_tensor(top) and top.dim() == 2 and 4 <= top.shape[1] <= S,
          f"split-attn: T's proof needs an FP8 latent cache of {SX_D}-byte rows ({kc.dtype}, {backend.kv_cache_dim}), "
          f"fused top-k page tables (use_fused_topk {backend.use_fused_topk}) and a 2-D top-k hand-off of 4 .. {S} "
          "columns")
    C = top.shape[1]
    lo, hi = max(SX_DEC_PROOF_SEQ[0], C), max(SX_DEC_PROOF_SEQ[1], C)
    gen = torch.Generator(device=dev).manual_seed(SX_DEC_PROOF_SEED)
    pool = (torch.randn((S, SX_D), generator=gen, device=dev) * 2).to(_FP8).view(kc.dtype)
    r = torch.arange(rows, device=dev)
    ctx = torch.randint(lo, hi + 1, (rows,), generator=gen, device=dev)
    nval = C - r % 4
    if SX_DEC_PROOF_SHORT:
        short = r % SX_DEC_PROOF_SHORT == SX_DEC_PROOF_SHORT - 1
        sl = torch.randint(1, C, (rows,), generator=gen, device=dev)
        ctx, nval = torch.where(short, sl, ctx), torch.where(short, sl, nval)
    seq = nval
    keys = torch.rand((rows, S), generator=gen, device=dev)
    tab = keys.argsort(dim=1)[:, :C].to(torch.int32)
    tab.masked_fill_(torch.arange(C, device=dev) >= nval[:, None], -1)
    del keys
    sc = torch.tensor(SX_DEC_PROOF_QSCALE, device=dev).repeat(-(-rows // len(SX_DEC_PROOF_QSCALE)))[:rows]
    qall = torch.randn((rows, SPLIT_HEADS, SX_D), generator=gen, device=dev) * sc[:, None, None]
    qr = qall[:, ex.rank * hl:(ex.rank + 1) * hl].to(torch.bfloat16).contiguous()
    from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
    return types.SimpleNamespace(pool=pool, kv=sx_kv(backend, pool), seq=seq.to(torch.int32), tab=tab, q=qr,
                                 k=torch.zeros((rows, 1, SX_D), dtype=torch.bfloat16, device=dev),
                                 counter=make_persistent_multi_ctas_kv_counter_buffer(dev, hl, rows),
                                 max_seq=sx_dec_max_seq(backend, ctx))


def sx_dec_max_seq(backend, ctx):
    rt = getattr(backend, "req_to_token", None)
    return int(rt.shape[1]) if torch.is_tensor(rt) and rt.dim() == 2 else int(ctx.max())


def sx_dec_ref(sg, backend, fi, d, M):
    hl = d.q.shape[1]
    q8 = torch.empty((M, hl, SX_D), dtype=_FP8, device=d.q.device)
    sx_pack(d.q, d.k, q8, 0, M)
    pt, top_k = backend._pad_trtllm_sparse_page_table(backend._get_fused_topk_page_table(
        backend._pad_topk_indices(d.tab[:M].clone(), M)))
    lens = sg.db.prepare_trtllm_nope_sparse_metadata(pt)
    return sg.fid.trtllm_batch_decode_with_kv_cache_mla(
        query=q8.view(M, 1, hl, SX_D), kv_cache=d.kv, block_tables=pt.unsqueeze(1), seq_lens=d.seq[:M],
        sparse_mla_top_k=top_k, sparse_mla_top_k_lens=lens, **dict(fi, multi_ctas_kv_counter_buffer=d.counter))


def sx_dec_cmp(got, want):
    g, w = got.reshape(-1, got.shape[-1]), want.reshape(-1, want.shape[-1])
    if not g.numel():
        return 0, 0, 0.0, 0, 0
    eq = (g.view(torch.int16) == w.view(torch.int16)).all(dim=1)
    fg, fw = torch.isfinite(g).all(dim=1), torch.isfinite(w).all(dim=1)
    both = fg & fw
    gf, wf = g[both].float(), w[both].float()
    err = float(((gf - wf).norm(dim=1) / wf.norm(dim=1).clamp_min(1e-30)).max()) if gf.shape[0] else 0.0
    return int(eq.sum()), eq.numel(), err, int((~fw).sum()), int((~fg).sum())


def sx_dec_proof(attn, sg, backend, ex, q, kwargs):
    _SX["dproof"] = "running"
    W, dev = ex.world, q.device
    sizes = sx_dec_sizes(backend, W)
    if not sizes:
        _SX["dproof"] = "no sizes"
        return
    why, d, fi, ref0 = None, None, None, None
    try:
        d = sx_dec_data(attn, backend, ex, q, kwargs, sizes[-1])
        fi = sx_fi(sg, backend, attn.stock, ex.counter, d.max_seq)
        ref0 = sx_dec_ref(sg, backend, fi, d, sizes[0])
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(sg.tp.all_gather_object(why)) if w is not None]
    acc = dict(off=0, err=0.0, diff=0, pairs=0, nf_ref=0, nf_got=0, first=None)
    if not bad:
        for M in sizes:
            ref = ref0 if M == sizes[0] else sx_dec_ref(sg, backend, fi, d, M)
            ptab = backend._get_fused_topk_page_table(backend._pad_topk_indices(d.tab[:M].clone(), M))
            got = sx_rounds(ex, sg, backend, "dec", d.q[:M], d.k[:M], torch.empty((M, 1, SX_D), dtype=_FP8, device=dev),
                            ptab, d.seq[:M], fi, sx_block(backend, M, "dec", W), lambda: d.kv)
            n_eq, n, err, nf_ref, nf_got = sx_dec_cmp(got, ref.view(got.shape))
            acc["err"], acc["diff"], acc["pairs"] = max(acc["err"], err), acc["diff"] + n - n_eq, acc["pairs"] + n
            acc["nf_ref"], acc["nf_got"] = acc["nf_ref"] + nf_ref, acc["nf_got"] + nf_got
            if n_eq != n or nf_ref:
                acc["off"] += 1
                if acc["first"] is None:
                    acc["first"] = (M, n - n_eq, round(err, 7), nf_ref, nf_got)
        verdicts = sg.tp.all_gather_object((acc["off"] == 0, acc["nf_ref"], round(acc["err"], 7), acc["diff"],
                                            acc["pairs"], acc["nf_got"], acc["off"], acc["first"]))
        if not all(v[0] for v in verdicts):
            what = ("(ok, stock non-finite pairs, rel-L2 max over finite pairs, differing pairs, pairs, split non-finite "
                    "pairs, sizes off, first size off (rows, differing pairs, rel-L2, stock nf, split nf))")
            if any(v[1] for v in verdicts):
                bad = [("all", f"the proof's inputs are invalid: stock's own call is non-finite on the seeded inputs, "
                        f"no evidence either way {what} {verdicts}")]
            else:
                bad = [("all", f"split != stock on the seeded inputs {what} {verdicts}")]
    del d, ref0
    if bad:
        why = (f"T proof over {len(sizes)} verify sizes ({sizes[0]}..{sizes[-1]} rows, tile {SPLIT_DEC_TILE}): rank "
               f"{bad[0][0]} {bad[0][1]}")
        _SX["dproof"] = why
        _need(not SPLIT_STRICT, f"split-attn: {why} (SPLIT_STRICT; set SPLIT_DEC False)")
        _SX["off"] = why
        _log.warning("split-attn: OFF on every rank (voted T proof), PackAttn's path from here: %s", why)
        return
    ex.warm.update(("dec", M) for M in sizes)
    _SX["dproof"] = "ok"
    _log.warning("split-attn: tp%d/%d T proof: %d verify sizes (%d..%d rows, tile %s) split == stock on seeded inputs, "
                 "bitwise on all %d (row, head) pairs (0 differing, rel-L2 max %.3g; stock finite everywhere); every size "
                 "warm", ex.rank, W, len(sizes), sizes[0], sizes[-1], SPLIT_DEC_TILE, acc["pairs"], acc["err"])


class SplitAttn(PackAttn):
    def forward(self, q, k, v, forward_batch, save_kv_cache=True, key_value_num_tokens=None, **kwargs):
        p = sf_take(self)
        if p is not None:
            return sf_attend(self, p, q, k, v, forward_batch, save_kv_cache, key_value_num_tokens, kwargs)
        ng = g3_take(self)
        why, kind = sx_why(self, q, k, forward_batch, key_value_num_tokens, kwargs)
        sx_note(self, kind, why, q)
        if ng is not None:
            g3_hold(self, ng, why, kind, q)
        if why is not None:
            return super().forward(q, k, v, forward_batch, save_kv_cache, key_value_num_tokens, **kwargs)
        return sx_attend(self, kind, q, k, v, forward_batch, save_kv_cache, kwargs)


def sx_attn(stock, env, prefix, role):
    split = SPLIT_DEC if role == "verify" else (SPLIT_PF or SPLIT_DEC)
    _need(not (SPLIT_DEC and split) or SPLIT_DEC_TILE == "q32x2", f"split-attn: SPLIT_DEC_TILE {SPLIT_DEC_TILE!r}: T "
          "ships only the bitwise q32x2 tile (the TP2 audit never runs T: its verify blocks are below "
          "SPLIT_DEC_MIN_BLOCK)")
    return (SplitAttn if split and FP8PACK == 2 else PackAttn)(stock, env, prefix, role)

SF_FRONT = True
SF_CLASS = "bitwise"
SF_ROW_TOL = 1e-2
SF_CHECKS = 1
SF_STRICT = False
_SF = {"seen": set()}


@triton.jit(do_not_specialize=["nq"])
def _sf_pack_kernel(q_ptr, qo_ptr, nq, H, sqt, sqh, ho, HO, D: tl.constexpr, R: tl.constexpr):
    pid, c = tl.program_id(0), tl.arange(0, D)[None, :]
    r = pid.to(tl.int64) * R + tl.arange(0, R)
    m = (r < nq)[:, None]
    x = tl.load(q_ptr + ((r // H) * sqt + (r % H) * sqh)[:, None] + c, mask=m)
    tl.store(qo_ptr + ((r // H) * HO + ho + r % H)[:, None] * D + c, x.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
             mask=m)


def sf_pack(q, qo, ho):
    n, H, D = q.shape
    nq = n * H
    if nq:
        _sf_pack_kernel[(triton.cdiv(nq, PK_R),)](q, qo, nq, H, q.stride(0), q.stride(1), ho, qo.shape[-2], D=D, R=PK_R,
                                                  num_warps=PK_WARPS)


def sf_pack_probe(dev, H):
    _need(not torch.cuda.is_current_stream_capturing(), "split-front: the pack probe is inside a CUDA-graph capture")
    x = torch.arange(1 << 16, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16).view(-1, SX_D)
    q = x.expand(H, *x.shape).contiguous().transpose(0, 1)
    want, _ = pk_pack(q, x.unsqueeze(1))
    got = torch.zeros((q.shape[0], 1, 2 * H, SX_D), dtype=torch.uint8, device=dev).view(_FP8)
    sf_pack(q, got, 0)
    sf_pack(q, got, H)
    w8 = want.view(torch.uint8)
    _need(torch.equal(got[:, 0, :H].view(torch.uint8), w8) and torch.equal(got[:, 0, H:].view(torch.uint8), w8),
          "split-front: the 64-head pack differs from FP8-PACK's pk_pack on the BF16 pattern table")
    _SF.setdefault("probe", set()).add(dev)


class SfLayer:

    def __init__(self, prefix, proxy, sx, sb):
        self.prefix, self.proxy, self.sx, self.sb = prefix, proxy, sx, sb
        self.lin = self.wkc = self.stock_qb = self.stock_prep = self.halves = None
        self.plan = self.pending = self.chk = self.settle = self.bad = None
        self.checks, self.on = 0, False
        self.hl, self.dn = getattr(proxy, "num_local_heads", None), getattr(proxy, "qk_nope_head_dim", None)


def sf_facts(proxy, sb):
    from sglang.srt.lora.utils import kv_b_lora_correction
    from sglang.srt.models import deepseek_v2
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla as fm
    mx, cls = fm.DeepseekMLAForwardMixin, type(proxy)
    fu = sb.ar.sg.fp8_utils
    qb, w = proxy._modules.get("q_b_proj"), getattr(proxy, "w_kc", None)
    lin = getattr(qb, "inner", None)
    qm = getattr(lin, "quant_method", None)
    fn = getattr(getattr(qm, "w8a8_block_fp8_linear", None), "func", getattr(qm, "w8a8_block_fp8_linear", None))
    hl, dn = proxy.num_local_heads, proxy.qk_nope_head_dim
    _need(cls.forward_absorb_prepare is mx.forward_absorb_prepare and cls.forward_absorb_core is mx.forward_absorb_core
          and cls._split_q_nope_pe is mx._split_q_nope_pe
          and cls.q_b_proj_forward is deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_forward
          and proxy.rotary_emb is None and proxy.qk_rope_head_dim == 0 and proxy.q_lora_rank is not None
          and proxy.use_dsa and not fm._ENABLE_DSA_Q8KV8_BORN_FP8_Q and not fm._ENABLE_DSA_Q8KV8_QPREP_OVERLAP
          and not fm._SGLANG_EXPERIMENTAL_LORA_OPTI and kv_b_lora_correction(proxy) is None
          and not proxy.use_deep_gemm_bmm and not getattr(proxy, "_kimi_split_gguf_kv_b", False)
          and proxy.dcp_size == 1 and proxy.q_b_proj_qrep is None and proxy.q_b_proj_qrep_weight is None
          and proxy._use_min_latency_q_b_gemm is not True and proxy._q_b_proj_verified_shape is False,
          "split-front: the attention's q path is not stock's absorb prepare -> q_b_proj_forward -> split -> BF16 W_UK "
          "bmm without rope (or born-FP8 q / LoRA / DCP / a replicated or min-latency q_b is on)")
    _need(type(qb) is Fp8Proj and fn is fu.deepgemm_w8a8_block_fp8_linear_with_fallback and lin.bias is None
          and getattr(lin, "_block_fp8_bf16_weight", None) is None and not fu.get_platform().is_sm90
          and sb.ar.sg.dgw.ENABLE_JIT_DEEPGEMM and lin.weight.dtype == _FP8
          and tuple(lin.weight.shape) == (hl * dn, proxy.q_lora_rank)
          and fu._deepgemm_block_fp8_supported(tuple(lin.weight.shape), [BLOCK, BLOCK], torch.bfloat16)
          and type(sb.zc) is ZcProj and sb.zc.quant is fu.sglang_per_token_group_quant_fp8,
          "split-front: q_b_proj is not b13's DeepGEMM block-FP8 linear (or SB's ZcProj calls differ)")
    _need(torch.is_tensor(w) and w.dtype == torch.bfloat16 and tuple(w.shape) == (hl, dn, SX_D)
          and tuple(w.stride()) == (dn * SX_D, 1, dn) and 2 * hl == SPLIT_HEADS and hl == sb.hl
          and dn == proxy.qk_head_dim, f"split-front: w_kc {getattr(w, 'dtype', None)} {tuple(getattr(w, 'shape', ()))} "
          "is not the loader's BF16 [32, 256, 512] (stride 256 x 512, 1, 256) on 2 x 32 heads")


def sf_attach(state):
    attn = state.attn
    if not SF_FRONT or attn.kind != "mla" or attn.proxy is None:
        return
    proxy = attn.proxy
    sx = proxy._modules.get("attn_mqa")
    sb = getattr(sx, "sb", None)
    if type(sx) is not SplitAttn or sx.role != "fp8proxy" or sb is None:
        return
    sf = SfLayer(sx.prefix, proxy, sx, sb)
    sf.stock_qb = types.MethodType(type(proxy).q_b_proj_forward, proxy)
    sf.stock_prep = types.MethodType(type(proxy).forward_absorb_prepare, proxy)
    try:
        _need(sb.bad is None, f"split-front: SB not attached ({sb.bad})")
        sf_facts(proxy, sb)
        sf.lin, sf.wkc = proxy._modules["q_b_proj"].inner, proxy.w_kc
    except Exception as exc:
        sf.bad = f"{type(exc).__name__}: {exc}"[:400]
        _log.warning("split-front: %s not attached (%s): S's q pull; voted off at its first eligible call", sx.prefix,
                     sf.bad)
    sx.sf = sf
    object.__setattr__(proxy, "forward_absorb_prepare", functools.partial(sf_prepare, sf))
    object.__setattr__(proxy, "q_b_proj_forward", functools.partial(sf_qb, sf))


def sf_note(sf, M, why):
    key = (sf.prefix, why)
    if why not in ("switch", "mode", "rows", "off") and key not in _SF["seen"]:
        _SF["seen"].add(key)
        _log.warning("split-front: %s rows=%d: S's q pull (%s)", sf.prefix, M, why)


def sf_plan(sf, hs, fb):
    if isinstance(hs, tuple):
        hs = hs[0]
    if not SF_FRONT:
        return "switch", None
    if "off" in _SF:
        return "off", None
    if not (SPLIT_PF and SB_BACK and FP8PACK == 2):
        return "split-off", None
    if not (torch.is_tensor(hs) and hs.dim() == 2) or fb is None:
        return "inputs", None
    M = hs.shape[0]
    if not max(PF_MIN_ROWS, SPLIT_PF_MIN_ROWS) <= M <= SPLIT_PF_MAX_ROWS:
        return "rows", None
    if not fb.forward_mode.is_extend_without_speculative() or torch.cuda.is_current_stream_capturing():
        return "mode", None
    ex, sb = _SX.get("ex"), sf.sb
    if "off" in _SX or "off" in _SB or ex is None:
        return "split-refused", None
    if _SX["checks"].get("pf", 0) < SPLIT_CHECKS or "pf" not in ex.kwarm:
        return "split-checks", None
    if sb.bad is not None or sb.halves is None or sb.checks < SB_CHECKS:
        return "sb-checks", None
    if sb.pending is not None or sb.bcheck is not None:
        return "sb-busy", None
    env = sf.sx.env
    if env.in_graph_scope() or env.padded_rows(M, fb) is not None or sb.ffn.sg.in_graph():
        return "graph-or-padded", None
    sg = sx_env()
    if ex.world != 2 or ex.hl != sf.hl or ex.hl * ex.world != SPLIT_HEADS:
        return "heads", None
    if (getattr(fb, "attn_cp_metadata", None) is not None or sg.db.dsa_use_prefill_cp(fb)
            or sg.db.get_in_autotune_dummy_run()):
        return "backend", None
    bq = sx_block(sx_backend(env), M, "pf", ex.world)
    if bq > ex.O or M < PF_MIN_ROWS:
        return "shape", None
    rng = [(min(M, s * bq), max(0, min(M, (s + 1) * bq) - min(M, s * bq))) for s in range(ex.world)]
    mode = "check" if sf.halves is None or sf.checks < SF_CHECKS else "serve"
    return None, types.SimpleNamespace(mode=mode, M=M, bq=bq, rng=rng, me=ex.rank)


def sf_prepare(sf, *args, **kwargs):
    hs = args[1] if len(args) > 1 else kwargs.get("hidden_states")
    fb = args[2] if len(args) > 2 else kwargs.get("forward_batch")
    _need(sf.pending is None and sf.plan is None, f"split-front: {sf.prefix}: the previous call's hand-off was never "
          "consumed")
    why, plan = sf_plan(sf, hs, fb)
    if why is not None:
        sf_note(sf, hs.shape[0] if torch.is_tensor(hs) and hs.dim() else 0, why)
    sf.plan = plan
    try:
        return sf.stock_prep(*args, **kwargs)
    finally:
        sf.plan = None


def sf_qb(sf, q_lora):
    plan, sf.plan = sf.plan, None
    if plan is None:
        return sf.stock_qb(q_lora)
    M = plan.M
    x = q_lora if q_lora.is_contiguous() else q_lora.contiguous()
    lin = sf.lin
    if (sf.bad is None and (x.dim() != 2 or x.shape[0] != M or x.dtype != torch.bfloat16
                            or sf.sb.ar.sg.fp8_utils._use_fp8_block_gemv(x, lin.weight, lin.weight_scale_inv,
                                                                         [BLOCK, BLOCK]))):
        sf_note(sf, M, "q_lora rows / gemv")
        return sf.stock_qb(q_lora)
    if sf.halves is None:
        sf_fetch(sf)
        if sf.halves is None:
            return sf.stock_qb(q_lora)
    if plan.mode == "check":
        sf.pending = types.SimpleNamespace(stage="check", x=x, ver=_rq_ver(x), M=M, bq=plan.bq, rng=plan.rng)
        return sf.stock_qb(q_lora)
    a, n = plan.rng[plan.me]
    sf.pending = types.SimpleNamespace(stage="serve", x=x, ver=_rq_ver(x), M=M, bq=plan.bq, rng=plan.rng,
                                       q64=sf_front(sf, x, a, n))
    if not sf.on:
        sf.on = True
        _log.warning("split-front: %s [fp8proxy pf] rows=%d rank=%d: on (q for own rows [%d, %d) x %d heads: q_b + w_kc "
                     "halves in rank order; no q stage, no q pull, one barrier)", sf.prefix, M, plan.me, a, a + n,
                     SPLIT_HEADS)
    return x.new_empty((0, sf.hl, sf.dn))


def sf_front(sf, x, a, n):
    zc = sf.sb.zc
    q64 = torch.empty((n, 1, SPLIT_HEADS, SX_D), dtype=_FP8, device=x.device)
    if n:
        q8, s8 = zc.quant(x[a:a + n], BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=zc.ue8m0)
        for r, (w8, ws, wkc) in enumerate(sf.halves):
            y = torch.empty((n, w8.shape[0]), dtype=torch.bfloat16, device=x.device)
            zc.gemm(q8, s8, w8, ws, y, BLOCK, BLOCK)
            qn = y.view(-1, sf.hl, sf.dn).split([sf.dn, 0], dim=-1)[0]
            b = torch.bmm(qn.transpose(0, 1), wkc)
            del y, qn
            sf_pack(b.transpose(0, 1), q64, r * sf.hl)
            del b
        del q8, s8
    return q64


def sf_fetch(sf):
    tp = sx_env().tp
    me, W = tp.rank_in_group, tp.world_size
    why, own, spans, gots = None, None, None, None
    try:
        _need(sf.bad is None, f"split-front: not attached ({sf.bad})")
        own = (sf.lin.weight, sf.lin.weight_scale_inv, sf.wkc)
        spans = [sb_span(t) for t in own]
        gots = [torch.empty((W, u8.numel()), dtype=torch.uint8, device=u8.device) for u8 in spans]
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    if bad:
        return sf_refuse(f"{sf.prefix} peer-half fetch: rank {bad[0][0]} {bad[0][1]} (voted before the gathers)")
    for u8, got in zip(spans, gots):
        tp.all_gather_into_tensor(got.view(-1), u8)
    halves, sums = [[None] * len(own) for _ in range(W)], None
    try:
        sums = []
        for i, (t, u8, got) in enumerate(zip(own, spans, gots)):
            for s in range(W):
                halves[s][i] = t if s == me else got[s].clone().view(t.dtype).as_strided(t.shape, t.stride(), 0)
            sums.append([sb_sum(u8)] + [sb_sum(sb_span(halves[s][i])) for s in range(W)])
        del gots
        dev = own[0].device
        if dev not in _SF.get("probe", ()):
            sf_pack_probe(dev, sf.hl)
    except Exception as exc:
        why, sums = f"{type(exc).__name__}: {exc}", None
    votes = tp.all_gather_object((why, sums))
    bad = [(r, w) for r, (w, _) in enumerate(votes) if w is not None]
    if not bad:
        bad = [(r, f"tensor {i}: its copy of rank {s}'s bytes differs from rank {s}'s own")
               for r, (_, sm) in enumerate(votes) for i in range(len(own)) for s in range(W)
               if sm[i][1 + s] != votes[s][1][i][0]]
    if bad:
        return sf_refuse(f"{sf.prefix} peer-half fetch: rank {bad[0][0]} {bad[0][1]}")
    sf.halves = [tuple(h) for h in halves]
    nb = sum(sb_span(t).numel() for s, h in enumerate(sf.halves) if s != me for t in h)
    _log.warning("split-front: %s tp%d/%d fetched the peer halves (FP8 q_b, scales, w_kc: %.1f MiB), checksums agree; "
                 "pack probe passed; device free %d MiB", sf.prefix, me, W, nb / 2 ** 20, sb_free_mib(own[0].device))


def sf_refuse(why):
    _need(not SF_STRICT, f"split-front: {why} (SF_STRICT; set SF_FRONT False)")
    _SF["off"] = why
    _log.warning("split-front: OFF on every rank (voted), S's q pull from here: %s", why)


def sf_take(attn):
    sf = getattr(attn, "sf", None)
    if sf is None or sf.pending is None:
        return None
    p, sf.pending = sf.pending, None
    if p.stage == "check":
        sf.chk = p
        return None
    return p


def sf_qstand(dev, hl):
    t = _SF.get(("q", dev))
    if t is None:
        t = _SF[("q", dev)] = torch.empty((hl, 2, SX_D), dtype=torch.bfloat16, device=dev).transpose(0, 1)
    return t


def sf_attn_why(attn, p, q, k, fb, kvn, kwargs, ex, backend, sg):
    M, sb = p.M, attn.sb
    if ex is None or "off" in _SX or "off" in _SB or "off" in _SF:
        return "refused"
    if not (torch.is_tensor(q) and q.dim() == 3 and q.shape[0] == 0 and q.shape[1] == ex.hl and q.shape[2] == SX_D
            and p.q64.shape[0] == p.rng[ex.rank][1]):
        return "hand-off"
    if kvn is not None or attn.env.in_graph_scope() or attn.env.padded_rows(M, fb) is not None:
        return "graph-or-padded"
    why = qpad_skip(attn.env, attn.stock, sf_qstand(q.device, ex.hl), fb, kwargs)
    if why is not None:
        return why
    if not (torch.is_tensor(k) and k.dtype == torch.bfloat16 and k.shape[0] == M and k.shape[-1] == SX_D
            and k.numel() == M * SX_D and k.stride(-1) == 1 and k.stride(0) % 16 == 0 and k.data_ptr() % 16 == 0):
        return "k-layout"
    if FP8PACK != 2 or not SPLIT_PF_MIN_ROWS <= M <= SPLIT_PF_MAX_ROWS or not fb.forward_mode.is_extend_without_speculative():
        return "rows-mode"
    md, topk = backend.forward_metadata, kwargs.get("topk_indices")
    seq = getattr(md, "dsa_cache_seqlens_int32", None)
    if not (torch.is_tensor(topk) and topk.dim() == 2 and topk.dtype == torch.int32 and topk.shape[0] == M
            and torch.is_tensor(seq) and seq.shape[0] == M and backend.workspace_buffer is not None
            and getattr(fb, "attn_cp_metadata", None) is None and not sg.db.dsa_use_prefill_cp(fb)
            and not sg.db.get_in_autotune_dummy_run() and torch.is_tensor(kwargs.get("k_rope"))
            and not torch.cuda.is_current_stream_capturing()):
        return "backend-rows"
    if p.bq != sx_block(backend, M, "pf", ex.world) or "pf" not in ex.kwarm:
        return "block"
    if sb_why(attn, "pf", M, p.bq, ex, False) is not None or sb.pending is not None or sb.bcheck is not None:
        return "sb-path"
    return None


def sf_stock_q(sf, p):
    _need(p.ver is None or _rq_ver(p.x) == p.ver, f"split-front: {sf.prefix}: q_lora was written after its q_b call")
    q = sf.stock_qb(p.x)
    q_nope, q_pe = q.split([sf.proxy.qk_nope_head_dim, sf.proxy.qk_rope_head_dim], dim=-1)
    return torch.bmm(q_nope.transpose(0, 1), sf.wkc).transpose(0, 1), q_pe


def sf_attend(attn, p, q, k, v, fb, save, kvn, kwargs):
    sf, sg, backend, ex = attn.sf, sx_env(), sx_backend(attn.env), _SX.get("ex")
    why = sf_attn_why(attn, p, q, k, fb, kvn, kwargs, ex, backend, sg)
    if why is not None:
        sf_note(sf, p.M, f"attention: {why}")
        q_full, q_pe = sf_stock_q(sf, p)
        del p
        return SplitAttn.forward(attn, q_full, k, v, fb, save, kvn, **dict(kwargs, q_rope=q_pe))
    st, md, M, me = attn.stock, backend.forward_metadata, p.M, ex.rank
    topk = kwargs["topk_indices"]
    k8, kr8 = torch.empty((M, 1, SX_D), dtype=_FP8, device=k.device), kwargs["k_rope"].to(_FP8)
    if backend.use_fused_topk:
        ptab = backend._get_fused_topk_page_table(backend._pad_topk_indices(topk, M))
    else:
        ptab = sg.db.transform_index_page_table_prefill(
            page_table=md.page_table_1, topk_indices=topk, extend_lens_cpu=md.dsa_extend_seq_lens_list, page_size=1,
            output_num_tokens=M, page_table_is_expanded=False, cu_seqlens_q=md.cu_seqlens_q)
    fi = sx_fi(sg, backend, st, ex.counter, md.max_seq_len_k)
    sx_pack(sf_qstand(k.device, ex.hl), k[:M], ex.qslot(0, 0), 0, 0, k8)
    if save:
        backend.kv_layers.layer(st.layer_id).store_mla(st, fb.out_cache_loc, k8, kr8)
    kv = sx_kv(backend, backend.kv_layers.read(st.layer_id).key())
    view = sb_view_of(ex, attn.sb, M, False)
    out = torch.empty((M, 1, ex.hl, SX_D), dtype=torch.bfloat16, device=k.device)
    ex.sync()
    a, n = p.rng[me]
    if n:
        pt, top_k = backend._pad_trtllm_sparse_page_table(ptab[a:a + n])
        sx_kernel(sg, "pf", ex, p.q64, kv, pt, top_k, sg.db.prepare_trtllm_nope_sparse_metadata(pt),
                  md.dsa_cache_seqlens_int32[a:a + n], ex.ostage(n), fi)
    view.oget(out, p.rng)
    orp_done(kwargs)
    return out


def sf_view(attn, kind, M, ex, check):
    sf = getattr(attn, "sf", None)
    c = None if sf is None else sf.chk
    if c is None:
        return ex
    sf.chk = None
    if kind != "pf" or check or getattr(ex, "sb_mode", None) != "serve" or c.M != M:
        sf_note(sf, M, "live check deferred: not an SB-served call")
        return ex
    view, base, me = copy.copy(ex), ex.qget, ex.rank

    def qget(dst, at, n, delta=None):
        base(dst, at, n, delta)
        if delta is None and not hasattr(c, "res"):
            c.res = sf_cmp(sf, c, dst, me)
    view.qget = qget
    sf.settle = c
    return view


def sf_cmp(sf, c, dst, me):
    a, n = c.rng[me]
    if c.ver is not None and _rq_ver(c.x) != c.ver:
        return types.SimpleNamespace(nd=None, why="q_lora was written after its q_b call")
    got = sf_front(sf, c.x, a, n)
    nd = (got.view(torch.uint8) != dst.view(torch.uint8)).sum()
    return types.SimpleNamespace(nd=nd, size=got.numel(), got=got, dst=dst, why=None, a=a, n=n)


def sf_err(got, dst, rows=1024):
    fin, err = True, 0.0
    for i in range(0, got.shape[0], rows):
        g, w = got[i:i + rows].reshape(-1, SX_D).float(), dst[i:i + rows].reshape(-1, SX_D).float()
        fin = fin and bool(torch.isfinite(g).all())
        e = (g - w).norm(dim=1) / w.norm(dim=1).clamp_min(1e-30)
        err = max(err, float(e.max()) if e.numel() else 0.0)
    return fin, err


def sf_settle(attn):
    sf = getattr(attn, "sf", None)
    c = None if sf is None else sf.settle
    if c is None:
        return
    sf.settle = None
    r = getattr(c, "res", None)
    if r is None or r.nd is None:
        local = (False, -1, 0, float("inf"), (r.why if r is not None else "the q pull was never reached"))
    else:
        nd = int(r.nd)
        fin, err = sf_err(r.got, r.dst) if nd else (True, 0.0)
        ok = nd == 0 if SF_CLASS == "bitwise" else (nd == 0 or (fin and err <= SF_ROW_TOL))
        local = (ok, nd, r.size, round(err, 7), None)
        r.got = r.dst = None
    votes = sx_env().tp.all_gather_object(local)
    if not all(v[0] for v in votes):
        return sf_refuse(f"{sf.prefix} rows={c.M}: SF's q64 differs from S's pulled q64 (ok, differing bytes, bytes, "
                         f"rel-L2 max, why) {votes} (class {SF_CLASS}, bound {SF_ROW_TOL})")
    sf.checks += 1
    me = sx_env().tp.rank_in_group
    a, n = c.rng[me]
    cls = "bitwise" if all(v[1] == 0 for v in votes) else "audit"
    _log.warning("split-front: %s rows=%d rank=%d check %d/%d pass: own-rows q64 [%d, %d) x %d heads == S's pulled q64 "
                 "(differing bytes %s of %s by rank; rel-L2 max %.3g); class %s", sf.prefix, c.M, me, sf.checks,
                 SF_CHECKS, a, a + n, SPLIT_HEADS, [v[1] for v in votes], [v[2] for v in votes],
                 max(v[3] for v in votes), cls)

SPLIT_DEC = True
SPLIT_DEC_MIN_ROWS = 192
SPLIT_DEC_MIN_ROWS_TP2 = 6
SPLIT_DEC_MIN_BLOCK = 75
SPLIT_DEC_G3 = True
SPLIT_DEC_TILE = "q32x2"
SX_DEC_PROOF_SEED = 20261182
SX_DEC_PROOF_SLOTS = 8192
SX_DEC_PROOF_SEQ = (4096, 131072)
SX_DEC_PROOF_SHORT = 8
SX_DEC_PROOF_QSCALE = (0.05, 0.3, 2.0)
G3_BLOCK, G3_WARPS = 4096, 4


def sx_dec_rows(role):
    return (SPLIT_DEC_MIN_ROWS, DEC_MAX_ROWS) if role == "verify" else (SPLIT_DEC_MIN_ROWS_TP2, SPLIT_PF_MAX_ROWS)


def sx_counter32(ex, dev):
    c = getattr(ex, "c32", None)
    if c is None:
        _need(not torch.cuda.is_current_stream_capturing(), "split-attn: the q32x2 counter's first use is inside a capture")
        from sglang.srt.layers.attention.trtllm_mla_backend import make_persistent_multi_ctas_kv_counter_buffer
        c = ex.c32 = make_persistent_multi_ctas_kv_counter_buffer(dev, SPLIT_HEADS // 2, 2 * ex.O)
    return c


def sx_kernel(sg, kind, ex, q, kv, pt, top_k, lens, seq, out, fi):
    if kind == "dec" and SPLIT_DEC_TILE == "q32x2":
        n, h = q.shape[0], SPLIT_HEADS // 2
        return sg.fid.trtllm_batch_decode_with_kv_cache_mla(
            query=q.view(2 * n, 1, h, SX_D), kv_cache=kv, block_tables=pt.repeat_interleave(2, dim=0).unsqueeze(1),
            seq_lens=seq.repeat_interleave(2), sparse_mla_top_k=top_k, sparse_mla_top_k_lens=lens.repeat_interleave(2),
            out=out.view(2 * n, 1, h, SX_D), **dict(fi, multi_ctas_kv_counter_buffer=sx_counter32(ex, q.device)))
    return sg.fid.trtllm_batch_decode_with_kv_cache_mla(query=q, kv_cache=kv, block_tables=pt.unsqueeze(1), seq_lens=seq,
                                                        sparse_mla_top_k=top_k, sparse_mla_top_k_lens=lens, out=out, **fi)


def sx_dec_blocks(backend, M, W):
    bq = sx_block(backend, M, "dec", W)
    return min(bq, M - (W - 1) * bq) >= SPLIT_DEC_MIN_BLOCK


def sx_graph_ready(kind, M):
    ex = _SX.get("ex")
    return "off" not in _SX and ex is not None and (kind, M) in ex.warm


@triton.jit(do_not_specialize=["a", "n", "total"])
def _g3_own_kernel(src_ptr, out_ptr, a, n, total, BLOCK: tl.constexpr):
    f = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    own = (f >= a) & (f < a + n)
    x = tl.load(src_ptr + (f - a), mask=own, other=-1)
    tl.store(out_ptr + f, x, mask=f < total)


def g3_register(attn, pk, layer_id, role, tp):
    if role != "verify" or tp is None or type(pk) is not SplitAttn or pk.role != "verify":
        return
    facts = (getattr(attn, "next_skip_topk", None), getattr(attn, "skip_topk", None),
             getattr(attn, "is_nextn", None))
    if facts == (False, False, False):
        _SX.setdefault("g3", {})[layer_id] = pk
    else:
        _log.warning("split-attn G3: model.layers.%d keeps ROWSHARD's top-k gather (next_skip_topk / skip_topk / "
                     "is_nextn = %s)", layer_id, facts)


def g3_serves(tp, layer_id, R, rows, pad, fb):
    if not (SPLIT_DEC and SPLIT_DEC_G3 and FP8PACK == 2 and tp is not None
            and torch.cuda.is_current_stream_capturing() and fb.forward_mode.is_target_verify()):
        return False
    attn = _SX.get("g3", {}).get(layer_id)
    if attn is None or rows != pad:
        return False
    lo, hi = sx_dec_rows(attn.role)
    if not (lo <= pad <= hi and sx_graph_ready("dec", pad)):
        return False
    ex, backend = _SX["ex"], sx_backend(attn.env)
    return (bool(getattr(backend, "use_fused_topk", False)) and ex.world == tp.world_size
            and sx_dec_blocks(backend, pad, tp.world_size) and R == sx_block(backend, pad, "dec", tp.world_size) <= ex.O)


def g3_precompile(dev):
    done = _SX.setdefault("g3k", set())
    if dev in done:
        return
    _need(not torch.cuda.is_current_stream_capturing(), "split-attn G3: the own-rows kernel's first launch is inside a "
          "CUDA-graph capture (SGLang's runner warms every graph eagerly first)")
    src = torch.arange(1, 17, dtype=torch.int32, device=dev)
    out = torch.zeros(64, dtype=torch.int32, device=dev)
    _g3_own_kernel[(triton.cdiv(64, G3_BLOCK),)](src, out, 16, 16, 64, BLOCK=G3_BLOCK, num_warps=G3_WARPS)
    want = torch.full((64,), -1, dtype=torch.int32, device=dev)
    want[16:32] = src
    _need(torch.equal(out, want), "split-attn G3: the own-rows kernel's self-check failed")
    done.add(dev)


def g3_own(tp, send, layer_id, R, rows, pad):
    C, me = send.shape[1], tp.rank_in_group
    _need(send.dtype == torch.int32 and send.dim() == 2 and send.shape[0] == R and send.is_contiguous()
          and 0 < rows <= pad and send.device in _SX.get("g3k", ()), f"split-attn G3: layer {layer_id} shard "
          f"{send.dtype} {tuple(send.shape)} (R {R}, rows {rows}, pad {pad}) or an uncompiled own-rows kernel")
    r0 = min(rows, me * R)
    n = min(rows, r0 + R) - r0
    out = torch.empty((pad, C), dtype=torch.int32, device=send.device)
    _g3_own_kernel[(triton.cdiv(pad * C, G3_BLOCK),)](send, out, r0 * C, n * C, pad * C, BLOCK=G3_BLOCK,
                                                      num_warps=G3_WARPS)
    _SX["ng"][layer_id] = (pad, r0, n)
    return out


def rs_gather(tp, send, layer_id, R, rows, pad, fb=None):
    _SX.setdefault("ng", {}).pop(layer_id, None)
    if fb is not None and g3_serves(tp, layer_id, R, rows, pad, fb):
        return g3_own(tp, send, layer_id, R, rows, pad)
    if SPLIT_DEC and SPLIT_DEC_G3 and not torch.cuda.is_current_stream_capturing():
        g3_precompile(send.device)
    return ags_gather(tp, send, layer_id, R, rows, pad)


def g3_take(attn):
    ng = _SX.get("ng")
    return ng.pop(attn.layer_id, None) if ng else None


def g3_hold(attn, ng, why, kind, q):
    ex, M = _SX.get("ex"), (q.shape[0] if torch.is_tensor(q) else -1)
    blk = None
    if ex is not None and M > 0:
        bq = sx_block(sx_backend(attn.env), M, "dec", ex.world)
        a = min(M, ex.rank * bq)
        blk = (M, a, max(0, min(M, a + bq) - a))
    _need(why is None and kind == "dec" and attn.role == "verify" and torch.cuda.is_current_stream_capturing()
          and blk == tuple(ng), f"split-attn G3: {attn.prefix}: the indexer kept only its block's top-k rows {ng} (no "
          f"gather) but this attention call does not split exactly that block (why {why}, kind {kind}, block {blk}): "
          "predicate disagreement")


def g3_guard(attn):
    ng = _SX.get("ng")
    _need(not ng or attn.layer_id not in ng, f"split-attn G3: {attn.prefix} [{attn.role}]: whole-batch attention on a "
          "layer whose indexer kept only its own block's top-k rows: predicate disagreement")


def tp2_verify_prepare(layer):
    attn = layer._modules.get("self_attn")
    kids = getattr(attn, "_modules", {})
    swaps = {}
    nn = wtc_verify_nn(kids.get("indexer"), layer.layer_id)
    if nn is not None:
        swaps["indexer"] = nn
    m = kids.get("attn_mqa")
    if SPLIT_DEC and FP8PACK == 2 and m is not None:
        _need(SPLIT_DEC_TILE == "q32x2", f"split-attn: SPLIT_DEC_TILE {SPLIT_DEC_TILE!r}: T ships only the q32x2 tile")
        swaps["attn_mqa"] = SplitAttn(m, _QpadEnv(), f"model.layers.{layer.layer_id}.self_attn", "verify2")
    return pj_proxy(layer, {"self_attn": pj_proxy(attn, swaps)}) if swaps else None

SB_BACK = True
SB_CLASS = "auto"
SB_ROW_TOL = 1e-2
SB_CHECKS = 1
SB_PULL = 0
SB_PULL_BLOCK = 4096
SB_STRICT = False
_SB = {"seen": set()}


@triton.jit
def _sb_add_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    f = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = f < n
    x = tl.load(a_ptr + f, mask=m).to(tl.float32) + tl.load(b_ptr + f, mask=m).to(tl.float32)
    tl.store(o_ptr + f, x.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)


@triton.jit
def _sb_pull_kernel(buf_ptr, delta_ptr, dst_ptr, src_off, n, s, steps, nprog, BLOCK: tl.constexpr):
    d = tl.multiple_of(tl.load(delta_ptr + s), 4)
    for i in range(0, steps):
        f = (i * nprog + tl.program_id(0)).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = f < n
        tl.store(dst_ptr + f, tl.load(buf_ptr + d + src_off + f, mask=m), mask=m)


def sb_add(a, b, out):
    n = a.numel()
    _need(a.dtype == b.dtype == out.dtype == torch.bfloat16 and b.numel() == n and out.numel() == n
          and a.is_contiguous() and b.is_contiguous() and out.is_contiguous(), "split-back: the partial sum's operands")
    if n:
        _sb_add_kernel[(triton.cdiv(n, 4096),)](a, b, out, n, BLOCK=4096, num_warps=8)
    return out


class SbLayer:

    def __init__(self, prefix, proxy, zc, ffn, stock_values=None):
        self.prefix, self.proxy, self.zc, self.ffn, self.ar = prefix, proxy, zc, ffn, ffn.ar
        self.stock_values, self.halves, self.pending, self.checks, self.on = stock_values, None, None, 0, False
        self.hl, self.v = getattr(proxy, "num_local_heads", None), getattr(proxy, "v_head_dim", None)
        self.bcheck, self.bad = None, None


class SbProj(torch.nn.Module):

    def __init__(self, zc, sb):
        super().__init__()
        self.zc = zc
        object.__setattr__(self, "sb", sb)

    def forward(self, x, *rest, **extra):
        p = self.sb.pending
        if p is None:
            return self.zc(x, *rest, **extra)
        _need(p.stage == "values" and not rest and not extra and torch.is_tensor(x) and x.data_ptr() == p.ptr,
              f"split-back: {self.sb.prefix} o_proj is not called on the SB call's hand-off ({p.stage})")
        p.stage = "proj"
        return self.zc.ar.buf[:p.M], None


class SbFfn:
    __slots__ = ("ffn", "sb")

    def __init__(self, ffn, sb):
        self.ffn, self.sb = ffn, sb

    def prepare(self, hidden_states, forward_batch):
        p = self.sb.pending
        if p is None:
            b = self.sb.bcheck
            if b is None:
                return self.ffn.prepare(hidden_states, forward_batch)
            self.sb.bcheck = None
            return sb_bound(self.sb, b, hidden_states, forward_batch)
        _need(p.stage == "proj", f"split-back: {self.sb.prefix} FFN entry reached at hand-off stage {p.stage}")
        self.sb.pending = None
        return sb_prepare(self.ffn, p, hidden_states, forward_batch)

    def finish(self, hidden_states, forward_batch):
        return self.ffn.finish(hidden_states, forward_batch)


def sb_values(sb, attn_output, zero_allocator):
    p = sb.pending
    if p is None:
        return sb.stock_values(attn_output, zero_allocator)
    _need(p.stage == "attn" and torch.is_tensor(attn_output) and attn_output.data_ptr() == p.ptr,
          f"split-back: {sb.prefix} project_mla_values is not called on the SB call's hand-off ({p.stage})")
    p.stage = "values"
    return attn_output


def sb_facts(proxy, zc, ffn):
    from sglang.srt.lora.utils import kv_b_lora_correction
    from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla as fm
    mx, w = fm.DeepseekMLAForwardMixin, getattr(proxy, "w_vc", None)
    hl, v, r = proxy.num_local_heads, proxy.v_head_dim, proxy.kv_lora_rank
    _need(type(proxy).project_mla_values is mx.project_mla_values and type(proxy).forward_absorb_core
          is mx.forward_absorb_core and fm._finish_absorb_output.__module__ == fm.__name__
          and not hasattr(proxy, "prepare_attention_output_gate") and not getattr(proxy, "_kimi_split_gguf_kv_b", False)
          and not proxy.use_deep_gemm_bmm and not fm._is_musa and not fm._SGLANG_EXPERIMENTAL_LORA_OPTI
          and kv_b_lora_correction(proxy) is None and torch.is_tensor(w) and w.dtype == torch.bfloat16
          and tuple(w.shape) == (hl, r, v) and tuple(w.stride()) == (r * v, 1, r) and r == SX_D
          and 2 * hl == SPLIT_HEADS and hl * v == zc.inner.weight.shape[1] and zc.slot is None
          and ffn.mega is not None and ffn.ar.tp.world_size == 2,
          "split-back: the attention tail is not stock's BF16 W_UV bmm -> ZcProj on 2 x 32 heads with W's mega "
          "boundary (or LoRA / an output gate / DeepGEMM bmm is on)")


def sb_attach(state):
    attn, ffn = state.attn, state.ffn
    if not SB_BACK or attn.kind != "mla" or ffn is None or ffn.mega is None or state.proxy is None:
        return
    proxy = attn.proxy
    sx, zc = proxy._modules.get("attn_mqa"), proxy._modules.get("o_proj")
    if type(sx) is not SplitAttn or sx.role != "fp8proxy":
        return
    sb = SbLayer(sx.prefix, proxy, zc, ffn)
    try:
        _need(type(zc) is ZcProj and state.proxy.ffn_boundary is ffn,
              "split-back: o_proj / ffn_boundary are not ZcProj / ArFfn")
        sb_facts(proxy, zc, ffn)
        sb.stock_values = types.MethodType(type(proxy).project_mla_values, proxy)
    except Exception as exc:
        sb.bad = f"{type(exc).__name__}: {exc}"[:400]
        sx.sb = sb
        _log.warning("split-back: %s not attached (%s): S's path; voted off at its first SB call", sx.prefix, sb.bad)
        return
    sx.sb = sb
    object.__setattr__(proxy, "project_mla_values", functools.partial(sb_values, sb))
    proxy._modules["o_proj"] = SbProj(zc, sb)
    object.__setattr__(state.proxy, "ffn_boundary", SbFfn(ffn, sb))


def sb_why(attn, kind, M, bq, ex, check):
    sb = getattr(attn, "sb", None)
    if sb is None or not SB_BACK or kind != "pf":
        return "-"
    if "off" in _SB:
        return "refused"
    if check:
        return "split-check"
    if bq > ex.O or ex.world != 2 or M < PF_MIN_ROWS:
        return "shape"
    if sb.ffn.sg.in_graph() or torch.cuda.is_current_stream_capturing():
        return "graph"
    return None


def sb_view(attn, kind, fb, M, bq, ex, check):
    why = sb_why(attn, kind, M, bq, ex, check)
    if why == "-":
        return ex
    sb = attn.sb
    key = (sb.prefix, why)
    if why is not None:
        if key not in _SB["seen"] and why != "split-check":
            _SB["seen"].add(key)
            _log.warning("split-back: %s rows=%d: S's path (%s)", sb.prefix, M, why)
        return ex
    _need(sb.pending is None and sb.bcheck is None,
          f"split-back: {sb.prefix}: the previous SB call's hand-off was never consumed")
    check = sb.checks < SB_CHECKS
    if check and sx_placeholder(sx_env(), fb, M):
        return ex
    return sb_view_of(ex, sb, M, check)


def sb_view_of(ex, sb, M, check):
    view = copy.copy(ex)
    done = []

    def oget(out, rng, delta=None):
        _need(delta is None and not done, "split-back: a second o pull in one call (SB serves one round)")
        done.append(True)
        sb_oget(ex, sb, M, check, out, rng)
    view.oget = oget
    view.sb_mode = "check" if check else "serve"
    return view


def sb_oget(ex, sb, M, check, out, rng):
    if sb.halves is None:
        sb_fetch(sb, ex)
        if "off" in _SB:
            return ex.oget(out, rng)
    if check:
        ex.oget(out, rng)
        return sb_check(sb, ex, out, rng, M)
    me = ex.rank
    a, n = rng[me]
    h = sb.ar.hidden
    _need(n * h // 2 <= ex.owords and rng[1 - me][1] * h // 2 <= ex.owords, "split-back: rows exceed the o stage")
    sb_back(sb, ex.ostage(n).view(n, SPLIT_HEADS, SX_D), sb_f1(ex, me, n, h))
    sb.pending = types.SimpleNamespace(stage="attn", ptr=out.data_ptr(), M=M, rng=[tuple(r) for r in rng], ex=ex)
    if not sb.on:
        sb.on = True
        _log.warning("split-back: %s [fp8proxy pf] rows=%d rank=%d: on (own rows [%d, %d) x %d heads: W_UV + o_proj "
                     "K-halves in rank order, one fp32 rounding; AG boundary, pull %s)", sb.prefix, M, me, a, a + n,
                     SPLIT_HEADS, "copy engine")


def sb_f1(ex, s, n, h):
    return ex.owner[2][s][ex.qwords:ex.qwords + n * h // 2].view(torch.bfloat16).view(n, h)


def sb_back(sb, o3, f1):
    n, zc, hl, v = o3.shape[0], sb.zc, sb.hl, sb.v
    parts = []
    for s, (wv, w8, s8) in enumerate(sb.halves):
        x = torch.empty((n, hl * v), dtype=torch.bfloat16, device=o3.device)
        torch.bmm(o3[:, s * hl:(s + 1) * hl].transpose(0, 1), wv, out=x.view(n, hl, v).transpose(0, 1))
        q, sc = zc.quant(x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=zc.ue8m0)
        del x
        y = torch.empty((n, f1.shape[1]), dtype=torch.bfloat16, device=o3.device)
        zc.gemm(q, sc, w8, s8, y, BLOCK, BLOCK)
        del q, sc
        parts.append(y)
    return sb_add(parts[0], parts[1], f1)


def sb_span(t):
    span = 1 + sum((d - 1) * st for d, st in zip(t.shape, t.stride()) if d > 0)
    return t.as_strided((span,), (1,)).view(torch.uint8)


def sb_sum(u8):
    x = u8.view(torch.int32) if u8.numel() % 4 == 0 else u8.to(torch.int32)
    w = torch.arange(x.numel(), device=x.device, dtype=torch.int32) % 65521 + 1
    return int(x.sum()), int((x * w).sum())


def sb_fetch(sb, ex):
    tp, me, W = sx_env().tp, ex.rank, ex.world
    why, own, spans, gots = None, None, None, None
    try:
        _need(sb.bad is None, f"split-back: not attached ({sb.bad})")
        own = (sb.proxy.w_vc, sb.zc.inner.weight, sb.zc.inner.weight_scale_inv)
        spans = [sb_span(t) for t in own]
        gots = [torch.empty((W, u8.numel()), dtype=torch.uint8, device=u8.device) for u8 in spans]
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
    bad = [(r, w) for r, w in enumerate(tp.all_gather_object(why)) if w is not None]
    if bad:
        return sb_refuse(f"{sb.prefix} peer-half fetch: rank {bad[0][0]} {bad[0][1]} (voted before the gathers)")
    for u8, got in zip(spans, gots):
        tp.all_gather_into_tensor(got.view(-1), u8)
    halves, sums = [[None] * len(own) for _ in range(W)], None
    try:
        sums = []
        for i, (t, u8, got) in enumerate(zip(own, spans, gots)):
            for s in range(W):
                halves[s][i] = t if s == me else got[s].clone().view(t.dtype).as_strided(t.shape, t.stride(), 0)
            sums.append([sb_sum(u8)] + [sb_sum(sb_span(halves[s][i])) for s in range(W)])
        del gots
    except Exception as exc:
        why, sums = f"{type(exc).__name__}: {exc}", None
    votes = tp.all_gather_object((why, sums))
    bad = [(r, w) for r, (w, _) in enumerate(votes) if w is not None]
    if not bad:
        bad = [(r, f"tensor {i}: its copy of rank {s}'s bytes differs from rank {s}'s own")
               for r, (_, sm) in enumerate(votes) for i in range(len(own)) for s in range(W)
               if sm[i][1 + s] != votes[s][1][i][0]]
    if bad:
        return sb_refuse(f"{sb.prefix} peer-half fetch: rank {bad[0][0]} {bad[0][1]}")
    sb.halves = [tuple(h) for h in halves]
    nb = sum(sb_span(t).numel() for s, h in enumerate(sb.halves) if s != me for t in h)
    _log.warning("split-back: %s tp%d/%d fetched the peer halves (w_vc, FP8 o_proj, scales: %.1f MiB), checksums agree; "
                 "device free %d MiB", sb.prefix, me, W, nb / 2 ** 20, sb_free_mib(own[0].device))


def sb_free_mib(dev):
    try:
        return int(torch.cuda.mem_get_info(dev)[0]) >> 20
    except Exception:
        return -1


def sb_refuse(why):
    _need(not SB_STRICT, f"split-back: {why} (SB_STRICT; set SB_BACK False)")
    _SB["off"] = why
    _log.warning("split-back: OFF on every rank (voted), S's path from here: %s", why)


def sb_cmp(got, want):
    g, w = got.reshape(got.shape[0], -1), want.reshape(want.shape[0], -1)
    v = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[g.element_size()]
    if g.shape == w.shape and g.dtype == w.dtype and torch.equal(g.view(v), w.view(v)):
        return True, True, 0.0, 1.0
    if g.shape != w.shape or not g.numel():
        return False, False, float("inf"), 0.0
    err = (g.float() - w.float()).norm(dim=1) / w.float().norm(dim=1).clamp_min(1e-30)
    same = float((g.view(v) == w.view(v)).all(dim=1).float().mean())
    fin = bool(torch.isfinite(g).all()) and bool(torch.isfinite(w).all())
    return False, fin, float(err.max()), same


def sb_ok(bit, fin, err):
    return bool(bit) if SB_CLASS == "bitwise" else bool(bit or (fin and err <= SB_ROW_TOL))


def sb_check(sb, ex, out, rng, M):
    me, ar = ex.rank, sb.ar
    a, n = rng[me]
    got = torch.empty((n, ar.hidden), dtype=torch.bfloat16, device=out.device)
    sb_back(sb, ex.ostage(n).view(n, SPLIT_HEADS, SX_D), got)
    ref = sb.zc(sb.stock_values(out.view(M, sb.hl, SX_D), None))[0]
    ar.zc(ref)
    bit, fin, err, same = sb_cmp(got, ref[a:a + n])
    del got
    votes = sx_env().tp.all_gather_object((sb_ok(bit, fin, err), bit, round(err, 7), round(same, 6)))
    if not all(v[0] for v in votes):
        return sb_refuse(f"{sb.prefix} rows={M}: the back half differs from stock's tail (ok, bitwise, rel-L2 max, "
                         f"bitwise row share) {votes} (class {SB_CLASS}, bound {SB_ROW_TOL})")
    sb.bcheck = types.SimpleNamespace(M=M, rng=[tuple(r) for r in rng], ex=ex, bit=all(v[1] for v in votes),
                                      err=max(v[2] for v in votes), same=min(v[3] for v in votes))


def sb_entry_facts(ffn, M, hidden_states, forward_batch):
    stream = forward_batch.residual_stream
    c = stream.pending
    _need(c is not None and ffn._takes(c.value, ffn.limit1) and type(hidden_states) is ffn.sg.OwedOutput
          and hidden_states.contribution is c and stream.transport is None and type(c.owed) is ffn.sg.DeclaredSum
          and c.owed.group is ffn.group1 and c.owed.complete_sum is None, "split-back: the attention output is not the "
          "declared plain sum the FFN entry was checked for")
    x = c.value
    _need(x.shape[0] == M and x.data_ptr() == ffn.ar.buf.data_ptr() and c.update is ffn.au and ffn.mega is not None
          and not (ffn.sg.in_graph() or torch.cuda.is_current_stream_capturing()),
          "split-back: the FFN entry is not W's eager boundary on the SB call's buffer head")
    return x


def sb_bound(sb, b, hidden_states, forward_batch):
    ffn, why, x, st, res = sb.ffn, None, None, None, None
    try:
        x = sb_entry_facts(ffn, b.M, hidden_states, forward_batch)
        st, res = ffn._m1_state(forward_batch, x.shape[0])
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"[:300]
    bad = [(r, w) for r, w in enumerate(sx_env().tp.all_gather_object(why)) if w is not None]
    if bad:
        sb_refuse(f"{sb.prefix} rows={b.M}: the check call's FFN entry, rank {bad[0][0]}: {bad[0][1]}")
        return ffn.prepare(hidden_states, forward_batch)
    new, x, st.h_res, st.h_post = sb_bound_core(sb, b, x, res, st.h_res, st.h_post, ffn.mega, ffn.stock.norm)
    forward_batch.residual_stream.write(new)
    return x


def sb_bound_core(sb, b, x, resid, h_res, h_post, layer, norm):
    ar, t, h = sb.ar, b.M, sb.ar.hidden
    out = mega_entry(ar, x, resid, h_res, h_post, layer, norm)
    want = (out[0].view(t, 4, h), out[1], out[3].view(t, 4, 1), out[2].view(t, 4, 4))
    res3, comb0, post0, me = resid.view(t, 4, h), h_res.view(t, 4, 4), h_post.view(t, 4, 1), b.ex.rank
    bit, fin, err = True, True, 0.0
    with ar.sg.dg_sms(MEGAPF_SMS):
        for a, n in [b.rng[me]] + [r for s, r in enumerate(b.rng) if s != me]:
            if not n:
                continue
            got = (torch.empty((n, 4, h), dtype=torch.bfloat16, device=x.device),
                   torch.empty((n, h), dtype=torch.bfloat16, device=x.device),
                   torch.empty((n, 4, 1), dtype=torch.float32, device=x.device),
                   torch.empty((n, 4, 4), dtype=torch.float32, device=x.device))
            sb_mega_rows(ar, x[a:a + n], res3[a:a + n], post0[a:a + n], comb0[a:a + n], layer, norm, *got)
            for g, w in zip(got, want):
                bi, fi, er, _ = sb_cmp(g, w[a:a + n])
                bit, fin, err = bit and bi, fin and fi, max(err, er)
            del got
    votes = sx_env().tp.all_gather_object((sb_ok(bit, fin, err), bit, round(err, 7)))
    if not all(v[0] for v in votes):
        sb_refuse(f"{sb.prefix} rows={t}: SB's AG row plan (mega_mhc on S's row halves) differs from W's mega_entry "
                  f"(ok, bitwise, rel-L2 max) {votes} (class {SB_CLASS}, bound {SB_ROW_TOL})")
        return out
    sb.checks += 1
    cls = "bitwise" if b.bit and all(v[1] for v in votes) else "audit"
    _log.warning("split-back: %s rows=%d rank=%d check %d/%d pass: back half == stock's W_UV + o_proj + sum (rel-L2 max "
                 "%.3g, bitwise rows %.4f); AG row plan == W's mega_entry (rel-L2 max %.3g); class %s", sb.prefix, t, me,
                 sb.checks, SB_CHECKS, b.err, b.same, max(v[2] for v in votes), cls)
    return out


def sb_pull(ex, s, dst, n, delta=None, programs=None):
    p = SB_PULL if programs is None else programs
    if not n:
        return
    if not p:
        dst[:n].copy_(ex.owner[2][s][ex.qwords:ex.qwords + n])
        return
    steps = triton.cdiv(triton.cdiv(n, SB_PULL_BLOCK), p)
    _sb_pull_kernel[(p,)](ex.buf, ex.delta if delta is None else delta, dst, ex.qwords, n, s, steps, p,
                          BLOCK=SB_PULL_BLOCK, num_warps=SX_WARPS)


def sb_mega(ar, x, a, b, res3, post0, comb0, layer, norm, new, li, post, comb):
    sb_mega_rows(ar, x, res3[a:b], post0[a:b], comb0[a:b], layer, norm, new[a:b], li[a:b], post[a:b], comb[a:b])


def sb_mega_rows(ar, x, res3, post0, comb0, layer, norm, new, li, post, comb):
    cfg = layer.config
    ar.sg.dg.mega_mhc(x, res3, None, post0, comb0, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, 4,
                      cfg.rms_norm_eps, cfg.hc_eps, 2.0, cfg.hc_eps, cfg.hc_sinkhorn_iters, norm.weight.data,
                      norm.variance_epsilon, 1.0, new, None, post, comb, y_bf16=li)


def sb_entry(ar, p, resid, h_res, h_post, layer, norm):
    ex, t, h = p.ex, p.M, ar.hidden
    me, f32, dev = ex.rank, torch.float32, resid.device
    (a, n), peers = p.rng[me], [(s, r[0], r[1]) for s, r in enumerate(p.rng) if s != me and r[1]]
    cur = torch.cuda.current_stream()
    new, li = torch.empty((t, 4, h), dtype=torch.bfloat16, device=dev), torch.empty((t, h), dtype=torch.bfloat16, device=dev)
    post, comb = torch.empty((t, 4, 1), dtype=f32, device=dev), torch.empty((t, 4, 4), dtype=f32, device=dev)
    ar.side.wait_stream(cur)
    with torch.cuda.stream(ar.side):
        ex.sync()
        for s, b0, m in peers:
            sb_pull(ex, s, ar.buf[b0:b0 + m].view(torch.int32).view(-1), m * h // 2)
        ar.evs[0].record(ar.side)
    res3, comb0, post0 = resid.view(t, 4, h), h_res.view(t, 4, 4), h_post.view(t, 4, 1)
    with ar.sg.dg_sms(MEGAPF_SMS):
        if n:
            sb_mega(ar, sb_f1(ex, me, n, h), a, a + n, res3, post0, comb0, layer, norm, new, li, post, comb)
        cur.wait_event(ar.evs[0])
        for s, b0, m in peers:
            sb_mega(ar, ar.buf[b0:b0 + m], b0, b0 + m, res3, post0, comb0, layer, norm, new, li, post, comb)
    return new.view(t, 4 * h), li, comb.view(t, 16), post.view(t, 4)


def sb_prepare(ffn, p, hidden_states, forward_batch):
    x = sb_entry_facts(ffn, p.M, hidden_states, forward_batch)
    st, res = ffn._m1_state(forward_batch, x.shape[0])
    new, x, st.h_res, st.h_post = sb_entry(ffn.ar, p, res, st.h_res, st.h_post, ffn.mega, ffn.stock.norm)
    forward_batch.residual_stream.write(new)
    return x

OWN_ROWS_PF = True
ORP_GATES = "probe"
ORP_TOPK_INTO = True
ORP_FILL = "auto"
ORP_CHECKS = 2
ORP_HEAL_MAX = 4
ORP_STRICT = False
_ORP = {"checks": 0, "heals": 0, "seen": set(), "cut": None, "gp": {}}


def orp_mods():
    m = _ORP.get("mods")
    if m is None:
        from sglang.srt.layers.attention.dsa import kpool_fp8_index as kif
        from sglang.srt.state_capturer import indexer_topk as itk
        try:
            from sglang.kernels.ops.attention.dsv4.topk import topk_transform_kpool_ragged_v2 as v2
        except ImportError:
            v2 = None
        m = _ORP["mods"] = types.SimpleNamespace(kif=kif, itk=itk, v2=v2)
        _need(callable(kif.gather_index_k_scale_prefix_into) and callable(itk.get_global_indexer_capturer),
              "own-rows-pf: SGLang's K-pool prefix gather / indexer capturer accessor are missing")
    return m


def orp_env():
    env = _ORP.get("env")
    if env is None:
        env = _ORP["env"] = _QpadEnv()
    return env


def orp_why(sh, x, q_lora, fb, return_indices):
    if not OWN_ROWS_PF:
        return "switch", None
    if "off" in _ORP:
        return "off", None
    if not (SPLIT_PF and FP8PACK == 2 and ROTQ):
        return "split-off", None
    if not return_indices:
        return "no-indices", None
    if not fb.forward_mode.is_extend_without_speculative() or torch.cuda.is_current_stream_capturing():
        return "mode", None
    if not (torch.is_tensor(x) and torch.is_tensor(q_lora) and x.dim() == 2 and q_lora.dim() == 2
            and q_lora.shape[0] == x.shape[0] and q_lora.is_contiguous()):
        return "inputs", None
    M = x.shape[0]
    if not SPLIT_PF_MIN_ROWS <= M <= SPLIT_PF_MAX_ROWS:
        return "rows", None
    ex = _SX.get("ex")
    if "off" in _SX or ex is None:
        return "no-exchange", None
    if _SX["checks"].get("pf", 0) < SPLIT_CHECKS:
        return "split-checks", None
    if ex.hl * ex.world != SPLIT_HEADS:
        return "heads", None
    spec = getattr(fb, "spec_info", None)
    if spec is not None and getattr(spec, "dsa_seed_topk_capture", None) is not None:
        return "seed-capture", None
    m = orp_mods()
    if m.itk.get_global_indexer_capturer() is not None:
        return "capturer", None
    env = orp_env()
    if env.in_graph_scope() or env.padded_rows(M, fb) is not None:
        return "graph-or-padded", None
    backend = sx_backend(env)
    if type(backend) is not env.Dsa:
        return "backend-class", None
    md = backend.forward_metadata
    plan, seq = getattr(md, "kpool_extend_plan", None), getattr(md, "dsa_cache_seqlens_int32", None)
    if plan is None:
        return "no-plan", None
    if not (torch.is_tensor(seq) and seq.shape[0] == M and plan.seq_lens_expanded.shape[0] == M
            and plan.pooled_seq_lens_expanded.shape[0] == M and plan.ragged_q_ks.shape[0] == M):
        return "plan-rows", None
    sg = sx_env()
    if (getattr(fb, "attn_cp_metadata", None) is not None or sg.db.dsa_use_prefill_cp(fb)
            or sg.db.get_in_autotune_dummy_run() or backend.workspace_buffer is None or backend.use_mha):
        return "backend", None
    bq = sx_block(backend, M, "pf", ex.world)
    r0, r1 = min(M, ex.rank * bq), min(M, (ex.rank + 1) * bq)
    if r1 <= r0:
        return "empty", None
    fill = ORP_FILL == "always" or (ORP_FILL == "auto" and not backend.use_fused_topk)
    return None, types.SimpleNamespace(M=M, r0=r0, r1=r1, x=x, q_lora=q_lora, plan=plan, md=md, fill=fill, cut=False,
                                       qk=0, vx=_rq_ver(x), vq=_rq_ver(q_lora))


def orp_note(layer_id, why, p, M):
    key = (why,) if why is not None else ("on", layer_id)
    if why in ("switch", "mode", "no-indices", "rows") or key in _ORP["seen"]:
        return
    _ORP["seen"].add(key)
    if why is None:
        _log.warning("own-rows-pf: layer %d rows=%d: indexer q path + top-k on rows [%d, %d) only (fill %s)", layer_id, M,
                     p.r0, p.r1, p.fill)
    else:
        _log.warning("own-rows-pf: whole-batch indexer (%s) rows=%d", why, M)


def orp_extend(nn, x, q_lora, positions, fb, layer_id, return_indices):
    sh, sg = nn.shim, nn.sg
    _ORP["cut"] = None
    why, p = orp_why(sh, x, q_lora, fb, return_indices)
    orp_note(layer_id, why, p, x.shape[0] if torch.is_tensor(x) and x.dim() else 0)
    sh._or = p
    try:
        return sg.Indexer._forward_cuda_impl(sh, x, q_lora, positions, fb, layer_id, return_indices)
    finally:
        sh._or = None


def orp_rows(sh, q_lora):
    p = sh._or
    if p is None:
        return q_lora
    p.qk += 1
    _need(p.qk == 1 and q_lora is p.q_lora, f"own-rows-pf: query call {p.qk} on q_lora {tuple(q_lora.shape)}, planned "
          f"for {p.M} rows")
    p.cut = True
    return q_lora[p.r0:p.r1]


def orp_gate_probe(st, x, a, b):
    key = (x.shape[0], a, b)
    v = _ORP["gp"].get(key)
    if v is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        g_all, g_own = wtc_project(st, x), wtc_project(st, x[a:b])
        v = _ORP["gp"][key] = g_all is not None and g_own is not None and _rsq_same(g_own, g_all[a:b])
        _log.warning("own-rows-pf: gate GEMM rows [%d, %d) of %d alone %s the whole batch's rows: %s gates", a, b,
                     x.shape[0], "==" if v else "!=", "own-row" if v else "whole-batch")
    return v


def orp_gate_rows(st, x, q_scale, a, b, head_weights=None):
    if head_weights is not None:
        hw = head_weights[a:b] if head_weights.shape[0] == x.shape[0] else head_weights
    elif ORP_GATES == "own" or (ORP_GATES == "probe" and orp_gate_probe(st, x, a, b)):
        hw = wtc_project(st, x[a:b])
    else:
        hw = wtc_project(st, x)
        if hw is None:
            hw = st._project_and_scale_head_gates(x)
        hw = hw[a:b]
    _need(hw is None or hw.shape[0] == q_scale.shape[0] == b - a, "own-rows-pf: head gates / q scales are not the rows'")
    return ge_scale(st, x[a:b], q_scale, hw)


def orp_gates(sh, x, q_scale, head_weights):
    p = sh._or
    if p is None or not p.cut:
        return rsq_gates(sh, x, q_scale, head_weights)
    _need(x.shape[0] == p.M and q_scale.shape[0] == p.r1 - p.r0, f"own-rows-pf: gates on x {tuple(x.shape)} / q scales "
          f"{tuple(q_scale.shape)} for rows [{p.r0}, {p.r1}) of {p.M}")
    return orp_gate_rows(sh._s, x, q_scale, p.r0, p.r1, head_weights)


def orp_qpath(sh, q_lora, x, a, b):
    st, sg = sh._s, sh._sg
    q = st.wq_b(q_lora[a:b])[0]
    q = rq_call(sh, q.reshape(q.shape[0], -1, st.head_dim))
    q8, qs = sg.act_quant(q, st.block_size, st.scale_fmt)
    return q8, orp_gate_rows(st, x, qs, a, b)


def orp_topk_into(st, logits, out, pool_lens, seq, pt, pt_ri, offs, starts):
    kpool = st.index_kpool
    v2, kif = orp_mods().v2, orp_mods().kif
    if offs is not None and offs.dim() == 2:
        _need(offs.shape[1] == 1, "own-rows-pf: top-k offsets are not one per row")
        offs = offs.squeeze(1)
    if (ORP_TOPK_INTO and v2 is not None and seq is not None and kif._kpool_topk_v2_layout_ok(logits=logits,
                                                                                             pool_size=kpool)
            and out.is_contiguous() and tuple(out.shape) == (logits.shape[0], st.index_topk + kpool - 1)
            and st.index_topk % kpool == 0):
        v2(scores=logits, pool_lens=pool_lens.to(torch.int32), out=out, pool_size=kpool,
           row_starts=None if starts is None else starts.to(torch.int32), token_seq_lens=seq.to(torch.int32),
           page_table=pt, page_table_row_index=None if pt_ri is None else pt_ri.to(torch.int32), out_offsets=offs)
        return
    r = st._topk_from_kpool_logits(logits=logits, pool_lens=pool_lens, seq_lens=seq, page_table=pt,
                                   topk_offsets=offs, row_starts=starts, out_rows=None, page_table_row_index=pt_ri)
    out.copy_(r)


def orp_rows_topk(sh, layer_id, q8, w, metadata, plan, a, b, out):
    st, ik, kif = sh._s, sh._sg.ik, orp_mods().kif
    M, n = out.shape[0], b - a
    _need(w.dim() == 3 and q8.shape[0] == w.shape[0] == n and plan.seq_lens_expanded.shape[0] == M,
          f"own-rows-pf: rows [{a}, {b}) query {tuple(q8.shape)} / weights {tuple(w.shape)} vs plan rows "
          f"{plan.seq_lens_expanded.shape[0]}")
    w = w.squeeze(-1)
    total_k = plan.ragged_total_k_rows
    kv = None
    if total_k > 0:
        k_u8, k_scale = plan.ragged_k_u8, plan.ragged_k_scale
        _need(k_u8 is not None and k_scale is not None, "own-rows-pf: the plan has no ragged K scratch")
        pool = ik.get_token_to_kv_pool()
        kif.gather_index_k_scale_prefix_into(pool=pool, buf=st._get_index_k_read_buffer(pool, layer_id),
                                             page_indices=plan.ragged_concat_page_table, seq_len=total_k, k_out=k_u8,
                                             scale_out=k_scale)
        kv = (k_u8.view(_FP8).contiguous(), k_scale.contiguous())
    method, amd = metadata.topk_transform_method, metadata.attn_metadata
    pt = pt_ri = offs = None
    if ik._should_fuse_kpool_topk(metadata):
        if method == ik.TopkTransformMethod.PAGED:
            pt, pt_ri = plan.ragged_paged_page_table, plan.ragged_paged_page_table_row_index
        elif method == ik.TopkTransformMethod.RAGGED:
            offs = amd.topk_indices_offset
    _need(pt is None or pt_ri is not None, "own-rows-pf: a fused page table without its row index")
    ks, ke = plan.ragged_q_ks, plan.ragged_q_ke
    pls, seq = plan.pooled_seq_lens_expanded, plan.seq_lens_expanded
    for c in ik._mqa_logits_row_chunks(num_rows=n, num_cols=total_k, device=q8.device):
        rows = slice(a + c.start, a + c.stop)
        if kv is None:
            logits = torch.empty((c.stop - c.start, 0), dtype=torch.float32, device=q8.device)
        else:
            logits = st._fp8_mqa_logits(q_fp8=q8[c].contiguous(), k_fp8=kv[0], k_scale=kv[1], weights=w[c].contiguous(),
                                        starts=ks[rows], ends=ke[rows], clean_logits=True)
        orp_topk_into(st, logits, out[rows], pls[rows], seq[rows], pt, None if pt_ri is None else pt_ri[rows],
                      None if offs is None else offs[rows], ks[rows])
        del logits
    return out


def orp_miss(want, got):
    rows = torch.arange(want.shape[0], device=want.device, dtype=torch.int64)[:, None] << 33
    ka, kb = rows + want.to(torch.int64) + 1, rows + got.to(torch.int64) + 1
    valid = want >= 0
    return int((~torch.isin(ka[valid], kb[got >= 0])).sum()), int(valid.sum())


def orp_reference(sh, p, fb, layer_id, metadata):
    st, sg = sh._s, sh._sg
    q = st.wq_b(p.q_lora)[0]
    q = rq_call(sh, q.reshape(q.shape[0], -1, st.head_dim))
    q8, qs = sg.act_quant(q, st.block_size, st.scale_fmt)
    w = rsq_gates(sh, p.x, qs, None)
    return q8, w, st._get_topk_ragged_kpool_plan(fb, layer_id, q8, w, metadata)


def orp_topk_plan(sh, fb, layer_id, q8, w, metadata):
    p, st = sh._or, sh._s
    if p is None or not p.cut:
        return st._get_topk_ragged_kpool_plan(fb, layer_id, q8, w, metadata)
    plan = metadata.attn_metadata.kpool_extend_plan
    if plan is not p.plan or q8.shape[0] != p.r1 - p.r0:
        orp_off(f"layer {layer_id}: the extend plan / query rows changed after the predicate")
        Q, Wt = orp_full(sh, p, q8, w)
        return st._get_topk_ragged_kpool_plan(fb, layer_id, Q, Wt, metadata)
    out = torch.empty((p.M, st.index_topk + st.index_kpool - 1), dtype=torch.int32, device=q8.device)
    if p.fill:
        out[:p.r0].fill_(-1)
        out[p.r1:].fill_(-1)
    orp_rows_topk(sh, layer_id, q8, w, metadata, plan, p.r0, p.r1, out)
    if _ORP["checks"] < ORP_CHECKS:
        _ORP["checks"] += 1
        rq8, rw, ref = orp_reference(sh, p, fb, layer_id, metadata)
        a, b = p.r0, p.r1
        same_q = _rsq_same(q8, rq8[a:b])
        wd = float((w - rw[a:b]).abs().max()) if w.numel() else 0.0
        wb = float(rw[a:b].abs().max()) if w.numel() else 0.0
        miss, valid = orp_miss(ref[a:b], out[a:b])
        ok = (bool(torch.isfinite(w).all()) and wd <= WTC_W_TOL * wb and miss <= WTC_TOPK_MAX * max(1, valid)
              and ref.shape == out.shape and ref.dtype == out.dtype)
        _log.warning("own-rows-pf: check %d/%d layer %d rows=%d rows [%d, %d): FP8 query bitwise %s, head weights max "
                     "|diff| %.3g (bitwise %s), top-k slots missing %d of %d (ordered %s): %s", _ORP["checks"],
                     ORP_CHECKS, layer_id, p.M, a, b, same_q, wd, bool(wd == 0.0), miss, valid,
                     torch.equal(ref[a:b], out[a:b]), "pass" if ok else "FAIL")
        if not ok:
            why = (f"layer {layer_id} rows={p.M}: own-row top-k / weights differ from the whole-batch indexer "
                   f"(missing {miss} of {valid}, weights {wd:.3g} vs {wb:.3g})")
            _need(not ORP_STRICT, f"own-rows-pf: {why} (ORP_STRICT; set OWN_ROWS_PF False)")
            orp_off(why)
            return ref
        del rq8, rw, ref
    _ORP["cut"] = types.SimpleNamespace(topk=out, sh=sh, layer_id=layer_id, metadata=metadata, plan=plan, M=p.M,
                                        r0=p.r0, r1=p.r1, x=p.x, q_lora=p.q_lora, vx=p.vx, vq=p.vq)
    return out


def orp_full(sh, p, q8, w):
    qs, ws = [], []
    for a, b in ((0, p.r0), (p.r0, p.r1), (p.r1, p.M)):
        if b <= a:
            continue
        if a == p.r0:
            qs.append(q8)
            ws.append(w)
        else:
            mq, mw = orp_qpath(sh, p.q_lora, p.x, a, b)
            qs.append(mq)
            ws.append(mw)
    return torch.cat(qs, 0), torch.cat(ws, 0)


def orp_topk_ragged(sh, args, kwargs):
    p, st = sh._or, sh._s
    if p is not None and p.cut:
        _need(len(args) >= 5, "own-rows-pf: _get_topk_ragged was not called positionally")
        fb, layer_id, q8, w, metadata = args[:5]
        orp_off(f"layer {layer_id}: an extend call without a K-pool plan after the predicate saw one")
        q8, w = orp_full(sh, p, q8, w)
        args = (fb, layer_id, q8, w, metadata) + tuple(args[5:])
    return st._get_topk_ragged(*args, **kwargs)


def orp_off(why):
    if "off" not in _ORP:
        _ORP["off"] = why
        _log.warning("own-rows-pf: OFF for this process (whole-batch indexer from here): %s", why)


def orp_heal(attn, kwargs):
    cut = _ORP.get("cut")
    if cut is None or kwargs.get("topk_indices") is not cut.topk:
        return
    _ORP["cut"] = None
    _need((cut.vx is None or _rq_ver(cut.x) == cut.vx) and (cut.vq is None or _rq_ver(cut.q_lora) == cut.vq),
          f"own-rows-pf: layer {cut.layer_id}: the held indexer inputs were written after the cut (cannot heal)")
    for a, b in ((0, cut.r0), (cut.r1, cut.M)):
        if b > a:
            q8, w = orp_qpath(cut.sh, cut.q_lora, cut.x, a, b)
            orp_rows_topk(cut.sh, cut.layer_id, q8, w, cut.metadata, cut.plan, a, b, cut.topk)
    _ORP["heals"] += 1
    why = f"{getattr(attn, 'prefix', '?')} [{getattr(attn, 'role', '?')}] rows={cut.M}"
    _log.warning("own-rows-pf: heal %d: %s reached whole-batch attention with own rows [%d, %d): missing rows recomputed",
                 _ORP["heals"], why, cut.r0, cut.r1)
    if _ORP["heals"] >= ORP_HEAL_MAX:
        orp_off(f"{_ORP['heals']} heals (the predicate disagrees with the split attention; last: {why})")


def orp_done(kwargs):
    cut = _ORP.get("cut")
    if cut is not None and kwargs.get("topk_indices") is cut.topk:
        _ORP["cut"] = None

FINADD_PF = True
FP_OVL = True
FP_CHECKS = 2
FP_STRICT = False
FP_CALL = "direct"
FP_OVL_GROUP = 1
FP_BH, FP_WARPS = 2048, 8
_FP = {"checks": 0, "off": None, "seen": set(), "ready": None, "evs": {}, "served": 0, "ovl": 0}


@triton.jit
def _fp_kernel(g_ptr, idx_ptr, w_ptr, s_ptr, o_ptr, gs, ss, osd, row0, H: tl.constexpr, K: tl.constexpr,
               BH: tl.constexpr):
    t = tl.program_id(0).to(tl.int64) + row0
    c = tl.program_id(1) * BH + tl.arange(0, BH)
    m = c < H
    acc = tl.zeros([BH], dtype=tl.float32)
    for k in tl.static_range(K):
        p = tl.load(idx_ptr + t * K + k)
        w = tl.load(w_ptr + t * K + k).to(tl.float32)
        ok = p != -1
        x = tl.load(g_ptr + tl.maximum(p, 0).to(tl.int64) * gs + c, mask=m & ok, other=0.0).to(tl.float32)
        acc = tl.where(ok, tl.fma(w, x, acc), acc)
    sh = tl.load(s_ptr + t * ss + c, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + t * osd + c, (acc.to(tl.bfloat16).to(tl.float32) + sh).to(tl.bfloat16), mask=m)


def fp_chunks(t):
    return 4 if t >= 12288 else 3 if t >= 6144 else 2


def fp_facts(state):
    f = getattr(state, "fp", None)
    if f is not None:
        return f[0]
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as fit
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
    px = state.proxy
    ex = px.experts
    facts = dict(
        nvfp4=isinstance(getattr(ex, "quant_method", None), ModelOptNvFp4FusedMoEMethod),
        deferrable=bool(getattr(ex, "supports_deferred_finalize", False)
                        or getattr(ex, "supports_routed_deferred_finalize", False)),
        helpers=callable(getattr(ex, "forward_deferred_finalize", None)),
        dwdp=bool(getattr(ex, "_dwdp_bound", False)), fuseep=bool(getattr(ex, "_use_ascend_fuseep", False)),
        shared=px.shared_experts is not None and px.num_fused_shared_experts == 0, tp1=bool(px._shared_expert_tp1),
        reduce=bool(px.reduce_results) or bool(getattr(ex, "reduce_results", False)
                                               and (getattr(ex, "moe_tp_size", 1) > 1 or getattr(ex, "moe_ep_size", 1) > 1)))
    ok = (facts["nvfp4"] and facts["deferrable"] and facts["helpers"] and not facts["dwdp"] and not facts["fuseep"]
          and facts["shared"] and not facts["tp1"] and not facts["reduce"])
    own = None
    if ok and FP_CALL in ("own", "direct"):
        from sglang.srt.layers.moe import utils as mu
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.moe.topk import TopKOutputChecker
        qm = ex.quant_method
        runner = getattr(qm, "runner", None)
        backend = getattr(qm, "_moe_runner_backend", None) or mu.get_moe_runner_backend()
        facts.update(
            backend=backend.is_flashinfer_trtllm() and not backend.is_flashinfer_trtllm_routed(),
            fused=getattr(runner, "fused_func", None) is fit.fused_experts_none_to_flashinfer_trtllm
            and not getattr(runner, "lora_enabled", True),
            mega=bool(getattr(ex, "_mega_moe_nvfp4", False)), pad=getattr(ex, "trtllm_padded_hidden_size", None),
            disp=type(getattr(ex, "dispatcher", None)) is StandardDispatcher,
            act=getattr(getattr(runner, "config", None), "activation", None) in ("silu", "relu2", "gelu", "situ"))
        ok = (facts["backend"] and facts["fused"] and not facts["mega"] and facts["pad"] is None and facts["disp"]
              and facts["act"])
        own = types.SimpleNamespace(fit=fit, bypassed=TopKOutputChecker.format_is_bypassed)
    why = None if ok else f"facts {facts}"
    state.fp = (why, fit, own)
    return why


def fp_why(state, hidden):
    x = state.x
    if not FINADD_PF:
        return "switch"
    if _FP["off"] is not None:
        return "off"
    if not ZC_AR2 or MOE_SIDE:
        return "route"
    if x.in_graph() or torch.cuda.is_current_stream_capturing():
        return "graph"
    t = hidden.shape[0]
    if not (PF_MIN_ROWS <= t <= x.ar.rows) or hidden.shape[1] != x.ar.hidden:
        return "rows"
    return fp_facts(state)


def fp_note(state, why, t):
    key = (why if why is None or len(why) < 16 else why[:16], getattr(state.module, "layer_id", None))
    if key not in _FP["seen"] and (why is None or why not in ("switch", "graph")):
        _FP["seen"].add(key)
        _log.warning("finadd-pf: model.layers.%s rows=%d %s", key[1], t, "on" if why is None else f"X's route: {why}")


def fp_valid(g, idx, w, shared, top_k, out, t):
    H = shared.shape[1]
    return (g.dtype == shared.dtype == out.dtype == torch.bfloat16 and g.dim() == 2 and g.stride(1) == 1
            and g.shape[1] >= H and idx.dtype == torch.int32 and idx.numel() >= t * top_k and idx.is_contiguous()
            and w.dtype in (torch.bfloat16, torch.float32) and w.shape[0] >= t and w.shape[-1] == top_k
            and w.is_contiguous() and shared.shape[0] == t and shared.stride(1) == 1 and out.shape[0] == t
            and out.shape[1] == H and out.stride(1) == 1)


def fp_refuse(g, idx, w, shared, out, t):
    _need(False, f"finadd-pf: deferred outputs g {g.dtype} {tuple(g.shape)} idx {idx.dtype} {tuple(idx.shape)} w {w.dtype} "
                 f"{tuple(w.shape)} shared {shared.dtype} {tuple(shared.shape)} out {out.dtype} {tuple(out.shape)} rows {t}")


def fp_events(dev):
    evs = _FP["evs"].get(dev)
    if evs is None:
        evs = _FP["evs"][dev] = [torch.cuda.Event() for _ in range(4)]
    return evs


def fp_experts(state, hidden, topk_output):
    ex = state.proxy.experts
    if FP_CALL not in ("own", "direct"):
        return ex.forward_deferred_finalize(hidden, topk_output, pre_quant_input=None)
    own = state.fp[2]
    fit = own.fit
    disp = ex._dispatch_with_pre_quant(hidden, topk_output, None)
    hs, topk = disp.hidden_states, disp.topk_output
    _need(disp.hidden_states_scale is None and hs.dtype == torch.bfloat16 and hs.dim() == 2 and own.bypassed(topk),
          "finadd-pf: the dispatched MoE input is not bf16 rows with a bypassed (router-logits) top-k")
    rc, qc = ex.quant_method.runner.config, ex.quant_method.quant_config
    t, h = hs.shape
    if qc.use_per_token_activation:
        from flashinfer import SfLayout, nvfp4_quantize
        from sglang.srt.environ import envs
        e4m3_max = 256.0 if envs.FLASHINFER_NVFP4_4OVER6.get() and envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get() else 448.0
        inv = torch.full((1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32, device=hs.device)
        q4, sf, per_token_scale = nvfp4_quantize(hs, inv, sfLayout=SfLayout.layout_linear, per_token_activation=True,
                                                 backend="cute-dsl")
        hs_fp4, hs_scale = q4.reshape(t, h // 2), sf.view(torch.float8_e4m3fn).reshape(t, h // 16)
    else:
        per_token_scale = None
        hs_fp4, hs_scale = fit.quantize_hidden_states_fp4(hs, ex.w13_input_scale_quant)
    hs_scale = hs_scale.view(torch.float8_e4m3fn).reshape(*hs_scale.shape[:-1], -1)
    tc = topk.topk_config
    g1c, g1a, g1b = getattr(ex, "gemm1_clamp_limit", None), getattr(ex, "gemm1_alpha", None), getattr(ex, "gemm1_beta", None)
    rmt = getattr(ex, "routing_method_type", None)
    if per_token_scale is None and fd_on():
        d = fd_experts(ex, fit, rc, tc, topk.router_logits, hs_fp4, hs_scale, t, h)
        if d is not None:
            return d
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.utils.triton_utils import next_power_of_2
    result = trtllm_fp4_block_scale_moe(
        routing_logits=topk.router_logits, routing_bias=tc.correction_bias, hidden_states=hs_fp4,
        hidden_states_scale=hs_scale, gemm1_weights=ex.w13_weight.data,
        gemm1_weights_scale=ex.w13_weight_scale.data.view(torch.float8_e4m3fn), gemm1_bias=None,
        gemm1_alpha=None if g1a is None else g1a.data, gemm1_beta=None if g1b is None else g1b.data,
        gemm1_clamp_limit=None if g1c is None else g1c.data, gemm2_weights=ex.w2_weight.data,
        gemm2_weights_scale=ex.w2_weight_scale.data.view(torch.float8_e4m3fn), gemm2_bias=None,
        output1_scale_scalar=ex.g1_scale_c.data, output1_scale_gate_scalar=ex.g1_alphas.data,
        output2_scale_scalar=ex.g2_alphas.data, per_token_scale=per_token_scale, num_experts=ex.num_experts,
        top_k=tc.top_k, n_group=tc.num_expert_group, topk_group=tc.topk_group,
        intermediate_size=ex.intermediate_size_per_partition, local_expert_offset=ex.moe_ep_rank * ex.num_local_experts,
        local_num_experts=ex.num_local_experts, routed_scaling_factor=rc.routed_scaling_factor,
        routing_method_type=rmt if rmt is not None else RoutingMethodType.Default, do_finalize=False,
        activation_type=fit.get_activation_type(rc.activation, is_gated=rc.is_gated),
        tune_max_num_tokens=next_power_of_2(t), enable_pdl=fit.trtllm_moe_enable_pdl(t),
        output=fit._get_fp4_moe_output_buffer(num_tokens=t, hidden_size=h, dtype=torch.bfloat16, device=hs.device,
                                              symmetric=True))
    return fit._make_deferred_finalize_output(result, top_k=tc.top_k)


def fp_ours(state, hidden, gza, out, publish):
    px, x = state.proxy, state.x
    t = hidden.shape[0]
    router_logits = px.gate(hidden, gza)
    topk_output = px.topk(hidden, router_logits)
    d = fp_experts(state, hidden, topk_output)
    shared = hz_shared(state, px, hidden, gza)
    if not (type(d) is state.fp[1].FlashInferTrtllmDeferredFinalizeOutput and torch.is_tensor(shared)):
        _need(False, f"finadd-pf: the deferred experts call returned {type(d).__name__} / shared {type(shared).__name__}")
    return fp_finish(x.ar, d, shared, t, out, publish)


def fp_finish(ar, d, shared, t, out, publish):
    k = d.top_k
    g = d.gemm2_out
    g = g.view(-1, g.shape[-1])
    idx, w = d.expanded_idx_to_permuted_idx.view(-1), d.expert_weights.view(-1, k)
    if not fp_valid(g, idx, w, shared, k, out, t):
        fp_refuse(g, idx, w, shared, out, t)
    H = shared.shape[1]
    nb, gs, ss, osd = triton.cdiv(H, FP_BH), g.stride(0), shared.stride(0), out.stride(0)
    if not FP_OVL:
        _fp_kernel[(t, nb)](g, idx, w, shared, out, gs, ss, osd, 0, H=H, K=k, BH=FP_BH, num_warps=FP_WARPS)
        return out
    plan = ar.plan(t, fp_chunks(t), True)
    evs, cur = (fp_events(shared.device), torch.cuda.current_stream()) if publish else (None, None)
    grp = max(1, FP_OVL_GROUP)
    waits = []
    for j in range(0, len(plan), grp):
        a, b = plan[j][0], plan[min(j + grp, len(plan)) - 1][1]
        _fp_kernel[(b - a, nb)](g, idx, w, shared, out, gs, ss, osd, a, H=H, K=k, BH=FP_BH, num_warps=FP_WARPS)
        if evs is not None:
            evs[j // grp].record(cur)
            waits.extend([evs[j // grp]] * (min(j + grp, len(plan)) - j))
    if evs is None:
        return out
    _FP["ready"] = (out.data_ptr(), t, plan, waits)
    _FP["ovl"] += 1
    return out


def fp_placeholder(x, fb, t):
    loc = getattr(fb, "out_cache_loc", None)
    local = not (torch.is_tensor(loc) and loc.dim() == 1 and loc.shape[0] >= t
                 and int(torch.unique(loc[:t]).numel()) == t)
    return any(x.tp.all_gather_object(local))


def fp_same(want, got):
    return bool(torch.is_tensor(want) and torch.is_tensor(got) and want.dtype == got.dtype == torch.bfloat16
                and want.shape == got.shape and torch.equal(want.view(torch.int16), got.view(torch.int16)))


def fp_check(state, hidden, gza, fb):
    x = state.x
    t = hidden.shape[0]
    if fp_placeholder(x, fb, t):
        return moe_x_forward(state, hidden, gza)
    t0 = time.perf_counter()
    fault, got = None, None
    try:
        got = fp_ours(state, hidden, gza, torch.empty((t, x.ar.hidden), dtype=torch.bfloat16, device=hidden.device),
                      False)
    except Exception as exc:
        fault = f"{type(exc).__name__}: {exc}"[:300]
    want = moe_x_forward(state, hidden, gza)
    same = fault is None and fp_same(want, got)
    nz = None if got is None else int((got.view(torch.int16) != want.view(torch.int16)).sum().item())
    del got
    verdicts = x.tp.all_gather_object((same, nz, fault))
    lid = getattr(state.module, "layer_id", None)
    if not all(v[0] for v in verdicts):
        _FP["off"] = f"model.layers.{lid} rows={t}: ranks (bitwise, differing elements, fault) {verdicts}"
        _log.warning("finadd-pf: OFF on every rank (voted live check: deferred finalize + shared add != X's MoE output "
                     "bitwise, or a fault of FP's path: %s); X's route from here", _FP["off"])
        return want
    _FP["checks"] += 1
    _log.warning("finadd-pf: check %d/%d model.layers.%s rows=%d: deferred finalize + shared add (%d row chunks) == "
                 "X's MoE output bitwise (ranks %s); %.1f ms", _FP["checks"], FP_CHECKS, lid, t, fp_chunks(t),
                 [v[0] for v in verdicts], (time.perf_counter() - t0) * 1e3)
    return want


def fp_forward(state, hidden, gza, args, kwargs):
    try:
        why = fp_why(state, hidden)
    except Exception as exc:
        _FP["off"] = f"fp_why raised {type(exc).__name__}: {exc}"[:400]
        _log.warning("finadd-pf: OFF (fp_why fault), X's route from here: %s", _FP["off"])
        why = "off"
    fp_note(state, why, hidden.shape[0])
    if why is not None:
        return moe_x_forward(state, hidden, gza)
    if _FP["checks"] < FP_CHECKS:
        fb = kwargs["forward_batch"] if "forward_batch" in kwargs else (args[1] if len(args) > 1 else None)
        return fp_check(state, hidden, gza, fb)
    _FP["served"] += 1
    return hz_served(state, hidden, gza)


def fp_m1(ar, x, resid, h_res, h_post, mode):
    r, _FP["ready"] = _FP["ready"], None
    t = x.shape[0]
    if (r is None or mode != "zc" or r[0] != x.data_ptr() or r[1] != t or torch.cuda.is_current_stream_capturing()
            or r[2] is not ar.plan(t, fp_chunks(t), True)):
        return ar.m1(x, resid, h_res, h_post, mode)
    h, plan, evs = x.shape[1], r[2], r[3]
    y, cur = ar.buf[:t], torch.cuda.current_stream()
    out = torch.empty((t, 4, h), dtype=torch.bfloat16, device=x.device)
    with torch.cuda.stream(ar.side):
        ar.v2.obj.set_pull_blocks(M1_PULL_BLOCKS)
        for k, (a, b, row) in enumerate(plan):
            ar.side.wait_event(evs[k])
            ar.sg.car(ar.v2.obj, y[a:b], algo=ar.sg.TWO_SHOT_PULL, graph_params=row)
            ar.evs[k].record(ar.side)
        ar.v2.obj.set_pull_blocks(None)
    res3, comb, post = resid.view(t, 4, h), h_res.view(t, 4, 4), h_post.view(t, 4)
    for k, (a, b, _) in enumerate(plan):
        cur.wait_event(ar.evs[k])
        ar.sg.kmhc.mhc_post_tilelang(comb[a:b], res3[a:b], post[a:b], y[a:b], out[a:b], 4, h, num_h_blocks=1)
    return out.view(t, 4 * h)


FD_CHECKS = 2
SH_DIRECT = True
SD_PROBE_ROWS = (16384, 12544, 8192)
IDX_GATE_EAGER = True
GE_CHECKS = 2
NORMQ_PDL = False
_FD = {"op": None, "lay": {}, "tac": {}, "mode": None, "checks": 0, "off": None, "served": 0, "fresh": 0}
_SD = {"lay": {}, "proved": None, "served": 0}
_GE = {"checks": 0, "off": None, "served": 0}


def fd_op():
    op = _FD["op"]
    if op is None:
        from flashinfer.fused_moe import core as fic
        from sglang.srt.utils.triton_utils import next_power_of_2
        ns = fic.get_trtllm_moe_sm100_module()
        op = _FD["op"] = types.SimpleNamespace(run=ns.moe_op.trtllm_fp4_block_scale_moe, moe_op=ns.moe_op, fic=fic,
                                               npow2=next_power_of_2, rim=fic.RoutingInputMode.FromLogits)
    return op


def fd_layer(ex, fit, rc, tc, hs4):
    f = _FD["lay"].get(id(ex))
    if f is None:
        from sglang.srt.layers.moe.utils import RoutingMethodType
        op = fd_op()
        g1c, g1a, g1b = getattr(ex, "gemm1_clamp_limit", None), getattr(ex, "gemm1_alpha", None), getattr(ex, "gemm1_beta", None)
        rmt = getattr(ex, "routing_method_type", None)
        f = types.SimpleNamespace(
            ex=ex, bias=tc.correction_bias, w1=ex.w13_weight.data, s1=ex.w13_weight_scale.data.view(torch.float8_e4m3fn),
            a=None if g1a is None else g1a.data, b=None if g1b is None else g1b.data,
            c=None if g1c is None else g1c.data, w2=ex.w2_weight.data,
            s2=ex.w2_weight_scale.data.view(torch.float8_e4m3fn), o1=ex.g1_scale_c.data, o1g=ex.g1_alphas.data,
            o2=ex.g2_alphas.data, E=ex.num_experts, K=tc.top_k, ng=tc.num_expert_group, tg=tc.topk_group,
            I=ex.intermediate_size_per_partition, off=ex.moe_ep_rank * ex.num_local_experts, nl=ex.num_local_experts,
            rsf=rc.routed_scaling_factor, rmt=rmt if rmt is not None else RoutingMethodType.Default,
            act=fit.get_activation_type(rc.activation, is_gated=rc.is_gated))
        f.nfs = op.fic._validate_fused_shared_experts(None, f.off, f.nl, f.E, f.rmt)
        f.v46 = op.fic._trtllm_fp4_nvfp4_4over6_code(op.fic._UNSET, None, hs4)
        f.sig = (f.E, f.K, f.ng, f.tg, f.I, f.off, f.nl, int(f.rmt), int(f.act), f.nfs, f.v46, tuple(f.w1.shape),
                 tuple(f.w2.shape), str(f.w1.dtype), f.a is None, f.b is None, f.c is None, f.bias is None)
        _FD["lay"][id(ex)] = f
    return f


def fd_gen(tuner):
    try:
        if tuner.is_tuning_mode:
            return None
        w = tuner._winner_cache()
        return (id(w), len(w), len(tuner.profiling_cache), len(getattr(tuner, "_file_configs", None) or ()))
    except Exception:
        return None


def fd_choose(f, op, logits, hs4, hsf, ids, wts, out, t, pdl):
    fic = op.fic
    hidden = hs4.shape[-1] * 2 if hs4.dtype == torch.uint8 else hs4.shape[-1]
    moe_runner = fic.TrtllmMoERunner(
        op.moe_op, top_k=f.K, num_local_experts=f.nl, dtype_act=fic.deduce_trtllm_gen_tensor_dtype(hs4, hsf),
        dtype_weights=fic.deduce_trtllm_gen_tensor_dtype(f.w1, f.s1),
        fp8_quantization_type=fic.Fp8QuantizationType.NoneFp8, hidden_size=hidden, hidden_size_output=hidden,
        intermediate_size=f.I, activation_type=f.act, weight_layout=fic.WeightLayout.MajorK, use_shuffled_weight=True,
        use_per_token_scaling=False, num_experts=f.E, num_fused_shared_experts=f.nfs, nvfp4_4over6_code=f.v46)
    moe_inputs = fic.MoeRunnerInputs(output=out, routing_logits=logits, topk_ids=ids, expert_weights=wts,
                                     hidden_states=hs4, hidden_states_scale=hsf, gemm1_lora_delta=None,
                                     per_token_scale=None)
    tuning_config = moe_runner._make_tuning_config(moe_inputs, tune_max_num_tokens=op.npow2(t),
                                                   routing_input_mode=fic.RoutingInputMode(op.rim),
                                                   hidden_states_scale_layout=None, use_cold_l2_cache=True,
                                                   use_cuda_graph=True)
    runner_kwargs = {
        "workspace_buffer": None, "routing_input_mode": op.rim, "num_experts": f.E, "routing_bias": f.bias,
        "gemm1_weights": f.w1, "gemm1_weights_scale": f.s1, "gemm1_bias": None, "gemm1_alpha": f.a, "gemm1_beta": f.b,
        "gemm1_clamp_limit": f.c, "gemm2_weights": f.w2, "gemm2_weights_scale": f.s2, "gemm2_bias": None,
        "output1_scale_scalar": f.o1, "output1_scale_gate_scalar": f.o1g, "output2_scale_scalar": f.o2,
        "n_group": f.ng, "topk_group": f.tg, "local_expert_offset": f.off, "routed_scaling_factor": f.rsf,
        "routing_method_type": f.rmt, "enable_pdl": pdl, "do_finalize": False, "activation_type": f.act,
        "num_fused_shared_experts": f.nfs, "norm_topk_prob": True, "routing_replay_out": None,
        "valid_hidden_size": None, "valid_intermediate_size": None,
    }
    _, tactic = fic.AutoTuner.get().choose_one("flashinfer::trtllm_fp4_block_scale_moe", [moe_runner], tuning_config,
                                               moe_inputs.to_list(), **runner_kwargs)
    return tactic


def fd_on():
    return FP_CALL == "direct" and _FD["off"] is None and _FD["mode"] in (True, "check")


def fd_experts(ex, fit, rc, tc, logits, hs4, hsf, t, h):
    op = fd_op()
    f = fd_layer(ex, fit, rc, tc, hs4)
    tuner = op.fic.AutoTuner.get()
    gen = fd_gen(tuner)
    if gen is None and tuner.is_tuning_mode:
        return None
    dev = hs4.device
    ids = torch.empty(t, f.K + f.nfs, dtype=torch.int32, device=dev)
    wts = torch.empty(t, f.K + f.nfs, dtype=torch.bfloat16, device=dev)
    out = fit._get_fp4_moe_output_buffer(num_tokens=t, hidden_size=h, dtype=torch.bfloat16, device=dev, symmetric=True)
    pdl = fit.trtllm_moe_enable_pdl(t)
    key = (t, f.sig, pdl)
    hit = _FD["tac"].get(key)
    if hit is None or gen is None or hit[1] != gen or _FD["mode"] == "check":
        tac = fd_choose(f, op, logits, hs4, hsf, ids, wts, out, t, pdl)
        _FD["fresh"] += 1
        if gen is not None:
            _FD["tac"][key] = (tac, gen)
    else:
        tac = hit[0]
    io = op.run(op.rim, logits, ids, wts, f.bias, hs4, hsf,
                f.w1, f.s1, None, None, f.a, f.b, f.c, f.w2, f.s2, None,
                f.o1, f.o1g, f.o2, None, f.E, f.K, f.nfs, f.ng, f.tg, f.I,
                f.off, f.nl, f.rsf, f.rmt, False, pdl, f.act, out,
                [-1, -1] if tac == -1 else tac, True, None, [], [], False,
                None, None, None, False, f.v46)
    g, idx = io[0], io[2]
    if type(g) is not torch.Tensor:
        g = torch.from_dlpack(g)
    if type(idx) is not torch.Tensor:
        idx = torch.from_dlpack(idx)
    _FD["served"] += 1
    return fit._make_deferred_finalize_output([g, wts, idx], top_k=tc.top_k)


def sd_layer(state):
    hit = _SD["lay"].get(id(state))
    if hit is not None:
        return hit[0]
    f, why = None, None
    try:
        ours, sg, ar = state.proxy.shared_experts, state.sg, state.x.ar
        fu = ar.sg.fp8_utils
        gu, dn = ours.gate_up_proj, ours.down_proj
        lim = ours.swiglu_limit
        for name, lin in (("gate_up_proj", gu), ("down_proj", dn)):
            qm = lin.quant_method
            fn = getattr(qm.w8a8_block_fp8_linear, "func", qm.w8a8_block_fp8_linear)
            if not (fn is fu.deepgemm_w8a8_block_fp8_linear_with_fallback and lin.bias is None
                    and lin.weight.dtype == torch.float8_e4m3fn and lin.weight_scale_inv.dtype == torch.int32
                    and lin.input_scale is None and not qm.use_marlin and not qm.use_mxfp8
                    and not qm.block_fp8_as_mxfp8 and list(qm.quant_config.weight_block_size) == [BLOCK, BLOCK]
                    and fu._deepgemm_block_fp8_supported(tuple(lin.weight.shape), [BLOCK, BLOCK], torch.bfloat16)):
                why = f"{name}: not the DeepGEMM block-FP8 linear with packed UE8M0 scales"
        if why is None and (gu.gather_output or not dn.input_is_parallel or dn.reduce_results
                            or getattr(dn, "use_decode_attn_tp", False) or dn.use_dp_attention_reduce
                            or ours.is_replicated or getattr(ours, "_enable_nvfp4_gemm_swiglu_fusion", False)
                            or ours.use_fused_clamp_act_mul or fu.get_platform().is_sm90 or fu._is_musa
                            or not ar.sg.dgw.ENABLE_JIT_DEEPGEMM):
            why = "module facts (gather / reduce / replicated / fused routes / platform)"
        if why is None and lim is not None and ours.allow_fused_down and dn.weight.dtype == torch.uint8:
            why = "the MLP's fused down route"
        if why is None:
            f = types.SimpleNamespace(
                q=fu.sglang_per_token_group_quant_fp8, u=ar.sg.dgw.DEEPGEMM_SCALE_UE8M0, g=ar.sg.dg_nt,
                w1=gu.weight, s1=gu.weight_scale_inv, n1=gu.weight.shape[0], w2=dn.weight, s2=dn.weight_scale_inv,
                n2=dn.weight.shape[0], lim=None if lim is None else float(lim), clamp=sg.dsv2.silu_and_mul_clamp,
                act=ours.act_fn, mod=ours)
    except Exception as exc:
        why = f"facts raised {type(exc).__name__}: {exc}"[:300]
    _SD["lay"][id(state)] = (f, why)
    if why is not None:
        _log.warning("sh-direct: model.layers.%s stock's shared-expert call: %s", getattr(state.module, "layer_id", None),
                     why)
    return f


def sd_forward(f, x):
    t = x.shape[0]
    if not x.is_contiguous():
        x = x.contiguous()
    q, s = f.q(x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=f.u)
    y = torch.empty((t, f.n1), dtype=torch.bfloat16, device=x.device)
    f.g(q, s, f.w1, f.s1, y, BLOCK, BLOCK)
    if f.lim is None:
        a = f.act(y)
    else:
        a = y.new_empty((t, f.n1 // 2))
        f.clamp(y, a, f.lim)
    q, s = f.q(a, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=f.u)
    z = torch.empty((t, f.n2), dtype=torch.bfloat16, device=x.device)
    f.g(q, s, f.w2, f.s2, z, BLOCK, BLOCK)
    return z


def sd_probe(state, f, hidden):
    t0 = time.perf_counter()
    res = []
    try:
        if f is None:
            raise RuntimeError(f"facts: {_SD['lay'].get(id(state), (None, None))[1]}")
        gen = torch.Generator(device=hidden.device).manual_seed(SELF_CHECK_SEED + 7)
        with torch.no_grad():
            for m in SD_PROBE_ROWS:
                x = torch.randn(m, hidden.shape[1], generator=gen, device=hidden.device).to(hidden.dtype)
                want, got = f.mod(x), sd_forward(f, x)
                res.append(int((want.view(torch.int16) != got.view(torch.int16)).sum().item())
                           if want.shape == got.shape and want.dtype == got.dtype else -1)
                del x, want, got
        mine = (all(v == 0 for v in res), res, None)
    except Exception as exc:
        mine = (False, res, f"{type(exc).__name__}: {exc}"[:300])
    verdicts = state.x.tp.all_gather_object(mine)
    ok = all(v[0] for v in verdicts)
    _SD["proved"] = ok
    _log.warning("sh-direct: probe rows %s: direct FP8 shared expert %s the module (differing elements per rank %s); "
                 "%.1f ms", list(SD_PROBE_ROWS), "==" if ok else "!= (SD OFF on every rank)",
                 [(v[1], v[2]) for v in verdicts], (time.perf_counter() - t0) * 1e3)
    return ok


def hz_shared(state, px, hidden, gza):
    if SH_DIRECT and fd_on() and hidden.shape[0] > 256:
        if _SD["proved"] is None:
            sd_probe(state, sd_layer(state), hidden)
        f = sd_layer(state) if _SD["proved"] else None
        if f is not None:
            _SD["served"] += 1
            return sd_forward(f, hidden)
    return px._forward_shared_experts(hidden, gza)


def hz_check(state, hidden, gza):
    x = state.x
    t = hidden.shape[0]
    t0 = time.perf_counter()
    fault, got = None, None
    _FD["mode"] = "check"
    try:
        got = fp_ours(state, hidden, gza, torch.empty((t, x.ar.hidden), dtype=torch.bfloat16, device=hidden.device),
                      False)
    except Exception as exc:
        fault = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        _FD["mode"] = None
    want = fp_ours(state, hidden, gza, x.ar.buf[:t], True)
    if got is not None and (got.shape != want.shape or got.dtype != want.dtype):
        fault, got = f"shape / dtype {tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}", None
    nz = None if got is None else int((got.view(torch.int16) != want.view(torch.int16)).sum().item())
    same = fault is None and nz == 0
    del got
    verdicts = x.tp.all_gather_object((same, nz, fault, _SD["served"] > 0))
    lid = getattr(state.module, "layer_id", None)
    if not all(v[0] for v in verdicts):
        _FD["off"] = f"model.layers.{lid} rows={t}: ranks (bitwise, differing elements, fault, sd) {verdicts}"
        _FD["mode"] = False
        _log.warning("hz-moe: OFF on every rank (voted live check: FD + SD path != FP's own path, or a fault): %s; FP's "
                     "own path from here", _FD["off"])
        return want
    _FD["checks"] += 1
    if _FD["checks"] >= FD_CHECKS:
        _FD["mode"] = True
    _log.warning("hz-moe: check %d/%d model.layers.%s rows=%d: FD (raw FlashInfer op, cached tactic) + SD (%s) == FP's own "
                 "path bitwise (ranks %s, differing elements %s); %.1f ms%s", _FD["checks"], FD_CHECKS, lid, t,
                 "direct FP8 shared expert" if verdicts[0][3] else "stock shared expert", [v[0] for v in verdicts],
                 [v[1] for v in verdicts], (time.perf_counter() - t0) * 1e3,
                 "; serving from the next call" if _FD["mode"] is True else "")
    return want


def hz_served(state, hidden, gza):
    if FP_CALL == "direct" and _FD["off"] is None and _FD["mode"] is None:
        return hz_check(state, hidden, gza)
    return fp_ours(state, hidden, gza, state.x.ar.buf[:hidden.shape[0]], True)


def ge_scale(st, x, q_scale, hw):
    if (not IDX_GATE_EAGER or hw is None or _GE["off"] is not None or hw.dtype != torch.float32
            or q_scale.dtype != torch.float32 or type(getattr(st, "softmax_scale", None)) is not float
            or torch.cuda.is_current_stream_capturing()):
        return st._resolve_head_gate_weights(x, q_scale, hw)
    if _GE["checks"] < GE_CHECKS:
        return ge_check(st, x, q_scale, hw)
    _GE["served"] += 1
    return hw.unsqueeze(-1).mul(q_scale).mul_(st.softmax_scale)


def ge_check(st, x, q_scale, hw):
    want = st._resolve_head_gate_weights(x, q_scale, hw)
    try:
        got = hw.unsqueeze(-1).mul(q_scale).mul_(st.softmax_scale)
        nz = (int((got.view(torch.int32) != want.view(torch.int32)).sum().item())
              if got.shape == want.shape and got.dtype == want.dtype == torch.float32 else -1)
    except Exception as exc:
        nz, got = f"{type(exc).__name__}: {exc}"[:200], None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if nz != 0:
        _GE["off"] = f"rows={q_scale.shape[0]}: {nz} differing words"
        _log.warning("idx-gate-eager: OFF for this process (rank %d): eager fp32 multiply != the compiled one: %s; stock's "
                     "compiled call from here", rank, _GE["off"])
        return want
    _GE["checks"] += 1
    _log.warning("idx-gate-eager: check %d/%d rank %d rows=%d heads=%d: eager fp32 multiply == stock's compiled "
                 "_apply_q_scale_and_softmax_scale, 0 differing words", _GE["checks"], GE_CHECKS, rank,
                 q_scale.shape[0], hw.shape[-1] if hw.dim() > 1 else 1)
    return want

AR_OPROJ_OVL = True
AO_SMS = 0
_AO = {"ready": None, "evs": {}, "n": 0}


def ao_events(dev):
    evs = _AO["evs"].get(dev)
    if evs is None:
        evs = _AO["evs"][dev] = [torch.cuda.Event() for _ in range(2)]
    return evs


def ao_gemm(zc, q, s, out):
    ar, t = zc.ar, q.shape[0]
    if (not AR_OPROJ_OVL or not MEGA_PF or not (PF_MIN_ROWS <= t <= ar.rows) or torch.cuda.is_current_stream_capturing()
            or ar.sg.in_graph()):
        zc.gemm(q, s, zc.inner.weight, zc.inner.weight_scale_inv, out, BLOCK, BLOCK)
        return out
    plan = ar.plan(t, 2, True)
    evs, cur = ao_events(out.device), torch.cuda.current_stream()
    for k, (a, b, _) in enumerate(plan):
        zc.gemm(q[a:b], s[a:b], zc.inner.weight, zc.inner.weight_scale_inv, out[a:b], BLOCK, BLOCK)
        evs[k].record(cur)
    _AO["ready"] = (out.data_ptr(), t, plan, evs)
    _AO["n"] += 1
    return out


def ao_pull(ar, x, plan, cur):
    r, _AO["ready"] = _AO["ready"], None
    hit = (r is not None and r[0] == x.data_ptr() and r[1] == x.shape[0] and r[2] is plan
           and not torch.cuda.is_current_stream_capturing())
    if not hit:
        ar.side.wait_stream(cur)
    with torch.cuda.stream(ar.side):
        ar.v2.obj.set_pull_blocks(M1_PULL_BLOCKS)
        for k, (a, b, row) in enumerate(plan):
            if hit:
                ar.side.wait_event(r[3][k])
            ar.sg.car(ar.v2.obj, x[a:b], algo=ar.sg.TWO_SHOT_PULL, graph_params=row)
            ar.evs[k].record(ar.side)
        ar.v2.obj.set_pull_blocks(None)


KDA_DYN = True
try:
    from .kdadyn import _KW, KwFresh, kw_run_dspark
except ImportError:
    from layerar.kdadyn import _KW, KwFresh, kw_run_dspark

class ZcProj(torch.nn.Module):
    def __init__(self, inner, ar):
        super().__init__()
        self.inner = inner
        self.ar, self.quant, self.gemm = ar, ar.sg.fp8_utils.sglang_per_token_group_quant_fp8, ar.sg.dg_nt
        self.ue8m0 = ar.sg.dgw.DEEPGEMM_SCALE_UE8M0
        self.slot = None
    def forward(self, x, *rest, **extra):
        _need(not rest and not extra and x.dim() == 2 and x.shape[0] <= self.ar.rows, f"o_proj call {tuple(x.shape)}")
        if self.slot is None:
            if not x.is_contiguous():
                x = x.contiguous()
            q, s = self.quant(x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=self.ue8m0)
        else:
            q, s = self.slot.q, self.slot.s
            self.slot.q = self.slot.s = None
            _need(q is not None and x.dtype == _FP8 and x.data_ptr() == q.data_ptr() and x.numel() == q.numel()
                  and x.shape[0] == s.shape[0] and x.is_contiguous(), "K2 o_proj: input is not this forward's NormQ output")
            q = x
        out = self.ar.buf[:x.shape[0]]
        ao_gemm(self, q, s, out)
        return out, None

def _attach_zc(state):
    ar, attn = state.ffn.ar, state.attn
    if attn.proxy is None:
        pj_build(attn)
    kdae = KdaeAttn(attn, ar.tp) if (KDA_EAGER or K2_CONV) and attn.kind == "kda" else None
    old = attn.proxy._modules["o_proj"]
    lin = getattr(old, "inner", None)
    fu, qm = ar.sg.fp8_utils, getattr(lin, "quant_method", None)
    fn = getattr(getattr(qm, "w8a8_block_fp8_linear", None), "func", getattr(qm, "w8a8_block_fp8_linear", None))
    _need(type(old) is Fp8Proj and fn is fu.deepgemm_w8a8_block_fp8_linear_with_fallback and lin.bias is None
          and not fu.get_platform().is_sm90 and not fu._is_musa and ar.sg.dgw.ENABLE_JIT_DEEPGEMM
          and lin.weight.shape[0] == ar.hidden
          and fu._deepgemm_block_fp8_supported(tuple(lin.weight.shape), [BLOCK, BLOCK], torch.bfloat16),
          "the attention proxy's o_proj is not b13's DeepGEMM block-FP8 linear on this platform")
    zc = ZcProj(lin, ar)
    gen = torch.Generator(device=lin.weight.device).manual_seed(PROBE_SEED)
    x = torch.randn(PF_MIN_ROWS, lin.weight.shape[1], generator=gen, device=lin.weight.device).to(torch.bfloat16)
    with torch.no_grad():
        want, got = lin(x)[0], zc(x)[0]
    _need(got.data_ptr() == ar.buf.data_ptr() and torch.equal(want.view(torch.int16), got.view(torch.int16)),
          f"{getattr(attn.proxy, 'prefix', 'self_attn')}.o_proj: ZcProj's probe output differs from the FP8 linear's")
    swaps = _attach_k2(attn, zc, kdae) if attn.kind == "kda" and (K2_CONV or K2_NORMQ) else {}
    attn.proxy._modules["o_proj"] = zc
    for name, mod in swaps.items():
        attn.proxy._modules[name] = mod
    if kdae is not None:
        attn.proxy._modules["attn"] = kdae
    ko_attach(state, zc)
    state.zc = True
    sb_attach(state)
    sf_attach(state)

def _layer_proxy(state):
    layer = state.layer
    swaps = {"self_attn": (pj_forward, state.attn)}
    if state.mlp.sg is not None:
        swaps["mlp"] = (sh_forward, state.mlp)
    for name, (entry, node_state) in swaps.items():
        swaps[name] = pj_proxy(layer._modules[name], {})
        object.__setattr__(swaps[name], "forward", functools.partial(entry, node_state))
    proxy = pj_proxy(layer, swaps)
    if state.ffn is not None:
        object.__setattr__(proxy, "ffn_boundary", state.ffn)
    _need(layer.ffn_boundary is not state.ffn and proxy.attn_boundary is layer.attn_boundary, "layer proxy boundaries")
    return proxy

def prepare(layer):
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.models import glm5_next
    _need(type(layer) is glm5_next.Glm5NextDecoderLayer, f"node is {type(layer).__name__}")
    _need(bool(layer.config.mhc) and layer.config.hc_mult == 4, "the layer is not mHC with 4 streams")
    stock_forward = type(layer).forward
    wrapped = getattr(layer.forward, "__wrapped__", None)
    if wrapped is not None:
        _need(getattr(wrapped, "__func__", None) is stock_forward and getattr(wrapped, "__self__", None) is layer,
              "the forward the dispatcher wraps is not the class forward bound to this layer")
    attn, mlp = pj_prepare(layer.self_attn), sh_prepare(layer.mlp)
    tp = ps.get_tp_group()
    ffn = _prepare_ar(layer, ps, tp) if tp.world_size == 2 else None
    if ffn is not None and mlp.sg is not None and (MOE_SIDE or ZC_AR2):
        _moe_facts(layer.mlp, mlp, ffn)
    if ffn is None and "off" not in _PROCESS:
        _PROCESS["off"] = True
        _log.warning("layerar: exact-ar2 off on this engine (tp=%d; the proof covers 2 ranks)", tp.world_size)
    dec4 = ffn is None and tp.world_size == 4
    _need(not (dec4 and ROWSHARD) or ps.get_attn_tp_group().world_size == tp.world_size,
          "idx-rowshard: attention TP is not the TP group (attention DP would shard other requests)")
    idx = _idx_prepare(layer, tp if dec4 and ROWSHARD else None) if ffn is None else None
    if idx is None and dec4:
        idx = kd_prepare(layer, tp)
    small = _idx_prepare(layer, role="small") if ffn is not None and (ROTQ or FP8PACK) else None
    if small is not None:
        attn.rq, attn.small = small.nn, small.proxy._modules["self_attn"]
    spv = tp2_verify_prepare(layer) if ffn is not None and (IDX_WPROJ_TC or SPLIT_DEC) else None
    return types.SimpleNamespace(layer=layer, stock_forward=stock_forward, attn=attn, mlp=mlp, ffn=ffn, proxy=None,
                                 spv=spv, zc=False, min_rows=PF_MIN_ROWS if ffn is None else ffn.ar.low, idx=idx,
                                 dec=_dec_prepare(layer, idx) if dec4 and (FASTPOST or MEGA_DEC) else None)

def forward(state, *args, **kwargs):
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else (args[1] if len(args) > 1 else None)
    rows = hs.shape[0] if torch.is_tensor(hs) and hs.dim() >= 1 else 0
    if rows < state.min_rows:
        if state.idx is not None or state.dec is not None:
            fb = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
            if fb.forward_mode.is_target_verify():
                if state.dec is not None and DEC_MIN_ROWS < rows <= DEC_MAX_ROWS:
                    return state.stock_forward(state.dec.proxy, *args, **kwargs)
                if state.idx is not None:
                    return state.stock_forward(state.idx.proxy, *args, **kwargs)
        if getattr(state, "spv", None) is not None:
            fb = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
            if fb.forward_mode.is_target_verify():
                return state.stock_forward(state.spv, *args, **kwargs)
        return state.stock_forward(state.layer, *args, **kwargs)
    if state.proxy is None:
        state.proxy = _layer_proxy(state)
    if state.ffn is not None:
        if state.ffn.ar.n64 is None:
            _build_comms(state.ffn.ar)
        if state.ffn.mega is not None and "mega" not in _PROCESS:
            _PROCESS["mega"] = _megapf_check(state.ffn.ar, state.ffn)
        if rows >= PF_MIN_ROWS and not state.zc:
            _attach_zc(state)
    return state.stock_forward(state.proxy, *args, **kwargs)
