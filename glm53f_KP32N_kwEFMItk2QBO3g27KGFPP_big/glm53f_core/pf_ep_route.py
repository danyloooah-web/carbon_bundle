"""r61 PF-EP2 (ours; derived by r61/pfep/derive_pfep.py from OUR r59 moe_ep_route.py, literals only): the TP2 PREFILL and AUDIT
engines' routed NVFP4 experts resharded once, in place, from tensor-parallel (every rank: all 288 experts,
intermediate 1024) to expert-parallel INSIDE the TP2 group (rank r: experts [144 r, 144 r + 144) whole, intermediate
2,048), at a target MoE layer's prepare, BEFORE the king's moe_dsl_route builds its CuTe-DSL layout from them (our
moe_dsl_route edit passes num_experts 288 / local_expert_offset 144 r / num_local_experts 144 to FlashInfer's
moe_sort, which masks non-local picks). Per rank: the same weight bytes and FLOPs, half the FC1 gather rows and half
the FC2 finalize scatter-add rows; FC2 K 1024 -> 2048. Reorder class (one FP32 K = 2048 accumulation instead of two
halves summed by the all-reduce). Design: r61/pfep/DESIGN.md. Census lines "pfep: ...". The decode route's design text
follows (same mechanism; read "TP 4" as TP 2, 72 as 144, 512 as 1024, "the PD decode engine" as the prefill / audit engines).

r59 EP (ours; reorder class: FC2's sum order only): the TP4 decode engine's routed NVFP4 experts resharded once, in
place, from tensor-parallel (every rank: all 288 experts, intermediate 512) to expert-parallel INSIDE the TP group (rank
r: experts [72 r, 72 r + 72) whole, intermediate 2,048), at a target MoE layer's prepare. Design and evidence:
r59/ep/README.md (hypotheses H1-H9 and what falsifies each) and r59/ep/BRIEF.md.

Stock (SGLang 314bce7c / image 93a5b07 and FlashInfer 0.7.3, read as text). After loading, a decode rank's FusedMoE holds
(flashinfer_trtllm.align_fp4_moe_weights_for_flashinfer_trtllm -> quantization/utils.py
prepare_static_weights_for_trtllm_fp4_moe; no padding at intermediate 512 or 2,048 (is_gated: alignment 16) or hidden
4,096 (alignment 1)): w13_weight uint8 [288, 1024, 2048] (per expert: the TP slice's two halves [512 + 512 rows] as the
loader placed them, rows permuted by FlashInfer's _maybe_get_cached_w3_w1_permute_indices = gated reorder then
shuffle_matrix_a at epilogue_tile_m 128), w13_weight_scale fp8 [288, 1024, 256] (rows permuted with num_elts_per_sf 16,
then nvfp4_block_scale_interleave), w2_weight uint8 [288, 4096, 256] (get_w2_permute_indices_with_cache), w2_weight_scale
fp8 [288, 4096, 32] (likewise + interleave); per expert g1_alphas / g1_alphas_up / g2_alphas / g1_scale_c /
w13_weight_scale_2 / w2_weight_scale_2; GLOBAL-expert input scales w13_input_scale / w2_input_scale (never sliced); the
scalars w13_input_scale_quant / w2_input_scale_quant. ModelOptNvFp4FusedMoEMethod.apply builds the FlashInfer call from
the LAYER at every call (modelopt_quant.py :3267-3285: local_expert_offset = moe_ep_rank * num_local_experts,
local_num_experts, intermediate_size_per_partition, the weights' .data); the standard dispatcher keeps global expert ids
for flashinfer_trtllm (standard.py _keeps_global_expert_ids), so its EP fields are never read on this path.

What this component does (NODE "layer"; listed in ACTIVE right BEFORE moe_prepared_route, whose prepare reads the local
expert range once: nodes.prepare_layer prepares the layer's components in ACTIVE order at the layer's first dispatched
call, which cacheon never makes inside FlashInfer's tuning pass), once per target MoE layer (3..44) of the PD DECODE
engine (TP 4), eagerly, never inside a stream capture:
1. local checks of the layer's form (shapes, dtypes, contiguity, the TP layout, every per-expert parameter known), the
   inverse maps (built once from FlashInfer's own index helpers on dummies of the TP shapes, inverted by argsort; the
   scale interleave inverted by applying the real nvfp4_block_scale_interleave to the base-256 digit planes of an iota),
   and every buffer allocated up front (the EP tensors, the round buffers, the probe's) -> vote V0 on the TP group's CPU
   group (all_gather_object): any miss -> no other collective for this layer;
2. ceil(72 / ROUND) rounds x 4 tensors of torch.distributed.all_to_all_single over the TP device group (expert e's TP
   slice to rank e // 72), each received slice canonicalized with the inverse maps and placed in the EP canonical
   layout (gate / up halves kept in their stock order, the four intermediate slices in rank order; w2 slices as column
   blocks), then the forward permutes for the EP shape with FlashInfer's helpers (SGLang's per-expert loop, into the EP
   buffers); round 0 also proves, byte for byte, forward_TP(canonical) == the received TP bytes for every received
   expert (the inverse maps against the real forward) and our EP bytes == SGLang's own
   prepare_static_weights_for_trtllm_fp4_moe on one expert;
3. the TILE CHECK (the first target layer only; local work, no collective of its own): the king's prepared decode path
   (moe_prepared_route.py) builds FlashInfer's canonical routing record for the tile default_tile(rows, 8,
   local_num_experts) (:250-256, read as a module global ONCE, in _prepare :262-266, a shape's first call, eager) and
   runs FlashInfer's FP4 body from it with tactic -1 (:400-401), whose tile the C++ launcher resolves on its own (the
   metadata slot's tensors carry none: fused_moe/core.py :134-179). Under TP local == global == 288; under EP (72 /
   288) this image's FlashInfer fork may resolve from the GLOBAL count (backends/trtllm/sm100_runner.py :229-232 /
   :639-646 pass it "to offer the tiles that fit expert-parallel inputs"; prims_ts/moe/config_mapper.py :198-201
   selects from it) -> a record built for another tile than the body's makes every captured decode graph read wrong
   expert rows per CTA, which the probe (FlashInfer's full entry: routing and body consistent by construction) cannot
   see. So for EVERY row count the king's path serves (MIN_ROWS .. MAX_ROWS, its literals; apply_ep.py pins them and
   default_tile below to the king's text, AST for AST) ONE eager launch of the real launcher: FlashInfer's MoERunner
   built as the king builds it (:276-283; local 72, global 288, intermediate 2,048, the EP buffers about to be
   committed) and run with tactic -1 (no AutoTuner), from logits, do_finalize False, local_expert_offset 72 r, the
   layer's own routing method / n_group / topk_group / routed scale as stock passes them (flashinfer_trtllm.py
   :2127-2167) and crafted routing (_tile_inputs: logits +4 on 8 picks a row, -4 elsewhere, zero bias, verified on the
   CPU under DeepSeekV3's sigmoid + bias top-8): on THIS rank exactly two picks are local, row 0's = local expert 0
   and row 1's = local expert 1. Its expanded -> permuted map then holds 0 for row 0's pick and, since every CTA
   tile holds one expert's rows and the experts' tile-padded segments follow in local order (trtllm_gen_routing.py
   :276-282, :340-350; the king's routingPermutation transcriptions moe_entry_native.cu :365-418, moe_wide_native.cu
   :433-477), exactly the launcher's resolved tile for row 1's: recorded into this rank's TABLE T (rows -> tile). A
   layout that does not read that way, a tile outside the king's FP4_TILES (its kernels take any tile >= 1 as a
   launch parameter, its record is allocated for that tile: FP4_TILES is the set it is written for) or an exception
   -> this rank's miss. T's runs ("8 at 17-576, ...": lossless) travel in V1; they must be the same on every rank,
   else OFF everywhere (nothing committed yet). T == default_tile(rows, 8, 72) at every row count: nothing more (the
   king's record agrees). Else TILE FOLLOW (r59d): the king's record follows the launcher. Before V1 this rank checks
   that the king's module is loaded as <our package>.moe_prepared_route (the validator's materialized name: looked up
   in sys.modules, as dsa_rtk_route / dsa_kattn_route find their siblings) with the default_tile apply_ep pinned (a
   miss otherwise); after the commit routed() swaps that module's global default_tile, for the duration of each EP
   layer's call (ours is entered BEFORE moe_prepared_route's: ACTIVE order, outer first, nodes.forward_layer), for
   one that answers T[rows] at (8, 72) and the king's own otherwise; restored in a finally (the framework's swap
   pattern), so the record the king's _prepare allocates, the tile its kernels write (shape["tile"]) and the body's
   tactic -1 tile are the launcher's;
4. the probe: the same crafted logits (every expert picked PROBE_ROWS * 8 / 288 times) and inputs through FlashInfer's
   trtllm_fp4_block_scale_moe in TP form (the old buffers) and EP form (the new ones), outside the AutoTuner's tuning
   (lookups only: no cache entry is made; EP keys differ from TP keys in shapes and runner extras), both summed over
   the TP group by one torch.distributed.all_reduce (FP32; not SGLang's all_reduce: its custom all-reduce returns
   zeros in a capture warm-up, custom_all_reduce.py :347-360), compared: max|d| / max|tp| <= BAR_MAX and
   mean|d| / mean|tp| <= BAR_MEAN (CPU model r59/ep/README.md: honest 4.5e-3 / 3.3e-3; one wrong expert 0.25-0.4 max);
5. vote V1: every rank passes -> commit: the EP bytes copied into the OLD storages (same byte counts; the parameters
   keep their identity and pointers), .data re-viewed as [72, 4096, ...], the per-expert vectors sliced in place
   (each Parameter object once, however many names it has; V0 refuses vectors sharing a storage with a weight, a
   global input scale or another vector at a different view), moe_ep_rank / moe_ep_size / moe_tp_rank / moe_tp_size /
   num_local_experts / _num_local_routed / _expert_storage_rank / intermediate_size_per_partition and
   moe_runner_config's and the dispatcher's copies set. Any miss -> the EP buffers dropped, the layer stays TP.
6. once nothing more will be resharded (after the last target layer's commit, or after a first-layer voted miss), every
   reshard buffer and view dropped: torch.cuda.empty_cache() outside any capture (eager, as SGLang's own graph-pool
   measure() calls it right before each warm-up: model_executor/runner_utils/pool.py :145,
   runner_backend/full_cuda_graph_backend.py :164-167), returning the ~1.3 GB a rank of transient memory.
Every rank takes the same collectives in the same order: every refusal before V0 is rank-uniform (literals, the engine
role, the group size, the layer id, the capture / tuning state); local work between collectives (an exception or an
out-of-memory error included) is recorded as this rank's miss and never skips a collective; a collective that raises is
a communicator fault (not recovered). ALL OR NOTHING across layers: the king's moe_prepared_route keeps ONE shape key
per row count for every layer (_SHAPES, moe_prepared_route.py :328-404), so a mixed EP / TP engine raises in its first
captured call. The first target layer decides: a refusal or a voted miss there turns EP OFF on every rank for every
layer (the king's bundle exactly); after a commit, a later layer that cannot follow raises here, on every rank after the
same vote, with the reason (a loud boot failure instead of moe_prepared_route's).
PINS FOLLOW (r59i, the v32 king dbe67160): that king pins decode MoE tactics at five row counts (moe_prepared_route.py's
module global PINS = {576: (64, 18), ..., 768: (64, 19)}, read once in _replaced_with_tactic at a shape's first call,
its get_valid_tactics asserted) - tactics chosen for the TP layout, a path the tile check above never exercises (its
launches are tactic -1). At the first target layer's tile check the king's module (its sibling name, as for the tile
follow) is read for PINS: a non-empty dict -> its fingerprint travels in V1 (the same on every rank, or a miss) and,
after the commit, for the duration of every EP layer's call (routed) the king's PINS is {} (the one found at entry back
after it, also when the call raises), so those row counts take the tactic -1 path this route validates (the record and
the body on the launcher's tile for the EP shape). With PINS emptied the v32 king's decode MoE path is the v339 king's
this route was reviewed against (r59h: PINS is that file's only functional change). A PINS that is not a {rows: tactic}
dict is a tile-check miss (EP OFF everywhere); absent / empty / the module not loaded (the v339 king) -> nothing to swap.
apply_ep.py pins the one read (pins_point).
Census (engine log, WARNING; this module's logger keeps WARNING under the arena's log_level error): "pfep: ...".
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time

import torch
from flashinfer import nvfp4_block_scale_interleave
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import core as moe_core
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.utils import prepare_static_weights_for_trtllm_fp4_moe
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_disagg, get_parallel

NODE = "layer"  # the claimed node this component's prepare runs on (nodes.py): a decoder layer
EP = True        # False = this component does nothing (the base bundle's behaviour exactly)
TP = 2           # the prefill / audit engines' TP group (the EP group is the same ranks)
ROLES = ("prefill", "null")  # the TP2 engines the king's moe_dsl_route serves (live prefill + the audit engine)
EXPERTS, HIDDEN, INTER_TP, TOP_K = 288, 4096, 1024, 8  # GLM-5.3-Flash-NVFP4's routed experts at TP 2
FIRST_LAYER, LAST_LAYER = 3, 44  # the target MoE layers (0-2 dense; 45 is the MTP draft layer, never a "layer" node)
ROUND = 8        # experts a destination rank per all_to_all round (transient ~0.26 GB a rank)
TILE_M = 128     # epilogue_tile_m of prepare_static_weights_for_trtllm_fp4_moe
SF_VEC = 16      # num_elts_per_sf of its scale permutes
PROBE_ROWS = 144  # every expert picked PROBE_ROWS * TOP_K / EXPERTS = 4 times
PROBE_SEED = 20261119
BAR_MAX, BAR_MEAN = 2e-2, 1e-2  # max|d| / max|tp| and mean|d| / mean|tp| of the probe (README: CPU model)
# the king's moe_prepared_route.py literals (pinned to its text by apply_ep.py, AST for AST, with default_tile below):
MIN_ROWS, MAX_ROWS = 17, 768  # the rows its prepared decode path serves
FP4_TILES = (8, 16, 32, 64, 128, 192, 256)  # the NVFP4 tiles its default_tile resolves over
TILE_LOGIT = 4.0  # the tile check's crafted logits: +4 picked, -4 not (sigmoid 0.982 / 0.018; zero bias)
_STOCK_QUANT = flashinfer_trtllm.quantize_hidden_states_fp4  # SGLang's MoE input quantize (import time)
_MOE = moe_core.trtllm_fp4_block_scale_moe                      # FlashInfer's entry (core's, not the package alias)
_WEIGHTS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
_ALIASES = ("w13_blockscale_swizzled", "w2_blockscale_swizzled")  # set to the scale parameters (modelopt :2985-2986)
_SLICED = ("g1_alphas", "g1_alphas_up", "g2_alphas", "g1_scale_c", "w13_weight_scale_2", "w2_weight_scale_2",
           "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")  # per LOCAL expert (created with num_local_experts)
_GLOBAL = ("w13_input_scale", "w2_input_scale")  # [num_experts, ...] (_sglang_require_global_experts): never sliced
_PKG = __name__.rsplit(".", 1)[0]  # this package as the validator materialized it (dsa_rtk_route.py :110's form)
_KING = ".moe_prepared_route"      # the sibling whose global default_tile the tile follow swaps
_PCACHE: dict = {}  # FlashInfer's permute-index cache for the EP forward (keyed by shape, as SGLang's loop keys it)
_LAYERS: dict = {}  # id(decoder layer) -> "EP" | "OFF" | "dense"
_ST = {"mode": None, "why": None, "maps": None, "ep": 0, "ms": 0.0, "served": False, "rank": None,
       "follow": None, "follow_runs": None, "followed": False,  # follow: (the king's module, our default_tile)
       "pins": None, "pins_line": None}  # pins (r59i): the king's module whose PINS stands down inside EP layer calls
_SAID: set = set()
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)  # the arena's log_level "error" sets the ROOT to ERROR; our census lines are WARNING


class _Kind:
    """One of the four routed-expert tensors: its parameter, its TP and EP per-expert [rows, cols] byte shapes."""

    def __init__(self, name, gated, scale, tp_shape, ep_shape):
        self.name, self.gated, self.scale = name, gated, scale
        self.tp, self.ep = tp_shape, ep_shape
        self.u_tp, self.u_ep = tp_shape[0] * tp_shape[1], ep_shape[0] * ep_shape[1]


def _kinds():
    """The four tensors at the literals (TP rows / cols -> EP rows / cols, bytes)."""
    i_ep = INTER_TP * TP
    return (_Kind("w13_weight", True, False, (2 * INTER_TP, HIDDEN // 2), (2 * i_ep, HIDDEN // 2)),
            _Kind("w13_weight_scale", True, True, (2 * INTER_TP, HIDDEN // SF_VEC), (2 * i_ep, HIDDEN // SF_VEC)),
            _Kind("w2_weight", False, False, (HIDDEN, INTER_TP // 2), (HIDDEN, i_ep // 2)),
            _Kind("w2_weight_scale", False, True, (HIDDEN, INTER_TP // SF_VEC), (HIDDEN, i_ep // SF_VEC)))


def _param(experts, name):
    """The routed-expert parameter `name` (literal attribute reads only)."""
    if name == "w13_weight":
        return experts.w13_weight
    if name == "w13_weight_scale":
        return experts.w13_weight_scale
    if name == "w2_weight":
        return experts.w2_weight
    return experts.w2_weight_scale


def _vectors(experts) -> dict:
    """The per-local-expert parameters present on the layer (name -> Parameter)."""
    found = {"g1_alphas": getattr(experts, "g1_alphas", None), "g1_alphas_up": getattr(experts, "g1_alphas_up", None),
             "g2_alphas": getattr(experts, "g2_alphas", None), "g1_scale_c": getattr(experts, "g1_scale_c", None),
             "w13_weight_scale_2": getattr(experts, "w13_weight_scale_2", None),
             "w2_weight_scale_2": getattr(experts, "w2_weight_scale_2", None),
             "gemm1_alpha": getattr(experts, "gemm1_alpha", None), "gemm1_beta": getattr(experts, "gemm1_beta", None),
             "gemm1_clamp_limit": getattr(experts, "gemm1_clamp_limit", None)}
    return {k: v for k, v in found.items() if v is not None}


def _unique(vec: dict) -> list:
    """The vectors as (first name, Parameter) pairs, each Parameter object once (an object registered under two names
    is sliced once: a second slice of an already sliced [72] view would be another expert range or empty)."""
    seen, out = set(), []
    for name, p in vec.items():
        if id(p) not in seen:
            seen.add(id(p))
            out.append((name, p))
    return out


def default_tile(rows: int, top_k: int, local_experts: int) -> int:
    """moe_prepared_route.py's default_tile (:250-256; apply_ep.py refuses a build whose body differs from the king's):
    the record tile of a tactic -1 shape."""
    average = float(rows) * float(top_k) / local_experts
    i = next((k for k, tile in enumerate(FP4_TILES) if not tile < average), len(FP4_TILES) - 1)
    return min({FP4_TILES[i], *FP4_TILES[i + 1:i + 3], *((FP4_TILES[i - 1],) if i > 0 else ())})


def _runs(tiles) -> str:
    """A tile a row count over MIN_ROWS .. MAX_ROWS as runs ("8 at 17-144, 16 at 145-288, ..."): lossless, so equal
    runs on two ranks = equal tables (the tile check's fingerprint in V1)."""
    runs = []
    for rows, t in zip(range(MIN_ROWS, MAX_ROWS + 1), tiles):
        if runs and runs[-1][0] == t:
            runs[-1][2] = rows
        else:
            runs.append([t, rows, rows])
    return ", ".join(f"{t} at {a}-{b}" for t, a, b in runs)


def _tile_table(local: int) -> str:
    """default_tile(rows, TOP_K, local) over MIN_ROWS .. MAX_ROWS as runs."""
    return _runs([default_tile(rows, TOP_K, local) for rows in range(MIN_ROWS, MAX_ROWS + 1)])


def _king_m1():
    """The king's moe_prepared_route module under this package's materialized name, or None."""
    return sys.modules.get(_PKG + _KING)


def _following(king_tile, table: dict):
    """The king's default_tile while an EP layer's call runs under the tile follow: the launcher's measured tile at
    (TOP_K, the local count) for the row counts the tile check measured, the king's own (king_tile) otherwise."""
    local = EXPERTS // TP

    def followed_tile(rows: int, top_k: int, local_experts: int) -> int:
        if top_k == TOP_K and local_experts == local and rows in table:
            if not _ST["followed"]:
                _ST["followed"] = True
                _log.warning("pfep: tile follow: the king's record tile follows the launcher (%s)", _ST["follow_runs"])
            return table[rows]
        return king_tile(rows, top_k, local_experts)

    return followed_tile


def _say(tag: str, detail: str) -> None:
    """One log line a kind of event."""
    if tag not in _SAID:
        _SAID.add(tag)
        _log.warning("%s", detail)


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def _uniform_refusal(layer, experts):
    """Why this layer cannot be resharded, from rank-uniform facts only (no collective has run), or None."""
    mode = get_disagg().disaggregation_mode
    if str(getattr(mode, "value", mode)) not in ROLES:
        return "inactive", f"not a TP2 prefill / audit engine (disaggregation_mode {mode!r})"
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return "inactive", "torch.distributed is not initialized"
    group = get_tp_group()
    if group.world_size != TP:
        return "inactive", f"the TP group has {group.world_size} ranks, not {TP}"
    lid = getattr(layer, "layer_id", None)
    if not (type(lid) is int and FIRST_LAYER <= lid <= LAST_LAYER) or getattr(experts, "is_draft", False):
        return "layer", f"layer {lid!r} is not a target MoE layer {FIRST_LAYER}..{LAST_LAYER} (or a draft layer)"
    if torch.cuda.is_current_stream_capturing():
        return "capture", "the layer's first call is inside a stream capture"
    if deepseek_v2.is_in_tc_piecewise_cuda_graph():  # r61: a breakable graph's eager warmup is this engine's first call
        return "capture", "the layer's first call is inside a piecewise graph"
    if AutoTuner.get().is_tuning_mode:
        return "tuning", "FlashInfer's AutoTuner is in tuning mode"
    return None


def _check_form(layer, experts, group) -> None:
    """The layer is the stock TP4 NVFP4 FlashInfer TRT-LLM MoE this reshard is written for (raises ValueError)."""
    method = experts.quant_method
    backend = getattr(getattr(method, "runner", None), "runner_backend", None)
    cfg = experts.moe_runner_config
    disp = experts.dispatcher
    par = get_parallel()
    _need(isinstance(method, ModelOptNvFp4FusedMoEMethod) and getattr(method, "use_flashinfer_trtllm_weight_layout",
                                                                        False)
          and callable(getattr(backend, "is_flashinfer_trtllm", None)) and backend.is_flashinfer_trtllm()
          and not method.quant_config.use_per_token_activation,
          "not the NVFP4 FlashInfer TRT-LLM MoE (static activation scales)")
    _need(par.moe_ep_size == 1 and par.moe_tp_size == TP and experts.moe_ep_size == 1 and experts.moe_ep_rank == 0
          and experts.moe_tp_size == TP and experts.moe_tp_rank == group.rank_in_group,
          f"the layer is not TP{TP} (moe_ep {experts.moe_ep_size}, moe_tp {experts.moe_tp_size}, moe_tp_rank "
          f"{experts.moe_tp_rank} vs group rank {group.rank_in_group})")
    _need(experts.num_experts == EXPERTS and experts.num_local_experts == EXPERTS and experts.top_k == TOP_K
          and experts.hidden_size == HIDDEN and experts.intermediate_size_per_partition == INTER_TP
          and cfg.num_local_experts == EXPERTS and cfg.intermediate_size_per_partition == INTER_TP
          and experts.num_fused_shared_experts == 0 and cfg.is_gated and cfg.activation == "silu"
          and getattr(experts, "trtllm_padded_hidden_size", None) is None and EXPERTS % TP == 0,
          f"the MoE is not {EXPERTS} experts top {TOP_K}, hidden {HIDDEN}, intermediate {INTER_TP} a rank, gated silu, "
          "unpadded, no fused shared expert")
    _need(isinstance(disp, StandardDispatcher) and disp.skip_local_expert_mapping is True
          and disp.local_expert_mapping is None and disp.moe_ep_size == 1,
          "the dispatcher is not the standard one keeping global expert ids (EP fields would remap topk ids)")
    for kd in _kinds():
        p = _param(experts, kd.name)
        want = torch.float8_e4m3fn if kd.scale else torch.uint8
        t = p.data
        _need(isinstance(p, torch.nn.Parameter) and t.dtype == want and tuple(t.shape) == (EXPERTS, *kd.tp)
              and t.is_contiguous() and t.storage_offset() == 0
              and t.untyped_storage().nbytes() == EXPERTS * kd.u_tp,
              f"{kd.name} is {t.dtype} {tuple(t.shape)} (contiguous {t.is_contiguous()}), not {want} "
              f"[{EXPERTS}, {kd.tp[0]}, {kd.tp[1]}] owning its storage")
    _need(experts.w13_blockscale_swizzled is experts.w13_weight_scale
          and experts.w2_blockscale_swizzled is experts.w2_weight_scale,
          "the swizzled-blockscale aliases are not the scale parameters")
    known = set(_WEIGHTS) | set(_ALIASES) | set(_SLICED) | set(_GLOBAL)
    for name, t in list(experts.named_parameters(recurse=False)) + list(experts.named_buffers(recurse=False)):
        if t is not None and t.dim() >= 1 and t.shape[0] == EXPERTS:
            _need(name in known, f"an unknown per-expert tensor {name} {tuple(t.shape)} (its EP form is unknown)")
    vec = _vectors(experts)
    for name, p in vec.items():
        _need(isinstance(p, torch.nn.Parameter) and p.data.dim() >= 1 and p.data.shape[0] == EXPERTS
              and p.data.is_contiguous(), f"{name} is not a contiguous per-expert parameter [{EXPERTS}, ...]")
    for name in ("g1_alphas", "g2_alphas", "g1_scale_c"):
        _need(name in vec, f"{name} missing")
    # the in-place slice of a vector must change nothing else: no storage shared with a weight or a global input
    # scale, and two vectors on one storage only as the same view (the same object is sliced once: _unique; distinct
    # objects on one view slice it idempotently: each copies its own untouched [a, b) into [0, 72))
    fixed = {_param(experts, kd.name).data.untyped_storage().data_ptr(): kd.name for kd in _kinds()}
    for name, t in (("w13_input_scale", getattr(experts, "w13_input_scale", None)),
                    ("w2_input_scale", getattr(experts, "w2_input_scale", None))):
        if isinstance(t, torch.Tensor):
            fixed[t.untyped_storage().data_ptr()] = name
    first: dict = {}
    for name, p in vec.items():
        key = p.data.untyped_storage().data_ptr()
        _need(key not in fixed, f"{name} shares its storage with {fixed.get(key)} (an in-place slice would change it)")
        if key in first:
            q_name, q = first[key]
            _need(q is p or (p.data.data_ptr() == q.data.data_ptr() and p.data.shape == q.data.shape
                             and p.data.stride() == q.data.stride()),
                  f"{name} and {q_name} share a storage at different views")
        else:
            first[key] = (name, p)
    gs = getattr(experts, "w13_input_scale_quant", None)
    _need(isinstance(gs, torch.Tensor) and gs.numel() == 1, "the MoE input scale is not one scalar")
    cfg_topk = layer.mlp.topk.topk_config
    _need(isinstance(cfg_topk.correction_bias, torch.Tensor) and tuple(cfg_topk.correction_bias.shape) == (EXPERTS,),
          "the router's correction bias is not [num_experts]")
    # the routing the king's prepared path serves (moe_prepared_route.py :400-401) and the tile check's crafted routing
    # is written for: DeepSeekV3 (sigmoid + bias, top-k) in one group
    _need(experts.routing_method_type == moe_core.RoutingMethodType.DeepSeekV3.value
          and cfg_topk.num_expert_group in (None, 0, 1) and cfg_topk.topk_group in (None, 0, 1),
          f"the routing is method {experts.routing_method_type!r} at n_group {cfg_topk.num_expert_group!r} / topk_group "
          f"{cfg_topk.topk_group!r}, not DeepSeekV3 in one group")


def _uninterleave(rows: int, cols: int, dev) -> torch.Tensor:
    """inv with pre_flat = stored_flat[inv] for stored = nvfp4_block_scale_interleave(pre) of a [rows, cols] uint8
    matrix: the real function applied to the base-256 digit planes of an iota (no layout formula transcribed)."""
    n = rows * cols
    iota = torch.arange(n, dtype=torch.int64, device=dev)
    src = torch.zeros(n, dtype=torch.int64, device=dev)
    p = 0
    while (n - 1) >> (8 * p):
        plane = ((iota >> (8 * p)) & 255).to(torch.uint8).view(rows, cols)
        out = nvfp4_block_scale_interleave(plane)
        _need(out.dtype == torch.uint8 and out.numel() == n, f"the interleave of [{rows}, {cols}] pads ({out.numel()})")
        src += out.reshape(-1).to(torch.int64) << (8 * p)
        p += 1
    _need(torch.equal(torch.sort(src).values, iota), f"the interleave of [{rows}, {cols}] is not a permutation")
    inv = torch.empty_like(src)
    inv[src] = iota
    return inv


def _maps(dev) -> dict:
    """The TP shapes' inverse maps (built once): rows kinds -> the inverse row permutation; scale kinds -> the flat
    gather from stored bytes to the canonical [rows, cols] bytes."""
    if _ST["maps"] is not None:
        return _ST["maps"]
    cache: dict = {}
    maps = {}
    for kd in _kinds():
        rows, cols = kd.tp
        dummy = torch.empty((rows, cols), dtype=torch.uint8)  # only its shape / dtype / device are read
        if kd.gated:
            perm = moe_core._maybe_get_cached_w3_w1_permute_indices(
                cache, dummy, TILE_M, num_elts_per_sf=SF_VEC if kd.scale else None, is_gated_act_gemm=True)
        else:
            perm = moe_core.get_w2_permute_indices_with_cache(cache, dummy, TILE_M,
                                                              num_elts_per_sf=SF_VEC if kd.scale else None)
        perm = perm.to(torch.int64).cpu()
        _need(torch.equal(torch.sort(perm).values, torch.arange(rows)), f"{kd.name}'s TP row permute is not one")
        inv_rows = torch.argsort(perm).to(dev)
        if kd.scale:
            inv_ilv = _uninterleave(rows, cols, dev)
            pre = (inv_rows[:, None] * cols + torch.arange(cols, device=dev)[None, :]).reshape(-1)
            maps[kd.name] = inv_ilv[pre]
        else:
            maps[kd.name] = inv_rows
    _ST["maps"] = maps
    return maps


def _forward(kd, src, out_flat, scratch) -> None:
    """SGLang's per-expert loop of prepare_static_weights_for_trtllm_fp4_moe for one tensor of one expert, with
    FlashInfer's helpers, into out_flat (no allocation but the cached index tensors)."""
    if kd.gated:
        p = moe_core._maybe_get_cached_w3_w1_permute_indices(
            _PCACHE, src, TILE_M, num_elts_per_sf=SF_VEC if kd.scale else None, is_gated_act_gemm=True)
    else:
        p = moe_core.get_w2_permute_indices_with_cache(_PCACHE, src, TILE_M,
                                                       num_elts_per_sf=SF_VEC if kd.scale else None)
    p = p.to(src.device)
    if kd.scale:
        torch.index_select(src, 0, p, out=scratch)
        nvfp4_block_scale_interleave(scratch, out=out_flat)
    else:
        torch.index_select(src, 0, p, out=out_flat.view(src.shape))


def _local(fail, fn):
    """Run local work unless this rank already missed; an exception (OOM included) or a returned reason is a miss."""
    if fail is not None:
        return fail
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - this rank's miss, voted like a difference; no collective is skipped
        return f"{type(exc).__name__}: {exc}"[:300]


def _probe_inputs():
    """The probe's router logits and inputs (CPU, seeded: the same on every rank): pass p of PROBE_ROWS / (EXPERTS /
    TOP_K) rows picks every expert once (8 consecutive entries of a seeded expert order a row, shifted by
    EXPERTS / 4 + 1 a pass), logits +4 picked / -4 not (sigmoid scores 0.98 / 0.02, the bias zero)."""
    g = torch.Generator().manual_seed(PROBE_SEED)
    order = torch.randperm(EXPERTS, generator=g)
    logits = torch.randn(PROBE_ROWS, EXPERTS, generator=g) * 0.01 - 4.0
    per_pass = EXPERTS // TOP_K
    for i in range(PROBE_ROWS):
        base = (i % per_pass) * TOP_K + (i // per_pass) * (EXPERTS // 4 + 1)
        pick = order[(base + torch.arange(TOP_K)) % EXPERTS]
        logits[i, pick] = 4.0 + 0.01 * torch.randn(TOP_K, generator=g)
    return logits, torch.randn(PROBE_ROWS, HIDDEN, generator=g).to(torch.bfloat16)


def _tile_inputs(rank: int) -> torch.Tensor:
    """The tile check's router logits for this rank (CPU, deterministic) [MAX_ROWS, EXPERTS]: TILE_LOGIT on 8 picks a
    row, -TILE_LOGIT elsewhere; row 0 picks local expert 0 (global 72 r), row 1 local expert 1, every other pick of
    every row another rank's expert (8 consecutive of the 216 in a row-shifted order). Verified here under the routing
    stock passes (DeepSeekV3 in one group: sigmoid + the zero bias, top 8): the selection is exactly the picks."""
    local = EXPERTS // TP
    a = rank * local
    others = torch.tensor([e for e in range(EXPERTS) if not a <= e < a + local], dtype=torch.int64)
    idx = (torch.arange(MAX_ROWS)[:, None] * TOP_K + torch.arange(TOP_K)[None, :]) % others.numel()
    picks = others[idx]
    picks[0, 0], picks[1, 0] = a, a + 1
    logits = torch.full((MAX_ROWS, EXPERTS), -TILE_LOGIT, dtype=torch.float32)
    logits.scatter_(1, picks, TILE_LOGIT)
    sel = torch.topk(torch.sigmoid(logits), TOP_K, dim=1).indices
    hit = (picks >= a) & (picks < a + local)
    _need(torch.equal(sel.sort(1).values, picks.sort(1).values) and int(hit.sum()) == 2
          and bool(hit[0, 0]) and bool(hit[1, 0]),
          "the tile check's crafted routing does not select exactly its picks (two local: rows 0 and 1)")
    return logits


def _alloc(experts, mlp, rank, dev, tile) -> dict:
    """Every buffer of the reshard, the tile check (`tile`: the first target layer) and the probe, before the first
    vote (an OOM here is this rank's V0 miss)."""
    kinds, local = _kinds(), EXPERTS // TP
    a, b = rank * local, rank * local + local
    big = max(kd.u_tp for kd in kinds)
    clones: dict = {}
    vec = {}
    for name, p in _vectors(experts).items():  # one clone a Parameter object (aliases share it)
        if id(p) not in clones:
            clones[id(p)] = p.data[a:b].clone()
        vec[name] = clones[id(p)]
    bufs = {"final": {kd.name: torch.empty((local, kd.u_ep), dtype=torch.uint8, device=dev) for kd in kinds},
            "canon": {kd.name: torch.empty((ROUND, kd.u_ep), dtype=torch.uint8, device=dev) for kd in kinds},
            "ep_scr": {kd.name: torch.empty(kd.ep, dtype=torch.uint8, device=dev) for kd in kinds if kd.scale},
            "tp_scr": {kd.name: torch.empty(kd.tp, dtype=torch.uint8, device=dev) for kd in kinds if kd.scale},
            "send": torch.empty(TP * ROUND * big, dtype=torch.uint8, device=dev),
            "recv": torch.empty(TP * ROUND * big, dtype=torch.uint8, device=dev),
            "tp_canon": torch.empty(ROUND * big, dtype=torch.uint8, device=dev),
            "chk": torch.empty(big, dtype=torch.uint8, device=dev),
            "vec": vec,
            "sum": torch.zeros((2, PROBE_ROWS, HIDDEN), dtype=torch.float32, device=dev)}
    logits, x = _probe_inputs()
    bufs["logits"] = logits.to(dev)
    bufs["x"] = x.to(dev)
    bufs["bias"] = torch.zeros_like(mlp.topk.topk_config.correction_bias)
    if tile:  # the tile check's inputs at MAX_ROWS (each row count a prefix view): logits, zero NVFP4 activations
        bufs["t_logits"] = _tile_inputs(rank).to(dev)
        bufs["t_ids"] = torch.empty((MAX_ROWS, TOP_K), dtype=torch.int32, device=dev)  # FromLogits outputs
        bufs["t_w"] = torch.empty((MAX_ROWS, TOP_K), dtype=torch.bfloat16, device=dev)
        bufs["t_xq"] = torch.zeros((MAX_ROWS, HIDDEN // 2), dtype=torch.uint8, device=dev)
        bufs["t_xsf"] = torch.zeros((MAX_ROWS, HIDDEN // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        bufs["t_sum"] = torch.zeros((MAX_ROWS - MIN_ROWS + 1, 3), dtype=torch.int64, device=dev)
    return bufs


def _gather_send(send, old, r0, c) -> None:
    """The round's send buffer: for every destination d, its experts [d * 72 + r0, + c) of this rank's TP slice."""
    send.copy_(old[:, r0:r0 + c])


def _assemble(kd, recv, c, maps, bufs, sender_check) -> str | None:
    """The c received experts of every sender: canonical TP bytes (inverse maps), placed in the EP canonical layout;
    sender_check proves forward_TP(canonical) == the received bytes for each of them."""
    rows, cols = kd.tp
    canon = bufs["canon"][kd.name][:c]
    for s in range(TP):
        src = recv[s]  # [c, u_tp]
        tpc = bufs["tp_canon"][:c * kd.u_tp].view(c, kd.u_tp)
        if kd.scale:
            torch.index_select(src, 1, maps[kd.name], out=tpc)
        else:
            torch.index_select(src.view(c, rows, cols), 1, maps[kd.name], out=tpc.view(c, rows, cols))
        if sender_check:
            for i in range(c):
                chk = bufs["chk"][:kd.u_tp]
                _forward(kd, tpc[i].view(rows, cols), chk, bufs["tp_scr"].get(kd.name))
                if not torch.equal(chk, src[i]):
                    return f"{kd.name}: forward_TP(inverse_TP(bytes)) != the bytes (sender {s}, expert {i})"
        if kd.gated:  # EP row h * (rows_ep / 2) + s * (rows / 2) + j <- sender s's canonical row h * (rows / 2) + j
            canon.view(c, 2, TP, rows // 2, cols)[:, :, s].copy_(tpc.view(c, 2, rows // 2, cols))
        else:         # EP column block s <- sender s's canonical columns
            canon.view(c, rows, TP, cols)[:, :, s].copy_(tpc.view(c, rows, cols))
    return None


def _forward_chunk(kinds, r0, c, bufs, sglang_check) -> str | None:
    """The EP forward of the chunk's c experts into the EP buffers; sglang_check: expert r0 == SGLang's own
    prepare_static_weights_for_trtllm_fp4_moe on the same canonical bytes."""
    for i in range(c):
        for kd in kinds:
            _forward(kd, bufs["canon"][kd.name][i].view(kd.ep), bufs["final"][kd.name][r0 + i], bufs["ep_scr"].get(
                kd.name))
    if sglang_check:
        cw = {kd.name: bufs["canon"][kd.name][0].view(1, *kd.ep) for kd in kinds}
        got = prepare_static_weights_for_trtllm_fp4_moe(
            cw["w13_weight"], cw["w2_weight"], cw["w13_weight_scale"], cw["w2_weight_scale"], HIDDEN, INTER_TP * TP, 1,
            is_gated=True)
        for kd, t in zip(kinds, (got[0], got[1], got[2], got[3])):
            if not torch.equal(t.contiguous().view(torch.uint8).reshape(-1), bufs["final"][kd.name][r0]):
                return f"{kd.name}: our EP forward != SGLang's prepare_static_weights_for_trtllm_fp4_moe (expert {r0})"
    return None


def _tile_verdict(got) -> tuple:
    """got[j] = (row 0's permuted row, row 1's permuted row, local picks) of the launch at MIN_ROWS + j rows ->
    (None, the launcher's tile a row count) when every launch reads as the expected layout (two local picks, row 0's
    at 0, row 1's at the launcher's tile) with a tile of the king's FP4_TILES, else (the first reason, None)."""
    tiles = []
    for j, (p0, p1, n) in enumerate(got):
        rows = MIN_ROWS + j
        if n != 2 or p0 != 0 or p1 <= 0:
            return (f"tile check: {rows} rows: a layout I cannot interpret ({n} local picks, rows 0 / 1 permuted to "
                    f"{p0} / {p1}; want 2: 0 / a tile of {FP4_TILES})"), None
        if p1 not in FP4_TILES:
            return (f"tile check: {rows} rows: the launcher's tile {p1} is not one of the king's FP4_TILES {FP4_TILES} "
                    "(the tiles its record and kernels are written for): no tile follow for it"), None
        tiles.append(p1)
    return None, tuple(tiles)


def _tile_diff(tiles, local) -> list:
    """(rows, the launcher's tile, the record's default_tile(rows, TOP_K, local)) where they differ."""
    return [(rows, t, default_tile(rows, TOP_K, local)) for rows, t in zip(range(MIN_ROWS, MAX_ROWS + 1), tiles)
            if t != default_tile(rows, TOP_K, local)]


def _tile_decision(votes) -> tuple:
    """From V1's votes alone (rank-uniform): (the tile check's census line, the misses it adds [(rank, why)], follow).
    follow: every rank measured the SAME table and it differs from the record's default_tile somewhere."""
    local = EXPERTS // TP
    tiles = [v[3] for v in votes]
    if not all(t is not None for t in tiles):
        miss = [(r, v[0]) for r, v in enumerate(votes) if isinstance(v[0], str) and v[0].startswith("tile check")]
        if miss:
            return f"pfep: tile check: MISS on rank(s) {[r for r, _ in miss]}: {miss[0][1]}", [], False
        return "pfep: tile check: not completed on every rank (an earlier miss: the OFF line says which)", [], False
    runs = [t[1] for t in tiles]
    modal = max(runs, key=runs.count)
    m = runs.index(modal)
    odd = [(r, f"tile check: rank {r}'s launcher tiles ({x}) differ from rank {m}'s ({modal})")
           for r, x in enumerate(runs) if x != modal]
    if odd:
        return f"pfep: tile check: MISS on rank(s) {[r for r, _ in odd]}: {odd[0][1]}", odd, False
    n, _, ndiff, first = tiles[0]
    if not ndiff:
        return (f"pfep: tile check: {n} row counts ({MIN_ROWS}..{MAX_ROWS}), launcher tile == record tile "
                f"default_tile(rows, {TOP_K}, {local}) on every rank ({modal})"), [], False
    return (f"pfep: tile check: {n} row counts ({MIN_ROWS}..{MAX_ROWS}), launcher tile != record tile default_tile(rows, "
            f"{TOP_K}, {local}) at {ndiff} row counts, the same table on every rank (first {first[0]} rows: launcher "
            f"{first[1]}, record {first[2]}): the king's record follows the launcher ({modal})"), [], True


def _tile_check(experts, mlp, bufs, stats) -> str | None:
    """Step 3 of the module docstring + the PINS FOLLOW's lookup (this rank's verdict; an exception is a miss named as
    the tile check's)."""
    try:
        why = _tile_launches(experts, mlp, bufs, stats)
        return why if why is not None else _pins_lookup(stats)
    except Exception as exc:  # noqa: BLE001 - this rank's miss, voted at V1; no collective is skipped
        return f"tile check: {type(exc).__name__}: {exc}"[:300]


def _pins_lookup(stats) -> str | None:
    """The PINS FOLLOW's lookup (module docstring): stats["pins"] = (the king's module when its PINS is a non-empty dict,
    else None; the fingerprint V1 compares). A PINS that is not a {rows: tactic} dict: this rank's miss ("pins: ...",
    r59j: the tile table itself passed, so its census line reads as a pass; the OFF line names the PINS miss)."""
    king = _king_m1()
    if king is None:
        stats["pins"] = (None, "no king module")
        return None
    pins = getattr(king, "PINS", None)
    if pins is None:
        stats["pins"] = (None, "no PINS")
        return None
    if not (isinstance(pins, dict) and all(isinstance(k, int) for k in pins)):
        return (f"pins: {_PKG}{_KING}.PINS is a {type(pins).__name__}, not the {{rows: (tile_N, config)}} dict "
                "apply_ep pinned")
    if not pins:
        stats["pins"] = (None, "empty PINS")
        return None
    stats["pins"] = (king, f"{len(pins)} pinned row counts {sorted(pins)}")
    return None


def _tile_launches(experts, mlp, bufs, stats) -> str | None:
    """One eager tactic -1 launch of FlashInfer's FP4 MoE (the king's MoERunner form, from logits, unfinalized) on the
    EP buffers a row count; the expanded -> permuted maps reduced on the device, one host read at the end."""
    rank, local = _ST["rank"], EXPERTS // TP
    cfg, topk = experts.moe_runner_config, mlp.topk.topk_config
    w = {kd.name: bufs["final"][kd.name].view(local, *kd.ep) for kd in _kinds()}
    w13s = w["w13_weight_scale"].view(torch.float8_e4m3fn)
    w2s = w["w2_weight_scale"].view(torch.float8_e4m3fn)
    v = bufs["vec"]
    xq, xsf = bufs["t_xq"], bufs["t_xsf"]
    sm100 = moe_core.get_trtllm_moe_sm100_module()
    runner = sm100.MoERunner(
        sm100.moe_op, top_k=TOP_K, num_local_experts=local,
        dtype_act=moe_core.deduce_trtllm_gen_tensor_dtype(xq, xsf),
        dtype_weights=moe_core.deduce_trtllm_gen_tensor_dtype(w["w13_weight"], w13s),
        fp8_quantization_type=moe_core.Fp8QuantizationType.NoneFp8, hidden_size=HIDDEN, hidden_size_output=HIDDEN,
        intermediate_size=INTER_TP * TP,
        activation_type=flashinfer_trtllm.get_activation_type(cfg.activation, is_gated=cfg.is_gated),
        weight_layout=moe_core.WeightLayout.MajorK, use_shuffled_weight=True, use_per_token_scaling=False,
        num_experts=EXPERTS, num_fused_shared_experts=0)
    call = dict(routing_input_mode=moe_core.RoutingInputMode.FromLogits, num_experts=EXPERTS, routing_bias=bufs["bias"],
                gemm1_weights=w["w13_weight"], gemm1_weights_scale=w13s, gemm1_bias=None,
                gemm1_alpha=v.get("gemm1_alpha"), gemm1_beta=v.get("gemm1_beta"),
                gemm1_clamp_limit=v.get("gemm1_clamp_limit"), gemm2_weights=w["w2_weight"], gemm2_weights_scale=w2s,
                gemm2_bias=None, output1_scale_scalar=v["g1_scale_c"], output1_scale_gate_scalar=v["g1_alphas"],
                output2_scale_scalar=v["g2_alphas"], n_group=topk.num_expert_group, topk_group=topk.topk_group,
                local_expert_offset=rank * local, routed_scaling_factor=cfg.routed_scaling_factor,
                routing_method_type=experts.routing_method_type, do_finalize=False, enable_pdl=False,
                num_fused_shared_experts=0, norm_topk_prob=True)
    summ = bufs["t_sum"]
    for j, rows in enumerate(range(MIN_ROWS, MAX_ROWS + 1)):
        inputs = moe_core.MoeRunnerInputs(
            output=moe_core._alloc_trtllm_moe_output(rows, HIDDEN, False, xq.device),
            routing_logits=bufs["t_logits"][:rows], topk_ids=bufs["t_ids"][:rows], expert_weights=bufs["t_w"][:rows],
            hidden_states=xq[:rows], hidden_states_scale=xsf[:rows], gemm1_lora_delta=None,
            per_token_scale=None).to_list()
        res = runner.forward(inputs, tactic=-1, **call)
        e2p = res[2] if len(res) == 3 else None
        # r59k (box11 11-20, measured): the image's FlashInfer returns the expanded -> permuted map FLAT, [rows * TOP_K],
        # token-major (t * TOP_K + k) -- exactly the layout the king's moe_finalize_native.cu consumes (rows.dim() == 1,
        # numel == tokens * TOP_K, token t's slots at [8t, 8t + 8)); the r59c check expected [rows, TOP_K] and turned EP OFF
        if e2p is not None and e2p.dim() == 1 and e2p.numel() == rows * TOP_K:
            e2p = e2p.view(rows, TOP_K)
        if e2p is None or tuple(e2p.shape) != (rows, TOP_K):
            return (f"tile check: {rows} rows: the unfinalized call returned {len(res)} tensors, the last "
                    f"{tuple(res[-1].shape) if res else ()}, not [rows, weights, expanded -> permuted [{rows}, {TOP_K}] "
                    f"or flat [{rows * TOP_K}]] (a layout I cannot interpret)")
        summ[j, 0] = e2p[0].max()
        summ[j, 1] = e2p[1].max()
        summ[j, 2] = (e2p >= 0).sum()
        res = e2p = None
    why, tiles = _tile_verdict([tuple(int(x) for x in row) for row in summ.tolist()])  # one host read
    if why is not None:
        return why
    diff = _tile_diff(tiles, local)
    if diff:  # the record must follow the launcher: the king's module and its pinned default_tile must be reachable
        king = _king_m1()
        king_tile = getattr(king, "default_tile", None)
        if not (callable(king_tile) and all(king_tile(rows, TOP_K, local) == default_tile(rows, TOP_K, local)
                                            for rows in range(MIN_ROWS, MAX_ROWS + 1))):
            return (f"tile check: the launcher's tile != the record's default_tile(rows, {TOP_K}, {local}) at "
                    f"{len(diff)} of {len(tiles)} row counts (first {diff[0][0]} rows: launcher {diff[0][1]}, record "
                    f"{diff[0][2]}) and no tile follow is possible: {_PKG}{_KING} "
                    f"{'is not loaded' if king is None else 'has no default_tile apply_ep pinned'}")
        stats["king"] = (king, king_tile)
    stats["table"] = tiles
    stats["tiles"] = (len(tiles), _runs(tiles), len(diff), diff[0] if diff else None)
    return None


def _probe_call(experts, mlp, bufs, form):
    """One FlashInfer MoE call of the probe in `form` ("tp": the layer's buffers; "ep": the new ones)."""
    cfg, topk = experts.moe_runner_config, mlp.topk.topk_config
    local, rank = EXPERTS // TP, _ST["rank"]
    vec = _vectors(experts)
    if form == "tp":
        w = {kd.name: _param(experts, kd.name).data for kd in _kinds()}
        v = {k: p.data for k, p in vec.items()}
        inter, offset, n_local = INTER_TP, 0, EXPERTS
    else:
        w = {kd.name: bufs["final"][kd.name].view(local, *kd.ep) for kd in _kinds()}
        w["w13_weight_scale"] = w["w13_weight_scale"].view(torch.float8_e4m3fn)
        w["w2_weight_scale"] = w["w2_weight_scale"].view(torch.float8_e4m3fn)
        v = bufs["vec"]
        inter, offset, n_local = INTER_TP * TP, rank * local, local
    hs, hsf = _STOCK_QUANT(bufs["x"], experts.w13_input_scale_quant)
    out = _MOE(routing_logits=bufs["logits"], routing_bias=bufs["bias"], hidden_states=hs,
               hidden_states_scale=hsf.view(torch.float8_e4m3fn).reshape(PROBE_ROWS, -1),
               gemm1_weights=w["w13_weight"], gemm1_weights_scale=w["w13_weight_scale"].view(torch.float8_e4m3fn),
               gemm1_bias=None, gemm1_alpha=v.get("gemm1_alpha"), gemm1_beta=v.get("gemm1_beta"),
               gemm1_clamp_limit=v.get("gemm1_clamp_limit"), gemm2_weights=w["w2_weight"],
               gemm2_weights_scale=w["w2_weight_scale"].view(torch.float8_e4m3fn), gemm2_bias=None,
               output1_scale_scalar=v["g1_scale_c"], output1_scale_gate_scalar=v["g1_alphas"],
               output2_scale_scalar=v["g2_alphas"], num_experts=EXPERTS, top_k=TOP_K, n_group=topk.num_expert_group,
               topk_group=topk.topk_group, intermediate_size=inter, local_expert_offset=offset,
               local_num_experts=n_local, routed_scaling_factor=cfg.routed_scaling_factor,
               routing_method_type=experts.routing_method_type, do_finalize=True, enable_pdl=False,
               activation_type=flashinfer_trtllm.get_activation_type(cfg.activation, is_gated=cfg.is_gated),
               tune_max_num_tokens=1 << (PROBE_ROWS - 1).bit_length())
    return out[0]


def _probe(experts, mlp, bufs) -> str | None:
    """Both forms' outputs into the sum buffer (FP32), outside the AutoTuner's tuning."""
    _need(not AutoTuner.get().is_tuning_mode, "the AutoTuner entered tuning mode")
    bufs["sum"][0].copy_(_probe_call(experts, mlp, bufs, "tp").float())
    bufs["sum"][1].copy_(_probe_call(experts, mlp, bufs, "ep").float())
    return None


def _compare(bufs, stats) -> str | None:
    tp, ep = bufs["sum"][0], bufs["sum"][1]
    if not (torch.isfinite(tp).all().item() and torch.isfinite(ep).all().item()):
        return "the probe's outputs are not finite"
    top, mean = tp.abs().max().item(), tp.abs().mean().item()
    if not top > 0.0:
        return "the TP probe output is zero (the probe tells nothing)"
    d = (ep - tp).abs()
    stats["max"], stats["mean"] = d.max().item() / top, d.mean().item() / mean
    if stats["max"] > BAR_MAX or stats["mean"] > BAR_MEAN:
        return f"probe EP vs TP max {stats['max']:.3e} (bar {BAR_MAX}), mean {stats['mean']:.3e} (bar {BAR_MEAN})"
    return None


def _commit(experts, rank, bufs) -> None:
    """The EP bytes into the old storages (same byte counts: the parameters keep identity and pointers), the vectors
    sliced in place, every EP attribute and its copies set."""
    local = EXPERTS // TP
    a, b = rank * local, rank * local + local
    for kd in _kinds():
        p = _param(experts, kd.name)
        flat = p.data.view(torch.uint8).reshape(-1)
        flat.copy_(bufs["final"][kd.name].reshape(-1))
        view = flat.view(torch.float8_e4m3fn) if kd.scale else flat
        p.data = view.view(local, *kd.ep)
    for _name, p in _unique(_vectors(experts)):  # each Parameter object once
        if a:
            p.data[:local].copy_(p.data[a:b].clone())
        p.data = p.data[:local]
    i_ep = INTER_TP * TP
    experts.moe_ep_size, experts.moe_ep_rank, experts.moe_tp_size, experts.moe_tp_rank = TP, rank, 1, 0
    experts.num_local_experts, experts._num_local_routed, experts._expert_storage_rank = local, local, rank
    experts.intermediate_size_per_partition = i_ep
    experts.moe_runner_config.num_local_experts = local
    experts.moe_runner_config.intermediate_size_per_partition = i_ep
    disp = experts.dispatcher
    disp.moe_ep_size, disp.moe_ep_rank, disp.num_local_experts, disp.num_local_routed_experts = TP, rank, local, local


def _settle(lid, why, after_commit) -> None:
    """A refusal or a voted miss: OFF on every rank before any commit; a raise on every rank after one."""
    if _ST["mode"] == "EP" or after_commit:
        _log.warning("pfep: layer %s cannot follow after %d layers committed EP (%s): a mixed EP / TP engine cannot "
                     "serve (the king's moe_prepared_route keeps one shape key for every layer); stopping the boot",
                     lid, _ST["ep"], why)
        raise RuntimeError(f"r59 EP: layer {lid} cannot follow the {_ST['ep']} EP layers: {why}")
    if _ST["mode"] is None:
        _ST.update(mode="OFF", why=f"layer {lid}: {why}")
        _log.warning("pfep: OFF on every rank from layer %s (the king's TP layout everywhere, nothing changed): %s", lid,
                     why)


def _release(lid, what) -> None:
    """Step 6 of the module docstring: called with every reshard buffer and view dropped (the caller's bufs, the
    rounds' send / recv / parameter views live in _rounds' frame only); the committed parameters are views of their
    OLD storages (live). Eager: prepare never runs inside a capture (refused before any allocation), checked again."""
    _PCACHE.clear()
    _ST["maps"] = None
    if torch.cuda.is_current_stream_capturing():
        return
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    _log.warning("pfep: transient memory released after layer %s (%s; torch.cuda.empty_cache outside capture: reserved "
                 "%.2f -> %.2f GiB)", lid, what, before / 2 ** 30, torch.cuda.memory_reserved() / 2 ** 30)


def _rounds(experts, kinds, bufs, maps, group, fail):
    """Step 2 of the module docstring: every round's all_to_all and local assembly, then the EP forward of the round's
    experts (the round's views die with this frame)."""
    local = EXPERTS // TP
    for r0 in range(0, local, ROUND):
        c = min(ROUND, local - r0)
        for kd in kinds:
            send = bufs["send"][:TP * c * kd.u_tp].view(TP, c, kd.u_tp)
            recv = bufs["recv"][:TP * c * kd.u_tp].view(TP, c, kd.u_tp)
            old = _param(experts, kd.name).data.view(torch.uint8).view(TP, local, kd.u_tp)
            fail = _local(fail, lambda: _gather_send(send, old, r0, c))
            torch.distributed.all_to_all_single(recv.view(-1), send.view(-1), group=group.device_group)
            fail = _local(fail, lambda: _assemble(kd, recv, c, maps, bufs, r0 == 0))
        fail = _local(fail, lambda: _forward_chunk(kinds, r0, c, bufs, r0 == 0))
    return fail


def prepare(layer) -> None:
    """The layer's routed experts resharded TP -> EP-inside-TP on the PD decode engine (module docstring)."""
    if not EP:
        return
    if id(layer) in _LAYERS:
        return  # prepared once (a second impl's prepare of the same node): nothing, no collective
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, deepseek_v2.DeepseekV2MoE):
        _LAYERS[id(layer)] = "dense"
        return
    experts = mlp.experts
    lid = getattr(layer, "layer_id", None)
    _LAYERS[id(layer)] = "OFF"
    if _ST["mode"] == "OFF":
        return
    why = _uniform_refusal(layer, experts)
    if why is not None:
        if why[0] == "inactive" and _ST["mode"] is None:
            _ST.update(mode="OFF", why=why[1])
            _say("inactive", f"pfep: inactive ({why[1]}): the king's TP layout")
            return
        _settle(lid, why[1], False)
        return
    t0 = time.perf_counter()
    group = get_tp_group()
    rank = group.rank_in_group
    _ST["rank"] = rank
    dev = experts.w13_weight.device
    kinds, local = _kinds(), EXPERTS // TP
    tile = _ST["mode"] is None  # the first target layer (rank-uniform: the mode changes only after a vote)
    bufs, stats = None, {}
    with torch.no_grad():
        fail = _local(None, lambda: _check_form(layer, experts, group))
        maps = None
        if fail is None:
            try:
                maps = _maps(dev)
            except Exception as exc:  # noqa: BLE001 - the maps' own work: this rank's V0 miss
                fail = f"maps: {type(exc).__name__}: {exc}"[:300]
        if fail is None:
            try:
                bufs = _alloc(experts, mlp, rank, dev, tile)
            except Exception as exc:  # noqa: BLE001 - an out-of-memory error here: this rank's V0 miss
                bufs, fail = None, f"alloc: {type(exc).__name__}: {exc}"[:300]
        votes = group.all_gather_object(fail)  # V0
        bad = [(r, v) for r, v in enumerate(votes) if v is not None]
        if bad:
            bufs = maps = None
            _settle(lid, f"V0 {bad}", False)
            _release(lid, "EP OFF")
            return
        fail = _rounds(experts, kinds, bufs, maps, group, fail)
        maps = None
        if tile:
            fail = _local(fail, lambda: _tile_check(experts, mlp, bufs, stats))
        fail = _local(fail, lambda: _probe(experts, mlp, bufs))
        torch.distributed.all_reduce(bufs["sum"], group=group.device_group)
        fail = _local(fail, lambda: _compare(bufs, stats))
        votes = group.all_gather_object((fail, stats.get("max"), stats.get("mean"), stats.get("tiles"),
                                         (stats.get("pins") or (None, None))[1]))  # V1
        bad = [(r, v[0]) for r, v in enumerate(votes) if v[0] is not None]
        follow = False
        if tile:  # rank-uniform: from the votes alone (every rank's table fingerprint)
            line, odd, follow = _tile_decision(votes)
            _log.warning("%s", line)
            bad += odd
            pfp = [v[4] for v in votes]
            if not bad and any(x != pfp[0] for x in pfp):  # the PINS FOLLOW: the same king PINS on every rank
                pm = max(pfp, key=pfp.count)
                pr = pfp.index(pm)
                bad += [(r, f"pins: rank {r}'s king PINS ({x}) differ from rank {pr}'s ({pm})")
                        for r, x in enumerate(pfp) if x != pm]
        if bad:
            bufs = None
            _settle(lid, f"V1 {bad}", False)
            _release(lid, "EP OFF")
            return
        try:
            _commit(experts, rank, bufs)
        except Exception as exc:  # the vote passed everywhere: a failed local commit leaves this rank inconsistent
            raise RuntimeError(f"r59 EP: layer {lid}'s commit failed after a passing vote: {exc}") from exc
        if follow:  # every rank: the same table (V1), the king's module checked before V1 (stats["king"])
            king, king_tile = stats["king"]
            _ST.update(follow=(king, _following(king_tile, dict(zip(range(MIN_ROWS, MAX_ROWS + 1), stats["table"])))),
                       follow_runs=votes[0][3][1])
        if tile and stats.get("pins") and stats["pins"][0] is not None:  # every rank: the same PINS (V1)
            _ST.update(pins=stats["pins"][0], pins_line=stats["pins"][1])
            _log.warning("pfep: pins follow: the king's decode MoE tactic pins (%s) stand down inside every EP layer's "
                         "call: those row counts take the tactic -1 path the tile check validated", stats["pins"][1])
    bufs = None
    _LAYERS[id(layer)] = "EP"
    ms = (time.perf_counter() - t0) * 1000.0
    _ST.update(mode="EP", ep=_ST["ep"] + 1, ms=_ST["ms"] + ms)
    if _ST["ep"] == 1:
        _log.warning("pfep: prefill MoE expert-parallel inside TP%d: rank %d holds experts [%d, %d) whole (intermediate "
                     "%d), the king's moe_prepared_route reads that range at its prepare", TP, rank, rank * local,
                     rank * local + local, INTER_TP * TP)
    mx = max(v[1] for v in votes)
    mn = max(v[2] for v in votes)
    _log.warning("pfep: layer %s ON (experts [%d, %d), probe max %.2e mean %.2e on %d rows; %.0f ms, %d layers %.0f ms)",
                 lid, rank * local, rank * local + local, mx, mn, PROBE_ROWS, ms, _ST["ep"], _ST["ms"])
    if _ST["ep"] == LAST_LAYER - FIRST_LAYER + 1:  # the last target layer committed: nothing more to reshard
        _release(lid, f"{_ST['ep']} layers EP")


@contextlib.contextmanager
def routed(layer, forward_batch):
    """The reshard is the lever; one census line at the first call of a resharded layer. Under the tile follow
    (step 3), for the duration of an EP layer's call, the king's moe_prepared_route global default_tile is ours (its
    _prepare reads it at a shape's first call, inside that call: this route is entered before that route's), the one
    found at entry back after it, also when the call raises. Under the PINS FOLLOW the same for its global PINS ({}:
    _replaced_with_tactic reads it at a shape's first call, inside that call)."""
    ep = _LAYERS.get(id(layer)) == "EP"
    if not _ST["served"] and ep:
        _ST["served"] = True
        _log.warning("pfep: served (layer %s's first call after the reshard; %d layers EP)",
                     getattr(layer, "layer_id", "?"), _ST["ep"])
    follow = _ST["follow"] if ep else None
    pins_king = _ST["pins"] if ep else None
    if follow is None and pins_king is None:
        yield
        return
    found = found_pins = stand_in = None
    swapped_tile = swapped_pins = False
    try:  # r59j: every install inside the try; the finally restores exactly what was installed
        if follow is not None:
            found = follow[0].default_tile
            follow[0].default_tile = follow[1]
            swapped_tile = True
        if pins_king is not None:  # the PINS FOLLOW (module docstring): the king's pins stand down for this call
            found_pins = pins_king.PINS
            stand_in = _AskedPins()
            pins_king.PINS = stand_in
            swapped_pins = True
        yield
    finally:
        if swapped_pins:
            pins_king.PINS = found_pins
            _pins_evidence(stand_in, found_pins)
        if swapped_tile:
            follow[0].default_tile = found


class _AskedPins(dict):
    """The PINS FOLLOW's stand-in for the king's PINS inside an EP layer call (r59j): an EMPTY dict (no row count is
    pinned) whose get() also records the row counts the king asked for -- the evidence that a pinned row count took the
    AutoTuner's tactic inside an EP call (census "pfep: pins follow: asked for <rows> rows ...")."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def get(self, key, default=None):
        self.asked.append(key)
        return default


_PSEEN: set = set()  # the pinned row counts the king has asked for inside EP layer calls (census once each)


def _pins_evidence(stand_in, pins) -> None:
    """Census of the PINS FOLLOW's effect: each pinned row count the king asked for inside an EP layer call (once a row
    count), and one line when all of them have been."""
    for rows in stand_in.asked:
        if rows in pins and rows not in _PSEEN:
            _PSEEN.add(rows)
            _log.warning("pfep: pins follow: asked for %s rows inside an EP layer call: no pin (the AutoTuner's tactic, the "
                         "tile check's path)", rows)
            if len(_PSEEN) == len(pins):
                _log.warning("pfep: pins follow: all %d pinned row counts asked inside EP layer calls %s", len(pins),
                             sorted(_PSEEN))
