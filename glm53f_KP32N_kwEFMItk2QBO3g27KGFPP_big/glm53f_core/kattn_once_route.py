"""kattn_once_route.py: r60 KATTN2 (-kset: KATTN-ONCE, set-exact checks) -- ONE top-k selection per K' layer call
instead of the king's two. SET-EXACT, with voted runtime checks per path kind: the attention kernel gets, row for row,
the same SET of K' picks as the king's (the same -1 slots, the same lengths); the ORDER of the picks inside a row is as
arbitrary as the king's own. Why not bytes (KATTN-ONCE, -kattn1): the ragged v2 top-k writes its picks "in no
particular order" (kernels/ops/attention/dsv4/topk.py:108; SGLANG_TOPK_V2_MODE = 1, environ.py:1621, mode 2 is the
index-ordered one), so the KING's K' selection run twice on the same logits differs in byte order (box11 042 boot,
11-20: "K' reruns byte-identical: NO") and a byte-exact check vs the king can never pass (-kattn1 went OFF at its first
voted check on every prefill rank). Comparing sorted rows (plus the -1 mask in place and the lengths exactly) accepts
exactly the king's own run-to-run freedom and nothing more: a different pick, a moved -1 slot or a length change is
still a miss -> OFF on every rank. The census reports bytes AND sets separately.
Python only: no native unit, no JIT of its own, no engine patch or flag.

The king's dsa_kattn_route (v32, dbe67160; pinned below) serves eager EXTEND calls of the 8 K' layers (the 11 sparse
layers minus SKIP {19, 23, 31}; the draft layer kept, DRAFT False; decode-side modes kept, DECODE False; graph / MHA /
unfused / DCP / hisparse calls kept: dsa_kattn_route.py:53-54, :59-74). For such a call its routed() installs the
INSTANCE attribute ``indexer._topk_from_kpool_logits = lambda *args, **kw: _select(args, kw)`` (:240). ``_select`` is
a module global of dsa_kattn_route looked up when the lambda RUNS (it is not a local of routed()), so rebinding
``dsa_kattn_route._select`` for the layer call swaps it. The king's _select (:93-103) runs stock's 2048-key selection
(:96) AND the K'=1024 one on the same keywords (:97), keeps the K' picks in _CALL["parts"] and returns STOCK's picks;
_attention (:180-219) then hands the kernel the K' table: block_tables, sparse_mla_top_k, sparse_mla_top_k_lens all
replaced (:218-219).

What this component does (NODE "dsa", right AFTER dsa_kattn_route in ACTIVE, so its routed() runs inside kattn's):
for a call kattn serves, ``dsa_kattn_route._select`` is ours: the K' selection ONLY (kattn's own _kprime, same
keywords), appended to kattn's parts exactly as the king does, and returned padded with -1 to stock's width
(TOPK + its tail = 2051 columns, the width idx_split_attn_route._kept_rows checks, :81) and stock's rows. Restored in
a finally (with idx_split_node._why_stock and kattn's _CALL["entry"], both wrapped only to observe).

WHY NOTHING READS STOCK'S PICKS on the paths served (engine 314bce7c = imgsrc 93a5b07; K = the king's package):
  1. The backend (layers/attention/dsa_backend.py): forward_extend -> _forward_trtllm (:2388-2406). Picks-derived:
     _pad_topk_indices (:4291-4292, :4424-4445, a cat), _get_fused_topk_page_table (:4309-4310, identity :1156-1164),
     _pad_trtllm_sparse_page_table (:4330-4332, :1049-1062, a cat to 2052), prepare_trtllm_nope_sparse_metadata
     (:4343-4345; kernels/ops/attention/dsa/transform_index.py:84-108: lengths, and col 0 of EMPTY rows of its OWN
     input written) -> block_tables (:4366), sparse_mla_top_k, sparse_mla_top_k_lens (:4372-4386) -> the kernel
     (:4414-4419). Every other kernel argument (query, kv_cache, workspace, seq_lens, max_seq_len, scales, multi_ctas
     buffer) is metadata, not picks. _resolve/_check_kpool_tail_backend read only None-ness and config
     (dsa/dsa_backend_kpool.py:37-79). DCP's localize (:4333-4342) is excluded by kattn (attn_dcp_size != 1, K :71-73).
     kattn replaces all three (K :218-219).
  2. PAD-ZC (K idx_split_attn_route._padded :28-45): stock's half -> _kept_rows (:69-94; the native table copies the
     values and counts v >= 0, K dsa_pagetable_native.cu:21-41) -> _lengths (:97-103) -> the kernel: kattn's "table"
     record (K :127-144, :196-198) replaces all three. idx_split_node._gather (K :61-62, :115-136): stock's halves
     all-gathered (same bytes: our pad has stock's shape) -> the kernel: kattn's "gathered" record (K :147-158,
     :208-217). Whole rows / no split: kattn stacks its parts (K :208-217).
  3. The excluded branches. "own" (K :199-207) needs a record from idx_nextn_route._exchange (K :161-168), which only
     idx_nextn_route._split_call reaches (K idx_nextn_route.py:134-149, served() :152-185): target-verify (:188-192,
     :205-212) and the draft extend (K idx_draft_route.py) -- never an EXTEND call of a target layer, which is all kattn
     serves (model_executor/forward_mode.py:85-90; K :61-64). So "own elsewhere" (:201-204) cannot occur here.
     "again" (K :183-185) needs a second kernel call in one layer call: _forward_trtllm makes one (:4414); the token
     split calls its own held _STOCK, not the module name (K dsa_split_route.py:14, :153, :176). Both are also caught
     at run time (below).
  4. Leaving the layer: _finish_absorb_output returns the picks only when next_skip_topk (models/deepseek_common/
     attention_forward_methods/forward_mla.py:160-168); indexer_types is all "full" (hf config; configs/
     model_config.py:593-601) so no layer reuses them (models/deepseek_v2.py:2164-2178, :3298, :3306). The MTP seed
     (layers/attention/index_topk_share.py:36-47, :62-69, :94-114) is written only from a seed buffer that the DRAFT
     extend sets (speculative/eagle_worker_v2.py:1048-1057, :1199-1204; forward_mha.py:107-120 is the MHA path kattn
     excludes); calls that share top-k (reuse_dsa_topk_indices / dsa_seed_topk_capture, K idx_split_attn_route.py:63-66)
     take the king's path here anyway. The indexer top-k capturer (state_capturer/indexer_topk.py:65-77, the
     --enable-return-indexer-topk feature) -> the king's path whenever one is installed.
  5. Stock's selection leaves nothing K' relies on (kpool_fp8_index.py:764-854): its output is a fresh tensor
     (_alloc_kpool_topk_out :638-643, the fused path :823-846, unfused :857-916); page_table / offsets / seq_lens /
     row_starts / out_rows / topk_v2_plan are read only (no plan is passed on these paths: dsa_indexer_kpool.py:1144-
     1159). The ragged v2 kernel (SGLANG_OPT_USE_TOPK_V2 True, environ.py:1616) writes the LOGITS in place -- only the
     <= 3 columns AHEAD of each row's window (kernels/ops/attention/dsv4/topk.py:202-205, :443) -- outside both
     selections' windows, masked by each kernel itself. That is the one claim the source cannot prove (topk_v2.cuh is
     not in the image snapshot): the CHECKS below test it byte for byte (ONCE's K' on a pristine copy of the logits vs
     the king's K' run after stock's selection).
  6. Widths: every stock implementation emits topk + (pool - 1 if seq_lens else 0) columns (kpool_fp8_index.py:662,
     :700; kpool_topk_transform.py:56; unfused :893-908), so stock's width = K' width + (TOPK - K'), and both emit the
     same rows (out_rows). The non-plan ragged path writes picks into a TOPK + POOL - 1 buffer (dsa_indexer_kpool.py:
     1284-1288, :1504) and the multi-chunk path into a chunk-width one (:1166-1175): unchanged by the pad.

THE RUNTIME CHECKS (rank-uniform triggers only; votes = all_gather_object on the attention TP group's CPU group):
  * prepare (the first DSA node): the source of dsa_kattn_route / idx_split_attn_route / idx_split_node pinned by
    sha256 (package- and materialization-normalized: the validator's rewritten native and package import lines put
    back), kattn's literals and _STOCK_SELECT checked, voted: any miss -> OFF on every rank.
  * The first CHECKS calls of each path kind ("table" = PAD-ZC, "gathered" = split + all-gather, "rows" = whole rows)
    serve the KING's result (stock + K') and, before it, ONCE's K' on a stride-exact copy of the logits; at the kernel
    the K' table / top-k width / lengths that ONCE's picks give (kattn's own _stack / _lay / _width, the record's
    layout) are compared as exact bytes with the king's final kwargs. Mismatch, exception, a pass-through branch or a
    second kernel call on any rank -> OFF on every rank for the process (census). Logits over CHECK_BYTES or a 0-row
    call -> the check is not counted.
  * Re-checks: every RECHECK_EVERY-th serve-mode call of a kind is a voted check again (later real traffic: deep
    prefixes, multi-request batches with unaligned row starts), the king's result served.
  * RANK-UNIFORM PATHS: a layer call's path (king / check / serve) is decided from voted state only (the switches,
    the vote-driven counts, the call's kind -- a host-side fact the TP pair shares). A rank-local miss (the boot
    probe, a served call whose kernel call does not take the K' table) is only RECORDED ("kattn-once: miss on this
    rank ..."); it reaches every rank through the next voted check (no served call can precede the first one; the
    re-checks bound the rest), which turns ONCE OFF everywhere at once. No collective one rank alone would call.
BOOT TIMING (the timed window's stderr is not kept, so the effect is logged at prepare): at the first eager prepare of
a K' layer on a rank that is not a PD decode server, _boot times the king's path (stock + K') against ONCE's (K' + pad)
on fresh synthetic tensors shaped as a PAD-ZC half hands them to _topk_from_kpool_logits (BOOT_ROWS x BOOT_CONTEXT/4),
then probes the byte-determinism of the K' selection and the king's order (stock, then K') against ONCE's on a
two-request layout with an unaligned row start; a NO = a rank-local miss: OFF on every rank at the first voted check,
before any call is served.
Census (WARNING; this logger keeps WARNING under the arena's log_level error): "kattn-once: on ...",
"kattn-once: boot timing 8192 rows: king X us vs once Y us (saves Z us a layer x 8 K' layers; ...)" (or "... boot
timing unavailable (...)"), "kattn-once: check i/N <kind> ... equal on every rank", "kattn-once: served xN",
"kattn-once: re-check xN ...", "kattn-once: miss on this rank (...)", "kattn-once: OFF (...)"."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import re

import flashinfer.decode
import torch
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

from glm53f_core import dsa_kattn_route, idx_split_attn_route, idx_split_node

NODE = "dsa"
ONCE = True       # False = the king's dsa_kattn_route exactly (this component inert)
TABLE = True      # PAD-ZC calls (idx_split_attn_route._padded: this rank's half into the kept table)
GATHERED = True   # indexer-split calls with their all-gather (idx_split_node._splitting)
ROWS = True       # whole-row calls (no split, or idx_split_node._why_stock's whole rows)
CHECKS = 2        # voted checks per path kind before ONCE serves it
RECHECK_EVERY = 256   # then every Nth serve-mode call of a kind is a voted re-check too (0 = never)
CHECK_BYTES = 1 << 30   # a check copies the layer's logits for ONCE's K'; larger -> that check is not counted
BOOT_TIMING = True      # one synthetic timing at the first eager prepare of a K' layer on a prefill-capable rank
BOOT_ROWS = 8192        # this rank's PAD-ZC half of a 16K chunk
BOOT_CONTEXT = 65536    # tokens through the chunk's last row: 16384 pooled keys (a mid-depth cold chunk)
BOOT_WARM, BOOT_REPS = 2, 7
PROBE_ROWS, PROBE_KEYS = 256, 3000   # the boot probe: two requests, the second's rows starting unaligned
KINDS = ("table", "gathered", "rows")
RELEASE_PKG = "glm53f_core"
KING_LITERALS = (2048, 4, 1024, False, False, frozenset({19, 23, 31}))   # TOPK, POOL, K_ATTN, DECODE, DRAFT, SKIP
PINS = {  # sha256 of _canonical(source) of the v32 king's files (file sha256 8e722e50 / c9db9694 / 86a86e63); the
    # apply tool checks the base's text against these
    "dsa_kattn_route": "304a62fbee11bf2cfeb4ce22e92b6665465f572029c1cea3124bbcfdb3c354d4",
    "idx_split_attn_route": "19766b2c0e80f42735eb43f06e09741008793f3ad1657ee202c759348d2a49a2",
    "idx_split_node": "0c8c02fb403d5d59934af88e5c4f1b9fe11276036019e669bf7494e97c5a97de",
}
_K, _SPLIT, _NODE = dsa_kattn_route, idx_split_attn_route, idx_split_node
_NATIVE = r"^([ \t]*)import cacheon_c_\w+ as (\w+)[ \t]*$"   # (re.sub patterns, multiline)
_FROM = r"^([ \t]*)from cacheon_c_[0-9A-Za-z]+\."
_ST: dict = {"pinned": False, "on": False, "local": None, "passed": {k: 0 for k in KINDS}, "served": 0, "rank": 0,
             "group": None, "select": None, "why": None, "capturer": None, "capture_mode": None, "said": set(),
             "timed": False, "kprepared": 0, "serves": {k: 0 for k in KINDS}, "rechecks": 0, "order_only": 0}
_NODES: dict = {}
_L: dict = {}   # the live layer call (empty outside routed)
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


def _say(tag: str, msg: str) -> None:
    if tag not in _ST["said"]:
        _ST["said"].add(tag)
        _log.warning("%s", msg)


def _canonical(text: str, pkg: str = RELEASE_PKG) -> str:
    """The source as the bundle ships it: the validator's materialized import lines put back (engine_tree
    _rewrite_python: ``import dsa_pagetable_native`` -> ``import cacheon_c_<d>__dsa_pagetable_native_<h> as
    dsa_pagetable_native``; ``from glm53f_core import x`` -> ``from cacheon_c_<d>.glm53f_core import x``), the package
    name normalized, trailing newlines dropped."""
    text = re.sub(_NATIVE, r"\1import \2", text, flags=re.M)
    text = re.sub(_FROM, r"\1from ", text, flags=re.M)
    if pkg != RELEASE_PKG:
        text = text.replace(f"from {pkg} import ", f"from {RELEASE_PKG} import ")
    return text.rstrip("\n")


def source_sha(module) -> str:
    pkg = module.__name__.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in module.__name__ else RELEASE_PKG
    try:
        text = inspect.getsource(module)
    except OSError:
        spec = getattr(module, "__spec__", None)
        text = spec.loader.get_source(module.__name__) if spec is not None and spec.loader is not None else None
        if text is None:
            raise
    return hashlib.sha256(_canonical(text, pkg).encode()).hexdigest()


def _group():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    from sglang.srt.distributed.parallel_state import get_attn_tp_group

    return get_attn_tp_group()


def _vote(obj) -> list:
    group = _ST["group"]
    return group.all_gather_object(obj) if group is not None and group.world_size > 1 else [obj]


def _engine_fns():
    from sglang.srt.model_executor.runner import get_is_capture_mode

    try:
        from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
    except ImportError:
        get_global_indexer_capturer = None
    return get_is_capture_mode, get_global_indexer_capturer


def _drift():
    """None if this rank may serve ONCE (pins, literals, the king's functions in place), else why not."""
    try:
        for module, name in ((_K, "dsa_kattn_route"), (_SPLIT, "idx_split_attn_route"), (_NODE, "idx_split_node")):
            got = source_sha(module)
            if got != PINS[name]:
                return f"{name} is not the pinned v32 text ({got[:12]} != {PINS[name][:12]})"
        lits = (_K.TOPK, _K.POOL, _K.K_ATTN, _K.DECODE, _K.DRAFT, frozenset(_K.SKIP))
        if lits != KING_LITERALS:
            return f"dsa_kattn_route's literals are {lits}"
        if _K._STOCK_SELECT is not IndexerKPool._topk_from_kpool_logits:
            return "dsa_kattn_route._STOCK_SELECT is not IndexerKPool._topk_from_kpool_logits"
        if not (callable(_K._select) and callable(_NODE._why_stock) and callable(_K._kprime)):
            return "dsa_kattn_route._select / _kprime or idx_split_node._why_stock missing"
        _ST["capture_mode"], _ST["capturer"] = _engine_fns()
        _ST["select"], _ST["why"] = _K._select, _NODE._why_stock
        return None
    except Exception as e:  # noqa: BLE001  (a failed check is a refusal, never a crash)
        return f"the pin check raised {type(e).__name__}: {e}"


def prepare(node) -> None:
    if not _ST["pinned"]:
        _ST["pinned"] = True
        if not ONCE:
            return
        why = _drift()
        try:
            group = _group()
            _ST["group"], _ST["rank"] = group, (group.rank_in_group if group is not None else 0)
        except Exception as e:  # noqa: BLE001
            _ST["group"], why = None, why or f"the attention TP group: {type(e).__name__}: {e}"
        bad = [(r, w) for r, w in enumerate(_vote(why)) if w is not None]
        if bad:
            _log.warning("kattn-once: OFF (rank %d: %s) -- the king's two selections on every rank", *bad[0])
            return
        _ST["on"] = True
        _log.warning("kattn-once: on (the %d K' layers' eager extend calls: the K'=%d selection only, returned padded "
                     "with -1 to stock's width; paths table=%s gathered=%s rows=%s; %d voted SET-EXACT checks a path "
                     "first (-kset); rank %d of %d)", 11 - len(_K.SKIP), _K.K_ATTN, TABLE, GATHERED, ROWS, CHECKS, _ST["rank"],
                     _ST["group"].world_size if _ST["group"] is not None else 1)
    _NODES[id(node)] = getattr(node, "indexer", None)
    if _ST["on"] and BOOT_TIMING and not _ST["timed"]:
        _boot(node)


# ------------------------------------------------------------------------------------------- the boot timing
def _cuda_ok() -> bool:
    return torch.cuda.is_available()


def _timer():
    """(device, an event maker, synchronize) on this rank's CUDA device."""
    return (torch.device("cuda", torch.cuda.current_device()), lambda: torch.cuda.Event(enable_timing=True),
            torch.cuda.synchronize)


def _disagg_mode():
    from sglang.srt.runtime_context import get_disagg

    return get_disagg().disaggregation_mode


def _boot(node) -> None:
    """At the first EAGER prepare of a K' layer (kattn's _NODES entry kept None) on a rank that is not a PD decode
    server: one synthetic timing of the selection as a PAD-ZC half hands it to _topk_from_kpool_logits (the plan
    path's RAGGED keywords, dsa_indexer_kpool.py:1144-1159, :1229-1256: fp32 clean logits, pool / seq lengths,
    zero row starts and offsets of one request, out_rows = the rows, no page table, no plan) on FRESH tensors only:
    (a) the king's path = stock's selection + the K' one (dsa_kattn_route.py:96-97), (b) ONCE's = the K' one + the -1
    pad. CUDA events, the median of BOOT_REPS after BOOT_WARM warm calls each, alternated. Census at prepare (the
    timed window's stderr is not kept): "kattn-once: boot timing ..." or "kattn-once: boot timing unavailable (...)".
    Nothing a real call reads is touched (no pool, page table, kattn or split state; the indexer's config only)."""
    held = _K._NODES.get(id(node))
    if held is None or held[2] is not None:
        return   # not a K' layer: the next K' layer's prepare times it
    _ST["kprepared"] += 1
    last = _ST["kprepared"] >= 11 - len(_K.SKIP)
    try:
        if _disagg_mode() == "decode":
            _ST["timed"] = True   # a PD decode server: kattn serves no call here
            return
        if not _cuda_ok():
            _ST["timed"] = True
            _log.warning("kattn-once: boot timing unavailable (no CUDA device)")
            return
        if torch.cuda.is_current_stream_capturing():
            if last:
                _ST["timed"] = True
                _log.warning("kattn-once: boot timing unavailable (every K' layer was prepared under a stream capture)")
            return
        _ST["timed"] = True
        king, once, cols, rerun_bytes, rerun, after_bytes, after = _boot_measure(held[0], held[1])
        _log.warning("kattn-once: boot timing %d rows: king %.0f us vs once %.0f us (saves %.0f us a layer x %d K' "
                     "layers; %d x %d pooled keys = this rank's half of the 16K chunk ending a %d-token context, "
                     "median of %d, rank %d; K' reruns byte-identical: %s, as sets: %s; ONCE's K' == the king's "
                     "after stock (2 requests, unaligned row starts): bytes %s, as sets %s)", BOOT_ROWS, king, once,
                     king - once, 11 - len(_K.SKIP), BOOT_ROWS, cols, BOOT_CONTEXT, BOOT_REPS, _ST["rank"],
                     "yes" if rerun_bytes else "NO", "yes" if rerun else "NO", "yes" if after_bytes else "NO",
                     "yes" if after else "NO")
        if not (rerun and after):
            _miss_here("the boot probe: " + ("the K' selection is not deterministic even as a set" if not rerun else
                                             "stock's selection changes the K' picks it leaves for kattn's second one"))
    except Exception as e:  # noqa: BLE001  (a timing never fails a prepare)
        _ST["timed"] = True
        _log.warning("kattn-once: boot timing unavailable (%s: %s)", type(e).__name__, str(e)[:200])


def _boot_measure(indexer, k: int):
    device, event, sync = _timer()
    pool = _K.POOL
    rows, cols = BOOT_ROWS, BOOT_CONTEXT // pool
    seq = torch.arange(BOOT_CONTEXT - rows + 1, BOOT_CONTEXT + 1, dtype=torch.int32, device=device)
    plen = torch.div(seq, pool, rounding_mode="floor").to(torch.int32)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    logits = torch.randn((rows, cols), generator=gen, dtype=torch.float32, device=device)
    logits.masked_fill_(torch.arange(cols, device=device).unsqueeze(0) >= plen.unsqueeze(1), float("-inf"))
    zeros = torch.zeros((rows,), dtype=torch.int32, device=device)
    kw = dict(logits=logits, pool_lens=plen, seq_lens=seq, page_table=None, topk_offsets=zeros, row_starts=zeros,
              out_rows=rows, page_table_row_index=None)

    def king():
        _K._STOCK_SELECT(indexer, **kw)
        _K._kprime(indexer, k, **kw)

    def once():
        _pad(_K._kprime(indexer, k, **kw), k)

    for _ in range(BOOT_WARM):
        king()
        once()
    sync()
    times = {king: [], once: []}
    for _ in range(BOOT_REPS):
        for fn in (king, once):
            a, b = event(), event()
            a.record()
            fn()
            b.record()
            b.synchronize()
            times[fn].append(a.elapsed_time(b) * 1000.0)
    r1, r2 = _K._kprime(indexer, k, **kw), _K._kprime(indexer, k, **kw)
    rerun_bytes, rerun = torch.equal(r1, r2), _same_set(r1, r2)
    del logits, kw, r1, r2
    after_bytes, after = _boot_probe(indexer, k, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def median(v):
        return sorted(v)[len(v) // 2]
    return median(times[king]), median(times[once]), cols, rerun_bytes, rerun, after_bytes, after


def _boot_probe(indexer, k: int, device) -> bool:
    """The in-place claim on rows the first checks may not see: two requests of PROBE_ROWS / 2 rows, the second's
    window starting at an UNALIGNED column (the ragged v2 kernel writes the <= 3 columns ahead of it in place,
    topk.py:202-205), the RAGGED offsets = the row starts (dsa_indexer_kpool.py:1229-1232); the king's order (stock,
    then K' on the same logits) vs ONCE's (K' on an identical copy), exact bytes. Fresh tensors only."""
    pool, rows, keys = _K.POOL, PROBE_ROWS, PROBE_KEYS
    half = rows // 2
    start1 = keys + 1
    cols = (start1 + keys + 3) // 4 * 4
    j = torch.arange(rows, device=device)
    starts = torch.where(j < half, torch.zeros_like(j), torch.full_like(j, start1)).to(torch.int32)
    plen = (keys - half + 1 + j % half).to(torch.int32)
    seq = (plen * pool + 3).to(torch.int32)
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    logits = torch.randn((rows, cols), generator=gen, dtype=torch.float32, device=device)
    col = torch.arange(cols, device=device).unsqueeze(0)
    logits.masked_fill_((col < starts.unsqueeze(1)) | (col >= (starts + plen).unsqueeze(1)), float("-inf"))
    copy = logits.clone()
    kw = dict(pool_lens=plen, seq_lens=seq, page_table=None, topk_offsets=starts, row_starts=starts, out_rows=rows,
              page_table_row_index=None)
    _K._STOCK_SELECT(indexer, logits=logits, **kw)
    king = _K._kprime(indexer, k, logits=logits, **kw)
    once = _K._kprime(indexer, k, logits=copy, **kw)
    return tuple(king.shape) == tuple(once.shape) and torch.equal(king, once), _same_set(king, once)


def _ineligible(node, forward_batch):
    """None if ONCE may take this layer call (every fact is host-side and rank-uniform), else why not."""
    kc = _K._CALL
    indexer = _NODES.get(id(node))
    if indexer is None or kc.get("indexer") is not indexer:
        return "a call kattn's state does not hold"
    if kc.get("parts") != [] or kc.get("record") is not None or kc.get("armed"):
        return "kattn's call state not fresh"
    if _K._select is not _ST["select"] or _NODE._why_stock is not _ST["why"]:
        return "another route holds dsa_kattn_route._select or idx_split_node._why_stock"
    if kc.get("entry") is not flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla:
        return "kattn's held kernel entry is not the live name"
    mode = forward_batch.forward_mode
    if not mode.is_extend_or_draft_extend_or_mixed() or mode.is_mixed():
        return "a non-EXTEND call"
    if torch.cuda.is_current_stream_capturing() or _ST["capture_mode"]():
        return "a capture"
    if _SPLIT._shares_topk(forward_batch):
        return "a call that shares its top-k"
    if _ST["capturer"] is not None and _ST["capturer"]() is not None:
        return "an indexer top-k capturer"
    return None


@contextlib.contextmanager
def routed(node, forward_batch):
    if not _ST["on"] or not _K._CALL or _L:
        yield
        return
    if _ineligible(node, forward_batch) is not None:
        yield
        return
    kc = _K._CALL
    _L.update(real=kc["entry"], decided=False, why=None, kind=None, mode=None, kinds=set(), once=[], served=0,
              width=None, miss=None, incomplete=None, compared=False, calls=0, rows=0)
    held_select, held_why = _K._select, _NODE._why_stock
    _K._select = _select
    _NODE._why_stock = _why
    kc["entry"] = _entry
    try:
        yield
        _settle()
    finally:
        _K._select = held_select
        _NODE._why_stock = held_why
        if _K._CALL.get("entry") is _entry:
            _K._CALL["entry"] = _L["real"]
        _L.clear()


def _why(kw: dict, n: int):
    """idx_split_node._why_stock, observed: its verdict tells this layer call's path kind."""
    why = _ST["why"](kw, n)
    if _L:
        _L["decided"], _L["why"] = True, why
    return why


def _kind() -> str:
    if not _L["decided"] or _L["why"] is not None:
        return "rows"
    return "table" if _SPLIT._CALL.get("skip") else "gathered"


def _decide(kind: str) -> str:
    """The layer call's path, from VOTED state only (rank-uniform: the switches, the vote-driven counts and the
    kind, a host-side fact): a rank-local miss never changes it (it rides the next voted check instead)."""
    if not {"table": TABLE, "gathered": GATHERED, "rows": ROWS}[kind]:
        return "king"
    if _ST["passed"][kind] < CHECKS:
        return "check"
    _ST["serves"][kind] += 1
    return "check" if RECHECK_EVERY and _ST["serves"][kind] % RECHECK_EVERY == 0 else "serve"


def _same_set(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal as the attention kernel reads them: same shape / dtype / device and, row for row (the last dimension), the
    same multiset of values with the -1 (empty) slots at the same places; 1-D tensors (lengths) must be byte-equal.
    Byte-equal tensors pass at once; _ST["order_only"] counts the compares that passed only as sets."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype or a.device != b.device:
        return False
    if torch.equal(a, b):
        return True
    if a.dim() < 2 or not torch.equal(a < 0, b < 0):
        return False
    if not torch.equal(torch.sort(a, dim=-1).values, torch.sort(b, dim=-1).values):
        return False
    _ST["order_only"] += 1
    return True


def _pad(part: torch.Tensor, k: int) -> torch.Tensor:
    """The K' picks at stock's width (its tail is K's: width + TOPK - K'), -1 beyond; never a view of the part."""
    w = part.shape[1]
    out = part.new_empty((part.shape[0], w + _K.TOPK - k))
    out[:, :w].copy_(part)
    out[:, w:].fill_(-1)
    return out


def _arm(kc: dict) -> None:
    if not kc["armed"]:
        _K._require(flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is _L["real"],
                    "the attention kernel's entry changed inside the layer's call before its picks were ranked")
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla = _K._attention
        kc["armed"] = True


def _select(args: tuple, kw: dict) -> torch.Tensor:
    """dsa_kattn_route._select for a call ONCE takes: serve = the K' selection only (stock's picks never ranked),
    check = the king's two selections served plus ONCE's on a copy, king = dsa_kattn_route._select's statements."""
    kc = _K._CALL
    _K._require(not args, "IndexerKPool._topk_from_kpool_logits called with positional arguments")
    indexer, k = kc["indexer"], kc["k"]
    kind = _kind()
    if _L["mode"] is None:
        _L["mode"], _L["kind"] = _decide(kind), kind
    _L["kinds"].add(kind)
    if _L["mode"] == "check":
        picks = _check(indexer, k, kw)
    elif _L["mode"] == "serve" and kind == _L["kind"]:
        part = _K._kprime(indexer, k, **kw)
        kc["parts"].append(part)
        picks = _pad(part, k)
        _L["served"] += 1
        _L["width"] = _K._width(part)
    else:
        picks = _K._STOCK_SELECT(indexer, **kw)
        kc["parts"].append(_K._kprime(indexer, k, **kw))
    _arm(kc)
    return picks


def _miss(why: str) -> None:
    if _L.get("miss") is None:
        _L["miss"] = why


def _check(indexer, k: int, kw: dict) -> torch.Tensor:
    kc = _K._CALL
    once = pad = None
    try:
        logits = kw["logits"]
        size = logits.numel() * logits.element_size()
        if size > CHECK_BYTES:
            _L["incomplete"] = _L["incomplete"] or f"{size} bytes of logits (over CHECK_BYTES)"
        else:
            copy = torch.empty_strided(tuple(logits.shape), tuple(logits.stride()), dtype=logits.dtype,
                                       device=logits.device)
            copy.copy_(logits)
            once = _K._kprime(indexer, k, **dict(kw, logits=copy))
            pad = _pad(once, k)
    except Exception as e:  # noqa: BLE001
        once = None
        _miss(f"ONCE's selection raised {type(e).__name__}: {e}")
    picks = _K._STOCK_SELECT(indexer, **kw)
    part = _K._kprime(indexer, k, **kw)
    kc["parts"].append(part)
    _L["once"].append(once)
    _L["width"] = _K._width(part)
    if once is not None:
        try:
            if tuple(pad.shape) != tuple(picks.shape) or pad.dtype != picks.dtype or pad.device != picks.device:
                _miss(f"ONCE's padded picks are {pad.dtype} {tuple(pad.shape)}, stock's {picks.dtype} "
                      f"{tuple(picks.shape)}")
            elif not _same_set(once, part):
                _miss(f"ONCE's K' picks ({tuple(once.shape)}) differ from the king's ({tuple(part.shape)})")
        except Exception as e:  # noqa: BLE001
            _miss(f"the picks compare raised {type(e).__name__}: {e}")
    return picks


def _entry(*args, **kwargs):
    """kattn's held kernel entry for the layer call: the real one, observed."""
    real = _L["real"]
    if _L["mode"] == "check":
        out = real(*args, **kwargs)
        try:
            _compare(args, kwargs)
        except Exception as e:  # noqa: BLE001
            _miss(f"the kernel compare raised {type(e).__name__}: {e}")
        return out
    if _L["served"] and not _kprime_call(args, kwargs):
        _miss_here("a kernel call of a ONCE-served layer call did not take the K' table (width "
                  f"{kwargs.get('sparse_mla_top_k')!r}, want {_L['width']})")
    return real(*args, **kwargs)


def _kprime_call(args: tuple, kwargs: dict) -> bool:
    tables = kwargs.get("block_tables")
    if args or not isinstance(tables, torch.Tensor):
        return False
    if tables.shape[0] == 0:
        return True
    return kwargs.get("sparse_mla_top_k") == _L["width"] and tables.shape[-1] == _L["width"]


def _miss_here(why: str) -> None:
    """A rank-local miss (the boot probe, a served call's pass-through): recorded, NOT acted on here -- every rank's
    path stays the voted one; the next voted check (a pending check, or a re-check every RECHECK_EVERY serve-mode
    calls of a kind) carries it and turns ONCE OFF on every rank at once. No collective of its own."""
    if _ST["local"] is None:
        _ST["local"] = why
        _log.warning("kattn-once: miss on this rank (%s) -- every rank goes OFF at the next voted check", why)


def _compare(args: tuple, kwargs: dict) -> None:
    """The kernel's K' table / top-k width / lengths that ONCE's picks give (kattn's _stack / _lay / _width in the
    record's layout: dsa_kattn_route.py:134-141, :152-157, :209-219) vs the king's final kwargs, exact bytes."""
    _L["calls"] += 1
    if _L["calls"] > 1:
        _miss("a second kernel call in the layer's call (kattn keeps the backend's table there)")
        return
    tables, lens, top = kwargs.get("block_tables"), kwargs.get("sparse_mla_top_k_lens"), kwargs.get("sparse_mla_top_k")
    if args or not isinstance(tables, torch.Tensor) or not isinstance(lens, torch.Tensor):
        _miss("the kernel call's tables / lengths are not tensors passed by keyword")
        return
    rows = tables.shape[0]
    if rows == 0:
        _L["incomplete"] = _L["incomplete"] or "a 0-row kernel call"
        return
    if top != _L["width"] or tables.shape[-1] != _L["width"]:
        _miss(f"the kernel took a table {top} wide, not the K' {_L['width']} (a pass-through branch)")
        return
    if len(_L["kinds"]) != 1:
        _miss(f"one layer call took paths {sorted(_L['kinds'])}")
        return
    once = _L["once"]
    if not once or any(p is None for p in once):
        return   # a miss or an uncounted check is recorded already
    kind, rank = _L["kind"], _ST["rank"]
    if kind == "rows":
        picks = _K._stack([_K._stack(once, None)], rows)
        table, counts = picks.new_empty((rows, _K._width(picks))), picks.new_empty((rows,))
        _K._lay(picks, table, counts)
    elif kind == "table":
        h = rows // 2
        part = _K._stack(once, h)
        table, counts = part.new_empty((rows, _K._width(part))), part.new_empty((rows,))
        for at in (rank * h, (1 - rank) * h):
            _K._lay(part, table[at:at + h], counts[at:at + h])
    else:
        # gathered: the peer's half is the peer's (its own check votes for it); ours re-laid from ONCE's picks
        h = (rows + 1) // 2
        table, counts = tables[:, 0].clone(), lens.clone()
        own = _K._stack([_K._stack(once, None)], h)
        a, b = rank * h, min(rows, rank * h + h)
        if b > a:
            _K._lay(own[:b - a], table[a:b], counts[a:b])
    king = tables[:, 0]
    if (tuple(table.shape) != tuple(king.shape) or table.dtype != king.dtype or top != table.shape[1]
            or counts.dtype != lens.dtype or not _same_set(table, king) or not torch.equal(counts, lens)):
        _miss(f"the kernel's K' table / lengths from ONCE's picks differ from the king's ({kind}, {rows} rows)")
        return
    _L["compared"], _L["rows"] = True, rows


def _settle() -> None:
    if _L["served"]:
        _ST["served"] += 1
        n = _ST["served"]
        if n & (n - 1) == 0:
            _log.warning("kattn-once: served x%d (last: a %s call, %d selection(s), stock's never ranked)", n,
                         _L["kind"], _L["served"])
    if _L["mode"] != "check":
        return
    kind = _L["kind"]
    if _L["miss"] is None and not _L["compared"] and _L["incomplete"] is None:
        _L["incomplete"] = "no kernel call compared"
    votes = _vote((_ST["local"] or _L["miss"], _L["incomplete"], kind, _L["rows"]))
    bad = [(r, v[0]) for r, v in enumerate(votes) if v[0] is not None]
    kinds = sorted({v[2] for v in votes})
    if bad or len(kinds) != 1:
        _ST["on"] = False
        r, why = bad[0] if bad else (0, f"the ranks took paths {kinds}")
        _log.warning("kattn-once: OFF (check of a %s call, rank %d: %s) -- the king's two selections on every rank "
                     "from here on", kind, r, why)
        return
    short = [(r, v[1]) for r, v in enumerate(votes) if v[1] is not None]
    if short:
        _say(f"short {kind}", f"kattn-once: check {kind} not counted (rank {short[0][0]}: {short[0][1]}); the king's "
                              "result served")
        return
    if _ST["passed"][kind] >= CHECKS:
        _ST["rechecks"] += 1
        n = _ST["rechecks"]
        if n & (n - 1) == 0:
            _log.warning("kattn-once: re-check x%d (last: %s, %d rows) equal on every rank (as sets; %d compares so "
                         "far equal only as sets: order differs)", n, kind, _L["rows"], _ST["order_only"])
        return
    _ST["passed"][kind] += 1
    _log.warning("kattn-once: check %d/%d %s: the kernel's K' table, top-k width and lengths from ONCE's picks equal "
                 "the king's on every rank (%d rows; set-exact: %d compares so far equal only as sets, order differs)",
                 _ST["passed"][kind], CHECKS, kind, _L["rows"], _ST["order_only"])
